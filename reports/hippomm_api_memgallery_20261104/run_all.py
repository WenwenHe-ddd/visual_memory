#!/usr/bin/env python3
"""Supervise provider-specific MemGallery HippoMM experiments; persist progress and audit completion."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/hippomm_api_memgallery_20261104'
EXPECTED = {'gpt5': 1711, 'deepseek': 1711}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def count_lines(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def complete(dataset):
    directory = OUTPUT / dataset / 'online'
    manifest_path = directory / 'completion_manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete'
    assert manifest['questions'] == EXPECTED[dataset]
    for filename in manifest['artifacts']:
        assert (directory / filename).is_file(), filename
    questions = [json.loads(line) for line in (directory / 'questions.jsonl').read_text().splitlines()]
    answers = [json.loads(line) for line in (directory / 'qa_responses.jsonl').read_text().splitlines()]
    assert len(answers) == len(questions) == EXPECTED[dataset]
    assert len({r['id'] for r in answers}) == EXPECTED[dataset]
    assert {r['id'] for r in answers} == {r['id'] for r in questions}
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['questions'] == EXPECTED[dataset]
    return summary


def command(dataset, phase):
    args = [sys.executable, '-u', str(ROOT / 'scripts/visual_memory_pipeline.py'), phase,
            '--memory', 'hippomm_api', '--dataset', 'mem_gallery', '--hippomm-backbone', dataset,
            '--hippomm-max-new-tokens', '2048', '--hippomm-api-workers', '8', '--device', 'cuda:0',
            '--output', str(OUTPUT / dataset / phase)]
    if phase == 'online':
        args += ['--stage', 'all', '--memory-dir', str(OUTPUT / dataset / 'offline')]
    return args


def main():
    lock = (REPORT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1')
    state = {'started_at': now(), 'updated_at': now(), 'status': 'running',
             'models': ['gpt-5-2025-08-07', 'deepseek-v4.1-flash'], 'physical_gpu': 0,
             'output_root': str(OUTPUT), 'datasets': {}}
    jobs = {}

    def launch(dataset, phase):
        record = state['datasets'][dataset]
        log_path = REPORT / (dataset + '_' + phase + '.log')
        log = log_path.open('a', buffering=1)
        args = command(dataset, phase)
        log.write('\n' + now() + ' ' + json.dumps(args) + '\n')
        process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT)
        record.update(status='running', phase=phase, pid=process.pid,
                      phase_started_at=now(), command=args, log=str(log_path))
        jobs[dataset] = (process, log)

    def publish():
        state['updated_at'] = now()
        for dataset, record in state['datasets'].items():
            directory = OUTPUT / dataset
            record['answered_questions'] = count_lines(directory / 'online/qa_responses.jsonl')
            answer_path = directory / 'online/qa_responses.jsonl'
            record['provider_rejections'] = (sum(json.loads(line).get('status') == 'provider_rejection'
                for line in answer_path.read_text().splitlines()) if answer_path.exists() else 0)
            record['question_error_attempts'] = count_lines(directory / 'online/question_errors.jsonl')
            record['reader_calls_cached'] = len(list((directory / 'online/llm_cache').glob('*/call_*.json')))
            record['summary_events_cached'] = len(list((directory / 'offline/llm_cache').glob('*/call_0001.json')))
            memory = directory / 'offline/memory.json'
            if memory.exists():
                record['memory_counts'] = json.loads(memory.read_text())['events']
        save(REPORT / 'status.json', state)
        lines = ['HippoMM MemGallery provider-specific full-backbone evaluation', 'Updated (UTC): ' + state['updated_at'],
                 'Status: ' + state['status'], 'Models: GPT-5 / DeepSeek V4.1 Flash; CLIP physical GPU: 0', '']
        for dataset, record in state['datasets'].items():
            lines += [f"{dataset}: {record['status']}; phase={record.get('phase', '-')}; "
                      f"evaluated={record['answered_questions']}/{record['expected_questions']}; "
                      f"provider rejections={record['provider_rejections']}; summaries={record['summary_events_cached']}/860; "
                      f"cached LLM calls={record['reader_calls_cached']}; question errors={record['question_error_attempts']}"]
            if 'summary' in record:
                lines.append(json.dumps(record['summary'], ensure_ascii=False))
            if 'error' in record:
                lines.append('ERROR: ' + record['error'])
            lines += ['Results: ' + str(OUTPUT / dataset), 'Log: ' + record.get('log', '-'), '']
        temporary = REPORT / 'RESULTS.txt.tmp'
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(REPORT / 'RESULTS.txt')

    for dataset, expected in EXPECTED.items():
        state['datasets'][dataset] = {'expected_questions': expected}
        summary = complete(dataset)
        if summary is not None:
            state['datasets'][dataset].update(status='complete', summary=summary)
        else:
            launch(dataset, 'offline')
    publish()
    while jobs:
        time.sleep(15)
        for dataset, (process, log) in list(jobs.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del jobs[dataset]
            record = state['datasets'][dataset]
            record['exit_code'] = code
            if code != 0:
                tail = Path(record['log']).read_text(errors='replace')[-6000:]
                record.update(status='failed', failed_at=now(), error=tail)
                continue
            if record['phase'] == 'offline':
                launch(dataset, 'online')
            else:
                try:
                    summary = complete(dataset)
                    assert summary is not None, 'Missing completion manifest'
                    record.update(status='complete', completed_at=now(), summary=summary)
                except Exception as exc:
                    record.update(status='failed', failed_at=now(), error=repr(exc))
        publish()
    state['status'] = ('complete' if all(r['status'] == 'complete' for r in state['datasets'].values())
                       else 'failed_or_partial')
    if state['status'] == 'complete':
        verification = subprocess.run([sys.executable, str(REPORT / 'verify_results.py')],
                                      cwd=ROOT, env=environment, capture_output=True, text=True)
        (REPORT / 'verification.log').write_text(verification.stdout + verification.stderr)
        if verification.returncode:
            state['status'] = 'verification_failed'
    state['finished_at'] = now()
    publish()
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
