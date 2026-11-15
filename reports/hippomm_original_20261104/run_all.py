#!/usr/bin/env python3
"""Supervise the two requested GPU-0 experiments without changing method code."""
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
OUTPUT = ROOT / 'results/hippomm_original_20261104'
EXPECTED = {'memeye': 371, 'mem_gallery': 1711}


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
            '--memory', 'hippomm_original', '--dataset', dataset, '--device', 'cuda:0',
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
             'model': 'Qwen3-VL-8B-Instruct', 'physical_gpu': 0,
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
            record['question_error_attempts'] = count_lines(directory / 'online/question_errors.jsonl')
            record['reader_calls_cached'] = len(list((directory / 'online/llm_cache').glob('*.json')))
            memory = directory / 'offline/memory.json'
            if memory.exists():
                record['memory_counts'] = json.loads(memory.read_text())['counts']
        save(REPORT / 'status.json', state)
        lines = ['HippoMM representative-originals paired evaluation', 'Updated (UTC): ' + state['updated_at'],
                 'Status: ' + state['status'], 'Model: Qwen3-VL-8B-Instruct; physical GPU: 0', '']
        for dataset, record in state['datasets'].items():
            lines += [f"{dataset}: {record['status']}; phase={record.get('phase', '-')}; "
                      f"answers={record['answered_questions']}/{record['expected_questions']}; "
                      f"new reader calls={record['reader_calls_cached']}; question errors={record['question_error_attempts']}"]
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
    state['finished_at'] = now()
    publish()
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
