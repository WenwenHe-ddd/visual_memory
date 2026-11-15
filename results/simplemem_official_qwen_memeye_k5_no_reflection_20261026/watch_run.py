"""Record live progress; analyze and audit after this specific benchmark exits."""
import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/storage/hewenwen/VisualMemory')
RESULT = ROOT / 'top5/simplemem_official_text_context5'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def process_alive(pid):
    try:
        proc = Path('/proc') / str(pid)
        command = proc.joinpath('cmdline').read_bytes().decode().split('\0')
        return str(ROOT) in command and 'scripts.visual_memory_pipeline' in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def status(launch):
    protocol = read(RESULT / 'protocol.json')
    progress, answered, complete = {}, 0, []
    for task in protocol['tasks']:
        target = RESULT / task
        progress_path = target / 'progress.json'
        progress[task] = read(progress_path) if progress_path.exists() else {'phase': 'waiting'}
        response_path = target / 'responses.jsonl'
        if response_path.exists():
            answered += len(response_path.read_text().splitlines())
        if (target / 'complete.json').exists():
            complete.append(task)
    return {'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'started_at_utc': launch['started_at_utc'], 'pid': launch['pid'],
            'status': 'running' if process_alive(launch['pid']) else 'stopped',
            'expected_questions': protocol['questions'], 'answered': answered,
            'complete_tasks': complete, 'tasks': progress,
            'completed_llm_calls': len(list((RESULT / 'calls').glob('*.json'))),
            'experiment_signature': protocol['experiment_signature']}


def main():
    launch = read(ROOT / 'launch.json')
    while True:
        current = status(launch)
        write(ROOT / 'run_status.json', current)
        if current['status'] != 'running':
            break
        time.sleep(30)
    command = launch['command'].copy()
    command[command.index('--stage') + 1] = 'analyze'
    with (ROOT / 'completion_checks.log').open('a', buffering=1) as log:
        analyzed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        audited = subprocess.run(['/usr/bin/python3', '-m', 'scripts.analysis.audit_simplemem_official_run',
                                  str(RESULT)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    current = status(launch)
    summary = read(RESULT / 'summary.json') if (RESULT / 'summary.json').exists() else {}
    audit = read(RESULT / 'audit.json') if (RESULT / 'audit.json').exists() else {}
    current.update(status='complete' if analyzed.returncode == audited.returncode == 0
                   and summary.get('status') == audit.get('status') == 'complete'
                   and audit.get('audit_passed_so_far') else 'incomplete_or_failed',
                   analyze_exit_code=analyzed.returncode, audit_exit_code=audited.returncode,
                   correct=summary.get('correct'), accuracy=summary.get('accuracy'),
                   summary=str(RESULT / 'summary.json'), audit=str(RESULT / 'audit.json'))
    write(ROOT / 'run_status.json', current)
    print(json.dumps(current, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
