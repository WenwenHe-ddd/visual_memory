"""Export the complete ten-method Judge comparison after all scores are sealed."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/storage/hewenwen/VisualMemory')
RUN = Path(__file__).resolve().parent


def main():
    with (RUN / 'final_export.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        methods = json.loads((RUN / 'methods_manifest.json').read_text())['methods']
        while True:
            ready = True
            for method in methods:
                directory = Path(method['root']) / 'judge'
                paths = [directory / 'summary.json']
                if method['id'] != 'semrag':
                    paths.append(directory / 'llm_judge_summary.json')
                for path in paths:
                    if not path.exists():
                        ready = False
                        break
                    try:
                        state = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        ready = False  # A writer may still be completing the file.
                        break
                    if state.get('status') != 'complete' or state.get('questions_scored') != 371:
                        ready = False
                        break
            if ready:
                break
            time.sleep(15)
        subprocess.run([sys.executable, str(ROOT / 'scripts/analysis/export_memeye_open_llm_judge.py')],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
