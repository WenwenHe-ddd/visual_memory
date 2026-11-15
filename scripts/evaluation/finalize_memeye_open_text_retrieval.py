"""Finalize each running frozen-memory retrieval as soon as all 371 rows seal."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
import time
import subprocess
from pathlib import Path
from scripts.evaluation.memeye_open_text_retrieval import ROOT, OUT
from scripts.memory.reflexion_memory import write_json, sha


def main():
    pending = {'a_mem', 'simplemem'}
    while pending:
        for method in sorted(pending):
            folder = OUT / method
            if not (folder / 'complete.json').exists():
                continue
            for name in ('memeye_open_text_reader_inputs.py', 'audit_memeye_open_text_retrieval.py'):
                subprocess.run([sys.executable, str(ROOT / 'scripts/evaluation' / name),
                                '--method', method], check=True)
            logfile = OUT.parent / ('amem_retrieval.log' if method == 'a_mem' else 'simplemem_retrieval.log')
            warnings = [line for line in logfile.read_text().splitlines()
                        if line.startswith('Error during keyword search:')]
            write_json(folder / 'native_retrieval_warnings.json', {
                'status': 'complete', 'recoverable_keyword_search_errors': len(warnings),
                'messages': warnings, 'log_sha256': sha(logfile),
                'policy': 'Original upstream catches each lexical-search error and continues with other channels; no keyword rewriting or algorithm repair.'})
            write_json(folder / 'ready_for_reader.json', {
                'method': method, 'questions': 371, 'reader_manifest_sha256': sha(folder / 'reader_manifest.jsonl'),
                'retrieval_audit_sha256': sha(folder / 'validation_final.json'),
                'reader_input_audit_sha256': sha(folder / 'reader_input_audit.json'),
                'native_warnings_sha256': sha(folder / 'native_retrieval_warnings.json')})
            print(method, 'ready for the 371-question Open reader', flush=True)
            pending.remove(method)
        if pending:
            time.sleep(10)


if __name__ == '__main__':
    main()
