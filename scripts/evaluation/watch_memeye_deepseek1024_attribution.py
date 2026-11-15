"""Wait for the independent Open retrieval stage, then enter the guarded helper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/progress.json'


def main():
    while True:
        state = json.loads(PROGRESS.read_text())
        if state.get('pid') != 2825379 or state.get('status') != 'running':
            raise RuntimeError('Expected running dispatcher is no longer active')
        if state.get('current_stage') == 'simplemem_open_retrieve':
            os.execv(sys.executable, [sys.executable, '-m',
                     'scripts.evaluation.run_memeye_deepseek1024_attribution_parallel'])
        if state.get('stages', {}).get('simplemem_open_retrieve', {}).get('status') == 'complete':
            raise RuntimeError('Safe overlap stage has already finished')
        time.sleep(5)


if __name__ == '__main__':
    main()
