#!/usr/bin/env python3
"""Compatibility entrypoint; use visual_memory_pipeline.py for new runs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visual_memory_pipeline import main as pipeline_main


def main():
    args = sys.argv[1:]
    if args and args[0] in ('offline', 'online'):
        args[1:1] = ['--memory', 'caption_visual_master']
    pipeline_main(args)


if __name__ == '__main__':
    main()
