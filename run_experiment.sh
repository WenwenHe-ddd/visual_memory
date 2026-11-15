#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$experiment_root"
export PYTHONPATH="$experiment_root/.vendor${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python3 scripts/data/prepare.py
python3 -m unittest discover -s scripts/tests -t . -p 'test_*.py' -q
python3 scripts/data/generate.py
python3 scripts/analysis/analyze.py
python3 scripts/analysis/report.py
