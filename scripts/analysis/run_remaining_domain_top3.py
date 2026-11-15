"""Run the fixed 1..15 distractor protocol for untested MemEye domains on GPU 0."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "card": "Card_Playlog_Test",
    "health": "Personal_Health_Dashboard_Assistant",
    "multiscene": "Multi-Scene_Visual_Case_Archive_Assistant",
    "social": "Social_Chat_Memory_Test",
    "outdoor": "Outdoor_Navigation_Route_Memory_Assistant",
}


def run(label: str, stage: str, args: list[str], env: dict[str, str]) -> None:
    log = ROOT / "results/other_domain_top3_interference_20261101" / f"{label}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{label}: {stage} started", flush=True)
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        print(log.read_text()[-3000:], flush=True)
        raise RuntimeError(f"{label}: {stage} failed; see {log}")
    print(f"{label}: {stage} complete · {log.read_text().splitlines()[-1]}", flush=True)


def main() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    for label, task in TASKS.items():
        source = ROOT / "results" / f"{label}_all_questions_distractor_review_20261101"
        output = ROOT / "results" / f"{label}_top3_interference_15_20261101"
        if not (source / "protocol.json").exists():
            run(label, "prepare", ["scripts/analysis/prepare_home_all_distractors.py",
                                   "--task", task, "--n", "15", "--device", "cuda:0",
                                   "--out", str(source)], env)
        if not (source / "query_similarity_protocol.json").exists():
            run(label, "score", ["scripts/analysis/score_home_reviewed_candidates.py",
                                 "--task", task, "--device", "cuda:0",
                                 "--source", str(source)], env)
        run(label, "evaluate", ["scripts/analysis/evaluate_home_top3_interference.py",
                                "--task", task, "--source", str(source),
                                "--output", str(output)], env)
    print(json.dumps({"completed": list(TASKS), "gpu": 0}), flush=True)


if __name__ == "__main__":
    main()
