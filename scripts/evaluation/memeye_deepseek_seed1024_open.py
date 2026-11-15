"""Isolated seed1024 adaptation of the signed DeepSeek Open reader.

Native messages, output caps, retry policy and raw-output persistence are kept.
The original implementation is cloned in memory; no seed42 source is edited.
Only literal seed values and the transport binding change. Judge stays seed42.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
from functools import lru_cache
import json
from pathlib import Path
import types
from types import SimpleNamespace

from scripts.evaluation import memeye_deepseek_open_reader as original
from scripts.evaluation import memeye_deepseek_seed1024_transport as transport

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')


@lru_cache(maxsize=1)
def reader_module():
    class Seed(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 42:
                return ast.copy_location(ast.Constant(1024), node)
            return node

    path = Path(original.__file__)
    tree = ast.fix_missing_locations(Seed().visit(ast.parse(path.read_text())))
    value = types.ModuleType('memeye_deepseek_seed1024_reader_runtime')
    value.__file__ = str(path)
    value.__package__ = original.__package__
    exec(compile(tree, str(path), 'exec'), value.__dict__)

    def settings(model=transport.MODEL):
        transport.canonical_model(model)
        return dict(model=model, max_tokens=2048, temperature=0, seed=1024,
                    enable_thinking=False, n=1, stream=False)

    value.api = SimpleNamespace(MODEL=transport.MODEL, settings=settings,
        chat=transport.chat_request, load_config=transport.load_config,
        APIError=transport.APIError, vision=transport.original, __file__=transport.__file__)
    value.text_api = transport
    value.RUN = OUT
    value.code_files = lambda: [Path(__file__), Path(transport.__file__), path,
        Path(value.common.__file__), Path(transport.original.__file__),
        ROOT / 'scripts/evaluation/openai_chat_transport.py']
    return value


def run_reader(root, method_id, reference_correct=0, cache=None, workers=4,
               env_file=ENV, limit=0, phase='run'):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(root=root, method_id=method_id, reference_correct=reference_correct,
        cache=Path(cache or OUT / 'answer_api_cache').resolve(), workers=workers,
        env_file=Path(env_file), limit=limit)
    module = reader_module()
    if phase == 'prepare':
        return module.prepare(args)
    if phase == 'run':
        return module.run(args)
    if phase == 'collect':
        return module.collect(args)
    raise ValueError('Unexpected reader phase')


def judge(root, method_id, label, env_file=ENV, workers=8):
    from scripts.evaluation import memeye_deepseek_open_judge as original_judge
    root = Path(root).resolve()
    args = SimpleNamespace(input=root / 'judge_inputs.jsonl', output=root / 'judge',
        method_id=method_id, method_label=label, source_protocol=root / 'protocol.json',
        cache_root=[], env_file=Path(env_file), workers=workers, max_attempts=5, limit=0)
    protocol, requests, question_map = original_judge.prepare(args)
    if protocol['seed'] != 42:
        raise ValueError('Evaluator seed changed')
    report = original_judge.base.run(args, protocol, requests, question_map)
    original_judge.publish_summary(args, protocol, report)
    if report['status'] != 'complete':
        raise ValueError('Incomplete fixed-seed42 judge coverage; raw outputs retained')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'collect', 'judge'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method-id', required=True)
    parser.add_argument('--reference-correct', type=int, default=0)
    parser.add_argument('--label')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--env-file', type=Path, default=ENV)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'seed1024_open.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'judge':
            result = judge(args.root, args.method_id, args.label or args.method_id,
                           args.env_file, args.workers)
        else:
            result = run_reader(args.root, args.method_id, args.reference_correct,
                args.cache, args.workers, args.env_file, args.limit, args.phase)
        if isinstance(result, dict):
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
