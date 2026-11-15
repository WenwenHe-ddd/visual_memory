"""Run M2A with validated evidence IDs; same CLI as memeye_m2a.py.

Use a fresh output. Existing queue implementations and old artifacts stay intact.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_m2a as base
from scripts.memory.m2a_evidence import POLICY, install_evidence_policy
from scripts.memory.m2a_length_prompt import POLICY as LENGTH_POLICY, LengthLimitedQwen


def run(args):
    if (args.output / 'protocol.json').exists():
        raise ValueError('Use a fresh output root; existing experiments are immutable')
    original_prepare = base.prepare
    original_qwen = base.LocalQwen
    base.prepare_vendor()
    sys.path.insert(0, str(base.VENDOR / 'runtime'))

    def prepare(args, patches):
        observations, qs, protocol = original_prepare(args, patches)
        protocol['evidence_policy'] = POLICY
        protocol['output_length_policy'] = LENGTH_POLICY
        for path in [Path(__file__), ROOT / 'scripts/memory/m2a_evidence.py',
                     ROOT / 'scripts/memory/m2a_length_prompt.py']:
            name = str(path.relative_to(ROOT))
            protocol['implementation_sha256'][name] = base.sha(path)
            target = args.output / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        protocol.pop('signature')
        protocol['signature'] = base.digest(protocol)
        base.write(args.output / 'protocol.json', protocol)
        return observations, qs, protocol

    cleanup = install_evidence_policy(args.output)
    base.prepare = prepare
    base.LocalQwen = LengthLimitedQwen
    try:
        return original_run(args)
    finally:
        cleanup()
        base.prepare = original_prepare
        base.LocalQwen = original_qwen


original_run = base.run

if __name__ == '__main__':
    base.run = run
    base.main()
