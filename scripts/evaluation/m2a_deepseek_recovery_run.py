"""DeepSeek M2A task entrypoint with explicit image compatibility policy."""
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import m2a_deepseek_run as baseline
from scripts.memory import m2a_deepseek as transport
from scripts.memory import m2a_deepseek_recovery as images


def protocol_hook(args, protocol):
    baseline.protocol_hook(args, protocol)
    protocol.update(image_policy=images.POLICY,
                    recovery_amendment='image compatibility only; same source inputs, model, prompts and tools')
    protocol['memory_input'] = 'Original chronological dialogue; DeepSeek input images follow recorded recovery image policy; fresh facts/captions'
    for path in [Path(__file__), Path(images.__file__)]:
        name = str(path.relative_to(ROOT))
        protocol['implementation_sha256'][name] = baseline.single.base.sha(path)
        target = args.output/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def run(args):
    transport.image_part = images.image_part
    factory = partial(transport.DeepSeek, env_file=args.api_env_file or transport.DEFAULT_ENV,
                      cache_dir=args.api_cache_dir)
    return baseline.single.run(args, model_class=factory, protocol_hook=protocol_hook)


if __name__ == '__main__':
    baseline.single.base.run = run
    baseline.single.base.main()
