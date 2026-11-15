"""M2A test-time agents with single-pass construction; same CLI as memeye_m2a."""
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_m2a as base
from scripts.memory.m2a_evidence import POLICY, install_evidence_policy
from scripts.memory.m2a_length_prompt import POLICY as LENGTH_POLICY
from scripts.memory.m2a_single_pass import EXTRACTION_PROMPT, SinglePassQwen, RetrievalLimits, search


def run(args, model_class=None, protocol_hook=None):
    if args.memory_source:
        raise ValueError('Single-pass trials require fresh construction')
    args.single_pass = True
    args.mandatory_query = True
    base.prepare_vendor()
    sys.path.insert(0, str(base.VENDOR / 'runtime'))
    from agent.stores import SemanticStore
    original_prepare, original_qwen = base.prepare, base.LocalQwen
    original_search = SemanticStore.hybrid_search
    limits = RetrievalLimits()

    def prepare(args, patches):
        observations, qs, protocol = original_prepare(args, patches)
        protocol.update(method='M2A retrieval with single-pass append-only construction (adaptation)',
            construction_mode='single_pass', extraction_prompt=EXTRACTION_PROMPT,
            construction_policy='Exactly one Qwen call and one memory per original round; no tools, search, update or deletion; program-bound singleton evidence IDs',
            extraction_json_policy='Strict two-string schema; only a missing final object brace may be appended to an otherwise complete response; logged per round; no repair of truncated generations or string contents',
            no_image_caption_policy='Program forces image_caption empty when the original round has no image; any discarded model-generated caption is logged without another LLM call',
            retrieval_limits=asdict(limits),
            retrieval='BGE Top5 + BM25 Top5 -> RRF Top5; image Top3; final RRF Top5 PER SEARCH; adaptive union may exceed 5',
            evidence_policy=POLICY, output_length_policy=LENGTH_POLICY)
        protocol['memeye_adaptations'][1] = 'Native images go to Qwen and SigLIP2; one unbound extraction call generates dialogue facts and a short image caption.'
        protocol['extraction_max_attempts'] = getattr(args, 'extraction_attempts', 1)
        if protocol['extraction_max_attempts'] > 1:
            protocol['construction_policy'] = ('One accepted memory per original round; initial extraction unchanged; '
                'on invalid JSON/schema only, retry the original input with an explicit JSON format reminder, '
                f"at most {protocol['extraction_max_attempts']} total calls per round; every rejected response logged; "
                'no tools, retrieval, updates or deletion during construction; program-bound singleton evidence IDs')
            protocol['memeye_adaptations'].append('Optional bounded extraction format retries; extra model calls reported explicitly.')
        for path in [Path(__file__), ROOT / 'scripts/memory/m2a_single_pass.py',
                     ROOT / 'scripts/memory/m2a_evidence.py', ROOT / 'scripts/memory/m2a_length_prompt.py',
                     ROOT / 'scripts/evaluation/audit_memeye_m2a.py']:
            name = str(path.relative_to(ROOT))
            protocol['implementation_sha256'][name] = base.sha(path)
            target = args.output / 'code_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        if protocol_hook is not None:
            protocol_hook(args, protocol)
        protocol.pop('signature')
        protocol['signature'] = base.digest(protocol)
        base.write(args.output / 'protocol.json', protocol)
        return observations, qs, protocol

    cleanup = install_evidence_policy(args.output)
    base.prepare, base.LocalQwen = prepare, model_class or SinglePassQwen
    SemanticStore.hybrid_search = lambda self, *a, **kw: search(self, limits, *a, **kw)
    try:
        return original_run(args)
    finally:
        cleanup()
        base.prepare, base.LocalQwen = original_prepare, original_qwen
        SemanticStore.hybrid_search = original_search


original_run = base.run
if __name__ == '__main__':
    base.run = run
    base.main()
