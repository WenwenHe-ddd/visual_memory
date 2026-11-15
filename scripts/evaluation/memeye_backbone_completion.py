"""Complete the full backbone run with the original BGE length policy.

The original caption/reader runner and its signed inputs remain immutable.
Overlong retrieval text follows the prior experiment's audited model limit.
"""
from pathlib import Path
import json

from scripts.evaluation import memeye_backbone as run


def pipeline(args):
    from scripts.evaluation.openai_backbone_transport import canonical_model
    if args.api_workers < 1 or args.api_limit < 0:
        raise ValueError('workers must be positive and request limit must be nonnegative')
    model = canonical_model(args.api_model)
    output = Path(args.output or run.DEFAULT_ROOT / model).resolve()
    stage = args.stage
    if stage in ('prepare', 'all'):
        run.prepare(output, model, args.api_env_file, args.api_workers)
    elif run.verify(output)['contract']['model'] != model:
        raise ValueError('Requested backbone differs from prepared experiment')
    if stage in ('caption', 'all'):
        if not run.generate_captions(output, args.api_limit):
            return
    if stage in ('retrieve', 'all'):
        run.verify_captions(output)
        if (output / 'retrieval_protocol.json').exists():
            run.verify_retrieval(output)
        else:
            from scripts.memory.memeye_backbone_retrieval import rebuild_and_retrieve
            overflow = None
            try:
                rebuild_and_retrieve(output, device=args.device)
            except ValueError as error:
                if ' exceeds BGE token limit ' not in str(error):
                    raise
                overflow = str(error)
            if overflow:
                # The strict stage already completed chronological construction.
                # This continuation validates/reuses those graphs and loads only
                # text encoders. No caption is shortened and no old image read.
                run.freeze(output / 'long_text_transition.json', {
                    'reason': overflow,
                    'action': 'Restore the original BGE 512-token encoding limit with an explicit truncation audit; keep full reader evidence.',
                    'caption_generation_changed': False, 'memory_construction_changed': False,
                    'reader_evidence_changed': False,
                })
                from scripts.memory.memeye_backbone_long_text import complete_retrieval
                complete_retrieval(output, device=args.device)
        run.analyze(output, require_reader=False)
    if stage in ('reader', 'all'):
        run.prepare_reader(output)
        if not run.execute_stage(output, 'reader', args.api_limit):
            return
    if stage in ('analyze', 'all'):
        print(json.dumps(run.analyze(output), ensure_ascii=False, indent=2), flush=True)
