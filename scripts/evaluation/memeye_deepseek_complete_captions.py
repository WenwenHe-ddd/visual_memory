"""Restart DeepSeek captions with a documented, uniformly larger output budget.

The stopped 256-token experiment is immutable. This additive entry prepares a
fresh run, changes only caption max_tokens to 1024 before any API request, and
binds the observed length stop as evidence. It never imports old responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from scripts.evaluation import memeye_deepseek_backbone as run
from scripts.evaluation.deepseek_backbone_transport import canonical_model

MODEL = 'deepseek-v4.1-flash-1017'
DEFAULT_FAILED = run.DEFAULT_ROOT / MODEL
DEFAULT_OUTPUT = run.DEFAULT_ROOT / (MODEL + '_complete_captions')
CAPTION_LIMIT = 1024
SCHEMA = 'deepseek_caption_budget_restart_v1'


def failure_evidence(failed_output):
    """Read-only evidence: matching frozen 256-token request and length stop."""
    failed = Path(failed_output).resolve()
    original = run.verify(failed)
    c = original['contract']
    if c['model'] != MODEL or c['caption_max_output_tokens'] != 256 or c['reader_max_output_tokens'] != 16:
        raise ValueError('Amendment requires the original DeepSeek 256-caption/16-reader run')
    files = [failed / 'protocol.json'] + [failed / 'caption_generation' / name for name in (
        'protocol.json', 'requests.jsonl', 'fresh_responses.jsonl', 'api_attempts.jsonl', 'progress.json')]
    meta = run.read(files[1])
    requests = run.indexed(run.lines(files[2]), 'record_id')
    responses = run.lines(files[3])
    if (meta['run_id'] != original['run_id'] or meta['experiment_signature'] != original['experiment_signature']
            or meta['max_output_tokens'] != 256 or meta['requests_sha256'] != run.sha(files[2])):
        raise ValueError('Original caption stage does not match its frozen experiment')
    bad = []
    for row in responses:
        request = requests[row['record_id']]
        if (row['run_id'] != original['run_id'] or row['experiment_signature'] != original['experiment_signature']
                or row['returned_model'] != MODEL or row['request_sha256'] != request['request_sha256']
                or row['payload_sha256'] != request['payload_sha256'] or row['historical_response_reuse'] is not False):
            raise ValueError('Original caption response has mismatched provenance')
        if row['status'] != 'ok':
            if (row['status'] != 'invalid_response' or row['finish_reason'] != 'length'
                    or (row.get('usage') or {}).get('completion_tokens') != 256):
                raise ValueError('Amendment can address only observed 256-token caption length stops')
            bad.append({key: row[key] for key in (
                'record_id', 'request_sha256', 'payload_sha256', 'response_id', 'finish_reason')})
    if not bad:
        raise ValueError('No observed caption length stop supports this amendment')
    return {
        'schema': SCHEMA,
        'reason': 'The original 256-token caption budget produced an incomplete caption (finish_reason=length); regenerate every caption with 1024 tokens available.',
        'previous_caption_max_output_tokens': 256, 'caption_max_output_tokens': CAPTION_LIMIT,
        'reader_max_output_tokens': 16, 'caption_prompt_changed': False,
        'caption_prompt_word_request': '80-120 words',
        'all_captions_regenerated': True, 'previous_caption_response_reuse': False,
        'failed_run_root': str(failed), 'failed_run_id': original['run_id'],
        'failed_experiment_signature': original['experiment_signature'],
        'failed_run_response_count': len(responses),
        'failed_run_ok_count': sum(row['status'] == 'ok' for row in responses),
        'length_stopped_records': bad,
        'failed_artifact_sha256': {str(path): run.sha(path) for path in files},
    }


def verify(output, failed_output=DEFAULT_FAILED):
    output = Path(output).resolve()
    protocol = run.verify(output)
    c = protocol['contract']
    if (c['model'] != MODEL or c['caption_max_output_tokens'] != CAPTION_LIMIT
            or c['reader_max_output_tokens'] != 16
            or c.get('caption_budget_amendment') != failure_evidence(failed_output)
            or protocol['run_id'] == c['caption_budget_amendment']['failed_run_id']
            or output == Path(failed_output).resolve()
            or c['source_sha256'].get(str(Path(__file__).resolve())) != run.sha(__file__)):
        raise ValueError('Output is not this independently signed complete-caption restart')
    for path, expected in c['caption_budget_amendment']['failed_artifact_sha256'].items():
        if c['source_sha256'].get(path) != expected:
            raise ValueError('Failed-run evidence is not bound to the new experiment signature')
    for name in ('source_memories.jsonl', 'source_questions.jsonl', 'evaluation.jsonl',
                 'images.jsonl', 'api_requests_caption.jsonl'):
        if run.sha(output / 'inputs' / name) != run.sha(Path(failed_output) / 'inputs' / name):
            raise ValueError('Restart changed original inputs or caption prompt')
    return protocol


def prepare(output=DEFAULT_OUTPUT, model=MODEL, env_file=run.DEFAULT_ENV, workers=3,
            failed_output=DEFAULT_FAILED):
    """Amend only an empty destination; once frozen, verify it without rewriting."""
    output, failed = Path(output).resolve(), Path(failed_output).resolve()
    if canonical_model(model) != MODEL or output == failed:
        raise ValueError('Restart requires a distinct output directory and the same pinned DeepSeek model')
    if (output / 'protocol.json').exists():
        protocol = verify(output, failed)
        if (protocol['contract']['workers'] != workers
                or protocol['contract']['api_env_file'] != str(Path(env_file).resolve())):
            raise ValueError('Restart configuration changed')
        return protocol
    if output.exists() and any(output.iterdir()):
        raise ValueError('Caption budget may only be amended in a fresh empty output directory')
    evidence = failure_evidence(failed)
    # No request is issued by prepare. The base contract is replaced exactly
    # once before stage artifacts exist; its source code remains untouched.
    base = run.prepare(output, model, env_file, workers)
    if any((output / name).exists() for name in ('caption_generation', 'captions.jsonl', 'reader', 'domains')):
        raise ValueError('Cannot amend an experiment after generation or memory construction began')
    contract = dict(base['contract'])
    contract['caption_max_output_tokens'] = CAPTION_LIMIT
    contract['caption_budget_amendment'] = evidence
    contract['source_sha256'] = {**contract['source_sha256'], **evidence['failed_artifact_sha256'],
                               str(Path(__file__).resolve()): run.sha(__file__)}
    protocol = {**base, 'run_id': str(uuid.uuid4()), 'contract': contract,
                'experiment_signature': run.digest(contract)}
    run.dump(output / 'protocol.json', protocol)
    return verify(output, failed)


def pipeline(args):
    if args.api_workers < 1 or args.api_limit < 0:
        raise ValueError('workers must be positive and request limit must be nonnegative')
    model = canonical_model(args.api_model)
    output = Path(args.output or DEFAULT_OUTPUT).resolve()
    failed = Path(args.failed_output).resolve()
    stage = args.stage
    if stage in ('prepare', 'all'):
        prepare(output, model, args.api_env_file, args.api_workers, failed)
    elif verify(output, failed)['contract']['model'] != model:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    run.add_arguments(parser)
    parser.add_argument('--failed-output', type=Path, default=DEFAULT_FAILED)
    parser.set_defaults(output=DEFAULT_OUTPUT)
    pipeline(parser.parse_args())
