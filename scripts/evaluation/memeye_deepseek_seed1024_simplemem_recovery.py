"""Optional exact-wire proof adapter for an overstrict in-process fatal latch.

Default inspect is entirely offline. Run changes only acceptance of a recorded
data_inspection_failed event that already has a later, directly saved successful
native response AND matching successful broker cache. It adds no API retry,
waiting, prompt change, output repair, or new seed. Unknown failures still stop.
The signed original modules remain unchanged. Root coordinates any real run.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
SOURCE = OUT / 'text_methods/simplemem'
MODEL = 'deepseek-v4.1-flash'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, compact=True):
    options = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        options['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def proof(source, failure):
    """Return direct evidence only; an absent body/success/cache is unresolved."""
    source = Path(source)
    require(failure.get('status') == 'error' and failure.get('error_type') == 'http'
            and failure.get('provider_error_code') == 'data_inspection_failed'
            and failure.get('retryable') is False, 'Only the observed nonretryable inspection-error class is eligible')
    key = failure['request_sha256']; p = read(source / 'protocol.json')
    require(p['seed'] == 1024 and p['model'] == MODEL and p['mode'] == 'simplemem', 'Wrong source protocol')
    attempts = rows(source / 'api_attempts.jsonl')
    matching_errors = [(i, e) for i, e in enumerate(attempts, 1) if e['request_sha256'] == key and e['status'] == 'error']
    require(bool(matching_errors), 'Failure is not present in the actual append-only attempt ledger')
    error_line, latest = matching_errors[-1]
    require(all(latest.get(k) == v for k, v in failure.items()), 'Latest same-wire error differs from the current fatal')
    later = {e['response_id']: i for i, e in enumerate(attempts, 1)
             if i > error_line and e['request_sha256'] == key and e['status'] == 'ok'}
    require(bool(later), 'No later successful exact-wire attempt exists')
    raw = {r['output']['response_id']: r for r in rows(source / 'api_responses.jsonl')
           if r['output']['response_id'] in later}
    verified = {}
    for identity, record in raw.items():
        wire, out = record['request'], record['output']
        require(digest(wire) == key == out['request_sha256'] and wire['model'] == MODEL and wire['seed'] == 1024
                and wire['enable_thinking'] is False and wire['n'] == 1 and wire['stream'] is False
                and (wire['temperature'], wire['max_tokens']) in {(.1, 16384), (.2, 16384), (.3, 16384), (0., 16)},
                'Saved later wire differs from the original native request policy')
        require(out['seed'] == 1024 and out['requested_model'] == out['returned_model'] == MODEL
                and out['enable_thinking'] is False and isinstance(out['raw_response'], str)
                and out['hit_token_limit'] == (out['finish_reason'] == 'length'), 'Saved later provider output identity is invalid')
        verified[identity] = record
    for path in sorted((source / 'calls').glob('*.json')):
        call = read(path); req, out = call['request'], call['output']
        if out.get('response_id') not in verified:
            continue
        record = verified[out['response_id']]; wire = record['request']
        require(digest(req, False) == path.stem and req['seed'] == 1024 and req['experiment_signature'] == p['experiment_signature']
                and req['messages'] == wire['messages'] and req['max_new_tokens'] == wire['max_tokens']
                and req['temperature'] == wire['temperature'] and all(out.get(k) == v for k, v in record['output'].items())
                and 'inherited_from' not in call, 'Later broker cache does not contain that actual successful response')
        return dict(status='resolved_by_existing_native_success', error_attempt_line=error_line,
            successful_attempt_line=later[out['response_id']], wire_sha256=key,
            response_id=out['response_id'], broker_call_key=path.stem, broker_call_path=str(path.resolve()),
            broker_call_sha256=sha(path), seed=1024, model=MODEL, native_temperature=wire['temperature'],
            native_cap=wire['max_tokens'], finish_reason=out['finish_reason'], hit_token_limit=out['hit_token_limit'])
    raise ValueError('No matching successful native broker-cache output exists')


def inspect(source=SOURCE):
    source = Path(source).resolve()
    failures = [e for e in rows(source / 'api_attempts.jsonl') if e['status'] == 'error']
    results = []
    for event in failures:
        try:
            results.append(dict(failure=event, **proof(source, event)))
        except Exception as error:
            results.append(dict(status='unresolved_stop', failure=event, reason=str(error)))
    return dict(status='resolved_history' if results and all(r['status'] == 'resolved_by_existing_native_success' for r in results)
                else 'no_failures' if not results else 'unresolved_stop', source_root=str(source),
                historical_failures=len(failures), results=results, api_calls=0, model_loads=0, implementation_sha256=sha(__file__))


def run(source=SOURCE, recovery_root=None):
    """Root-only optional launch; original settings and append-only logs persist."""
    from scripts.evaluation import memeye_deepseek_seed1024_text_methods as original
    from scripts.evaluation import memeye_deepseek_seed1024_suite as suite
    source = Path(source).resolve()
    require(source == SOURCE.resolve(), 'Recovery run only supports the prepared canonical SimpleMem source')
    require(suite.export_allowed(), 'Prepared data-export authorization is missing or changed')
    recovery = Path(recovery_root or OUT / 'recovery/simplemem_exact_wire_acceptance_20261106')
    recovery.mkdir(parents=True, exist_ok=True)
    with (source.parent / 'simplemem.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = original.verify(source)
        contract = dict(kind='native_success_proven_terminal_latch_acceptance', seed=1024, model=MODEL,
            source_root=str(source), source_protocol_sha256=sha(source / 'protocol.json'),
            original_wrapper_sha256=sha(original.__file__), adapter_sha256=sha(__file__),
            unchanged='Original prompts, native temperatures/caps, native3-attempt client, task workers, construction/retrieval/reader and exact successful-call cache.',
            recovery_boundary='Only already proven later successful exact-wire inspection failures are resolved. Unknown failures stop immediately; no waits or API retries are added. Previously aborted tasks are rebuilt by the original run function.')
        path = recovery / 'protocol.json'
        if path.exists():
            require(read(path)['contract'] == contract, 'Recovery contract changed')
        else:
            path.write_text(json.dumps(dict(contract=contract, signature=digest(contract)), ensure_ascii=False, indent=2) + '\n')
        resolved = {}
        previous = original.check_fatal

        def check(model):
            for index, error in enumerate(model.fatal_errors):
                identity = (index, digest(error))
                if identity in resolved:
                    continue
                evidence = proof(source, error)  # Unknown/currently unproven failures still raise.
                event = dict(at_utc=datetime.now(timezone.utc).isoformat(), fatal_index=index,
                             original_failure=error, **evidence)
                with (recovery / 'resolved_native_failures.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                resolved[identity] = event
        try:
            original.check_fatal = check
            result = original.run('simplemem', source.parent)
        finally:
            original.check_fatal = previous
        report = dict(status='complete', source_summary_sha256=sha(source / 'summary.json'),
            recovery_protocol_sha256=sha(path), current_process_resolved_failures=len(resolved),
            historical_attempts_preserved=True, implementation_sha256=sha(__file__))
        (recovery / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('inspect', 'run'), default='inspect')
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--recovery-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.source_root) if args.phase == 'inspect' else run(args.source_root, args.recovery_root)), flush=True)
