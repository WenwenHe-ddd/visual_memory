#!/usr/bin/env python3
"""Offline audit of saved ours requests/results; never starts inference or HTTP."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_seed1024_ours as run
from scripts.evaluation import memeye_deepseek_seed1024_ours_open as open_run


def inspect(output=run.OUTPUT):
    output = Path(output).resolve()
    protocol = run.read(output / 'protocol.json')
    run.verify(protocol)
    retrieval = run.read(output / 'retrieval_audit.json')
    run.native.verify_hashes(retrieval['output_sha256'])
    historical = run.read(output / 'historical_caption_input_audit.json')
    for item in historical['checked']:
        assert run.sha(item['artifact']) == item['sha256']
    captions = run.lines(run.SOURCE / 'captions.jsonl')
    assert len(captions) == 437
    assert all(c['model'] == run.MODEL and c['seed'] == 42 for c in captions)
    historical_responses = run.native.unique(
        run.lines(run.SOURCE / 'caption_generation/fresh_responses.jsonl'), 'request_sha256')
    for caption in captions:
        response = historical_responses[caption['request_sha256']]
        assert caption['caption'] == response['raw_response'].strip()
        assert response['seed'] == 42 and response['returned_model'] == run.MODEL
    attached_memories, attached_questions, model = run.native.attach_captions(
        run.lines(run.SOURCE / 'inputs/source_memories.jsonl'),
        run.lines(run.SOURCE / 'inputs/source_questions.jsonl'), captions)
    assert model == run.MODEL
    assert attached_memories == run.lines(output / 'inputs/memories.jsonl')
    assert attached_questions == run.lines(output / 'inputs/questions.jsonl')
    assert protocol['contract']['caption_generation_calls'] == 0

    records = run.lines(output / 'reader/requests.jsonl')
    assert len(records) == len(run.native.unique(records, 'record_id')) == 1484
    cohort = {q['id'] for q in attached_questions}
    assert len(cohort) == 371
    for condition in run.CONDITIONS:
        group = [r for r in records if r['condition'] == condition]
        assert len(group) == 371 and {r['question_id'] for r in group} == cohort
    transport = run.transport()
    # Only pure local request construction is used. No config/credential read.
    expected = {}
    by_payload = {}
    for record in records:
        payload = {k: record[k] for k in ('system', 'user_text', 'query_images')}
        assert run.digest(payload) == record['payload_sha256']
        if record['payload_sha256'] in by_payload:
            assert by_payload[record['payload_sha256']] == payload
            continue
        by_payload[record['payload_sha256']] = payload
        wire = transport.build_request(payload, model=run.MODEL, max_output_tokens=16, image_detail='high')
        key = transport.request_sha256(wire)
        assert key not in expected
        expected[key] = wire
    assert len(expected) == 1041

    saved = sorted((output / 'api_calls').glob('*.json'))
    mcq_keys, counts, cap_counts, finishes, rejections = set(), Counter(), Counter(), Counter(), Counter()
    image_wires = 0
    for path in saved:
        cache = run.read(path)
        wire, response = cache['request'], cache['response']
        key = transport.request_sha256(wire)
        assert key == path.stem == cache['request_sha256']
        assert type(wire['seed']) is int and wire['seed'] == 1024
        assert wire['model'] == run.MODEL and wire['temperature'] == 0
        assert wire['enable_thinking'] is False and wire['n'] == 1 and wire['stream'] is False
        cap_counts[wire['max_tokens']] += 1
        assert wire['max_tokens'] in (16, 2048)
        assert response['seed'] == 1024
        if response['status'] == 'ok':
            assert response['returned_model'] == run.MODEL
            assert response['request_sha256'] == key
            assert response['generated_tokens'] <= wire['max_tokens']
            finishes[response['finish_reason']] += 1
        else:
            assert response['status'] == 'provider_input_rejected' and response['raw_response'] is None
            rejections[response.get('error_kind')] += 1
        if wire['max_tokens'] == 16:
            assert key in expected and wire == expected[key]
            mcq_keys.add(key)
            counts[response['status']] += 1
            image_wires += any(isinstance(message['content'], list) and any(
                part.get('type') == 'image_url' for part in message['content']) for message in wire['messages'])

    result = {
        'passed': True, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_kind': 'read-only local saved-wire audit; no API calls', 'api_calls_by_auditor': 0,
        'main_protocol_sha256': run.sha(output / 'protocol.json'),
        'caption_audit': {'passed': True, 'caption_seed': 42, 'caption_model': run.MODEL,
            'images': 437, 'source_sha256': run.sha(run.SOURCE / 'captions.jsonl'),
            'original_seed42_hashes_reverified': True,
            'attached_memories_match_original': len(attached_memories),
            'attached_questions_match_original': len(attached_questions), 'new_caption_generation_calls': 0},
        'mcq_audit': {'logical_records': len(records), 'questions_per_condition': 371,
            'expected_unique_wire_requests': len(expected), 'saved_unique_wire_requests': len(mcq_keys),
            'pending_unique_wire_requests': len(set(expected) - mcq_keys), 'saved_wire_statuses': dict(counts),
            'saved_image_wire_requests': image_wires,
            'actual_wire_seed': 1024, 'actual_wire_model': run.MODEL, 'actual_mcq_token_cap': 16,
            'actual_wires_equal_freshly_reconstructed_manifest_payloads': True,
            'final_results_available': False},
        'saved_cache_token_cap_counts': dict(cap_counts), 'saved_cache_finish_reasons': dict(finishes),
        'saved_cache_rejection_kinds': dict(rejections),
        'source_sha256': {str(Path(__file__)): run.sha(Path(__file__))}}
    final_paths = [output / n for n in ('summary.json', 'independent_audit.json', 'reader/responses.jsonl')]
    if all(p.is_file() for p in final_paths):
        reference = open_run.validate_mcq_reference(output)
        result['mcq_audit'].update(final_results_available=True,
            actual_correct_by_method=reference['per_method'], raw_responses_regraded=1484,
            final_artifact_sha256={str(p): run.sha(p) for p in final_paths})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.output)
    run.dump(args.report, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
