"""Independently reconstruct alias requests and scores from frozen original inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.audit_memeye_gpt54_seed1042_baselines import read, lines, sha, digest, index, choice
from scripts.evaluation.openai_backbone_transport import build_request

SOURCE = ROOT/'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
REFERENCE = ROOT/'results/memeye_gpt54_seed42_baselines_20261024'
REFERENCE1042 = ROOT/'results/memeye_gpt54_seed1042_baselines_20261024'
OUTPUT = ROOT/'results/memeye_gpt54_alias_seed42_baselines_20261024'
SNAPSHOT = 'gpt-5.4-2026-03-05'
ALIAS = 'gpt-5.4'


def audit(output=OUTPUT):
    output = Path(output)
    protocol = read(output/'protocol.json')
    c = protocol['contract']
    assert c['model'] == ALIAS and c['caption_model'] == SNAPSHOT and c['seed'] == 42
    assert c['caption_rebuilt'] is False and c['retrieval_rebuilt'] is False
    assert digest(c) == protocol['experiment_signature']
    for name, expected in c['source_sha256'].items():
        assert sha(Path(name)) == expected, name
    historical = index(lines(SOURCE/'reader/requests.jsonl'), 'record_id')
    records = index(lines(output/'inputs/requests.jsonl'), 'record_id')
    manifest = index(lines(SOURCE/'reader_manifest.jsonl'), 'record_id')
    wires = index(lines(output/'api_request_manifest.jsonl'), 'record_id')
    ref_wires = index(lines(REFERENCE/'api_request_manifest.jsonl'), 'record_id')
    responses = index(lines(output/'responses.jsonl'), 'record_id')
    fresh = index(lines(output/'fresh_responses.jsonl'), 'payload_sha256')
    gold = index(lines(SOURCE/'inputs/evaluation.jsonl'), 'id')
    pairs = index(read(output/'paired_results.json'), 'question_id')
    assert set(historical) == set(records) == set(responses) == set(wires) == set(manifest)
    assert len(records) == 742 and len(fresh) == 720 and len(gold) == 371
    assert lines(output/'inputs/requests.jsonl') == lines(REFERENCE/'inputs/requests.jsonl')
    assert lines(output/'inputs/gold.jsonl') == list(gold.values())
    assert digest(list(wires.values())) == c['request_manifest_sha256']
    counts = Counter()
    errors = empty = length = 0
    for rid, row in records.items():
        h = historical[rid]
        body = build_request({k: h[k] for k in ['system', 'user_text', 'query_images']},
            model=SNAPSHOT, max_output_tokens=16, image_detail='high')
        assert digest(body) == h['request_sha256'] == row['historical_request_sha256']
        body['seed'] = 42
        assert digest(body) == ref_wires[rid]['request_sha256']
        body['model'] = ALIAS
        wire_hash = digest(body)
        assert wire_hash == wires[rid]['request_sha256'] and wires[rid]['requested_model'] == ALIAS
        assert set(body) == {'model', 'messages', 'seed', 'temperature', 'reasoning_effort',
                            'max_completion_tokens', 'n', 'stream', 'store'}
        assert c['reader_system_prompt'] == h['system']
        assert row['reader_text'] == h['user_text'] and row['query_images'] == h['query_images']
        assert row['top_memory_ids'] == manifest[rid]['top_memory_ids'] and len(row['top_memory_ids']) == 5
        assert row['payload_sha256'] == h['payload_sha256']
        a, f = responses[rid], fresh[row['payload_sha256']]
        assert a['request_sha256'] == wire_hash and a['requested_model'] == ALIAS
        assert re.fullmatch(r'gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?', a['returned_model'])
        assert a['run_id'] == protocol['run_id'] and a['experiment_signature'] == protocol['experiment_signature']
        assert a['historical_response_reuse'] is False and a['top_memory_ids'] == row['top_memory_ids']
        assert a['fresh_inference_executed_for_this_record'] == (rid == f['record_id'])
        assert a['same_run_shared_inference_record_id'] == (None if rid == f['record_id'] else f['record_id'])
        for key in ['raw_response', 'response_id', 'request_sha256', 'request_id', 'usage', 'execution_id', 'returned_model']:
            assert a[key] == f[key]
        pred = choice(a['raw_response'])
        assert pred == a['predicted_choice']
        qid, condition = rid.rsplit('::', 1)
        correct = pred == gold[qid]['gold_choice']
        counts[condition] += correct
        prefix = 'baseline' if condition == 'baseline_question_only' else 'expanded'
        assert pairs[qid][prefix+'_answer'] == pred and pairs[qid][prefix+'_correct'] == correct
        errors += pred is None
        empty += not a['raw_response']
        length += a['finish_reason'] == 'length'
    fresh_rows = list(fresh.values())
    returned_models = sorted({r['returned_model'] for r in fresh_rows})
    assert len(returned_models) == 1
    new_ids = {r['response_id'] for r in fresh_rows}
    old_ids = {r['response_id'] for r in lines(SOURCE/'reader/fresh_responses.jsonl')}
    for directory in [REFERENCE, REFERENCE1042]:
        old_ids.update(r['response_id'] for r in lines(directory/'fresh_responses.jsonl'))
    probe = read(REFERENCE/'seed_support_probe.json')
    old_ids.update(r['response']['id'] for r in probe['attempts'])
    probe_path = ROOT/'reports/openai_gpt54_alias_probe_20261024.json'
    if probe_path.exists():
        old_ids.add(read(probe_path)['response']['id'])
    assert len(new_ids) == 720 and len({r['request_id'] for r in fresh_rows}) == 720
    assert not new_ids & old_ids
    attempts = lines(output/'api_attempts.jsonl')
    successful = [r for r in attempts if r['status'] == 'success']
    assert len(successful) == 720 and {r['response_id'] for r in successful} == new_ids
    assert all(r['requested_model'] == ALIAS for r in attempts)
    summary = read(output/'summary.json')
    v = summary['validation']
    assert summary['status'] == 'complete' and summary['seed'] == 42 and summary['requested_model'] == ALIAS
    assert summary['returned_models'] == v['returned_models'] == returned_models
    assert summary['overall']['questions'] == 371
    assert summary['overall']['baseline_correct'] == counts['baseline_question_only']
    assert summary['overall']['expanded_correct'] == counts['master_answer_query']
    reasoning = sum(r.get('reasoning_tokens') or 0 for r in fresh_rows)
    assert (v['format_errors'], v['empty_responses'], v['token_limit_records'], v['reasoning_tokens_reported']) == (errors, empty, length, reasoning)
    assert v['responses_sha256'] == sha(output/'responses.jsonl')
    assert v['fresh_responses_sha256'] == sha(output/'fresh_responses.jsonl')
    for name, annotation in summary['method_annotations'].items():
        assert len(annotation) == 5 and annotation['5. 回答方式']['seed'] == 42
        assert annotation['5. 回答方式']['requested_model'] == ALIAS
        assert annotation['5. 回答方式']['correct'] == counts[name]
        assert annotation['3. 检索方式']['zscore'] is False
    comparison = {}
    for label, directory in [('snapshot_seed42', REFERENCE), ('snapshot_seed1042', REFERENCE1042), ('snapshot_unseeded', SOURCE)]:
        path = directory/'paired_results.json'
        refs = index(read(path) if path.exists() else lines(directory/'paired_results.jsonl'), 'question_id')
        comparison[label] = {}
        for condition, prefix in [('baseline_question_only', 'baseline'), ('master_answer_query', 'expanded')]:
            old_correct = new_correct = wins = losses = changed = 0
            for qid, g in gold.items():
                before, after = refs[qid][prefix+'_answer'], pairs[qid][prefix+'_answer']
                a, b = before == g['gold_choice'], after == g['gold_choice']
                old_correct += a
                new_correct += b
                wins += not a and b
                losses += a and not b
                changed += before != after
            values = dict(reference_correct=old_correct, alias_seed42_correct=new_correct,
                wrong_to_right=wins, right_to_wrong=losses, net_correct=new_correct-old_correct, answer_changed=changed)
            assert all(summary['comparisons'][label][condition][k] == value for k, value in values.items())
            comparison[label][condition] = values
    proof = read(output/'source_validation.json')
    assert proof['passed'] and proof['unweighted_correct'] == {'baseline_question_only': 236, 'master_answer_query': 241}
    result = {'passed': True, 'requested_model': ALIAS, 'returned_models': returned_models, 'seed': 42,
        'questions': 371, 'records': 742, 'fresh_requests': 720, 'same_run_shared_records': 22,
        'api_requests_independently_reconstructed': 742, 'only_wire_change_from_snapshot_seed42': 'model',
        'all_frozen_sources_verified': True, 'caption_api_provenance_verified': proof['caption_unique_images'],
        'historical_and_probe_response_ids_excluded': True, 'format_errors': errors, 'empty_responses': empty,
        'token_limit_records': length, 'reasoning_tokens': reasoning,
        'api_attempts': len(attempts), 'transient_failed_attempts': sum(r['status'] != 'success' for r in attempts),
        'correct': dict(counts), 'comparisons': comparison,
        'artifact_sha256': {name: sha(output/name) for name in ['protocol.json', 'summary.json',
            'responses.jsonl', 'fresh_responses.jsonl', 'paired_results.json', 'paired_results.csv',
            'api_request_manifest.jsonl', 'api_attempts.jsonl', 'alias_comparison.csv', 'report.html']}}
    (output/'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    audit(parser.parse_args().output)
