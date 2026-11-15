"""Independently audit the GPT-5.4 MemEye answer-model ablation, without APIs.

Reads frozen artifacts and writes only this run's independent_audit.json after
the complete summary exists. Request construction does not import the runner
or transport implementation being audited.
"""
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_reader_ablation_20261021/gpt-5.4-2026-03-05_none'
MINI = ROOT / 'results/memeye_reader_ablation_20261021/gpt-4o-mini'
MODEL = 'gpt-5.4-2026-03-05'
OLD = ROOT / 'reports/all_master_answer_query_20261021/reader'
CONDITIONS = ('baseline_question_only', 'master_answer_query')


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(value, compact=True):
    kwargs = {'ensure_ascii': False, 'sort_keys': True, 'allow_nan': False}
    if compact:
        kwargs['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def index(rows, key):
    result = {r[key]: r for r in rows}
    assert len(result) == len(rows), ('duplicate', key)
    return result


def aggregate(pairs):
    n = len(pairs)
    b, m = (sum(r[key] for r in pairs) for key in ('baseline_correct', 'expanded_correct'))
    wins = sum(not r['baseline_correct'] and r['expanded_correct'] for r in pairs)
    losses = sum(r['baseline_correct'] and not r['expanded_correct'] for r in pairs)
    d = wins + losses
    p = min(1., 2 * sum(math.comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d) if d else 1.
    return {'questions': n, 'baseline_correct': b, 'expanded_correct': m,
            'baseline_accuracy': b / n, 'expanded_accuracy': m / n,
            'delta_percentage_points': 100 * (m - b) / n,
            'wrong_to_correct': wins, 'correct_to_wrong': losses, 'mcnemar_exact_p': p}


def expected_request(contract, content):
    assert contract['model'] == MODEL and contract['reasoning_effort'] == 'none'
    return {'model':MODEL, 'messages':[{'role':'system','content':contract['reader_system_prompt']},
             {'role':'user','content':content}], 'temperature':0, 'reasoning_effort':'none',
            'max_completion_tokens':16, 'n':1, 'stream':False, 'store':False}


def check_fresh_flags(responses, fresh):
    for rid, row in responses.items():
        source = fresh[row['payload_sha256']]
        expected_fresh = rid == source['record_id']
        assert row['fresh_inference_executed_for_this_record'] is expected_fresh, ('fresh_flag', rid)
        assert row['same_run_shared_inference_record_id'] == (None if expected_fresh else source['record_id']), ('shared_source', rid)


def reasoning_counts(rows):
    counts = {'reported_zero':0, 'reported_nonzero':0, 'unreported':0, 'reported_token_total':0}
    for row in rows:
        value = ((row.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens')
        if 'reasoning_tokens' in row:
            assert row['reasoning_tokens'] == value, 'Parsed reasoning token count differs from usage'
        if value is None:
            counts['unreported'] += 1
        else:
            assert type(value) is int and value >= 0, 'Invalid reasoning token count'
            counts['reported_zero' if value == 0 else 'reported_nonzero'] += 1
            counts['reported_token_total'] += value
    assert counts['reported_nonzero'] == 0, 'reasoning_effort=none returned nonzero reasoning tokens'
    return counts


def main():
    if not (OUT / 'summary.json').exists() or not (OUT / 'validation_output.json').exists():
        print(json.dumps({'audit_status': 'waiting_for_complete_outputs', 'progress': read(OUT / 'progress.json') if (OUT / 'progress.json').exists() else None}))
        return
    protocol, validation, summary = (read(OUT / name) for name in ('protocol.json', 'validation_output.json', 'summary.json'))
    contract = protocol['contract']
    assert digest(contract) == protocol['experiment_signature']
    assert contract['base_url'] == 'https://api.openai.com/v1'
    assert contract['model'] == MODEL and contract['reasoning_effort'] == 'none'
    assert contract['seed'] is None and contract['max_completion_tokens'] == 16 and contract['temperature'] == 0
    for path, expected in contract['source_sha256'].items():
        assert sha(path) == expected, ('frozen_source', path)
    snapshots = {name: sha(OUT / name) for name in ('protocol.json', 'api_request_manifest.jsonl',
        'responses.jsonl', 'fresh_responses.jsonl', 'api_attempts.jsonl', 'summary.json', 'validation_output.json')}
    for name in ('responses', 'fresh_responses'):
        assert snapshots[name + '.jsonl'] == validation[name + '_sha256']
    prepared_list = lines(OUT / 'inputs/requests.jsonl')
    prepared = index(prepared_list, 'record_id')
    old_rows = lines(OLD / 'qa_manifest.jsonl')
    old = index(old_rows, 'record_id')
    assert [r['record_id'] for r in prepared_list] == [r['record_id'] for r in old_rows]
    wire_list = lines(OUT / 'api_request_manifest.jsonl')
    wire = index(wire_list, 'record_id')
    assert digest(wire_list) == contract['request_manifest_sha256']
    response_list = lines(OUT / 'responses.jsonl')
    responses = index(response_list, 'record_id')
    fresh_list = lines(OUT / 'fresh_responses.jsonl')
    fresh = index(fresh_list, 'payload_sha256')
    attempts = lines(OUT / 'api_attempts.jsonl')
    assert len(prepared) == len(old) == len(wire) == len(responses) == 742
    assert set(prepared) == set(old) == set(wire) == set(responses)
    assert len(fresh) == 719 and len({r['response_id'] for r in fresh_list}) == 719
    originals = index(lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'), 'id')
    memory = index(lines(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl'), 'id')
    gold = index(lines(OUT / 'inputs/gold.jsonl'), 'id')
    assert set(originals) == set(gold) == {r['id'] for r in prepared_list} and len(gold) == 371
    old_protocol = read(OLD / 'protocol.json')
    assert contract['reader_system_prompt'] == old_protocol['reader_settings']['reader_system_prompt']
    representative, image_paths = {}, set()
    for row in prepared_list:
        rid = row['record_id']
        original = old[rid]
        assert set(row) == {'record_id','id','task','condition','reader_text','query_images',
                            'query_image_paths','top_memory_ids','payload_sha256'}
        for key, value in row.items():
            assert value == original[key], ('old_manifest_field', rid, key)
        representative.setdefault(row['payload_sha256'], rid)
        q = originals[row['id']]
        assert q['gold_choice'] == gold[q['id']]['gold_choice']
        assert q['gold_ids'] == gold[q['id']]['gold_ids']
        evidence = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        evidence += '\n\n'.join(f'Memory {i}: {memory[mid]["reader_text"]}' for i, mid in enumerate(row['top_memory_ids'], 1))
        evidence += '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
        evidence += '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD') + '\n\nAnswer with one letter only.'
        assert evidence == row['reader_text']
        payload = {'system': contract['reader_system_prompt'], 'user_text': evidence, 'query_images': row['query_images']}
        assert digest(payload, compact=False) == row['payload_sha256'] == wire[rid]['payload_sha256']
        content = []
        assert row['query_image_paths'] == q['query_image_paths'] == [r['path'] for r in row['query_images']]
        for image in row['query_images']:
            data = Path(image['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == image['sha256']
            mime = 'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else 'image/jpeg'
            assert mime == 'image/png' or data.startswith(b'\xff\xd8\xff')
            content.append({'type':'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(data).decode(),
                                                            'detail': contract['image_detail']}})
            image_paths.add(image['path'])
        content.append({'type':'text', 'text':evidence})
        body = expected_request(contract, content)
        expected_hash = digest(body)
        assert expected_hash == wire[rid]['request_sha256'] == responses[rid]['request_sha256'], ('actual_request', rid)
        assert wire[rid]['query_image_count'] == len(row['query_images']) == responses[rid]['query_image_count']
    assert len(image_paths) == 6 and sum(bool(r['query_images']) for r in prepared_list) == 16
    assert set(representative) == set(fresh)
    check_fresh_flags(responses, fresh)
    for payload, source in fresh.items():
        assert source['record_id'] == representative[payload], 'Fresh source must be canonical first occurrence'
        assert source['run_id'] == protocol['run_id'] and source['experiment_signature'] == protocol['experiment_signature']
        success = [r for r in attempts if r['payload_sha256'] == payload and r['status'] == 'success']
        assert len(success) == 1
        attempt = success[0]
        for key in ('record_id','payload_sha256','request_sha256','execution_id','run_id','response_id','request_id','returned_model'):
            assert source[key] == attempt[key], ('attempt_chain', key)
        assert source['attempts'] == attempt['attempt']
        execution_attempts = [r for r in attempts if r['payload_sha256'] == payload
                              and r['execution_id'] == source['execution_id']]
        assert sorted(r['attempt'] for r in execution_attempts) == list(range(1, source['attempts'] + 1))
    for attempt in attempts:
        assert attempt['run_id'] == protocol['run_id']
        assert attempt['record_id'] == representative[attempt['payload_sha256']]
        assert attempt['request_sha256'] == wire[attempt['record_id']]['request_sha256']
        assert attempt['status'] in {'success','api_error'}
        if attempt['status'] == 'api_error':
            assert attempt['retryable'] is True, 'Unexpected permanent API error in completed run'
    for rid, row in responses.items():
        request = prepared[rid]
        source = fresh[request['payload_sha256']]
        expected_fresh = rid == source['record_id']
        assert row['fresh_inference_executed_for_this_record'] is expected_fresh, ('fresh_flag', rid)
        assert row['same_run_shared_inference_record_id'] == (None if expected_fresh else source['record_id']), ('shared_source', rid)
        for key in ('id','task','condition','top_memory_ids','payload_sha256'):
            assert row[key] == request[key], ('logical_field', rid, key)
        for key, value in source.items():
            if key not in ('record_id','id','task','condition'):
                assert row[key] == value, ('actual_api_source_field', rid, key)
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', row['raw_response'].strip(), re.I)
        choice = match.group(1).upper() if match else None
        assert row['predicted_choice'] == choice
        assert row['status'] == ('ok' if choice else 'format_error')
        assert row['hit_token_limit'] == (row['finish_reason'] == 'length')
        assert row['historical_response_reuse'] is False
    assert sum(row['fresh_inference_executed_for_this_record'] for row in responses.values()) == 719
    assert sum(not row['fresh_inference_executed_for_this_record'] for row in responses.values()) == 23
    source_pairs = read(OUT / 'inputs/pairs.json')
    pairs = []
    for pair in source_pairs:
        qid = pair['question_id']
        a, b = [responses[qid+'::'+condition] for condition in CONDITIONS]
        correct = gold[qid]['gold_choice']
        pairs.append({**pair, 'baseline_correct':a['predicted_choice']==correct,'expanded_correct':b['predicted_choice']==correct})
        if pair['same_reader_payload']:
            assert a['response_id'] == b['response_id'] and a['raw_response'] == b['raw_response']
    assert len(pairs) == 371 and summary['overall'] == aggregate(pairs)
    for domain in {p['domain'] for p in pairs}:
        assert summary['by_domain'][domain] == aggregate([p for p in pairs if p['domain'] == domain])
    for name, value in (('text_only', False),('with_query_image', True)):
        assert summary['by_query_modality'][name] == aggregate([p for p in pairs if p['query_has_image'] == value])
    qwen = index(lines(OLD / 'responses.jsonl'), 'record_id')
    qwen_correct, model_changes = {}, {}
    for condition in CONDITIONS:
        subset = [r for r in prepared_list if r['condition'] == condition]
        qwen_correct[condition] = sum(qwen[r['record_id']]['predicted_choice'] == gold[r['id']]['gold_choice'] for r in subset)
        changes = Counter()
        for r in subset:
            g = gold[r['id']]['gold_choice']
            old_correct = qwen[r['record_id']]['predicted_choice'] == g
            new_correct = responses[r['record_id']]['predicted_choice'] == g
            changes['qwen_wrong_api_correct'] += not old_correct and new_correct
            changes['qwen_correct_api_wrong'] += old_correct and not new_correct
        model_changes[condition] = dict(changes)
    assert qwen_correct == dict(zip(CONDITIONS,[184,195]))
    assert model_changes == summary['reader_model_changes']
    mini_audit = read(MINI/'independent_audit.json')
    mini_artifact_names = ('summary.json','responses.jsonl','fresh_responses.jsonl','validation_output.json')
    for name in mini_artifact_names:
        assert sha(MINI/name) == mini_audit['artifact_sha256'][name], ('frozen_gpt4omini_reference', name)
    assert (MINI/'inputs/requests.jsonl').read_bytes() == (OUT/'inputs/requests.jsonl').read_bytes()
    mini_answers = index(lines(MINI/'responses.jsonl'), 'record_id')
    mini_pairs = []
    for pair in source_pairs:
        qid = pair['question_id']
        a, b = [mini_answers[qid+'::'+condition] for condition in CONDITIONS]
        correct = gold[qid]['gold_choice']
        mini_pairs.append({**pair, 'baseline_correct':a['predicted_choice']==correct,
                           'expanded_correct':b['predicted_choice']==correct})
        for answer in (a,b):
            assert answer['payload_sha256'] == prepared[answer['record_id']]['payload_sha256']
    mini_overall = aggregate(mini_pairs)
    assert (mini_overall['baseline_correct'], mini_overall['expanded_correct']) == (196,196)
    assert mini_overall == read(MINI/'summary.json')['overall']
    assert mini_overall == summary['gpt4o_mini_reference']
    mini_changes = {}
    for condition in CONDITIONS:
        outcomes = [(mini_answers[r['record_id']]['predicted_choice'] == gold[r['id']]['gold_choice'],
                     responses[r['record_id']]['predicted_choice'] == gold[r['id']]['gold_choice'])
                    for r in prepared_list if r['condition'] == condition]
        mini_changes[condition] = {'mini_wrong_gpt5_correct':sum(not old and new for old,new in outcomes),
                                   'mini_correct_gpt5_wrong':sum(old and not new for old,new in outcomes)}
    assert mini_changes == summary['gpt4o_mini_to_gpt5_changes']
    three_reader_table = [
        {'model':'Qwen/Qwen3-VL-8B-Instruct', 'questions':371,
         'baseline_correct':qwen_correct[CONDITIONS[0]], 'expanded_correct':qwen_correct[CONDITIONS[1]]},
        {'model':'gpt-4o-mini-2024-07-18', **{key:mini_overall[key] for key in ('questions','baseline_correct','expanded_correct')}},
        {'model':MODEL, **{key:aggregate(pairs)[key] for key in ('questions','baseline_correct','expanded_correct')}},
    ]
    for row in three_reader_table:
        row['baseline_accuracy'] = row['baseline_correct']/row['questions']
        row['expanded_accuracy'] = row['expanded_correct']/row['questions']
        row['method_delta_percentage_points'] = 100*(row['expanded_correct']-row['baseline_correct'])/row['questions']
    usage = {k: sum(r['usage'][k] for r in fresh_list) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    assert usage == summary['usage_fresh_requests_only']
    counts = {'questions':371,'logical_records':742,'fresh_requests':719,'shared_records':23,
              'format_errors':sum(r['status']!='ok' for r in response_list),
              'empty_responses':sum(not r['raw_response'] for r in response_list),
              'token_limit_records':sum(r['hit_token_limit'] for r in response_list)}
    for key, value in counts.items():
        assert validation[key] == value
    models = Counter(r['returned_model'] for r in fresh_list)
    assert dict(models) == {MODEL:719}
    assert validation['returned_models'] == sorted(models) == summary['model_returned']
    statuses = Counter(r['status'] for r in attempts)
    end_reasons = Counter(r['finish_reason'] for r in fresh_list)
    assert statuses['success'] == 719 and dict(end_reasons) == {'stop':719}
    assert counts['format_errors'] == counts['empty_responses'] == counts['token_limit_records'] == 0
    assert not any(r['refusal'] for r in fresh_list)
    reasoning = reasoning_counts(fresh_list)
    assert validation['reasoning_tokens_reported'] == reasoning['reported_token_total']
    assert validation['reasoning_usage_missing_requests'] == reasoning['unreported']
    for name, expected in snapshots.items():
        assert sha(OUT/name)==expected, 'Output changed during independent audit'
    report = {
        'passed':True,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'auditor':'memeye_gpt5_reader_independent_audit','counts':counts,
        'all_frozen_source_hashes_verified':len(contract['source_sha256']),
        'all_742_actual_request_object_hashes_independently_reconstructed':True,
        'request_hash_representation':'SHA256 of canonical sorted compact JSON of the exact API request object; not a captured raw network byte hash.',
        'request_execution_evidence':'Frozen sender implementation recomputes request object hash immediately before calling the API; all execution attempt and returned-result hashes match reconstruction.',
        'exact_qwen_system_user_text_top5_and_query_image_hashes_preserved':True,
        'api_message_allowlist_has_no_gold_or_retrieval_hypotheses':True,
        'all_fresh_flags_equal_record_id_matches_actual_source_record_id':True,
        'all_shared_source_ids_present_if_and_only_if_not_fresh':True,
        'all_719_fresh_responses_link_to_unique_successful_api_attempt':True,
        'same_run_deduplication_pairs':23,'historical_api_response_reuse':False,
        'image_condition_records':16,'unique_original_query_images':6,
        'returned_model_counts':dict(models),'finish_reason_counts':dict(end_reasons),
        'attempt_status_counts':dict(statuses),'refusals':0,
        'retried_requests':sum(r['attempts']>1 for r in fresh_list),
        'system_fingerprints':dict(Counter(r['system_fingerprint'] for r in fresh_list)),
        'api_overall_recomputed':aggregate(pairs),'qwen_correct_recomputed':qwen_correct,
        'reader_model_changes_recomputed':model_changes,'usage_fresh_requests_only_recomputed':usage,
        'reasoning_effort':'none','seed_sent':False,'max_tokens_sent':False,'max_completion_tokens':16,
        'reasoning_token_reporting':reasoning,'three_reader_table_recomputed':three_reader_table,
        'gpt4o_mini_to_gpt5_changes_recomputed':mini_changes,
        'gpt4omini_reference_artifact_sha256':{name:sha(MINI/name) for name in mini_artifact_names},
        'input_provenance_limitation':'Exact frozen Qwen experiment inputs retained. Two dialogues are local benchmark rewrites; full caption generation chain is partly retired; see the earlier GPT-4o-mini public_data_provenance.json.',
        'artifact_sha256':snapshots,
    }
    (OUT/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'counts':counts,'overall':report['api_overall_recomputed'],
                      'model_counts':dict(models),'reasoning_token_reporting':reasoning,
                      'three_reader_table':three_reader_table},ensure_ascii=False))


if __name__ == '__main__':
    main()
