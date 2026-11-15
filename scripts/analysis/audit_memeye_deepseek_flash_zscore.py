"""Independent, standard-library audit of saved DeepSeek z-score experiment.

Reconstructs complete API bodies, rankings, score corrections, caption lineage,
fresh/shared-answer provenance and both QA/retrieval metrics. No model calls.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'results/memeye_deepseek_flash_zscore_lambda0p03_alpha1_20261025'
NAMES = ('baseline_question_only', 'master_answer_query_zscore')
MODEL = 'deepseek-v4.1-flash'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    assert len(result) == len(rows)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def choice(text):
    m = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', text.strip(), re.I)
    return m.group(1).upper() if m else None


def wire(payload, max_tokens):
    assert set(payload) in ({'system', 'user_text', 'query_images'}, {'user_text', 'query_images'})
    parts = []
    for im in payload['query_images']:
        assert set(im) == {'path', 'sha256'}
        data = Path(im['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == im['sha256']
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        else:
            assert data.startswith(b'\xff\xd8\xff')
            mime = 'image/jpeg'
        parts.append({'type': 'image_url', 'image_url': {
            'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode(), 'detail': 'high'}})
    parts.append({'type': 'text', 'text': payload['user_text']})
    messages = [{'role': 'system', 'content': payload['system']}] if 'system' in payload else []
    messages.append({'role': 'user', 'content': parts})
    return {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens,
            'n': 1, 'stream': False, 'enable_thinking': False}


def audit(output=DEFAULT):
    output = Path(output).resolve(); folder = output/'reader'
    protocol = read(folder/'protocol.json'); c = protocol['contract']
    assert digest(c) == protocol['experiment_signature']
    assert c['model'] == MODEL and c['api_config']['model'] == MODEL
    assert c['api_config']['base_url'] == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    assert c['lambda'] == .03 and c['alpha'] == 1 and c['conditions'] == list(NAMES)
    assert c['seed'] is None and c['temperature'] == 0 and c['enable_thinking'] is False
    assert c['reader_max_output_tokens'] == 16
    assert c['gold_sent_to_api'] is False and c['historical_answers_reused'] is False
    for path, expected in c['source_sha256'].items():
        assert sha(path) == expected, path
    source = Path(c['memory_path'])
    caps = indexed(lines(source/'captions.jsonl'), 'id')
    history = indexed(lines(source/'inputs/source_memories.jsonl'), 'id')
    qs = indexed(lines(source/'inputs/source_questions.jsonl'), 'id')
    gold = indexed(lines(folder/'gold.jsonl'), 'id')
    original = indexed(lines(ROOT/'data/independent_visual_inputs/qa_manifest.jsonl'), 'id')
    assert set(qs) == set(gold) == set(original) and len(gold) == 371
    for qid, g in gold.items():
        for key in ('gold_choice', 'gold_ids'):
            assert g[key] == original[qid][key]
        assert qs[qid]['task'] == original[qid]['task']
    assert len(history) == 438 and len(caps) == 437
    needed = {m['image_sha256'] for m in history.values()} | {im['sha256'] for q in qs.values() for im in q['query_images']}
    assert set(caps) == needed
    cap_requests = indexed(lines(source/'caption_generation/requests.jsonl'), 'record_id')
    cap_fresh = indexed(lines(source/'caption_generation/fresh_responses.jsonl'), 'request_sha256')
    for cap in caps.values():
        row = cap_requests[cap['source_response_record_id']]
        ans = cap_fresh[cap['request_sha256']]
        payload = {k: row[k] for k in ('system', 'user_text', 'query_images') if k in row}
        assert digest(wire(payload, 1024)) == cap['request_sha256'] == row['request_sha256']
        assert cap['caption'] == ans['raw_response'].strip() and cap['model'] == ans['returned_model'] == MODEL
        assert ans['status'] == 'ok' and ans['finish_reason'] == 'stop' and not ans.get('refusal')
    memory_text = {mid: m['dialogue']+'\nMemory description: '+caps[m['image_sha256']]['caption'] for mid,m in history.items()}
    old_retrieval = indexed(lines(source/'retrieval_results.jsonl'), 'question_id')
    retrieval = indexed(lines(output/'retrieval_results.jsonl'), 'question_id')
    assert set(retrieval) == set(qs)
    group_checks = 0; max_error = 0.
    for qid, row in retrieval.items():
        conditions = indexed(row['conditions'], 'name'); assert set(conditions) == set(NAMES)
        old = old_retrieval[qid]
        old_raw = next(x for x in old['conditions'] if x['name'] == NAMES[0])
        assert conditions[NAMES[0]]['full_rank_memory_ids'] == old_raw['full_rank_memory_ids']
        weighted = conditions[NAMES[1]]
        assert len(weighted['leaves']) == len(old['leaves'])
        old_groups = indexed(old['masters'], 'master_id')
        assert {g['group_id'] for g in weighted['groups']} == set(old_groups)
        for i, leaf in enumerate(weighted['leaves']):
            assert leaf['leaf_index'] == i
            assert leaf['memory_id'] == old['leaves'][i]['memory_id']
            assert leaf['group_id'] == old['leaves'][i]['master_id']
            assert abs(leaf['raw'] - old['leaves'][i]['expanded_score']) < 2e-6
        for g in weighted['groups']:
            old_g = old_groups[g['group_id']]
            assert g['query'] == old_g['augmented_query']
            assert g['selected_choice'] == old_g['selected_choice'] and g['selected_statement'] == old_g['selected_statement']
            ls = [l for l in weighted['leaves'] if l['group_id'] == g['group_id']]
            assert len(ls) == g['size']
            std = statistics.pstdev(l['white'] for l in ls); raw_std = statistics.pstdev(l['raw'] for l in ls)
            mean = statistics.mean(l['white'] for l in ls)
            white_tol = 32*sys.float_info.epsilon*max(1., max(abs(l['white']) for l in ls))
            raw_tol = 32*sys.float_info.epsilon*max(1., max(abs(l['raw']) for l in ls))
            for l in ls:
                # The stored singleton diagnostic retains its raw score; the
                # correction is zero, as required for a one-leaf master.
                if len(ls)==1:
                    z = l['white']
                elif std<=white_tol:
                    z = 0.
                elif len(ls)==2:
                    z = 1. if l['white']>mean else -1.
                else:
                    z = (l['white'] - mean)/std
                correction = raw_std*math.tanh(z) if len(ls)>1 and raw_std>raw_tol and std>white_tol else 0.
                error = max(abs(z-l['z']), abs(correction-l['correction']), abs(l['raw']+correction-l['weighted']))
                assert error < 2e-10; max_error = max(max_error, error)
            group_checks += 1
        order = [l['memory_id'] for l in sorted(weighted['leaves'], key=lambda l:-l['weighted'])]
        assert order == weighted['full_rank_memory_ids'] and order[:5] == weighted['top5_memory_ids']
    requests = indexed(lines(folder/'requests.jsonl'), 'record_id')
    fresh_list = lines(folder/'fresh_responses.jsonl'); fresh = indexed(fresh_list, 'request_sha256')
    logical = indexed(lines(folder/'responses.jsonl'), 'record_id')
    assert len(requests) == len(logical) == 742 and set(requests) == set(logical)
    reps = {}
    for row in requests.values(): reps.setdefault(row['request_sha256'], row['record_id'])
    assert set(fresh) == set(reps) and len(fresh) == c['unique_requests']
    prior_responses = lines(source/'reader/fresh_responses.jsonl')
    assert not {r['response_id'] for r in prior_responses} & {r['response_id'] for r in fresh_list}
    assert len({r['response_id'] for r in fresh_list}) == len(fresh_list)
    successes = [r for r in lines(folder/'api_attempts.jsonl') if r['status'] == 'ok']
    assert Counter(r['request_sha256'] for r in successes) == dict.fromkeys(fresh, 1)
    old_requests = indexed(lines(source/'reader/requests.jsonl'), 'record_id')
    checked = Counter(); results = {n: {'correct':0, 'gold_hits_top5':0, 'questions_with_any_gold':0} for n in NAMES}
    for rid, row in requests.items():
        assert set(row) == {'record_id','question_id','task','condition','top_memory_ids','system','user_text','query_images','payload_sha256','request_sha256'}
        qid = row['question_id']; name = row['condition']; q = qs[qid]; g = gold[qid]
        top = next(x['top5_memory_ids'] for x in retrieval[qid]['conditions'] if x['name'] == name)
        assert row['top_memory_ids'] == top and len(top) == len(set(top)) == 5
        text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' +
                '\n\n'.join(f'Memory {i}: {memory_text[mid]}' for i,mid in enumerate(top,1)) +
                '\n\nQuestion:\n'+q['question']+'\n\nOptions:\n' +
                '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD')+'\n\nAnswer with one letter only.')
        assert row['user_text'] == text and row['query_images'] == q['query_images']
        assert row['system'] == read(source/'inputs/protocol.json')['reader_system_prompt']
        payload = {k: row[k] for k in ('system','user_text','query_images')}
        assert digest(wire(payload,16)) == row['request_sha256'] and digest(payload) == row['payload_sha256']
        if name == NAMES[0]: assert row['request_sha256'] == old_requests[rid]['request_sha256']
        ans = fresh[row['request_sha256']]; final = logical[rid]
        assert ans['record_id'] == reps[row['request_sha256']]
        assert ans['run_id'] == protocol['run_id'] and ans['experiment_signature'] == protocol['experiment_signature']
        assert ans['requested_model'] == ans['returned_model'] == MODEL and ans['status'] == 'ok'
        assert ans['finish_reason'] == 'stop' and not ans.get('refusal') and ans['historical_response_reuse'] is False
        assert ans['enable_thinking'] is False and ans['reasoning_content_present'] is False and ans['reasoning_tokens'] in (None,0)
        if row['query_images']: assert ans['usage']['prompt_tokens_details']['image_tokens'] > 0
        for key in ('raw_response','predicted_choice','response_id','request_sha256','payload_sha256','execution_id','run_id','experiment_signature'):
            assert final[key] == ans[key]
        pred = choice(ans['raw_response']); assert pred is not None and pred == ans['predicted_choice']
        assert final['correct'] == (pred==g['gold_choice']) and final['top_memory_ids'] == top
        is_fresh = rid == ans['record_id']; assert final['fresh_inference_executed_for_this_record'] == is_fresh
        assert final['same_run_shared_inference_record_id'] == (None if is_fresh else ans['record_id'])
        checked[name] += 1
        hits = len(set(top)&set(g['gold_ids']))
        results[name]['correct'] += pred == g['gold_choice']
        results[name]['gold_hits_top5'] += hits
        results[name]['questions_with_any_gold'] += hits>0
    summary = read(output/'summary.json'); assert all(v==371 for v in checked.values())
    total = sum(len(set(g['gold_ids'])) for g in gold.values()); assert total==1041
    for name, r in results.items():
        for key, value in r.items(): assert summary['overall'][name][key] == value
        assert summary['overall'][name]['recall_at_5_micro'] == r['gold_hits_top5']/total
        assert summary['overall'][name]['accuracy'] == r['correct']/371
    report = {'passed':True,'questions':371,'caption_responses_verified':437,'memory_leaves':438,
              'group_score_corrections_recomputed':group_checks,'max_score_correction_error':max_error,
              'complete_wire_requests_reconstructed':742,'fresh_requests':len(fresh),'shared_records':742-len(fresh),
              'gold_references':1041,'overall':results,'audit_script_sha256':sha(__file__),
              'source_experiment_signature':protocol['experiment_signature'],
              'artifact_sha256':{name:sha(output/name) for name in ('summary.json','reader/protocol.json','reader/requests.jsonl',
                  'reader/gold.jsonl','reader/responses.jsonl','reader/fresh_responses.jsonl','reader/api_attempts.jsonl','retrieval_results.jsonl')},
              'scope_note':'Whitening also checked against independent inverse ridge during GPU retrieval; this offline audit checks saved score correction and ranking.'}
    (output/'external_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    audit(parser.parse_args().output)
