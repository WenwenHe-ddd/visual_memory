"""Lambda=0.001, alpha=1 z-score on the GPT-5.4-caption memory whose unweighted baseline is 236/371.

Reuse the exact GPT-generated captions, dialogue, masters and option statements.
Re-encode retrieval on GPU, verify the unweighted rankings, then change only the
online weighting. Every answer is newly requested from the same GPT snapshot.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import html
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(key, value)
import numpy as np
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.memory.online_zscore import ZScoreWeighting

SOURCE = ROOT / 'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
BASE = ROOT / 'results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024'
RAW = ('baseline_question_only', 'master_answer_query')
NAMES = tuple(name + '_zscore' for name in RAW)
read, lines, sha, digest, dump = api.read, api.lines, api.sha, api.digest, api.dump


def verify_sources(source):
    """Read-only verification; preserve old signatures and disclose source drift."""
    source = Path(source).resolve()
    root = read(source / 'protocol.json')
    assert digest(root['contract']) == root['experiment_signature']
    assert root['contract']['model'] == api.MODEL
    assert root['contract']['scope'] == 'all_original_qwen_roles'
    audit = read(source / 'independent_audit.json')
    assert audit['audit_status'] == 'passed' and audit['errors'] == []
    paths = set()
    for filename, expected in audit['artifact_sha256'].items():
        assert sha(filename) == expected, filename
        paths.add(Path(filename))
    drift = []
    for name in ('protocol.json', 'inputs/protocol.json', 'retrieval_protocol.json',
                 'caption_generation/protocol.json', 'reader/protocol.json'):
        p = read(source / name)
        if 'contract' in p:
            assert digest(p['contract']) == p['experiment_signature']
        hashes = p.get('source_sha256', p.get('contract', {}).get('source_sha256', {}))
        for filename, expected in hashes.items():
            actual = sha(filename) if Path(filename).is_file() else None
            if actual != expected:
                assert Path(filename).suffix == '.py', ('changed data', filename)
                drift.append({'protocol': name, 'path': filename,
                              'historical_sha256': expected, 'current_sha256': actual})
            if actual is not None:
                paths.add(Path(filename))
        paths.add(source / name)
    from scripts.evaluation.memeye_backbone import verify_captions, verify_stage
    captions = verify_captions(source)
    old_requests, old_answers = verify_stage(source, 'reader', require_complete=True)
    gold = api.indexed(lines(source / 'inputs/evaluation.jsonl'), 'id')
    counts = {name: 0 for name in RAW}
    for r in old_requests:
        qid, condition = r['record_id'].rsplit('::', 1)
        counts[condition] += api.parse_choice(old_answers[r['request_sha256']]['raw_response']) == gold[qid]['gold_choice']
    assert list(counts.values()) == [236, 241], counts
    paths.update(source / n for n in ('independent_audit.json', 'summary.json', 'paired_results.jsonl',
                                     'reader/fresh_responses.jsonl', 'reader/requests.jsonl',
                                     'caption_generation/fresh_responses.jsonl', 'captions.jsonl'))
    return {'passed': True, 'source': str(source), 'caption_model': api.MODEL,
            'source_run_id': root['run_id'], 'unweighted_correct': counts,
            'caption_unique_images': len(captions), 'historical_source_drift': drift,
            'source_sha256': {str(p): sha(p) for p in sorted(paths)}}


def checked_weighting(query, vectors):
    result = ZScoreWeighting(.001, 1.).apply(query, vectors)
    # Independent dual ridge solve, without the production SVD implementation.
    centered = vectors - vectors.mean(axis=0)
    if len(vectors) == 1:
        check = vectors @ query
    else:
        adjusted = query - centered.T @ np.linalg.solve(
            centered @ centered.T + (len(vectors) - 1) * .001 * np.eye(len(vectors)),
            centered @ query)
        check = vectors @ adjusted
    np.testing.assert_allclose(result['white'], check, atol=2e-12, rtol=0)
    return result, float(np.max(np.abs(result['white'] - check)))


def retrieve(output, source, device):
    output, source = Path(output).resolve(), Path(source).resolve()
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        p = read(marker)
        assert p['memory_path'] == str(source)
        for filename, expected in {**p['source_sha256'], **p['output_sha256']}.items():
            assert sha(filename) == expected, filename
        return p
    assert not (output / 'protocol.json').exists(), 'Do not rebuild a signed API experiment.'
    proof = verify_sources(source)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    source_inputs = read(source / 'inputs/protocol.json')
    memories, questions, model_name = original.attach_captions(
        lines(source / 'inputs/source_memories.jsonl'), lines(source / 'inputs/source_questions.jsonl'),
        lines(source / 'captions.jsonl'))
    assert model_name == api.MODEL
    old = api.indexed(lines(source / 'retrieval_results.jsonl'), 'question_id')
    args = SimpleNamespace(device=device)
    original.configure_device(args)
    bge = original.load_text_encoder(source_inputs['text_encoder'], args.device)
    clip, tokenizer, limit = original.load_clip_text_encoder(source_inputs['image_encoder'], args.device)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'source_validation.json', proof)
    results, encoding, domains = [], [], sorted({m['task'] for m in memories})
    group_checks = leaf_checks = rank_checks = 0
    max_raw_error = max_inverse_error = 0.
    master_counts = {}
    for domain in domains:
        ms = [m for m in memories if m['task'] == domain]
        qs = [q for q in questions if q['task'] == domain]
        graph = read(source / 'domains' / domain / 'memory/graph.json')
        sums = np.load(source / 'domains' / domain / 'memory/master_sums.npy', allow_pickle=False)
        master_counts[domain] = len(graph['masters'])
        e, einfo = encode_complete(bge, [m['retrieval_text'] for m in ms],
                                  ids=[m['id'] for m in ms], field='dialogue+caption')
        query_texts, query_ids, statements = [], [], []
        for q in qs:
            candidates = [q['candidate_statements'][k] for k in 'ABCD']
            statements.extend(candidates)
            query_texts.extend([q['retrieval_query']] + [q['retrieval_query'] + '\nCandidate answer: ' + s for s in candidates])
            query_ids.extend(q['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D'))
        qv, qinfo = encode_complete(bge, query_texts, source_inputs['query_prefix'], ids=query_ids, field='query')
        cv, cinfo = original._clip_candidates(clip, tokenizer, limit, statements)
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        for qi, question in enumerate(qs):
            raw = original.rank_question(question, ms, graph, sums, qv[qi], e, cv[qi])
            prior = old[question['id']]
            for key in ('question', 'options', 'query_images', 'query_image_captions', 'baseline_retrieval_query', 'candidate_statements'):
                assert raw[key] == prior[key], (question['id'], key)
            for a, b in zip(raw['masters'], prior['masters']):
                for key in ('master_id', 'member_ids', 'selected_choice', 'selected_statement', 'augmented_query'):
                    assert a[key] == b[key], (question['id'], key)
                np.testing.assert_allclose(list(a['clip_candidate_scores'].values()), list(b['clip_candidate_scores'].values()), atol=2e-6, rtol=0)
            for a, b in zip(raw['conditions'], prior['conditions']):
                assert a['full_rank_memory_ids'] == b['full_rank_memory_ids'], (question['id'], a['name'], 'raw ranks')
                assert a['reader_context'] == b['reader_context']
                rank_checks += 1
            for field in ('baseline_score', 'expanded_score'):
                err = float(np.max(np.abs(np.array([l[field] for l in raw['leaves']]) - [l[field] for l in prior['leaves']])))
                assert err < 2e-6, (question['id'], field, err)
                max_raw_error = max(max_raw_error, err)
            conditions = []
            for name, raw_field in zip(NAMES, ('baseline_score', 'expanded_score')):
                groups = ([{'master_id': 'all_memory', 'selected_choice': None, 'selected_statement': None,
                            'augmented_query': question['retrieval_query']}]
                          if name == NAMES[0] else raw['masters'])
                leaves = [None] * len(ms)
                group_rows = []
                for g in groups:
                    indices = list(range(len(ms))) if name == NAMES[0] else [i for i, l in enumerate(raw['leaves']) if l['master_id'] == g['master_id']]
                    query_index = 0 if name == NAMES[0] else 1 + 'ABCD'.index(g['selected_choice'])
                    scores, error = checked_weighting(qv[qi, query_index], e[indices])
                    max_inverse_error = max(max_inverse_error, error)
                    np.testing.assert_allclose(scores['raw'], [raw['leaves'][i][raw_field] for i in indices], atol=2e-15, rtol=0)
                    group_rows.append({'group_id': g['master_id'], 'size': len(indices), 'query': g['augmented_query'],
                                       'selected_choice': g['selected_choice'], 'selected_statement': g['selected_statement'],
                                       **{k: scores[k] for k in ('raw_std', 'white_std', 'raw_status', 'z_status')}})
                    for j, i in enumerate(indices):
                        leaves[i] = {'memory_id': ms[i]['id'], 'leaf_index': i, 'group_id': g['master_id'],
                                     **{k: float(scores[k][j]) for k in ('raw', 'white', 'z', 'correction', 'weighted')}}
                    group_checks += 1
                    leaf_checks += len(indices)
                order = sorted(range(len(ms)), key=lambda i: -leaves[i]['weighted'])
                conditions.append({'name': name, 'groups': group_rows, 'leaves': leaves,
                                   'top5_memory_ids': [ms[i]['id'] for i in order[:5]],
                                   'top5_scores': [leaves[i]['weighted'] for i in order[:5]],
                                   'full_rank_memory_ids': [ms[i]['id'] for i in order]})
            results.append({'question_id': question['id'], 'task': domain, 'conditions': conditions})
        encoding.append({'domain': domain, 'leaf': einfo, 'query': qinfo, 'candidate': cinfo})
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'completed_questions': len(results),
                          'raw_rankings_verified': rank_checks, 'masters': len(graph['masters'])}), flush=True)
    assert len(results) == 371 and rank_checks == 742
    api.write_lines(output / 'retrieval_results.jsonl', results)
    sources = dict(proof['source_sha256'])
    for p in (Path(__file__), Path(original.__file__), ROOT / 'scripts/memory/memeye_backbone_long_text.py',
              ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py'):
        sources[str(p)] = sha(p)
    for filename, expected in sources.items():
        assert sha(filename) == expected, filename
    p = {'status': 'complete', 'memory_path': str(source), 'caption_model': api.MODEL, 'device': args.device,
         'lambda': .001, 'alpha': 1., 'questions': 371, 'memory_leaves': 438,
         'unweighted_full_rankings_equal': rank_checks, 'max_raw_score_difference': max_raw_error,
         'independent_inverse_ridge_groups': group_checks, 'group_leaf_scores_checked': leaf_checks,
         'max_inverse_ridge_difference': max_inverse_error, 'master_counts': master_counts, 'encoding': encoding,
         'vectors_persisted': False, 'memory_modified': False, 'old_images_reencoded': False,
         'gold_loaded_for_ranking': False, 'source_sha256': sources,
         'output_sha256': {str(output / n): sha(output / n) for n in ('retrieval_results.jsonl', 'source_validation.json')}}
    dump(marker, p)
    return p


def annotations(source, retrieval):
    result = {}
    for name in NAMES:
        master = name == NAMES[1]
        result[name] = {
            '1. 候选记忆结构': {'memory_path': str(source), 'offline_method': 'caption_visual_master' if master else 'natural',
                           'caption_model': api.MODEL, 'memory_leaves': 438,
                           'text': '原始对话＋此前 GPT-5.4 完整 caption',
                           'weighting_group': '每个已有 master 的全部叶子' if master else '同 domain 全部记忆为一个簇'},
            '2. 查询内容': '原问题＋已有 GPT-5.4 题图 caption' + ('＋该 master 视觉匹配选中的完整候选答案陈述' if master else '；不添加选项陈述'),
            '3. 检索方式': {'method': 'multi_path' if master else 'full_scan', 'top_k': 5, 'lambda': .001, 'alpha': 1.,
                          'score': 'raw + alpha * std_pop(raw in group) * tanh(white_z)',
                          'scope': '覆盖同 domain 全部叶子；按最终分数全局 Top-5；同分保留原顺序'},
            '4. 编码方式': {'BGE': 'BAAI/bge-small-en-v1.5, 384 dimensions, L2 normalized',
                          'query_prefix': 'Represent this sentence for searching relevant passages: ',
                          'CLIP': '沿用既有 master 视觉聚合量匹配完整陈述' if master else '不用于该条件的排序',
                          'device': retrieval['device'], 'vectors': '现场编码、用完即弃、不写回记忆',
                          'length_policy': '沿用 BGE 512-token 上限；仅 Card 一条记忆编码截断，reader 保留全文'},
            '5. 回答方式': {'model': api.MODEL, 'caption_model': api.MODEL, 'reasoning_effort': 'none',
                          'temperature': 0, 'max_completion_tokens': 16, 'seed': None,
                          'input': 'Top-5 原始对话＋GPT-5.4 完整 caption、问题、选项、原始题图；不传检索候选陈述',
                          'reader_called': False, 'correct': None}}
    return result


def prepare(output, source, env_file, workers):
    output, source = Path(output).resolve(), Path(source).resolve()
    if (output / 'protocol.json').exists():
        p = api.verify(output)
        assert p['contract']['memory_path'] == str(source)
        return p
    assert not (output / 'fresh_responses.jsonl').exists()
    ret = read(output / 'retrieval_protocol.json')
    for filename, expected in {**ret['source_sha256'], **ret['output_sha256']}.items():
        assert sha(filename) == expected, filename
    assert ret['memory_path'] == str(source)
    from scripts.evaluation.openai_gpt5_chat_transport import load_config
    config = load_config(env_file)
    questions = api.indexed(lines(source / 'inputs/source_questions.jsonl'), 'id')
    gold = api.indexed(lines(source / 'inputs/evaluation.jsonl'), 'id')
    memories = {m['id']: m['retrieval_text'] for m in lines(source / 'offline/memories.jsonl')}
    system = read(source / 'inputs/protocol.json')['reader_system_prompt']
    requests, evaluation = [], []
    for index, row in enumerate(lines(output / 'retrieval_results.jsonl')):
        q = questions[row['question_id']]
        evaluation.append({**{k: gold[q['id']][k] for k in ('id', 'gold_choice', 'gold_ids')}, 'task': q['task']})
        pair = []
        for condition in row['conditions']:
            ids = condition['top5_memory_ids']
            text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
                    + '\n\n'.join(f'Memory {i}: {memories[mid]}' for i, mid in enumerate(ids, 1))
                    + '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
                    + '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD') + '\n\nAnswer with one letter only.')
            payload = {'system': system, 'user_text': text, 'query_images': q['query_images']}
            pair.append({'record_id': q['id'] + '::' + condition['name'], 'id': q['id'], 'task': q['task'],
                         'condition': condition['name'], 'reader_text': text, 'query_images': q['query_images'],
                         'query_image_paths': [im['path'] for im in q['query_images']],
                         'top_memory_ids': ids, 'payload_sha256': digest(payload)})
        requests.extend(pair if index % 2 == 0 else pair[::-1])
    (output / 'inputs').mkdir(exist_ok=True)
    api.write_lines(output / 'inputs/requests.jsonl', requests)
    api.write_lines(output / 'inputs/gold.jsonl', evaluation)
    sources = {**ret['source_sha256'], **ret['output_sha256']}
    paths = [output / 'retrieval_protocol.json', output / 'inputs/requests.jsonl', output / 'inputs/gold.jsonl',
             Path(__file__), ROOT / 'scripts/analysis/audit_memeye_gpt54_caption_zscore_lambda0p001.py',
             Path(api.__file__), ROOT / 'scripts/evaluation/openai_gpt5_chat_transport.py',
             ROOT / 'scripts/evaluation/openai_chat_transport.py']
    paths.extend(Path(im['path']) for q in questions.values() for im in q['query_images'])
    from scripts.analysis.compare_memeye_gpt54_caption_lambda0p001 import PRIOR, PRIORS, REFERENCE_FILES
    paths.extend(reference / name for reference in PRIORS.values() for name in REFERENCE_FILES)
    paths.append(ROOT / 'scripts/analysis/compare_memeye_gpt54_caption_lambda0p001.py')
    sources.update({str(p): sha(p) for p in paths})
    contract = {'dataset': 'MemEye', 'questions': 371, 'domains': 8, 'logical_records': 742,
                'unique_payloads': len({r['payload_sha256'] for r in requests}), 'conditions': list(NAMES),
                'memory_path': str(source), 'caption_model': api.MODEL, 'model': config.model,
                'lambda': .001, 'alpha': 1., 'top_k': 5, 'base_url': config.base_url,
                'api_config': config.public_dict(), 'api_env_file': str(Path(env_file).resolve()),
                'temperature': 0, 'seed': None, 'reasoning_effort': 'none', 'max_completion_tokens': 16,
                'image_detail': 'high', 'workers': workers, 'timeout_seconds': 60, 'max_attempts': 3,
                'reader_system_prompt': system, 'source_sha256': sources, 'method_annotations': annotations(source, ret),
                'prior_unweighted_correct': {RAW[0]: 236, RAW[1]: 241},
                'prior_lambda_reference': str(PRIOR),
                'lambda_reference_paths': {key: str(value) for key, value in PRIORS.items()},
                'changed_component': 'Only online z-score weighting; reuse exact prior GPT-5.4 captions, memory masters, queries, encoder settings and answer settings.',
                'gold_in_api_payload': False, 'historical_api_answer_reuse': False,
                'caption_reuse': 'Exact GPT-5.4 caption outputs from the 236/371 source run; no Qwen captions.',
                'interpretation': 'Fixed-option MCQ accuracy on all 371 questions; historical unweighted reference, not a fresh unweighted control. Cross-run API determinism is not guaranteed.'}
    wire = [{'record_id': r['record_id'], 'payload_sha256': r['payload_sha256'],
             'request_sha256': digest(api.request_body({'contract': contract}, r)),
             'query_image_count': len(r['query_images'])} for r in requests]
    contract['request_manifest_sha256'] = digest(wire)
    api.write_lines(output / 'api_request_manifest.jsonl', wire)
    p = {'run_id': str(uuid.uuid4()), 'experiment_signature': digest(contract),
         'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'contract': contract}
    dump(output / 'protocol.json', p)
    api.verify(output)
    print(json.dumps({'stage': 'prepared', 'records': len(requests), 'unique_requests': contract['unique_payloads'],
                      'caption_model': api.MODEL, 'reader_model': api.MODEL, 'unweighted_reference': [236, 241]}), flush=True)
    return p


def analyze(output):
    output = Path(output)
    p = api.verify(output)
    c = p['contract']
    source = Path(c['memory_path'])
    validation = read(output / 'validation_output.json')
    assert validation['passed']
    for n in ('responses', 'fresh_responses'):
        assert sha(output / (n + '.jsonl')) == validation[n + '_sha256']
    requests = api.indexed(lines(output / 'inputs/requests.jsonl'), 'record_id')
    answers = api.indexed(lines(output / 'responses.jsonl'), 'record_id')
    gold = api.indexed(lines(output / 'inputs/gold.jsonl'), 'id')
    assert set(requests) == set(answers) and len(answers) == 742
    old_requests = api.indexed(lines(source / 'reader/requests.jsonl'), 'record_id')
    old_answers = api.indexed(lines(source / 'reader/fresh_responses.jsonl'), 'request_sha256')
    pairs = []
    for qid, g in gold.items():
        row = {'question_id': qid, 'domain': g['task'], 'gold_choice': g['gold_choice']}
        for name, raw in zip(NAMES, RAW):
            rid, oldrid = qid + '::' + name, qid + '::' + raw
            pred = api.parse_choice(answers[rid]['raw_response'])
            oldpred = api.parse_choice(old_answers[old_requests[oldrid]['request_sha256']]['raw_response'])
            ids = requests[rid]['top_memory_ids']
            row.update({name + '_answer': pred, name + '_correct': pred == g['gold_choice'],
                        name + '_top5': ids, name + '_gold_hits': len(set(ids) & set(g['gold_ids'])),
                        raw + '_historical_answer': oldpred, raw + '_historical_correct': oldpred == g['gold_choice'],
                        raw + '_identical_historical_payload': requests[rid]['payload_sha256'] == old_requests[oldrid]['payload_sha256']})
        pairs.append(row)
    # The existing aggregate only consumes these correctness booleans.
    def aggregate(rows):
        return api.aggregate([{'baseline_correct': r[NAMES[0] + '_correct'], 'expanded_correct': r[NAMES[1] + '_correct']} for r in rows])
    comparison, recall = {}, {}
    for name, raw in zip(NAMES, RAW):
        before = sum(r[raw + '_historical_correct'] for r in pairs)
        after = sum(r[name + '_correct'] for r in pairs)
        assert before == c['prior_unweighted_correct'][raw]
        comparison[name] = {'historical_unweighted_correct': before, 'current_weighted_correct': after,
                            'net_correct': after - before,
                            'wrong_to_right': sum(not r[raw + '_historical_correct'] and r[name + '_correct'] for r in pairs),
                            'right_to_wrong': sum(r[raw + '_historical_correct'] and not r[name + '_correct'] for r in pairs),
                            'identical_payload_questions': sum(r[raw + '_identical_historical_payload'] for r in pairs),
                            'identical_payload_answer_disagreements': [r['question_id'] for r in pairs if r[raw + '_identical_historical_payload'] and r[raw + '_historical_answer'] != r[name + '_answer']]}
        recall[name] = {'gold_hits_top5': sum(r[name + '_gold_hits'] for r in pairs), 'gold_references': 1041,
                        'questions_with_any_gold': sum(r[name + '_gold_hits'] > 0 for r in pairs)}
    methods = copy.deepcopy(c['method_annotations'])
    for name in NAMES:
        methods[name]['5. 回答方式'].update(reader_called=True, correct=comparison[name]['current_weighted_correct'])
    fresh = lines(output / 'fresh_responses.jsonl')
    summary = {'status': 'complete', 'caption_model': c['caption_model'], 'reader_model': c['model'],
               'memory_path': str(source), 'lambda': .001, 'alpha': 1., 'overall': aggregate(pairs),
               'by_domain': {d: aggregate([r for r in pairs if r['domain'] == d]) for d in sorted({r['domain'] for r in pairs})},
               'historical_comparisons': comparison, 'retrieval_stats': recall, 'method_annotations': methods,
               'validation': validation, 'interpretation': c['interpretation'],
               'usage': {k: sum((r.get('usage') or {}).get(k, 0) or 0 for r in fresh) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
    dump(output / 'summary.json', summary)
    dump(output / 'paired_results.json', pairs)
    with (output / 'paired_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT-5.4 caption + z-score</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
            '<h1>GPT-5.4 caption＋GPT-5.4 回答：λ=0.001、α=1</h1>',
            '<p>同此前 baseline 236/371、master 241/371 的完整流程：复用该轮 GPT caption 和记忆；重新编码检索并加入 z-score。本轮未重新生成 caption，全部答案重新调用 GPT-5.4。未加权的全部 742 个完整检索排序已核对一致。</p>',
            '<table><tr><th>方案</th><th>历史未加权</th><th>本轮加权</th><th>准确率</th><th>正确数变化</th></tr>']
    for name in NAMES:
        v = comparison[name]
        body.append(f'<tr><td>{name}</td><td>{v["historical_unweighted_correct"]}/371</td><td>{v["current_weighted_correct"]}/371</td><td>{v["current_weighted_correct"]/371:.4%}</td><td>{v["net_correct"]:+d}</td></tr>')
    body += ['</table><p>历史未加权对照未在本轮重跑。API 输出可能跨轮波动，差异不应全部归因于赋权；相同输入的跨轮答案变化单独记录。</p>',
             '<pre>' + html.escape(json.dumps(comparison, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>核验与方法标注</h2><p>固定选项排列的 371 题 MCQ accuracy。GPT-5.4-2026-03-05、reasoning=none、temperature=0、16-token 上限；标签不进入 API 请求。</p>',
             '<pre>' + html.escape(json.dumps(validation, ensure_ascii=False, indent=2)) + '</pre>']
    for name in NAMES:
        body.append('<details><summary>' + name + '</summary><pre>' + html.escape(json.dumps(methods[name], ensure_ascii=False, indent=2)) + '</pre></details>')
    body.append('<p><a href="summary.json">完整统计</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a> · <a href="retrieval_protocol.json">检索一致性</a></p></html>')
    (output / 'report.html').write_text('\n'.join(body))
    print(json.dumps({'stage': 'analyzed', 'overall': summary['overall'], 'comparisons': comparison}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('retrieve', 'prepare', 'smoke', 'run', 'analyze', 'all'), default='all')
    p.add_argument('--memory', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=BASE)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--api-env-file', type=Path, default=api.DEFAULT_ENV)
    p.add_argument('--api-workers', type=int, default=3)
    args = p.parse_args()
    assert args.api_workers > 0
    if args.stage in ('retrieve', 'all'):
        retrieve(args.output, args.memory, args.device)
    if args.stage in ('prepare', 'all'):
        prepare(args.output, args.memory, args.api_env_file, args.api_workers)
    from scripts.analysis.audit_memeye_gpt54_caption_zscore_lambda0p001 import audit
    from scripts.analysis.compare_memeye_gpt54_caption_lambda0p001 import compare
    if args.stage in ('prepare', 'smoke', 'run', 'all'):
        audit(args.output, inputs_only=True)
        compare(args.output, inputs_only=True)
    if args.stage in ('smoke', 'all'):
        api.run(args.output, smoke=True)
    if args.stage in ('run', 'all'):
        api.run(args.output)
    if args.stage in ('analyze', 'all'):
        analyze(args.output)
        audit(args.output)
        compare(args.output)


if __name__ == '__main__':
    main()
