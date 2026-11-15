"""Seed42 DeepSeek: freshly regenerated captions/graphs and both fresh readers.

Baseline question-only is unchanged; only the master path adds online z-score.
Caption and answer HTTP bodies explicitly contain seed=42. Earlier unseeded
artifacts are used only for post-hoc comparison, never as experiment inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import os
from pathlib import Path
import re
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
from scripts.evaluation import deepseek_flash_seed42_transport as transport
from scripts.evaluation.memeye_gpt5_reader import read, lines, sha, digest, dump, append, write_lines, indexed
from scripts.analysis.deepseek_seed42_audit_helpers import wire_request, validate_response
from scripts.memory.online_zscore import ZScoreWeighting

SOURCE = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
OUTPUT = SOURCE / 'weighted_evaluation'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
MODEL = 'deepseek-v4.1-flash'
BASELINE = 'baseline_question_only'
WEIGHTED = 'master_answer_query_zscore'
CONDITIONS = (BASELINE, WEIGHTED)
RIDGE, ALPHA = .03, 1.


def checked_hashes(hashes):
    for path, expected in hashes.items():
        assert sha(path) == expected, path


def freeze(path, value):
    if path.exists():
        assert read(path) == value, f'Frozen artifact changed: {path}'
    else:
        dump(path, value)


def choice(text):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', text.strip(), re.I)
    return match.group(1).upper() if match else None


def verify_source(source=SOURCE):
    """Verify every seed42 caption and the graph rebuilt from this fresh run."""
    from scripts.evaluation import memeye_deepseek_flash_seed42 as fresh
    source = Path(source).resolve()
    p = fresh.verify(source); c = p['contract']
    captions = fresh.verify_captions(source)
    ret = fresh.verify_retrieval(source)
    requests = indexed(lines(source/'caption_generation/requests.jsonl'), 'record_id')
    answers = indexed(lines(source/'caption_generation/fresh_responses.jsonl'), 'request_sha256')
    memories = lines(source/'inputs/source_memories.jsonl')
    questions = lines(source/'inputs/source_questions.jsonl')
    assert c['seed'] == 42 and c['caption_seed'] == c['reader_seed'] == 42
    needed = {m['image_sha256'] for m in memories} | {im['sha256'] for q in questions for im in q['query_images']}
    assert set(captions) == needed and len(captions) == len(answers) == 437
    for cap in captions.values():
        req = requests[cap['source_response_record_id']]; ans = answers[cap['request_sha256']]
        payload = {k:req[k] for k in ('system','user_text','query_images') if k in req}
        body = wire_request(payload, MODEL, 1024, 'high')
        assert body['seed'] == cap['seed'] == 42 and digest(body) == req['request_sha256']
        assert digest(payload) == req['payload_sha256']
        validate_response(ans, req, p, 'caption_generation')
        assert cap['caption'] == ans['raw_response'].strip() and cap['model'] == MODEL
    assert len(memories) == 438 and len(questions) == 371
    counts = {}
    for domain in sorted({m['task'] for m in memories}):
        graph = read(source/'domains'/domain/'memory/graph.json')
        by_image = {m['image_id']:m for m in memories if m['task'] == domain}
        assert len(graph['image_nodes']) == len(by_image)
        for node in graph['image_nodes']:
            m=by_image[node['id']]
            assert node['caption'] == captions[m['image_sha256']]['caption'] and node['dialogue'] == m['dialogue']
        counts[domain] = len(graph['masters'])
    paths = [source/n for n in ('protocol.json','captions.jsonl','retrieval_protocol.json',
        'caption_generation/requests.jsonl','caption_generation/fresh_responses.jsonl','caption_generation/protocol.json','seed42_input_audit.json')]
    hashes = {**c['source_sha256'], **ret['source_sha256'], **ret['output_sha256'], **{str(p):sha(p) for p in paths}}
    return {'passed':True,'caption_model':MODEL,'caption_seed':42,'caption_images':437,'memory_leaves':438,
        'questions':371,'master_counts':counts,'masters':sum(counts.values()),
        'caption_requests_reconstructed':437,'caption_model_consistent':True,
        'captions_regenerated':True,'graph_rebuilt_this_run':True,'historical_caption_or_graph_reuse':False,
        'source_sha256':hashes}


def checked_weighting(query, vectors):
    result = ZScoreWeighting(RIDGE, ALPHA).apply(query, vectors)
    center = vectors - vectors.mean(axis=0)
    check = vectors @ query if len(vectors) == 1 else vectors @ (
        query - center.T @ np.linalg.solve(center @ center.T + (len(vectors)-1)*RIDGE*np.eye(len(vectors)), center @ query))
    np.testing.assert_allclose(result['white'], check, rtol=0, atol=2e-12)
    # Independent standardization and final correction, including degeneracy.
    std = float(check.std())
    z = (check - check.mean()) / std if std > 1e-12 and len(check) > 1 else np.zeros_like(check)
    raw = vectors @ query
    correction = ALPHA * raw.std() * np.tanh(z) if len(raw) > 1 and raw.std() > 1e-12 else np.zeros_like(raw)
    np.testing.assert_allclose(result['weighted'], raw + correction, rtol=0, atol=2e-10)
    return result, float(np.max(np.abs(result['white'] - check)))


def retrieve(output=OUTPUT, source=SOURCE, device='cuda:1'):
    output, source = Path(output).resolve(), Path(source).resolve()
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        p = read(marker); checked_hashes(p['source_sha256']); checked_hashes(p['output_sha256'])
        assert p['memory_path'] == str(source) and p['lambda'] == RIDGE and p['alpha'] == ALPHA
        return p
    assert not (output / 'reader/protocol.json').exists()
    proof = verify_source(source)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    inp = read(source / 'inputs/protocol.json')
    memories, questions, model = original.attach_captions(lines(source / 'inputs/source_memories.jsonl'),
        lines(source / 'inputs/source_questions.jsonl'), lines(source / 'captions.jsonl'))
    assert model == MODEL
    previous = indexed(lines(source / 'retrieval_results.jsonl'), 'question_id')
    args = SimpleNamespace(device=device); original.configure_device(args)
    bge = original.load_text_encoder(inp['text_encoder'], args.device)
    clip, tokenizer, limit = original.load_clip_text_encoder(inp['image_encoder'], args.device)
    output.mkdir(parents=True, exist_ok=True); dump(output / 'source_validation.json', proof)
    results, encoding = [], []
    groups_checked = ranks_checked = 0; max_raw_error = max_inverse_error = 0.
    for domain in sorted({m['task'] for m in memories}):
        ms = [m for m in memories if m['task'] == domain]; qs = [q for q in questions if q['task'] == domain]
        graph = read(source / 'domains' / domain / 'memory/graph.json')
        sums = np.load(source / 'domains' / domain / 'memory/master_sums.npy', allow_pickle=False)
        e, einfo = encode_complete(bge, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='dialogue+caption')
        texts, text_ids, statements = [], [], []
        for q in qs:
            candidates = [q['candidate_statements'][k] for k in 'ABCD']
            statements.extend(candidates)
            texts.extend([q['retrieval_query']] + [q['retrieval_query'] + '\nCandidate answer: ' + s for s in candidates])
            text_ids.extend(q['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D'))
        qv, qinfo = encode_complete(bge, texts, inp['query_prefix'], ids=text_ids, field='query')
        cv, cinfo = original._clip_candidates(clip, tokenizer, limit, statements)
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        for qi, q in enumerate(qs):
            raw = original.rank_question(q, ms, graph, sums, qv[qi], e, cv[qi]); old = previous[q['id']]
            for field in ('question', 'options', 'query_images', 'query_image_captions', 'baseline_retrieval_query', 'candidate_statements'):
                assert raw[field] == old[field]
            assert len(raw['masters']) == len(old['masters'])
            for a, b in zip(raw['masters'], old['masters']):
                for key in ('master_id', 'member_ids', 'selected_choice', 'selected_statement', 'augmented_query'):
                    assert a[key] == b[key], (q['id'], key)
                np.testing.assert_allclose(list(a['clip_candidate_scores'].values()), list(b['clip_candidate_scores'].values()), atol=2e-6, rtol=0)
            for a, b in zip(raw['conditions'], old['conditions']):
                assert a['name'] == b['name'] and a['full_rank_memory_ids'] == b['full_rank_memory_ids']
                assert a['reader_context'] == b['reader_context']; ranks_checked += 1
            for field in ('baseline_score', 'expanded_score'):
                err = float(np.max(np.abs(np.array([l[field] for l in raw['leaves']]) - [l[field] for l in old['leaves']])))
                assert err < 2e-6; max_raw_error = max(max_raw_error, err)
            leaves = [None] * len(ms); groups = []
            for g in raw['masters']:
                indices = [i for i, l in enumerate(raw['leaves']) if l['master_id'] == g['master_id']]
                query_index = 1 + 'ABCD'.index(g['selected_choice'])
                scores, err = checked_weighting(qv[qi, query_index], e[indices])
                max_inverse_error = max(max_inverse_error, err)
                np.testing.assert_allclose(scores['raw'], [raw['leaves'][i]['expanded_score'] for i in indices], atol=2e-15, rtol=0)
                groups.append({'group_id': g['master_id'], 'size': len(indices), 'query': g['augmented_query'],
                    'selected_choice': g['selected_choice'], 'selected_statement': g['selected_statement'],
                    **{k: scores[k] for k in ('raw_std', 'white_std', 'raw_status', 'z_status')}})
                for j, i in enumerate(indices):
                    assert leaves[i] is None
                    leaves[i] = {'memory_id': ms[i]['id'], 'leaf_index': i, 'group_id': g['master_id'],
                                 **{k: float(scores[k][j]) for k in ('raw', 'white', 'z', 'correction', 'weighted')}}
                groups_checked += 1
            assert all(l is not None for l in leaves)
            order = sorted(range(len(ms)), key=lambda i: -leaves[i]['weighted'])
            baseline = next(c for c in raw['conditions'] if c['name'] == BASELINE)
            results.append({'question_id': q['id'], 'task': domain, 'conditions': [
                {k: baseline[k] for k in ('name', 'top5_memory_ids', 'top5_scores', 'full_rank_memory_ids')},
                {'name': WEIGHTED, 'groups': groups, 'leaves': leaves,
                 'top5_memory_ids': [ms[i]['id'] for i in order[:5]], 'top5_scores': [leaves[i]['weighted'] for i in order[:5]],
                 'full_rank_memory_ids': [ms[i]['id'] for i in order]}]})
        encoding.append({'domain': domain, 'leaf': einfo, 'query': qinfo, 'candidate': cinfo})
        print(json.dumps({'stage': 'retrieve', 'domain': domain, 'questions_done': len(results)}), flush=True)
    assert len(results) == 371 and ranks_checked == 742
    write_lines(output / 'retrieval_results.jsonl', results)
    source_hashes = dict(proof['source_sha256'])
    for module in (Path(__file__), Path(original.__file__), ROOT / 'scripts/memory/memeye_backbone_long_text.py',
                   ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py'):
        source_hashes[str(module)] = sha(module)
    checked_hashes(source_hashes)
    result = {'status': 'complete', 'memory_path': str(source), 'caption_model': MODEL, 'lambda': RIDGE, 'alpha': ALPHA,
              'conditions': list(CONDITIONS), 'questions': 371, 'memory_leaves': 438, 'masters': proof['masters'],
              'unweighted_rankings_reproduced': ranks_checked, 'independent_inverse_ridge_groups': groups_checked,
              'max_raw_score_difference': max_raw_error, 'max_inverse_ridge_difference': max_inverse_error,
              'encoding': encoding, 'device': args.device, 'score_formula': 'raw + alpha * std_pop(raw) * tanh(z_white)',
              'caption_regenerated': True, 'graph_rebuilt_this_run': True, 'graph_previously_built_from_same_model_captions': True,
              'memory_modified': False, 'vectors_persisted': False, 'old_images_reencoded': False, 'gold_loaded_for_ranking': False,
              'source_sha256': source_hashes, 'output_sha256': {str(output/n): sha(output/n) for n in ('source_validation.json', 'retrieval_results.jsonl')}}
    dump(marker, result)
    return result


def annotations(source, system):
    result = {}
    for name in CONDITIONS:
        weighted = name == WEIGHTED
        result[name] = {
            '1. 候选记忆结构': {'memory_path': str(source), 'caption_model': MODEL, 'memory_leaves': 438,
                'offline_method': 'caption_visual_master' if weighted else 'natural',
                'candidate_scope': '当前 domain 的全部图片记忆', 'master_count': sum(len(read(source/'domains'/d/'memory/graph.json')['masters']) for d in sorted({m['task'] for m in lines(source/'inputs/source_memories.jsonl')})) if weighted else None,
                'leaf_text': 'dialogue + "\\nMemory description: " + full caption'},
            '2. 查询内容': {'description': '问题＋已有题图 caption' + ('＋本 master 视觉匹配选择的完整答案陈述' if weighted else ''),
                'candidate_statements_used': weighted, 'gold_used': False},
            '3. 检索方式': {'retrieval': 'multi_path' if weighted else 'full_scan', 'top_k': 5,
                'online_weighting': 'z_score' if weighted else 'none', 'lambda': RIDGE if weighted else None,
                'alpha': ALPHA if weighted else None, 'group': '每个 master 全部叶子' if weighted else '当前 domain 全量记忆',
                'final_score': 'raw + alpha * std_pop(raw) * tanh(z_white)' if weighted else 'BGE cosine',
                'tie_break': '保留源记忆顺序'},
            '4. 编码方式': {'model': 'BAAI/bge-small-en-v1.5', 'dimension': 384, 'L2_normalized': True,
                'query_prefix': 'Represent this sentence for searching relevant passages: ',
                'max_length': 512, 'long_text': '保留历史 BGE 截断口径，完整文本送回答模型',
                'clip_role': '选择完整候选陈述，视觉向量来自已有 master；不与 BGE 分数融合' if weighted else None,
                'vectors_persisted': False},
            '5. 回答方式': {'model': MODEL, 'provider': 'DashScope International', 'temperature': 0,
                'enable_thinking': False, 'max_tokens': 16, 'seed': 42,
                'system_prompt': system, 'evidence': 'Top5 对话＋完整 caption，问题、选项及原始题图',
                'historical_answers_reused': False, 'identical_full_requests_share_one_fresh_answer': True}}
    return result


def verify_reader(output):
    folder = Path(output) / 'reader'; p = read(folder / 'protocol.json')
    assert digest(p['contract']) == p['experiment_signature']
    c = p['contract']; assert c['model'] == MODEL and c['lambda'] == RIDGE and c['alpha'] == ALPHA
    assert c['conditions'] == list(CONDITIONS) and c['seed'] == 42
    checked_hashes(c['source_sha256'])
    records = lines(folder / 'requests.jsonl'); assert len(records) == len(indexed(records, 'record_id')) == 742
    assert len({r['request_sha256'] for r in records}) == c['unique_requests']
    for row in records:
        payload = {k: row[k] for k in ('system', 'user_text', 'query_images')}
        wire = transport.build_request(payload, model=MODEL, max_output_tokens=16, image_detail='high')
        assert wire == wire_request(payload, MODEL, 16, 'high')
        assert digest(wire) == row['request_sha256'] and digest(payload) == row['payload_sha256']
    return p, records


def prepare(output=OUTPUT, source=SOURCE, workers=3, env_file=ENV):
    output, source = Path(output).resolve(), Path(source).resolve(); folder = output / 'reader'
    if (folder / 'protocol.json').exists():
        return verify_reader(output)[0]
    ret = read(output / 'retrieval_protocol.json'); checked_hashes(ret['source_sha256']); checked_hashes(ret['output_sha256'])
    assert ret['memory_path'] == str(source)
    config = transport.load_config(env_file)
    inp = read(source / 'inputs/protocol.json'); system = inp['reader_system_prompt']
    questions = indexed(lines(source / 'inputs/source_questions.jsonl'), 'id')
    gold = lines(source / 'inputs/evaluation.jsonl')
    memory = {m['id']: m['retrieval_text'] for m in lines(source / 'offline/memories.jsonl')}
    old = indexed(lines(source / 'retrieval_results.jsonl'), 'question_id')
    records = []; matches = 0
    for qi, result in enumerate(lines(output / 'retrieval_results.jsonl')):
        q = questions[result['question_id']]; batch = []
        for condition in result['conditions']:
            name = condition['name']; ids = condition['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5 and all(mid.split('/',1)[0] == q['task'] for mid in ids)
            context = '\n\n'.join(f'Memory {i}: {memory[mid]}' for i, mid in enumerate(ids, 1))
            text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context +
                    '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n' +
                    '\n'.join(f'{k}. {q["options"][k]}' for k in 'ABCD') + '\n\nAnswer with one letter only.')
            payload = {'system': system, 'user_text': text, 'query_images': q['query_images']}
            wire = transport.build_request(payload, model=MODEL, max_output_tokens=16, image_detail='high')
            assert wire == wire_request(payload, MODEL, 16, 'high')
            row = {'record_id': q['id']+'::'+name, 'question_id': q['id'], 'task': q['task'], 'condition': name,
                   'top_memory_ids': ids, **payload, 'payload_sha256': digest(payload), 'request_sha256': digest(wire)}
            if name == BASELINE:
                prev = next(r for r in old[q['id']]['conditions'] if r['name'] == BASELINE)
                assert ids == prev['top5_memory_ids'] and context == prev['reader_context']
                matches += 1
            batch.append(row)
        records.extend(batch if qi % 2 == 0 else reversed(batch))
    assert len(records) == 742 and matches == len(gold) == 371
    folder.mkdir(parents=True, exist_ok=True)
    write_lines(folder / 'requests.jsonl', records); write_lines(folder / 'gold.jsonl', gold)
    source_hashes = {**ret['source_sha256'], **ret['output_sha256']}
    for path in (output/'retrieval_protocol.json', folder/'requests.jsonl', folder/'gold.jsonl',
                 Path(__file__), Path(transport.__file__), ROOT/'scripts/analysis/audit_memeye_deepseek_flash_alias.py', ROOT/'scripts/analysis/deepseek_seed42_audit_helpers.py', ROOT/'scripts/analysis/audit_memeye_deepseek_flash_seed42_zscore.py',
                 ROOT/'scripts/evaluation/openai_chat_transport.py', ROOT/'scripts/evaluation/memeye_gpt5_reader.py'):
        source_hashes[str(path)] = sha(path)
    contract = {'model': MODEL, 'provider': 'DashScope International', 'api_config': config.public_dict(),
                'api_env_file': str(Path(env_file).resolve()), 'memory_path': str(source),
                'conditions': list(CONDITIONS), 'lambda': RIDGE, 'alpha': ALPHA, 'temperature': 0,
                'seed': 42, 'enable_thinking': False, 'reader_max_output_tokens': 16, 'image_detail': 'high',
                'workers': workers, 'max_attempts': 3, 'timeout_seconds': 60,
                'questions': 371, 'logical_records': 742, 'unique_requests': len({r['request_sha256'] for r in records}),
                'history_caption_regenerated_seed42': True, 'historical_answers_reused': False,
                'gold_sent_to_api': False, 'same_run_raw_baseline_context_matches': matches,
                'method_annotations': annotations(source, system), 'source_sha256': source_hashes}
    p = {'run_id': str(uuid.uuid4()), 'experiment_signature': digest(contract), 'contract': contract,
         'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump(folder / 'protocol.json', p); verify_reader(output)
    dump(output / 'independent_input_audit.json', {'passed': True, 'records': 742, 'questions': 371,
         'unique_requests': contract['unique_requests'], 'request_bodies_independently_reconstructed': 742,
         'same_run_raw_baseline_exact_context_matches': matches, 'gold_separate': True,
         'source_caption_and_memory_audit': str(output/'source_validation.json'),
         'independent_inverse_ridge_groups': ret['independent_inverse_ridge_groups'],
         'max_inverse_ridge_difference': ret['max_inverse_ridge_difference']})
    print(json.dumps({'stage': 'prepared', 'unique_requests': contract['unique_requests'], 'records': 742}), flush=True)
    return p


def run(output=OUTPUT, limit=0):
    output = Path(output); p, records = verify_reader(output); c = p['contract']; folder = output/'reader'
    config = transport.load_config(c['api_env_file']); assert config.public_dict() == c['api_config']
    representatives = {}
    for row in records:
        representatives.setdefault(row['request_sha256'], row)
    completed = lines(folder/'fresh_responses.jsonl') if (folder/'fresh_responses.jsonl').exists() else []
    fresh = indexed(completed, 'request_sha256')
    for h, ans in fresh.items():
        validate_response(ans, representatives[h], p, 'reader')
    pending = [r for h,r in representatives.items() if h not in fresh]
    if limit:
        pending = pending[:limit]
    execution_id = str(uuid.uuid4())

    def execute(row):
        payload = {k: row[k] for k in ('system','user_text','query_images')}
        body = transport.build_request(payload, model=MODEL, max_output_tokens=16, image_detail='high')
        assert digest(body) == row['request_sha256']
        attempts = []
        for attempt in range(1, c['max_attempts']+1):
            entry = {'record_id': row['record_id'], 'request_sha256': row['request_sha256'],
                     'execution_id': execution_id, 'attempt': attempt}
            try:
                response = transport.chat_request(config, body, timeout=c['timeout_seconds'])
            except transport.APIError as error:
                attempts.append({**entry, 'status': 'api_error', 'kind': error.kind,
                                 'http_status': error.http_status, 'message': error.message})
                if not error.retryable or attempt == c['max_attempts']:
                    return None, attempts
                time.sleep(min(2**attempt, 8)); continue
            valid = (choice(response['raw_response']) is not None and response['finish_reason']=='stop'
                     and not response.get('refusal') and response['returned_model']==MODEL)
            answer = {**response, 'record_id': row['record_id'], 'question_id': row['question_id'], 'condition': row['condition'],
                      'payload_sha256': row['payload_sha256'], 'request_sha256': row['request_sha256'],
                      'run_id': p['run_id'], 'experiment_signature': p['experiment_signature'], 'execution_id': execution_id,
                      'attempts': attempt, 'retry_count': attempt-1, 'status': 'ok' if valid else 'invalid_response',
                      'predicted_choice': choice(response['raw_response']), 'historical_response_reuse': False}
            attempts.append({**entry, 'status': answer['status'], 'response_id': answer['response_id']})
            return answer, attempts

    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for start in range(0, len(pending), c['workers']):
            failed = []
            futures = {pool.submit(execute, row): row for row in pending[start:start+c['workers']]}
            for future in as_completed(futures):
                row = futures[future]; answer, attempts = future.result()
                for a in attempts: append(folder/'api_attempts.jsonl', a)
                if answer is not None:
                    append(folder/'fresh_responses.jsonl', answer)
                    if answer['status']=='ok':
                        validate_response(answer, row, p, 'reader'); fresh[row['request_sha256']] = answer
                if answer is None or answer['status']!='ok': failed.append(row['record_id'])
            dump(output/'execution_status.json', {'status': 'stopped_on_error' if failed else 'running',
                 'fresh_completed': len(fresh), 'unique_requests': len(representatives), 'failed_records': failed})
            print(json.dumps({'stage': 'reader', 'fresh_completed': len(fresh), 'unique_requests': len(representatives), 'failed': failed}), flush=True)
            if failed: raise RuntimeError('API batch failed; stopped before any further batches.')
    if len(fresh) == len(representatives):
        return analyze(output)
    return {'status': 'partial', 'fresh_completed': len(fresh)}


def analyze(output=OUTPUT):
    output = Path(output); p, records = verify_reader(output); c=p['contract'];folder=output/'reader'
    gold = indexed(lines(folder/'gold.jsonl'), 'id'); assert len(gold)==371
    fresh_rows=lines(folder/'fresh_responses.jsonl'); fresh=indexed(fresh_rows,'request_sha256')
    reps={}
    for row in records: reps.setdefault(row['request_sha256'],row)
    assert set(fresh)==set(reps) and len(fresh)==c['unique_requests']
    for h,ans in fresh.items():
        validate_response(ans,reps[h],p,'reader'); assert choice(ans['raw_response'])==ans['predicted_choice']
    previous=lines(ROOT/'results/memeye_deepseek_flash_zscore_lambda0p03_alpha1_20261025/reader/fresh_responses.jsonl')
    questions=indexed(lines(Path(c['memory_path'])/'inputs/source_questions.jsonl'),'id')
    assert not ({r['response_id'] for r in previous}&{r['response_id'] for r in fresh_rows})
    assert len({r['response_id'] for r in fresh_rows})==len(fresh_rows)
    attempt_rows=lines(folder/'api_attempts.jsonl')
    assert Counter(a['request_sha256'] for a in attempt_rows if a['status']=='ok')==dict.fromkeys(fresh,1)
    answers={}; logical=[]; per_question=[]
    for row in records:
        ans=fresh[row['request_sha256']]; q=gold[row['question_id']]; predicted=choice(ans['raw_response'])
        r={**ans,'record_id':row['record_id'],'question_id':row['question_id'],'condition':row['condition'],
           'top_memory_ids':row['top_memory_ids'],'correct':predicted==q['gold_choice'],
           'fresh_inference_executed_for_this_record':ans['record_id']==row['record_id'],
           'same_run_shared_inference_record_id':None if ans['record_id']==row['record_id'] else ans['record_id']}
        logical.append(r);answers[row['question_id'],row['condition']]=r
    write_lines(folder/'responses.jsonl',logical)
    for qid,q in gold.items():
        row={'question_id':qid,'domain':questions[qid]['task'],'gold_choice':q['gold_choice'],'gold_count':len(q['gold_ids'])}
        for name in CONDITIONS:
            ans=answers[qid,name];hits=len(set(ans['top_memory_ids'])&set(q['gold_ids']))
            row.update({name+'_answer':ans['predicted_choice'],name+'_correct':ans['correct'],name+'_gold_hits':hits,
                        name+'_top5':ans['top_memory_ids']})
        per_question.append(row)
    total=sum(r['gold_count'] for r in per_question);assert total==1041
    overall={}
    for name in CONDITIONS:
        correct=sum(r[name+'_correct'] for r in per_question);hits=sum(r[name+'_gold_hits'] for r in per_question)
        overall[name]={'correct':correct,'questions':371,'accuracy':correct/371,'gold_hits_top5':hits,'gold_references':total,
                       'recall_at_5_micro':hits/total,'recall_at_5_macro':sum(r[name+'_gold_hits']/r['gold_count'] for r in per_question)/371,
                       'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in per_question)}
    wins=sum(not r[BASELINE+'_correct'] and r[WEIGHTED+'_correct'] for r in per_question)
    losses=sum(r[BASELINE+'_correct'] and not r[WEIGHTED+'_correct'] for r in per_question)
    method_annotations=c['method_annotations']
    for name in CONDITIONS:method_annotations[name]['5. 回答方式'].update(overall[name])
    summary={'status':'complete','model':MODEL,'caption_model':MODEL,'provider':c['provider'],'lambda':RIDGE,'alpha':ALPHA,
             'caption_source':c['memory_path'],'captions_regenerated':True,'caption_seed':42,'caption_images':437,'memory_leaves':438,'masters':read(output/'source_validation.json')['masters'],
             'overall':overall,'comparison':{'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses},
             'seed':42,'fresh_requests':len(fresh),'logical_records':742,
             'same_run_shared_records':742-len(fresh),'failed_attempts':sum(r['status']!='ok' for r in attempt_rows),
             'format_errors':0,'refusals':0,'length_stops':0,'historical_answers_reused':False,
             'method_annotations':method_annotations,'generation_parameters':{k:c[k] for k in ('temperature','seed','enable_thinking','reader_max_output_tokens')},
             'usage':{k:sum((r.get('usage') or {}).get(k,0) for r in fresh_rows) for k in ('prompt_tokens','completion_tokens','total_tokens')},
             'limitations':['API model is an undated alias; the provider does not establish a fixed weight revision.',
                            'Only the requested baseline and master+zscore conditions were freshly answered; historical unweighted master is not a matched new control.']}
    cap_responses=lines(Path(c['memory_path'])/'caption_generation/fresh_responses.jsonl')
    summary['caption_fresh_requests']=len(cap_responses)
    summary['total_fresh_requests']=len(cap_responses)+len(fresh)
    summary['caption_generation_parameters']={'model':MODEL,'seed':42,'temperature':0,'enable_thinking':False,'max_tokens':1024}
    summary['master_counts']=read(output/'source_validation.json')['master_counts']
    summary['by_domain']={}
    for domain in sorted({r['domain'] for r in per_question}):
        rows=[r for r in per_question if r['domain']==domain]; domain_total=sum(r['gold_count'] for r in rows)
        summary['by_domain'][domain]={name:{'correct':sum(r[name+'_correct'] for r in rows),
            'questions':len(rows),'accuracy':sum(r[name+'_correct'] for r in rows)/len(rows),
            'gold_hits_top5':sum(r[name+'_gold_hits'] for r in rows),'gold_references':domain_total,
            'recall_at_5_micro':sum(r[name+'_gold_hits'] for r in rows)/domain_total} for name in CONDITIONS}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',per_question)
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        table=[{'condition':name,**r} for name,r in overall.items()]
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    with (output/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_question[0]));writer.writeheader();writer.writerows(per_question)
    audit={'passed':True,'caption_and_graph_source_verified':True,'questions':371,'gold_references':1041,
           'records':742,'fresh_requests':len(fresh),'same_run_shared_records':742-len(fresh),
           'full_request_hashes_reconstructed':742,'historical_answer_reuse':False,
           'response_identity_and_disabled_thinking_verified':True,'metric_recomputed_from_raw_answers_and_retrieved_ids':True,
           'artifact_sha256':{n:sha(output/n) for n in ('reader/protocol.json','reader/requests.jsonl','reader/gold.jsonl',
               'reader/fresh_responses.jsonl','reader/responses.jsonl','reader/api_attempts.jsonl','summary.json','paired_results.json',
               'retrieval_protocol.json','retrieval_results.jsonl','source_validation.json','independent_input_audit.json')}}
    dump(output/'independent_audit.json',audit)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek Flash MemEye</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
          '<h1>DeepSeek V4.1 Flash：baseline 与 master + zscore</h1>',
          '<p>DashScope International，deepseek-v4.1-flash。重新生成全部 437 张去重图片的 caption，重新构建 438 条记忆及 master。全部 371 题重新回答；未加权 baseline 对比 master_answer_query + zscore（λ=0.03，α=1）。</p>',
          '<table><tr><th>方法</th><th>QA</th><th>Recall@5 micro</th><th>Recall@5 macro</th></tr>']
    for name,result in overall.items():
        body.append(f'<tr><td>{name}</td><td>{result["correct"]}/371 ({result["accuracy"]:.4%})</td><td>{result["gold_hits_top5"]}/1041 ({result["recall_at_5_micro"]:.4%})</td><td>{result["recall_at_5_macro"]:.4%}</td></tr>')
    body+=['</table>',f'<p>本轮 {len(fresh)} 次新推理，{742-len(fresh)} 条同请求共享。caption 与回答均显式发送 seed=42；本轮没有复用历史 caption 或回答。</p>',
           '<p>temperature=0，enable_thinking=false，回答预算 16 tokens；caption 和回答均固定 seed=42。接口模型为无日期别名，实际权重版本未由服务返回确认。</p>',
           '<p><a href="summary.json">统计和五项标注</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="external_audit.json">独立核验</a> · <a href="source_validation.json">Caption 来源核验</a></p>',
           '<pre>'+html.escape(json.dumps(method_annotations,ensure_ascii=False,indent=2))+'</pre></html>']
    (output/'report.html').write_text('\n'.join(body))
    dump(output/'execution_status.json',{'status':'complete','fresh_completed':len(fresh),'unique_requests':len(reps),'logical_records':742})
    print(json.dumps({'status':'complete','overall':overall,'fresh_requests':len(fresh)}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--memory',type=Path,default=SOURCE)
    parser.add_argument('--stage',choices=['verify','retrieve','prepare','run','analyze','all'],default='all')
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--env-file',type=Path,default=ENV);parser.add_argument('--limit',type=int,default=0)
    a=parser.parse_args();assert 1<=a.workers<=8 and a.limit>=0
    if a.stage=='verify':
        proof=verify_source(a.memory);a.output.mkdir(parents=True,exist_ok=True);dump(a.output/'source_validation.json',proof)
        print(json.dumps({k:v for k,v in proof.items() if k!='source_sha256'}),flush=True)
    if a.stage in ('retrieve','all'):retrieve(a.output,a.memory,a.device)
    if a.stage in ('prepare','all'):prepare(a.output,a.memory,a.workers,a.env_file)
    if a.stage in ('run','all'):run(a.output,a.limit)
    if a.stage=='analyze':analyze(a.output)


if __name__=='__main__':main()
