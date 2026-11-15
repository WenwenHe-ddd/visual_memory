#!/usr/bin/env python3
"""Fresh seed1024 four-condition MemEye flow with frozen DeepSeek captions.

All eight caption-Ward memories are rebuilt. Original external PCA bases,
CLIP/BGE weights, chronology, candidate statements and prompts remain pinned.
No caption-generation request is made by this driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import gc
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(key, value)
import numpy as np
from scripts.memory import memeye_backbone_retrieval as native
from scripts.memory.memeye_backbone_long_text import encode_complete, POLICY
from scripts.memory.online_zscore import ZScoreWeighting

SOURCE = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
OUTPUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/ours'
OPEN_SOURCE = ROOT / 'results/memeye_deepseek_open_20261106/caption_methods'
MODEL = 'deepseek-v4.1-flash'
SEED = 1024
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
TRANSPORT = 'scripts.evaluation.memeye_deepseek_seed1024_transport'
CONDITIONS = ('baseline_question_only', 'master_answer_query',
              'baseline_question_only_zscore', 'master_answer_query_zscore')
read, lines, sha, digest, dump, write_lines = native.read, native.lines, native.sha, native.digest, native.dump, native.write_lines


def transport():
    api = importlib.import_module(TRANSPORT)
    assert api.MODEL == MODEL and api.SEED == SEED
    return api


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, f'Frozen artifact differs: {path}'
    else:
        dump(path, value)


def verify(protocol):
    assert digest(protocol['contract']) == protocol['experiment_signature']
    native.verify_hashes(protocol['contract']['source_sha256'])
    assert protocol['contract']['model'] == MODEL and protocol['contract']['seed'] == SEED


def seed_all():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def historical_retrieval():
    result = {}
    for folder in ('weighted_evaluation', 'complementary_evaluation'):
        for row in lines(SOURCE / folder / 'retrieval_results.jsonl'):
            by_condition = result.setdefault(row['question_id'], {})
            for condition in row['conditions']:
                assert condition['name'] not in by_condition
                by_condition[condition['name']] = condition
    assert len(result) == 371 and all(set(x) == set(CONDITIONS) for x in result.values())
    return result


def prepare(output=OUTPUT, device='cuda:0', workers=4):
    output = Path(output).resolve()
    if (output / 'protocol.json').exists():
        protocol = read(output / 'protocol.json')
        verify(protocol)
        assert protocol['contract']['device'] == device
        return protocol
    api = transport()
    source_protocol = read(SOURCE / 'protocol.json')
    assert digest(source_protocol['contract']) == source_protocol['experiment_signature']
    assert source_protocol['contract']['model'] == MODEL and source_protocol['contract']['seed'] == 42
    metadata = read(SOURCE / 'inputs/protocol.json')
    captions = lines(SOURCE / 'captions.jsonl')
    by_sha = native.unique(captions)
    memories, questions, model = native.attach_captions(lines(SOURCE / 'inputs/source_memories.jsonl'),
        lines(SOURCE / 'inputs/source_questions.jsonl'), captions)
    assert model == MODEL and len(captions) == 437 and len(memories) == 438 and len(questions) == 371
    assert all(c['seed'] == 42 and c['model'] == MODEL for c in captions)
    requests = native.unique(lines(SOURCE / 'caption_generation/requests.jsonl'), 'record_id')
    responses = native.unique(lines(SOURCE / 'caption_generation/fresh_responses.jsonl'), 'request_sha256')
    for cap in captions:
        request = requests[cap['source_response_record_id']]
        response = responses[cap['request_sha256']]
        assert cap['request_sha256'] == request['request_sha256']
        assert cap['caption'] == response['raw_response'].strip() and response['returned_model'] == MODEL
        assert response['seed'] == 42
    code = [Path(__file__), Path(api.__file__), Path(native.__file__),
            ROOT / 'scripts/memory/memeye_backbone_long_text.py', ROOT / 'scripts/memory/online_zscore.py',
            ROOT / 'scripts/memory/ridge_whitening.py', ROOT / 'scripts/graph/caption_ward_visual.py',
            ROOT / 'scripts/graph/visual_image_encoder.py', ROOT / 'scripts/memory/caption_visual_master.py',
            ROOT / 'scripts/analysis/prepare_all_domain_adaptive_inputs.py']
    sources = [SOURCE / 'protocol.json', SOURCE / 'inputs/protocol.json', SOURCE / 'captions.jsonl',
        SOURCE / 'inputs/source_memories.jsonl', SOURCE / 'inputs/source_questions.jsonl',
        SOURCE / 'inputs/evaluation.jsonl', SOURCE / 'caption_generation/requests.jsonl',
        SOURCE / 'caption_generation/fresh_responses.jsonl', *code]
    for folder in ('weighted_evaluation', 'complementary_evaluation'):
        sources.extend(SOURCE / folder / name for name in ('summary.json', 'retrieval_results.jsonl',
            'reader/protocol.json', 'reader/requests.jsonl', 'reader/responses.jsonl', 'reader/gold.jsonl'))
    for domain in sorted({m['task'] for m in memories}):
        basis = metadata['basis_by_domain'][domain]
        native.verify_hashes({basis['path']: basis['sha256'], basis['protocol_path']: basis['protocol_sha256']})
        sources.extend([Path(basis['path']), Path(basis['protocol_path'])])
        build = read(SOURCE / 'domains' / domain / 'build_protocol.json')
        native.verify_hashes(build['artifact_sha256'])
        sources.extend(SOURCE / 'domains' / domain / 'memory' / f for f in ('graph.json', 'basis.npy', 'master_sums.npy'))
    hashes = {str(p): sha(p) for p in sources}
    for memory in memories:
        assert sha(memory['image_path']) == memory['image_sha256']
        hashes[memory['image_path']] = memory['image_sha256']
    output.mkdir(parents=True, exist_ok=True)
    write_lines(output / 'inputs/memories.jsonl', memories)
    write_lines(output / 'inputs/questions.jsonl', questions)
    write_lines(output / 'inputs/gold.jsonl', lines(SOURCE / 'inputs/evaluation.jsonl'))
    dump(output / 'inputs/encoding.json', metadata)
    for name in ('memories.jsonl', 'questions.jsonl', 'gold.jsonl', 'encoding.json'):
        path = output / 'inputs' / name
        hashes[str(path)] = sha(path)
    contract = {'model': MODEL, 'model_identity_kind': 'provider_alias', 'resolved_model_revision': None,
        'seed': SEED, 'caption_seed': 42, 'judge_seed': 42, 'caption_model': MODEL,
        'caption_source': str(SOURCE / 'captions.jsonl'), 'caption_generation_calls': 0,
        'device': device, 'workers': workers, 'questions': 371, 'memory_leaves': 438,
        'conditions': list(CONDITIONS), 'lambda': .03, 'alpha': 1.,
        'generation': {'temperature': 0, 'enable_thinking': False, 'mcq_max_tokens': 16,
                       'open_max_tokens': 2048, 'open_candidate_max_tokens': 2048, 'image_detail': 'high'},
        'offline': 'Fresh all-domain chronological caption-Ward rebuild; original fixed external PCA bases and batch1 caption encoding.',
        'historical_graphs_or_answers_reused': False, 'long_text_policy': POLICY,
        'source_sha256': hashes}
    protocol = {'run_id': str(uuid.uuid4()), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'contract': contract, 'experiment_signature': digest(contract)}
    dump(output / 'protocol.json', protocol)
    dump(output / 'validation_input.json', {'passed': True, 'caption_images': 437,
        'memory_leaves': 438, 'questions': 371, 'domains': 8, 'caption_model': MODEL,
        'caption_seed': 42, 'new_generation_seed': 1024, 'api_calls': 0})
    print(json.dumps({'stage': 'prepared', 'root': str(output), 'model': MODEL, 'seed': SEED}), flush=True)
    return protocol


def rebuild(output, protocol):
    marker = output / 'memory_manifest.json'
    if marker.exists():
        result = read(marker)
        assert result['experiment_signature'] == protocol['experiment_signature']
        native.verify_hashes(result['output_sha256'])
        return result
    seed_all()
    meta = read(output / 'inputs/encoding.json')
    memories = lines(output / 'inputs/memories.jsonl')
    device = protocol['contract']['device']
    args = type('Device', (), {'device': device})()
    native.configure_device(args)
    image_meta = meta['image_encoder']
    encoder = native.ImageEncoder(image_meta['model_name'], revision=image_meta['revision'], device=device, local_files_only=True)
    assert native._metadata_equal(encoder.metadata, image_meta)
    text_encoder = native.load_text_encoder(meta['text_encoder'], device)
    domains, hashes = [], {}
    for task in sorted({m['task'] for m in memories}):
        seed_all()
        rows = [m for m in memories if m['task'] == task]
        graph, sums, proof = native._build_domain(rows, output / 'domains' / task,
            np.load(meta['basis_by_domain'][task]['path'], allow_pickle=False), encoder, text_encoder,
            protocol['experiment_signature'])
        compared = {}
        for name in ('graph.json', 'basis.npy', 'master_sums.npy'):
            new, old = output / 'domains' / task / 'memory' / name, SOURCE / 'domains' / task / 'memory' / name
            detail = {'sha256': sha(new), 'seed42_sha256': sha(old), 'byte_identical': sha(new) == sha(old)}
            if name.endswith('.npy'):
                a, b = np.load(new), np.load(old)
                detail['same_shape'] = a.shape == b.shape
                detail['max_abs_difference'] = float(np.max(np.abs(a - b))) if a.shape == b.shape else None
            compared[name] = detail
        old_graph = read(SOURCE / 'domains' / task / 'memory/graph.json')
        domains.append({'task': task, 'images': len(rows), 'masters': len(graph['masters']),
            'seed42_masters': len(old_graph['masters']), 'files': compared,
            'same_memberships': [(m['id'], m['member_ids']) for m in graph['masters']] ==
                                [(m['id'], m['member_ids']) for m in old_graph['masters']]})
        hashes.update(proof['artifact_sha256'])
        hashes[str(output / 'domains' / task / 'build_protocol.json')] = sha(output / 'domains' / task / 'build_protocol.json')
        print(json.dumps({'stage': 'rebuild', **{k: domains[-1][k] for k in ('task', 'images', 'masters', 'same_memberships')}}), flush=True)
    del encoder, text_encoder
    gc.collect()
    result = {'completed': True, 'experiment_signature': protocol['experiment_signature'],
        'memory_leaves': sum(d['images'] for d in domains), 'masters': sum(d['masters'] for d in domains),
        'domains': domains, 'historical_graphs_reused': False, 'image_arrival_forwards': 438,
        'old_image_reencodes': 0, 'output_sha256': hashes}
    dump(marker, result)
    return result


def rank_four(question, memories, graph, sums, queries, leaves, candidates):
    raw = native.rank_question(question, memories, graph, sums, queries, leaves, candidates)
    global_scores = ZScoreWeighting(.03, 1.).apply(queries[0], leaves)
    master_scores = np.empty(len(memories))
    group_rows = []
    for group in raw['masters']:
        indices = [i for i, node in enumerate(graph['image_nodes']) if node['master_id'] == group['master_id']]
        query = queries[1 + 'ABCD'.index(group['selected_choice'])]
        adjusted = ZScoreWeighting(.03, 1.).apply(query, leaves[indices])
        master_scores[indices] = adjusted['weighted']
        group_rows.append({'master_id': group['master_id'], 'indices': indices,
            'query': group['augmented_query'], 'raw': adjusted['raw'].tolist(),
            'white': adjusted['white'].tolist(), 'z': adjusted['z'].tolist(),
            'weighted': adjusted['weighted'].tolist(), 'raw_std': adjusted['raw_std']})
    for name, scores in ((CONDITIONS[2], global_scores['weighted']), (CONDITIONS[3], master_scores)):
        order = np.argsort(-scores, kind='stable')
        raw['conditions'].append({'name': name, 'top5_memory_ids': [memories[i]['id'] for i in order[:5]],
            'top5_scores': [float(scores[i]) for i in order[:5]],
            'full_rank_memory_ids': [memories[i]['id'] for i in order]})
    raw['weighting_audit'] = {'global': {k: global_scores[k].tolist() for k in ('raw', 'white', 'z', 'weighted')},
                              'masters': group_rows}
    return raw


def retrieve(output, protocol):
    marker = output / 'retrieval_audit.json'
    if marker.exists():
        audit = read(marker)
        native.verify_hashes(audit['output_sha256'])
        return audit
    seed_all()
    metadata = read(output / 'inputs/encoding.json')
    memories = lines(output / 'inputs/memories.jsonl')
    questions = lines(output / 'inputs/questions.jsonl')
    device = protocol['contract']['device']
    bge = native.load_text_encoder(metadata['text_encoder'], device)
    clip, tokenizer, limit = native.load_clip_text_encoder(metadata['image_encoder'], device)
    previous = historical_retrieval()
    results, encoding = [], []
    for task in sorted({m['task'] for m in memories}):
        ms, qs = [m for m in memories if m['task'] == task], [q for q in questions if q['task'] == task]
        graph = read(output / 'domains' / task / 'memory/graph.json')
        sums = np.load(output / 'domains' / task / 'memory/master_sums.npy', allow_pickle=False)
        leaves, leaf_info = encode_complete(bge, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='dialogue+caption')
        query_texts = [t for q in qs for t in [q['retrieval_query'],
            *[q['retrieval_query'] + '\nCandidate answer: ' + q['candidate_statements'][c] for c in 'ABCD']]]
        qv, query_info = encode_complete(bge, query_texts, metadata['query_prefix'], field='MCQ queries')
        cv, candidate_info = native._clip_candidates(clip, tokenizer, limit, [q['candidate_statements'][c] for q in qs for c in 'ABCD'])
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        for i, q in enumerate(qs):
            results.append(rank_four(q, ms, graph, sums, qv[i], leaves, cv[i]))
        encoding.append({'task': task, 'leaf': leaf_info, 'query': query_info, 'candidate': candidate_info})
        print(json.dumps({'stage': 'retrieve', 'task': task, 'completed': len(results)}), flush=True)
    del bge, clip
    gc.collect()
    write_lines(output / 'retrieval_results.jsonl', results)
    gold = native.unique(lines(output / 'inputs/gold.jsonl'))
    metrics = {}
    for condition in CONDITIONS:
        selected = [(r, next(c for c in r['conditions'] if c['name'] == condition)) for r in results]
        hits = sum(len(set(c['top5_memory_ids']) & set(gold[r['id']]['gold_ids'])) for r, c in selected)
        metrics[condition] = {'questions': 371, 'gold_hits_top5': hits, 'gold_references': 1041,
            'recall_at5_micro': hits / 1041,
            'changed_top5_order_vs_seed42': sum(c['top5_memory_ids'] != previous[r['id']][condition]['top5_memory_ids'] for r, c in selected),
            'changed_top5_set_vs_seed42': sum(set(c['top5_memory_ids']) != set(previous[r['id']][condition]['top5_memory_ids']) for r, c in selected)}
    records = []
    texts = {m['id']: m['retrieval_text'] for m in memories}
    qs = native.unique(questions)
    system = metadata['reader_system_prompt']
    for i, result in enumerate(results):
        q = qs[result['id']]
        batch = []
        for condition in result['conditions']:
            ids = condition['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            context = '\n\n'.join(f'Memory {j}: {texts[mid]}' for j, mid in enumerate(ids, 1))
            text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context +
                '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n' +
                '\n'.join(f'{c}. {q["options"][c]}' for c in 'ABCD') + '\n\nAnswer with one letter only.')
            payload = {'system': system, 'user_text': text, 'query_images': q['query_images']}
            batch.append({'record_id': q['id'] + '::' + condition['name'], 'question_id': q['id'],
                'task': q['task'], 'condition': condition['name'], 'top_memory_ids': ids,
                **payload, 'payload_sha256': digest(payload)})
        offset = i % 4
        records.extend(batch[offset:] + batch[:offset])
    write_lines(output / 'reader/requests.jsonl', records)
    audit = {'passed': True, 'questions': 371, 'logical_records': 1484,
        'metrics': metrics, 'encoding': encoding, 'long_text_policy': POLICY,
        'source_signature': protocol['experiment_signature'], 'gold_loaded_for_ranking': False,
        'output_sha256': {str(p): sha(p) for p in (output / 'retrieval_results.jsonl', output / 'reader/requests.jsonl')}}
    dump(marker, audit)
    return audit


def api_call(output, payload, cap, *, env_file=ENV):
    """Run-local exact-wire cache; retain provider rejection, never invent text."""
    api = transport()
    body = api.build_request(payload, model=MODEL, max_output_tokens=cap, image_detail='high')
    assert body['seed'] == 1024 and body['model'] == MODEL and body['enable_thinking'] is False
    key = api.request_sha256(body)
    path = output / 'api_calls' / (key + '.json')
    if path.exists():
        cached = read(path)
        assert cached['request'] == body and cached['request_sha256'] == key
        return key, cached['response']
    config = api.load_config(env_file)
    attempts = []
    for attempt in range(1, 5):
        try:
            response = api.chat_request(config, body, timeout=120)
            assert response['returned_model'] == MODEL
            response = {**response, 'seed': 1024, 'status': 'ok'}
            break
        except api.APIError as exc:
            kind = getattr(exc, 'provider_error_code', None) or getattr(exc, 'kind', type(exc).__name__)
            attempts.append({'attempt': attempt, 'kind': kind, 'retryable': bool(getattr(exc, 'retryable', False))})
            dump(output / 'api_errors' / (key + '.json'), {'request_sha256': key, 'attempts': attempts})
            if kind in ('data_inspection_failed', 'input_inspection_failed', 'provider_input_rejected', 'content_filter'):
                response = {'status': 'provider_input_rejected', 'raw_response': None,
                            'returned_model': None, 'seed': SEED, 'error_kind': kind}
                break
            if not getattr(exc, 'retryable', False) or attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    freeze(path, {'request': body, 'request_sha256': key, 'response': response, 'attempts': attempts})
    return key, response


def run_mcq(output, protocol):
    requests = lines(output / 'reader/requests.jsonl')
    assert len(requests) == 1484 and len({r['record_id'] for r in requests}) == 1484
    by_payload = {}
    for r in requests:
        by_payload.setdefault(r['payload_sha256'], r)
    results = {}
    def one(row):
        return api_call(output, {k: row[k] for k in ('system', 'user_text', 'query_images')}, 16)
    with ThreadPoolExecutor(max_workers=protocol['contract']['workers']) as pool:
        futures = {pool.submit(one, row): key for key, row in by_payload.items()}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            if len(results) % 20 == 0 or len(results) == len(futures):
                print(json.dumps({'stage': 'mcq_api', 'completed': len(results), 'total': len(futures)}), flush=True)
    gold = native.unique(lines(output / 'inputs/gold.jsonl'))
    rows = []
    first = {}
    for source in requests:
        key, response = results[source['payload_sha256']]
        raw = response.get('raw_response')
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
        choice = match.group(1).upper() if match else None
        origin = first.setdefault(key, source['record_id'])
        rows.append({**response, 'record_id': source['record_id'], 'question_id': source['question_id'],
            'task': source['task'], 'condition': source['condition'], 'predicted_choice': choice,
            'gold_choice': gold[source['question_id']]['gold_choice'],
            'correct': choice == gold[source['question_id']]['gold_choice'],
            'payload_sha256': source['payload_sha256'], 'request_sha256': key,
            'experiment_signature': protocol['experiment_signature'], 'run_id': protocol['run_id'],
            'historical_answer_reuse': False, 'fresh_inference_for_record': origin == source['record_id'],
            'same_run_shared_request_record_id': origin if origin != source['record_id'] else None})
    write_lines(output / 'reader/responses.jsonl', rows)
    return analyze(output)


def analyze(output=OUTPUT):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    verify(protocol)
    retrieval = read(output / 'retrieval_audit.json')
    native.verify_hashes(retrieval['output_sha256'])
    responses = lines(output / 'reader/responses.jsonl')
    requests = native.unique(lines(output / 'reader/requests.jsonl'), 'record_id')
    assert len(responses) == len(native.unique(responses, 'record_id')) == len(requests) == 1484
    old = {}
    for folder in ('weighted_evaluation', 'complementary_evaluation'):
        for row in lines(SOURCE / folder / 'reader/responses.jsonl'):
            old[row['record_id']] = row
    metrics = copy.deepcopy(retrieval['metrics'])
    for response in responses:
        source = requests[response['record_id']]
        assert response['payload_sha256'] == source['payload_sha256']
        cache = read(output / 'api_calls' / (response['request_sha256'] + '.json'))
        assert cache['request']['seed'] == 1024 and cache['request']['model'] == MODEL
        assert cache['response'].get('raw_response') == response['raw_response']
        assert response['experiment_signature'] == protocol['experiment_signature'] and not response['historical_answer_reuse']
    for condition in CONDITIONS:
        group = [r for r in responses if r['condition'] == condition]
        assert len(group) == len({r['question_id'] for r in group}) == 371
        correct = sum(r['correct'] for r in group)
        metrics[condition].update({'correct': correct, 'accuracy': correct / 371,
            'seed42_correct': sum(old[r['record_id']].get('predicted_choice') == r['gold_choice'] for r in group),
            'changed_answers_vs_seed42': sum(r['predicted_choice'] != old[r['record_id']].get('predicted_choice') for r in group),
            'outcomes': dict(Counter(r['status'] for r in group)),
            'format_errors': sum(r['status'] == 'ok' and r['predicted_choice'] is None for r in group)})
    summary = {'status': 'complete', 'seed': 1024, 'caption_seed': 42, 'judge_seed': 42,
        'model': MODEL, 'questions': 371, 'overall': metrics, 'logical_records': 1484,
        'unique_requests': len({r['request_sha256'] for r in responses}),
        'historical_answers_reused': 0, 'memory_rebuild': read(output / 'memory_manifest.json'),
        'interpretation': 'Fixed-option MCQ accuracy; provider alias does not establish fixed weights across dates.'}
    dump(output / 'summary.json', summary)
    dump(output / 'independent_audit.json', {'passed': True, 'questions_per_condition': 371,
        'logical_records': 1484, 'seed1024_actual_request_bodies_checked': True,
        'frozen_caption_source_verified': True, 'artifact_sha256': {str(p): sha(p) for p in (
            output / 'summary.json', output / 'reader/responses.jsonl', output / 'retrieval_audit.json',
            output / 'memory_manifest.json')}})
    print(json.dumps({'stage': 'analyzed', 'correct': {c: metrics[c]['correct'] for c in CONDITIONS}}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'rebuild', 'retrieve', 'mcq', 'analyze', 'run'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = prepare(output, args.device, args.workers)
    if args.phase in ('rebuild', 'run'):
        rebuild(output, protocol)
    if args.phase in ('retrieve', 'run'):
        retrieve(output, protocol)
    if args.phase in ('mcq', 'run'):
        run_mcq(output, protocol)
    if args.phase == 'analyze':
        analyze(output)


if __name__ == '__main__':
    main()
