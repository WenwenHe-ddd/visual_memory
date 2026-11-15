"""Rebuild MemEye caption/visual memories from one backbone's fresh captions.

This stage never opens the separate evaluation labels. It retains full visual
vectors only for masters; each arriving image is encoded once and its leaf keeps
two coordinates. Atomic checkpoints resume from text and aggregates without
re-encoding images that have already entered this run's memory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np

from scripts.analysis.probe_caption_visual_online import normalize
from scripts.analysis.prepare_all_domain_adaptive_inputs import save_memory
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master import (
    configure_device, load_clip_text_encoder, load_text_encoder,
)

CONDITIONS = ('baseline_question_only', 'master_answer_query')
LETTERS = 'ABCD'
MEMORY_SEPARATOR = '\nMemory description: '
QUERY_IMAGE_SEPARATOR = "\nquestion's image:\nimage_caption: "


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    if path.exists():
        if path.read_text(encoding='utf-8') != content:
            raise ValueError(f'Refusing to overwrite changed artifact: {path}')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def verify_hashes(hashes, base=None):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute() and base is not None:
            path = Path(base) / path
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Input or artifact changed: {path}')


def unique(rows, key='id'):
    mapped = {row[key]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return mapped


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def attach_captions(memories, questions, captions):
    """Join fresh captions by image hash, never by labels or old retrieval."""
    unique(memories)
    unique(memories, 'image_id')
    unique(questions)
    by_sha = unique(captions)
    needed = {row['image_sha256'] for row in memories}
    needed.update(image['sha256'] for row in questions for image in row['query_images'])
    if set(by_sha) != needed:
        raise ValueError('Fresh caption coverage differs from history/query images')
    models = {_text(row['model'], 'caption model') for row in captions}
    if len(models) != 1:
        raise ValueError('Fresh captions must come from exactly one backbone')
    for row in captions:
        _text(row['caption'], 'caption')
        _text(row['source_response_record_id'], 'caption response provenance')
        _text(row['request_sha256'], 'caption request hash')
    stored = []
    for row in memories:
        if any(key in row for key in ('caption', 'retrieval_text', 'gold_ids', 'gold_choice')):
            raise ValueError('Source memory includes old caption or evaluation fields')
        caption = by_sha[row['image_sha256']]['caption']
        stored.append({**row, 'caption': caption,
                       'retrieval_text': row['dialogue'] + MEMORY_SEPARATOR + caption})
    enriched = []
    for row in questions:
        if any(key in row for key in ('retrieval_query', 'query_image_captions', 'gold_ids', 'gold_choice')):
            raise ValueError('Source question includes old query text or evaluation fields')
        statements = row['candidate_statements']
        if set(statements) != set(LETTERS):
            raise ValueError('Each question requires four frozen option statements')
        for key in LETTERS:
            _text(statements[key], 'candidate statement')
        query_captions = [by_sha[image['sha256']]['caption'] for image in row['query_images']]
        query = row['question'] + (QUERY_IMAGE_SEPARATOR + ' '.join(query_captions)
                                   if query_captions else '')
        enriched.append({**row, 'retrieval_query': query,
                         'query_image_captions': query_captions,
                         'query_image_paths': [image['path'] for image in row['query_images']]})
    return stored, enriched, next(iter(models))


def encode_checked(model, texts, prefix='', *, label='text'):
    """Check the complete BGE input, including query prefix, before encoding."""
    if not texts:
        raise ValueError('Cannot encode an empty text batch')
    inputs = [prefix + _text(text, label) for text in texts]
    lengths = [len(tokens) for tokens in model.tokenizer(inputs, truncation=False)['input_ids']]
    limit = int(model.max_seq_length)
    overflow = [(i, length) for i, length in enumerate(lengths) if length > limit]
    if overflow:
        raise ValueError(f'{label} exceeds BGE token limit {limit}; refusing silent truncation: {overflow[:5]}')
    vectors = normalize(model.encode(inputs, batch_size=16, normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False, prompt=''))
    if vectors.ndim != 2 or len(vectors) != len(inputs):
        raise ValueError('Text encoder returned an incompatible batch')
    return vectors, {'count': len(inputs), 'max_tokens': max(lengths),
                     'token_limit': limit, 'truncated_inputs': 0}


def _checkpoint(memory, audits, signature):
    return {'source_signature': signature, 'graph': memory.snapshot(),
            'master_sums': [master['visual_sum'].tolist() for master in memory.masters.values()],
            'next_master_index': memory._next_master_index, 'insertion_audit': audits}


def restore_checkpoint(value, basis, expected_rows, text_encoder, signature):
    """Recover old state using saved 2-D coordinates, master sums and captions."""
    if value['source_signature'] != signature:
        raise ValueError('Partial memory belongs to different inputs or code')
    graph, audits = value['graph'], value['insertion_audit']
    nodes = graph['image_nodes']
    if not nodes or len(nodes) != len(audits) or len(nodes) > len(expected_rows):
        raise ValueError('Invalid partial-memory length')
    for node, row in zip(nodes, expected_rows):
        if (set(node) != {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'}
                or node['id'] != row['image_id']
                or any(node[key] != row[key] for key in ('caption', 'dialogue', 'timestamp'))
                or len(node['visual_coords']) != 2):
            raise ValueError('Partial memory differs from chronological source')
    memory = CaptionWardVisualMemory(basis)
    sums = np.asarray(value['master_sums'], dtype=np.float64)
    masters = graph['masters']
    if sums.shape != (len(masters), memory.visual_dim) or not np.isfinite(sums).all():
        raise ValueError('Invalid partial master visual sums')
    if sorted(mid for master in masters for mid in master['member_ids']) != sorted(node['id'] for node in nodes):
        raise ValueError('Partial master membership is not a partition')
    memory.image_nodes = {node['id']: copy.deepcopy(node) for node in nodes}
    for master, visual_sum in zip(masters, sums):
        row = {key: copy.deepcopy(val) for key, val in master.items()
               if key not in ('visual_sum_ref', 'visual_mean_ref')}
        if row['n'] != len(row['member_ids']) or not all(
                memory.image_nodes[mid]['master_id'] == row['id'] for mid in row['member_ids']):
            raise ValueError('Partial master count or membership mismatch')
        coords = np.sum([memory.image_nodes[mid]['visual_coords'] for mid in row['member_ids']], axis=0)
        np.testing.assert_allclose(memory.basis.T @ visual_sum, coords, rtol=1e-10, atol=1e-10)
        row['visual_sum'] = visual_sum.copy()
        memory.masters[row['id']] = row
    # Repeat the original single-caption batch shape: batching differently can
    # introduce small floating-point differences that affect a Ward tie.
    text_vectors = np.concatenate([encode_checked(text_encoder, [node['caption']],
                                                 label='checkpoint caption')[0] for node in nodes])
    memory.text_embedding_cache = {node['id']: vector for node, vector in zip(nodes, text_vectors)}
    memory.text_dim = text_vectors.shape[1]
    if memory.text_dim != graph['text_dim']:
        raise ValueError('Restored caption embedding dimension changed')
    memory._next_master_index = int(value['next_master_index'])
    if memory._next_master_index <= max(int(master['id'].rsplit('_', 1)[1]) for master in masters):
        raise ValueError('Partial next-master ID would collide')
    return memory, copy.deepcopy(audits)


def _metadata_equal(actual, expected):
    return {key: value for key, value in actual.items() if key != 'device'} == {
        key: value for key, value in expected.items() if key != 'device'}


def _build_domain(rows, target, basis, image_encoder, text_encoder, signature):
    """One image forward per chronological arrival; completed domains resume."""
    target = Path(target)
    marker = target / 'build_protocol.json'
    if marker.exists():
        previous = read(marker)
        if previous['source_signature'] != signature:
            raise ValueError('Existing domain memory belongs to different inputs')
        verify_hashes(previous['artifact_sha256'])
        return read(target / 'memory/graph.json'), np.load(target / 'memory/master_sums.npy', allow_pickle=False), previous
    target.mkdir(parents=True, exist_ok=True)
    partial = target / 'build_checkpoint.json'
    if partial.exists():
        memory, audits = restore_checkpoint(read(partial), basis, rows, text_encoder, signature)
    else:
        memory, audits = CaptionWardVisualMemory(basis), []
    started = time.perf_counter()
    for row in rows[len(audits):]:
        if sha(row['image_path']) != row['image_sha256']:
            raise ValueError(f'Incoming image changed: {row["image_id"]}')
        # Complete caption validation precedes the image forward. Once add()
        # returns, the full incoming vector is discarded before checkpointing.
        caption_vector, token_info = encode_checked(text_encoder, [row['caption']], label=row['image_id'])
        visual = normalize(image_encoder.encode(row['image_path']))
        audit = memory.add(row['image_id'], row['caption'], caption_vector[0], visual,
                           dialogue=row['dialogue'], timestamp=row['timestamp'])
        del visual, caption_vector
        audit.update(insertion_index=row['insertion_index'], caption_encoding=token_info)
        audits.append(audit)
        dump(partial, _checkpoint(memory, audits, signature))
    graph, sums = save_memory(memory, target / 'memory')
    write_lines(target / 'insertion_audit.jsonl', audits)
    artifact_paths = [target / 'memory' / name for name in ('graph.json', 'basis.npy', 'master_sums.npy')]
    artifact_paths.append(target / 'insertion_audit.jsonl')
    protocol = {'source_signature': signature, 'images': len(rows), 'masters': len(graph['masters']),
                'insertion_order': [row['image_id'] for row in rows],
                'image_arrival_forwards': len(rows), 'old_image_reencodes': 0,
                'per_image_visual_embeddings_saved': False, 'leaf_visual_dimension': 2,
                'clustering_seconds': sum(row['timings_seconds']['clustering'] for row in audits),
                'seconds_this_invocation': time.perf_counter() - started,
                'artifact_sha256': {str(path): sha(path) for path in artifact_paths}}
    dump(marker, protocol)
    partial.unlink(missing_ok=True)
    return graph, sums, protocol


def rank_question(question, memories, graph, sums, query_vectors, leaf_vectors, candidate_vectors, top_k=5):
    """Pure ranking function: no evaluation fields, image I/O or model calls."""
    if not 1 <= top_k <= len(memories):
        raise ValueError('top_k must fit the domain memory pool')
    masters, nodes = graph['masters'], graph['image_nodes']
    if [node['id'] for node in nodes] != [row['image_id'] for row in memories]:
        raise ValueError('Graph leaves and source chronology differ')
    if any(node['caption'] != row['caption'] or node['dialogue'] != row['dialogue']
           for node, row in zip(nodes, memories)):
        raise ValueError('Graph text differs from fresh captions')
    query_vectors, leaf_vectors, candidate_vectors = map(np.asarray, (query_vectors, leaf_vectors, candidate_vectors))
    if (query_vectors.shape != (5, leaf_vectors.shape[1]) or leaf_vectors.shape[0] != len(memories)
            or candidate_vectors.shape != (4, sums.shape[1])):
        raise ValueError('Retrieval feature dimensions differ')
    lookup = {master['id']: index for index, master in enumerate(masters)}
    leaf_master = np.asarray([lookup[node['master_id']] for node in nodes], dtype=int)
    centers = normalize([sums[master['visual_mean_ref']['row']] * master['visual_mean_ref']['scale']
                         for master in masters])
    clip_scores = centers @ candidate_vectors.T
    choices = np.argmax(clip_scores, axis=1)
    all_scores = query_vectors @ leaf_vectors.T
    expanded = all_scores[1 + choices[leaf_master], np.arange(len(memories))]
    orders = [np.argsort(-scores, kind='stable') for scores in (all_scores[0], expanded)]
    ranks = []
    for order in orders:
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        ranks.append(rank)
    leaves = [{
        'leaf_index': index, 'image_id': row['image_id'], 'memory_id': row['id'],
        'filename': Path(row['image_path']).name, 'master_id': nodes[index]['master_id'],
        'master_choice': LETTERS[choices[leaf_master[index]]],
        'baseline_score': float(all_scores[0, index]), 'expanded_score': float(expanded[index]),
        'baseline_rank': int(ranks[0][index]), 'expanded_rank': int(ranks[1][index]),
    } for index, row in enumerate(memories)]
    conditions = []
    for name, order, scores in zip(CONDITIONS, orders, (all_scores[0], expanded)):
        indices = order[:top_k].tolist()
        reader_memories = [{'memory_id': memories[index]['id'],
                            'filename': Path(memories[index]['image_path']).name,
                            'reader_text': memories[index]['retrieval_text']} for index in indices]
        conditions.append({
            'name': name, 'top5_memory_ids': [memories[index]['id'] for index in indices],
            'top5_scores': scores[indices].tolist(), 'top5_leaf_indices': indices,
            'full_rank_memory_ids': [memories[index]['id'] for index in order],
            'top5': [{**leaves[index], 'rank': rank, 'score': float(scores[index])}
                     for rank, index in enumerate(indices, 1)],
            'reader_memories': reader_memories,
            'reader_context': '\n\n'.join(f'Memory {rank}: {row["reader_text"]}'
                                          for rank, row in enumerate(reader_memories, 1)),
        })
    master_rows = []
    for index, master in enumerate(masters):
        key = LETTERS[choices[index]]
        statement = question['candidate_statements'][key]
        ordered = np.sort(clip_scores[index])[::-1]
        master_rows.append({'master_id': master['id'], 'n': master['n'], 'member_ids': master['member_ids'],
                            'clip_candidate_scores': dict(zip(LETTERS, clip_scores[index].tolist())),
                            'selected_choice': key, 'selected_statement': statement,
                            'top_two_margin': float(ordered[0] - ordered[1]),
                            'augmented_query': question['retrieval_query'] + '\nCandidate answer: ' + statement})
    return {
        'id': question['id'], 'question_id': question['id'], 'task': question['task'],
        'question': question['question'], 'options': question['options'],
        'query_image_paths': question['query_image_paths'], 'query_images': question['query_images'],
        'query_image_captions': question['query_image_captions'],
        'baseline_retrieval_query': question['retrieval_query'],
        'candidate_statements': question['candidate_statements'], 'conditions': conditions,
        'masters': master_rows, 'leaves': leaves,
        'master_choice_counts': {key: sum(row['selected_choice'] == key for row in master_rows) for key in LETTERS},
        'top5_same_order': conditions[0]['top5_memory_ids'] == conditions[1]['top5_memory_ids'],
    }


def _clip_candidates(model, tokenizer, limit, statements):
    import torch
    lengths = [len(row) for row in tokenizer(statements, truncation=False)['input_ids']]
    if not lengths or max(lengths) > limit:
        raise ValueError('Frozen candidate statements exceed CLIP token limit')
    vectors = []
    for start in range(0, len(statements), 32):
        tokens = tokenizer(statements[start:start + 32], padding=True, truncation=False, return_tensors='pt')
        tokens = tokens.to(next(model.parameters()).device)
        with torch.inference_mode():
            vectors.append(model(**tokens, return_dict=True).text_embeds.double().cpu().numpy())
    return normalize(np.concatenate(vectors)), {'count': len(statements), 'max_tokens': max(lengths),
                                               'token_limit': limit, 'truncated_inputs': 0}


def rebuild_and_retrieve(output, device='cuda:0'):
    """Build all eight domains and both fresh Top5 rankings, without gold labels."""
    output = Path(output).resolve()
    inputs = output / 'inputs'
    preparation = read(inputs / 'protocol.json')
    run_protocol = read(output / 'protocol.json')
    if digest(run_protocol['contract']) != run_protocol['experiment_signature']:
        raise ValueError('Root experiment signature changed')
    verify_hashes(run_protocol['contract']['source_sha256'])
    verify_hashes(preparation['source_sha256'])
    if 'artifact_sha256' in preparation:
        verify_hashes({name: value for name, value in preparation['artifact_sha256'].items()
                       if name != 'evaluation.jsonl'}, inputs)
    memories, questions, model_name = attach_captions(
        lines(inputs / 'source_memories.jsonl'), lines(inputs / 'source_questions.jsonl'),
        lines(output / 'captions.jsonl'))
    if model_name != run_protocol['contract']['model']:
        raise ValueError('Fresh captions do not match this experiment backbone')
    domains = sorted({row['task'] for row in memories})
    if (len(memories) != 438 or len(questions) != 371 or len(domains) != 8
            or {row['task'] for row in questions} != set(domains)):
        raise ValueError('Full MemEye run requires all 438 image memories and 371 questions in eight domains')
    if sum(bool(row['query_images']) for row in questions) != 8:
        raise ValueError('Query image coverage differs from the frozen MemEye questions')
    source_paths = [output / 'protocol.json', inputs / 'protocol.json', inputs / 'source_memories.jsonl',
                    inputs / 'source_questions.jsonl', output / 'captions.jsonl', Path(__file__),
                    ROOT / 'scripts/graph/caption_ward_visual.py', ROOT / 'scripts/graph/visual_image_encoder.py',
                    ROOT / 'scripts/memory/caption_visual_master.py',
                    ROOT / 'scripts/analysis/prepare_all_domain_adaptive_inputs.py',
                    ROOT / 'scripts/analysis/probe_caption_visual_online.py']
    for domain in domains:
        basis_info = preparation['basis_by_domain'][domain]
        if sha(basis_info['path']) != basis_info['sha256'] or sha(basis_info['protocol_path']) != basis_info['protocol_sha256']:
            raise ValueError(f'Frozen encoder/basis provenance changed: {domain}')
        reference = read(basis_info['protocol_path'])
        if (reference['image_encoder'] != preparation['image_encoder']
                or reference['text_encoder']['files_sha256'] != preparation['text_encoder']['files_sha256']
                or reference['query_prefix'] != preparation['query_prefix']):
            raise ValueError(f'Encoder or query-prefix metadata differ: {domain}')
        source_paths.extend([Path(basis_info['path']), Path(basis_info['protocol_path'])])
        rows = [row for row in memories if row['task'] == domain]
        if [row['insertion_index'] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f'Invalid chronological source order: {domain}')
    source_hashes = {**preparation['source_sha256'], **run_protocol['contract']['source_sha256'],
                     **{str(inputs / name): value for name, value in preparation.get('artifact_sha256', {}).items()
                        if name != 'evaluation.jsonl'},
                     **{str(path): sha(path) for path in source_paths}}
    contract = {'source_sha256': source_hashes, 'device': device, 'caption_model': model_name, 'top_k': 5,
                'source_run_id': run_protocol['run_id'],
                'source_experiment_signature': run_protocol['experiment_signature'],
                'memory_count': 438, 'question_count': 371, 'domains': domains,
                'image_encoder': preparation['image_encoder'], 'text_encoder': preparation['text_encoder'],
                'query_prefix': preparation['query_prefix']}
    signature = digest(contract)
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        previous = read(marker)
        if previous['contract'] != contract or previous['experiment_signature'] != signature:
            raise ValueError('Completed retrieval belongs to a different experiment; use a fresh directory')
        verify_hashes(previous['artifact_sha256'])
        return previous
    runtime = SimpleNamespace(device=device)
    configure_device(runtime)
    image_meta = preparation['image_encoder']
    image_encoder = ImageEncoder(image_meta['model_name'], revision=image_meta['revision'],
                                 device=runtime.device, local_files_only=True)
    if not _metadata_equal(image_encoder.metadata, image_meta):
        raise ValueError('Actual CLIP image encoder differs from the frozen reference')
    text_encoder = load_text_encoder(preparation['text_encoder'], runtime.device)
    domain_states, domain_summaries, artifacts = {}, [], []
    started = time.perf_counter()
    for domain in domains:
        rows = [row for row in memories if row['task'] == domain]
        basis = np.load(preparation['basis_by_domain'][domain]['path'], allow_pickle=False)
        target = output / 'domains' / domain
        graph, sums, build = _build_domain(rows, target, basis, image_encoder, text_encoder, signature)
        domain_states[domain] = graph, sums
        domain_summaries.append({'task': domain, 'images': len(rows), 'masters': len(graph['masters']),
                                 'clustering_seconds': build['clustering_seconds']})
        artifacts.extend(Path(path) for path in build['artifact_sha256'])
        artifacts.append(target / 'build_protocol.json')
        print(json.dumps({'stage': 'backbone_memory_rebuilt', **domain_summaries[-1]}), flush=True)
    del image_encoder
    clip, tokenizer, clip_limit = load_clip_text_encoder(preparation['image_encoder'], runtime.device)
    results, encoding_audits = [], []
    for domain in domains:
        rows = [row for row in memories if row['task'] == domain]
        qs = [row for row in questions if row['task'] == domain]
        graph, sums = domain_states[domain]
        leaf_vectors, leaf_encoding = encode_checked(text_encoder, [row['retrieval_text'] for row in rows], label=f'{domain} dialogue+caption')
        query_texts, statements = [], []
        for question in qs:
            candidates = [question['candidate_statements'][key] for key in LETTERS]
            statements.extend(candidates)
            query_texts.extend([question['retrieval_query'], *[
                question['retrieval_query'] + '\nCandidate answer: ' + candidate for candidate in candidates]])
        query_vectors, query_encoding = encode_checked(text_encoder, query_texts, preparation['query_prefix'], label=f'{domain} queries')
        candidate_vectors, candidate_encoding = _clip_candidates(clip, tokenizer, clip_limit, statements)
        query_vectors = query_vectors.reshape(len(qs), 5, -1)
        candidate_vectors = candidate_vectors.reshape(len(qs), 4, -1)
        for index, question in enumerate(qs):
            results.append(rank_question(question, rows, graph, sums, query_vectors[index], leaf_vectors, candidate_vectors[index]))
        encoding_audits.append({'task': domain, 'leaf': leaf_encoding, 'query': query_encoding, 'candidate': candidate_encoding})
        print(json.dumps({'stage': 'backbone_retrieval', 'task': domain, 'questions': len(qs)}), flush=True)
    if {row['id'] for row in results} != {row['id'] for row in questions} or len(results) != 371:
        raise ValueError('Incomplete fresh retrieval coverage')
    # Store reconstructible textual memories; source pixels/paths belong only
    # in the separate audit inputs and are never copied into the memory store.
    write_lines(output / 'offline/memories.jsonl', [
        {key: value for key, value in row.items() if key not in ('image_path', 'image_sha256')}
        for row in memories])
    write_lines(output / 'retrieval_results.jsonl', results)
    artifacts.extend([output / 'offline/memories.jsonl', output / 'retrieval_results.jsonl'])
    verify_hashes(source_hashes)
    verify_hashes(preparation['source_sha256'])
    output_hashes = {str(path): sha(path) for path in artifacts}
    protocol = {'status': 'complete', 'contract': contract, 'experiment_signature': signature,
                'source_run_id': run_protocol['run_id'],
                'source_experiment_signature': run_protocol['experiment_signature'],
                'caption_model': model_name, 'questions': len(results), 'memory_nodes': len(memories),
                'domains': domain_summaries, 'source_sha256': source_hashes,
                'artifact_sha256': output_hashes, 'output_sha256': output_hashes,
                'results_sha256': sha(output / 'retrieval_results.jsonl'),
                'encoding_audits': encoding_audits, 'execution_device': runtime.device,
                'elapsed_seconds': time.perf_counter() - started,
                'gold_loaded_for_ranking': False, 'old_graphs_or_text_embeddings_reused': False,
                'fresh_caption_clustering_in_all_domains_including_home': True,
                'image_arrival_forwards': len(memories), 'old_image_reencodes': 0,
                'leaf_visual_dimension': 2, 'per_image_visual_embeddings_saved': False,
                'candidate_statements_in_reader_evidence': False,
                'candidate_policy': 'Frozen assistant-authored four option statements; unchanged between backbones.',
                'retrieval_policy': 'Full same-domain BGE scan; master CLIP candidate selects one expanded query per leaf.',
                'query_image_policy': 'Fresh caption in BGE query; original query pixels remain available to reader.',
                'tie_policy': 'Stable chronological memory order; ABCD order for equal master candidate scores.',
                'manual_similarity_thresholds': 0, 'score_fusion_weights': 0, 'leaf_distance_used': False,
                'identical_ordered_top5': sum(row['top5_same_order'] for row in results)}
    dump(marker, protocol)
    return protocol
