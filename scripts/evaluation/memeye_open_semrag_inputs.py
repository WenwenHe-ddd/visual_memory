#!/usr/bin/env python3
"""Prepare answer-free MemEye Open queries using the frozen 184/371 SemRAG memory."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
from scripts.memory.caption_visual_master import encode_text, load_text_encoder

ORIGINAL = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
FROZEN = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021/protocol.json'
STORES = ROOT / 'results/memory_stores_20261023'
SYSTEM = ('You answer questions about remembered images and conversations. Use only the retrieved '
          'dialogue and image descriptions and any query image provided. Give a concise, direct '
          'answer in natural language. If the evidence is insufficient, say so. Do not invent details.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, values):
    Path(path).write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


def payload_hash(system, text, images):
    payload = {'system': system, 'user_text': text,
               'query_images': [{'path': p, 'sha256': sha(p)} for p in images]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def prepare(output, device):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; frozen inputs are never overwritten.')
    mcq_path = ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'
    memories_path = ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl'
    source_paths = {Path(__file__).resolve(), mcq_path, memories_path, REFERENCE,
                    ORIGINAL / 'protocol.json', ORIGINAL / 'summary.json',
                    ORIGINAL / 'retrieval_results.jsonl', ORIGINAL / 'reader/protocol.json',
                    ORIGINAL / 'reader/qa_manifest.jsonl', ORIGINAL / 'reader/responses.jsonl',
                    ROOT / 'scripts/memory/caption_visual_master.py',
                    ROOT / 'scripts/analysis/probe_caption_visual_online.py'}
    pinned = read(REFERENCE)['source_sha256']
    source_protocol = read(ORIGINAL / 'protocol.json')
    source_summary = read(ORIGINAL / 'summary.json')['overall']['baseline_question_only']
    assert source_summary['correct'] == 184 and source_summary['questions'] == 371
    mcq = {r['id']: r for r in rows(mcq_path)}
    source_memories = {r['id']: r for r in rows(memories_path)}
    source_retrieval = {r['question_id']: r for r in rows(ORIGINAL / 'retrieval_results.jsonl')}
    source_reader = {r['id']: r for r in rows(ORIGINAL / 'reader/qa_manifest.jsonl')
                     if r['condition'] == 'baseline_question_only'}
    source_responses = [r for r in rows(ORIGINAL / 'reader/responses.jsonl')
                        if r['condition'] == 'baseline_question_only']
    assert len(source_reader) == len(source_responses) == len(mcq) == 371
    assert sum(bool(r['correct']) for r in source_responses) == 184
    for path, expected in source_protocol['source_sha256'].items():
        if Path(path) in (mcq_path, memories_path, REFERENCE):
            assert sha(path) == expected, path

    questions, gold, domains, differences = [], [], {}, []
    for task in sorted({q['task'] for q in mcq.values()}):
        path = ROOT / 'data/MemEye/data/dialog' / (task + '_Open.json')
        source_paths.add(path)
        opens = read(path)['human-annotated QAs']
        domains[task] = {'questions': len(opens), 'unchanged_questions': 0, 'changed_questions': 0}
        for raw in opens:
            qid = task + '/' + raw['question_id']
            old = mcq[qid]
            image = raw.get('question_image')
            relative_images = ([image] if isinstance(image, str) else image) or []
            image_paths = [str((ROOT / 'data/MemEye/data/image' / p).resolve()) for p in relative_images]
            assert image_paths == old['query_image_paths'], qid
            source_paths.update(map(Path, image_paths))
            assert old['retrieval_query'].startswith(old['question'])
            suffix = old['retrieval_query'][len(old['question']):]
            captions = old.get('query_image_captions', [])
            if image_paths:
                assert suffix == "\nquestion's image:\n" + '\n'.join('image_caption: ' + c for c in captions)
            else:
                assert not suffix and not captions
            point = '/'.join(str(v) for axis in raw['point'] for v in (axis if isinstance(axis, list) else [axis]))
            questions.append({'id': qid, 'task': task, 'point': point, 'question': raw['question'],
                              'query_image_paths': image_paths, 'query_image_captions': captions,
                              'retrieval_query': raw['question'] + suffix})
            gold_ids = [task + '/' + clue for clue in raw['clue']]
            assert set(gold_ids) == set(old['gold_ids']), qid
            gold.append({'id': qid, 'answer': raw['answer'], 'gold_ids': gold_ids, 'point': point})
            changed = raw['question'] != old['question']
            domains[task]['changed_questions' if changed else 'unchanged_questions'] += 1
            differences.append({'id': qid, 'question_changed': changed,
                                'open_gold_differs_from_mcq_correct_option': raw['answer'] != old['options'][old['gold_choice']]})
    assert len(questions) == len({q['id'] for q in questions}) == 371
    assert {q['id'] for q in questions} == set(mcq)
    changed_count = sum(r['question_changed'] for r in differences)
    assert changed_count == 234

    metadata = read(STORES / next(iter(domains)) / 'natural/memory.json')['encoding']
    encoder = load_text_encoder(metadata['text_encoder'], device)
    assert encoder.max_seq_length == 512
    import torch
    torch.set_num_threads(4)
    changed_questions = [q for q in questions if q['retrieval_query'] != mcq[q['id']]['retrieval_query']]
    new_vectors = encode_text(encoder, [q['retrieval_query'] for q in changed_questions], metadata['query_prefix'])
    encoded = {q['id']: vector for q, vector in zip(changed_questions, new_vectors)}
    query_tokens = [len(v) for v in encoder.tokenizer(
        [metadata['query_prefix'] + q['retrieval_query'] for q in questions], truncation=False)['input_ids']]
    assert max(query_tokens) <= 512
    retrieval, reader, memory_count, max_old_score_error, unchanged_matches = [], [], 0, 0., 0
    allowed_truncations = []
    for task in domains:
        frozen = FROZEN / task
        inputs_path, features_path = frozen / 'inputs.json', frozen / 'features.npz'
        store = STORES / task / 'natural'
        source_paths.update([inputs_path, features_path, frozen / 'protocol.json', store / 'memory.json', store / 'leaves.json'])
        for path in (inputs_path, features_path, frozen / 'protocol.json'):
            assert sha(path) == pinned[str(path)], path
        original_inputs, memory_manifest = read(inputs_path), read(store / 'memory.json')
        leaves = read(store / 'leaves.json')
        assert sha(store / 'leaves.json') == memory_manifest['files']['leaves']['sha256']
        assert memory_manifest['encoding']['text_encoder'] == metadata['text_encoder']
        assert memory_manifest['encoding']['query_prefix'] == metadata['query_prefix']
        allowed_truncations.extend(memory_manifest['encoding']['allowed_leaf_truncations'])
        assert len(leaves) == len(original_inputs['leaves']) == memory_manifest['leaf_count']
        for leaf, original_leaf in zip(leaves, original_inputs['leaves']):
            assert all(leaf[k] == original_leaf[k] for k in ('image_id', 'memory_id', 'dialogue', 'caption'))
        memory_count += len(leaves)
        texts = [leaf['dialogue'] + '\nMemory description: ' + leaf['caption'] for leaf in leaves]
        memory_ids = [leaf['memory_id'] for leaf in leaves]
        assert len(memory_ids) == len(set(memory_ids))
        assert all(text == source_memories[mid]['reader_text'] for mid, text in zip(memory_ids, texts))
        with np.load(features_path) as features:
            leaf_vectors = features['bge_leaf'].copy()
            old_vectors = features['bge_query'].copy()
        assert leaf_vectors.shape == (len(leaves), 384) and leaf_vectors.dtype == np.float64
        assert np.allclose(np.linalg.norm(leaf_vectors, axis=1), 1, atol=1e-7)
        old_queries = {q['id']: (q, old_vectors[i]) for i, q in enumerate(original_inputs['questions'])}
        for q in (q for q in questions if q['task'] == task):
            qid = q['id']
            old_q, old_vector = old_queries[qid]
            assert old_q['retrieval_query'] == mcq[qid]['retrieval_query']
            old_scores = leaf_vectors @ old_vector
            old_ref = source_retrieval[qid]['methods']['baseline_question_only']
            assert old_ref['memory_ids'] == memory_ids
            score_error = float(np.max(np.abs(old_scores - np.asarray(old_ref['raw']))))
            max_old_score_error = max(max_old_score_error, score_error)
            assert score_error < 1e-6, (qid, score_error)
            old_top = [memory_ids[i] for i in np.argsort(-old_scores, kind='stable')[:5]]
            assert old_top == source_reader[qid]['top_memory_ids']
            old_context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                f'Memory {rank}: {texts[memory_ids.index(mid)]}' for rank, mid in enumerate(old_top, 1))
            assert source_reader[qid]['reader_text'].split('\n\nQuestion:\n')[0] == old_context
            vector = encoded[qid] if qid in encoded else old_vector
            scores = leaf_vectors @ vector
            order = np.argsort(-scores, kind='stable')
            top = order[:5]
            top_ids = [memory_ids[i] for i in top]
            if qid not in encoded:
                assert top_ids == old_top
                unchanged_matches += 1
            context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                f'Memory {rank}: {texts[i]}' for rank, i in enumerate(top, 1))
            user_text = context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
            retrieval.append({'id': qid, 'task': task, 'retrieval_query': q['retrieval_query'],
                              'top_memory_ids': top_ids, 'top_scores': [float(scores[i]) for i in top],
                              'full_rank_memory_ids': [memory_ids[i] for i in order],
                              'full_rank_scores': [float(scores[i]) for i in order],
                              'query_vector_origin': 'new_open_question_encoding' if qid in encoded else 'unchanged_query_frozen_vector',
                              'mcq_top_memory_ids': old_top, 'top5_changed_from_mcq': top_ids != old_top})
            reader.append({'id': qid, 'task': task, 'point': q['point'], 'question': q['question'],
                           'top_memory_ids': top_ids, 'context': context, 'query_image_paths': q['query_image_paths'],
                           'system': SYSTEM, 'user_text': user_text,
                           'payload_sha256': payload_hash(SYSTEM, user_text, q['query_image_paths'])})
        print(json.dumps({'task': task, 'questions_prepared': len(old_queries), 'leaves': len(leaves)}), flush=True)
    assert memory_count == 438 and unchanged_matches == 137 and len(reader) == len(retrieval) == 371
    source_hashes = {str(p): sha(p) for p in sorted(source_paths)}
    output.mkdir(parents=True)
    (output / 'inputs').mkdir()
    write_rows(output / 'inputs/questions.jsonl', questions)
    write_rows(output / 'inputs/gold.jsonl', gold)
    write_rows(output / 'retrieval.jsonl', retrieval)
    write_rows(output / 'reader_manifest.jsonl', reader)
    audit = {'passed': True, 'questions': 371, 'memory_leaves': memory_count, 'domains': domains,
             'original_mcq_correct': 184, 'original_mcq_accuracy': 184 / 371,
             'unchanged_question_text': 137, 'changed_question_text': changed_count,
             'open_gold_differs_from_mcq_correct_option': sum(r['open_gold_differs_from_mcq_correct_option'] for r in differences),
             'query_image_questions': sum(bool(q['query_image_paths']) for q in questions),
             'unique_query_images': len({p for q in questions for p in q['query_image_paths']}),
             'all_open_images_match_original_mcq': True, 'all_open_clues_match_original_mcq': True,
             'old_mcq_top5_reproduced': 371, 'max_old_mcq_score_error': max_old_score_error,
             'unchanged_queries_top5_reproduced': unchanged_matches,
             'top5_changed_from_mcq': sum(r['top5_changed_from_mcq'] for r in retrieval),
             'all_contexts_exactly_five_memories': True, 'reader_has_no_options_or_gold_fields': True,
             'query_max_tokens': max(query_tokens), 'query_truncations': 0,
             'allowed_frozen_leaf_truncations': allowed_truncations, 'question_differences': differences}
    write(output / 'input_validation.json', audit)
    protocol = {'stage': 'memeye_open_semrag_inputs', 'dataset': 'MemEye', 'evaluation_type': 'open_ended',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'questions': 371,
                'method': 'semantic_rag', 'code_method': 'natural/full_scan/online_weighting=none',
                'original_source_root': str(ORIGINAL), 'original_condition': 'baseline_question_only',
                'original_mcq_correct': 184, 'original_mcq_questions': 371, 'top_k': 5,
                'memory_leaves': 438, 'memory_text': 'dialogue + newline + Memory description: + full Qwen caption',
                'memory_root': str(STORES), 'text_encoder': metadata['text_encoder'],
                'query_prefix': metadata['query_prefix'], 'retrieval_device': device,
                'memory_vectors': 'Frozen verified BGE float64 leaf vectors; no memory rewriting or rebuilding.',
                'query_vectors': '137 identical queries reuse frozen vectors; 234 changed Open questions freshly encoded.',
                'query_policy': 'Official _Open.json question + identical cached Qwen question-image caption suffix; no options, clues, gold, or history added to query.',
                'ranking': 'Within-domain full scan, normalized cosine dot product; descending stable source-order ties; top5.',
                'reader_system': SYSTEM,
                'reader_policy': 'Ranked top5 original dialogue+full caption; official Open question; original question images. Gold held in separate file.',
                'judge_inputs_policy': 'Question + official Open answer + generated response; never MCQ option outputs.',
                'source_sha256': source_hashes,
                'artifact_sha256': {str(p.relative_to(output)): sha(p) for p in [
                    output / 'inputs/questions.jsonl', output / 'inputs/gold.jsonl', output / 'retrieval.jsonl',
                    output / 'reader_manifest.jsonl', output / 'input_validation.json']}}
    write(output / 'protocol.json', protocol)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('question_differences', 'allowed_frozen_leaf_truncations', 'domains')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/memeye_qwen_semrag_open_20261105')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    prepare(args.output, args.device)
