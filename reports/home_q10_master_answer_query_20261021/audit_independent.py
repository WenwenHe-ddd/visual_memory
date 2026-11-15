"""Independent arithmetic, provenance, and text-input audit of the Q10 pilot.

This checker does not import the experiment's scoring implementation. It
reconstructs the stored master means, encodes the fixed candidate/query text
again, and recomputes all 100 CLIP and 445 BGE comparisons.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    inputs = read(SOURCE / 'inputs.json')
    graph = read(SOURCE / 'memory/graph.json')
    protocol = read(OUT / 'protocol.json')
    candidates = read(OUT / 'candidates.json')
    result = read(OUT / 'retrieval_results.json')
    saved = np.load(OUT / 'score_matrices.npz', allow_pickle=False)
    source_vectors = np.load(SOURCE / 'features.npz', allow_pickle=False)
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def close(name, actual, expected, tolerance=1e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        check(name + ':shape', actual.shape == expected.shape)
        delta = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        check(name + ':values', np.isfinite(actual).all() and delta <= tolerance)
        return delta

    for path, expected in protocol['source_sha256'].items():
        check('source hash: ' + path, sha(path) == expected)
    check('candidates hash', sha(OUT / 'candidates.json') == protocol['candidates_sha256'])
    questions, leaves, masters = inputs['questions'], inputs['leaves'], inputs['masters']
    qi = next(i for i, row in enumerate(questions) if row['id'] == result['question_id'])
    question = questions[qi]
    check('source question index', qi == protocol['baseline_source_question_index'])
    check('question text unchanged', result['question'] == candidates['question'] == question['question'])
    check('source options unchanged', result['options'] == candidates['options'] == question['options'])
    check('no query image', not question['query_image_paths'])
    check('25 masters, 89 leaves', len(masters) == 25 and len(leaves) == 89)
    check('master order', [m['id'] for m in masters] == [m['id'] for m in graph['masters']])
    check('leaf order', [leaf['image_id'] for leaf in leaves] == [node['id'] for node in graph['image_nodes']])
    expected_candidates = {
        'A': 'The window desk has a monitor, and the longer desk along the right wall has a monitor.',
        'B': 'The window desk has no monitor, and the longer desk along the right wall has no monitor.',
        'C': 'The window desk has a monitor, and the longer desk along the right wall has no monitor.',
        'D': 'The window desk has no monitor, and the longer desk along the right wall has a monitor.',
    }
    check('fixed parallel candidates', candidates['candidate_statements'] == expected_candidates)
    letters = list(expected_candidates)
    queries = [question['question'] + '\nCandidate answer: ' + expected_candidates[k] for k in letters]
    check('query appended correctly', list(candidates['augmented_queries'].values()) == queries)
    leaf_lookup = {leaf['image_id']: i for i, leaf in enumerate(leaves)}
    master_lookup = {master['id']: i for i, master in enumerate(masters)}
    leaf_master = np.full(len(leaves), -1, dtype=np.int64)
    for mi, master in enumerate(graph['masters']):
        check('master cardinality ' + master['id'], master['n'] == len(master['member_ids']))
        check('master members ' + master['id'], master['member_ids'] == masters[mi]['member_ids'])
        for member in master['member_ids']:
            li = leaf_lookup[member]
            check('unique membership ' + member, leaf_master[li] == -1)
            leaf_master[li] = mi
    check('all leaves assigned', np.all(leaf_master >= 0))
    check('saved membership', np.array_equal(saved['leaf_master_indices'], leaf_master))
    for li, (leaf, node) in enumerate(zip(leaves, graph['image_nodes'])):
        check('leaf text ' + leaf['image_id'], leaf['retrieval_text'] == node['dialogue'] + '\nMemory description: ' + node['caption'])
        check('leaf master ' + leaf['image_id'], master_lookup[leaf['master_id']] == leaf_master[li])

    sums = np.load(SOURCE / 'memory/master_sums.npy', allow_pickle=False)
    means = []
    for master in graph['masters']:
        ref = master['visual_mean_ref']
        check('actual mean reference ' + master['id'], ref['path'] == 'master_sums.npy' and ref['scale'] == 1 / master['n'])
        means.append(sums[ref['row']] / master['n'])
    clip_scores = unit(means) @ saved['clip_candidate_vectors'].T
    clip_delta = close('100 CLIP similarities', clip_scores, saved['clip_master_candidate'])
    choices = np.argmax(clip_scores, axis=1)
    check('25 stable argmax choices', np.array_equal(choices, saved['selected_choice_indices']))
    all_queries = np.vstack((source_vectors['bge_query'][qi], saved['bge_augmented_query_vectors']))
    all_scores = all_queries @ source_vectors['bge_leaf'].T
    bge_delta = close('445 BGE similarities', all_scores, saved['bge_all_query_leaf'])
    chosen = np.array([all_scores[1 + choices[mi], li] for li, mi in enumerate(leaf_master)])
    close('89 own-master scores', chosen, saved['selected_leaf_scores'])
    for mi, master in enumerate(result['masters']):
        check('master id ' + str(mi), master['master_id'] == masters[mi]['id'])
        check('master argmax ' + str(mi), master['selected_choice'] == letters[choices[mi]])
        check('master query ' + str(mi), master['augmented_query'] == queries[choices[mi]])
        close('master four scores ' + str(mi), list(master['clip_candidate_scores'].values()), clip_scores[mi])

    top5 = {}
    source_text = {leaf['memory_id']: leaf['retrieval_text'] for leaf in leaves}
    for condition, scores in zip(result['conditions'], (all_scores[0], chosen)):
        order = sorted(range(len(leaves)), key=lambda i: (-float(scores[i]), i))
        top5[condition['name']] = order[:5]
        check('stable global top5 ' + condition['name'], condition['top5_leaf_indices'] == order[:5])
        check('top5 IDs ' + condition['name'], condition['top5_memory_ids'] == [leaves[i]['memory_id'] for i in order[:5]])
        close('top5 scores ' + condition['name'], condition['top5_scores'], scores[order[:5]])
        ranks = {li: rank for rank, li in enumerate(order, 1)}
        rank_field = 'baseline_rank' if condition['name'] == 'baseline_question_only' else 'expanded_rank'
        check('all 89 ranks ' + condition['name'], all(row[rank_field] == ranks[row['leaf_index']] for row in result['leaves']))
        for memory in condition['reader_memories']:
            check('unmodified reader evidence ' + condition['name'] + memory['memory_id'], memory['reader_text'] == source_text[memory['memory_id']])
        context = '\n\n'.join(f'Memory {i}: {source_text[mid]}' for i, mid in enumerate(condition['top5_memory_ids'], 1))
        check('exact reader context ' + condition['name'], context == condition['reader_context'])
        check('no candidate prompt in evidence ' + condition['name'], 'Candidate answer:' not in context)

    # Static data-flow check: model scores and rankings are assigned before
    # the first source gold label is accessed in the producer's main routine.
    tree = ast.parse((ROOT / 'scripts/analysis/probe_master_answer_query.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    gold_reads = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Subscript)
                  and isinstance(node.slice, ast.Constant) and node.slice.value in ('gold_ids', 'gold_choice')]
    rank_writes = [node.lineno for node in ast.walk(function) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'expanded_order' for t in node.targets)]
    check('gold first accessed after final rankings', min(gold_reads) > max(rank_writes))
    check('no image encoding in pilot', all(not isinstance(node, ast.Name) or node.id not in ('Image', 'CLIPVisionModelWithProjection', 'CLIPImageProcessor') for node in ast.walk(function)))

    # Independent repeat of the four fixed candidate and four expanded-query
    # encodings. Cached source leaves are independently encoded from retained
    # text to establish that they are dialogue + full caption, not captions only.
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    clip_path = Path(protocol['image_encoder']['model_name'])
    for filename, expected in protocol['image_encoder']['files_sha256'].items():
        check('CLIP checkpoint hash ' + filename, sha(clip_path / filename) == expected)
    config = AutoConfig.from_pretrained(str(clip_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(str(clip_path), config=config.text_config, local_files_only=True)
    model.eval()
    tokenized = tokenizer(list(expected_candidates.values()), padding=True, return_tensors='pt', truncation=False)
    with torch.inference_mode():
        candidate_repeat = unit(model(**tokenized).text_embeds.double().numpy())
    repeat_clip_delta = close('repeat CLIP four embeddings', candidate_repeat, saved['clip_candidate_vectors'], 1e-7)
    del model
    bge_path = Path(protocol['text_encoder']['model_path'])
    for filename, expected in protocol['text_encoder']['files_sha256'].items():
        check('BGE checkpoint hash ' + filename, sha(bge_path / filename) == expected)
    encoder = SentenceTransformer(str(bge_path), device='cpu', local_files_only=True)
    query_texts = [protocol['query_prefix'] + query for query in queries]
    repeated_queries = unit(encoder.encode(query_texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False, prompt=''))
    repeat_query_delta = close('repeat BGE four queries', repeated_queries, saved['bge_augmented_query_vectors'], 1e-7)
    leaf_texts = [leaf['retrieval_text'] for leaf in leaves]
    repeated_leaves = unit(encoder.encode(leaf_texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False, prompt=''))
    repeat_leaf_delta = close('repeat BGE 89 leaves', repeated_leaves, source_vectors['bge_leaf'], 2e-6)
    lengths = [len(ids) for ids in encoder.tokenizer(query_texts + leaf_texts, truncation=False)['input_ids']]
    check('no BGE truncation', max(lengths) <= encoder.max_seq_length)
    check('no CLIP truncation', tokenized['input_ids'].shape[1] <= config.text_config.max_position_embeddings)

    validation = {
        'passed': True,
        'checks_passed': len(checks),
        'independent_implementation': True,
        'clip_comparisons_recomputed': 100,
        'bge_comparisons_recomputed': 445,
        'master_choices_verified': 25,
        'own_master_leaf_scores_verified': 89,
        'exact_top5_conditions_verified': 2,
        'clip_score_max_abs_difference': clip_delta,
        'bge_score_max_abs_difference': bge_delta,
        'repeat_clip_embedding_max_abs_difference': repeat_clip_delta,
        'repeat_bge_query_embedding_max_abs_difference': repeat_query_delta,
        'repeat_bge_leaf_embedding_max_abs_difference': repeat_leaf_delta,
        'master_choice_counts': {letter: int(np.count_nonzero(choices == i)) for i, letter in enumerate(letters)},
        'same_top5_members': set(top5['baseline_question_only']) == set(top5['master_answer_query']),
        'same_top5_order': top5['baseline_question_only'] == top5['master_answer_query'],
        'gold_first_read_after_ranking': True,
        'reader_memories_exact_original_text': True,
        'old_images_or_visual_leaf_embeddings_used': False,
        'limits': [
            'Single question. This validates the specified computation, not domain-level answer accuracy.',
            'All ten Top5 entries belong to the same master and use hypothesis B; this case alone cannot isolate visual selection benefit from fixed B expansion.',
            'Raw BGE similarities arise from different queries across masters; no probability calibration claimed.',
        ],
        'checks': checks,
    }
    (OUT / 'validation_independent.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in validation.items() if k != 'checks'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
