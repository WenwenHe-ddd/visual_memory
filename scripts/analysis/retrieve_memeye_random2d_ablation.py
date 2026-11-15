#!/usr/bin/env python3
"""Paired basis ablation using frozen Qwen captions and exact text-side scores.

Only rebuilt master visual centers affect candidate selection. This stage never
reads original images, invokes a model, or accesses labels before ranking.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
CACHE = ROOT / 'reports/all_master_answer_query_20261021'
DEFAULT_OUTPUT = ROOT / 'reports/memeye_qwen_random2d_ablation_20261105'
CONDITIONS = ['pca_2d', *[f'random_2d_seed_{seed}' for seed in range(10)]]


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.isfinite(x).all() and np.all(norms > 0)
    return x / norms


def centers(graph, sums):
    ids = [key for m in graph['masters'] for key in m['member_ids']]
    assert len(ids) == len(set(ids)) == len(graph['image_nodes'])
    assert set(ids) == {n['id'] for n in graph['image_nodes']}
    for m in graph['masters']:
        ref = m['visual_mean_ref']
        assert m['n'] == len(m['member_ids']) > 0
        assert ref['path'] == 'master_sums.npy' and 0 <= ref['row'] < len(sums)
        assert abs(ref['scale'] - 1 / m['n']) < 1e-15
    return np.array([sums[m['visual_mean_ref']['row']] * m['visual_mean_ref']['scale']
                     for m in graph['masters']])


def metrics(rows):
    targets = sum(len(r['gold_ids']) for r in rows)
    hits = sum(r['gold_hits_top5'] for r in rows)
    return {'questions': len(rows), 'gold_references': targets, 'gold_hits_top5': hits,
            'recall_at_5': hits / targets,
            'questions_with_gold_hit': sum(r['gold_hits_top5'] > 0 for r in rows)}


def verify_cache():
    protocol = read(CACHE / 'protocol.json')
    originals = {r['id']: r for r in lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')}
    historical = {r['question_id']: r for r in lines(CACHE / 'retrieval_results.jsonl')}
    candidates = {r['id']: r for r in lines(CACHE / 'candidates.jsonl')}
    assert len(originals) == len(historical) == len(candidates) == 371
    tracked = {str(p): sha(p) for p in [CACHE / 'protocol.json', CACHE / 'retrieval_results.jsonl',
               CACHE / 'candidates.jsonl', SOURCE / 'manifest.json',
               ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl', Path(__file__)]}
    domains, drift = {}, []
    for path, digest in protocol['source_sha256'].items():
        if not Path(path).exists() or sha(path) != digest:
            drift.append({'path': path, 'recorded_sha256': digest,
                          'current_sha256': sha(path) if Path(path).exists() else None})
    # Data used here must still match; the moved QA file is verified by content
    # field-for-field below. Historical code drift is recorded, not hidden.
    for domain_entry in read(SOURCE / 'manifest.json')['domains']:
        domain = domain_entry['domain']
        directory = SOURCE / domain
        for relative, digest in domain_entry['sha256'].items():
            assert sha(directory / relative) == digest, (domain, relative)
        for name in ('inputs.json', 'protocol.json', 'features.npz', 'memory/graph.json', 'memory/master_sums.npy'):
            p = directory / name
            assert sha(p) == protocol['source_sha256'][str(p)], p
            tracked[str(p)] = sha(p)
        inputs, graph = read(directory / 'inputs.json'), read(directory / 'memory/graph.json')
        with np.load(directory / 'features.npz', allow_pickle=False) as archive:
            features = {key: archive[key] for key in ('bge_query', 'bge_leaf')}
        cache_path = CACHE / 'domains' / domain / 'score_matrices.npz'
        tracked[str(cache_path)] = sha(cache_path)
        with np.load(cache_path, allow_pickle=False) as archive:
            cache = {key: archive[key] for key in archive.files}
        sums = np.load(directory / 'memory/master_sums.npy', allow_pickle=False)
        means = centers(graph, sums)
        clip_scores = np.einsum('md,qcd->qmc', normalize(means), cache['clip_candidate_vectors'])
        np.testing.assert_allclose(clip_scores, cache['clip_master_candidate'], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(np.argmax(clip_scores, axis=2), cache['selected_choice_indices'])
        all_scores = np.concatenate([features['bge_query'][:, None, :],
                                     cache['bge_augmented_query_vectors']], axis=1) @ features['bge_leaf'].T
        np.testing.assert_allclose(all_scores, cache['bge_all_query_leaf'], atol=1e-12, rtol=0)
        assert [n['id'] for n in graph['image_nodes']] == [n['image_id'] for n in inputs['leaves']]
        for leaf, node in zip(inputs['leaves'], graph['image_nodes']):
            assert leaf['caption'] == node['caption'] and leaf['dialogue'] == node['dialogue']
            assert leaf['retrieval_text'] == node['dialogue'] + '\nMemory description: ' + node['caption']
        for qi, q in enumerate(inputs['questions']):
            old, orig, candidate = historical[q['id']], originals[q['id']], candidates[q['id']]
            for key in ('question', 'options', 'query_image_paths'):
                assert q[key] == old[key] == orig[key], (q['id'], key)
            for key in ('gold_choice', 'gold_ids'):
                assert q[key] == old['evaluation'][key] == orig[key]
            assert candidate['question'] == q['question'] and candidate['options'] == q['options']
            assert candidate['candidate_statements'] == old['candidate_statements']
            assert [m['id'] for m in graph['masters']] == [m['master_id'] for m in old['masters']]
            selected = all_scores[qi, 1 + cache['selected_choice_indices'][qi, cache['leaf_master_indices']],
                                  np.arange(len(inputs['leaves']))]
            np.testing.assert_allclose(selected, cache['selected_leaf_scores'][qi], atol=1e-12, rtol=0)
            order = np.argsort(-selected, kind='stable')[:5]
            assert [inputs['leaves'][i]['memory_id'] for i in order] == old['conditions'][1]['top5_memory_ids']
        domains[domain] = (inputs, graph, means, cache)
    assert sum(len(x[0]['questions']) for x in domains.values()) == 371
    assert sum(len(x[0]['leaves']) for x in domains.values()) == 438
    return domains, originals, historical, tracked, drift


def run(output):
    if (output / 'retrieval_results.jsonl').exists():
        raise FileExistsError('Retrieval output already exists; do not overwrite a signed experiment')
    domains, originals, historical, tracked, drift = verify_cache()
    manifest_path = output / 'offline/manifest.json'
    assert manifest_path.is_file(), 'Wait for all offline builds to complete'
    assert read(manifest_path)['completed'] is True
    tracked[str(manifest_path)] = sha(manifest_path)
    records, master_rows, comparison_rows = [], [], []
    pca_records, pca_means_by_domain, pca_choices_by_domain = {}, {}, {}
    historical_pca_top5_matches = 0
    for condition in CONDITIONS:
        for domain, (inputs, frozen_graph, _, cache) in domains.items():
            directory = output / 'offline' / condition / domain
            graph = read(directory / 'memory/graph.json')
            sums = np.load(directory / 'memory/master_sums.npy', allow_pickle=False)
            for name in ('memory/graph.json', 'memory/basis.npy', 'memory/master_sums.npy', 'protocol.json'):
                p = directory / name
                tracked[str(p)] = sha(p)
            nodes, masters, leaves = graph['image_nodes'], graph['masters'], inputs['leaves']
            assert [n['id'] for n in nodes] == [l['image_id'] for l in leaves]
            assert {frozenset(m['member_ids']) for m in masters} == {frozenset(m['member_ids']) for m in frozen_graph['masters']}
            by_members = {frozenset(m['member_ids']): i for i, m in enumerate(masters)}
            reference_order = [by_members[frozenset(m['member_ids'])] for m in frozen_graph['masters']]
            means = centers(graph, sums)
            leaf_lookup = {key: i for i, m in enumerate(masters) for key in m['member_ids']}
            leaf_master = np.array([leaf_lookup[n['id']] for n in nodes])
            for node, leaf in zip(nodes, leaves):
                assert node['caption'] == leaf['caption'] and node['dialogue'] == leaf['dialogue']
            scores = np.einsum('md,qcd->qmc', normalize(means), cache['clip_candidate_vectors'])
            choices = np.argmax(scores, axis=2)
            if condition == 'pca_2d':
                pca_means_by_domain[domain] = means[reference_order]
                pca_choices_by_domain[domain] = choices[:, reference_order]
            else:
                reference = pca_means_by_domain[domain]
                cosines = np.sum(normalize(means[reference_order]) * normalize(reference), axis=1)
                for j, m in enumerate(frozen_graph['masters']):
                    master_rows.append({'condition': condition, 'domain': domain, 'reference_master_id': m['id'],
                        'n': m['n'], 'cosine_to_pca_master': float(cosines[j]),
                        'l2_to_pca_master': float(np.linalg.norm(means[reference_order[j]] - reference[j]))})
            for qi, q in enumerate(inputs['questions']):
                selected = cache['bge_all_query_leaf'][qi, 1 + choices[qi, leaf_master], np.arange(len(leaves))]
                order = np.argsort(-selected, kind='stable')[:5]
                top = [{**{key: leaves[i][key] for key in ('image_id', 'memory_id', 'filename', 'dialogue', 'caption')},
                        'master_id': nodes[i]['master_id'], 'rank': rank, 'score': float(selected[i]),
                        'master_choice': 'ABCD'[choices[qi, leaf_master[i]]]} for rank, i in enumerate(order, 1)]
                row = {'condition': condition, 'question_id': q['id'], 'domain': domain,
                       **{key: q[key] for key in ('question', 'options', 'query_image_paths')},
                       'top5': top, 'top5_memory_ids': [l['memory_id'] for l in top],
                       'master_choices': {m['id']: 'ABCD'[choices[qi, i]] for i, m in enumerate(masters)},
                       'master_candidate_scores': scores[qi].tolist(),
                       'gold_choice': q['gold_choice'], 'gold_ids': q['gold_ids']}
                assert len(set(row['top5_memory_ids'])) == 5
                row['gold_hits_top5'] = len(set(row['top5_memory_ids']) & set(row['gold_ids']))
                if condition == 'pca_2d':
                    pca_records[q['id']] = row
                    historical_pca_top5_matches += row['top5_memory_ids'] == historical[q['id']]['conditions'][1]['top5_memory_ids']
                else:
                    base = pca_records[q['id']]
                    comparison_rows.append({'condition': condition, 'question_id': q['id'], 'domain': domain,
                        'pca_hits': base['gold_hits_top5'], 'random_hits': row['gold_hits_top5'],
                        'gold_count': len(row['gold_ids']), 'same_top5_order': base['top5_memory_ids'] == row['top5_memory_ids'],
                        'same_top5_set': set(base['top5_memory_ids']) == set(row['top5_memory_ids']),
                        'top5_overlap_fraction': len(set(base['top5_memory_ids']) & set(row['top5_memory_ids'])) / 5,
                        'matching_master_choices': int(np.sum(choices[qi, reference_order] == pca_choices_by_domain[domain][qi])),
                        'masters': len(masters)})
                records.append(row)
        print(json.dumps({'condition': condition, **metrics([r for r in records if r['condition'] == condition])}), flush=True)
    assert len(records) == 4081 and len({(r['condition'], r['question_id']) for r in records}) == 4081
    overall = {condition: metrics([r for r in records if r['condition'] == condition]) for condition in CONDITIONS}
    domain_metrics = {condition: {domain: metrics([r for r in records if r['condition'] == condition and r['domain'] == domain])
                                 for domain in domains} for condition in CONDITIONS}
    for condition in CONDITIONS[1:]:
        paired = [r for r in comparison_rows if r['condition'] == condition]
        overall[condition].update(
            same_top5_order=sum(r['same_top5_order'] for r in paired),
            same_top5_set=sum(r['same_top5_set'] for r in paired),
            mean_top5_overlap_fraction=float(np.mean([r['top5_overlap_fraction'] for r in paired])),
            matching_master_choices=sum(r['matching_master_choices'] for r in paired),
            total_master_choices=sum(r['masters'] for r in paired))
    recalls = [overall[c]['recall_at_5'] for c in CONDITIONS[1:]]
    summary = {'completed': True, 'conditions': overall, 'by_domain': domain_metrics,
               'random_seeds': list(range(10)), 'random_recall_mean': float(np.mean(recalls)),
               'random_recall_std_sample': float(np.std(recalls, ddof=1)),
               'random_recall_min': min(recalls), 'random_recall_max': max(recalls),
               'pca_replay_ordered_top5_matches_historical': historical_pca_top5_matches,
               'metric': 'Micro source-memory recall: total gold memory IDs in Top5 / total gold references (1041)'}
    for path, digest in tracked.items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'retrieval_results.jsonl').open('w') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    dump(output / 'retrieval_summary.json', summary)
    csv_write(output / 'retrieval_per_question.csv', comparison_rows)
    csv_write(output / 'master_comparison.csv', master_rows)
    csv_write(output / 'retrieval_metrics.csv', [{'condition': c, **metrics([r for r in records if r['condition'] == c])} for c in CONDITIONS])
    dump(output / 'retrieval_protocol.json', {
        'conditions': CONDITIONS, 'source_sha256': tracked,
        'historical_protocol_drift': drift,
        'drift_handling': 'Moved QA source verified field-for-field against current independent_visual_inputs; historical code drift recorded. All used frozen input data hashes match.',
        'cache_validation': 'All 371 historical ordered Top5 reproduced; candidate CLIP scores and all BGE scores reconstructed from frozen vectors.',
        'changed_variable': 'Fixed 2D basis used in streamed master updates',
        'same_caption_partitions': True, 'retrieval': 'multi_path', 'online_weighting': 'none',
        'visibility': 'Full final memory of the same domain, matching the original experiment',
        'candidate_statements': 'Frozen original statements; query expansion only, never reader evidence',
        'memory_text': 'dialogue + newline + Memory description: + complete Qwen caption',
        'encoder_calls': 0, 'api_calls': 0, 'original_images_read': 0,
        'results_sha256': sha(output / 'retrieval_results.jsonl'),
        'summary_sha256': sha(output / 'retrieval_summary.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-cache-only', action='store_true')
    args = parser.parse_args()
    if args.verify_cache_only:
        domains, originals, _, _, drift = verify_cache()
        print(json.dumps({'passed': True, 'domains': len(domains), 'questions': len(originals), 'historical_drift': drift}, indent=2))
    else:
        run(args.output.resolve())
