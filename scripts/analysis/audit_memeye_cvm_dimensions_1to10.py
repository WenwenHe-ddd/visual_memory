#!/usr/bin/env python3
"""Independent dual-ridge/ranking and payload audit of PCA dimension ablation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.analysis.evaluate_adaptive_changed_qa import digest, dump, lines, read, reader_text, sha


def audit(output, inputs_only=False):
    conditions = [f'pca_{d}d' for d in range(1, 11)]
    source = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
    cache_root = ROOT / 'reports/all_master_answer_query_20261021/domains'
    retrieved = lines(output / 'retrieval_results.jsonl')
    index = {(r['question_id'], r['condition']): r for r in retrieved}
    assert len(index) == len(retrieved) == 3710
    p = read(output / 'reader/protocol.json')
    assert digest({k: v for k, v in p.items() if k != 'experiment_signature'}) == p['experiment_signature']
    for stage in ('offline_manifest.json', 'retrieval_manifest.json'):
        stage_manifest = read(output / stage)
        for path, expected in {**stage_manifest['source_sha256'], **stage_manifest['output_sha256']}.items():
            assert sha(path) == expected, path
    calibration = read(output / 'calibration/manifest.json')
    for path, expected in {**calibration['source_sha256'], **calibration['output_sha256']}.items():
        assert sha(path) == expected, path
    for path, expected in {**p['source_sha256'], **p['model_source_sha256']}.items():
        assert sha(path) == expected, path
    max_score_delta, checked = 0., 0
    for record in read(source / 'manifest.json')['domains']:
        domain = record['domain']
        original_graph = read(source / domain / 'memory/graph.json')
        data = read(source / domain / 'inputs.json')
        with np.load(source / domain / 'features.npz') as archive:
            leaf_vectors = archive['bge_leaf'].copy()
        with np.load(cache_root / domain / 'score_matrices.npz') as archive:
            query_vectors = archive['bge_augmented_query_vectors'].copy()
            candidate_vectors = archive['clip_candidate_vectors'].copy()
        leaf_vectors = leaf_vectors.astype(float)
        query_vectors = query_vectors.astype(float)
        leaf_vectors /= np.linalg.norm(leaf_vectors, axis=1, keepdims=True)
        query_vectors /= np.linalg.norm(query_vectors, axis=2, keepdims=True)
        basis10 = np.load(output / 'calibration' / (domain + '.npy'))
        for d, condition in zip(range(1, 11), conditions):
            folder = output / 'offline' / domain / condition
            graph = read(folder / 'graph.json')
            sums = np.load(folder / 'master_sums.npy')
            basis = np.load(folder / 'basis.npy')
            np.testing.assert_array_equal(basis, basis10[:, :d])
            np.testing.assert_allclose(basis.T @ basis, np.eye(d), rtol=0, atol=1e-12)
            assert [m['member_ids'] for m in graph['masters']] == [m['member_ids'] for m in original_graph['masters']]
            for n, original in zip(graph['image_nodes'], original_graph['image_nodes']):
                assert set(n) == {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'}
                assert len(n['visual_coords']) == d
                for key in ('id', 'caption', 'dialogue', 'timestamp', 'master_id'):
                    assert n[key] == original[key]
            for qi, q in enumerate(data['questions']):
                scores = np.empty(len(leaf_vectors))
                choices = {}
                for mi, master in enumerate(graph['masters']):
                    ids = [i for i, n in enumerate(graph['image_nodes']) if n['master_id'] == master['id']]
                    xx = leaf_vectors[ids]
                    visual = sums[mi] / np.linalg.norm(sums[mi])
                    choice = int(np.argmax(candidate_vectors[qi] @ visual))
                    choices[master['id']] = 'ABCD'[choice]
                    query = query_vectors[qi, choice]
                    raw = xx @ query
                    corrected = raw.copy()
                    if len(xx) > 1:
                        center = xx - xx.mean(0)
                        adjusted = query - center.T @ np.linalg.solve(
                            center @ center.T + (len(xx)-1)*.03*np.eye(len(xx)), center @ query)
                        white = xx @ adjusted
                        # Independent population standardization (no production helpers).
                        std = float(white.std())
                        threshold = 32*np.finfo(float).eps*max(1., float(np.abs(white).max()))
                        raw_threshold = 32*np.finfo(float).eps*max(1., float(np.abs(raw).max()))
                        if std > threshold and raw.std() > raw_threshold:
                            z = (white-white.mean())/std
                            if len(xx) == 2:
                                z = np.sign(white[0]-white[1])*np.array([1., -1.])
                            corrected += raw.std()*np.tanh(z)
                    scores[ids] = corrected
                order = np.argsort(-scores, kind='stable')[:5]
                row = index[q['id'], condition]
                assert row['master_choices'] == choices
                assert row['top5_memory_ids'] == [data['leaves'][i]['memory_id'] for i in order]
                delta = float(np.max(np.abs(scores[order]-row['top5_scores'])))
                assert delta < 2e-10
                max_score_delta = max(max_score_delta, delta)
                checked += 1
    questions = {r['id']: r for r in lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in lines(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl')}
    manifests = lines(output / 'reader/qa_manifest.jsonl')
    assert len(manifests) == len({r['record_id'] for r in manifests}) == 3710
    for m in manifests:
        q = questions[m['id']]
        assert m['top_memory_ids'] == index[m['id'], m['condition']]['top5_memory_ids']
        text = reader_text(q, m['top_memory_ids'], memories)
        images = [{'path': path, 'sha256': sha(path)} for path in q['query_image_paths']]
        assert m['reader_text'] == text and m['query_images'] == images
        assert m['payload_sha256'] == digest({'system': p['reader_settings']['reader_system_prompt'], 'user_text': text, 'query_images': images})
    prior_root = ROOT / 'results/memeye_qwen_cvm_pca_dimensions_zscore_l0p03_a1_20261114'
    prior_rows = lines(prior_root / 'retrieval_results.jsonl')
    assert len(prior_rows) == 1484
    for old in prior_rows:
        current = index[old['question_id'], old['condition']]
        assert current['top5_memory_ids'] == old['top5_memory_ids']
        assert current['master_choices'] == old['master_choices']
        np.testing.assert_allclose(current['top5_scores'], old['top5_scores'], atol=1e-12, rtol=0)
    result = {'passed': True, 'inputs_only': inputs_only, 'checked_rankings': checked,
              'checked_payloads': len(manifests), 'max_independent_score_delta': max_score_delta,
              'nested_bases': True, 'caption_membership_and_text_unchanged': True,
              'previous_2d_to_5d_rankings_reproduced': len(prior_rows),
              'previous_retrieval_sha256': sha(prior_root / 'retrieval_results.jsonl'),
              'source_code_sha256': sha(__file__)}
    if not inputs_only:
        rows = lines(output / 'reader/responses.jsonl')
        by_id = {r['record_id']: r for r in rows}
        assert len(rows) == len(by_id) == 3710
        fresh = {}
        for r in rows:
            assert r['experiment_signature'] == p['experiment_signature']
            assert r['run_id'] == p['experiment_run_id'] and not r['historical_response_reuse']
            if r['fresh_inference_executed_for_this_record']:
                assert r['payload_sha256'] not in fresh
                fresh[r['payload_sha256']] = r
            else:
                prior = fresh[r['payload_sha256']]
                assert r['same_run_shared_inference_record_id'] == prior['record_id']
                assert r['predicted_choice'] == prior['predicted_choice']
            assert r['correct'] == (r['predicted_choice'] == r['gold_choice'])
        for m in manifests:
            r = by_id[m['record_id']]
            assert r['payload_sha256'] == m['payload_sha256'] and r['gold_choice'] == m['gold_choice']
        result['unique_fresh_inferences'] = len(fresh)
        result['correct_counts'] = {c: sum(r['correct'] for r in rows if r['condition'] == c) for c in conditions}
        for summary in read(output / 'summary.json')['conditions']:
            assert summary['correct'] == result['correct_counts'][summary['condition']]
            assert summary['questions'] == 371 and summary['accuracy'] == summary['correct']/371
        result['responses_sha256'] = sha(output / 'reader/responses.jsonl')
        prior_responses = lines(prior_root / 'reader/responses.jsonl')
        assert len(prior_responses) == 1484
        result['answer_changes_vs_previous_2d_to_5d_run'] = {
            c: sum(by_id[r['question_id'] + '::' + c]['predicted_choice'] != r['predicted_choice']
                   for r in prior_responses if r['condition'] == c)
            for c in ['pca_2d', 'pca_3d', 'pca_4d', 'pca_5d']}
    dump(output / ('independent_input_audit.json' if inputs_only else 'independent_audit.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/memeye_qwen_cvm_pca_dimensions_1to10_zscore_l0p03_a1_20261114')
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    audit(args.output, args.inputs_only)
