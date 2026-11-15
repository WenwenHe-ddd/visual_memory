#!/usr/bin/env python3
"""Paired PCA 1 through 10-D CVM + z-score experiment, with frozen Qwen inputs.

External calibration and arrival replay use the historical CPU CLIP encoder.
The existing dimension-generic PartitionReplay implements the same aggregate
update without modifying the production two-dimensional memory implementation.
Reader responses are fresh; only identical payloads within this run are shared.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
from scripts.analysis.build_memeye_random2d_ablation import preflight, SOURCE, CAPTION, HOME
from scripts.analysis.compare_caption_visual_dimensions import PartitionReplay
from scripts.analysis.retrieve_memeye_random2d_ablation import verify_cache, centers, normalize, metrics
from scripts.analysis.evaluate_adaptive_changed_qa import digest, dump, lines, read, reader_text, sha
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.graph.caption_ward_visual import _unit
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.online_zscore import ZScoreWeighting
from scripts.evaluation.paired_single_gpu_reader import run as run_reader

DIMS = tuple(range(1, 11))
CONDITIONS = tuple(f'pca_{d}d' for d in DIMS)
HISTORY = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
HIST_CONDITION = 'master_answer_query__l0p03_a1'
OUTPUT = ROOT / 'results/memeye_qwen_cvm_pca_dimensions_1to10_zscore_l0p03_a1_20261114'
ORIGINAL = ROOT / 'data/independent_visual_inputs'


def log(**value):
    print(json.dumps(value), flush=True)


def write_lines(path, rows):
    with Path(path).open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def check_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, path


def signed(value):
    return {**value, 'experiment_signature': digest(value)}


def verify_signature(value):
    unsigned = dict(value)
    signature = unsigned.pop('experiment_signature')
    assert digest(unsigned) == signature


def calibrate(domains, encoder, output):
    """Fit nested bases with exactly the original per-domain calibration sets."""
    target = output / 'calibration'
    target.mkdir(exist_ok=True)
    manifest = target / 'manifest.json'
    if manifest.exists():
        saved = read(manifest)
        check_hashes(saved['source_sha256'])
        check_hashes(saved['output_sha256'])
        return {d['name']: np.load(target / (d['name'] + '.npy')) for d in domains}
    by_hash = {}
    excluded = {}
    for domain in domains:
        excluded[domain['name']] = {row['sha256'] for row in domain['rows']}
        for row in domain['rows']:
            by_hash.setdefault(row['sha256'], row['source_path'])
    assert len(by_hash) == 433
    home_manifest = ROOT / 'reports/home_caption_ward_visual_20261020/input_manifest.json'
    home_cal = read(home_manifest)['calibration']
    assert [r['image_sha256'] for r in home_cal] == [h for h in sorted(by_hash) if h not in excluded[HOME]]
    accum = {d['name']: [np.zeros(512), np.zeros((512, 512)), 0] for d in domains}
    for index, image_hash in enumerate(sorted(by_hash), 1):
        assert sha(by_hash[image_hash]) == image_hash
        visual = np.asarray(encoder.encode(by_hash[image_hash]), dtype=np.float64)
        for domain in domains:
            name = domain['name']
            if image_hash in excluded[name]:
                continue
            # Home's older pilot normalized once more before moments.
            v = visual / np.linalg.norm(visual) if name == HOME else visual
            state = accum[name]
            state[0] += v
            state[1] += np.outer(v, v)
            state[2] += 1
        del visual, v
        if index % 25 == 0 or index == 433:
            log(stage='calibration', completed=index, total=433)
    bases, info = {}, {}
    for domain in domains:
        name = domain['name']
        total, second, n = accum.pop(name)
        selected = [h for h in sorted(by_hash) if h not in excluded[name]]
        assert n == len(selected) and not set(selected) & excluded[name]
        if name == HOME:
            covariance = (second - np.outer(total, total) / n) / (n - 1)
        else:
            assert selected == domain['protocol']['calibration']['calibration_sha256']
            covariance = second / n - np.outer(total / n, total / n)
        eig, vec = np.linalg.eigh((covariance + covariance.T) / 2)
        order = np.argsort(-eig, kind='stable')
        eig = np.maximum(eig[order], 0)
        basis = vec[:, order[:max(DIMS)]].copy()
        for col in range(max(DIMS)):
            if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
                basis[:, col] *= -1
        np.testing.assert_allclose(basis.T @ basis, np.eye(max(DIMS)), rtol=0, atol=1e-12)
        original = np.load(domain['directory'] / 'memory/basis.npy')
        delta = float(np.max(np.abs(basis[:, :2] - original)))
        assert delta < 1e-8, (name, 'historical PCA basis mismatch', delta)
        bases[name] = basis
        np.save(target / (name + '.npy'), basis)
        info[name] = {'calibration_images': n, 'calibration_sha256': selected,
                      'target_hash_overlap': 0, 'top2_max_abs_delta_from_historical': delta,
                      'explained_variance_ratio': {str(d): float(eig[:d].sum()/eig.sum()) for d in DIMS}}
    sources = {str(home_manifest): sha(home_manifest), **{p: h for h, p in by_hash.items()}}
    dump(manifest, {'completed': True, 'image_forwards': 433, 'encoder': encoder.metadata,
                    'per_image_vectors_saved': False, 'source_sha256': sources, 'domains': info,
                    'output_sha256': {str(target / (name + '.npy')): sha(target / (name + '.npy')) for name in bases}})
    return bases


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    completion = output / 'offline_manifest.json'
    if completion.exists():
        saved = read(completion)
        check_hashes(saved['source_sha256'])
        check_hashes(saved['output_sha256'])
        return
    domains, sources, _ = preflight(SOURCE, CAPTION)
    sources.update({str(Path(__file__)): sha(__file__),
                    str(ROOT / 'scripts/analysis/compare_caption_visual_dimensions.py'):
                    sha(ROOT / 'scripts/analysis/compare_caption_visual_dimensions.py')})
    frozen = output / 'frozen_inputs.json'
    if frozen.exists():
        assert read(frozen)['source_sha256'] == sources
    else:
        dump(frozen, {'source_sha256': sources, 'dimensions': list(DIMS), 'lambda': .03, 'alpha': 1.,
                      'top_k': 5, 'encoder_device': 'cpu', 'caption_regeneration': False,
                      'historical_response_reuse': False, 'calibration_policy': 'same external-domain PCA data, nested leading components'})
    import torch
    torch.set_num_threads(4)
    meta = domains[0]['protocol']['image_encoder']
    encoder = ImageEncoder(meta['model_name'], revision=meta['revision'], device='cpu', local_files_only=True)
    assert encoder.metadata == meta
    bases = calibrate(domains, encoder, output)
    summaries, artifacts = {}, {}
    for domain in domains:
        name = domain['name']
        complete = output / 'offline' / name / 'completion.json'
        if complete.exists():
            info = read(complete)
            check_hashes(info['output_sha256'])
            summaries[name] = info
            artifacts.update(info['output_sha256'])
            artifacts[str(complete)] = sha(complete)
            continue
        parent = complete.parent
        parent.mkdir(parents=True, exist_ok=True)
        memories = {d: PartitionReplay(bases[name][:, :d]) for d in DIMS}
        total = np.zeros(512)
        checks = {str(d): {'max_global_sum_error': 0., 'max_projection_error': 0.} for d in DIMS}
        for index, (row, audit) in enumerate(zip(domain['rows'], domain['historical_audits']), 1):
            assert sha(row['source_path']) == row['sha256']
            current = _unit(_unit(encoder.encode(row['source_path']), 'arrival', 512), 'core visual', 512)
            total += current
            for d, memory in memories.items():
                memory.add(row, current, audit)
                err = float(np.linalg.norm(sum(m['visual_sum'] for m in memory.masters.values()) - total))
                proj = max(float(np.linalg.norm(memory.basis.T @ m['visual_sum'] -
                    np.sum([memory.image_nodes[i]['visual_coords'] for i in m['member_ids']], axis=0)))
                    for m in memory.masters.values())
                assert err < 1e-10 and proj < 1e-10
                checks[str(d)]['max_global_sum_error'] = max(checks[str(d)]['max_global_sum_error'], err)
                checks[str(d)]['max_projection_error'] = max(checks[str(d)]['max_projection_error'], proj)
            del current
            if index % 20 == 0 or index == len(domain['rows']):
                log(stage='build', domain=name, completed=index, total=len(domain['rows']))
        sums = np.stack([m['visual_sum'] for m in memories[2].masters.values()])
        frozen_sums = np.load(domain['directory'] / 'memory/master_sums.npy')
        delta = float(np.max(np.abs(sums - frozen_sums)))
        assert delta < 1e-8, (name, '2D replay differs', delta)
        outputs = {}
        for d, memory in memories.items():
            destination = parent / f'pca_{d}d'
            memory.save(destination)
            graph = read(destination / 'graph.json')
            assert graph['leaf_visual_dim'] == d
            assert all(len(n['visual_coords']) == d for n in graph['image_nodes'])
            for file in destination.iterdir():
                outputs[str(file)] = sha(file)
        info = {'completed': True, 'leaves': len(domain['rows']), 'masters': len(memories[2].masters),
                'historical_partition_replayed': True, 'image_arrival_forwards': len(domain['rows']),
                'full_leaf_visual_vectors_saved': False, 'two_dimensional_max_abs_sum_delta': delta,
                'checks': checks, 'output_sha256': outputs}
        dump(complete, info)
        summaries[name] = info
        artifacts.update(outputs)
        artifacts[str(complete)] = sha(complete)
    assert sum(d['leaves'] for d in summaries.values()) == 438
    assert sum(d['masters'] for d in summaries.values()) == 88
    artifacts[str(output / 'calibration/manifest.json')] = sha(output / 'calibration/manifest.json')
    dump(completion, {'completed': True, 'dimensions': list(DIMS), 'domains': summaries,
                      'source_sha256': sources, 'output_sha256': artifacts})


def retrieve(output):
    manifest = output / 'retrieval_manifest.json'
    if manifest.exists():
        saved = read(manifest)
        check_hashes(saved['source_sha256'])
        check_hashes(saved['output_sha256'])
        return
    offline = read(output / 'offline_manifest.json')
    check_hashes(offline['output_sha256'])
    domains, originals, _, tracked, drift = verify_cache()
    history = {r['question_id']: r for r in lines(HISTORY / 'retrieval_results.jsonl')}
    historical_protocol = read(HISTORY / 'protocol.json')
    assert sha(HISTORY / 'retrieval_results.jsonl') == historical_protocol['output_sha256'][str(HISTORY / 'retrieval_results.jsonl')]
    records, two_d = [], {}
    max_dual_error = max_historical_score_error = 0.
    choice_changes = {c: 0 for c in CONDITIONS}
    for domain, (inputs, frozen_graph, _, cache) in domains.items():
        leaves = inputs['leaves']
        x = normalize(np.load(SOURCE / domain / 'features.npz')['bge_leaf'])
        qvecs = normalize(cache['bge_augmented_query_vectors'])
        frozen_masters = frozen_graph['masters']
        indices = [np.array([i for i, n in enumerate(frozen_graph['image_nodes']) if n['master_id'] == m['id']]) for m in frozen_masters]
        # All four hypothetical query scores are independent of visual dimension.
        weighted = np.empty((len(qvecs), 4, len(x)))
        for ix in indices:
            xx = x[ix]
            center = xx - xx.mean(0)
            for qi, four in enumerate(qvecs):
                for ci, query in enumerate(four):
                    score = ZScoreWeighting(.03, 1.).apply(query, xx)
                    adjusted = query if len(xx) == 1 else query - center.T @ np.linalg.solve(
                        center @ center.T + (len(xx)-1)*.03*np.eye(len(xx)), center @ query)
                    error = float(np.max(np.abs(score['white'] - xx @ adjusted)))
                    max_dual_error = max(max_dual_error, error)
                    assert error < 2e-12
                    weighted[qi, ci, ix] = score['weighted']
        for d in (2, *[rank for rank in DIMS if rank != 2]):
            condition = f'pca_{d}d'
            folder = output / 'offline' / domain / condition
            graph = read(folder / 'graph.json')
            assert [m['id'] for m in graph['masters']] == [m['id'] for m in frozen_masters]
            assert [m['member_ids'] for m in graph['masters']] == [m['member_ids'] for m in frozen_masters]
            for n, old in zip(graph['image_nodes'], frozen_graph['image_nodes']):
                assert all(n[k] == old[k] for k in ('id', 'caption', 'dialogue', 'timestamp', 'master_id'))
            means = centers(graph, np.load(folder / 'master_sums.npy'))
            clip = np.einsum('md,qcd->qmc', normalize(means), cache['clip_candidate_vectors'])
            choices = clip.argmax(axis=2)
            if d == 2:
                np.testing.assert_array_equal(choices, cache['selected_choice_indices'])
                base_choices = choices
            choice_changes[condition] += int(np.sum(choices != base_choices))
            owner = cache['leaf_master_indices']
            for qi, q in enumerate(inputs['questions']):
                scores = weighted[qi, choices[qi, owner], np.arange(len(x))]
                order = np.argsort(-scores, kind='stable')[:5]
                ids = [leaves[i]['memory_id'] for i in order]
                row = {'condition': condition, 'dimension': d, 'question_id': q['id'], 'domain': domain,
                       'top5_memory_ids': ids, 'top5_scores': scores[order].tolist(),
                       'master_choices': {m['id']: 'ABCD'[choices[qi, j]] for j, m in enumerate(frozen_masters)},
                       'gold_ids': originals[q['id']]['gold_ids'], 'gold_choice': originals[q['id']]['gold_choice']}
                row['gold_hits_top5'] = len(set(ids) & set(row['gold_ids']))
                if d == 2:
                    previous = history[q['id']]['conditions'][HIST_CONDITION]
                    assert ids == previous['top5_memory_ids'], (q['id'], '2D ranking mismatch')
                    error = float(np.max(np.abs(scores[order] - previous['top5_scores'])))
                    max_historical_score_error = max(max_historical_score_error, error)
                    # Historical online encoding renormalizes the cached FP32 vectors;
                    # its independent audit uses the same 2e-6 CPU/GPU tolerance.
                    # Ordered Top5 must nevertheless match exactly on every question.
                    assert error < 2e-6, (q['id'], error)
                    two_d[q['id']] = row
                row['same_top5_order_as_2d'] = ids == two_d[q['id']]['top5_memory_ids']
                records.append(row)
        log(stage='retrieval', domain=domain, records=len(records))
    assert len(records) == len({(r['condition'], r['question_id']) for r in records}) == 3710
    write_lines(output / 'retrieval_results.jsonl', records)
    summaries = {c: {**metrics([r for r in records if r['condition'] == c]),
                     'changed_master_choices': choice_changes[c],
                     'changed_top5_order': sum(not r['same_top5_order_as_2d'] for r in records if r['condition'] == c)} for c in CONDITIONS}
    sources = {**tracked, str(output / 'offline_manifest.json'): sha(output / 'offline_manifest.json'),
               str(HISTORY / 'retrieval_results.jsonl'): sha(HISTORY / 'retrieval_results.jsonl')}
    for p in [Path(__file__), ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py']:
        sources[str(p)] = sha(p)
    dump(manifest, {'completed': True, 'conditions': summaries, 'two_dimensional_historical_top5_matches': 371,
                    'max_historical_top5_score_error': max_historical_score_error,
                    'max_independent_dual_ridge_error': max_dual_error, 'historical_source_drift': drift,
                    'source_sha256': sources, 'output_sha256': {str(output / 'retrieval_results.jsonl'): sha(output / 'retrieval_results.jsonl')}})
    log(stage='retrieval_complete', conditions=summaries)


def prepare_reader(output, gpu):
    out = output / 'reader'
    if (out / 'protocol.json').exists():
        protocol = read(out / 'protocol.json')
        verify_signature(protocol)
        verify(protocol)
        assert protocol['placement']['gpu_index'] == gpu
        return
    retrieval = read(output / 'retrieval_manifest.json')
    check_hashes(retrieval['source_sha256'])
    check_hashes(retrieval['output_sha256'])
    previous = read(HISTORY / 'reader/protocol.json')
    verify_signature(previous)
    settings = previous['reader_settings']
    assert settings['seed'] == 42 and settings['do_sample'] is False and settings['max_new_tokens'] == 16
    indexed = {(r['question_id'], r['condition']): r for r in lines(output / 'retrieval_results.jsonl')}
    questions = lines(ORIGINAL / 'qa_manifest.jsonl')
    memories = {m['id']: m['reader_text'] for m in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    old = {r['id']: r for r in lines(HISTORY / 'reader/qa_manifest.jsonl') if r['condition'] == HIST_CONDITION}
    manifest, images = [], {}
    for qi, q in enumerate(questions):
        for path in q['query_image_paths']:
            images.setdefault(path, {'path': path, 'sha256': sha(path)})
        image_rows = [images[p] for p in q['query_image_paths']]
        batch = []
        for c in CONDITIONS:
            r = indexed[q['id'], c]
            ids = r['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            text = reader_text(q, ids, memories)
            payload = digest({'system': settings['reader_system_prompt'], 'user_text': text, 'query_images': image_rows})
            if c == 'pca_2d':
                assert payload == old[q['id']]['payload_sha256']
            batch.append({**{k: q[k] for k in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')},
                          'record_id': q['id'] + '::' + c, 'condition': c, 'top_memory_ids': ids,
                          'top_query_scores': r['top5_scores'], 'reader_text': text, 'query_images': image_rows,
                          'payload_sha256': payload})
        offset = qi % len(CONDITIONS)
        manifest.extend(batch[offset:] + batch[:offset])
    assert len(manifest) == 3710
    check_hashes(previous['model_source_sha256'])
    for weight in previous['weight_files']:
        assert sha(weight['path']) == weight['sha256']
    out.mkdir(exist_ok=True)
    write_lines(out / 'qa_manifest.jsonl', manifest)
    source_files = [Path(__file__), output / 'retrieval_manifest.json', output / 'retrieval_results.jsonl',
                    out / 'qa_manifest.jsonl', ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl',
                    HISTORY / 'reader/protocol.json', HISTORY / 'reader/qa_manifest.jsonl',
                    ROOT / 'scripts/evaluation/paired_single_gpu_reader.py',
                    ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py',
                    ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py']
    protocol = signed({'design': 'Only PCA visual coordinate dimension changes; fixed Qwen CVM + zscore lambda=.03 alpha=1 Top5.',
                       'experiment_run_id': str(uuid.uuid4()), 'dimensions': list(DIMS), 'condition_names': list(CONDITIONS),
                       'questions': 371, 'logical_conditions': 3710, 'reader_settings': settings,
                       'placement': {'gpu_index': gpu, 'minimum_free_gib': 24}, 'attention_implementation': 'sdpa',
                       'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}),
                       'historical_response_reuse': False, 'two_dimensional_historical_payload_matches': 371,
                       'deduplication': 'Identical complete payloads share one fresh response within this experiment only.',
                       'source_sha256': {str(p): sha(p) for p in source_files},
                       'model_source_sha256': previous['model_source_sha256'], 'weight_files': previous['weight_files'],
                       'query_images': list(images.values())})
    dump(out / 'protocol.json', protocol)
    verify(protocol)
    log(stage='reader_prepared', logical_records=3710, unique_payloads=protocol['unique_reader_payloads'])


def analyze(reader):
    protocol = read(reader / 'protocol.json')
    verify_signature(protocol)
    verify(protocol)
    manifests = lines(reader / 'qa_manifest.jsonl')
    rows = lines(reader / 'responses.jsonl')
    by_id = {r['record_id']: r for r in rows}
    assert len(rows) == len(by_id) == len(manifests) == 3710
    payload_predictions = {}
    for m in manifests:
        r = by_id[m['record_id']]
        assert r['experiment_signature'] == protocol['experiment_signature'] and r['run_id'] == protocol['experiment_run_id']
        assert r['payload_sha256'] == m['payload_sha256'] and r['gold_choice'] == m['gold_choice']
        assert r['correct'] == (r['predicted_choice'] == m['gold_choice'])
        assert not r['historical_response_reuse']
        payload_predictions.setdefault(r['payload_sha256'], r['predicted_choice'])
        assert payload_predictions[r['payload_sha256']] == r['predicted_choice']
    retrieval = read(reader.parent / 'retrieval_manifest.json')['conditions']
    old = {r['question_id']: r for r in lines(HISTORY / 'reader/responses.jsonl') if r['condition'] == HIST_CONDITION}
    assert len(old) == 371 and sum(r['correct'] for r in old.values()) == 199
    summaries, paired = [], []
    for d, c in zip(DIMS, CONDITIONS):
        subset = [r for r in rows if r['condition'] == c]
        assert len(subset) == 371
        wins = losses = changes = 0
        for r in subset:
            b = by_id[r['question_id'] + '::pca_2d']
            wins += int(r['correct'] and not b['correct'])
            losses += int(b['correct'] and not r['correct'])
            changes += int(b['predicted_choice'] != r['predicted_choice'])
            paired.append({'dimension': d, 'question_id': r['question_id'], 'gold_choice': r['gold_choice'],
                           'prediction': r['predicted_choice'], 'correct': r['correct'],
                           'pca2_prediction': b['predicted_choice'], 'pca2_correct': b['correct'],
                           'same_payload_as_2d': r['payload_sha256'] == b['payload_sha256']})
        n = wins + losses
        p = min(1., 2*sum(math.comb(n, i) for i in range(min(wins, losses)+1))/2**n) if n else 1.
        count = sum(r['correct'] for r in subset)
        summaries.append({'dimension': d, 'condition': c, 'correct': count, 'questions': 371, 'accuracy': count/371,
                          **retrieval[c], 'wrong_to_correct': wins, 'correct_to_wrong': losses,
                          'answer_changes': changes, 'mcnemar_exact_two_sided_p': p,
                          'format_errors': sum(r['status'] != 'ok' for r in subset),
                          'hit_token_limit': sum(r['hit_token_limit'] for r in subset),
                          'visual_numeric_bytes': 8*(512*d + 88*512 + 438*d)})
    historical_changes = sum(by_id[q + '::pca_2d']['predicted_choice'] != r['predicted_choice'] for q, r in old.items())
    result = {'completed': True, 'conditions': summaries, 'historical_2d_correct': 199,
              'fresh_2d_vs_historical_answer_changes': historical_changes,
              'fresh_inferences': sum(r['fresh_inference_executed_for_this_record'] for r in rows),
              'logical_records': len(rows), 'historical_response_reuse': False}
    dump(reader.parent / 'summary.json', result)
    for filename, values in [('summary.csv', summaries), ('paired_results.csv', paired)]:
        with (reader.parent / filename).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    dump(reader.parent / 'completion_manifest.json', {
        'completed': True, 'questions_per_condition': 371, 'conditions': len(CONDITIONS), 'logical_records': 3710,
        'format_errors': sum(r['status'] != 'ok' for r in rows),
        'artifact_sha256': {str(p): sha(p) for p in [reader / 'protocol.json', reader / 'responses.jsonl',
                            reader.parent / 'summary.json', reader.parent / 'summary.csv', reader.parent / 'paired_results.csv']}})
    log(stage='complete', **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=['build', 'retrieve', 'prepare', 'reader', 'analyze', 'all'], default='all')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.stage in ('all', 'build'):
        build(out)
    if args.stage in ('all', 'retrieve'):
        retrieve(out)
    if args.stage in ('all', 'prepare'):
        prepare_reader(out, args.gpu)
    if args.stage in ('all', 'reader'):
        run_reader(out / 'reader', analyze, resume=args.resume)
    if args.stage == 'analyze':
        analyze(out / 'reader')


if __name__ == '__main__':
    main()
