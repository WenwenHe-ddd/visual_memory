#!/usr/bin/env python3
"""Matched, streaming PCA/Random-2D replay for a frozen MemEye API backbone.

No API calls. Reuse one backbone's captions and candidate statements, recompute
each caption partition, and encode each arriving image once for all four states.
Only master visual sums and two coordinates per leaf are persisted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(key, value)

import numpy as np

from scripts.memory import memeye_backbone_retrieval as original
from scripts.memory.memeye_backbone_long_text import encode_complete, POLICY
from scripts.memory.caption_visual_master_random_2d import make_random_basis
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory, choose_caption_partition, _unit
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.analysis.prepare_all_domain_adaptive_inputs import save_memory

read, lines, sha, digest, dump = original.read, original.lines, original.sha, original.digest, original.dump
SEEDS = [0, 1, 2]
CONDITIONS = ['pca_2d', *[f'random_2d_seed_{seed}' for seed in SEEDS]]


def verify_source(source):
    prep, root, retrieval = [read(source / p) for p in
                             ('inputs/protocol.json', 'protocol.json', 'retrieval_protocol.json')]
    assert digest(root['contract']) == root['experiment_signature']
    if 'experiment_signature' in retrieval:
        assert digest(retrieval['contract']) == retrieval['experiment_signature']
    hashes, drift, cached = {}, [], {}

    def check(mapping, base=None):
        for name, expected in mapping.items():
            path = Path(name)
            if not path.is_absolute():
                assert base is not None, name
                path = base / path
            path = path.resolve()
            actual = cached.setdefault(str(path), sha(path)) if str(path) not in cached else cached[str(path)]
            if actual != expected:
                allowed_drift = {ROOT / 'scripts/memory/caption_visual_master.py',
                                 ROOT / 'scripts/visual_memory_pipeline.py'}
                assert path in allowed_drift, f'UNAPPROVED SOURCE CHANGE: {path}'
                item = {'path': str(path), 'historical_sha256': expected, 'current_sha256': actual}
                if item not in drift:
                    drift.append(item)
            hashes[str(path)] = actual

    for mapping in (root['contract']['source_sha256'], prep['source_sha256'],
                    retrieval['source_sha256'], retrieval['output_sha256']):
        check(mapping)
    check(prep['artifact_sha256'], source / 'inputs')
    for domain in prep['basis_by_domain']:
        directory = source / 'domains' / domain
        check(read(directory / 'build_protocol.json')['artifact_sha256'])
        for name in ('build_protocol.json', 'insertion_audit.jsonl', 'memory/graph.json',
                     'memory/basis.npy', 'memory/master_sums.npy'):
            path = directory / name
            hashes[str(path)] = sha(path)
    for name in ('protocol.json', 'inputs/protocol.json', 'retrieval_protocol.json',
                 'captions.jsonl', 'retrieval_results.jsonl'):
        path = source / name
        hashes[str(path)] = sha(path)
    for path in (Path(__file__).resolve(), ROOT / 'scripts/memory/caption_visual_master_random_2d.py'):
        hashes[str(path)] = sha(path)
    memories, questions, model = original.attach_captions(
        lines(source / 'inputs/source_memories.jsonl'), lines(source / 'inputs/source_questions.jsonl'),
        lines(source / 'captions.jsonl'))
    assert len(memories) == 438 and len(questions) == 371
    assert model == root['contract']['model']
    domains = sorted({m['task'] for m in memories})
    assert len(domains) == 8 and sum(bool(q['query_images']) for q in questions) == 8
    proof = {'passed': True, 'source': str(source), 'caption_model': model,
             'source_run_id': root['run_id'], 'source_experiment_signature': root['experiment_signature'],
             'historical_code_drift': drift, 'source_sha256': hashes,
             'data_hash_failures': 0, 'questions': 371, 'memories': 438, 'domains': domains,
             'seeds': SEEDS, 'conditions': CONDITIONS, 'long_text_policy': POLICY,
             'online_weighting': 'none', 'leaf_visual_dimension': 2}
    return prep, memories, questions, proof


def verify_file_map(hashes):
    for path, expected in hashes.items():
        assert sha(path) == expected, f'Artifact changed: {path}'


def build_domain(source, output, domain, rows, encoder, bge, signature):
    completion = output / 'offline' / 'domain_completion' / f'{domain}.json'
    if completion.exists():
        previous = read(completion)
        assert previous['source_signature'] == signature
        verify_file_map(previous['artifact_sha256'])
        return previous
    historical = lines(source / 'domains' / domain / 'insertion_audit.jsonl')
    historical_graph = read(source / 'domains' / domain / 'memory/graph.json')
    assert len(historical) == len(rows)
    assert [r['insertion_index'] for r in rows] == list(range(1, len(rows) + 1))
    assert [n['id'] for n in historical_graph['image_nodes']] == [r['image_id'] for r in rows]
    for node, row in zip(historical_graph['image_nodes'], rows):
        for key in ('caption', 'dialogue', 'timestamp'):
            assert node[key] == row[key]
    pca_basis = np.load(source / 'domains' / domain / 'memory/basis.npy', allow_pickle=False)
    bases = {'pca_2d': pca_basis, **{f'random_2d_seed_{s}': make_random_basis(512, s) for s in SEEDS}}
    states = {name: CaptionWardVisualMemory(basis) for name, basis in bases.items()}
    audits = {name: [] for name in CONDITIONS}
    total = np.zeros(512, dtype=np.float64)
    score_delta = 0.0
    started = time.perf_counter()
    for index, row in enumerate(rows):
        caption, token_info = original.encode_checked(bge, [row['caption']], label=row['image_id'])
        text = _unit(caption[0], 'caption', 384)
        choice = choose_caption_partition(np.stack([*states['pca_2d'].text_embedding_cache.values(), text]))
        ids = [r['image_id'] for r in rows[:index + 1]]
        partition = [[ids[i] for i in group] for group in choice['partition']]
        assert partition == historical[index]['clusters_partition'], (domain, index + 1, 'caption prefix changed')
        assert choice['selected_k'] == historical[index]['selected_k']
        if choice.get('selected_score') is not None:
            score_delta = max(score_delta, abs(choice['selected_score'] - historical[index]['selected_score']))
        assert sha(row['image_path']) == row['image_sha256']
        visual = original.normalize(encoder.encode(row['image_path']))
        total += _unit(visual, 'visual', 512)
        with patch('scripts.graph.caption_ward_visual.choose_caption_partition', return_value=choice):
            for name, memory in states.items():
                audit = memory.add(row['image_id'], row['caption'], caption[0], visual,
                                   dialogue=row['dialogue'], timestamp=row['timestamp'])
                assert audit['clusters_partition'] == historical[index]['clusters_partition']
                assert audit['clusters'] == historical[index]['clusters'], (domain, name, index + 1)
                error = float(np.linalg.norm(sum(m['visual_sum'] for m in memory.masters.values()) - total))
                assert error < 1e-10 and audit['max_projection_consistency_l2_error'] < 1e-10
                audit.update(insertion_index=index + 1, caption_encoding=token_info,
                             caption_partition_matches_historical=True, visual_forward_shared=True,
                             cumulative_global_sum_l2_error=error)
                audits[name].append(audit)
        del visual, caption, text
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(json.dumps({'stage': 'offline', 'domain': domain, 'arrivals': index + 1,
                              'total': len(rows), 'masters': len(states['pca_2d'].masters)}), flush=True)
    artifacts, stats, comparisons = {}, {}, []
    caption_cache = output / 'offline' / 'caption_features' / f'{domain}.npz'
    caption_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(caption_cache, image_ids=np.asarray([r['image_id'] for r in rows]),
                        caption_vectors=np.stack(list(states['pca_2d'].text_embedding_cache.values())))
    artifacts[str(caption_cache)] = sha(caption_cache)
    historical_sums = np.load(source / 'domains' / domain / 'memory/master_sums.npy', allow_pickle=False)
    pca_sums = np.stack([m['visual_sum'] for m in states['pca_2d'].masters.values()])
    pca_coords = np.asarray([n['visual_coords'] for n in states['pca_2d'].image_nodes.values()])
    historical_coords = np.asarray([n['visual_coords'] for n in historical_graph['image_nodes']])
    sum_delta = float(np.max(np.abs(pca_sums - historical_sums)))
    coord_delta = float(np.max(np.abs(pca_coords - historical_coords)))
    assert sum_delta < 1e-4 and coord_delta < 1e-5, (domain, sum_delta, coord_delta)
    for name, memory in states.items():
        directory = output / 'offline' / name / domain
        graph, sums = save_memory(memory, directory / 'memory')
        original.write_lines(directory / 'insertion_audit.jsonl', audits[name])
        assert [m['id'] for m in graph['masters']] == [m['id'] for m in historical_graph['masters']]
        assert [m['member_ids'] for m in graph['masters']] == [m['member_ids'] for m in historical_graph['masters']]
        for node in graph['image_nodes']:
            assert set(node) == {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'}
            assert len(node['visual_coords']) == 2
        orth_error = float(np.max(np.abs(memory.basis.T @ memory.basis - np.eye(2))))
        assert orth_error < 1e-12
        np.testing.assert_allclose(sums.sum(axis=0), pca_sums.sum(axis=0), atol=1e-10, rtol=0)
        for i, master in enumerate(graph['masters']):
            comparisons.append({'condition': name, 'domain': domain, 'master_id': master['id'],
                                'members': master['n'], 'cosine_to_pca': float(original.normalize(sums[i]) @ original.normalize(pca_sums[i])),
                                'sum_l2_to_pca': float(np.linalg.norm(sums[i] - pca_sums[i]))})
        stats[name] = {'leaves': len(rows), 'masters': len(graph['masters']),
                       'split_steps': sum(bool(a['split_master_ids']) for a in audits[name]),
                       'basis_orthogonality_max_error': orth_error,
                       'max_global_sum_l2_error': max(a['cumulative_global_sum_l2_error'] for a in audits[name]),
                       'max_projection_error': max(a['max_projection_consistency_l2_error'] for a in audits[name])}
        for filename in ('memory/graph.json', 'memory/basis.npy', 'memory/master_sums.npy', 'insertion_audit.jsonl'):
            path = directory / filename
            artifacts[str(path)] = sha(path)
    result = {'completed': True, 'source_signature': signature, 'domain': domain,
              'image_arrival_forwards': len(rows), 'old_image_reencodes': 0,
              'per_image_visual_embeddings_saved': False, 'original_images_saved': False,
              'historical_prefix_partitions_identical': len(rows), 'caption_silhouette_max_delta': score_delta,
              'pca_master_sum_max_abs_delta': sum_delta, 'pca_coords_max_abs_delta': coord_delta,
              'seconds': time.perf_counter() - started, 'conditions': stats,
              'master_comparisons': comparisons, 'artifact_sha256': artifacts}
    dump(completion, result)
    return result


def run(source, output, device, stage):
    started = time.perf_counter()
    prep, memories, questions, proof = verify_source(source)
    signature = digest({'proof': proof, 'device': device})
    output.mkdir(parents=True, exist_ok=True)
    proof_path = output / 'source_validation.json'
    if proof_path.exists():
        assert read(proof_path) == proof, 'Source or replay code changed; use a fresh run directory'
    else:
        dump(proof_path, proof)
    if stage == 'verify':
        return proof
    runtime = SimpleNamespace(device=device)
    original.configure_device(runtime)
    bge = original.load_text_encoder(prep['text_encoder'], runtime.device)
    assert bge.max_seq_length == prep['text_encoder']['max_seq_length']
    builds = []
    if stage in ('all', 'offline'):
        meta = prep['image_encoder']
        encoder = ImageEncoder(meta['model_name'], revision=meta['revision'], device=runtime.device, local_files_only=True)
        assert original._metadata_equal(encoder.metadata, meta), 'Actual CLIP configuration changed'
        for domain in proof['domains']:
            builds.append(build_domain(source, output, domain, [m for m in memories if m['task'] == domain],
                                       encoder, bge, signature))
        del encoder
        dump(output / 'offline/manifest.json', {'completed': True, 'source_signature': signature,
             'source': str(source), 'conditions': CONDITIONS, 'seeds': SEEDS,
             'domains': builds, 'image_arrival_forwards': sum(b['image_arrival_forwards'] for b in builds),
             'old_image_reencodes': 0, 'per_image_visual_embeddings_saved': False})
        if stage == 'offline':
            return {'completed': True, 'stage': 'offline'}
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        prior = read(marker)
        assert prior['source_signature'] == signature
        verify_file_map(prior['artifact_sha256'])
        return prior
    manifest = read(output / 'offline/manifest.json')
    assert manifest['completed'] and manifest['source_signature'] == signature
    for build in manifest['domains']:
        verify_file_map(build['artifact_sha256'])
    clip, tokenizer, limit = original.load_clip_text_encoder(prep['image_encoder'], runtime.device)
    historical = {r['question_id']: r for r in lines(source / 'retrieval_results.jsonl')}
    results, encoding, comparisons, feature_paths = [], [], [], []
    max_score_delta = 0.0
    for domain in proof['domains']:
        ms = [m for m in memories if m['task'] == domain]
        qs = [q for q in questions if q['task'] == domain]
        e, einfo = encode_complete(bge, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='dialogue+caption')
        query_texts, query_ids, statements = [], [], []
        for q in qs:
            candidates = [q['candidate_statements'][key] for key in 'ABCD']
            statements.extend(candidates)
            query_texts.extend([q['retrieval_query']] + [q['retrieval_query'] + '\nCandidate answer: ' + s for s in candidates])
            query_ids.extend(q['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D'))
        qv, qinfo = encode_complete(bge, query_texts, prep['query_prefix'], ids=query_ids, field='query')
        cv, cinfo = original._clip_candidates(clip, tokenizer, limit, statements)
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        feature_path = output / 'retrieval_features' / f'{domain}.npz'
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feature_path, question_ids=np.asarray([q['id'] for q in qs]),
                            memory_ids=np.asarray([m['id'] for m in ms]),
                            leaf_vectors=e, query_vectors=qv, candidate_text_vectors=cv)
        feature_paths.append(feature_path)
        states = {name: (read(output / 'offline' / name / domain / 'memory/graph.json'),
                        np.load(output / 'offline' / name / domain / 'memory/master_sums.npy', allow_pickle=False))
                  for name in CONDITIONS}
        for qi, question in enumerate(qs):
            condition_rows, raw_rows = [], {}
            for name, (graph, sums) in states.items():
                raw = original.rank_question(question, ms, graph, sums, qv[qi], e, cv[qi])
                selected = next(c for c in raw['conditions'] if c['name'] == 'master_answer_query')
                condition_rows.append({**selected, 'name': name, 'masters': raw['masters'],
                                       'leaves': raw['leaves'], 'master_choice_counts': raw['master_choice_counts']})
                raw_rows[name] = raw
            pca, prior = raw_rows['pca_2d'], historical[question['id']]
            for key in ('question', 'options', 'query_images', 'query_image_captions', 'baseline_retrieval_query', 'candidate_statements'):
                assert pca[key] == prior[key], (question['id'], key)
            for a, b in zip(pca['masters'], prior['masters']):
                for key in ('master_id', 'member_ids', 'selected_choice', 'selected_statement', 'augmented_query'):
                    assert a[key] == b[key], (question['id'], key)
                np.testing.assert_allclose(list(a['clip_candidate_scores'].values()), list(b['clip_candidate_scores'].values()), atol=2e-6, rtol=0)
            for a, b in zip(pca['conditions'], prior['conditions']):
                assert a['full_rank_memory_ids'] == b['full_rank_memory_ids'], (question['id'], a['name'], 'rank changed')
                assert a['reader_context'] == b['reader_context']
                delta = float(np.max(np.abs(np.asarray(a['top5_scores']) - b['top5_scores'])))
                assert delta < 2e-6
                max_score_delta = max(max_score_delta, delta)
            for condition in condition_rows[1:]:
                comparisons.append({'question_id': question['id'], 'task': domain, 'condition': condition['name'],
                                    'ordered_top5_changed': condition['top5_memory_ids'] != condition_rows[0]['top5_memory_ids'],
                                    'master_choices_changed': sum(a['selected_choice'] != b['selected_choice']
                                                                 for a, b in zip(condition['masters'], pca['masters']))})
            results.append({**{k: pca[k] for k in ('id', 'question_id', 'task', 'question', 'options',
                           'query_image_paths', 'query_images', 'query_image_captions', 'baseline_retrieval_query', 'candidate_statements')},
                           'conditions': condition_rows})
        encoding.append({'task': domain, 'leaf': einfo, 'query': qinfo, 'candidate': cinfo})
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'questions': len(qs), 'pca_historical_exact_rank_parity': True}), flush=True)
    assert len(results) == 371 and {r['id'] for r in results} == {q['id'] for q in questions}
    # Publish answer-free rankings before opening the separate evaluation labels.
    original.write_lines(output / 'retrieval_results.jsonl', results)
    original.write_lines(output / 'retrieval_comparisons.jsonl', comparisons)
    gold = {r['id']: r for r in lines(source / 'inputs/evaluation.jsonl')}
    assert set(gold) == {r['id'] for r in results}
    metrics = []
    for name in CONDITIONS:
        for domain in ['ALL', *proof['domains']]:
            subset = [r for r in results if domain == 'ALL' or r['task'] == domain]
            hits = sum(len(set(gold[r['id']]['gold_ids']) & set(next(c for c in r['conditions'] if c['name'] == name)['top5_memory_ids'])) for r in subset)
            total = sum(len(set(gold[r['id']]['gold_ids'])) for r in subset)
            metrics.append({'condition': name, 'domain': domain, 'questions': len(subset), 'gold_hits': hits,
                            'gold_total': total, 'recall_at_5': hits / total, 'recall_at_5_percent': 100 * hits / total})
    with (output / 'retrieval_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    summary = {'completed': True, 'caption_model': proof['caption_model'], 'conditions': CONDITIONS, 'seeds': SEEDS,
               'questions': 371, 'condition_question_records': 1484, 'memories': 438,
               'masters': sum(b['conditions']['pca_2d']['masters'] for b in manifest['domains']),
               'metrics': metrics, 'historical_pca_exact_full_rank_and_context_parity_questions': 371,
               'historical_top5_score_max_abs_delta': max_score_delta,
               'changes_by_condition': {name: {'ordered_top5_changed': sum(r['ordered_top5_changed'] for r in comparisons if r['condition'] == name),
                                               'master_choices_changed': sum(r['master_choices_changed'] for r in comparisons if r['condition'] == name)} for name in CONDITIONS[1:]},
               'encoding_audits': encoding, 'gold_loaded_for_ranking': False, 'online_weighting': 'none'}
    dump(output / 'retrieval_summary.json', summary)
    verify_file_map(proof['source_sha256'])
    artifacts = {str(output / p): sha(output / p) for p in ('source_validation.json', 'offline/manifest.json',
                 'retrieval_results.jsonl', 'retrieval_comparisons.jsonl', 'retrieval_metrics.csv', 'retrieval_summary.json')}
    artifacts.update({str(path): sha(path) for path in feature_paths})
    protocol = {'completed': True, 'status': 'complete', 'source_signature': signature,
                'source': str(source), 'source_sha256': proof['source_sha256'], 'artifact_sha256': artifacts,
                'output_sha256': artifacts,
                'conditions': CONDITIONS, 'seeds': SEEDS, 'caption_model': proof['caption_model'],
                'questions': 371, 'gold_loaded_for_ranking': False, 'device': runtime.device,
                'reader_schema': 'one question per row; conditions[].name, top5, top5_memory_ids, reader_context, reader_memories; query_images include path and sha256',
                'elapsed_seconds': time.perf_counter() - started}
    dump(marker, protocol)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('verify', 'offline', 'retrieve', 'all'), default='all')
    args = parser.parse_args()
    result = run(args.source.resolve(), args.output.resolve(), args.device, args.stage)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'encoding_audits', 'metrics', 'artifact_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
