"""Full Mem-Gallery paired retrieval with chronological caption/visual memory.

Image memories use the existing two-coordinate visual update implementation.
Text-only memories keep baseline query scores. Evaluation labels are read only
after both rankings have been frozen. No individual image vectors are saved.
"""
from __future__ import annotations

import collections
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

from scripts.memory.caption_visual_master import (
    configure_device, encode_text, load_clip_text_encoder, load_text_encoder,
)
from scripts.analysis.probe_caption_visual_online import normalize
from scripts.analysis.prepare_all_domain_adaptive_inputs import save_memory
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder

DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_master_answer_20261021'
REFERENCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/protocol.json'
BASIS = ROOT / 'reports/home_caption_ward_visual_20261020/memory/basis.npy'
CALIBRATION = ROOT / 'reports/home_caption_ward_visual_20261020/input_manifest.json'
CONDITIONS = ('baseline_question_only', 'master_answer_query')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    temporary.replace(path)


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Input or artifact changed: {path}')


def build(output=DEFAULT_OUTPUT, device='cuda:0'):
    from scripts.data.mem_gallery_inputs import prepare
    output = Path(output).resolve()
    prepare(output / 'inputs')
    source = output / 'inputs/memories.jsonl'
    records = lines(source)
    assert len(records) == 3962 and sum(r['is_visual'] for r in records) == 1003
    tracked = [source, output / 'inputs/protocol.json', BASIS, REFERENCE, CALIBRATION,
               Path(__file__), ROOT / 'scripts/graph/caption_ward_visual.py',
               ROOT / 'scripts/graph/visual_image_encoder.py']
    hashes = {str(p): sha(p) for p in tracked}
    destination = output / 'offline'
    previous = destination / 'protocol.json'
    if previous.exists():
        protocol = read(previous)
        if protocol['source_sha256'] != hashes:
            raise ValueError('Existing memory belongs to another input or implementation; use a fresh output.')
        verify_hashes(protocol['artifact_sha256'])
        print('Mem-Gallery offline: verified existing complete memory.', flush=True)
        return
    runtime = SimpleNamespace(device=device)
    configure_device(runtime)
    reference = read(REFERENCE)
    calibration_hashes = {r['image_sha256'] for r in read(CALIBRATION)['calibration']}
    assert not calibration_hashes & {r['image_sha256'] for r in records if r['is_visual']}
    basis = np.load(BASIS, allow_pickle=False)
    meta = reference['image_encoder']
    encoder = ImageEncoder(meta['model_name'], revision=meta['revision'],
                           device=runtime.device, local_files_only=True)
    assert {k: v for k, v in encoder.metadata.items() if k != 'device'} == {
        k: v for k, v in meta.items() if k != 'device'}
    text_encoder = load_text_encoder(reference['text_encoder'], runtime.device)
    destination.mkdir(parents=True, exist_ok=True)
    # This is the persistent textual memory store. Source pixels/paths are only
    # present in the separate input audit, not in this store or visual graph.
    stored = [{k: v for k, v in r.items() if k not in ('image_path', 'image_sha256')}
              for r in records]
    write_lines(destination / 'memories.jsonl', stored)
    domains = list(dict.fromkeys(r['task'] for r in records))
    summaries, artifacts = [], [destination / 'memories.jsonl']
    started = time.perf_counter()
    for domain in domains:
        visual = [r for r in records if r['task'] == domain and r['is_visual']]
        memory = CaptionWardVisualMemory(basis)
        audits = []
        for index, row in enumerate(visual, 1):
            assert sha(row['image_path']) == row['image_sha256']
            vector = normalize(encoder.encode(row['image_path']))
            text_vector = encode_text(text_encoder, [row['caption']])[0]
            audit = memory.add(row['node_id'], row['caption'], text_vector, vector,
                               dialogue=row['dialogue'], timestamp=row['timestamp'])
            del vector, text_vector
            audits.append(audit)
        target = destination / 'memory' / domain
        graph, sums = save_memory(memory, target)
        assert len(graph['image_nodes']) == len(visual) and sums.shape[1] == 512
        write_lines(target / 'insertion_audit.jsonl', audits)
        artifacts.extend(target / name for name in ('graph.json', 'basis.npy', 'master_sums.npy', 'insertion_audit.jsonl'))
        summaries.append({'task': domain, 'images': len(visual), 'masters': len(graph['masters']),
                          'rounds': sum(r['task'] == domain for r in records),
                          'clustering_seconds': sum(a['timings_seconds']['clustering'] for a in audits)})
        print(json.dumps({'stage': 'mem_gallery_offline', **summaries[-1]}, ensure_ascii=False), flush=True)
    protocol = {
        'dataset': 'Mem-Gallery', 'domains': len(domains), 'rounds': len(records),
        'image_rounds': 1003, 'text_only_rounds': 2959, 'domain_summaries': summaries,
        'image_encoder': encoder.metadata, 'text_encoder': reference['text_encoder'],
        'query_prefix': reference['query_prefix'], 'device': runtime.device,
        'basis_source': str(BASIS), 'basis_calibration': 'Fixed external MemEye calibration, disjoint from Mem-Gallery; no refit or replay of old images.',
        'leaf_visual_dimension': 2, 'per_image_visual_embeddings_saved': False,
        'image_copies': 0, 'old_image_reencodes': 0,
        'source_sha256': hashes, 'artifact_sha256': {str(p): sha(p) for p in artifacts},
        'elapsed_seconds': time.perf_counter() - started,
    }
    verify_hashes(hashes)
    dump(previous, protocol)
    print('Mem-Gallery offline complete.', flush=True)


def retrieve(output=DEFAULT_OUTPUT, device='cuda:0', top_k=5):
    import torch
    output = Path(output).resolve()
    if top_k < 1:
        raise ValueError('top_k must be positive')
    offline = read(output / 'offline/protocol.json')
    verify_hashes(offline['source_sha256'])
    verify_hashes(offline['artifact_sha256'])
    records = lines(output / 'offline/memories.jsonl')
    questions = lines(output / 'inputs/questions.jsonl')
    candidates = lines(output / 'candidates.jsonl')
    candidate_map = {r['id']: r['candidate_statements'] for r in candidates}
    assert len(candidate_map) == len(candidates) == len(questions) == 1711
    assert set(candidate_map) == {q['id'] for q in questions}
    sources = [output / 'offline/protocol.json', output / 'offline/memories.jsonl',
               output / 'inputs/questions.jsonl', output / 'candidates.jsonl',
               output / 'candidate_generation_protocol.json', Path(__file__)]
    source_hashes = {str(p): sha(p) for p in sources}
    existing = output / 'retrieval_protocol.json'
    if existing.exists():
        previous = read(existing)
        if previous['source_sha256'] != source_hashes or previous['top_k'] != top_k:
            raise ValueError('Existing retrieval belongs to a different experiment.')
        assert sha(output / 'retrieval_results.jsonl') == previous['results_sha256']
        print('Mem-Gallery retrieval: verified existing results.', flush=True)
        return analyze_retrieval(output)
    runtime = SimpleNamespace(device=device)
    configure_device(runtime)
    model = load_text_encoder(offline['text_encoder'], runtime.device)
    clip, tokenizer, clip_limit = load_clip_text_encoder(offline['image_encoder'], runtime.device)
    results, domain_times = [], []
    started = time.perf_counter()
    for domain in dict.fromkeys(r['task'] for r in records):
        tick = time.perf_counter()
        memories = [r for r in records if r['task'] == domain]
        qs = [q for q in questions if q['task'] == domain]
        target = output / 'offline/memory' / domain
        graph = read(target / 'graph.json')
        masters = graph['masters']
        sums = np.load(target / 'master_sums.npy', allow_pickle=False)
        centers = normalize([sums[m['visual_mean_ref']['row']] / m['n'] for m in masters])
        master_indices = {m['id']: i for i, m in enumerate(masters)}
        nodes = {n['id']: n for n in graph['image_nodes']}
        visual_indices = [i for i, m in enumerate(memories) if m['is_visual']]
        leaf_master_indices = np.array([master_indices[nodes[memories[i]['node_id']]['master_id']]
                                        for i in visual_indices])
        for i in visual_indices:
            row, node = memories[i], nodes[memories[i]['node_id']]
            assert row['caption'] == node['caption'] and row['dialogue'] == node['dialogue']
        # Text vectors are re-encodable; no original image is read online.
        memory_vectors = encode_text(model, [r['retrieval_text'] for r in memories])
        hypotheses, query_inputs = [], []
        for q in qs:
            h = candidate_map[q['id']]
            if isinstance(h, dict):
                h = [h[k] for k in 'ABCD']
            assert len(h) == len(set(h)) == 4 and all(isinstance(s, str) and s.strip() for s in h)
            hypotheses.extend(h)
            query_inputs.extend([q['retrieval_query'], *[
                q['retrieval_query'] + '\nCandidate answer: ' + s for s in h]])
        queries = encode_text(model, query_inputs, offline['query_prefix']).reshape(len(qs), 5, -1)
        candidate_vectors = []
        for start in range(0, len(hypotheses), 64):
            tokens = tokenizer(hypotheses[start:start+64], padding=True, truncation=False, return_tensors='pt')
            assert tokens['input_ids'].shape[1] <= clip_limit
            tokens = tokens.to(next(clip.parameters()).device)
            with torch.inference_mode():
                candidate_vectors.extend(clip(**tokens, return_dict=True).text_embeds.double().cpu().numpy())
        candidate_vectors = normalize(candidate_vectors).reshape(len(qs), 4, -1)
        for qi, q in enumerate(qs):
            clip_scores = centers @ candidate_vectors[qi].T
            choices = np.argmax(clip_scores, axis=1)
            scores_by_query = queries[qi] @ memory_vectors.T
            baseline = scores_by_query[0]
            expanded = baseline.copy()
            expanded[visual_indices] = scores_by_query[1 + choices[leaf_master_indices], visual_indices]
            pure_text = [i for i, m in enumerate(memories) if not m['is_visual']]
            np.testing.assert_array_equal(expanded[pure_text], baseline[pure_text])
            conditions = []
            for name, scores in zip(CONDITIONS, (baseline, expanded)):
                order = np.argsort(-scores, kind='stable')
                top = order[:top_k]
                conditions.append({'name': name, 'top5_memory_ids': [memories[i]['id'] for i in top],
                                   'top5_scores': scores[top].tolist(),
                                   'ranking_memory_ids': [memories[i]['id'] for i in order]})
            results.append({'id': q['id'], 'task': domain, 'candidate_count': len(memories),
                            'conditions': conditions,
                            'master_choices': [{'master_id': m['id'], 'candidate_index': int(choices[i]),
                                                'clip_scores': clip_scores[i].tolist()} for i, m in enumerate(masters)]})
        domain_times.append({'task': domain, 'seconds': time.perf_counter()-tick, 'questions': len(qs)})
        print(json.dumps({'stage': 'mem_gallery_retrieval', **domain_times[-1]}, ensure_ascii=False), flush=True)
    assert {r['id'] for r in results} == {q['id'] for q in questions}
    assert len(results) == 1711
    write_lines(output / 'retrieval_results.jsonl', results)
    verify_hashes(source_hashes)
    dump(existing, {
        'dataset': 'Mem-Gallery', 'questions': len(results), 'top_k': top_k,
        'baseline': 'Full scan of all same-domain dialogue+full-caption memories using original query.',
        'method': 'Same full pool; image leaves use own visual master selected candidate expanded query; pure text leaves retain baseline scores.',
        'candidate_policy': 'Four frozen model-generated hypotheses from question and supplied query-image caption only; open-ended adaptation, no answer options supplied.',
        'reader_evidence': 'Identical original dialogue/full caption and date/image identifiers; hypotheses are retrieval-only.',
        'tie_break': 'Stable source memory order; four candidate ties follow candidate order.',
        'manual_similarity_thresholds': 0, 'score_fusion_weights': 0, 'leaf_distance_used': False,
        'query_image_policy': 'Provided caption for both retrieval conditions; original pixels only at reader.',
        'source_sha256': source_hashes, 'results_sha256': sha(output / 'retrieval_results.jsonl'),
        'execution_device': runtime.device, 'elapsed_seconds': time.perf_counter()-started,
        'domain_times': domain_times, 'gold_used_for_ranking': False,
    })
    return analyze_retrieval(output)


def analyze_retrieval(output=DEFAULT_OUTPUT):
    output = Path(output)
    ranked = lines(output / 'retrieval_results.jsonl')
    assert sha(output / 'retrieval_results.jsonl') == read(output / 'retrieval_protocol.json')['results_sha256']
    gold = {r['id']: r for r in lines(output / 'inputs/gold.jsonl')}
    questions = {r['id']: r for r in lines(output / 'inputs/questions.jsonl')}
    memories = {r['id']: r for r in lines(output / 'offline/memories.jsonl')}
    assert set(gold) == set(questions) == {r['id'] for r in ranked}
    rows = []
    for r in ranked:
        ids = set(gold[r['id']]['gold_ids'])
        q = questions[r['id']]
        visual = {mid for mid in ids if memories[mid]['is_visual']}
        pure = ids-visual
        cohort = 'no_clue' if not ids else 'mixed' if visual and pure else 'visual_only' if visual else 'text_only'
        row = {'id': r['id'], 'task': r['task'], 'point': q['point'],
               'query_has_image': bool(q['query_image_paths']), 'cohort': cohort,
               'gold_count': len(ids), 'visual_gold_count': len(visual), 'text_gold_count': len(pure)}
        for c in r['conditions']:
            top = set(c['top5_memory_ids'])
            rank = {mid: i for i, mid in enumerate(c['ranking_memory_ids'], 1)}
            assert set(rank) == {mid for mid, m in memories.items() if m['task'] == r['task']}
            row[c['name']] = {'hits': len(top & ids), 'visual_hits': len(top & visual),
                              'text_hits': len(top & pure), 'any_hit': bool(top & ids),
                              'all_hit': bool(ids) and ids <= top,
                              'reciprocal_rank': 1/min(rank[mid] for mid in ids) if ids else None}
        row['same_top5_order'] = r['conditions'][0]['top5_memory_ids'] == r['conditions'][1]['top5_memory_ids']
        rows.append(row)
    def summarize(group):
        labeled = [r for r in group if r['gold_count']]
        total = sum(r['gold_count'] for r in labeled)
        result = {'questions': len(group), 'questions_with_clues': len(labeled), 'gold_references': total,
                  'identical_ordered_top5': sum(r['same_top5_order'] for r in group)}
        for name in CONDITIONS:
            hits = sum(r[name]['hits'] for r in labeled)
            result[name] = {'gold_hits': hits, 'recall_at_5': hits/total if total else None,
                            'any_hit_rate': sum(r[name]['any_hit'] for r in labeled)/len(labeled) if labeled else None,
                            'all_hit_rate': sum(r[name]['all_hit'] for r in labeled)/len(labeled) if labeled else None,
                            'mrr': float(np.mean([r[name]['reciprocal_rank'] for r in labeled])) if labeled else None}
            for kind in ('visual', 'text'):
                n = sum(r[f'{kind}_gold_count'] for r in labeled)
                result[name][f'{kind}_recall_at_5'] = sum(r[name][f'{kind}_hits'] for r in labeled)/n if n else None
        return result
    summary = {'overall': summarize(rows),
               'by_domain': {v: summarize([r for r in rows if r['task'] == v]) for v in sorted({r['task'] for r in rows})},
               'by_point': {v: summarize([r for r in rows if r['point'] == v]) for v in sorted({r['point'] for r in rows})},
               'by_cohort': {v: summarize([r for r in rows if r['cohort'] == v]) for v in sorted({r['cohort'] for r in rows})},
               'no_clue_policy': 'All 184 no-clue questions remain in reader evaluation and are excluded only from supervised retrieval denominators.'}
    write_lines(output / 'retrieval_per_question.jsonl', rows)
    dump(output / 'retrieval_summary.json', summary)
    print(json.dumps({'stage': 'retrieval_summary', **summary['overall']}, ensure_ascii=False), flush=True)
    return summary


def offline(args):
    build(args.output, args.device)


def online(args):
    from scripts.evaluation.mem_gallery_master_reader import generate_candidates, run_reader, analyze
    if args.stage in ('prepare', 'all'):
        generate_candidates(args.output, device=args.device)
        retrieve(args.output, args.device, args.top_k)
    if args.stage in ('run', 'all'):
        run_reader(args.output, device=args.device)
    if args.stage in ('analyze', 'all'):
        analyze_retrieval(args.output)
        analyze(args.output)
