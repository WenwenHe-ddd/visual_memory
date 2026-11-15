"""Matched Mem-Gallery online z-score addition; frozen captions/graphs/queries.

Run prepare, reader, analyze separately or with --stage all. The original reader
module is reused unchanged, with only its two condition labels set in this
process. No original result file or offline memory is modified.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
from scripts.memory import mem_gallery_master_answer as original
from scripts.memory.online_zscore import ZScoreWeighting
from scripts.evaluation import mem_gallery_master_reader as reader

SOURCE = original.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_qwen_zscore_l0p03_a1_20261027'
BASE_CONDITIONS = original.CONDITIONS
CONDITIONS = tuple(c + '+zscore' for c in BASE_CONDITIONS)
TEXT_METHODS = ROOT / 'results/mem_gallery_qwen_seed42_text_methods_20261026_v2'
read, lines, dump, write_lines = original.read, original.lines, original.dump, original.write_lines
sha, verify_hashes = original.sha, original.verify_hashes


def group_scores(vectors, queries, groups, ridge, alpha):
    """Apply the production component to each supplied full group, before Top-5.

    Each group is (indices, query_index). Ungrouped leaves retain query 0 cosine;
    this is the original Gallery fallback for text-only memories without masters.
    """
    raw = vectors @ queries[0]
    weighted = raw.copy()
    details = []
    component = ZScoreWeighting(ridge, alpha)
    covered = set()
    for indices, qi in groups:
        indices = np.asarray(indices, dtype=int)
        assert len(indices) and not covered.intersection(indices.tolist())
        covered.update(indices.tolist())
        stats = component.apply(queries[qi], vectors[indices])
        raw[indices] = stats['raw']
        weighted[indices] = stats['weighted']
        details.append({'indices': indices.tolist(), 'query_index': int(qi),
                        **{k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in stats.items()}})
    return raw, weighted, details


def prepare(output, device, ridge, alpha):
    output = Path(output).resolve()
    protocol_path = output / 'retrieval_protocol.json'
    if protocol_path.exists():
        p = read(protocol_path)
        assert p['lambda'] == ridge and p['alpha'] == alpha
        verify_hashes(p['source_sha256'])
        verify_hashes(p['artifact_sha256'])
        return p
    offline = read(SOURCE / 'offline/protocol.json')
    reference = read(SOURCE / 'retrieval_protocol.json')
    verify_hashes(offline['source_sha256'])
    verify_hashes(offline['artifact_sha256'])
    verify_hashes(reference['source_sha256'])
    assert sha(SOURCE / 'retrieval_results.jsonl') == reference['results_sha256']
    input_protocol = read(SOURCE / 'inputs/protocol.json')
    verify_hashes({str(ROOT / p): v for p, v in input_protocol['source_sha256'].items()})
    sources = {**offline['source_sha256'], **offline['artifact_sha256'],
               **reference['source_sha256']}
    for path in [Path(__file__), Path(reader.__file__),
                 ROOT / 'scripts/memory/online_zscore.py',
                 ROOT / 'scripts/memory/ridge_whitening.py',
                 ROOT / 'scripts/memory/caption_visual_master.py',
                 SOURCE / 'retrieval_results.jsonl', SOURCE / 'reader/protocol.json',
                 ROOT / 'results/memeye_qwen_zscore_joint_search_20261024/search_results.csv']:
        sources[str(path)] = sha(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inputs').mkdir(exist_ok=True)
    for name in ('questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json'):
        src, dst = SOURCE / 'inputs' / name, output / 'inputs' / name
        if dst.exists():
            assert sha(src) == sha(dst)
        else:
            shutil.copyfile(src, dst)
        sources[str(src)] = sha(src)
        sources[str(dst)] = sha(dst)
    questions = lines(output / 'inputs/questions.jsonl')
    records = lines(SOURCE / 'offline/memories.jsonl')
    candidate_map = {r['id']: r['candidate_statements'] for r in lines(SOURCE / 'candidates.jsonl')}
    ref = {r['id']: r for r in lines(SOURCE / 'retrieval_results.jsonl')}
    assert len(questions) == len(candidate_map) == len(ref) == 1711
    assert len(records) == 3962
    assert not any({'answer', 'gold_ids', 'clue'} & set(q) for q in questions)
    args = SimpleNamespace(device=device)
    original.configure_device(args)
    encoder = original.load_text_encoder(offline['text_encoder'], args.device)
    results, diagnostics = [], []
    raw_top5_checks = 0
    changed = dict.fromkeys(CONDITIONS, 0)
    changed_sets = dict.fromkeys(CONDITIONS, 0)
    full_rank_changes = 0
    max_score_error = 0.
    for domain in dict.fromkeys(m['task'] for m in records):
        memories = [m for m in records if m['task'] == domain]
        qs = [q for q in questions if q['task'] == domain]
        graph = read(SOURCE / 'offline/memory' / domain / 'graph.json')
        node_map = {n['id']: n for n in graph['image_nodes']}
        groups = {m['id']: [] for m in graph['masters']}
        for i, m in enumerate(memories):
            if m['is_visual']:
                n = node_map[m['node_id']]
                assert (m['caption'], m['dialogue']) == (n['caption'], n['dialogue'])
                groups[n['master_id']].append(i)
        vectors = original.encode_text(encoder, [m['retrieval_text'] for m in memories])
        query_inputs = []
        for q in qs:
            candidates = candidate_map[q['id']]
            if isinstance(candidates, dict):
                candidates = [candidates[k] for k in 'ABCD']
            assert len(candidates) == len(set(candidates)) == 4
            query_inputs.extend([q['retrieval_query'], *[
                q['retrieval_query'] + '\nCandidate answer: ' + s for s in candidates]])
        queries = original.encode_text(encoder, query_inputs, offline['query_prefix']).reshape(len(qs), 5, -1)
        for qi, q in enumerate(qs):
            # Reuse already frozen CLIP selections: weighting does not select candidates.
            old = ref[q['id']]
            choices = {c['master_id']: c['candidate_index'] for c in old['master_choices']}
            assert set(choices) == set(groups)
            condition_groups = [([(list(range(len(memories))), 0)]),
                                [(idx, 1 + choices[mid]) for mid, idx in groups.items()]]
            conditions, detail = [], {'id': q['id'], 'memory_ids': [m['id'] for m in memories], 'conditions': {}}
            for name, previous_name, supplied_groups in zip(CONDITIONS, BASE_CONDITIONS, condition_groups):
                raw, weighted, stats = group_scores(vectors, queries[qi], supplied_groups, ridge, alpha)
                previous = next(c for c in old['conditions'] if c['name'] == previous_name)
                raw_order = np.argsort(-raw, kind='stable')
                raw_ids = [memories[i]['id'] for i in raw_order]
                assert raw_ids[:5] == previous['top5_memory_ids'], (q['id'], previous_name)
                error = float(np.max(np.abs(raw[raw_order[:5]] - previous['top5_scores'])))
                assert error < 2e-6, (q['id'], error)
                max_score_error = max(error, max_score_error)
                full_rank_changes += raw_ids != previous['ranking_memory_ids']
                raw_top5_checks += 1
                order = np.argsort(-weighted, kind='stable')
                ids = [memories[i]['id'] for i in order]
                changed[name] += ids[:5] != raw_ids[:5]
                changed_sets[name] += set(ids[:5]) != set(raw_ids[:5])
                conditions.append({'name': name, 'top5_memory_ids': ids[:5],
                                   'top5_scores': weighted[order[:5]].tolist(),
                                   'ranking_memory_ids': ids})
                detail['conditions'][name] = {'raw': raw.tolist(), 'weighted': weighted.tolist(), 'groups': stats}
            results.append({'id': q['id'], 'task': domain, 'candidate_count': len(memories),
                            'conditions': conditions, 'master_choices': old['master_choices']})
            diagnostics.append(detail)
        print(json.dumps({'stage': 'zscore_retrieval', 'domain': domain, 'completed': len(results), 'total': 1711}), flush=True)
    assert len(results) == len({r['id'] for r in results}) == 1711
    write_lines(output / 'retrieval_results.jsonl', results)
    write_lines(output / 'weighting_diagnostics.jsonl', diagnostics)
    verify_hashes(sources)
    dimensions = {
        CONDITIONS[0]: {
            '1_candidate_memory_structure': 'All same-scenario original dialogue rounds, flat; visual rounds contain full Qwen caption. One whitening group per scenario.',
            '2_query_content': 'Original question plus supplied official query-image caption when available; no answer statement.',
            '3_retrieval_method': 'Original BGE cosine plus online ridge/z-score bounded correction over the entire pool, then stable Top-5.',
        },
        CONDITIONS[1]: {
            '1_candidate_memory_structure': 'The same full pool; existing caption_visual_master groups for visual rounds. Text-only rounds have no master.',
            '2_query_content': 'Frozen original question/caption plus the original CLIP-selected complete candidate statement per master; original query for text-only rounds.',
            '3_retrieval_method': 'Original BGE cosine plus within-master online ridge/z-score correction for visual rounds. Text-only scores unchanged. Merge and take stable global Top-5.',
        },
    }
    for d in dimensions.values():
        d['4_encoding'] = 'Pinned BAAI/bge-small-en-v1.5, 384D L2, dialogue+full caption, original query prefix; refuse truncation. Original frozen CLIP master choices.'
        d['5_answering'] = 'Qwen3-VL-8B-Instruct, seed42, greedy, bf16 SDPA, 256 output tokens, original reader prompts and five full original memories; original query pixels supplied. No candidate statements in reader.'
    p = {'dataset': 'Mem-Gallery', 'questions': 1711, 'top_k': 5,
         'lambda': ridge, 'alpha': alpha, 'online_weighting': 'z_score',
         'formula': 'raw + alpha * population_std(raw_in_group) * tanh(population_zscore(ridge_whitened_scores_in_group))',
         'parameter_policy': 'Fixed before Gallery scoring; lambda=.03 alpha=1 matches prior Qwen MemEye 185/371 and 199/371 comparison, not a Gallery search.',
         'source_run': str(SOURCE), 'conditions': dimensions,
         'caption_graph_candidates_reused_unchanged': True, 'gold_read_by_retrieval': False,
         'vectors_persisted': False, 'device': args.device,
         'raw_top5_checks': raw_top5_checks, 'raw_full_rank_changes': full_rank_changes,
         'max_raw_top5_score_error': max_score_error, 'changed_ordered_top5': changed, 'changed_top5_sets': changed_sets,
         'source_sha256': sources, 'artifact_sha256': {str(output / f): sha(output / f) for f in
             ['retrieval_results.jsonl', 'weighting_diagnostics.jsonl']}}
    dump(protocol_path, p)
    return p


def verify(output):
    p = read(output / 'retrieval_protocol.json')
    verify_hashes(p['source_sha256'])
    verify_hashes(p['artifact_sha256'])
    return p


def run_reader(output, device):
    verify(output)
    # Runtime-only condition names; all prompting, inference, and dedup code is unchanged.
    reader.CONDITIONS = CONDITIONS
    result = reader.run_reader(output, device=device, batch_size=16, image_batch_size=4)
    current = read(output / 'reader/protocol.json')['contract']
    old = read(SOURCE / 'reader/protocol.json')['contract']
    for key in ('model', 'system_prompt', 'task_instructions', 'official_prompt_revision',
                'seed', 'do_sample', 'max_new_tokens', 'batch_size_text', 'batch_size_images',
                'dtype', 'query_image_max_pixels', 'input_truncation', 'candidate_expansion_in_reader'):
        assert current[key] == old[key], key
    verify(output)
    return result


def analyze(output):
    protocol = verify(output)
    reader.CONDITIONS = CONDITIONS
    summary = reader.analyze(output)
    pairs = read(output / 'reader/paired_results.json')
    old_pairs = {r['id']: r for r in read(SOURCE / 'reader/paired_results.json')}
    assert len(pairs) == len(old_pairs) == 1711
    rows, comparisons = [], {}
    for base, weighted in zip(BASE_CONDITIONS, CONDITIONS):
        for name, selected in ((base, [old_pairs[p['id']][base] for p in pairs]),
                               (weighted, [p[weighted] for p in pairs])):
            clues = [r for r in selected if r['gold_clues']]
            rows.append(dict(method=name, correct=sum(int(r['em']) for r in selected), questions=len(selected),
                em=sum(r['em'] for r in selected)/len(selected), f1=sum(r['f1'] for r in selected)/len(selected),
                recall_at_5_micro=sum(r['gold_hits'] for r in clues)/sum(r['gold_clues'] for r in clues),
                recall_at_5_macro=sum(r['gold_hits']/r['gold_clues'] for r in clues)/len(clues),
                gold_hits=sum(r['gold_hits'] for r in clues), gold_links=sum(r['gold_clues'] for r in clues),
                empty_answers=sum(not r['raw_response'] for r in selected),
                token_limit_records=sum(r['hit_token_limit'] for r in selected),
                result_source=str(output if name == weighted else SOURCE)))
        comparisons[weighted] = {'wrong_to_right': sum(p[weighted]['em'] > old_pairs[p['id']][base]['em'] for p in pairs),
                                'right_to_wrong': sum(p[weighted]['em'] < old_pairs[p['id']][base]['em'] for p in pairs),
                                'same_ordered_top5': sum(p[weighted]['top5_memory_ids'] == old_pairs[p['id']][base]['top5_memory_ids'] for p in pairs)}
    result = {'status': 'complete', 'lambda': protocol['lambda'], 'alpha': protocol['alpha'], 'model': 'Qwen3-VL-8B-Instruct',
              'seed': 42, 'conditions': protocol['conditions'], 'results': rows, 'paired_vs_historical': comparisons,
              'metric_policy': summary['metric_policy'], 'retrieval_denominator_policy': summary['retrieval_denominator_policy'],
              'comparison_note': 'Unweighted answers are the original frozen run; both z-score variants receive fresh inference. Identical inputs can still differ across batches under greedy GPU execution.'}
    dump(output / 'comparison.json', result)
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    table = ''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{r["correct"]}/1711</td><td>{100*r["em"]:.4f}%</td><td>{100*r["f1"]:.4f}%</td><td>{100*r["recall_at_5_micro"]:.4f}%</td></tr>' for r in rows)
    extra = ''
    if (TEXT_METHODS / 'final_results.json').exists():
        native = read(TEXT_METHODS / 'final_results.json')
        for name, v in native['methods'].items():
            s = v['overall']; k5 = name in ('simplemem', 'a_mem')
            extra += f'<tr><td>{html.escape(v["name"])}</td><td>{s["correct"]}/1711</td><td>{100*s["em"]:.4f}%</td><td>{100*s["f1"]:.4f}%</td><td>{100*s["source_recall_micro"]:.4f}%'+('' if k5 else ' (native context, not R@5)')+'</td></tr>'
        result['previous_four_methods_source'] = str(TEXT_METHODS / 'final_results.json')
        result['previous_four_methods_sha256'] = sha(TEXT_METHODS / 'final_results.json')
        dump(output / 'comparison.json', result)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Mem-Gallery Qwen + z-score</title><style>body{font:16px system-ui;margin:36px;max-width:1200px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}code{background:#eee}</style><h1>Mem-Gallery · Qwen3-VL-8B</h1>'
        f'<p>λ={protocol["lambda"]}, α={protocol["alpha"]}, seed=42. 1711 questions. Normalized exact match; no LLM judge.</p><table><tr><th>Method</th><th>Correct</th><th>EM</th><th>F1</th><th>Recall@5 (micro)</th></tr>'+table+extra+'</table><p>Recall uses 4109 gold source-round links across 1527 questions; all 184 empty-clue questions remain in EM/F1. Original no-z results are frozen historical references. Z-score variants receive fresh answers. The master variant retains original scores for text-only memories outside visual masters.</p><p><a href="comparison.json">Protocol and results</a> · <a href="metrics.csv">CSV</a> · <a href="reader/paired_results.csv">Per-question results</a></p>')
    print(json.dumps({'stage': 'complete', 'results': rows}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'reader', 'analyze', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--whitening-lambda', type=float, default=.03)
    parser.add_argument('--reweight-alpha', type=float, default=1.)
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.output, args.device, args.whitening_lambda, args.reweight_alpha)
    if args.stage in ('reader', 'all'):
        run_reader(args.output, args.device)
    if args.stage in ('analyze', 'all'):
        analyze(args.output)


if __name__ == '__main__':
    main()
