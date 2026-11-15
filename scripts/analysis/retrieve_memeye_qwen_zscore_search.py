"""Fixed 25-pair Qwen search: two retrievals, one shared pair, frozen Qwen memory."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for k, v in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from scripts.analysis.evaluate_adaptive_changed_qa import read, lines, dump, sha, digest
from scripts.memory import online_retrieval as online
from scripts.memory.online_zscore import ZScoreWeighting

BASE = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
ORIGINAL = ROOT / 'data/independent_visual_inputs'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021'
MASTER_REF = ROOT / 'reports/master_answer_query_zscore_20261023'
FULL_REF = ROOT / 'reports/baseline_question_only_lambda_ablation_20261023'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
STORES = ROOT / 'results/memory_stores_20261023'
METHODS = ('baseline_question_only', 'master_answer_query')
LAMBDAS = (.001, .003, .01, .03, .1)
ALPHAS = (.1, .3, .6, 1., 2.)


def tag(x):
    return format(x, 'g').replace('.', 'p')


def condition(method, ridge, alpha):
    return method if alpha == 0 else method + '__l' + tag(ridge) + '_a' + tag(alpha)


SPECS = {m: {'method': m, 'lambda': None, 'alpha': 0.} for m in METHODS}
SPECS.update({condition(m, l, a): {'method': m, 'lambda': l, 'alpha': a}
             for l in LAMBDAS for a in ALPHAS for m in METHODS})
SELECTION = {
    'shared_pair': True,
    'eligibility': ['baseline_zscore > baseline_unweighted', 'master_zscore > master_unweighted', 'master_zscore > baseline_zscore'],
    'priority': ['highest_master_correct', 'highest_baseline_correct', 'smaller_alpha', 'smaller_lambda'],
    'gpt5_validation': 'Freeze one Qwen-selected pair before API calls. One fresh four-condition GPT-5 run, sharing identical full requests. No GPT-5-guided parameter reselection.',
    'metric': 'Fixed-option MCQ accuracy, 371 questions; invalid outputs remain incorrect in the denominator.',
    'scope': 'Exploratory search and backbone-transfer validation on the same benchmark, not held-out generalization.'}


def write_lines(p, records):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))


def verify_hashes(hashes):
    for p, h in hashes.items():
        assert sha(p) == h, p


def retrieve(output=BASE, device='cuda:1'):
    output = Path(output).resolve()
    if (output / 'protocol.json').exists():
        p = read(output / 'protocol.json')
        verify_hashes(p['source_sha256']); verify_hashes(p['output_sha256'])
        assert p['condition_specs'] == SPECS and p['selection_rule'] == SELECTION
        return p
    assert not (output / 'reader/protocol.json').exists()
    qs = lines(ORIGINAL / 'qa_manifest.jsonl')
    raw_ref = {r['question_id']: r for r in lines(REFERENCE / 'retrieval_results.jsonl')}
    master_ref = {r['question_id']: r for r in lines(MASTER_REF / 'retrieval_results.jsonl')}
    full_ref = {r['question_id']: r for r in lines(FULL_REF / 'retrieval_results.jsonl')}
    assert len(qs) == len(raw_ref) == len(master_ref) == len(full_ref) == 371
    sources = [Path(__file__), ORIGINAL / 'qa_manifest.jsonl', ORIGINAL / 'retrieval_memories.jsonl',
               REFERENCE / 'protocol.json', REFERENCE / 'retrieval_results.jsonl', REFERENCE / 'candidates.jsonl',
               MASTER_REF / 'retrieval_results.jsonl', FULL_REF / 'retrieval_results.jsonl',
               ROOT / 'scripts/memory/online_retrieval.py', ROOT / 'scripts/memory/online_zscore.py',
               ROOT / 'scripts/memory/ridge_whitening.py', ROOT / 'scripts/memory/caption_visual_master.py',
               ROOT / 'scripts/memory/memory_store.py']
    hashes = {str(p): sha(p) for p in sources}
    pinned = read(REFERENCE / 'protocol.json')['source_sha256']
    assert hashes[str(REFERENCE / 'candidates.jsonl')] == pinned[str(REFERENCE / 'candidates.jsonl')]
    args = SimpleNamespace(device=device)
    online.configure_device(args)
    output.mkdir(parents=True, exist_ok=True)
    results = {}; raw_checks = group_checks = 0
    max_raw_error = max_white_error = max_z_error = max_inverse_error = 0.
    master_counts = {}
    source_texts = {m['id']: m['reader_text'] for m in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    for domain in sorted({q['task'] for q in qs}):
        src = INPUTS / domain / 'inputs.json'
        assert sha(src) == pinned[str(src)]
        hashes[str(src)] = sha(src)
        question_rows = read(src)['questions']
        # Gold fields are removed before planning, encoding, scoring or ranking.
        questions = [{k: v for k, v in q.items() if k not in ('gold_ids', 'gold_choice')} for q in question_rows]
        args.mode = 'caption_visual_master'; args.memory_dir = STORES / domain / args.mode
        args.candidates = REFERENCE / 'candidates.jsonl'; args.retrieval = 'multi_path'
        memory, metadata, paths, devices = online.load_memory(args, questions)
        hashes.update({str(p): sha(p) for p in paths})
        natural_args = SimpleNamespace(mode='natural', memory_dir=STORES / domain / 'natural')
        natural, natural_metadata, paths, _ = online.load_memory(natural_args, questions)
        hashes.update({str(p): sha(p) for p in paths})
        assert memory.texts == natural.texts and metadata['text_encoder'] == natural_metadata['text_encoder']
        assert all(text == source_texts[leaf['memory_id']] for text, leaf in zip(memory.texts, memory.leaves))
        master_counts[domain] = len(memory.masters)
        encoder = online.load_text_encoder(metadata['text_encoder'], args.device)
        for q in questions:
            qid = q['id']; old = raw_ref[qid]
            assert q['retrieval_query'] == old['baseline_retrieval_query']
            vectors = online.encode_memory_text(memory, encoder)
            item = {'question_id': qid, 'domain': domain, 'methods': {}, 'conditions': {}}
            for method, strategy in zip(METHODS, (online.FullScanRetrieval(), online.MultiPathRetrieval())):
                obj = natural if method == METHODS[0] else memory
                texts, groups = strategy.plan(q, obj)
                queries = online.encode_text(encoder, texts, metadata['query_prefix'])
                ids = [l['memory_id'] for l in memory.leaves]
                raws = np.empty(len(ids)); stats = {}; group_info = []
                for ridge in LAMBDAS:
                    white, z, unit = (np.empty(len(ids)) for _ in range(3))
                    for g in groups:
                        ix = np.array(g.indices); x = vectors[ix]; query = queries[g.query_index]
                        scores = ZScoreWeighting(ridge, 1.).apply(query, x)
                        centered = x - x.mean(0)
                        adjusted = query if len(x) == 1 else query - centered.T @ np.linalg.solve(
                            centered @ centered.T + (len(x) - 1) * ridge * np.eye(len(x)), centered @ query)
                        error = float(np.max(np.abs(scores['white'] - x @ adjusted)))
                        assert error < 2e-12; max_inverse_error = max(max_inverse_error, error)
                        raws[ix], white[ix], z[ix], unit[ix] = (scores[k] for k in ('raw', 'white', 'z', 'correction'))
                        if ridge == LAMBDAS[0]:
                            group_info.append({'id': g.id, 'indices': list(g.indices), 'query': texts[g.query_index],
                                               'raw_std': scores['raw_std'], 'raw_status': scores['raw_status'], **g.details})
                        group_checks += 1
                    stats[tag(ridge)] = {'white': white.tolist(), 'z': z.tolist(), 'unit_correction': unit.tolist()}
                baseline = next(c for c in old['conditions'] if c['name'] == method)
                raw_order = np.argsort(-raws, kind='stable')
                assert [ids[i] for i in raw_order[:5]] == baseline['top5_memory_ids']
                raw_checks += 1
                ref = master_ref[qid] if method == METHODS[1] else full_ref[qid]['weighting']['lambda_0p01']
                assert ids == [l['memory_id'] for l in ref['leaf_scores']]
                error = max(abs(raws[i] - l['raw_score']) for i, l in enumerate(ref['leaf_scores']))
                assert error < 2e-6; max_raw_error = max(max_raw_error, error)
                for key, field in [('white', 'white_score'), ('z', 'white_z')]:
                    error = max(abs(stats['0p01'][key][i] - l[field]) for i, l in enumerate(ref['leaf_scores']))
                    assert error < 2e-6
                    if key == 'white': max_white_error = max(max_white_error, error)
                    else: max_z_error = max(max_z_error, error)
                for g in group_info:
                    ref_g = next(t for t in ref['groups'] if t['group_id'] == g['id'])
                    assert g['query'] == ref_g['query'] and len(g['indices']) == ref_g['size']
                    if method == METHODS[1]:
                        assert all(g[k] == ref_g[k] for k in ('choice', 'selected_statement'))
                        np.testing.assert_allclose(list(g['clip_statement_scores'].values()), list(ref_g['clip_statement_scores'].values()), rtol=0, atol=2e-6)
                item['methods'][method] = {'memory_ids': ids, 'raw': raws.tolist(), 'groups': group_info, 'weighting': stats}
                for name, spec in SPECS.items():
                    if spec['method'] != method: continue
                    if spec['alpha'] == 0:
                        values = raws
                    else:
                        values = raws.copy(); zs = np.asarray(stats[tag(spec['lambda'])]['z'])
                        units = np.asarray(stats[tag(spec['lambda'])]['unit_correction'])
                        for g in group_info:
                            ix = np.asarray(g['indices'])
                            # Same multiplication order as the online component; zero groups stay zero.
                            correction = spec['alpha'] * g['raw_std'] * np.tanh(zs[ix])
                            correction = np.where(units[ix] == 0., 0., correction)
                            values[ix] += correction
                    order = np.argsort(-values, kind='stable')
                    item['conditions'][name] = {'top5_memory_ids': [ids[i] for i in order[:5]],
                                                'top5_scores': [float(values[i]) for i in order[:5]],
                                                'full_rank_memory_ids': [ids[i] for i in order]}
            results[qid] = item
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'questions': len(results), 'raw_checks': raw_checks}), flush=True)
        del encoder, memory, natural, vectors, queries
        gc.collect()
    assert len(results) == 371 and raw_checks == 742 and sum(master_counts.values()) == 88
    write_lines(output / 'retrieval_results.jsonl', [results[q['id']] for q in qs])
    verify_hashes(hashes)
    protocol = {'stage': 'qwen_joint_search', 'questions': 371, 'memory_path': str(STORES), 'memory_leaves': 438,
                'masters': 88, 'master_counts': master_counts, 'lambdas': list(LAMBDAS), 'alphas': list(ALPHAS),
                'condition_specs': SPECS, 'selection_rule': SELECTION, 'device': args.device,
                'source_sha256': hashes, 'output_sha256': {str(output / 'retrieval_results.jsonl'): sha(output / 'retrieval_results.jsonl')},
                'raw_top5_verified': raw_checks, 'max_raw_error_vs_reference': max_raw_error,
                'max_white_error_vs_reference': max_white_error, 'max_z_error_vs_reference': max_z_error,
                'independent_dual_ridge_groups': group_checks, 'max_inverse_error': max_inverse_error,
                'vectors_persisted': False, 'memory_modified': False, 'caption_regenerated': False,
                'gold_used_for_ranking': False, 'score_formula': 'raw + alpha * std_pop(raw in group) * tanh(z_white)'}
    dump(output / 'protocol.json', protocol)
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE); p.add_argument('--device', default='cuda:1')
    a = p.parse_args(); result = retrieve(a.output, a.device)
    print(json.dumps({k: result[k] for k in ('questions', 'masters', 'raw_top5_verified', 'max_raw_error_vs_reference', 'max_inverse_error')}), flush=True)
