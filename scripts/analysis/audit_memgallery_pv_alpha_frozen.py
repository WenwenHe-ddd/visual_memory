"""Independent ranking, cached-answer provenance and metric audit for Gallery PV alpha."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.analysis.audit_memgallery_positive_variance_ablation import read, lines, sha, digest, check, normalize

DEFAULT = ROOT / 'results/memgallery_qwen_pv_original_answers_alpha_ablation_l0p03_20261107'
ALPHAS = [.1, .3, .6, 1., 3., 6., 10.]
METHODS = ['baseline_question_only', 'master_answer_query']


def audit_retrieval(base):
    p = read(base / 'retrieval_protocol.json')
    check(p['source_sha256']); check(p['artifact_sha256'])
    assert p['lambda'] == .03 and p['alphas'] == ALPHAS
    specs = p['condition_specs']; assert len(specs) == 16
    questions = {q['id']: q for q in lines(base / 'inputs/questions.jsonl')}
    memories = {m['id']: m for m in lines(base / 'inputs/memories.jsonl')}
    rows = lines(base / 'retrieval_results.jsonl')
    ranked = {r['id']: r for r in rows}
    assert len(rows) == len(ranked) == len(questions) == 1711 and set(questions) == set(ranked)
    source = Path(p['source_run'])
    prior = {q['id']: q for q in lines(Path(p['answer_cache_run']) / 'retrieval_results.jsonl')}
    candidates = {q['id']: q['candidate_statements'] for q in lines(source / 'candidates.jsonl')}
    checked = set(); rankings = raw_matches = alpha1_matches = group_checks = 0
    for path in sorted(Path(p['fixed_lambda_diagnostics']).glob('*.jsonl')):
        graph = read(source / 'offline/memory' / path.stem / 'graph.json')
        nodes = {n['id']: n for n in graph['image_nodes']}
        for d in lines(path):
            qid = d['id']; assert qid not in checked; checked.add(qid)
            q = questions[qid]; ids = d['memory_ids']
            assert len(ids) == len(set(ids)) and set(ids) == {mid for mid, m in memories.items() if m['task'] == q['task']}
            assert path.stem == d['task'] == ranked[qid]['task'] == q['task']
            assert not {'answer', 'gold_ids', 'clue', 'gold_choice'} & set(q)
            text_indices = [i for i, mid in enumerate(ids) if not memories[mid]['is_visual']]
            assert text_indices == d['text_only_indices']
            choices = {r['master_id']: r['candidate_index'] for r in prior[qid]['master_choices']}
            assert ranked[qid]['master_choices'] == prior[qid]['master_choices']
            statements = candidates[qid]
            if isinstance(statements, dict): statements = [statements[k] for k in 'ABCD']
            actual = {r['name']: r for r in ranked[qid]['conditions']}
            anchors = {r['name']: r for r in prior[qid]['conditions']}
            assert set(actual) == set(specs)
            for method, data in d['methods'].items():
                raw = np.asarray(data['raw']); stats = data['weighting']['0.03']
                unit = np.asarray(stats['correction']); z = np.asarray(stats['z'])
                enhanced = np.asarray(stats['enhanced'])
                assert len(raw) == len(ids) and all(np.isfinite(x).all() for x in [raw, unit, z, enhanced])
                groups = data['groups']; covered = [i for group in groups for i in group['indices']]
                assert len(covered) == len(set(covered))
                assert sorted(covered) == (list(range(len(ids))) if method == METHODS[0] else [i for i in range(len(ids)) if i not in text_indices])
                expected_unit = np.zeros(len(ids))
                for group in groups:
                    ix = group['indices']
                    if method == METHODS[0]:
                        assert len(groups) == 1 and group['query'] == q['retrieval_query'] and group['query_index'] == 0
                    else:
                        ci = choices[group['id']]
                        assert group['query'] == q['retrieval_query'] + '\nCandidate answer: ' + statements[ci]
                        assert group['query_index'] == 1 + ci
                        assert all(nodes[memories[ids[i]]['node_id']]['master_id'] == group['id'] for i in ix)
                    local = enhanced[ix]; std = np.std(local)
                    tol = 32 * np.finfo(float).eps * max(1., max(abs(local)))
                    expected_z = local if len(ix) == 1 else (np.zeros(len(ix)) if std <= tol else (local - local.mean()) / std)
                    assert np.allclose(z[ix], expected_z, rtol=0, atol=1e-10)
                    if len(ix) > 1 and group['raw_status'] != 'zero_variance':
                        expected_unit[ix] = np.std(raw[ix]) * np.tanh(expected_z)
                    group_checks += 1
                assert np.allclose(unit, expected_unit, rtol=0, atol=1e-12)
                assert np.array_equal(raw + unit, np.asarray(stats['weighted']))
                if method == METHODS[1]: assert not unit[text_indices].any()
                for name, spec in specs.items():
                    if spec['method'] != method: continue
                    if spec['lambda'] is not None: assert spec['lambda'] == .03 and spec['alpha'] in ALPHAS
                    values = raw + spec['alpha'] * unit
                    order = sorted(range(len(ids)), key=lambda i: (-values[i], i))[:5]
                    assert actual[name]['top5_memory_ids'] == [ids[i] for i in order]
                    assert np.allclose(actual[name]['top5_scores'], values[order], rtol=0, atol=1e-12)
                    if method == METHODS[1]: assert np.array_equal(values[text_indices], raw[text_indices])
                    if spec['lambda'] is None or spec['alpha'] == 1.:
                        assert actual[name] == anchors[name]
                        raw_matches += spec['lambda'] is None
                        alpha1_matches += spec['lambda'] is not None
                    rankings += 1
    assert len(checked) == 1711 and rankings == 27376
    assert raw_matches == alpha1_matches == 3422
    return {'passed': True, 'rankings_checked': rankings, 'raw_top5_matches': raw_matches,
        'alpha1_top5_matches': alpha1_matches, 'group_checks': group_checks,
        'only_alpha_scaled': True, 'master_groups_queries_and_text_only_fallback_unchanged': True}


def audit(base):
    from nltk.stem import PorterStemmer
    retrieval = audit_retrieval(base)
    p = read(base / 'reader/protocol.json'); c = p['contract']
    assert digest(c) == p['experiment_signature']
    check(c['source_sha256'])
    assert c['seed'] == 42 and c['do_sample'] is False and c['historical_response_reuse']
    assert not c['gold_loaded_before_analysis'] and not c['candidate_expansion_in_reader']
    roots = [Path(r) for r in c['answer_cache_priority']]
    cache, source_rows, source_manifests, source_protocols = {}, {}, {}, {}
    for root in roots:
        sp = read(root / 'reader/protocol.json')
        assert digest(sp['contract']) == sp['experiment_signature']
        check(sp['contract']['source_sha256'])
        for key in ['model', 'system_prompt', 'task_instructions', 'seed', 'do_sample', 'max_new_tokens',
                    'batch_size_text', 'batch_size_images', 'dtype', 'query_image_max_pixels', 'input_truncation']:
            assert c[key] == sp['contract'][key]
        sm = lines(root / 'reader/qa_manifest.jsonl')
        assert digest(sm) == sp['contract']['manifest_sha256']
        sr = {r['record_id']: r for r in lines(root / 'reader/responses.jsonl')}
        sm = {m['record_id']: m for m in sm}
        assert set(sm) == set(sr)
        for rid, r in sr.items():
            h = r['payload_sha256']; assert h == sm[rid]['payload_sha256']
            if h in cache: assert cache[h]['answer'] == r['raw_response']
            else: cache[h] = {'root': str(root), 'answer': r['raw_response']}
        source_rows[str(root)] = sr; source_manifests[str(root)] = sm; source_protocols[str(root)] = sp
    assert len(cache) == 4588
    original_m, original_r = source_manifests[str(roots[0])], source_rows[str(roots[0])]
    manifest = lines(base / 'reader/qa_manifest.jsonl')
    assert digest(manifest) == c['manifest_sha256']
    responses = lines(base / 'reader/responses.jsonl')
    mm = {m['record_id']: m for m in manifest}; rr = {r['record_id']: r for r in responses}
    assert len(manifest) == len(mm) == len(responses) == len(rr) == 27376 and set(mm) == set(rr)
    plan = read(base / 'reader/inference_plan.json')
    assert sha(base / 'reader/inference_plan.json') == c['batch_plan_sha256']
    assert sha(base / 'reader/reuse_manifest.jsonl') == c['reuse_manifest_sha256']
    assignments = {a['record_id']: a for a in lines(base / 'reader/reuse_manifest.jsonl')}
    assert set(assignments) == set(mm)
    fresh = {}
    for batch in plan['batches']:
        saved = read(base / 'reader/batches' / f'{batch["batch_index"]:04d}.json')
        assert saved['plan'] == batch and saved['experiment_signature'] == p['experiment_signature']
        assert len(saved['responses']) == len(batch['record_ids']) == len(batch['payloads'])
        assert len(batch['record_ids']) <= (4 if batch['has_image'] else 16)
        for rid, h, r in zip(batch['record_ids'], batch['payloads'], saved['responses']):
            assert rid == r['record_id'] and h == r['payload_sha256'] == mm[rid]['payload_sha256']
            assert h not in cache and h not in fresh
            spec = c['condition_specs'][mm[rid]['condition']]
            assert spec['lambda'] == .03 and spec['alpha'] != 1.
            assert mm[rid]['condition'] == batch['owner_condition']
            assert bool(mm[rid]['query_image_paths']) == batch['has_image']
            assert r['batch_size'] == len(batch['record_ids']) and r['batch_index'] == batch['batch_index']
            fresh[h] = r
    assert len(fresh) == c['new_unique_payloads'] == plan['new_unique_payloads']
    questions = {q['id']: q for q in lines(base / 'inputs/questions.jsonl')}
    memories = {m['id']: m for m in lines(base / 'inputs/memories.jsonl')}
    gold = {g['id']: g for g in lines(base / 'inputs/gold.jsonl')}
    ranked = {q['id']: {x['name']: x for x in q['conditions']} for q in lines(base / 'retrieval_results.jsonl')}
    pairs = {r['id']: r for r in read(base / 'reader/paired_results.json')}
    summary = read(base / 'summary.json')
    assert set(questions) == set(gold) == set(pairs) and len(questions) == 1711
    counts, correct, hits, totals, f1s, origins = [Counter() for _ in range(6)]
    stemmer = PorterStemmer(); images = {}; outputs = {}
    controls = alpha1_checks = same_input_checks = 0
    fields = ['raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit', 'execution_id', 'batch_size', 'batch_elapsed_seconds']
    for rid, m in mm.items():
        r = rr[rid]; q = questions[m['id']]; name = m['condition']; spec = c['condition_specs'][name]
        assert not {'answer', 'gold_ids', 'gold_choice', 'clue'} & set(m)
        text = 'The retrieved memory contents are as follows:\n\n'
        text += '\n\n'.join(f'Memory {i}: {memories[mid]["reader_text"]}' for i, mid in enumerate(m['top5_memory_ids'], 1))
        text += '\n\nPlease answer the question about the conversation using the retrieved memories. Return only the answer content, concisely but completely. For dates or times, provide a specific date or time whenever possible.\n\nQuestion:\n' + q['question']
        if instruction := c['task_instructions'].get(q['point'].upper()): text += '\n\n' + instruction
        assert text == m['reader_text'] and q['query_image_paths'] == m['query_image_paths']
        im = []
        for path in q['query_image_paths']:
            if path not in images: images[path] = sha(path)
            im.append({'path': path, 'sha256': images[path]})
        assert im == m['query_images']
        h = digest({'system': c['system_prompt'], 'user_text': text, 'query_images': im})
        assert h == m['payload_sha256'] == r['payload_sha256']
        assert m['top5_memory_ids'] == r['top5_memory_ids'] == ranked[m['id']][name]['top5_memory_ids']
        assert r['run_id'] == p['run_id'] and r['experiment_signature'] == p['experiment_signature']
        a = assignments[rid]; assert a['payload_sha256'] == h
        assert r['historical_response_reused'] == (h in cache)
        if h in cache:
            root = cache[h]['root']
            assert r['historical_root'] == a['source_root'] == root
            origin = source_rows[root][r['historical_record_id']]
            label = 'original' if root == str(roots[0]) else 'lambda_ablation'
            assert r['cache_source'] == a['origin'] == label
            assert r['historical_run_id'] == source_protocols[root]['run_id']
            assert r['historical_record_id'] == a['source_record_id']
            assert origin['payload_sha256'] == h and not r['fresh_inference_executed_for_this_record']
            assert r['same_run_shared_inference_record_id'] is None
        else:
            origin = fresh[h]; label = 'new'
            assert r['cache_source'] is None and r['historical_root'] is None and r['historical_record_id'] is None
            assert a['origin'] == 'new' and a['source_root'] == str(base) and a['source_record_id'] == origin['record_id']
            assert r['fresh_inference_executed_for_this_record'] == (rid == origin['record_id'])
            assert r['same_run_shared_inference_record_id'] == (None if rid == origin['record_id'] else origin['record_id'])
            assert r['inference_batch_index'] == origin['batch_index']
        for field in fields: assert r[field] == origin[field], (rid, field)
        raw_rid = m['id'] + '::' + spec['method']
        same = h == original_m[raw_rid]['payload_sha256']
        assert r['same_as_own_unweighted_input'] == a['same_as_own_unweighted_input'] == same
        if same:
            assert r['raw_response'] == original_r[raw_rid]['raw_response']; same_input_checks += 1
        if spec['lambda'] is None:
            assert same and label == 'original'; controls += 1
        elif spec['alpha'] == 1.:
            assert h in cache and r['raw_response'] == source_rows[str(roots[1])][rid]['raw_response']; alpha1_checks += 1
        if h in outputs: assert outputs[h] == r['raw_response']
        outputs[h] = r['raw_response']
        pred, target = normalize(r['raw_response']), normalize(gold[m['id']]['answer'])
        pt, gt = pred.split(), target.split()
        common = sum((Counter(stemmer.stem(t) for t in pt) & Counter(stemmer.stem(t) for t in gt)).values())
        f1 = 2 * common / (len(pt) + len(gt)) if common else 0.
        em = int(pred == target)
        hit = len(set(gold[m['id']]['gold_ids']) & set(m['top5_memory_ids'])); total = len(set(gold[m['id']]['gold_ids']))
        score = pairs[m['id']][name]
        assert score['em'] == em and abs(score['f1'] - f1) < 1e-15
        assert score['gold_hits'] == hit and score['gold_clues'] == total
        counts[name] += 1; correct[name] += em; hits[name] += hit; totals[name] += total; f1s[name] += f1; origins[name, label] += 1
    assert controls == alpha1_checks == 3422
    assert correct[METHODS[0]] == 778 and correct[METHODS[1]] == 798
    for name, metrics in summary['overall'].items():
        assert counts[name] == metrics['questions'] == 1711 and totals[name] == metrics['gold_links'] == 4109
        assert correct[name] == metrics['correct'] and hits[name] == metrics['gold_hits']
        assert metrics['em'] == correct[name] / 1711 and metrics['recall_at_5_micro'] == hits[name] / 4109
        assert abs(metrics['f1'] - f1s[name] / 1711) < 1e-14
        for label, field in [('original', 'original_answer_reused'), ('lambda_ablation', 'lambda_answer_reused'), ('new', 'new_answer_records')]:
            assert origins[name, label] == metrics[field]
    result = {'passed': True, 'retrieval': retrieval, 'logical_records': len(responses),
        'control_records_identical': controls, 'alpha1_records_identical': alpha1_checks,
        'control_model_calls': 0, 'alpha1_model_calls': 0, 'fresh_unique_payloads': len(fresh),
        'original_cached_records': sum(v for (n, label), v in origins.items() if label == 'original'),
        'lambda_cached_records': sum(v for (n, label), v in origins.items() if label == 'lambda_ablation'),
        'same_run_shared_records': sum(r['same_run_shared_inference_record_id'] is not None for r in responses),
        'same_as_own_unweighted_input_answer_checks': same_input_checks,
        'all_complete_payloads_reconstructed': True, 'all_identical_payload_answers_identical': True,
        'independent_em_f1_recall_verified': True, 'new_inputs_only_in_generation': True,
        'empty_responses': sum(not r['raw_response'] for r in responses),
        'token_limit_records': sum(r['hit_token_limit'] for r in responses), 'audit_script_sha256': sha(__file__)}
    check(c['source_sha256'])
    (base / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [f for f in base.rglob('*') if f.is_file() and f.name != 'completion_manifest.json'
             and f.suffix in ['.json', '.jsonl', '.csv', '.html', '.png', '.pdf', '.svg']]
    completion = {'status': 'complete', 'conditions': 16, 'questions_per_condition': 1711,
        'independent_audit_passed': True, 'control_model_calls': 0, 'alpha1_model_calls': 0,
        'artifact_sha256': {str(f): sha(f) for f in files}}
    (base / 'completion_manifest.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    audit(parser.parse_args().output.resolve())
