"""Fresh local-Qwen joint search and preregistered shared-parameter selection."""
from __future__ import annotations
import argparse
import copy
import csv
import html
import json
import math
from pathlib import Path
import re
import sys
import uuid
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis.retrieve_memeye_qwen_zscore_search import (
    BASE, ORIGINAL, REFERENCE, MASTER_REF, FULL_REF, METHODS, LAMBDAS, ALPHAS, SPECS, SELECTION,
    condition, read, lines, dump, sha, digest, write_lines, verify_hashes)
from scripts.analysis.evaluate_adaptive_changed_qa import reader_text
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.evaluation.paired_single_gpu_reader import run as run_reader


def prepare(base=BASE, device='cuda:1'):
    base = Path(base).resolve(); out = base / 'reader'
    if (out / 'protocol.json').exists():
        p = read(out / 'protocol.json'); verify(p)
        assert p['condition_specs'] == SPECS and p['selection_rule'] == SELECTION
        return p
    method = read(base / 'protocol.json')
    verify_hashes(method['source_sha256']); verify_hashes(method['output_sha256'])
    previous = read(MASTER_REF / 'reader/protocol.json')
    signed = dict(previous); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    settings = previous['reader_settings']
    assert settings['do_sample'] is False and settings['seed'] == 42 and settings['max_new_tokens'] == 16
    original = {q['id']: q for q in lines(ORIGINAL / 'qa_manifest.jsonl')}
    memories = {m['id']: m['reader_text'] for m in lines(ORIGINAL / 'retrieval_memories.jsonl')}
    old = {r['record_id']: r for r in lines(REFERENCE / 'reader/qa_manifest.jsonl')}
    manifest = []; images_by_path = {}; baseline_matches = 0
    for qi, row in enumerate(lines(base / 'retrieval_results.jsonl')):
        q = original[row['question_id']]; batch = []
        assert set(row['conditions']) == set(SPECS)
        for p in q['query_image_paths']:
            images_by_path.setdefault(p, {'path': p, 'sha256': sha(p)})
        images = [images_by_path[p] for p in q['query_image_paths']]
        for name in SPECS:
            result = row['conditions'][name]; ids = result['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5 and all(mid.split('/', 1)[0] == q['task'] for mid in ids)
            text = reader_text(q, ids, memories)
            payload = {'system': settings['reader_system_prompt'], 'user_text': text, 'query_images': images}
            if name in METHODS:
                prior = old[q['id'] + '::' + name]
                assert prior['payload_sha256'] == digest(payload) and prior['top_memory_ids'] == ids
                baseline_matches += 1
            batch.append({**{k: q[k] for k in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')},
                          'record_id': q['id'] + '::' + name, 'condition': name, 'top_memory_ids': ids,
                          'top_query_scores': result['top5_scores'], 'reader_text': text, 'query_images': images,
                          'payload_sha256': digest(payload)})
        offset = qi % len(SPECS)
        manifest.extend(batch[offset:] + batch[:offset])
    assert len(manifest) == 371 * 52 and baseline_matches == 742
    out.mkdir(parents=True, exist_ok=True)
    write_lines(out / 'qa_manifest.jsonl', manifest)
    verify_hashes(previous['model_source_sha256'])
    for weight in previous['weight_files']:
        assert sha(weight['path']) == weight['sha256']
    sources = [Path(__file__), Path(__file__).with_name('audit_memeye_qwen_zscore_search.py'),
               ROOT / 'scripts/tests/test_memeye_qwen_zscore_search.py', ROOT / 'scripts/evaluation/paired_single_gpu_reader.py',
               ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py', ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py',
               base / 'protocol.json', base / 'retrieval_results.jsonl', out / 'qa_manifest.jsonl',
               MASTER_REF / 'reader/protocol.json', MASTER_REF / 'reader/summary.json', FULL_REF / 'reader/summary.json',
               REFERENCE / 'reader/qa_manifest.jsonl', REFERENCE / 'reader/responses.jsonl']
    full_annotations = read(FULL_REF / 'reader/summary.json')['method_annotations']
    master_annotations = read(MASTER_REF / 'reader/summary.json')['method_annotations']
    annotations = {}
    for name, spec in SPECS.items():
        weighted = spec['alpha'] > 0
        template = (full_annotations['lambda_0p01' if weighted else METHODS[0]] if spec['method'] == METHODS[0]
                    else master_annotations['master_answer_query_zscore' if weighted else METHODS[1]])
        m = copy.deepcopy(template); m['scope'] = name
        m['3. 检索方式'].update({'lambda': spec['lambda'], 'alpha': spec['alpha'], 'online_weighting': 'z_score' if weighted else 'none'})
        m['5. 回答方式'] = {**settings, 'reader_called': False, 'correct': None,
                            'note': 'Fresh local Qwen; same full payload shares one answer within this 52-condition run.'}
        annotations[name] = m
    p = {'design': '25 shared lambda/alpha pairs and two fresh unweighted controls; same frozen Qwen caption and graph.',
         'experiment_run_id': str(uuid.uuid4()), 'condition_names': list(SPECS), 'condition_specs': SPECS,
         'selection_rule': SELECTION, 'questions': 371, 'logical_conditions': len(manifest),
         'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}), 'reader_settings': settings,
         'placement': {'gpu_index': int(device.split(':')[1]), 'minimum_free_gib': 24}, 'attention_implementation': 'sdpa',
         'query_images': list(images_by_path.values()), 'historical_response_reuse': False,
         'deduplication': 'Identical full reader payloads share one fresh inference within this run only.',
         'source_sha256': {**method['source_sha256'], **{str(p): sha(p) for p in sources}},
         'model_source_sha256': previous['model_source_sha256'], 'weight_files': previous['weight_files'],
         'method_annotations': annotations}
    p['experiment_signature'] = digest(p); dump(out / 'protocol.json', p)
    verify(p)
    dump(out / 'validation_input.json', {'passed': True, 'records': len(manifest), 'unique_payloads': p['unique_reader_payloads'],
                                       'historical_baseline_payload_matches': baseline_matches, 'model_weight_hashes_verified': True})
    print(json.dumps({'stage': 'prepared', 'records': len(manifest), 'unique_payloads': p['unique_reader_payloads']}), flush=True)
    return p


def paired(answers, before, after, ids):
    wins = sum(not answers[q, before]['correct'] and answers[q, after]['correct'] for q in ids)
    losses = sum(answers[q, before]['correct'] and not answers[q, after]['correct'] for q in ids)
    n = wins + losses
    return {'wrong_to_right': wins, 'right_to_wrong': losses, 'net_correct': wins - losses,
            'mcnemar_exact_p': min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.}


def choose_shared(rows):
    eligible = [r for r in rows if r['passes_all_three']]
    return sorted(eligible, key=lambda r: (-r['master_correct'], -r['baseline_correct'], r['alpha'], r['lambda']))[0] if eligible else None


def analyze(out):
    out = Path(out).resolve(); base = out.parent
    p = read(out / 'protocol.json'); verify(p)
    signed = dict(p); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    manifest = {r['record_id']: r for r in lines(out / 'qa_manifest.jsonl')}
    responses = lines(out / 'responses.jsonl'); by_id = {r['record_id']: r for r in responses}
    assert len(by_id) == len(responses) == len(manifest) == 19292 and set(by_id) == set(manifest)
    answers = {}; fresh = {}; shared = 0
    for r in responses:
        source = manifest[r['record_id']]
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', r['raw_response'].strip(), re.I)
        choice = match.group(1).upper() if match else None
        assert r['predicted_choice'] == choice and r['correct'] == (choice == source['gold_choice'])
        assert r['payload_sha256'] == source['payload_sha256'] and r['experiment_signature'] == signature
        assert r['run_id'] == p['experiment_run_id'] and not r['historical_response_reuse']
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh; fresh[r['payload_sha256']] = r
        else:
            origin = by_id[r['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            for k in ('payload_sha256', 'raw_response', 'generated_token_ids', 'execution_id'):
                assert r[k] == origin[k]
            shared += 1
        answers[source['id'], source['condition']] = r
    assert len(fresh) == p['unique_reader_payloads'] and len(fresh) + shared == 19292
    qs = lines(ORIGINAL / 'qa_manifest.jsonl'); ids = [q['id'] for q in qs]
    overall = {}; annotations = copy.deepcopy(p['method_annotations'])
    for name in SPECS:
        chosen = [manifest[q + '::' + name] for q in ids]
        hits = [len(set(r['top_memory_ids']) & set(r['gold_ids'])) for r in chosen]
        correct = sum(answers[q, name]['correct'] for q in ids)
        overall[name] = {'correct': correct, 'questions': 371, 'accuracy': correct / 371,
                         'gold_hits_top5': sum(hits), 'gold_references': 1041, 'questions_with_any_gold': sum(x > 0 for x in hits)}
        annotations[name]['5. 回答方式'].update(reader_called=True, correct=correct, questions=371, measured_qa_accuracy=correct / 371)
    rows = []
    for ridge in LAMBDAS:
        for alpha in ALPHAS:
            b, m = (condition(method, ridge, alpha) for method in METHODS)
            bc, mc = overall[b]['correct'], overall[m]['correct']
            b0, m0 = (overall[name]['correct'] for name in METHODS)
            rows.append({'lambda': ridge, 'alpha': alpha, 'baseline_correct': bc, 'master_correct': mc,
                         'baseline_accuracy': bc / 371, 'master_accuracy': mc / 371,
                         'baseline_gain': bc - b0, 'master_gain': mc - m0, 'master_minus_baseline': mc - bc,
                         'passes_all_three': bc > b0 and mc > m0 and mc > bc,
                         'baseline_gold_hits': overall[b]['gold_hits_top5'], 'master_gold_hits': overall[m]['gold_hits_top5'],
                         'baseline_comparison': paired(answers, METHODS[0], b, ids),
                         'master_comparison': paired(answers, METHODS[1], m, ids), 'cross_method_comparison': paired(answers, b, m, ids)})
    selected = choose_shared(rows)
    old = {r['record_id']: r for r in lines(REFERENCE / 'reader/responses.jsonl')}
    disagreements = {n: [qid for qid in ids if answers[qid, n]['predicted_choice'] != old[qid + '::' + n]['predicted_choice']] for n in METHODS}
    pairs = []
    for q in qs:
        row = {'question_id': q['id'], 'domain': q['task'], 'gold_choice': q['gold_choice']}
        for n in SPECS:
            row[n + '_answer'] = answers[q['id'], n]['predicted_choice']; row[n + '_correct'] = answers[q['id'], n]['correct']
        pairs.append(row)
    summary = {'status': 'complete', 'overall': overall, 'search_results': rows, 'selected': selected,
               'selection_rule': SELECTION, 'logical_records': 19292, 'fresh_inferences': len(fresh), 'shared_records': shared,
               'baseline_disagreements_vs_historical': disagreements,
               'format_errors': sum(r['status'] != 'ok' for r in responses), 'token_limit_hits': sum(r['hit_token_limit'] for r in responses),
               'method_annotations': annotations, 'historical_answers_reused': 0}
    dump(base / 'summary.json', summary); dump(base / 'paired_results.json', pairs)
    with (base / 'search_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        keys = [k for k in rows[0] if not k.endswith('_comparison')]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: r[k] for k in keys} for r in rows)
    with (base / 'paired_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    selection = {'status': 'selected' if selected else 'no_eligible_pair', 'selection_rule': SELECTION,
                 'selected': selected, 'qwen_controls': {n: overall[n] for n in METHODS},
                 'eligible_pairs': sum(r['passes_all_three'] for r in rows), 'search_pairs': 25,
                 'source_run_id': p['experiment_run_id'], 'source_experiment_signature': signature,
                 'source_sha256': {str(base / n): sha(base / n) for n in ('protocol.json', 'reader/protocol.json', 'reader/responses.jsonl', 'summary.json')},
                 'gpt5_results_used_for_selection': False}
    dump(base / 'selected_parameters.json', selection)
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Qwen 联合参数搜索</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>',
            '<h1>Qwen：25 组共享 λ/α 搜索</h1><p>固定 Qwen caption、438 条记忆、88 个 master；371 题。两种未赋权对照与 50 个赋权条件均在本轮重新回答，相同完整请求共享一次本轮推理。先要求两种赋权分别超过各自未赋权且 master 赋权超过 baseline 赋权，再优先选择 master 准确率最高、baseline 次高、α 更小、λ 更小的组合。GPT-5 不参与选参。</p>',
            '<h2>本轮未赋权对照</h2><pre>' + html.escape(json.dumps(selection['qwen_controls'], indent=2)) + '</pre>',
            '<h2>固定的候选参数</h2><pre>' + html.escape(json.dumps(selected, ensure_ascii=False, indent=2)) + '</pre>',
            '<table><tr><th>λ</th><th>α</th><th>baseline 答对</th><th>master 答对</th><th>baseline 增益</th><th>master 增益</th><th>master−baseline</th><th>全部通过</th></tr>']
    for r in rows:
        body.append('<tr>' + ''.join('<td>' + html.escape(str(r[k])) + '</td>' for k in ('lambda', 'alpha', 'baseline_correct', 'master_correct', 'baseline_gain', 'master_gain', 'master_minus_baseline', 'passes_all_three')) + '</tr>')
    body.append('</table><p>这是同一测试集上的探索性搜索；选定参数在 GPT-5 上的验证属于 backbone 迁移验证，不是独立测试集泛化证明。</p><p><a href="summary.json">完整统计与五项标注</a> · <a href="search_results.csv">搜索表</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a></p></html>')
    (base / 'report.html').write_text('\n'.join(body))
    print(json.dumps({'stage': 'analyzed', 'controls': selection['qwen_controls'], 'selected': selected, 'fresh': len(fresh)}), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, default=BASE)
    p.add_argument('--device', default='cuda:1'); p.add_argument('--stage', choices=['prepare', 'run', 'analyze', 'all'], default='all'); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    from scripts.analysis.audit_memeye_qwen_zscore_search import audit
    if a.stage in ('prepare', 'all'): prepare(a.output, a.device)
    if a.stage in ('prepare', 'run', 'all'): audit(a.output, inputs_only=True)
    if a.stage in ('run', 'all'): run_reader(a.output / 'reader', analyze, a.resume)
    elif a.stage == 'analyze': analyze(a.output / 'reader')
    if a.stage in ('run', 'analyze', 'all'): audit(a.output)


if __name__ == '__main__': main()
