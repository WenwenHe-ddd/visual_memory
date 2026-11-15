"""Matched full-MemEye Qwen evaluation of the independent positive_variance path.

Runs the actual pipeline CLI for both memories, then four fresh reader conditions
(two positive-variance treatments and their unweighted controls). Frozen Qwen
captions, memory graphs, candidate statements, prompts and model weights are
verified against the original 184/371 and 195/371 experiment. Only identical
full payloads within this new run share an answer; historical answers are not
reused. Existing signed experimental results are never rewritten.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import html
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE':'1', 'TOKENIZERS_PARALLELISM':'false',
                   'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4'}.items():
    os.environ.setdefault(key, value)

import numpy as np
from scripts import visual_memory_pipeline as pipeline
from scripts.analysis.evaluate_adaptive_changed_qa import read, lines, dump, sha, digest, reader_text
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.memory.result_method_annotations import online_method_annotations
from scripts.memory.ridge_whitening import standardize
from scripts.evaluation.paired_single_gpu_reader import run as run_reader

DEFAULT_OUTPUT = ROOT / 'results/memeye_qwen_positive_variance_l0p03_a1_20261107_v2'
ORIGINAL = ROOT / 'data/independent_visual_inputs'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021'
SEARCH = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024/protocol.json'
MASTER_REF = ROOT / 'reports/master_answer_query_zscore_20261023/retrieval_results.jsonl'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
METHODS = ('baseline_question_only', 'master_answer_query')
SPECS = {
    'baseline_question_only': {'memory':'natural', 'retrieval':'full_scan', 'weighting':'none'},
    'baseline_question_only+positive_variance': {'memory':'natural', 'retrieval':'full_scan', 'weighting':'positive_variance'},
    'master_answer_query': {'memory':'caption_visual_master', 'retrieval':'multi_path', 'weighting':'none'},
    'master_answer_query+positive_variance': {'memory':'caption_visual_master', 'retrieval':'multi_path', 'weighting':'positive_variance'},
}
CODE = [Path(__file__), ROOT/'scripts/visual_memory_pipeline.py',
        *[ROOT/'scripts/memory'/name for name in ('online_retrieval.py','online_positive_variance.py',
           'online_zscore.py','ridge_whitening.py','result_method_annotations.py','caption_visual_master.py','memory_store.py')],
        ROOT/'scripts/evaluation/paired_single_gpu_reader.py',
        ROOT/'scripts/analysis/evaluate_adaptive_changed_qa.py',
        ROOT/'scripts/analysis/evaluate_all_master_answer_reader.py',
        ROOT/'scripts/tests/test_positive_variance.py']


def write_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))


def check_hashes(hashes):
    for path, expected in hashes.items():
        assert sha(path) == expected, f'Changed source: {path}'


def retrieve(base, device, ridge, alpha):
    target = base/'retrieval_protocol.json'
    if target.exists():
        p = read(target)
        assert (p['lambda'],p['alpha']) == (ridge,alpha)
        check_hashes(p['source_sha256']); check_hashes(p['output_sha256'])
        return p
    pinned = read(REFERENCE/'protocol.json')
    search = read(SEARCH)
    stores = Path(search['memory_path'])
    sources = {str(p):sha(p) for p in [*CODE, SEARCH, REFERENCE/'protocol.json']}
    # Pin immutable DATA against old protocols; do not require obsolete runtime
    # hashes for files deliberately extended to implement the new component.
    for p in [ORIGINAL/'qa_manifest.jsonl', ORIGINAL/'retrieval_memories.jsonl',
              REFERENCE/'retrieval_results.jsonl', REFERENCE/'candidates.jsonl', MASTER_REF]:
        expected = search['source_sha256'][str(p)]
        assert sha(p) == expected, str(p)
        sources[str(p)] = expected
    original_qs = lines(ORIGINAL/'qa_manifest.jsonl')
    originals = {q['id']:q for q in original_qs}
    old = {r['question_id']:r for r in lines(REFERENCE/'retrieval_results.jsonl')}
    old_master = {r['question_id']:r for r in lines(MASTER_REF)}
    source_texts = {r['id']:r['reader_text'] for r in lines(ORIGINAL/'retrieval_memories.jsonl')}
    results, annotations, raw_checks, group_checks = {}, {}, 0, 0
    max_raw_error = 0.
    base.mkdir(parents=True, exist_ok=True)
    for domain in sorted(search['master_counts']):
        src = INPUTS/domain/'inputs.json'
        assert sha(src) == pinned['source_sha256'][str(src)]
        sources[str(src)] = sha(src)
        for method, memory_type in zip(METHODS, ['natural','caption_visual_master']):
            state = stores/domain/memory_type
            manifest = read(state/'memory.json')
            for path in [state/'memory.json', *[state/ref['path'] for ref in manifest['files'].values()]]:
                assert sha(path) == search['source_sha256'][str(path)], str(path)
                sources[str(path)] = sha(path)
            out = base/'domains'/domain/method
            report_path = out/'retrieval_results.json'
            if report_path.exists():
                report = read(report_path); check_hashes(report['source_sha256'])
            else:
                pipeline.main(['online','--memory-path',str(state),'--question-source',str(src),
                               '--retrieval',SPECS[method]['retrieval'],'--online-weighting','positive_variance',
                               '--whitening-lambda',str(ridge),'--reweight-alpha',str(alpha),
                               '--top-k','5','--device',device,'--stage','prepare','--output',str(out)])
                report = read(report_path)
            assert report['online_weighting'] == 'positive_variance' and report['svd_term_sign'] == '+'
            assert (report['ridge_lambda'],report['alpha']) == (ridge,alpha)
            assert report['memory_unchanged'] and not report['reader_called'] and not report['vectors_persisted']
            sources.update(report['source_sha256'])
            sources[str(report_path)] = sha(report_path)
            for row in report['questions']:
                qid = row['question_id']; original = originals[qid]; prior = old[qid]
                assert row['retrieval_query'] == prior['baseline_retrieval_query']
                for key in ['question','options','query_image_paths']:
                    assert row[key] == original[key], (qid,key)
                actual_raw = row['conditions'][0]
                reference_raw = next(c for c in prior['conditions'] if c['name'] == method)
                assert actual_raw['top5_memory_ids'] == reference_raw['top5_memory_ids'], (qid,method)
                error = max(abs(a['score']-b) for a,b in zip(actual_raw['top5'],reference_raw['top5_scores']))
                assert error < 2e-6, (qid,error)
                max_raw_error = max(max_raw_error,error); raw_checks += 1
                leaves = row['leaf_scores']
                for g in row['groups']:
                    group_leaves = [leaf for leaf in leaves if leaf['group_id'] == g['group_id']]
                    assert len(group_leaves) == g['size']
                    raw = np.array([leaf['raw_score'] for leaf in group_leaves])
                    enhanced = np.array([leaf['positive_variance_score'] for leaf in group_leaves])
                    z, status = standardize(enhanced)
                    np.testing.assert_allclose(z,[leaf['positive_variance_z'] for leaf in group_leaves],atol=1e-12)
                    expected = np.zeros(len(raw))
                    if len(raw)>1 and g['raw_status'] != 'zero_variance' and status != 'zero_variance':
                        expected = alpha*raw.std()*np.tanh(z)
                    np.testing.assert_allclose(expected,[leaf['correction'] for leaf in group_leaves],atol=1e-12)
                    np.testing.assert_allclose(raw+expected,[leaf['weighted_score'] for leaf in group_leaves],atol=1e-12)
                    assert all('white_score' not in leaf for leaf in group_leaves)
                    if method == METHODS[1]:
                        previous = next(g0 for g0 in old_master[qid]['groups'] if g0['group_id'] == g['group_id'])
                        assert all(g[key] == previous[key] for key in ['size','query','choice','selected_statement'])
                    else:
                        assert g['query'] == row['retrieval_query'] and g['size'] == report['images']
                    group_checks += 1
                record = results.setdefault(qid,{'question_id':qid,'domain':domain,'conditions':{}})
                for condition, retrieved in zip([method,method+'+positive_variance'],row['conditions']):
                    for memory in retrieved['top5']:
                        assert memory['reader_text'] == source_texts[memory['memory_id']]
                    record['conditions'][condition] = {
                        'top5_memory_ids':retrieved['top5_memory_ids'],
                        'top5_scores':[m['score'] for m in retrieved['top5']]}
            for condition in [method,method+'+positive_variance']:
                description = copy.deepcopy(report)
                if condition == method:
                    description.update(online_weighting='none',formula='raw',ridge_lambda=None,alpha=None)
                annotations.setdefault(condition,{})[domain] = online_method_annotations(description,manifest['encoding'])
            del report
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        print(json.dumps({'stage':'domain_complete','domain':domain,'raw_top5_matches':raw_checks}),flush=True)
    assert len(results) == 371 and raw_checks == 742
    assert all(set(r['conditions']) == set(SPECS) for r in results.values())
    write_lines(base/'retrieval_results.jsonl',[results[q['id']] for q in original_qs])
    before_path = base/'implementation_before_sha256.json'
    if before_path.exists():
        before = read(before_path)
        for name in ['online_zscore.py','ridge_whitening.py']:
            path = ROOT/'scripts/memory'/name
            assert sha(path) == before[str(path)], f'Existing z-score implementation modified: {path}'
    check_hashes(sources)
    p = {'scheme':'positive_variance','lambda':ridge,'alpha':alpha,'condition_specs':SPECS,
         'questions':371,'domains':8,'memory_leaves':438,'masters':88,
         'svd_term_sign':'+','intermediate_score':'raw + sum_j v_j/(v_j+lambda)*(q dot u_j)*(e dot u_j)',
         'final_score':'raw + alpha * std_pop(raw in group) * tanh(z_positive_variance)',
         'source_sha256':sources,'output_sha256':{str(base/'retrieval_results.jsonl'):sha(base/'retrieval_results.jsonl')},
         'method_annotations':annotations,'raw_top5_matches_historical':raw_checks,'max_raw_score_error':max_raw_error,
         'group_queries_and_arithmetic_verified':group_checks,'caption_regenerated':False,'memory_modified':False,
         'existing_zscore_implementation_unchanged':True,'vectors_persisted':False,'device':device}
    dump(target,p)
    print(json.dumps({'stage':'retrieval_complete','raw_checks':raw_checks,'groups_checked':group_checks}),flush=True)
    return p


def prepare_reader(base, device):
    out = base/'reader'; target = out/'protocol.json'
    if target.exists():
        p = read(target); verify(p)
        return p
    method = read(base/'retrieval_protocol.json')
    check_hashes(method['source_sha256']); check_hashes(method['output_sha256'])
    previous = read(REFERENCE/'reader/protocol.json')
    signed = dict(previous); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    settings = previous['reader_settings']
    assert settings['seed'] == 42 and settings['do_sample'] is False and settings['max_new_tokens'] == 16
    original = {q['id']:q for q in lines(ORIGINAL/'qa_manifest.jsonl')}
    memories = {m['id']:m['reader_text'] for m in lines(ORIGINAL/'retrieval_memories.jsonl')}
    old_manifest = {r['record_id']:r for r in lines(REFERENCE/'reader/qa_manifest.jsonl')}
    rows, image_records, payload_matches = [], {}, 0
    for qi, item in enumerate(lines(base/'retrieval_results.jsonl')):
        q = original[item['question_id']]; batch = []
        for path in q['query_image_paths']:
            image_records[path] = {'path':path,'sha256':sha(path)}
        images = [image_records[path] for path in q['query_image_paths']]
        for name in SPECS:
            retrieved = item['conditions'][name]; ids = retrieved['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            assert all(mid.split('/',1)[0] == q['task'] for mid in ids)
            text = reader_text(q,ids,memories)
            payload = {'system':settings['reader_system_prompt'],'user_text':text,'query_images':images}
            record_id = q['id']+'::'+name
            if name in METHODS:
                assert digest(payload) == old_manifest[record_id]['payload_sha256']
                payload_matches += 1
            batch.append({**{key:q[key] for key in ['id','task','question','options','gold_choice','gold_ids','query_image_paths']},
                          'record_id':record_id,'condition':name,'top_memory_ids':ids,'top_query_scores':retrieved['top5_scores'],
                          'reader_text':text,'query_images':images,'payload_sha256':digest(payload)})
        offset = qi % len(SPECS)
        rows.extend(batch[offset:]+batch[:offset])
    image_questions = sum(bool(q['query_image_paths']) for q in original.values())
    assert len(rows) == 1484 and payload_matches == 742 and image_questions == 8
    assert set(image_records) == {path for q in original.values() for path in q['query_image_paths']}
    out.mkdir(parents=True,exist_ok=True)
    write_lines(out/'qa_manifest.jsonl',rows)
    check_hashes(previous['model_source_sha256'])
    weights = copy.deepcopy(previous['weight_files'])
    for w in weights:
        assert sha(w['path']) == w['sha256'], w['path']
        stat = Path(w['path']).stat(); w.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    sources = {**method['source_sha256'], **method['output_sha256']}
    for path in [base/'retrieval_protocol.json',out/'qa_manifest.jsonl',REFERENCE/'reader/protocol.json',
                 REFERENCE/'reader/qa_manifest.jsonl',REFERENCE/'reader/responses.jsonl',REFERENCE/'reader/summary.json']:
        sources[str(path)] = sha(path)
    p = {'design':'Four-condition fresh matched Qwen MemEye evaluation; positive SVD term, then positive tanh correction.',
         'experiment_run_id':str(uuid.uuid4()),'condition_names':list(SPECS),'condition_specs':SPECS,
         'lambda':method['lambda'],'alpha':method['alpha'],'questions':371,'logical_conditions':1484,
         'unique_reader_payloads':len({r['payload_sha256'] for r in rows}),
         'reader_settings':settings,'placement':{'gpu_index':int(device.split(':')[1]),'minimum_free_gib':24},
         'attention_implementation':'sdpa','query_images':list(image_records.values()),
         'image_questions':image_questions,'unique_query_images':len(image_records),
         'historical_response_reuse':False,'deduplication':'Identical full reader payloads share one fresh inference within this run only.',
         'historical_reader_placement':previous['placement'],
         'source_sha256':sources,'model_source_sha256':previous['model_source_sha256'],'weight_files':weights,
         'method_annotations':method['method_annotations']}
    p['experiment_signature'] = digest(p)
    dump(target,p); verify(p)
    dump(out/'validation_input.json',{'passed':True,'records':len(rows),'unique_payloads':p['unique_reader_payloads'],
                                     'historical_baseline_payload_matches':payload_matches,'model_weight_hashes_verified':True})
    print(json.dumps({'stage':'reader_prepared','records':len(rows),'unique_payloads':p['unique_reader_payloads']}),flush=True)
    return p


def analyze(out):
    out = Path(out); base = out.parent
    p = read(out/'protocol.json'); verify(p)
    signed = dict(p); signature = signed.pop('experiment_signature'); assert digest(signed) == signature
    manifest = {r['record_id']:r for r in lines(out/'qa_manifest.jsonl')}
    responses = lines(out/'responses.jsonl'); answers = {r['record_id']:r for r in responses}
    assert len(manifest) == len(responses) == len(answers) == 1484 and set(manifest) == set(answers)
    fresh = {}; shared = 0
    for row in responses:
        source = manifest[row['record_id']]
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',row['raw_response'].strip(),re.I)
        choice = match.group(1).upper() if match else None
        assert row['predicted_choice'] == choice and row['correct'] == (choice == source['gold_choice'])
        assert row['payload_sha256'] == source['payload_sha256'] and row['experiment_signature'] == signature
        assert row['run_id'] == p['experiment_run_id'] and not row['historical_response_reuse']
        if row['fresh_inference_executed_for_this_record']:
            assert row['payload_sha256'] not in fresh
            fresh[row['payload_sha256']] = row
        else:
            origin = answers[row['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            assert all(row[key] == origin[key] for key in ['payload_sha256','raw_response','generated_token_ids','execution_id'])
            shared += 1
    assert len(fresh) == p['unique_reader_payloads'] and len(fresh)+shared == 1484
    qs = lines(ORIGINAL/'qa_manifest.jsonl'); overall = {}; per_domain = {}
    annotations = copy.deepcopy(p['method_annotations'])
    for name in SPECS:
        for domain in [None,*sorted({q['task'] for q in qs})]:
            selected = [q for q in qs if domain is None or q['task'] == domain]
            chosen = [manifest[q['id']+'::'+name] for q in selected]
            hits = [len(set(r['top_memory_ids']) & set(r['gold_ids'])) for r in chosen]
            totals = [len(set(r['gold_ids'])) for r in chosen]
            correct = sum(answers[r['record_id']]['correct'] for r in chosen)
            metrics = {'correct':correct,'questions':len(chosen),'accuracy':correct/len(chosen),
                       'gold_hits_top5':sum(hits),'gold_references':sum(totals),
                       'recall_at_5_micro':sum(hits)/sum(totals),
                       'recall_at_5_macro':float(np.mean([h/t for h,t in zip(hits,totals) if t])),
                       'questions_with_any_gold':sum(h>0 for h in hits),
                       'hit_at_5':sum(h>0 for h in hits)/len(chosen)}
            if domain is None:
                assert sum(totals) == 1041
                overall[name] = metrics
            else:
                per_domain.setdefault(domain,{})[name] = metrics
                annotations[name][domain]['5. 回答方式'] = {
                    '说明':'本轮新调用本地 Qwen3-VL-8B；原始问题/选项、Top-5 原始对话和 caption；带图问题传原图。',
                    **p['reader_settings'],'reader_called':True,'correct':correct,'questions':len(chosen),
                    'measured_qa_accuracy':correct/len(chosen),'candidate_statements_in_reader':False}
    old_answers = {r['record_id']:r for r in lines(REFERENCE/'reader/responses.jsonl')}
    comparisons = {}; paired_rows = []
    for method in METHODS:
        weighted = method+'+positive_variance'
        wins=[]; losses=[]; changed=[]; drift=[]; same=[]
        for q in qs:
            a,b = (answers[q['id']+'::'+name] for name in [method,weighted])
            if not a['correct'] and b['correct']: wins.append(q['id'])
            if a['correct'] and not b['correct']: losses.append(q['id'])
            if a['payload_sha256'] != b['payload_sha256']: changed.append(q['id'])
            else:
                same.append(q['id']); assert a['raw_response'] == b['raw_response']
            if a['predicted_choice'] != old_answers[a['record_id']]['predicted_choice']: drift.append(q['id'])
        comparisons[method] = {'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':len(wins)-len(losses),
                               'changed_payload_questions':changed,'unchanged_payload_questions':same,
                               'fresh_control_answer_changes_vs_historical':drift}
    for q in qs:
        row = {'question_id':q['id'],'domain':q['task'],'gold_choice':q['gold_choice']}
        for name in SPECS:
            a=answers[q['id']+'::'+name]; m=manifest[a['record_id']]
            row[name+'_answer']=a['predicted_choice']; row[name+'_correct']=a['correct']
            row[name+'_gold_hits']=len(set(m['top_memory_ids']) & set(m['gold_ids']))
            row[name+'_top5']=' | '.join(m['top_memory_ids'])
        paired_rows.append(row)
    summary = {'status':'complete','scheme':'positive_variance','lambda':p['lambda'],'alpha':p['alpha'],
               'overall':overall,'per_domain':per_domain,'paired_comparisons':comparisons,
               'historical_controls':{'baseline_question_only':184,'master_answer_query':195},
               'fresh_inferences':len(fresh),'shared_records':shared,'logical_records':1484,
               'format_errors':sum(r['status'] != 'ok' for r in responses),
               'token_limit_hits':sum(r['hit_token_limit'] for r in responses),
               'empty_responses':sum(not r['raw_response'].strip() for r in responses),
               'historical_answers_reused':0,'method_annotations':annotations,
               'recall_definition':'micro = total target memories in Top-5 / 1041 gold references; macro = mean per-question recall',
               'reader_settings':p['reader_settings']}
    dump(base/'summary.json',summary)
    for filename,records in [('metrics.csv',[{'condition':name,**values} for name,values in overall.items()]),
                             ('paired_results.csv',paired_rows)]:
        with (base/filename).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemEye Positive Variance</title>',
          '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
          '<h1>Qwen MemEye：Positive Variance</h1>',
          f'<p>λ={p["lambda"]}，α={p["alpha"]}；371 题，438 条原始 Qwen 图片记忆，88 个 master。SVD 项改为加号，后续保持 +tanh。原 caption、图结构、master 候选陈述和 reader 设置固定。</p>',
          '<p>h = raw + Σ[v/(v+λ)](q·u)(e·u)；s = raw + α·std(raw)·tanh(z(h))。</p>',
          '<p>下表未赋权对照在本轮重新回答；历史对照分别是 184/371 与 195/371。同一完整请求仅在本轮共享回答。</p>',
          '<table><tr><th>条件</th><th>答对 / 371</th><th>准确率</th><th>Recall@5（micro）</th><th>Hit@5</th></tr>']
    for name,m in overall.items():
        body.append(f'<tr><td>{html.escape(name)}</td><td>{m["correct"]}/371</td><td>{m["accuracy"]:.2%}</td><td>{m["recall_at_5_micro"]:.2%}</td><td>{m["hit_at_5"]:.2%}</td></tr>')
    body+=['</table><h2>配对变化</h2><pre>'+html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))+'</pre>',
           '<p><a href="summary.json">完整结果和五项方法标注</a> · <a href="metrics.csv">指标 CSV</a> · <a href="paired_results.csv">逐题结果</a></p></html>']
    (base/'report.html').write_text('\n'.join(body))
    check_hashes(read(base/'retrieval_protocol.json')['source_sha256'])
    audit={'passed':True,'logical_records':1484,'questions_per_condition':371,'fresh_inferences':len(fresh),
           'shared_records':shared,'input_hashes_unchanged':True,'raw_top5_matches_historical':742,
           'reader_payloads_reconstructed_from_original_text':True,'response_ids_and_signatures_verified':True,
           'same_payload_responses_identical':True,'no_historical_answer_reuse':True,
           'format_errors':summary['format_errors'],'token_limit_hits':summary['token_limit_hits'],
           'empty_responses':summary['empty_responses']}
    dump(base/'final_audit.json',audit)
    artifacts=[base/n for n in ['retrieval_protocol.json','retrieval_results.jsonl','summary.json','metrics.csv','paired_results.csv','report.html','final_audit.json']]
    artifacts += [out/n for n in ['protocol.json','qa_manifest.jsonl','responses.jsonl','validation_input.json','generation_environment.json']]
    dump(base/'completion_manifest.json',{'status':'complete','experiment_signature':signature,
                                         'artifact_sha256':{str(path):sha(path) for path in artifacts}})
    print(json.dumps({'stage':'complete','overall':overall,'audit':audit}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--ridge',type=float,default=.03)
    parser.add_argument('--alpha',type=float,default=1.)
    parser.add_argument('--stage',choices=['retrieve','prepare','run','analyze','all'],default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();base=args.output.resolve()
    if args.stage in ['retrieve','all']: retrieve(base,args.device,args.ridge,args.alpha)
    if args.stage in ['prepare','all']: prepare_reader(base,args.device)
    if args.stage in ['run','all']: run_reader(base/'reader',analyze,args.resume)
    if args.stage == 'analyze': analyze(base/'reader')


if __name__ == '__main__':
    main()
