"""Fresh four-condition GPT-5 validation after Qwen-only shared parameter selection."""
from __future__ import annotations
import argparse
import copy
import csv
import html
import json
from pathlib import Path
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.analysis import evaluate_memeye_gpt5_base as previous
from scripts.analysis.retrieve_memeye_gpt5_qwen_selected import DEFAULT_OUTPUT, SELECTION, RAW, CONDITIONS, selected_pair, verify_source
read,lines,sha,digest,dump=api.read,api.lines,api.sha,api.digest,api.dump
indexed,write_lines,verify=api.indexed,api.write_lines,api.verify
SOURCE=previous.BASE
run=previous.run


def prepare(output=DEFAULT_OUTPUT,source=SOURCE,workers=3):
    output,source=Path(output).resolve(),Path(source).resolve();out=output/'reader'
    ret=read(output/'retrieval_protocol.json');ridge,alpha,selection_hashes=selected_pair(ret['selection_path'])
    if (out/'protocol.json').exists():
        p=verify(out);c=p['contract']
        assert (c['lambda'],c['alpha'],c['memory_path'])==(ridge,alpha,str(source))
        return p
    proof=verify_source(source);pc=verify(source/'reader')['contract']
    assert (ret['lambda'],ret['alpha'],ret['memory_path'])==(ridge,alpha,str(source))
    for path,h in {**ret['source_sha256'],**ret['output_sha256']}.items():assert sha(path)==h,path
    qs=indexed(lines(source/'inputs/source_questions.jsonl'),'id');gold=lines(source/'reader/inputs/gold.jsonl')
    memories={m['id']:m['retrieval_text'] for m in lines(source/'offline/memories.jsonl')}
    old=indexed(lines(source/'reader/inputs/requests.jsonl'),'record_id')
    records=[];raw_matches=0
    for i,row in enumerate(lines(output/'retrieval_results.jsonl')):
        q=qs[row['question_id']];conditions=indexed(row['conditions'],'name');assert set(conditions)==set(CONDITIONS)
        batch=[]
        for name in CONDITIONS:
            ids=conditions[name]['top5_memory_ids'];assert len(ids)==len(set(ids))==5
            text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
            text+='\n\n'.join(f'Memory {j}: {memories[mid]}' for j,mid in enumerate(ids,1))
            text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
            payload={'system':pc['reader_system_prompt'],'user_text':text,'query_images':q['query_images']}
            record={'record_id':q['id']+'::'+name,'id':q['id'],'task':q['task'],'condition':name,
                    'reader_text':text,'query_images':q['query_images'],'query_image_paths':[im['path'] for im in q['query_images']],
                    'top_memory_ids':ids,'payload_sha256':digest(payload)}
            if name in RAW:assert record==old[record['record_id']];raw_matches+=1
            batch.append(record)
        offset=i%4;records.extend(batch[offset:]+batch[:offset])
    assert len(records)==1484 and len(gold)==371 and raw_matches==742
    (out/'inputs').mkdir(parents=True,exist_ok=True)
    write_lines(out/'inputs/requests.jsonl',records);write_lines(out/'inputs/gold.jsonl',gold)
    sources={**proof['source_sha256'],**ret['source_sha256'],**ret['output_sha256'],**selection_hashes}
    paths=[Path(__file__),ROOT/'scripts/analysis/audit_memeye_gpt5_qwen_selected.py',ROOT/'scripts/tests/test_memeye_gpt5_qwen_selected.py',
           ROOT/'scripts/analysis/retrieve_memeye_gpt5_qwen_selected.py',ROOT/'scripts/analysis/evaluate_memeye_gpt5_base.py',
           ROOT/'scripts/analysis/audit_memeye_gpt5_base.py',output/'retrieval_protocol.json',out/'inputs/requests.jsonl',out/'inputs/gold.jsonl']
    paths.extend(Path(im['path']) for q in qs.values() for im in q['query_images'])
    sources.update({str(p):sha(p) for p in paths})
    keys=('model','requested_model','expected_returned_model','caption_model','seed','caption_seed','temperature','temperature_wire_policy',
          'reasoning_effort','max_completion_tokens','image_detail','base_url','api_config','api_env_file','reader_system_prompt','timeout_seconds','max_attempts')
    c={k:copy.deepcopy(pc[k]) for k in keys};annotations={}
    for name in CONDITIONS:
        m=copy.deepcopy(pc['method_annotations'][name]);weighted=name.endswith('_zscore')
        m['1. 候选记忆结构'].update(caption_source='同一批已核验 GPT-5 caption',text='原始对话＋已有完整 GPT-5 caption')
        m['1. 候选记忆结构']['weighting_group']=m['1. 候选记忆结构']['weighting_group'].replace('新 master','已有 master')
        m['2. 查询内容']=m['2. 查询内容'].replace('本轮 GPT-5','已有 GPT-5')
        m['3. 检索方式'].update({'lambda':ridge if weighted else None,'alpha':alpha if weighted else None})
        m['4. 编码方式']['CLIP']='沿用 GPT-5 caption 重建所得的 72 个 master 及视觉聚合量；仅重新编码检索文本'
        m['5. 回答方式'].update(same_run_identical_requests='四个条件中完整请求相同则共享本轮新回答；不复用历史回答',correct=None)
        annotations[name]=m
    c.update(questions=371,logical_records=1484,unique_payloads=len({r['payload_sha256'] for r in records}),conditions=list(CONDITIONS),
             memory_path=str(source),source_run_id=read(source/'protocol.json')['run_id'],top_k=5,workers=workers,source_sha256=sources,
             method_annotations=annotations,historical_api_answer_reuse=False,gold_in_api_payload=False,caption_regenerated=False,memory_rebuilt=False,
             selection_path=ret['selection_path'],selection_sha256=sha(ret['selection_path']),qwen_only_parameter_selection=True,
             same_run_controls=True,within_run_sharing='One fresh answer per complete request across all four conditions.',
             interpretation='One frozen shared Qwen-selected pair; fresh GPT-5 controls and treatments. All three strict accuracy comparisons required. No GPT-5-guided reselection.')
    c['lambda']=ridge;c['alpha']=alpha
    manifest=[{'record_id':r['record_id'],'payload_sha256':r['payload_sha256'],
               'request_sha256':digest(previous.request_body({'contract':c},r)),
               'query_image_count':len(r['query_images']),'requested_model':c['model']} for r in records]
    write_lines(out/'api_request_manifest.jsonl',manifest);c['request_manifest_sha256']=digest(manifest)
    p={'run_id':str(uuid.uuid4()),'experiment_signature':digest(c),'contract':c,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/'protocol.json',p);verify(out)
    print(json.dumps({'stage':'prepared','lambda':ridge,'alpha':alpha,'records':1484,'unique_requests':c['unique_payloads']}),flush=True)
    return p


def success_checks(overall):
    b,m=RAW;bz,mz=b+'_zscore',m+'_zscore'
    return {'baseline_zscore_beats_baseline':overall[bz]['correct']>overall[b]['correct'],
            'master_zscore_beats_master':overall[mz]['correct']>overall[m]['correct'],
            'master_zscore_beats_baseline_zscore':overall[mz]['correct']>overall[bz]['correct']}


def analyze(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();out=output/'reader';p=verify(out);c=p['contract']
    validation=read(out/'validation_output.json');assert validation['passed']
    for n in ('responses','fresh_responses'):assert sha(out/(n+'.jsonl'))==validation[n+'_sha256']
    requests=indexed(lines(out/'inputs/requests.jsonl'),'record_id');answers=indexed(lines(out/'responses.jsonl'),'record_id');gold=lines(out/'inputs/gold.jsonl')
    assert len(gold)==371 and len(answers)==1484 and set(requests)==set(answers)
    pairs=[]
    for g in gold:
        qid=g['id'];row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for n in CONDITIONS:
            rid=qid+'::'+n;answer=api.parse_choice(answers[rid]['raw_response']);ids=requests[rid]['top_memory_ids']
            row.update({n+'_answer':answer,n+'_correct':answer==g['gold_choice'],n+'_top5':ids,n+'_gold_hits':len(set(ids)&set(g['gold_ids']))})
        for n in RAW:
            same=requests[qid+'::'+n]['payload_sha256']==requests[qid+'::'+n+'_zscore']['payload_sha256']
            row[n+'_identical_payload']=same
            if same:assert row[n+'_answer']==row[n+'_zscore_answer']
        pairs.append(row)
    overall={n:{'correct':sum(r[n+'_correct'] for r in pairs),'questions':371,'accuracy':sum(r[n+'_correct'] for r in pairs)/371,
                'gold_hits_top5':sum(r[n+'_gold_hits'] for r in pairs),'gold_references':1041,'questions_with_any_gold':sum(r[n+'_gold_hits']>0 for r in pairs)} for n in CONDITIONS}
    comparisons={}
    for a,b in [(n,n+'_zscore') for n in RAW]+[(RAW[0]+'_zscore',RAW[1]+'_zscore')]:
        result=previous.compare(pairs,a,b)
        same=[r for r in pairs if requests[r['question_id']+'::'+a]['payload_sha256']==requests[r['question_id']+'::'+b]['payload_sha256']]
        assert all(r[a+'_answer']==r[b+'_answer'] for r in same)
        result.update(identical_request_questions=len(same),changed_request_questions=371-len(same),identical_request_answer_disagreements=0)
        comparisons[a+'__'+b]=result
    checks=success_checks(overall);methods=copy.deepcopy(c['method_annotations'])
    for n in CONDITIONS:methods[n]['5. 回答方式'].update(correct=overall[n]['correct'],returned_models=validation['returned_models'])
    fresh=lines(out/'fresh_responses.jsonl')
    summary={'status':'complete','lambda':c['lambda'],'alpha':c['alpha'],'selection_path':c['selection_path'],'selection_sha256':c['selection_sha256'],
             'requested_model':c['model'],'returned_model':c['expected_returned_model'],'caption_model':c['caption_model'],'reader_seed':42,
             'overall':overall,'comparisons':comparisons,'success_checks':checks,'validation_passed':all(checks.values()),
             'by_domain':{d:{n:{'correct':sum(r[n+'_correct'] for r in pairs if r['domain']==d),'questions':sum(r['domain']==d for r in pairs)} for n in CONDITIONS} for d in sorted({r['domain'] for r in pairs})},
             'method_annotations':methods,'validation':validation,'interpretation':c['interpretation'],
             'usage':{k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',pairs)
    with (output/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Qwen 选参 → GPT-5 验证</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 16px}td,th{padding:9px;border-bottom:1px solid #ccc}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}</style>',
          f'<h1>Qwen 选参 → GPT-5 验证</h1><p>固定共享参数 λ={c["lambda"]}、α={c["alpha"]}；由 Qwen 的 25 组搜索选出，GPT-5 不参与调参。沿用同一批 GPT-5 caption、438 条记忆和 72 个 master。371 题，四个条件均重新回答，相同完整请求共享一次本轮回答。</p>',
          '<table><tr><th>方案</th><th>正确数</th><th>准确率</th><th>目标记忆命中数</th></tr>']
    for n in CONDITIONS:
        r=overall[n];body.append(f'<tr><td>{n}</td><td>{r["correct"]}/371</td><td>{r["accuracy"]:.4%}</td><td>{r["gold_hits_top5"]}/1041</td></tr>')
    body+=['</table><h2>预先约定的验证条件</h2><pre>'+html.escape(json.dumps(checks,ensure_ascii=False,indent=2))+'</pre>',
           '<h2>逐题配对比较</h2><pre>'+html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))+'</pre>',
           '<p>同一基准上的 backbone 迁移验证；一次运行中的观察，不等于独立测试集泛化或稳定统计显著提升。</p>',
           '<p><a href="summary.json">完整统计及五项方法标注</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a></p></html>']
    (output/'report.html').write_text('\n'.join(body));print(json.dumps({'stage':'analyzed','overall':overall,'success_checks':checks,'passed':all(checks.values())}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--memory',type=Path,default=SOURCE)
    p.add_argument('--stage',choices=['prepare','smoke','run','analyze','all'],default='all');p.add_argument('--api-workers',type=int,default=3);a=p.parse_args()
    from scripts.analysis.audit_memeye_gpt5_qwen_selected import audit
    if a.stage in ('prepare','all'):prepare(a.output,a.memory,a.api_workers)
    if a.stage in ('prepare','smoke','run','all'):audit(a.output,inputs_only=True)
    if a.stage in ('smoke','all'):run(a.output/'reader',smoke=True)
    if a.stage in ('run','all'):run(a.output/'reader')
    if a.stage in ('analyze','all'):
        analyze(a.output);proof=audit(a.output);dump(a.output/'execution_status.json',{'status':'complete','independent_audit_passed':proof['passed'],'validation_passed':proof['validation_passed']})

if __name__=='__main__':main()
