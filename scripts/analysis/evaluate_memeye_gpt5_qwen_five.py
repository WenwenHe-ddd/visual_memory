"""Fresh GPT-5 follow-up for all five Qwen-eligible shared parameter pairs."""
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
from scripts.analysis.retrieve_memeye_gpt5_qwen_five import DEFAULT_OUTPUT, SEARCH, FIRST, RAW, CONDITIONS, SPECS, PAIRS, condition, eligible_pairs, verify_source
read,lines,sha,digest,dump=api.read,api.lines,api.sha,api.digest,api.dump
indexed,write_lines,verify=api.indexed,api.write_lines,api.verify
SOURCE=previous.BASE
run=previous.run


def prepare(output=DEFAULT_OUTPUT,source=SOURCE,workers=3):
    output,source=Path(output).resolve(),Path(source).resolve();out=output/'reader'
    ret=read(output/'retrieval_protocol.json');candidates,selection_hashes=eligible_pairs(ret['qwen_search_path'])
    if (out/'protocol.json').exists():
        p=verify(out);c=p['contract']
        assert c['condition_specs']==SPECS and c['memory_path']==str(source)
        return p
    proof=verify_source(source);pc=verify(source/'reader')['contract']
    assert ret['condition_specs']==SPECS and ret['memory_path']==str(source)
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
        offset=i%len(CONDITIONS);records.extend(batch[offset:]+batch[:offset])
    assert len(records)==4452 and len(gold)==371 and raw_matches==742
    (out/'inputs').mkdir(parents=True,exist_ok=True)
    write_lines(out/'inputs/requests.jsonl',records);write_lines(out/'inputs/gold.jsonl',gold)
    sources={**proof['source_sha256'],**ret['source_sha256'],**ret['output_sha256'],**selection_hashes}
    paths=[Path(__file__),ROOT/'scripts/analysis/audit_memeye_gpt5_qwen_five.py',ROOT/'scripts/tests/test_memeye_gpt5_qwen_five.py',
           ROOT/'scripts/analysis/retrieve_memeye_gpt5_qwen_five.py',ROOT/'scripts/analysis/evaluate_memeye_gpt5_base.py',
           ROOT/'scripts/analysis/audit_memeye_gpt5_base.py',output/'retrieval_protocol.json',out/'inputs/requests.jsonl',out/'inputs/gold.jsonl']
    paths.extend(Path(im['path']) for q in qs.values() for im in q['query_images'])
    sources.update({str(p):sha(p) for p in paths})
    keys=('model','requested_model','expected_returned_model','caption_model','seed','caption_seed','temperature','temperature_wire_policy',
          'reasoning_effort','max_completion_tokens','image_detail','base_url','api_config','api_env_file','reader_system_prompt','timeout_seconds','max_attempts')
    c={k:copy.deepcopy(pc[k]) for k in keys};annotations={}
    for name in CONDITIONS:
        spec=SPECS[name];weighted=spec['alpha']>0
        base_name=spec['method']+('_zscore' if weighted else '')
        m=copy.deepcopy(pc['method_annotations'][base_name])
        m['1. 候选记忆结构'].update(caption_source='同一批已核验 GPT-5 caption',text='原始对话＋已有完整 GPT-5 caption')
        m['1. 候选记忆结构']['weighting_group']=m['1. 候选记忆结构']['weighting_group'].replace('新 master','已有 master')
        m['2. 查询内容']=m['2. 查询内容'].replace('本轮 GPT-5','已有 GPT-5')
        m['3. 检索方式'].update({'lambda':spec['lambda'],'alpha':spec['alpha'] if weighted else None})
        m['4. 编码方式']['CLIP']='沿用 GPT-5 caption 重建所得的 72 个 master 及视觉聚合量；仅重新编码检索文本'
        m['5. 回答方式'].update(same_run_identical_requests='十二个条件中完整请求相同则共享本轮新回答；不复用历史回答',correct=None)
        annotations[name]=m
    c.update(questions=371,logical_records=4452,unique_payloads=len({r['payload_sha256'] for r in records}),conditions=list(CONDITIONS),
             memory_path=str(source),source_run_id=read(source/'protocol.json')['run_id'],top_k=5,workers=workers,source_sha256=sources,
             method_annotations=annotations,historical_api_answer_reuse=False,gold_in_api_payload=False,caption_regenerated=False,memory_rebuilt=False,
             qwen_search_path=ret['qwen_search_path'],parameter_pairs=candidates,condition_specs=SPECS,first_validation_path=str(FIRST),
             candidate_list_from_qwen_only=True,expanded_after_first_gpt5_failure=True,
             same_run_controls=True,within_run_sharing='One fresh answer per complete request across all twelve conditions.',
             interpretation='User-requested follow-up after the first GPT-5 candidate failed: test all five Qwen-eligible shared pairs. Fresh common controls and treatments. No historical answers reused. Report all candidates and all three strict comparisons; this expanded comparison is exploratory, not a held-out confirmatory test.')
    manifest=[{'record_id':r['record_id'],'payload_sha256':r['payload_sha256'],
               'request_sha256':digest(previous.request_body({'contract':c},r)),
               'query_image_count':len(r['query_images']),'requested_model':c['model']} for r in records]
    write_lines(out/'api_request_manifest.jsonl',manifest);c['request_manifest_sha256']=digest(manifest)
    p={'run_id':str(uuid.uuid4()),'experiment_signature':digest(c),'contract':c,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    dump(out/'protocol.json',p);verify(out)
    print(json.dumps({'stage':'prepared','parameter_pairs':len(PAIRS),'conditions':len(CONDITIONS),'records':4452,'unique_requests':c['unique_payloads']}),flush=True)
    return p


def candidate_table(overall):
    b0,m0=(overall[n]['correct'] for n in RAW)
    table=[]
    for rank,(ridge,alpha) in enumerate(PAIRS,1):
        b,m=(overall[condition(n,ridge,alpha)]['correct'] for n in RAW)
        checks={'baseline_zscore_beats_baseline':b>b0,'master_zscore_beats_master':m>m0,'master_zscore_beats_baseline_zscore':m>b}
        table.append({'qwen_rank':rank,'lambda':ridge,'alpha':alpha,'baseline_correct':b,'master_correct':m,
                      'baseline_gain':b-b0,'master_gain':m-m0,'master_minus_baseline':m-b,
                      **checks,'passes_all_three':all(checks.values())})
    return table


def analyze(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();out=output/'reader';p=verify(out);c=p['contract']
    validation=read(out/'validation_output.json');assert validation['passed']
    for n in ('responses','fresh_responses'):assert sha(out/(n+'.jsonl'))==validation[n+'_sha256']
    requests=indexed(lines(out/'inputs/requests.jsonl'),'record_id');answers=indexed(lines(out/'responses.jsonl'),'record_id');gold=lines(out/'inputs/gold.jsonl')
    assert len(gold)==371 and len(answers)==4452 and set(requests)==set(answers)
    pairs=[]
    for g in gold:
        qid=g['id'];row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        same_answers={}
        for n in CONDITIONS:
            rid=qid+'::'+n;answer=api.parse_choice(answers[rid]['raw_response']);ids=requests[rid]['top_memory_ids']
            row.update({n+'_answer':answer,n+'_correct':answer==g['gold_choice'],n+'_top5':ids,n+'_gold_hits':len(set(ids)&set(g['gold_ids']))})
            h=requests[rid]['payload_sha256']
            if h in same_answers:assert answer==same_answers[h]
            same_answers[h]=answer
        pairs.append(row)
    overall={n:{'correct':sum(r[n+'_correct'] for r in pairs),'questions':371,'accuracy':sum(r[n+'_correct'] for r in pairs)/371,
                'gold_hits_top5':sum(r[n+'_gold_hits'] for r in pairs),'gold_references':1041,'questions_with_any_gold':sum(r[n+'_gold_hits']>0 for r in pairs)} for n in CONDITIONS}
    comparisons={}
    for ridge,alpha in PAIRS:
        bz,mz=(condition(n,ridge,alpha) for n in RAW)
        for a,b in ((RAW[0],bz),(RAW[1],mz),(bz,mz)):
            result=previous.compare(pairs,a,b)
            same=[r for r in pairs if requests[r['question_id']+'::'+a]['payload_sha256']==requests[r['question_id']+'::'+b]['payload_sha256']]
            assert all(r[a+'_answer']==r[b+'_answer'] for r in same)
            result.update(identical_request_questions=len(same),changed_request_questions=371-len(same),identical_request_answer_disagreements=0)
            comparisons[a+'__'+b]=result
    table=candidate_table(overall);methods=copy.deepcopy(c['method_annotations'])
    for n in CONDITIONS:methods[n]['5. 回答方式'].update(correct=overall[n]['correct'],returned_models=validation['returned_models'])
    fresh=lines(out/'fresh_responses.jsonl')
    summary={'status':'complete','parameter_pairs':c['parameter_pairs'],'condition_specs':SPECS,'qwen_search_path':c['qwen_search_path'],
             'requested_model':c['model'],'returned_model':c['expected_returned_model'],'caption_model':c['caption_model'],'reader_seed':42,
             'overall':overall,'comparisons':comparisons,'candidate_results':table,'passing_pairs':[r for r in table if r['passes_all_three']],
             'by_domain':{d:{n:{'correct':sum(r[n+'_correct'] for r in pairs if r['domain']==d),'questions':sum(r['domain']==d for r in pairs)} for n in CONDITIONS} for d in sorted({r['domain'] for r in pairs})},
             'method_annotations':methods,'validation':validation,'interpretation':c['interpretation'],'previous_first_validation_retained':True,
             'usage':{k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh) for k in ('prompt_tokens','completion_tokens','total_tokens')}}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',pairs)
    for filename,records in (('paired_results.csv',pairs),('candidate_results.csv',table)):
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    old_answers=indexed(lines(FIRST/'reader/responses.jsonl'),'record_id')
    old_requests=indexed(lines(FIRST/'reader/inputs/requests.jsonl'),'record_id');repeats={}
    for old_name in (RAW[0],RAW[0]+'_zscore',RAW[1],RAW[1]+'_zscore'):
        name=condition(old_name.removesuffix('_zscore'),.01,1.) if old_name.endswith('_zscore') else old_name
        rr=[]
        for g in gold:
            rid=g['id']+'::'+name;oid=g['id']+'::'+old_name
            assert requests[rid]['payload_sha256']==old_requests[oid]['payload_sha256']
            a,b=api.parse_choice(old_answers[oid]['raw_response']),api.parse_choice(answers[rid]['raw_response'])
            rr.append({'question_id':g['id'],'old_answer':a,'new_answer':b,'old_correct':a==g['gold_choice'],'new_correct':b==g['gold_choice']})
        repeats[old_name]={**previous.compare(rr,'old','new'),'identical_requests':371,'changed_questions':[r['question_id'] for r in rr if r['old_answer']!=r['new_answer']]}
    dump(output/'previous_first_repeatability.json',{'previous_path':str(FIRST),'conditions':repeats,'all_requests_identical':True,'historical_answers_reused':False})
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Qwen 五组候选 → GPT-5 验证</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:32px auto;padding:0 16px}td,th{padding:8px;border-bottom:1px solid #ccc}table{border-collapse:collapse;width:100%}pre{white-space:pre-wrap}.pass{background:#e6f5e9}</style>',
          '<h1>Qwen 五组候选 → GPT-5 验证</h1><p>按用户要求，在首个候选验证未通过后，扩展测试 Qwen 中满足条件的全部 5 组共享参数。沿用同一批 GPT-5 caption、438 条记忆和 72 个 master；每组两种检索各 371 题，加上两个本轮未赋权对照，共 4,452 条记录。完整请求相同则共享一次本轮新回答。</p>',
          f'<p>本轮对照：baseline_question_only = {overall[RAW[0]]["correct"]}/371；master_answer_query = {overall[RAW[1]]["correct"]}/371。</p>',
          '<table><tr><th>Qwen 排名</th><th>λ</th><th>α</th><th>baseline + zscore</th><th>master + zscore</th><th>baseline 增益</th><th>master 增益</th><th>master − baseline</th><th>全部通过</th></tr>']
    for row in table:
        body.append('<tr'+(' class="pass"' if row['passes_all_three'] else '')+'>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in ('qwen_rank','lambda','alpha','baseline_correct','master_correct','baseline_gain','master_gain','master_minus_baseline','passes_all_three'))+'</tr>')
    body+=['</table><p>通过要求：baseline 加权严格优于其未赋权对照；master 加权严格优于其未赋权对照；master 加权严格优于 baseline 加权。所有分母固定为 371，异常回答计错。</p>',
           '<p>这是看到首轮结果后扩展的同基准探索性比较；不将挑选出的结果称为独立验证或稳定提升。此前单候选失败记录保持原样。</p>',
           '<details><summary>15 项逐题配对比较</summary><pre>'+html.escape(json.dumps(comparisons,indent=2))+'</pre></details>',
           '<details><summary>与前一轮相同请求的回答波动</summary><pre>'+html.escape(json.dumps(repeats,indent=2))+'</pre></details>',
           '<p><a href="summary.json">完整统计及五项方法标注</a> · <a href="candidate_results.csv">五组比较 CSV</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a></p></html>']
    (output/'report.html').write_text('\n'.join(body));print(json.dumps({'stage':'analyzed','controls':{n:overall[n] for n in RAW},'candidate_results':table,'passing_pairs':len(summary['passing_pairs'])}),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--memory',type=Path,default=SOURCE)
    p.add_argument('--stage',choices=['prepare','smoke','run','analyze','all'],default='all');p.add_argument('--api-workers',type=int,default=3);a=p.parse_args()
    assert a.api_workers>0
    from scripts.analysis.audit_memeye_gpt5_qwen_five import audit
    if a.stage in ('prepare','all'):prepare(a.output,a.memory,a.api_workers)
    if a.stage in ('prepare','smoke','run','all'):audit(a.output,inputs_only=True)
    if a.stage in ('smoke','all'):run(a.output/'reader',smoke=True)
    if a.stage in ('run','all'):run(a.output/'reader')
    if a.stage in ('analyze','all'):
        analyze(a.output);proof=audit(a.output);dump(a.output/'execution_status.json',{'status':'complete','independent_audit_passed':proof['passed'],'passing_pairs':proof['passing_pairs']})

if __name__=='__main__':main()
