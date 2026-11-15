"""MemEye GPT-5.4 reader evaluation on frozen, correct z-score retrievals.

Fixed lambda=.01, alpha=1; fresh API answers for both 371-question conditions.
The established GPT-5.4 runtime/transport and historical outputs stay unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.evaluation import memeye_gpt5_reader as runtime
from scripts.evaluation.memeye_reader_inputs import request_from_row, validate_responses, digest as payload_digest
from scripts.evaluation.openai_gpt5_chat_transport import load_config

BASE=ROOT/'results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024'
ORIGINAL=ROOT/'data/independent_visual_inputs'
RAW=ROOT/'reports/all_master_answer_query_20261021'
REFERENCE=runtime.DEFAULT_OUTPUT
CONDITIONS=('baseline_question_only_zscore','master_answer_query_zscore')
SOURCES={CONDITIONS[0]:(ROOT/'reports/baseline_question_only_lambda_ablation_20261023','lambda_0p01',188),
         CONDITIONS[1]:(ROOT/'reports/master_answer_query_zscore_20261023','master_answer_query_zscore',201)}
LABELS={CONDITIONS[0]:'baseline_question_only + z-score',CONDITIONS[1]:'master_answer_query + z-score'}
read,lines,sha,digest,dump=runtime.read,runtime.lines,runtime.sha,runtime.digest,runtime.dump


def load_frozen_inputs():
    originals=runtime.indexed(lines(ORIGINAL/'qa_manifest.jsonl'),'id')
    memories={r['id']:r['reader_text'] for r in lines(ORIGINAL/'retrieval_memories.jsonl')}
    raw_rows=runtime.indexed(lines(RAW/'retrieval_results.jsonl'),'question_id')
    assert len(originals)==len(raw_rows)==371 and len(memories)==438
    requests_by={}; qwen={}; annotations={}; paths=[ORIGINAL/'qa_manifest.jsonl',ORIGINAL/'retrieval_memories.jsonl',
        ORIGINAL/'protocol.json',RAW/'retrieval_results.jsonl',RAW/'candidates.jsonl']
    drift=[]; rank_checks=0; group_checks=0; leaf_checks=0; settings=None
    retrieval_stats={}
    for name,(base,source_condition,expected_correct) in SOURCES.items():
        rp=read(base/'reader/protocol.json');unsigned=dict(rp);signature=unsigned.pop('experiment_signature')
        assert payload_digest(unsigned)==signature
        root=read(base/'protocol.json');audit=read(base/'reader/independent_audit.json');assert audit['passed']
        if settings is None:settings=rp['reader_settings']
        assert rp['reader_settings']==settings
        for rel in ('protocol.json','retrieval_results.jsonl','retrieval_validation.json','reader/protocol.json',
                    'reader/qa_manifest.jsonl','reader/responses.jsonl','reader/summary.json','reader/independent_audit.json'):
            path=base/rel;paths.append(path)
            if str(path) in rp['source_sha256']:assert sha(path)==rp['source_sha256'][str(path)]
        # Old executable source changes are disclosed; immutable data drift fails.
        for label,protocol in (('retrieval',root),('reader',rp)):
            for filename,expected in protocol['source_sha256'].items():
                path=Path(filename);actual=sha(path) if path.is_file() else None
                if actual!=expected:
                    assert path.suffix=='.py',('changed non-code input',filename)
                    drift.append({'condition':name,'stage':label,'path':filename,'historical_sha256':expected,'current_sha256':actual})
        all_manifest=lines(base/'reader/qa_manifest.jsonl');all_responses=lines(base/'reader/responses.jsonl')
        answers=validate_responses(all_manifest,all_responses,rp)
        selected=[r for r in all_manifest if r['condition']==source_condition]
        assert len(selected)==371 and {r['id'] for r in selected}==set(originals)
        retrieval=runtime.indexed(lines(base/'retrieval_results.jsonl'),'question_id');assert set(retrieval)==set(originals)
        if name==CONDITIONS[0]:assert root['alpha']==1 and .01 in root['new_lambdas']
        else:assert root['lambda']==.01 and root['alpha']==1
        requests_by[name]={};qwen[name]={};hits=any_hits=0
        for row in selected:
            qid=row['id'];q=originals[qid];ret=retrieval[qid]
            for key in ('question','options','query_image_paths'):assert ret[key]==q[key]
            assert ret['evaluation']=={key:q[key] for key in ('gold_choice','gold_ids')}
            c=next(x for x in ret['conditions'] if x['name']==source_condition)
            weighting=ret['weighting'][source_condition] if 'weighting' in ret else ret
            groups=weighting['groups'];leaves=weighting['leaf_scores']
            if name==CONDITIONS[0]:
                assert c['lambda']==.01 and len(groups)==1 and groups[0]['group_id']=='all_memory'
                assert groups[0]['query']==ret['retrieval_query']==raw_rows[qid]['baseline_retrieval_query']
            else:
                old={m['master_id']:m for m in raw_rows[qid]['masters']}
                assert {g['group_id'] for g in groups}==set(old)
                for g in groups:
                    master=old[g['group_id']]
                    assert g['choice']==master['selected_choice'] and g['selected_statement']==master['selected_statement']
                    assert g['query']==master['augmented_query']
            assert sum(g['size'] for g in groups)==len(leaves)
            for group in groups:
                subset=[l for l in leaves if l['group_id']==group['group_id']];assert len(subset)==group['size']
                raw=np.array([l['raw_score'] for l in subset]);white=np.array([l['white_score'] for l in subset])
                z=np.array([l['white_z'] for l in subset]);correction=np.array([l['correction'] for l in subset])
                if len(subset)==1:expected_z=white.copy()
                elif group['z_status']=='zero_variance':expected_z=np.zeros(len(subset))
                else:expected_z=(white-white.mean())/white.std()
                np.testing.assert_allclose(z,expected_z,rtol=0,atol=1e-10)
                expected=np.zeros(len(subset)) if len(subset)==1 or group['raw_status']=='zero_variance' or group['z_status']=='zero_variance' else raw.std()*np.tanh(z)
                np.testing.assert_allclose(correction,expected,rtol=0,atol=1e-14)
                np.testing.assert_allclose([l['weighted_score'] for l in subset],raw+correction,rtol=0,atol=1e-15)
                group_checks+=1;leaf_checks+=len(subset)
            ids=[l['memory_id'] for l in sorted(leaves,key=lambda l:-l['weighted_score'])[:5]]
            assert ids==c['top5_memory_ids']==row['top_memory_ids']
            assert [l['reader_text'] for l in c['top5']]==[memories[mid] for mid in ids]
            assert [l['score'] for l in c['top5']]==row['top_query_scores']
            request=request_from_row(row,q,memories,settings['reader_system_prompt'])
            request.update(record_id=qid+'::'+name,condition=name)
            requests_by[name][qid]=request
            answer=answers[row['record_id']]
            qwen[name][qid]={'predicted_choice':answer['predicted_choice'],'correct':answer['correct'],
                'payload_sha256':answer['payload_sha256'],'source_record_id':answer['record_id']}
            h=len(set(ids)&set(q['gold_ids']));assert h==c['gold_hits_top5'];hits+=h;any_hits+=h>0;rank_checks+=1
        assert sum(x['correct'] for x in qwen[name].values())==expected_correct
        a=copy.deepcopy(read(base/'reader/summary.json')['method_annotations'][source_condition])
        a['scope']=name+'; frozen retrieval, GPT-5.4 reader evaluation'
        assert a['3. 检索方式']['lambda']==.01 and a['3. 检索方式']['alpha']==1
        a['5. 回答方式']={'reader_model':runtime.MODEL,'reasoning_effort':'none','temperature':0,
            'max_completion_tokens':16,'seed':None,'input':'Same ranked Top-5 original dialogue + full caption, question, options, native query images.',
            'reader_called':False,'measured_qa_accuracy':None,'historical_qwen_correct':expected_correct}
        annotations[name]=a;retrieval_stats[name]={'gold_hits_top5':hits,'gold_references':1041,'questions_with_any_gold':any_hits}
    records=[]
    for i,qid in enumerate(originals):
        pair=[requests_by[name][qid] for name in CONDITIONS]
        records.extend(pair if i%2==0 else pair[::-1])
    assert len(records)==742
    image_paths={p for r in records for p in r['query_image_paths']};paths.extend(Path(p) for p in image_paths)
    validation={'passed':True,'questions':371,'records':742,'ordered_top5_verified':rank_checks,
        'groups_checked':group_checks,'leaf_scores_checked':leaf_checks,'unique_payloads':len({r['payload_sha256'] for r in records}),
        'image_questions':sum(bool(q['query_image_paths']) for q in originals.values()),'unique_image_files':len(image_paths),
        'qwen_correct':{n:sum(r['correct'] for r in d.values()) for n,d in qwen.items()},'retrieval_stats':retrieval_stats,
        'historical_source_drift':drift,'frozen_retrieval':True,'gold_excluded_from_requests':True}
    return records,originals,qwen,settings,annotations,paths,validation


def prepare(output=BASE,env_file=runtime.DEFAULT_ENV,workers=3):
    output=Path(output).resolve()
    if (output/'protocol.json').exists():return runtime.verify(output)
    if output.exists():raise FileExistsError('Use a fresh output directory.')
    config=load_config(env_file);assert workers>0
    records,originals,qwen,settings,annotations,paths,validation=load_frozen_inputs()
    # Preserve the prior unweighted GPT-5.4 run only as an attributed reference.
    ref_protocol=read(REFERENCE/'protocol.json');assert digest(ref_protocol['contract'])==ref_protocol['experiment_signature']
    assert ref_protocol['contract']['model']==runtime.MODEL and ref_protocol['contract']['reasoning_effort']=='none'
    for rel in ('protocol.json','responses.jsonl','fresh_responses.jsonl','summary.json','independent_audit.json','validation_output.json','inputs/requests.jsonl','inputs/gold.jsonl'):
        paths.append(REFERENCE/rel)
    assert read(REFERENCE/'independent_audit.json')['passed']
    ref_validation=read(REFERENCE/'validation_output.json')
    assert sha(REFERENCE/'responses.jsonl')==ref_validation['responses_sha256']
    output.mkdir(parents=True);(output/'inputs').mkdir()
    runtime.write_lines(output/'inputs/requests.jsonl',records)
    runtime.write_lines(output/'inputs/gold.jsonl',[{k:q[k] for k in ('id','task','gold_choice','gold_ids')} for q in originals.values()])
    dump(output/'inputs/qwen_reference.json',qwen);dump(output/'inputs/input_audit.json',validation)
    paths.extend(output/'inputs'/n for n in ('requests.jsonl','gold.jsonl','qwen_reference.json','input_audit.json'))
    paths.extend([Path(__file__),ROOT/'scripts/analysis/audit_memeye_gpt54_zscore.py',
        ROOT/'scripts/evaluation/memeye_gpt5_reader.py',ROOT/'scripts/evaluation/memeye_reader_inputs.py',
        ROOT/'scripts/evaluation/openai_gpt5_chat_transport.py',ROOT/'scripts/evaluation/openai_chat_transport.py'])
    contract={'dataset':'MemEye','questions':371,'domains':8,'logical_records':742,'unique_payloads':validation['unique_payloads'],
        'conditions':list(CONDITIONS),'lambda':.01,'alpha':1.,'top_k':5,'memory_leaves':438,
        'model':config.model,'base_url':config.base_url,'api_config':config.public_dict(),'api_env_file':str(Path(env_file).resolve()),
        'api':'chat/completions','temperature':0,'seed':None,'reasoning_effort':'none','max_completion_tokens':16,'image_detail':'high',
        'workers':workers,'timeout_seconds':60,'max_attempts':3,'retry_policy':'Only transient transport errors; same request; never retry answer format or correctness.',
        'reader_system_prompt':settings['reader_system_prompt'],'qwen_reference_settings':settings,
        'source_sha256':{str(p):sha(p) for p in paths},'method_annotations':annotations,
        'retrieval_sources':{n:{'directory':str(p),'condition':c} for n,(p,c,_) in SOURCES.items()},
        'prior_unweighted_gpt54_reference':str(REFERENCE),'historical_api_answer_reuse':False,'gold_in_api_payload':False,
        'changed_component':'Only final reader on existing lambda=.01 alpha=1 retrieval. All Top5, question/options, text and native query images frozen.',
        'deduplication':'Identical complete requests share a fresh result only within this signed run.',
        'interpretation':'Fixed option-order MCQ accuracy, all 371 questions; not paper four-rotation EM. API determinism is not guaranteed.',
        'official_model_documentation':'https://developers.openai.com/api/docs/models/gpt-5.4'}
    provisional={'contract':contract};wire=[]
    for row in records:
        wire.append({'record_id':row['record_id'],'payload_sha256':row['payload_sha256'],
            'request_sha256':digest(runtime.request_body(provisional,row)),'query_image_count':len(row['query_images'])})
    contract['request_manifest_sha256']=digest(wire)
    runtime.write_lines(output/'api_request_manifest.jsonl',wire)
    protocol={'run_id':str(uuid.uuid4()),'experiment_signature':digest(contract),
        'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'contract':contract}
    dump(output/'protocol.json',protocol);runtime.verify(output)
    print(json.dumps({'stage':'prepared','questions':371,'records':742,'unique_requests':validation['unique_payloads'],
        'lambda':.01,'alpha':1,'qwen_reference':validation['qwen_correct'],'gold_in_requests':False}),flush=True)
    return protocol


def aggregate(pairs):
    n=len(pairs);result={'questions':n,'conditions':{}}
    for name in CONDITIONS:
        correct=sum(p[name+'_correct'] for p in pairs)
        result['conditions'][name]={'correct':correct,'accuracy':correct/n}
    wins=sum(not p[CONDITIONS[0]+'_correct'] and p[CONDITIONS[1]+'_correct'] for p in pairs)
    losses=sum(p[CONDITIONS[0]+'_correct'] and not p[CONDITIONS[1]+'_correct'] for p in pairs)
    m=wins+losses;p=min(1.,2*sum(math.comb(m,k) for k in range(min(wins,losses)+1))/2**m) if m else 1.
    result['master_vs_baseline']={'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses,
        'delta_percentage_points':100*(wins-losses)/n,'mcnemar_exact_two_sided_p':p}
    return result


def analyze(output=BASE):
    output=Path(output);protocol=runtime.verify(output);c=protocol['contract'];validation=read(output/'validation_output.json')
    assert validation['passed']
    for name in ('responses','fresh_responses'):assert sha(output/(name+'.jsonl'))==validation[name+'_sha256']
    requests=runtime.indexed(lines(output/'inputs/requests.jsonl'),'record_id')
    answers=runtime.indexed(lines(output/'responses.jsonl'),'record_id')
    gold=runtime.indexed(lines(output/'inputs/gold.jsonl'),'id')
    assert set(answers)==set(requests) and len(answers)==742 and len(gold)==371
    pairs=[]
    for qid,g in gold.items():
        p={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for name in CONDITIONS:
            rid=qid+'::'+name;r=answers[rid];q=requests[rid]
            assert r['payload_sha256']==q['payload_sha256'] and r['predicted_choice']==runtime.parse_choice(r['raw_response'])
            p.update({name+'_answer':r['predicted_choice'],name+'_correct':r['predicted_choice']==g['gold_choice'],
                name+'_top5':q['top_memory_ids'],name+'_gold_hits':len(set(q['top_memory_ids'])&set(g['gold_ids']))})
        pairs.append(p)
    old=runtime.indexed(lines(REFERENCE/'responses.jsonl'),'record_id');old_requests=runtime.indexed(lines(REFERENCE/'inputs/requests.jsonl'),'record_id')
    old_gold=runtime.indexed(lines(REFERENCE/'inputs/gold.jsonl'),'id');assert set(old_gold)==set(gold)
    comparisons={}
    for name,raw_name in zip(CONDITIONS,('baseline_question_only','master_answer_query')):
        wins=losses=identical=0;disagreements=[];old_correct=0
        for qid,g in gold.items():
            assert old_gold[qid]=={k:g[k] for k in ('id','gold_choice','gold_ids')}
            rid=qid+'::'+name;prior=qid+'::'+raw_name;a=answers[rid];b=old[prior]
            before=b['predicted_choice']==g['gold_choice'];after=a['predicted_choice']==g['gold_choice']
            old_correct+=before;wins+=not before and after;losses+=before and not after
            if requests[rid]['payload_sha256']==old_requests[prior]['payload_sha256']:
                identical+=1
                if a['predicted_choice']!=b['predicted_choice']:disagreements.append(qid)
        comparisons[name]={'prior_unweighted_correct':old_correct,'wrong_to_right':wins,'right_to_wrong':losses,
            'net_correct':wins-losses,'identical_payload_questions':identical,'identical_payload_answer_disagreements':disagreements,
            'reference_source':str(REFERENCE),'caveat':'Prior run, no new unweighted control. Same model settings do not guarantee identical API responses.'}
    assert [comparisons[n]['prior_unweighted_correct'] for n in CONDITIONS]==[227,235]
    annotations=copy.deepcopy(c['method_annotations']);overall=aggregate(pairs)
    for name in CONDITIONS:
        a=overall['conditions'][name];annotations[name]['5. 回答方式'].update(reader_called=True,correct=a['correct'],measured_qa_accuracy=a['accuracy'])
    fresh=lines(output/'fresh_responses.jsonl')
    usage={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in fresh) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    summary={'status':'complete','model_requested':c['model'],'model_returned':validation['returned_models'],
        'lambda':.01,'alpha':1.,'overall':overall,'by_domain':{d:aggregate([p for p in pairs if p['domain']==d]) for d in sorted({p['domain'] for p in pairs})},
        'method_annotations':annotations,'retrieval_stats':read(output/'inputs/input_audit.json')['retrieval_stats'],
        'qwen_reference_correct':read(output/'inputs/input_audit.json')['qwen_correct'],'prior_unweighted_gpt54_comparisons':comparisons,
        'usage_fresh_requests_only':usage,'validation':validation,'metric_policy':c['interpretation']}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',pairs)
    with (output/'paired_results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    render(output,summary)
    print(json.dumps({'stage':'complete','overall':overall,'validation':validation},ensure_ascii=False),flush=True)
    return summary


def render(output,summary):
    import html
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemEye GPT-5.4 z-score</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#20333d}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>MemEye：GPT-5.4，λ=0.01、α=1</h1><p>全部 371 题、8 个 domain。冻结两套正确方案的已有 Top-5，仅替换回答模型。每条记忆保留对话＋完整 caption；full_scan 使用原查询，multi_path 使用 master 选中的完整候选陈述。</p>',
        '<table><tr><th>检索方案</th><th>GPT-5.4 答对</th><th>准确率</th><th>同输入 Qwen 参照</th><th>Top-5 目标召回</th></tr>']
    for name in CONDITIONS:
        v=summary['overall']['conditions'][name];r=summary['retrieval_stats'][name]
        parts.append(f'<tr><td>{LABELS[name]}</td><td>{v["correct"]}/371</td><td>{v["accuracy"]:.4%}</td><td>{summary["qwen_reference_correct"][name]}/371</td><td>{r["gold_hits_top5"]}/1041</td></tr>')
    change=summary['overall']['master_vs_baseline'];v=summary['validation']
    parts.append(f'</table><p>加权多路径相对加权全量：{change["wrong_to_right"]} 题错→对，{change["right_to_wrong"]} 题对→错，净变化 {change["net_correct"]:+d} 题；McNemar 双侧 p={change["mcnemar_exact_two_sided_p"]:.4g}，未作多重比较校正。</p>')
    parts.append('<h2>此前未赋权 GPT-5.4 参照</h2><p>此前全量为 227/371、多路径为 235/371，来自独立历史运行。API 不保证确定性；不能把跨轮差异全部归因于 z-score。</p><pre>'+html.escape(json.dumps(summary['prior_unweighted_gpt54_comparisons'],ensure_ascii=False,indent=2))+'</pre>')
    parts.append(f'<p>本轮 {v["logical_records"]} 条回答记录，{v["fresh_requests"]} 次新 API 调用、{v["shared_records"]} 条同轮相同输入共享；无历史答案复用。格式错误 {v["format_errors"]}，空答 {v["empty_responses"]}，截断 {v["token_limit_records"]}。</p>')
    parts.append('<p>固定 gpt-5.4-2026-03-05，官方端点，reasoning_effort=none、temperature=0、max_completion_tokens=16，不发送 seed。题目原图按原字节传入；不传历史图片或检索扩展陈述。固定选项排列的 MCQ accuracy，非四次旋转 EM。</p>')
    parts.append('<h2>分 domain</h2><table><tr><th>Domain</th><th>题数</th><th>全量＋z-score</th><th>多路径＋z-score</th></tr>')
    for domain,values in summary['by_domain'].items():
        parts.append(f'<tr><td>{html.escape(domain)}</td><td>{values["questions"]}</td>'+''.join(f'<td>{values["conditions"][n]["correct"]}</td>' for n in CONDITIONS)+'</tr>')
    parts.append('</table><h2>五项方法标注</h2>')
    for name,a in summary['method_annotations'].items():parts.append('<details><summary>'+LABELS[name]+'</summary><pre>'+html.escape(json.dumps(a,ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('<p>历史上游脚本变动记录于 input_audit.json；冻结检索结果、逐题原始证据、来源协议签名和输入哈希经过核对，旧协议未改写。</p><p><a href="summary.json">完整统计</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="independent_audit.json">独立核验</a> · <a href="protocol.json">协议</a> · <a href="https://developers.openai.com/api/docs/models/gpt-5.4">OpenAI 模型文档</a></p></html>')
    (output/'report.html').write_text('\n'.join(parts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','smoke','run','analyze','all'),default='all')
    p.add_argument('--output',type=Path,default=BASE);p.add_argument('--api-env-file',type=Path,default=runtime.DEFAULT_ENV)
    p.add_argument('--api-workers',type=int,default=3);args=p.parse_args()
    if args.stage in ('prepare','all'):prepare(args.output,args.api_env_file,args.api_workers)
    from scripts.analysis.audit_memeye_gpt54_zscore import audit
    if args.stage in ('prepare','smoke','run','all'):audit(args.output,inputs_only=True)
    if args.stage in ('smoke','all'):runtime.run(args.output,smoke=True)
    if args.stage in ('run','all'):runtime.run(args.output)
    if args.stage in ('analyze','all'):
        analyze(args.output)
        audit(args.output)


if __name__=='__main__':main()
