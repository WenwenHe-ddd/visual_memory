"""Seed 1042 rerun on the exact frozen inputs of both seed 42 baselines."""
import argparse
import copy
import csv
import html
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis import rerun_memeye_gpt54_seeded_baselines as previous
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.evaluation import openai_gpt54_seeded_transport as transport

SOURCE=previous.SOURCE
REFERENCE=previous.OUTPUT
OUTPUT=ROOT/'results/memeye_gpt54_seed1042_baselines_20261024'
SEED=1042
read,lines,sha,digest,dump=api.read,api.lines,api.sha,api.digest,api.dump
indexed,write_lines=api.indexed,api.write_lines
run,verify,request_body=previous.run,previous.verify,previous.request_body


def prepare(output=OUTPUT,seed=SEED):
    output=Path(output).resolve()
    assert seed==SEED,'This version is frozen to seed 1042'
    if (output/'protocol.json').exists():
        p=verify(output);assert p['contract']['seed']==seed
        return p
    assert not (output/'fresh_responses.jsonl').exists()
    old=verify(REFERENCE)
    assert old['contract']['seed']==42
    audit=read(REFERENCE/'independent_audit.json')
    assert audit['passed'] and audit['correct']=={'baseline_question_only':233,'master_answer_query':231}
    for name,expected in audit['artifact_sha256'].items():
        assert sha(REFERENCE/name)==expected,name
    records=lines(REFERENCE/'inputs/requests.jsonl')
    gold=lines(REFERENCE/'inputs/gold.jsonl')
    assert len(records)==742 and len(gold)==371
    output.mkdir(parents=True,exist_ok=True);(output/'inputs').mkdir(exist_ok=True)
    write_lines(output/'inputs/requests.jsonl',records)
    write_lines(output/'inputs/gold.jsonl',gold)
    proof=read(REFERENCE/'source_validation.json')
    dump(output/'source_validation.json',proof)
    c=copy.deepcopy(old['contract'])
    c.update(seed=seed,changed_component='Only provider seed changes from 42 to 1042; all complete reader inputs and other generation parameters are identical.',
        method_annotations=previous.method_annotations(seed),seed42_reference=str(REFERENCE),
        seed_caveat='Seed 42 probes already demonstrated differing answers on identical requests. Seed 1042 does not guarantee reproducibility.',
        seed_support_evidence=str(REFERENCE/'seed_support_probe.json'),probe_calls_this_experiment=0)
    wire=[]
    old_wire=indexed(lines(REFERENCE/'api_request_manifest.jsonl'),'record_id')
    for r in records:
        before=request_body(old,r)
        after=request_body({'contract':c},r)
        assert before['seed']==42 and after['seed']==1042
        assert {k:v for k,v in before.items() if k!='seed'}=={k:v for k,v in after.items() if k!='seed'}
        assert digest(before)==old_wire[r['record_id']]['request_sha256']
        wire.append({'record_id':r['record_id'],'payload_sha256':r['payload_sha256'],
                     'request_sha256':digest(after),'query_image_count':len(r['query_images'])})
    c['request_manifest_sha256']=digest(wire)
    sources=[Path(__file__),Path(transport.__file__),output/'inputs/requests.jsonl',output/'inputs/gold.jsonl',
        output/'source_validation.json',*(REFERENCE/n for n in ['protocol.json','summary.json','paired_results.json',
            'responses.jsonl','fresh_responses.jsonl','independent_audit.json','api_request_manifest.jsonl'])]
    c['source_sha256'].update({str(p):sha(p) for p in sources})
    p={'run_id':str(uuid.uuid4()),'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'contract':c,'experiment_signature':digest(c)}
    write_lines(output/'api_request_manifest.jsonl',wire);dump(output/'protocol.json',p)
    dump(output/'input_audit.json',{'passed':True,'seed':seed,'questions':371,'records':742,
        'unique_requests':720,'all_inputs_equal_to_seed42':True,'wire_only_difference_from_seed42':'seed',
        'requests_individually_checked':742,'gold_fields_absent_from_wire':True,
        'caption_memory_and_retrieval_frozen':True,'historical_or_probe_answer_reuse':False})
    print(api.json.dumps({'stage':'prepared','seed':seed,'records':742,'unique_requests':720}),flush=True)
    return p


def analyze(output=OUTPUT):
    output=Path(output);p=verify(output)
    validation=read(output/'validation_output.json')
    records=indexed(lines(output/'inputs/requests.jsonl'),'record_id')
    responses=indexed(lines(output/'responses.jsonl'),'record_id')
    fresh=indexed(lines(output/'fresh_responses.jsonl'),'payload_sha256')
    wire=indexed(lines(output/'api_request_manifest.jsonl'),'record_id')
    gold=indexed(lines(output/'inputs/gold.jsonl'),'id')
    assert set(records)==set(responses) and len(responses)==742 and len(fresh)==720
    for name in ['responses','fresh_responses']:
        assert sha(output/(name+'.jsonl'))==validation[name+'_sha256']
    old42=indexed(read(REFERENCE/'paired_results.json'),'question_id')
    old0=indexed(lines(SOURCE/'paired_results.jsonl'),'question_id')
    for rid,r in responses.items():
        q=records[rid];f=fresh[q['payload_sha256']]
        assert r['request_sha256']==wire[rid]['request_sha256']==digest(request_body(p,q))
        assert r['run_id']==p['run_id'] and r['experiment_signature']==p['experiment_signature']
        assert r['returned_model']==api.MODEL and r['historical_response_reuse'] is False
        assert r['predicted_choice']==api.parse_choice(r['raw_response'])
        assert r['top_memory_ids']==q['top_memory_ids']
        assert r['fresh_inference_executed_for_this_record']==(rid==f['record_id'])
        assert r['same_run_shared_inference_record_id']==(None if rid==f['record_id'] else f['record_id'])
        for key in ['response_id','raw_response','request_sha256','usage','execution_id']:
            assert r[key]==f[key]
    pairs=[];long_rows=[]
    for qid,g in gold.items():
        pair={'question_id':qid,'domain':qid.rsplit('/',1)[0],'gold_choice':g['gold_choice']}
        for condition,prefix in zip(previous.CONDITIONS,['baseline','expanded']):
            r=responses[qid+'::'+condition];pred=r['predicted_choice'];correct=pred==g['gold_choice']
            pair.update({prefix+'_answer':pred,prefix+'_correct':correct,
                'seed42_'+prefix+'_answer':old42[qid][prefix+'_answer'],
                'seed42_'+prefix+'_correct':old42[qid][prefix+'_correct'],
                'unseeded_'+prefix+'_answer':old0[qid][prefix+'_answer'],
                'unseeded_'+prefix+'_correct':old0[qid][prefix+'_correct']})
            long_rows.append({'question_id':qid,'condition':condition,'gold_choice':g['gold_choice'],
                'unseeded_answer':old0[qid][prefix+'_answer'],'seed42_answer':old42[qid][prefix+'_answer'],
                'seed1042_answer':pred,'seed42_correct':old42[qid][prefix+'_correct'],'seed1042_correct':correct,
                'answer_changed_vs_seed42':pred!=old42[qid][prefix+'_answer']})
        pairs.append(pair)
    comparisons={}
    for label in ['seed42','unseeded']:
        comparisons[label]={}
        for condition,prefix in zip(previous.CONDITIONS,['baseline','expanded']):
            x=api.aggregate([{'baseline_correct':r[label+'_'+prefix+'_correct'],
                              'expanded_correct':r[prefix+'_correct']} for r in pairs])
            comparisons[label][condition]={'reference_correct':x['baseline_correct'],
                'seed1042_correct':x['expanded_correct'],'wrong_to_right':x['wrong_to_correct'],
                'right_to_wrong':x['correct_to_wrong'],'net_correct':x['expanded_correct']-x['baseline_correct'],
                'mcnemar_exact_p':x['mcnemar_exact_p'],
                'answer_changed':sum(r[prefix+'_answer']!=r[label+'_'+prefix+'_answer'] for r in pairs)}
    annotations=previous.method_annotations(SEED)
    for condition in previous.CONDITIONS:
        annotations[condition]['5. 回答方式']['correct']=comparisons['seed42'][condition]['seed1042_correct']
    summary={'status':'complete','model':api.MODEL,'caption_model':api.MODEL,'seed':SEED,
        'overall':api.aggregate(pairs),'by_domain':{d:api.aggregate([r for r in pairs if r['domain']==d])
            for d in sorted({r['domain'] for r in pairs})},'comparisons':comparisons,
        'method_annotations':annotations,'validation':validation,
        'retrieval_metrics':read(REFERENCE/'summary.json')['retrieval_metrics'],
        'usage_scored_fresh_calls_only':{k:sum((r.get('usage') or {}).get(k,0) for r in fresh.values())
            for k in ['prompt_tokens','completion_tokens','total_tokens']},
        'seed_caveat':p['contract']['seed_caveat'],
        'interpretation':'One run per seed; differences mix seed choice with cross-call variability. No best-seed selection claim.'}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',pairs)
    for name,rows in [('paired_results.csv',pairs),('seed_comparison.csv',long_rows)]:
        with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    parts=['<!doctype html><meta charset="utf-8"><title>MemEye seed=1042 基线复测</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:35px auto}td,th{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
        '<h1>GPT-5.4：seed=42 与 seed=1042</h1>',
        '<p>371 道 MemEye 题，两套未加权基线。GPT caption、memory、完整排名、Top-5、原图、提示词和其他回答参数全部冻结，只将回答 seed 从 42 改为 1042。</p>',
        '<table><tr><th>方案</th><th>历史未传 seed</th><th>seed=42</th><th>seed=1042</th><th>相对 seed=42：错→对 / 对→错</th></tr>']
    for condition in previous.CONDITIONS:
        c=comparisons['seed42'][condition];raw=comparisons['unseeded'][condition]
        parts.append(f'<tr><td>{condition}</td><td>{raw["reference_correct"]}/371</td><td>{c["reference_correct"]}/371</td><td>{c["seed1042_correct"]}/371 ({c["seed1042_correct"]/371:.2%})</td><td>{c["wrong_to_right"]} / {c["right_to_wrong"]}</td></tr>')
    parts += ['</table><p>720 次本轮新回答支撑 742 条逻辑记录，22 条同轮相同请求共享。没有复用历史 seed=42、未传 seed 或探测答案。每个 seed 只有一轮，分差不能完全归因于 seed，不能据此挑选最优 seed。</p>',
        '<p><a href="seed_comparison.csv">逐题 seed 对照</a> · <a href="summary.json">完整统计</a> · <a href="protocol.json">协议</a></p>',
        '<h2>五项方法标注</h2><pre>'+html.escape(api.json.dumps(annotations,ensure_ascii=False,indent=2))+'</pre>']
    (output/'report.html').write_text('\n'.join(parts)+'\n')
    dump(output/'execution_status.json',{'status':'complete','seed':SEED,'overall':summary['overall']})
    print(api.json.dumps({'stage':'complete','seed':SEED,'overall':summary['overall'],'validation':validation}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','run','analyze','all'],default='all')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();assert args.seed==SEED
    if args.stage in ['prepare','all']:prepare(args.output,args.seed)
    if args.stage in ['run','all']:
        assert verify(args.output)['contract']['seed']==SEED
        run(args.output,smoke=args.smoke)
    if args.stage in ['analyze','all'] and not args.smoke:analyze(args.output)


if __name__=='__main__':
    main()
