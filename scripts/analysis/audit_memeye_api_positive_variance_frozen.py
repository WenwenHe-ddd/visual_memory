"""Independent frozen-input, score-sign, wire-byte and result audit."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.audit_memeye_gpt5_base import wire as gpt_wire
from scripts.analysis.deepseek_seed42_audit_helpers import wire_request as ds_wire
DEFAULT=ROOT/'results/memeye_api_pv_frozen_l0p03_a1_20261108'
RAW='master_answer_query'; PV=RAW+'+positive_variance'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def index(rows,key):
    d={r[key]:r for r in rows};assert len(d)==len(rows);return d
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def choice(s):
    m=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',(s or '').strip(),re.I)
    return m.group(1).upper() if m else None
def std(values):
    values=np.asarray(values);s=values.std()
    if len(values)==1:return values.copy()
    if s<=32*np.finfo(float).eps*max(1.,abs(values).max()):return np.zeros(len(values))
    if len(values)==2:return np.array([1.,-1.])*np.sign(values[0]-values[1])
    return (values-values.mean())/s


def audit(out,inputs_only=False):
    p=read(out/'reader/protocol.json');c=p['contract'];assert digest(c)==p['experiment_signature']
    assert (c['lambda'],c['alpha'],c['seed'],c['conditions'])==(.03,1.,42,[RAW,PV])
    for path,h in c['source_sha256'].items():assert sha(path)==h,path
    source=Path(c['memory_path']);ref=Path(c['control_reference']);name=c['name']
    rt=read(out/'retrieval_protocol.json')
    assert rt['svd_term_sign']=='+' and not rt['memory_modified'] and not rt['vectors_persisted']
    assert rt['raw_full_rankings_verified']==371 and rt['max_dual_solve_error']<2e-12
    old=index(lines(source/'retrieval_results.jsonl'),'question_id')
    qmap=index(lines(source/'inputs/source_questions.jsonl'),'id')
    memory={m['id']:m['retrieval_text'] for m in lines(source/'offline/memories.jsonl')}
    gold=index(lines(out/'reader/gold.jsonl'),'id');assert set(gold)==set(qmap)==set(old) and len(gold)==371
    assert sum(len(g['gold_ids']) for g in gold.values())==1041
    system=read(source/'inputs/protocol.json')['reader_system_prompt']
    whitepath=ref/'retrieval_results.jsonl' if name=='gpt5' else source/'weighted_evaluation/retrieval_results.jsonl'
    whiteprotocol=read(whitepath.parent/'retrieval_protocol.json')
    assert whiteprotocol['output_sha256'][str(whitepath)]==sha(whitepath)
    whites=index(lines(whitepath),'question_id')
    ret=index(lines(out/'retrieval_results.jsonl'),'question_id');assert set(ret)==set(gold)
    groupcount=leafcount=0;maxsignerror=0.
    for qid,row in ret.items():
        oldrow=old[qid];assert row['task']==qmap[qid]['task']
        conditions=index(row['conditions'],'name');raw=next(x for x in oldrow['conditions'] if x['name']==RAW)
        assert all(conditions[RAW][k]==raw[k] for k in ['top5_memory_ids','top5_scores','full_rank_memory_ids'])
        wh=next(x for x in whites[qid]['conditions'] if x['name']==('master_answer_query__l0p03_a1' if name=='gpt5' else 'master_answer_query_zscore'))
        wm=index(wh['leaves'],'memory_id');om=index(oldrow['masters'],'master_id')
        assert len(row['leaves'])==len(wm)==len(oldrow['leaves'])
        for group in row['groups']:
            prior=om[group['master_id']]
            for k in ['member_ids','selected_choice','selected_statement','augmented_query']:
                assert group[k]==prior[k]
            ls=[r for r in row['leaves'] if r['group_id']==group['master_id']]
            rawv=np.array([r['raw'] for r in ls]);pv=np.array([r['positive_variance'] for r in ls])
            # Independent historical minus-sign result yields the plus result.
            expected=np.array([2*wm[r['memory_id']]['raw']-wm[r['memory_id']]['white'] for r in ls])
            err=float(abs(pv-expected).max());assert err<2e-6;maxsignerror=max(maxsignerror,err)
            np.testing.assert_allclose(rawv,[wm[r['memory_id']]['raw'] for r in ls],atol=2e-6,rtol=0)
            z=std(pv);correction=np.zeros(len(ls))
            if len(ls)>1 and np.any(std(rawv)) and np.any(z):correction=rawv.std()*np.tanh(z)
            for field,values in [('z',z),('correction',correction),('weighted',rawv+correction)]:
                np.testing.assert_allclose([r[field] for r in ls],values,atol=2e-12,rtol=0)
            groupcount+=1;leafcount+=len(ls)
        ranked=sorted(row['leaves'],key=lambda x:-x['weighted'])
        assert conditions[PV]['full_rank_memory_ids']==[r['memory_id'] for r in ranked]
        assert conditions[PV]['top5_memory_ids']==[r['memory_id'] for r in ranked[:5]]
        assert conditions[PV]['top5_scores']==[r['weighted'] for r in ranked[:5]]
    records=index(lines(out/'reader/requests.jsonl'),'record_id');assert len(records)==742
    for row in records.values():
        q=qmap[row['question_id']];ids=row['top_memory_ids']
        condition=next(x for x in ret[q['id']]['conditions'] if x['name']==row['condition'])
        assert ids==condition['top5_memory_ids'] and len(set(ids))==len(ids)==5
        assert all(mid.split('/')[0]==q['task'] for mid in ids)
        context='\n\n'.join('Memory '+str(i)+': '+memory[mid] for i,mid in enumerate(ids,1))
        text=('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'+context+
              '\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+
              '\n\nAnswer with one letter only.')
        payload={'system':system,'user_text':text,'query_images':q['query_images']}
        assert payload=={k:row[k] for k in payload}
        body=gpt_wire(payload,2048,42) if name=='gpt5' else ds_wire(payload,c['model'],16,'high')
        assert digest(payload)==row['payload_sha256'] and digest(body)==row['request_sha256']
    oldans=index(lines(ref/'reader/responses.jsonl'),'record_id')
    reused=index(lines(out/'reader/reused_answers.jsonl'),'record_id')
    pending=index(lines(out/'reader/pending_requests.jsonl'),'request_sha256')
    assert len(reused)==c['reused_records'] and len(pending)==c['new_unique_requests']
    for rid,r in reused.items():
        assert r['source_path']==str(ref/'reader/responses.jsonl')
        assert r['answer']==oldans[r['source_record_id']]
        assert r['request_sha256']==records[rid]['request_sha256']==r['answer']['request_sha256']
        if records[rid]['condition']==RAW:assert r['source_record_id']==rid
    assert all(r['condition']==PV and r['record_id'] not in reused for r in pending.values())
    for rid,r in records.items():
        assert rid in reused or r['request_sha256'] in pending
    report={'passed':True,'inputs_only':inputs_only,'source_hashes_verified':len(c['source_sha256']),
        'reader_requests_independently_rebuilt':742,'raw_rankings_preserved':371,'master_groups_checked':groupcount,
        'leaf_scores_checked':leafcount,'max_error_vs_2raw_minus_historical_white':maxsignerror,
        'old_control_answers_preserved':371,'reused_records':len(reused),'planned_new_requests':len(pending),
        'independent_wire_reconstruction':True,'control_reference':str(ref),
        'audit_source_sha256':{str(Path(__file__)):sha(__file__),str(whitepath):sha(whitepath)}}
    if inputs_only:
        dump(out/'preflight_audit.json',report);return report
    outcomes=index(lines(out/'reader/outcomes.jsonl'),'request_sha256');assert set(outcomes)==set(pending)
    responses=index(lines(out/'reader/responses.jsonl'),'record_id');assert set(responses)==set(records)
    for h,a in outcomes.items():
        assert a['run_id']==p['run_id'] and a['experiment_signature']==p['experiment_signature'] and a['seed']==42
        assert a['record_id']==pending[h]['record_id'] and a['payload_sha256']==pending[h]['payload_sha256']
        if a['status']=='provider_input_rejected':
            assert a['attempts']==3 and a['raw_response'] is None and a['predicted_choice'] is None
        else:
            assert a['returned_model']==c['expected_returned_model'] and a['requested_model']==c['model']
            assert a['response_id']
            valid=choice(a['raw_response']) is not None and a['finish_reason']=='stop' and not a.get('refusal')
            assert (a['status']=='ok')==valid
            assert a['predicted_choice']==(choice(a['raw_response']) if valid else None)
    for rid,r in responses.items():
        a=reused[rid]['answer'] if rid in reused else outcomes[r['request_sha256']]
        for k,v in records[rid].items():assert r[k]==v
        for k in ['raw_response','predicted_choice','status','response_id']:assert r[k]==a[k]
        assert r['origin']==('historical' if rid in reused else 'new')
    summary=read(out/'summary.json');pairs=index(read(out/'paired_results.json'),'question_id')
    for method in [RAW,PV]:
        correct=hits=anygold=failed=reusedn=0;recalls=[]
        for qid,g in gold.items():
            a=responses[qid+'::'+method];pred=a['predicted_choice'];hit=len(set(g['gold_ids'])&set(a['top_memory_ids']))
            correct+=pred==g['gold_choice'];hits+=hit;anygold+=hit>0;recalls.append(hit/len(g['gold_ids']))
            failed+=a['status']!='ok';reusedn+=a['origin']=='historical'
            assert pairs[qid][method]['correct']==(pred==g['gold_choice'])
            assert pairs[qid][method]['prediction']==pred and pairs[qid][method]['top5']==a['top_memory_ids']
        expected={'correct':correct,'questions':371,'accuracy':correct/371,'gold_hits_top5':hits,'gold_references':1041,
                  'recall_at_5_micro':hits/1041,'recall_at_5_macro':float(np.mean(recalls)),
                  'questions_with_any_gold':anygold,'failed_records':failed,'reused_answers':reusedn}
        assert expected==summary['overall'][method]
    comparison={'wrong_to_right':sum(not r[RAW]['correct'] and r[PV]['correct'] for r in pairs.values()),
        'right_to_wrong':sum(r[RAW]['correct'] and not r[PV]['correct'] for r in pairs.values()),
        'changed_ordered_top5':sum(r[RAW]['top5']!=r[PV]['top5'] for r in pairs.values()),
        'answer_changes':sum(r[RAW]['prediction']!=r[PV]['prediction'] for r in pairs.values())}
    assert comparison==summary['comparison']
    assert summary['overall'][RAW]['correct']==c['expected_control_correct']
    for path,h in c['source_sha256'].items():assert sha(path)==h,path
    summary['status']='complete';summary['independent_audit_passed']=True;dump(out/'summary.json',summary)
    report.update(completed_new_requests=len(outcomes),new_outcome_counts=dict(Counter(a['status'] for a in outcomes.values())),
                  returned_models=dict(Counter(a.get('returned_model') for a in outcomes.values())),
                  overall=summary['overall'],comparison=comparison,all_371_questions_retained=True)
    report['artifact_sha256']={str(p):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['completion_manifest.json','independent_audit.json']}
    dump(out/'independent_audit.json',report)
    dump(out/'completion_manifest.json',{'status':'complete','completed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         'independent_audit_passed':True,'questions_per_condition':371,'conditions':[RAW,PV],
         'source_sha256':c['source_sha256'],'artifact_sha256':{**report['artifact_sha256'],str(out/'independent_audit.json'):sha(out/'independent_audit.json')}})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--model',choices=['gpt5','deepseek'],required=True);p.add_argument('--inputs-only',action='store_true')
    a=p.parse_args();r=audit(a.output.resolve()/a.model,a.inputs_only)
    print(json.dumps({k:v for k,v in r.items() if k not in ['artifact_sha256','audit_source_sha256']},ensure_ascii=False))


if __name__=='__main__':main()
