"""Frozen observational PCA diagnosis; no reader calls or memory mutations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
for k,v in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MPLCONFIGDIR':'/tmp/visualmemory-matplotlib'}.items():os.environ.setdefault(k,v)
import numpy as np
from scripts.evaluation.memeye_gpt5_reader import read,lines,sha,digest,dump,write_lines,indexed
from scripts.memory import memeye_backbone_retrieval as original
from scripts.memory.memeye_backbone_long_text import encode_complete
from scripts.memory.ridge_whitening import fit
from scripts.memory.online_zscore import ZScoreWeighting
from scripts.memory.online_positive_variance import PositiveVarianceWeighting
SOURCE=ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
PVROOT=ROOT/'results/memeye_api_pv_frozen_l0p03_a1_20261108/gpt5'
ZROOT=ROOT/'results/memeye_gpt5_qwen_five_validation_20261024'
OUT=ROOT/'reports/gpt5_variance_observation_20261108'
RAW='master_answer_query';Z=RAW+'__l0p03_a1';PV=RAW+'+positive_variance'


def table(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def aggregate(rows):
    if not rows:return {'pairs':0,'questions':0}
    byq=defaultdict(list)
    for r in rows:byq[r['question_id']].append(r)
    result={'pairs':len(rows),'questions':len(byq),'masters':len({(r['domain'],r['master_id']) for r in rows}),
        'raw_target_ahead_pairs':sum(r['raw_margin']>1e-12 for r in rows),
        'white_target_ahead_pairs':sum(r['white_margin']>1e-12 for r in rows),
        'pv_target_ahead_pairs':sum(r['pv_margin']>1e-12 for r in rows),
        'zscore_final_target_ahead_pairs':sum(r['zscore_final_margin']>1e-12 for r in rows),
        'pv_final_target_ahead_pairs':sum(r['pv_final_margin']>1e-12 for r in rows),
        'white_rescues_raw_wrong_pairs':sum(r['raw_margin']<=0<r['white_margin'] for r in rows),
        'white_harms_raw_right_pairs':sum(r['white_margin']<=0<r['raw_margin'] for r in rows),
        'white_margin_improved_pairs':sum(r['white_margin']-r['raw_margin']>1e-12 for r in rows),
        'mean_white_minus_raw_margin_question_macro':float(np.mean([np.mean([r['white_margin']-r['raw_margin'] for r in group]) for group in byq.values()]))}
    for name in ['raw','white','pv','zscore_final','pv_final']:
        result[name+'_target_ahead_fraction_question_macro']=float(np.mean([np.mean([r[name+'_margin']>1e-12 for r in group]) for group in byq.values()]))
    bandrows=[r for r in rows if r['high_contribution'] is not None]
    bq=defaultdict(list)
    for r in bandrows:bq[r['question_id']].append(r)
    result['band_eligible_pairs']=len(bandrows);result['band_eligible_questions']=len(bq)
    for band in ['high','middle','low','rest']:
        result[band]={'supports_target_pairs':sum(r[band+'_contribution']>1e-12 for r in bandrows),
            'supports_distractor_pairs':sum(r[band+'_contribution']<-1e-12 for r in bandrows),
            'supports_target_fraction_question_macro':float(np.mean([np.mean([r[band+'_contribution']>1e-12 for r in group]) for group in bq.values()])) if bq else None,
            'signed_mean_question_macro':float(np.mean([np.mean([r[band+'_contribution'] for r in group]) for group in bq.values()])) if bq else None,
            'absolute_mean_question_macro':float(np.mean([np.mean([abs(r[band+'_contribution']) for r in group]) for group in bq.values()])) if bq else None}
    for key,pred in [('high_distractor_rest_target',lambda r:r['high_contribution']<-1e-12 and r['rest_contribution']>1e-12),
                     ('high_target_rest_distractor',lambda r:r['high_contribution']>1e-12 and r['rest_contribution']<-1e-12)]:
        result[key+'_pairs']=sum(pred(r) for r in bandrows)
        result[key+'_fraction_question_macro']=float(np.mean([np.mean([pred(r) for r in group]) for group in bq.values()])) if bq else None
    return result


def run(out,device):
    plan=read(out/'observation_plan.json');assert plan['lambda']==.03 and plan['alpha']==1
    prior=read(PVROOT/'retrieval_protocol.json');sources={**prior['source_sha256'],**prior['output_sha256']}
    for p,h in sources.items():assert sha(p)==h,p
    for p in [out/'observation_plan.json',PVROOT/'retrieval_protocol.json',ZROOT/'retrieval_results.jsonl',Path(__file__),
              ROOT/'scripts/memory/online_zscore.py',ROOT/'scripts/memory/online_positive_variance.py']:
        sources[str(p)]=sha(p)
    data,questions,model=original.attach_captions(lines(SOURCE/'inputs/source_memories.jsonl'),
        lines(SOURCE/'inputs/source_questions.jsonl'),lines(SOURCE/'captions.jsonl'))
    assert model=='gpt-5-2025-08-07'
    old=indexed(lines(SOURCE/'retrieval_results.jsonl'),'question_id')
    pvrows=indexed(lines(PVROOT/'retrieval_results.jsonl'),'question_id')
    zrows=indexed(lines(ZROOT/'retrieval_results.jsonl'),'question_id')
    gold=indexed(lines(SOURCE/'inputs/evaluation.jsonl'),'id')
    inp=read(SOURCE/'inputs/protocol.json');args=SimpleNamespace(device=device);original.configure_device(args)
    encoder=original.load_text_encoder(inp['text_encoder'],args.device)
    allpairs=[];largestpairs=[];pcrows=[];spectra=[];members=[];encoding=[];largestcoords=[]
    max_raw_error=max_margin_error=max_inverse_error=0.;checked_groups=0;counts=[]
    selected=plan['selected']
    for domain in sorted({m['task'] for m in data}):
        ms=[m for m in data if m['task']==domain];qs=[q for q in questions if q['task']==domain]
        graph=read(SOURCE/'domains'/domain/'memory/graph.json')
        nodes=indexed(graph['image_nodes'],'id');mi={m['image_id']:i for i,m in enumerate(ms)}
        for g in graph['masters']:counts.append({'domain':domain,'master_id':g['id'],'size':len(g['member_ids'])})
        e,ei=encode_complete(encoder,[m['retrieval_text'] for m in ms],ids=[m['id'] for m in ms],field='dialogue+caption')
        texts=[];ids=[]
        for q in qs:
            texts.extend([q['retrieval_query']]+[q['retrieval_query']+'\nCandidate answer: '+q['candidate_statements'][k] for k in 'ABCD'])
            ids.extend(q['id']+s for s in ['/baseline','/A','/B','/C','/D'])
        qv,qi=encode_complete(encoder,texts,inp['query_prefix'],ids=ids,field='query');qv=qv.reshape(len(qs),5,-1)
        fits={}
        for group in graph['masters']:
            indices=[mi[mid] for mid in group['member_ids']];x=e[indices]
            u,v=fit(x);r=int((v>0).sum())
            u=u.copy()
            for comp in u:
                if comp[np.argmax(abs(comp))]<0:comp*=-1
            coords=(x-x.mean(0))@u.T
            fits[group['id']]=(indices,u,v,r,coords)
            isselected=domain==selected['domain'] and group['id']==selected['master_id']
            width=math.ceil(.2*r) if r>=5 else None
            for j,value in enumerate(v):
                band='excluded_low_rank' if r<5 else 'high' if j<width else 'low' if j>=r-width and j<r else 'middle' if j<r else 'null'
                sr={'domain':domain,'master_id':group['id'],'size':len(x),'rank':r,'pc':j+1,'band':band,'variance':float(value),
                    'variance_share':float(value/v.sum()) if v.sum() else 0.,'white_retention':float(.03/(value+.03)),
                    'pv_multiplier':float(1+value/(value+.03))}
                spectra.append(sr)
            if isselected:
                assert len(indices)==48
                members=[{**ms[i],'master_id':group['id']} for i in indices]
                for ix,m in enumerate(members):
                    largestcoords.append({'memory_id':m['id'],'coordinates':[float(c) for c in coords[ix]]})
        for iq,q in enumerate(qs):
            qid=q['id'];targets=set(gold[qid]['gold_ids']);before=old[qid]
            zl=indexed(next(c for c in zrows[qid]['conditions'] if c['name']==Z)['leaves'],'memory_id')
            pl=indexed(pvrows[qid]['leaves'],'memory_id')
            original_top5=next(c for c in before['conditions'] if c['name']==RAW)['top5_memory_ids']
            rawrank={mid:i+1 for i,mid in enumerate(next(c for c in before['conditions'] if c['name']==RAW)['full_rank_memory_ids'])}
            for group in before['masters']:
                inds,u,v,r,coords=fits[group['master_id']];x=e[inds]
                query=qv[iq,1+'ABCD'.index(group['selected_choice'])]
                z=ZScoreWeighting(.03,1).apply(query,x);p=PositiveVarianceWeighting(.03,1).apply(query,x)
                for field,calc,sourcefield,stored in [('raw',z['raw'],'raw',zl),('white',z['white'],'white',zl),
                    ('zscore_final',z['weighted'],'weighted',zl),('pv',p['positive_variance'],'positive_variance',pl),('pv_final',p['weighted'],'weighted',pl)]:
                    expected=np.array([stored[ms[i]['id']][sourcefield] for i in inds]);err=float(abs(expected-calc).max());assert err<2e-6
                    max_raw_error=max(max_raw_error,err)
                centered=x-x.mean(0)
                if len(x)>1:
                    dq=centered.T@np.linalg.solve(centered@centered.T+(len(x)-1)*.03*np.eye(len(x)),centered@query)
                    err=float(abs(z['white']-x@(query-dq)).max());assert err<2e-12;max_inverse_error=max(max_inverse_error,err)
                checked_groups+=1
                ti=[j for j,i in enumerate(inds) if ms[i]['id'] in targets]
                di=[j for j,i in enumerate(inds) if ms[i]['id'] not in targets]
                if not ti or not di:continue
                isselected=domain==selected['domain'] and group['master_id']==selected['master_id']
                qp=u@query;cs=[]
                for t in ti:
                    for d in di:
                        c=qp*(u@(x[t]-x[d]));cs.append(c)
                        margin=float(query@(x[t]-x[d]));err=abs(c.sum()-margin);assert err<2e-12;max_margin_error=max(max_margin_error,float(err))
                        np.testing.assert_allclose((c*(.03/(v+.03))).sum(),z['white'][t]-z['white'][d],atol=2e-12,rtol=0)
                        np.testing.assert_allclose((c*(1+v/(v+.03))).sum(),p['positive_variance'][t]-p['positive_variance'][d],atol=2e-12,rtol=0)
                        width=math.ceil(.2*r) if r>=5 else None
                        bands={band+'_contribution':None for band in ['high','middle','low','rest']}
                        if width:
                            bands={'high_contribution':float(c[:width].sum()),'middle_contribution':float(c[width:r-width].sum()),
                                   'low_contribution':float(c[r-width:r].sum()),'rest_contribution':float(c[width:r].sum())}
                        row={'question_id':qid,'domain':domain,'master_id':group['master_id'],'master_size':len(x),'rank':r,
                             'target_id':ms[inds[t]]['id'],'distractor_id':ms[inds[d]]['id'],
                             'target_raw_global_rank':rawrank[ms[inds[t]]['id']],'distractor_raw_global_rank':rawrank[ms[inds[d]]['id']],
                             'distractor_in_original_top5':ms[inds[d]]['id'] in original_top5,
                             'raw_margin':margin,'white_margin':float(z['white'][t]-z['white'][d]),
                             'pv_margin':float(p['positive_variance'][t]-p['positive_variance'][d]),
                             'zscore_final_margin':float(z['weighted'][t]-z['weighted'][d]),
                             'pv_final_margin':float(p['weighted'][t]-p['weighted'][d]),**bands}
                        allpairs.append(row)
                        if isselected:largestpairs.append({**row,'contributions':[float(v0) for v0 in c]})
                cs=np.asarray(cs)
                for j,value in enumerate(v):
                    pcrows.append({'question_id':qid,'domain':domain,'master_id':group['master_id'],'pc':j+1,'variance':float(value),
                        'pair_count':len(cs),'signed_mean_contribution':float(cs[:,j].mean()),'mean_absolute_contribution':float(abs(cs[:,j]).mean()),
                        'target_support_fraction':float((cs[:,j]>1e-12).mean()),'distractor_support_fraction':float((cs[:,j]<-1e-12).mean())})
        encoding.append({'domain':domain,'leaf':ei,'query':qi})
        print(json.dumps({'stage':'domain_complete','domain':domain,'pairs':len(allpairs),'groups_verified':checked_groups}),flush=True)
    counts.sort(key=lambda x:(-x['size'],x['domain'],x['master_id']))
    assert counts[0]=={'domain':selected['domain'],'master_id':selected['master_id'],'size':selected['members']}
    assert len(counts)==72 and sum(c['size'] for c in counts)==438
    subset=[{k:v for k,v in row.items() if k!='contributions'} for row in largestpairs]
    metrics={'all_pairs':aggregate(allpairs),'largest_master':aggregate(subset),
        'all_original_top5_distractors':aggregate([r for r in allpairs if r['distractor_in_original_top5']]),
        'largest_original_top5_distractors':aggregate([r for r in subset if r['distractor_in_original_top5']])}
    sortedexamples=sorted(largestpairs,key=lambda r:(r['white_margin']-r['raw_margin'],r['question_id'],r['target_id'],r['distractor_id']))
    examples={'max_whitening_harm':sortedexamples[0],'max_whitening_help':sortedexamples[-1]}
    for k,row in examples.items():
        q=next(q for q in questions if q['id']==row['question_id']);g=next(g for g in old[q['id']]['masters'] if g['master_id']==selected['master_id'])
        row['question']=q['question'];row['master_query']=g['augmented_query'];row['gold_choice']=gold[q['id']]['gold_choice'];row['options']=q['options']
    table(out/'pairs.csv',allpairs);table(out/'pc_by_question.csv',pcrows);table(out/'spectra.csv',spectra);table(out/'master_sizes.csv',counts)
    write_lines(out/'largest_pairs_with_contributions.jsonl',largestpairs)
    dump(out/'largest_members.json',members);dump(out/'largest_coordinates.json',largestcoords);dump(out/'examples.json',examples)
    p={'status':'numeric_complete','plan':plan,'metrics':metrics,'selected_master':selected,'master_counts':len(counts),
       'verified_query_master_groups':checked_groups,'max_original_score_error':max_raw_error,
       'max_pair_margin_decomposition_error':max_margin_error,'max_inverse_solve_error':max_inverse_error,
       'encoding_audits':encoding,'api_calls':0,'vectors_written_to_memory':False,'memory_modified':False,
       'source_sha256':sources,'output_sha256':{str(p):sha(p) for p in out.iterdir() if p.is_file()}}
    for path,h in sources.items():assert sha(path)==h,path
    dump(out/'numeric_summary.json',p)
    print(json.dumps({'selected':selected,'metrics':metrics,'examples':{k:{f:v[f] for f in ['question_id','target_id','distractor_id','raw_margin','white_margin','pv_margin']} for k,v in examples.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(a.output.resolve(),a.device)
