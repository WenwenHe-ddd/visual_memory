"""Read-only paired diagnosis of GPT-5 CVM, ridge z-score and positive variance."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ZROOT=ROOT/'results/memeye_gpt5_qwen_five_validation_20261024'
PROOT=ROOT/'results/memeye_api_pv_frozen_l0p03_a1_20261108/gpt5'
OUT=ROOT/'reports/gpt5_cvm_zscore_vs_positive_variance_20261108'
RAW='master_answer_query';Z=RAW+'__l0p03_a1';PV=RAW+'+positive_variance'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def index(rows,key):
    d={r[key]:r for r in rows};assert len(rows)==len(d);return d
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for root in [ZROOT,PROOT]:
        a=read(root/'independent_audit.json');assert a['passed']
        for name,h in a['artifact_sha256'].items():
            p=Path(name);p=p if p.is_absolute() else root/p
            assert sha(p)==h,p
    zg=index(read(ZROOT/'paired_results.json'),'question_id')
    pg=index(read(PROOT/'paired_results.json'),'question_id')
    zr=index(lines(ZROOT/'retrieval_results.jsonl'),'question_id')
    pr=index(lines(PROOT/'retrieval_results.jsonl'),'question_id')
    za=index(lines(ZROOT/'reader/responses.jsonl'),'record_id')
    pa=index(lines(PROOT/'reader/responses.jsonl'),'record_id')
    assert set(zg)==set(pg)==set(zr)==set(pr) and len(zg)==371
    rows=[];groups=[];overall={}
    for qid,g in pg.items():
        ids=set(g['gold_ids']);row={'question_id':qid,'domain':g['task'],'gold_ids':g['gold_ids'],'gold_choice':g['gold_choice']}
        zcs=index(zr[qid]['conditions'],'name');pcs=index(pr[qid]['conditions'],'name')
        for label,c,a,correct in [('raw',zcs[RAW],za[qid+'::'+RAW],zg[qid][RAW+'_correct']),
                                  ('zscore',zcs[Z],za[qid+'::'+Z],zg[qid][Z+'_correct']),
                                  ('pv',pcs[PV],pa[qid+'::'+PV],g[PV]['correct'])]:
            ranks={mid:i+1 for i,mid in enumerate(c['full_rank_memory_ids'])};top=c['top5_memory_ids']
            assert correct==(a['predicted_choice']==g['gold_choice'])
            row[label]={'top5':top,'correct':correct,'prediction':a['predicted_choice'],
                'request_sha256':a['request_sha256'],'hits':len(set(top)&ids),
                'top1_gold':top[0] in ids,'best_gold_rank':min(ranks[mid] for mid in ids),
                'gold_rank_sum':sum(ranks[mid] for mid in ids),'top5_positions_of_gold':[i+1 for i,mid in enumerate(top) if mid in ids]}
        assert pa[qid+'::'+RAW]['raw_response']==za[qid+'::'+RAW]['raw_response']
        zl=index(zcs[Z]['leaves'],'memory_id');pl=index(pr[qid]['leaves'],'memory_id')
        assert set(zl)==set(pl)
        for group in zcs[Z]['groups']:
            mids=[mid for mid,l in zl.items() if l['group_id']==group['group_id']]
            raw=np.array([zl[mid]['raw'] for mid in mids]);z=np.array([zl[mid]['weighted'] for mid in mids]);p=np.array([pl[mid]['weighted'] for mid in mids])
            np.testing.assert_allclose(raw,[pl[mid]['raw'] for mid in mids],rtol=0,atol=2e-12)
            non_tied=raw[:,None]-raw[None,:];mask=np.triu(abs(non_tied)>1e-12,k=1)
            n=int(mask.sum());entry={'question_id':qid,'group_id':group['group_id'],'size':len(mids),'non_tied_pairs':n}
            for name,values in [('zscore',z),('pv',p)]:
                diff=values[:,None]-values[None,:]
                entry[name+'_raw_rank_inversions']=int(((diff*non_tied<0)&mask).sum())
                entry[name+'_raw_correlation']=float(np.corrcoef(raw,values)[0,1]) if len(mids)>1 and raw.std()>1e-12 and values.std()>1e-12 else None
            groups.append(entry)
        rows.append(row)
    for name in ['raw','zscore','pv']:
        overall[name]={'correct':sum(r[name]['correct'] for r in rows),'questions':371,
            'hits':sum(r[name]['hits'] for r in rows),'gold_references':1041,
            'top1_gold_questions':sum(r[name]['top1_gold'] for r in rows),
            'any_gold_questions':sum(r[name]['hits']>0 for r in rows),
            'mean_best_gold_rank':sum(r[name]['best_gold_rank'] for r in rows)/371,
            'mean_gold_rank':sum(r[name]['gold_rank_sum'] for r in rows)/1041}
        assert overall[name]['correct']==(221 if name=='raw' else 222 if name=='zscore' else 211)
    comparisons={}
    for before,after in [('raw','zscore'),('raw','pv'),('zscore','pv')]:
        gains=[r for r in rows if not r[before]['correct'] and r[after]['correct']]
        losses=[r for r in rows if r[before]['correct'] and not r[after]['correct']]
        n=len(gains)+len(losses)
        v={'wrong_to_right':len(gains),'right_to_wrong':len(losses),
            'mcnemar_exact_p':min(1.,2*sum(math.comb(n,i) for i in range(min(len(gains),len(losses))+1))/2**n) if n else 1.,
            'changed_ordered_top5':sum(r[before]['top5']!=r[after]['top5'] for r in rows),
            'changed_top5_set':sum(set(r[before]['top5'])!=set(r[after]['top5']) for r in rows),
            'same_payload':sum(r[before]['request_sha256']==r[after]['request_sha256'] for r in rows),
            'same_payload_answer_changes':sum(r[before]['request_sha256']==r[after]['request_sha256'] and r[before]['prediction']!=r[after]['prediction'] for r in rows),
            'more_gold_questions':sum(r[after]['hits']>r[before]['hits'] for r in rows),
            'fewer_gold_questions':sum(r[after]['hits']<r[before]['hits'] for r in rows),
            'gold_added':sum(len(set(r[after]['top5'])&set(r['gold_ids'])-set(r[before]['top5'])) for r in rows),
            'gold_removed':sum(len(set(r[before]['top5'])&set(r['gold_ids'])-set(r[after]['top5'])) for r in rows)}
        for label,subset in [('gains',gains),('losses',losses)]:
            v[label]={'question_ids':[r['question_id'] for r in subset],
                'more_gold':sum(r[after]['hits']>r[before]['hits'] for r in subset),
                'same_gold_count':sum(r[after]['hits']==r[before]['hits'] for r in subset),
                'fewer_gold':sum(r[after]['hits']<r[before]['hits'] for r in subset),
                'same_gold_set':sum(set(r[after]['top5'])&set(r['gold_ids'])==set(r[before]['top5'])&set(r['gold_ids']) for r in subset),
                'same_top5_set':sum(set(r[after]['top5'])==set(r[before]['top5']) for r in subset),
                'same_gold_positions':sum(r[after]['top5_positions_of_gold']==r[before]['top5_positions_of_gold'] for r in subset),
                'zero_gold_after':sum(r[after]['hits']==0 for r in subset)}
        comparisons[before+'_to_'+after]=v
    geometry={'query_master_groups':len(groups),'non_tied_leaf_pairs':sum(g['non_tied_pairs'] for g in groups)}
    for method in ['zscore','pv']:
        vals=[g[method+'_raw_correlation'] for g in groups if g[method+'_raw_correlation'] is not None]
        geometry[method]={'within_master_pair_inversions_vs_raw':sum(g[method+'_raw_rank_inversions'] for g in groups),
                          'raw_rank_changed_groups':sum(g[method+'_raw_rank_inversions']>0 for g in groups),
                          'mean_raw_final_score_correlation':float(np.mean(vals)),'nonconstant_groups':len(vals)}
    paths=[ZROOT/'summary.json',ZROOT/'paired_results.json',ZROOT/'retrieval_results.jsonl',ZROOT/'reader/responses.jsonl',
           PROOT/'summary.json',PROOT/'paired_results.json',PROOT/'retrieval_results.jsonl',PROOT/'reader/responses.jsonl',Path(__file__)]
    summary={'lambda':.03,'alpha':1.,'source_sha256':{str(p):sha(p) for p in paths},'overall':overall,
             'comparisons':comparisons,'geometry':geometry,'api_calls':0,
             'interpretation_limits':'Descriptive paired observations on one evaluated question set; no semantic labels on PCA directions or repeated-run causal identification.'}
    dump(OUT/'summary.json',summary);dump(OUT/'per_question.json',rows);dump(OUT/'per_group.json',groups)
    flat=[{'question_id':r['question_id'],'domain':r['domain'],'gold_choice':r['gold_choice'],
           **{name+'_'+k:r[name][k] for name in ['raw','zscore','pv'] for k in ['correct','prediction','hits','top1_gold','best_gold_rank','top5']}} for r in rows]
    with (OUT/'per_question.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},ensure_ascii=False))


if __name__=='__main__':main()
