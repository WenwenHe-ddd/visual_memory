"""Caption-only retrieval and non-embedding lexical cue-sharing diagnostics."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/natural_observation'
CORE={'scene','person','object'}
GENERIC={'image','photo','photograph','picture','screenshot','question','information','detail','thing',
         'what','which','when','where','who','whom','whose','how'}
# Fixed before outcome analysis. Only simple inflection and a small synonym list.
IRREGULAR={'people':'person','persons':'person','men':'man','women':'woman','children':'child',
           'shelves':'shelf','knives':'knife','leaves':'leaf','feet':'foot','teeth':'tooth',
           'glasses':'glasses','glass':'glass','dress':'dress','status':'status','ethos':'ethos'}
SYNONYMS={'couch':'sofa','television':'tv','refrigerator':'fridge','automobile':'car','photograph':'photo',
          'ad':'advertisement','advert':'advertisement'}

def read(name):return [json.loads(x) for x in (OUT/name).read_text().splitlines() if x.strip()]
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def word(w):
    if w in IRREGULAR:return IRREGULAR[w]
    if len(w)>4 and w.endswith('ies'):w=w[:-3]+'y'
    elif len(w)>4 and w.endswith(('ches','shes','xes','zes')):w=w[:-2]
    elif len(w)>3 and w.endswith('s') and not w.endswith(('ss','us','is')):w=w[:-1]
    return SYNONYMS.get(w,w)

def normalize(text):
    tokens=re.findall(r"[a-z0-9]+",unicodedata.normalize('NFKC',text).casefold())
    return tuple(word(w) for w in tokens)

def match(tokens, phrase):
    return bool(phrase) and any(tokens[i:i+len(phrase)]==phrase for i in range(len(tokens)-len(phrase)+1))

def rank_metrics(scores,gold_indices):
    # Equal rounded cosine scores get midranks, not arbitrary filename ordering.
    rounded=np.round(scores.astype(float),6)
    ranks=rankdata(-rounded,method='average')
    gs=np.asarray(gold_indices,dtype=int)
    best_gold_score=rounded[gs].max()
    best_tie_size=int(np.sum(rounded==best_gold_score))
    best_tie_gold=int(np.sum(rounded[gs]==best_gold_score))
    expected_first=int(np.sum(rounded>best_gold_score))+(best_tie_size+1)/(best_tie_gold+1)
    result={'first_gold_rank':float(expected_first),'mean_gold_rank':float(ranks[gs].mean()),
            'mean_gold_rank_fraction':float(((ranks[gs]-1)/(len(scores)-1)).mean()),
            'gold_rank_ties':int(sum(np.sum(rounded==rounded[g])>1 for g in gs))}
    for k in (1,5,10):
        probs=[]
        for g in gs:
            better=int(np.sum(rounded>rounded[g]));tied=int(np.sum(rounded==rounded[g]))
            probs.append(float(np.clip((k-better)/tied,0,1)))
        result[f'recall_at_{k}']=float(np.mean(probs))
        allhit=1.
        for score in set(rounded[gs]):
            needed=int(np.sum(rounded[gs]==score));total=int(np.sum(rounded==score))
            taken=max(0,min(total,k-int(np.sum(rounded>score))))
            allhit*=math.comb(total-needed,taken-needed)/math.comb(total,taken) if taken>=needed else 0.
        result[f'all_annotated_hit_at_{k}']=allhit
    return result,ranks

def rho(rows,load='all_cue_load',outcome='mean_gold_rank_fraction'):
    valid=[r for r in rows if r.get('competitor_count',1)>0 and r.get(load) is not None and r.get(outcome) is not None]
    if len(valid)<3:return None
    x=[r[load] for r in valid];y=[r[outcome] for r in valid]
    if len(set(x))<2 or len(set(y))<2:return None
    return float(spearmanr(x,y).statistic)

def partial_rho(rows,load):
    rows=[r for r in rows if r.get('competitor_count',1)>0 and r.get(load) is not None and r.get('gold_cue_coverage') is not None]
    if len(rows)<30:return None
    x=rankdata([r[load] for r in rows]);y=rankdata([r['mean_gold_rank_fraction'] for r in rows])
    cov=[np.ones(len(rows))]
    for field in ('task','x','y'):
        for value in sorted(set(r[field] for r in rows))[1:]:cov.append(np.array([float(r[field]==value) for r in rows]))
    for field in ('gold_count','primary_cue_count','gold_caption_words','gold_cue_coverage','has_query_image'):
        v=rankdata([r[field] for r in rows]);cov.append((v-v.mean())/(v.std() or 1))
    z=np.column_stack(cov)
    xr=x-z@np.linalg.lstsq(z,x,rcond=None)[0]
    yr=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    if xr.std()<1e-8 or yr.std()<1e-8:return None
    return float(np.corrcoef(xr,yr)[0,1])

def bootstrap(rows,load,partial=False,n_boot=1000):
    rng=np.random.default_rng(42)
    tasks=sorted(set(r['task'] for r in rows));groups={t:[r for r in rows if r['task']==t] for t in tasks}
    values=[]
    for _ in range(n_boot):
        sample=[r for t in rng.choice(tasks,size=len(tasks),replace=True) for r in groups[t]]
        v=partial_rho(sample,load) if partial else rho(sample,load)
        if v is not None:values.append(v)
    return {'low':float(np.percentile(values,2.5)),'high':float(np.percentile(values,97.5)),
            'valid_replicates':len(values),'clusters':len(tasks)} if values else None

def summarize(label,rows):
    return {'group':label,'n':len(rows),'n_with_primary_cue':sum(r['all_cue_load'] is not None for r in rows),
            'n_with_supported_cue':sum(r['supported_cue_load'] is not None for r in rows),
            'n_correlation_eligible':sum(r['all_cue_load'] is not None and r['competitor_count']>0 for r in rows),
            'mean_gold_count':float(np.mean([r['gold_count'] for r in rows])) if rows else None,
            'exploratory_small_cell':len(rows)<10,
            'mean_recall_at_5':float(np.mean([r['recall_at_5'] for r in rows])) if rows else None,
            'mean_recall_at_10':float(np.mean([r['recall_at_10'] for r in rows])) if rows else None,
            'rho_all_load_rank':rho(rows),
            'rho_supported_load_rank':rho(rows,'supported_cue_load'),
            'rho_all_load_recall5':rho(rows,outcome='recall_at_5')}

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    config=json.loads((OUT/'protocol.json').read_text())
    memories=read('memories.jsonl');queries=read('queries.jsonl');gold={r['id']:r for r in read('gold.jsonl')}
    cr=read('captions.jsonl');qr=read('query_cues.jsonl')
    assert len(cr)==len({r['id'] for r in cr})==len(read('images.jsonl'))
    assert len(qr)==len({r['id'] for r in qr})==len(queries)
    captions={r['id']:r['caption'] for r in cr};cues={r['id']:r for r in qr}
    assert all(not r['parse_error'] for r in qr),'Invalid cue JSON; inspect before continuing'
    assert all(not r['hit_token_limit'] for r in cr+qr),'Generation truncation; inspect before continuing'
    texts=[' '.join(captions[x] for x in m['images']) for m in memories]
    query_texts=[cues[q['id']]['source_text'] for q in queries]
    model=SentenceTransformer(config['retriever_model_path'],device='cpu',local_files_only=True)
    lengths=[len(model.tokenizer.encode(s)) for s in texts]
    qlengths=[len(model.tokenizer.encode(config['query_prefix']+s)) for s in query_texts]
    assert max(lengths+qlengths)<=model.max_seq_length,'BGE truncation would invalidate cue matching'
    emb=model.encode(texts,batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    qemb=model.encode([config['query_prefix']+s for s in query_texts],batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    np.savez_compressed(OUT/'embeddings.npz',memory=emb,query=qemb,
                        memory_ids=np.array([m['id'] for m in memories]),query_ids=np.array([q['id'] for q in queries]))
    tokenized=[normalize(t) for t in texts]
    task_idxs={t:[i for i,m in enumerate(memories) if m['task']==t] for t in sorted(set(m['task'] for m in memories))}
    rows=[];cue_rows=[];details=[];rejected=[];duplicate_sensitivity=[]
    for qi,q in enumerate(queries):
        g=gold[q['id']];inds=task_idxs[q['task']];mids=[memories[i]['id'] for i in inds]
        localgold=[mids.index(mid) for mid in g['clue_ids']];gset=set(localgold)
        ncompetitors=len(inds)-len(gset)
        scores=emb[inds]@qemb[qi]
        metrics,ranks=rank_metrics(scores,localgold)
        effective=[];seen=set()
        for cue in cues[q['id']]['cues']:
            norm=normalize(cue['text']);key=(cue['category'],norm)
            if not norm or (len(norm)==1 and norm[0] in GENERIC) or key in seen:
                rejected.append({'id':q['id'],'cue':cue,'reason':'generic_or_duplicate'});continue
            seen.add(key)
            hits=[j for j,i in enumerate(inds) if match(tokenized[i],norm)]
            ghits=[j for j in hits if j in gset];others=[j for j in hits if j not in gset]
            row={'id':q['id'],'task':q['task'],'x':g['x'],'y':g['y'],'category':cue['category'],
                 'cue':cue['text'],'normalized_cue':' '.join(norm),'gold_matches':len(ghits),
                 'other_matches':len(others),'other_match_fraction':len(others)/ncompetitors if ncompetitors else None,
                 'matching_gold_ids':[mids[j] for j in ghits],'matching_other_ids':[mids[j] for j in others]}
            cue_rows.append(row);effective.append(row)
        core=[c for c in effective if c['category'] in CORE]
        supported=[c for c in core if c['gold_matches']>0]
        union=set(mid for c in core for mid in c['matching_other_ids'])
        row={'id':q['id'],'task':q['task'],'x':g['x'],'y':g['y'],'has_query_image':int(bool(q['images'])),
             'memory_count':len(inds),'gold_count':len(localgold),'primary_cue_count':len(core),
             'competitor_count':ncompetitors,
             'supported_cue_count':len(supported),'gold_cue_coverage':len(supported)/len(core) if core else None,
             'all_cue_load':float(np.mean([c['other_matches'] for c in core])) if core else None,
             'supported_cue_load':float(np.mean([c['other_matches'] for c in supported])) if supported else None,
             'all_cue_load_fraction':float(np.mean([c['other_match_fraction'] for c in core])) if core and ncompetitors else None,
             'shared_cue_union_count':len(union),
             'gold_caption_words':float(np.mean([len(texts[inds[j]].split()) for j in localgold])),
             **metrics}
        for category in ('scene','person','object','attribute','time'):
            subset=[c for c in effective if c['category']==category]
            row[category+'_cue_count']=len(subset)
            row[category+'_load']=float(np.mean([c['other_matches'] for c in subset])) if subset else None
        rows.append(row)
        order=np.argsort(-np.round(scores,6),kind='stable')
        details.append({'id':q['id'],'question':q['question'],'query_text':query_texts[qi],
           'x':g['x'],'y':g['y'],'gold':[{'id':mids[j],'midrank':float(ranks[j]),'caption':texts[inds[j]]} for j in localgold],
           'ranking':[{'id':mids[j],'midrank':float(ranks[j]),'score':float(scores[j]),'is_gold':j in gset} for j in order],
           'cues':effective,'metrics':row})
        # Exact duplicate images can be visually indistinguishable across rounds.
        gold_images={tuple(memories[inds[j]]['images']) for j in localgold}
        expanded=[j for j,i in enumerate(inds) if tuple(memories[i]['images']) in gold_images]
        sm,_=rank_metrics(scores,expanded)
        duplicate_sensitivity.append({'id':q['id'],'original_gold_count':len(localgold),
            'content_equivalent_gold_count':len(expanded),'first_equivalent_gold_rank':sm['first_gold_rank']})
    groups=[summarize('all',rows),summarize('text_only_queries',[r for r in rows if not r['has_query_image']])]
    for field in ('x','y','task'):
        for value in sorted(set(r[field] for r in rows)):groups.append(summarize(value,[r for r in rows if r[field]==value]))
    for x in ('X1','X2','X3','X4'):
        for y in ('Y1','Y2','Y3'):groups.append(summarize(x+'/'+y,[r for r in rows if r['x']==x and r['y']==y]))
    associations={}
    for load in ('all_cue_load','supported_cue_load'):
        associations[load]={'n':sum(r[load] is not None and r['competitor_count']>0 for r in rows),
                            'spearman':rho(rows,load),'cluster_bootstrap_95ci':bootstrap(rows,load),
                            'partial_rank_correlation':partial_rho(rows,load),
                            'partial_cluster_bootstrap_95ci':bootstrap(rows,load,partial=True)}
    audit={
       'captions':len(cr),'queries':len(rows),'caption_token_limit_hits':sum(r['hit_token_limit'] for r in cr),
       'cue_parse_errors':sum(bool(r['parse_error']) for r in qr),'cue_token_limit_hits':sum(r['hit_token_limit'] for r in qr),
       'rejected_nonliteral_or_schema_cues':sum(len(r['rejected_cues']) for r in qr),
       'rejected_generic_duplicate_cues':rejected,
       'questions_without_primary_cues':sum(not r['primary_cue_count'] for r in rows),
       'questions_without_competitors':sum(not r['competitor_count'] for r in rows),
       'questions_without_any_gold_supported_primary_cue':sum(not r['supported_cue_count'] for r in rows),
       'questions_with_tied_gold_ranks':sum(r['gold_rank_ties']>0 for r in rows),
       'questions_with_duplicate_image_gold_alternatives':sum(r['content_equivalent_gold_count']>r['original_gold_count'] for r in duplicate_sensitivity),
       'memory_max_bge_tokens':max(lengths),'query_max_bge_tokens':max(qlengths),
       'bge_max_seq_length':model.max_seq_length,'normalization_irregular':IRREGULAR,'normalization_synonyms':SYNONYMS,
       'input_sha256':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['protocol.json','captions.jsonl','query_cues.jsonl','gold.jsonl','memories.jsonl','queries.jsonl']},
    }
    write_csv('per_question.csv',rows);write_csv('per_cue.csv',cue_rows);write_csv('group_summary.csv',groups)
    write_csv('duplicate_image_sensitivity.csv',duplicate_sensitivity)
    (OUT/'retrieval_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    dump('summary.json',{'associations':associations,'groups':groups,'audit':audit})
    print(json.dumps({'associations':associations,'all':groups[0],'audit':{k:v for k,v in audit.items() if not isinstance(v,(dict,list))}},indent=2))

if __name__=='__main__':main()
