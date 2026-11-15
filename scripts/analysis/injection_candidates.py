"""Score-blind candidate pools and caption-grounded negative validation."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import collections
import hashlib
import json
import math
import os
import random
from pathlib import Path
import re

from scripts.analysis.injection_design import ROOT,NAT,OUT,LocalQwen,read,write,anchor_signature,ANCHOR_KINDS
from scripts.analysis.analyze import normalize,match,GENERIC

TEXT_JUDGE='''You are auditing NEGATIVE retrieval evidence for a controlled experiment. The benchmark question, original answer and original evidence refer to a fixed original episode; an extra memory is not a new event in that episode.
Check the FULL question and the required locator_cues supplied below. The candidate caption must preserve those person/event/time/order/reference cues; merely depicting another object or room does not establish that it refers to the queried event. Never infer a person's name, move-in day, or upload order from room appearance. Accept a high-overlap distractor ONLY if these locators are shared and its caption explicitly describes the SAME TYPE of decisive property asked in the question, with a DIFFERENT value, and it is not valid evidence for the original question. Sharing objects without an explicit different decisive value is insufficient. An unrelated number/color elsewhere is insufficient. Do not mark a legitimate earlier state needed for an evolution question as negative. For counting/comparison questions, reject if a specific contradictory property cannot be identified. Be conservative: uncertainty means reject.
Return JSON only: {"valid_high":true_or_false,"locator_cues_shared":true_or_false,"different_decisive_value":true_or_false,"not_valid_original_evidence":true_or_false,"candidate_value":"short value or empty","evidence_quote":"EXACT candidate-caption span demonstrating that property and value, or empty","reason":"brief reason"}.
'''

VISUAL_JUDGE='''Verify the proposed high-overlap negative against the attached ORIGINAL image. This is evaluation annotation, not caption generation. The benchmark answer is fixed to its original episode. Do NOT assume the proposed annotation is true. Check whether the image visibly supports the candidate value for the same decisive property and whether this differs from the original correct value. Reject uncertain, unreadable, or uncountable details. Merely sharing a scene/object is insufficient. Do not count unrelated items or accept a valid earlier state needed by the original question.
Person names, event dates and upload order must already be supported by the candidate retrieval text. Visual appearance cannot establish them. Never upgrade an unverified contextual cue on the basis of the image.
Return JSON only: {"visually_supported":true_or_false,"different_decisive_value":true_or_false,"valid_negative":true_or_false,"observed_value":"short value","reason":"brief reason"}.
'''

def load_data():
    memories=read(NAT/'memories.jsonl');captions={r['id']:r['caption'] for r in read(NAT/'captions.jsonl')}
    mems={m['id']:{**m,'caption':' '.join(captions[x] for x in m['images'])} for m in memories}
    return mems,{t['id']:t for t in read(OUT/'targets.jsonl')}

def seed(qid):return int(hashlib.sha256(qid.encode()).hexdigest()[:12],16)+42

def judge_signature(stage):
    prompt=TEXT_JUDGE if stage=='text' else VISUAL_JUDGE
    return hashlib.sha256((anchor_signature()+prompt).encode()).hexdigest()

def question_reviews():
    return {r['question_id']:r for r in read(OUT/'question_cue_reviews.jsonl')}

def cue_gate(target,row,mems,review=None):
    """Fail closed when event locators were omitted or lost during captioning."""
    if review and review.get('decision')=='exclude_caption_only':
        return {'passed':False,'reason':review['reason'],'locator_cues':review.get('locator_cues',[])}
    if not row or row.get('anchor_signature')!=anchor_signature():
        return {'passed':False,'reason':'Legacy anchor extraction excluded temporal/reference cues; full-question extraction pending.','locator_cues':[]}
    parsed=row.get('parsed') or {}
    locators=parsed.get('locator_cues')
    anchors=row.get('anchors',[])
    if row.get('parse_error') or row.get('truncated') or row.get('rejected') or not isinstance(locators,list):
        return {'passed':False,'reason':'Incomplete or invalid full-question anchor annotation.','locator_cues':[]}
    for cue in locators:
        if not isinstance(cue,dict) or cue.get('kind') not in ANCHOR_KINDS or cue not in anchors:
            return {'passed':False,'reason':'Locator is missing from validated question anchors.','locator_cues':[]}
    # Known proper names and temporal/reference anchors cannot silently disappear
    # from the locator subset, even if an annotation returns an empty list.
    contextual=[a for a in anchors if a['kind'] in {'person','time','order','event','reference'}]
    if any(a not in locators for a in contextual):
        return {'passed':False,'reason':'Required contextual anchors omitted from locator_cues.','locator_cues':locators}
    gold_tokens=[normalize(mems[mid]['caption']) for mid in target['clue_ids']]
    missing=[a['text'] for a in locators if not any(match(ts,normalize(a['text'])) for ts in gold_tokens)]
    return {'passed':not missing,'reason':('Indexed gold captions omit required locators: '+', '.join(missing)) if missing else '',
            'locator_cues':locators,'missing_gold_locators':missing}

def valid_pair_ids(visual,text,human):
    """Current prompt versions and human exclusions override legacy VLM accepts."""
    latest_text={r['id']:r for r in text};latest_visual={r['id']:r for r in visual}
    text_ids={r['id'] for r in latest_text.values() if r['accepted'] and r.get('judge_signature')==judge_signature('text')}
    latest_human={r['review_id']:r for r in human}
    excluded={r['review_id'] for r in latest_human.values() if r.get('decision')=='exclude'}
    return {r['id'] for r in latest_visual.values() if r['accepted'] and r.get('judge_signature')==judge_signature('visual')
            and r['id'] in text_ids and r['id'] not in excluded}

def pools():
    os.environ['HF_HUB_OFFLINE']='1'
    from transformers import AutoTokenizer
    natural=json.loads((NAT/'protocol.json').read_text())
    tok=AutoTokenizer.from_pretrained(natural['retriever_model_path'],local_files_only=True)
    mems,targets=load_data();anchors={r['id']:r for r in read(OUT/'anchors.jsonl')}
    assert len(anchors)==371
    for m in mems.values():m['tokens']=len(tok.encode(m['caption']));m['normalized']=normalize(m['caption'])
    result=[];pairs=[];audit=collections.Counter();reviews=question_reviews()
    for qid,t in targets.items():
        row=anchors[qid];terms=[];seen=set()
        for a in row['anchors']:
            term=normalize(a['text'])
            if not term or (len(term)==1 and term[0] in GENERIC) or term in seen:continue
            seen.add(term);terms.append({**a,'normalized':term})
        gate=cue_gate(t,row,mems,reviews.get(qid))
        glens=sorted(mems[i]['tokens'] for i in t['clue_ids']);n=len(glens)
        ref=(glens[(n-1)//2]+glens[n//2])/2
        bounds=(math.ceil(.9*ref),math.floor(1.1*ref))
        banned_images={im for g in t['clue_ids'] for im in mems[g]['images']}|set(t['images'])
        used_contents=set();pool={'high':[],'low':[],'unrelated':[]}
        for mid,m in sorted(mems.items()):
            if mid in t['clue_ids'] or any(im in banned_images for im in m['images']):continue
            if not bounds[0]<=m['tokens']<=bounds[1]:continue
            content=tuple(m['images'])
            if content in used_contents:continue
            used_contents.add(content)
            hits=[a for a in terms if match(m['normalized'],tuple(a['normalized']))]
            locator_hits=all(match(m['normalized'],normalize(a['text'])) for a in gate['locator_cues'])
            if terms and locator_hits and len(hits)/len(terms)>=.5 and any(a['kind'] in ('object','person','place') for a in hits):
                kind='high'
            elif terms and not hits:kind='low' if m['task']==t['task'] else 'unrelated'
            else:continue
            pool[kind].append({'id':mid,'tokens':m['tokens'],'shared_anchors':[a['text'] for a in hits]})
        rng=random.Random(seed(qid));background_pool=list(pool['unrelated']);rng.shuffle(background_pool)
        background=[m['id'] for m in background_pool[:5]]
        pool['unrelated']=[m for m in pool['unrelated'] if m['id'] not in background]
        ready=gate['passed'] and bool(terms) and len(background)==5 and min(map(len,pool.values()))>=5 and not row['parse_error'] and not row['truncated']
        record={'id':qid,'task':t['task'],'x':t['x'],'y':t['y'],'anchors':terms,'reference_tokens':ref,'length_bounds':bounds,
                'base_ids':t['clue_ids']+background,'pools':pool,'ready_for_real_min5':ready,
                'anchor_signature':anchor_signature(),'cue_gate':gate}
        result.append(record)
        audit['questions']+=1;audit['ready_for_real_min5']+=ready
        if ready:
            for candidate in pool['high']:
                pairs.append({'id':qid+'||'+candidate['id'],'qid':qid,'memory_id':candidate['id']})
    (OUT/'pools.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    (OUT/'high_candidate_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    audit['high_pairs_to_verify']=len(pairs)
    audit['prevalidation_complete_grid_counts']={str(k):sum(len(r['base_ids'])>=len(targets[r['id']]['clue_ids'])+5 and min(map(len,r['pools'].values()))>=k for r in result) for k in (5,10,20,50)}
    write(OUT/'pool_audit.json',dict(audit));print(json.dumps(dict(audit),indent=2))

def text_judge():
    mems,targets=load_data();existing=read(OUT/'high_text_judgments.jsonl');done={r['id'] for r in existing}
    done={r['id'] for r in existing if r.get('judge_signature')==judge_signature('text')}
    pending=[p for p in read(OUT/'high_candidate_pairs.jsonl') if p['id'] not in done]
    anchors={r['id']:r for r in read(OUT/'anchors.jsonl')};reviews=question_reviews()
    pending=[p for p in pending if cue_gate(targets[p['qid']],anchors.get(p['qid']),mems,reviews.get(p['qid']))['passed']]
    write(OUT/'judge_prompts.json',{'text_judge':TEXT_JUDGE,'visual_judge':VISUAL_JUDGE})
    if not pending:print('No pending text judgments with complete cue coverage');return
    model=LocalQwen()
    for offset in range(0,len(pending),8):
        batch=pending[offset:offset+8];prompts=[]
        for p in batch:
            t=targets[p['qid']];m=mems[p['memory_id']]
            payload={'question':t['retrieval_text'],'correct_answer':t['answer'],
                     'original_evidence_captions':[mems[i]['caption'] for i in t['clue_ids']],
                     'candidate_caption':m['caption'],'candidate_task':m['task'],'candidate_session':m['session_id'],
                     'locator_cues':anchors[p['qid']]['parsed']['locator_cues']}
            prompts.append(TEXT_JUDGE+json.dumps(payload,ensure_ascii=False))
        results=model.generate(prompts,limit=384)
        with (OUT/'high_text_judgments.jsonl').open('a') as f:
            for p,r in zip(batch,results):
                d=r['parsed'] or {};quote=d.get('evidence_quote','');caption=mems[p['memory_id']]['caption']
                accepted=not r['parse_error'] and not r['truncated'] and all(d.get(k) is True for k in ['valid_high','locator_cues_shared','different_decisive_value','not_valid_original_evidence']) and isinstance(quote,str) and bool(quote.strip()) and quote.casefold() in caption.casefold()
                f.write(json.dumps({**p,**r,'accepted':accepted,'judge_signature':judge_signature('text')},ensure_ascii=False)+'\n')
        print('high_text',len(done)+offset+len(batch),'/',len(done)+len(pending),flush=True)

def visual_judge():
    mems,targets=load_data();latest={r['id']:r for r in read(OUT/'high_text_judgments.jsonl')}
    text=[r for r in latest.values() if r['accepted'] and r.get('judge_signature')==judge_signature('text')]
    nbyq=collections.Counter(r['qid'] for r in text)
    # No real N>=5 arm is possible below five validated high-overlap items.
    eligible={q for q,n in nbyq.items() if n>=5}
    existing=read(OUT/'high_visual_judgments.jsonl');done={r['id'] for r in existing if r.get('judge_signature')==judge_signature('visual')}
    pending=[r for r in text if r['qid'] in eligible and r['id'] not in done]
    images={r['id']:r['path'] for r in read(NAT/'images.jsonl')}
    write(OUT/'text_judge_audit.json',{'accepted_pairs':len(text),'eligible_questions':len(eligible),'visual_pairs':len(pending)+len(done)})
    if not pending:print('No pending visual judgments');return
    model=LocalQwen()
    for offset in range(0,len(pending),4):
        batch=pending[offset:offset+4];prompts=[];paths=[]
        for p in batch:
            t=targets[p['qid']];m=mems[p['memory_id']]
            payload={'question':t['retrieval_text'],'correct_answer':t['answer'],'candidate_caption':m['caption'],
                     'proposed_different_value':p['parsed']['candidate_value'],'claimed_evidence':p['parsed']['evidence_quote']}
            prompts.append(VISUAL_JUDGE+json.dumps(payload,ensure_ascii=False));paths.append(images[m['images'][0]])
        results=model.generate(prompts,paths=paths,limit=256)
        with (OUT/'high_visual_judgments.jsonl').open('a') as f:
            for p,r in zip(batch,results):
                d=r['parsed'] or {};accepted=not r['parse_error'] and not r['truncated'] and all(d.get(k) is True for k in ['visually_supported','different_decisive_value','valid_negative'])
                f.write(json.dumps({'id':p['id'],'qid':p['qid'],'memory_id':p['memory_id'],**r,'accepted':accepted,'judge_signature':judge_signature('visual')},ensure_ascii=False)+'\n')
        print('high_visual',len(done)+offset+len(batch),'/',len(done)+len(pending),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pools','text','visual']);a=p.parse_args()
    {'pools':pools,'text':text_judge,'visual':visual_judge}[a.stage]()
