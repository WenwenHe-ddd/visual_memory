"""Frozen design and question-only anchors for controlled distractor injection."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
NAT=ROOT/'results/natural_observation'
OUT=ROOT/'results/controlled_injection'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'.vendor'))

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

ANCHOR_KINDS={'object','person','place','attribute','time','order','event','reference'}
ANCHOR_PROMPT='''Extract ALL explicit retrieval anchors of the question below, including the information that locates the particular remembered event or image. We are designing a controlled retrieval experiment, not answering the question. Return JSON only: {"anchors":[{"text":"exact short span of the question","kind":"object|person|place|attribute|time|order|event|reference"}],"locator_cues":[{"text":"exact span also listed in anchors","kind":"same kind as that anchor"}],"critical_property":"short description of the property asked about"}.
Keep concrete objects, named people, places, known attributes, dates, event descriptions, temporal order, episode/session references, and image-order qualifiers. Locator cues identify the particular target event or image; include named people and applicable event/time/order/reference qualifiers. An empty locator_cues list is allowed only when the question has no such qualifier. Do not discard a locator because it cannot be seen in an image. Split descriptive noun phrases: "yellow cards held by Player 2" -> "cards" (object), "yellow" (attribute), "Player 2" (person). For "What type of room did Hannah show as the very first image on move-in day?", anchors must include "room" (object), "Hannah" (person), "very first image" (order), "move-in day" (event); the last three are locator cues. For brands/products retain proper names and the object head separately. Exclude standalone generic image/photo/screenshot/frame/detail/information words, question words, and unspecified answer placeholders, but retain qualified references such as "first image". Each anchor should normally be 1-3 words except proper names and complete references. Use at most 12 anchors; all text values must occur verbatim in the question. Do not use an answer or infer an unstated fact.
QUESTION:
'''

def anchor_signature():return hashlib.sha256(ANCHOR_PROMPT.encode()).hexdigest()

class LocalQwen:
    def __init__(self):
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        self.torch=torch;torch.manual_seed(42);torch.set_num_threads(8)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(ROOT/'models/Qwen3-VL-8B-Instruct'),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        self.processor=AutoProcessor.from_pretrained(str(ROOT/'models/Qwen3-VL-8B-Instruct'),local_files_only=True)
        self.processor.tokenizer.padding_side='left'
        self.processor.image_processor.size={'shortest_edge':65536,'longest_edge':1638400}

    def generate(self,texts,paths=None,limit=256):
        from PIL import Image
        messages=[[{'role':'user','content':([{'type':'image'}] if paths else [])+[{'type':'text','text':t}]}] for t in texts]
        rendered=[self.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        images=[Image.open(p).convert('RGB') for p in paths] if paths else None
        inputs=self.processor(text=rendered,images=images,padding=True,return_tensors='pt').to('cuda:0')
        with self.torch.inference_mode():
            output=self.model.generate(**inputs,do_sample=False,max_new_tokens=limit,pad_token_id=self.processor.tokenizer.pad_token_id)
        tokens=output[:,inputs['input_ids'].shape[1]:]
        decoded=self.processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        results=[]
        for text,row in zip(decoded,tokens.tolist()):
            n=next((i+1 for i,t in enumerate(row) if t==self.processor.tokenizer.eos_token_id),len(row))
            try:
                raw=text.strip()
                if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
                parsed=json.loads(raw);error=None
            except Exception as e:parsed=None;error=type(e).__name__
            results.append({'raw':text,'parsed':parsed,'parse_error':error,'tokens':n,'truncated':n>=limit})
        return results

def prepare():
    natural=json.loads((NAT/'protocol.json').read_text())
    protocol={
      'study':'Controlled injection of original-image distractors into caption retrieval',
      'status':'real-image first; synthetic items require explicit human review before formal use',
      'provenance':natural['provenance'],'caption_policy':'reuse identical cached Qwen3-VL ordinary captions; no query-aware recaptioning',
      'retriever':natural['retriever'],'retriever_revision':natural['retriever_revision'],
      'conditions':['high_overlap','low_overlap_same_task','unrelated_other_task'],
      'distractor_counts':[0,5,10,20,50],'seeds':list(range(42,62)),
      'base_memory':'all gold clue rounds plus 5 fixed unrelated other-task items, same base in every condition/count/seed',
      'base_scope':'Controlled retrieval collection; original benchmark episode, question, answer, evidence IDs and query text stay fixed. This is not a rewritten conversational history.',
      'real_pool':'original image-bearing MemEye rounds outside gold and fixed background; exclude identical-image contents from all pools and sampling without replacement',
      'high_rule':'>=50% of complete query anchors shared and >=1 object/person/place anchor shared; all required locator cues must be present in actual retrieval text; caption explicitly binds the queried entity to the same decisive property and a different value supported by its image; no alternate valid evidence',
      'cue_coverage_gate':'Extract person, event, time, order and reference locators from the full question. Every required locator must be represented in at least one indexed gold caption. Missing contextual cues exclude a question from this caption-only experiment; do not invent event metadata or infer it from room appearance.',
      'anchor_signature':anchor_signature(),
      'low_rule':'same original task; zero anchor-word overlap; excludes semantically relevant or answer-bearing candidates in later review',
      'unrelated_rule':'different original task; zero anchor-word overlap; excludes semantically relevant or answer-bearing candidates in later review',
      'length_control':'BGE-token lengths within +/-10% of median gold caption length; matched triplets differ by at most 5 BGE tokens',
      'sampling':'match high/low/unrelated triplets without retrieval scores; seeded random orders create nested 5/10/20/50 prefixes',
      'anchor_prompt':ANCHOR_PROMPT,
      'score_blinding':'candidate annotation and matching never access BGE similarity or retrieval ranks',
      'primary_outcome':'increase in mean unnormalized gold rank from fixed N=0 base; rounded-score ties use midranks',
      'secondary_outcomes':['annotated_evidence_recall@5','annotated_evidence_recall@10','expected_first_gold_rank'],
      'interaction':'paired high-minus-low and high-minus-unrelated difference in rank degradation at each N; divide by N for per-item slope; average seeds within question',
      'uncertainty':'bootstrap whole source-task clusters; never count random seeds as independent questions; show cluster and question counts',
      'cohort':'each N uses only questions with all 3 matched conditions; full dose-response comparison uses same cohort eligible at largest displayed N',
      'flatness_criterion':'practical equivalence would require 95% interval within +/-1 mean-rank position at N=50; no nonsignificant-test-as-flat claim',
      'synthetic_gate':'draft only; human review must confirm changed decisive attribute, preserved nondecisive cues and length, and no correct answer/alternate evidence leakage before inclusion',
      'scope_of_inference':'causal effect of this injection policy on fixed BGE caption retrieval; not evidence of an internal human-like cognitive mechanism',
      'source_hashes':{f:hashlib.sha256((NAT/f).read_bytes()).hexdigest() for f in ['captions.jsonl','queries.jsonl','gold.jsonl','memories.jsonl','embeddings.npz']},
    }
    previous=json.loads((OUT/'protocol.json').read_text()) if (OUT/'protocol.json').exists() else {}
    for key in ('human_review_policy','human_reviewed_candidates','withdrawals'):
        if key in previous:protocol[key]=previous[key]
    write(OUT/'protocol.json',protocol)
    qmap={q['id']:q for q in read(NAT/'queries.jsonl')}
    gold={q['id']:q for q in read(NAT/'gold.jsonl')}
    sources={q['id']:q['source_text'] for q in read(NAT/'query_cues.jsonl')}
    targets=[]
    for f in sorted((ROOT/'data/MemEye/data/dialog').glob('*_Open.json')):
        task=f.stem.removesuffix('_Open');data=json.loads(f.read_text())
        for q in data['human-annotated QAs']:
            qid=task+'/'+q['question_id']
            targets.append({**qmap[qid],**gold[qid],'answer':q['answer'],'retrieval_text':sources[qid]})
    (OUT/'targets.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in targets))
    print('Prepared',len(targets),'targets')

def annotate():
    targets=read(OUT/'targets.jsonl');done={r['id'] for r in read(OUT/'anchors.jsonl') if r.get('anchor_signature')==anchor_signature()}
    targets=[t for t in targets if t['id'] not in done]
    if not targets:print('No pending anchors');return
    model=LocalQwen()
    for start in range(0,len(targets),4):
        batch=targets[start:start+4]
        results=model.generate([ANCHOR_PROMPT+t['retrieval_text'] for t in batch],limit=768)
        with (OUT/'anchors.jsonl').open('a') as f:
            for t,r in zip(batch,results):
                accepted=[];rejected=[]
                if isinstance(r['parsed'],dict):
                    for a in r['parsed'].get('anchors',[]):
                        if isinstance(a,dict) and a.get('kind') in ANCHOR_KINDS and isinstance(a.get('text'),str) and a['text'].strip() and a['text'].casefold() in t['retrieval_text'].casefold():accepted.append(a)
                        else:rejected.append(a)
                f.write(json.dumps({'id':t['id'],'anchors':accepted,'rejected':rejected,'anchor_signature':anchor_signature(),**r},ensure_ascii=False)+'\n')
        print('anchors',len(done)+start+len(batch),'/371',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','annotate']);args=p.parse_args()
    prepare() if args.stage=='prepare' else annotate()
