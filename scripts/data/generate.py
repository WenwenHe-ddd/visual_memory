"""Local, resumable captioning and question-only cue extraction. No API calls."""

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
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/natural_observation'

def read(name):
    p = OUT/name
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []

def signature(config):
    fields = ['provenance','caption_prompt','cue_prompt','image_max_pixels','caption_max_new_tokens',
              'cue_max_new_tokens','seed','do_sample','dtype']
    return hashlib.sha256(json.dumps({k:config[k] for k in fields},sort_keys=True).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stage',choices=['all','caption','cue'],default='all')
    args = ap.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    config=json.loads((OUT/'protocol.json').read_text())
    sig=signature(config)
    torch.manual_seed(config['seed'])
    torch.set_num_threads(8)
    device='cuda:0'
    assert torch.cuda.is_available()
    print('Loading model',flush=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(config['caption_model_path'],
                  dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval()
    processor=AutoProcessor.from_pretrained(config['caption_model_path'],local_files_only=True)
    processor.tokenizer.padding_side='left'
    processor.image_processor.size={'shortest_edge':65536,'longest_edge':config['image_max_pixels']}
    (OUT/'generation_environment.json').write_text(json.dumps({
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'protocol_signature':sig,'processor_size':processor.image_processor.size},indent=2))

    def generate(texts, paths=None, max_new_tokens=256):
        messages=[]
        for text in texts:
            content=([{'type':'image'}] if paths else [])+[{'type':'text','text':text}]
            messages.append([{'role':'user','content':content}])
        rendered=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
        images=[Image.open(p).convert('RGB') for p in paths] if paths else None
        inputs=processor(text=rendered,images=images,padding=True,return_tensors='pt').to(device)
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,
                                      pad_token_id=processor.tokenizer.pad_token_id)
        outputs=generated[:,inputs['input_ids'].shape[1]:]
        texts=processor.batch_decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        lengths=[]
        for row in outputs.tolist():
            n=next((i+1 for i,t in enumerate(row) if t==processor.tokenizer.eos_token_id),len(row))
            lengths.append(n)
        return texts,lengths

    if args.stage in ('all','caption'):
        existing=read('captions.jsonl')
        assert all(r['protocol_signature']==sig for r in existing), 'Caption config mismatch'
        done={r['id'] for r in existing}
        pending=[r for r in read('images.jsonl') if r['id'] not in done]
        if args.limit:pending=pending[:args.limit]
        for offset in range(0,len(pending),config['batch_size']):
            batch=pending[offset:offset+config['batch_size']]
            start=time.time()
            texts,lens=generate([config['caption_prompt']]*len(batch),[r['path'] for r in batch],config['caption_max_new_tokens'])
            with (OUT/'captions.jsonl').open('a') as f:
                for r,text,n in zip(batch,texts,lens):
                    f.write(json.dumps({'id':r['id'],'caption':text.strip(),'tokens':n,
                        'hit_token_limit':n>=config['caption_max_new_tokens'],
                        'protocol_signature':sig},ensure_ascii=False)+'\n')
            print(f'captions {len(done)+offset+len(batch)}/{len(done)+len(pending)} batch_seconds={time.time()-start:.1f}',flush=True)
    if args.stage in ('all','cue'):
        captions={r['id']:r['caption'] for r in read('captions.jsonl')}
        existing=read('query_cues.jsonl')
        assert all(r['protocol_signature']==sig for r in existing),'Cue config mismatch'
        done={r['id'] for r in existing}
        pending=[q for q in read('queries.jsonl') if q['id'] not in done]
        if args.limit:pending=pending[:args.limit]
        for offset in range(0,len(pending),config['batch_size']):
            batch=pending[offset:offset+config['batch_size']]
            sources=[q['question']+(' Query image description: '+' '.join(captions[x] for x in q['images']) if q['images'] else '') for q in batch]
            texts,lens=generate([config['cue_prompt']+s for s in sources],max_new_tokens=config['cue_max_new_tokens'])
            with (OUT/'query_cues.jsonl').open('a') as f:
                for q,source,text,n in zip(batch,sources,texts,lens):
                    error=None; cues=[]; rejected=[]
                    try:
                        raw=text.strip()
                        if raw.startswith('```'):raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
                        obj=json.loads(raw)
                        assert isinstance(obj,dict) and isinstance(obj.get('cues'),list)
                        for cue in obj['cues']:
                            if not isinstance(cue,dict) or cue.get('category') not in ['scene','person','object','attribute','time'] or not isinstance(cue.get('text'),str):
                                rejected.append({'cue':cue,'reason':'invalid_schema'});continue
                            if not cue['text'].strip() or cue['text'].casefold() not in source.casefold():
                                rejected.append({'cue':cue,'reason':'not_literal_source_span'});continue
                            if cue not in cues:cues.append(cue)
                    except Exception as exc:error=type(exc).__name__+': '+str(exc)
                    f.write(json.dumps({'id':q['id'],'source_text':source,'cues':cues,
                       'rejected_cues':rejected,'raw_output':text,'parse_error':error,
                       'tokens':n,'hit_token_limit':n>=config['cue_max_new_tokens'],
                       'protocol_signature':sig},ensure_ascii=False)+'\n')
            print(f'query_cues {len(done)+offset+len(batch)}/{len(done)+len(pending)}',flush=True)

if __name__=='__main__':main()
