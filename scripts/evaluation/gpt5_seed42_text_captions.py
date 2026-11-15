"""Fresh seed42 GPT-5 captions for the four native text-memory experiments."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import threading
import time
from scripts.evaluation import openai_gpt5_base_transport as api
from scripts.evaluation.openai_backbone_transport import request_sha256
from scripts.evaluation.gpt5_text_memory_transport import classify_http_error

OUTPUT=ROOT/'results/memeye_gpt5_seed42_text_methods_20261025'
REFERENCE=ROOT/'results/memeye_deepseek_flash_seed42_full_20261025'
ENV=Path('/storage/hewenwen/NTU Decentralized Agent/.env')

def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def freeze(p,d):
    if Path(p).exists(): assert read(p)==d,str(p)
    else: dump(p,d)

def prepare():
    root=OUTPUT/'caption_source';root.mkdir(parents=True,exist_ok=True)
    source=read(REFERENCE/'inputs/protocol.json')
    for p,h in source['source_sha256'].items(): assert sha(p)==h,p
    for name in ['source_questions.jsonl','source_memories.jsonl','evaluation.jsonl','images.jsonl','api_requests_caption.jsonl']:
        data=(REFERENCE/'inputs'/name).read_bytes();p=root/'inputs'/name;p.parent.mkdir(exist_ok=True)
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
    source=dict(source,model='gpt-5',caption_max_new_tokens=2048,caption_seed=42)
    freeze(root/'inputs/protocol.json',source)
    provenance=read(ROOT/'results/memeye_deepseek_flash_seed42_text_methods_20261025/public_input_provenance.json')
    for item in provenance['verified_public_files']:
        p=ROOT/'data/MemEye'/item['relative_path'];b=p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['git_blob_id']
    freeze(OUTPUT/'public_input_provenance.json',provenance)
    sources=[Path(__file__),ROOT/'scripts/evaluation/gpt5_text_memory_transport.py',ROOT/'scripts/evaluation/openai_gpt5_base_transport.py',ROOT/'scripts/evaluation/openai_backbone_transport.py',ROOT/'scripts/evaluation/openai_chat_transport.py',ROOT/'scripts/evaluation/openai_gpt54_seeded_transport.py',root/'inputs/protocol.json',root/'inputs/api_requests_caption.jsonl',OUTPUT/'public_input_provenance.json']
    contract=dict(model='gpt-5',seed=42,reasoning_effort='minimal',temperature='omitted',max_completion_tokens=2048,
        endpoint='https://api.openai.com/v1/chat/completions',image_detail='high',images=437,
        prior_captions_reused=False,gold_in_payload=False,workers=8,
        source_sha256={str(p):sha(p) for p in sources})
    freeze(root/'protocol.json',contract)
    return root,contract

def run(limit=0):
    root,p=prepare();config=api.load_config(ENV)
    # This process-local adapter leaves all prior signed experiment files intact.
    api._http_error=classify_http_error
    for f,h in p['source_sha256'].items():assert sha(f)==h
    records=rows(root/'inputs/api_requests_caption.jsonl');assert len(records)==437
    lock=threading.Lock();stopped=threading.Event()
    def log(name,d):
        with lock:
            with (root/name).open('a') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    def one(r):
        if stopped.is_set():raise RuntimeError('Caption batch stopped after API failure')
        payload={k:r[k] for k in ('user_text','query_images')}
        wire=api.build_request(payload,seed=42,max_output_tokens=2048,image_detail='high')
        key=request_sha256(wire);path=root/'responses'/(r['id']+'.json')
        if path.exists():
            result=read(path);assert result['request_sha256']==key and result['seed']==42
            return result
        for attempt in range(1,4):
            if stopped.is_set():raise RuntimeError('Caption batch stopped after API failure')
            try:
                out=api.chat_request(config,wire,timeout=120)
                result=dict(out,id=r['id'],request_sha256=key,seed=42,historical_response_reuse=False)
                log('attempts.jsonl',dict(id=r['id'],attempt=attempt,status='ok',response_id=out['response_id']))
                if out['finish_reason']!='stop' or out.get('refusal') or not out['raw_response'].strip():
                    dump(root/'invalid_responses'/(r['id']+'.json'),result)
                    stopped.set()
                    raise RuntimeError('Invalid caption; inspect archived raw output')
                dump(path,result);return result
            except api.APIError as e:
                log('attempts.jsonl',dict(id=r['id'],attempt=attempt,status='error',kind=e.kind,retryable=e.retryable,provider_error_code=getattr(e,'provider_error_code',e.code)))
                if not e.retryable or attempt==3:
                    stopped.set();raise
                time.sleep(2*attempt)
    todo=records[:limit] if limit else records
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(one,r) for r in todo]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%20==0 or i==len(todo):print(f'GPT-5 captions: {i}/{len(todo)}',flush=True)
    if limit:return
    output=[one(r) for r in records]
    models=sorted({r['returned_model'] for r in output});assert len(models)==1
    caps=[dict(id=r['id'],caption=r['raw_response'].strip(),model=r['returned_model'],requested_model='gpt-5',seed=42,request_sha256=r['request_sha256'],response_id=r['response_id']) for r in output]
    (root/'captions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in caps))
    artifacts=[root/'captions.jsonl',root/'protocol.json',*sorted((root/'inputs').glob('*')),*sorted((root/'responses').glob('*.json'))]
    dump(root/'completion_manifest.json',dict(complete=True,caption_count=437,seed=42,returned_model=models[0],artifacts_sha256={str(x):sha(x) for x in artifacts}))
    print(json.dumps(dict(complete=True,captions=437,seed=42,models=models)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if args.prepare_only:prepare()
    else:run(args.limit)
