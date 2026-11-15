"""Model-matched native API settings with an explicit actual-wire experiment seed."""
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import threading
import time
from scripts.evaluation import required_grid_common as c

ENV=Path('/storage/hewenwen/NTU Decentralized Agent/.env')

class Gateway:
    def __init__(self,job):
        if job['backbone'] not in ('gpt5','deepseek'):raise ValueError('API backbone required')
        self.job=job;self.SEED=job['seed'];self.MODEL=c.MODELS[job['backbone']]
        self.__file__=__file__
        from scripts.evaluation import memeye_deepseek_seed2026_transport as images
        self.images=images
        if job['backbone']=='deepseek':
            path=Path(images.__file__)
            lines={n.lineno for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==2026}
            if len(lines)!=3:raise ValueError('DeepSeek seed literal boundary changed')
            self.api=c.clone(path,'grid_deepseek_transport',self.SEED,lines,old_seed=2026)
            self.expected='deepseek-v4.1-flash'
        else:
            from scripts.evaluation import gpt5_text_memory_transport as native
            path=Path(native.__file__)
            lines={n.lineno for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==42}
            self.api=c.clone(path,'grid_gpt5_transport',self.SEED,lines,old_seed=42)
            self.api.build_request=self.build_message_request
            self.expected='gpt-5-2025-08-07'
        self.APIError=self.api.APIError if hasattr(self.api,'APIError') else self.images.APIError
        self.config=self.api.load_config(ENV)

    def build_message_request(self,messages,max_tokens=2048,temperature=0,**kwargs):
        if kwargs.get('seed',self.SEED)!=self.SEED:raise ValueError('Wrong actual-wire seed')
        if self.job['backbone']=='deepseek':
            return self.api.build_message_request(messages,max_tokens,temperature,seed=self.SEED,model=self.MODEL)
        return dict(model='gpt-5',messages=self.images._messages(messages),
            max_completion_tokens=max(2048,int(max_tokens)),seed=self.SEED,
            reasoning_effort='minimal',n=1,stream=False,store=False)

    def build_request(self,payload,max_tokens=2048,temperature=0,**kwargs):
        cap=kwargs.get('max_output_tokens',max_tokens)
        if isinstance(payload,list):return self.build_message_request(payload,cap,temperature,**kwargs)
        if 'reader_messages' in payload:messages=payload['reader_messages']
        else:
            messages=[]
            if payload.get('system'):messages.append({'role':'system','content':payload['system']})
            content=[{'type':'image','path':i['path'],'sha256':i['sha256']} for i in payload.get('query_images',[])]
            content.append({'type':'text','text':payload['user_text']})
            messages.append({'role':'user','content':content if payload.get('query_images') else payload['user_text']})
        return self.build_message_request(messages,cap,temperature,**kwargs)

    def wire_from_native(self,request):
        """Retain native image preprocessing/message order, changing only seed."""
        body=dict(request);body['seed']=self.SEED
        if body['model']!=self.MODEL:raise ValueError('Wrong requested backbone')
        expected=self.build_message_request(body['messages'],body.get('max_tokens',body.get('max_completion_tokens')),body.get('temperature',0))
        if expected!=body:raise ValueError('Native provider fields differ from the original condition')
        return body

    def chat_request(self,config,request,timeout=120):
        body=self.wire_from_native(request)
        result=self.api.chat_request(config,body,timeout)
        if result['returned_model']!=self.expected:
            raise self.APIError('model_mismatch',retryable=False,message='Returned model revision differs from the seed42 caption backbone')
        usage=result.get('usage') or {}
        return dict(result,seed=self.SEED,requested_model=self.MODEL,request_sha256=c.digest(body),
            input_tokens=usage.get('prompt_tokens',result.get('input_tokens',0)),
            generated_tokens=usage.get('completion_tokens',result.get('generated_tokens',0)),
            hit_token_limit=result['finish_reason']=='length')

    def call(self,directory,request,attempts=4,allow_rejection=False):
        directory=Path(directory);key=c.digest(request);target=directory/key
        with c.lock(directory/'locks'/(key+'.lock')):
            if (target/'response.json').exists():
                raw=gzip.decompress((target/'wire.json.gz').read_bytes())
                if json.loads(raw)!=request:raise ValueError('Exact-wire cache differs')
                return c.read(target/'response.json')
            target.mkdir(parents=True,exist_ok=True)
            encoded=json.dumps(request,ensure_ascii=False,allow_nan=False).encode()
            (target/'wire.json.gz').write_bytes(gzip.compress(encoded,mtime=0))
            previous=c.rows(target/'attempts.jsonl') if (target/'attempts.jsonl').exists() else []
            if previous and previous[-1]['status']=='ok':
                c.freeze(target/'response.json',previous[-1]['output']);return previous[-1]['output']
            if len(previous)>=attempts or (previous and not previous[-1].get('retryable',False)):
                raise RuntimeError('Native attempt policy exhausted; failure evidence retained')
            for number in range(len(previous)+1,attempts+1):
                try:
                    result=self.chat_request(self.config,request)
                    event=dict(attempt=number,status='ok',output=result,retryable=False)
                except self.APIError as error:
                    code=getattr(error,'provider_error_code',None) or getattr(error,'code',None)
                    event=dict(attempt=number,status='error',kind=error.kind,code=code,
                        http_status=getattr(error,'http_status',None),retryable=bool(error.retryable))
                    self.append(target/'attempts.jsonl',event)
                    if allow_rejection and code in ('data_inspection_failed','input_inspection_failed'):
                        result=dict(status='provider_rejection',raw_response='',seed=self.SEED,requested_model=self.MODEL,
                            returned_model=None,hit_token_limit=False,request_sha256=key,finish_reason='provider_rejection')
                        c.freeze(target/'response.json',result);return result
                    if not event['retryable'] or number==attempts:raise
                    time.sleep(min(2**number,8));continue
                self.append(target/'attempts.jsonl',event)
                c.freeze(target/'response.json',result);return result

    @staticmethod
    def append(path,row):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            import os
            os.fsync(f.fileno())

class APIModel:
    def __init__(self,root,job,workers=4,expected=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.gateway=Gateway(job);self.workers=workers;self.fatal_errors=[];self.lock=threading.Lock()
        self.expected_model=expected or self.gateway.expected
        import torch
        self.torch=torch
        self.environment=dict(model=self.gateway.MODEL,seed=job['seed'],expected_model=self.expected_model,
            native_temperature_policy='omitted' if job['backbone']=='gpt5' else 'native',api_workers=workers)
    def log(self,name,value):
        with self.lock:self.gateway.append(self.root/name,value)
    def invoke(self,request,stage='memory',record_id=None):
        request=self.gateway.wire_from_native(request)
        for attempt in range(1,4):
            try:
                result=self.gateway.chat_request(self.gateway.config,request)
                self.log('api_attempts.jsonl',dict(status='ok',stage=stage,id=record_id,attempt=attempt,
                    request_sha256=c.digest(request),response_id=result['response_id']))
                self.log('api_responses.jsonl',dict(request=request,output=result,stage=stage,id=record_id))
                return result
            except self.gateway.APIError as exc:
                failure=dict(status='error',stage=stage,id=record_id,attempt=attempt,request_sha256=c.digest(request),
                    error_type=exc.kind,retryable=exc.retryable,http_status=getattr(exc,'http_status',None),
                    provider_error_code=getattr(exc,'provider_error_code',None))
                self.log('api_attempts.jsonl',failure)
                if not exc.retryable or attempt==3:
                    with self.lock:self.fatal_errors.append(failure)
                    raise
                time.sleep(2*attempt)
    def one(self,messages,cap,temperature):
        return self.invoke(self.gateway.build_message_request(messages,cap,temperature))
    def generate(self,messages,max_new_tokens,sampling_temperature=0.):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda m:self.one(m,max_new_tokens,sampling_temperature),messages))
