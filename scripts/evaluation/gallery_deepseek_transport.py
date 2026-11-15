"""Seed42 DeepSeek V4.1 Flash transport with byte-preserving Gallery image support."""
import argparse
import http.client
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from scripts.evaluation import mem_gallery_gpt5_gen_agents as common
from scripts.evaluation import mem_gallery_gpt5_image_formats as image_wire
from scripts.evaluation import deepseek_flash_seed42_audited_transport as provider

MODEL = provider.DEEPSEEK_MODEL
ENDPOINT = provider.DASHSCOPE_BASE_URL + '/chat/completions'
request_sha256 = provider.request_sha256


def wire_for(payload, cap=2048):
    if set(payload) - {'record_id', 'user_text', 'system', 'query_images'}:
        raise ValueError('Unexpected generation payload fields')
    # Reuse only the verified original-byte message builder; no OpenAI request
    # is sent, and credentials are bound separately to the DeepSeek endpoint.
    messages = image_wire.wire_for(payload, cap)['messages']
    return dict(model=MODEL, messages=messages, temperature=0, max_tokens=cap,
                n=1, stream=False, enable_thinking=False, seed=42)


def validate(config, request):
    provider._validate_config(config)
    assert set(request) == {'model','messages','temperature','max_tokens','n','stream','enable_thinking','seed'}
    assert request['model'] == MODEL and type(request['seed']) is int and request['seed'] == 42
    assert request['temperature'] == 0 and request['enable_thinking'] is False
    assert request['n'] == 1 and request['stream'] is False
    assert type(request['max_tokens']) is int and request['max_tokens'] > 0
    ms = request['messages']
    assert isinstance(ms, list) and len(ms) in (1, 2)
    if len(ms) == 2:
        assert set(ms[0]) == {'role','content'} and ms[0]['role'] == 'system' and isinstance(ms[0]['content'], str)
    user = ms[-1]
    assert set(user) == {'role','content'} and user['role'] == 'user'
    parts = user['content']
    assert isinstance(parts, list) and parts
    assert set(parts[-1]) == {'type','text'} and parts[-1]['type'] == 'text' and isinstance(parts[-1]['text'], str)
    for part in parts[:-1]:
        assert set(part) == {'type','image_url'} and part['type'] == 'image_url'
        item = part['image_url']
        assert set(item) == {'url','detail'} and item['detail'] == 'high'
        assert item['url'].startswith(tuple('data:image/'+kind+';base64,' for kind in ('jpeg','png','webp','gif')))


def chat(config, request):
    validate(config, request)
    req = urllib.request.Request(ENDPOINT, data=json.dumps(request,ensure_ascii=False,allow_nan=False).encode(),
        method='POST',headers={'Authorization':'Bearer '+config.api_key,'Content-Type':'application/json','Accept':'application/json'})
    started=time.monotonic()
    try:
        with urllib.request.build_opener(provider._NoRedirect()).open(req,timeout=120) as response:
            body=response.read();request_id=response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:body=exc.read(65536)
        finally:exc.close()
        error=provider._http_error(exc.code,body)
        error.provider_error_code=None
        try:
            if json.loads(body).get('error',{}).get('code')=='data_inspection_failed':
                error.provider_error_code='data_inspection_failed'
        except (ValueError,AttributeError,UnicodeError):pass
        raise error from None
    except (urllib.error.URLError,OSError,http.client.HTTPException):
        raise provider.APIError('network',retryable=True,message='DeepSeek network request failed') from None
    try:
        d=json.loads(body);assert len(d['choices'])==1
        c=d['choices'][0];message=c['message'];refusal=message.get('refusal')
        raw=message.get('content')
        if raw is None and refusal:raw=''
        assert isinstance(raw,str) and isinstance(c['finish_reason'],str)
        assert isinstance(d['id'],str) and d['id']
        usage=d.get('usage') or {};reasoning=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    except (AssertionError,KeyError,ValueError,TypeError,AttributeError):
        raise provider.APIError('protocol',retryable=False,message='Invalid DeepSeek response') from None
    if d['model']!=MODEL:
        raise provider.APIError('model_mismatch',retryable=False,message='Unexpected DeepSeek model identifier')
    if message.get('reasoning_content') or reasoning not in (None,0):
        raise provider.APIError('unexpected_reasoning',retryable=False,message='Reasoning returned despite thinking=false')
    return dict(raw_response=raw,requested_model=MODEL,returned_model=d['model'],
        model_identity_kind='provider_alias',resolved_model_revision=None,
        response_id=d['id'],request_id=request_id,finish_reason=c['finish_reason'],refusal=refusal,
        system_fingerprint=d.get('system_fingerprint'),usage=usage,
        input_tokens=usage.get('prompt_tokens',0),generated_tokens=usage.get('completion_tokens',0),
        seed=42,temperature=0,enable_thinking=False,reasoning_tokens=reasoning,reasoning_content_present=False,
        request_sha256=request_sha256(request),hit_token_limit=c['finish_reason']=='length',elapsed_seconds=time.monotonic()-started)


class Model:
    def __init__(self,root,env_file,expected_model=MODEL):
        assert expected_model==MODEL
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.config=provider.load_config(env_file)
        self.lock=threading.Lock();self.fatal_errors=[]

    def log(self,name,value):
        with self.lock:
            with (self.root/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')

    def invoke(self,request,stage,record_id=None):
        key=request_sha256(request)
        for attempt in range(1,4):
            if self.fatal_errors:raise RuntimeError('Stopped after fatal DeepSeek error; see api_attempts.jsonl')
            try:
                out=chat(self.config,request)
                self.log('api_attempts.jsonl',dict(stage=stage,id=record_id,request_sha256=key,
                    attempt=attempt,status='ok',response_id=out['response_id']))
                self.log('api_responses.jsonl',dict(stage=stage,id=record_id,output=out))
                return out
            except provider.APIError as exc:
                failure=dict(stage=stage,id=record_id,request_sha256=key,attempt=attempt,status='error',
                    error_type=exc.kind,http_status=exc.http_status,retryable=exc.retryable,
                    provider_error_code=getattr(exc,'provider_error_code',None))
                self.log('api_attempts.jsonl',failure)
                if not exc.retryable or attempt==3:
                    with self.lock:self.fatal_errors.append(failure)
                    raise
                time.sleep(2*attempt)


def probe(output,env_file):
    from concurrent.futures import ThreadPoolExecutor
    source=common.DEFAULT_OUTPUT
    jobs=common.rows(source/'caption_jobs.jsonl')
    payloads={}
    prompt=common.read(source/'protocol.json')['caption_prompt']
    for job in jobs:
        payload=dict(record_id=job['id'],user_text=prompt,query_images=[dict(path=job['path'],sha256=job['sha256'])])
        wire=wire_for(payload)
        mime=wire['messages'][-1]['content'][0]['image_url']['url'].split(';')[0]
        payloads.setdefault(mime,payload)
    model=Model(output/'probe',env_file)
    def one(item):
        mime,payload=item;wire=wire_for(payload)
        out=model.invoke(wire,'probe',mime)
        assert out['finish_reason']=='stop' and not out['refusal'] and out['raw_response'].strip()
        return dict(mime=mime,payload=payload,output=out)
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(one,payloads.items()))
    common.dump(output/'api_probe.json',dict(status='passed',model=MODEL,requests=result))
    print(dict(status='passed',model=MODEL,formats=list(payloads)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path,default=common.DEFAULT_ENV)
    args=parser.parse_args();probe(args.output.resolve(),args.env_file)
