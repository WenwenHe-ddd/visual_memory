"""Auditable GPT5 native-tool transport for M2A, with original image bytes."""
import base64
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from scripts.evaluation import gpt5_text_memory_transport as provider
from scripts.memory.m2a_local import append_json, safe_content
from scripts.memory.m2a_length_prompt import constrain_messages, AGENT_LIMIT, OPEN_LIMIT, MCQ_LIMIT

MODEL = provider.MODEL
DEFAULT_ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')


def image_part(block):
    from PIL import Image
    ref = block.get('url') or block.get('image_url') or block.get('image')
    if isinstance(ref, dict):
        ref = ref['url']
    if ref.startswith('data:'):
        data = base64.b64decode(ref.split(',', 1)[1], validate=True)
        metadata = {}
    else:
        data = Path(ref).read_bytes()
        metadata = {'path': ref}
    with Image.open(io.BytesIO(data)) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('Animated images require a separate explicit frame policy')
        mime = {'JPEG':'image/jpeg','PNG':'image/png','WEBP':'image/webp','GIF':'image/gif'}.get(im.format)
        if mime is None:
            raise ValueError('Unsupported original image format')
    metadata.update(bytes_sha256=hashlib.sha256(data).hexdigest(), mime_type=mime, bytes=len(data))
    return {'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(data).decode(),
                                         'detail':'high'}}, metadata


def build_request(messages, schemas, max_tokens):
    prepared, image_audit, pending_images = [], [], []
    roles = {'human':'user','ai':'assistant','system':'system','tool':'tool'}

    def flush_tool_images():
        if pending_images:
            prepared.append({'role':'user','content':[{'type':'text','text':
                'Images returned by the preceding memory tools, with matching attachment labels:'}]+pending_images[:]})
            pending_images.clear()

    for message in messages:
        role = roles[message.type]
        if role != 'tool':
            flush_tool_images()
        entry = {'role':role, 'content':message.content}
        if getattr(message, 'tool_calls', None):
            entry['tool_calls'] = [{'id':c['id'],'type':'function','function':{
                'name':c['name'],'arguments':json.dumps(c['args'],ensure_ascii=False)}} for c in message.tool_calls]
        if role == 'tool':
            entry['tool_call_id'] = message.tool_call_id
        if isinstance(message.content, list):
            parts = []
            for block in message.content:
                if block.get('type') in ('image','image_url'):
                    visual, metadata = image_part(block)
                    image_audit.append(metadata)
                    if role == 'tool':
                        label = f'[Tool image attachment {len(image_audit)}]'
                        parts.append({'type':'text','text':label})
                        pending_images.extend([{'type':'text','text':label}, visual])
                    else:
                        parts.append(visual)
                elif block.get('type') == 'text':
                    parts.append({'type':'text','text':block['text']})
                else:
                    raise ValueError('Unsupported message block')
            entry['content'] = '\n'.join(p['text'] for p in parts) if role == 'tool' else parts
        prepared.append(entry)
    flush_tool_images()
    wire = dict(model=MODEL, messages=prepared, max_completion_tokens=max(2048, max_tokens),
                n=1, stream=False, reasoning_effort='minimal', store=False, seed=42)
    if schemas:
        wire.update(tools=schemas, tool_choice='auto', parallel_tool_calls=False)
    return wire, image_audit


def chat(config, wire, timeout=120):
    provider._validate_config(config)
    if (wire['model'] != MODEL or wire['seed'] != 42 or wire['reasoning_effort'] != 'minimal'
            or 'temperature' in wire or wire['stream'] is not False):
        raise ValueError('GPT5 request configuration differs')
    req = urllib.request.Request(config.base_url+'/chat/completions',
        data=json.dumps(wire,ensure_ascii=False,allow_nan=False).encode(), method='POST',
        headers={'Authorization':'Bearer '+config.api_key,'Content-Type':'application/json','Accept':'application/json'})
    start = time.monotonic()
    try:
        with urllib.request.build_opener(provider._NoRedirect()).open(req,timeout=timeout) as response:
            body = response.read()
            request_id = response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        finally:
            exc.close()
        error = provider.classify_http_error(exc.code,body)
        error.provider_error_code = None
        try:
            code = json.loads(body).get('error',{}).get('code')
            if code in ('data_inspection_failed','input_inspection_failed'):
                error.provider_error_code = code
        except (ValueError,AttributeError,UnicodeError):
            pass
        raise error from None
    except (urllib.error.URLError,OSError):
        raise provider.APIError('network',retryable=True,message='GPT5 request failed or timed out') from None
    try:
        data = json.loads(body)
        assert len(data['choices']) == 1 and data['id']
        choice = data['choices'][0]
        message = choice['message']
        raw = message.get('content') or ''
        assert isinstance(raw,str)
        usage = data.get('usage') or {}
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        if not provider.valid_returned_model(data['model']):
            raise provider.APIError('model_mismatch',retryable=False,message='Unexpected returned model')
        allowed = {s['function']['name'] for s in wire.get('tools',[])}
        calls = []
        for call in message.get('tool_calls') or []:
            args = json.loads(call['function']['arguments'])
            assert call['function']['name'] in allowed and isinstance(args,dict) and call['id']
            calls.append(dict(name=call['function']['name'],args=args,id=call['id'],type='tool_call'))
        assert choice['finish_reason'] in ('stop','tool_calls','length','content_filter')
    except (AssertionError,KeyError,TypeError,ValueError):
        raise provider.APIError('protocol',retryable=False,message='Invalid GPT5 response structure') from None
    return dict(raw_response=raw.strip(),tool_calls=calls,requested_model=MODEL,returned_model=data['model'],
        response_id=data['id'],request_id=request_id,system_fingerprint=data.get('system_fingerprint'),
        finish_reason=choice['finish_reason'],refusal=message.get('refusal'),usage=usage,
        input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),
        hit_token_limit=choice['finish_reason']=='length',seed=42,reasoning_effort='minimal',reasoning_tokens=reasoning,
        model_identity_kind='provider_alias',resolved_model_revision=data['model'],
        elapsed_seconds=time.monotonic()-start,request_sha256=provider.request_sha256(wire))


class GPT5:
    def __init__(self, model_path, out, device='cuda:0', seed=42, max_new_tokens=1200,
                 image_max_pixels=262144, env_file=DEFAULT_ENV, cache_dir=None):
        if seed != 42:
            raise ValueError('This comparison fixes seed=42')
        self.config = provider.load_config(env_file)
        self.out = Path(out)
        self.cache_dir = Path(cache_dir) if cache_dir else self.out/'api_cache'
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.max_new_tokens, self.schemas = max_new_tokens, []
        self.stage, self.sample_id, self.counter = 'init', None, [0]

    def bind_tools(self, tools, **kwargs):
        from langchain_core.utils.function_calling import convert_to_openai_tool
        bound = copy.copy(self)
        bound.schemas = [convert_to_openai_tool(t) for t in tools]
        return bound

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        if self.stage != 'construction':
            instruction = (MCQ_LIMIT if self.max_new_tokens == 16 else OPEN_LIMIT) if self.stage == 'reader' else AGENT_LIMIT
            messages = constrain_messages(messages,instruction)
        elif self.schemas:
            raise ValueError('Construction cannot bind tools')
        wire, image_audit = build_request(messages,self.schemas,self.max_new_tokens)
        key = provider.request_sha256(wire)
        folder = self.cache_dir/key
        folder.mkdir(exist_ok=True)
        response_path = folder/'response.json'
        self.counter[0] += 1
        replay = response_path.exists()
        print(f'GPT5 {self.counter[0]} {self.stage} {self.sample_id} images={len(image_audit)} cached={replay}',flush=True)
        if replay:
            if json.loads(gzip.decompress((folder/'request.json.gz').read_bytes())) != wire:
                raise ValueError('Cached request does not match exact current wire')
            result = json.loads(response_path.read_text())
        else:
            (folder/'request.json.gz').write_bytes(gzip.compress(json.dumps(wire,ensure_ascii=False).encode(),mtime=0))
            for attempt in range(1,4):
                try:
                    result = chat(self.config,wire)
                    append_json(self.out/'api_attempts.jsonl',dict(sample_id=self.sample_id,stage=self.stage,
                        request_sha256=key,attempt=attempt,status='ok',response_id=result['response_id']))
                    temporary = folder/'response.tmp'
                    temporary.write_text(json.dumps(result,ensure_ascii=False)+'\n')
                    temporary.replace(response_path)
                    break
                except provider.APIError as exc:
                    append_json(self.out/'api_attempts.jsonl',dict(sample_id=self.sample_id,stage=self.stage,
                        request_sha256=key,attempt=attempt,status='error',kind=exc.kind,
                        http_status=exc.http_status,retryable=exc.retryable,
                        provider_error_code=getattr(exc,'provider_error_code',None)))
                    if not exc.retryable or attempt == 3:
                        raise
                    time.sleep(attempt*2)
        valid = (provider.valid_returned_model(result['returned_model']) and result['finish_reason'] in ('stop','tool_calls')
                 and not result.get('refusal') and (result['raw_response'] or result['tool_calls']))
        record = dict(result,call=self.counter[0],stage=self.stage,sample_id=self.sample_id,
            messages=safe_content([m.model_dump() for m in messages]),tools=self.schemas,
            images=image_audit,request=safe_content(wire),cache_replayed=replay,cache_directory=str(folder),
            status='ok' if valid else 'error')
        append_json(self.out/'llm_calls.jsonl',record)
        if not valid:
            raise ValueError('GPT5 response incomplete, refused, empty or model mismatch; retained in logs')
        return AIMessage(content=result['raw_response'],tool_calls=result['tool_calls'])
