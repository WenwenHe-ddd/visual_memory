"""Retry one frozen request, retaining a redacted provider error code if rejected.

Do not change model, prompt, seed, or payload. A successful response is persisted
with the original run provenance; a rejection never becomes a fake answer.
"""
from pathlib import Path
import json
import time
import uuid
import urllib.error
import urllib.request
from unittest.mock import patch
from scripts.analysis.memeye_deepseek_flash_seed42_zscore import (
    OUTPUT, MODEL, verify_reader, transport, digest, append, dump, choice, validate_response,
)


def main():
    output = OUTPUT
    p, records = verify_reader(output);c = p['contract']
    record_id = 'Outdoor_Navigation_Route_Memory_Assistant/Q9::master_answer_query_zscore'
    row = next(r for r in records if r['record_id'] == record_id)
    payload = {k:row[k] for k in ('system','user_text','query_images')}
    body = transport.build_request(payload,model=MODEL,max_output_tokens=16,image_detail='high')
    assert digest(body) == row['request_sha256'] and body['seed'] == 42
    config = transport.load_config(c['api_env_file'])
    execution_id = str(uuid.uuid4())
    detail = {}
    opener_factory = urllib.request.build_opener

    class AuditedOpener:
        def __init__(self, original): self.original = original
        def open(self, *args, **kwargs):
            try:
                return self.original.open(*args, **kwargs)
            except urllib.error.HTTPError as error:
                raw = error.read(65536)
                try:
                    obj = json.loads(raw); err = obj.get('error') or obj
                    for key in ('code','type','message'):
                        v = err.get(key)
                        if isinstance(v,(str,int)):
                            detail[key] = str(v).replace(config.api_key,'[REDACTED]')[:500]
                except (ValueError,AttributeError):
                    detail['unparsed_body'] = True
                # Preserve transport's normal HTTP status handling without
                # logging headers, credentials, or complete provider bodies.
                import io
                raise urllib.error.HTTPError(error.url,error.code,'API rejected request',{},io.BytesIO(raw)) from None

    attempt = {'record_id':record_id,'request_sha256':row['request_sha256'],
        'execution_id':execution_id,'attempt':1,'diagnostic_retry':True}
    try:
        with patch('urllib.request.build_opener',side_effect=lambda *a,**kw:AuditedOpener(opener_factory(*a,**kw))):
            response = transport.chat_request(config,body,timeout=c['timeout_seconds'])
    except transport.APIError as error:
        append(output/'reader/api_attempts.jsonl',{**attempt,'status':'api_error','kind':error.kind,
            'http_status':error.http_status,'message':error.message,'provider_error':detail})
        result={'status':'rejected','record_id':record_id,'http_status':error.http_status,
            'request_sha256':row['request_sha256'],'unchanged_seed42_request':True,'provider_error':detail}
    else:
        valid = choice(response['raw_response']) is not None and response['finish_reason']=='stop' and not response.get('refusal')
        if not valid: raise ValueError('Diagnostic retry returned an invalid answer')
        ans={**response,'record_id':record_id,'question_id':row['question_id'],'condition':row['condition'],
            'payload_sha256':row['payload_sha256'],'request_sha256':row['request_sha256'],
            'run_id':p['run_id'],'experiment_signature':p['experiment_signature'],'execution_id':execution_id,
            'attempts':1,'retry_count':0,'status':'ok','predicted_choice':choice(response['raw_response']),
            'historical_response_reuse':False}
        validate_response(ans,row,p,'reader')
        append(output/'reader/fresh_responses.jsonl',ans)
        append(output/'reader/api_attempts.jsonl',{**attempt,'status':'ok','response_id':ans['response_id']})
        result={'status':'ok','record_id':record_id,'unchanged_seed42_request':True}
    result['script_sha256']=transport.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(output/'reader/provider_error_diagnostic.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
