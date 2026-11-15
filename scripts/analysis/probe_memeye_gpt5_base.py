"""Verify the exact GPT-5 alias and seed42 before a fresh benchmark directory."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation import openai_gpt5_base_transport as transport
from scripts.evaluation.memeye_gpt5_base_captions import DEFAULT_OUTPUT,DEFAULT_ENV


def probe(output=DEFAULT_OUTPUT,env_file=DEFAULT_ENV):
    output=Path(output);path=output/'availability_probe.json'
    body=transport.build_request({'user_text':'Reply with the single uppercase letter A.','query_images':[]},seed=42,max_output_tokens=2048)
    if path.exists():
        result=json.loads(path.read_text())
        assert result['passed'] and result['request']==body
        assert transport.valid_returned_model(result['response']['returned_model'])
        return result
    config=transport.load_config(env_file)
    try:
        response=transport.chat_request(config,body)
        result={'passed':response['raw_response'].strip()=='A' and response['finish_reason']=='stop',
            'request':body,'response':response,'scope':'Synthetic availability and parameter probe; excluded from benchmark answers.'}
    except transport.APIError as error:
        result={'passed':False,'request':body,'error':{'kind':error.kind,'http_status':error.http_status,
            'code':error.code,'message':error.message,'retryable':error.retryable}}
    output.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert result['passed'],'Availability probe failed; inspect availability_probe.json before a paid full run'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--api-env-file',type=Path,default=DEFAULT_ENV)
    a=p.parse_args();r=probe(a.output,a.api_env_file)
    print(json.dumps({'passed':r['passed'],'requested_model':r['request']['model'],
        'returned_model':r['response']['returned_model'],'seed':r['request']['seed']}))
