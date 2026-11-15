import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from scripts.evaluation import gallery_deepseek_transport as api
from scripts.evaluation import mem_gallery_deepseek_caption_visual_master as run
p=run.OUTPUT
protocol=run.verify(p)
job=next(j for j in run.rows(p/'caption_jobs.jsonl') if j['id']=='29d1b35ff3f1fd21daeeb3a47f23f088c5bd3e9316bbf3c13ac37708aeabbab1')
wire=api.wire_for(run.caption_payload(job,protocol))
config=api.provider.load_config(run.ga.DEFAULT_ENV)
api.validate(config,wire)
request=urllib.request.Request(api.ENDPOINT,data=json.dumps(wire,ensure_ascii=False).encode(),method='POST',headers={'Authorization':'Bearer '+config.api_key,'Content-Type':'application/json'})
try:
    with urllib.request.build_opener(api.provider._NoRedirect()).open(request,timeout=120) as response:
        data=json.loads(response.read());result=dict(status='unexpected_success',returned_model=data.get('model'),response_id=data.get('id'))
except urllib.error.HTTPError as exc:
    data=json.loads(exc.read(65536));err=data.get('error',{})
    message=str(err.get('message',''))
    message=re.sub(r'(?:https?://|data:)[^\s]+','[url omitted]',message)
    message=re.sub(r'[A-Za-z0-9+/=_-]{80,}','[long value omitted]',message)
    result=dict(status='rejected',http_status=exc.code,error_code=err.get('code'),error_type=err.get('type'),message=message[:1000])
run.dump(p/'image_limit_diagnostic.json',result)
print(result)
