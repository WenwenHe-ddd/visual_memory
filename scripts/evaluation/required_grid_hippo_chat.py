"""Native HippoMM role interface, original provider images, explicit wire seed."""
from pathlib import Path
import gzip
import json
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway

def chat_class(job):
    g=Gateway(job)
    class Chat:
        def __init__(self,directory,backbone=None,env_file=None,token_limit=2048,allow_api=True):
            self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
            self.backbone=backbone or job['backbone'];self.token_limit=token_limit;self.allow_api=allow_api
            if self.backbone!=job['backbone'] or token_limit!=2048:raise ValueError('HippoMM native backbone/budget changed')
            self.last_call_id=None;self.index={}
            for path in self.directory.glob('call_*.json'):
                row=c.read(path);key=c.digest(row['request'])
                if key in self.index:raise ValueError('Duplicate native role request')
                self.index[key]=path.stem
        def __call__(self,role,prompt,image_paths=()):
            from scripts.evaluation.mem_gallery_master_reader import READER_SYSTEM
            paths=[str(p) for p in image_paths]
            request=dict(role=role,prompt=prompt,image_paths=paths,backbone=self.backbone,max_output_tokens=self.token_limit,seed=job['seed'])
            encodings=[]
            if job['dataset']=='memgallery':
                from scripts.memory import hippomm_gallery_transport as native
                api=native.gpt if self.backbone=='gpt5' else native.deep
                wire,encodings=native.build_wire(api,self.backbone,prompt,paths,self.token_limit)
                wire=g.wire_from_native(wire)
            else:
                wire=g.build_request(dict(system=READER_SYSTEM,user_text=prompt,
                    query_images=[dict(path=p,sha256=c.sha(p)) for p in paths]),self.token_limit)
            key=c.digest(request);call_id=self.index.get(key,f'call_{len(self.index)+1:04d}')
            path=self.directory/(call_id+'.json');wirepath=self.directory/(call_id+'.wire.json.gz')
            if path.exists():
                row=c.read(path)
                if row['request']!=request or row['request_sha256']!=c.digest(wire) or json.loads(gzip.decompress(wirepath.read_bytes()))!=wire:
                    raise ValueError('Cached native role/wire differs')
            else:
                if not self.allow_api:raise ValueError('Offline role replay requires an existing call')
                out=g.call(self.directory/'wire_cache',wire,attempts=4,
                    allow_rejection=job['dataset']=='memgallery' and self.backbone=='deepseek')
                wirepath.write_bytes(gzip.compress(json.dumps(wire,ensure_ascii=False).encode(),mtime=0))
                row=dict(call_id=call_id,request=request,response=out,request_sha256=c.digest(wire),
                    wire_file_sha256=c.sha(wirepath),image_sha256={p:c.sha(p) for p in paths},image_encoding=encodings)
                c.freeze(path,row);self.index[key]=call_id
            self.last_call_id=call_id;out=row['response']
            if out.get('status')=='provider_rejection':
                from scripts.memory.hippomm_gallery_transport import ProviderRejected
                raise ProviderRejected('Provider rejection retained: '+str(path))
            if out['seed']!=job['seed'] or out['returned_model']!=g.expected:raise ValueError('Wrong actual role seed/model')
            if out['finish_reason']!='stop' or out.get('refusal') or not out['raw_response'].strip():raise ValueError('Native role response incomplete; raw retained')
            return out['raw_response']
    return Chat
