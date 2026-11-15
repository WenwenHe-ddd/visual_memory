"""Resume the Gallery official run after two diagnosed construction failures.

Keep all completed scenarios and original calls. A deterministic location-field
serialization repair preserves every other field; one truncated extraction is
regenerated with explicit finite-output instructions, using original input only.
"""
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from scripts.evaluation import mem_gallery_simplemem_official as run
from scripts.memory.text_memory_transport import Broker

ROOT=run.OUTPUT
LOCATION='8a875341162362563787044467b1c005eb57427b6dd09c5ed3688ba60df734ce'
TRUNCATED='35e39e3614a45782a341b53f3c33a7ab25e4dad9f5d8a259a6d5388555855a00'
INSTRUCTION=('Your earlier extraction exceeded the output length limit. Generate a complete, concise JSON array from ONLY the supplied dialogue window. '
'Include each supported fact once; do not enumerate imagined observations, times, or repeated variants. '
'Keep lossless_restatement concise and keywords short. All scalar fields (timestamp, location, topic) must be a single string or null, never an array. '
'Close the JSON array and stop. Do not add Markdown or explanations.')

def repair_location(raw):
    text=raw.strip()
    if text.startswith('```json') and text.endswith('```'):
        text=text[len('```json'):-3].strip()
    values=json.loads(text);assert isinstance(values,list)
    fixed=copy.deepcopy(values);changes=[]
    for i,entry in enumerate(fixed):
        value=entry.get('location')
        if isinstance(value,list):
            assert value and all(isinstance(s,str) for s in value)
            entry['location']='; '.join(value);changes.append(dict(index=i,before=value,after=entry['location']))
    assert changes
    return json.dumps(fixed,ensure_ascii=False),changes

def main():
    while subprocess.run(['systemctl','--user','is-active','--quiet','memgallery-simplemem-qwen-20261030.service']).returncode==0:
        time.sleep(30)
    base=run.base;root=ROOT/'simplemem'
    run.verify(ROOT,run.METHOD)
    originals={k:base.read(root/'calls'/(k+'.json')) for k in (LOCATION,TRUNCATED)}
    raw=originals[LOCATION]['output']['raw_response']
    corrected,changes=repair_location(raw)
    assert originals[TRUNCATED]['output']['hit_token_limit']
    amendment=dict(status='recovery_started',source_sha256=base.sha(Path(__file__)),original_calls={k:base.sha(root/'calls'/(k+'.json')) for k in originals},location_field_changes=changes,truncated_retry_instruction=INSTRUCTION,policy='Preserve successful scenarios and cached outputs. Serialize only list-valued location; retry one truncated extraction using its original gold-free input. No changes to retriever, reflection, RRF, reader, model or token cap.')
    path=ROOT/'recovery_amendment.json'
    if path.exists():
        previous=base.read(path)
        assert previous['source_sha256']==amendment['source_sha256']
    else:base.dump(path,amendment)
    if (root/'run_errors.json').exists():
        archive=ROOT/'initial_run_errors.json'
        if not archive.exists():archive.write_bytes((root/'run_errors.json').read_bytes())
        (root/'run_errors.json').unlink()
    class RepairedMemory(run.GalleryMemory):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            original_parse=self.core.memory_builder._parse_llm_response
            def parse(response,ids):
                if response==raw:
                    result=original_parse(corrected,ids)
                    base.dump(self.workdir/'location_serialization_repair.json',dict(original_call=LOCATION,original_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),corrected_response=corrected,changes=changes,dialogue_ids=ids))
                    return result
                return original_parse(response,ids)
            self.core.memory_builder._parse_llm_response=parse
    original_request=Broker.request
    def request(self,messages,*args,**kwargs):
        if messages==originals[TRUNCATED]['request']['messages']:
            repaired=copy.deepcopy(messages)
            repaired[-1]['content']+='\n\n'+INSTRUCTION
            result=original_request(self,repaired,*args,**kwargs)
            base.dump(ROOT/'truncated_window_retry.json',dict(original_call=TRUNCATED,new_call=result['call_key'],hit_token_limit=result['hit_token_limit']))
            return result
        return original_request(self,messages,*args,**kwargs)
    Broker.request=request;base.Memory=RepairedMemory;base.verify=run.verify
    base.run(ROOT,run.METHOD,'cuda:0')
    run.analyze(ROOT)
    amendment['status']='complete';base.dump(path,amendment)
    result=base.read(ROOT/'final_results.json');result['recovery_amendment']=str(path);base.dump(ROOT/'final_results.json',result)

if __name__=='__main__':main()
