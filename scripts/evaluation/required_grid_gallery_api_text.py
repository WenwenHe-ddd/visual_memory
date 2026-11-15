"""Fresh native Gallery text memories on fixed same-backbone history captions."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_api import Gateway,APIModel,ENV
from scripts.evaluation import required_grid_gallery_common as shared
OLD=c.ROOT/'scripts/evaluation/mem_gallery_deepseek_text_methods.py'
REFS={
 'gpt5':dict(gen_agents='mem_gallery_gpt5_gen_agents_20261028',memoryos='mem_gallery_gpt5_memoryos_20261030',
  simplemem='mem_gallery_gpt5_simplemem030_20261031',a_mem='mem_gallery_amem_backbones_20261104/gpt5'),
 'deepseek':dict(gen_agents='mem_gallery_deepseek_text_methods_20261031/gen_agents',memoryos='mem_gallery_deepseek_text_methods_20261031/memoryos',
  simplemem='mem_gallery_deepseek_text_methods_20261031/simplemem',a_mem='mem_gallery_amem_backbones_20261104/deepseek')}

def runtime(job):
    native=c.clone(OLD,'grid_gallery_api_text_native')
    def model(root,env,workers,floor):
        class Strict(APIModel):
            def one(self,messages,cap,temperature):
                if self.fatal_errors:raise RuntimeError('Stopped after terminal memory failure')
                out=super().one(messages,max(cap,floor),temperature)
                if out['finish_reason']!='stop' or out.get('refusal') or not out['raw_response'].strip():
                    self.fatal_errors.append(dict(error_type='invalid_completion',response_id=out['response_id']))
                    raise ValueError('Native invalid memory response retained')
                return out
        return Strict(root,job,job['workers'])
    native.memory_api=SimpleNamespace(Model=model)
    native.reader_api=SimpleNamespace(MODEL=c.MODELS[job['backbone']])
    native.verify=shared.verify_protocol
    if job['method']=='a_mem':
        from scripts.evaluation.mem_gallery_amem_api import AMem
        native.NativeMemory=AMem
    return native

def prepare(job,native):
    root=Path(job['method_root'])/'native';ref=c.ROOT/'results'/REFS[job['backbone']][job['method']]
    if (root/'protocol.json').exists():return shared.verify_protocol(root)
    parent=c.read(ref/'protocol.json');g=Gateway(job)
    if parent['seed']!=42 or parent['expected_model']!=g.expected:raise ValueError('Wrong model-matched reference')
    questions=c.rows(ref/'inputs/questions.jsonl');obs=c.read(ref/'observations.json');shared.verify_inputs(questions,obs)
    for q in questions:
        expected=q['question']+("\nquestion's image:\nimage_caption: "+q['query_image_caption'] if q['query_image_caption'] else '')
        if q['retrieval_query']!=expected:raise ValueError('Native retrieval query changed')
    p=copy.deepcopy(parent)
    for key in ('experiment_signature','source_sha256','input_sha256','recovery'):p.pop(key,None)
    p.update(seed=job['seed'],mode=job['method'],backbone=job['backbone'],requested_model=g.MODEL,expected_model=g.expected,
        task_workers=job['workers'],api_workers=job['workers'],embedding_device='cpu',
        memory_output_floor=parent.get('memory_output_floor',0 if job['method']=='simplemem' else 2048),
        caption_source_run_seed=42,caption_policy='Fixed original same-backbone cached history captions, verbatim.',
        generated_memory_reused=False,prior_answers_reused=False,
        study='Required seed grid: fresh native construction/retrieval/reader; evaluator seed42')
    files=[Path(__file__),OLD,Path(shared.__file__),c.ROOT/'scripts/evaluation/required_grid_api.py',
        ref/'protocol.json',ref/'observations.json',ref/'inputs/questions.jsonl',ref/'inputs/gold.jsonl',
        Path(native.adapters.__file__),Path(native.native.__file__),Path(native.official.__file__)]
    files += [f for f in Path(p['embedding_model_path']).rglob('*') if f.is_file()]
    if job['method']=='a_mem':files.append(c.ROOT/'scripts/evaluation/mem_gallery_amem_api.py')
    p['source_sha256']={str(f.resolve()):c.sha(f) for f in files};p['experiment_signature']=c.digest(p)
    c.freeze(root/'observations.json',obs);c.write_rows(root/'inputs/questions.jsonl',questions)
    c.write_rows(root/'inputs/gold.jsonl',c.rows(ref/'inputs/gold.jsonl'));c.freeze(root/'protocol.json',p)
    return shared.verify_protocol(root)

def audit(job,root,p,native):
    c.hashes(p['source_sha256']);qs=c.rows(root/'inputs/questions.jsonl');obs=c.read(root/'observations.json')
    retrieved=c.rows(root/'memory/retrieval.jsonl');rmap={r['id']:r for r in retrieved}
    if len(retrieved)!=1711 or set(rmap)!={q['id'] for q in qs}:raise ValueError('Incomplete native retrieval')
    done=c.read(root/'memory/complete.json');c.hashes(done['artifact_sha256'])
    if done['experiment_signature']!=p['experiment_signature']:raise ValueError('Wrong native run')
    for task,group in obs.items():
        stats=c.read(root/'memory'/task/'memory_stats.json')
        if stats['stored_rounds']!=len(group):raise ValueError('Original memory rounds missing')
    for q in qs:
        r=rmap[q['id']]
        if r['retrieval_query']!=q['retrieval_query'] or r['experiment_signature']!=p['experiment_signature']:
            raise ValueError('Query/protocol changed')
        ids=r['retrieval_detail'].get('context_memory_ids')
        if job['method'] in ('a_mem','simplemem') and (ids is None or len(ids)!=len(set(ids)) or len(ids) not in (0,5)):
            raise ValueError('Native final-five context policy changed')
        if job['method']=='a_mem':
            from scripts.evaluation.mem_gallery_amem_api import render_notes
            notes=c.read(root/'memory'/q['task']/'final_memory.json')
            if len(ids)!=5 or render_notes(notes,ids)!=r['retrieved_context']:raise ValueError('A-Mem notes/context mismatch')
    raw={r['output']['response_id']:r for r in c.rows(root/'memory/api_responses.jsonl')}
    for path in (root/'memory/calls').glob('*.json'):
        value=c.read(path);request=value['request'];out=value['output'];logged=raw.get(out['response_id'])
        if (request['seed']!=job['seed'] or out['seed']!=job['seed'] or logged is None
            or out['returned_model']!=p['expected_model'] or logged['request']['seed']!=job['seed']
            or out['request_sha256']!=c.digest(logged['request']) or logged['output']['raw_response']!=out['raw_response']):
            raise ValueError('Native memory output lacks new-seed actual-wire provenance')
    proof=dict(status='passed',seed=job['seed'],questions=1711,memory_rounds=3962,native_memory_calls=len(list((root/'memory/calls').glob('*.json'))))
    c.dump(root/'independent_audit.json',proof);return proof

def stage(job,phase):
    native=runtime(job);p=prepare(job,native);root=Path(job['method_root'])/'native'
    if phase=='native':native.memory_stage(root,ENV);audit(job,root,p,native);return
    if phase=='reader':
        audit(job,root,p,native)
        return shared.api_answers(job,c.rows(root/'inputs/questions.jsonl'),c.rows(root/'memory/retrieval.jsonl'),root/'reader',p['max_new_tokens'])
    if phase!='evaluation':raise ValueError('Unknown stage')
    audit(job,root,p,native)
    return shared.publish(job,c.rows(root/'reader/responses.jsonl'),
        [root/n for n in ('protocol.json','memory/complete.json','memory/retrieval.jsonl','reader/responses.jsonl','reader/manifest.jsonl','independent_audit.json')],
        dict(caption_source_run_seed=42))

def run(job,path):
    if job['dataset']!='memgallery' or job['backbone'] not in REFS:raise ValueError('Wrong API Gallery handler')
    c.job_protocol(job,[Path(__file__),OLD,Path(shared.__file__),c.ROOT/'scripts/evaluation/required_grid_api.py'])
    for phase in ('native','reader','evaluation'):
        print(dict(id=job['id'],stage=phase),flush=True)
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        n=runtime(j);prepare(j,n);print(dict(status='prepared',id=j['id'],api_calls=0))
    elif a.stage:stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
