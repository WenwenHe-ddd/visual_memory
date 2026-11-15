"""Explicit, audited continuation for MemoryOS profile completion overflow."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import threading

from scripts.evaluation import mem_gallery_deepseek_text_methods as run
from scripts.evaluation import gallery_deepseek_memory_transport as api

CEILING=32768


class BudgetModel(api.APIModel):
    def __init__(self,root,env_file,workers=8,min_output_tokens=8192):
        super().__init__(root,env_file,workers)
        self.min_output_tokens=min_output_tokens
        assert min_output_tokens==8192

    def one(self,messages,cap,temperature):
        if self.fatal_errors:raise RuntimeError('Terminal memory failure')
        cap=max(cap,self.min_output_tokens)
        out=super().one(messages,cap,temperature)
        if out['finish_reason']=='length' and not out.get('refusal') and cap<CEILING:
            before=out
            self.log('budget_retry_events.jsonl',dict(status='started',
                discarded_response_id=before['response_id'],discarded_request_sha256=before['request_sha256'],
                old_cap=cap,new_cap=CEILING))
            out=super().one(messages,CEILING,temperature)
            self.log('budget_retry_events.jsonl',dict(status='completed',
                discarded_response_id=before['response_id'],discarded_request_sha256=before['request_sha256'],
                replacement_response_id=out['response_id'],replacement_request_sha256=out['request_sha256'],
                old_cap=cap,new_cap=CEILING))
        if out['finish_reason']!='stop' or out.get('refusal') or not out['raw_response'].strip():
            with self.lock:self.fatal_errors.append(dict(error_type='invalid_completion',response_id=out['response_id']))
            raise RuntimeError('No complete memory response; partial output is not accepted')
        return out


def verify_extension(output,p):
    protocol=run.read(output/'budget_extension_protocol.json')
    assert protocol['base_protocol_sha256']==run.sha(output/'protocol.json')
    for name,digest in protocol['source_sha256'].items():assert run.sha(name)==digest,name
    root=output/'memory';calls=run.rows(root/'api_responses.jsonl')
    by_id={r['output']['response_id']:r for r in calls};assert len(by_id)==len(calls)
    index={r['output']['response_id']:i for i,r in enumerate(calls)}
    discarded={identity for identity,r in by_id.items() if r['output']['finish_reason']=='length'}
    for identity,r in by_id.items():
        w=r['request'];o=r['output']
        assert w==api.build_request(w['messages'],w['max_tokens'],w['temperature'])
        assert w['max_tokens'] in (8192,CEILING)
        assert o['request_sha256']==api.request_sha256(w)
        if identity in discarded:assert w['max_tokens']==8192 and o['generated_tokens']==8192
    events=run.rows(root/'budget_retry_events.jsonl') if (root/'budget_retry_events.jsonl').exists() else []
    completed=[e for e in events if e['status']=='completed'];replacements={}
    for e in completed:
        before,after=by_id[e['discarded_response_id']],by_id[e['replacement_response_id']]
        assert e['discarded_response_id'] in discarded
        assert index[e['discarded_response_id']]<index[e['replacement_response_id']]
        assert before['request']['max_tokens']==e['old_cap']==8192
        assert after['request']['max_tokens']==e['new_cap']==CEILING
        assert after['request']==dict(before['request'],max_tokens=CEILING)
        assert before['output']['request_sha256']==e['discarded_request_sha256']
        assert after['output']['request_sha256']==e['replacement_request_sha256']
        assert after['output']['finish_reason']=='stop'
        assert any(s['status']=='started' and s['discarded_response_id']==e['discarded_response_id'] for s in events)
        replacements[e['replacement_response_id']]=e
    assert {identity for identity,r in by_id.items() if r['request']['max_tokens']==CEILING}==set(replacements)
    for file in (root/'calls').glob('*.json'):
        assert run.read(file)['output']['response_id'] not in discarded
    return dict(by_id=by_id,discarded=discarded,replacements=replacements,
        summary=dict(discarded_truncated_completions=len(discarded),expanded_complete_responses=len(replacements),
                     original_floor=8192,overflow_ceiling=CEILING,final_cache_contains_no_truncation=True))


def audited_wire(request,out,p,proof):
    canonical=api.build_request(request['messages'],max(request['max_new_tokens'],p['memory_output_floor']),request['temperature'])
    actual=proof['by_id'][out['response_id']]['request']
    if actual!=canonical:
        assert out['response_id'] in proof['replacements']
        assert actual==dict(canonical,max_tokens=CEILING)
    assert out['response_id'] not in proof['discarded']
    return actual


def prepare(output):
    p=run.verify(output);assert p['mode']=='memoryos' and p['memory_output_floor']==8192
    sources=[Path(__file__),run.ROOT/'scripts/analysis/audit_mem_gallery_deepseek_memoryos_extended.py',
             run.ROOT/'scripts/tests/test_mem_gallery_deepseek_budget_extension.py']
    revision=output/'budget_extension_protocol.json'
    if revision.exists():
        for f,d in run.read(revision)['source_sha256'].items():assert run.sha(f)==d
        return
    assert run.read(output/'execution.json')['status']=='failed'
    root=output/'memory';completed={f.parent.name:run.sha(f) for f in root.glob('*/complete.json')}
    run.freeze(revision,dict(created=datetime.now(timezone.utc).isoformat(),base_protocol_sha256=run.sha(output/'protocol.json'),
        source_sha256={str(f.resolve()):run.sha(f) for f in sources},
        policy='Keep native 8192-token memory requests. Only finish_reason=length triggers the exact same request with max_tokens=32768. No partial completion is returned to the core. Other failures remain terminal.',
        retrieval_and_reader_unchanged=True,retained_complete_tasks=completed,
        reuse_policy='Completed scenarios and exact full-response request cache retained. Incomplete native states are archived and rebuilt; no truncated profile is reused.',
        old_api_responses_sha256=run.sha(root/'api_responses.jsonl')))
    errors=root/'run_errors.json'
    if errors.exists():errors.rename(root/'run_errors_before_budget_extension.json')
    run.freeze(output/'execution_before_budget_extension.json',run.read(output/'execution.json'))


def main():
    output=run.OUTPUT/'memoryos';prepare(output)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    os.environ.update(env)
    stage='memory'
    try:
        run.dump(output/'execution.json',dict(status='running',stage=stage,pid=os.getpid()))
        run.dump(output/'stages/memory.json',dict(status='running',amendment='budget_extension_protocol.json'))
        api.Model=BudgetModel
        run.memory_stage(output,run.ga.DEFAULT_ENV)
        run.dump(output/'stages/memory.json',dict(status='complete',amendment='budget_extension_protocol.json'))
        print('memoryos extended memory complete',flush=True)
        for stage,module in [('answers','scripts.evaluation.mem_gallery_deepseek_text_methods'),
                             ('audit','scripts.analysis.audit_mem_gallery_deepseek_memoryos_extended')]:
            run.dump(output/'execution.json',dict(status='running',stage=stage,pid=os.getpid()))
            command=[sys.executable,'-u','-m',module]
            if stage=='answers':command.append(stage)
            command+=['--method','memoryos','--output',str(run.OUTPUT)]
            with (output/(stage+'.log')).open('a') as log:subprocess.run(command,cwd=run.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('memoryos',stage,'complete',flush=True)
    except BaseException as exc:
        run.dump(output/'execution.json',dict(status='failed',stage=stage,error=str(exc)));raise
    run.dump(output/'execution.json',dict(status='complete',completed=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
