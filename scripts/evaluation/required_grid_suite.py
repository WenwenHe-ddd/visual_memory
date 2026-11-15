"""Fill the 2-dataset/3-backbone/3-seed/eight-method grid, two methods at once."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c

BACKBONES={'Qwen3-VL-8B':'qwen','GPT-5':'gpt5','DeepSeek Flash':'deepseek'}

def prepare():
    inventory=c.ROOT/'reports/experiment_inventory_20261113/inventory.json'
    old=c.read(inventory)
    existing={(r['dataset'].lower(),BACKBONES[r['backbone']],r['seed'],r['method_id']):r
              for r in old['main_method_records'] if r['method_id'] in c.METHODS and r['seed'] in (42,1024,2026)}
    jobs=[]
    for ds in ('memeye','memgallery'):
        for bb in ('qwen','gpt5','deepseek'):
            for seed in (42,1024,2026):
                for method in c.METHODS:
                    identity=f'{ds}__{bb}__s{seed}__{method}'
                    old_row=existing.get((ds,bb,seed,method))
                    root=c.RUN/'new_runs'/ds/bb/f'seed{seed}'
                    job=dict(id=identity,dataset=ds,backbone=bb,seed=seed,method=method,
                        questions=371 if ds=='memeye' else 1711,run_root=str(root),
                        method_root=str(root/'methods'/method),workers=4,judge_seed=42,
                        status='reused_complete' if old_row and old_row['judge_status']=='complete' else 'pending',
                        existing_source_paths=old_row['source_paths'] if old_row else [],
                        existing_note=old_row.get('note','') if old_row else '')
                    c.freeze(c.RUN/'jobs'/f'{identity}.json',job)
                    jobs.append(job)
    plan=dict(dataset=['memeye','memgallery'],backbone=list(c.MODELS),seeds=[42,1024,2026],
        methods=list(c.METHODS),jobs=jobs,total_required=144,reused=sum(j['status']=='reused_complete' for j in jobs),
        missing=sum(j['status']=='pending' for j in jobs),maximum_active_methods=2,
        caption_policy='Preserve same-backbone seed42 captions; fresh native construction/retrieval/answers for new seeds.',
        judge_seed=42,source_sha256={str(p):c.sha(p) for p in [Path(__file__),Path(c.__file__),inventory]},
        user_authorization='Explicit request to retain all existing results and run every missing requested combination.')
    c.freeze(c.RUN/'plan.json',plan)
    return plan

def handler(job):
    directory=c.RUN/'ready'
    for pattern in (job['id'],f"{job['dataset']}__{job['backbone']}__{job['method']}",
                    f"{job['dataset']}__{job['method']}"):
        path=directory/(pattern+'.json')
        if path.exists():
            found=c.read(path)
            c.hashes(found['source_sha256'])
            return found
    return None

def completed(job):
    if job['status']=='reused_complete':return True
    p=Path(job['method_root'])/'completion.json'
    if not p.exists():return False
    x=c.read(p)
    if (x.get('status'),x.get('seed'),x.get('method'),x.get('questions'),x.get('open_questions'),x.get('judge_questions')) != (
        'complete',job['seed'],job['method'],job['questions'],job['questions'],job['questions']):
        raise ValueError('Incomplete final job evidence: '+job['id'])
    c.hashes(x['source_sha256']);c.hashes(x['artifact_sha256'])
    return True

def event(kind,**values):
    import json
    path=c.RUN/'scheduler/events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    row=dict(time=c.now(),event=kind,**values)
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,ensure_ascii=False),flush=True)

def publish(plan,state):
    items=[]
    for j in plan['jobs']:
        s=state['jobs'].get(j['id'],{})
        items.append(dict(dataset=j['dataset'],backbone=j['backbone'],seed=j['seed'],method=j['method'],
            questions=j['questions'],status=s.get('status',j['status']),
            new_completion=str(Path(j['method_root'])/'completion.json') if j['status']!='reused_complete' else '',
            existing_sources=' | '.join(j['existing_source_paths'])))
    state.update(updated_at=c.now(),methods_complete=sum(x['status'] in ('complete','reused_complete') for x in items),
        methods_total=144,maximum_active_methods=2)
    c.dump(c.RUN/'progress.json',state)
    tmp=c.RUN/'coverage.csv.tmp'
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    tmp.replace(c.RUN/'coverage.csv')

def run(plan,retry=False):
    c.RUN.mkdir(parents=True,exist_ok=True)
    with c.lock(c.RUN/'scheduler/suite.lock',False):
        c.hashes(plan['source_sha256'])
        state=c.read(c.RUN/'progress.json') if (c.RUN/'progress.json').exists() else {'jobs':{}}
        # systemd KillMode=control-group terminates children with the supervisor.
        for key,old in state['jobs'].items():
            if old.get('status')=='running':
                try:os.kill(old['pid'],0)
                except ProcessLookupError:old['status']='pending'
                else:raise ValueError('Prior method still running: '+key)
        for j in plan['jobs']:
            if completed(j):state['jobs'][j['id']]={'status':j['status'] if j['status']=='reused_complete' else 'complete'}
            elif retry or state['jobs'].get(j['id'],{}).get('status')!='failed':state['jobs'][j['id']]={'status':'pending'}
        state.update(status='running',supervisor_pid=os.getpid(),active=[])
        publish(plan,state)
        pending=[j for j in plan['jobs'] if state['jobs'][j['id']]['status']=='pending']
        # Finish the one remaining method in the latest seed study first.
        pending.sort(key=lambda j:(not(j['dataset']=='memeye' and j['backbone']=='deepseek' and j['seed']==2026),
            j['dataset']!='memeye',j['backbone']!='qwen',j['seed'],list(c.METHODS).index(j['method'])))
        running={};stopping=False
        def stop(sig,frame):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        event('suite_started',pid=os.getpid(),reused=plan['reused'],missing=plan['missing'])
        try:
            while pending or running:
                while len(running)<2 and pending and not stopping:
                    selected=next(((j,handler(j)) for j in pending if handler(j)),None)
                    if selected is None:break
                    j,h=selected;pending.remove(j);c.hashes(plan['source_sha256'])
                    path=c.RUN/'jobs'/(j['id']+'.json')
                    c.freeze(Path(j['method_root'])/'handler.json',h)
                    log=c.RUN/'logs'/(j['id']+'_'+str(time.time_ns())+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    stream=log.open('ab',buffering=0)
                    cmd=[sys.executable,'-u',h['script'],'--job',str(path)]
                    proc=subprocess.Popen(cmd,cwd=c.ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1',
                        'TOKENIZERS_PARALLELISM':'false','PYTHONHASHSEED':str(j['seed']),
                        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    running[j['id']]=(j,proc,stream)
                    state['jobs'][j['id']]=dict(status='running',pid=proc.pid,log=str(log),command=cmd,started_at=c.now())
                    state['active']=list(running);publish(plan,state)
                    event('method_started',id=j['id'],pid=proc.pid,active=list(running),log=str(log))
                for key,(j,proc,stream) in list(running.items()):
                    code=proc.poll()
                    if code is None:continue
                    stream.close();del running[key]
                    try:
                        if code!=0:raise ValueError('Native method exited '+str(code))
                        if not completed(j):raise ValueError('Method exited without final cohort evidence')
                        state['jobs'][key].update(status='complete',exit_code=code,ended_at=c.now())
                    except Exception as exc:
                        state['jobs'][key].update(status='failed',exit_code=code,error=f'{type(exc).__name__}: {exc}',ended_at=c.now())
                    event('method_finished',id=key,status=state['jobs'][key]['status'],exit_code=code,active=list(running))
                    state['active']=list(running);publish(plan,state)
                if stopping:break
                if pending or running:time.sleep(5)
        finally:
            for key,(j,proc,stream) in running.items():
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                stream.close();state['jobs'][key].update(status='pending',interrupted_at=c.now())
                event('method_interrupted',id=key)
            state['active']=[]
            state['status']='interrupted' if stopping else 'complete' if all(
                v['status'] in ('complete','reused_complete') for v in state['jobs'].values()) else 'incomplete'
            publish(plan,state)
        event('suite_finished',status=state['status'])
        if state['status']=='complete':
            c.dump(c.RUN/'completion.json',dict(status='complete',total=144,reused=plan['reused'],
                newly_completed=plan['missing'],maximum_active_methods=2,completed_at=c.now(),
                artifact_sha256={str(p):c.sha(p) for p in [c.RUN/'plan.json',c.RUN/'coverage.csv',c.RUN/'scheduler/events.jsonl']}))
        return 0 if state['status']=='complete' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('prepare','run','status'),default='status');p.add_argument('--retry-failed',action='store_true');a=p.parse_args()
    plan=prepare()
    if a.phase=='run':raise SystemExit(run(plan,a.retry_failed))
    if a.phase=='prepare':print({k:plan[k] for k in ('total_required','reused','missing')})
    else:print(c.read(c.RUN/'progress.json') if (c.RUN/'progress.json').exists() else {'status':'prepared','missing':plan['missing']})
