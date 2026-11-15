"""Frozen two-dataset DeepSeek M2A queue, task audits and Gallery ID rescoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation.m2a_suite import prepare as prepare_plan, read, write, sha, now
from scripts.evaluation.audit_memeye_m2a import audit, rows
from scripts.evaluation.m2a_inputs import load_task
from scripts.memory.m2a_deepseek import MODEL, DEFAULT_ENV, provider


def prepare(root, workers=4, datasets=('memeye', 'memgallery')):
    if (root/'plan.json').exists():
        raise ValueError('Use a fresh suite root')
    plan = prepare_plan(root,runner='single_pass',datasets_to_run=datasets)
    plan['configuration'].update(model=MODEL,llm_backend='deepseek_api',provider='DashScope International',
        endpoint=provider.DASHSCOPE_BASE_URL+'/chat/completions',seed=42,enable_thinking=False,temperature=0,
        image_max_pixels=None,image_policy='original image bytes; provider detail high',
        extraction_max_attempts=3,workers=workers,physical_gpus=[0,1],
        api_request_attempts=3,cache_policy='Exact wire hashes per task; reusable across fresh attempts',
        gallery_output='raw plus deterministic image ID mapped scores')
    for path in [Path(__file__),ROOT/'scripts/evaluation/m2a_deepseek_run.py',
                 ROOT/'scripts/memory/m2a_deepseek.py',Path(provider.__file__),
                 ROOT/'scripts/evaluation/openai_chat_transport.py',
                 ROOT/'scripts/evaluation/m2a_gallery_image_id_postprocess.py']:
        target = root/'code_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        plan['implementation_sha256'][str(path)] = sha(path)
    plan['known_source_issues'] = [dict(dataset='memgallery',id='Music_Dance_Theater_Performance_Learning/QA_0048',
        issue='Question field is 2024-11-13; answer field contains the question text',handling='Preserve source fields and full denominator')]
    write(root/'plan.json',plan)
    write(root/'status.json',dict(status='prepared_pending_api_probe',datasets={k:dict(tasks=len(v['tasks']),
        rounds=v['rounds'],questions=v['questions']) for k,v in plan['datasets'].items()}))
    return plan


def verify(root):
    plan = read(root/'plan.json')
    for path, expected in plan['implementation_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen implementation changed: '+path)
    for ds in plan['datasets'].values():
        for task in ds['tasks']:
            if sha(task['source']) != task['source_sha256']:
                raise ValueError('Frozen input changed: '+task['source'])
    return plan


def command(task, out, cache, env_file):
    return [sys.executable,'-u',str(ROOT/'scripts/evaluation/m2a_deepseek_run.py'),
        '--dataset',task['dataset'],'--task',task['task'],'--output',str(out),'--device','cuda:0',
        '--extraction-attempts','3','--api-env-file',str(env_file),'--api-cache-dir',str(cache)]


def probe(root, env_file):
    plan = verify(root)
    results=[]
    for dataset, task in [('memeye','Social_Chat_Memory_Test'),('memgallery','Fashion_Personal_Care_Lifestyle_Shopping')]:
        if dataset not in plan['datasets']:
            continue
        _, questions, _ = load_task(dataset,task)
        question = next(q for q in questions if q['query_image_paths'])
        target = root/'smoke'/dataset
        target.mkdir(parents=True,exist_ok=True)
        attempts = sorted(target.glob('attempt_*'))
        out = target/f'attempt_{len(attempts)+1:03d}'
        cmd = command(dict(dataset=dataset,task=task),out,target/'api_cache',env_file)
        cmd += ['--limit-rounds','4','--question-id',question['id']]
        with (target/(out.name+'.log')).open('w') as log:
            rc = subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
        if rc != 0 or not (out/'completion_manifest.json').exists():
            raise RuntimeError('API pipeline smoke failed: '+str(out))
        checked = audit(out)
        if not checked['passed']:
            raise RuntimeError('API pipeline smoke audit failed: '+str(checked['errors']))
        results.append(dict(dataset=dataset,output=str(out),audit=checked,completion_sha256=sha(out/'completion_manifest.json')))
    write(root/'probe.json',dict(passed=True,results=results,plan_sha256=sha(root/'plan.json')))


def finalize(root, entries):
    plan = verify(root)
    summaries={}
    for dataset, specification in plan['datasets'].items():
        tasks=[t for t in entries if t['dataset']==dataset and t['status'] in ('passed','audit_failed')]
        answers=[];task_results=[];issues=[]
        for task in tasks:
            out=Path(task['output'])
            check=audit(out)
            obs,labels,_=load_task(dataset,task['task'])
            if obs!=read(out/'observations.json') or labels!=read(out/'evaluation_labels.json'):
                raise ValueError('Full source coverage mismatch: '+task['task'])
            answer=rows(out/'responses.jsonl')
            if [a['id'] for a in answer]!=[q['id'] for q in labels]:
                raise ValueError('Full question coverage mismatch')
            answers.extend(answer)
            task_results.append(dict(task=task['task'],questions=len(answer),correct=check['correct'],
                strict_audit_passed=check['passed'],output=str(out),completion_sha256=sha(out/'completion_manifest.json')))
            if not check['passed']:
                issues.append(dict(task=task['task'],errors=check['errors']))
        n=len(answers)
        complete=n==specification['questions'] and len(tasks)==len(specification['tasks'])
        report=dict(dataset=dataset,model=MODEL,created_at=now(),questions=n,expected_questions=specification['questions'],
            tasks=task_results,correct=sum(a['correct'] for a in answers),accuracy=sum(a['correct'] for a in answers)/n if n else None,
            coverage_complete=complete,strict_audit_passed=complete and not issues,audit_issues=issues,
            known_source_issues=[i for i in plan['known_source_issues'] if i['dataset']==dataset],
            plan_sha256=sha(root/'plan.json'))
        if dataset=='memgallery' and n:
            report['metrics']={k:sum(a[k] for a in answers)/n for k in ('em','f1','bleu','bleu_1','bleu_2')}
        (root/dataset).mkdir(exist_ok=True)
        write(root/dataset/'summary.json',report)
        if dataset=='memgallery' and complete:
            write(root/dataset/'recovery_summary.json',report)
            from scripts.evaluation.m2a_gallery_image_id_postprocess import run as postprocess
            target=root/dataset/'image_id_postprocessed'
            if not target.exists():
                postprocess(root/dataset,target)
            report['image_id_mapped_metrics']=read(target/'summary.json')['metrics']['after']
            write(root/dataset/'summary.json',report)
        summaries[dataset]=report
    complete=all(s['coverage_complete'] for s in summaries.values())
    strict=all(s['strict_audit_passed'] for s in summaries.values())
    status='completed' if strict else ('completed_with_audit_issues' if complete else 'incomplete')
    final=dict(status=status,created_at=now(),datasets=summaries,plan_sha256=sha(root/'plan.json'))
    write(root/'suite_summary.json',final)
    if complete:
        write(root/'completion_manifest.json',dict(status=status,plan_sha256=sha(root/'plan.json'),
            suite_summary_sha256=sha(root/'suite_summary.json')))
    return final


def run(root, env_file):
    import fcntl
    plan=verify(root)
    if not (root/'probe.json').exists() or not read(root/'probe.json')['passed']:
        raise ValueError('Selected dataset smoke audits must pass before the full run')
    lock=(root/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=[]
    # Interleave the two datasets so both begin immediately.
    groups=[[dict(t,dataset=ds) for t in spec['tasks']] for ds,spec in plan['datasets'].items()]
    for i in range(max(map(len,groups))):
        pending.extend(g[i] for g in groups if i<len(g))
    entries=[];active={};started=time.time()
    state=dict(status='running',pid=os.getpid(),started_at=now())

    def save():
        state.update(updated_at=now(),elapsed_seconds=time.time()-started,tasks=entries,
            active=[{k:v for k,v in a.items() if k not in ('process','log')} for a in active.values()],
            completed_tasks=sum(t['status']=='passed' for t in entries),expected_tasks=sum(len(s['tasks']) for s in plan['datasets'].values()),
            completed_questions=sum(t.get('questions',0) for t in entries),expected_questions=sum(s['questions'] for s in plan['datasets'].values()))
        write(root/'status.json',state)

    def record(task,out):
        if not (out/'completion_manifest.json').exists():
            return dict(dataset=task['dataset'],task=task['task'],status='failed',output=str(out))
        checked=audit(out)
        return dict(dataset=task['dataset'],task=task['task'],status='passed' if checked['passed'] and checked['is_complete_task'] else 'audit_failed',
            output=str(out),questions=checked['questions'],correct=checked['correct'],audit=checked,
            completion_sha256=sha(out/'completion_manifest.json'))

    save()
    try:
        while pending or active:
            for slot in range(plan['configuration']['workers']):
                if slot in active or not pending:continue
                verify(root)
                task=pending.pop(0);target=root/task['dataset']/task['task'];target.mkdir(parents=True,exist_ok=True)
                attempts=sorted(p for p in target.glob('attempt_*') if p.is_dir())
                accepted=next((p for p in attempts if (p/'completion_manifest.json').exists() and audit(p)['passed']),None)
                if accepted:
                    entries.append(record(task,accepted));continue
                out=target/f'attempt_{len(attempts)+1:03d}'
                cmd=command(task,out,target/'api_cache',env_file)
                log=(target/(out.name+'.log')).open('w')
                proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot%2)),stdout=log,stderr=subprocess.STDOUT)
                active[slot]=dict(dataset=task['dataset'],task=task['task'],output=str(out),pid=proc.pid,
                    gpu=slot%2,process=proc,log=log)
            for slot,a in list(active.items()):
                out=Path(a['output']);proc=a['process']
                if (out/'progress.json').exists():a['progress']=read(out/'progress.json')
                if (out/'failure.json').exists() and proc.poll() is None:proc.terminate()
                if proc.poll() is None:continue
                a['log'].close();entries.append(record(a,out));del active[slot]
            save()
            if pending or active:time.sleep(5)
        state['status']='generation_finished_auditing';save()
        final=finalize(root,entries);state['status']=final['status'];save()
    except BaseException as exc:
        for a in active.values():a['process'].terminate()
        state.update(status='failed',error=repr(exc));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','probe','run','finalize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--datasets',nargs='+',choices=['memeye','memgallery'],default=['memeye','memgallery'])
    p.add_argument('--env-file',type=Path,default=DEFAULT_ENV)
    args=p.parse_args();root=args.output.resolve()
    if args.stage=='prepare':prepare(root,args.workers,args.datasets)
    elif args.stage=='probe':probe(root,args.env_file)
    elif args.stage=='run':run(root,args.env_file)
    else:finalize(root,read(root/'status.json')['tasks'])
