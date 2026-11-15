"""Complete and aggregate the authorized one-round memory-gap variant."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import m2a_deepseek_suite as common
from scripts.evaluation import m2a_deepseek_recovery_suite as recovery
from scripts.evaluation import m2a_deepseek_memory_gap_run as variant


def finalize(root, inherited, output):
    checked = variant.audit(output)
    if not checked['passed']:
        raise ValueError('Memory-gap variant failed audit: '+str(checked['errors']))
    entry = dict(dataset='memgallery', task=variant.TASK, output=str(output), status='passed',
                 questions=checked['questions'], correct=checked['correct'], audit=checked,
                 completion_sha256=common.sha(output/'completion_manifest.json'))
    entries = inherited+[entry]
    datasets = {}
    for ds in ('memeye', 'memgallery'):
        expected = 371 if ds == 'memeye' else 1711
        selected = [e for e in entries if e['dataset'] == ds]
        answers, tasks = [], []
        for e in selected:
            out = Path(e['output'])
            if e is not entry:
                recovery.verify_inherited(e)
            full, labels, _ = common.load_task(ds, e['task'])
            if common.read(out/'evaluation_labels.json') != labels:
                raise ValueError('Source QA changed')
            if e is not entry and common.read(out/'observations.json') != full:
                raise ValueError('Inherited history changed')
            values = common.rows(out/'responses.jsonl')
            if [a['id'] for a in values] != [q['id'] for q in labels]:
                raise ValueError('QA coverage/order mismatch')
            answers.extend(values)
            tasks.append(dict(task=e['task'], output=str(out), questions=len(values),
                correct=e['correct'], strict_audit_passed=e is not entry,
                variant_audit_passed=e['audit']['passed'], completion_sha256=e['completion_sha256']))
        if len(answers) != expected or len({a['id'] for a in answers}) != expected:
            raise ValueError('Full question denominator mismatch')
        report = dict(dataset=ds, model=common.MODEL, questions=len(answers), expected_questions=expected,
            correct=sum(a['correct'] for a in answers), accuracy=sum(a['correct'] for a in answers)/expected,
            tasks=tasks, coverage_complete=True, question_coverage_complete=True,
            memory_coverage_complete=ds=='memeye', strict_audit_passed=ds=='memeye',
            variant_audit_passed=True, missing_memory_ids=[variant.OMITTED_ID] if ds=='memgallery' else [],
            evaluation_variant=variant.VARIANT if ds=='memgallery' else 'Unchanged completed MemEye run')
        specification=common.read(root/'plan.json')
        report['original_rounds']=specification['datasets'][ds]['rounds']
        report['available_memory_rounds']=report['original_rounds']-(ds=='memgallery')
        report['known_source_issues']=[i for i in specification['known_source_issues'] if i['dataset']==ds]
        directory = root/ds
        directory.mkdir(exist_ok=True)
        if ds == 'memgallery':
            report['metrics'] = {k:sum(a[k] for a in answers)/expected for k in ('em','f1','bleu','bleu_1','bleu_2')}
            common.write(directory/'recovery_summary.json', report)
            from scripts.evaluation.m2a_gallery_image_id_postprocess import run as postprocess
            postprocess(directory, directory/'image_id_postprocessed')
            report['image_id_mapped_metrics'] = common.read(directory/'image_id_postprocessed/summary.json')['metrics']['after']
        common.write(directory/'summary.json', report)
        datasets[ds] = report
    result = dict(status='completed_with_memory_gap', datasets=datasets,
                  question_coverage_complete=True, memory_coverage_complete=False,
                  variant_audit_passed=True, missing_memory_ids=[variant.OMITTED_ID],
                  created_at=common.now(), plan_sha256=common.sha(root/'plan.json'))
    common.write(root/'suite_summary.json', result)
    common.write(root/'completion_manifest.json', dict(status=result['status'],
        question_coverage_complete=True, memory_coverage_complete=False, variant_audit_passed=True,
        plan_sha256=common.sha(root/'plan.json'), suite_summary_sha256=common.sha(root/'suite_summary.json'),
        missing_memory_ids=[variant.OMITTED_ID]))
    return result


def run(root):
    parent = variant.PARENT
    parent_plan = common.verify(parent)
    state = common.read(parent/'status.json')
    if state['active']:
        raise ValueError('Previous queue is active')
    inherited = [e for e in state['tasks'] if e['status']=='passed']
    if len(inherited) != 27:
        raise ValueError('Expected 27 completed tasks to inherit')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent/'code_snapshot',root/'code_snapshot')
    full, labels, _ = common.load_task('memgallery', variant.TASK)
    omission = variant.omission_record(full)
    plan = copy.deepcopy(parent_plan)
    plan.update(parent=str(parent), parent_plan_sha256=common.sha(parent/'plan.json'),
                evaluation_variant=variant.VARIANT, missing_memory_ids=[variant.OMITTED_ID],
                qa_denominator=2082, memory_gap_policy=omission,
                inherited_completion_hashes={e['output']:e['completion_sha256'] for e in inherited})
    for path in [Path(__file__).resolve(), Path(variant.__file__).resolve()]:
        plan['implementation_sha256'][str(path)] = common.sha(path)
        target=root/'code_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    common.write(root/'plan.json', plan)
    common.write(root/'memory_omissions.json', [omission])
    common.write(root/'inherited_tasks.json', inherited)
    out = root/'memgallery'/variant.TASK/'attempt_001'
    out.parent.mkdir(parents=True)
    cache = ROOT/'results/m2a_single_pass_deepseek_20261113/memgallery'/variant.TASK/'api_cache'
    cmd = common.command(dict(dataset='memgallery', task=variant.TASK), out, cache, common.DEFAULT_ENV)
    cmd[2] = str(ROOT/'scripts/evaluation/m2a_deepseek_memory_gap_run.py')
    status = dict(status='running', pid=os.getpid(), started_at=common.now(),
                  completed_tasks=27, expected_tasks=28, completed_questions=2004, expected_questions=2082,
                  missing_memory_ids=[variant.OMITTED_ID], active=[])
    with (out.parent/'attempt_001.log').open('w') as log:
        proc=subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='1'),
                              stdout=log, stderr=subprocess.STDOUT)
        status['active']=[dict(task=variant.TASK, pid=proc.pid, output=str(out))]
        while proc.poll() is None:
            if (out/'progress.json').exists():
                status['active'][0]['progress']=common.read(out/'progress.json')
            status['updated_at']=common.now()
            common.write(root/'status.json',status)
            if (out/'failure.json').exists():
                proc.terminate()
            time.sleep(5)
    status['active']=[]
    if proc.returncode != 0 or not (out/'completion_manifest.json').exists():
        status.update(status='failed', returncode=proc.returncode)
        common.write(root/'status.json', status)
        return
    status['status']='auditing'
    common.write(root/'status.json',status)
    try:
        result=finalize(root, inherited, out)
    except BaseException as exc:
        status.update(status='audit_failed',error=repr(exc))
        common.write(root/'status.json',status)
        raise
    status.update(status=result['status'], completed_tasks=28, completed_questions=2082, updated_at=common.now())
    common.write(root/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.output.resolve())
