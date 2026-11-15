"""One user-authorized retry of the remaining Astronomy scenario, unchanged input."""
import shutil
from pathlib import Path
from scripts.evaluation import retry_amem_deepseek_three_scenarios as prior

run = prior.run
SOURCE = prior.OUTPUT
OUTPUT = run.OUTPUT / 'deepseek_astronomy_retry_20261105'
TASK = 'Astronomy_Physics_Scientific_Experiments_Cosmology'


def prepare():
    p = run.gpt.verify(SOURCE)
    if (OUTPUT/'retry_amendment.json').exists():
        amendment = run.read(OUTPUT/'retry_amendment.json')
        assert amendment['script_sha256'] == run.sha(__file__)
        for f,h in amendment['input_sha256'].items():
            assert run.sha(f) == h
        return run.gpt.verify(OUTPUT)
    assert run.read(SOURCE/'combined_results.json')['omitted_tasks'] == [TASK]
    copies = ['protocol.json','source_snapshot/manifest.json','observations.json',
              'inputs/questions.jsonl','inputs/gold.jsonl']
    for rel in copies:
        dest = OUTPUT/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(SOURCE/rel,dest)
    (OUTPUT/'memory/calls').mkdir(parents=True,exist_ok=True)
    keys = {e['key'] for e in run.rows(SOURCE/'memory/call_events.jsonl') if e['owner']==TASK}
    provenance=[]; wanted=set()
    for key in sorted(keys):
        f=SOURCE/'memory/calls'/(key+'.json');c=run.read(f);out=c['output']
        assert run.obj(c['request'])==key
        assert out.get('structured_output_valid') is not False and out['finish_reason']=='stop'
        assert out['returned_model']==p['expected_model']
        shutil.copyfile(f,OUTPUT/'memory/calls'/f.name)
        provenance.append(dict(source=str(f),sha256=run.sha(f)))
        wanted.add((out['request_sha256'],out['response_id']))
    inherited=[r for r in run.rows(SOURCE/'memory/api_responses.jsonl')
               if (r['output']['request_sha256'],r['output']['response_id']) in wanted]
    assert wanted=={(r['output']['request_sha256'],r['output']['response_id']) for r in inherited}
    run.write_rows(OUTPUT/'memory/api_responses.jsonl',inherited)
    events=[e for e in run.rows(SOURCE/'memory/budget_recovery_events.jsonl')
            if (e['attempts'][-1]['request_sha256'],e['attempts'][-1]['response_id']) in wanted]
    run.write_rows(OUTPUT/'memory/budget_recovery_events.jsonl',events)
    (OUTPUT/'memory/api_attempts.jsonl').write_text('')
    run.freeze(OUTPUT/'retry_amendment.json',dict(
        user_authorization='Try the remaining scenario again.',task=TASK,
        script_sha256=run.sha(__file__), reused_runner_sha256=run.sha(prior.__file__),
        input_sha256={str(OUTPUT/f):run.sha(OUTPUT/f) for f in copies},reused_calls=provenance,
        policy='One fresh submission of previously rejected input with identical messages and settings; no content rewriting. Fresh rejection stops this attempt.',
        prior_1657_answers_preserved=True))
    return p


def combine(p,completed):
    audit=run.read(SOURCE/'combined_audit.json');assert audit['status']=='passed'
    assert run.sha(SOURCE/'combined_results.json')==audit['result_sha256']
    assert run.sha(SOURCE/'combined_scored_questions.jsonl')==audit['scored_sha256']
    for f,h in audit['source_artifact_sha256'].items():
        assert run.sha(f)==h
    scored=run.rows(SOURCE/'combined_scored_questions.jsonl');assert len(scored)==1657
    if completed:
        fresh=run.read(OUTPUT/'partial_audit.json');assert fresh['status']=='passed'
        for f,h in fresh['artifact_sha256'].items():
            assert run.sha(f)==h
        scored.extend(run.rows(OUTPUT/'partial_scored_questions.jsonl'))
    included=set(run.read(SOURCE/'combined_results.json')['completed_tasks'])|set(completed)
    expected={q['id'] for q in run.rows(OUTPUT/'inputs/questions.jsonl') if q['task'] in included}
    assert len(scored)==len(expected) and {r['id'] for r in scored}==expected
    evidence=[r for r in scored if r['gold_count']]
    overall=dict(questions=len(scored),correct=sum(int(r['em']) for r in scored),
        **{k:sum(r[k] for r in scored)/len(scored) for k in ('em','f1','bleu_1','bleu_2','bleu')},
        source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in evidence)/len(evidence),
        source_recall_micro=sum(r['gold_hits'] for r in evidence)/sum(r['gold_count'] for r in evidence),
        provider_rejections=sum(r['provider_rejection'] for r in scored),
        empty_answers=sum(r['empty_answer'] for r in scored))
    full=len(scored)==1711
    run.write_rows(OUTPUT/'combined_scored_questions.jsonl',scored)
    run.dump(OUTPUT/'combined_results.json',dict(status='complete' if full else 'partial',full_dataset=full,
        model=p['expected_model'],overall=overall,completed_tasks=sorted(included),
        omitted_tasks=sorted(set(p['tasks'])-included),prior_results_reused=1657,retry_questions=len(scored)-1657))
    run.dump(OUTPUT/'combined_audit.json',dict(status='passed',full_dataset=full,questions=len(scored),
        original_1657_and_new_subset_verified=True,unique_question_ids_verified=True,
        parent_audit_sha256=run.sha(SOURCE/'combined_audit.json'),
        subset_audit_sha256=run.sha(OUTPUT/'partial_audit.json') if completed else None,
        result_sha256=run.sha(OUTPUT/'combined_results.json'),scored_sha256=run.sha(OUTPUT/'combined_scored_questions.jsonl')))


def main():
    p=prepare()
    prior.OUTPUT=OUTPUT;prior.TASKS={TASK}
    run.dump(OUTPUT/'execution_status.json',dict(status='running',phase='memory',task=TASK))
    errors=prior.memory_stage(p)
    completed=[TASK] if (OUTPUT/'memory'/TASK/'complete.json').exists() else []
    if completed:
        run.dump(OUTPUT/'execution_status.json',dict(status='running',phase='answers',task=TASK))
        prior.isolated.recovery.audit_recovery(OUTPUT)
        prior.isolated.partial_answers(OUTPUT,p,completed)
    combine(p,completed)
    run.dump(OUTPUT/'execution_status.json',dict(status='finished',completed_tasks=completed,
        failed_tasks=[e['task'] for e in errors],full_dataset_score_available=bool(completed)))


if __name__=='__main__':
    main()
