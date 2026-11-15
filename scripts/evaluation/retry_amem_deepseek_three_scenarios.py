"""User-authorized fresh attempt of three missing Gallery scenarios.

Preserve the original run. Reuse its valid calls, regenerate the single malformed
evolution response, and allow one fresh submission of previously rejected input.
No prompt/content rewriting or alternate provider routing is applied.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import shutil
import traceback
from scripts.evaluation import continue_amem_deepseek_isolated as isolated

run = isolated.run
SOURCE = run.OUTPUT / 'deepseek'
OUTPUT = run.OUTPUT / 'deepseek_retry3_20261104'
TASKS = {'Astronomy_Physics_Scientific_Experiments_Cosmology',
         'Landscape_Travel_Architecture_Nature', 'Nature_Economics_Programming_Student_Life'}


def prepare():
    p = run.gpt.verify(SOURCE)
    if (OUTPUT/'retry_amendment.json').exists():
        amendment = run.read(OUTPUT/'retry_amendment.json')
        assert amendment['script_sha256'] == run.sha(__file__)
        for f, h in amendment['input_sha256'].items():
            assert run.sha(f) == h
        return run.gpt.verify(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copies = ['protocol.json', 'source_snapshot/manifest.json', 'observations.json',
              'inputs/questions.jsonl', 'inputs/gold.jsonl']
    for rel in copies:
        dest = OUTPUT/rel; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/rel, dest)
    assert TASKS == set(run.read(SOURCE/'partial_results.json')['omitted_tasks'])
    owners = {}
    for event in run.rows(SOURCE/'memory/call_events.jsonl'):
        if event['owner'] in TASKS:
            owners.setdefault(event['key'], set()).add(event['owner'])
    provenance = []; excluded = []; wanted = set()
    (OUTPUT/'memory/calls').mkdir(parents=True, exist_ok=True)
    for key, tasks in owners.items():
        f = SOURCE/'memory/calls'/(key+'.json'); saved = run.read(f)
        assert run.obj(saved['request']) == key
        if saved['output'].get('structured_output_valid') is False:
            excluded.append(dict(key=key, tasks=sorted(tasks), sha256=run.sha(f)))
            continue
        assert saved['output']['finish_reason'] == 'stop'
        assert saved['output']['returned_model'] == p['expected_model']
        shutil.copyfile(f, OUTPUT/'memory/calls'/f.name)
        provenance.append(dict(key=key, tasks=sorted(tasks), source=str(f), sha256=run.sha(f)))
        wanted.add((saved['output']['request_sha256'], saved['output']['response_id']))
    assert len(excluded) == 1 and excluded[0]['tasks'] == ['Landscape_Travel_Architecture_Nature']
    # Copy response evidence, but not rejection state, into this distinct attempt.
    inherited = []
    for row in run.rows(SOURCE/'memory/api_responses.jsonl'):
        o = row['output']
        if (o['request_sha256'], o['response_id']) in wanted:
            inherited.append(row)
    assert wanted == {(r['output']['request_sha256'], r['output']['response_id']) for r in inherited}
    run.write_rows(OUTPUT/'memory/api_responses.jsonl', inherited)
    events_path = SOURCE/'memory/budget_recovery_events.jsonl'
    if events_path.exists():
        inherited_events = [e for e in run.rows(events_path)
            if (e['attempts'][-1]['request_sha256'], e['attempts'][-1]['response_id']) in wanted]
        run.write_rows(OUTPUT/'memory/budget_recovery_events.jsonl', inherited_events)
    (OUTPUT/'memory/api_attempts.jsonl').write_text('')
    run.freeze(OUTPUT/'retry_amendment.json', dict(
        user_authorization='Retry the three missing DeepSeek scenarios.',
        script_sha256=run.sha(__file__), tasks=sorted(TASKS),
        source=str(SOURCE), source_protocol_sha256=run.sha(SOURCE/'protocol.json'),
        inherited_signature_policy='Same semantic configuration; distinct retry namespace and amendment identify the new attempt.',
        input_sha256={str(OUTPUT/f):run.sha(OUTPUT/f) for f in copies},
        adapter_sha256={str(Path(m.__file__)):run.sha(m.__file__) for m in (isolated, isolated.recovery)},
        excluded_invalid_calls=excluded, reused_valid_calls=provenance,
        rejection_policy='One new submission of previously rejected input; unchanged messages/model/sampling. Any fresh content rejection stops only that scenario.',
        budget_policy='Original 2048-token cap; length-only retries at 4096/8192/16384 as previously documented.',
        source_17_scenarios_unchanged=True))
    return p


def memory_stage(p):
    root = OUTPUT/'memory'
    model = isolated.IsolatedModel(root, run.base.DEFAULT_ENV, 8, 2048)
    model.torch.set_num_threads(2)
    broker = isolated.ScenarioBroker(model, root, p, 8)
    errors = []
    try:
        encoder = run.runner.adapters.install_runtime('a_mem', broker, p)
        run.dump(root/'generation_environment.json', model.environment | dict(embedding_device=str(encoder.device)))
        obs = run.read(OUTPUT/'observations.json'); qs = run.rows(OUTPUT/'inputs/questions.jsonl')

        def one(task):
            dest = root/task
            if (dest/'complete.json').exists():
                return
            broker.local.owner = task; broker.local.rejected = False
            runtime = dest/'runtime'; runtime.mkdir(parents=True, exist_ok=True)
            memory = run.AMem(p, runtime)
            for i, row in enumerate(obs[task], 1):
                if model.fatal_errors:
                    raise RuntimeError('Terminal API failure')
                memory.store(row)
                if broker.local.rejected or model.fatal_errors:
                    raise isolated.ScenarioRejected('Operation failed; scenario not sealed')
                run.dump(dest/'progress.json', dict(phase='ingest', done=i, total=len(obs[task])))
            selected = [q for q in qs if q['task'] == task]; retrieved = []
            for i, q in enumerate(selected, 1):
                context, detail = memory.recall(q['retrieval_query'])
                if broker.local.rejected or model.fatal_errors:
                    raise isolated.ScenarioRejected('Retrieval failed; scenario not sealed')
                retrieved.append(dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                    retrieved_context=context, retrieval_detail=detail, experiment_signature=p['experiment_signature']))
                run.dump(dest/'progress.json', dict(phase='retrieval', done=i, total=len(selected)))
            run.write_rows(dest/'retrieval.jsonl', retrieved); memory.export(dest)
            files = [dest/'retrieval.jsonl', dest/'memory_stats.json', dest/'final_memory.json', *runtime.rglob('*.json')]
            run.freeze(dest/'complete.json', dict(experiment_signature=p['experiment_signature'],
                stored_rounds=memory.stats['stored_rounds'], questions=len(selected),
                artifact_sha256={str(f):run.sha(f) for f in files}))
            run.dump(dest/'progress.json', dict(phase='complete', done=len(selected), total=len(selected)))
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(one, task):task for task in sorted(TASKS)}
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    errors.append(dict(task=futures[f], traceback=traceback.format_exc()))
                    run.dump(root/'run_errors.json', errors)
    finally:
        broker.close()
    return errors


def combine(p, completed):
    sources = [SOURCE] + ([OUTPUT] if completed else [])
    merged = []; checks = {}
    for source in sources:
        audit = run.read(source/'partial_audit.json')
        assert audit['status'] == 'passed'
        for f, h in audit['artifact_sha256'].items():
            assert run.sha(f) == h, f
            checks[f] = h
        merged.extend(run.rows(source/'partial_scored_questions.jsonl'))
    ids = {r['id'] for r in merged}
    assert len(ids) == len(merged)
    all_qs = run.rows(OUTPUT/'inputs/questions.jsonl')
    included_tasks = set(run.read(SOURCE/'partial_results.json')['completed_tasks']) | set(completed)
    assert ids == {q['id'] for q in all_qs if q['task'] in included_tasks}
    evidence = [r for r in merged if r['gold_count']]
    overall = dict(questions=len(merged), correct=sum(int(r['em']) for r in merged),
        **{k:sum(r[k] for r in merged)/len(merged) for k in ('em','f1','bleu_1','bleu_2','bleu')},
        source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in evidence)/len(evidence),
        source_recall_micro=sum(r['gold_hits'] for r in evidence)/sum(r['gold_count'] for r in evidence),
        provider_rejections=sum(r['provider_rejection'] for r in merged),
        empty_answers=sum(r['empty_answer'] for r in merged))
    full = len(merged) == 1711
    run.write_rows(OUTPUT/'combined_scored_questions.jsonl', merged)
    run.dump(OUTPUT/'combined_results.json', dict(status='complete' if full else 'partial',
        full_dataset=full, model=p['expected_model'], overall=overall,
        completed_tasks=sorted(included_tasks), omitted_tasks=sorted(set(p['tasks'])-included_tasks),
        prior_results_reused=1429, retry_questions=len(merged)-1429))
    run.dump(OUTPUT/'combined_audit.json', dict(status='passed', full_dataset=full, questions=len(merged),
        unique_question_ids_verified=True, source_artifact_sha256=checks,
        result_sha256=run.sha(OUTPUT/'combined_results.json'), scored_sha256=run.sha(OUTPUT/'combined_scored_questions.jsonl')))


def main():
    p = prepare()
    run.dump(OUTPUT/'execution_status.json', dict(status='running', phase='memory', tasks=sorted(TASKS)))
    errors = memory_stage(p)
    completed = sorted(f.parent.name for f in (OUTPUT/'memory').glob('*/complete.json'))
    run.dump(OUTPUT/'execution_status.json', dict(status='running', phase='answers', completed_tasks=completed,
        failed_tasks=[e['task'] for e in errors]))
    if completed:
        isolated.recovery.audit_recovery(OUTPUT)
        isolated.partial_answers(OUTPUT, p, completed)
    combine(p, completed)
    run.dump(OUTPUT/'execution_status.json', dict(status='finished', completed_tasks=completed,
        failed_tasks=[e['task'] for e in errors], full_dataset_score_available=len(completed)==3))


if __name__ == '__main__':
    main()
