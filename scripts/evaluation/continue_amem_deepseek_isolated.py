"""Isolate rejected scenarios and evaluate only fully completed scenario memories.

Provider-rejected requests are never resubmitted. Original notes, prompts,
sampling, successful call cache, and native retrieval remain unchanged.
Outputs here are explicitly partial and cannot be reported as a full score.
"""
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from types import SimpleNamespace
import time

from scripts.evaluation import recover_mem_gallery_amem_api as recovery

run = recovery.run
api = run.deep_memory
KNOWN_BLOCKED = {'Astronomy_Physics_Scientific_Experiments_Cosmology',
                 'Nature_Economics_Programming_Student_Life'}


class ScenarioRejected(RuntimeError):
    pass


class IsolatedModel(recovery.DeepModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rejected = {r['request_sha256'] for r in run.rows(self.root / 'api_attempts.jsonl')
                         if r.get('provider_error_code') == 'data_inspection_failed'}

    def unchecked_one(self, messages, cap, temperature):
        request = self.wire(messages, cap, temperature)
        key = api.request_sha256(request)
        if key in self.rejected:
            raise ScenarioRejected('Previously rejected request; not resubmitted: ' + key)
        for attempt in range(1, 4):
            try:
                out = api.chat_request(self.config, request)
                self.log('api_attempts.jsonl', dict(request_sha256=key, attempt=attempt,
                         status='ok', response_id=out['response_id']))
                self.log('api_responses.jsonl', dict(stage='memory', request=request, output=out))
                return out
            except api.APIError as exc:
                failure = dict(request_sha256=key, attempt=attempt, status='error',
                    error_type=exc.kind, retryable=exc.retryable,
                    provider_error_code=getattr(exc, 'provider_error_code', None))
                self.log('api_attempts.jsonl', failure)
                if failure['provider_error_code'] == 'data_inspection_failed':
                    with self.lock:
                        self.rejected.add(key)
                    raise ScenarioRejected('Provider rejected this scenario operation: ' + key) from exc
                if not exc.retryable or attempt == 3:
                    with self.lock:
                        self.fatal_errors.append(failure)
                    raise
                time.sleep(attempt * 2)


class ScenarioBroker(run.runner.adapters.StreamingBroker):
    def request(self, *args, **kwargs):
        if getattr(self.local, 'rejected', False):
            raise ScenarioRejected('Scenario already blocked; no additional calls')
        try:
            return super().request(*args, **kwargs)
        except ScenarioRejected as exc:
            self.local.rejected = True
            self.model.log('scenario_rejections.jsonl', dict(
                task=getattr(self.local, 'owner', None), reason=str(exc)))
            raise


class AvailableExecutor(ThreadPoolExecutor):
    def submit(self, fn, name):
        if name in KNOWN_BLOCKED:
            future = Future()
            future.set_exception(ScenarioRejected('Known rejected scenario; not resubmitted'))
            return future
        return super().submit(fn, name)


def partial_answers(root, p, completed):
    """Use the existing reader exactly; audit/score only the explicit subset."""
    outdir = root / 'partial_reader'
    outdir.mkdir(exist_ok=True)
    qs = [q for q in run.rows(root / 'inputs/questions.jsonl') if q['task'] in completed]
    retrieved = {}
    notes = {}
    obs = run.read(root / 'observations.json')
    helper = run.base.native.helpers('a_mem')
    for task in completed:
        dest = root / 'memory' / task
        seal = run.read(dest / 'complete.json')
        assert seal['experiment_signature'] == p['experiment_signature']
        for f, h in seal['artifact_sha256'].items():
            assert run.sha(f) == h, f
        notes[task] = run.read(dest / 'final_memory.json')
        assert set(notes[task]) == {r['id'] for r in obs[task]}
        for r in obs[task]:
            assert notes[task][r['id']]['content'] == helper['build_a_mem_note_text'](r, r['speaker'], 'assistant')
            assert notes[task][r['id']]['timestamp'] == r['timestamp']
        retrieved.update({r['id']: r for r in run.rows(dest / 'retrieval.jsonl')})
    assert len(qs) == len(retrieved) and qs
    payloads = []
    for q in qs:
        r = retrieved[q['id']]
        ids = r['retrieval_detail']['context_memory_ids']
        assert len(ids) == len(set(ids)) == 5
        assert r['retrieved_context'] == run.render_notes(notes[q['task']], ids)
        payloads.append(run.base.reader_payload(q, r['retrieved_context']))
    run.write_rows(outdir / 'manifest.jsonl', payloads)
    model = run.deep_reader.Model(outdir, run.base.DEFAULT_ENV, p['expected_model'])

    def one(payload):
        wire = run.deep_images.wire_for(payload, p['max_new_tokens'])
        path = outdir / 'responses' / (run.obj(payload['record_id']) + '.json')
        if path.exists():
            saved = run.read(path)
            assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
            out = saved['output']
        else:
            out = model.invoke(wire, 'answer', payload['record_id'])
            saved = dict(payload=payload, output=out, experiment_signature=p['experiment_signature'])
            if out.get('status') != 'provider_rejection':
                assert out['returned_model'] == p['expected_model']
                if out['finish_reason'] != 'stop' or out.get('refusal'):
                    run.dump(outdir / 'invalid' / path.name, saved)
                    raise RuntimeError('Invalid partial answer archived')
            # Empty stop answers retain their original text and score zero.
            run.dump(path, saved)
        assert out['request_sha256'] == run.deep_reader.request_sha256(wire)
        return dict(id=payload['record_id'], **out)

    answers = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, f in enumerate(as_completed([pool.submit(one, row) for row in payloads]), 1):
            a = f.result(); answers[a['id']] = a
            run.dump(outdir / 'progress.json', dict(done=i, total=len(qs), full_dataset=False))
    run.write_rows(outdir / 'responses.jsonl', [answers[q['id']] for q in qs])
    gold = {r['id']: r for r in run.rows(root / 'inputs/gold.jsonl')}
    scored = []
    for q in qs:
        a = answers[q['id']]; g = gold[q['id']]
        gids = set(g['gold_ids'])
        sources = {q['task'] + '/' + k for k in retrieved[q['id']]['retrieval_detail']['context_memory_ids']}
        scored.append(dict(id=q['id'], **run.base.gallery.score_answer(a['raw_response'], g['answer']),
            gold_hits=len(gids & sources), gold_count=len(gids),
            provider_rejection=a.get('status') == 'provider_rejection', empty_answer=not a['raw_response'].strip()))
    evidence = [r for r in scored if r['gold_count']]
    metrics = dict(questions=len(qs), correct=sum(int(r['em']) for r in scored),
        **{k:sum(r[k] for r in scored)/len(scored) for k in ('em','f1','bleu_1','bleu_2','bleu')},
        source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in evidence)/len(evidence),
        provider_rejections=sum(r['provider_rejection'] for r in scored),
        empty_answers=sum(r['empty_answer'] for r in scored))
    run.write_rows(root / 'partial_scored_questions.jsonl', scored)
    run.dump(root / 'partial_results.json', dict(status='partial', full_dataset=False,
        model=p['expected_model'], completed_tasks=sorted(completed),
        omitted_tasks=sorted(set(p['tasks'])-set(completed)), overall=metrics))
    files = [root/'partial_results.json', root/'partial_scored_questions.jsonl',
             outdir/'manifest.jsonl', outdir/'responses.jsonl', *outdir.glob('responses/*.json')]
    run.dump(root / 'partial_audit.json', dict(status='passed', full_dataset=False,
        questions=len(qs), contexts_reconstructed=len(qs), original_notes_verified=True,
        artifact_sha256={str(f):run.sha(f) for f in files}))


def main():
    root = run.OUTPUT / 'deepseek'
    p = run.gpt.verify(root)
    run.freeze(root / 'scenario_isolation_amendment.json', dict(
        script_sha256=run.sha(__file__), base_protocol_sha256=run.sha(root/'protocol.json'),
        recovery_script_sha256=run.sha(recovery.__file__), known_blocked=sorted(KNOWN_BLOCKED),
        policy='Never resend provider-rejected requests; stop only affected scenario. '
               'Preserve original algorithm and cache. Evaluate completed scenarios as explicit partial subset; empty answers score zero.'))
    old_errors = root / 'memory/run_errors.json'
    if old_errors.exists():
        old_errors.rename(old_errors.with_name('run_errors_before_isolation_'+str(time.time_ns())+'.json'))
    run.configure(p)
    brokers = []
    def make_broker(method, model, dest, protocol):
        broker = ScenarioBroker(model, dest, protocol, 8)
        brokers.append(broker)
        return broker
    class GuardedMemory(run.AMem):
        def store(self, row):
            result = super().store(row)
            if getattr(brokers[0].local, 'rejected', False):
                raise ScenarioRejected('Native metadata fallback followed provider rejection; scenario not sealed')
            return result
        def recall(self, query):
            result = super().recall(query)
            if getattr(brokers[0].local, 'rejected', False):
                raise ScenarioRejected('Native retrieval fallback followed provider rejection; scenario not sealed')
            return result
    class TaskExecutor(AvailableExecutor):
        def submit(self, fn, name):
            def isolated(task):
                brokers[0].local.rejected = False
                return fn(task)
            return super().submit(isolated, name)
    run.runner.NativeMemory = GuardedMemory
    run.runner.memory_api = SimpleNamespace(Model=IsolatedModel)
    run.runner.adapters.make_broker = make_broker
    run.runner.ThreadPoolExecutor = TaskExecutor
    try:
        run.runner.memory_stage(root, run.base.DEFAULT_ENV)
    except RuntimeError:
        if not (root/'memory/run_errors.json').exists():
            raise
    completed = [f.parent.name for f in (root/'memory').glob('*/complete.json')]
    run.dump(root/'partial_execution_status.json', dict(status='partial', phase='answering_completed_scenarios',
        completed_tasks=sorted(completed), blocked_or_failed_tasks=sorted(set(p['tasks'])-set(completed)),
        full_dataset_score_available=False))
    partial_answers(root, p, completed)
    run.dump(root/'partial_execution_status.json', dict(status='partial', phase='finished_available_scenarios',
        completed_tasks=sorted(completed), blocked_or_failed_tasks=sorted(set(p['tasks'])-set(completed)),
        full_dataset_score_available=False))


if __name__ == '__main__':
    main()
