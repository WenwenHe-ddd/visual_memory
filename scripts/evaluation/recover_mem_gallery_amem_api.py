"""Resume A-Mem with audited output-budget escalation for truncated memory JSON.

Original signed source, inputs, successful calls and sampling stay unchanged.
Only finish_reason=length permits increasing the cap, up to 16384 tokens.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import time

from scripts.evaluation import mem_gallery_amem_api as run

CAPS = (2048, 4096, 8192, 16384)


class BudgetRecovery:
    def init_recovery(self):
        self.truncated = {}
        path = self.root / 'api_responses.jsonl'
        for row in run.rows(path) if path.exists() else []:
            out = row['output']
            if row.get('request') and out['finish_reason'] == 'length':
                self.truncated[out['request_sha256']] = out
        self.environment['memory_completion_caps_on_length'] = list(CAPS)

    def one(self, messages, cap, temperature):
        if self.fatal_errors:
            raise RuntimeError('Stopped after terminal memory API error')
        original = self.wire(messages, max(2048, cap), temperature)
        original_key = run.base.request_sha256(original)
        attempts = []
        for limit in (v for v in CAPS if v >= max(2048, cap)):
            request = self.wire(messages, limit, temperature)
            key = run.base.request_sha256(request)
            out = self.truncated.get(key)
            archived = out is not None
            if out is None:
                out = self.unchecked_one(messages, limit, temperature)
            assert out['request_sha256'] == key
            attempts.append(dict(cap=limit, request_sha256=key,
                                 response_id=out['response_id'], archived=archived,
                                 finish_reason=out['finish_reason']))
            if out['finish_reason'] == 'stop' and not out.get('refusal') and out['raw_response'].strip():
                if limit != max(2048, cap):
                    self.log('budget_recovery_events.jsonl', dict(
                        original_request_sha256=original_key, attempts=attempts,
                        input_messages_unchanged=True, successful_cap=limit))
                    out = dict(out, budget_recovery=dict(original_request_sha256=original_key,
                                                        successful_cap=limit))
                return out
            if out['finish_reason'] != 'length' or out.get('refusal'):
                break
            self.truncated[key] = out
        failure = dict(error_type='unrecoverable_memory_completion', attempts=attempts)
        with self.lock:
            self.fatal_errors.append(failure)
        self.log('budget_recovery_failures.jsonl', failure)
        raise RuntimeError('Memory response invalid after bounded budget recovery; raw responses retained')


class DeepModel(BudgetRecovery, run.deep_memory.APIModel):
    wire = staticmethod(run.deep_memory.build_request)

    def __init__(self, root, env_file, workers=8, floor=2048):
        assert floor == 2048
        super().__init__(root, env_file, workers)
        self.init_recovery()

    def unchecked_one(self, messages, cap, temperature):
        return run.deep_memory.APIModel.one(self, messages, cap, temperature)


class GPTModel(BudgetRecovery, run.gpt.transport.Model):
    wire = staticmethod(run.gpt.transport.build_request)

    def __init__(self, root, env_file, expected_model):
        super().__init__(root, env_file, expected_model)
        self.init_recovery()

    def unchecked_one(self, messages, cap, temperature):
        return run.gpt.transport.Model.one(self, messages, cap, temperature)


def audit_recovery(output):
    p = run.gpt.verify(output)
    events_path = output / 'memory/budget_recovery_events.jsonl'
    events = run.rows(events_path) if events_path.exists() else []
    api = {r['output']['request_sha256']: r for r in run.rows(output / 'memory/api_responses.jsonl') if r.get('request')}
    wire = GPTModel.wire if p['backbone'] == 'gpt5' else DeepModel.wire
    recovered = []
    for f in (output / 'memory/calls').glob('*.json'):
        c = run.read(f); req, out = c['request'], c['output']
        limit = out.get('budget_recovery', {}).get('successful_cap', max(2048, req['max_new_tokens']))
        expected_wire = wire(req['messages'], limit, req['temperature'])
        key = run.base.request_sha256(expected_wire)
        assert out['request_sha256'] == key
        assert api[key]['request'] == expected_wire
        assert api[key]['output']['raw_response'] == out['raw_response']
        if out.get('budget_recovery'):
            original = wire(req['messages'], max(2048, req['max_new_tokens']), req['temperature'])
            assert run.base.request_sha256(original) == out['budget_recovery']['original_request_sha256']
            assert any(e['original_request_sha256'] == out['budget_recovery']['original_request_sha256'] and e['attempts'][-1]['request_sha256'] == key for e in events)
            recovered.append(f.stem)
    run.dump(output / 'budget_recovery_audit.json', dict(status='passed',
        messages_unchanged=True, wire_requests_verified=True, recovered_calls=len(recovered),
        recovered_call_keys=recovered, caps=list(CAPS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=['gpt5', 'deepseek'], required=True)
    parser.add_argument('--env-file', type=Path, default=run.base.DEFAULT_ENV)
    args = parser.parse_args()
    output = run.OUTPUT / args.backbone
    p = run.gpt.verify(output)
    amendment = dict(kind='retry_truncated_memory_with_larger_cap',
        base_protocol_sha256=run.sha(output / 'protocol.json'),
        script_sha256=run.sha(__file__), caps=list(CAPS),
        retry_trigger='finish_reason=length only; never provider refusal',
        original_prompts_sampling_and_algorithm_unchanged=True,
        existing_successful_call_cache_retained=True,
        cached_request_identity='Original logical request plus explicit recovery metadata and wire hashes')
    run.freeze(output / 'budget_recovery_amendment.json', amendment)
    old_errors = output / 'memory/run_errors.json'
    if old_errors.exists():
        old_errors.rename(old_errors.with_name('run_errors_before_recovery_' + str(time.time_ns()) + '.json'))
    run.configure(p)
    run.runner.memory_api = SimpleNamespace(Model=DeepModel if args.backbone == 'deepseek' else
        lambda root, env, workers, floor: GPTModel(root, env, p['expected_model']))
    run.runner.memory_stage(output, args.env_file)
    audit_recovery(output)
    run.runner.answers(output, args.env_file)
    run.analyze(output)


if __name__ == '__main__':
    main()
