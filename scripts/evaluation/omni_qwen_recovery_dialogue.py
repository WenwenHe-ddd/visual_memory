"""One bounded corrective dialogue after the six preserved format attempts."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluation import omni_qwen_recovery_20261105 as recovery

OUTPUT = recovery.OUTPUT
SUCCESS_KEY = '43339008e7f8bfa271bf603111c128ab7656abd45ec6ec150626c1b9fb9488ef'
POLICY = (recovery.POLICY + ' After six unsuccessful format attempts, allow one corrective dialogue '
          'containing the original source, the first failed raw response as assistant content, and '
          'structural validation feedback as user content. Maximum seven attempts; no manual JSON edits.')


def correction_prompt(out, response_format):
    feedback = recovery.validation_feedback(out, response_format)
    errors = feedback.get('errors', [])
    if errors and errors[0]['actual_type'] == 'null' and errors[0]['expected'] == 'string':
        explanation = 'at ' + errors[0]['path'] + ' because null is not a string'
    else:
        explanation = 'with these structural diagnostics: ' + json.dumps(feedback, sort_keys=True)
    return ('Your JSON failed schema validation ' + explanation + '. Return a corrected complete JSON object '
            'using only the original source. Every relation must have two explicitly named entity endpoints. '
            'Represent unary states such as being switched on as attributes of the original entity. '
            'Do not invent an endpoint or leave any null subject, predicate, or object. Output JSON only.')


class DialogueBroker(recovery.OriginalBroker):
    def client(self):
        broker = self
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, *, messages, model=None, max_tokens=None, temperature=0,
                       response_format=None, **kwargs):
                assert model in ('local-qwen', 'Qwen3-VL-8B-Instruct')
                if response_format == {'type': 'json_object'}:
                    assert any('expert knowledge extraction system' in m['content'] for m in messages
                               if isinstance(m['content'], str))
                    response_format = dict(type='json_schema', json_schema=dict(name='omni_entities', schema=recovery.ENTITY_SCHEMA))
                cap = max_tokens or broker.protocol['memory_max_new_tokens']
                attempt_keys, causes = [], []
                request_messages = copy.deepcopy(messages)
                out, first_out = None, None
                for attempt in range(7):
                    if attempt == 6 and causes[-1] == 'format' and response_format:
                        request_messages = copy.deepcopy(messages) + [
                            dict(role='assistant', content=first_out['raw_response']),
                            dict(role='user', content=correction_prompt(first_out, response_format))]
                    elif attempt and causes[-1] != 'length':
                        request_messages = copy.deepcopy(messages)
                        request_messages.insert(0, dict(role='system', content=recovery.retry_message(attempt, out, response_format)))
                    out = broker.request(request_messages, cap, response_format=response_format,
                        requested_temperature=temperature, allow_invalid=True)
                    first_out = first_out or out
                    attempt_keys.append(out['call_key'])
                    if (out['raw_response'].strip() and not out['hit_token_limit']
                            and out.get('structured_output_valid') is not False):
                        if attempt:
                            record = dict(owner=getattr(broker.local, 'owner', 'internal'),
                                attempt_keys=attempt_keys, causes=causes, final_key=out['call_key'], status='recovered')
                            if attempt >= 3:
                                record['recovery_amendment'] = '20261105'
                            if attempt == 6:
                                record['dialogue_amendment'] = '20261105-dialogue'
                            with broker.log_lock:
                                with (broker.root / 'transport_recoveries.jsonl').open('a') as stream:
                                    stream.write(json.dumps(record) + '\n')
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out['raw_response']))])
                    cause = 'length' if out['hit_token_limit'] else 'format'
                    causes.append(cause)
                    if cause == 'length':
                        cap = min(16384, max(4096, cap * 2))
                failure = dict(owner=getattr(broker.local, 'owner', 'internal'), attempt_keys=attempt_keys,
                               causes=causes, error='All seven memory generation attempts failed under audited amendments')
                broker.errors.append(failure)
                recovery.prior.base.dump(broker.root / 'generation_errors.json', broker.errors)
                raise RuntimeError(failure['error'])
        return Client()


def prepare():
    base = recovery.prior.base
    original = recovery.verify_amendment()
    path = OUTPUT / 'recovery_dialogue_amendment.json'
    if path.exists():
        return verify_amendment()
    sources = [Path(__file__), base.ROOT / 'scripts/analysis/audit_omni_recovery_dialogue.py']
    proof = base.read(OUTPUT / recovery.ARCHIVE / 'dialogue_preflight.json')
    assert proof['structured_output_valid'] and proof['call_key'] == SUCCESS_KEY
    extra = base.read(OUTPUT / 'memory/generation_errors.json')[-1]['attempt_keys'][3:] + [SUCCESS_KEY]
    value = dict(policy=POLICY, prior_amendment_sha256=base.sha(OUTPUT / 'recovery_amendment.json'),
        source_sha256={str(p): base.sha(p) for p in sources},
        preserved_preflight_calls={str(OUTPUT / 'memory/calls' / (k + '.json')):
                                  base.sha(OUTPUT / 'memory/calls' / (k + '.json')) for k in extra},
        original_calls_preserved=original['original_calls_preserved'],
        success_key=SUCCESS_KEY)
    value['signature'] = base.obj(value)
    base.freeze(path, value)
    for source in sources:
        recovery.shutil.copyfile(source, OUTPUT / recovery.ARCHIVE / source.name)
    recovery.shutil.copyfile(OUTPUT / 'memory/generation_errors.json',
                             OUTPUT / recovery.ARCHIVE / 'six_attempts_failure.json')
    return value


def verify_amendment(output=OUTPUT):
    base = recovery.prior.base
    recovery.verify_amendment(output)
    value = base.read(output / 'recovery_dialogue_amendment.json')
    assert value['signature'] == base.obj({k: v for k, v in value.items() if k != 'signature'})
    assert base.sha(output / 'recovery_amendment.json') == value['prior_amendment_sha256']
    for section in ('source_sha256', 'preserved_preflight_calls'):
        for path, expected in value[section].items():
            assert base.sha(path) == expected, path
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'preflight', 'all', 'analyze'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    verify_amendment()
    base = recovery.prior.base
    if args.stage == 'preflight':
        recovery.RecoveryBroker = DialogueBroker
        recovery.preflight()
        return
    recovery.prior.install()
    base.OmniBroker = DialogueBroker
    try:
        if args.stage == 'all':
            assert base.read(OUTPUT / recovery.ARCHIVE / 'gpu_preflight.json')['status'] == 'passed'
            for name in ('errors.json', 'native_errors.json', 'memory/generation_errors.json'):
                (OUTPUT / name).unlink(missing_ok=True)
            base.dump(OUTPUT / 'execution.json', dict(status='resuming', recovery_amendment='20261105-dialogue'))
            base.run(OUTPUT, 'cuda:0')
        recovery.prior.audit_transport(OUTPUT)
        from scripts.analysis.audit_omni_recovery_dialogue import analyze
        analyze(OUTPUT)
    except BaseException as exc:
        base.dump(OUTPUT / 'execution.json', dict(status='failed', error=str(exc), recovery_amendment='20261105-dialogue'))
        raise


if __name__ == '__main__':
    main()
