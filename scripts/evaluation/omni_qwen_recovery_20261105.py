"""Audited continuation of the frozen Omni v3 run after exhausted JSON retries.

Keep the original three attempts exactly. Up to three additional attempts use
schema-validator feedback, original source text, and unchanged model settings.
No response is edited, no invalid relation is silently dropped, and no gold is
read during inference. Completed scenes and exact request caches are retained.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from scripts.evaluation import mem_gallery_qwen_omni_v3 as prior
from scripts.evaluation.omni_qwen_transport_v3 import ENTITY_SCHEMA, OmniBroker as OriginalBroker

OUTPUT = prior.OUTPUT
ARCHIVE = 'recovery_20261105'
FIRST_FAILED = 'ebb5a94cbf323bf7e912fc9a305da710a0b204e60c82f5fdd2e3c22c770f52d7'
POLICY = ('Preserve original attempts 0..2; allow attempts 3..5 only after all three fail. '
          'Additional format prompts contain schema-validation feedback and original source only. '
          'Keep model, temperature, seed, schema, retrieval and reader settings unchanged; '
          'length recovery remains capped at 16384. Preserve all raw generations.')


def validation_feedback(out, response_format):
    """Return structural diagnostics only, never new entity facts or answers."""
    import jsonschema
    try:
        value = json.loads(out['raw_response'])
    except json.JSONDecodeError as exc:
        return dict(error='invalid JSON', line=exc.lineno, column=exc.colno)
    schema = (response_format or {}).get('json_schema', {}).get('schema')
    if not schema:
        return dict(error='empty or incomplete generation')
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
    return dict(errors=[dict(path='/' + '/'.join(map(str, e.absolute_path)),
                            validator=e.validator, expected=e.validator_value,
                            actual_type='null' if e.instance is None else type(e.instance).__name__)
                       for e in errors[:8]])


def retry_message(attempt, out=None, response_format=None):
    original = (f'Format retry {attempt}: return one JSON object matching the supplied schema. '
                'Entities must be objects with a string name; relations need subject, predicate, object. '
                'Use only facts in the original source text. Do not use strings in place of entity objects.')
    if attempt < 3:
        return original
    return (original + ' Audited retry extension 20261105. Validator feedback: ' +
            json.dumps(validation_feedback(out, response_format), sort_keys=True) +
            '. The subject, predicate, and object fields must each be strings, never null. '
            'A relation needs two named endpoints from the source. Unary properties or states '
            'belong in the source entity attributes, not in a relation with a missing endpoint. '
            'Regenerate from the original source; do not invent an endpoint to fill a missing value.')


class RecoveryBroker(OriginalBroker):
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
                    response_format = dict(type='json_schema', json_schema=dict(name='omni_entities', schema=ENTITY_SCHEMA))
                cap = max_tokens or broker.protocol['memory_max_new_tokens']
                attempt_keys, causes = [], []
                request_messages = copy.deepcopy(messages)
                out = None
                for attempt in range(6):
                    if attempt and causes[-1] != 'length':
                        request_messages = copy.deepcopy(messages)
                        request_messages.insert(0, dict(role='system', content=retry_message(attempt, out, response_format)))
                    out = broker.request(request_messages, cap, response_format=response_format,
                        requested_temperature=temperature, allow_invalid=True)
                    attempt_keys.append(out['call_key'])
                    if (out['raw_response'].strip() and not out['hit_token_limit']
                            and out.get('structured_output_valid') is not False):
                        if attempt:
                            record = dict(owner=getattr(broker.local, 'owner', 'internal'),
                                attempt_keys=attempt_keys, causes=causes, final_key=out['call_key'], status='recovered')
                            if attempt >= 3:
                                record['recovery_amendment'] = '20261105'
                            with broker.log_lock:
                                with (broker.root / 'transport_recoveries.jsonl').open('a') as stream:
                                    stream.write(json.dumps(record) + '\n')
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out['raw_response']))])
                    cause = 'length' if out['hit_token_limit'] else 'format'
                    causes.append(cause)
                    if cause == 'length':
                        cap = min(16384, max(4096, cap * 2))
                failure = dict(owner=getattr(broker.local, 'owner', 'internal'), attempt_keys=attempt_keys,
                               causes=causes, error='All six memory generation attempts failed under audited amendment')
                broker.errors.append(failure)
                prior.base.dump(broker.root / 'generation_errors.json', broker.errors)
                raise RuntimeError(failure['error'])
        return Client()


def verify_amendment(output=OUTPUT):
    base = prior.base
    amendment = base.read(output / 'recovery_amendment.json')
    assert amendment['signature'] == base.obj({k: v for k, v in amendment.items() if k != 'signature'})
    assert base.sha(output / 'protocol.json') == amendment['original_protocol_sha256']
    for path, expected in amendment['source_sha256'].items():
        assert base.sha(path) == expected, path
    snapshot = base.read(output / ARCHIVE / 'snapshot.json')
    assert base.sha(output / ARCHIVE / 'snapshot.json') == amendment['snapshot_sha256']
    for section in ('original_calls', 'completed_artifacts', 'archived_diagnostics'):
        for path, expected in snapshot[section].items():
            assert base.sha(path) == expected, path
    return amendment


def prepare(output=OUTPUT):
    base = prior.base
    if (output / 'recovery_amendment.json').exists():
        return verify_amendment(output)
    protocol = base.verify(output)
    assert base.read(output / 'execution.json')['status'] == 'failed'
    assert not (output / 'completion_manifest.json').exists()
    archive = output / ARCHIVE
    archive.mkdir(exist_ok=True)
    snapshot = dict(original_calls={}, completed_artifacts={}, archived_diagnostics={})
    for path in sorted((output / 'memory/calls').glob('*.json')):
        record = base.read(path)
        assert base.obj(record['request']) == path.stem
        assert record['request']['experiment_signature'] == protocol['experiment_signature']
        snapshot['original_calls'][str(path)] = base.sha(path)
    completed = []
    for manifest in sorted((output / 'memory').glob('*/complete.json')):
        completed.append(manifest.parent.name)
        snapshot['completed_artifacts'][str(manifest)] = base.sha(manifest)
        for path, expected in base.read(manifest)['artifact_sha256'].items():
            assert base.sha(path) == expected, path
            snapshot['completed_artifacts'][path] = expected
    for name in ('execution.json', 'errors.json', 'native_errors.json', 'memory/generation_errors.json',
                 'generation_environment.json', 'memory/transport_recoveries.jsonl', 'run.log', 'native.log'):
        source = output / name
        if source.exists():
            target = archive / 'diagnostics' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            snapshot['archived_diagnostics'][str(target)] = base.sha(target)
    base.freeze(archive / 'snapshot.json', snapshot)
    sources = [Path(__file__), base.ROOT / 'scripts/analysis/audit_omni_recovery_20261105.py']
    source_sha = {str(p): base.sha(p) for p in sources}
    for path in sources:
        shutil.copyfile(path, archive / path.name)
    amendment = dict(status='prepared', created_at=datetime.now(timezone.utc).isoformat(),
        original_protocol_sha256=base.sha(output / 'protocol.json'), policy=POLICY,
        source_sha256=source_sha, snapshot_sha256=base.sha(archive / 'snapshot.json'),
        completed_scenes_preserved=completed, original_calls_preserved=len(snapshot['original_calls']),
        original_progress=dict(scenes=13, answers=1056, rounds=3300, total_questions=1711, total_rounds=3962))
    amendment['signature'] = base.obj(amendment)
    base.freeze(output / 'recovery_amendment.json', amendment)
    return amendment


def preflight(output=OUTPUT):
    base = prior.base
    verify_amendment(output)
    p = base.verify(output)
    model = prior.LocalMultimodalQwen('cuda:0')
    broker = RecoveryBroker(model, output / 'memory', p)
    original = base.read(output / 'memory/calls' / (FIRST_FAILED + '.json'))
    messages = original['request']['messages'][1:]
    try:
        broker.local.owner = 'Nature_Economics_Programming_Student_Life'
        response = broker.client().create(messages=messages, model='local-qwen',
                                         response_format={'type': 'json_object'}, temperature=0)
        value = json.loads(response.choices[0].message.content)
        broker.check()
    finally:
        broker.close()
    verify_amendment(output)
    base.dump(output / ARCHIVE / 'gpu_preflight.json', dict(status='passed', original_key=FIRST_FAILED,
        entities=len(value['entities']), relations=len(value['relations']), raw_responses_unchanged=True,
        environment=model.environment))
    print('Actual failed input recovered with local Qwen; raw outputs preserved.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'preflight', 'all', 'analyze'])
    args = parser.parse_args()
    base = prior.base
    if args.stage == 'prepare':
        prepare()
        return
    if args.stage == 'preflight':
        preflight()
        return
    verify_amendment()
    prior.install()
    base.OmniBroker = RecoveryBroker
    try:
        if args.stage == 'all':
            assert base.read(OUTPUT / ARCHIVE / 'gpu_preflight.json')['status'] == 'passed'
            # The failed diagnostics are sealed in the recovery snapshot above.
            for name in ('errors.json', 'native_errors.json', 'memory/generation_errors.json'):
                (OUTPUT / name).unlink(missing_ok=True)
            base.dump(OUTPUT / 'execution.json', dict(status='resuming', recovery_amendment='20261105'))
            base.run(OUTPUT, 'cuda:0')
        prior.audit_transport(OUTPUT)
        from scripts.analysis.audit_omni_recovery_20261105 import analyze
        analyze(OUTPUT)
    except BaseException as exc:
        base.dump(OUTPUT / 'execution.json', dict(status='failed', error=str(exc), recovery_amendment='20261105'))
        raise


if __name__ == '__main__':
    main()
