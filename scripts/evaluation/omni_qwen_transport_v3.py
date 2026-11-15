"""Omni-only transport recovery: preserve every system instruction in Qwen.

The pinned Qwen template only renders the first system message.  The v2
transport inserted schema/retry instructions as separate system messages,
which hid the upstream extraction prompt and made retries ineffective.
Old transport files and prior experiment records remain unchanged.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

from scripts.evaluation import omni_qwen_transport as previous

ENTITY_SCHEMA = copy.deepcopy(previous.ENTITY_SCHEMA)
# Native ExtractedEntity.entity_type is an arbitrary string, not an enum.
ENTITY_SCHEMA['properties']['entities']['items']['properties']['type'] = {'type': 'string'}
for node in (ENTITY_SCHEMA, ENTITY_SCHEMA['properties']['entities']['items'],
             ENTITY_SCHEMA['properties']['relations']['items']):
    node['additionalProperties'] = True  # The native parser ignores extra keys.


def merge_system_messages(messages):
    """Coalesce leading system messages losslessly; reject hidden later ones."""
    messages = copy.deepcopy(messages)
    count = 0
    while count < len(messages) and messages[count]['role'] == 'system':
        assert isinstance(messages[count]['content'], str)
        count += 1
    assert not any(m['role'] == 'system' for m in messages[count:])
    if count <= 1:
        return messages
    return [dict(role='system', content='\n\n'.join(m['content'] for m in messages[:count]))] + messages[count:]


class MergedSystemProcessor:
    def __init__(self, processor):
        self.original = processor

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        return self.original.apply_chat_template(merge_system_messages(messages), *args, **kwargs)


class LocalMultimodalQwen(previous.LocalMultimodalQwen):
    def __init__(self, device):
        super().__init__(device)
        self.processor = MergedSystemProcessor(self.processor)
        self.environment['system_message_policy'] = 'Merge all leading system messages before native Qwen template.'

    def generate(self, conversations, cap, sampling_temperature=0.):
        results = super().generate(conversations, cap, sampling_temperature)
        for messages, out in zip(conversations, results):
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            out['rendered_prompt_sha256'] = hashlib.sha256(rendered.encode()).hexdigest()
        return results


class OmniBroker(previous.OmniBroker):
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
                for attempt in range(3):
                    if attempt and causes[-1] != 'length':
                        request_messages = copy.deepcopy(messages)
                        request_messages.insert(0, dict(role='system', content=
                            f'Format retry {attempt}: return one JSON object matching the supplied schema. '
                            'Entities must be objects with a string name; relations need subject, predicate, object. '
                            'Use only facts in the original source text. Do not use strings in place of entity objects.'))
                    out = broker.request(request_messages, cap, response_format=response_format,
                        requested_temperature=temperature, allow_invalid=True)
                    attempt_keys.append(out['call_key'])
                    if (out['raw_response'].strip() and not out['hit_token_limit']
                            and out.get('structured_output_valid') is not False):
                        if attempt:
                            record = dict(owner=getattr(broker.local, 'owner', 'internal'),
                                attempt_keys=attempt_keys, causes=causes, final_key=out['call_key'], status='recovered')
                            with broker.log_lock:
                                with (broker.root / 'transport_recoveries.jsonl').open('a') as stream:
                                    stream.write(json.dumps(record) + '\n')
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out['raw_response']))])
                    cause = 'length' if out['hit_token_limit'] else 'format'
                    causes.append(cause)
                    if cause == 'length':
                        cap = min(16384, max(4096, cap * 2))
                failure = dict(owner=getattr(broker.local, 'owner', 'internal'), attempt_keys=attempt_keys,
                               causes=causes, error='All three memory generation attempts failed')
                broker.errors.append(failure)
                previous.write_json(broker.root / 'generation_errors.json', broker.errors)
                raise RuntimeError(failure['error'])
        return Client()
