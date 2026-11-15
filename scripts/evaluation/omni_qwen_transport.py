"""Offline Qwen transport for native Omni OpenAI-style text/image requests."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from concurrent.futures import Future
from types import SimpleNamespace

from scripts.memory.text_memory_transport import Broker, PerRowTemperatureSampler
from scripts.memory.reflexion_memory import hash_object, write_json

ENTITY_SCHEMA = {
    'type': 'object', 'required': ['entities', 'relations'], 'additionalProperties': False,
    'properties': {
        'entities': {'type': 'array', 'items': {'type': 'object', 'required': ['name'],
            'additionalProperties': False, 'properties': {
                'name': {'type': 'string'},
                'type': {'type': 'string', 'enum': ['PERSON', 'OBJECT', 'LOCATION', 'EVENT', 'CONCEPT', 'TIME', 'ORGANIZATION']},
                'attributes': {'type': 'object'}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}},
        'relations': {'type': 'array', 'items': {'type': 'object', 'required': ['subject', 'predicate', 'object'],
            'additionalProperties': False, 'properties': {
                'subject': {'type': 'string'}, 'predicate': {'type': 'string'}, 'object': {'type': 'string'},
                'attributes': {'type': 'object'}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}
    }
}


class LocalMultimodalQwen:
    def __init__(self, device):
        from scripts.evaluation.mem_gallery_master_reader import load_runtime
        import torch
        self.processor, self.model, self.environment = load_runtime(device)
        self.torch = torch

    def generate(self, conversations, cap, sampling_temperature=0.):
        has_images = any(isinstance(m['content'], list) and any(p['type'] == 'image' for p in m['content'])
                         for messages in conversations for m in messages)
        if has_images and len(conversations) > 2:
            return [output for start in range(0, len(conversations), 2)
                    for output in self.generate(conversations[start:start + 2], cap, sampling_temperature)]
        from PIL import Image
        rendered, images = [], []
        for messages in conversations:
            clean = copy.deepcopy(messages)
            for message in clean:
                if not isinstance(message['content'], list):
                    continue
                for part in message['content']:
                    if part['type'] == 'image':
                        path = Path(part.pop('path'))
                        expected = part.pop('sha256')
                        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                            raise ValueError('Image changed: ' + str(path))
                        with Image.open(path) as im:
                            images.append(im.convert('RGB'))
            rendered.append(self.processor.apply_chat_template(clean, tokenize=False, add_generation_prompt=True))
        try:
            inputs = self.processor(text=rendered, images=images or None, padding=True,
                                    truncation=False, return_tensors='pt').to(self.model.device)
        finally:
            for im in images:
                im.close()
        if inputs['input_ids'].shape[1] + cap > self.model.config.text_config.max_position_embeddings:
            raise RuntimeError('Refusing Qwen context truncation')
        lengths = inputs['attention_mask'].sum(1).tolist()
        options = {}
        if sampling_temperature > 0:
            options['logits_processor'] = [PerRowTemperatureSampler(sampling_temperature, 42)]
        self.torch.manual_seed(42)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=cap,
                pad_token_id=self.processor.tokenizer.pad_token_id, **options)
        tails = generated[:, inputs['input_ids'].shape[1]:]
        texts = self.processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.model.generation_config.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.processor.tokenizer.eos_token_id}
        outputs = []
        for text, tokens, length in zip(texts, tails.tolist(), lengths):
            stop = next((i for i, token in enumerate(tokens) if token in stop_ids), None)
            outputs.append(dict(raw_response=text.strip(), input_tokens=int(length),
                generated_tokens=stop + 1 if stop is not None else len(tokens),
                hit_token_limit=stop is None and len(tokens) >= cap, batch_size=len(texts)))
        return outputs


class OmniBroker(Broker):
    def __init__(self, model, root, protocol):
        self.errors = []
        self.inflight = {}
        super().__init__(model, root, protocol, protocol['generation_batch_size'])

    def normalize(self, messages):
        messages = copy.deepcopy(messages)
        for message in messages:
            if isinstance(message['content'], str):
                continue
            for part in message['content']:
                if part['type'] != 'image_url':
                    continue
                uri = part['image_url']['url']
                if not uri.startswith('data:image/') or ';base64,' not in uri:
                    raise ValueError('Only inline local image input is permitted')
                data = base64.b64decode(uri.split(';base64,', 1)[1], validate=True)
                key = hashlib.sha256(data).hexdigest()
                target = self.root / 'transport_images' / (key + '.image')
                target.parent.mkdir(parents=True, exist_ok=True)
                with self.log_lock:
                    if target.exists():
                        assert target.read_bytes() == data
                    else:
                        target.write_bytes(data)
                part.clear()
                part.update(type='image', path=str(target), sha256=key)
        return messages

    def request(self, messages, max_tokens, *, kind='memory', response_format=None,
                requested_temperature=0, owner=None, allow_invalid=False):
        messages = self.normalize(messages)
        if response_format:
            schema = response_format.get('json_schema', {}).get('schema')
            instruction = 'Return only valid JSON, without Markdown fences.'
            if schema:
                instruction += ' Follow this JSON schema exactly: ' + json.dumps(schema)
            messages.insert(0, dict(role='system', content=instruction))
        request = dict(messages=messages, max_new_tokens=int(max_tokens), kind=kind,
                       do_sample=kind == 'memory' and requested_temperature > 0,
                       temperature=float(requested_temperature) if kind == 'memory' else 0.,
                       seed=42, response_format=response_format,
                       experiment_signature=self.protocol['experiment_signature'])
        key = hash_object(request)
        path = self.root / 'calls' / (key + '.json')
        owner = owner or getattr(self.local, 'owner', 'internal')
        future, first = None, False
        with self.log_lock:
            if path.exists():
                result = json.loads(path.read_text())
                assert result['request'] == request
                reused = True
            else:
                future = self.inflight.get(key)
                first = future is None
                if first:
                    future = self.inflight[key] = Future()
                    self.pending.put((request, future))
                reused = not first
        if future is not None:
            try:
                result = future.result()
            except BaseException as exc:
                self.errors.append(dict(owner=owner, key=key, error=str(exc)))
                raise
            if first:
                with self.log_lock:
                    write_json(path, result)
                    self.inflight.pop(key, None)
        with self.log_lock:
            with (self.root / 'call_events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(owner=owner, key=key, kind=kind, reused=reused)) + '\n')
        out = result['output']
        if out['hit_token_limit'] or not out['raw_response'].strip() or out.get('structured_output_valid') is False:
            if allow_invalid:
                return dict(out, call_key=key)
            error = dict(owner=owner, key=key, output=out)
            self.errors.append(error)
            write_json(self.root / 'generation_errors.json', self.errors)
            raise RuntimeError('Invalid/truncated Omni generation; raw result retained: ' + key)
        return dict(out, call_key=key)

    def client(self):
        broker = self
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
            def create(self, *, messages, model=None, max_tokens=None, temperature=0,
                       response_format=None, **kwargs):
                assert model in ('local-qwen', 'Qwen3-VL-8B-Instruct')
                if response_format == {'type': 'json_object'}:
                    assert any('expert knowledge extraction system' in m['content'] for m in messages if isinstance(m['content'], str))
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
                    valid = (out['raw_response'].strip() and not out['hit_token_limit']
                             and out.get('structured_output_valid') is not False)
                    if valid:
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
                write_json(broker.root / 'generation_errors.json', broker.errors)
                raise RuntimeError(failure['error'])
        return Client()

    def check(self):
        if self.errors:
            raise RuntimeError('Native core caught a failed generation; see generation_errors.json')
