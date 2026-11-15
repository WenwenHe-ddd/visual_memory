"""Local, audited Qwen transport for pinned third-party memory algorithms.

Workers own independent task states. Only this broker calls the shared model;
no OpenAI HTTP requests are issued. Identical requests can replay their outputs.
"""
import copy
import json
import queue
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from scripts.memory.reflexion_memory import hash_object, write_json


class PerRowTemperatureSampler:
    """Independent categorical RNG streams, unaffected by neighboring rows."""
    def __init__(self, temperature, seed):
        self.temperature, self.seed = temperature, seed
        self.generators = None

    def __call__(self, input_ids, scores):
        import torch
        if self.generators is None:
            self.generators = [torch.Generator(device=scores.device).manual_seed(self.seed)
                               for _ in range(scores.shape[0])]
        probabilities = torch.softmax(scores.float() / self.temperature, dim=-1)
        selected = torch.stack([torch.multinomial(probabilities[i], 1, generator=generator)
                                for i, generator in enumerate(self.generators)]).squeeze(-1)
        forced = torch.full_like(scores, -float('inf'))
        forced.scatter_(1, selected[:, None], 0.0)
        return forced


class Broker:
    def __init__(self, model, root, protocol, batch_size=4):
        self.model, self.root, self.protocol = model, root, protocol
        self.batch_size = batch_size
        self.pending = queue.Queue()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.local = threading.local()
        self.log_lock = threading.Lock()
        self.thread.start()

    def request(self, messages, max_tokens, *, kind='memory', response_format=None,
                requested_temperature=0, owner=None):
        messages = copy.deepcopy(messages)
        if any(not isinstance(m['content'], str) for m in messages):
            raise ValueError('text-memory transport refuses native images/non-text payloads')
        # API structured output is represented as an explicit schema instruction.
        # Do not fabricate valid JSON or silently retry malformed generations.
        if response_format:
            schema = response_format.get('json_schema', {}).get('schema')
            instruction = 'Return only valid JSON, without Markdown fences.'
            if schema:
                instruction += ' Follow this JSON schema exactly: ' + json.dumps(schema)
            messages.insert(0, {'role': 'system', 'content': instruction})
        request = {'messages': messages, 'max_new_tokens': int(max_tokens),
                   'kind': kind, 'do_sample': kind == 'memory' and requested_temperature > 0,
                   'temperature': float(requested_temperature) if kind == 'memory' else 0.0,
                   'seed': self.protocol['seed'],
                   'response_format': response_format,
                   'experiment_signature': self.protocol['experiment_signature']}
        key = hash_object(request)
        path = self.root / 'calls' / (key + '.json')
        owner = owner or getattr(self.local, 'owner', 'internal')
        if path.exists():
            result = json.loads(path.read_text())
            if result['request'] != request:
                raise ValueError('cached request mismatch')
            reused = True
        else:
            future = Future()
            self.pending.put((request, future))
            result = future.result()
            with self.log_lock:
                write_json(path, result)
            reused = False
        with self.log_lock:
            with (self.root / 'call_events.jsonl').open('a') as f:
                f.write(json.dumps({'owner': owner, 'key': key, 'kind': kind,
                    'reused': reused, 'requested_temperature': requested_temperature}) + '\n')
        return {**result['output'], 'call_key': key}

    def _loop(self):
        while not self.closed.is_set() or not self.pending.empty():
            try:
                first = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + 0.06
            while len(batch) < self.batch_size and time.monotonic() < deadline:
                try:
                    batch.append(self.pending.get(timeout=max(.001, deadline-time.monotonic())))
                except queue.Empty:
                    break
            try:
                # Different generation budgets must not alter a method's cap.
                groups = {}
                for req, future in batch:
                    groups.setdefault((req['max_new_tokens'], req['temperature']), []).append((req, future))
                for (cap, temperature), group in groups.items():
                    # Bound the padded KV allocation as independent tasks batch up.
                    # Split batches only; no prompt truncation or changed token caps.
                    chunks, chunk, longest = [], [], 0
                    for req, future in group:
                        if hasattr(self.model, 'processor'):
                            rendered = self.model.processor.apply_chat_template(req['messages'], tokenize=False, add_generation_prompt=True)
                            length = len(self.model.processor.tokenizer.encode(rendered, add_special_tokens=False))
                        else:
                            length = 0  # Test transport without a model tokenizer.
                        size = max(longest, length) + cap
                        if chunk and size * (len(chunk) + 1) > self.protocol.get('max_padded_batch_tokens', 48000):
                            chunks.append(chunk)
                            chunk, longest = [], 0
                        chunk.append((req, future))
                        longest = max(longest, length)
                    if chunk:
                        chunks.append(chunk)
                    for chunk in chunks:
                        outputs = self.model.generate([r['messages'] for r, _ in chunk], cap,
                                                      sampling_temperature=temperature)
                        for (req, future), out in zip(chunk, outputs):
                            if req['response_format']:
                                try:
                                    value = json.loads(out['raw_response'].strip())
                                    schema = req['response_format'].get('json_schema', {}).get('schema')
                                    if schema:
                                        import jsonschema
                                        jsonschema.validate(value, schema)
                                    out['structured_output_valid'] = True
                                except Exception as exc:
                                    out['structured_output_valid'] = False
                                    out['structured_output_error'] = str(exc)[:400]
                            future.set_result({'request': req, 'output': out})
            except BaseException as exc:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)

    def close(self):
        self.closed.set()
        self.thread.join()


def client_factory(broker):
    class LocalClient:
        def __init__(self, *args, **kwargs):
            self.owner = getattr(broker.local, 'owner', 'internal')
            self.chat = SimpleNamespace(completions=self)

        def create(self, *, messages, max_tokens=None, temperature=0, response_format=None, **kwargs):
            out = broker.request(messages, max_tokens or broker.protocol['memory_max_new_tokens'],
                response_format=response_format, requested_temperature=temperature, owner=self.owner)
            return SimpleNamespace(id=out['call_key'], created=0,
                choices=[SimpleNamespace(message=SimpleNamespace(content=out['raw_response']))],
                usage=SimpleNamespace(prompt_tokens=out['input_tokens'], completion_tokens=out['generated_tokens']))

        def close(self):
            pass
    return LocalClient
