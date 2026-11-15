"""Retry identical failed wires; cache the first structurally valid response."""
import json, gzip, sys, time, urllib.request
from unittest.mock import patch
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path[:0] = [str(ROOT), str(ROOT / '.vendor_m2a'), str(ROOT / '.vendor')]
from scripts.memory import m2a_deepseek as api
RUN = Path(__file__).resolve().parent
config = api.provider.load_config(api.DEFAULT_ENV)
original_opener = urllib.request.build_opener
local = threading.local()
class Tap:
    def __init__(self, response): self.response = response
    def __enter__(self): self.response.__enter__(); return self
    def __exit__(self, *args): return self.response.__exit__(*args)
    def __getattr__(self, name): return getattr(self.response, name)
    def read(self, *args):
        data = self.response.read(*args)
        payload = json.loads(data)
        report = dict(request_sha256=local.key, choices=[])
        for choice in payload.get('choices', []):
            message = choice.get('message', {})
            report['choices'].append(dict(finish_reason=choice.get('finish_reason'),
                content_type=type(message.get('content')).__name__,
                tool_calls=message.get('tool_calls')))
        local.diagnostic = report
        return data
class Opener:
    def __init__(self, *args, **kwargs): self.opener = original_opener(*args, **kwargs)
    def open(self, *args, **kwargs): return Tap(self.opener.open(*args, **kwargs))
targets = []
for task in (RUN / 'memeye').iterdir():
    if not task.is_dir(): continue
    attempts = sorted(task.glob('attempt_*'))
    attempts = [p for p in attempts if p.is_dir()]
    if any((p / 'completion_manifest.json').exists() for p in attempts): continue
    attempt = attempts[-1]
    failure = json.loads((attempt / 'failure.json').read_text())
    errors = [json.loads(l) for l in (attempt / 'api_attempts.jsonl').read_text().splitlines()
              if json.loads(l)['status'] == 'error']
    last = errors[-1]
    if last['kind'] != 'protocol': raise ValueError('Only identical protocol-error retries allowed')
    targets.append((task, failure['sample_id'], last['request_sha256']))
def retry(item):
    task, question, key = item
    folder = task / 'api_cache' / key
    wire = json.loads(gzip.decompress((folder / 'request.json.gz').read_bytes()))
    assert api.provider.request_sha256(wire) == key
    events = []
    for n in range(1, 2):
        if (folder / 'response.json').exists(): break
        event = dict(task=task.name, sample_id=question, request_sha256=key, attempt=n)
        try:
            local.key = key
            result = api.chat(config, wire)
            if result['finish_reason'] not in ('stop', 'tool_calls') or result.get('refusal') or not (result['raw_response'] or result['tool_calls']):
                raise ValueError('Non-complete response')
            temporary = folder / 'recovery_response.tmp'
            temporary.write_text(json.dumps(result, ensure_ascii=False) + '\n')
            temporary.replace(folder / 'response.json')
            event.update(status='ok', response_id=result['response_id'])
            events.append(event)
            break
        except api.provider.APIError as error:
            event.update(status='error', kind=error.kind, diagnostic=getattr(local, 'diagnostic', None))
            events.append(event)
            if error.kind != 'protocol': break
            time.sleep(2)
    return events
with patch('urllib.request.build_opener', Opener), ThreadPoolExecutor(max_workers=3) as pool:
    events = [event for group in pool.map(retry, targets) for event in group]
with (RUN / 'identical_wire_retries.jsonl').open('a') as log:
    for event in events:
        log.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
