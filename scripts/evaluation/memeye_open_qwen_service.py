"""Loopback-only, serialized local Qwen inference for MemEye Open experiments.

The service preserves native messages and generation parameters. Every request
is cached by its full content plus the signed model/runtime contract; cache
recovery also prevents duplicate generation after a disconnected client.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_master_reader as common

DEFAULT_URL = 'http://127.0.0.1:18761'


class RPCQwen:
    def __init__(self, url=DEFAULT_URL):
        self.url = url.rstrip('/')
        with urlopen(self.url + '/health', timeout=15) as response:
            health = json.load(response)
        self.environment = health['environment']
        self.service_signature = health['service_signature']

    def generate(self, conversations, cap, sampling_temperature=0.):
        body = {'conversations': conversations, 'cap': int(cap),
                'sampling_temperature': float(sampling_temperature),
                'service_signature': self.service_signature}
        request = Request(self.url + '/generate', data=json.dumps(body, ensure_ascii=False).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=3600) as response:
            result = json.load(response)
        if result.get('error'):
            raise RuntimeError(result['error'])
        if result['service_signature'] != self.service_signature:
            raise ValueError('Qwen service contract changed')
        return result['outputs']


def serve(args):
    from scripts.evaluation.omni_qwen_transport import LocalMultimodalQwen
    args.output.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), Path(common.__file__), ROOT / 'scripts/evaluation/omni_qwen_transport.py',
            ROOT / 'scripts/memory/text_memory_transport.py']
    contract = {'model': common.model_provenance(), 'device': args.device,
        'seed': 42, 'dtype': 'bfloat16', 'attention_implementation': 'sdpa',
        'image_max_pixels': common.QUERY_IMAGE_MAX_PIXELS, 'input_truncation': False,
        'template': 'Unchanged native Qwen template, supplied messages preserved.',
        'batch_policy': 'Explicit caller batch, requests serialized; image batches split at two by original transport.',
        'implementation_sha256': {str(p): common.sha(p) for p in code}}
    protocol = common.ensure_protocol(args.output / 'protocol.json', contract)
    model = LocalMultimodalQwen(args.device)
    signature = protocol['experiment_signature']
    common.dump(args.output / 'environment.json', model.environment)
    call_dir = args.output / 'calls'; call_dir.mkdir(exist_ok=True)
    mutex = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            pass

        def reply(self, data, status=200):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Durable call file remains available to the retrying client.

        def do_GET(self):
            if self.path != '/health':
                return self.reply({'error': 'unknown endpoint'}, 404)
            self.reply({'status': 'ready', 'service_signature': signature,
                        'environment': model.environment})

        def do_POST(self):
            if self.path != '/generate':
                return self.reply({'error': 'unknown endpoint'}, 404)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(size))
                if set(request) != {'conversations', 'cap', 'sampling_temperature', 'service_signature'}:
                    raise ValueError('Unexpected inference request fields')
                if request['service_signature'] != signature:
                    raise ValueError('Service signature mismatch')
                conversations = request['conversations']
                if not isinstance(conversations, list) or not 1 <= len(conversations) <= 32:
                    raise ValueError('Invalid batch size')
                if type(request['cap']) is not int or not 1 <= request['cap'] <= 8192:
                    raise ValueError('Invalid generation limit')
                if not isinstance(request['sampling_temperature'], (float, int)) or not 0 <= request['sampling_temperature'] <= 2:
                    raise ValueError('Invalid sampling temperature')
                for messages in conversations:
                    for message in messages:
                        if isinstance(message['content'], list):
                            for part in message['content']:
                                if part['type'] == 'image':
                                    path = Path(part['path']).resolve()
                                    if not path.is_relative_to(ROOT) or common.sha(path) != part['sha256']:
                                        raise ValueError('Image path/hash differs')
                key = common.digest(request)
                path = call_dir / (key + '.json')
                started = time.time()
                with mutex:
                    reused = path.exists()
                    if reused:
                        result = json.loads(path.read_text())
                        if result['request'] != request:
                            raise ValueError('Cached inference request differs')
                    else:
                        outputs = model.generate(conversations, request['cap'], request['sampling_temperature'])
                        if len(outputs) != len(conversations):
                            raise ValueError('Wrong output batch size')
                        result = {'request': request, 'outputs': outputs, 'service_signature': signature,
                                  'request_sha256': key, 'elapsed_seconds': time.time() - started}
                        common.dump(path, result)
                    with (args.output / 'events.jsonl').open('a') as stream:
                        stream.write(json.dumps({'request_sha256': key, 'reused': reused,
                            'batch_size': len(conversations), 'cap': request['cap'], 'at_unix': started}) + '\n')
                        stream.flush(); os.fsync(stream.fileno())
                self.reply(result)
            except Exception as exc:
                self.reply({'error': f'{type(exc).__name__}: {exc}'}, 500)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(json.dumps({'status': 'ready', 'port': args.port, 'device': args.device,
                      'service_signature': signature}), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=18761)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/memeye_qwen_other_open_20261105/qwen_service')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'service.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        serve(args)


if __name__ == '__main__':
    main()
