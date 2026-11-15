"""Fresh seed-1024 Open answers with the original MemEye messages and judge.

The frozen, shared captions and evaluation gold are inputs. Answer caches belong
only to this run. Judge requests retain the original prompt and seed 42.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for key, value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(key, value)

from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_method_reader import messages
from scripts.memory.text_memory_transport import PerRowTemperatureSampler

SEED = 1024
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs'
DEFAULT = ROOT / 'results/memeye_qwen_seed1024_full_20261105'
sha, digest, rows, dump = common.sha, common.digest, common.read_lines, common.dump


def append(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def frozen_rows(path, values):
    path = Path(path)
    data = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values)
    if path.exists() and path.read_text() != data:
        raise ValueError('Frozen output differs: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


class SeededMultimodalQwen:
    """Same native Qwen transport; both RNG boundaries use the recorded seed."""

    def __init__(self, device='cuda:0', seed=SEED):
        import numpy as np
        import torch
        self.seed = int(seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        original = common.SEED
        try:
            common.SEED = self.seed
            self.processor, self.model, self.environment = common.load_runtime(device)
        finally:
            common.SEED = original
        self.environment = {**self.environment, 'seed': self.seed,
                            'seeded_transport_sha256': sha(__file__)}
        self.torch = torch

    def generate(self, conversations, cap, sampling_temperature=0.):
        from PIL import Image
        has_images = any(isinstance(m['content'], list) and any(p['type'] == 'image' for p in m['content'])
                         for conversation in conversations for m in conversation)
        if has_images and len(conversations) > 2:
            return [out for i in range(0, len(conversations), 2)
                    for out in self.generate(conversations[i:i + 2], cap, sampling_temperature)]
        rendered, images = [], []
        for conversation in conversations:
            clean = copy.deepcopy(conversation)
            for message in clean:
                if not isinstance(message['content'], list):
                    continue
                for part in message['content']:
                    if part['type'] == 'image':
                        path = Path(part.pop('path'))
                        if sha(path) != part.pop('sha256'):
                            raise ValueError('Image changed: ' + str(path))
                        with Image.open(path) as source:
                            images.append(source.convert('RGB'))
            rendered.append(self.processor.apply_chat_template(clean, tokenize=False, add_generation_prompt=True))
        try:
            inputs = self.processor(text=rendered, images=images or None, padding=True,
                                    truncation=False, return_tensors='pt').to(self.model.device)
        finally:
            for image in images:
                image.close()
        if inputs['input_ids'].shape[1] + cap > self.model.config.text_config.max_position_embeddings:
            raise ValueError('Refusing Qwen context truncation')
        lengths = inputs['attention_mask'].sum(1).tolist()
        options = {}
        if sampling_temperature > 0:
            options['logits_processor'] = [PerRowTemperatureSampler(sampling_temperature, self.seed)]
        self.torch.manual_seed(self.seed)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=cap,
                pad_token_id=self.processor.tokenizer.pad_token_id, **options)
        tails = generated[:, inputs['input_ids'].shape[1]:]
        texts = self.processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos]) | {self.processor.tokenizer.eos_token_id}
        result = []
        for text, tokens, length in zip(texts, tails.tolist(), lengths):
            stop = next((i for i, token in enumerate(tokens) if token in stops), None)
            result.append({'raw_response': text.strip(), 'input_tokens': int(length),
                'generated_tokens': stop + 1 if stop is not None else len(tokens),
                'hit_token_limit': stop is None and len(tokens) >= cap, 'batch_size': len(texts)})
        return result


def run_manifest(root, method, device='cuda:0', model=None):
    """Generate all 371 Open answers, preserving native method message roles."""
    root = Path(root)
    manifest = rows(root / 'reader_manifest.jsonl')
    canonical = common.by_id(rows(CANONICAL / 'questions.jsonl'))
    by_id = common.by_id(manifest)
    if len(manifest) != 371 or set(by_id) != set(canonical):
        raise ValueError('Incomplete Open cohort')
    image_hashes = {}
    for row in manifest:
        if any(row[k] != canonical[row['id']][k] for k in ('question', 'task', 'point')):
            raise ValueError('Open question metadata changed')
        if set(row) & {'options', 'gold_choice', 'ground_truth', 'answer', 'clue', 'gold_ids'}:
            raise ValueError('Evaluation-only metadata in reader manifest')
        for message in messages(row):
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image':
                        image_hashes[part['path']] = part['sha256']
    out = root / 'reader'
    contract = {'dataset': 'MemEye Open', 'method_id': method, 'questions': 371,
        'seed': SEED, 'do_sample': False, 'max_new_tokens': 512, 'batch_size': 1,
        'answer_model': 'Qwen/Qwen3-VL-8B-Instruct', 'model_revision': common.MODEL_REVISION,
        'source_sha256': {str(p.resolve()): sha(p) for p in [Path(__file__),
            Path(common.__file__), root / 'reader_manifest.jsonl', root / 'protocol.json',
            CANONICAL / 'questions.jsonl']}, 'image_sha256': image_hashes,
        'gold_sha256': sha(CANONICAL / 'gold.jsonl'),
        'historical_answer_reuse': False, 'gold_parsed_during_generation': False}
    protocol = common.ensure_protocol(out / 'protocol.json', contract)
    path = out / 'responses.jsonl'
    existing = rows(path) if path.exists() else []
    done = common.by_id(existing)
    calls = out / 'calls'
    calls.mkdir(exist_ok=True)
    for row in existing:
        request = {'messages': messages(by_id[row['id']]), 'seed': SEED, 'cap': 512,
                   'experiment_signature': protocol['experiment_signature']}
        call = json.loads((calls / (digest(request) + '.json')).read_text())
        if (row['request_sha256'] != digest(request) or call['request'] != request
                or any(row[k] != v for k, v in call['output'].items())):
            raise ValueError('Saved Open answer provenance differs')
    pending = [row for row in manifest if row['id'] not in done]
    if pending:
        model = model or SeededMultimodalQwen(device)
        if model.seed != SEED:
            raise ValueError('Wrong runtime seed')
        dump(out / 'environment.json', model.environment)
    for row in pending:
        request = {'messages': messages(row), 'seed': SEED, 'cap': 512,
                   'experiment_signature': protocol['experiment_signature']}
        key = digest(request)
        call_path = calls / (key + '.json')
        started = time.monotonic()
        if call_path.exists():
            call = json.loads(call_path.read_text())
            if call['request'] != request:
                raise ValueError('Run-local Open cache differs')
            answer = call['output']
        else:
            answer = model.generate([request['messages']], 512)[0]
            dump(call_path, {'request': request, 'output': answer})
        result = {**answer, **{k: row[k] for k in ('id', 'task', 'point')},
            'method_id': method, 'seed': SEED, 'request_sha256': key,
            'experiment_signature': protocol['experiment_signature'],
            'run_id': protocol['run_id'], 'historical_answer_reuse': False,
            'elapsed_seconds': time.monotonic() - started}
        append(path, result)
        done[row['id']] = result
        if len(done) % 10 == 0 or len(done) == 371:
            print(json.dumps({'stage': 'open_answers', 'method': method, 'completed': len(done), 'total': 371}), flush=True)
    status = {'status': 'complete' if len(done) == 371 else 'partial',
        'questions': 371, 'answered': len(done), 'seed': SEED,
        'empty_answers': sum(not r['raw_response'] for r in done.values()),
        'token_limit_hits': sum(r['hit_token_limit'] for r in done.values()),
        'responses_sha256': sha(path), 'protocol_sha256': sha(out / 'protocol.json')}
    dump(out / 'completion.json', status)
    return status


def collect(root, hippo=False):
    root = Path(root)
    location = root if hippo else root / 'reader'
    seal = json.loads((location / 'completion.json').read_text())
    response_path = location / ('responses.jsonl' if not hippo else 'qa_responses.jsonl')
    answers = common.by_id(rows(response_path))
    if seal['status'] != 'complete' or len(answers) != 371 or seal['responses_sha256'] != sha(response_path):
        raise ValueError('All Open answers must be completed and sealed before judging')
    questions = rows(CANONICAL / 'questions.jsonl')
    gold = common.by_id(rows(CANONICAL / 'gold.jsonl'))
    values = [{**{k: q[k] for k in ('id', 'task', 'point', 'question')},
        'ground_truth': gold[q['id']]['answer'], 'model_output': answers[q['id']]['raw_response'],
        'source_record_id': q['id'],
        'answer_model': 'Qwen/Qwen3-VL-8B-Instruct', 'source_seed': SEED,
        'source_responses_sha256': sha(response_path),
        'source_hit_token_limit': answers[q['id']]['hit_token_limit']} for q in questions]
    frozen_rows(root / 'judge_inputs.jsonl', values)
    return root / 'judge_inputs.jsonl'


def judge(root, method, label, workers=8):
    from types import SimpleNamespace
    from scripts.evaluation import memeye_open_method_judge as original
    root = Path(root)
    args = SimpleNamespace(input=root / 'judge_inputs.jsonl', output=root / 'judge',
        method_id=method, method_label=label, source_protocol=root / 'protocol.json',
        cache_root=[], env_file=Path('/storage/hewenwen/NTU Decentralized Agent/.env'),
        workers=workers, max_attempts=5, limit=0)
    protocol, requests, question_map = original.prepare(args)
    if protocol['seed'] != 42:
        raise ValueError('Judge seed must stay 42')
    report = original.base.run(args, protocol, requests, question_map)
    original.publish_summary(args, protocol, report)
    if report['status'] != 'complete':
        raise RuntimeError('Incomplete judge coverage')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'collect', 'judge'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--label')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hippomm', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'seed1024_open.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'run':
            print(json.dumps(run_manifest(args.root, args.method, args.device)), flush=True)
        elif args.phase == 'collect':
            print(collect(args.root, args.hippomm), flush=True)
        else:
            print(json.dumps(judge(args.root, args.method, args.label or args.method)), flush=True)


if __name__ == '__main__':
    main()
