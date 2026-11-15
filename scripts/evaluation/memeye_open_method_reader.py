"""Read frozen method-specific Open contexts with the shared pinned Qwen model."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_qwen_service import RPCQwen, DEFAULT_URL

CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
sha, digest, rows, dump = common.sha, common.digest, common.read_lines, common.dump


def append(path, data):
    with path.open('a') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def write_frozen(path, data):
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in data)
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen output differs: ' + str(path))
    path.write_text(text)


def messages(row):
    if 'reader_messages' in row:
        value = row['reader_messages']
    else:
        images = [{'type': 'image', 'path': path, 'sha256': sha(path)}
                  for path in row['query_image_paths']]
        value = [{'role': 'system', 'content': row['system']},
                 {'role': 'user', 'content': images + [{'type': 'text', 'text': row['user_text']}]}]
    for message in value:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'image' and sha(part['path']) != part['sha256']:
                    raise ValueError('Reader image changed')
    return value


def prepare(args, rpc):
    source = args.root / 'reader_manifest.jsonl'
    manifest = rows(source); by_id = common.by_id(manifest)
    canonical = common.by_id(rows(CANONICAL / 'inputs/questions.jsonl'))
    if len(manifest) != 371 or set(by_id) != set(canonical):
        raise ValueError('Method must supply the complete 371-question Open cohort')
    image_hashes = {}
    for row in manifest:
        if any(row[key] != canonical[row['id']][key] for key in ('question', 'task', 'point')):
            raise ValueError('Method changed original Open question metadata')
        if set(row) & {'options', 'gold_choice', 'ground_truth', 'answer', 'clue', 'gold_ids'}:
            raise ValueError('Reader input contains evaluation-only fields')
        for message in messages(row):
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part['type'] == 'image':
                        image_hashes[part['path']] = part['sha256']
    files = [source, args.root / 'protocol.json', CANONICAL / 'inputs/questions.jsonl',
             Path(__file__), ROOT / 'scripts/evaluation/memeye_open_qwen_service.py',
             args.service_root / 'protocol.json']
    contract = {'dataset': 'MemEye Open', 'method_id': args.method_id,
        'answer_model': MODEL, 'questions': 371, 'reference_mcq_correct': args.reference_correct,
        'source_sha256': {str(p.resolve()): sha(p) for p in files},
        'image_sha256': image_hashes, 'gold_sha256': sha(CANONICAL / 'inputs/gold.jsonl'),
        'service_signature': rpc.service_signature, 'service_root': str(args.service_root.resolve()),
        'max_new_tokens': 512, 'seed': 42, 'do_sample': False, 'batch_size': 1,
        'policy': 'Frozen method-specific evidence and message roles; Open question, no choices; original raw answers retained.',
        'gold_parsed_during_generation': False}
    out = args.root / 'reader'; out.mkdir(exist_ok=True)
    return common.ensure_protocol(out / 'protocol.json', contract), manifest


def validate(root, protocol, manifest):
    contract = protocol['contract']
    if digest(contract) != protocol['experiment_signature']:
        raise ValueError('Reader contract signature changed')
    for field in ('source_sha256', 'image_sha256'):
        for path, expected in contract[field].items():
            if sha(path) != expected:
                raise ValueError('Reader source changed: ' + path)
    source = common.by_id(manifest)
    path = root / 'reader/responses.jsonl'
    existing = rows(path) if path.exists() else []
    common.by_id(existing)
    for row in existing:
        if row['id'] not in source or row['experiment_signature'] != protocol['experiment_signature']:
            raise ValueError('Response belongs to another cohort/run')
        request = {'conversations': [messages(source[row['id']])], 'cap': 512,
                   'sampling_temperature': 0.0, 'service_signature': contract['service_signature']}
        if row['service_request_sha256'] != digest(request):
            raise ValueError('Response input hash differs')
        call = json.loads((Path(contract['service_root']) / 'calls' / (digest(request) + '.json')).read_text())
        if call['request'] != request or any(row[key] != value for key, value in call['outputs'][0].items()):
            raise ValueError('Response differs from original Qwen service output')
    return existing


def run(args):
    rpc = RPCQwen(args.url)
    protocol, manifest = prepare(args, rpc)
    existing = validate(args.root, protocol, manifest)
    done = {row['id'] for row in existing}
    pending = [row for row in manifest if row['id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    for row in pending:
        conversation = messages(row)
        request = {'conversations': [conversation], 'cap': 512,
                   'sampling_temperature': 0.0, 'service_signature': rpc.service_signature}
        started = time.monotonic()
        answer = rpc.generate([conversation], 512)[0]
        result = {**answer, **{k: row[k] for k in ('id', 'task', 'point')},
            'method_id': args.method_id, 'answer_model': MODEL,
            'top_memory_ids': row.get('top_memory_ids', []),
            'run_id': protocol['run_id'], 'experiment_signature': protocol['experiment_signature'],
            'service_request_sha256': digest(request), 'elapsed_seconds': time.monotonic() - started}
        append(args.root / 'reader/responses.jsonl', result)
        done.add(row['id'])
        if len(done) % 10 == 0 or len(done) == 371:
            print(json.dumps({'method': args.method_id, 'completed': len(done), 'total': 371}), flush=True)
    existing = validate(args.root, protocol, manifest)
    status = {'status': 'complete' if len(existing) == 371 else 'partial',
        'questions': 371, 'questions_answered': len(existing),
        'token_limit_hits': sum(row['hit_token_limit'] for row in existing),
        'empty_answers': sum(not row['raw_response'] for row in existing),
        'responses_sha256': sha(args.root / 'reader/responses.jsonl') if existing else None,
        'protocol_sha256': sha(args.root / 'reader/protocol.json')}
    dump(args.root / 'reader/completion.json', status)
    print(json.dumps(status), flush=True)


def collect(args):
    protocol = json.loads((args.root / 'reader/protocol.json').read_text())
    manifest = rows(args.root / 'reader_manifest.jsonl')
    responses = common.by_id(validate(args.root, protocol, manifest))
    seal = json.loads((args.root / 'reader/completion.json').read_text())
    if seal['status'] != 'complete' or len(responses) != 371 or seal['responses_sha256'] != sha(args.root / 'reader/responses.jsonl'):
        raise ValueError('All 371 original answers must be sealed')
    gold_path = CANONICAL / 'inputs/gold.jsonl'
    if sha(gold_path) != protocol['contract']['gold_sha256']:
        raise ValueError('Open gold changed')
    gold = common.by_id(rows(gold_path))
    judge = []
    for q in manifest:
        response = responses[q['id']]
        judge.append({'id': q['id'], 'task': q['task'], 'point': q['point'], 'question': q['question'],
            'ground_truth': gold[q['id']]['answer'], 'model_output': response['raw_response'],
            'answer_model': MODEL, 'source_record_id': q['id'],
            'source_hit_token_limit': response['hit_token_limit'],
            'source_responses_sha256': seal['responses_sha256'],
            'source_service_request_sha256': response['service_request_sha256']})
    write_frozen(args.root / 'judge_inputs.jsonl', judge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'collect'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method-id', required=True)
    parser.add_argument('--reference-correct', type=int, required=True)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--service-root', type=Path, default=ROOT / 'results/memeye_qwen_other_open_20261105/qwen_service')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(); args.root = args.root.resolve()
    with (args.root / 'reader.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run if args.phase == 'run' else collect)(args)


if __name__ == '__main__':
    main()
