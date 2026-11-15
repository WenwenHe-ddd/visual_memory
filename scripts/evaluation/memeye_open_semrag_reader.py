"""Fresh Qwen open answers for the original MemEye Semantic RAG memory.

The reader consumes only the sanitized manifest. Gold is loaded exclusively by
the separate analyze phase after all 371 original responses have been sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_master_reader as common

DEFAULT = ROOT / 'results/memeye_qwen_semrag_open_20261105'
REFERENCE = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024/reader/protocol.json'
ANSWER_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
sha, digest, read_rows, dump = common.sha, common.digest, common.read_lines, common.dump


def write_rows(path, rows):
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Existing frozen artifact differs: {path}')
    path.write_text(text)


def manifest(root):
    rows = read_rows(root / 'reader_manifest.jsonl')
    if len(rows) != 371 or len(common.by_id(rows)) != 371:
        raise ValueError('Expected exactly 371 unique Open questions')
    for row in rows:
        if set(row) & {'answer', 'ground_truth', 'gold_ids', 'gold_choice', 'options', 'clue'}:
            raise ValueError('Gold or options present in reader manifest')
        if len(row['top_memory_ids']) != 5 or len(set(row['top_memory_ids'])) != 5:
            raise ValueError('Reader needs five distinct memories')
        images = [{'path': path, 'sha256': sha(path)} for path in row['query_image_paths']]
        payload = {'system': row['system'], 'user_text': row['user_text'], 'query_images': images}
        if digest(payload) != row['payload_sha256']:
            raise ValueError('Reader payload or image bytes changed')
    return rows


def prepare(args):
    rows = manifest(args.root)
    reference = json.loads(REFERENCE.read_text())
    settings = reference['reader_settings']
    model = common.model_provenance()
    if model['revision'] != settings['reader_model']['revision'] or model['path'] != settings['reader_model_path']:
        raise ValueError('Reader model differs from original 184/371 baseline')
    for weight in reference['weight_files']:
        if model['files'][Path(weight['path']).name]['sha256'] != weight['sha256']:
            raise ValueError('Model weights differ from baseline')
    for path, expected in reference['model_source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Model tokenizer/processor differs from baseline')
    sources = [args.root / 'protocol.json', args.root / 'reader_manifest.jsonl',
               args.root / 'inputs/questions.jsonl', args.root / 'retrieval.jsonl',
               REFERENCE, Path(__file__), Path(common.__file__)]
    images = {path for row in rows for path in row['query_image_paths']}
    contract = {'dataset': 'MemEye', 'stage': 'semantic_rag_open_reader',
        'condition': 'baseline_question_only_open', 'questions': 371,
        'reference_mcq_correct': 184, 'reference_mcq_questions': 371,
        'source_sha256': {str(p): sha(p) for p in sources},
        'gold_file_sha256': sha(args.root / 'inputs/gold.jsonl'),
        'query_image_sha256': {path: sha(path) for path in sorted(images)},
        'model': model, 'seed': 42, 'do_sample': False, 'dtype': 'bfloat16',
        'attention_implementation': 'sdpa', 'device': args.device,
        'max_new_tokens': args.max_new_tokens, 'batch_size': args.batch_size,
        'query_image_max_pixels': common.QUERY_IMAGE_MAX_PIXELS,
        'input_truncation': False, 'top_k': 5,
        'system_prompts': sorted({r['system'] for r in rows}),
        'answer_policy': 'Generate concise natural-language answers to original Open questions; no MCQ options.',
        'retrieval_policy': 'Same 438 original dialogue/full-caption memories and BGE baseline; true Open queries.',
        'gold_loaded_before_analysis': False, 'historical_response_reuse': False,
        'partial_or_empty_outputs': 'Preserve raw output and report status; never discard from denominator.'}
    out = args.root / 'reader'
    out.mkdir(parents=True, exist_ok=True)
    return common.ensure_protocol(out / 'protocol.json', contract), rows


def validate(root, protocol, rows):
    if digest(protocol['contract']) != protocol['experiment_signature']:
        raise ValueError('Reader protocol signature changed')
    for field in ('source_sha256', 'query_image_sha256'):
        for path, expected in protocol['contract'][field].items():
            if sha(path) != expected:
                raise ValueError('Frozen reader source changed: ' + path)
    out = root / 'reader/responses.jsonl'
    responses = read_rows(out) if out.exists() else []
    done, inputs = common.by_id(responses), common.by_id(rows)
    for qid, row in done.items():
        if qid not in inputs or row['payload_sha256'] != inputs[qid]['payload_sha256']:
            raise ValueError('Response ID or payload changed')
        if row['experiment_signature'] != protocol['experiment_signature'] or row['run_id'] != protocol['run_id']:
            raise ValueError('Response belongs to another run')
        if row['top_memory_ids'] != inputs[qid]['top_memory_ids'] or not isinstance(row['raw_response'], str):
            raise ValueError('Response evidence or text malformed')
    return responses


def run(args):
    protocol, rows = prepare(args)
    responses = validate(args.root, protocol, rows)
    done = {r['id'] for r in responses}
    pending = [r for r in rows if r['id'] not in done]
    if args.limit:
        pending = pending[:args.limit]
    if pending:
        runtime = common.load_runtime(args.device)
        execution_id = str(uuid.uuid4())
        dump(args.root / 'reader' / f'environment_{execution_id}.json', runtime[2])
        completed = len(done)
        for has_image in (False, True):
            selected = [r for r in pending if bool(r['query_image_paths']) == has_image]
            for start in range(0, len(selected), args.batch_size):
                batch = selected[start:start + args.batch_size]
                outputs = common.generate_batch(runtime, batch, args.max_new_tokens)
                with (args.root / 'reader/responses.jsonl').open('a') as handle:
                    for item, answer in zip(batch, outputs):
                        response = {**answer, **{k: item[k] for k in ('id', 'task', 'point', 'top_memory_ids', 'payload_sha256')},
                            'status': 'ok' if answer['raw_response'] else 'empty_response',
                            'run_id': protocol['run_id'], 'experiment_signature': protocol['experiment_signature'],
                            'execution_id': execution_id, 'answer_model': ANSWER_MODEL,
                            'historical_response_reuse': False}
                        handle.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                completed += len(batch)
                print(json.dumps({'completed': completed, 'total': 371,
                    'last_id': batch[-1]['id'], 'token_limit_hits_in_batch': sum(r['hit_token_limit'] for r in outputs)}), flush=True)
    responses = validate(args.root, protocol, rows)
    status = {'status': 'complete' if len(responses) == 371 else 'partial',
        'questions_expected': 371, 'questions_answered': len(responses),
        'token_limit_hits': sum(r['hit_token_limit'] for r in responses),
        'empty_answers': sum(not r['raw_response'] for r in responses)}
    if len(responses) == 371:
        status['responses_sha256'] = sha(args.root / 'reader/responses.jsonl')
        status['protocol_sha256'] = sha(args.root / 'reader/protocol.json')
    dump(args.root / 'reader/completion.json', status)
    print(json.dumps(status), flush=True)


def analyze(args):
    root = args.root
    protocol = json.loads((root / 'reader/protocol.json').read_text())
    rows = manifest(root)
    responses = common.by_id(validate(root, protocol, rows))
    seal = json.loads((root / 'reader/completion.json').read_text())
    if len(responses) != 371 or seal['status'] != 'complete' or seal['responses_sha256'] != sha(root / 'reader/responses.jsonl'):
        raise ValueError('All original answers must be sealed before loading gold')
    if sha(root / 'inputs/gold.jsonl') != protocol['contract']['gold_file_sha256']:
        raise ValueError('Frozen gold changed before analysis')
    gold = common.by_id(read_rows(root / 'inputs/gold.jsonl'))
    if set(responses) != set(gold):
        raise ValueError('Gold IDs differ from the generated cohort')
    scored, judge = [], []
    for item in rows:
        qid = item['id']; answer = responses[qid]; target = gold[qid]
        metrics = common.score_answer(answer['raw_response'], target['answer'])
        hits = len(set(item['top_memory_ids']) & set(target['gold_ids']))
        scored.append({'id': qid, 'task': item['task'], 'point': item['point'],
            'question': item['question'], 'answer': target['answer'], 'raw_response': answer['raw_response'],
            **metrics, 'gold_hits': hits, 'gold_count': len(target['gold_ids']),
            'top_memory_ids': item['top_memory_ids'], 'hit_token_limit': answer['hit_token_limit']})
        judge.append({'id': qid, 'task': item['task'], 'point': item['point'],
            'question': item['question'], 'ground_truth': target['answer'],
            'model_output': answer['raw_response'], 'em': int(metrics['em']), 'source_record_id': qid,
            'source_hit_token_limit': answer['hit_token_limit'], 'source_payload_sha256': item['payload_sha256'],
            'source_responses_sha256': seal['responses_sha256'], 'answer_model': ANSWER_MODEL})
    write_rows(root / 'scored_questions.jsonl', scored)
    write_rows(root / 'judge_inputs.jsonl', judge)
    summary = {'status': 'complete', 'dataset': 'MemEye Open', 'method': 'Semantic RAG (caption baseline)',
        'code_method': 'baseline_question_only_open', 'answer_model': ANSWER_MODEL,
        'questions': 371, 'correct_em': sum(int(r['em']) for r in scored),
        **{key: sum(r[key] for r in scored) / 371 for key in ('em', 'f1', 'bleu_1')},
        'gold_hits_top5': sum(r['gold_hits'] for r in scored), 'gold_references': sum(r['gold_count'] for r in scored),
        'top_k': 5, 'token_limit_hits': sum(r['hit_token_limit'] for r in scored),
        'empty_answers': sum(not r['raw_response'] for r in scored),
        'reference_mcq': {'correct': 184, 'questions': 371, 'accuracy': 184/371,
            'note': 'Separate fixed-option MCQ result; Open questions and response format differ.'},
        'artifact_sha256': {name: sha(root / name) for name in ('reader/protocol.json', 'reader/responses.jsonl',
            'inputs/gold.jsonl', 'judge_inputs.jsonl', 'scored_questions.jsonl', 'reader_manifest.jsonl')}}
    summary['recall_at5_micro'] = summary['gold_hits_top5'] / summary['gold_references']
    dump(root / 'answer_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze'), required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(); args.root = args.root.resolve()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        parser.error('Invalid batch size, output budget or limit')
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'reader.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            prepare(args)
        elif args.phase == 'run':
            run(args)
        else:
            analyze(args)


if __name__ == '__main__':
    main()
