"""Question-only CLIP retrieval followed by native-image Qwen-VL MCQ answering."""
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from scripts.evaluation import mcq_reader as io

ROOT = io.ROOT


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def output_dir(args):
    return args.output_root / f'top{args.top_k}' / 'clip_image'


def image_candidates(memories, images, tasks):
    """Index only historical images, once per task, retaining all source rounds."""
    by_image = {r['id']: r for r in images}
    grouped = {}
    for row in memories:
        if row['task'] not in tasks:
            continue
        for image_id in row['images']:
            key = (row['task'], image_id)
            if key not in grouped:
                grouped[key] = {'task': row['task'], 'image_id': image_id,
                                'path': by_image[image_id]['path'], 'source_memory_ids': []}
            grouped[key]['source_memory_ids'].append(row['id'])
    return list(grouped.values())


def rank_images(question, candidates, vectors, query_vector, top_k):
    indices = [i for i, c in enumerate(candidates) if c['task'] == question['task']]
    if len(indices) < top_k:
        raise ValueError('fewer than Top-K distinct historical images in task')
    scores = vectors[indices] @ query_vector
    order = np.argsort(-scores, kind='stable')[:top_k]
    return [{**candidates[indices[j]], 'score': float(scores[j])} for j in order]


def reader_messages(question, retrieved, system):
    content = [{'type': 'text', 'text': 'Retrieved historical images (ranked by relevance):'}]
    for i, item in enumerate(retrieved, 1):
        content += [{'type': 'text', 'text': f'Historical image {i}:'},
                    {'type': 'image', 'path': item['path']}]
    for i, path in enumerate(question['query_image_paths'], 1):
        content += [{'type': 'text', 'text': f'Question image {i} (part of the question):'},
                    {'type': 'image', 'path': path}]
    text = 'Question: ' + question['question'] + '\nOptions:\n'
    text += '\n'.join(f'{letter}. {question["options"][letter]}' for letter in 'ABCD')
    text += '\nReturn only one uppercase letter: A, B, C, or D.'
    content.append({'type': 'text', 'text': text})
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def prepare(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from scripts.graph.visual_image_encoder import ImageEncoder

    target = output_dir(args)
    if (target / 'protocol.json').exists():
        raise FileExistsError('Prepared experiment exists; use --stage run/analyze or a new --output-root')
    questions = io.read(args.benchmark_manifest)
    if args.task:
        questions = [q for q in questions if q['task'] == args.task]
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('empty or duplicate question IDs')
    candidates = image_candidates(io.read(args.natural_root / 'memories.jsonl'),
                                  io.read(args.natural_root / 'images.jsonl'),
                                  {q['task'] for q in questions})
    checkpoint = ROOT / 'models/clip-vit-base-patch32'
    torch.set_num_threads(8)
    encoder = ImageEncoder(str(checkpoint), device=args.device, local_files_only=True)
    cache = {}
    for item in candidates:
        if item['image_id'] not in cache:
            if io.sha(Path(item['path'])) != item['image_id']:
                raise ValueError('historical image content changed: ' + item['path'])
            cache[item['image_id']] = encoder.encode(item['path'])
    vectors = np.array([cache[item['image_id']] for item in candidates], dtype=np.float64)
    metadata = encoder.metadata
    del encoder
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        checkpoint, config=config.text_config, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('CLIP text/projection weights are incomplete')
    model.to(device=args.device, dtype=torch.float32).eval()
    limit = config.text_config.max_position_embeddings
    texts = [q['question'] for q in questions]
    lengths = [len(tokenizer.encode(text, truncation=False)) for text in texts]
    query_vectors = []
    for start in range(0, len(texts), 32):
        inputs = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                           max_length=limit, return_tensors='pt').to(args.device)
        with torch.inference_mode():
            embedding = model(**inputs, return_dict=True).text_embeds.double().cpu().numpy()
        query_vectors.extend(embedding / np.linalg.norm(embedding, axis=1, keepdims=True))
    query_vectors = np.asarray(query_vectors)
    if not np.isfinite(query_vectors).all():
        raise ValueError('non-finite CLIP query embeddings')
    baseline = json.loads((args.benchmark_manifest.parent / 'protocol.json').read_text())
    system = baseline['reader_system_prompt'].replace('memory captions', 'historical images')
    rows = []
    for q, vector, length in zip(questions, query_vectors, lengths):
        retrieved = rank_images(q, candidates, vectors, vector, args.top_k)
        row = {k: q[k] for k in ('id', 'task', 'x', 'y', 'question', 'options',
                                'gold_choice', 'gold_ids', 'query_image_paths')}
        messages = reader_messages(row, retrieved, system)
        image_paths = [c['path'] for c in messages[1]['content'] if c['type'] == 'image']
        image_hashes = {p: io.sha(Path(p)) for p in image_paths}
        row.update(retrieval_query=q['question'], query_token_count=length,
                   query_truncated=length > limit, retrieved_images=retrieved,
                   top_memory_ids=list(dict.fromkeys(mid for r in retrieved for mid in r['source_memory_ids'])),
                   reader_messages=messages, reader_image_paths=image_paths,
                   reader_image_hashes=image_hashes,
                   payload_sha256=digest({'messages': messages, 'image_hashes': image_hashes}))
        rows.append(row)
    target.mkdir(parents=True, exist_ok=True)
    io.jsonl(target / 'image_candidates.jsonl', candidates)
    np.savez(target / 'clip_embeddings.npz', image_vectors=vectors, query_vectors=query_vectors,
             image_ids=np.array([c['image_id'] for c in candidates]),
             query_ids=np.array([q['id'] for q in questions]))
    io.jsonl(target / 'qa_manifest.jsonl', rows)
    files = [Path(__file__), ROOT / 'scripts/graph/visual_image_encoder.py',
             ROOT / 'scripts/evaluation/mcq_reader.py', args.benchmark_manifest,
             args.benchmark_manifest.parent / 'protocol.json',
             args.natural_root / 'memories.jsonl', args.natural_root / 'images.jsonl',
             target / 'image_candidates.jsonl', target / 'clip_embeddings.npz', target / 'qa_manifest.jsonl']
    files += [p for p in checkpoint.iterdir() if p.is_file()]
    files += [Path(c['path']) for c in candidates]
    files += [Path(p) for q in questions for p in q['query_image_paths']]
    protocol = {k: baseline[k] for k in ('reader_model', 'reader_model_path', 'seed', 'dtype',
                                       'max_new_tokens', 'query_image_max_pixels')}
    protocol.update(mode='clip_image', retrieval_top_k=args.top_k, questions=len(rows),
        candidate_images=len(candidates), source_memory_rounds=len({m for c in candidates for m in c['source_memory_ids']}),
        retrieval_scope='distinct historical images within original task; no query-only images',
        query_policy='original question text only; no options, answers, captions or query images',
        answer_policy='Top-K native historical images plus question/options and native query images; no caption/dialogue',
        reader_system_prompt=system, do_sample=False, batch_size=1,
        clip_image_encoder=metadata, clip_text_token_limit=limit,
        query_truncated_count=sum(n > limit for n in lengths), query_max_tokens=max(lengths),
        clip_truncation_policy='right truncation at CLIP text context limit',
        duplicate_image_policy='deduplicate by content hash within task; retain all source round IDs for recall',
        input_hashes={str(p.resolve()): io.sha(p) for p in files})
    protocol['experiment_signature'] = digest(protocol)
    io.dump(target / 'protocol.json', protocol)
    print(json.dumps({'stage': 'prepared', 'questions': len(rows), 'candidate_images': len(candidates),
                      'query_truncated_count': protocol['query_truncated_count']}), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify(target):
    protocol = json.loads((target / 'protocol.json').read_text())
    if digest({k: v for k, v in protocol.items() if k != 'experiment_signature'}) != protocol['experiment_signature']:
        raise ValueError('protocol signature mismatch')
    for name, expected in protocol['input_hashes'].items():
        if io.sha(Path(name)) != expected:
            raise ValueError('input changed: ' + name)
    rows = io.read(target / 'qa_manifest.jsonl')
    if len(rows) != protocol['questions'] or len({q['id'] for q in rows}) != len(rows):
        raise ValueError('manifest IDs/count mismatch')
    for row in rows:
        if row['reader_messages'] != reader_messages(row, row['retrieved_images'], protocol['reader_system_prompt']):
            raise ValueError('reader messages mismatch')
        if len(row['retrieved_images']) != protocol['retrieval_top_k']:
            raise ValueError('retrieved image count mismatch')
        paths = [c['path'] for c in row['reader_messages'][1]['content'] if c['type'] == 'image']
        hashes = {p: io.sha(Path(p)) for p in paths}
        if (paths != row['reader_image_paths'] or hashes != row['reader_image_hashes'] or
                digest({'messages': row['reader_messages'], 'image_hashes': hashes}) != row['payload_sha256']):
            raise ValueError('image payload mismatch: ' + row['id'])
    return protocol, rows


def checked_responses(target, protocol, rows):
    path = target / 'qa_responses.jsonl'
    responses = io.read(path) if path.exists() else []
    by_id = {q['id']: q for q in rows}
    if len({r['id'] for r in responses}) != len(responses):
        raise ValueError('duplicate response IDs')
    for r in responses:
        if (r['id'] not in by_id or r['experiment_signature'] != protocol['experiment_signature'] or
                r['payload_sha256'] != by_id[r['id']]['payload_sha256']):
            raise ValueError('response provenance mismatch')
    return responses


def run(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    target = output_dir(args)
    protocol, rows = verify(target)
    responses = checked_responses(target, protocol, rows)
    done = {r['id'] for r in responses}
    pending = [q for q in rows if q['id'] not in done]
    if not pending:
        return
    torch.set_num_threads(8)
    torch.manual_seed(protocol['seed'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('native-image reader requires a CUDA device')
    processor = AutoProcessor.from_pretrained(protocol['reader_model_path'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': protocol['query_image_max_pixels']}
    print('Loading Qwen-VL native-image reader', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(protocol['reader_model_path'],
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, device_map={'': str(device)})
    model.eval()
    io.dump(target / 'generation_environment.json', {'torch': torch.__version__,
        'transformers': transformers.__version__, 'pid': os.getpid(), 'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(device), 'processor_size': processor.image_processor.size,
        'parameter_devices': sorted({str(p.device) for p in model.parameters()}),
        'experiment_signature': protocol['experiment_signature']})
    for count, row in enumerate(pending, len(done) + 1):
        started = time.perf_counter()
        rendered = processor.apply_chat_template(row['reader_messages'], tokenize=False, add_generation_prompt=True)
        images = []
        for path in row['reader_image_paths']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        encoded = processor(text=[rendered], images=images, padding=True, return_tensors='pt').to(device)
        input_tokens = int(encoded['attention_mask'].sum())
        if input_tokens + protocol['max_new_tokens'] > model.config.text_config.max_position_embeddings:
            raise ValueError('reader context would truncate')
        if encoded['image_grid_thw'].shape[0] != len(images):
            raise ValueError('processor image count mismatch')
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=protocol['max_new_tokens'],
                                       pad_token_id=processor.tokenizer.pad_token_id)
        tokens = generated[0, encoded['input_ids'].shape[1]:].tolist()
        raw = processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw, re.IGNORECASE)
        eos = processor.tokenizer.eos_token_id
        response = {'id': row['id'], 'raw_response': raw,
            'predicted_choice': match.group(1).upper() if match else None,
            'status': 'ok' if match else 'format_error', 'input_tokens': input_tokens,
            'generated_tokens': next((i + 1 for i, t in enumerate(tokens) if t == eos), len(tokens)),
            'hit_token_limit': len(tokens) >= protocol['max_new_tokens'] and eos not in tokens,
            'reader_image_count': len(images), 'image_grid_thw': encoded['image_grid_thw'].tolist(),
            'elapsed_seconds': time.perf_counter() - started, 'payload_sha256': row['payload_sha256'],
            'experiment_signature': protocol['experiment_signature']}
        with (target / 'qa_responses.jsonl').open('a') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')
        print(f'clip_image QA {count}/{len(rows)} {row["id"]}: {response["predicted_choice"]}', flush=True)
        del encoded, generated, images


def analyze(args):
    target = output_dir(args)
    protocol, rows = verify(target)
    responses = checked_responses(target, protocol, rows)
    answers = {r['id']: r for r in responses}
    if set(answers) != {q['id'] for q in rows}:
        raise ValueError('incomplete answer set')
    results = []
    for q in rows:
        gold, selected = set(q['gold_ids']), set(q['top_memory_ids'])
        a = answers[q['id']]
        if a['reader_image_count'] != args.top_k + len(q['query_image_paths']):
            raise ValueError('actual reader image count mismatch')
        results.append({'id': q['id'], 'task': q['task'], 'gold_choice': q['gold_choice'],
            'predicted_choice': a['predicted_choice'], 'correct': a['predicted_choice'] == q['gold_choice'],
            'gold_count': len(gold), 'gold_hits': len(gold & selected),
            'any_gold_topk': bool(gold & selected), 'all_gold_topk': gold <= selected})
    def stats(group):
        return {'questions': len(group), 'correct': sum(r['correct'] for r in group),
                'accuracy': sum(r['correct'] for r in group) / len(group),
                'any_gold_topk': sum(r['any_gold_topk'] for r in group),
                'all_gold_topk': sum(r['all_gold_topk'] for r in group),
                'gold_hits': sum(r['gold_hits'] for r in group),
                'gold_total': sum(r['gold_count'] for r in group),
                'micro_recall': sum(r['gold_hits'] for r in group) / sum(r['gold_count'] for r in group)}
    summary = {**stats(results), 'mode': 'clip_image', 'top_k': args.top_k,
        'format_errors': sum(r['status'] != 'ok' for r in responses),
        'token_limit_hits': sum(r['hit_token_limit'] for r in responses),
        'query_truncated_count': protocol['query_truncated_count'],
        'experiment_signature': protocol['experiment_signature'],
        'by_task': {t: stats([r for r in results if r['task'] == t]) for t in sorted({r['task'] for r in results})}}
    io.jsonl(target / 'per_question.jsonl', results)
    io.dump(target / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def pipeline(args):
    for stage in ('prepare', 'run', 'analyze'):
        if args.stage in (stage, 'all'):
            globals()[stage](args)
