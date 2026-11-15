"""Frozen CLIP Top-K images, with their exact source dialogues added for answering."""
import copy
import json
import os
from pathlib import Path
import re
import time

from scripts.memory import clip_image_memory as original

io = original.io
ROOT = original.ROOT
digest = original.digest


def output_dir(args):
    return args.output_root / f'top{args.top_k}' / 'clip_image_dialogue'


def source_dialogues(natural_root, tasks):
    config = json.loads((natural_root / 'protocol.json').read_text())
    memories = {m['id']: m for m in io.read(natural_root / 'memories.jsonl') if m['task'] in tasks}
    images = {p: i['id'] for i in io.read(natural_root / 'images.jsonl') for p in i['source_paths']}
    dialogues, sources = {}, []
    for task in sorted(tasks):
        matches = [s for s in config['source_files'] if Path(s['path']).name == task + '_Open.json']
        if len(matches) != 1:
            raise ValueError('expected one pinned dialogue file for ' + task)
        source = ROOT / matches[0]['path']
        if io.sha(source) != matches[0]['sha256']:
            raise ValueError('dialogue source changed: ' + str(source))
        sources.append(source)
        for session in json.loads(source.read_text())['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = task + '/' + turn['round']
                if mid not in memories:
                    continue
                refs = turn.get('input_image') or []
                if isinstance(refs, str):
                    refs = [refs]
                if [images[r] for r in refs] != memories[mid]['images'] or mid in dialogues:
                    raise ValueError('source image/round mismatch: ' + mid)
                dialogues[mid] = {'text': f"User: {turn['user']}\nAssistant: {turn['assistant']}",
                                  'image_ids': memories[mid]['images']}
    if set(dialogues) != set(memories):
        raise ValueError('missing source dialogue rounds')
    return dialogues, sources


def attach_dialogues(row, dialogues):
    selected = {}
    for item in row['retrieved_images']:
        for mid in item['source_memory_ids']:
            entry = dialogues[mid]
            if item['image_id'] not in entry['image_ids']:
                raise ValueError('dialogue belongs to a different image: ' + mid)
            selected[mid] = entry['text']
    return selected


def reader_messages(row, system):
    messages = original.reader_messages(row, row['retrieved_images'], system)
    content = messages[1]['content']
    # Insert after each historical image, leaving query-image and question blocks intact.
    for i in reversed(range(len(row['retrieved_images']))):
        item = row['retrieved_images'][i]
        dialogue = '\n\n'.join(row['dialogue_by_memory_id'][mid] for mid in item['source_memory_ids'])
        content.insert(3 + 2 * i, {'type': 'text',
            'text': f'Dialogue associated with historical image {i + 1}:\n' + dialogue})
    return messages


def prepare(args):
    source = args.clip_image_source or args.output_root / f'top{args.top_k}' / 'clip_image'
    target = output_dir(args)
    if (target / 'protocol.json').exists():
        raise FileExistsError('Prepared experiment exists; use --stage run/analyze or a new output root')
    source_protocol, frozen = original.verify(source)
    source_answers = original.checked_responses(source, source_protocol, frozen)
    if {a['id'] for a in source_answers} != {q['id'] for q in frozen}:
        raise ValueError('source image experiment must have complete answers')
    if source_protocol['retrieval_top_k'] != args.top_k:
        raise ValueError('Top-K disagrees with frozen retrieval')
    if args.task or args.limit_questions:
        raise ValueError('This method keeps the complete frozen question set; omit question filters')
    dialogues, files = source_dialogues(args.natural_root, {q['task'] for q in frozen})
    system = source_protocol['reader_system_prompt'].replace(
        'retrieved historical images', 'retrieved historical images and their accompanying dialogue')
    rows = []
    for old in frozen:
        row = copy.deepcopy(old)
        row['dialogue_by_memory_id'] = attach_dialogues(row, dialogues)
        row['reader_messages'] = reader_messages(row, system)
        row['source_payload_sha256'] = old['payload_sha256']
        row['payload_sha256'] = digest({'messages': row['reader_messages'],
                                        'image_hashes': row['reader_image_hashes']})
        if row['payload_sha256'] == old['payload_sha256']:
            raise ValueError('dialogue was not added')
        rows.append(row)
    target.mkdir(parents=True, exist_ok=True)
    io.jsonl(target / 'qa_manifest.jsonl', rows)
    files += [Path(__file__), args.natural_root / 'protocol.json',
              source / 'protocol.json', source / 'qa_manifest.jsonl', source / 'qa_responses.jsonl',
              target / 'qa_manifest.jsonl']
    protocol = copy.deepcopy(source_protocol)
    protocol.pop('experiment_signature')
    protocol.update(mode='clip_image_dialogue', reader_system_prompt=system,
        answer_policy='frozen Top-K native historical images with exact corresponding User/Assistant dialogue; question/options and native query image; no captions',
        frozen_retrieval_source=str(source.resolve()),
        frozen_retrieval_signature=source_protocol['experiment_signature'],
        dialogue_policy='all source rounds attached to each deduplicated image, in original order; exactly K images',
        source_dialogue_files=[str(p.resolve()) for p in files if p.name.endswith('_Open.json')])
    protocol['input_hashes'].update({str(p.resolve()): io.sha(p) for p in files})
    protocol['experiment_signature'] = digest(protocol)
    io.dump(target / 'protocol.json', protocol)
    io.dump(target / 'preparation_audit.json', {
        'questions': len(rows), 'unchanged_retrieval_queries': True,
        'unchanged_retrieved_images_scores_order': True, 'unchanged_question_options_and_query_images': True,
        'dialogue_rounds_per_question': sorted({len(q['dialogue_by_memory_id']) for q in rows}),
        'experiment_signature': protocol['experiment_signature']})
    print(f'clip_image_dialogue: prepared {len(rows)} questions with frozen Top-{args.top_k} images', flush=True)


def verify(target):
    protocol = json.loads((target / 'protocol.json').read_text())
    if digest({k: v for k, v in protocol.items() if k != 'experiment_signature'}) != protocol['experiment_signature']:
        raise ValueError('protocol signature mismatch')
    for filename, expected in protocol['input_hashes'].items():
        if io.sha(Path(filename)) != expected:
            raise ValueError('input changed: ' + filename)
    rows = io.read(target / 'qa_manifest.jsonl')
    source = io.read(Path(protocol['frozen_retrieval_source']) / 'qa_manifest.jsonl')
    if len(rows) != protocol['questions'] or [r['id'] for r in rows] != [r['id'] for r in source]:
        raise ValueError('question set/order changed')
    for row, old in zip(rows, source):
        for key in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths',
                    'retrieval_query', 'retrieved_images', 'top_memory_ids', 'reader_image_paths', 'reader_image_hashes'):
            if row[key] != old[key]:
                raise ValueError('frozen field changed: ' + key)
        if set(row['dialogue_by_memory_id']) != set(row['top_memory_ids']):
            raise ValueError('dialogue/image source coverage mismatch')
        messages = reader_messages(row, protocol['reader_system_prompt'])
        paths = [c['path'] for c in messages[1]['content'] if c['type'] == 'image']
        if messages != row['reader_messages'] or paths != row['reader_image_paths']:
            raise ValueError('reader payload changed')
        hashes = {p: protocol['input_hashes'][str(Path(p).resolve())] for p in paths}
        if hashes != row['reader_image_hashes'] or digest({'messages': messages, 'image_hashes': hashes}) != row['payload_sha256']:
            raise ValueError('reader image hash mismatch')
    return protocol, rows


def run(args):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    target = output_dir(args)
    protocol, rows = verify(target)
    responses = original.checked_responses(target, protocol, rows)
    done = {r['id'] for r in responses}
    pending = [q for q in rows if q['id'] not in done]
    if not pending:
        return
    torch.set_num_threads(8)
    torch.manual_seed(protocol['seed'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('native-image reader requires CUDA')
    processor = AutoProcessor.from_pretrained(protocol['reader_model_path'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536, 'longest_edge': protocol['query_image_max_pixels']}
    print('Loading Qwen-VL for native images with source dialogue', flush=True)
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
            'dialogue_round_count': len(row['dialogue_by_memory_id']),
            'elapsed_seconds': time.perf_counter() - started, 'payload_sha256': row['payload_sha256'],
            'experiment_signature': protocol['experiment_signature']}
        with (target / 'qa_responses.jsonl').open('a') as f:
            f.write(json.dumps(response, ensure_ascii=False) + '\n')
        print(f'clip_image_dialogue QA {count}/{len(rows)} {row["id"]}: {response["predicted_choice"]}', flush=True)
        del encoded, generated, images


def analyze(args):
    target = output_dir(args)
    protocol, rows = verify(target)
    answers = {a['id']: a for a in original.checked_responses(target, protocol, rows)}
    if set(answers) != {q['id'] for q in rows}:
        raise ValueError('incomplete answer set')
    source = Path(protocol['frozen_retrieval_source'])
    before = {a['id']: a for a in io.read(source / 'qa_responses.jsonl')}
    results = []
    for q in rows:
        a = answers[q['id']]
        if a['reader_image_count'] != protocol['retrieval_top_k'] + len(q['query_image_paths']):
            raise ValueError('actual image count mismatch')
        if a['dialogue_round_count'] != len(q['dialogue_by_memory_id']):
            raise ValueError('dialogue round count mismatch')
        results.append({'id': q['id'], 'task': q['task'], 'gold_choice': q['gold_choice'],
            'predicted_choice': a['predicted_choice'], 'correct': a['predicted_choice'] == q['gold_choice'],
            'without_dialogue_correct': before[q['id']]['predicted_choice'] == q['gold_choice']})
    def stats(group):
        return {'questions': len(group), 'correct': sum(r['correct'] for r in group),
                'accuracy': sum(r['correct'] for r in group) / len(group)}
    summary = {**stats(results), 'mode': 'clip_image_dialogue', 'top_k': protocol['retrieval_top_k'],
        'format_errors': sum(a['status'] != 'ok' for a in answers.values()),
        'token_limit_hits': sum(a['hit_token_limit'] for a in answers.values()),
        'unchanged_retrieval': True, 'query_truncated_count': protocol['query_truncated_count'],
        'without_dialogue_correct': sum(r['without_dialogue_correct'] for r in results),
        'wrong_to_correct': sum(r['correct'] and not r['without_dialogue_correct'] for r in results),
        'correct_to_wrong': sum(not r['correct'] and r['without_dialogue_correct'] for r in results),
        'experiment_signature': protocol['experiment_signature'],
        'by_task': {t: stats([r for r in results if r['task'] == t]) for t in sorted({r['task'] for r in results})}}
    io.jsonl(target / 'per_question.jsonl', results)
    io.dump(target / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def pipeline(args):
    for stage in ('prepare', 'run', 'analyze'):
        if args.stage in (stage, 'all'):
            globals()[stage](args)
