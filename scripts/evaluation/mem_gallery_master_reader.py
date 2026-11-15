"""Gold-free candidate generation and paired open-answer Mem-Gallery evaluation.

Candidates see only sanitized questions and supplied question-image captions.
Readers see only original questions, question pixels and original Top-5 memories.
Only analyze() opens gold.jsonl. Historical responses are never imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

MODEL = ROOT / 'models/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
CLIP = ROOT / 'models/clip-vit-base-patch32'
DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_master_answer_20261021'
CONDITIONS = ('baseline_question_only', 'master_answer_query')
SEED = 42
QUERY_IMAGE_MAX_PIXELS = 1638400
OFFICIAL_REVISION = 'a93959e1e978a6a7d77798ae92c2ffe41c538c62'

CANDIDATE_SYSTEM = '''You propose hypothetical visual evidence for memory retrieval.
You do not have the conversation history or the correct answer. Never claim that
a hypothesis is known to be true. Return ONLY a JSON array of exactly four
distinct short English declarative statements, without markdown or explanation.
Each statement must be at most 24 words and describe a different plausible
answer or visual scene that could help answer the question. Preserve the
question's subject and requested distinctions. Avoid lists of unrelated guesses.
For image-search questions, describe possible scenes or objects, never invent an
image ID, filename, conversation identifier or session identifier. If a supplied
question-image caption exists, use it as query context, not as historical evidence.
Do not refer to choices A/B/C/D. The four statements are retrieval hypotheses,
not final answers.'''

# Official sys_prompt.txt and task-specific instructions, frozen at OFFICIAL_REVISION.
READER_SYSTEM = '''You are an AI assistant evaluated on multimodal long-term conversational memory.
For the given question-answering task, your responses must be concise, yet complete enough to accurately answer the questions.
If multiple pieces of information about the same event appear in the conversation, always rely on the most recent information.

The question-answering evaluation will contain several multimodal task types:

Factual Retrieval: Retrieve explicit facts mentioned in the conversation for the answer.

Multi-entity Reasoning: Combine the retrieved information to reason and infer an answer.

Temporal Reasoning: Resolve time-dependent questions.

Visual-centric Reasoning: Besides textual information, answer questions using visual images in the conversation.

Test-time Learning: Learn new visual knowledge from provided images within historical dialogue and use it in question-answering.

Visual-centric Search: Find the image(s) that match the information in a given query and return their image ID(s).

Conflict Detection: Detect contradictions between the conversation history and the information provided in the question.

Knowledge Resolution: Resolve knowledge conflicts or updates by prioritizing the most recent information.

Answer Refusal: Decline to answer when the information does not exist in the conversation history.

Follow all instructions strictly. Only answer using information contained within the multimodal conversation. Do not hallucinate. Always remain consistent and grounded in the dialogue history.'''
TASK_INSTRUCTIONS = {
    'VS': 'Return the image_id of the image(s). If there are multiple images, sort them in ascending order and separate them by commas. Format example: “D2:IMG_003, D2:IMG_010, D10:IMG_002” (for format reference only).',
    'CD': 'Please check whether this information conflicts with the conversation, and reply strictly with either “Yes.” or “No.”',
    'AR': 'Provide your answer based on the information in the conversation. Only if the information about the question is not present in the conversation, reply with: “Not mentioned.”',
}


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, rows):
    with Path(path).open('a') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def by_id(rows, key='id'):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def model_provenance():
    paths = sorted(path for path in MODEL.iterdir()
                   if path.is_file() and path.suffix in ('.json', '.txt', '.safetensors'))
    if not any(path.suffix == '.safetensors' for path in paths):
        raise FileNotFoundError(f'No local model weights in {MODEL}')
    return {'repo': 'Qwen/Qwen3-VL-8B-Instruct', 'revision': MODEL_REVISION,
            'path': str(MODEL), 'files': {path.name: {'bytes': path.stat().st_size,
            'sha256': sha(path)} for path in paths}}


def ensure_protocol(path, contract):
    """Resume only the identical experiment, including source and model hashes."""
    signature = digest(contract)
    if Path(path).exists():
        protocol = json.loads(Path(path).read_text())
        if protocol['experiment_signature'] != signature or protocol['contract'] != contract:
            raise ValueError(f'Existing experiment differs; use a fresh output directory: {path}')
    else:
        protocol = {'run_id': str(uuid.uuid4()), 'experiment_signature': signature,
                    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'contract': contract}
        dump(path, protocol)
    return protocol


def load_runtime(device):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    requested = torch.device(device)
    if requested.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This evaluation requires CUDA; no silent CPU fallback is permitted.')
    if requested.index is None:
        requested = torch.device('cuda:0')
    if requested.index >= torch.cuda.device_count():
        raise ValueError(f'Unavailable GPU: {requested}')
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = {'shortest_edge': 65536,
                                      'longest_edge': QUERY_IMAGE_MAX_PIXELS}
    started = time.perf_counter()
    free_before = torch.cuda.mem_get_info(requested)[0]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa', device_map={'': str(requested)})
    model.eval()
    if next(model.parameters()).device != requested:
        raise RuntimeError('Reader did not load on the requested GPU')
    environment = {'torch': torch.__version__, 'transformers': transformers.__version__,
        'device': str(requested), 'gpu': torch.cuda.get_device_name(requested),
        'free_gpu_bytes_before_loading': free_before, 'dtype': str(model.dtype),
        'device_map': getattr(model, 'hf_device_map', None), 'pid': os.getpid(),
        'load_seconds': time.perf_counter() - started,
        'context_limit': model.config.text_config.max_position_embeddings,
        'generation_config': model.generation_config.to_dict(),
        'query_image_max_pixels': QUERY_IMAGE_MAX_PIXELS}
    return processor, model, environment


def generate_batch(runtime, rows, max_new_tokens):
    """Rows contain system, user_text, and query_image_paths only."""
    import torch
    from PIL import Image
    processor, model, _ = runtime
    rendered, images = [], []
    for row in rows:
        paths = row.get('query_image_paths', [])
        content = [{'type': 'image'} for _ in paths]
        content.append({'type': 'text', 'text': row['user_text']})
        rendered.append(processor.apply_chat_template([
            {'role': 'system', 'content': row['system']},
            {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True))
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert('RGB'))
    try:
        inputs = processor(text=rendered, images=images or None, padding=True,
                           truncation=False, return_tensors='pt').to(next(model.parameters()).device)
    finally:
        for image in images:
            image.close()
    lengths = inputs['attention_mask'].sum(dim=1).tolist()
    context_limit = model.config.text_config.max_position_embeddings
    if inputs['input_ids'].shape[1] + max_new_tokens > context_limit:
        raise ValueError(f'Refusing input truncation: {max(lengths)} + {max_new_tokens} > {context_limit}')
    torch.manual_seed(SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                   pad_token_id=processor.tokenizer.pad_token_id)
    elapsed = time.perf_counter() - started
    tails = generated[:, inputs['input_ids'].shape[1]:]
    decoded = processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(processor.tokenizer.eos_token_id)
    output = []
    for text, tokens, input_length in zip(decoded, tails.tolist(), lengths):
        stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        emitted = stop + 1 if stop is not None else len(tokens)
        output.append({'raw_response': text.strip(), 'input_tokens': int(input_length),
            'generated_tokens': emitted, 'hit_token_limit': stop is None and emitted >= max_new_tokens,
            'batch_elapsed_seconds': elapsed, 'batch_size': len(rows)})
    return output


def parse_candidates(raw, clip_tokenizer, *, return_format=False):
    """Accept JSON or exactly four unambiguous lines; never invent or truncate.

    Plain lines may all be unmarked, bullet-prefixed, or numbered 1 through 4.
    Malformed JSON/bracket lists, fences and explanatory framing are rejected.
    """
    text = raw.strip()
    response_format = 'json_array'
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if any(character in text for character in ('[', ']', '{', '}', '`')):
            raise ValueError('Malformed JSON/bracket list or fenced response')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 4:
            raise ValueError('Expected JSON or exactly four nonempty statement lines')
        values, markers = [], []
        for index, line in enumerate(lines, 1):
            numbered = re.match(r'^(\d+)[.)]\s+(.+)$', line)
            bullet = re.match(r'^[-*•]\s+(.+)$', line)
            if numbered:
                if int(numbered.group(1)) != index:
                    raise ValueError('Numbered candidates must run from 1 through 4')
                line, marker = numbered.group(2).strip(), 'numbered'
            elif bullet:
                line, marker = bullet.group(1).strip(), 'bulleted'
            else:
                marker = 'unmarked'
            if (not line or line[0] in ('"', "'") or
                    re.match(r'^(?:here (?:are|is)|(?:candidates?|hypotheses|statements|note|answer|explanation)\s*:)', line, re.I)):
                raise ValueError('Ambiguous statement framing')
            values.append(line)
            markers.append(marker)
        if len(set(markers)) != 1:
            raise ValueError('Mixed candidate line formatting')
        value, response_format = values, 'four_lines_' + markers[0]
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('Expected a JSON array of four statements')
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError('Every candidate must be a nonempty string')
    value = [item.strip() for item in value]
    if len({' '.join(item.casefold().split()) for item in value}) != 4:
        raise ValueError('The four candidates must be distinct')
    if any(re.search(r'\b(?:D\d+[:_]IMG[_:]?\d+|QA_IMG_\d+)\b|\.(?:jpg|jpeg|png)\b', item, re.I)
           for item in value):
        raise ValueError('Candidates must describe scenes, not invent image IDs or filenames')
    lengths = [len(clip_tokenizer(item, truncation=False)['input_ids']) for item in value]
    if max(lengths) > 77:
        raise ValueError(f'Candidate exceeds CLIP limit: {lengths}')
    return (value, lengths, response_format) if return_format else (value, lengths)


def candidate_user_text(row, attempt=1):
    text = 'Question:\n' + row['question']
    if row.get('query_image_caption'):
        text += '\n\nSupplied question-image caption:\n' + row['query_image_caption']
    text += ('\n\nRequired output format: a JSON array containing exactly four strings. '
             'Use this syntax, replacing every placeholder with your own short statement:\n'
             '["First distinct plausible statement.", "Second distinct plausible statement.", '
             '"Third distinct plausible statement.", "Fourth distinct plausible statement."]\n'
             'Output only the JSON array. No introduction, markdown, image IDs or filenames.')
    if attempt > 1:
        text += ('\nKeep all four candidates different. Include the opening and closing square '
                 'brackets, quote each string with double quotes, and separate strings with commas.')
    if attempt > 2:
        text += ('\nUse no more than twelve words per statement. Count exactly four '
                 'distinct alternatives before producing the JSON array. Do not add prose.')
    return text


def generate_candidates(root, device='cuda:0', batch_size=32, max_attempts=3):
    """Generate frozen hypotheses from the single sanitized input file."""
    from transformers import AutoTokenizer
    root = Path(root).resolve()
    source = root / 'inputs/sanitized_candidate_inputs.jsonl'
    rows = read_lines(source)
    by_id(rows)
    if not rows or any(set(row) != {'id', 'question', 'query_image_caption'} for row in rows):
        raise ValueError('Candidate input must contain only id, question and query_image_caption')
    if batch_size < 1 or max_attempts < 1:
        raise ValueError('Batch size and max attempts must be positive')
    tokenizer = AutoTokenizer.from_pretrained(str(CLIP), local_files_only=True)
    contract = {'stage': 'gold_free_open_ended_candidates', 'questions': len(rows),
        'source_sha256': {str(source): sha(source), str(Path(__file__)): sha(__file__)},
        'model': model_provenance(), 'system_prompt': CANDIDATE_SYSTEM,
        'seed': SEED, 'do_sample': False, 'max_new_tokens': 384, 'batch_size': batch_size,
        'max_attempts': max_attempts, 'device': str(device), 'dtype': 'bfloat16',
        'candidate_count': 4, 'clip_token_limit': 77, 'truncation': False,
        'accepted_response_formats': ['json_array', 'four_lines_unmarked',
                                     'four_lines_bulleted', 'four_lines_numbered'],
        'line_parser_policy': 'Exactly four nonempty lines; remove only uniform bullets or sequential 1-4 prefixes. Reject malformed JSON, brackets, fences, and explanatory framing.',
        'clip_tokenizer_sha256': {p.name: sha(p) for p in sorted(CLIP.iterdir())
            if p.name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')},
        'input_fields': ['id', 'question', 'query_image_caption'],
        'gold_or_clues_or_history_used': False,
        'limitation': 'Open-ended hypothetical candidates, not the provided MCQ options used in MemEye.'}
    protocol = ensure_protocol(root / 'candidate_generation_protocol.json', contract)
    output = root / 'candidates.jsonl'
    completed = read_lines(output) if output.exists() else []
    completed_by_id = by_id(completed)
    source_ids = {row['id'] for row in rows}
    for row in completed:
        if row['id'] not in source_ids or row['experiment_signature'] != protocol['experiment_signature']:
            raise ValueError('Candidate resume signature or ID mismatch')
        values, lengths = parse_candidates(json.dumps(row['candidate_statements']), tokenizer)
        if lengths != row['clip_token_lengths']:
            raise ValueError('Candidate tokenizer lengths changed')
    pending = [row for row in rows if row['id'] not in completed_by_id]
    if pending:
        runtime = load_runtime(device)
        dump(root / f'candidate_environment_{uuid.uuid4()}.json', runtime[2])
        failed = pending
        for attempt in range(1, max_attempts + 1):
            retry = []
            for start in range(0, len(failed), batch_size):
                batch = failed[start:start + batch_size]
                responses = generate_batch(runtime, [{'system': CANDIDATE_SYSTEM,
                    'user_text': candidate_user_text(row, attempt)} for row in batch], 384)
                for row, response in zip(batch, responses):
                    error = None
                    response_format = None
                    try:
                        if response['hit_token_limit']:
                            raise ValueError('Candidate generation hit its token limit')
                        values, lengths, response_format = parse_candidates(response['raw_response'], tokenizer, return_format=True)
                    except (ValueError, TypeError, KeyError) as exc:
                        error = str(exc)
                        retry.append(row)
                    attempt_row = {'id': row['id'], 'attempt': attempt,
                        'run_id': protocol['run_id'], 'experiment_signature': protocol['experiment_signature'],
                        'prompt_sha256': digest({'system': CANDIDATE_SYSTEM,
                            'user_text': candidate_user_text(row, attempt)}),
                        **response, 'validation_error': error, 'response_format': response_format}
                    append(root / 'candidate_attempts.jsonl', [attempt_row])
                    if error is None:
                        result = {'id': row['id'], 'candidate_statements': values,
                            'clip_token_lengths': lengths, 'source_input_sha256': digest(row),
                            'attempt': attempt, 'prompt_sha256': attempt_row['prompt_sha256'],
                            'response_format': response_format,
                            'run_id': protocol['run_id'], 'experiment_signature': protocol['experiment_signature'],
                            'gold_or_history_used': False}
                        append(output, [result])
                        completed_by_id[row['id']] = result
                progress = {'stage': 'candidates', 'completed': len(completed_by_id),
                            'total': len(rows), 'attempt': attempt, 'failed_this_pass': len(retry)}
                dump(root / 'candidate_progress.json', progress)
                print(json.dumps(progress), flush=True)
            failed = retry
            if not failed:
                break
        del runtime
        if failed:
            dump(root / 'candidate_failures.json', {'ids': [row['id'] for row in failed]})
            raise RuntimeError(f'{len(failed)} candidate records still invalid; raw attempts retained')
    if set(completed_by_id) != source_ids:
        raise ValueError('Incomplete candidate coverage')
    if sha(source) != contract['source_sha256'][str(source)]:
        raise ValueError('Sanitized candidate source changed during generation')
    # Canonical source order makes downstream indexing independent of retry order.
    temp = output.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(completed_by_id[row['id']], ensure_ascii=False) + '\n' for row in rows))
    temp.replace(output)
    report = {'passed': True, 'questions': len(rows), 'candidates': 4 * len(rows),
        'max_clip_tokens': max(max(row['clip_token_lengths']) for row in completed_by_id.values()),
        'candidates_sha256': sha(output), 'gold_or_history_used': False}
    dump(root / 'candidate_validation.json', report)
    return report


def reader_text(question, memory_ids, memories):
    context = '\n\n'.join(f'Memory {i}: {memories[mid]["reader_text"]}'
                           for i, mid in enumerate(memory_ids, 1))
    suffix = ('Please answer the question about the conversation using the retrieved memories. '
              'Return only the answer content, concisely but completely. For dates or times, '
              'provide a specific date or time whenever possible.\n\nQuestion:\n' + question['question'])
    constraint = TASK_INSTRUCTIONS.get(question['point'].upper())
    if constraint:
        suffix += '\n\n' + constraint
    return 'The retrieved memory contents are as follows:\n\n' + context + '\n\n' + suffix


def build_reader_manifest(questions, memories, retrieval, image_hash=sha):
    """Build signed paired payloads without access to candidates or gold answers."""
    qmap, mmap, rmap = by_id(questions), by_id(memories), by_id(retrieval)
    if set(qmap) != set(rmap) or not qmap:
        raise ValueError('Retrieval question coverage differs from input questions')
    manifest, image_cache = [], {}
    for index, question in enumerate(questions):
        if {'answer', 'gold_choice', 'gold_ids', 'clue'} & set(question):
            raise ValueError('Reader questions must be gold-free')
        rid = rmap[question['id']]
        if rid['task'] != question['task']:
            raise ValueError('Retrieval domain mismatch')
        conditions = by_id(rid['conditions'], 'name')
        if set(conditions) != set(CONDITIONS):
            raise ValueError('Expected the two frozen retrieval conditions')
        image_records = []
        for path in question['query_image_paths']:
            if path not in image_cache:
                image_cache[path] = {'path': path, 'sha256': image_hash(path)}
            image_records.append(image_cache[path])
        pair = []
        for condition in CONDITIONS:
            row = conditions[condition]
            ids = row['top5_memory_ids']
            if len(ids) != 5 or len(set(ids)) != 5 or len(row['top5_scores']) != 5:
                raise ValueError('Each condition requires five distinct memories and scores')
            if any(mid not in mmap or mmap[mid]['task'] != question['task'] for mid in ids):
                raise ValueError('Retrieved memory is missing or from another domain')
            text = reader_text(question, ids, mmap)
            payload = {'system': READER_SYSTEM, 'user_text': text, 'query_images': image_records}
            pair.append({'id': question['id'], 'task': question['task'], 'point': question['point'],
                'record_id': question['id'] + '::' + condition, 'condition': condition,
                'top5_memory_ids': ids, 'top5_scores': row['top5_scores'],
                'reader_text': text, 'query_image_paths': question['query_image_paths'],
                'query_images': image_records, 'payload_sha256': digest(payload)})
        manifest.extend(pair if index % 2 == 0 else pair[::-1])
    return manifest


def run_reader(root, device='cuda:0', batch_size=16, image_batch_size=4):
    root = Path(root).resolve()
    output = root / 'reader'
    output.mkdir(parents=True, exist_ok=True)
    sources = [root / 'inputs/questions.jsonl', root / 'inputs/memories.jsonl', root / 'retrieval_results.jsonl']
    questions, memories, retrieval = [read_lines(path) for path in sources]
    manifest = build_reader_manifest(questions, memories, retrieval)
    contract = {'stage': 'paired_open_ended_reader', 'questions': len(questions),
        'logical_conditions': len(manifest), 'unique_payloads': len({row['payload_sha256'] for row in manifest}),
        'source_sha256': {str(path): sha(path) for path in [*sources, Path(__file__)]},
        'manifest_sha256': digest(manifest), 'model': model_provenance(),
        'system_prompt': READER_SYSTEM, 'task_instructions': TASK_INSTRUCTIONS,
        'official_prompt_revision': OFFICIAL_REVISION, 'seed': SEED, 'do_sample': False,
        'max_new_tokens': 256, 'batch_size_text': batch_size, 'batch_size_images': image_batch_size,
        'device': str(device), 'dtype': 'bfloat16', 'query_image_max_pixels': QUERY_IMAGE_MAX_PIXELS,
        'input_truncation': False, 'generation_limit_reported': True,
        'candidate_expansion_in_reader': False, 'gold_loaded_before_analysis': False,
        'historical_response_reuse': False,
        'deduplication': 'Identical ordered reader payloads share fresh inference within this run only.',
        'inference_order': 'Alternating A/B then B/A per source question, grouped by image presence.'}
    if batch_size < 1 or image_batch_size < 1:
        raise ValueError('Batch sizes must be positive')
    protocol = ensure_protocol(output / 'protocol.json', contract)
    manifest_path = output / 'qa_manifest.jsonl'
    if manifest_path.exists():
        if read_lines(manifest_path) != manifest:
            raise ValueError('Reader manifest changed')
    else:
        append(manifest_path, manifest)
    existing = read_lines(output / 'responses.jsonl') if (output / 'responses.jsonl').exists() else []
    done, cache, mindex = by_id(existing, 'record_id'), {}, by_id(manifest, 'record_id')
    for row in existing:
        if (row['record_id'] not in mindex or row['run_id'] != protocol['run_id'] or
                row['experiment_signature'] != protocol['experiment_signature'] or
                row['payload_sha256'] != mindex[row['record_id']]['payload_sha256']):
            raise ValueError('Reader resume provenance mismatch')
        if row['fresh_inference_executed_for_this_record']:
            if row['payload_sha256'] in cache:
                raise ValueError('Duplicate fresh inference for one payload')
            cache[row['payload_sha256']] = row
    for row in existing:
        if not row['fresh_inference_executed_for_this_record']:
            fresh = cache.get(row['payload_sha256'])
            if fresh is None or row['same_run_shared_inference_record_id'] != fresh['record_id']:
                raise ValueError('Shared response has no matching fresh source')
            if row['raw_response'] != fresh['raw_response']:
                raise ValueError('Shared response differs from fresh response')

    def flush_available():
        emitted = []
        for item in manifest:
            if item['record_id'] in done or item['payload_sha256'] not in cache:
                continue
            original = cache[item['payload_sha256']]
            fresh = item['record_id'] == original['record_id']
            row = {key: original[key] for key in ('raw_response', 'input_tokens', 'generated_tokens',
                'hit_token_limit', 'batch_elapsed_seconds', 'batch_size', 'execution_id')}
            row.update({key: item[key] for key in ('id', 'task', 'point', 'record_id', 'condition',
                'payload_sha256', 'top5_memory_ids')})
            row.update(run_id=protocol['run_id'], experiment_signature=protocol['experiment_signature'],
                status='ok' if row['raw_response'] else 'empty_response',
                fresh_inference_executed_for_this_record=fresh,
                same_run_shared_inference_record_id=None if fresh else original['record_id'])
            done[item['record_id']] = row
            emitted.append(row)
        if emitted:
            append(output / 'responses.jsonl', emitted)

    flush_available()
    pending = {}
    for row in manifest:
        if row['record_id'] not in done:
            pending.setdefault(row['payload_sha256'], row)
    if pending:
        runtime = load_runtime(device)
        execution_id = str(uuid.uuid4())
        dump(output / f'generation_environment_{execution_id}.json', runtime[2])
        for has_image in (False, True):
            selected = [row for row in pending.values() if bool(row['query_image_paths']) == has_image]
            size = image_batch_size if has_image else batch_size
            for start in range(0, len(selected), size):
                batch = selected[start:start + size]
                responses = generate_batch(runtime, [{'system': READER_SYSTEM,
                    'user_text': row['reader_text'], 'query_image_paths': row['query_image_paths']}
                    for row in batch], 256)
                for row, response in zip(batch, responses):
                    cache[row['payload_sha256']] = {**response, 'record_id': row['record_id'],
                                                  'execution_id': execution_id}
                flush_available()
                progress = {'stage': 'reader', 'completed': len(done), 'total': len(manifest),
                    'fresh_inferences': len(cache), 'empty_responses': sum(not row['raw_response'] for row in done.values()),
                    'token_limit_records': sum(row['hit_token_limit'] for row in done.values())}
                dump(output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        del runtime
    if set(done) != set(mindex):
        raise ValueError('Incomplete paired reader coverage')
    report = {'passed': True, 'questions': len(questions), 'logical_records': len(done),
        'fresh_inferences': len(cache), 'shared_records': len(done) - len(cache),
        'gold_loaded': False, 'candidate_expansion_in_reader': False,
        'empty_responses': sum(not row['raw_response'] for row in done.values()),
        'token_limit_records': sum(row['hit_token_limit'] for row in done.values()),
        'responses_sha256': sha(output / 'responses.jsonl')}
    dump(output / 'validation_output.json', report)
    return report


def normalize_answer_universal(value):
    """Official Mem-Gallery normalization, including ID/decimal protection."""
    import regex
    text = str(value).lower()
    text = regex.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', text)
    text = text.replace('_', 'UNDERSCOREPLACEHOLDER')
    text = regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    text = ''.join(character if character not in string.punctuation else ' ' for character in text)
    text = text.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_')
    return ' '.join(text.split())


def score_answer(prediction, answer):
    from nltk.stem import PorterStemmer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    stemmer = PorterStemmer()
    p, g = normalize_answer_universal(prediction), normalize_answer_universal(answer)
    pt, gt = p.split(), g.split()
    same = sum((Counter(stemmer.stem(w) for w in pt) & Counter(stemmer.stem(w) for w in gt)).values())
    f1 = 2 * same / (len(pt) + len(gt)) if same else 0.0
    smoothing = SmoothingFunction().method1
    result = {'em': float(p == g), 'f1': f1}
    for name, weights in [('bleu', (.25, .25, .25, .25)), ('bleu_1', (1, 0, 0, 0)), ('bleu_2', (.5, .5, 0, 0))]:
        result[name] = float(sentence_bleu([gt], pt, weights=weights, smoothing_function=smoothing)) if pt and gt else 0.0
    return result


def summarize_pairs(rows):
    result = {'questions': len(rows)}
    for condition in CONDITIONS:
        selected = [row[condition] for row in rows]
        metrics = ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')
        result[condition] = {key: sum(row[key] for row in selected) / len(selected) if selected else None for key in metrics}
        with_gold = [row for row in selected if row['gold_clues']]
        result[condition].update(
            nonempty_clue_questions=len(with_gold),
            recall_at_5_macro_nonempty=sum(row['gold_hits'] / row['gold_clues'] for row in with_gold) / len(with_gold) if with_gold else None,
            recall_at_5_micro=sum(row['gold_hits'] for row in with_gold) / sum(row['gold_clues'] for row in with_gold) if with_gold else None,
            hit_rate_at_5_nonempty=sum(row['gold_hits'] > 0 for row in with_gold) / len(with_gold) if with_gold else None,
            token_limit_records=sum(row['hit_token_limit'] for row in selected),
            empty_responses=sum(not row['raw_response'] for row in selected))
    result['same_reader_payload'] = sum(row['same_reader_payload'] for row in rows)
    result['f1_improved'] = sum(row[CONDITIONS[1]]['f1'] > row[CONDITIONS[0]]['f1'] for row in rows)
    result['f1_worsened'] = sum(row[CONDITIONS[1]]['f1'] < row[CONDITIONS[0]]['f1'] for row in rows)
    result['em_gained'] = sum(row[CONDITIONS[1]]['em'] > row[CONDITIONS[0]]['em'] for row in rows)
    result['em_lost'] = sum(row[CONDITIONS[1]]['em'] < row[CONDITIONS[0]]['em'] for row in rows)
    return result


def analyze(root):
    """Only this post-generation stage reads gold answers and clue IDs."""
    root = Path(root).resolve()
    output = root / 'reader'
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['contract']['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Reader source changed: {path}')
    manifest = read_lines(output / 'qa_manifest.jsonl')
    if digest(manifest) != protocol['contract']['manifest_sha256']:
        raise ValueError('Reader manifest hash mismatch')
    responses = read_lines(output / 'responses.jsonl')
    rmap, mmap = by_id(responses, 'record_id'), by_id(manifest, 'record_id')
    if set(rmap) != set(mmap):
        raise ValueError('Cannot score incomplete paired responses')
    gold_path = root / 'inputs/gold.jsonl'
    gold = by_id(read_lines(gold_path))
    ids = list(dict.fromkeys(row['id'] for row in manifest))
    if set(gold) != set(ids):
        raise ValueError('Gold question set mismatch')
    pairs = []
    for qid in ids:
        target = gold[qid]
        pair = {'id': qid, 'answer': target['answer'], 'gold_ids': target['gold_ids']}
        for condition in CONDITIONS:
            key = qid + '::' + condition
            response, source = rmap[key], mmap[key]
            if (response['experiment_signature'] != protocol['experiment_signature'] or
                    response['run_id'] != protocol['run_id'] or response['payload_sha256'] != source['payload_sha256'] or
                    response['top5_memory_ids'] != source['top5_memory_ids']):
                raise ValueError('Response provenance or retrieval mismatch')
            pair['task'], pair['point'] = source['task'], source['point']
            if target['point'] != pair['point']:
                raise ValueError('Gold category mismatch')
            pair[condition] = {**score_answer(response['raw_response'], target['answer']),
                'raw_response': response['raw_response'], 'top5_memory_ids': source['top5_memory_ids'],
                'payload_sha256': source['payload_sha256'], 'hit_token_limit': response['hit_token_limit'],
                'gold_hits': len(set(target['gold_ids']) & set(source['top5_memory_ids'])),
                'gold_clues': len(set(target['gold_ids']))}
        pair['same_reader_payload'] = pair[CONDITIONS[0]]['payload_sha256'] == pair[CONDITIONS[1]]['payload_sha256']
        pairs.append(pair)
    summary = {'status': 'complete', 'metric_policy': 'Official deterministic normalized EM, stemmed token F1 and smoothed BLEU; not an LLM judge or MCQ accuracy.',
        'official_scoring_revision': OFFICIAL_REVISION, 'official_scoring_path': 'benchmark/memengine/evaluate/evaluation.py',
        'gold_loaded_only_after_generation': True, 'retrieval_denominator_policy': 'Exclude empty clue sets from recall and hit-rate denominators; include every QA in answer metrics.',
        'overall': summarize_pairs(pairs),
        'by_domain': {task: summarize_pairs([row for row in pairs if row['task'] == task]) for task in sorted({row['task'] for row in pairs})},
        'by_question_type': {point: summarize_pairs([row for row in pairs if row['point'] == point]) for point in sorted({row['point'] for row in pairs})},
        'artifacts_sha256': {str(path): sha(path) for path in [gold_path, output / 'responses.jsonl', output / 'qa_manifest.jsonl']}}
    dump(output / 'paired_results.json', pairs)
    dump(output / 'summary.json', summary)
    with (output / 'paired_results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        fields = ['id', 'task', 'point', 'answer', 'baseline_answer', 'method_answer', 'baseline_em', 'method_em', 'baseline_f1', 'method_f1']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pairs:
            base, method = row[CONDITIONS[0]], row[CONDITIONS[1]]
            writer.writerow({**{key: row[key] for key in ('id', 'task', 'point', 'answer')},
                'baseline_answer': base['raw_response'], 'method_answer': method['raw_response'],
                'baseline_em': base['em'], 'method_em': method['em'],
                'baseline_f1': base['f1'], 'method_f1': method['f1']})
    print(json.dumps(summary['overall']), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('candidates', 'reader', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, help='Default: 32 for candidates, 16 for reader')
    parser.add_argument('--image-batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'candidates':
        generate_candidates(args.output, args.device, batch_size=args.batch_size if args.batch_size is not None else 32)
    elif args.stage == 'reader':
        run_reader(args.output, args.device, batch_size=args.batch_size if args.batch_size is not None else 16,
                   image_batch_size=args.image_batch_size)
    else:
        analyze(args.output)


if __name__ == '__main__':
    main()
