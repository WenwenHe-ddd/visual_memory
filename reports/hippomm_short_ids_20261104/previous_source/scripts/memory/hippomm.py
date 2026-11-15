"""Isolated HippoMM image adaptation: CLIP, single-image events, no pixel replay.

Access through visual_memory_pipeline.py --memory hippomm. Existing methods,
shared caption caches and dataset files are read-only. No model is loaded at import.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts.memory import hippomm_core as core
from scripts.memory import hippomm_inputs as inputs

ROOT = inputs.ROOT
CLIP = ROOT / 'models/clip-vit-base-patch32'
MODEL = ROOT / 'models/Qwen3-VL-8B-Instruct'
SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_lines(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    temp.replace(path)


def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()


def implementation_hashes():
    # Do not sign the shared dispatcher: adding an unrelated method must not
    # invalidate this method's existing artifacts.
    names = [Path(__file__), Path(core.__file__), Path(inputs.__file__),
             ROOT / 'scripts/graph/visual_image_encoder.py',
             ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    return {str(p): inputs.sha(p) for p in names}


def model_hashes():
    result = {}
    for directory in (CLIP, MODEL):
        files = [p for p in directory.iterdir() if p.is_file() and
                 p.suffix in ('.json', '.txt', '.safetensors', '.bin', '.jinja')]
        if not any(p.suffix in ('.safetensors', '.bin') for p in files):
            raise ValueError('Missing model weights: ' + str(directory))
        result.update({str(p): inputs.sha(p) for p in sorted(files)})
    return result


def verify_hashes(values):
    for name, expected in values.items():
        if inputs.sha(name) != expected:
            raise ValueError('Changed signed file: ' + name)


def contract_at(path, contract):
    record = {'contract': contract, 'signature': digest(contract)}
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError('Existing run has different inputs/settings/code; use a fresh output: ' + str(path))
    else:
        dump(path, record)
    return record


class LocalChat:
    """Lazy Qwen runtime with signed call cache and full raw-response audit."""

    def __init__(self, directory, contract, device, token_limit=512, context_budget=120000):
        self.directory, self.device = Path(directory), device
        self.token_limit, self.context_budget = token_limit, context_budget
        self.signature = digest(contract)
        self.runtime = None
        self.cache_dir = self.directory / 'llm_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, role, prompt, image_paths=()):
        from scripts.evaluation import mem_gallery_master_reader as runtime
        # File paths are ephemeral inference inputs; caches retain hashes only.
        request = {'role': role, 'prompt': prompt, 'image_sha256': [inputs.sha(p) for p in image_paths],
                   'run_signature': self.signature, 'max_new_tokens': self.token_limit}
        key = digest(request)
        path = self.cache_dir / (key + '.json')
        if path.exists():
            row = json.loads(path.read_text())
            if row['request'] != request or row['response_sha256'] != digest(row['response']):
                raise ValueError('Corrupt LLM cache: ' + key)
        else:
            if self.runtime is None:
                self.runtime = runtime.load_runtime(self.device)
                dump(self.directory / 'generation_environment.json', self.runtime[2])
            processor, model, _ = self.runtime
            # Count the exact multimodal input, including original query images.
            from PIL import Image
            content = [{'type': 'image'} for _ in image_paths] + [{'type': 'text', 'text': prompt}]
            rendered = processor.apply_chat_template([
                {'role': 'system', 'content': runtime.READER_SYSTEM},
                {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            images = []
            try:
                for image_path in image_paths:
                    with Image.open(image_path) as image:
                        images.append(image.convert('RGB'))
                encoded = processor(text=[rendered], images=images or None, padding=True,
                                    truncation=False, return_tensors='pt')
                length = int(encoded['attention_mask'].sum())
                del encoded
            finally:
                for image in images:
                    image.close()
            if length + self.token_limit > self.context_budget:
                raise ValueError('HippoMM full context exceeds budget; no silent summary/dialogue truncation')
            response = runtime.generate_batch(self.runtime, [{'system': runtime.READER_SYSTEM,
                'user_text': prompt, 'query_image_paths': list(image_paths)}], self.token_limit)[0]
            row = {'request': request, 'response': response, 'response_sha256': digest(response)}
            dump(path, row)
        if row['response']['hit_token_limit']:
            raise ValueError('LLM output hit token limit; raw output preserved in ' + str(path))
        append(self.directory / 'llm_calls.jsonl', {'role': role, 'request_sha256': key,
               'response_sha256': row['response_sha256'],
               'input_tokens': row['response']['input_tokens'],
               'generated_tokens': row['response']['generated_tokens']})
        return row['response']['raw_response']

    def close(self):
        self.runtime = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ClipText:
    def __init__(self, device):
        self.device = device
        self.model = None

    def __call__(self, text):
        import torch
        if self.model is None:
            from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
            config = AutoConfig.from_pretrained(CLIP, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(CLIP, local_files_only=True)
            self.model, info = CLIPTextModelWithProjection.from_pretrained(CLIP,
                config=config.text_config, local_files_only=True, output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError('Incomplete CLIP text checkpoint')
            self.model.to(device=self.device, dtype=torch.float32).eval()
            self.limit = config.text_config.max_position_embeddings
        length = len(self.tokenizer.encode(text, truncation=False))
        if length > self.limit:
            raise ValueError('Rewritten visual query exceeds CLIP token limit')
        encoded = self.tokenizer([text], padding=True, truncation=False, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            vector = self.model(**encoded).text_embeds[0].float().cpu().numpy()
        return core.unit_vector(vector), {'tokens': length, 'token_limit': self.limit, 'truncated': False}


def construct_events(rounds, images, encode_image, *, duplicate_threshold=.9):
    """Sequential novelty filter; merge source links, never discard dialogue."""
    dialogues = inputs.unique(rounds)
    events, vectors, mapping = [], {}, []
    for image in images:
        path = Path(image['path'])
        if inputs.sha(path) != image['image_sha256']:
            raise ValueError('Historical image hash mismatch during ingestion')
        vector = core.unit_vector(encode_image(path))
        local = [e for e in events if e['task'] == image['task']]
        similarities = [float(np.clip(np.dot(vectors[e['event_id']], vector), -1., 1.)) for e in local]
        best = int(np.argmax(similarities)) if local else None
        redundant = best is not None and similarities[best] >= duplicate_threshold
        if redundant:
            event = local[best]
        else:
            eid = image['task'] + '/theta_' + image['image_sha256']
            event = {'event_id': eid, 'task': image['task'],
                     'representative_image': image['image_sha256'], 'caption': image['caption'],
                     'source_images': [], 'source_round_ids': [], 'summary': ''}
            events.append(event)
            vectors[eid] = vector
        for mid in image['source_round_ids']:
            row = dialogues[mid]
            if row['task'] != event['task']:
                raise ValueError('Cross-task image/dialogue association')
            if mid not in event['source_round_ids']:
                event['source_round_ids'].append(mid)
            for ref in row['images']:
                if ref['image_sha256'] == image['image_sha256']:
                    event['source_images'].append({**ref, 'round_id': mid, 'timestamp': row['timestamp'],
                        'relation': 'visually_redundant' if redundant else 'representative'})
        mapping.append({'task': image['task'], 'image_sha256': image['image_sha256'],
                        'event_id': event['event_id'], 'retained': not redundant,
                        'source_round_ids': image['source_round_ids'],
                        'similarity_to_representative': similarities[best] if redundant else 1.})
        # No rejected image vector, image path or bytes enter persistent memory.
    for event in events:
        event['source_round_ids'].sort(key=lambda mid: dialogues[mid]['insertion_index'])
    expected = {r['id'] for r in rounds if r['images']}
    actual = {mid for e in events for mid in e['source_round_ids']}
    if expected != actual:
        raise ValueError('Lost image-associated dialogue during consolidation')
    return events, vectors, mapping


def build(args):
    rounds, images, sources = inputs.memory_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {'dialogue_rounds': len(rounds), 'image_rounds': sum(bool(r['images']) for r in rounds),
              'distinct_task_images': len(images), 'tasks': sorted({r['task'] for r in rounds})}
    if args.hippomm_stage == 'prepare':
        # Read-only source readiness, not a completed memory or model evaluation.
        dump(args.output / 'readiness.json', {'status': 'inputs_ready_not_built', **counts,
             'source_hashes': sources, 'image_copies': 0})
        print(json.dumps({'stage': 'inputs_ready_not_built', **counts}), flush=True)
        return
    contract = {'method': 'hippomm_image_clip_no_originals', 'schema': SCHEMA, 'dataset': args.dataset,
        'task': args.task, 'duplicate_threshold': args.hippomm_duplicate_threshold,
        'max_new_tokens': args.hippomm_max_new_tokens, 'context_budget': args.hippomm_context_budget,
        'caption_policy': 'reuse verified local Qwen captions, retained representative images only',
        'source_hashes': sources, 'implementation': implementation_hashes(), 'models': model_hashes()}
    signed = contract_at(args.output / 'build_contract.json', contract)
    if (args.output / 'memory.json').exists():
        verify_memory(args.output)
        print('HippoMM: verified existing complete memory.', flush=True)
        return
    from scripts.graph.visual_image_encoder import ImageEncoder
    encoder = ImageEncoder(str(CLIP), device=args.device, local_files_only=True)
    events, vectors, mapping = construct_events(rounds, images, encoder.encode,
                                               duplicate_threshold=args.hippomm_duplicate_threshold)
    encoder_metadata = encoder.metadata
    del encoder, images
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    chat = LocalChat(args.output, contract, args.device, args.hippomm_max_new_tokens, args.hippomm_context_budget)
    dialogues = inputs.unique(rounds)
    try:
        for i, event in enumerate(events, 1):
            event['summary'] = chat('summary', core.summary_prompt(event, dialogues)).strip()
            if not event['summary']:
                raise ValueError('Empty ThetaEvent summary')
            print(f'HippoMM summary {i}/{len(events)}: {event["event_id"]}', flush=True)
    finally:
        chat.close()
    write_lines(args.output / 'events.jsonl', events)
    write_lines(args.output / 'dialogues.jsonl', rounds)
    write_lines(args.output / 'image_mapping.jsonl', mapping)
    np.savez(args.output / 'visual_embeddings.npz',
             event_ids=np.array([e['event_id'] for e in events]),
             vectors=np.stack([vectors[e['event_id']] for e in events]).astype(np.float32))
    artifacts = ('events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz')
    memory = {'offline_method': 'hippomm', 'schema': SCHEMA, 'dataset': args.dataset,
        'build_signature': signed['signature'], 'counts': {**counts, 'events': len(events)},
        'clip_encoder': encoder_metadata, 'history_image_paths_saved': False, 'image_copies': 0,
        'text_only_policy': 'stored for provenance; no standalone event, no text-only retrieval candidates',
        'artifact_sha256': {name: inputs.sha(args.output / name) for name in artifacts}}
    memory['signature'] = digest(memory)
    dump(args.output / 'memory.json', memory)
    verify_memory(args.output)
    print(json.dumps({'stage': 'memory_complete', **memory['counts']}), flush=True)


def verify_memory(directory):
    directory = Path(directory)
    memory = json.loads((directory / 'memory.json').read_text())
    if memory['offline_method'] != 'hippomm' or memory['schema'] != SCHEMA:
        raise ValueError('Not a supported HippoMM memory store')
    if digest({k: v for k, v in memory.items() if k != 'signature'}) != memory['signature']:
        raise ValueError('Memory signature mismatch')
    signed = json.loads((directory / 'build_contract.json').read_text())
    if digest(signed['contract']) != signed['signature'] or signed['signature'] != memory['build_signature']:
        raise ValueError('Build contract signature mismatch')
    # Deliberately do NOT re-open original image files or ingestion sources.
    verify_hashes({str(directory / name): h for name, h in memory['artifact_sha256'].items()})
    events = inputs.read_lines(directory / 'events.jsonl')
    dialogues = inputs.unique(inputs.read_lines(directory / 'dialogues.jsonl'))
    inputs.unique(events, 'event_id')
    if len(events) != memory['counts']['events'] or len(dialogues) != memory['counts']['dialogue_rounds']:
        raise ValueError('Memory counts differ')
    with np.load(directory / 'visual_embeddings.npz', allow_pickle=False) as data:
        ids, matrix = data['event_ids'].tolist(), data['vectors']
    if ids != [e['event_id'] for e in events] or matrix.ndim != 2 or len(matrix) != len(ids):
        raise ValueError('Event/vector alignment mismatch')
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-5):
        raise ValueError('Invalid stored CLIP vectors')
    for e in events:
        if not e['caption'].strip() or not e['summary'].strip():
            raise ValueError('Missing event text')
        if len(e['source_round_ids']) != len(set(e['source_round_ids'])):
            raise ValueError('Duplicate source dialogue')
        if any(mid not in dialogues or dialogues[mid]['task'] != e['task'] for mid in e['source_round_ids']):
            raise ValueError('Invalid event provenance')
    expected = {mid for mid, r in dialogues.items() if r['images']}
    if expected != {mid for e in events for mid in e['source_round_ids']}:
        raise ValueError('Image-associated dialogue coverage mismatch')
    return memory, events, dict(zip(ids, matrix)), dialogues


def prepare(args):
    memory, events, _, _ = verify_memory(args.memory_dir)
    if memory['dataset'] != args.dataset:
        raise ValueError('Dataset differs from offline memory')
    questions, sources = inputs.questions_input(args)
    if not {q['task'] for q in questions} <= {e['task'] for e in events}:
        raise ValueError('Offline memory does not cover requested tasks')
    models = model_hashes()
    build_record = json.loads((args.memory_dir / 'build_contract.json').read_text())
    if models != build_record['contract']['models']:
        raise ValueError('Models differ from offline build; rebuild in a new memory directory')
    contract = {'method': 'hippomm_image_clip_no_originals', 'schema': SCHEMA, 'dataset': args.dataset,
        'memory_dir': str(args.memory_dir.resolve()), 'memory_signature': memory['signature'],
        'questions_sha256': digest(questions), 'question_source_hashes': sources,
        'evaluation_source': str((args.hippomm_gallery_inputs if args.dataset == 'mem_gallery'
                                  else args.benchmark_manifest).resolve()),
        'top_k': args.top_k, 'confidence_threshold': args.hippomm_confidence_threshold,
        'fallback_threshold': args.hippomm_fallback_threshold, 'fallback_score': args.hippomm_fallback_score,
        'max_new_tokens': args.hippomm_max_new_tokens, 'context_budget': args.hippomm_context_budget,
        'fast_policy': 'all same-task ThetaEvent summaries only; no detail text or options; no truncation',
        'retrieval_policy': '2-5 word rewritten question; CLIP cosine; per-image caption fallback; global Top-K',
        'query_image_policy': 'original query pixels may be used by fast/answer LLM; never by CLIP rewrite or fallback',
        'mcq_policy': 'choices only in final answer or fast-answer-to-choice formatting',
        'implementation': implementation_hashes(), 'models': models}
    args.output.mkdir(parents=True, exist_ok=True)
    signed = contract_at(args.output / 'protocol.json', contract)
    path = args.output / 'questions.jsonl'
    if path.exists() and inputs.read_lines(path) != questions:
        raise ValueError('Prepared questions changed')
    write_lines(path, questions)
    print(json.dumps({'stage': 'prepared', 'questions': len(questions), 'signature': signed['signature']}), flush=True)


def load_run(args, *, verify_models=False):
    signed = json.loads((args.output / 'protocol.json').read_text())
    c = signed['contract']
    if digest(c) != signed['signature'] or c['dataset'] != args.dataset:
        raise ValueError('Run protocol mismatch')
    if Path(c['memory_dir']) != args.memory_dir.resolve():
        raise ValueError('--memory-dir differs from prepared run')
    values = {'top_k': args.top_k, 'confidence_threshold': args.hippomm_confidence_threshold,
              'fallback_threshold': args.hippomm_fallback_threshold, 'fallback_score': args.hippomm_fallback_score,
              'max_new_tokens': args.hippomm_max_new_tokens, 'context_budget': args.hippomm_context_budget}
    if any(c[key] != value for key, value in values.items()):
        raise ValueError('Runtime settings differ from prepared run')
    verify_hashes(c['implementation'])
    if verify_models:
        verify_hashes(c['models'])
    memory, events, vectors, dialogues = verify_memory(args.memory_dir)
    if memory['signature'] != c['memory_signature']:
        raise ValueError('Offline memory changed')
    questions = inputs.read_lines(args.output / 'questions.jsonl')
    if digest(questions) != c['questions_sha256']:
        raise ValueError('Question artifact changed')
    inputs.unique(questions)
    for q in questions:
        verify_hashes(q['query_image_hashes'])
    return signed, questions, events, vectors, dialogues


def checked_responses(output, signed, questions):
    path = output / 'qa_responses.jsonl'
    responses = inputs.read_lines(path) if path.exists() else []
    inputs.unique(responses)
    qmap = inputs.unique(questions)
    for r in responses:
        if r['id'] not in qmap or r['experiment_signature'] != signed['signature']:
            raise ValueError('Response identity/signature mismatch')
        if r['question_sha256'] != digest(qmap[r['id']]):
            raise ValueError('Response question mismatch')
        if digest({k: v for k, v in r.items() if k != 'record_sha256'}) != r['record_sha256']:
            raise ValueError('Response trace changed')
    return responses


def run(args):
    signed, questions, events, vectors, dialogues = load_run(args, verify_models=True)
    responses = checked_responses(args.output, signed, questions)
    done = {r['id'] for r in responses}
    from scripts.evaluation.mem_gallery_master_reader import TASK_INSTRUCTIONS
    chat = LocalChat(args.output, signed['contract'], args.device,
                     args.hippomm_max_new_tokens, args.hippomm_context_budget)
    encoder = ClipText(args.device)
    try:
        for q in questions:
            if q['id'] in done:
                continue
            start = time.perf_counter()
            call_keys = []

            def llm(role, prompt):
                paths = q['query_image_paths'] if role in ('fast', 'answer') else []
                if role == 'fast' and q['point'] in TASK_INSTRUCTIONS:
                    prompt += '\nAnswer formatting: ' + TASK_INSTRUCTIONS[q['point']]
                result = chat(role, prompt, paths)
                # Bind each final answer to the exact cached requests and outputs.
                request = {'role': role, 'prompt': prompt, 'image_sha256': [inputs.sha(p) for p in paths],
                           'run_signature': chat.signature, 'max_new_tokens': chat.token_limit}
                key = digest(request)
                cached = json.loads((chat.cache_dir / (key + '.json')).read_text())
                call_keys.append({'request_sha256': key, 'response_sha256': cached['response_sha256']})
                return result

            trace = core.answer_question(q, events, vectors, dialogues, llm, encoder,
                top_k=args.top_k, confidence_threshold=args.hippomm_confidence_threshold,
                fallback_threshold=args.hippomm_fallback_threshold, fallback_score=args.hippomm_fallback_score,
                task_instruction=TASK_INSTRUCTIONS.get(q['point'], ''))
            trace.update(experiment_signature=signed['signature'], question_sha256=digest(q),
                         elapsed_seconds=time.perf_counter() - start, llm_calls=call_keys,
                         historical_image_reads=0)
            trace['record_sha256'] = digest(trace)
            append(args.output / 'qa_responses.jsonl', trace)
            print(json.dumps({'id': q['id'], 'route': trace['route'], 'answer': trace['answer']}), flush=True)
    finally:
        chat.close()


def analyze(args):
    signed, questions, events, _, dialogues = load_run(args)
    responses = checked_responses(args.output, signed, questions)
    if {r['id'] for r in responses} != {q['id'] for q in questions}:
        raise ValueError('Incomplete answers; no final metrics written')
    # Gold enters only after answers have been frozen. Verify the source used
    # at prepare before reading it for evaluation.
    verify_hashes(signed['contract']['question_source_hashes'])
    source = args.hippomm_gallery_inputs if args.dataset == 'mem_gallery' else args.benchmark_manifest
    if str(source.resolve()) != signed['contract']['evaluation_source']:
        raise ValueError('Evaluation source differs from prepared run')
    gold = inputs.evaluation_gold(args)
    from scripts.evaluation.mem_gallery_master_reader import score_answer
    rows = []
    total_gold = total_hit = 0
    for r in responses:
        g = gold[r['id']]
        labels = set(g['gold_ids'])
        context = set(r['context_round_ids'])
        hit = len(labels & context)
        total_gold += len(labels)
        total_hit += hit
        for call in r['llm_calls']:
            cache = json.loads((args.output / 'llm_cache' / (call['request_sha256'] + '.json')).read_text())
            if digest(cache['request']) != call['request_sha256'] or digest(cache['response']) != call['response_sha256']:
                raise ValueError('LLM provenance mismatch')
        metrics = ({'accuracy': float(r['answer'] == g['gold_choice']),
                    'format_error': r['answer'] not in 'ABCD' or len(r['answer']) != 1}
                   if args.dataset == 'memeye' else score_answer(r['answer'], g['answer']))
        rows.append({'id': r['id'], 'task': r['task'], 'route': r['route'], **metrics,
                     'context_gold_hits': hit, 'gold_count': len(labels),
                     'source_round_recall': hit / len(labels) if labels else None})
    metric_names = ('accuracy',) if args.dataset == 'memeye' else ('em', 'f1', 'bleu')
    summary = {'method': 'hippomm_image_clip_no_originals', 'questions': len(rows),
        'experiment_signature': signed['signature'],
        'fast_answers': sum(r['route'] == 'fast' for r in rows),
        'detailed_answers': sum(r['route'] == 'detailed' for r in rows),
        'format_errors': sum(r.get('format_error', False) for r in rows),
        'mean_answer_seconds': sum(r['elapsed_seconds'] for r in responses) / len(responses),
        'context_source_micro_recall': total_hit / total_gold if total_gold else None,
        'recall_note': 'Source-round coverage of supplied context; summary provenance is not proof that every fact survived summarization.',
        'historical_image_reads_during_query': 0,
        'text_only_rounds_outside_image_events': sum(not r['images'] for r in dialogues.values()),
        'events': len(events), 'retained_visual_bytes': (args.memory_dir / 'visual_embeddings.npz').stat().st_size,
        'memory_artifact_bytes': sum((args.memory_dir / name).stat().st_size for name in
                                    ('events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz'))}
    for key in metric_names:
        summary[key] = sum(r[key] for r in rows) / len(rows)
    write_lines(args.output / 'per_question.jsonl', rows)
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'completion_manifest.json', {'status': 'complete', 'questions': len(rows),
        'experiment_signature': signed['signature'], 'artifacts': {name: inputs.sha(args.output / name)
        for name in ('questions.jsonl', 'qa_responses.jsonl', 'per_question.jsonl', 'summary.json')}})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def add_arguments(parser, phase):
    group = parser.add_argument_group('hippomm: isolated single-image CLIP memory, no historical pixel replay')
    group.add_argument('--hippomm-gallery-inputs', type=Path,
                       default=ROOT / 'results/mem_gallery_master_answer_20261021/inputs')
    group.add_argument('--hippomm-max-new-tokens', type=int, default=512)
    group.add_argument('--hippomm-context-budget', type=int, default=120000)
    if phase == 'offline':
        group.add_argument('--hippomm-stage', choices=['prepare', 'build'], default='build',
                           help='prepare validates input schemas only; build encodes and summarizes')
        group.add_argument('--hippomm-duplicate-threshold', type=float, default=.9)
    else:
        group.add_argument('--hippomm-confidence-threshold', type=float, default=.75)
        group.add_argument('--hippomm-fallback-threshold', type=float, default=.4)
        group.add_argument('--hippomm-fallback-score', type=float, default=.6)


def pipeline(args, parser):
    default = ROOT / 'results/debug_caption_master_answer'
    if args.output == default / args.phase:
        args.output = ROOT / 'results/visual_memory_pipeline/hippomm' / args.dataset / args.phase
    if args.hippomm_max_new_tokens < 1 or args.hippomm_context_budget <= args.hippomm_max_new_tokens:
        parser.error('HippoMM requires positive generation tokens and a larger context budget')
    if args.phase == 'offline':
        if args.memory_source or args.image_files or args.limit or args.from_start:
            parser.error('HippoMM uses full task histories; use --task, not legacy offline filters/export')
        threshold = args.hippomm_duplicate_threshold
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            parser.error('--hippomm-duplicate-threshold must be in (0, 1]')
        return build(args)
    if args.memory_path is not None:
        if args.memory_dir != default / 'offline/memory':
            parser.error('Choose --memory-path or --memory-dir')
        args.memory_dir = args.memory_path
    elif args.memory_dir == default / 'offline/memory':
        args.memory_dir = ROOT / 'results/visual_memory_pipeline/hippomm' / args.dataset / 'offline'
    if args.output.resolve() == args.memory_dir.resolve():
        parser.error('HippoMM online output must differ from the offline memory directory')
    if (args.reader_backend != 'local' or args.online_weighting is not None or args.retrieval is not None
            or args.retrieval_scoring == 'z_reweight' or args.whitening_lambda is not None
            or args.reweight_alpha is not None or args.query_image_captions or args.encode_queries
            or args.simplemem_backbone != 'local-qwen' or args.output_root != ROOT / 'results/visual_memory_pipeline'):
        parser.error('HippoMM uses its own retrieval and local reader; use --output and omit other-method flags')
    args.top_k = 5 if args.top_k is None else args.top_k
    if args.top_k < 1 or args.limit_questions < 0:
        parser.error('HippoMM requires positive --top-k and nonnegative --limit-questions')
    for name in ('hippomm_confidence_threshold', 'hippomm_fallback_threshold', 'hippomm_fallback_score'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error('--' + name.replace('_', '-') + ' must be finite and in [0, 1]')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('run', 'all'):
        run(args)
    if args.stage in ('analyze', 'all'):
        analyze(args)
