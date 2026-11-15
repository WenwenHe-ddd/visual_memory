"""Generate objects, generate relations, then complete objects from relation endpoints."""

import fcntl
import json
from collections import Counter
from pathlib import Path

FORMAT = 'objects_first'
OBJECT_RETRY_LIMIT_ERROR = 'object retry returned more than 30 distinct objects'
OBJECT_RULES = '''List the distinct objects/entities visible in the image or explicitly named in the caption.
Use short, consistent names; distinguish different instances. Do not invent objects.
Return only JSON: {"objects": ["person", "cup"]}. No relations, IDs or evidence.
'''
RELATION_RULES = '''Using the image and caption, extract factual relationships.
OBJECTS_JSON is a starting point; new subject/object names are allowed.
Reuse existing names for the same entity. Include only facts supported by the image or caption.
Return only JSON: {"relations": [{"subject": "person", "predicate": "holds", "object": "cup"}]}.
'''


def cache_path(root, name):
    if (not isinstance(name, str) or not name or name.strip() != name
            or name in ('.', '..') or any(c in name for c in ('/', '\\', '\0'))):
        raise ValueError('cache name must be a valid filename component')
    return root / 'generations' / (name + '.json')


def prompt(caption):
    return OBJECT_RULES + '\nCAPTION_JSON:\n' + json.dumps(caption, ensure_ascii=False)


def relation_prompt(caption, objects):
    return (RELATION_RULES + '\nOBJECTS_JSON:\n' + json.dumps(objects, ensure_ascii=False)
            + '\nCAPTION_JSON:\n' + json.dumps(caption, ensure_ascii=False))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def parse_objects(raw):
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'objects'} or not isinstance(obj['objects'], list):
        raise ValueError('expected an objects list')
    if any(not isinstance(name, str) or not name.strip() for name in obj['objects']):
        raise ValueError('object names must be nonempty strings')
    return list(dict.fromkeys(clean_name(name) for name in obj['objects']))


def parse_output(raw):
    text = raw.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'relations'} or not isinstance(obj['relations'], list):
        raise ValueError('expected a relations list')
    for row in obj['relations']:
        if not isinstance(row, dict) or set(row) != {'subject', 'predicate', 'object'}:
            raise ValueError('each relation needs subject, predicate and object')
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError('relation fields must be nonempty strings')
    return obj['relations']


def clean_name(value):
    return ' '.join(value.split())


def relation_sentences(relations):
    sentences = []
    for relation in relations:
        sentence = ' '.join(clean_name(relation[key]) for key in ('subject', 'predicate', 'object'))
        if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
            sentence += '.'
        sentences.append(sentence)
    return sentences


def organize_relations(relations):
    """Trim whitespace and remove duplicate directed triples; preserve distinct predicates."""
    organized, seen = [], set()
    for row in relations:
        triple = tuple(clean_name(row[key]) for key in ('subject', 'predicate', 'object'))
        if triple not in seen:
            seen.add(triple)
            organized.append(dict(zip(('subject', 'predicate', 'object'), triple)))
    return organized


def complete_objects(objects, relations):
    """Stage 3: keep initial objects and append unseen relation endpoints in order."""
    names = dict.fromkeys(clean_name(name) for name in objects)
    for relation in relations:
        for key in ('subject', 'object'):
            names.setdefault(clean_name(relation[key]), None)
    return list(names)


def process_output(raw, objects):
    generated = parse_output(raw)
    relations = organize_relations(generated)
    initial_objects = list(dict.fromkeys(clean_name(name) for name in objects))
    completed = complete_objects(initial_objects, relations)
    return {'generated_relations': relation_sentences(generated), 'initial_objects': initial_objects,
            'added_objects': completed[len(initial_objects):],
            'objects': completed, 'relations': list(dict.fromkeys(relation_sentences(relations)))}


def unique_rows(rows, label):
    mapped = {row['id']: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError('duplicate ' + label + ' IDs')
    return mapped


def prepare(args):
    root = args.natural_root
    paths = [root / name for name in ('protocol.json', 'captions.jsonl', 'memories.jsonl', 'images.jsonl')]
    config = json.loads(paths[0].read_text())
    captions = unique_rows(read_jsonl(paths[1]), 'caption')
    memories = read_jsonl(paths[2])
    unique_rows(memories, 'memory')
    images = unique_rows(read_jsonl(paths[3]), 'image')
    name_counts = Counter(Path(image['path']).stem for image in images.values())
    tasks = sorted({m['task'] for m in memories})
    if args.task and args.task not in tasks:
        raise ValueError('unknown task: ' + args.task)
    memories = [m for m in memories if not args.task or m['task'] == args.task]
    refs = {}
    for memory in memories:
        for image_id in dict.fromkeys(memory['images']):
            refs.setdefault(image_id, []).append({'memory_id': memory['id'], 'task': memory['task']})
    if not refs:
        raise ValueError('no memory images selected')
    requests = []
    for image_id, references in refs.items():
        row = captions.get(image_id)
        if row is None or image_id not in images:
            raise ValueError('missing cached caption or image manifest entry: ' + image_id)
        caption = row.get('caption')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('empty caption: ' + image_id)
        if row.get('hit_token_limit') is not False:
            raise ValueError('truncated caption: ' + image_id)
        image_path = Path(images[image_id]['path']).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        name = image_path.stem
        if name_counts[name] > 1:
            name = image_path.parent.name + '__' + name
        cache_path(args.object_relation_root, name)
        requests.append({'image_id': image_id, 'image_path': str(image_path),
                         'cache_name': name,
                         'references': references, 'caption': caption, 'prompt': prompt(caption)})
    if len({r['cache_name'] for r in requests}) != len(requests):
        raise ValueError('duplicate image filenames within the same source directory')
    protocol = {'format': FORMAT, 'tasks': sorted({m['task'] for m in memories}),
                'model_path': config['caption_model_path'], 'seed': config['seed'],
                'model_provenance': config['provenance']['caption_model'],
                'dtype': 'bfloat16', 'do_sample': False,
                'image_max_pixels': config['image_max_pixels'],
                'max_new_tokens': args.extraction_max_new_tokens,
                'object_prompt_template': OBJECT_RULES, 'relation_prompt_template': RELATION_RULES,
                'generation_input': 'both stages receive original image and cached caption; second stage also receives first-stage objects',
                'object_completion': 'third stage appends unseen relation endpoints to initial objects without another model call',
                'unit': 'one extraction per unique memory image; retain all memory references',
                'retry_policy': 'one retry per stage for invalid format or empty output; accept empty after recheck',
                'validation': 'nonempty names; new relation endpoints allowed and added to objects; exact deduplication; no semantic verification',
                'source_files': [str(p.resolve()) for p in paths]}
    return config, protocol, requests


def generate_stage(writer, text, image_path, stage, max_tokens, parser, is_empty, attempts, checkpoint):
    compact_objects = stage == 'object_extraction' and (
        (len(attempts) >= 2 and all('token limit' in (row.get('error') or '') for row in attempts[-2:]))
        or (bool(attempts) and attempts[-1].get('error') == OBJECT_RETRY_LIMIT_ERROR)
        or (bool(attempts) and attempts[-1].get('retry_strategy') == 'main_objects'))
    for attempt in range(2):
        request_text = text
        generation_options = {}
        stage_max_tokens = max_tokens
        if compact_objects:
            request_text += ('\nFor this retry, list only the most prominent visible entities.'
                             ' Use at most 5 short object names; do not fill the list to reach five.')
            generation_options['repetition_penalty'] = 1.0
            stage_max_tokens = min(max_tokens, 512)
        elif attempt or (attempts and attempts[-1].get('error')):
            request_text += ('\nRecheck both the image and caption. Previous issue: '
                             + attempts[-1]['error'] + '.')
            if stage == 'relation_extraction' and attempts[-1]['error'] == 'relation fields must be nonempty strings':
                request_text += (' Extract only relationships between two entities.'
                                 ' Exclude standalone descriptions or attributes of a single entity.'
                                 ' Both endpoints must be nonempty entity names, never null.')
            if 'token limit' in attempts[-1]['error']:
                request_text += (' Return at most 30 distinct entries, each only once.'
                                 ' Use concise names and predicates; close the JSON and stop.')
                generation_options['repetition_penalty'] = 1.1
        attempt_options = {'generation_options': generation_options} if generation_options else {}
        if compact_objects:
            attempt_options = {'retry_strategy': 'main_objects', 'generation_options': {
                **generation_options, 'max_new_tokens': stage_max_tokens}}
        raw = None
        writer.last_raw = None
        try:
            raw = writer._ask_plain(request_text, stage, stage_max_tokens, image_path=image_path,
                                    **generation_options)
            parsed = parser(raw)
            if compact_objects and len(parsed) > 5:
                raise ValueError('main-object recovery returned more than 5 objects')
            if generation_options.get('repetition_penalty') == 1.1:
                distinct_count = len(parsed) if stage == 'object_extraction' else len(organize_relations(parsed))
                if distinct_count > 30:
                    raise ValueError(OBJECT_RETRY_LIMIT_ERROR if stage == 'object_extraction'
                                     else 'relation retry returned more than 30 distinct relations')
            if is_empty(parsed) and attempt == 0:
                raise ValueError('empty ' + stage + ' output; recheck once')
        except Exception as exc:
            attempts.append({'prompt': request_text, 'raw_output': raw or writer.last_raw,
                             'error': str(exc), **attempt_options})
            checkpoint()
            if attempt == 0 and isinstance(exc, ValueError):
                continue
            raise
        attempts.append({'prompt': request_text, 'raw_output': raw, 'error': None, **attempt_options})
        checkpoint()
        return raw, parsed


def build(args, writer_factory=None):
    root = args.object_relation_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return build_locked(args, writer_factory)


def build_locked(args, writer_factory):
    config, protocol, requests = prepare(args)
    root = args.object_relation_root
    run_path = root / 'run_config.json'
    if run_path.exists() and json.loads(run_path.read_text()) != protocol:
        raise ValueError('extraction sources/settings changed; review existing cache before reusing it')
    requests_path = root / 'requests.jsonl'
    if requests_path.exists() and read_jsonl(requests_path) != requests:
        raise ValueError('extraction sources/settings changed; review existing requests before reusing them')
    write_json(run_path, protocol)
    write_jsonl(requests_path, requests)
    rows, previous = {}, {}
    for request in requests:
        image_id = request['image_id']
        path = cache_path(root, request['cache_name'])
        if not path.exists():
            continue
        cached = json.loads(path.read_text())
        if cached.get('request') != request:
            raise ValueError('cached extraction request mismatch: ' + image_id)
        previous[image_id] = cached
        if cached.get('status') == 'completed':
            objects = parse_objects(cached['raw_objects_output'])
            parsed = process_output(cached['raw_output'], objects)
            if any(cached.get(key) != value for key, value in parsed.items()):
                legacy = {**parsed, 'relations': organize_relations(parse_output(cached['raw_output']))}
                if any(cached.get(key) != value for key, value in legacy.items()):
                    raise ValueError('cached extraction differs from raw output: ' + image_id)
                cached.update(parsed)
                write_json(path, cached)
            rows[image_id] = {**request, **parsed}
        elif cached.get('status') not in ('failed', 'incomplete'):
            raise ValueError('invalid extraction cache status: ' + image_id)

    def snapshot():
        output = [rows[r['image_id']] for r in requests if r['image_id'] in rows]
        write_jsonl(root / 'extractions.jsonl', output)
        summary = {'format': FORMAT, 'status': 'complete' if len(output) == len(requests) else 'incomplete',
                   'requested_images': len(requests), 'completed_images': len(output),
                   'pending_images': len(requests) - len(output),
                   'failed_images': sum(r.get('status') == 'failed' for r in previous.values()),
                   'memory_references': sum(len(r['references']) for r in requests),
                   'empty_images': sum(not r['relations'] for r in output),
                   'objects': sum(len(r['objects']) for r in output),
                   'relations': sum(len(r['relations']) for r in output)}
        write_json(root / 'summary.json', summary)
        return summary

    summary = snapshot()
    if args.extraction_stage == 'prepare':
        print(json.dumps(summary), flush=True)
        return summary
    writer, generated = None, 0
    try:
        for request in requests:
            image_id = request['image_id']
            if image_id in rows and not args.from_start:
                continue
            if args.limit and generated >= args.limit:
                break
            if writer is None:
                if writer_factory is None:
                    from scripts.memory.qwen_generation import QwenGenerator
                    writer = QwenGenerator(config['caption_model_path'], config['image_max_pixels'],
                                             args.extraction_max_new_tokens, config['seed'], '', 256)
                else:
                    writer = writer_factory()
            rows.pop(image_id, None)
            old = {} if args.from_start else previous.get(image_id, {})
            cached = {'request': request, 'status': 'incomplete',
                      'object_attempts': list(old.get('object_attempts', [])),
                      'attempts': list(old.get('attempts', []))}
            path = cache_path(root, request['cache_name'])
            def checkpoint():
                write_json(path, cached)
            try:
                raw_objects, objects = generate_stage(
                    writer, request['prompt'], request['image_path'], 'object_extraction',
                    args.extraction_max_new_tokens, parse_objects, lambda value: not value,
                    cached['object_attempts'], checkpoint)
                cached.update(raw_objects_output=raw_objects, initial_objects=objects, objects=objects)
                checkpoint()
                raw, generated_relations = generate_stage(
                    writer, relation_prompt(request['caption'], objects), request['image_path'],
                    'relation_extraction', args.extraction_max_new_tokens,
                    parse_output, lambda value: not value, cached['attempts'], checkpoint)
                cached.update(raw_output=raw, generated_relations=relation_sentences(generated_relations))
                checkpoint()
                parsed = process_output(raw, objects)
                cached.update(status='completed', raw_output=raw, **parsed)
                checkpoint()
                previous[image_id] = cached
                rows[image_id] = {**request, **parsed}
            except Exception as exc:
                cached['status'] = 'failed'
                checkpoint()
                previous[image_id] = cached
                if getattr(args, 'continue_on_error', False) and isinstance(exc, ValueError):
                    generated += 1
                    snapshot()
                    print(f'object_relation failed: {request["cache_name"]}: {exc}', flush=True)
                    continue
                raise
            generated += 1
            snapshot()
            print(f'object_relation {len(rows)}/{len(requests)}: {request["cache_name"]}', flush=True)
    finally:
        summary = snapshot()
    print(json.dumps(summary), flush=True)
    return summary
