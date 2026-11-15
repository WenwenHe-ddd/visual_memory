"""Read-only input adapters; historical image paths exist only during ingestion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, key='id'):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate ' + key)
    return result


def gallery_files(directory, names):
    directory = Path(directory)
    manifest = json.loads((directory / 'artifact_manifest.json').read_text())
    for name in names:
        if sha(directory / name) != manifest.get(name):
            raise ValueError('Changed Mem-Gallery input: ' + name)
    return {str(directory / name): sha(directory / name) for name in [*names, 'artifact_manifest.json']}


def memory_inputs(args):
    """Return whitelisted dialogue records and temporary, distinct image inputs."""
    rounds, images, sources = [], {}, {}
    if args.dataset == 'mem_gallery':
        directory = args.hippomm_gallery_inputs
        sources = gallery_files(directory, ['memories.jsonl', 'protocol.json'])
        raw = read_lines(directory / 'memories.jsonl')
        unique(raw)
        for row in raw:
            if args.task and row['task'] != args.task:
                continue
            refs = []
            if row['is_visual']:
                key = (row['task'], row['image_sha256'])
                refs = [{'image_sha256': row['image_sha256'], 'image_id': row['image_id']}]
                item = images.setdefault(key, {'task': row['task'], 'image_sha256': row['image_sha256'],
                    'path': row['image_path'], 'caption': row['caption'], 'source_round_ids': []})
                if item['caption'] != row['caption']:
                    raise ValueError('Same image has inconsistent cached captions')
                item['source_round_ids'].append(row['id'])
            rounds.append({k: row[k] for k in ('id', 'task', 'round', 'session_id', 'timestamp', 'dialogue', 'insertion_index')}
                          | {'images': refs})
    else:
        root = args.natural_root
        config = json.loads((root / 'protocol.json').read_text())
        from scripts.data.generate import signature
        caption_rows = read_lines(root / 'captions.jsonl')
        captions = unique(caption_rows)
        expected = signature(config)
        if any(r.get('protocol_signature') != expected or not r['caption'].strip() for r in caption_rows):
            raise ValueError('Natural caption cache provenance mismatch')
        image_rows = read_lines(root / 'images.jsonl')
        by_hash = unique(image_rows)
        by_relative = {rel: r for r in image_rows for rel in r['source_paths']}
        memories = unique(read_lines(root / 'memories.jsonl'))
        sources = {str(root / name): sha(root / name) for name in
                   ('protocol.json', 'captions.jsonl', 'images.jsonl', 'memories.jsonl')}
        for source in config['source_files']:
            path = ROOT / source['path']
            task = path.stem.removesuffix('_Open')
            if args.task and task != args.task:
                continue
            if sha(path) != source['sha256']:
                raise ValueError('Changed source dialogue: ' + str(path))
            sources[str(path)] = source['sha256']
            index = 0
            # Only dialogue fields are consumed; all QA content is ignored.
            for session in json.loads(path.read_text())['multi_session_dialogues']:
                for row in session['dialogues']:
                    mid = task + '/' + row['round']
                    paths = row.get('input_image') or []
                    paths = [paths] if isinstance(paths, str) else paths
                    official_ids = row.get('image_id') or []
                    official_ids = [official_ids] if isinstance(official_ids, str) else official_ids
                    refs = []
                    for n, relative in enumerate(paths):
                        image = by_relative[relative]
                        image_hash = image['id']
                        label = official_ids[n] if n < len(official_ids) else relative
                        refs.append({'image_sha256': image_hash, 'image_id': label})
                        key = (task, image_hash)
                        item = images.setdefault(key, {'task': task, 'image_sha256': image_hash,
                            'path': by_hash[image_hash]['path'], 'caption': captions[image_hash]['caption'],
                            'source_round_ids': []})
                        if mid not in item['source_round_ids']:
                            item['source_round_ids'].append(mid)
                    if paths and (mid not in memories or memories[mid]['images'] != [r['image_sha256'] for r in refs]):
                        raise ValueError('Dialogue/image manifest mismatch: ' + mid)
                    rounds.append({'id': mid, 'task': task, 'round': row['round'],
                        'session_id': session['session_id'], 'timestamp': session.get('date', ''),
                        'dialogue': 'User: ' + row.get('user', '') + '\nAssistant: ' + row.get('assistant', ''),
                        'insertion_index': index, 'images': refs})
                    index += 1
    unique(rounds)
    if not rounds or not images:
        raise ValueError('No historical images/dialogues match --task')
    return rounds, list(images.values()), sources


def questions_input(args):
    if args.dataset == 'mem_gallery':
        sources = gallery_files(args.hippomm_gallery_inputs, ['questions.jsonl', 'protocol.json'])
        raw = read_lines(args.hippomm_gallery_inputs / 'questions.jsonl')
    else:
        raw = read_lines(args.benchmark_manifest)
        sources = {str(args.benchmark_manifest): sha(args.benchmark_manifest)}
    unique(raw)
    questions = []
    for row in raw:
        if args.task and row['task'] != args.task:
            continue
        if args.question_ids and row['id'] not in args.question_ids and row['id'].split('/')[-1] not in args.question_ids:
            continue
        q = {k: row[k] for k in ('id', 'task', 'question', 'query_image_paths')}
        q['point'] = row.get('point', '')
        if args.dataset == 'memeye':
            q['options'] = {k: row['options'][k] for k in 'ABCD'}
        q['query_image_hashes'] = {p: sha(p) for p in q['query_image_paths']}
        questions.append(q)
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    if not questions:
        raise ValueError('No questions match the requested filters')
    return questions, sources


def evaluation_gold(args):
    """Called by analyze only; never imported into any model prompt."""
    if args.dataset == 'mem_gallery':
        gallery_files(args.hippomm_gallery_inputs, ['gold.jsonl'])
        return unique(read_lines(args.hippomm_gallery_inputs / 'gold.jsonl'))
    return unique(read_lines(args.benchmark_manifest))
