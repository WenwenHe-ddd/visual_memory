"""Prepare audited Mem-Gallery inputs with gold annotations kept separately.

Historical captions use the existing Qwen cache. Query-image captions remain
the original supplied captions. This module only reads data; it loads no model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CAPTION_FIELD = 'image_caption_qwen3_vl_8b_instruct'
DEFAULT_CAPTIONS = ROOT / 'data/Mem-Gallery/captions/qwen3-vl-8b-instruct_20261018_v1'
DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_master_answer_20261021/inputs'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                   for row in rows).encode('utf-8')


def prepare(output_dir: Path = DEFAULT_OUTPUT, *, root: Path = ROOT,
            caption_dir: Path | None = None) -> dict:
    """Validate every source, then create deterministic, resumable input files.

    Existing matching output files are left untouched. If any expected file
    differs, nothing is overwritten, including when source manifests change.
    """
    root, output_dir = root.resolve(), output_dir.resolve()
    base = root / 'data/Mem-Gallery'
    caption_dir = (caption_dir or base / 'captions/qwen3-vl-8b-instruct_20261018_v1').resolve()
    source_path, image_path = base / 'source_manifest.json', base / 'image_manifest.json'
    source_manifest, image_manifest = read_json(source_path), read_json(image_path)
    cache_manifest_path = caption_dir / 'artifact_manifest.json'
    cache_manifest = read_json(cache_manifest_path)
    require(source_manifest['revision'] == image_manifest['revision'], 'Source/image revision mismatch')
    source_hashes: dict[str, str] = {}

    def remember(path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        require(path.is_file(), f'Missing input: {path}')
        digest = sha256(path)
        require(expected is None or digest == expected, f'Input hash mismatch: {path}')
        try:
            key = str(path.relative_to(root))
        except ValueError:
            key = str(path)
        source_hashes[key] = digest
        return digest

    for path in (source_path, image_path, cache_manifest_path):
        remember(path)
    for relative, expected in cache_manifest.items():
        remember(caption_dir / relative, expected)
    audit = read_json(caption_dir / 'independent_audit.json')
    require(audit.get('status') == 'passed', 'Qwen caption cache has not passed its audit')

    image_hashes: dict[Path, str] = {}
    for entry in image_manifest['files']:
        path = (base / entry['path']).resolve()
        require(path not in image_hashes, f'Duplicate image manifest path: {path}')
        image_hashes[path] = remember(path, entry['sha256'])

    memories, questions, gold, candidate_inputs = [], [], [], []
    totals, by_type = Counter(), Counter()
    scenario_counts = []
    all_ids: set[str] = set()
    seen_images: set[Path] = set()
    for entry in source_manifest['files']:
        path = (root / entry['path']).resolve()
        remember(path, entry['sha256'])
        original = read_json(path)
        derived_path = caption_dir / 'dialog_with_qwen_caption' / path.name
        require(str(derived_path.relative_to(caption_dir)) in cache_manifest,
                f'Derived JSON missing from cache manifest: {derived_path}')
        derived = read_json(derived_path)
        stripped = copy.deepcopy(derived)
        for session in stripped['multi_session_dialogues']:
            for row in session['dialogues']:
                row.pop(CAPTION_FIELD, None)
        for qa in stripped['human-annotated QAs']:
            qa.pop(CAPTION_FIELD, None)
        require(stripped == original, f'Derived JSON changed original fields: {derived_path}')
        task, counts, round_ids, visual_ids = path.stem, Counter(), set(), set()

        def resolve_image(relative: str) -> Path:
            resolved = (path.parent / relative).resolve()
            require(resolved in image_hashes, f'Image reference absent from manifest: {resolved}')
            seen_images.add(resolved)
            return resolved

        for session in derived['multi_session_dialogues']:
            counts['sessions'] += 1
            for row in session['dialogues']:
                mid = task + '/' + row['round']
                require(mid not in all_ids, f'Duplicate memory ID: {mid}')
                all_ids.add(mid)
                round_ids.add(mid)
                images = row.get('input_image') or []
                require(len(images) <= 1, f'Multiple images require a separate memory-unit policy: {mid}')
                visual = bool(images)
                caption, actual_image, image_id = '', None, None
                if visual:
                    captions = row.get(CAPTION_FIELD) or []
                    ids = row.get('image_id') or []
                    require(len(captions) == len(ids) == 1 and
                            isinstance(captions[0], str) and bool(captions[0].strip()),
                            f'Missing image caption/ID: {mid}')
                    require(isinstance(ids[0], str) and bool(ids[0]), f'Invalid image ID: {mid}')
                    caption, image_id = captions[0], ids[0]
                    actual_image = resolve_image(images[0])
                    visual_ids.add(mid)
                dialogue = 'User: ' + row['user'] + '\nAssistant: ' + row['assistant']
                text = dialogue + ('\nMemory description: ' + caption if visual else '')
                metadata = ['Date: ' + session['date'], 'Round: ' + row['round']]
                if image_id:
                    metadata.append('Image ID: ' + image_id)
                memories.append({
                    'id': mid, 'task': task, 'round': row['round'],
                    'session_id': session['session_id'], 'timestamp': session['date'],
                    'dialogue': dialogue, 'caption': caption, 'retrieval_text': text,
                    'reader_text': '\n'.join([*metadata, text]), 'is_visual': visual,
                    'image_path': str(actual_image) if actual_image else None,
                    'image_id': image_id, 'node_id': mid + '#image1' if visual else None,
                    'image_sha256': image_hashes[actual_image] if actual_image else None,
                    'insertion_index': counts['memories'],
                })
                counts['memories'] += 1
                counts['visual_memories' if visual else 'text_only_memories'] += 1

        for index, qa in enumerate(original['human-annotated QAs']):
            qid = task + f'/QA_{index:04d}'
            query_paths, query_caption = [], ''
            if qa.get('question_image'):
                query_paths = [str(resolve_image(qa['question_image']))]
                query_caption = qa.get('image_caption')
                require(isinstance(query_caption, str) and bool(query_caption.strip()),
                        f'Missing original query-image caption: {qid}')
            retrieval_query = qa['question']
            if query_paths:
                retrieval_query += "\nquestion's image:\nimage_caption: " + query_caption
            questions.append({
                'id': qid, 'task': task, 'question': qa['question'], 'point': qa['point'],
                'query_image_paths': query_paths, 'query_image_caption': query_caption,
                'retrieval_query': retrieval_query,
            })
            gids = [task + '/' + item for item in qa['clue']]
            require(len(gids) == len(set(gids)) and set(gids) <= round_ids,
                    f'Invalid or duplicate gold clue IDs: {qid}')
            gold.append({'id': qid, 'answer': qa['answer'], 'gold_ids': gids, 'point': qa['point']})
            candidate_inputs.append({'id': qid, 'question': qa['question'],
                                     'query_image_caption': query_caption})
            counts['questions'] += 1
            counts['questions_with_image'] += bool(query_paths)
            counts['questions_with_gold' if gids else 'questions_without_gold'] += 1
            counts['gold_links'] += len(gids)
            counts['visual_gold_links'] += sum(item in visual_ids for item in gids)
            counts['nonvisual_gold_links'] += sum(item not in visual_ids for item in gids)
            by_type[qa['point']] += 1
        totals.update(counts)
        scenario_counts.append({'task': task, **dict(counts)})

    qids = [row['id'] for row in questions]
    require(len(qids) == len(set(qids)), 'Duplicate question IDs')
    require(qids == [row['id'] for row in gold] == [row['id'] for row in candidate_inputs],
            'Question/gold/candidate-input alignment differs')
    require(seen_images == set(image_hashes), 'Some manifest images are not referenced by the selected data')
    totals.update(scenarios=len(scenario_counts), image_files=len(image_hashes))
    protocol = {
        'schema_version': 1,
        'dataset': source_manifest['repo'], 'dataset_revision': source_manifest['revision'],
        'source_sha256': dict(sorted(source_hashes.items())),
        'implementation_sha256': sha256(Path(__file__)),
        'counts': dict(totals), 'question_types': dict(sorted(by_type.items())),
        'by_scenario': scenario_counts,
        'policy': {
            'memory_unit': 'Every original dialogue round, including all text-only rounds',
            'history_caption': 'Cached Qwen image_caption_qwen3_vl_8b_instruct, verbatim; empty for text-only rounds',
            'query_caption': 'Original official QA image_caption, verbatim; no regeneration',
            'retrieval_text': 'User and assistant dialogue, then full historical caption when present',
            'reader_text': 'Same original dialogue/caption plus date, round and official image ID metadata',
            'query_text': "question + (when image exists) literal \\nquestion's image:\\nimage_caption: caption",
            'candidate_pool': 'Entire same-scenario history; no QA session_id filtering',
            'insertion_order': 'Source scenario/session/round order; insertion_index is zero-based per scenario over all rounds',
            'sanitization': 'Questions exclude answer, clue and session_id; candidate inputs contain only ID/question/query-image caption',
            'gold_isolation': 'Answers and clue IDs appear only in gold.jsonl; never in candidate generation or retrieval inputs',
            'no_clue_questions': 'Retain all questions including AR; exclude no-clue questions only from supervised retrieval denominators',
            'image_paths': 'Read-only source references for experiment ingestion; these inputs are not persisted graph memory',
            'id_policy': 'Memory task/round; question task/QA_ plus zero-based four-digit index',
        },
    }
    files = {
        'memories.jsonl': jsonl_bytes(memories), 'questions.jsonl': jsonl_bytes(questions),
        'gold.jsonl': jsonl_bytes(gold), 'sanitized_candidate_inputs.jsonl': jsonl_bytes(candidate_inputs),
        'protocol.json': json_bytes(protocol),
    }
    artifact_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files['artifact_manifest.json'] = json_bytes(artifact_hashes)
    # Validate every pre-existing file before writing any missing files.
    for name, content in files.items():
        target = output_dir / name
        require(not target.exists() or target.read_bytes() == content,
                f'Refusing to overwrite a different input artifact: {target}')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = output_dir / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(target)
    return {'output_dir': str(output_dir), 'counts': dict(totals), 'artifact_sha256': artifact_hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--caption-dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, root=args.root, caption_dir=args.caption_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
