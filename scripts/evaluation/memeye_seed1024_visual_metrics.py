"""Measure query-free visual memory storage and complete MCQ context recall.

CLIP exports the same history-only four-file package used by the original
storage audit. HippoMM counts its actual retained representative images and
persisted memory artifacts. Questions, answers, query embeddings, model weights,
logs, and LLM caches are excluded from memory storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
DEFAULT = ROOT / 'results/memeye_qwen_seed1024_full_20261105/visual_methods'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
NATURAL = ROOT / 'results/natural_observation'
MIB = 1024 ** 2
HIPPO_FILES = ('events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl',
               'visual_embeddings.npz', 'representative_images.jsonl')


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def unique(values, field='id'):
    result = {r[field]: r for r in values}
    if len(result) != len(values):
        raise ValueError('Duplicate ' + field)
    return result


def cohort(source, responses_name, questions_name):
    gold = unique(rows(BENCHMARK))
    questions = unique(rows(source / questions_name))
    responses = unique(rows(source / responses_name))
    if len(gold) != 371 or set(questions) != set(gold) or set(responses) != set(gold):
        raise ValueError('Metrics require all 371 unique canonical MCQ questions and responses')
    for key, q in questions.items():
        if q['task'] != gold[key]['task'] or q['question'] != gold[key]['question']:
            raise ValueError('MCQ question identity changed')
        if q.get('options') != gold[key]['options']:
            raise ValueError('MCQ choices changed')
    return gold, questions, responses


def component(path, category):
    path = Path(path).resolve()
    return dict(path=str(path), category=category, bytes=path.stat().st_size, sha256=sha(path))


def clip_metrics(args):
    from scripts.memory.clip_image_dialogue import source_dialogues
    source = args.clip_mcq_source or args.visual_root / 'clip_full/top5/clip_image_dialogue'
    source = source.resolve()
    protocol = read(source / 'protocol.json')
    image_source = Path(protocol['frozen_retrieval_source'])
    source_summary = read(source / 'summary.json')
    gold, questions, responses = cohort(source, 'qa_responses.jsonl', 'qa_manifest.jsonl')
    if source_summary['questions'] != 371:
        raise ValueError('Incomplete CLIP source summary')
    candidates = rows(image_source / 'image_candidates.jsonl')
    with np.load(image_source / 'clip_embeddings.npz', allow_pickle=False) as archive:
        vectors = archive['image_vectors'].copy()
        ids = archive['image_ids'].tolist()
    if vectors.shape != (433, 512) or vectors.dtype != np.float64 or ids != [c['image_id'] for c in candidates]:
        raise ValueError('CLIP history-vector shape/dtype/identity mismatch')
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite CLIP history vectors')
    dialogues, original_sources = source_dialogues(NATURAL, {c['task'] for c in candidates})
    if len(dialogues) != 438:
        raise ValueError('CLIP source-dialogue coverage mismatch')
    output = args.visual_root / 'clip_image_dialogue'
    export = output / 'history_only_memory'
    export.mkdir(parents=True, exist_ok=True)
    # Keep the historical audit's JSON formatting and NPY serialization exactly.
    shutil.copyfile(image_source / 'image_candidates.jsonl', export / 'image_candidates.jsonl')
    np.save(export / 'image_vectors.npy', vectors)
    with (export / 'dialogues.jsonl').open('w') as handle:
        for mid, data in dialogues.items():
            handle.write(json.dumps({'memory_id': mid, **data}, ensure_ascii=False, separators=(',', ':')) + '\n')
    dump(export / 'memory.json', {'method': 'clip_image_dialogue', 'image_count': len(candidates),
        'dialogue_rounds': len(dialogues), 'encoder': 'CLIP ViT-B/32', 'vector_dtype': str(vectors.dtype),
        'image_index': 'image_candidates.jsonl', 'image_vectors': 'image_vectors.npy',
        'dialogues': 'dialogues.jsonl', 'image_policy': 'referenced original files; count each content hash once',
        'export_policy': 'history only; excludes query vectors, query images, QA and predictions'})
    components = [component(p, 'history_vectors' if p.suffix == '.npy' else 'history_index_and_text')
                  for p in sorted(export.iterdir()) if p.is_file()]
    seen = set()
    for candidate in candidates:
        key = candidate['image_id']
        if key in seen:
            continue
        record = component(candidate['path'], 'historical_original_image')
        if record['sha256'] != key:
            raise ValueError('Historical CLIP image content changed')
        seen.add(key)
        components.append(record)
    if len(seen) != 433 or len(components) != 437:
        raise ValueError('Unexpected CLIP storage component coverage')
    hits = total_gold = correct = 0
    for key, q in questions.items():
        if q['gold_ids'] != gold[key]['gold_ids'] or q['gold_choice'] != gold[key]['gold_choice']:
            raise ValueError('CLIP gold provenance changed')
        labels = set(gold[key]['gold_ids'])
        hits += len(labels & set(q['top_memory_ids']))
        total_gold += len(labels)
        response = responses[key]
        if response.get('payload_sha256') != q['payload_sha256']:
            raise ValueError('CLIP response payload provenance differs')
        if response.get('experiment_signature') != protocol['experiment_signature']:
            raise ValueError('CLIP response experiment provenance differs')
        correct += response['predicted_choice'] == gold[key]['gold_choice']
    if correct != round(source_summary['accuracy'] * 371):
        raise ValueError('CLIP accuracy does not reproduce')
    total = sum(c['bytes'] for c in components)
    report = dict(status='complete', method='clip_image_dialogue', questions=371,
        seed=protocol['seed'], correct=correct, accuracy=correct / 371,
        recall_at5_micro=hits / total_gold, recall_numerator=hits, recall_denominator=total_gold,
        recall_metric='Source-round coverage of five retrieved distinct historical images and associated dialogue.',
        storage_bytes=total, storage_mib=total / MIB, storage_unit='MiB = bytes / 1048576',
        storage_policy='Four history-only export files plus 433 content-deduplicated original image files.',
        excluded='Question/query vectors and images, QA/predictions, replay caches, logs, model weights.',
        historical_images=433, historical_dialogue_rounds=438, history_vector_dtype=str(vectors.dtype),
        image_bytes=sum(c['bytes'] for c in components if c['category'] == 'historical_original_image'),
        memory_artifact_bytes=sum(c['bytes'] for c in components if c['category'] != 'historical_original_image'),
        metric_source=str(source), components=components,
        source_sha256={str(p.resolve()): sha(p) for p in [Path(__file__), BENCHMARK,
            source / 'protocol.json', source / 'summary.json', source / 'qa_manifest.jsonl',
            source / 'qa_responses.jsonl', image_source / 'image_candidates.jsonl',
            image_source / 'clip_embeddings.npz', *original_sources]})
    dump(output / 'metric_summary.json', report)
    return report


def hippomm_metrics(args):
    source = (args.hippomm_source or args.visual_root / 'hippomm').resolve()
    offline, online = source / 'offline', source / 'online'
    memory, summary, protocol = (read(p) for p in
        (offline / 'memory.json', online / 'summary.json', online / 'protocol.json'))
    completion = read(online / 'completion_manifest.json')
    if completion['status'] != 'complete' or completion['questions'] != 371 or summary['questions'] != 371:
        raise ValueError('HippoMM metrics require completed 371-question MCQ evaluation')
    gold, questions, responses = cohort(online, 'qa_responses.jsonl', 'questions.jsonl')
    events = unique(rows(offline / 'events.jsonl'), 'event_id')
    images = unique(rows(offline / 'representative_images.jsonl'), 'event_id')
    if set(events) != set(images) or len(events) != memory['counts']['events']:
        raise ValueError('HippoMM representative-image/event identity mismatch')
    components = [component(offline / name, 'persisted_memory_artifact') for name in HIPPO_FILES]
    image_root = (offline / 'images').resolve()
    actual = {p.resolve() for p in image_root.iterdir() if p.is_file()}
    expected = set()
    for key, image in images.items():
        path = (offline / image['file']).resolve()
        if not path.is_relative_to(image_root) or image['task'] != events[key]['task'] or image['image_ref'] != events[key]['representative_image']:
            raise ValueError('HippoMM representative path/task/image identity mismatch')
        if path.stat().st_size != image['bytes']:
            raise ValueError('HippoMM representative image size changed')
        expected.add(path)
        components.append(component(path, 'historical_representative_image'))
    if actual != expected:
        raise ValueError('Unexpected or missing retained HippoMM image files')
    image_bytes = sum(c['bytes'] for c in components if c['category'] == 'historical_representative_image')
    artifact_bytes = sum(c['bytes'] for c in components if c['category'] == 'persisted_memory_artifact')
    total = image_bytes + artifact_bytes
    if (image_bytes != memory['original_image_bytes'] or artifact_bytes != memory['memory_artifact_bytes']
            or total != summary['total_memory_bytes']):
        raise ValueError('HippoMM measured bytes disagree with native artifact accounting')
    hits = total_gold = correct = 0
    for key, response in responses.items():
        if response['task'] != questions[key]['task']:
            raise ValueError('HippoMM response task identity mismatch')
        labels = set(gold[key]['gold_ids'])
        hits += len(labels & set(response['context_round_ids']))
        total_gold += len(labels)
        correct += response['answer'] == gold[key]['gold_choice']
    recall = hits / total_gold
    if correct != round(summary['accuracy'] * 371) or abs(recall - summary['context_source_micro_recall']) > 1e-12:
        raise ValueError('HippoMM source accuracy/context recall does not reproduce')
    report = dict(status='complete', method='hippomm', questions=371,
        seed=protocol['contract'].get('seed', 42), correct=correct, accuracy=correct / 371,
        recall_at5_micro=recall, context_source_micro_recall=recall,
        recall_numerator=hits, recall_denominator=total_gold,
        recall_metric='Context source-round coverage; fast summaries include all same-task summary provenance; detailed retrieval selects at most five events.',
        recall_is_literal_top5=False,
        storage_bytes=total, storage_mib=total / MIB, storage_unit='MiB = bytes / 1048576',
        storage_decimal_mb=total / 1000000,
        storage_policy='Retained representative original image files plus five native persisted memory artifacts.',
        excluded='Question/query images, QA/predictions, original ingestion caches, LLM replay caches, logs, model weights.',
        seed42_user_storage_value=168.82, seed42_user_storage_actual_unit='decimal MB',
        seed42_storage_correction='User table 168.82 corresponds to 168822299 / 1000000 MB; the same bytes are 161.00149059295654 MiB.',
        historical_images=len(images), events=len(events), image_bytes=image_bytes,
        memory_artifact_bytes=artifact_bytes, metric_source=str(online), components=components,
        source_sha256={str(p.resolve()): sha(p) for p in [Path(__file__), BENCHMARK,
            offline / 'memory.json', online / 'protocol.json', online / 'completion_manifest.json',
            online / 'summary.json', online / 'questions.jsonl', online / 'qa_responses.jsonl']})
    dump(args.visual_root / 'hippomm/metric_summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('clip_image_dialogue', 'hippomm', 'all'), required=True)
    parser.add_argument('--phase', choices=('measure',), required=True)
    parser.add_argument('--visual-root', type=Path, default=DEFAULT)
    parser.add_argument('--clip-mcq-source', type=Path)
    parser.add_argument('--hippomm-source', type=Path)
    args = parser.parse_args()
    args.visual_root = args.visual_root.resolve()
    for method in ('clip_image_dialogue', 'hippomm') if args.method == 'all' else (args.method,):
        report = (clip_metrics if method == 'clip_image_dialogue' else hippomm_metrics)(args)
        print(json.dumps({k: report[k] for k in ('method', 'status', 'questions', 'correct',
            'recall_numerator', 'recall_denominator', 'storage_bytes', 'storage_mib')}), flush=True)


if __name__ == '__main__':
    main()
