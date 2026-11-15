"""Measure fresh DeepSeek seed1024 memories with the audited seed42 policies.

All storage figures are logical bytes, not RAM or VRAM.  SimpleMem attribution
uses the original local Qwen seed42 evaluator on NEW memory entries.  Its default
flow stops at a gold-blind review queue; no review decisions are manufactured.
This module makes no HTTP requests and never changes a historical experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
REFERENCE = ROOT / 'reports/memory_storage_deepseek_memeye_20261102'
OLD_FULL = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
OLD_TEXT = ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025'
OLD_SIMPLE = ROOT / 'results/simplemem_official_deepseek_memeye_k10_rrf5_reflection_20261029'
OLD_HIP = ROOT / 'results/hippomm_api_memeye_20261104/deepseek'
EVALUATOR = ROOT / 'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/source_attribution_20261028'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
MODEL, SEED, MIB = 'deepseek-v4.1-flash', 1024, 1048576
OURS = {
    'semrag': ('Semantic RAG (baseline)', 'baseline_question_only', 'natural'),
    'ours_1': ('ours_1', 'master_answer_query', 'caption_visual_master'),
    'ours_2': ('ours_2 (z-score)', 'baseline_question_only_zscore', 'natural'),
    'ours_1_ours_2': ('ours_1 + ours_2', 'master_answer_query_zscore', 'caption_visual_master'),
}
TEXT = {'gen_agents': 'Generative Agents (Top10)', 'memoryos': 'MemoryOS',
        'simplemem': 'SimpleMem (official, context 5)', 'a_mem': 'A-Mem (context 5)'}
METHODS = (*OURS, 'gen_agents', 'memoryos', 'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm')
HIP_FILES = ('events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl',
             'visual_embeddings.npz', 'representative_images.jsonl', 'caption_provenance.jsonl')
STORAGE_POLICY = {
    'gen_agents': 'Saved final_memory.json snapshots once; unpersisted vectors and importance/recency tensors excluded.',
    'a_mem': 'Saved final_memory.json snapshots once; unpersisted retrieval vectors excluded.',
    'memoryos': 'Exactly four native runtime JSON layers per task including decimal embedding arrays.',
    'simplemem': 'All native lancedb files including retained versions/FTS plus resolved_config.json; duplicate final_memory export excluded.',
}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    from scripts.memory.reflexion_memory import hash_object
    return hash_object(value)


def compact_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def unique(values, key='id'):
    result = {r[key]: r for r in values}
    if len(result) != len(values):
        raise ValueError('Duplicate identity: ' + key)
    return result


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha(name) != expected:
            raise ValueError('Artifact changed: ' + name)


def components(files):
    paths = [Path(f).resolve() for f in files]
    if len(paths) != len(set(paths)):
        raise ValueError('A memory file was counted twice')
    return [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths]


def cohort(values, prediction='predicted_choice'):
    canonical, found = unique(rows(BENCHMARK)), unique(values)
    if len(canonical) != 371 or set(found) != set(canonical):
        raise ValueError('Expected all 371 unique canonical MCQ identities')
    for identity, row in found.items():
        for key in ('task', 'question', 'options', 'gold_choice', 'gold_ids'):
            if key in row and row[key] != canonical[identity][key]:
                raise ValueError('Canonical question changed: ' + identity + '/' + key)
    correct = sum(r.get(prediction) == canonical[i]['gold_choice'] for i, r in found.items())
    return canonical, found, correct


def metric_path(method, output):
    group = 'ours' if method in OURS else 'text_methods' if method in TEXT else 'visual_methods'
    return Path(output) / group / method / 'metric_summary.json'


def reference_mapping():
    historical = {r['method']: r for r in read(REFERENCE / 'storage_summary.json')['rows']}
    mapped = {}
    for method in METHODS:
        if method in OURS:
            label, condition, _ = OURS[method]
            item = historical[label]
            source = OLD_FULL
        elif method in TEXT:
            item = historical[TEXT[method]]
            source = OLD_SIMPLE / 'top5/simplemem_official_text_context5' if method == 'simplemem' else OLD_TEXT / method
        elif method == 'clip_image_dialogue':
            item = historical['CLIP → 原图＋历史对话']
            source = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue_api/deepseek-v4.1-flash'
        else:
            item = read(OLD_HIP / 'online/summary.json')
            source = OLD_HIP
            item = dict(item, storage_bytes=item['total_memory_bytes'], memory_count=item['representative_images'],
                        recall_at5_micro=None, context_source_micro_recall=item['context_source_micro_recall'])
        mapped[method] = {k: item.get(k) for k in ('correct', 'questions', 'accuracy', 'storage_bytes',
                                                  'memory_count', 'recall_at5_micro')}
        mapped[method].update(source_root=str(source), storage_mib=item['storage_bytes'] / MIB,
                              model=MODEL, seed=42, reference_report=str(REFERENCE / 'storage_summary.json'))
        mapped[method]['correct'] = int(mapped[method]['correct'])
        if method in OURS:
            mapped[method]['condition'] = OURS[method][1]
        if method == 'hippomm':
            mapped[method].update(context_source_micro_recall=item['context_source_micro_recall'],
                recall_is_literal_top5=False, reference_report=str(OLD_HIP / 'online/summary.json'))
    return mapped


def text_storage(method, source, protocol):
    files, count = [], 0
    for task in protocol['tasks']:
        work = source / task
        if method in ('gen_agents', 'a_mem'):
            files.append(work / 'final_memory.json')
            count += len(read(files[-1]))
        elif method == 'memoryos':
            group = sorted((work / 'runtime').rglob('*.json'))
            expected = {'short_term.json', 'mid_term.json', 'long_term_user.json', 'long_term_assistant.json'}
            if len(group) != 4 or {p.name for p in group} != expected:
                raise ValueError('MemoryOS must persist exactly four native layers per task')
            files.extend(group)
        elif method == 'simplemem':
            group = sorted(p for p in (work / 'runtime/lancedb').rglob('*') if p.is_file())
            if not group:
                raise ValueError('Missing SimpleMem native database')
            files.extend([*group, work / 'runtime/resolved_config.json'])
            count += len(read(work / 'final_memory.json'))
        else:
            raise ValueError(method)
    return files, count if method != 'memoryos' else None


def export_ours(source, destination, *, historical=False):
    """Use the original four-field leaf/manifest schema, including provenance bytes."""
    import numpy as np
    if historical:
        memories = {r['image_id']: r for r in rows(source / 'inputs/source_memories.jsonl')}
        metadata = read(source / 'inputs/protocol.json')
        captions = {r['id']: r['caption'] for r in rows(source / 'captions.jsonl')}
        encoding = read(source / 'retrieval_protocol.json')['encoding_audits']
        provenance = [source / n for n in ('inputs/protocol.json', 'inputs/source_memories.jsonl', 'captions.jsonl')]
    else:
        memories = {r['image_id']: r for r in rows(source / 'inputs/memories.jsonl')}
        metadata = read(source / 'inputs/encoding.json')
        captions = {r['image_sha256']: r['caption'] for r in memories.values()}
        encoding = read(source / 'retrieval_audit.json')['encoding']
        provenance = [source / n for n in ('inputs/encoding.json', 'inputs/memories.jsonl')]
        provenance.append(OLD_FULL / 'captions.jsonl')
    encodings = {r['task']: r['leaf'] for r in encoding}
    stores, leaves_total, masters_total = {}, 0, 0
    for domain in sorted((source / 'domains').iterdir()):
        if not domain.is_dir():
            continue
        graph_path, sums_path = domain / 'memory/graph.json', domain / 'memory/master_sums.npy'
        graph, build = read(graph_path), read(domain / 'build_protocol.json')
        verify_hashes({str(p): build['artifact_sha256'][str(p)] for p in (graph_path, sums_path)})
        leaves = []
        for node in graph['image_nodes']:
            original = memories[node['id']]
            if node['dialogue'] != original['dialogue'] or node['caption'] != captions[original['image_sha256']]:
                raise ValueError('Persisted leaf differs from full dialogue/caption')
            leaves.append(dict(image_id=node['id'], memory_id=original['id'], dialogue=node['dialogue'], caption=node['caption']))
        sums = np.load(sums_path, allow_pickle=False)
        if sums.shape != (len(graph['masters']), 512) or sums.dtype != np.float64:
            raise ValueError('Unexpected master-vector representation')
        leaves_total += len(leaves)
        masters_total += len(graph['masters'])
        for kind in ('natural', 'caption_visual_master'):
            dest = destination / domain.name / kind
            dump(dest / 'leaves.json', leaves)
            files = {'leaves': 'leaves.json'}
            config = {k: metadata[k] for k in ('text_encoder', 'query_prefix')}
            config['allowed_leaf_truncations'] = [dict(id=leaf['image_id'],
                original_tokens=record['original_tokens'], retained_limit=record['retained_limit'],
                text_sha256=hashlib.sha256((leaf['dialogue'] + '\nMemory description: ' + leaf['caption']).encode()).hexdigest())
                for record in encodings[domain.name]['affected_inputs'] for leaf in leaves
                if record['id'] in (leaf['memory_id'], leaf['image_id'])]
            sources = {str(p): sha(p) for p in provenance}
            if kind == 'caption_visual_master':
                for key, path in (('graph', graph_path), ('visual_sums', sums_path)):
                    shutil.copyfile(path, dest / path.name)
                    files[key] = path.name
                    sources[str(path)] = sha(path)
                config['image_encoder'] = metadata['image_encoder']
            dump(dest / 'memory.json', dict(schema_version=1, offline_method=kind, domain=domain.name,
                leaf_count=len(leaves), leaf_text_format='dialogue + newline + Memory description: + full caption',
                files={k: dict(path=name, sha256=sha(dest / name)) for k, name in files.items()},
                encoding=config, source_sha256=sources,
                export_policy='existing DeepSeek Flash offline memory; no new clustering; no QA data included'))
            stores.setdefault(kind, []).append(dest)
    if leaves_total != 438 or len(stores.get('natural', [])) != 8:
        raise ValueError('Expected eight domains and 438 full caption/dialogue leaves')
    return stores, leaves_total, masters_total


def reference_validation(output=OUT):
    """Re-measure historical fixtures read-only; original exports are never rewritten."""
    reference = reference_mapping()
    with (REFERENCE / 'storage_files.csv').open(encoding='utf-8-sig') as stream:
        old_components = list(csv.DictReader(stream))
    for item in old_components:
        if Path(item['path']).stat().st_size != int(item['bytes']) or sha(item['path']) != item['sha256']:
            raise ValueError('Historical audited storage fixture changed')
    checks = []
    for method, label in TEXT.items():
        source = Path(reference[method]['source_root'])
        files, count = text_storage(method, source, read(source / 'protocol.json'))
        measured = sum(p.stat().st_size for p in files)
        expected_paths = {r['path'] for r in old_components if r['method'] == label}
        if measured != reference[method]['storage_bytes'] or {str(p.resolve()) for p in files} != expected_paths:
            raise ValueError('Historical native storage policy mismatch: ' + method)
        if count != reference[method]['memory_count']:
            raise ValueError('Historical memory count mismatch: ' + method)
        checks.append(dict(method=method, storage_bytes=measured, files=len(files), passed=True))
    with tempfile.TemporaryDirectory(prefix='deepseek_storage_fixture_') as tmp:
        stores, _, masters = export_ours(OLD_FULL, Path(tmp), historical=True)
        for method, (label, _, kind) in OURS.items():
            files = [p for d in stores[kind] for p in d.iterdir() if p.is_file()]
            original = {Path(r['path']).relative_to(REFERENCE / 'memory_stores'): r
                        for r in old_components if r['method'] == label}
            for p in files:
                record = original[p.relative_to(Path(tmp))]
                if p.stat().st_size != int(record['bytes']) or sha(p) != record['sha256']:
                    raise ValueError('Original export byte parity failed: ' + str(p))
            measured = sum(p.stat().st_size for p in files)
            if measured != reference[method]['storage_bytes']:
                raise ValueError('Original export size parity failed')
            checks.append(dict(method=method, storage_bytes=measured, files=len(files), masters=masters, passed=True))
    clip_files = [r for r in old_components if r['method'] == 'CLIP → 原图＋历史对话']
    if sum(int(r['bytes']) for r in clip_files) != reference['clip_image_dialogue']['storage_bytes']:
        raise ValueError('CLIP reference accounting mismatch')
    checks.append(dict(method='clip_image_dialogue', storage_bytes=reference['clip_image_dialogue']['storage_bytes'],
                       files=len(clip_files), passed=True))
    hip_files, count = hip_storage(OLD_HIP / 'offline')
    if sum(p.stat().st_size for p in hip_files) != reference['hippomm']['storage_bytes']:
        raise ValueError('HippoMM reference accounting mismatch')
    checks.append(dict(method='hippomm', storage_bytes=reference['hippomm']['storage_bytes'], files=len(hip_files), passed=True))
    report = dict(status='passed', passed=True, reference_methods=reference, checks=checks,
        historical_component_hashes_verified=len(old_components), historical_experiments_modified=False,
        export_fixture_byte_parity=True, measurement_code_sha256=sha(__file__),
        reference_sha256={str(REFERENCE / name): sha(REFERENCE / name)
                          for name in ('storage_summary.json', 'storage_files.csv', 'storage_audit.json')})
    dump(Path(output) / 'metrics/reference_policy_validation.json', report)
    return report


def completed_text(method, output):
    from scripts.evaluation import memeye_deepseek_seed1024_text_methods as run
    source = Path(output) / 'text_methods' / method
    if not (source / 'completion_manifest.json').exists():
        return source, None, None
    p, complete = run.verify(source), read(source / 'completion_manifest.json')
    if complete.get('status') != 'complete' or complete.get('passed') is not True:
        raise ValueError('Text completion audit did not pass')
    verify_hashes({str(source / name): complete[key] for name, key in (
        ('protocol.json', 'protocol_sha256'), ('qa_responses.jsonl', 'responses_sha256'),
        ('stability_audit.json', 'audit_sha256'))})
    audit = read(source / 'stability_audit.json')
    if not audit['passed'] or audit['questions'] != 371 or audit['experiment_signature'] != p['experiment_signature']:
        raise ValueError('Text source audit mismatch')
    _, _, correct = cohort(rows(source / 'qa_responses.jsonl'))
    for row in rows(source / 'qa_responses.jsonl'):
        raw = row['raw_response'].strip()
        prediction = raw if raw in ('A', 'B', 'C', 'D') and not row['hit_token_limit'] and not row.get('refusal') else None
        if prediction != row['predicted_choice']:
            raise ValueError('Text MCQ raw output does not reproduce its native grade')
    if correct != audit['correct']:
        raise ValueError('Text accuracy does not reproduce')
    return source, p, rows(source / 'qa_responses.jsonl')


def a_mem_recall(source, protocol, values):
    canonical, _, _ = cohort(values)
    notes = {task: read(source / task / 'final_memory.json') for task in protocol['tasks']}
    evaluated = []
    for row in values:
        identities = row['retrieval_detail']['context_memory_ids']
        if len(identities) > 5 or len(identities) != len(set(identities)):
            raise ValueError('A-Mem native context exceeds five distinct whole notes')
        selected = [notes[row['task']][i] for i in identities]
        rendered = ''.join('talk start time:' + n['timestamp'] + 'memory content: ' + n['content']
            + 'memory context: ' + n['context'] + 'memory keywords: ' + str(n['keywords'])
            + 'memory tags: ' + str(n['tags']) + '\n' for n in selected)
        if rendered != row['retrieved_context']:
            raise ValueError('A-Mem source-round mapping does not reconstruct the actual reader context')
        rounds = {row['task'] + '/' + identity for identity in identities}
        gold = set(canonical[row['id']]['gold_ids'])
        evaluated.append(dict(id=row['id'], task=row['task'], context_entries=len(identities),
            retrieved_round_ids=sorted(rounds), gold_ids=sorted(gold), hit_ids=sorted(rounds & gold),
            gold_count=len(gold), hit_count=len(rounds & gold)))
    denominator, hits = sum(r['gold_count'] for r in evaluated), sum(r['hit_count'] for r in evaluated)
    if denominator != 1041:
        raise ValueError('Expected 1041 canonical gold-round references')
    report = dict(questions=371, gold_round_hits=hits, gold_round_references=denominator,
        recall_at5_micro=hits / denominator, context_entry_counts=dict(Counter(r['context_entries'] for r in evaluated)),
        formula='sum_i |final context source rounds intersect gold_i| / sum_i |gold_i|',
        context_policy='Native up-to-five whole notes, including preserved short or empty contexts.',
        responses_sha256=sha(source / 'qa_responses.jsonl'), canonical_sha256=sha(BENCHMARK))
    write_rows(source / 'source_recall_at5_per_question.jsonl', evaluated)
    dump(source / 'source_recall_at5.json', report)
    return report


def hip_storage(offline):
    memory = read(offline / 'memory.json')
    events, images = unique(rows(offline / 'events.jsonl'), 'event_id'), unique(rows(offline / 'representative_images.jsonl'), 'event_id')
    if set(events) != set(images) or len(events) != memory['events']:
        raise ValueError('HippoMM event/representative coverage mismatch')
    files = [offline / name for name in HIP_FILES]
    image_root, expected = (offline / 'images').resolve(), set()
    for identity, image in images.items():
        path = (offline / image['file']).resolve()
        if (not path.is_relative_to(image_root) or image['task'] != events[identity]['task']
                or image['image_ref'] != events[identity]['representative_image']
                or path.stat().st_size != image['bytes']):
            raise ValueError('HippoMM representative image provenance changed')
        expected.add(path)
        files.append(path)
    if {p.resolve() for p in image_root.iterdir() if p.is_file()} != expected:
        raise ValueError('Unexpected retained HippoMM images')
    image_bytes = sum(p.stat().st_size for p in expected)
    artifact_bytes = sum((offline / n).stat().st_size for n in HIP_FILES)
    if image_bytes != memory['original_image_bytes'] or artifact_bytes != memory['memory_artifact_bytes']:
        raise ValueError('HippoMM storage differs from its actual native artifact accounting')
    return files, len(images)


def measure(method, output=OUT, attribution_root=None):
    output = Path(output)
    attribution_root = Path(attribution_root or output / 'metrics/simplemem_source_attribution')
    reference = reference_mapping()[method]
    report = dict(method=method, seed=SEED, model=MODEL, status='awaiting_complete_source',
        reference=reference, storage_unit='MiB = logical file bytes / 1048576',
        storage_excluded='Weights, code, questions, query images/vectors, answers, inference caches, logs and provenance annotations.',
        measurement_code_sha256=sha(__file__), reference_policy_source=str(ROOT / 'scripts/analysis/audit_deepseek_memeye_storage.py'))
    files, memory_count = [], None
    if method in TEXT:
        source, p, values = completed_text(method, output)
        if p is None:
            dump(metric_path(method, output), report)
            return report
        files, memory_count = text_storage(method, source, p)
        _, _, correct = cohort(values)
        summary = read(source / 'summary.json')
        report.update(questions=371, correct=correct, accuracy=correct / 371,
            memory_token_limit_hits=summary['memory_token_limit_hits'],
            answer_token_limit_hits=summary['answer_token_limit_hits'],
            invalid_structured_outputs=summary['invalid_structured_outputs'], format_errors=summary['format_errors'],
            source_protocol_sha256=sha(source / 'protocol.json'), source_responses_sha256=sha(source / 'qa_responses.jsonl'),
            storage_policy=STORAGE_POLICY[method], recall_at5_micro=None, status='complete')
        if method == 'a_mem':
            recall = a_mem_recall(source, p, values)
            report.update(recall_at5_micro=recall['recall_at5_micro'], gold_round_hits=recall['gold_round_hits'],
                          gold_round_references=1041, recall_status='complete', context_entry_counts=recall['context_entry_counts'])
        elif method == 'simplemem':
            rp, lp = attribution_root / 'recall_summary.json', attribution_root / 'annotation_lock.json'
            if rp.exists() and lp.exists():
                from scripts.analysis.simplemem_source_attribution import verify_inputs
                ap = read(attribution_root / 'protocol.json')
                verify_inputs(ap, attribution_root)
                recall = read(rp)
                if (ap['seed'] != 42 or Path(ap['source_root']).resolve() != source.resolve()
                        or recall['source_experiment_signature'] != p['experiment_signature']
                        or recall['responses_sha256'] != report['source_responses_sha256']
                        or recall['annotation_lock_sha256'] != sha(lp)):
                    raise ValueError('SimpleMem annotation seal belongs to a different source run')
                report.update(recall_at5_micro=recall['recall_at5_micro'], gold_round_hits=recall['gold_round_hits'],
                    gold_round_references=recall['gold_round_references'], recall_status='complete_conservative_posthoc_attribution',
                    attribution_lock_sha256=sha(lp), recall_summary_sha256=sha(rp), attribution_evaluator_seed=42,
                    attribution_limitations=recall['limitations'])
            else:
                report.update(status='accuracy_storage_complete_attribution_pending',
                              recall_status='pending_fresh_gold_blind_evidence_annotation_and_review')
        else:
            report['recall_status'] = 'not_reported_at5_native_context_top10_or_dynamic'
    elif method in OURS:
        source = output / 'ours'
        if not (source / 'memory_manifest.json').exists() or not (source / 'retrieval_audit.json').exists():
            dump(metric_path(method, output), report)
            return report
        build, retrieval = read(source / 'memory_manifest.json'), read(source / 'retrieval_audit.json')
        protocol = read(source / 'protocol.json')
        contract = protocol['contract']
        if (compact_digest(contract) != protocol['experiment_signature']
                or contract['model'] != MODEL or contract['seed'] != SEED
                or build['experiment_signature'] != protocol['experiment_signature']
                or retrieval['source_signature'] != protocol['experiment_signature']):
            raise ValueError('Fresh ours build/retrieval seed or signature binding differs')
        verify_hashes({str(source / 'inputs' / name): contract['source_sha256'][str(source / 'inputs' / name)]
                       for name in ('memories.jsonl', 'questions.jsonl', 'gold.jsonl', 'encoding.json')})
        if not build['completed'] or not retrieval['passed']:
            raise ValueError('Fresh build/retrieval did not pass')
        verify_hashes(build['output_sha256']); verify_hashes(retrieval['output_sha256'])
        stores, memory_count, masters = export_ours(source, source / 'memory_stores')
        label, condition, kind = OURS[method]
        files = [p for d in stores[kind] for p in sorted(d.iterdir()) if p.is_file()]
        canonical = unique(rows(BENCHMARK))
        retrieved = unique(rows(source / 'retrieval_results.jsonl'))
        if set(retrieved) != set(canonical):
            raise ValueError('Ours retrieved cohort is incomplete')
        selected = {identity: next(c for c in r['conditions'] if c['name'] == condition)
                    for identity, r in retrieved.items()}
        if any(len(c['top5_memory_ids']) != 5 or len(set(c['top5_memory_ids'])) != 5 for c in selected.values()):
            raise ValueError('Ours final Top5 source identities are incomplete or duplicated')
        hits = sum(len(set(c['top5_memory_ids']) & set(canonical[i]['gold_ids'])) for i, c in selected.items())
        if hits != retrieval['metrics'][condition]['gold_hits_top5']:
            raise ValueError('Ours recall does not reproduce from actual selected source IDs')
        payloads = [p for p in files if p.name != 'memory.json']
        comparisons = [sha(p) == sha(REFERENCE / 'memory_stores' / p.relative_to(source / 'memory_stores')) for p in payloads]
        report.update(label=label, condition=condition, memory_count=memory_count, masters=masters if kind != 'natural' else None,
            storage_policy='Eight online-schema packages: memory.json plus leaves.json; master methods also graph.json and float64 master_sums.npy. Full duplicate graph/leaf serialization counted; BGE/basis arrays excluded.',
            recall_status='complete', recall_at5_micro=retrieval['metrics'][condition]['recall_at5_micro'],
            gold_round_hits=hits, gold_round_references=1041,
            payload_files_compared_to_seed42=len(comparisons), payload_files_byte_identical_to_seed42=sum(comparisons),
            memory_payload_bytes=sum(p.stat().st_size for p in payloads),
            storage_serialization_policy='Provenance paths and export-policy text remain in counted memory.json bytes; a manifest-only size delta does not indicate changed memory retention.')
        if (source / 'summary.json').exists():
            summary, audit = read(source / 'summary.json'), read(source / 'independent_audit.json')
            if summary['status'] != 'complete' or not audit['passed']:
                raise ValueError('Ours full MCQ audit did not pass')
            verify_hashes(audit['artifact_sha256'])
            values = [dict(r, id=r['question_id']) for r in rows(source / 'reader/responses.jsonl') if r['condition'] == condition]
            for r in values:
                raw = r.get('raw_response')
                match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
                if r['predicted_choice'] != (match.group(1).upper() if match else None):
                    raise ValueError('Ours raw MCQ output does not reproduce its native grade')
            _, _, correct = cohort(values)
            if correct != summary['overall'][condition]['correct']:
                raise ValueError('Ours accuracy does not reproduce')
            report.update(status='complete', questions=371, correct=correct, accuracy=correct / 371)
        report['source_protocol_sha256'] = sha(source / 'protocol.json')
    elif method == 'clip_image_dialogue':
        source = output / 'visual_methods/clip_image_dialogue/online'
        if not (source / 'summary.json').exists():
            dump(metric_path(method, output), report)
            return report
        from scripts.evaluation import memeye_seed1024_visual_metrics as shared
        from types import SimpleNamespace
        protocol = read(source / 'protocol.json')
        if (protocol['seed'] != SEED or protocol['answer_model'] != MODEL
                or compact_digest({k: v for k, v in protocol.items() if k != 'experiment_signature'}) != protocol['experiment_signature']):
            raise ValueError('CLIP answer model/seed/signature differs')
        report.update(shared.clip_metrics(SimpleNamespace(visual_root=output / 'visual_methods', clip_mcq_source=source)))
        memory_count = report['historical_images']
        files = [Path(c['path']) for c in report.pop('components')]
        report.update(model=MODEL, seed=SEED, reference=reference)
    else:
        source = output / 'visual_methods/hippomm'
        offline, online = source / 'offline', source / 'online'
        if not (online / 'completion_manifest.json').exists():
            dump(metric_path(method, output), report)
            return report
        completion, summary = read(online / 'completion_manifest.json'), read(online / 'summary.json')
        protocol = read(online / 'protocol.json')['contract']
        if protocol['seed'] != SEED or protocol['model'] != MODEL or summary['model'] != MODEL:
            raise ValueError('HippoMM answer model/seed differs')
        if completion['status'] != 'complete' or completion['questions'] != 371 or summary['questions'] != 371:
            raise ValueError('HippoMM MCQ is incomplete')
        values = rows(online / 'qa_responses.jsonl')
        canonical, _, correct = cohort(values, 'answer')
        questions = unique(rows(online / 'questions.jsonl'))
        if set(questions) != set(canonical):
            raise ValueError('HippoMM source question cohort mismatch')
        for identity, q in questions.items():
            if any(q[k] != canonical[identity][k] for k in ('task', 'question', 'options')):
                raise ValueError('HippoMM fixed question/options changed')
        files, memory_count = hip_storage(offline)
        hits = sum(len(set(r['context_round_ids']) & set(canonical[r['id']]['gold_ids'])) for r in values)
        recall = hits / 1041
        if correct != summary['correct'] or abs(recall - summary['context_source_micro_recall']) > 1e-12:
            raise ValueError('HippoMM accuracy/context recall does not reproduce')
        report.update(status='complete', questions=371, correct=correct, accuracy=correct / 371,
            recall_at5_micro=None, context_source_micro_recall=recall, recall_is_literal_top5=False,
            gold_round_hits=hits, gold_round_references=1041,
            recall_status='complete_context_source_coverage_not_literal_top5',
            storage_policy='Retained representative image files plus six native artifacts, including provider-caption provenance; memory.json metadata excluded as in original DeepSeek42 accounting.',
            source_protocol_sha256=sha(online / 'protocol.json'), source_responses_sha256=sha(online / 'qa_responses.jsonl'))
    measured = components(files)
    total = sum(c['bytes'] for c in measured)
    report.update(storage_bytes=total, storage_mib=total / MIB, memory_count=memory_count, storage_components=measured,
                  metric_source=str(source), storage_delta_bytes_vs_seed42=total - reference['storage_bytes'])
    if method == 'hippomm' and total != summary['total_memory_bytes']:
        raise ValueError('HippoMM complete summary size differs from measured retained memory')
    dump(metric_path(method, output), report)
    return report


def attribution(stage='prepare', output=OUT, attribution_root=None, evaluator=EVALUATOR, device='cuda:1', model=None):
    from scripts.evaluation import memeye_deepseek_seed1024_text_methods as run
    from scripts.evaluation.memeye_seed1024_text_metrics import fusion_audit
    from scripts.analysis import simplemem_deepseek_source_attribution as adapter
    from scripts.analysis import simplemem_source_attribution as annotation
    from scripts.analysis import simplemem_attribution_recall as recall
    source, p, values = completed_text('simplemem', output)
    if p is None:
        return dict(method='simplemem', status='awaiting_complete_source')
    root = Path(attribution_root or Path(output) / 'metrics/simplemem_source_attribution')
    if stage in ('prepare', 'all') and not (root / 'protocol.json').exists():
        # The shared DeepSeek prepare adapter requires this historical audit
        # schema. Reconstruct it truthfully from this run's native call ledger.
        calls, failures, settings = run.validate_calls(source, p)
        if failures:
            raise ValueError('Actual seed1024 call audit failed before attribution')
        fusion = fusion_audit(source, save=False)
        audit = dict(passed=True, complete=True, questions=371, experiment_signature=p['experiment_signature'],
            seed=SEED, actual_api_calls=len(calls), source_generation='fresh_seed1024',
            final_context_audit=fusion, actual_wire_model_seed_join_passed=True,
            source_protocol_sha256=sha(source / 'protocol.json'), source_responses_sha256=sha(source / 'qa_responses.jsonl'),
            audit_code_sha256=sha(__file__))
        audit_path = source / 'backbone_audit.json'
        if audit_path.exists() and read(audit_path) != audit:
            raise ValueError('Existing attribution source audit changed')
        dump(audit_path, audit)
        evaluation = read(Path(evaluator) / 'protocol.json')
        if evaluation['seed'] != 42:
            raise ValueError('Matched local Qwen evaluator must use seed42')
        adapter.prepare(source, root, Path(evaluator))
        ap = read(root / 'protocol.json')
        ap.update(reader_device_map={'': int(device.split(':')[1])}, seed=42,
            evaluation_scope='Fresh DeepSeek seed1024 memories/contexts; original fixed local Qwen42 attribution evaluator.',
            memory_generation_seed=SEED, evaluator_seed_policy='fixed seed42 independently of construction seed1024',
            fresh_source_only=True)
        ap['input_hashes'][str(Path(__file__).resolve())] = sha(__file__)
        ap['signature'] = digest({k: v for k, v in ap.items() if k != 'signature'})
        dump(root / 'protocol.json', ap)
        shutil.copyfile(__file__, root / 'stability_source_snapshot.py')
        jobs = rows(root / 'jobs.jsonl')
        if any(set(j) != {'id', 'task', 'window_index', 'sources', 'targets'} for j in jobs):
            raise ValueError('Attribution jobs contain non-gold-blind top-level fields')
        actual_ids = {entry['entry_id'] for task in p['tasks'] for entry in read(source / task / 'final_memory.json')}
        if not set(read(root / 'entry_index.json')).issubset(actual_ids):
            raise ValueError('Attribution contains memories outside the fresh source store')
        dump(root / 'fresh_source_audit.json', dict(status='passed', passed=True, gold_blind=True,
            construction_seed=SEED, evaluation_seed=42, selected_entries=ap['selected_entries'],
            annotation_entries=ap['annotation_entries'], jobs_per_pass=len(jobs), historical_annotations_reused=False,
            source_root=str(source.resolve()), protocol_sha256=sha(root / 'protocol.json'),
            jobs_sha256=sha(root / 'jobs.jsonl'), entry_index_sha256=sha(root / 'entry_index.json')))
    ap = read(root / 'protocol.json')
    annotation.verify_inputs(ap, root)
    if ap['seed'] != 42 or ap.get('memory_generation_seed') != 1024 or Path(ap['source_root']).resolve() != source.resolve():
        raise ValueError('Wrong attribution evaluator/source seed binding')
    if stage in ('run', 'all'):
        factory = annotation.LocalQwen
        try:
            if model is not None:
                for key in ('reader_model_path', 'dtype', 'context_budget'):
                    if model.protocol[key] != ap[key]:
                        raise ValueError('Injected local evaluator differs: ' + key)
                model.protocol = ap
                annotation.LocalQwen = lambda protocol: model
            annotation.run(root)
        finally:
            annotation.LocalQwen = factory
        recall.review(root)
    elif stage == 'review':
        recall.review(root)
    elif stage == 'freeze':
        recall.freeze(root)  # Requires genuine evidence-bearing review for every flagged entry.
    elif stage == 'score':
        recall.score(root)
        measure('simplemem', output, root)
    progress = read(root / ('review_status.json' if (root / 'review_status.json').exists() else 'progress.json'))
    status = ('complete' if (root / 'recall_summary.json').exists() else
              'sealed_before_scoring' if (root / 'annotation_lock.json').exists() else
              'awaiting_explicit_review' if progress.get('review_entries') else
              'ready_to_freeze' if (root / 'review_status.json').exists() else 'prepared')
    result = dict(progress, method='simplemem', stage=stage, status=status, seed=42, memory_seed=1024, attribution_root=str(root))
    dump(root / 'stage_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure', 'reference', 'attribution'), required=True)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--attribution-root', type=Path)
    parser.add_argument('--evaluator-control', type=Path, default=EVALUATOR)
    parser.add_argument('--attribution-stage', choices=('prepare', 'run', 'review', 'freeze', 'score', 'all'), default='prepare')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.phase == 'reference':
        report = reference_validation(args.output_root)
        print(json.dumps({k: report[k] for k in ('status', 'passed', 'historical_component_hashes_verified')}), flush=True)
    elif args.phase == 'attribution':
        if args.method not in ('simplemem', 'all'):
            raise ValueError('Source attribution applies only to official SimpleMem')
        print(json.dumps(attribution(args.attribution_stage, args.output_root, args.attribution_root,
                                     args.evaluator_control, args.device)), flush=True)
    else:
        results = []
        for method in METHODS if args.method == 'all' else (args.method,):
            try:
                result = measure(method, args.output_root, args.attribution_root)
            except Exception as error:
                result = dict(method=method, status='failed', error_type=type(error).__name__, error=str(error))
                dump(metric_path(method, args.output_root), result)
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ('storage_components', 'source_sha256', 'reference')}), flush=True)
        report = dict(status='failed' if any(r['status'] == 'failed' for r in results) else
                      'complete' if all(r['status'] == 'complete' for r in results) else 'partial',
                      mcq_storage_status='complete' if all(r['status'] in ('complete', 'accuracy_storage_complete_attribution_pending') for r in results) else 'partial',
                      methods=results, seed=SEED, model=MODEL, measurement_code_sha256=sha(__file__))
        dump(args.output_root / 'metrics/metric_summary.json', report)
        if report['status'] == 'failed':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
