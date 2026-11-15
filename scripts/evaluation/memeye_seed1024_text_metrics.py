"""Measure seed1024 text methods with the original MemEye storage/recall policy.

SimpleMem source attribution uses the existing two-pass, quote-checked protocol
with evaluation seed42.  The default attribution flow stops at the review queue;
it never invents review decisions or credits ambiguous source alternatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_seed1024_text_methods as runs
from scripts.memory import simplemem_official as official
from scripts.memory.simplemem_fusion import fuse_candidates

base = runs.baseline
OUT = runs.OUTPUT.parent / 'metrics/simplemem_source_attribution'
MIB = 1024 ** 2


def read(path):
    return json.loads(Path(path).read_text())


def completed_source(method, output_root=runs.OUTPUT):
    source = runs.target(method, output_root)
    summary_path = source / 'summary.json'
    if not summary_path.exists() or read(summary_path)['status'] != 'complete':
        return source, None, None
    p = read(source / 'protocol.json')
    device = 'cuda:' + str(p['reader_device_map'][''])
    report = runs.audit(method, output_root, device)
    if not report['coverage_complete'] or not report['audit_passed_so_far']:
        raise ValueError('Source must pass full-flow coverage and provenance audit')
    rows = base.read_rows(source / 'qa_responses.jsonl')
    return source, p, rows


def storage_files(method, source, protocol):
    files = []
    memory_count = 0
    for task in protocol['tasks']:
        work = source / task
        if method in ('gen_agents', 'a_mem'):
            files.append(work / 'final_memory.json')
            memory_count += len(read(files[-1]))
        elif method == 'memoryos':
            group = sorted((work / 'runtime').rglob('*.json'))
            expected = {'short_term.json', 'mid_term.json', 'long_term_user.json', 'long_term_assistant.json'}
            if len(group) != 4 or {p.name for p in group} != expected:
                raise ValueError('MemoryOS requires exactly four native runtime JSON layers per task')
            files.extend(group)
        else:
            group = sorted(f for f in (work / 'runtime/lancedb').rglob('*') if f.is_file())
            if not group:
                raise ValueError('SimpleMem persisted database is missing')
            files.extend([*group, work / 'runtime/resolved_config.json'])
            memory_count += len(read(work / 'final_memory.json'))
    if len(files) != len(set(files)):
        raise ValueError('Duplicate storage artifact')
    return files, memory_count if method != 'memoryos' else None


def a_mem_recall(source, protocol, rows):
    canonical_path = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    canonical = {q['id']: q for q in base.read_rows(canonical_path)}
    memories = {task: read(source / task / 'final_memory.json') for task in protocol['tasks']}
    results = []
    for row in rows:
        q = canonical[row['id']]
        if any(row[k] != q[k] for k in ('task', 'question', 'options', 'gold_choice', 'gold_ids')):
            raise ValueError('Canonical MCQ changed: ' + row['id'])
        ids = row['retrieval_detail']['context_memory_ids']
        if len(ids) != len(set(ids)) or len(ids) != 5:
            raise ValueError('A-Mem must have five distinct final notes')
        selected = [memories[row['task']][identity] for identity in ids]
        rendered = ''.join('talk start time:' + n['timestamp'] + 'memory content: ' + n['content']
            + 'memory context: ' + n['context'] + 'memory keywords: ' + str(n['keywords'])
            + 'memory tags: ' + str(n['tags']) + '\n' for n in selected)
        if rendered != row['retrieved_context']:
            raise ValueError('A-Mem final context differs from selected whole notes')
        rounds = {row['task'] + '/' + identity for identity in ids}
        gold = set(q['gold_ids'])
        results.append({'id': row['id'], 'task': row['task'], 'retrieved_round_ids': sorted(rounds),
                        'gold_ids': sorted(gold), 'hit_ids': sorted(rounds & gold),
                        'gold_count': len(gold), 'hit_count': len(rounds & gold)})
    denominator = sum(r['gold_count'] for r in results)
    numerator = sum(r['hit_count'] for r in results)
    if len(results) != 371 or denominator != 1041:
        raise ValueError('Expected 371 questions and 1041 gold source-round references')
    base.write_rows(source / 'source_recall_at5_per_question.jsonl', results)
    report = {'questions': len(results), 'gold_round_hits': numerator,
              'gold_round_references': denominator, 'recall_at5_micro': numerator / denominator,
              'formula': 'sum_i |retrieved_source_rounds_i intersect gold_i| / sum_i |gold_i|',
              'canonical_sha256': base.sha(canonical_path),
              'responses_sha256': base.sha(source / 'qa_responses.jsonl')}
    base.write_json(source / 'source_recall_at5.json', report)
    return report


def fusion_audit(source, save=True):
    """Recompute RRF, full entries, rendered contexts and actual reader payloads."""
    protocol = read(source / 'protocol.json')
    official.verify_installation()
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    rows = base.read_rows(source / 'qa_responses.jsonl')
    memories = {task: {e['entry_id']: e for e in read(source / task / 'final_memory.json')}
                for task in protocol['tasks']}
    checks = Counter()
    failures = []

    def check(ok, name, identity):
        checks[name] += 1
        if not ok:
            failures.append({'check': name, 'id': identity})

    for row in rows:
        identity, detail = row['id'], row['retrieval_detail']
        entries = memories[row['task']]
        batches, ids = detail['search_batches'], detail['context_memory_ids']
        original = [MemoryEntry(**entries[i]) for i in detail['original_item_ids']]
        ranked, fusion = fuse_candidates(original, batches, 60)
        check(fusion == detail['fusion'], 'recomputed_channel_rrf', identity)
        empty = not ids and detail.get('source_empty_preserved') and not original
        check(ids == [e.entry_id for e in ranked[:5]] and (empty or len(ids) == len(set(ids)) == 5),
              'five_distinct_fused_entries_or_native_empty', identity)
        check(detail['items'] == [dict(entries[i], id=i) for i in ids], 'whole_stored_entries', identity)
        for batch in batches:
            k = 10 if batch['channel'] == 'semantic' else 5
            check(batch['top_k'] == k and len(batch['result_ids']) <= k, 'native_channel_search_k', identity)
        context = formatter._format_contexts([MemoryEntry(**entries[i]) for i in ids])
        check(context == row['retrieved_context'], 'reconstructed_reader_context', identity)
        messages = base.reader_messages(row, context, protocol)
        check(base.hash_object(messages) == row['reader_messages_sha256'], 'reader_message_hash', identity)
        call_path = source / 'calls' / (row['call_key'] + '.json')
        call = read(call_path)
        check(base.hash_object(call['request']) == row['call_key'], 'reader_call_hash', identity)
        check(call['request']['messages'] == messages and call['request']['kind'] == 'answer',
              'actual_reader_payload', identity)
        check(call['request']['seed'] == protocol['seed'], 'reader_seed', identity)
        check(row['raw_response'] == call['output']['raw_response'], 'reader_output', identity)
    check(len(rows) == len({r['id'] for r in rows}) == 371, 'complete_unique_cohort', 'all')
    report = {'status': 'complete', 'audit_passed': not failures, 'questions': len(rows),
              'seed': protocol['seed'], 'checks': dict(checks), 'failures': failures,
              'interpretation': 'Independent final-context audit of the fresh full-flow run; no frozen-construction claim',
              'protocol_sha256': base.sha(source / 'protocol.json'),
              'responses_sha256': base.sha(source / 'qa_responses.jsonl'),
              'audit_code_sha256': base.sha(Path(__file__))}
    if save:
        base.write_json(source / 'fusion_audit.json', report)
    if failures:
        raise ValueError(f'{len(failures)} SimpleMem context/fusion provenance failure(s)')
    return report


def measure(method, output_root=runs.OUTPUT, attribution_root=OUT):
    source, protocol, rows = completed_source(method, output_root)
    if protocol is None:
        return {'method': method, 'status': 'awaiting_complete_source'}
    files, count = storage_files(method, source, protocol)
    components = [{'path': str(f.resolve()), 'bytes': f.stat().st_size, 'sha256': base.sha(f)} for f in files]
    total = sum(f['bytes'] for f in components)
    correct = sum(r['predicted_choice'] == r['gold_choice'] for r in rows)
    report = {'method': method, 'seed': runs.SEED, 'status': 'complete', 'questions': len(rows),
        'correct': correct, 'accuracy': correct / len(rows), 'storage_bytes': total, 'storage_mib': total / MIB,
        'memory_count': count, 'recall_at5_micro': None,
        'storage_policy': {
            'gen_agents': 'saved final_memory snapshots once; vectors and importance/recency tensors were not persisted',
            'memoryos': 'four native per-task runtime JSON layers including their decimal embedding arrays',
            'a_mem': 'saved final_memory snapshots once; retrieval vectors were not persisted',
            'simplemem': 'all files in the native per-task lancedb plus resolved_config.json; retained versions included; duplicate final_memory export excluded',
        }[method],
        'storage_unit': 'MiB = logical file bytes / 1048576; weights, code, calls, QA, logs and provenance annotations excluded',
        'reference_policy_source': str(ROOT / 'scripts/analysis/audit_memory_storage.py'),
        'source_protocol_sha256': base.sha(source / 'protocol.json'),
        'source_responses_sha256': base.sha(source / 'qa_responses.jsonl'),
        'measurement_code_sha256': base.sha(Path(__file__)), 'storage_components': components}
    if method == 'a_mem':
        recall = a_mem_recall(source, protocol, rows)
        report.update(recall_at5_micro=recall['recall_at5_micro'], recall_status='complete',
                      gold_round_hits=recall['gold_round_hits'], gold_round_references=recall['gold_round_references'])
    elif method == 'simplemem':
        recall_path = Path(attribution_root) / 'recall_summary.json'
        lock_path = Path(attribution_root) / 'annotation_lock.json'
        if recall_path.exists() and lock_path.exists():
            recall = read(recall_path)
            if recall['source_experiment_signature'] != protocol['experiment_signature'] or recall['responses_sha256'] != report['source_responses_sha256']:
                raise ValueError('Attribution belongs to a different SimpleMem run')
            report.update(recall_at5_micro=recall['recall_at5_micro'], recall_status='complete_conservative_posthoc_attribution',
                          gold_round_hits=recall['gold_round_hits'], gold_round_references=recall['gold_round_references'],
                          attribution_lock_sha256=base.sha(lock_path), recall_summary_sha256=base.sha(recall_path))
        else:
            report.update(status='accuracy_storage_complete_attribution_pending', recall_status='pending_evidence_annotation_and_review')
    else:
        report['recall_status'] = 'not_reported_at5_native_context_is_top10_or_dynamic'
    base.write_json(source / 'metric_summary.json', report)
    return report


def attribution(stage='prepare', output_root=runs.OUTPUT, attribution_root=OUT, device='cuda:0', model=None):
    from scripts.analysis import simplemem_source_attribution as annotation
    from scripts.analysis import simplemem_attribution_recall as recall
    source, protocol, _ = completed_source('simplemem', output_root)
    if protocol is None:
        return {'method': 'simplemem', 'status': 'awaiting_complete_source'}
    root = Path(attribution_root)
    if stage in ('prepare', 'all'):
        if not (root / 'protocol.json').exists():
            fusion_audit(source)
            annotation.prepare(source, root)
            p = read(root / 'protocol.json')
            p.update(seed=42, reader_device_map={'': int(device.split(':')[1])},
                     evaluation_seed_policy='Use seed42 for source-attribution evaluation, independently of memory-generation seed1024')
            p['input_hashes'][str(Path(__file__).resolve())] = base.sha(Path(__file__))
            p['signature'] = base.hash_object({k: v for k, v in p.items() if k != 'signature'})
            base.write_json(root / 'protocol.json', p)
        annotation.verify_inputs(read(root / 'protocol.json'), root)
    annotation.verify_inputs(read(root / 'protocol.json'), root)
    if read(root / 'protocol.json')['seed'] != 42:
        raise ValueError('Attribution evaluator must use seed42')
    if stage in ('run', 'all'):
        factory = annotation.LocalQwen
        try:
            if model is not None:
                p = read(root / 'protocol.json')
                for key in ('reader_model_path', 'dtype', 'context_budget'):
                    if model.protocol[key] != p[key]:
                        raise ValueError('Injected attribution model differs: ' + key)
                model.protocol = p
                annotation.LocalQwen = lambda p: model
            annotation.run(root)
        finally:
            annotation.LocalQwen = factory
        recall.review(root)
    elif stage == 'review':
        recall.review(root)
    elif stage == 'freeze':
        # Existing freeze() requires explicit evidence-bearing decisions for
        # every flagged entry; this wrapper does not create such decisions.
        recall.freeze(root)
    elif stage == 'score':
        recall.score(root)
        measure('simplemem', output_root, root)
    progress_path = root / 'review_status.json' if (root / 'review_status.json').exists() else root / 'progress.json'
    progress = read(progress_path)
    if (root / 'recall_summary.json').exists():
        status = 'complete'
    elif (root / 'annotation_lock.json').exists():
        status = 'sealed_before_scoring'
    elif (root / 'review_status.json').exists():
        status = 'awaiting_explicit_review' if progress['review_entries'] else 'ready_to_freeze'
    else:
        status = progress.get('status', 'partial')
    return {'method': 'simplemem', 'attribution_root': str(root), 'stage': stage,
            'seed': 42, **progress, 'status': status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure', 'attribution'), required=True)
    parser.add_argument('--method', choices=(*runs.METHODS, 'all'), default='all')
    parser.add_argument('--output-root', type=Path, default=runs.OUTPUT)
    parser.add_argument('--attribution-root', type=Path, default=OUT)
    parser.add_argument('--attribution-stage', choices=('prepare', 'run', 'review', 'freeze', 'score', 'all'), default='prepare')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.phase == 'attribution':
        print(json.dumps(attribution(args.attribution_stage, args.output_root, args.attribution_root, args.device), ensure_ascii=False), flush=True)
    else:
        methods = runs.METHODS if args.method == 'all' else (args.method,)
        for method in methods:
            result = measure(method, args.output_root, args.attribution_root)
            print(json.dumps({k: v for k, v in result.items() if k != 'storage_components'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
