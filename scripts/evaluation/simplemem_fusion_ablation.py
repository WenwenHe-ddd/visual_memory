"""Rerun official SimpleMem K10/channel-RRF on frozen MemEye memories.

Initial planning calls are inherited with provenance; retrieval, reflection and
answers run again. Reflection batches up to eight; the final reader uses the
control batch size of two. Independent worker databases keep traces isolated.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import traceback
from types import SimpleNamespace

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.evaluation.simplemem_reflection_ablation import operation
from scripts.memory import simplemem_official as official
from scripts.memory import text_memory_baselines as baseline
from scripts.memory.simplemem_fusion import fuse_candidates


def read(path):
    return json.loads(path.read_text())


def signature(protocol):
    return baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})


def checkpoint(root, stage, identity):
    return root / stage / (baseline.hash_object(identity) + '.json')


def verify_inputs(p):
    if signature(p) != p['experiment_signature']:
        raise ValueError('protocol signature differs')
    for name, digest in p['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('input changed: ' + name)
    if official.verify_installation() != p['upstream']:
        raise ValueError('official runtime changed')


def prepare(source, root):
    if root.exists():
        raise ValueError('output already exists; resume with --stage run')
    parent = read(source / 'protocol.json')
    summary = read(source / 'summary.json')
    if summary['status'] != 'complete' or summary['questions'] != 371 or not read(source / 'audit.json')['audit_passed_so_far']:
        raise ValueError('control must contain all 371 audited questions')
    if signature(parent) != parent['experiment_signature'] or official.verify_installation() != parent['upstream']:
        raise ValueError('control protocol or runtime differs')
    if parent['method_defaults']['ENABLE_REFLECTION'] or parent['method_defaults']['SEMANTIC_TOP_K'] != 5:
        raise ValueError('expected audited K5 no-reflection control')
    p = copy.deepcopy(parent)
    p.pop('recovery', None)
    p.update(study='Frozen-memory official SimpleMem K10 equal-channel RRF, reflection on, final five entries',
             context_fusion='channel_rrf', fusion_rank_constant=60, reader_device_map={'': 0},
             generation_batch_size=2, max_padded_batch_tokens=48000,
             retrieval_generation_batch_size=8, retrieval_max_padded_batch_tokens=196608,
             task_workers=8)
    p['method_defaults'].update(SEMANTIC_TOP_K=10, ENABLE_REFLECTION=True)
    p['retrieval_count_policy'] = ('semantic K10 per query; keyword K5; structured K5; official planning and '
        'up to two reflection rounds; deduplicate and apply two-stage equal-channel RRF; final five complete entries')
    p['differences_from_paper'] = [s.replace('semantic top-k=5', 'semantic top-k=10')
        .replace('reflection=False', 'reflection=True') for s in p['differences_from_paper']]
    p['differences_from_paper'].append('Adapter two-stage equal-channel RRF replaces ordered first-five selection; '
        'reflection queries contribute to the semantic channel; this is an adapter variant, not upstream default fusion')
    p['ablation'] = {
        'control_root': str(source.resolve()), 'control_signature': parent['experiment_signature'],
        'control_protocol_sha256': baseline.sha(source / 'protocol.json'),
        'control_responses_sha256': baseline.sha(source / 'qa_responses.jsonl'),
        'changed_method_settings': {'SEMANTIC_TOP_K': [5, 10], 'ENABLE_REFLECTION': [False, True]},
        'changed_context_policy': ['ordered', 'channel_rrf'],
        'policy': 'freeze construction, questions and primary planning calls; run all retrieval and reflection again; '
                  'generate all answers fresh, in consecutive pairs after retrieval finishes',
        'execution_changes': {'retrieval_batch_size': [parent['generation_batch_size'], 8],
                              'reader_batch_size': [parent['generation_batch_size'], 2]},
        'reader_note': 'same maximum reader batch size as control; historical asynchronous batch composition is unavailable'}
    allowed = {Path(official.__file__).resolve(), Path(baseline.__file__).resolve()}
    changes = []
    for name, digest in p['input_hashes'].items():
        current = baseline.sha(Path(name))
        if current != digest:
            if Path(name) not in allowed:
                raise ValueError('unexpected source input change: ' + name)
            changes.append({'path': name, 'old_sha256': digest, 'new_sha256': current})
        p['input_hashes'][name] = current
    for path in (Path(__file__), Path(__file__).with_name('simplemem_reflection_ablation.py'),
                 *official.source_files()):
        p['input_hashes'][str(path.resolve())] = baseline.sha(path)
    p['ablation']['source_changes'] = changes
    p['experiment_signature'] = signature(p)
    root.mkdir(parents=True)
    baseline.write_json(root / 'protocol.json', p)
    for name in ('questions.jsonl', 'observations.json'):
        shutil.copyfile(source / name, root / name)
    if baseline.hash_object(baseline.read_rows(root / 'questions.jsonl')) != p['prepared_questions_sha256']:
        raise ValueError('question hash differs')
    if baseline.hash_object(read(root / 'observations.json')) != p['observations_sha256']:
        raise ValueError('observation hash differs')
    snapshots = {}
    for name, digest in p['input_hashes'].items():
        if Path(name).suffix == '.py':
            dest = root / 'source_snapshot' / (digest[:12] + '_' + Path(name).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
            snapshots[name] = {'snapshot': str(dest.resolve()), 'sha256': digest}
    baseline.write_json(root / 'source_snapshot/manifest.json', snapshots)
    inherited = 0
    for path in sorted((source / 'calls').glob('*.json')):
        call = read(path)
        if operation(call['request']) not in ('information_plan', 'query_plan', 'query_analysis'):
            continue
        if baseline.hash_object(call['request']) != path.stem:
            raise ValueError('control call hash differs')
        call['request']['experiment_signature'] = p['experiment_signature']
        call['inherited_from'] = {'path': str(path.resolve()), 'sha256': baseline.sha(path)}
        baseline.write_json(root / 'calls' / (baseline.hash_object(call['request']) + '.json'), call)
        inherited += 1
    assets = {}
    for task in p['tasks']:
        target = root / task
        target.mkdir()
        shutil.copytree(source / task / 'runtime', target / 'runtime')
        (target / 'runtime/query_analysis_repairs.json').unlink(missing_ok=True)
        for name in ('final_memory.json', 'memory_stats.json'):
            shutil.copyfile(source / task / name, target / name)
        assets[task] = {str(f.relative_to(source / task)): baseline.sha(f)
            for f in (source / task / 'runtime/lancedb').rglob('*') if f.is_file()}
        assets[task]['final_memory.json'] = baseline.sha(target / 'final_memory.json')
    baseline.write_json(root / 'frozen_memory_assets.json', assets)
    baseline.write_json(root.parent.parent / 'run_status.json', {'status': 'prepared',
        'expected_questions': 371, 'retrieved': 0, 'answered': 0, 'inherited_primary_calls': inherited})
    print(f'Prepared 371 questions with {inherited} frozen primary calls', flush=True)


def instrument(memory, trace):
    retriever = memory.core.hybrid_retriever
    reflect = retriever._retrieve_with_intelligent_reflection
    assess = retriever._analyze_information_completeness
    queries = retriever._generate_missing_info_queries

    def reflection(query, entries, plan):
        trace['initial_ids'] = [e.entry_id for e in entries]
        trace['information_plan_sha256'] = baseline.hash_object(plan)
        result = reflect(query, entries, plan)
        trace['final_ids'] = [e.entry_id for e in result]
        return result

    def assessment(query, entries, plan):
        result = assess(query, entries, plan)
        trace['assessments'].append({'status': result, 'candidate_count': len(entries)})
        return result

    def missing_queries(query, entries, plan):
        result = queries(query, entries, plan)
        trace['additional_queries'].append(result)
        return result

    retriever._retrieve_with_intelligent_reflection = reflection
    retriever._analyze_information_completeness = assessment
    retriever._generate_missing_info_queries = missing_queries


def run(root):
    p = read(root / 'protocol.json')
    verify_inputs(p)
    questions = baseline.read_rows(root / 'questions.jsonl')
    model = baseline.LocalMemoryQwen(p)
    baseline.write_json(root / 'generation_environment.json', model.environment)
    retrieval_p = dict(p, max_padded_batch_tokens=p['retrieval_max_padded_batch_tokens'])
    broker = baseline.Broker(model, root, retrieval_p, p['retrieval_generation_batch_size'])
    baseline.install_runtime(broker, p)
    lock, errors = threading.Lock(), []
    done = {q['id'] for q in questions if checkpoint(root, 'retrieval_checkpoints', q['id']).exists()}
    answered = {q['id'] for q in questions if checkpoint(root, 'answer_checkpoints', q['id']).exists()}

    def progress(phase):
        baseline.write_json(root.parent.parent / 'run_status.json', {
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
            'status': 'running_with_errors' if errors else 'running', 'phase': phase,
            'expected_questions': len(questions), 'retrieved': len(done), 'answered': len(answered),
            'by_task_retrieved': dict(Counter(q['task'] for q in questions if q['id'] in done)),
            'failed_workers': errors, 'reflection_enabled': True, 'semantic_top_k': 10,
            'context_memory_limit': 5, 'context_fusion': 'channel_rrf'})

    def worker(index):
        memories = {}
        for q in questions[index::p['task_workers']]:
            if q['id'] in done:
                row = read(checkpoint(root, 'retrieval_checkpoints', q['id']))
                if row['experiment_signature'] != p['experiment_signature'] or any(row[k] != v for k, v in q.items()):
                    raise ValueError('retrieval checkpoint changed')
                continue
            task = q['task']
            if task not in memories:
                target = root / 'worker_runtime' / str(index) / task
                if not target.exists():
                    shutil.copytree(root / task / 'runtime', target / 'runtime')
                broker.local.owner = f'worker{index}/{task}'
                memory = official.OfficialSimpleMemMemory(p, target / 'runtime')
                expected = {e['entry_id']: e for e in read(root / task / 'final_memory.json')}
                actual = {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()}
                if actual != expected or not memory.core.hybrid_retriever.enable_reflection:
                    raise ValueError('restored memory contents or reflection setting differs')
                memory.finalized = True
                trace = {}
                instrument(memory, trace)
                memories[task] = memory, trace
            memory, trace = memories[task]
            trace.clear()
            trace.update(question_id=q['id'], assessments=[], additional_queries=[])
            query = baseline.retrieval_query(q)
            context, detail = memory.recall(query)
            detail['reflection_trace'] = copy.deepcopy(trace)
            row = {**q, 'retrieval_query': query, 'retrieved_context': context, 'retrieval_detail': detail,
                   'reader_messages_sha256': baseline.hash_object(baseline.reader_messages(q, context, p)),
                   'experiment_signature': p['experiment_signature']}
            baseline.write_json(checkpoint(root, 'retrieval_checkpoints', q['id']), row)
            with lock:
                done.add(q['id'])
                progress('retrieval')
                print(f'Retrieved {len(done)}/371: {q["id"]}; rounds={len(trace["assessments"])}', flush=True)

    progress('retrieval')
    try:
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            futures = {pool.submit(worker, i): i for i in range(p['task_workers'])}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    with lock:
                        errors.append({'worker': futures[future], 'error': str(exc), 'traceback': traceback.format_exc()})
                        baseline.write_json(root / 'run_errors.json', errors)
                        progress('retrieval')
                    print(errors[-1]['traceback'], flush=True)
    finally:
        broker.close()
    if errors:
        raise RuntimeError('retrieval workers failed; inspect run_errors.json')
    retrieved = [read(checkpoint(root, 'retrieval_checkpoints', q['id'])) for q in questions]
    baseline.write_rows(root / 'retrieval_results.jsonl', retrieved)
    progress('answer')
    batches = read(root / 'reader_batches.json') if (root / 'reader_batches.json').exists() else []
    # Keep pair boundaries fixed across resumes. A partly committed pair may
    # regenerate its missing member together with the committed neighbour.
    for start in range(0, len(retrieved), p['generation_batch_size']):
        group = retrieved[start:start + p['generation_batch_size']]
        if all(r['id'] in answered for r in group):
            continue
        messages = [baseline.reader_messages(r, r['retrieved_context'], p) for r in group]
        lengths = [len(model.processor.tokenizer.encode(model.processor.apply_chat_template(m,
            tokenize=False, add_generation_prompt=True), add_special_tokens=False)) for m in messages]
        if (max(lengths) + p['max_new_tokens']) * len(group) > p['max_padded_batch_tokens']:
            raise ValueError('reader batch exceeds control token budget')
        outputs = model.generate(messages, p['max_new_tokens'], sampling_temperature=0.0)
        if len(outputs) != len(group):
            raise ValueError('reader returned incomplete batch')
        batches.append({'ids': [r['id'] for r in group], 'input_tokens': lengths})
        for row, message, out in zip(group, messages, outputs):
            if row['id'] in answered:
                continue
            request = {'messages': message, 'max_new_tokens': p['max_new_tokens'], 'kind': 'answer',
                'do_sample': False, 'temperature': 0.0, 'seed': p['seed'], 'response_format': None,
                'experiment_signature': p['experiment_signature']}
            key = baseline.hash_object(request)
            baseline.write_json(root / 'calls' / (key + '.json'), {'request': request, 'output': out})
            prediction = out['raw_response'].strip()
            result = {**row, **out, 'call_key': key,
                      'predicted_choice': prediction if prediction in ('A', 'B', 'C', 'D') else None}
            baseline.write_json(checkpoint(root, 'answer_checkpoints', row['id']), result)
            answered.add(row['id'])
        baseline.write_json(root / 'reader_batches.json', batches)
        progress('answer')
        print(f'Answered {len(answered)}/371; reader batch=2', flush=True)
    rows = [read(checkpoint(root, 'answer_checkpoints', q['id'])) for q in questions]
    for task in p['tasks']:
        target = root / task
        task_rows = [r for r in rows if r['task'] == task]
        baseline.write_rows(target / 'responses.jsonl', task_rows)
        baseline.write_json(target / 'complete.json', {'experiment_signature': p['experiment_signature'],
            'questions': len(task_rows), 'responses_sha256': baseline.sha(target / 'responses.jsonl')})
    baseline.analyze(SimpleNamespace(mode='simplemem', simplemem_backend='official_text',
        simplemem_context_limit=5, output_root=root.parent.parent))
    report = audit(root)
    status = read(root.parent.parent / 'run_status.json')
    status.update(status='complete' if report['audit_passed'] else 'audit_failed', phase='finished',
                  correct=report['correct'], accuracy=report['accuracy'],
                  summary=str(root / 'summary.json'), audit=str(root / 'fusion_audit.json'))
    baseline.write_json(root.parent.parent / 'run_status.json', status)
    if not report['audit_passed']:
        raise RuntimeError('fusion audit failed')
    print(json.dumps({k: report[k] for k in ('questions', 'correct', 'accuracy', 'unchanged_contexts')}, ensure_ascii=False), flush=True)


def audit(root):
    p = read(root / 'protocol.json')
    verify_inputs(p)
    source = Path(p['ablation']['control_root'])
    parent = read(source / 'protocol.json')
    failures, checks = [], Counter()

    def check(condition, name, identity=''):
        checks[name] += 1
        if not condition:
            failures.append({'check': name, 'id': identity})

    check(baseline.sha(source / 'protocol.json') == p['ablation']['control_protocol_sha256'], 'control_protocol_hash')
    check(baseline.sha(source / 'qa_responses.jsonl') == p['ablation']['control_responses_sha256'], 'control_response_hash')
    changes = {k: [v, p['method_defaults'][k]] for k, v in parent['method_defaults'].items() if v != p['method_defaults'][k]}
    check(changes == p['ablation']['changed_method_settings'], 'declared_method_changes')
    for name in ('reader_model', 'reader_system_prompt', 'seed', 'dtype', 'max_new_tokens', 'memory_max_new_tokens',
                 'context_memory_limit', 'preserve_empty_context', 'generation_batch_size', 'max_padded_batch_tokens',
                 'prepared_questions_sha256', 'observations_sha256'):
        check(p[name] == parent[name], 'matched_control_setting', name)
    questions = baseline.read_rows(root / 'questions.jsonl')
    check(baseline.hash_object(questions) == p['prepared_questions_sha256'], 'questions_hash')
    check(baseline.hash_object(read(root / 'observations.json')) == p['observations_sha256'], 'observations_hash')
    expected = {q['id']: q for q in questions}
    control = {r['id']: r for r in baseline.read_rows(source / 'qa_responses.jsonl')}
    rows = baseline.read_rows(root / 'qa_responses.jsonl')
    calls, operations, fresh = {}, Counter(), Counter()
    for path in (root / 'calls').glob('*.json'):
        call = read(path)
        calls[path.stem] = call
        check(baseline.hash_object(call['request']) == path.stem, 'request_hash', path.stem)
        check(call['request']['experiment_signature'] == p['experiment_signature'], 'request_signature', path.stem)
        op = operation(call['request'])
        operations[op] += 1
        check(not call['output']['hit_token_limit'], 'no_generation_truncation', path.stem)
        if 'inherited_from' in call:
            origin = call['inherited_from']
            original = read(Path(origin['path']))
            check(baseline.sha(Path(origin['path'])) == origin['sha256'], 'inherited_file_hash', path.stem)
            original['request']['experiment_signature'] = p['experiment_signature']
            check(original['request'] == call['request'] and original['output'] == call['output'], 'inherited_payload', path.stem)
            check(op in ('information_plan', 'query_plan', 'query_analysis'), 'only_primary_calls_inherited', path.stem)
        else:
            fresh[op] += 1
    assets = read(root / 'frozen_memory_assets.json')
    for task, files in assets.items():
        for name, digest in files.items():
            check(baseline.sha(source / task / name) == digest, 'source_memory_frozen', task + '/' + name)
            check(baseline.sha(root / task / name) == digest, 'memory_copy_frozen', task + '/' + name)
        for worker in (root / 'worker_runtime').iterdir():
            db = worker / task
            if not db.exists():
                continue
            actual_files = {str(f.relative_to(db)) for f in (db / 'runtime/lancedb').rglob('*') if f.is_file()}
            check(actual_files == set(files) - {'final_memory.json'}, 'worker_db_file_set', str(db))
            for name, digest in files.items():
                if name != 'final_memory.json':
                    check(baseline.sha(db / name) == digest, 'worker_db_frozen', str(db / name))
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    memories = {t: {e['entry_id']: e for e in read(root / t / 'final_memory.json')} for t in p['tasks']}
    comparisons = []
    for row in rows:
        identity, task = row['id'], row['task']
        q, before, detail = expected[identity], control[identity], row['retrieval_detail']
        trace = detail['reflection_trace']
        check(all(row[k] == v for k, v in q.items()), 'original_question_fields', identity)
        check(row['experiment_signature'] == p['experiment_signature'], 'response_signature', identity)
        check(row['retrieval_query'] == baseline.retrieval_query(q), 'answer_free_retrieval_query', identity)
        check(trace['final_ids'] == detail['original_item_ids'], 'reflection_results_match', identity)
        check(trace['final_ids'][:len(trace['initial_ids'])] == trace['initial_ids'], 'initial_union_preserved', identity)
        check(1 <= len(trace['assessments']) <= p['method_defaults']['MAX_REFLECTION_ROUNDS'], 'reflection_executed', identity)
        batches, ids = detail['search_batches'], detail['context_memory_ids']
        check(set().union(*(set(b['result_ids']) for b in batches)) == set(detail['original_item_ids']), 'search_union', identity)
        check({b['channel'] for b in batches} == {'semantic', 'keyword', 'structured'}, 'all_three_channels_called', identity)
        for b in batches:
            k = 10 if b['channel'] == 'semantic' else 5
            check(b['top_k'] == k and len(b['result_ids']) <= k, 'actual_search_k', identity)
        entries = memories[task]
        ranked, fusion = fuse_candidates([MemoryEntry(**entries[i]) for i in detail['original_item_ids']], batches, 60)
        check(fusion == detail['fusion'], 'recomputed_fusion', identity)
        check(ids == [e.entry_id for e in ranked[:5]] and len(ids) == len(set(ids)) == 5, 'five_distinct_fused_entries', identity)
        check(detail['items'] == [dict(entries[i], id=i) for i in ids], 'whole_entry_payload', identity)
        context = formatter._format_contexts([MemoryEntry(**entries[i]) for i in ids])
        check(context == row['retrieved_context'], 'formatted_context', identity)
        messages = baseline.reader_messages(q, context, p)
        check(baseline.hash_object(messages) == row['reader_messages_sha256'], 'reader_message_hash', identity)
        call = calls[row['call_key']]
        check(call['request']['messages'] == messages and call['request']['kind'] == 'answer', 'actual_reader_input', identity)
        check('inherited_from' not in call, 'fresh_answer', identity)
        check(all(row[k] == v for k, v in call['output'].items()), 'answer_output', identity)
        check(row['predicted_choice'] in ('A', 'B', 'C', 'D') and row['predicted_choice'] == row['raw_response'].strip(), 'answer_format', identity)
        selected = detail['fusion']['ranked_candidates'][:5]
        sem = set(fusion['channel_rankings']['semantic'])
        first_sem = next(b['result_ids'][:5] for b in batches if b['channel'] == 'semantic')
        comparisons.append({'id': identity, 'task': task,
            'control_correct': before['predicted_choice'] == before['gold_choice'],
            'fusion_correct': row['predicted_choice'] == row['gold_choice'],
            'context_unchanged': context == before['retrieved_context'],
            'prediction_unchanged': row['predicted_choice'] == before['predicted_choice'],
            'context_same_as_first_semantic_five': ids == first_sem,
            'selected_outside_first_semantic_five': len(set(ids) - set(first_sem)),
            'selected_absent_from_all_semantic': len(set(ids) - sem),
            'selected_added_by_reflection': len(set(ids) - set(trace['initial_ids'])),
            'selected_channel_hits': {c: sum(c in x['channels'] for x in selected) for c in ('semantic', 'keyword', 'structured')},
            'channel_candidate_counts': {c: len(v) for c, v in fusion['channel_rankings'].items()},
            'added_candidates': len(trace['final_ids']) - len(trace['initial_ids']),
            'reflection_rounds': len(trace['assessments']),
            'reflection_searches': sum(b['phase'] == 'reflection' for b in batches)})
    check(len(rows) == len({r['id'] for r in rows}) == len(expected) == 371, 'all_371_unique_questions')
    check({r['id'] for r in rows} == set(expected), 'question_id_set')
    check(fresh['answer'] == 371, 'all_371_answers_fresh')
    check(not any(fresh[k] for k in ('construction', 'information_plan', 'query_plan', 'query_analysis')), 'construction_and_primary_planning_frozen')
    reader_batches = read(root / 'reader_batches.json')
    check(all(1 <= len(b['ids']) <= 2 for b in reader_batches), 'reader_batch_two')
    check({i for b in reader_batches for i in b['ids']} == set(expected), 'reader_batch_coverage')
    correct = sum(r['predicted_choice'] == r['gold_choice'] for r in rows)
    summary = read(root / 'summary.json')
    check(summary['status'] == 'complete' and summary['correct'] == correct, 'summary_recomputed')
    report = {'status': 'complete' if len(rows) == 371 else 'partial', 'audit_passed': not failures,
        'questions': len(rows), 'correct': correct, 'accuracy': correct / len(rows) if rows else None,
        'control_correct': sum(x['control_correct'] for x in comparisons),
        'unchanged_contexts': sum(x['context_unchanged'] for x in comparisons),
        'unchanged_predictions': sum(x['prediction_unchanged'] for x in comparisons),
        'wrong_to_right': sum(not x['control_correct'] and x['fusion_correct'] for x in comparisons),
        'right_to_wrong': sum(x['control_correct'] and not x['fusion_correct'] for x in comparisons),
        'questions_with_selected_reflection_additions': sum(x['selected_added_by_reflection'] > 0 for x in comparisons),
        'selected_reflection_addition_total': sum(x['selected_added_by_reflection'] for x in comparisons),
        'questions_with_selected_nonsemantic_entries': sum(x['selected_absent_from_all_semantic'] > 0 for x in comparisons),
        'questions_differing_from_first_semantic_five': sum(not x['context_same_as_first_semantic_five'] for x in comparisons),
        'selected_channel_hits': {c: sum(x['selected_channel_hits'][c] for x in comparisons) for c in ('semantic', 'keyword', 'structured')},
        'questions_with_channel_hits': {c: sum(x['selected_channel_hits'][c] > 0 for x in comparisons) for c in ('semantic', 'keyword', 'structured')},
        'questions_with_nonempty_channel': {c: sum(x['channel_candidate_counts'][c] > 0 for x in comparisons) for c in ('semantic', 'keyword', 'structured')},
        'reflection_rounds': dict(Counter(x['reflection_rounds'] for x in comparisons)),
        'reflection_searches': sum(x['reflection_searches'] for x in comparisons),
        'llm_operations': dict(operations), 'fresh_llm_operations': dict(fresh),
        'checks': dict(checks), 'failures': failures, 'by_task': summary['by_task'],
        'experiment_signature': p['experiment_signature']}
    baseline.write_rows(root / 'paired_comparison.jsonl', comparisons)
    baseline.write_json(root / 'fusion_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'run', 'audit', 'all'), default='all')
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        if args.source is None:
            parser.error('--source is required for prepare')
        prepare(args.source, args.root)
    if args.stage in ('run', 'all'):
        run(args.root)
    if args.stage == 'audit':
        print(json.dumps(audit(args.root), ensure_ascii=False))
