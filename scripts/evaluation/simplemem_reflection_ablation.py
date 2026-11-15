"""Toggle official SimpleMem reflection on frozen memory and initial retrieval.

Replays identical primary-planning LLM requests, runs real Qwen reflection,
and freshly generates every answer. The completed control stays untouched.
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
from scripts.memory import simplemem_official as official
from scripts.memory import text_memory_baselines as baseline


def read(path):
    return json.loads(path.read_text())


def operation(request):
    if request['kind'] == 'answer':
        return 'answer'
    system = '\n'.join(m['content'] for m in request['messages'] if m['role'] == 'system')
    for marker, name in (
        ('information extraction assistant', 'construction'),
        ('information requirement analyst', 'information_plan'),
        ('query generation specialist', 'query_plan'),
        ('query analysis assistant', 'query_analysis'),
        ('information completeness evaluator', 'reflection_assessment'),
        ('missing information query generator', 'reflection_queries')):
        if marker in system:
            return name
    raise ValueError('unrecognized LLM operation: ' + system)


def prepare(source, root):
    if root.exists():
        raise ValueError('output exists; resume with --stage run')
    parent = read(source / 'protocol.json')
    summary, control_audit = read(source / 'summary.json'), read(source / 'audit.json')
    if summary['status'] != 'complete' or summary['questions'] != 371 or not control_audit['audit_passed_so_far']:
        raise ValueError('control must be complete and audited')
    if baseline.hash_object({k: v for k, v in parent.items() if k != 'experiment_signature'}) != parent['experiment_signature']:
        raise ValueError('control protocol signature differs')
    for name, digest in parent['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('control code or input changed: ' + name)
    if official.verify_installation() != parent['upstream']:
        raise ValueError('official runtime changed')
    if parent['method_defaults']['ENABLE_REFLECTION'] is not False:
        raise ValueError('control already has reflection enabled')
    protocol = copy.deepcopy(parent)
    protocol.pop('recovery', None)
    protocol['method_defaults']['ENABLE_REFLECTION'] = True
    # Execution throughput only: per-request RNG, temperatures and token caps
    # stay fixed; the shared broker never truncates prompts when batching.
    protocol['generation_batch_size'] = 8
    protocol['max_padded_batch_tokens'] = 196608
    protocol['retrieval_count_policy'] = protocol['retrieval_count_policy'].replace('reflection=False', 'reflection=True')
    protocol['differences_from_paper'] = [x.replace('reflection=False', 'reflection=True') for x in protocol['differences_from_paper']]
    protocol['ablation'] = {
        'control_root': str(source.resolve()), 'control_signature': parent['experiment_signature'],
        'control_protocol_sha256': baseline.sha(source / 'protocol.json'),
        'control_responses_sha256': baseline.sha(source / 'qa_responses.jsonl'),
        'changed_method_settings': {'ENABLE_REFLECTION': [False, True]},
        'execution_changes': {'generation_batch_size': [parent['generation_batch_size'], 8],
                              'max_padded_batch_tokens': [parent['max_padded_batch_tokens'], 196608]},
        'policy': 'freeze stored memories and initial planning/query analysis via audited request cache; '
                  'execute upstream reflection with real Qwen; freshly generate all final answers'}
    protocol['input_hashes'][str(Path(__file__).resolve())] = baseline.sha(Path(__file__))
    protocol['experiment_signature'] = baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})
    root.mkdir(parents=True)
    baseline.write_json(root / 'protocol.json', protocol)
    for name in ('questions.jsonl', 'observations.json'):
        shutil.copyfile(source / name, root / name)
    snapshots = {}
    for name, digest in protocol['input_hashes'].items():
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
        call['request']['experiment_signature'] = protocol['experiment_signature']
        call['inherited_from'] = {'path': str(path.resolve()), 'sha256': baseline.sha(path)}
        baseline.write_json(root / 'calls' / (baseline.hash_object(call['request']) + '.json'), call)
        inherited += 1
    assets = {}
    for task in parent['tasks']:
        target = root / task
        target.mkdir()
        shutil.copytree(source / task / 'runtime', target / 'runtime')
        # Only traces produced by this run belong in the current runtime log.
        repair_log = target / 'runtime/query_analysis_repairs.json'
        if repair_log.exists():
            repair_log.unlink()
        for name in ('final_memory.json', 'memory_stats.json'):
            shutil.copyfile(source / task / name, target / name)
        baseline.write_rows(target / 'responses.jsonl', [])
        baseline.write_json(target / 'progress.json', {'phase': 'waiting', 'done': 0,
            'total': summary['by_task'][task]['questions']})
        assets[task] = {str(p.relative_to(source / task)): baseline.sha(p)
            for p in (source / task / 'runtime/lancedb').rglob('*') if p.is_file()}
        assets[task]['final_memory.json'] = baseline.sha(source / task / 'final_memory.json')
    baseline.write_json(root / 'frozen_memory_assets.json', assets)
    print(f'Prepared reflection ablation: 371 fresh answers; {inherited} cached primary LLM calls', flush=True)


def instrument(memory, trace, control):
    retriever = memory.core.hybrid_retriever
    semantic = retriever.vector_store.semantic_search
    reflect = retriever._retrieve_with_intelligent_reflection
    assess = retriever._analyze_information_completeness
    queries = retriever._generate_missing_info_queries

    def semantic_search(query, top_k=5):
        result = semantic(query, top_k=top_k)
        trace['searches'].append({'phase': trace['phase'], 'query': query,
            'top_k': top_k, 'result_ids': [e.entry_id for e in result]})
        if top_k != 5 or len(result) > 5:
            raise ValueError('semantic search violates K=5')
        return result

    def reflection(query, entries, plan):
        ids = [e.entry_id for e in entries]
        if ids != control[trace['question_id']]['retrieval_detail']['original_item_ids']:
            raise ValueError('initial retrieval differs from frozen control: ' + trace['question_id'])
        trace['initial_ids'] = ids
        trace['information_plan_sha256'] = baseline.hash_object(plan)
        trace['phase'] = 'reflection'
        result = reflect(query, entries, plan)
        trace['final_ids'] = [e.entry_id for e in result]
        trace['phase'] = 'finished'
        return result

    def assessment(query, entries, plan):
        result = assess(query, entries, plan)
        trace['assessments'].append({'status': result, 'candidate_count': len(entries)})
        return result

    def missing_queries(query, entries, plan):
        result = queries(query, entries, plan)
        trace['additional_queries'].append(result)
        return result

    retriever.vector_store.semantic_search = semantic_search
    retriever._retrieve_with_intelligent_reflection = reflection
    retriever._analyze_information_completeness = assessment
    retriever._generate_missing_info_queries = missing_queries


def audit(root):
    p = read(root / 'protocol.json')
    source = Path(p['ablation']['control_root'])
    old_p = read(source / 'protocol.json')
    failures, checks = [], Counter()

    def check(condition, name, identity=''):
        checks[name] += 1
        if not condition:
            failures.append({'check': name, 'id': identity})

    check(baseline.hash_object({k: v for k, v in p.items() if k != 'experiment_signature'}) == p['experiment_signature'], 'protocol_signature')
    check(baseline.sha(source / 'protocol.json') == p['ablation']['control_protocol_sha256'], 'control_protocol_hash')
    check(baseline.sha(source / 'qa_responses.jsonl') == p['ablation']['control_responses_sha256'], 'control_responses_hash')
    for name, digest in p['input_hashes'].items():
        check(baseline.sha(Path(name)) == digest, 'input_hash', name)
    check(official.verify_installation() == p['upstream'], 'pinned_runtime')
    differences = {k: [v, p['method_defaults'][k]] for k, v in old_p['method_defaults'].items()
                   if v != p['method_defaults'][k]}
    check(differences == {'ENABLE_REFLECTION': [False, True]}, 'only_reflection_setting_changed')
    for name in ('reader_model', 'reader_system_prompt', 'seed', 'dtype', 'max_new_tokens', 'memory_max_new_tokens',
                 'context_memory_limit', 'preserve_empty_context',
                 'prepared_questions_sha256', 'observations_sha256'):
        check(p[name] == old_p[name], 'matched_control_setting', name)
    questions = baseline.read_rows(root / 'questions.jsonl')
    check(baseline.hash_object(questions) == p['prepared_questions_sha256'], 'question_hash')
    check(baseline.hash_object(read(root / 'observations.json')) == p['observations_sha256'], 'observation_hash')
    expected = {q['id']: q for q in questions}
    control = {r['id']: r for r in baseline.read_rows(source / 'qa_responses.jsonl')}
    calls, operations, fresh = {}, Counter(), Counter()
    for path in (root / 'calls').glob('*.json'):
        call = read(path)
        calls[path.stem] = call
        check(baseline.hash_object(call['request']) == path.stem, 'call_request_hash', path.stem)
        check(call['request']['experiment_signature'] == p['experiment_signature'], 'call_signature', path.stem)
        op = operation(call['request'])
        operations[op] += 1
        if 'inherited_from' in call:
            origin = call['inherited_from']
            original = read(Path(origin['path']))
            check(baseline.sha(Path(origin['path'])) == origin['sha256'], 'inherited_source_hash', path.stem)
            original['request']['experiment_signature'] = p['experiment_signature']
            check(original['request'] == call['request'] and original['output'] == call['output'], 'inherited_call_unchanged', path.stem)
            check(op in ('information_plan', 'query_plan', 'query_analysis'), 'only_primary_calls_inherited', path.stem)
        else:
            fresh[op] += 1
        check(not call['output']['hit_token_limit'], 'no_output_truncation', path.stem)
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    assets = read(root / 'frozen_memory_assets.json')
    rows, comparisons = [], []
    for task in p['tasks']:
        target = root / task
        for relative, digest in assets[task].items():
            check(baseline.sha(target / relative) == digest, 'memory_assets_unchanged', task + '/' + relative)
        settings = read(target / 'runtime/resolved_config.json')['settings']
        check(settings == p['method_defaults'], 'resolved_settings', task)
        entries = {e['entry_id']: e for e in read(target / 'final_memory.json')}
        task_rows = baseline.read_rows(target / 'responses.jsonl')
        checkpoint = read(target / 'complete.json') if (target / 'complete.json').exists() else {}
        check(checkpoint.get('responses_sha256') == baseline.sha(target / 'responses.jsonl')
              and checkpoint.get('experiment_signature') == p['experiment_signature'], 'task_checkpoint', task)
        rows.extend(task_rows)
        for row in task_rows:
            identity = row['id']
            q, before, detail = expected[identity], control[identity], row['retrieval_detail']
            trace = detail['reflection_trace']
            check(all(row[k] == v for k, v in q.items()), 'original_question_fields', identity)
            check(row['experiment_signature'] == p['experiment_signature'], 'response_signature', identity)
            check(row['retrieval_query'] == baseline.retrieval_query(q), 'answer_free_query', identity)
            check(trace['initial_ids'] == before['retrieval_detail']['original_item_ids'], 'initial_retrieval_frozen', identity)
            check(detail['original_item_ids'] == trace['final_ids'], 'reflection_results_match', identity)
            check(trace['final_ids'][:len(trace['initial_ids'])] == trace['initial_ids'], 'reflection_preserves_initial_prefix', identity)
            check(1 <= len(trace['assessments']) <= p['method_defaults']['MAX_REFLECTION_ROUNDS'], 'actual_reflection_executed', identity)
            for search in trace['searches']:
                check(search['top_k'] == 5 and len(search['result_ids']) <= 5, 'actual_semantic_k5', identity)
            ids = detail['context_memory_ids']
            check(len(ids) == len(set(ids)) == 5 and ids == list(dict.fromkeys(trace['final_ids']))[:5], 'final_five_complete_entries', identity)
            check(detail['items'] == [dict(entries[i], id=i) for i in ids], 'context_entry_payloads', identity)
            context = formatter._format_contexts([MemoryEntry(**entries[i]) for i in ids])
            check(row['retrieved_context'] == context, 'formatted_context', identity)
            messages = baseline.reader_messages(q, context, p)
            check(row['reader_messages_sha256'] == baseline.hash_object(messages), 'reader_payload_hash', identity)
            call = calls[row['call_key']]
            check(call['request']['messages'] == messages and call['request']['kind'] == 'answer', 'actual_reader_request', identity)
            check('inherited_from' not in call, 'fresh_answer', identity)
            check(all(row[k] == v for k, v in call['output'].items()), 'answer_matches_output', identity)
            check(row['predicted_choice'] in ('A', 'B', 'C', 'D'), 'answer_format', identity)
            comparisons.append({'id': identity, 'task': task,
                'control_correct': before['predicted_choice'] == before['gold_choice'],
                'reflection_correct': row['predicted_choice'] == row['gold_choice'],
                'context_unchanged': row['retrieved_context'] == before['retrieved_context'],
                'prediction_unchanged': row['predicted_choice'] == before['predicted_choice'],
                'added_candidates': len(trace['final_ids']) - len(trace['initial_ids']),
                'reflection_rounds': len(trace['assessments']),
                'additional_searches': sum(s['phase'] == 'reflection' for s in trace['searches'])})
    check(len(rows) == len({r['id'] for r in rows}) == len(expected) == 371, 'all_371_unique_questions')
    check({r['id'] for r in rows} == set(expected), 'question_id_set')
    check(not fresh['information_plan'] and not fresh['query_plan'] and not fresh['query_analysis'], 'primary_llm_calls_frozen')
    check(fresh['answer'] == 371, 'all_answers_fresh')
    summary = read(root / 'summary.json')
    correct = sum(r['predicted_choice'] == r['gold_choice'] for r in rows)
    check(summary['status'] == 'complete' and summary['correct'] == correct, 'summary_recomputed')
    report = {'status': 'complete' if len(rows) == 371 else 'partial', 'audit_passed': not failures,
        'questions': len(rows), 'correct': correct, 'accuracy': correct / len(rows) if rows else None,
        'control_correct': sum(x['control_correct'] for x in comparisons),
        'unchanged_contexts': sum(x['context_unchanged'] for x in comparisons),
        'unchanged_predictions': sum(x['prediction_unchanged'] for x in comparisons),
        'questions_with_added_candidates': sum(x['added_candidates'] > 0 for x in comparisons),
        'additional_candidate_total': sum(x['added_candidates'] for x in comparisons),
        'additional_semantic_searches': sum(x['additional_searches'] for x in comparisons),
        'reflection_rounds': dict(Counter(x['reflection_rounds'] for x in comparisons)),
        'wrong_to_right': sum(not x['control_correct'] and x['reflection_correct'] for x in comparisons),
        'right_to_wrong': sum(x['control_correct'] and not x['reflection_correct'] for x in comparisons),
        'llm_operations': dict(operations), 'fresh_llm_operations': dict(fresh),
        'checks': dict(checks), 'failures': failures, 'experiment_signature': p['experiment_signature']}
    baseline.write_rows(root / 'paired_comparison.jsonl', comparisons)
    baseline.write_json(root / 'reflection_audit.json', report)
    return report


def run(root):
    p = read(root / 'protocol.json')
    for name, digest in p['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('input changed: ' + name)
    if official.verify_installation() != p['upstream']:
        raise ValueError('official runtime changed')
    source = Path(p['ablation']['control_root'])
    control = {r['id']: r for r in baseline.read_rows(source / 'qa_responses.jsonl')}
    questions = baseline.read_rows(root / 'questions.jsonl')
    model = baseline.LocalMemoryQwen(p)
    baseline.write_json(root / 'generation_environment.json', model.environment)
    broker = baseline.Broker(model, root, p, p['generation_batch_size'])
    baseline.install_runtime(broker, p)
    lock, errors = threading.Lock(), []
    done = {task: len(baseline.read_rows(root / task / 'responses.jsonl')) for task in p['tasks']}

    def progress():
        baseline.write_json(root.parent.parent / 'run_status.json', {
            'updated_at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
            'status': 'running_with_errors' if errors else 'running', 'expected_questions': 371,
            'answered': sum(done.values()), 'by_task': done, 'failed_tasks': errors,
            'reflection_enabled': True, 'semantic_top_k': 5, 'context_memory_limit': 5})

    def task_run(task):
        broker.local.owner = task
        target = root / task
        if (target / 'complete.json').exists():
            return
        memory = official.OfficialSimpleMemMemory(p, target / 'runtime')
        memory.stats = read(target / 'memory_stats.json')
        expected_entries = {e['entry_id']: e for e in read(target / 'final_memory.json')}
        actual_entries = {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()}
        if actual_entries != expected_entries or not memory.core.hybrid_retriever.enable_reflection:
            raise ValueError('restored memory or reflection setting differs')
        memory.finalized = True
        trace = {}
        instrument(memory, trace, control)
        rows = baseline.read_rows(target / 'responses.jsonl')
        answered = {r['id'] for r in rows}
        selected = [q for q in questions if q['task'] == task]
        for q in selected:
            if q['id'] in answered:
                continue
            trace.clear()
            trace.update(question_id=q['id'], phase='initial', searches=[], assessments=[], additional_queries=[])
            query = baseline.retrieval_query(q)
            context, detail = memory.recall(query)
            detail['reflection_trace'] = copy.deepcopy(trace)
            messages = baseline.reader_messages(q, context, p)
            output = broker.request(messages, p['max_new_tokens'], kind='answer', owner=q['id'])
            raw = output['raw_response'].strip()
            rows.append({**q, **output, 'predicted_choice': raw if len(raw) == 1 and raw in 'ABCD' else None,
                'retrieval_query': query, 'retrieved_context': context, 'retrieval_detail': detail,
                'reader_messages_sha256': baseline.hash_object(messages), 'experiment_signature': p['experiment_signature']})
            baseline.write_rows(target / 'responses.jsonl', rows)
            baseline.write_json(target / 'progress.json', {'phase': 'answer', 'done': len(rows), 'total': len(selected)})
            with lock:
                done[task] = len(rows)
                progress()
            print(f'{task}: {len(rows)}/{len(selected)}; reflection rounds={len(trace["assessments"])}', flush=True)
        baseline.write_json(target / 'complete.json', {'experiment_signature': p['experiment_signature'],
            'responses_sha256': baseline.sha(target / 'responses.jsonl'), 'questions': len(rows)})

    progress()
    try:
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            futures = {pool.submit(task_run, task): task for task in p['tasks']}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    with lock:
                        errors.append({'task': futures[future], 'error': str(exc), 'traceback': traceback.format_exc()})
                        baseline.write_json(root / 'run_errors.json', errors)
                        progress()
                    print(errors[-1]['traceback'], flush=True)
    finally:
        broker.close()
    args = SimpleNamespace(mode='simplemem', simplemem_backend='official_text', simplemem_context_limit=5,
                           output_root=root.parent.parent)
    baseline.analyze(args)
    if errors:
        raise RuntimeError('reflection ablation has failed tasks')
    report = audit(root)
    status = read(root.parent.parent / 'run_status.json')
    status.update(status='complete' if report['audit_passed'] else 'audit_failed',
                  correct=report['correct'], accuracy=report['accuracy'],
                  unchanged_contexts=report['unchanged_contexts'],
                  summary=str(root / 'summary.json'), audit=str(root / 'reflection_audit.json'))
    baseline.write_json(root.parent.parent / 'run_status.json', status)
    if not report['audit_passed']:
        raise RuntimeError('reflection ablation audit failed')
    print(json.dumps({k: report[k] for k in ('questions', 'correct', 'accuracy', 'unchanged_contexts', 'questions_with_added_candidates')}, ensure_ascii=False), flush=True)


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
