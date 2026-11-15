"""Audit a real official-SimpleMem run, including the exact Qwen answer payloads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.memory import simplemem_official as official
from scripts.memory import text_memory_baselines as baseline


def audit(root):
    def read(path):
        return json.loads(path.read_text())

    failures = []
    checks = Counter()

    def check(condition, name, detail=''):
        checks[name] += 1
        if not condition:
            failures.append({'check': name, 'detail': str(detail)})

    protocol = read(root / 'protocol.json')
    signature = protocol['experiment_signature']
    check(baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})
          == signature, 'protocol_signature')
    check(official.verify_installation() == protocol['upstream'], 'pinned_official_runtime')
    for path, digest in protocol['input_hashes'].items():
        check(baseline.sha(Path(path)) == digest, 'input_hashes', path)
    questions = baseline.read_rows(root / 'questions.jsonl')
    observations = read(root / 'observations.json')
    check(baseline.hash_object(questions) == protocol['prepared_questions_sha256'], 'questions_hash')
    check(baseline.hash_object(observations) == protocol['observations_sha256'], 'observations_hash')
    check(len(questions) == len({q['id'] for q in questions}) == 371, 'full_benchmark_371_ids')
    check(sum(map(len, observations.values())) == protocol['memory_rounds'] == 848, 'full_history_848_rounds')
    defaults = protocol['method_defaults']
    check(defaults['SEMANTIC_TOP_K'] == 5 and defaults['ENABLE_REFLECTION'] is False
          and defaults['ENABLE_PLANNING'] is True, 'requested_retrieval_settings')
    check(protocol['context_memory_limit'] == 5 and protocol['preserve_empty_context'], 'final_context_policy')

    calls = {}
    operations = Counter()
    token_totals = Counter()
    for path in sorted((root / 'calls').glob('*.json')):
        call = read(path)
        request = call['request']
        check(baseline.hash_object(request) == path.stem, 'call_request_hash', path.name)
        check(request['experiment_signature'] == signature, 'call_signature', path.name)
        if 'inherited_from' in call:
            provenance = call['inherited_from']
            original_path = Path(provenance['path'])
            check(baseline.sha(original_path) == provenance['sha256'], 'inherited_call_source_hash', path.name)
            original = read(original_path)
            original['request']['experiment_signature'] = signature
            check(original['request'] == request and original['output'] == call['output'],
                  'inherited_call_payload_unchanged', path.name)
        calls[path.stem] = call
        systems = '\n'.join(m['content'] for m in request['messages'] if m['role'] == 'system')
        if request['kind'] == 'answer':
            operation = 'answer'
        elif 'information extraction assistant' in systems:
            operation = 'construction'
        elif 'information requirement analyst' in systems:
            operation = 'information_plan'
        elif 'query generation specialist' in systems:
            operation = 'search_query_plan'
        elif 'query analysis assistant' in systems:
            operation = 'structured_query_analysis'
        else:
            operation = 'unexpected_memory_operation'
        operations[operation] += 1
        check(operation != 'unexpected_memory_operation', 'only_expected_llm_operations', path.name)
        check(not any(marker in systems for marker in (
            'information completeness evaluator', 'missing information query generator',
            'information adequacy evaluator', 'search strategy assistant')), 'no_reflection_llm_calls', path.name)
        token_totals[request['kind'] + '_input_tokens'] += call['output']['input_tokens']
        token_totals[request['kind'] + '_generated_tokens'] += call['output']['generated_tokens']
        token_totals[request['kind'] + '_token_limit_hits'] += int(call['output']['hit_token_limit'])

    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    expected = {q['id']: q for q in questions}
    rows, task_stats, completed = [], {}, []
    for task in protocol['tasks']:
        target = root / task
        resolved_path = target / 'runtime/resolved_config.json'
        if resolved_path.exists():
            resolved = read(resolved_path)
            check(resolved['settings'] == defaults, 'resolved_task_settings', task)
            check(resolved['embedding_dimension'] == 384, 'resolved_embedding_dimension', task)
        path = target / 'responses.jsonl'
        task_rows = baseline.read_rows(path) if path.exists() else []
        rows.extend(task_rows)
        task_stats[task] = {'answered': len(task_rows),
                            'expected': sum(q['task'] == task for q in questions),
                            'correct': sum(r['predicted_choice'] == r['gold_choice'] for r in task_rows)}
        checkpoint = target / 'complete.json'
        if checkpoint.exists():
            completed.append(task)
            value = read(checkpoint)
            check(value['experiment_signature'] == signature and value['responses_sha256'] == baseline.sha(path),
                  'completed_task_checkpoint', task)
            check(len(task_rows) == task_stats[task]['expected'] == value['questions'], 'completed_task_count', task)
        memory_path = target / 'final_memory.json'
        memories = {e['entry_id']: e for e in read(memory_path)} if memory_path.exists() else {}
        if memory_path.exists():
            sources = read(target / 'runtime/dialogue_sources.json')
            windows = read(target / 'runtime/construction_windows.json')
            stats = read(target / 'memory_stats.json')
            check(stats['stored_rounds'] == len(observations[task]), 'all_history_ingested', task)
            check(stats['memory_entries'] == len(memories), 'memory_entry_count', task)
            check({i for w in windows for i in w['dialogue_ids']} == {int(i) for i in sources},
                  'construction_covers_all_dialogues', task)
            check({i for w in windows for i in w['source_round_ids']} == {r['id'] for r in observations[task]},
                  'construction_covers_all_rounds', task)
            check(all(w['parsed_successfully'] for w in windows), 'construction_parse_success', task)
            check({i for w in windows for i in w['entry_ids']} == set(memories), 'construction_entry_provenance', task)
            task_stats[task].update(memory_entries=len(memories), construction_windows=len(windows))
        for row in task_rows:
            identity = row['id']
            q = expected[identity]
            check(all(row[k] == v for k, v in q.items()), 'source_question_fields', identity)
            check(row['experiment_signature'] == signature, 'response_signature', identity)
            check(row['retrieval_query'] == baseline.retrieval_query(q), 'answer_free_retrieval_query', identity)
            detail = row['retrieval_detail']
            ids = detail['context_memory_ids']
            original_ids = list(dict.fromkeys(detail['original_item_ids']))
            check(ids == original_ids[:5] and len(ids) == len(set(ids)), 'first_five_distinct_entries', identity)
            check(len(ids) == 5 or (not original_ids and detail['source_empty_preserved']), 'actual_context_count', identity)
            check(len(detail['items']) == detail['context_memory_count'] == len(ids), 'context_detail_count', identity)
            check(all(i in memories for i in original_ids), 'retrieval_ids_in_task_memory', identity)
            for item, entry_id in zip(detail['items'], ids):
                check(item == dict(memories[entry_id], id=entry_id), 'whole_memory_entry_unchanged', identity)
            context = formatter._format_contexts([MemoryEntry(**memories[i]) for i in ids])
            check(row['retrieved_context'] == context, 'actual_formatted_context', identity)
            messages = baseline.reader_messages(q, context, protocol)
            check(row['reader_messages_sha256'] == baseline.hash_object(messages), 'reader_payload_hash', identity)
            call = calls[row['call_key']]
            check(call['request']['messages'] == messages, 'actual_qwen_reader_payload', identity)
            check(call['request']['kind'] == 'answer' and call['request']['temperature'] == 0
                  and call['request']['max_new_tokens'] == 16 and call['request']['seed'] == 42,
                  'common_reader_generation_settings', identity)
            check(all(row[k] == v for k, v in call['output'].items()), 'response_matches_llm_output', identity)
            check(row['predicted_choice'] in ('A', 'B', 'C', 'D'), 'answer_format', identity)

    check(len(rows) == len({r['id'] for r in rows}), 'response_id_uniqueness')
    check({r['id'] for r in rows} <= set(expected), 'no_unexpected_question_ids')
    if 'recovery' in protocol:
        recovery = protocol['recovery']
        source = Path(recovery['source_root'])
        check(baseline.sha(source / 'protocol.json') == recovery['source_protocol_sha256'],
              'recovery_parent_protocol_hash')
        provenance = read(root / 'recovery_provenance.json')
        inherited_count = 0
        current = {r['id']: r for r in rows}
        for task, record in provenance.items():
            check(baseline.sha(source / task / 'responses.jsonl') == record['source_responses_sha256'],
                  'recovery_parent_responses_hash', task)
            check(baseline.sha(root / task / 'final_memory.json') == record['source_memory_sha256'],
                  'recovery_memory_unchanged', task)
            for old in baseline.read_rows(source / task / 'responses.jsonl'):
                new = current[old['id']]
                ignored = {'call_key', 'experiment_signature'}
                check({k: v for k, v in old.items() if k not in ignored}
                      == {k: v for k, v in new.items() if k not in ignored},
                      'inherited_answer_and_retrieval_unchanged', old['id'])
                inherited_count += 1
        check(inherited_count == recovery['inherited_answers'], 'inherited_answer_count')
    complete = set(completed) == set(protocol['tasks']) and {r['id'] for r in rows} == set(expected)
    summary_path = root / 'summary.json'
    if complete and summary_path.exists():
        summary = read(summary_path)
        check(summary['status'] == 'complete' and summary['questions'] == len(rows)
              and summary['correct'] == sum(r['predicted_choice'] == r['gold_choice'] for r in rows),
              'recomputed_summary_accuracy')
    errors_path = root / 'run_errors.json'
    run_errors = read(errors_path) if errors_path.exists() else []
    check(not run_errors, 'no_task_errors', run_errors)
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root.resolve()),
              'status': 'complete' if complete else 'in_progress', 'audit_passed_so_far': not failures,
              'experiment_signature': signature, 'expected_questions': len(questions), 'answered': len(rows),
              'correct': sum(r['predicted_choice'] == r['gold_choice'] for r in rows),
              'completed_tasks': completed, 'by_task': task_stats, 'llm_operations': dict(operations),
              'token_totals': dict(token_totals), 'context_counts': dict(Counter(str(len(r['retrieval_detail']['items'])) for r in rows)),
              'check_counts': dict(checks), 'failures': failures}
    result['inherited_llm_calls'] = sum('inherited_from' in c for c in calls.values())
    result['fresh_llm_calls'] = len(calls) - result['inherited_llm_calls']
    result['query_analysis_format_repairs'] = [dict(item, task=task)
        for task in protocol['tasks']
        if (root / task / 'runtime/query_analysis_repairs.json').exists()
        for item in read(root / task / 'runtime/query_analysis_repairs.json')]
    baseline.write_json(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='directory containing protocol.json')
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps({k: report[k] for k in ('status', 'audit_passed_so_far', 'answered', 'correct', 'failures')}, ensure_ascii=False))
    raise SystemExit(0 if report['audit_passed_so_far'] else 1)
