"""Recompute full MemEye coverage, shared protocol and metrics from audited tasks."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.audit_memeye_m2a import audit, read, rows, sha
from scripts.evaluation.m2a_inputs import load_task


def run(root):
    plan = read(root/'plan.json')
    finished = read(root/'memeye/completion_manifest.json')
    expected = plan['datasets']['memeye']
    assert finished['status'] == 'completed'
    assert finished['plan_sha256'] == sha(root/'plan.json')
    recorded = {t['task']: t for t in finished['tasks']}
    assert set(recorded) == {t['task'] for t in expected['tasks']}
    common = None
    implementation_reference = None
    all_ids, expected_ids, per_task, failures = [], [], [], []
    protocol_fields = ['construction_mode', 'extraction_prompt', 'construction_policy',
        'extraction_json_policy', 'retrieval_limits', 'memory_manager_max_tools',
        'chat_max_query_tools', 'mandatory_initial_query', 'seed', 'do_sample',
        'llm_path', 'llm_config_sha256', 'bge_path', 'bge_dimension', 'bge_query_prefix',
        'siglip', 'max_new_tokens', 'reader_max_new_tokens', 'image_max_pixels',
        'evidence_policy', 'output_length_policy']
    for task in expected['tasks']:
        item = recorded[task['task']]
        out = Path(item['output'])
        assert sha(out/'completion_manifest.json') == item['completion_sha256']
        checked = audit(out)
        assert checked['passed'], checked['errors']
        protocol = read(out/'protocol.json')
        settings = {k: protocol[k] for k in protocol_fields}
        if common is None:
            common = settings
        assert settings == common, 'Mixed experimental protocols'
        implementation = protocol['implementation_sha256']
        if implementation_reference is None:
            implementation_reference = implementation
        allowed_changes = {'scripts/memory/m2a_single_pass.py',
            'scripts/evaluation/m2a_single_pass_run.py', 'scripts/evaluation/audit_memeye_m2a.py'}
        assert {k:v for k,v in implementation.items() if k not in allowed_changes} == {
            k:v for k,v in implementation_reference.items() if k not in allowed_changes}, 'Unrelated implementation drift'
        obs, queries, _ = load_task('memeye', task['task'])
        answers, build = rows(out/'responses.jsonl'), rows(out/'construction.jsonl')
        calls, searches = rows(out/'llm_calls.jsonl'), rows(out/'searches.jsonl')
        # Prove the deterministic normalization is identical on every retained
        # old/new model output; inherited tasks already had empty no-image captions.
        from scripts.memory.m2a_single_pass import parse_extraction
        extractions = rows(out/'extractions.jsonl')
        build_calls = [c for c in calls if c['stage'] == 'construction']
        for observation, extraction, call in zip(obs, extractions, build_calls):
            assert parse_extraction(call['raw_response'], bool(observation['image'])) == {
                k:extraction[k] for k in ('facts','image_caption')}
        assert [a['id'] for a in answers] == [q['id'] for q in queries]
        assert [b['id'] for b in build] == [o['id'] for o in obs]
        all_ids += [a['id'] for a in answers]
        expected_ids += [q['id'] for q in queries]
        correct = sum(a['predicted_choice'] == q['gold_choice'] for a, q in zip(answers, queries))
        summary = read(out/'summary.json')
        row = dict(task=task['task'], rounds=len(obs), memories=summary['semantic_memories'],
            questions=len(answers), correct=correct, accuracy=correct/len(answers),
            construction_calls=sum(b['calls'] for b in build),
            construction_seconds=sum(b['elapsed_seconds'] for b in build),
            elapsed_seconds=summary['elapsed_seconds'], total_llm_calls=len(calls),
            json_brace_completions=sum(bool(b.get('json_repair')) for b in build),
            no_image_captions_discarded=sum(bool(b.get('discarded_caption_without_image')) for b in build),
            search_calls=len(searches),
            image_search_calls=sum(bool(s['kwargs'].get('query_image_path')) for s in searches),
            max_memories_per_search=max(len(s['results']) for s in searches),
            llm_errors=sum(c['status'] != 'ok' for c in calls),
            token_limit_errors=sum(c['hit_token_limit'] for c in calls),
            tool_errors=len(checked['unique_tool_failures']),
            source_hits=sum(a['gold_source_hits'] for a in answers),
            source_targets=sum(len(set(a['gold_ids'])) for a in answers),
            independent_audit_passed=True, output=str(out),
            completion_sha256=item['completion_sha256'])
        assert row['rounds'] == row['memories'] == row['construction_calls']
        assert not row['llm_errors'] and not row['token_limit_errors']
        assert row['max_memories_per_search'] <= common['retrieval_limits']['final']
        per_task.append(row)
    assert all_ids == expected_ids and len(all_ids) == len(set(all_ids)) == expected['questions'] == 371
    assert sum(t['rounds'] for t in per_task) == expected['rounds'] == 848
    correct = sum(t['correct'] for t in per_task)
    assert correct == finished['correct']
    for attempt in sorted((root/'memeye').glob('*/attempt_*')):
        if attempt.is_dir() and (attempt/'failure.json').exists():
            failures.append(dict(output=str(attempt), failure=read(attempt/'failure.json')))
    result = dict(passed=True, method='M2A retrieval with single-pass construction; Qwen/BGE/SigLIP2 adaptation',
        completed_tasks=len(per_task), completed_questions=len(all_ids), correct=correct,
        accuracy=correct/len(all_ids), original_rounds=848,
        construction_calls=sum(t['construction_calls'] for t in per_task),
        construction_seconds=sum(t['construction_seconds'] for t in per_task),
        total_llm_calls=sum(t['total_llm_calls'] for t in per_task),
        retrieval_limits=common['retrieval_limits'],
        source_recall_label='Union of source IDs from adaptive searches, not fixed Recall@5',
        source_hits=sum(t['source_hits'] for t in per_task),
        source_targets=sum(t['source_targets'] for t in per_task),
        shared_protocol_verified=True, exact_qa_coverage=True, per_task=per_task,
        failed_attempts_excluded=failures, auditor_sha256=sha(__file__),
        suite_completion_sha256=sha(root/'memeye/completion_manifest.json'),
        parallel_execution_sha256=sha(root/'parallel_execution.json') if (root/'parallel_execution.json').exists() else None)
    (root/'full_memeye_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    with (root/'per_task_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader()
        writer.writerows(per_task)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))
