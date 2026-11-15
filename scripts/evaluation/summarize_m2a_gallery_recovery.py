"""Combine immutable original outputs and construction retries, retaining the bad source QA."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.audit_memeye_m2a import audit, read, rows, sha
from scripts.evaluation.m2a_inputs import load_task, task_names, gallery_reader
from scripts.evaluation.m2a_suite import write, now


def summarize(original, retry):
    state = read(retry/'memgallery/status.json')
    overrides = {t['task']:Path(t['output']) for t in state['tasks'] if t['status'] == 'passed'}
    source_state = read(original/'memgallery/status.json')
    old = {t['task']:Path(t['output']) for t in source_state['tasks']}
    scorer = gallery_reader().score_answer
    totals, by_type = Counter(), defaultdict(Counter)
    task_results, failures, known_issues = [], [], []
    expected_questions = expected_rounds = 0
    for task in task_names('memgallery'):
        obs, labels, _ = load_task('memgallery', task)
        expected_questions += len(labels)
        expected_rounds += len(obs)
        out = overrides.get(task, old[task])
        if not (out/'completion_manifest.json').exists():
            failures.append(dict(task=task, error='No complete output', output=str(out)))
            continue
        checked = audit(out)
        if read(out/'evaluation_labels.json') != labels or read(out/'observations.json') != obs:
            failures.append(dict(task=task, error='Original inputs or labels differ'))
            continue
        answers = rows(out/'responses.jsonl')
        if [a['id'] for a in answers] != [q['id'] for q in labels]:
            failures.append(dict(task=task, error='Question coverage differs'))
            continue
        no_search = [a['id'] for a in answers if not a['search_calls']]
        known_id = 'Music_Dance_Theater_Performance_Learning/QA_0048'
        known = (checked['errors'] == ['Some questions did not search memory'] and no_search == [known_id]
                 and labels[48]['question'] == '2024-11-13'
                 and labels[48]['answer'] == 'When did the user sign up for a dance training class? Return in the format YYYY-MM-DD.')
        if known:
            known_issues.append(dict(id=known_id, issue='Original source has question and answer reversed; no search occurred',
                original_question=labels[48]['question'], original_answer=labels[48]['answer'],
                handling='Original record and generated answer retained in the full benchmark denominator; not marked as audit-passing'))
        elif not checked['passed']:
            failures.append(dict(task=task, error='Task audit failed', details=checked['errors']))
            continue
        scores = Counter()
        for answer, label in zip(answers, labels):
            score = scorer(answer['raw_answer'], label['answer'])
            scores.update(score)
            by_type[label['point']].update(score)
            by_type[label['point']]['questions'] += 1
        totals.update(scores)
        totals['questions'] += len(answers)
        totals['rounds'] += len(obs)
        search_turns = Counter()
        for c in rows(out/'llm_calls.jsonl'):
            totals['llm_calls'] += 1
            if c['stage'] == 'construction':
                totals['construction_calls'] += 1
            if any(t['name'] == 'search_semantic_memories' for t in c['tool_calls']):
                search_turns[c['sample_id']] += 1
        totals['questions_with_multiple_search_turns'] += sum(n > 1 for n in search_turns.values())
        totals['questions_without_search'] += len(no_search)
        totals['strictly_audited_tasks'] += int(checked['passed'])
        summary = read(out/'summary.json')
        task_results.append(dict(task=task, questions=len(answers), correct=int(scores['em']),
            strict_audit_passed=checked['passed'], known_source_issue=known, output=str(out),
            metrics=summary['metrics'], completion_sha256=sha(out/'completion_manifest.json')))
    complete = totals['questions'] == expected_questions and totals['rounds'] == expected_rounds and not failures
    keys = ('em','f1','bleu','bleu_1','bleu_2')
    result = dict(created_at=now(), status=('all_outputs_available_with_known_source_issue' if known_issues else 'complete')
                  if complete else 'incomplete', all_original_questions_have_outputs=complete,
        strict_audit_passed=complete and not known_issues, expected_questions=expected_questions,
        expected_rounds=expected_rounds, totals=dict(totals), extra_construction_calls=totals['construction_calls']-totals['rounds'],
        metrics={k:totals[k]/totals['questions'] for k in keys} if totals['questions'] else {},
        known_source_issues=known_issues, failures=failures, tasks=task_results,
        by_type={p:dict(questions=c['questions'],metrics={k:c[k]/c['questions'] for k in keys}) for p,c in by_type.items()},
        accounting='Counts include accepted original tasks and retry tasks; failed prior attempts and smoke tests excluded from scores and accepted-run call counts',
        original_plan_sha256=sha(original/'plan.json'), retry_plan_sha256=sha(retry/'plan.json'))
    write(retry/'recovery_summary.json',result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--retry',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    args = p.parse_args()
    if args.wait:
        deadline = time.monotonic() + 14400
        while read(args.retry/'memgallery/status.json')['status'] == 'running':
            if time.monotonic() > deadline:
                raise TimeoutError('Retry coordinator did not finish within four hours')
            time.sleep(15)
    result = summarize(args.original.resolve(),args.retry.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','by_type')},indent=2))
