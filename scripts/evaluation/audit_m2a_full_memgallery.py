"""Independently audit full Gallery coverage and recompute pooled open-answer scores."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.audit_memeye_m2a import audit, read, rows, sha
from scripts.evaluation.m2a_inputs import load_task, task_names, gallery_reader
from scripts.evaluation.m2a_suite import write


def audit_full(root):
    plan = read(root/'plan.json')
    state = read(root/'memgallery/status.json')
    expected_tasks = task_names('memgallery')
    entries = [x for x in state['tasks'] if x['status'] == 'passed']
    errors, totals, by_type, task_results = [], Counter(), defaultdict(Counter), []
    names = [x['task'] for x in entries]
    if len(names) != len(set(names)) or set(names) != set(expected_tasks):
        errors.append('Missing or duplicate tasks')
    scorer = gallery_reader().score_answer
    for entry in entries:
        out = Path(entry['output'])
        checked = audit(out)
        if not checked['passed'] or not checked['is_complete_task']:
            errors.append('Task audit failed: ' + entry['task'])
        if sha(out/'completion_manifest.json') != entry['completion_sha256']:
            errors.append('Task completion hash changed: ' + entry['task'])
        observations, labels, _ = load_task('memgallery', entry['task'])
        if observations != read(out/'observations.json') or labels != read(out/'evaluation_labels.json'):
            errors.append('Task inputs/labels differ from full original data: ' + entry['task'])
        answers = rows(out/'responses.jsonl')
        if [a['id'] for a in answers] != [q['id'] for q in labels]:
            errors.append('Question ID coverage differs: ' + entry['task'])
        task_scores = Counter()
        for answer, label in zip(answers, labels):
            score = scorer(answer['raw_answer'], label['answer'])
            task_scores.update(score)
            by_type[label['point']].update(score)
            by_type[label['point']]['questions'] += 1
        totals.update(task_scores)
        totals['questions'] += len(answers)
        totals['rounds'] += len(observations)
        totals['construction_calls'] += sum(c['stage'] == 'construction' for c in rows(out/'llm_calls.jsonl'))
        task_results.append(dict(task=entry['task'], questions=len(answers), rounds=len(observations),
            metrics={k:v/len(answers) for k,v in task_scores.items()},
            completion_sha256=entry['completion_sha256'], output=str(out)))
    expected = plan['datasets']['memgallery']
    if totals['questions'] != expected['questions'] or totals['rounds'] != expected['rounds']:
        errors.append('Aggregate coverage differs from frozen plan')
    if totals['construction_calls'] != totals['rounds']:
        errors.append('Construction is not one call per original round')
    for path, expected_hash in plan['implementation_sha256'].items():
        snapshot = root/'code_snapshot'/Path(path).relative_to(ROOT)
        if not snapshot.is_file() or sha(snapshot) != expected_hash:
            errors.append('Changed code snapshot: ' + path)
    keys = ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')
    result = dict(passed=not errors, errors=errors, tasks=task_results,
        completed_tasks=len(entries), questions=totals['questions'], rounds=totals['rounds'],
        construction_calls=totals['construction_calls'], correct=int(totals['em']),
        metrics={k:totals[k]/totals['questions'] for k in keys} if totals['questions'] else {},
        by_type={p:dict(questions=s['questions'], correct=int(s['em']),
                       metrics={k:s[k]/s['questions'] for k in keys}) for p,s in sorted(by_type.items())},
        metric_aggregation='Sum of per-question scores divided by number of questions; no LLM judge',
        plan_sha256=sha(root/'plan.json'))
    write(root/'full_memgallery_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    result = audit_full(parser.parse_args().output.resolve())
    print({k:v for k,v in result.items() if k not in ('tasks', 'by_type')})
    raise SystemExit(0 if result['passed'] else 1)
