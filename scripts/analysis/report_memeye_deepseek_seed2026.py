"""Independently check seed2026 results; never run inference or fill pending scores."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'results/memeye_deepseek_seed2026_full_20261113'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
OPEN_GOLD = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/gold.jsonl'
REFERENCE = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/comparison.json'
METHODS = ('semrag', 'ours_1_ours_2', 'gen_agents', 'memoryos', 'simplemem', 'a_mem', 'clip_image_dialogue')
LABELS = dict(zip(METHODS, ('Semantic RAG (caption baseline)', 'ours_1 + ours_2 (lambda=0.03, alpha=1)',
    'Generative Agents (Top10)', 'MemoryOS', 'SimpleMem (final context <=5)', 'A-Mem (final context <=5)',
    'VisRAG (CLIP -> original images + historical dialogue)')))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(values):
    for name, expected in values.items():
        require(Path(name).is_absolute() and sha(name) == expected, 'Hash differs: ' + name)


def indexed(values):
    result = {r['id']: r for r in values}
    require(len(values) == len(result) == 371, 'Require 371 distinct records')
    return result


def audit_method(output, method, canonical):
    """Recompute from records, then check their native evidence and completion seal."""
    base = output / 'evaluations' / method
    marker = output / 'methods' / method / 'completion.json'
    if not marker.exists():
        return {'method': method, 'status': 'pending'}
    completion, metrics = read(marker), read(base / 'metrics.json')
    require(completion.get('status') == metrics.get('status') == 'complete', method + ': incomplete seal')
    if 'metrics_sha256' in completion:
        require(completion['metrics_sha256'] == sha(base / 'metrics.json'), method + ': metric seal differs')
    hashes(completion.get('output_sha256', {}))
    hashes(completion.get('artifact_sha256', {}))
    require((metrics['seed'], metrics['caption_seed'], metrics['judge_seed'], metrics['model']) ==
            (2026, 42, 42, 'deepseek-v4.1-flash'), method + ': seed/backbone differs')
    require(metrics['questions'] == metrics['open_questions'] == metrics['judge_questions'] == 371,
            method + ': coverage counter differs')
    require(bool(metrics.get('source_sha256')), method + ': missing native evidence')
    hashes(metrics['source_sha256'])
    for filename, field in (('mcq_records.jsonl', 'mcq_records_sha256'), ('open_records.jsonl', 'open_records_sha256')):
        actual = sha(base / filename)
        pinned = (metrics['source_sha256'].get(str(base / filename))
                  or metrics.get(field) or completion.get('output_sha256', {}).get(str(base / filename)))
        require(pinned == actual, method + ': uniform records are not sealed')
    mcq, opened = indexed(rows(base / 'mcq_records.jsonl')), indexed(rows(base / 'open_records.jsonl'))
    require(set(mcq) == set(opened) == set(canonical), method + ': canonical identities differ')
    for identity, row in mcq.items():
        source = canonical[identity]
        require(all(row[k] == source[k] for k in ('task', 'question', 'gold_choice')),
                method + ': question/gold changed ' + identity)
        require(row['predicted_choice'] in ('A', 'B', 'C', 'D', None), 'Invalid choice encoding')
        correct = row['predicted_choice'] == source['gold_choice']
        require(type(row['correct']) is bool and row['correct'] == correct, 'MCQ correctness differs')
        require('raw_response' in row and 'native_outcome' in row, 'Native MCQ outcome missing')
    allowed = {0., .25, .5, .75, 1.}
    gold_open = indexed(rows(OPEN_GOLD))
    require(set(gold_open) == set(canonical), 'Canonical Open cohort differs')
    for identity, row in opened.items():
        require(row['task'] == canonical[identity]['task'], 'Open task differs')
        # The Open wording can differ from its paired MCQ question; it is audited natively.
        require(isinstance(row['model_output'], str) and isinstance(row['ground_truth'], str),
                'Open output/reference must be preserved as text')
        require(row['judge_score'] in allowed, 'Judge is not a five-grade score')
        require(row['ground_truth'] == gold_open[identity]['answer'], 'Canonical Open reference differs')
        require(row.get('judge_seed', 42) == 42, 'Judge seed differs')
    correct = sum(r['correct'] for r in mcq.values())
    score = sum(r['judge_score'] for r in opened.values()) / 371
    require(metrics['correct'] == correct and math.isclose(metrics['accuracy'], correct / 371, abs_tol=1e-12),
            method + ': MCQ aggregate differs')
    require(math.isclose(metrics['llm_judge'], score, abs_tol=1e-12), method + ': Judge mean differs')
    artifacts = {str(p): sha(p) for p in (marker, base / 'metrics.json', base / 'mcq_records.jsonl', base / 'open_records.jsonl')}
    return dict(method=method, status='complete', correct=correct, questions=371, accuracy=correct / 371,
        llm_judge=score, judge_distribution=dict(Counter(str(r['judge_score']) for r in opened.values())),
        native_outcomes=dict(Counter(str(r['native_outcome']) for r in mcq.values())),
        empty_open_answers=sum(not r['model_output'].strip() for r in opened.values()),
        source_files_checked=len(metrics['source_sha256']), artifact_sha256=artifacts)


def concurrency_audit(output):
    path = output / 'scheduler/events.jsonl'
    if not path.exists():
        return {'status': 'pending', 'maximum_active_methods': 0}
    active, maximum = set(), 0
    for event in rows(path):
        if event['event'] == 'method_started':
            require(event['method'] not in active, 'Method launched twice concurrently')
            active.add(event['method'])
            maximum = max(maximum, len(active))
            require(len(active) <= 2, 'More than two methods active')
        elif event['event'] in ('method_finished', 'method_interrupted'):
            require(event['method'] in active, 'Method finish lacks matching launch')
            active.remove(event['method'])
    return {'status': 'passed', 'maximum_active_methods': maximum, 'active_methods': sorted(active),
            'events_sha256': sha(path)}


def export(output=DEFAULT, require_complete=False):
    output = Path(output).resolve()
    canonical = indexed(rows(BENCHMARK))
    reference = read(REFERENCE)
    require(reference['status'] == 'complete', 'Historical comparison incomplete')
    historical = {r['method']: r for r in reference['methods']}
    results = [audit_method(output, method, canonical) for method in METHODS]
    parallel = concurrency_audit(output)
    ready = (all(r['status'] == 'complete' for r in results) and parallel['status'] == 'passed'
             and not parallel.get('active_methods'))
    if require_complete:
        require(ready, 'All seven methods must pass independent coverage checks')
    plan = output / 'plan.json'
    if plan.exists():
        hashes(read(plan)['source_sha256'])
    for path in (output / 'scheduler/method_drivers').glob('*.json'):
        driver = read(path)
        require(driver['status'] == 'ready' and sha(driver['path']) == driver['sha256'], 'Reviewed method driver changed')
    comparison = []
    paired = []
    # The original seed42 parser is read-only and is used only after a new method finishes.
    for item in results:
        method, old = item['method'], historical[item['method']]
        row = dict(method=method, label=LABELS[method], status=item['status'], questions=371,
            seed42_correct=old['seed42_correct'], seed1024_correct=old['seed1024_correct'],
            seed2026_correct=item.get('correct'), seed42_accuracy=old['seed42_accuracy'],
            seed1024_accuracy=old['seed1024_accuracy'], seed2026_accuracy=item.get('accuracy'),
            seed42_llm_judge=old['seed42_llm_judge'], seed1024_llm_judge=old['seed1024_llm_judge'],
            seed2026_llm_judge=item.get('llm_judge'))
        if item['status'] == 'complete':
            import sys
            sys.path.insert(0, str(ROOT))
            from scripts.analysis.audit_memeye_deepseek_seed1024 import read_mcq, prediction
            prior = indexed(read_mcq(method, reference=True))
            current = indexed(rows(output / 'evaluations' / method / 'mcq_records.jsonl'))
            require(set(prior) == set(current), 'Paired seed42 cohort differs')
            transitions = Counter()
            changed = 0
            for identity, now in current.items():
                before = prediction(method, prior[identity])
                before_correct = before == canonical[identity]['gold_choice']
                # Original VisRAG responses preserve the prediction, but have no
                # precomputed correctness field. Always derive it from gold.
                if 'correct' in prior[identity]:
                    require(bool(prior[identity]['correct']) == before_correct, 'Historical prediction differs')
                transitions[(before_correct, now['correct'])] += 1
                changed += before != now['predicted_choice']
                paired.append(dict(method=method, id=identity, gold_choice=now['gold_choice'],
                    seed42_prediction=before, seed2026_prediction=now['predicted_choice'],
                    seed42_correct=before_correct, seed2026_correct=now['correct']))
            require(sum(k[0] * n for k, n in transitions.items()) == old['seed42_correct'],
                    'Historical correct total differs')
            row.update(accuracy_delta_pp=(item['accuracy'] - old['seed42_accuracy']) * 100,
                changed_answers_vs42=changed, both_correct=transitions[(True, True)],
                seed42_only_correct=transitions[(True, False)], seed2026_only_correct=transitions[(False, True)],
                both_wrong=transitions[(False, False)], native_outcomes=item['native_outcomes'],
                empty_open_answers=item['empty_open_answers'])
        comparison.append(row)
    result = dict(status='complete' if ready else 'partial', dataset='MemEye', model='deepseek-v4.1-flash',
        seed=2026, caption_seed=42, judge_seed=42, methods_complete=sum(r['status'] == 'complete' for r in results),
        requested_methods=7, questions_per_method=371, concurrency=parallel, methods=comparison)
    dump(output / 'comparison.json', result)
    fields = list(dict.fromkeys(k for r in comparison for k in r))
    csv_path = output / 'comparison.csv'
    csv_tmp = csv_path.with_name(csv_path.name + f'.{os.getpid()}.tmp')
    with csv_tmp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
                          for k, v in r.items()} for r in comparison])
    csv_tmp.replace(csv_path)
    dump(output / 'audit/independent_results.json', dict(status='passed' if ready else 'partial',
        methods=results, concurrency=parallel, source_sha256={str(BENCHMARK): sha(BENCHMARK),
        str(REFERENCE): sha(REFERENCE), str(OPEN_GOLD): sha(OPEN_GOLD), str(Path(__file__)): sha(__file__)}))
    dump(output / 'audit/paired_mcq_vs42.json', dict(status='complete' if ready else 'partial', records=paired))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = export(args.run_root, args.require_complete)
    print(json.dumps({k: result[k] for k in ('status', 'methods_complete', 'requested_methods')}, ensure_ascii=False))
