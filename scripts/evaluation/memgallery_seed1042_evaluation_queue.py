"""Durable Judge/report queue, independent of the serial MemGallery GPU queue.

The original answers are copied verbatim into independent Judge inputs. Only EM
metadata is made a strict integer. Missing generations or grades never become0.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memgallery_seed1042_judge as judge
from scripts.evaluation.mem_gallery_master_reader import score_answer
from scripts.analysis import report_memgallery_seed1042_stability as report
from scripts.analysis.prepare_memgallery_seed1042_reference import METHOD_IDS, LABELS, SOURCE

RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
JUDGE = ROOT / 'scripts/evaluation/memgallery_seed1042_judge.py'
REPORT = ROOT / 'scripts/analysis/report_memgallery_seed1042_stability.py'
PYTHON = '/usr/bin/python3'
INTERVAL = 15


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value, frozen=False):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if frozen and path.exists():
        if path.read_text() != raw:
            raise ValueError('Frozen evaluation artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def freeze_rows(path, values):
    path = Path(path)
    raw = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in values)
    if path.exists():
        if path.read_text() != raw:
            raise ValueError('Frozen Judge input differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def unique(values):
    result = {r['id']: r for r in values}
    if len(values) != len(result):
        raise ValueError('Duplicate evaluation question IDs')
    return result


def verify_hashes(values):
    if not isinstance(values, dict) or not values:
        raise ValueError('Nonempty source_sha256 provenance is required')
    checked = {}
    for name, expected in values.items():
        path = Path(name)
        if not path.is_absolute() or sha(path) != expected:
            raise ValueError('Evaluation source hash differs: ' + str(path))
        checked[str(path)] = expected
    return checked


def canonical():
    base = SOURCE / 'inputs'
    shared = RUN / 'inputs'
    if (shared / 'questions.jsonl').exists() and (shared / 'gold.jsonl').exists():
        for name in ('questions.jsonl', 'gold.jsonl'):
            if sha(shared / name) != sha(base / name):
                raise ValueError('Seed1042 canonical questions/gold differ from the fixed seed42 cohort')
    questions, gold = unique(rows(base / 'questions.jsonl')), unique(rows(base / 'gold.jsonl'))
    if len(questions) != 1711 or set(questions) != set(gold):
        raise ValueError('Canonical cohort must contain1711 distinct questions')
    result = {qid: {**{k: q[k] for k in ('id', 'task', 'point', 'question')},
                    'ground_truth': gold[qid]['answer']} for qid, q in questions.items()}
    return result, {str(base / name): sha(base / name) for name in ('questions.jsonl', 'gold.jsonl')}


def normalized_records(values, canon, expected, seed):
    by_id = unique(values)
    if len(values) != 1711 or set(by_id) != set(canon):
        raise ValueError('Method does not cover the canonical1711-question cohort')
    output, scored = [], []
    for qid, question in canon.items():
        row = by_id[qid]
        if any(row.get(k) != v or not isinstance(row.get(k), str) for k, v in question.items()):
            raise ValueError('Method changed canonical question/gold metadata: ' + qid)
        if not isinstance(row.get('model_output'), str):
            raise ValueError('Original model_output must be text')
        for key in ('source_seed', 'answer_seed'):
            if key in row and row[key] != seed:
                raise ValueError('Source answer seed differs')
        metric = score_answer(row['model_output'], question['ground_truth'])
        em = int(judge.normalize_answer(row['model_output']) == judge.normalize_answer(question['ground_truth']))
        if int(metric['em']) != em:
            raise ValueError('Official scorer and Judge EM metadata disagree')
        for key in ('em', 'f1', 'bleu_1'):
            if key in row and (not isinstance(row[key], (int, float)) or not math.isfinite(row[key])
                               or abs(row[key] - metric[key]) > 1e-12):
                raise ValueError('Source per-question metric differs: ' + key)
        source_id = row.get('source_record_id', qid)
        if not isinstance(source_id, str) or not source_id:
            raise ValueError('Invalid original source_record_id')
        output.append({**row, 'em': em, 'source_record_id': source_id,
                       'source_hit_token_limit': row.get('source_hit_token_limit')})
        scored.append(metric)
    actual = {'correct': sum(int(r['em']) for r in scored),
              **{k: sum(r[k] for r in scored) / 1711 for k in ('em', 'f1', 'bleu_1')}}
    if type(expected['correct']) is not int or actual['correct'] != expected['correct']:
        raise ValueError('Aggregate correct count does not reproduce')
    for key in ('em', 'f1', 'bleu_1'):
        if not isinstance(expected[key], (int, float)) or not math.isfinite(expected[key]) or abs(actual[key] - expected[key]) > 1e-12:
            raise ValueError('Aggregate metric does not reproduce: ' + key)
    return output, actual


def reference_ready():
    path = RUN / 'reference_ready.json'
    if not path.exists():
        return False
    marker = read(path)
    if marker.get('status') != 'complete':
        return False
    if marker.get('methods') != list(METHOD_IDS) or marker.get('questions_per_method') != 1711:
        raise ValueError('Reference completion covers a different cohort')
    verify_hashes(marker['output_sha256'])
    return True


def upstream_seal(mid):
    if mid in ('semrag', 'ours_1', 'ours_1_ours_2'):
        return RUN / 'caption_methods/stages/audit.json', 'outputs_sha256'
    if mid in ('clip_image_dialogue', 'hippomm'):
        return RUN / 'visual_methods' / mid / 'completion.json', None
    return RUN / 'text_methods' / mid / 'completion_manifest.json', None


def generation_ready(mid):
    metric_path = RUN / 'evaluations' / mid / 'metrics.json'
    records_path = metric_path.with_name('records.jsonl')
    marker, _ = upstream_seal(mid)
    if not all(p.exists() for p in (metric_path, records_path, marker)):
        return False
    seal, metric = read(marker), read(metric_path)
    if seal.get('status') not in ('complete', 'passed'):
        return False
    return metric.get('status') == 'complete'


def prepare_input(mid, seed=1042):
    canon, canonical_hashes = canonical()
    if seed == 42:
        if mid != 'ours_1' or not reference_ready():
            raise ValueError('Only the completed frozen seed42 ours_1 row is judged here')
        reference = read(RUN / 'reference_seed42.json')
        metric = next(m for m in reference['methods'] if m['id'] == mid)
        if metric['seed'] != 42 or metric['correct'] != 798:
            raise ValueError('Wrong original ours_1 row')
        source_path = Path(metric['records_path'])
        if sha(source_path) != metric['records_sha256']:
            raise ValueError('Frozen seed42 source records changed')
        hashes = verify_hashes(metric['source_sha256'])
        hashes.update({str(p): sha(p) for p in (RUN / 'reference_seed42.json', RUN / 'reference_ready.json', source_path)})
        destination = RUN / 'judge_inputs/seed42_ours_1'
    else:
        if not generation_ready(mid):
            raise ValueError('Upstream completion seal and full metrics are required')
        metric_path = RUN / 'evaluations' / mid / 'metrics.json'
        source_path = metric_path.with_name('records.jsonl')
        metric = read(metric_path)
        if metric.get('seed') != 1042 or metric.get('questions') != 1711 or metric.get('status') != 'complete':
            raise ValueError('Wrong seed1042 metric cohort')
        hashes = verify_hashes(metric['source_sha256'])
        marker, seal_hash_field = upstream_seal(mid)
        seal = read(marker)
        expected = metric.get('records_sha256')
        if seal_hash_field:
            proof = seal[seal_hash_field]
            if proof.get(str(metric_path)) != sha(metric_path) or proof.get(str(source_path)) != sha(source_path):
                raise ValueError('Caption evaluation outputs differ from the completed audit seal')
        elif mid in ('clip_image_dialogue', 'hippomm'):
            if seal['evaluation_metrics_sha256'] != sha(metric_path) or seal['evaluation_records_sha256'] != sha(source_path):
                raise ValueError('Visual evaluation completion hashes differ')
        else:
            if seal['metrics_sha256'] != sha(metric_path):
                raise ValueError('Text evaluation completion metric hash differs')
            expected = expected or seal['records_sha256']
        if not expected or sha(source_path) != expected:
            raise ValueError('Seed1042 evaluation records_sha256 differs')
        hashes.update({str(p): sha(p) for p in (metric_path, source_path, marker)})
        destination = RUN / 'judge_inputs' / mid
    values, actual = normalized_records(rows(source_path), canon, metric, seed)
    if any(type(row['em']) is not int for row in values):
        raise ValueError('Judge EM must be a strict integer')
    copied = destination / 'records.jsonl'
    freeze_rows(copied, values)
    hashes.update(canonical_hashes)
    hashes[str(copied)] = sha(copied)
    manifest = {'questions_per_method': 1711, 'source_seed': seed, 'methods': [{
        'id': mid, 'label': LABELS[METHOD_IDS.index(mid)], 'source_seed': seed,
        'expected_em_correct': actual['correct'], **{k: actual[k] for k in ('em', 'f1', 'bleu_1')},
        'records_path': str(copied), 'records_sha256': sha(copied), 'source_sha256': hashes}]}
    path = destination / 'manifest.json'
    dump(path, manifest, frozen=True)
    return path


def verify_plan(plan, expected_sha):
    if sha(RUN / 'evaluation_plan.json') != expected_sha:
        raise ValueError('Evaluation queue plan changed after startup')
    if any(plan[k] != value for k, value in (('seed', 1042), ('reference_seed', 42), ('judge_seed', 42))):
        raise ValueError('Wrong generation/reference/Judge seeds')
    if [m['id'] for m in plan['methods']] != list(METHOD_IDS):
        raise ValueError('Plan methods differ from the fixed nine-method cohort')
    verify_hashes(plan['source_sha256'])
    gpu = read(RUN / 'plan.json')
    if sha(RUN / 'plan.json') != plan['gpu_plan_sha256']:
        raise ValueError('Pinned GPU plan changed')
    verify_hashes(gpu['source_sha256'])
    for path in (Path(__file__).resolve(), JUDGE, REPORT):
        if plan['source_sha256'].get(str(path)) != sha(path):
            raise ValueError('Evaluation plan must pin the queue, Judge and report implementations')


def judge_complete(mid, old=False):
    root = RUN / ('judge_seed42_ours_1' if old else 'judge/' + mid)
    paths = [root / 'summary.json', root / 'methods' / mid / 'summary.json',
             root / 'methods' / mid / 'per_question.jsonl', root / 'protocol.json']
    if not all(p.exists() for p in paths):
        return False
    overall, summary, protocol = read(paths[0]), read(paths[1]), read(paths[3])
    if (overall.get('status') != 'complete' or summary.get('status') != 'complete'
            or summary.get('questions_scored') != 1711 or summary.get('questions_missing') != 0
            or overall.get('total_scored_method_questions') != 1711 or overall.get('unique_requests_missing') != 0):
        return False
    values = rows(paths[2])
    if len(values) != 1711:
        return False  # analyze() publishes the method summary before its rows.
    canon, _ = canonical()
    if set(unique(values)) != set(canon):
        raise ValueError('Completed Judge rows cover a different cohort')
    manifest_path = RUN / 'judge_inputs' / ('seed42_ours_1' if old else mid) / 'manifest.json'
    source = read(manifest_path)['methods'][0]
    verify_hashes(source['source_sha256'])
    originals = unique(rows(source['records_path']))
    for key, expected in judge.judge_settings().items():
        if protocol.get(key) != expected:
            raise ValueError('Completed Judge changed the fixed evaluator settings')
    if protocol['answer_generation_seed'] != (42 if old else 1042):
        raise ValueError('Completed Judge source answer seed differs')
    requests = rows(root / 'requests.jsonl')
    saved = judge.engine.validated_results(root, requests)
    if len(saved) != len(requests):
        return False
    template = (root / 'judge_prompt.txt').read_text()
    for row in values:
        original = originals[row['id']]
        if (row.get('judge_status') != 'ok' or row.get('score') not in judge.GRADES
                or any(row[k] != original[k] for k in ('task', 'point', 'question', 'ground_truth', 'model_output', 'em', 'source_record_id'))):
            raise ValueError('Completed Judge differs from the original copied answer')
        key = judge.transport.request_sha256(judge.build_request(row, template))
        if row['request_sha256'] != key or key not in saved or row['score'] != saved[key]['score']:
            raise ValueError('Completed Judge row does not match the real scored request')
    if sum(r['score'] for r in values) != summary['score_sum'] or summary['llm_judge'] != summary['score_sum'] / 1711:
        raise ValueError('Completed Judge aggregate differs from raw grades')
    for name, expected in overall['artifact_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError('Completed Judge artifact hash differs')
    return True


def update_progress(state):
    state['updated_at_utc'] = now()
    for name, job in state['jobs'].items():
        mid, old = ('ours_1', True) if name == 'seed42_ours_1' else (name, False)
        root = RUN / ('judge_seed42_ours_1' if old else 'judge/' + mid)
        path = root / 'methods' / mid / 'summary.json'
        if path.exists():
            summary = read(path)
            job['questions_scored'] = summary.get('questions_scored', 0)
            job['questions_missing'] = summary.get('questions_missing', 1711)
            job['llm_judge'] = summary.get('llm_judge') if summary.get('status') == 'complete' else None
    dump(RUN / 'evaluation_progress.json', state)


def child(argv, name, state, plan, plan_sha):
    verify_plan(plan, plan_sha)
    path = RUN / 'logs/evaluation' / (name + '.log')
    path.parent.mkdir(parents=True, exist_ok=True)
    state['active_child'] = {'name': name, 'argv': argv, 'log': str(path), 'started_at_utc': now()}
    with path.open('ab') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        state['active_child']['pid'] = process.pid
        while process.poll() is None:
            update_progress(state)
            time.sleep(INTERVAL)
    state['active_child'].update(returncode=process.returncode, finished_at_utc=now())
    update_progress(state)
    if process.returncode:
        raise RuntimeError(f'{name} exited with code{process.returncode}; inspect {path}')


def finish(state, plan, plan_sha):
    verify_plan(plan, plan_sha)
    seal_path = RUN / 'generation_completion.json'
    if not seal_path.exists() or read(seal_path).get('status') != 'generation_complete':
        raise ValueError('A full successful GPU-generation seal is required before report export')
    if not judge_complete('ours_1', old=True) or not all(judge_complete(mid) for mid in METHOD_IDS):
        raise ValueError('Every complete1711-row Judge is required before report export')
    if not report.status(RUN)['ready_for_export']:
        raise ValueError('Independent report readiness gate is not satisfied')
    for phase in ('export', 'audit'):
        child([PYTHON, '-u', str(REPORT), '--phase', phase, '--run', str(RUN)], 'report_' + phase, state, plan, plan_sha)
    completion = read(report.OUTPUT / 'completion.json')
    audit = read(report.OUTPUT / 'audit.json')
    if completion['status'] != 'complete' or audit['status'] != 'passed' or completion['paired_questions'] != 9 * 1711:
        raise ValueError('Report independent completion/audit did not pass')
    hashes = verify_hashes(completion['output_sha256'])
    # The explicit second audit rewrites audit.json deterministically; verify all
    # report completion hashes again rather than trusting its status alone.
    for p in (RUN / 'plan.json', RUN / 'evaluation_plan.json', RUN / 'generation_completion.json', RUN / 'reference_ready.json',
              RUN / 'reference_seed42.json', report.OUTPUT / 'completion.json'):
        hashes[str(p)] = sha(p)
    for mid, old in [('ours_1', True)] + [(mid, False) for mid in METHOD_IDS]:
        root = RUN / ('judge_seed42_ours_1' if old else 'judge/' + mid)
        for filename in ('protocol.json', 'summary.json', 'requests.jsonl', 'question_map.jsonl',
                         'scores.jsonl', 'reused_scores.jsonl', 'attempts.jsonl',
                         'methods/' + mid + '/summary.json', 'methods/' + mid + '/per_question.jsonl'):
            p = root / filename
            if p.exists():
                hashes[str(p)] = sha(p)
    final = {'status': 'complete', 'seeds': [42, 1042], 'seed': 1042, 'reference_seed': 42,
        'judge_seed': 42, 'methods': list(METHOD_IDS), 'method_count': 9,
        'questions_per_method': 1711, 'seed1042_answer_and_judge_rows': 9 * 1711,
        'paired_questions': 9 * 1711, 'additional_seed42_ours_1_judge_rows': 1711,
        'independent_report_audit_passed': True, 'no_missing_score_imputation': True,
        'report_path': str(report.OUTPUT / report.WORKBOOK_NAME),
        'report_completion_path': str(report.OUTPUT / 'completion.json'),
        'source_sha256': hashes, 'completed_at_utc': now()}
    dump(RUN / 'completion.json', final)
    state.update(status='complete', active_child=None, completed_at_utc=now())
    update_progress(state)
    return 0


def run_queue():
    plan_path = RUN / 'evaluation_plan.json'
    plan, plan_sha = read(plan_path), sha(plan_path)
    verify_plan(plan, plan_sha)
    dump(RUN / 'evaluation_queue_protocol.json', {'seed': 1042, 'reference_seed': 42,
        'judge_seed': 42, 'workers': 8, 'max_attempts_per_wire': 5, 'plan_sha256': plan_sha,
        'implementation_sha256': {str(p): sha(p) for p in (Path(__file__).resolve(), JUDGE, REPORT)},
        'policy': 'Frozen original answers only; independent per-method failures; complete1711 grades and generation seal required for export.'}, frozen=True)
    path = RUN / 'evaluation_progress.json'
    state = read(path) if path.exists() else {'jobs': {}}
    state.update(status='running', pid=os.getpid(), plan_sha256=plan_sha, resumed_at_utc=now())
    attempted = set()
    names = ['seed42_ours_1', *METHOD_IDS]
    for name in names:
        state['jobs'].setdefault(name, {'status': 'pending', 'questions_expected': 1711})
    while True:
        verify_plan(plan, plan_sha)
        ready = reference_ready()
        for name in names:
            if name in attempted:
                continue
            old, mid = name == 'seed42_ours_1', 'ours_1' if name == 'seed42_ours_1' else name
            if not ready or (not old and not generation_ready(mid)):
                continue
            job = state['jobs'][name]
            attempted.add(name)  # A permanent failure is retried only after an explicit queue restart.
            try:
                verify_plan(plan, plan_sha)
                manifest = prepare_input(mid, 42 if old else 1042)
                if not judge_complete(mid, old):
                    job.update(status='running', manifest=str(manifest), manifest_sha256=sha(manifest), started_at_utc=now())
                    update_progress(state)
                    output = RUN / ('judge_seed42_ours_1' if old else 'judge/' + mid)
                    child([PYTHON, '-u', str(JUDGE), '--phase', 'run', '--manifest', str(manifest),
                           '--output', str(output), '--workers', '8', '--max-attempts', '5'], name, state, plan, plan_sha)
                if not judge_complete(mid, old):
                    raise ValueError('Judge returned without a complete1711-row scored cohort')
                job.update(status='complete', finished_at_utc=now())
            except Exception as exc:
                job.update(status='failed', error=str(exc), traceback=traceback.format_exc(), finished_at_utc=now())
            state['active_child'] = None
            update_progress(state)
        if all(state['jobs'][name]['status'] == 'complete' for name in names):
            generation = RUN / 'generation_completion.json'
            if generation.exists() and read(generation).get('status') == 'generation_complete':
                return finish(state, plan, plan_sha)
        generation = RUN / 'generation_completion.json'
        if generation.exists():
            final_generation = read(generation).get('status')
            if final_generation in ('generation_complete', 'partial_failure'):
                # A final generation marker means absent method seals will not
                # appear later in this invocation. Keep their missing status.
                if all(name in attempted or not (ready and (name == 'seed42_ours_1' or generation_ready(name))) for name in names):
                    if not all(state['jobs'][n]['status'] == 'complete' for n in names) or final_generation != 'generation_complete':
                        state.update(status='partial_failure', active_child=None, finished_at_utc=now())
                        update_progress(state)
                        return 2
        state['status'] = 'waiting_reference' if not ready else 'waiting_generation_or_judge'
        update_progress(state)
        time.sleep(INTERVAL)


def evaluation_plan(preflight_path):
    gpu_path = RUN / 'plan.json'
    gpu = read(gpu_path)
    if any(gpu[k] != value for k, value in (('seed', 1042), ('reference_seed', 42), ('judge_seed', 42))):
        raise ValueError('Incorrect frozen GPU plan seeds')
    verify_hashes(gpu['source_sha256'])
    dependencies = [Path(__file__).resolve(), JUDGE, REPORT,
        Path(judge.engine.__file__).resolve(), Path(judge.original.__file__).resolve(),
        Path(judge.transport.__file__).resolve(),
        ROOT / 'scripts/evaluation/openai_chat_transport.py',
        ROOT / 'scripts/evaluation/mem_gallery_master_reader.py',
        ROOT / 'scripts/analysis/export_memgallery_five_grade_judge.py',
        ROOT / 'scripts/analysis/prepare_memgallery_seed1042_reference.py',
        gpu_path, RUN / 'reference_seed42.json', RUN / 'reference_ready.json',
        RUN / 'judge_inputs/seed42_ours_1/manifest.json',
        RUN / 'judge_inputs/seed42_ours_1/records.jsonl', preflight_path]
    plan = {'seed': 1042, 'reference_seed': 42, 'judge_seed': 42,
        'gpu_plan_sha256': sha(gpu_path), 'methods': [{'id': mid, 'label': label} for mid, label in zip(METHOD_IDS, LABELS)],
        'workers': 8, 'max_attempts_per_wire': 5,
        'source_sha256': {str(path): sha(path) for path in dependencies},
        'policy': 'GPU and API queues have independent plans. This plan pins the frozen GPU plan and evaluator/report implementations; original generation drivers remain unchanged.'}
    dump(RUN / 'evaluation_plan.json', plan, frozen=True)
    verify_plan(plan, sha(RUN / 'evaluation_plan.json'))
    return plan


def preflight():
    frozen_plan = RUN / 'evaluation_plan.json'
    if frozen_plan.exists():
        verify_plan(read(frozen_plan), sha(frozen_plan))
        result = read(RUN / 'evaluation_queue_preflight.json')
        print(json.dumps(result), flush=True)
        return
    canon, _ = canonical()
    manifest_path = prepare_input('ours_1', 42)
    manifest = read(manifest_path)
    method = manifest['methods'][0]
    values = rows(method['records_path'])
    reference = next(m for m in read(RUN / 'reference_seed42.json')['methods'] if m['id'] == 'ours_1')
    original = unique(rows(reference['records_path']))
    if sum(v['em'] for v in values) != 798 or any(type(v['em']) is not int for v in values):
        raise ValueError('Old ours_1 strict integer EM conversion failed')
    if any(any(v[k] != original[v['id']][k] for k in ('question', 'ground_truth', 'model_output', 'source_record_id')) for v in values):
        raise ValueError('Original ours_1 text or provenance changed')
    bad = copy.deepcopy(values)
    bad[0]['question'] += ' altered'
    try:
        normalized_records(bad, canon, reference, 42)
    except ValueError:
        wrong_cohort_rejected = True
    else:
        raise ValueError('Wrong canonical question was accepted')
    try:
        verify_hashes({str(manifest_path): '0' * 64})
    except ValueError:
        wrong_source_hash_rejected = True
    else:
        raise ValueError('Wrong source hash was accepted')
    with tempfile.TemporaryDirectory(prefix='gallery1042_no_premature_report_', dir='/tmp') as temporary:
        incomplete = Path(temporary)
        dump(incomplete / 'reference_seed42.json', {'status': 'complete'})
        if report.status(incomplete)['ready_for_export']:
            raise ValueError('Incomplete report was marked ready')
        try:
            report.collect(incomplete)
        except ValueError:
            no_premature_report = True
        else:
            raise ValueError('Incomplete generations produced a report')
    result = {'status': 'passed', 'seed': 1042, 'reference_seed': 42, 'judge_seed': 42,
        'canonical_questions': len(canon), 'seed42_ours_1_correct': 798,
        'strict_integer_em': True, 'original_answers_and_source_ids_preserved': True,
        'wrong_canonical_question_rejected': wrong_cohort_rejected,
        'wrong_source_hash_rejected': wrong_source_hash_rejected,
        'no_premature_report': no_premature_report, 'no_api_or_gpu_calls': True,
        'seed42_ours_1_manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'implementation_sha256': {str(p): sha(p) for p in (Path(__file__).resolve(), JUDGE, REPORT)}, 'checked_at_utc': now()}
    preflight_path = RUN / 'evaluation_queue_preflight.json'
    dump(preflight_path, result)
    evaluation_plan(preflight_path)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'preflight'), required=True)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.phase == 'preflight':
        preflight()
        return
    with (RUN / 'evaluation_queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise SystemExit(run_queue())


if __name__ == '__main__':
    main()
