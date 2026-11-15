"""Fresh seed-1024 construction, retrieval and reading on fixed MemEye captions.

Existing implementations and historical protocols remain untouched.  Each
method receives a new protocol, task stores and request cache.  The official
SimpleMem variant uses K10, reflection and channel RRF with five final entries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.evaluation import mcq_reader  # Dependency bootstrap, without models.
from scripts.memory import text_memory_baselines as baseline

SEED = 1024
OUTPUT = ROOT / 'results/memeye_qwen_seed1024_full_20261105/text_methods'
OPEN_OUTPUT = OUTPUT.parent / 'open/text_methods'
REFERENCE = {
    'gen_agents': ROOT / 'results/visual_memory_pipeline/gen_agents',
    'memoryos': ROOT / 'results/visual_memory_pipeline/memoryos',
    'a_mem': ROOT / 'results/visual_memory_pipeline/top5/a_mem_context5',
    'simplemem': ROOT / 'results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/top5/simplemem_official_text_context5',
}
METHODS = tuple(REFERENCE)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def arguments(method, output_root=OUTPUT, device='cuda:0'):
    if method not in METHODS:
        raise ValueError('Unknown method: ' + method)
    if not re.fullmatch(r'cuda:\d+', device):
        raise ValueError('device must be a visible CUDA device such as cuda:0')
    return SimpleNamespace(
        mode=method, output_root=Path(output_root), device=device,
        benchmark_manifest=ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl',
        natural_root=ROOT / 'results/natural_observation', task=None,
        limit_questions=0, top_k=5 if method == 'a_mem' else None,
        a_mem_context_limit=5 if method == 'a_mem' else None,
        simplemem_backend='official_text' if method == 'simplemem' else None,
        simplemem_semantic_top_k=10, simplemem_reflection=True,
        simplemem_context_fusion='channel_rrf',
        simplemem_context_limit=5 if method == 'simplemem' else None,
        simplemem_preserve_empty=True, context_budget=128000,
        memory_max_new_tokens=16384 if method == 'simplemem' else 2048,
        memory_workers=4 if method == 'gen_agents' else 8,
        memory_batch_size=4 if method == 'gen_agents' else 2 if method == 'simplemem' else 8,
        memory_device_map=None,
    )


def target(method, output_root=OUTPUT):
    return baseline.output_dir(arguments(method, output_root))


def signature(protocol):
    return baseline.hash_object({k: v for k, v in protocol.items() if k != 'experiment_signature'})


def verify(method, output_root=OUTPUT, device='cuda:0'):
    args = arguments(method, output_root, device)
    directory = baseline.output_dir(args)
    p = read(directory / 'protocol.json')
    if p['seed'] != SEED or p['experiment_signature'] != signature(p):
        raise ValueError('Wrong seed or invalid protocol signature')
    if p['reader_device_map'] != {'': int(device.split(':')[1])}:
        raise ValueError('Device differs from the prepared protocol')
    for name, digest in p['input_hashes'].items():
        if baseline.sha(Path(name)) != digest:
            raise ValueError('Pinned input changed: ' + name)
    q = baseline.read_rows(directory / 'questions.jsonl')
    observations = read(directory / 'observations.json')
    if baseline.hash_object(q) != p['prepared_questions_sha256'] or baseline.hash_object(observations) != p['observations_sha256']:
        raise ValueError('Prepared question or observation hash differs')
    reference = read(REFERENCE[method] / 'protocol.json')
    if p['method_defaults'] != reference['method_defaults']:
        raise ValueError('Method defaults differ from the seed-42 table row')
    for key in ('prepared_questions_sha256', 'observations_sha256', 'reader_model',
                'reader_system_prompt', 'dtype', 'max_new_tokens', 'context_budget',
                'memory_max_new_tokens', 'generation_batch_size', 'task_workers'):
        if p[key] != reference[key]:
            raise ValueError('Unexpected change from seed-42 reference: ' + key)
    if len(q) != 371 or len({r['id'] for r in q}) != 371 or sum(map(len, observations.values())) != 848:
        raise ValueError('MemEye coverage must be 371 questions and 848 rounds')
    if set(observations) != {r['task'] for r in q} or len(observations) != 8:
        raise ValueError('Expected the same eight MemEye tasks')
    return args, p, q, observations


def prepare(method, output_root=OUTPUT, device='cuda:0'):
    args = arguments(method, output_root, device)
    directory = baseline.output_dir(args)
    if (directory / 'protocol.json').exists():
        verify(method, output_root, device)
        return directory
    if directory.exists() and any(p.name != 'stage_status.json' for p in directory.iterdir()):
        raise ValueError('Refuse to prepare over nonempty output: ' + str(directory))
    baseline.prepare(args)
    p = read(directory / 'protocol.json')
    reference = REFERENCE[method]
    p.update(seed=SEED, reader_device_map={'': int(device.split(':')[1])},
             study='MemEye fixed-caption full-flow seed-1024 stability replication')
    p['memory_sampling'] = p['memory_sampling'].replace('seed42', 'seed1024')
    p['differences_from_paper'] = [s.replace('seed42', 'seed1024') for s in p['differences_from_paper']]
    if method == 'gen_agents':
        p['memory_sampling'] = 'native temperature=0; greedy memory operations and final reader; seed1024 recorded'
    p['stability_replication'] = {
        'reference_root': str(reference), 'reference_seed': 42, 'seed': SEED,
        'reference_protocol_sha256': baseline.sha(reference / 'protocol.json'),
        'reference_responses_sha256': baseline.sha(reference / 'qa_responses.jsonl'),
        'caption_policy': 'same frozen historical and query-image Qwen captions as seed42',
        'construction_policy': 'fresh task stores; no inherited memories or calls',
        'retrieval_policy': 'rerun all native retrieval, planning, reflection and query-keyword calls',
        'reader_policy': 'fresh greedy predictions; fixed question order and presented options',
        'judge_seed': 42,
        'execution_note': ('explicit selected GPU; same per-method construction workers and generation batch limit; '
                           'generic native runner interleaves retrieval and reading within each task; '
                           'historical SimpleMem ablation batched retrieval separately at up to eight'),
        'native_nondeterminism': {
            'gen_agents': 'memory and reader are greedy; native numeric backend effects remain possible',
            'memoryos': 'upstream uuid4 IDs and current maintenance/visit timestamps are preserved and are not set by torch seed',
            'a_mem': 'note IDs and timestamps use fixed source rounds; last_accessed uses wall clock',
            'simplemem': 'upstream MemoryEntry uuid4 IDs are preserved and are not set by torch seed',
        }[method],
    }
    p['input_hashes'].update({
        str(Path(__file__).resolve()): baseline.sha(Path(__file__)),
        str(reference / 'protocol.json'): baseline.sha(reference / 'protocol.json'),
        str(reference / 'qa_responses.jsonl'): baseline.sha(reference / 'qa_responses.jsonl'),
    })
    p['experiment_signature'] = signature(p)
    baseline.write_json(directory / 'protocol.json', p)
    snapshots = {}
    for name, digest in p['input_hashes'].items():
        if Path(name).suffix != '.py':
            continue
        snapshot = directory / 'source_snapshot' / (digest[:16] + '_' + Path(name).name)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(Path(name).read_bytes())
        snapshots[name] = {'snapshot': str(snapshot), 'sha256': digest}
    baseline.write_json(directory / 'source_snapshot/manifest.json', snapshots)
    verify(method, output_root, device)
    return directory


@contextmanager
def injected_runtime(model, protocol):
    """Allow a caller to reuse one local model across serial method runs."""
    import openai
    import sentence_transformers
    factory = baseline.LocalMemoryQwen
    openai_factory = openai.OpenAI
    encoder_factory = sentence_transformers.SentenceTransformer
    try:
        if model is not None:
            for key in ('reader_model_path', 'dtype', 'context_budget', 'reader_device_map'):
                if model.protocol.get(key) != protocol.get(key):
                    raise ValueError('Injected model setting differs: ' + key)
            model.protocol = protocol
            baseline.LocalMemoryQwen = lambda p: model
        yield
    finally:
        baseline.LocalMemoryQwen = factory
        openai.OpenAI = openai_factory
        sentence_transformers.SentenceTransformer = encoder_factory


def run_method(method, output_root=OUTPUT, device='cuda:0', model=None):
    args, p, _, _ = verify(method, output_root, device)
    with injected_runtime(model, p):
        baseline.run(args)
    return audit(method, output_root, device)


def audit(method, output_root=OUTPUT, device='cuda:0'):
    args, p, questions, observations = verify(method, output_root, device)
    directory = baseline.output_dir(args)
    failures = []
    calls = {}
    sampling = Counter()
    for path in (directory / 'calls').glob('*.json'):
        call = read(path)
        req = call['request']
        if (baseline.hash_object(req) != path.stem or req['seed'] != SEED
                or req['experiment_signature'] != p['experiment_signature'] or 'inherited_from' in call):
            failures.append({'check': 'fresh_seed1024_request', 'path': str(path)})
        sampling[(req['kind'], req.get('temperature'), req['do_sample'])] += 1
        calls[path.stem] = call
    rows = []
    completed_tasks = []
    construction = {}
    for task in p['tasks']:
        work = directory / task
        stats_path = work / 'memory_stats.json'
        if stats_path.exists():
            construction[task] = read(stats_path)
        marker = work / 'complete.json'
        if not marker.exists():
            continue
        complete = read(marker)
        responses = baseline.read_rows(work / 'responses.jsonl')
        if (complete['experiment_signature'] != p['experiment_signature']
                or complete['responses_sha256'] != baseline.sha(work / 'responses.jsonl')
                or complete['questions'] != len(responses)):
            failures.append({'check': 'task_completion', 'task': task})
        completed_tasks.append(task)
        rows.extend(responses)
        if construction.get(task, {}).get('stored_rounds') != len(observations[task]):
            failures.append({'check': 'all_source_rounds_ingested', 'task': task})
    qmap = {q['id']: q for q in questions}
    seen = Counter(r['id'] for r in rows)
    for identity, count in seen.items():
        if identity not in qmap or count != 1:
            failures.append({'check': 'unique_expected_response', 'id': identity, 'count': count})
    for row in rows:
        identity = row['id']
        if identity not in qmap:
            continue
        if any(row.get(k) != v for k, v in qmap[identity].items()):
            failures.append({'check': 'fixed_question', 'id': identity})
        call = calls.get(row.get('call_key'))
        if (call is None or call['request']['kind'] != 'answer'
                or call['request']['messages'] != baseline.reader_messages(qmap[identity], row['retrieved_context'], p)
                or row['reader_messages_sha256'] != baseline.hash_object(call['request']['messages'])
                or row['raw_response'] != call['output']['raw_response']):
            failures.append({'check': 'reader_payload_provenance', 'id': identity})
        if row['experiment_signature'] != p['experiment_signature']:
            failures.append({'check': 'response_signature', 'id': identity})
        if method in ('a_mem', 'simplemem'):
            ids = row['retrieval_detail']['context_memory_ids']
            empty = method == 'simplemem' and row['retrieval_detail'].get('source_empty_preserved') and not ids
            if not empty and (len(ids) != 5 or len(set(ids)) != 5):
                failures.append({'check': 'five_distinct_final_entries', 'id': identity})
    expected_ids = set(qmap)
    coverage_complete = set(seen) == expected_ids and len(rows) == len(questions) and len(completed_tasks) == len(p['tasks'])
    report = {
        'updated_at': now(), 'method': method, 'seed': SEED,
        'status': 'complete' if coverage_complete and not failures else 'failed_audit' if failures else 'partial',
        'audit_passed_so_far': not failures, 'coverage_complete': coverage_complete,
        'expected_questions': len(questions), 'answered_questions': len(rows),
        'expected_tasks': len(p['tasks']), 'completed_tasks': completed_tasks,
        'memory_rounds': sum(map(len, observations.values())),
        'unique_llm_calls': len(calls),
        'memory_calls': sum(x['request']['kind'] == 'memory' for x in calls.values()),
        'reader_calls': sum(x['request']['kind'] == 'answer' for x in calls.values()),
        'memory_token_limit_hits': sum(x['output']['hit_token_limit'] for x in calls.values() if x['request']['kind'] == 'memory'),
        'answer_token_limit_hits': sum(x['hit_token_limit'] for x in rows),
        'format_errors': sum(x['predicted_choice'] not in ('A', 'B', 'C', 'D') for x in rows),
        'sampling_calls': [{'kind': k[0], 'temperature': k[1], 'do_sample': k[2], 'count': n} for k, n in sampling.items()],
        'construction_stats': construction, 'failures': failures,
        'experiment_signature': p['experiment_signature'],
    }
    baseline.write_json(directory / 'stability_audit.json', report)
    if coverage_complete and not failures:
        baseline.write_json(directory / 'completion_manifest.json', {
            'status': 'complete', 'seed': SEED, 'questions': len(rows),
            'experiment_signature': p['experiment_signature'],
            'protocol_sha256': baseline.sha(directory / 'protocol.json'),
            'responses_sha256': baseline.sha(directory / 'qa_responses.jsonl'),
            'audit_sha256': baseline.sha(directory / 'stability_audit.json'),
            'completed_at': now(),
        })
    if failures:
        raise ValueError(f'{method}: {len(failures)} audit failure(s); inspect stability_audit.json')
    return report


@contextmanager
def open_sources(method, output_root=OUTPUT, open_output=OPEN_OUTPUT):
    """Bind the existing Open adapter to the newly built seed-1024 memory."""
    from scripts.evaluation import memeye_open_text_retrieval as retrieval
    from scripts.evaluation import memeye_open_text_reader_inputs as reader_inputs
    source = target(method, output_root)
    device = 'cuda:' + str(read(source / 'protocol.json')['reader_device_map'][''])
    _, p, _, _ = verify(method, output_root, device)
    summary = read(source / 'summary.json')
    seal = read(source / 'completion_manifest.json')
    if (summary['status'] != 'complete' or summary['questions'] != 371 or seal['status'] != 'complete'
            or seal['responses_sha256'] != baseline.sha(source / 'qa_responses.jsonl') or p['seed'] != SEED):
        raise ValueError('Open retrieval requires a complete audited seed1024 MCQ source')
    sources = dict(retrieval.SOURCES, **{method: (str(source.resolve()), summary['correct'])})
    old = retrieval.SOURCES, retrieval.OUT, reader_inputs.SOURCES, reader_inputs.OUT
    retrieval.SOURCES, retrieval.OUT = sources, Path(open_output)
    reader_inputs.SOURCES, reader_inputs.OUT = sources, Path(open_output)
    try:
        yield retrieval, reader_inputs
    finally:
        retrieval.SOURCES, retrieval.OUT, reader_inputs.SOURCES, reader_inputs.OUT = old


def prepare_open(method, output_root=OUTPUT, device='cuda:0', open_output=OPEN_OUTPUT):
    verify(method, output_root, device)
    with open_sources(method, output_root, open_output) as (retrieval, reader_inputs):
        destination = Path(open_output) / method
        _, protocol, questions = retrieval.make_protocol(method, destination)
        if protocol['seed'] != SEED or len(questions) != 371:
            raise ValueError('Wrong Open seed or cohort')
        if (destination / 'complete.json').exists():
            reader_inputs.prepare(method)
        return {'method': method, 'seed': SEED, 'status': 'prepared',
                'expected_questions': len(questions), 'answered_questions': 0,
                'reader_manifest': str(destination / 'reader_manifest.jsonl')
                    if (destination / 'reader_manifest.jsonl').exists() else None}


def run_open(method, output_root=OUTPUT, device='cuda:0', open_output=OPEN_OUTPUT, model=None):
    _, p, _, _ = verify(method, output_root, device)
    with open_sources(method, output_root, open_output) as (retrieval, reader_inputs):
        with injected_runtime(model, p):
            retrieval.run(method, Path(open_output), workers=p['task_workers'], allow_gpu=True)
        reader_inputs.prepare(method)
        destination = Path(open_output) / method
        manifest = baseline.read_rows(destination / 'reader_manifest.jsonl')
        for row in manifest:
            if set(row) & {'options', 'gold_choice', 'answer', 'ground_truth', 'clue', 'gold_ids'}:
                raise ValueError('Open reader manifest contains evaluation-only fields')
        return {'method': method, 'seed': SEED, 'status': 'retrieval_complete',
                'expected_questions': 371, 'answered_questions': 0,
                'retrieved_questions': len(manifest), 'reader_manifest': str(destination / 'reader_manifest.jsonl')}


def execute(method, phase, output_root=OUTPUT, device='cuda:0', model=None):
    directory = target(method, output_root)
    baseline.write_json(directory / 'stage_status.json', {'phase': phase, 'status': 'running', 'updated_at': now()})
    try:
        if phase in ('open-prepare', 'open-run'):
            result = (prepare_open(method, output_root, device) if phase == 'open-prepare' else
                      run_open(method, output_root, device, model=model))
            baseline.write_json(directory / 'stage_status.json', dict(result, phase=phase, updated_at=now()))
            return result
        if phase in ('prepare', 'all'):
            prepare(method, output_root, device)
        if phase in ('run', 'all'):
            run_method(method, output_root, device, model)
        if phase == 'analyze':
            args, _, _, _ = verify(method, output_root, device)
            baseline.analyze(args)
        result = audit(method, output_root, device)
        baseline.write_json(directory / 'stage_status.json', {
            'phase': phase, 'status': result['status'], 'updated_at': now(),
            'answered_questions': result['answered_questions'], 'expected_questions': result['expected_questions'],
        })
        return result
    except BaseException as exc:
        baseline.write_json(directory / 'stage_failure.json', {
            'phase': phase, 'status': 'failed', 'updated_at': now(),
            'error': str(exc), 'traceback': traceback.format_exc(),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=(*METHODS, 'all'), required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze', 'audit', 'all', 'open-prepare', 'open-run'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    args = parser.parse_args()
    errors = []
    shared_model = None
    for method in METHODS if args.method == 'all' else (args.method,):
        try:
            if args.phase in ('run', 'all', 'open-run') and shared_model is None:
                if args.phase == 'all':
                    prepare(method, args.output_root, args.device)
                _, protocol, _, _ = verify(method, args.output_root, args.device)
                shared_model = baseline.LocalMemoryQwen(protocol)
            report = execute(method, args.phase, args.output_root, args.device, shared_model)
            print(json.dumps({k: report[k] for k in ('method', 'seed', 'status', 'answered_questions', 'expected_questions')}, ensure_ascii=False), flush=True)
        except Exception as exc:
            errors.append({'method': method, 'error': str(exc)})
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    if errors:
        raise SystemExit(json.dumps(errors, ensure_ascii=False))


if __name__ == '__main__':
    main()
