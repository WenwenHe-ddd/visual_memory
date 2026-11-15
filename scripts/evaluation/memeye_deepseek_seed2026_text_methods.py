"""Fresh MemEye DeepSeek seed2026 native text memories, MCQ and Open Judge42.

Signed seed42/1024 sources stay read-only. Only raw observations, canonical
questions and seed42 captions are shared. Every generated memory, native LLM
request, retrieval and answer lives below this run root. Open retrieval restores
only this run's freshly completed memory. No historical generated calls enter it.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.memory import text_memory_baselines as base

SEED = 2026
MODEL = 'deepseek-v4.1-flash'
RUN = ROOT / 'results/memeye_deepseek_seed2026_full_20261113'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
TRANSPORT = 'scripts.evaluation.memeye_deepseek_seed2026_transport'
OLD = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_text_methods.py'
OLD_READER = ROOT / 'scripts/evaluation/memeye_deepseek_open_reader.py'
RECOVERY = ROOT / 'scripts/evaluation/memeye_deepseek_open_reader_recovery.py'
OLD_LATCH = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_simplemem_recovery.py'
LABELS = dict(gen_agents='Generative Agents (Top10)', memoryos='MemoryOS',
              simplemem='SimpleMem official (final context <=5)', a_mem='A-Mem (final context <=5)')
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: base.sha(Path(path))
rows, digest = base.read_rows, base.hash_object


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen seed2026 contract differs: ' + str(path))
    else:
        dump(path, value)


@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def clone(path, name, old_seed=None, seed_lines=(), *, derived_guard=False, workers=None):
    """Only reviewed seed literal positions change; native token caps stay fixed."""
    source = ast.parse(path.read_text())
    seen = {n.lineno for n in ast.walk(source) if isinstance(n, ast.Constant)
            and type(n.value) is int and n.value == old_seed} if old_seed is not None else set()
    if seen != set(seed_lines):
        raise ValueError('Review semantic seed locations before adapting ' + str(path) + ': ' + str(seen))
    class Adapt(ast.NodeTransformer):
        guard_count = 0
        worker_count = 0
        def visit_Constant(self, node):
            if old_seed is not None and type(node.value) is int and node.lineno in seed_lines and node.value == old_seed:
                return ast.copy_location(ast.Constant(SEED), node)
            if isinstance(node.value, str) and path == OLD:
                value = node.value.replace('seed1024', 'seed2026').replace('full_20261106', 'full_20261113')
                if value != node.value:
                    return ast.copy_location(ast.Constant(value), node)
            return node
        def visit_keyword(self, node):
            self.generic_visit(node)
            if (workers is not None and node.arg in ('task_workers', 'api_workers')
                    and isinstance(node.value, ast.Constant) and node.value.value == 8):
                node.value = ast.copy_location(ast.Constant(workers), node.value)
                self.worker_count += 1
            return node
        def visit_GeneratorExp(self, node):
            self.generic_visit(node)
            if (derived_guard and len(node.generators) == 1
                    and ast.unparse(node.generators[0].target) == '(k, v)'
                    and ast.unparse(node.generators[0].iter) == "question_map[row['id']].items()"
                    and ast.unparse(node.elt) == 'row[k] != v'):
                if node.generators[0].ifs:
                    raise ValueError('Derived query guard structure changed')
                node.generators[0].ifs.append(ast.parse("k != 'retrieval_query'", mode='eval').body)
                self.guard_count += 1
            return node
    change = Adapt()
    tree = ast.fix_missing_locations(change.visit(source))
    if derived_guard and change.guard_count != 1:
        raise ValueError('Expected exactly one derived-query field guard')
    if workers is not None and change.worker_count != 2:
        raise ValueError('Expected only task/API worker settings to change')
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), 'scripts.evaluation'
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=8)
def runtime(run_root, workers):
    run_root = Path(run_root).resolve()
    api = importlib.import_module(TRANSPORT)
    if api.SEED != SEED or api.MODEL != MODEL:
        raise ValueError('Wrong actual-wire seed/model transport')
    native = clone(OLD, 'memeye_seed2026_text_runtime', 1024, (33,),
                   derived_guard=True, workers=workers)
    native.OUTPUT, native.OPEN_OUTPUT = run_root / 'text_methods', run_root / 'open/text_methods'
    native.TRANSPORT = TRANSPORT
    old_code = native.code_files
    native.code_files = lambda: list(dict.fromkeys([*old_code(), Path(__file__), OLD_READER,
        RECOVERY, OLD_LATCH, ROOT / 'scripts/evaluation/memeye_deepseek_open_judge.py',
        ROOT / 'scripts/evaluation/memeye_open_judge_engine.py']))
    native.frozen = freeze
    # At prepare/run time these functions are always called with explicit new
    # roots. Their original evaluated defaults are never used.
    native.SEED = SEED
    reader = clone(OLD_READER, 'memeye_seed2026_text_reader_runtime', 42, (150, 202))
    def settings(model=MODEL):
        api.canonical_model(model)
        return dict(model=model, max_tokens=2048, temperature=0, seed=SEED,
                    enable_thinking=False, n=1, stream=False)
    reader.api = SimpleNamespace(MODEL=MODEL, settings=settings, chat=api.chat_request,
        load_config=api.load_config, APIError=api.APIError, vision=api.original, __file__=api.__file__)
    reader.text_api, reader.RUN, reader.dump = api, run_root, dump
    reader.code_files = lambda: list(dict.fromkeys(native.code_files() + [Path(reader.common.__file__),
        Path(api.original.__file__), ROOT / 'scripts/evaluation/openai_chat_transport.py']))
    # Retain the completed original HTTP400 accounting policy. Permanent
    # provider rejections are explicit empty outcomes, sent to the real judge;
    # successes stay original append-only records and are never fabricated.
    recovery = clone(RECOVERY, 'memeye_seed2026_text_reader_projection')
    recovery.original, recovery.dump = reader, dump
    latch = clone(OLD_LATCH, 'memeye_seed2026_simplemem_latch', 1024, (57, 71, 75, 84, 91, 122))
    old_check = native.check_fatal
    def check_fatal(model):
        if not model.fatal_errors:
            return
        protocol = read(model.root / 'protocol.json')
        if protocol['mode'] != 'simplemem':
            return old_check(model)
        # Unknown errors still fail. This adds no retries or synthetic outputs;
        # it accepts only an already saved later successful exact-wire native
        # attempt joined to a matching broker-cache provider response.
        evidence = [latch.proof(model.root, failure) for failure in model.fatal_errors]
        dump(model.root / 'resolved_native_failure_proof.json', {'status': 'proven', 'seed': SEED,
             'api_retries_added': 0, 'evidence': evidence})
    native.check_fatal = check_fatal
    return native, reader, recovery, api


def prepare(run_root, method, device, workers):
    run_root = Path(run_root).resolve()
    if 'seed2026' not in str(run_root) or run_root == RUN.parent:
        raise ValueError('An isolated seed2026 output root is required')
    native, reader, recovery, api = runtime(str(run_root), workers)
    source = native.prepare(method, native.OUTPUT, embedding_device=device)
    p = native.verify(source)
    contract = {'dataset': 'MemEye', 'method': method, 'label': LABELS[method], 'model': MODEL,
        'seed': SEED, 'caption_seed': 42, 'judge_seed': 42, 'questions': 371, 'device': device,
        'workers': workers, 'memory_rounds': 848, 'native_settings': sorted(map(list, native.NATIVE_MEMORY_SETTINGS[method])),
        'historical_llm_calls_imported': False, 'historical_generated_memories_imported': False,
        'construction_root': str(source), 'open_retrieval_root': str(native.OPEN_OUTPUT / method),
        'answer_cache_root': str(run_root / 'text_open_answer_api_cache' / method),
        'memory_policy': 'Fresh native construction, retrieval, MCQ; Open restores only this run new memory.',
        'derived_query_audit': 'Only retrieval_query is excluded from copied-field equality; actual query must independently equal base.retrieval_query(q).',
        'terminal_http400_policy': recovery.POLICY,
        'source_sha256': {str(path): sha(path) for path in [*native.code_files(), source / 'protocol.json']}}
    if p['seed'] != SEED or p['caption_seed'] != 42 or p['task_workers'] != workers or p['api_workers'] != workers:
        raise ValueError('Native prepared execution seed/settings differ')
    if str(p['stability_replication']['reference_root']).startswith(str(run_root)):
        raise ValueError('Reference metadata should identify original frozen seed42 observations')
    result = {'contract': contract, 'experiment_signature': digest(contract)}
    freeze(run_root / 'methods' / method / 'protocol.json', result)
    return result


def verify_method(run_root, method):
    p = read(run_root / 'methods' / method / 'protocol.json')
    if digest(p['contract']) != p['experiment_signature'] or p['contract']['seed'] != SEED:
        raise ValueError('Method protocol signature/seed differs')
    for name, expected in p['contract']['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen text source changed: ' + name)
    return p


def open_args(run_root, method, workers, correct):
    return SimpleNamespace(root=run_root / 'open/text_methods' / method, method_id=method,
        reference_correct=correct, cache=run_root / 'text_open_answer_api_cache' / method,
        workers=workers, env_file=ENV, limit=0)


def judge(run_root, method, workers):
    from scripts.evaluation import memeye_deepseek_open_judge as original
    opened = run_root / 'open/text_methods' / method
    args = SimpleNamespace(input=opened / 'judge_inputs.jsonl', output=opened / 'judge',
        method_id=method, method_label=LABELS[method], source_protocol=opened / 'protocol.json',
        cache_root=[], env_file=ENV, workers=workers, max_attempts=5, limit=0)
    protocol, requests, question_map = original.prepare(args)
    if protocol['seed'] != 42:
        raise ValueError('Judge seed must stay42')
    result = original.base.run(args, protocol, requests, question_map)
    original.publish_summary(args, protocol, result)
    if result['status'] != 'complete':
        raise ValueError('Incomplete real Judge coverage')
    return result


def audit(run_root, method, workers):
    run_root = Path(run_root).resolve()
    preparation = verify_method(run_root, method)
    native, reader, recovery, api = runtime(str(run_root), workers)
    source, opened = native.OUTPUT / method, native.OPEN_OUTPUT / method
    p = native.verify(source)
    summary = native.audit(method, native.OUTPUT)
    if summary['status'] != 'complete' or summary['questions'] != 371:
        raise ValueError('MCQ construction/readers incomplete')
    seal = read(opened / 'complete.json')
    if seal['seed'] != SEED or seal['status'] != 'complete' or seal['questions'] != 371:
        raise ValueError('Open native retrieval incomplete')
    if seal['reader_manifest_sha256'] != sha(opened / 'reader_manifest.jsonl') or seal['retrieval_sha256'] != sha(opened / 'retrieval.jsonl'):
        raise ValueError('Open native retrieval hash changed')
    open_protocol = read(opened / 'protocol.json')
    if open_protocol['experiment_signature'] != native.signature(open_protocol):
        raise ValueError('Open source protocol signature differs')
    for path, expected in open_protocol['open_evaluation']['source_hashes'].items():
        if sha(path) != expected:
            raise ValueError('New MCQ memory source changed')
    manifest = {r['id']: r for r in rows(opened / 'reader_manifest.jsonl')}
    retrieval = rows(opened / 'retrieval.jsonl')
    canonical_open = {r['id']: r for r in rows(native.OPEN_QUESTIONS)}
    for row in retrieval:
        identity = row['id']
        if row['question'] != canonical_open[identity]['question'] or row['context_sha256'] != digest(row['context']):
            raise ValueError('Open original question/context changed')
        if row['retrieval_query'] != base.retrieval_query(manifest[identity]):
            raise ValueError('Open retrieval query is not the fixed DeepSeek-caption derivation')
        if method in ('a_mem', 'simplemem'):
            detail = row['retrieval_detail']; ids = detail['context_memory_ids']
            if len(ids) != len(set(ids)) or len(ids) > 5:
                raise ValueError('Native Open final context exceeds5')
            if method == 'simplemem' and len(ids) != 5 and not (not ids and detail.get('source_empty_preserved')):
                raise ValueError('SimpleMem Open context not native five/empty')
    outcomes, provenance = recovery.verified_evaluation(opened, with_provenance=True)
    question_map = {r['id']: r for r in rows(source / 'questions.jsonl')}
    mcq = rows(source / 'qa_responses.jsonl')
    scores = {r['id']: r for r in rows(opened / 'judge/per_question.jsonl')}
    judge_summary, judge_protocol = read(opened / 'judge/summary.json'), read(opened / 'judge/protocol.json')
    if (judge_protocol['seed'] != 42 or judge_summary['status'] != 'complete'
            or judge_summary['questions_scored'] != 371 or judge_summary['questions_missing'] != 0):
        raise ValueError('Full fixed Judge42 coverage required')
    from scripts.evaluation import memeye_deepseek_open_judge as original_judge
    requests = rows(opened / 'judge/requests.jsonl')
    actual_grades = original_judge.shared.validated_results(opened / 'judge', requests)
    judged_sources = {r['id']: r for r in rows(opened / 'judge/question_map.jsonl')}
    gold_open = {r['id']: r for r in rows(reader.CANONICAL / 'inputs/gold.jsonl')}
    ids = set(question_map)
    if not all(len(data) == 371 and {r['id'] for r in data} == ids for data in (mcq, outcomes, list(scores.values()), list(manifest.values()))):
        raise ValueError('MCQ/Open/Judge canonical371 identity mismatch')
    for r in outcomes:
        identity = r['id']; saved = scores[identity]; mapped = judged_sources[identity]
        if (saved['model_output'] != r['raw_response'] or saved['question'] != canonical_open[identity]['question']
                or saved['ground_truth'] != gold_open[identity]['answer'] or saved['score'] not in (0., .25, .5, .75, 1.)
                or saved['score'] != actual_grades[mapped['request_sha256']]['score']):
            raise ValueError('Judge does not grade original generated Open answer')
        if r['source_status'] == 'answered' and (r['seed'] != SEED or r['returned_model'] != MODEL):
            raise ValueError('Open original successful provider metadata differs')
    correct, mean = sum(r['correct'] for r in mcq), sum(r['score'] for r in scores.values()) / 371
    if summary['correct'] != correct or abs(mean - judge_summary['llm_judge']) > 1e-12:
        raise ValueError('Recomputed MCQ/Judge aggregates differ')
    mcq_path, open_path = source / 'qa_responses.jsonl', opened / 'reader/evaluation_responses.jsonl'
    mcq_records = [{**r, 'question': question_map[r['id']]['question'],
        'native_outcome': 'invalid_response' if r['predicted_choice'] is None else 'ok',
        'source_file': str(mcq_path), 'source_sha256': sha(mcq_path), 'source_record_id': r['id']} for r in mcq]
    open_records = [{**r, 'question': canonical_open[r['id']]['question'],
        'ground_truth': gold_open[r['id']]['answer'], 'model_output': r['raw_response'],
        'judge_score': scores[r['id']]['score'], 'judge_seed': 42,
        'source_file': str(open_path), 'source_sha256': sha(open_path), 'source_record_id': r['id'],
        'judge_source_file': str(opened / 'judge/per_question.jsonl'),
        'judge_source_sha256': sha(opened / 'judge/per_question.jsonl'), **provenance[r['id']]} for r in outcomes]
    evaluation = run_root / 'evaluations' / method
    base.write_rows(evaluation / 'mcq_records.jsonl', mcq_records)
    base.write_rows(evaluation / 'open_records.jsonl', open_records)
    artifacts = [run_root / 'methods' / method / 'protocol.json', source / 'protocol.json',
        source / 'summary.json', source / 'stability_audit.json', source / 'completion_manifest.json', mcq_path,
        opened / 'protocol.json', opened / 'complete.json', opened / 'retrieval.jsonl', opened / 'reader_manifest.jsonl',
        opened / 'reader/protocol.json', opened / 'reader/evaluation_protocol.json', open_path,
        opened / 'reader/evaluation_completion.json', opened / 'reader/evaluation_collection_audit.json',
        opened / 'judge/summary.json', opened / 'judge/protocol.json', opened / 'judge/per_question.jsonl',
        evaluation / 'mcq_records.jsonl', evaluation / 'open_records.jsonl']
    metrics = dict(status='complete', passed=True, dataset='MemEye', method=method, model=MODEL,
        seed=SEED, caption_seed=42, judge_seed=42, questions=371, correct=correct, accuracy=correct/371,
        open_questions=371, judge_questions=371, llm_judge=mean, historical_answers_reused=0,
        recall_at5_micro=None, recall_policy='No new artificial attribution flow; native contexts preserved.',
        native_mcq_outcomes=dict(Counter(r['native_outcome'] for r in mcq_records)),
        native_open_outcomes=dict(Counter(r['source_status'] for r in outcomes)),
        method_experiment_signature=preparation['experiment_signature'],
        source_sha256={str(path): sha(path) for path in artifacts})
    dump(evaluation / 'metrics.json', metrics)
    completion = dict(metrics, metrics_path=str(evaluation / 'metrics.json'),
        metrics_sha256=sha(evaluation / 'metrics.json'), completion_time_utc=datetime.now(timezone.utc).isoformat())
    dump(run_root / 'methods' / method / 'completion.json', completion)
    return completion


def native_stage(run_root, method, stage, workers, judge_workers):
    verify_method(run_root, method)
    native, reader, recovery, api = runtime(str(run_root), workers)
    if stage == 'mcq':
        result = native.run(method, native.OUTPUT, ENV)
        if result['status'] != 'complete':
            raise ValueError('Incomplete native MCQ phase')
        return result
    if stage == 'open-retrieval':
        return native.open_run(method, native.OUTPUT, native.OPEN_OUTPUT, ENV)
    if stage == 'open-answers':
        summary = read(native.OUTPUT / method / 'summary.json')
        args = open_args(run_root, method, workers, summary['correct'])
        recovery.run(args)
        recovery.collect(args)
        return read(args.root / 'reader/evaluation_completion.json')
    if stage == 'judge':
        return judge(run_root, method, judge_workers)
    if stage == 'audit':
        return audit(run_root, method, workers)
    raise ValueError('Unknown internal native stage')


def run(run_root, method, device, workers, judge_workers):
    prepare(run_root, method, device, workers)
    # Native transports and encoder factories are process-global monkeypatches.
    # Keep the same process separation as the completed original pipeline: an
    # Open restoration cannot accidentally retain the now-closed MCQ broker.
    for stage in ('mcq', 'open-retrieval', 'open-answers', 'judge', 'audit'):
        state = dict(status='running', method=method, seed=SEED, native_stage=stage,
                     supervisor_pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat())
        dump(run_root / 'methods' / method / 'progress.json', state)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', 'run',
            '--run-root', str(run_root), '--method', method, '--device', device,
            '--workers', str(workers), '--judge-workers', str(judge_workers),
            '--native-stage', stage, '--owner-pid', str(os.getpid())]
        print(json.dumps({'method': method, 'seed': SEED, 'native_stage': stage, 'status': 'starting'}), flush=True)
        code = subprocess.call(command, cwd=ROOT)
        if code:
            dump(run_root / 'methods' / method / 'progress.json', dict(state, status='failed', returncode=code))
            raise RuntimeError(f'{method}: native stage {stage} exited {code}; original evidence retained')
    return read(run_root / 'methods' / method / 'completion.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit'), required=True)
    parser.add_argument('--method', choices=tuple(LABELS), required=True)
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--judge-workers', type=int, default=4)
    parser.add_argument('--native-stage', choices=('mcq', 'open-retrieval', 'open-answers', 'judge', 'audit'),
                        help=argparse.SUPPRESS)
    parser.add_argument('--owner-pid', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.judge_workers < 1:
        parser.error('Positive worker counts required')
    run_root = args.run_root.resolve()
    if args.native_stage:
        if args.phase != 'run' or not args.owner_pid or os.getppid() != args.owner_pid:
            raise ValueError('Internal native stages require their live locked method supervisor')
        result = native_stage(run_root, args.method, args.native_stage, args.workers, args.judge_workers)
        print(json.dumps({'method': args.method, 'native_stage': args.native_stage,
                          'seed': SEED, 'status': result.get('status', 'complete')}), flush=True)
        return
    with lock(run_root / 'methods' / args.method / 'execution.lock'):
        if args.phase == 'prepare':
            result = prepare(run_root, args.method, args.device, args.workers)
        elif args.phase == 'audit':
            result = audit(run_root, args.method, args.workers)
        else:
            result = run(run_root, args.method, args.device, args.workers, args.judge_workers)
        print(json.dumps({'method': args.method, 'phase': args.phase, 'seed': SEED,
                          'status': result.get('status', 'prepared')}), flush=True)


if __name__ == '__main__':
    main()
