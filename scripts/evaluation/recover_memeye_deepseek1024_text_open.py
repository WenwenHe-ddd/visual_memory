"""CPU-only recovery of the stale derived Open-query validation contract.

Preserve every frozen question, protocol and native retrieval byte.  The only
source adaptation excludes retrieval_query from the frozen-field comparison;
the original independent equality to the DeepSeek-caption query remains.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import os
from pathlib import Path
import shutil
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
from scripts.evaluation import memeye_deepseek_seed1024_text_methods as native
from scripts.evaluation import memeye_deepseek_seed1024_suite as suite

METHODS = ('gen_agents', 'memoryos', 'a_mem', 'simplemem')
OUT = suite.OUT


def prohibited(*args, **kwargs):
    raise RuntimeError('Recovery must not start inference, an encoder, or any API request')


def validate_rows(questions, result):
    qmap = native.base.by_id(questions) if hasattr(native.base, 'by_id') else {q['id']: q for q in questions}
    if len(questions) != len(qmap) or len(result) != len({r['id'] for r in result}) or set(qmap) != {r['id'] for r in result}:
        raise ValueError('Question/retrieval identity coverage differs')
    mismatches = []
    for row in result:
        q = qmap[row['id']]
        fields = [k for k, v in q.items() if row[k] != v]
        if any(k != 'retrieval_query' for k in fields):
            raise ValueError('Non-derived frozen question field changed: ' + row['id'])
        if row['retrieval_query'] != native.base.retrieval_query(q):
            raise ValueError('Actual retrieval query differs from its frozen DeepSeek-caption derivation')
        if row['context'] != row['retrieved_context'] or row['context_sha256'] != native.base.hash_object(row['context']):
            raise ValueError('Native context or its hash changed')
        if fields:
            mismatches.append({'id': row['id'], 'fields': fields,
                'preserved_frozen_query_sha256': native.base.hash_object(q['retrieval_query']),
                'validated_actual_deepseek_query_sha256': native.base.hash_object(row['retrieval_query'])})
    return mismatches


def tail():
    tree = ast.parse(Path(native.__file__).read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'open_run')
    start = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'manifest' for t in n.targets))
    body = copy.deepcopy(original.body[start:])

    class DerivedField(ast.NodeTransformer):
        count = 0
        def visit_GeneratorExp(self, node):
            self.generic_visit(node)
            if (len(node.generators) == 1 and isinstance(node.generators[0].target, ast.Tuple)
                    and [n.id for n in node.generators[0].target.elts if isinstance(n, ast.Name)] == ['k', 'v']
                    and isinstance(node.generators[0].iter, ast.Call)
                    and isinstance(node.generators[0].iter.func, ast.Attribute)
                    and node.generators[0].iter.func.attr == 'items'
                    and isinstance(node.elt, ast.Compare)
                    and ast.unparse(node.elt) == 'row[k] != v'):
                if node.generators[0].ifs:
                    raise ValueError('Unexpected source guard structure')
                node.generators[0].ifs.append(ast.Compare(left=ast.Name(id='k', ctx=ast.Load()),
                    ops=[ast.NotEq()], comparators=[ast.Constant(value='retrieval_query')]))
                self.count += 1
            return node

    change = DerivedField()
    function = ast.FunctionDef(name='recovered_tail',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in ('method', 'source', 'target', 'p', 'questions', 'result')],
                           kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    module = change.visit(ast.Module(body=[function], type_ignores=[]))
    if change.count != 1:
        raise ValueError('Expected exactly one stale-field guard adaptation')
    module = ast.fix_missing_locations(module)
    namespace = dict(vars(native))
    namespace.update(new_model=prohibited, install_runtime=prohibited, memory_for=prohibited, FrozenBroker=prohibited)
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['recovered_tail'], ast.unparse(module)


def check_no_live_method(method, state):
    dispatcher = Path('/proc') / str(state['pid']) / 'cmdline'
    if not dispatcher.is_file():
        raise ValueError('Host process view required to exclude a live same-method process')
    command = dispatcher.read_bytes().split(b'\0')
    if command[1:5] != [b'-m', b'scripts.evaluation.memeye_deepseek_seed1024_suite', b'--phase', b'run']:
        raise ValueError('Dispatcher identity differs')
    active = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = [s.decode(errors='replace') for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
        except (OSError, ProcessLookupError):
            continue
        if not any('deepseek1024' in a or 'deepseek_seed1024' in a for a in args):
            continue
        for flag in ('--method', '--method-id'):
            if flag in args and args.index(flag) + 1 < len(args) and args[args.index(flag) + 1] == method:
                active.append({'pid': int(proc.name), 'command': args})
                break
    if active:
        raise ValueError('A same-method process is active: ' + str(active))
    return {'same_method_active_processes': [], 'host_dispatcher_pid_verified': state['pid']}


def audit(method, require_process_check=True):
    plan = suite.read(OUT / 'plan.json')
    suite.verify(plan)
    source, target = OUT / 'text_methods' / method, OUT / 'open/text_methods' / method
    state = suite.read(OUT / 'progress.json')
    entry = state['stages'][method + '_open_retrieve']
    if entry.get('status') != 'failed' or entry.get('returncode') == 0:
        raise ValueError('Only an ended failed native retrieval stage is recoverable')
    if state.get('current_stage') == method + '_open_retrieve':
        raise ValueError('Native retrieval stage is still current')
    processes = check_no_live_method(method, state) if require_process_check else {'process_check': 'not_requested_audit_only'}
    p = suite.read(target / 'protocol.json')
    questions, result = native.base.read_rows(target / 'questions.jsonl'), native.base.read_rows(target / 'retrieval.jsonl')
    if len(questions) != 371 or p['experiment_signature'] != native.signature(p):
        raise ValueError('Frozen source protocol/cohort differs')
    if p['open_evaluation']['questions_sha256'] != native.base.hash_object(questions):
        raise ValueError('Frozen questions changed')
    mismatches = validate_rows(questions, result)
    source_questions = {q['id']: q for q in native.base.read_rows(source / 'questions.jsonl')}
    canonical_questions = {q['id']: q for q in native.base.read_rows(native.OPEN_QUESTIONS)}
    if any(q['query_image_captions'] != source_questions[q['id']]['query_image_captions']
           or q['retrieval_query'] != canonical_questions[q['id']]['retrieval_query'] for q in questions):
        raise ValueError('Expected exact frozen DeepSeek captions and preserved historical derived query')
    paths = [target / n for n in ('protocol.json', 'questions.jsonl', 'retrieval.jsonl', 'stage_failure.json', 'replayed_calls.jsonl')]
    task_rows = []
    for task in p['tasks']:
        work = target / task
        marker = suite.read(work / 'complete.json')
        rows = native.base.read_rows(work / 'retrieval.jsonl')
        if marker['status'] != 'complete' or marker['questions'] != len(rows) or marker['retrieval_sha256'] != suite.sha(work / 'retrieval.jsonl'):
            raise ValueError('Task-native completion differs: ' + task)
        paths.extend([work / 'complete.json', work / 'retrieval.jsonl'])
        if method in ('gen_agents', 'a_mem'):
            restored = suite.read(work / 'restored_memory.json')
            expected = suite.read(source / task / 'final_memory.json')
            if method == 'a_mem':
                strip = lambda notes: {k: {f: v for f, v in n.items() if f != 'last_accessed'} for k, n in notes.items()}
                restored, expected = strip(restored), strip(expected)
            if restored != expected:
                raise ValueError('Restored native memory differs')
            paths.append(work / 'restored_memory.json')
        else:
            copied = suite.read(work / 'runtime_copy_audit.json')
            if not copied['passed']:
                raise ValueError('Native database copy audit did not pass')
            paths.append(work / 'runtime_copy_audit.json')
        task_rows.extend(rows)
    order = {q['id']: i for i, q in enumerate(questions)}
    task_rows.sort(key=lambda row: order[row['id']])
    if task_rows != result:
        raise ValueError('Global retrieval differs from ordered native task outputs')
    for name, expected in p['open_evaluation']['source_hashes'].items():
        if suite.sha(name) != expected:
            raise ValueError('New MCQ source changed: ' + name)
    evidence = {'method': method, 'questions': 371, 'different_frozen_fields': dict(Counter(k for x in mismatches for k in x['fields'])),
        'mismatches': mismatches, 'unchanged_other_records': 371 - len(mismatches),
        'actual_query_derivation_passed_all_371': True, 'native_task_global_retrieval_and_context_hashes_passed': True,
        'frozen_protocol_and_question_bytes_preserved': True, 'source_hashes_passed': True,
        'root_cause': 'Frozen Open questions retain a historical canonical Qwen-derived retrieval_query after their captions are replaced with DeepSeek captions; the actual native retrieval correctly recomputes the DeepSeek-caption query.',
        'api_calls': 0, 'gpu_model_loads': 0, **processes,
        'preserved_input_sha256': {str(path): suite.sha(path) for path in paths},
        'source_code_sha256': suite.sha(native.__file__), 'adapter_sha256': suite.sha(Path(__file__))}
    return source, target, p, questions, result, evidence


def recover(method):
    source, target, p, questions, result, evidence = audit(method)
    evidence_root = OUT / 'metrics/recovery_diagnostics' / (method + '_open_failure_20261106')
    evidence_root.mkdir(parents=True, exist_ok=True)
    snapshot = evidence_root / 'target_snapshot'
    if not snapshot.exists():
        shutil.copytree(target, snapshot)
        shutil.copyfile(OUT / 'logs' / (method + '_open_retrieve.log'), evidence_root / 'stage_log.txt')
        suite.dump(evidence_root / 'suite_progress_snapshot.json', suite.read(OUT / 'progress.json'))
    if not (evidence_root / 'diagnosis.json').exists():
        suite.dump(evidence_root / 'diagnosis.json', evidence)
    suite.dump(evidence_root / 'cpu_recovery_preflight.json', evidence)
    api = native.api()
    api.chat_request = api.load_config = prohibited
    if hasattr(api, 'original'):
        api.original.chat_request = api.original.load_config = prohibited
    function, adapted_source = tail()
    (evidence_root / 'adapted_native_tail.py').write_text(adapted_source + '\n')
    outputs = ('reader_manifest.jsonl', 'complete.json', 'ready_for_reader.json')
    if any((target / n).exists() for n in outputs):
        raise ValueError('Recovery only commits absent completion outputs; existing outputs require independent review')
    with tempfile.TemporaryDirectory(prefix=method + '-open-recovery-', dir=str(evidence_root)) as temporary:
        staging = Path(temporary) / 'target'
        shutil.copytree(target, staging, copy_function=os.link)
        marker = function(method, source, staging, p, questions, copy.deepcopy(result))
        if not marker.get('passed') or marker.get('questions') != 371 or marker.get('status') != 'complete':
            raise ValueError('Original native tail did not produce audited full completion')
        for name, expected in evidence['preserved_input_sha256'].items():
            if suite.sha(name) != expected:
                raise ValueError('An original immutable input changed during recovery')
        for name in outputs:
            staged = staging / name
            if not staged.is_file():
                raise ValueError('Missing validated native completion output')
            temporary_output = target / (name + '.recovery-tmp')
            shutil.copyfile(staged, temporary_output)
            temporary_output.replace(target / name)
    evidence.update(status='cpu_recovered', completed_at_utc=suite.now(),
        recovery_boundary='Only exclude the stale derived retrieval_query from frozen-field equality. Its separate exact DeepSeek derivation, every native context and all original completion audits remain.',
        output_sha256={str(target / n): suite.sha(target / n) for n in outputs},
        native_marker=marker, original_failure_snapshot=str(snapshot),
        no_native_retrieval_or_generation_rerun=True, original_protocol_signature_unchanged=p['experiment_signature'])
    suite.dump(evidence_root / 'recovery_evidence.json', evidence)
    return evidence


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--phase', choices=('audit', 'recover'), default='audit')
    args = parser.parse_args()
    value = recover(args.method) if args.phase == 'recover' else audit(args.method)[-1]
    print(json.dumps({k: value[k] for k in ('method', 'questions', 'different_frozen_fields', 'api_calls', 'gpu_model_loads')}, ensure_ascii=False))
