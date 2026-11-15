"""Fresh DeepSeek full-flow seed1024 text-memory stability runs on frozen captions.

Run one method per process: upstream libraries retain transport module globals.
The reference construction inputs/defaults are read-only; no old calls or
generated memories enter the new construction. Open retrieval restores only
this run's completed seed1024 stores. API credentials never enter artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mcq_reader
from scripts.memory import text_memory_baselines as base, simplemem_official as official
from scripts.evaluation import memeye_deepseek_text_methods as legacy
from scripts.evaluation.gpt5_text_memory_adapters import install_runtime, make_broker
from scripts.evaluation.continue_deepseek_ga_format import instruction_for
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM as OPEN_SYSTEM

SEED = 1024
MODEL = 'deepseek-v4.1-flash'
TRANSPORT = 'scripts.evaluation.memeye_deepseek_seed1024_transport'
NATIVE_MEMORY_SETTINGS = {
    'gen_agents': {(0., 2048)},
    'memoryos': {(.7, 2000), (.3, 100), (0., 10)},
    'a_mem': {(.7, 1000)},
    'simplemem': {(.1, 16384), (.2, 16384), (.3, 16384)},
}
OUTPUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/text_methods'
OPEN_OUTPUT = OUTPUT.parent / 'open/text_methods'
CAPTIONS = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
CANONICAL = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
OPEN_QUESTIONS = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
REFERENCES = {
    'gen_agents': ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025/gen_agents',
    'memoryos': ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025/memoryos',
    'a_mem': ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025/a_mem',
    'simplemem': ROOT / 'results/simplemem_official_deepseek_memeye_k10_rrf5_reflection_20261029/top5/simplemem_official_text_context5',
}


def api():
    module = importlib.import_module(TRANSPORT)
    if module.SEED != SEED or module.DEEPSEEK_MODEL != MODEL:
        raise ValueError('Shared transport model/seed differs')
    return module


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(p):
    return base.hash_object({k: v for k, v in p.items() if k != 'experiment_signature'})


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Prepared artifact differs: ' + str(path))
    else:
        base.write_json(path, value)


def wire(messages, cap, temperature):
    return api().build_request(messages, cap, temperature, seed=SEED, model=MODEL)


def code_files():
    names = [Path(__file__), Path(base.__file__), Path(official.__file__), Path(legacy.__file__),
             Path(api().__file__), ROOT / 'scripts/memory/text_memory_transport.py',
             ROOT / 'scripts/memory/reflexion_memory.py', ROOT / 'scripts/memory/simplemem_fusion.py',
             ROOT / 'scripts/evaluation/gpt5_text_memory_adapters.py',
             ROOT / 'scripts/evaluation/continue_deepseek_text_methods_streaming.py',
             ROOT / 'scripts/evaluation/continue_deepseek_text_methods_parallel.py',
             ROOT / 'scripts/evaluation/continue_deepseek_ga_format.py',
             ROOT / 'scripts/evaluation/probe_deepseek_ga_format.py',
             ROOT / 'scripts/evaluation/continue_deepseek_memoryos_format.py',
             ROOT / 'scripts/evaluation/deepseek_flash_seed42_audited_transport.py',
             ROOT / 'scripts/evaluation/openai_chat_transport.py']
    return names


def verify(root):
    p = read(root / 'protocol.json')
    if p['seed'] != SEED or p['model'] != MODEL or p['experiment_signature'] != signature(p):
        raise ValueError('Invalid protocol seed/model/signature')
    for name, digest in p['input_sha256'].items():
        if base.sha(Path(name)) != digest:
            raise ValueError('Pinned input changed: ' + name)
    if base.hash_object(read(root / 'observations.json')) != p['observations_sha256']:
        raise ValueError('Construction inputs changed')
    if base.hash_object(base.read_rows(root / 'questions.jsonl')) != p['questions_sha256']:
        raise ValueError('Questions changed')
    if base.hash_object(base.read_rows(root / 'gold.jsonl')) != p['gold_sha256']:
        raise ValueError('Evaluation labels changed')
    upstream = official.verify_installation() if p['mode'] == 'simplemem' else base.verify_vendor()
    if upstream != p['upstream']:
        raise ValueError('Pinned method implementation changed')
    return p


def prepare(method, output=OUTPUT, embedding_device=None):
    target = Path(output) / method
    if (target / 'protocol.json').exists():
        verify(target)
        return target
    if target.exists() and any(target.iterdir()):
        raise ValueError('Refuse nonempty unprepared directory')
    source = REFERENCES[method]
    parent = read(source / 'protocol.json')
    if parent['seed'] != 42 or parent['model'] != MODEL or read(source / 'summary.json')['status'] != 'complete':
        raise ValueError('Expected the completed exact-model seed42 reference')
    if parent['experiment_signature'] != signature(parent):
        raise ValueError('Reference protocol signature differs')
    captions = base.read_rows(CAPTIONS / 'captions.jsonl')
    if len(captions) != 437 or len({c['id'] for c in captions}) != 437:
        raise ValueError('Expected 437 frozen historical/query caption images')
    if any(c['seed'] != 42 or c['model'] != MODEL or not c['caption'].strip() for c in captions):
        raise ValueError('Frozen captions differ from the reference model/seed')
    expected_caption_sha = parent.get('caption_sha256') or parent['input_hashes'][str(CAPTIONS / 'captions.jsonl')]
    if base.sha(CAPTIONS / 'captions.jsonl') != expected_caption_sha:
        raise ValueError('Caption bytes differ from the completed seed42 method')
    observations = read(source / 'observations.json')
    questions = base.read_rows(source / 'questions.jsonl')
    if (base.hash_object(observations) != parent['observations_sha256']
            or base.hash_object(questions) != parent.get('questions_sha256', parent.get('prepared_questions_sha256'))):
        raise ValueError('Reference prepared observations/questions differ')
    canonical = {q['id']: q for q in base.read_rows(CANONICAL)}
    if len(questions) != 371 or {q['id'] for q in questions} != set(canonical) or sum(map(len, observations.values())) != 848:
        raise ValueError('Expected all 371 questions and 848 original dialogue rounds')
    for q in questions:
        if any(q[k] != canonical[q['id']][k] for k in ('question', 'options', 'task')):
            raise ValueError('Reference MCQ question/options changed')
    gold = [dict(id=q['id'], **{k: canonical[q['id']][k] for k in ('gold_choice', 'gold_ids', 'x', 'y')}) for q in questions]
    p = copy.deepcopy(parent)
    p.pop('experiment_signature', None)
    p.pop('input_hashes', None)
    for key in ('control', 'ablation', 'recovery'):
        p.pop(key, None)
    for key in ('memory_sampling', 'answer_sampling', 'answer_prompt_policy'):
        if isinstance(p.get(key), str):
            p[key] = p[key].replace('wire seed42', 'wire seed1024').replace('; seed42', '; seed1024')
    if isinstance(p.get('differences_from_paper'), list):
        p['differences_from_paper'] = [text if 'caption' in text.lower() else text.replace('seed42', 'seed1024')
                                       for text in p['differences_from_paper']]
    p.update(mode=method, seed=SEED, caption_seed=42, model=MODEL,
             caption_model=MODEL, expected_returned_model=MODEL, enable_thinking=False,
             study='DeepSeek fixed-caption fresh full-flow seed1024 stability replication',
             task_workers=8, api_workers=8, generation_batch_size=8,
             embedding_device=embedding_device or ('cpu' if method == 'simplemem' else 'cuda:1'),
             max_padded_batch_tokens=131072 if method == 'simplemem' else 48000,
             observations_sha256=base.hash_object(observations), questions_sha256=base.hash_object(questions),
             prepared_questions_sha256=base.hash_object(questions), gold_sha256=base.hash_object(gold),
             questions=371, memory_rounds=848, tasks=sorted(observations),
             memory_max_new_tokens=16384 if method == 'simplemem' else 2048,
             max_new_tokens=16, context_budget=128000,
             memory_sampling='native per-call temperatures and budgets, explicit wire seed1024, thinking=false',
             answer_sampling='temperature0, max_tokens16, explicit wire seed1024, thinking=false',
             caption_sha256=base.sha(CAPTIONS / 'captions.jsonl'),
             prior_generated_memory_reused=False, prior_reader_answers_reused=False,
             stability_replication={'reference_root': str(source), 'reference_seed': 42,
                 'reference_protocol_sha256': base.sha(source / 'protocol.json'),
                 'captions_frozen_at_seed': 42, 'memory_and_reader_seed': SEED, 'judge_seed': 42,
                 'fresh_construction': True, 'old_calls_or_stores_imported': False,
                 'effective_legacy_execution': 'eight workers with original streaming/format amendments',
                 'ga_format': 'original system constraints and at most three identical format attempts',
                 'memoryos_format': 'original lossless singleton summary container/key normalization',
                 'native_nondeterminism': 'UUID IDs, native wall-clock state and provider alias/backend variation are preserved'})
    files = [*code_files(), source / 'protocol.json', source / 'observations.json', source / 'questions.jsonl',
             CANONICAL, CAPTIONS / 'captions.jsonl', CAPTIONS / 'completion_manifest.json']
    files += sorted(source.glob('*amendment*.json'))
    files += [f for f in Path(p['embedding_model_path']).rglob('*') if f.is_file()]
    if method == 'simplemem':
        p['upstream'] = official.verify_installation()
        files += official.source_files()
        p.update(simplemem_backend='official_text', context_fusion='channel_rrf', fusion_rank_constant=60,
                 context_memory_limit=5, preserve_empty_context=True)
    else:
        p['upstream'] = base.verify_vendor()
    p['input_sha256'] = {str(f.resolve()): base.sha(f) for f in files}
    p['experiment_signature'] = signature(p)
    target.mkdir(parents=True, exist_ok=True)
    base.write_json(target / 'protocol.json', p)
    base.write_json(target / 'observations.json', observations)
    base.write_rows(target / 'questions.jsonl', questions)
    base.write_rows(target / 'gold.jsonl', gold)
    snapshots = {}
    for name, digest in p['input_sha256'].items():
        if Path(name).suffix == '.py':
            snapshot = target / 'source_snapshot' / (digest[:16] + '_' + Path(name).name)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(Path(name).read_bytes())
            snapshots[name] = {'path': str(snapshot), 'sha256': digest}
    base.write_json(target / 'source_snapshot/manifest.json', snapshots)
    verify(target)
    return target


def memory_for(method, p, runtime):
    if method != 'simplemem':
        return legacy.Memory(method, p, runtime)
    memory = official.OfficialSimpleMemMemory(p, runtime)
    memory.core.llm_client.model = MODEL
    config = read(runtime / 'resolved_config.json')
    config.update(actual_model=MODEL, llm_model=MODEL, seed=SEED,
                  transport=TRANSPORT, enable_thinking=False)
    base.write_json(runtime / 'resolved_config.json', config)
    return memory


def new_model(root, p, env_file=ENV):
    return api().APIModel(root, env_file, workers=p['api_workers'], seed=SEED, model=MODEL)


def check_fatal(model):
    if model.fatal_errors:
        raise RuntimeError('Terminal API failure swallowed upstream; task cannot be accepted')


def run(method, output=OUTPUT, env_file=ENV):
    root = Path(output) / method
    p = verify(root)
    observations = read(root / 'observations.json')
    questions = base.read_rows(root / 'questions.jsonl')
    gold = {q['id']: q for q in base.read_rows(root / 'gold.jsonl')}
    pending = [t for t in p['tasks'] if not (root / t / 'complete.json').exists()]
    if not pending:
        return audit(method, output)
    model = new_model(root, p, env_file)
    broker = make_broker(method, model, root, p)
    try:
        encoder = install_runtime(method, broker, p)
        encoder.to(p['embedding_device'])
        base.write_json(root / 'generation_environment.json', dict(model.environment, embedding_device=str(encoder.device)))

        def one(task):
            broker.local.owner = task
            target, runtime = root / task, root / task / 'runtime'
            if runtime.exists():
                shutil.rmtree(runtime)  # only interrupted new-run task state
            runtime.mkdir(parents=True, exist_ok=True)
            memory = memory_for(method, p, runtime)
            for i, observation in enumerate(observations[task], 1):
                check_fatal(model)
                memory.store(observation)
                check_fatal(model)
                base.write_json(target / 'progress.json', {'phase': 'construction', 'done': i, 'total': len(observations[task])})
            if method == 'simplemem':
                memory.finalize()
                memory.export(target)
            answers = []
            for q in [q for q in questions if q['task'] == task]:
                query = base.retrieval_query(q)
                context, detail = memory.recall(query)
                check_fatal(model)
                context = str(context)
                messages = base.reader_messages(q, context, p)
                out = broker.request(messages, p['max_new_tokens'], kind='answer', owner=q['id'])
                check_fatal(model)
                text = out['raw_response'].strip()
                pred = text if text in ('A', 'B', 'C', 'D') and not out['hit_token_limit'] and not out.get('refusal') else None
                row = dict(q, **out)
                row.update(gold[q['id']], predicted_choice=pred, correct=pred == gold[q['id']]['gold_choice'],
                           retrieval_query=query, retrieved_context=context, retrieval_detail=detail,
                           reader_messages_sha256=base.hash_object(messages), experiment_signature=p['experiment_signature'])
                answers.append(row)
                base.write_rows(target / 'responses.jsonl', answers)
                base.write_json(target / 'progress.json', {'phase': 'reader', 'done': len(answers)})
            memory.export(target)
            base.write_json(target / 'complete.json', {'status': 'complete', 'questions': len(answers),
                'experiment_signature': p['experiment_signature'], 'responses_sha256': base.sha(target / 'responses.jsonl')})
            print(f'{method} {task}: complete {len(answers)} questions', flush=True)

        errors = []
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            futures = {pool.submit(one, task): task for task in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append({'task': futures[future], 'error': str(exc), 'traceback': traceback.format_exc()})
                    base.write_json(root / 'run_errors.json', errors)
        if errors:
            raise RuntimeError(f'{len(errors)} construction/retrieval/reader task(s) failed')
    finally:
        broker.close()
    return audit(method, output)


def validate_calls(root, p):
    """Join every successful broker output to an actual seed1024 wire response."""
    raw_path = root / 'api_responses.jsonl'
    raw = base.read_rows(raw_path) if raw_path.exists() else []
    by_response = {r['output']['response_id']: r for r in raw}
    calls, failures, counts = {}, [], Counter()
    for path in sorted((root / 'calls').glob('*.json')):
        call = read(path)
        request, out = call['request'], call['output']
        actual_wire = wire(request['messages'], request['max_new_tokens'], request['temperature'])
        logged = by_response.get(out['response_id'])
        valid = (base.hash_object(request) == path.stem and request['seed'] == SEED
                 and request['experiment_signature'] == p['experiment_signature']
                 and out['seed'] == SEED and out['requested_model'] == out['returned_model'] == MODEL
                 and out['enable_thinking'] is False
                 and out['request_sha256'] == api().request_sha256(actual_wire)
                 and logged is not None and logged['request'] == actual_wire
                 and all(out.get(k) == v for k, v in logged['output'].items())
                 and 'inherited_from' not in call)
        if not valid:
            failures.append({'check': 'actual_wire_seed_model_output_provenance', 'call': path.stem})
        setting = (request['temperature'], request['max_new_tokens'])
        if ((request['kind'] == 'answer' and setting != (0., 16))
                or (request['kind'] == 'memory' and setting not in NATIVE_MEMORY_SETTINGS[p['mode']])):
            failures.append({'check': 'unchanged_native_temperature_and_cap', 'call': path.stem})
        if out.get('finish_reason') is not None and out['hit_token_limit'] != (out['finish_reason'] == 'length'):
            failures.append({'check': 'finish_reason_cap_flag', 'call': path.stem})
        if out['input_tokens'] + request['max_new_tokens'] > p['context_budget']:
            failures.append({'check': 'native_context_budget', 'call': path.stem})
        counts[(request['kind'], request['temperature'], request['max_new_tokens'])] += 1
        calls[path.stem] = call
    return calls, failures, counts


def audit(method, output=OUTPUT):
    root = Path(output) / method
    p = verify(root)
    questions = base.read_rows(root / 'questions.jsonl')
    qmap = {q['id']: q for q in questions}
    gold = {q['id']: q for q in base.read_rows(root / 'gold.jsonl')}
    obs = read(root / 'observations.json')
    calls, failures, settings = validate_calls(root, p)
    responses, completed = [], []
    for task in p['tasks']:
        work = root / task
        if not (work / 'complete.json').exists():
            continue
        marker = read(work / 'complete.json')
        if marker['experiment_signature'] != p['experiment_signature'] or marker['responses_sha256'] != base.sha(work / 'responses.jsonl'):
            failures.append({'check': 'task_completion_hash', 'task': task})
        if read(work / 'memory_stats.json')['stored_rounds'] != len(obs[task]):
            failures.append({'check': 'full_task_construction', 'task': task})
        responses.extend(base.read_rows(work / 'responses.jsonl'))
        completed.append(task)
    seen = Counter(r['id'] for r in responses)
    for row in responses:
        identity = row['id']
        call = calls.get(row['call_key'])
        if (identity not in qmap or seen[identity] != 1 or call is None
                or call['request']['messages'] != base.reader_messages(qmap[identity], row['retrieved_context'], p)
                or row['reader_messages_sha256'] != base.hash_object(call['request']['messages'])
                or row['raw_response'] != call['output']['raw_response']
                or row['correct'] != (row['predicted_choice'] == gold[identity]['gold_choice'])):
            failures.append({'check': 'fixed_mcq_payload_output_grade', 'id': identity})
        if method in ('a_mem', 'simplemem'):
            detail = row['retrieval_detail']
            ids = detail['context_memory_ids']
            if (len(ids) != len(set(ids)) or len(ids) > 5
                    or (method == 'simplemem' and (len(ids) not in (0, 5)
                        or (not ids and not detail.get('source_empty_preserved'))))):
                failures.append({'check': 'native_context_limit', 'id': identity})
    complete = len(responses) == 371 and set(seen) == set(qmap) and len(completed) == 8
    summary = {'method': method, 'seed': SEED, 'model': MODEL,
        'status': 'complete' if complete and not failures else 'failed_audit' if failures else 'partial',
        'passed': complete and not failures, 'questions': len(responses), 'expected_questions': 371,
        'correct': sum(r['correct'] for r in responses),
        'accuracy': sum(r['correct'] for r in responses) / len(responses) if responses else None,
        'complete_tasks': completed, 'memory_rounds': 848, 'unique_llm_calls': len(calls),
        'memory_calls': sum(c['request']['kind'] == 'memory' for c in calls.values()),
        'reader_calls': sum(c['request']['kind'] == 'answer' for c in calls.values()),
        'format_errors': sum(r['predicted_choice'] is None for r in responses),
        'answer_token_limit_hits': sum(r['hit_token_limit'] for r in responses),
        'memory_token_limit_hits': sum(c['output']['hit_token_limit'] for c in calls.values() if c['request']['kind'] == 'memory'),
        'invalid_structured_outputs': sum(c['output'].get('structured_output_valid') is False for c in calls.values()),
        'native_call_settings': [{'kind': k[0], 'temperature': k[1], 'cap': k[2], 'calls': n} for k, n in settings.items()],
        'failures': failures, 'experiment_signature': p['experiment_signature']}
    order = {q['id']: i for i, q in enumerate(questions)}
    responses.sort(key=lambda r: order[r['id']])
    base.write_rows(root / 'qa_responses.jsonl', responses)
    base.write_json(root / 'summary.json', summary)
    base.write_json(root / 'stability_audit.json', summary)
    if summary['passed']:
        base.write_json(root / 'completion_manifest.json', {'status': 'complete', 'passed': True,
            'questions': 371, 'seed': SEED, 'model': MODEL, 'protocol_sha256': base.sha(root / 'protocol.json'),
            'responses_sha256': base.sha(root / 'qa_responses.jsonl'), 'audit_sha256': base.sha(root / 'stability_audit.json')})
    if failures:
        raise ValueError(f'{method}: {len(failures)} audit failure(s)')
    return summary


class FrozenBroker:
    """Rebuild ephemeral indices from this run's exact cached construction only."""
    def __init__(self, method, source, target, p, model):
        self.method, self.source, self.root, self.protocol = method, source, target, p
        self.model, self.local, self.lock = model, threading.local(), threading.RLock()
        self.live = make_broker('a_mem', model, target, p)
        self.cache = {}
        source_protocol = read(source / 'protocol.json')
        for path in sorted((source / 'calls').glob('*.json')):
            call = read(path)
            if call['request']['kind'] != 'memory':
                continue
            if (base.hash_object(call['request']) != path.stem
                    or call['request']['experiment_signature'] != source_protocol['experiment_signature']
                    or call['output']['request_sha256'] != api().request_sha256(wire(
                        call['request']['messages'], call['request']['max_new_tokens'], call['request']['temperature']))
                    or call['output']['returned_model'] != MODEL):
                raise ValueError('New-run source memory-call provenance differs')
            req = {k: v for k, v in call['request'].items() if k != 'experiment_signature'}
            if req['seed'] != SEED or call['output']['seed'] != SEED:
                raise ValueError('Cannot replay non-seed1024 memory')
            self.cache[base.hash_object(req)] = path, call

    def request(self, messages, max_tokens, *, kind='memory', response_format=None, requested_temperature=0, owner=None):
        messages = copy.deepcopy(messages)
        if self.method == 'gen_agents' and kind == 'memory':
            instruction, _ = instruction_for(messages)
            messages.insert(0, {'role': 'system', 'content': instruction})
        prepared = copy.deepcopy(messages)
        if response_format:
            schema = response_format.get('json_schema', {}).get('schema')
            instruction = 'Return only valid JSON, without Markdown fences.'
            if schema:
                instruction += ' Follow this JSON schema exactly: ' + json.dumps(schema)
            prepared.insert(0, {'role': 'system', 'content': instruction})
        req = dict(messages=prepared, max_new_tokens=int(max_tokens), kind=kind,
                   do_sample=kind == 'memory' and requested_temperature > 0,
                   temperature=float(requested_temperature) if kind == 'memory' else 0., seed=SEED,
                   response_format=response_format)
        key = base.hash_object(req)
        if key in self.cache:
            path, call = self.cache[key]
            with self.lock:
                with (self.root / 'replayed_calls.jsonl').open('a') as stream:
                    stream.write(json.dumps({'canonical_request_sha256': key, 'source_path': str(path),
                        'source_sha256': base.sha(path), 'source_call_key': path.stem,
                        'stage': 'construction' if getattr(self.local, 'construction_only', True) else 'retrieval'}) + '\n')
            return dict(call['output'], call_key=path.stem)
        if getattr(self.local, 'construction_only', True):
            base.write_json(self.root / 'missing_replay_request.json', req)
            raise ValueError('Exact new-run construction request absent from cache')
        if self.method not in ('a_mem', 'simplemem') or kind != 'memory':
            raise ValueError('Unexpected new generation in native Open retrieval')
        return self.live.request(messages, max_tokens, kind=kind, response_format=response_format,
                                 requested_temperature=requested_temperature, owner=owner)

    def close(self):
        self.live.close()


def open_prepare(method, output=OUTPUT, open_output=OPEN_OUTPUT):
    source, target = Path(output) / method, Path(open_output) / method
    p = verify(source)
    if read(source / 'completion_manifest.json')['responses_sha256'] != base.sha(source / 'qa_responses.jsonl'):
        raise ValueError('A complete audited new MCQ memory is required')
    old = {q['id']: q for q in base.read_rows(source / 'questions.jsonl')}
    questions = base.read_rows(OPEN_QUESTIONS)
    for q in questions:
        if set(q) & {'options', 'gold_choice', 'gold_ids', 'answer', 'ground_truth', 'clue'}:
            raise ValueError('Evaluation-only Open field')
        q['query_image_captions'] = copy.deepcopy(old[q['id']]['query_image_captions'])
    files = [OPEN_QUESTIONS, source / 'protocol.json', source / 'qa_responses.jsonl', source / 'observations.json']
    for task in p['tasks']:
        if method in ('gen_agents', 'a_mem', 'simplemem'):
            files.append(source / task / 'final_memory.json')
        if method in ('memoryos', 'simplemem'):
            files += [f for f in (source / task / 'runtime').rglob('*') if f.is_file()]
    p = copy.deepcopy(p)
    p['open_evaluation'] = {'source_root': str(source), 'source_seed': SEED,
        'source_hashes': {str(f): base.sha(f) for f in files},
        'questions': 371, 'questions_sha256': base.hash_object(questions),
        'policy': 'Restore new seed1024 native memory; reretrieve canonical Open questions with same fixed captions; no options/gold; Open answers generated separately.'}
    p['experiment_signature'] = signature(p)
    target.mkdir(parents=True, exist_ok=True)
    frozen(target / 'protocol.json', p)
    qpath = target / 'questions.jsonl'
    if qpath.exists() and base.read_rows(qpath) != questions:
        raise ValueError('Prepared Open questions differ')
    base.write_rows(qpath, questions)
    return source, target, p, questions


def open_run(method, output=OUTPUT, open_output=OPEN_OUTPUT, env_file=ENV):
    source, target, p, questions = open_prepare(method, output, open_output)
    if (target / 'complete.json').exists():
        marker = read(target / 'complete.json')
        if marker['retrieval_sha256'] != base.sha(target / 'retrieval.jsonl'):
            raise ValueError('Completed Open retrieval changed')
        return marker
    model = new_model(target, p, env_file)
    broker = FrozenBroker(method, source, target, p, model)
    observations = read(source / 'observations.json')
    try:
        encoder = install_runtime(method, broker, p)
        encoder.to(p['embedding_device'])
        base.write_json(target / 'generation_environment.json', dict(model.environment, embedding_device=str(encoder.device)))

        def one(task):
            work, runtime = target / task, target / task / 'runtime'
            work.mkdir(parents=True, exist_ok=True)
            if (work / 'complete.json').exists():
                marker = read(work / 'complete.json')
                if marker['retrieval_sha256'] != base.sha(work / 'retrieval.jsonl'):
                    raise ValueError('Task retrieval changed')
                return base.read_rows(work / 'retrieval.jsonl')
            if runtime.exists():
                shutil.rmtree(runtime)
            if method in ('memoryos', 'simplemem'):
                shutil.copytree(source / task / 'runtime', runtime)
                copied = {str(f.relative_to(runtime)): base.sha(f) for f in runtime.rglob('*') if f.is_file()}
                expected = {str(f.relative_to(source / task / 'runtime')): base.sha(f) for f in (source / task / 'runtime').rglob('*') if f.is_file()}
                if copied != expected:
                    raise ValueError('Native database copy differs')
                base.write_json(work / 'runtime_copy_audit.json', {'passed': True, 'files': copied})
            runtime.mkdir(parents=True, exist_ok=True)
            broker.local.owner, broker.local.construction_only = task, True
            memory = memory_for(method, p, runtime)
            if method in ('gen_agents', 'a_mem'):
                for row in observations[task]:
                    memory.store(row)
                actual = ([memory.core.storage.get_memory_element_by_mid(i) for i in range(memory.core.storage.get_element_number())]
                          if method == 'gen_agents' else {k: vars(v) for k, v in memory.core.memory_system.memories.items()})
                expected = read(source / task / 'final_memory.json')
                ignore_access = lambda notes: {k: {f: v for f, v in n.items() if f != 'last_accessed'} for k, n in notes.items()}
                if (actual != expected if method == 'gen_agents' else ignore_access(actual) != ignore_access(expected)):
                    raise ValueError('Restored native memory differs')
                base.write_json(work / 'restored_memory.json', actual)
                if method == 'gen_agents':
                    native = memory.core.recall_op.__retention__
                    def retention(ids, timestamp):
                        memory.last_ids = [int(i) for i in ids]
                        return native(ids, timestamp)
                    memory.core.recall_op.__retention__ = retention
            elif method == 'simplemem':
                if {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()} != {e['entry_id']: e for e in read(source / task / 'final_memory.json')}:
                    raise ValueError('Restored SimpleMem entries differ')
                memory.finalized = True
            broker.local.construction_only = False
            result = []
            for q in [q for q in questions if q['task'] == task]:
                query = base.retrieval_query(q)
                context, detail = memory.recall(query)
                check_fatal(model)
                if method == 'gen_agents':
                    detail = {'context_memory_count': len(memory.last_ids), 'context_memory_indices': memory.last_ids}
                result.append(dict(q, method=method, retrieval_query=query, context=str(context),
                                   retrieved_context=str(context), retrieval_detail=detail, context_sha256=base.hash_object(str(context))))
                base.write_rows(work / 'retrieval.jsonl', result)
            if method == 'memoryos':
                memory.core.client.shutdown()
            base.write_json(work / 'complete.json', {'status': 'complete', 'questions': len(result),
                                                   'retrieval_sha256': base.sha(work / 'retrieval.jsonl')})
            return result

        result = []
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            for future in as_completed([pool.submit(one, task) for task in p['tasks']]):
                result.extend(future.result())
    finally:
        broker.close()
    order = {q['id']: i for i, q in enumerate(questions)}
    if len(result) != 371 or {q['id'] for q in result} != set(order):
        raise ValueError('Incomplete Open retrieval')
    result.sort(key=lambda q: order[q['id']])
    base.write_rows(target / 'retrieval.jsonl', result)
    manifest = []
    question_map = {q['id']: q for q in questions}
    for row in result:
        if (any(row[k] != v for k, v in question_map[row['id']].items())
                or row['retrieval_query'] != base.retrieval_query(question_map[row['id']])
                or row['context_sha256'] != base.hash_object(row['context'])):
            raise ValueError('Native Open retrieval changed its frozen question/context')
        if method in ('a_mem', 'simplemem'):
            detail = row['retrieval_detail']
            ids = detail['context_memory_ids']
            if len(ids) != len(set(ids)) or len(ids) > 5:
                raise ValueError('Native Open context bound violated')
            if method == 'simplemem' and not (len(ids) == 5 or (not ids and detail.get('source_empty_preserved'))):
                raise ValueError('SimpleMem Open context is neither five entries nor preserved native empty')
        messages = [{'role': 'system', 'content': OPEN_SYSTEM},
                    {'role': 'assistant', 'content': 'Retrieved memory context:\n' + row['context']},
                    {'role': 'user', 'content': row['retrieval_query']}]
        manifest.append(dict(row, system=OPEN_SYSTEM, user_text=row['retrieval_query'], reader_messages=messages,
            source_query_image_paths=row['query_image_paths'], query_image_paths=[],
            reader_messages_sha256=base.hash_object(messages), payload_sha256=base.hash_object(messages)))
    base.write_rows(target / 'reader_manifest.jsonl', manifest)
    _, failures, settings = validate_calls(target, p)
    replays = base.read_rows(target / 'replayed_calls.jsonl') if (target / 'replayed_calls.jsonl').exists() else []
    for event in replays:
        path = Path(event['source_path'])
        req = read(path)['request']
        canonical = {k: v for k, v in req.items() if k != 'experiment_signature'}
        if (path.parent != source / 'calls' or base.sha(path) != event['source_sha256']
                or req['seed'] != SEED or base.hash_object(canonical) != event['canonical_request_sha256']):
            failures.append({'check': 'new_seed1024_construction_replay_provenance', 'path': str(path)})
    for path, digest in p['open_evaluation']['source_hashes'].items():
        if base.sha(Path(path)) != digest:
            failures.append({'check': 'new_mcq_source_unchanged', 'path': path})
    if failures:
        base.write_json(target / 'audit_failures.json', failures)
        raise ValueError('Open retrieval audit failed')
    marker = {'status': 'complete', 'passed': True, 'method': method, 'seed': SEED, 'questions': 371,
              'retrieval_sha256': base.sha(target / 'retrieval.jsonl'),
              'reader_manifest_sha256': base.sha(target / 'reader_manifest.jsonl'),
              'protocol_sha256': base.sha(target / 'protocol.json'),
              'replayed_call_events': len(replays), 'construction_api_calls': 0,
              'new_call_settings': [{'kind': k[0], 'temperature': k[1], 'cap': k[2], 'calls': n} for k, n in settings.items()]}
    base.write_json(target / 'complete.json', marker)
    base.write_json(target / 'ready_for_reader.json', marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(REFERENCES), required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit', 'analyze', 'open-prepare', 'open-run'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--open-output', type=Path, default=OPEN_OUTPUT)
    parser.add_argument('--embedding-device')
    parser.add_argument('--env-file', type=Path, default=ENV)
    args = parser.parse_args()
    try:
        if args.phase == 'prepare':
            path = prepare(args.method, args.output, args.embedding_device)
            report = {'status': 'prepared', 'method': args.method, 'seed': SEED, 'output': str(path), 'questions': 371}
        elif args.phase == 'run':
            report = run(args.method, args.output, args.env_file)
        elif args.phase in ('audit', 'analyze'):
            report = audit(args.method, args.output)
        elif args.phase == 'open-prepare':
            _, path, p, qs = open_prepare(args.method, args.output, args.open_output)
            report = {'status': 'prepared', 'method': args.method, 'seed': SEED, 'output': str(path), 'questions': len(qs)}
        else:
            report = open_run(args.method, args.output, args.open_output, args.env_file)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        target = (args.open_output if args.phase.startswith('open-') else args.output) / args.method
        base.write_json(target / 'stage_failure.json', {'phase': args.phase, 'error': str(exc), 'traceback': traceback.format_exc(), 'time': now()})
        raise


if __name__ == '__main__':
    main()
