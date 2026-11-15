"""Official MemEye Open retrieval from the frozen GPT-5 text-method memories.

The screenshot MCQ runs identify sources only. Public Open questions are used
without options or gold, with the original GPT-5 question-image captions.
"""
from __future__ import annotations
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import threading

from scripts.evaluation import mcq_reader  # dependency bootstrap
from scripts.memory import text_memory_baselines as base
from scripts.memory import simplemem_official as official
from scripts.evaluation import gpt5_text_memory_transport as api
from scripts.evaluation.gpt5_text_memory_adapters import install_runtime
from scripts.evaluation.continue_deepseek_text_methods_streaming import StreamingBroker
from scripts.evaluation.continue_deepseek_ga_format import instruction_for
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM
from scripts.evaluation.simplemem_reflection_ablation import operation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_gpt5_open_20261105/text_methods'
QUESTIONS = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
SOURCES = {
    'gen_agents': ('results/memeye_gpt5_seed42_text_methods_20261025/gen_agents', 177),
    'memoryos': ('results/memeye_gpt5_seed42_text_methods_20261025/memoryos', 204),
    'a_mem': ('results/memeye_gpt5_seed42_text_methods_20261025/a_mem', 199),
    'simplemem': ('results/simplemem_official_gpt5_memeye_k10_rrf5_reflection_20261029/top5/simplemem_official_text_context5', 191),
}


def read(path):
    return json.loads(Path(path).read_text())


def normalized(request):
    value = copy.deepcopy(request)
    value.pop('experiment_signature', None)
    return value


def freeze(path, value):
    if path.exists():
        assert read(path) == value, 'existing protocol differs: ' + str(path)
    else:
        base.write_json(path, value)


class FrozenGPTBroker:
    """Exact source-cache replay during construction; new GPT calls only for queries."""
    def __init__(self, method, source, target, protocol, model):
        self.method, self.source, self.root, self.protocol = method, source, target, protocol
        self.model, self.local, self.lock = model, threading.local(), threading.RLock()
        self.live = StreamingBroker(model, target, protocol, 8)
        self.cache = {}
        for path in sorted((source / 'calls').glob('*.json')):
            call = read(path)
            assert base.hash_object(call['request']) == path.stem, str(path)
            if call['request']['kind'] != 'memory':
                continue
            key = base.hash_object(normalized(call['request']))
            if key in self.cache:
                assert self.cache[key][1]['output'] == call['output'], key
            wire = api.build_request(call['request']['messages'], call['request']['max_new_tokens'], call['request']['temperature'])
            assert api.request_sha256(wire) == call['output']['request_sha256']
            assert call['output']['returned_model'] == protocol['caption_model']
            self.cache[key] = (path, call)

    def request(self, messages, max_tokens, *, kind='memory', response_format=None,
                requested_temperature=0, owner=None):
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
        request = {'messages': prepared, 'max_new_tokens': int(max_tokens),
                   'kind': kind, 'do_sample': kind == 'memory' and requested_temperature > 0,
                   'temperature': float(requested_temperature) if kind == 'memory' else 0.0,
                   'seed': self.protocol['seed'], 'response_format': response_format}
        key = base.hash_object(request)
        if key in self.cache:
            path, call = self.cache[key]
            with self.lock:
                with (self.root / 'replayed_calls.jsonl').open('a') as stream:
                    stream.write(json.dumps({'owner': owner or getattr(self.local, 'owner', 'internal'),
                        'stage': 'construction' if getattr(self.local, 'construction_only', True) else 'retrieval',
                        'canonical_request_sha256': key, 'source_path': str(path),
                        'source_sha256': base.sha(path), 'source_call_key': path.stem}) + '\n')
            return dict(call['output'], call_key=path.stem)
        if getattr(self.local, 'construction_only', True):
            base.write_json(self.root / 'missing_replay_requests' / (key + '.json'), request)
            raise ValueError('missing cached GPT construction request: ' + key)
        if self.method not in ('a_mem', 'simplemem') or kind != 'memory':
            raise ValueError('Only A-Mem/SimpleMem native query LLM calls are allowed')
        if self.model.fatal_errors:
            raise RuntimeError('terminal API failure; inspect api_attempts.jsonl')
        return self.live.request(messages, max_tokens, kind=kind, response_format=response_format,
            requested_temperature=requested_temperature, owner=owner or getattr(self.local, 'owner', 'internal'))

    def close(self):
        self.live.close()


def without_access(notes):
    return {k: {f: v for f, v in value.items() if f != 'last_accessed'} for k, value in notes.items()}


def prepare(method, output, embedding_device):
    source_name, expected = SOURCES[method]
    source, target = ROOT / source_name, output / method
    parent, summary = read(source / 'protocol.json'), read(source / 'summary.json')
    qa = base.read_rows(source / 'qa_responses.jsonl')
    assert summary['status'] == 'complete' and len(qa) == len({q['id'] for q in qa}) == 371
    assert sum(bool(q['correct']) for q in qa) == expected
    assert parent['model'] == 'gpt-5' and parent['seed'] == 42
    assert parent['caption_model'] == 'gpt-5-2025-08-07'
    original = {q['id']: q for q in qa}
    canonical = base.read_rows(QUESTIONS)
    questions = []
    for q in canonical:
        assert set(q).isdisjoint({'options', 'gold_choice', 'gold_ids', 'answer', 'ground_truth'})
        row = copy.deepcopy(q)
        row['query_image_captions'] = copy.deepcopy(original[q['id']]['query_image_captions'])
        assert len(row['query_image_captions']) == len(row['query_image_paths'])
        questions.append(row)
    assert len(questions) == len({q['id'] for q in questions}) == 371
    files = [source / 'protocol.json', source / 'summary.json', source / 'qa_responses.jsonl',
             source / 'questions.jsonl', QUESTIONS]
    for task in parent['tasks']:
        if method in ('gen_agents', 'a_mem'):
            files.append(source / task / 'final_memory.json')
        else:
            files.extend(v for v in (source / task / 'runtime').rglob('*') if v.is_file())
            if method == 'simplemem':
                files.append(source / task / 'final_memory.json')
    if method in ('gen_agents', 'a_mem'):
        files.append(source / 'observations.json')
    # Freeze all cached construction/planning outputs and the actual transport.
    files.extend(sorted((source / 'calls').glob('*.json')))
    code = [Path(__file__), Path(base.__file__), Path(official.__file__), Path(api.__file__),
            ROOT/'scripts/evaluation/gpt5_text_memory_adapters.py',
            ROOT/'scripts/evaluation/continue_deepseek_text_methods_streaming.py',
            ROOT/'scripts/evaluation/continue_deepseek_text_methods_parallel.py',
            ROOT/'scripts/evaluation/continue_deepseek_ga_format.py',
            ROOT/'scripts/evaluation/continue_deepseek_memoryos_format.py',
            ROOT/'scripts/memory/text_memory_transport.py']
    p = copy.deepcopy(parent)
    p.update(task_workers=8, api_workers=8, embedding_device=embedding_device,
             open_evaluation={'method': method, 'original_mcq_correct': expected,
        'source_root': str(source), 'questions': 371,
        'source_hashes': {str(f): base.sha(f) for f in files},
        'code_sha256': {str(f): base.sha(f) for f in code},
        'restoration': 'Exact GPT-5 cached construction replay; frozen final-memory equality check'
            if method in ('gen_agents', 'a_mem') else 'Byte-copy original persisted memory; validate before retrieval',
        'query_policy': 'Official Open question plus the original GPT-5 seed42 question-image captions; no options or gold',
        'context_policy': 'Original native retrieval and original final-context bound',
        'original_embedding_device': parent.get('embedding_device', 'cpu'),
        'embedding_device_note': 'Same frozen MiniLM weights; requested CUDA device preserves original CUDA computation for the three legacy adapters; official SimpleMem remains CPU.',
        'gold_access': 'none in retrieval or reader-input generation',
        'reader_policy': 'Open system, assistant retrieved memory, user Open question plus original GPT-5 query-image captions; no native images or options',
        'generation': 'Official GPT-5, seed42, minimal reasoning, temperature omitted, max(2048,native logical budget); context128000',
        'questions_sha256': base.hash_object(questions)})
    p['experiment_signature'] = base.hash_object({k: v for k, v in p.items() if k != 'experiment_signature'})
    target.mkdir(parents=True, exist_ok=True)
    freeze(target / 'protocol.json', p)
    if (target / 'questions.jsonl').exists():
        assert base.read_rows(target / 'questions.jsonl') == questions
    else:
        base.write_rows(target / 'questions.jsonl', questions)
    return source, target, p, questions


def run(method, output, embedding_device):
    source, target, protocol, questions = prepare(method, output, embedding_device)
    if (target / 'complete.json').exists():
        return finalize(method, target, source, protocol, questions)
    model = api.APIModel(target, ENV, workers=8)
    model.torch.set_num_threads(2)
    broker = FrozenGPTBroker(method, source, target, protocol, model)
    encoder = install_runtime(method, broker, protocol)
    if embedding_device != 'cpu':
        assert model.torch.cuda.is_available(), 'GPU required for source-matched MiniLM computation'
        encoder.to(embedding_device)
    base.write_json(target / 'generation_environment.json', model.environment | {
        'embedding_device': str(encoder.device), 'embedding_model_path': protocol['embedding_model_path']})
    if method == 'simplemem':
        assert official.verify_installation() == protocol['upstream']
    else:
        assert base.verify_vendor() == protocol['upstream']
    observations = read(source / 'observations.json') if method in ('gen_agents', 'a_mem') else {}

    def one(task):
        dest = target / task
        dest.mkdir(parents=True, exist_ok=True)
        broker.local.owner, broker.local.construction_only = task, True
        if (dest / 'complete.json').exists():
            done = read(dest / 'complete.json')
            assert done['retrieval_sha256'] == base.sha(dest / 'retrieval.jsonl')
            return base.read_rows(dest / 'retrieval.jsonl')
        runtime = dest / 'runtime'
        if method in ('memoryos', 'simplemem'):
            if runtime.exists():
                shutil.rmtree(runtime)
            shutil.copytree(source / task / 'runtime', runtime)
            copied = {str(f.relative_to(runtime)): base.sha(f) for f in runtime.rglob('*') if f.is_file()}
            expected = {str(f.relative_to(source / task / 'runtime')): base.sha(f)
                        for f in (source / task / 'runtime').rglob('*') if f.is_file()}
            assert copied == expected
            base.write_json(dest / 'runtime_copy_audit.json', {'files': copied, 'source': str(source / task / 'runtime')})
        runtime.mkdir(parents=True, exist_ok=True)
        memory = official.OfficialSimpleMemMemory(protocol, runtime) if method == 'simplemem' else base.Memory(method, protocol, runtime)
        if method in ('gen_agents', 'a_mem'):
            for index, row in enumerate(observations[task], 1):
                memory.store(row)
                if index % 25 == 0:
                    print(f'{method} {task}: frozen memory replay {index}/{len(observations[task])}', flush=True)
            if method == 'gen_agents':
                actual = [memory.core.storage.get_memory_element_by_mid(i) for i in range(memory.core.storage.get_element_number())]
                assert actual == read(source / task / 'final_memory.json'), task
                native = memory.core.recall_op.__retention__
                def retention(ids, timestamp):
                    memory.last_ids = [int(i) for i in ids]
                    return native(ids, timestamp)
                memory.core.recall_op.__retention__ = retention
            else:
                actual = {k: vars(v) for k, v in memory.core.memory_system.memories.items()}
                assert without_access(actual) == without_access(read(source / task / 'final_memory.json')), task
            base.write_json(dest / 'restored_memory.json', actual)
        elif method == 'simplemem':
            actual = {e.entry_id: e.model_dump() for e in memory.core.get_all_memories()}
            assert actual == {e['entry_id']: e for e in read(source / task / 'final_memory.json')}, task
            memory.core.llm_client.model = 'gpt-5'
            memory.finalized = True
        broker.local.construction_only = False
        result = []
        selected = [q for q in questions if q['task'] == task]
        for i, q in enumerate(selected):
            query = base.retrieval_query(q)
            context, detail = memory.recall(query)
            if model.fatal_errors:
                raise RuntimeError('native method swallowed terminal API failure; cannot accept retrieval')
            if method == 'gen_agents':
                detail = {'context_memory_count': len(memory.last_ids), 'context_memory_indices': memory.last_ids,
                          'context_memories': [memory.core.storage.get_memory_element_by_mid(j) for j in memory.last_ids]}
            context = str(context)
            result.append(dict(q, method=method, retrieval_query=query, context=context,
                retrieved_context=context, retrieval_detail=detail, context_sha256=base.hash_object(context)))
            base.write_rows(dest / 'retrieval.jsonl', result)
            if i % 10 == 0 or i + 1 == len(selected):
                print(f'{method} {task}: Open retrieval {i+1}/{len(selected)}', flush=True)
        if method == 'memoryos':
            memory.core.client.shutdown()
        base.write_json(dest / 'complete.json', {'questions': len(result),
            'retrieval_sha256': base.sha(dest / 'retrieval.jsonl'), 'memory_rebuilt_with_new_llm_calls': False})
        return result

    results = []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(one, task) for task in protocol['tasks']]):
                results.extend(future.result())
    finally:
        broker.close()
    order = {q['id']: i for i, q in enumerate(questions)}
    assert len(results) == len({r['id'] for r in results}) == 371
    results.sort(key=lambda r: order[r['id']])
    base.write_rows(target / 'retrieval.jsonl', results)
    base.write_json(target / 'complete.json', {'method': method, 'questions': 371,
        'retrieval_sha256': base.sha(target / 'retrieval.jsonl'), 'protocol_sha256': base.sha(target / 'protocol.json')})
    finalize(method, target, source, protocol, questions)


def finalize(method, target, source, protocol, questions):
    done = read(target / 'complete.json')
    assert done['questions'] == 371 and done['retrieval_sha256'] == base.sha(target / 'retrieval.jsonl')
    assert done['protocol_sha256'] == base.sha(target / 'protocol.json')
    for name, expected in (protocol['open_evaluation']['source_hashes'] | protocol['open_evaluation']['code_sha256']).items():
        assert base.sha(Path(name)) == expected, name
    rows = base.read_rows(target / 'retrieval.jsonl')
    original = {q['id']: q for q in base.read_rows(source / 'qa_responses.jsonl')}
    canonical = {q['id']: q for q in questions}
    assert len(rows) == len({r['id'] for r in rows}) == 371
    counts, calls, manifest, evidence = Counter(), Counter(), [], {}
    for row in rows:
        q = canonical[row['id']]
        assert all(row[k] == v for k, v in q.items())
        assert row['retrieval_query'] == base.retrieval_query(q)
        assert row['context_sha256'] == base.hash_object(row['context'])
        assert set(row).isdisjoint({'options', 'gold_choice', 'gold_ids', 'ground_truth', 'answer'})
        detail = row['retrieval_detail']
        if method in ('gen_agents', 'a_mem', 'simplemem'):
            count = detail['context_memory_count']
            if method == 'gen_agents':
                assert count == 10
            elif method == 'a_mem':
                assert count == len(set(detail['context_memory_ids'])) == 5
            else:
                assert count == len(set(detail['context_memory_ids']))
                assert count == 5 or (count == 0 and detail['source_empty_preserved'])
        else:
            count = len(detail['retrieved_pages'])
            assert count <= 10
            assert len(detail['retrieved_user_knowledge']) <= 20 and len(detail['retrieved_assistant_knowledge']) <= 20
        counts[count] += 1
        old = original[row['id']]
        call_path = source / 'calls' / (old['call_key'] + '.json')
        call = read(call_path)
        assert base.hash_object(call['request']) == call_path.stem
        assert call['request']['messages'] == base.reader_messages(old, old['retrieved_context'], read(source / 'protocol.json'))
        evidence[str(call_path)] = base.sha(call_path)
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'assistant', 'content': 'Retrieved memory context:\n' + row['context']},
                    {'role': 'user', 'content': row['retrieval_query']}]
        manifest.append(dict(row, system=SYSTEM, user_text=row['retrieval_query'], reader_messages=messages,
            source_query_image_paths=row['query_image_paths'], query_image_paths=[],
            query_image_policy='Original GPT-5 seed42 query-image captions in user text; no native query images.',
            reader_messages_sha256=base.hash_object(messages), payload_sha256=base.hash_object(messages)))
    for path in (target / 'calls').glob('*.json'):
        call = read(path)
        req, out = call['request'], call['output']
        assert base.hash_object(req) == path.stem and req['experiment_signature'] == protocol['experiment_signature']
        assert req['kind'] == 'memory' and req['seed'] == 42
        wire = api.build_request(req['messages'], req['max_new_tokens'], req['temperature'])
        assert api.request_sha256(wire) == out['request_sha256']
        assert out['returned_model'] == protocol['caption_model'] and out['requested_model'] == 'gpt-5'
        assert out['input_tokens'] + wire['max_completion_tokens'] <= protocol['context_budget']
        if method == 'a_mem':
            users = [m['content'] for m in req['messages'] if m['role'] == 'user']
            assert len(users) == 1 and users[0].startswith('Given the following question, generate several keywords')
            assert req['max_new_tokens'] == 1000
            calls['query_keywords'] += 1
        elif method == 'simplemem':
            name = operation(req)
            assert name in ('information_plan', 'query_plan', 'query_analysis', 'reflection_assessment', 'reflection_queries')
            assert req['max_new_tokens'] == 16384
            calls[name] += 1
        else:
            raise AssertionError('GA and MemoryOS must not need new generation for retrieval')
    replay = base.read_rows(target / 'replayed_calls.jsonl') if (target / 'replayed_calls.jsonl').exists() else []
    for item in replay:
        assert base.sha(Path(item['source_path'])) == item['source_sha256']
    base.write_rows(target / 'reader_manifest.jsonl', manifest)
    base.write_json(target / 'reader_input_audit.json', {'method': method, 'questions': 371,
        'source_mcq_correct': SOURCES[method][1], 'native_roles': ['system', 'assistant', 'user'],
        'source_reader_payloads_verified': True, 'source_hashes': evidence,
        'original_question_image_questions': sum(bool(r['source_query_image_paths']) for r in manifest),
        'query_image_caption_model': protocol['caption_model'], 'native_reader_images': 0,
        'reader_manifest_sha256': base.sha(target / 'reader_manifest.jsonl')})
    report = {'status': 'complete', 'method': method, 'questions': 371,
        'source_mcq_correct': SOURCES[method][1], 'source_files_unchanged': True,
        'memory_construction_new_llm_calls': 0, 'new_llm_calls': dict(calls),
        'replayed_call_events': len(replay), 'context_count_distribution': dict(counts),
        'open_queries_changed': sum(r['retrieval_query'] != original[r['id']]['retrieval_query'] for r in rows),
        'reader_manifest_sha256': base.sha(target / 'reader_manifest.jsonl'),
        'script_sha256': base.sha(Path(__file__))}
    base.write_json(target / 'validation_final.json', report)
    base.write_json(target / 'ready_for_reader.json', {'method': method, 'questions': 371,
        'reader_manifest_sha256': base.sha(target / 'reader_manifest.jsonl'),
        'retrieval_audit_sha256': base.sha(target / 'validation_final.json'),
        'reader_input_audit_sha256': base.sha(target / 'reader_input_audit.json')})
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SOURCES), required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--embedding-device', default=None)
    parser.add_argument('--phase', choices=('prepare', 'run', 'finalize'), default='run')
    args = parser.parse_args()
    device = args.embedding_device or ('cpu' if args.method == 'simplemem' else 'cuda:0')
    if args.phase == 'run':
        run(args.method, args.output, device)
    else:
        source, target, protocol, questions = prepare(args.method, args.output, device)
        if args.phase == 'finalize':
            finalize(args.method, target, source, protocol, questions)
        else:
            print(json.dumps({'method': args.method, 'source': str(source), 'questions': len(questions), 'correct': SOURCES[args.method][1]}))


if __name__ == '__main__':
    main()
