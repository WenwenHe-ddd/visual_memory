"""Reretrieve official MemEye Open queries from the frozen Qwen text memories.

Old memory-building calls are replayed exactly when an index was not saved;
no memory text is regenerated. Native retrievers and context bounds are kept.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.evaluation import mcq_reader
from scripts.memory import text_memory_baselines as base
from scripts.memory import simplemem_official as official

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_qwen_other_open_20261105/text_methods'
QUESTIONS = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
SOURCES = {
    'gen_agents': ('results/visual_memory_pipeline/gen_agents', 174),
    'memoryos': ('results/visual_memory_pipeline/memoryos', 177),
    'a_mem': ('results/visual_memory_pipeline/top5/a_mem_context5', 195),
    'simplemem': ('results/simplemem_official_qwen_memeye_k10_rrf5_reflection_20261028/top5/simplemem_official_text_context5', 186),
}


def read(path):
    return json.loads(Path(path).read_text())


def normalized(request):
    value = copy.deepcopy(request)
    value.pop('experiment_signature', None)
    value.setdefault('temperature', 0.0)
    return value


class FrozenBroker:
    """Serve exact historical memory requests; generate only genuinely new queries."""
    def __init__(self, source, target, protocol, model=None):
        import torch
        self.source, self.root, self.protocol = source, target, protocol
        self.local, self.lock = threading.local(), threading.RLock()
        self.model = model or SimpleNamespace(torch=torch)
        self.live = base.Broker(model, target, protocol, protocol['generation_batch_size']) if model else None
        self.cache = {}
        for path in sorted((source / 'calls').glob('*.json')):
            call = read(path)
            if base.hash_object(call['request']) != path.stem:
                raise ValueError('source call hash differs: ' + str(path))
            if call['request']['kind'] != 'memory':
                continue
            key = base.hash_object(normalized(call['request']))
            if key in self.cache and self.cache[key][1]['output'] != call['output']:
                raise ValueError('ambiguous source call: ' + key)
            self.cache[key] = (path, call)
        self.local.construction_only = True

    def request(self, messages, max_tokens, *, kind='memory', response_format=None,
                requested_temperature=0, owner=None):
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
                        'canonical_request_sha256': key, 'source_path': str(path),
                        'source_sha256': base.sha(path), 'source_call_key': path.stem}) + '\n')
            return dict(call['output'], call_key=path.stem)
        if getattr(self.local, 'construction_only', True) or self.live is None:
            base.write_json(self.root / 'missing_replay_request.json', request)
            raise ValueError('missing cached construction/request: ' + key)
        return self.live.request(messages, max_tokens, kind=kind, response_format=response_format,
            requested_temperature=requested_temperature, owner=owner or getattr(self.local, 'owner', 'internal'))

    def close(self):
        if self.live:
            self.live.close()


def without_access(notes):
    return {k: {f: v for f, v in value.items() if f != 'last_accessed'} for k, value in notes.items()}


def make_protocol(method, target):
    source_name, expected = SOURCES[method]
    source = ROOT / source_name
    p = read(source / 'protocol.json')
    summary = read(source / 'summary.json')
    # Original reader records provide the screenshot identity independent of labels.
    qa = base.read_rows(source / 'qa_responses.jsonl')
    assert len(qa) == 371 and sum(bool(q['correct']) for q in qa) == expected
    questions = base.read_rows(QUESTIONS)
    assert len(questions) == 371 and len({q['id'] for q in questions}) == 371
    qold = {q['id']: q for q in qa}
    for q in questions:
        assert q['query_image_captions'] == qold[q['id']]['query_image_captions']
        assert q['query_image_paths'] == qold[q['id']]['query_image_paths']
    files = [source / 'protocol.json', source / 'summary.json', source / 'qa_responses.jsonl', QUESTIONS]
    for task in p['tasks']:
        if method in ('gen_agents', 'a_mem'):
            files.append(source / task / 'final_memory.json')
        else:
            files.extend(v for v in (source / task / 'runtime').rglob('*') if v.is_file())
            if method == 'simplemem':
                files.append(source / task / 'final_memory.json')
    if method in ('gen_agents', 'a_mem'):
        files.append(source / 'observations.json')
    p = copy.deepcopy(p)
    p['open_evaluation'] = {'method': method, 'original_mcq_correct': expected,
        'source_root': str(source), 'questions': 371,
        'source_hashes': {str(f): base.sha(f) for f in files},
        'restoration': 'Exact cached construction replay; frozen final-memory equality check'
            if method in ('gen_agents', 'a_mem') else 'Byte-copy original persisted memory; validate before retrieval',
        'query_policy': 'Official Open question plus the same cached Qwen question-image captions; no options or gold',
        'context_policy': 'Original native algorithm and original final-context bound',
        'code_sha256': base.sha(Path(__file__))}
    p['experiment_signature'] = base.hash_object({k: v for k, v in p.items() if k != 'experiment_signature'})
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'protocol.json').exists():
        assert read(target / 'protocol.json') == p, 'existing Open protocol differs'
    else:
        base.write_json(target / 'protocol.json', p)
    return source, p, questions


def run(method, output, workers=4, allow_gpu=False, limit=None, rpc_url=None):
    target = output / method
    source, protocol, questions = make_protocol(method, target)
    if rpc_url:
        from scripts.evaluation.memeye_open_qwen_service import RPCQwen
        import torch
        model = RPCQwen(rpc_url)
        model.torch = torch
        protocol['generation_batch_size'] = 1
    else:
        model = base.LocalMemoryQwen(protocol) if allow_gpu else None
    if model:
        base.write_json(target / 'generation_environment.json', model.environment)
    broker = FrozenBroker(source, target, protocol, model)
    encoder = base.install_runtime(broker, protocol)
    if method == 'simplemem':
        official.verify_installation()
    observations = read(source / 'observations.json') if method in ('gen_agents', 'a_mem') else {}
    all_results = []

    def one(task):
        dest = target / task
        dest.mkdir(parents=True, exist_ok=True)
        broker.local.owner = task
        broker.local.construction_only = True
        if (dest / 'complete.json').exists():
            done = read(dest / 'complete.json')
            assert done['retrieval_sha256'] == base.sha(dest / 'retrieval.jsonl')
            return base.read_rows(dest / 'retrieval.jsonl')
        runtime = dest / 'runtime'
        if method in ('memoryos', 'simplemem') and not runtime.exists():
            shutil.copytree(source / task / 'runtime', runtime)
        runtime.mkdir(parents=True, exist_ok=True)
        memory = official.OfficialSimpleMemMemory(protocol, runtime) if method == 'simplemem' else base.Memory(method, protocol, runtime)
        if method in ('gen_agents', 'a_mem'):
            for row in observations[task]:
                memory.store(row)
            if method == 'gen_agents':
                actual = [memory.core.storage.get_memory_element_by_mid(i) for i in range(memory.core.storage.get_element_number())]
                assert actual == read(source / task / 'final_memory.json'), task
                # Capture rankings before the original native recall performs recency retention.
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
            expected = {e['entry_id']: e for e in read(source / task / 'final_memory.json')}
            assert actual == expected, task
            memory.finalized = True
        # Only query generation may hit the live model after exact frozen construction.
        broker.local.construction_only = False
        result = []
        selected = [q for q in questions if q['task'] == task]
        if limit:
            selected = selected[:limit]
        for i, q in enumerate(selected):
            query = base.retrieval_query(q)
            context, detail = memory.recall(query)
            if method == 'gen_agents':
                detail = {'context_memory_count': len(memory.last_ids), 'context_memory_indices': memory.last_ids,
                          'context_memories': [memory.core.storage.get_memory_element_by_mid(j) for j in memory.last_ids]}
            result.append(dict(q, method=method, retrieval_query=query, context=context,
                retrieved_context=context, retrieval_detail=detail, context_sha256=base.hash_object(context)))
            base.write_rows(dest / 'retrieval.jsonl', result)
            if i % 10 == 0 or i + 1 == len(selected):
                print(f'{method} {task}: {i+1}/{len(selected)}', flush=True)
        if method == 'memoryos':
            memory.core.client.shutdown()
        base.write_json(dest / 'complete.json', {'questions': len(result),
            'retrieval_sha256': base.sha(dest / 'retrieval.jsonl'), 'memory_rebuilt_with_new_llm_calls': False})
        return result

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(one, task) for task in protocol['tasks']]
            for future in as_completed(jobs):
                all_results.extend(future.result())
    finally:
        broker.close()
    order = {q['id']: i for i, q in enumerate(questions)}
    all_results.sort(key=lambda q: order[q['id']])
    if not limit:
        assert len(all_results) == 371 and {q['id'] for q in all_results} == set(order)
    base.write_rows(target / 'retrieval.jsonl', all_results)
    for path, expected in protocol['open_evaluation']['source_hashes'].items():
        assert base.sha(Path(path)) == expected, path
    base.write_json(target / 'complete.json', {'method': method, 'questions': len(all_results),
        'retrieval_sha256': base.sha(target / 'retrieval.jsonl'), 'source_files_unchanged': True,
        'source_mcq_correct': SOURCES[method][1], 'protocol_sha256': base.sha(target / 'protocol.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(SOURCES), required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--allow-gpu', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rpc-url')
    args = parser.parse_args()
    run(args.method, args.output, args.workers, args.allow_gpu, args.limit, args.rpc_url)
