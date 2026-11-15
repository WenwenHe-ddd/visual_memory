"""Mem-Gallery MemoryOS with the verified GPT-5 GA captions and reader policy.

Keep the completed GA experiment read-only. Build isolated native MemoryOS
states, preserving source metadata and the official question-image captions.
Only the separate analysis module opens gold labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from scripts.evaluation import mem_gallery_gpt5_gen_agents as ga
from scripts.evaluation import mem_gallery_gpt5_image_formats as images
from scripts.evaluation import gpt5_text_memory_adapters as adapters

ROOT, SOURCE = ga.ROOT, ga.DEFAULT_OUTPUT
OUTPUT = ROOT / 'results/mem_gallery_gpt5_memoryos_20261030'
read, rows, sha, obj = ga.read, ga.rows, ga.sha, ga.obj
dump, write_rows, freeze = ga.dump, ga.write_rows, ga.freeze
native, transport = ga.native, ga.transport


def user_text(row):
    """Same lossless metadata adaptation as the Gallery Qwen MemoryOS control."""
    raw = row['raw']
    assert len(raw['image_id']) == len(raw['image_caption']) == len(raw['input_image'])
    parts = ['Date: ' + row['timestamp'], 'Round: ' + row['id'], row['user'].strip()]
    for identity, caption in zip(raw['image_id'], raw['image_caption']):
        assert caption.strip()
        parts.extend(['image:', 'image_id: ' + identity, 'image_caption: ' + caption])
    return '\n'.join(parts)


def verify_ga(source):
    """Validate the completed control, its immutable inputs and caption calls."""
    p = ga.verify(source)
    assert read(source / 'execution_status.json')['status'] == 'complete'
    audit = read(source / 'audit.json')
    assert audit['status'] == read(source / 'independent_audit.json')['status'] == 'passed'
    for group in ('artifact_sha256', 'memory_call_sha256'):
        for path, digest in audit[group].items():
            assert sha(path) == digest, path
    amendment = read(source / 'image_format_amendment.json')
    assert amendment['adapter_sha256'] == sha(Path(images.__file__))
    c = read(source / 'caption_source/complete.json')
    assert c['observations_sha256'] == sha(source / 'observations.json')
    assert c['captions_sha256'] == sha(source / 'caption_source/captions.jsonl')
    caps = rows(source / 'caption_source/captions.jsonl')
    jobs = rows(source / 'caption_jobs.jsonl')
    capmap = {c['id']: c for c in caps}
    assert len(capmap) == len(caps) == len(jobs) == 979
    for job in jobs:
        saved = read(source / 'caption_source/responses' / (job['id'] + '.json'))
        payload = dict(user_text=p['caption_prompt'], query_images=[dict(path=job['path'], sha256=job['sha256'])])
        out = saved['output']
        assert saved['payload'] == payload
        assert out['request_sha256'] == ga.request_sha256(images.wire_for(payload))
        assert out['returned_model'] == p['expected_model'] and out['seed'] == 42
        assert out['finish_reason'] == 'stop' and not out['refusal']
        assert capmap[job['id']]['caption'] == out['raw_response'].strip()
    observations = read(source / 'observations.json')
    assert len(observations) == 20 and sum(map(len, observations.values())) == 3962
    for group in observations.values():
        for row in group:
            assert row['raw']['image_caption'] == [capmap[i]['caption'] for i in row['caption_ids']]
    return p


def prepare(output, source=SOURCE):
    if (output / 'protocol.json').exists():
        return verify(output)
    parent = verify_ga(source)
    upstream = native.verify_vendor()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('observations.json', 'inputs/questions.jsonl', 'inputs/gold.jsonl'):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    qs = rows(output / 'inputs/questions.jsonl')
    assert len(qs) == len({q['id'] for q in qs}) == 1711
    allowed = {'id', 'task', 'question', 'point', 'query_image_paths', 'query_image_caption', 'retrieval_query'}
    for q in qs:
        assert set(q) == allowed
        assert q['retrieval_query'] == q['question'] + (
            "\nquestion's image:\nimage_caption: " + q['query_image_caption'] if q['query_image_caption'] else '')
    hashes = dict(parent['source_sha256'])
    paths = {Path(__file__), Path(images.__file__), Path(adapters.__file__),
             native.VENDOR / 'memoryos.yaml', source / 'protocol.json', source / 'audit.json',
             source / 'independent_audit.json', source / 'final_results.json',
             source / 'observations.json', source / 'image_format_amendment.json',
             source / 'caption_source/complete.json', source / 'caption_source/captions.jsonl',
             source / 'gen_agents/reader/responses.jsonl',
             output / 'observations.json', output / 'inputs/questions.jsonl', output / 'inputs/gold.jsonl'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).startswith(str(ROOT / 'scripts')) and str(filename).endswith('.py'):
            paths.add(Path(filename))
    hashes.update({str(path): sha(path) for path in paths})
    snapshots = {}
    for path, digest in hashes.items():
        if Path(path).suffix == '.py':
            target = output / 'source_snapshot' / (digest[:16] + '_' + Path(path).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            snapshots[path] = dict(path=str(target), sha256=digest)
    freeze(output / 'source_snapshot/manifest.json', snapshots)
    p = dict(schema='mem_gallery_gpt5_memoryos_v1', mode='memoryos',
        requested_model='gpt-5', expected_model=parent['expected_model'],
        seed=42, reasoning_effort='minimal', temperature_wire_policy='omitted',
        endpoint=parent['endpoint'], image_detail='high', max_new_tokens=2048,
        memory_max_new_tokens=2048, completion_budget_policy='max(2048, native requested cap)',
        tasks=parent['tasks'], questions=1711, memory_rounds=3962,
        task_workers=8, api_workers=8, embedding_device='cpu',
        embedding_model_path=parent['embedding_model_path'],
        method_defaults=native.effective_method_defaults('memoryos'), upstream=upstream,
        ga_control_root=str(source), ga_signature=parent['experiment_signature'],
        caption_policy='Reuse all 979 hash-verified GPT-5 history captions from the complete GA control (1003 image occurrences).',
        query_policy=parent['query_policy'], reader_policy=parent['reader_policy'],
        reader_system_prompt=parent['reader_system_prompt'], task_instructions=parent['task_instructions'],
        metadata_policy='Preserve date, round ID, image ID, original dialogue and GPT-5 caption; identical to Gallery Qwen metadata adaptation.',
        context_policy='Native MemoryOS retrieved pages and user/assistant knowledge; no added top-five truncation.',
        summary_format_policy='Reuse existing GPT-5 MemoryOS adapter: single summary object converted losslessly to list; native fallback retained and logged.',
        failure_policy='Abort on terminal API errors, refusals, empty or truncated API responses; preserve diagnostics.',
        gold_policy='Copied gold is hashed only; no label contents opened before the separate analysis stage.',
        source_sha256=hashes, questions_sha256=obj(qs), observations_sha256=sha(output / 'observations.json'))
    p['experiment_signature'] = obj(p)
    freeze(output / 'protocol.json', p)
    dump(output / 'ga_reuse_validation.json', dict(status='passed', questions=1711, memory_rounds=3962,
        captions=979, source=str(source), source_signature=parent['experiment_signature'],
        ga_result=read(source / 'final_results.json')['overall'], checked_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='prepared', questions=1711, rounds=3962, captions=979)), flush=True)
    return p


def verify(output):
    p = read(output / 'protocol.json')
    assert p['experiment_signature'] == obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    for path, digest in p['source_sha256'].items():
        assert sha(path) == digest, path
    for record in read(output / 'source_snapshot/manifest.json').values():
        assert sha(record['path']) == record['sha256']
    return p


class Model(transport.Model):
    def one(self, messages, cap, temperature):
        out = super().one(messages, cap, temperature)
        if out['finish_reason'] != 'stop' or out.get('refusal') or not out['raw_response'].strip():
            failure = dict(error_type='invalid_completion', response_id=out['response_id'])
            with self.lock:
                self.fatal_errors.append(failure)
            raise RuntimeError('Invalid memory response; archived in api_responses.jsonl')
        return out


class Memory(native.Memory):
    def __init__(self, protocol, workdir):
        super().__init__('memoryos', protocol, workdir)
        self.helper['_round_user_with_captions'] = user_text
        self.core.llm_model = self.core.updater.llm_model = 'gpt-5'
        dump(workdir / 'resolved_config.json', dict(method='memoryos', requested_model='gpt-5',
            expected_model=protocol['expected_model'], defaults=protocol['method_defaults'],
            embedding_model_path=protocol['embedding_model_path']))


def memory_stage(output, env_file, task=None):
    p = verify(output)
    observations, qs = read(output / 'observations.json'), rows(output / 'inputs/questions.jsonl')
    root = output / 'memoryos'
    root.mkdir(exist_ok=True)
    pending = [t for t in p['tasks'] if (task is None or task == t) and not (root / t / 'complete.json').exists()]
    if pending:
        model = Model(root, env_file, p['expected_model'])
        model.torch.set_num_threads(2)
        broker = adapters.make_broker('memoryos', model, root, p)
        try:
            encoder = adapters.install_runtime('memoryos', broker, p)
            dump(root / 'generation_environment.json', model.environment | dict(embedding_device=str(encoder.device)))
            def one(name):
                broker.local.owner = name
                target = root / name
                runtime = target / 'runtime'
                if runtime.exists():
                    runtime.rename(target / ('runtime_incomplete_' + str(time.time_ns())))
                runtime.mkdir(parents=True)
                memory = Memory(p, runtime)
                try:
                    for i, row in enumerate(observations[name], 1):
                        if model.fatal_errors:
                            raise RuntimeError('Fatal API error; stopping memory construction')
                        memory.store(row)
                        if model.fatal_errors:
                            raise RuntimeError('MemoryOS swallowed a terminal API error; rejecting task')
                        dump(target / 'progress.json', dict(phase='ingest', done=i, total=len(observations[name])))
                        if i % 25 == 0:
                            print(f'MemoryOS {name}: {i}/{len(observations[name])} rounds', flush=True)
                    recalled = []
                    selected = [q for q in qs if q['task'] == name]
                    for q in selected:
                        context, detail = memory.recall(q['retrieval_query'])
                        if model.fatal_errors:
                            raise RuntimeError('Fatal API error during retrieval')
                        recalled.append(dict(id=q['id'], task=name, retrieval_query=q['retrieval_query'],
                            retrieved_context=context, retrieval_detail=detail, experiment_signature=p['experiment_signature']))
                    write_rows(target / 'retrieval.jsonl', recalled)
                    memory.export(target)
                    state_hashes = {str(f): sha(f) for f in runtime.rglob('*.json')}
                    dump(target / 'complete.json', dict(experiment_signature=p['experiment_signature'],
                        questions=len(recalled), stored_rounds=memory.stats['stored_rounds'],
                        retrieval_sha256=sha(target / 'retrieval.jsonl'), state_sha256=state_hashes))
                    dump(target / 'progress.json', dict(phase='complete', done=len(selected), total=len(selected)))
                    print(f'MemoryOS {name}: complete ({len(recalled)} questions)', flush=True)
                finally:
                    memory.core.client.shutdown()
            errors = []
            with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
                futures = {pool.submit(one, name): name for name in pending}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        errors.append(dict(task=futures[future], traceback=traceback.format_exc()))
                        dump(root / 'run_errors.json', errors)
                        print(errors[-1]['traceback'], flush=True)
            if errors:
                raise RuntimeError(f'{len(errors)} memory tasks failed')
        finally:
            broker.close()
    if task:
        return
    recalled = []
    for name in p['tasks']:
        done = read(root / name / 'complete.json')
        assert done['experiment_signature'] == p['experiment_signature']
        assert done['retrieval_sha256'] == sha(root / name / 'retrieval.jsonl')
        for path, digest in done['state_sha256'].items():
            assert sha(path) == digest
        recalled.extend(rows(root / name / 'retrieval.jsonl'))
    by_id = {r['id']: r for r in recalled}
    assert len(recalled) == len(by_id) == len(qs) and set(by_id) == {q['id'] for q in qs}
    write_rows(root / 'retrieval.jsonl', [by_id[q['id']] for q in qs])


def answers(output, env_file, limit=0):
    p = verify(output)
    qs = rows(output / 'inputs/questions.jsonl')
    retrieved = rows(output / 'memoryos/retrieval.jsonl')
    rmap = {r['id']: r for r in retrieved}
    assert len(rmap) == len(retrieved) == len(qs) == 1711
    root = output / 'memoryos/reader'
    model = Model(root, env_file, p['expected_model'])
    manifest = [ga.reader_payload(q, rmap[q['id']]['retrieved_context']) for q in qs]
    if (root / 'manifest.jsonl').exists():
        assert rows(root / 'manifest.jsonl') == manifest
    else:
        write_rows(root / 'manifest.jsonl', manifest)
    def one(payload):
        path = root / 'responses' / (obj(payload['record_id']) + '.json')
        wire = images.wire_for(payload, p['max_new_tokens'])
        if path.exists():
            saved = read(path)
            assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
            out = saved['output']
            assert out['request_sha256'] == ga.request_sha256(wire)
        else:
            out = model.invoke(wire, 'answer', payload['record_id'])
            saved = dict(id=payload['record_id'], payload=payload, output=out, experiment_signature=p['experiment_signature'])
            if out['finish_reason'] != 'stop' or out.get('refusal') or not out['raw_response'].strip():
                dump(path.parent / 'invalid' / path.name, saved)
                raise RuntimeError('Invalid answer archived')
            dump(path, saved)
        assert out['returned_model'] == p['expected_model']
        assert out['finish_reason'] == 'stop' and not out.get('refusal') and out['raw_response'].strip()
        return dict(id=payload['record_id'], **out, experiment_signature=p['experiment_signature'])
    selected = manifest[:limit] if limit else manifest
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        futures = [pool.submit(one, payload) for payload in selected]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            dump(root / 'progress.json', dict(phase='answer', done=i, total=len(qs)))
            if i % 50 == 0 or i == len(selected):
                print(f'GPT-5 MemoryOS answers {i}/{len(qs)}', flush=True)
    if not limit:
        write_rows(root / 'responses.jsonl', [one(payload) for payload in manifest])
        freeze(root / 'complete.json', dict(count=len(qs), responses_sha256=sha(root / 'responses.jsonl'),
            manifest_sha256=sha(root / 'manifest.jsonl'), experiment_signature=p['experiment_signature']))


def probe(output, env_file):
    p = verify(output)
    model = Model(output / 'probe', env_file, p['expected_model'])
    request = transport.build_request([dict(role='user', content='Reply with exactly OK.')], 2048, 0)
    out = model.invoke(request, 'probe')
    assert out['raw_response'].strip() == 'OK' and out['finish_reason'] == 'stop'
    dump(output / 'api_probe.json', dict(status='passed', output=out))
    print(json.dumps(dict(status='passed', returned_model=out['returned_model'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'probe', 'memory', 'answers', 'all'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--env-file', type=Path, default=ga.DEFAULT_ENV)
    parser.add_argument('--task')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    stages = ['prepare', 'probe', 'memory', 'answers'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        dump(output / 'execution_status.json', dict(status='running', stage=stage, pid=os.getpid(), updated=time.time()))
        try:
            if stage == 'prepare': prepare(output, args.source.resolve())
            elif stage == 'probe': probe(output, args.env_file)
            elif stage == 'memory': memory_stage(output, args.env_file, args.task)
            else: answers(output, args.env_file, args.limit)
        except BaseException as exc:
            dump(output / 'execution_status.json', dict(status='stopped_before_completion', stage=stage,
                error=str(exc), updated=time.time(), results_available=False))
            raise
        dump(output / 'execution_status.json', dict(status='stage_complete', stage=stage, updated=time.time()))


if __name__ == '__main__':
    main()
