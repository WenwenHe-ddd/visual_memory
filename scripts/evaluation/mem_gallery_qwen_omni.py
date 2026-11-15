"""Separate Omni-SimpleMem 0.3.0 visual experiment; no text-mode mutations.

Fresh Qwen captions through native add_image, shared CLIP text/image vectors,
native selective ingestion, semantic10 + BM25 5, RRF60 context5, raw evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  LITELLM_LOCAL_MODEL_COST_MAP='True', ANONYMIZED_TELEMETRY='False')
from scripts.evaluation import mem_gallery_qwen_text_methods as base
from scripts.memory import omni_simplemem as omni
from scripts.evaluation.omni_qwen_transport import LocalMultimodalQwen, OmniBroker

SOURCE = ROOT / 'results/simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029'
OUTPUT = ROOT / 'results/mem_gallery_qwen_omni_simplemem030_v2_20261104'
read, rows, sha, obj = base.read, base.rows, base.sha, base.obj
dump, write_rows, freeze = base.dump, base.write_rows, base.freeze


def guard_paths():
    paths = set()
    for name in ('simplemem_official.py', 'simplemem_official_sources.json', 'simplemem_fusion.py'):
        paths.add(ROOT / 'scripts/memory' / name)
    for name in ('mem_gallery_simplemem_official.py', 'mem_gallery_gpt5_simplemem030.py',
                 'mem_gallery_deepseek_text_methods.py'):
        paths.add(ROOT / 'scripts/evaluation' / name)
    text_lock = read(ROOT / 'scripts/memory/simplemem_official_sources.json')
    # Lock format is shared with the prior official text installation.
    for name in text_lock['git_blob_hashes']:
        paths.add(ROOT / '.vendor/simplemem_official_runtime' / name)
    for root in (SOURCE, ROOT / 'results/mem_gallery_gpt5_simplemem030_20261031',
                 ROOT / 'results/mem_gallery_simplemem030_backbones_20261031'):
        for name in ('protocol.json', 'simplemem/protocol.json', 'final_results.json',
                     'audit.json', 'completion_manifest.json', 'comparison.json'):
            if (root / name).exists():
                paths.add(root / name)
    return sorted(paths)


def prepare(output, pilot=False):
    if (output / 'protocol.json').exists():
        return verify(output)
    upstream = omni.verify_installation()
    parent = read(SOURCE / 'simplemem/protocol.json')
    for path, digest in parent['input_sha256'].items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=True)
    guard = {str(p): sha(p) for p in guard_paths()}
    freeze(output / 'text_control_guard.json', guard)
    for name in ('observations.json', 'inputs/questions.jsonl', 'inputs/memories.jsonl', 'inputs/gold.jsonl'):
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, dest)
    observations, questions = read(output / 'observations.json'), rows(output / 'inputs/questions.jsonl')
    assert len(questions) == 1711 and sum(map(len, observations.values())) == 3962
    for q in questions:
        assert set(q) == {'id', 'task', 'point', 'question', 'query_image_paths', 'query_image_caption', 'retrieval_query'}
    tasks = sorted(observations)
    if pilot:
        tasks = tasks[:1]
        # Exercise real images and at least two original query records.
        observations = {tasks[0]: observations[tasks[0]][:8]}
        questions = [q for q in questions if q['task'] == tasks[0]][:3]
        dump(output / 'observations.json', observations)
        write_rows(output / 'inputs/questions.jsonl', questions)
    memories = rows(output / 'inputs/memories.jsonl')
    mapping = {m['id']: m for m in memories}
    images = {}
    for task, records in observations.items():
        for row in records:
            original = mapping[task + '/' + row['id']]
            assert len(row['raw']['input_image']) == len(row['raw']['image_id']) <= 1
            if original['is_visual']:
                assert sha(original['image_path']) == original['image_sha256']
                images[original['id']] = [dict(image_id=original['image_id'], image_path=original['image_path'],
                                               sha256=original['image_sha256'])]
            else:
                images[original['id']] = []
    dump(output / 'inputs/history_images.json', images)
    p = dict(schema='omni_simplemem030_qwen_visual_v1', mode='omni', method='Omni-SimpleMem 0.3.0 (visual, RRF5)',
        upstream=upstream, seed=42, model=parent['model'], reader_model='Qwen3-VL-8B-Instruct',
        query_image_max_pixels=base.gallery.QUERY_IMAGE_MAX_PIXELS, max_new_tokens=256,
        memory_max_new_tokens=4096, generation_batch_size=8, max_padded_batch_tokens=48000,
        task_workers=8, api_calls_permitted=False, tasks=tasks, questions=len(questions),
        memory_rounds=sum(map(len, observations.values())), pilot=pilot,
        reader_system_prompt=base.gallery.READER_SYSTEM, task_instructions=base.gallery.TASK_INSTRUCTIONS,
        embedding_model='openai/clip-vit-base-patch32', embedding_dim=512,
        embedding_policy='Official unified CLIP backend. Native 77-token CLIP text truncation. Local pinned weights; no embedding API.',
        compatibility_policy='Instance-local Transformers5 projected pooler_output unwrapping; shared encoder loading; isolated graph constructor; null related-list restored to empty list. Upstream files unmodified.',
        caption_policy='Fresh native image captions from local Qwen. Prior text-mode captions are excluded from ingestion.',
        trigger_policy='Native force=False. Text redundancy and visual 0.9/0.7 thresholds. Reset at original session boundaries.',
        retrieval_policy='Native query parser/filters, semantic candidate K10, native BM25 K5. Equal-channel RRF60, final <=5 whole MAUs.',
        graph_policy='Native entity/relationship construction and query executed; graph diagnostics retained. No extra graph facts appended outside final five.',
        expansion_policy='Native expand(EVIDENCE) for selected MAUs; stored JPEG image bytes are passed to reader alongside original query images.',
        reader_policy='Same official Gallery system/task instructions and Qwen greedy256. Labels identify history images and query images.',
        consolidation_policy='Native event summaries at session close; no self-evolution, gold optimization, or optional parametric distillation.',
        source_policy='Dates/IDs copied from original source. Independent text/image MAUs map exactly to one source round. Related MAUs do not consume extra context.',
        failure_policy='Memory: at most three audited attempts, schema reinforcement for format errors, identical-message token-cap extension to <=16384 for truncation. Reader stays greedy256. Stop on exhausted retries, missing visual embeddings/evidence, or source mismatch.',
        json_policy='Entity JSON schema matches native parser fields. Invalid generations retained verbatim; retry prompts contain original source only, never repaired entities or gold.',
        control_root=str(SOURCE), differences_from_text=['CLIP512 instead of MiniLM384',
            'Native selective per-modality MAUs instead of text-core window compression',
            'Actual history pixels in selected visual evidence', 'Native Omni query plus two-channel RRF; no text-core reflection loop'],
        prior_answers_reused=False, prior_memory_reused=False)
    paths = {Path(__file__), ROOT / 'scripts/evaluation/omni_qwen_transport.py',
        ROOT / 'scripts/memory/omni_simplemem.py', ROOT / 'scripts/analysis/audit_mem_gallery_qwen_omni.py',
        omni.LOCK, *[omni.RUNTIME / f for f in upstream['files']],
        output / 'observations.json', *list((output / 'inputs').glob('*')), output / 'text_control_guard.json'}
    paths.update(base.gallery.CLIP.glob('*'))
    paths.update(base.gallery.MODEL / name for name in parent['model']['files'])
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and str(path).startswith(str(ROOT / 'scripts')) and str(path).endswith('.py'):
            paths.add(Path(path))
    paths.update(Path(i['image_path']) for group in images.values() for i in group)
    paths.update(Path(path) for q in questions for path in q['query_image_paths'])
    p['source_sha256'] = {str(path): sha(path) for path in sorted(paths) if path.is_file()}
    p['experiment_signature'] = obj(p)
    snapshots = {}
    for path, digest in p['source_sha256'].items():
        if Path(path).suffix == '.py':
            target = output / 'source_snapshot' / (digest[:16] + '_' + Path(path).name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            snapshots[path] = dict(path=str(target), sha256=digest)
    freeze(output / 'source_snapshot/manifest.json', snapshots)
    freeze(output / 'protocol.json', p)
    print('Prepared', p['questions'], 'questions,', p['memory_rounds'], 'rounds:', output, flush=True)
    return p


def verify(output):
    p = read(output / 'protocol.json')
    assert p['experiment_signature'] == obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    assert omni.verify_installation() == p['upstream']
    for path, digest in p['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in read(output / 'text_control_guard.json').items():
        assert sha(path) == digest, 'text control changed: ' + path
    return p


def reader_messages(q, context, detail):
    content = []
    for image in detail['history_images']:
        content.append(dict(type='text', text='Historical image from memory ' + image['mau_id'] +
                            ', image_id: ' + image['image_id']))
        content.append(dict(type='image', path=image['path'], sha256=image['sha256']))
    if q['query_image_paths']:
        content.append(dict(type='text', text='Question image(s):'))
    for path in q['query_image_paths']:
        content.append(dict(type='image', path=path, sha256=sha(path)))
    content.append(dict(type='text', text=base.reader_text(q, context)))
    return [dict(role='system', content=base.gallery.READER_SYSTEM), dict(role='user', content=content)]


def run(output, device='cuda:0'):
    p = verify(output)
    if (output / 'completion_manifest.json').exists():
        print('Already complete:', output, flush=True)
        return
    root = output / 'memory'
    root.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING, handlers=[logging.FileHandler(output / 'native.log'), logging.StreamHandler()])
    omni.activate()
    omni.install_isolation()
    model = LocalMultimodalQwen(device)
    dump(output / 'generation_environment.json', model.environment)
    clip = omni.SharedCLIP(base.gallery.CLIP, 'cpu')
    broker = OmniBroker(model, root, p)
    class NativeErrorGuard(logging.Handler):
        def emit(self, record):
            failure = dict(component=record.name, error=record.getMessage())
            with broker.log_lock:
                broker.errors.append(failure)
                dump(output / 'native_errors.json', broker.errors)
    native_guard = NativeErrorGuard(level=logging.ERROR)
    logging.getLogger('simplemem').addHandler(native_guard)
    observations, questions = read(output / 'observations.json'), rows(output / 'inputs/questions.jsonl')
    images = read(output / 'inputs/history_images.json')
    dump(output / 'execution.json', dict(status='running', pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
    try:
        def one(task):
            broker.local.owner = task
            dest = root / task
            if (dest / 'complete.json').exists():
                for path, digest in read(dest / 'complete.json')['artifact_sha256'].items():
                    assert sha(path) == digest, path
                return
            dest.mkdir(exist_ok=True)
            workdir = dest / 'runtime'
            if workdir.exists():
                workdir.rename(dest / ('runtime_incomplete_' + str(time.time_ns())))
            workdir.mkdir()
            memory = omni.OmniMemory(workdir, broker.client(), clip, dump, task)
            for i, row in enumerate(observations[task], 1):
                broker.check()
                memory.store(row, images[task + '/' + row['id']])
                broker.check()
                dump(dest / 'progress.json', dict(phase='ingest', done=i, total=len(observations[task])))
                if i % 20 == 0:
                    print(task, 'ingest', i, '/', len(observations[task]), flush=True)
            memory.finalize()
            broker.check()
            retrieval, answers = [], []
            task_questions = [q for q in questions if q['task'] == task]
            for i, q in enumerate(task_questions, 1):
                context, detail = memory.recall(q['retrieval_query'])
                broker.check()
                record = dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                    retrieved_context=context, retrieval_detail=detail, experiment_signature=p['experiment_signature'])
                retrieval.append(record)
                messages = reader_messages(q, context, detail)
                answer = broker.request(messages, p['max_new_tokens'], kind='answer', owner=q['id'])
                answers.append(dict(id=q['id'], task=task, output=answer, messages=messages,
                                    experiment_signature=p['experiment_signature']))
                dump(dest / 'checkpoints' / (obj(q['id']) + '.json'), dict(retrieval=record, answer=answers[-1]))
                dump(dest / 'progress.json', dict(phase='answer', done=i, total=len(task_questions)))
                if i % 10 == 0:
                    print(task, 'answers', i, '/', len(task_questions), flush=True)
            memory.core.close()
            broker.check()
            write_rows(dest / 'retrieval.jsonl', retrieval)
            write_rows(dest / 'answers.jsonl', answers)
            files = [f for f in dest.rglob('*') if f.is_file() and not any('runtime_incomplete_' in a for a in f.parts)]
            # Progress is sealed as complete before computing artifact hashes.
            dump(dest / 'progress.json', dict(phase='complete', done=len(answers), total=len(answers)))
            freeze(dest / 'complete.json', dict(experiment_signature=p['experiment_signature'],
                questions=len(answers), rounds=len(observations[task]), artifact_sha256={str(f): sha(f) for f in files}))
            print(task, 'COMPLETE', len(answers), flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            futures = {pool.submit(one, task): task for task in p['tasks']}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    failure = dict(task=futures[future], traceback=traceback.format_exc())
                    failures.append(failure)
                    dump(output / 'errors.json', failures)
                    print(failure['traceback'], flush=True)
        if failures:
            raise RuntimeError('Omni tasks failed; diagnostics preserved')
        retrieval, answers = [], []
        for task in p['tasks']:
            retrieval.extend(rows(root / task / 'retrieval.jsonl'))
            answers.extend(rows(root / task / 'answers.jsonl'))
        rm, am = {r['id']: r for r in retrieval}, {r['id']: r for r in answers}
        assert len(rm) == len(am) == len(questions)
        write_rows(output / 'retrieval.jsonl', [rm[q['id']] for q in questions])
        write_rows(output / 'answers.jsonl', [am[q['id']] for q in questions])
        dump(output / 'execution.json', dict(status='inference_complete', questions=len(answers)))
    except BaseException:
        dump(output / 'execution.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
    finally:
        broker.close()
        logging.getLogger('simplemem').removeHandler(native_guard)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'analyze', 'all'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        prepare(args.output, args.pilot)
    if args.stage in ('run', 'all'):
        run(args.output, args.device)
    if args.stage in ('analyze', 'all'):
        from scripts.analysis.audit_mem_gallery_qwen_omni import analyze
        analyze(args.output)


if __name__ == '__main__':
    main()
