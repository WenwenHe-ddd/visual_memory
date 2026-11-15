"""Natural Top-5 Mem-Gallery baseline, using the completed GPT-5 caption control.

Keep the existing BGE full-scan recipe and official reader prompt. Only the
history captions and answer backbone differ from the original Qwen baseline.
Gold labels are copied/hashed, and opened only by the separate audit module.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import sys
import time

from scripts.evaluation import mem_gallery_gpt5_memoryos as shared
from scripts.memory import caption_visual_master as bge

ga, images, transport = shared.ga, shared.images, shared.transport
read, rows, sha, obj = shared.read, shared.rows, shared.sha, shared.obj
dump, write_rows, freeze = shared.dump, shared.write_rows, shared.freeze
ROOT, SOURCE = shared.ROOT, shared.SOURCE
REFERENCE = ROOT / 'results/mem_gallery_master_answer_20261021'
OUTPUT = ROOT / 'results/mem_gallery_gpt5_baseline_20261030'


def memory_records(observations):
    """Match the existing Natural text formatting, substituting GPT-5 captions."""
    records = []
    for task, group in observations.items():
        for index, row in enumerate(group):
            captions, ids = row['raw']['image_caption'], row['raw']['image_id']
            assert len(captions) == len(ids) == len(row['raw']['input_image']) <= 1
            caption = captions[0] if captions else ''
            dialogue = 'User: ' + row['user'] + '\nAssistant: ' + row['assistant']
            text = dialogue + ('\nMemory description: ' + caption if captions else '')
            metadata = ['Date: ' + row['timestamp'], 'Round: ' + row['id']]
            if ids:
                metadata.append('Image ID: ' + ids[0])
            records.append(dict(id=task + '/' + row['id'], task=task,
                round=row['id'], session_id=row['session_id'], timestamp=row['timestamp'],
                dialogue=dialogue, caption=caption, retrieval_text=text,
                reader_text='\n'.join([*metadata, text]), is_visual=bool(captions),
                insertion_index=index))
    assert len(records) == len({r['id'] for r in records})
    return records


def verify(output):
    p = read(output / 'protocol.json')
    assert obj({k: v for k, v in p.items() if k != 'experiment_signature'}) == p['experiment_signature']
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, path
    for item in read(output / 'source_snapshot/manifest.json').values():
        assert sha(item['path']) == item['sha256']
    return p


def prepare(output):
    if (output / 'protocol.json').exists():
        return verify(output)
    parent = shared.verify_ga(SOURCE)
    reference = read(REFERENCE / 'offline/protocol.json')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('observations.json', 'inputs/questions.jsonl', 'inputs/gold.jsonl'):
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, dest)
    qs = rows(output / 'inputs/questions.jsonl')
    memories = memory_records(read(output / 'observations.json'))
    assert len(qs) == len({q['id'] for q in qs}) == 1711 and len(memories) == 3962
    allowed = {'id', 'task', 'point', 'question', 'retrieval_query', 'query_image_caption', 'query_image_paths'}
    for q in qs:
        assert set(q) == allowed
        assert q['retrieval_query'] == q['question'] + (
            "\nquestion's image:\nimage_caption: " + q['query_image_caption'] if q['query_image_caption'] else '')
    write_rows(output / 'memories.jsonl', memories)
    hashes = dict(parent['source_sha256'])
    paths = {Path(__file__), Path(shared.__file__), Path(images.__file__), Path(bge.__file__),
             ROOT / 'scripts/analysis/audit_mem_gallery_gpt5_baseline.py',
             ROOT / 'scripts/tests/test_mem_gallery_gpt5_baseline.py',
             REFERENCE / 'offline/protocol.json', REFERENCE / 'retrieval_protocol.json',
             SOURCE / 'protocol.json', SOURCE / 'audit.json', SOURCE / 'observations.json',
             SOURCE / 'caption_source/complete.json', SOURCE / 'caption_source/captions.jsonl',
             output / 'observations.json', output / 'memories.jsonl',
             output / 'inputs/questions.jsonl', output / 'inputs/gold.jsonl'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).startswith(str(ROOT / 'scripts')) and str(filename).endswith('.py'):
            paths.add(Path(filename))
    meta = reference['text_encoder']
    for filename, digest in meta['files_sha256'].items():
        path = Path(meta['model_path']) / filename
        assert sha(path) == digest
        paths.add(path)
    hashes.update({str(path): sha(path) for path in paths})
    snapshots = {}
    for path, digest in hashes.items():
        if Path(path).suffix == '.py':
            dest = output / 'source_snapshot' / (digest[:16] + '_' + Path(path).name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dest)
            snapshots[path] = dict(path=str(dest), sha256=digest)
    freeze(output / 'source_snapshot/manifest.json', snapshots)
    p = dict(schema='mem_gallery_gpt5_natural_v1', method='Natural baseline',
        requested_model='gpt-5', expected_model=parent['expected_model'],
        endpoint=parent['endpoint'], seed=42, reasoning_effort='minimal',
        temperature_wire_policy='omitted', max_new_tokens=2048, image_detail='high', api_workers=8,
        questions=1711, memory_rounds=3962, tasks=parent['tasks'], top_k=5,
        text_encoder=meta, embedding_device='cpu', query_prefix=reference['query_prefix'],
        retrieval_policy='Original Natural BGE normalized cosine full scan over every same-scenario round; stable source order for ties; Top5.',
        retrieval_text_policy='User and assistant text plus full GPT-5 history caption; metadata added only to reader context.',
        caption_policy='Reuse 979 verified GPT-5 historical captions from GA, covering 1003 image occurrences; identical to MemoryOS.',
        query_policy=parent['query_policy'], reader_policy=parent['reader_policy'],
        reader_system_prompt=parent['reader_system_prompt'], task_instructions=parent['task_instructions'],
        context_policy='Exactly five original dialogue rounds in retrieval rank order, with date/round/image ID and full captions.',
        no_hypotheses=True, no_reflection=True, no_zscore=True, truncation=False,
        gold_policy='Only the separate audit opens gold label contents, after all answers are frozen.',
        caption_control_root=str(SOURCE), caption_control_signature=parent['experiment_signature'],
        source_sha256=hashes)
    p['experiment_signature'] = obj(p)
    freeze(output / 'protocol.json', p)
    print('Prepared Natural GPT-5: 3962 memories, 1711 questions.', flush=True)
    return p


def context_for(ids, memory_map):
    return '\n\n'.join(f'Memory {i}: {memory_map[mid]["reader_text"]}' for i, mid in enumerate(ids, 1))


def retrieve(output):
    import numpy as np
    import torch
    p = verify(output)
    done_path = output / 'retrieval_complete.json'
    if done_path.exists():
        done = read(done_path)
        assert done['experiment_signature'] == p['experiment_signature']
        for path, digest in done['artifact_sha256'].items():
            assert sha(path) == digest
        return
    torch.set_num_threads(4)
    encoder = bge.load_text_encoder(p['text_encoder'], 'cpu')
    qs, memories = rows(output / 'inputs/questions.jsonl'), rows(output / 'memories.jsonl')
    mmap = {m['id']: m for m in memories}
    results, artifacts, lengths = {}, {}, []
    for task in p['tasks']:
        ms, selected = [m for m in memories if m['task'] == task], [q for q in qs if q['task'] == task]
        texts = [m['retrieval_text'] for m in ms]
        queries = [q['retrieval_query'] for q in selected]
        vectors = bge.encode_text(encoder, texts)
        qvectors = bge.encode_text(encoder, queries, p['query_prefix'])
        all_texts = texts + [p['query_prefix'] + q for q in queries]
        lengths.extend(len(x) for x in encoder.tokenizer(all_texts, truncation=False)['input_ids'])
        dest = output / 'embeddings' / (task + '.npz')
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, memory_ids=np.array([m['id'] for m in ms]),
            question_ids=np.array([q['id'] for q in selected]), memory_vectors=vectors, query_vectors=qvectors)
        artifacts[str(dest)] = sha(dest)
        for q, vector in zip(selected, qvectors):
            scores = vectors @ vector
            order = np.argsort(-scores, kind='stable')
            ranked = [ms[int(i)]['id'] for i in order]
            ids = ranked[:p['top_k']]
            results[q['id']] = dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                candidate_count=len(ms), source_ids=ids, ranking_memory_ids=ranked,
                ranking_scores=scores[order].tolist(), retrieved_context=context_for(ids, mmap),
                experiment_signature=p['experiment_signature'])
        print(f'Natural retrieval: {len(results)}/1711', flush=True)
    assert set(results) == {q['id'] for q in qs}
    write_rows(output / 'retrieval.jsonl', [results[q['id']] for q in qs])
    artifacts[str(output / 'retrieval.jsonl')] = sha(output / 'retrieval.jsonl')
    freeze(done_path, dict(count=len(results), max_embedding_tokens=max(lengths), truncated_inputs=0,
        artifact_sha256=artifacts, experiment_signature=p['experiment_signature']))


def answers(output, env_file, limit=0):
    p = verify(output)
    complete = read(output / 'retrieval_complete.json')
    assert complete['artifact_sha256'][str(output / 'retrieval.jsonl')] == sha(output / 'retrieval.jsonl')
    qs = rows(output / 'inputs/questions.jsonl')
    retrieval = {r['id']: r for r in rows(output / 'retrieval.jsonl')}
    assert set(retrieval) == {q['id'] for q in qs}
    root = output / 'reader'
    model = transport.Model(root, env_file, p['expected_model'])
    manifest = [ga.reader_payload(q, retrieval[q['id']]['retrieved_context']) for q in qs]
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
        assert out['returned_model'] == p['expected_model'] and out['seed'] == 42
        assert out['finish_reason'] == 'stop' and not out.get('refusal') and out['raw_response'].strip()
        return dict(id=payload['record_id'], **out, experiment_signature=p['experiment_signature'])
    selected = manifest[:limit] if limit else manifest
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        futures = [pool.submit(one, payload) for payload in selected]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            dump(root / 'progress.json', dict(stage='answer', done=i, total=len(qs)))
            if i % 50 == 0 or i == len(selected):
                print(f'Natural GPT-5 answers: {i}/1711', flush=True)
    if not limit:
        write_rows(root / 'responses.jsonl', [one(payload) for payload in manifest])
        freeze(root / 'complete.json', dict(count=len(qs), responses_sha256=sha(root / 'responses.jsonl'),
            manifest_sha256=sha(root / 'manifest.jsonl'), experiment_signature=p['experiment_signature']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'retrieve', 'answers', 'all'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--env-file', type=Path, default=ga.DEFAULT_ENV)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    for stage in (['prepare', 'retrieve', 'answers'] if args.stage == 'all' else [args.stage]):
        dump(output / 'execution_status.json', dict(status='running', stage=stage, pid=os.getpid(), updated=time.time()))
        try:
            if stage == 'prepare': prepare(output)
            elif stage == 'retrieve': retrieve(output)
            else: answers(output, args.env_file, args.limit)
        except BaseException as exc:
            dump(output / 'execution_status.json', dict(status='stopped_before_completion', stage=stage,
                error=str(exc), updated=time.time(), results_available=False))
            raise
        dump(output / 'execution_status.json', dict(status='stage_complete', stage=stage, updated=time.time()))


if __name__ == '__main__':
    main()
