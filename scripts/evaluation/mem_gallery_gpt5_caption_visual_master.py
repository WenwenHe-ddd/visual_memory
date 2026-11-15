"""Paired GPT-5 caption_visual_master and within-master z-score on Mem-Gallery.

The signed Natural run supplies identical captions, queries, evidence formatting
and BGE vectors. Rebuild the visual masters and generate four fresh, gold-free
GPT-5 hypotheses per question. Both conditions share masters and hypotheses.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scripts.evaluation import mem_gallery_gpt5_baseline as base
from scripts.evaluation.mem_gallery_zscore import group_scores
from scripts.memory import mem_gallery_master_answer as original
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory

ROOT, SOURCE = base.ROOT, base.OUTPUT
OUTPUT = ROOT / 'results/mem_gallery_gpt5_caption_visual_master_20261030'
CONDITIONS = ('caption_visual_master', 'caption_visual_master+zscore')
read, rows, sha, obj = base.read, base.rows, base.sha, base.obj
dump, write_rows, freeze = base.dump, base.write_rows, base.freeze
ga, images, transport = base.ga, base.images, base.transport
verify = base.verify


def check_marker(output, name, p):
    path = output / name
    if not path.exists():
        return False
    marker = read(path)
    assert marker['experiment_signature'] == p['experiment_signature']
    for filename, digest in marker['artifact_sha256'].items():
        assert sha(filename) == digest, filename
    return marker


def seal(output, name, files, p, **fields):
    marker = dict(**fields, artifact_sha256={str(path): sha(path) for path in files},
                  experiment_signature=p['experiment_signature'])
    freeze(output / name, marker)
    return marker


def prepare(output, ridge=.03, alpha=1.):
    if (output / 'protocol.json').exists():
        p = verify(output)
        assert p['lambda'] == ridge and p['alpha'] == alpha
        return p
    parent = base.verify(SOURCE)
    completed = read(SOURCE / 'completion_manifest.json')
    assert completed['status'] == 'complete'
    for path, digest in completed['artifacts_sha256'].items():
        assert sha(path) == digest, path
    output.mkdir(parents=True, exist_ok=True)
    for name in ('observations.json', 'memories.jsonl', 'inputs/questions.jsonl', 'inputs/gold.jsonl'):
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, dest)
    obs = read(output / 'observations.json')
    jobs = {r['id']: r for r in rows(base.SOURCE / 'caption_jobs.jsonl')}
    historical = []
    for task, group in obs.items():
        for r in group:
            for digest in r['caption_ids']:
                job = jobs[digest]
                assert sha(job['path']) == digest
                historical.append(dict(id=task + '/' + r['id'], node_id=task + '/' + r['id'] + '#image1',
                    task=task, path=job['path'], sha256=digest))
    assert len(historical) == 1003 and len({r['id'] for r in historical}) == 1003
    calibration = {r['image_sha256'] for r in read(original.CALIBRATION)['calibration']}
    assert not calibration & {r['sha256'] for r in historical}
    write_rows(output / 'history_image_inputs.jsonl', historical)
    reference = read(base.REFERENCE / 'offline/protocol.json')
    paths = {Path(__file__), ROOT / 'scripts/analysis/audit_mem_gallery_gpt5_caption_visual_master.py',
        ROOT / 'scripts/tests/test_mem_gallery_gpt5_caption_visual_master.py',
        original.BASIS, original.CALIBRATION, base.SOURCE / 'caption_jobs.jsonl',
        SOURCE / 'protocol.json', SOURCE / 'completion_manifest.json',
        output / 'observations.json', output / 'memories.jsonl', output / 'history_image_inputs.jsonl',
        output / 'inputs/questions.jsonl', output / 'inputs/gold.jsonl'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).startswith(str(ROOT / 'scripts')) and str(filename).endswith('.py'):
            paths.add(Path(filename))
    clip_path = Path(reference['image_encoder']['model_name'])
    paths.update(path for path in clip_path.iterdir() if path.is_file())
    hashes = dict(parent['source_sha256']) | completed['artifacts_sha256']
    hashes.update({str(path): sha(path) for path in paths})
    snapshots = {}
    for path, digest in hashes.items():
        if Path(path).suffix == '.py':
            dest = output / 'source_snapshot' / (digest[:16] + '_' + Path(path).name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, dest)
            snapshots[path] = dict(path=str(dest), sha256=digest)
    freeze(output / 'source_snapshot/manifest.json', snapshots)
    p = {k: parent[k] for k in ('requested_model', 'expected_model', 'endpoint', 'seed',
        'reasoning_effort', 'temperature_wire_policy', 'max_new_tokens', 'image_detail', 'api_workers',
        'questions', 'memory_rounds', 'tasks', 'top_k', 'text_encoder', 'query_prefix',
        'caption_policy', 'query_policy', 'reader_policy', 'reader_system_prompt', 'task_instructions', 'context_policy')}
    p.update(schema='mem_gallery_gpt5_caption_visual_master_v1', conditions=list(CONDITIONS),
        baseline_root=str(SOURCE), baseline_signature=parent['experiment_signature'],
        image_encoder=reference['image_encoder'], embedding_device='cpu',
        **{'lambda': ridge, 'alpha': alpha}, parameter_policy='User selected lambda=.03, alpha=1 before generation/scoring.',
        memory_policy='Fresh chronological caption-only Ward/silhouette partitions; original two-coordinate leaf/aggregate-master visual update; fixed disjoint MemEye basis.',
        retrieval_policy='Each visual master selects one GPT-5 hypothesis by CLIP; BGE scores original query plus that hypothesis; text-only rounds keep original baseline scores; global Top5.',
        zscore_policy='Only visual master leaf scores receive the existing ZScoreWeighting correction. Same masters and selected hypotheses in both conditions.',
        candidate_system=ga.gallery.CANDIDATE_SYSTEM, candidate_max_attempts=3, candidate_max_tokens=2048,
        candidate_policy='Four fresh GPT-5 hypotheses from only question and official question-image caption; existing Gallery parser and retry prompts, CLIP limit77, no fabricated image IDs.',
        answer_cache_policy='Fresh API answers, shared only for identical wire requests within these two conditions; no historical answers reused.',
        gold_policy='Gold copied/hashed only; first read by separate audit after answers are sealed.',
        truncation=False, source_sha256=hashes)
    p['experiment_signature'] = obj(p)
    freeze(output / 'protocol.json', p)
    print('Prepared GPT-5 master pair: 1711 questions, lambda=' + str(ridge) + ', alpha=' + str(alpha), flush=True)
    return p


def candidate_payload(q, attempt):
    clean = {k: q[k] for k in ('id', 'question', 'query_image_caption')}
    return dict(record_id=q['id'], system=ga.gallery.CANDIDATE_SYSTEM,
                user_text=ga.gallery.candidate_user_text(clean, attempt), query_images=[])


def candidates(output, env_file):
    from transformers import AutoTokenizer
    p = verify(output)
    if check_marker(output, 'candidates_complete.json', p):
        return
    qs = rows(output / 'inputs/questions.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(p['image_encoder']['model_name'], local_files_only=True)
    root = output / 'candidate_generation'
    model = transport.Model(root, env_file, p['expected_model'])
    def one(q):
        failures = []
        for attempt in range(1, p['candidate_max_attempts'] + 1):
            payload = candidate_payload(q, attempt)
            wire = images.wire_for(payload, p['candidate_max_tokens'])
            path = root / 'responses' / (obj([q['id'], attempt]) + '.json')
            if path.exists():
                saved = read(path)
                assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
                out = saved['output']
                assert out['request_sha256'] == ga.request_sha256(wire)
            else:
                out = model.invoke(wire, 'candidate', q['id'])
                dump(path, dict(id=q['id'], attempt=attempt, payload=payload, output=out,
                    experiment_signature=p['experiment_signature']))
            assert out['returned_model'] == p['expected_model'] and not out['refusal']
            assert out['finish_reason'] == 'stop' and out['raw_response'].strip()
            try:
                statements, lengths, fmt = ga.gallery.parse_candidates(out['raw_response'], tokenizer, return_format=True)
            except ValueError as exc:
                failures.append(dict(attempt=attempt, error=str(exc), response_id=out['response_id']))
                continue
            return dict(id=q['id'], candidate_statements=statements, clip_token_lengths=lengths,
                response_format=fmt, accepted_attempt=attempt, failed_attempts=failures,
                response_file=str(path), experiment_signature=p['experiment_signature'])
        raise RuntimeError('Candidate format attempts exhausted: ' + q['id'] + ': ' + str(failures))
    result = {}
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        fs = [pool.submit(one, q) for q in qs]
        for i, f in enumerate(as_completed(fs), 1):
            r = f.result()
            result[r['id']] = r
            dump(root / 'progress.json', dict(done=i, total=len(qs)))
            if i % 50 == 0 or i == len(qs): print(f'GPT-5 hypotheses {i}/1711', flush=True)
    write_rows(output / 'candidates.jsonl', [result[q['id']] for q in qs])
    seal(output, 'candidates_complete.json', [output / 'candidates.jsonl', *root.glob('responses/*.json'),
        root / 'api_attempts.jsonl', root / 'api_responses.jsonl'], p, questions=1711,
        format_retries=sum(len(r['failed_attempts']) for r in result.values()))


def offline(output):
    import torch
    torch.set_num_threads(4)
    p = verify(output)
    if check_marker(output, 'offline_complete.json', p): return
    memories = rows(output / 'memories.jsonl')
    imap = {r['id']: r for r in rows(output / 'history_image_inputs.jsonl')}
    basis = np.load(original.BASIS, allow_pickle=False)
    image_encoder = original.ImageEncoder(p['image_encoder']['model_name'],
        revision=p['image_encoder']['revision'], device='cpu', local_files_only=True)
    for key in ('files_sha256', 'processor_config_sha256', 'embedding_dim', 'dtype'):
        assert image_encoder.metadata[key] == p['image_encoder'][key], key
    encoder = base.bge.load_text_encoder(p['text_encoder'], 'cpu')
    files, counts = [], {}
    for task in p['tasks']:
        dest = output / 'offline' / task
        if (dest / 'complete.json').exists():
            done = check_marker(dest, 'complete.json', p)
            files.extend(Path(s) for s in done['artifact_sha256'])
            files.append(dest / 'complete.json')
            counts[task] = done['masters']
            continue
        visual = [m for m in memories if m['task'] == task and m['is_visual']]
        memory = CaptionWardVisualMemory(basis)
        audits = []
        total_sum = np.zeros(512)
        for m in visual:
            source = imap[m['id']]
            assert sha(source['path']) == source['sha256']
            vector = original.normalize(image_encoder.encode(source['path']))
            # Match the production stable normalization for aggregate checks.
            unit = vector / np.max(np.abs(vector)); unit /= np.linalg.norm(unit)
            total_sum += unit
            caption_vector = base.bge.encode_text(encoder, [m['caption']])[0]
            a = memory.add(source['node_id'], m['caption'], caption_vector, vector,
                dialogue=m['dialogue'], timestamp=m['timestamp'])
            assert a['global_conservation_max_abs_error'] < 1e-10
            assert a['max_projection_consistency_l2_error'] < 1e-10
            audits.append(a)
            del vector, unit, caption_vector
        graph, sums = original.save_memory(memory, dest)
        np.testing.assert_allclose(sums.sum(axis=0), total_sum, atol=1e-10, rtol=0)
        write_rows(dest / 'insertion_audit.jsonl', audits)
        task_files = [dest / n for n in ('graph.json', 'basis.npy', 'master_sums.npy', 'insertion_audit.jsonl')]
        seal(dest, 'complete.json', task_files, p, images=len(visual), masters=len(graph['masters']),
            aggregate_conservation_error=float(np.max(np.abs(sums.sum(axis=0)-total_sum))))
        files.extend([*task_files, dest / 'complete.json'])
        counts[task] = len(graph['masters'])
        print(f'Masters {task}: {len(visual)} images, {counts[task]} masters', flush=True)
    dump(output / 'offline/environment.json', image_encoder.metadata)
    seal(output, 'offline_complete.json', [*files, output / 'offline/environment.json'], p,
        image_occurrences=1003, master_counts=counts, individual_image_vectors_saved=False)


def clip_vectors(model, tokenizer, limit, statements):
    import torch
    vectors = []
    for start in range(0, len(statements), 64):
        tokens = tokenizer(statements[start:start+64], padding=True, truncation=False, return_tensors='pt')
        assert tokens['input_ids'].shape[1] <= limit
        with torch.inference_mode():
            vectors.extend(model(**tokens, return_dict=True).text_embeds.double().cpu().numpy())
    return original.normalize(vectors)


def retrieve(output):
    import torch
    torch.set_num_threads(4)
    p = verify(output)
    if check_marker(output, 'retrieval_complete.json', p): return
    assert check_marker(output, 'candidates_complete.json', p)
    assert check_marker(output, 'offline_complete.json', p)
    qs, memories = rows(output / 'inputs/questions.jsonl'), rows(output / 'memories.jsonl')
    mmap = {m['id']: m for m in memories}
    candidates_by_id = {r['id']: r for r in rows(output / 'candidates.jsonl')}
    baseline = {r['id']: r for r in rows(SOURCE / 'retrieval.jsonl')}
    encoder = base.bge.load_text_encoder(p['text_encoder'], 'cpu')
    clip, tokenizer, limit = base.bge.load_clip_text_encoder(p['image_encoder'], 'cpu')
    results, diagnostics, files = {}, [], []
    for task in p['tasks']:
        ms, selected = [m for m in memories if m['task'] == task], [q for q in qs if q['task'] == task]
        dest = output / 'offline' / task
        graph = read(dest / 'graph.json')
        sums = np.load(dest / 'master_sums.npy', allow_pickle=False)
        masters, nodes = graph['masters'], {n['id']: n for n in graph['image_nodes']}
        centers = original.normalize([sums[m['visual_mean_ref']['row']] / m['n'] for m in masters])
        groups = [[i for i, m in enumerate(ms) if m['is_visual'] and nodes[m['id']+'#image1']['master_id'] == master['id']] for master in masters]
        with np.load(SOURCE / 'embeddings' / (task + '.npz'), allow_pickle=False) as saved:
            assert saved['memory_ids'].tolist() == [m['id'] for m in ms]
            assert saved['question_ids'].tolist() == [q['id'] for q in selected]
            vectors, original_queries = saved['memory_vectors'], saved['query_vectors']
        hypotheses = [h for q in selected for h in candidates_by_id[q['id']]['candidate_statements']]
        augmented = [q['retrieval_query']+'\nCandidate answer: '+h for q in selected for h in candidates_by_id[q['id']]['candidate_statements']]
        expanded = base.bge.encode_text(encoder, augmented, p['query_prefix']).reshape(len(selected), 4, -1)
        queries = np.concatenate([original_queries[:, None, :], expanded], axis=1)
        cv = clip_vectors(clip, tokenizer, limit, hypotheses).reshape(len(selected), 4, -1)
        cache = output / 'embeddings' / (task + '.npz'); cache.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache, memory_vectors=vectors, query_vectors=queries, candidate_vectors=cv,
            memory_ids=np.array([m['id'] for m in ms]), question_ids=np.array([q['id'] for q in selected]))
        files.append(cache)
        for qi, q in enumerate(selected):
            original_scores = vectors @ original_queries[qi]
            order = np.argsort(-original_scores, kind='stable')
            assert [ms[i]['id'] for i in order] == baseline[q['id']]['ranking_memory_ids']
            clips = centers @ cv[qi].T
            choices = np.argmax(clips, axis=1)
            raw, weighted, details = group_scores(vectors, queries[qi],
                [(indices, 1 + int(choices[i])) for i, indices in enumerate(groups)], p['lambda'], p['alpha'])
            pure = [i for i, m in enumerate(ms) if not m['is_visual']]
            np.testing.assert_array_equal(raw[pure], original_scores[pure])
            np.testing.assert_array_equal(weighted[pure], original_scores[pure])
            condition_rows = []
            for name, scores in zip(CONDITIONS, (raw, weighted)):
                order = np.argsort(-scores, kind='stable')
                ids = [ms[i]['id'] for i in order]
                condition_rows.append(dict(name=name, source_ids=ids[:5], ranking_memory_ids=ids,
                    ranking_scores=scores[order].tolist(), retrieved_context=base.context_for(ids[:5], mmap)))
            results[q['id']] = dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                candidate_count=len(ms), conditions=condition_rows,
                master_choices=[dict(master_id=m['id'], candidate_index=int(choices[i]),
                    clip_scores=clips[i].tolist()) for i, m in enumerate(masters)], experiment_signature=p['experiment_signature'])
            diagnostics.append(dict(id=q['id'], groups=details))
        print(f'Master paired retrieval {len(results)}/1711', flush=True)
    write_rows(output / 'retrieval.jsonl', [results[q['id']] for q in qs])
    write_rows(output / 'weighting_diagnostics.jsonl', diagnostics)
    seal(output, 'retrieval_complete.json', [*files, output / 'retrieval.jsonl', output / 'weighting_diagnostics.jsonl'], p,
        questions=1711, baseline_full_rankings_verified=1711,
        same_ordered_top5=sum(r['conditions'][0]['source_ids'] == r['conditions'][1]['source_ids'] for r in results.values()))


def answers(output, env_file):
    p = verify(output)
    if check_marker(output, 'reader/complete.json', p): return
    assert check_marker(output, 'retrieval_complete.json', p)
    qs = rows(output / 'inputs/questions.jsonl')
    rmap = {r['id']: r for r in rows(output / 'retrieval.jsonl')}
    manifest, jobs = [], {}
    for q in qs:
        for c in rmap[q['id']]['conditions']:
            payload = ga.reader_payload(q, c['retrieved_context'])
            key = ga.request_sha256(images.wire_for(payload, p['max_new_tokens']))
            job = dict(condition=c['name'], payload=payload, key=key)
            manifest.append(job)
            jobs.setdefault(key, job)
    root = output / 'reader'
    if (root / 'manifest.jsonl').exists(): assert rows(root / 'manifest.jsonl') == manifest
    else: write_rows(root / 'manifest.jsonl', manifest)
    model = transport.Model(root, env_file, p['expected_model'])
    def one(job):
        payload, key = job['payload'], job['key']
        path = root / 'responses' / (key + '.json')
        wire = images.wire_for(payload, p['max_new_tokens'])
        assert ga.request_sha256(wire) == key
        if path.exists():
            saved = read(path)
            assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
            out = saved['output']
        else:
            out = model.invoke(wire, 'answer', job['condition']+'::'+payload['record_id'])
            saved = dict(payload=payload, output=out, experiment_signature=p['experiment_signature'])
            if out['finish_reason'] != 'stop' or out['refusal'] or not out['raw_response'].strip():
                dump(root / 'invalid' / path.name, saved)
                raise RuntimeError('Invalid answer archived')
            dump(path, saved)
        assert out['request_sha256'] == key and out['returned_model'] == p['expected_model']
        assert out['finish_reason'] == 'stop' and not out['refusal'] and out['raw_response'].strip()
        return key, out
    outputs = {}
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        fs = [pool.submit(one, job) for job in jobs.values()]
        for i, f in enumerate(as_completed(fs), 1):
            key, out = f.result(); outputs[key] = out
            dump(root / 'progress.json', dict(done=i, total=len(jobs), evaluation_records=len(manifest)))
            if i % 50 == 0 or i == len(jobs): print(f'Master GPT-5 answers {i}/{len(jobs)} unique requests', flush=True)
    write_rows(root / 'responses.jsonl', [dict(id=m['payload']['record_id'], condition=m['condition'],
        **outputs[m['key']], experiment_signature=p['experiment_signature']) for m in manifest])
    seal(output, 'reader/complete.json', [root / 'manifest.jsonl', root / 'responses.jsonl',
        *root.glob('responses/*.json'), root / 'api_attempts.jsonl', root / 'api_responses.jsonl'], p,
        count=len(manifest), unique_requests=len(jobs), shared_requests=len(manifest)-len(jobs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'candidates', 'offline', 'retrieve', 'answers'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--env-file', type=Path, default=ga.DEFAULT_ENV)
    parser.add_argument('--whitening-lambda', type=float, default=.03)
    parser.add_argument('--reweight-alpha', type=float, default=1.)
    a = parser.parse_args(); output = a.output.resolve()
    status = output / 'stages' / (a.stage + '.json')
    dump(status, dict(status='running', pid=os.getpid(), updated=time.time()))
    try:
        if a.stage == 'prepare': prepare(output, a.whitening_lambda, a.reweight_alpha)
        elif a.stage == 'offline': offline(output)
        elif a.stage == 'retrieve': retrieve(output)
        else: globals()[a.stage](output, a.env_file)
    except BaseException as exc:
        dump(status, dict(status='failed', error=str(exc), updated=time.time())); raise
    dump(status, dict(status='complete', updated=time.time()))


if __name__ == '__main__': main()
