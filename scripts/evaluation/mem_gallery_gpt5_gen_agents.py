"""Full Mem-Gallery GA run with fresh GPT-5 history captions and GPT-5 answers.

Official question captions stay verbatim; native image bytes reach the reader.
Gold is opened only by analyze(). Each scene owns an isolated native GA memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_qwen_text_methods as gallery_run
from scripts.evaluation import gallery_gpt5_transport as transport
from scripts.evaluation.gpt5_text_memory_adapters import make_broker
from scripts.evaluation.openai_backbone_transport import request_sha256

native, gallery = gallery_run.native, gallery_run.gallery
read, rows, sha, obj = gallery_run.read, gallery_run.rows, gallery_run.sha, gallery_run.obj
dump, write_rows, freeze = gallery_run.dump, gallery_run.write_rows, gallery_run.freeze
SOURCE = gallery_run.SOURCE / 'inputs'
DEFAULT_OUTPUT = ROOT / 'results/mem_gallery_gpt5_gen_agents_20261028'
DEFAULT_ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')


def round_text(row, speaker, assistant):
    result = ['Date: ' + row['timestamp'], 'Round: ' + row['id']]
    if row['user'].strip():
        result.append(speaker + ': ' + row['user'].strip())
    if row['assistant'].strip():
        result.append(assistant + ': ' + row['assistant'].strip())
    raw = row['raw']
    assert len(raw['image_id']) == len(raw['image_caption']) == len(raw['input_image'])
    for image_id, caption in zip(raw['image_id'], raw['image_caption']):
        assert caption.strip()
        result.extend(['image:', 'image_id: ' + image_id, 'image_caption: ' + caption])
    return '\n'.join(result)


def wire_for(payload, cap=2048):
    return transport.api.build_request(payload, seed=42, max_output_tokens=cap, image_detail='high')


def prepare(output):
    if (output / 'protocol.json').exists():
        return verify(output)
    upstream = native.verify_vendor()
    source = read(SOURCE / 'protocol.json')
    for name, expected in source['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in read(SOURCE / 'artifact_manifest.json').items():
        assert sha(SOURCE / name) == expected, name
    questions = rows(SOURCE / 'questions.jsonl')
    assert len(questions) == len({q['id'] for q in questions}) == 1711
    assert all(set(q) == {'id', 'task', 'question', 'point', 'query_image_paths',
                         'query_image_caption', 'retrieval_query'} for q in questions)
    originals = read(ROOT / 'data/Mem-Gallery/source_manifest.json')
    observations, images, source_hashes = {}, {}, {}
    for entry in originals['files']:
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256']
        source_hashes[str(path)] = entry['sha256']
        data, task = read(path), path.stem
        name = str((data.get('character_profile') or {}).get('name') or '').strip()
        observations[task] = []
        for session in data['multi_session_dialogues']:
            for turn in session['dialogues']:
                refs, ids = turn.get('input_image') or [], turn.get('image_id') or []
                refs = [refs] if isinstance(refs, str) else refs
                ids = [ids] if isinstance(ids, str) else ids
                assert len(refs) == len(ids)
                hashes = []
                for ref in refs:
                    image = (path.parent / ref).resolve()
                    digest = sha(image)
                    source_hashes[str(image)] = digest
                    images.setdefault(digest, dict(id=digest, path=str(image), sha256=digest))
                    hashes.append(digest)
                observations[task].append(dict(id=turn['round'], session_id=session['session_id'],
                    timestamp=str(session.get('date') or '').strip(), speaker=f'user ({name})' if name else 'user',
                    user=turn['user'], assistant=turn['assistant'], caption_ids=hashes,
                    raw=dict(input_image=refs, image_id=ids)))
    assert len(observations) == 20 and sum(map(len, observations.values())) == 3962
    assert sum(len(o['caption_ids']) for v in observations.values() for o in v) == 1003
    for q in questions:
        expected = q['question']
        if q['query_image_caption']:
            expected += "\nquestion's image:\nimage_caption: " + q['query_image_caption']
        assert q['retrieval_query'] == expected
        for path in q['query_image_paths']:
            source_hashes[path] = sha(path)
    prompt_source = ROOT / 'results/natural_observation/protocol.json'
    prompt = read(prompt_source)['caption_prompt']
    embedding = native.EMBEDDING / 'snapshots' / (native.EMBEDDING / 'refs/main').read_text().strip()
    paths = {Path(__file__), Path(transport.__file__), prompt_source,
             ROOT / 'scripts/evaluation/continue_deepseek_text_methods_streaming.py',
             ROOT / 'scripts/evaluation/continue_deepseek_ga_format.py',
             ROOT / 'scripts/evaluation/probe_deepseek_ga_format.py',
             native.VENDOR / 'UPSTREAM.json', native.VENDOR / 'gen_agents.yaml',
             ROOT / 'data/Mem-Gallery/source_manifest.json'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and str(filename).startswith(str(ROOT / 'scripts')) and str(filename).endswith('.py'):
            paths.add(Path(filename))
    paths.update(native.VENDOR / name for name in upstream['files'])
    paths.update(f for f in embedding.rglob('*') if f.is_file())
    source_hashes.update({str(f): sha(f) for f in paths})
    output.mkdir(parents=True, exist_ok=True)
    for name in ('questions.jsonl', 'gold.jsonl', 'protocol.json', 'artifact_manifest.json'):
        target = output / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
        source_hashes[str(target)] = sha(target)
    freeze(output / 'observations_without_captions.json', observations)
    write_rows(output / 'caption_jobs.jsonl', list(images.values()))
    for name in ('observations_without_captions.json', 'caption_jobs.jsonl'):
        source_hashes[str(output / name)] = sha(output / name)
    snapshots = {}
    for f in paths:
        if f.suffix == '.py':
            digest = sha(f)
            dest = output / 'source_snapshot' / (digest[:16] + '_' + f.name)
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(f, dest)
            snapshots[str(f)] = dict(path=str(dest), sha256=digest)
    freeze(output / 'source_snapshot/manifest.json', snapshots)
    probe = read(output / 'api_probe.json')
    assert probe['status'] == 'passed'
    protocol = dict(schema='mem_gallery_gpt5_ga_v1', mode='gen_agents',
        requested_model='gpt-5', expected_model=probe['output']['returned_model'],
        seed=42, reasoning_effort='minimal', temperature_wire_policy='omitted',
        endpoint='https://api.openai.com/v1/chat/completions', image_detail='high',
        caption_prompt=prompt, caption_max_tokens=2048, memory_max_new_tokens=2048, max_new_tokens=2048,
        tasks=sorted(observations), questions=1711, memory_rounds=3962,
        history_image_occurrences=1003, unique_history_images=len(images),
        task_workers=8, api_workers=8, embedding_device='cpu', embedding_model_path=str(embedding),
        method_defaults=native.effective_method_defaults('gen_agents'), upstream=upstream,
        history_caption_policy='Fresh GPT-5 captions of every unique historical image; no previous caption responses reused.',
        query_policy='Original question plus supplied official question caption; no answers/options/clues or session filtering.',
        reader_policy='Official Gallery system/task prompts; retrieved text and original query image bytes.',
        reader_system_prompt=gallery.READER_SYSTEM, task_instructions=gallery.TASK_INSTRUCTIONS,
        metadata_policy='Preserve source date, round, image ID, dialogue and fresh image caption verbatim.',
        memory_format_policy='Uniform explicit output-format system message for every GA call; up to three identical-request format attempts; archive invalid outputs, never fabricate.',
        gold_policy='Only analyze opens copied gold.jsonl; never used in captions, ingestion, retrieval or answering.',
        context_policy='Native K=10; native 4000 space-separated-word truncation; original and reflection memories compete.',
        source_sha256=source_hashes, questions_sha256=obj(questions),
        differences_from_qwen_reference=['GPT-5 fresh historical captions, importance, reflection, and answers.',
            'GPT-5 minimal reasoning, omitted temperature, explicit seed42; 2048-token completion budget.',
            'Original query image bytes with API high detail instead of Qwen image processor.',
            'Uniform GA format instructions as in the existing GPT-5 MemEye experiment.'])
    protocol['experiment_signature'] = obj(protocol)
    freeze(output / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('tasks', 'questions', 'memory_rounds', 'unique_history_images')}), flush=True)
    return protocol


def verify(output):
    p = read(output / 'protocol.json')
    assert p['experiment_signature'] == obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    for path, expected in p['source_sha256'].items():
        assert sha(path) == expected, path
    for entry in read(output / 'source_snapshot/manifest.json').values():
        assert sha(entry['path']) == entry['sha256']
    return p


def cached_completion(model, path, payload, protocol, stage, record_id):
    wire = wire_for(payload, protocol['caption_max_tokens'] if stage == 'caption' else protocol['max_new_tokens'])
    key = request_sha256(wire)
    if path.exists():
        saved = read(path)
        assert saved['payload'] == payload and saved['output']['request_sha256'] == key
        assert saved['experiment_signature'] == protocol['experiment_signature']
        assert saved['output']['returned_model'] == protocol['expected_model']
        return saved['output']
    out = model.invoke(wire, stage, record_id)
    saved = dict(id=record_id, payload=payload, output=out, experiment_signature=protocol['experiment_signature'])
    if out['finish_reason'] != 'stop' or out.get('refusal') or not out['raw_response'].strip():
        dump(path.parent / 'invalid' / path.name, saved)
        raise RuntimeError('Invalid/empty/truncated generation archived: ' + str(path))
    dump(path, saved)
    return out


def captions(output, env_file, limit=0):
    p = verify(output)
    jobs = rows(output / 'caption_jobs.jsonl')
    model = transport.Model(output / 'caption_source', env_file, p['expected_model'])
    def one(job):
        payload = dict(user_text=p['caption_prompt'], query_images=[dict(path=job['path'], sha256=job['sha256'])])
        out = cached_completion(model, output / 'caption_source/responses' / (job['id'] + '.json'),
                                payload, p, 'caption', job['id'])
        return dict(id=job['id'], caption=out['raw_response'].strip(), **out)
    selected = jobs[:limit] if limit else jobs
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        futures = [pool.submit(one, job) for job in selected]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            dump(output / 'caption_source/progress.json', dict(done=i, total=len(jobs), phase='caption'))
            if i % 20 == 0 or i == len(selected):
                print(f'GPT-5 captions {i}/{len(jobs)}', flush=True)
    if limit:
        return
    all_captions = [one(job) for job in jobs]
    write_rows(output / 'caption_source/captions.jsonl', all_captions)
    capmap = {c['id']: c['caption'] for c in all_captions}
    observations = read(output / 'observations_without_captions.json')
    for group in observations.values():
        for row in group:
            row['raw']['image_caption'] = [capmap[key] for key in row['caption_ids']]
    freeze(output / 'observations.json', observations)
    freeze(output / 'caption_source/complete.json', dict(count=len(all_captions),
        captions_sha256=sha(output / 'caption_source/captions.jsonl'),
        observations_sha256=sha(output / 'observations.json'), experiment_signature=p['experiment_signature']))


class Memory(native.Memory):
    def __init__(self, protocol, workdir):
        super().__init__('gen_agents', protocol, workdir)
        self.helper['_build_round_text'] = round_text
        original = self.core.recall_op.__retention__
        def capture(ids, timestamp):
            self.selected_ids = [int(i) for i in ids]
            original(ids, timestamp)
        self.core.recall_op.__retention__ = capture

    def recall(self, query):
        self.selected_ids = []
        context, _ = super().recall(query)
        return str(context), dict(selected_memory_ids=self.selected_ids)


def memory_stage(output, env_file, only_task=None):
    p = verify(output)
    c = read(output / 'caption_source/complete.json')
    assert c['observations_sha256'] == sha(output / 'observations.json')
    assert c['captions_sha256'] == sha(output / 'caption_source/captions.jsonl')
    observations, qs = read(output / 'observations.json'), rows(output / 'inputs/questions.jsonl')
    root = output / 'gen_agents'
    root.mkdir(exist_ok=True)
    pending = [t for t in p['tasks'] if (not only_task or t == only_task) and not (root / t / 'complete.json').exists()]
    if pending:
        model = transport.Model(root, env_file, p['expected_model'])
        model.torch.set_num_threads(2)
        broker = make_broker('gen_agents', model, root, p)
        try:
            encoder = native.install_runtime(broker, p)
            dump(root / 'generation_environment.json', model.environment | dict(embedding_device=str(encoder.device)))
            def one(task):
                broker.local.owner = task
                target = root / task
                runtime = target / 'runtime'
                runtime.mkdir(parents=True, exist_ok=True)
                memory = Memory(p, runtime)
                for i, row in enumerate(observations[task], 1):
                    if model.fatal_errors:
                        raise RuntimeError('Fatal API error')
                    memory.store(row)
                    dump(target / 'progress.json', dict(phase='ingest', done=i, total=len(observations[task])))
                    if i % 25 == 0:
                        print(f'GA {task}: {i}/{len(observations[task])} rounds', flush=True)
                recalled = []
                selected = [q for q in qs if q['task'] == task]
                for q in selected:
                    context, detail = memory.recall(q['retrieval_query'])
                    recalled.append(dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                        retrieved_context=context, retrieval_detail=detail, experiment_signature=p['experiment_signature']))
                write_rows(target / 'retrieval.jsonl', recalled)
                memory.export(target)
                dump(target / 'complete.json', dict(experiment_signature=p['experiment_signature'], questions=len(recalled),
                    retrieval_sha256=sha(target / 'retrieval.jsonl'), final_memory_sha256=sha(target / 'final_memory.json')))
                dump(target / 'progress.json', dict(phase='complete', done=len(selected), total=len(selected)))
                print(f'GA {task}: retrieval complete ({len(selected)} questions)', flush=True)
            errors = []
            with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
                jobs = {pool.submit(one, t): t for t in pending}
                for future in as_completed(jobs):
                    try:
                        future.result()
                    except Exception:
                        errors.append(dict(task=jobs[future], traceback=traceback.format_exc()))
                        dump(root / 'run_errors.json', errors)
            if errors:
                raise RuntimeError(f'{len(errors)} memory tasks failed; see gen_agents/run_errors.json')
        finally:
            broker.close()
    if only_task:
        return
    all_rows = []
    for task in p['tasks']:
        done = read(root / task / 'complete.json')
        assert done['experiment_signature'] == p['experiment_signature']
        assert done['retrieval_sha256'] == sha(root / task / 'retrieval.jsonl')
        assert done['final_memory_sha256'] == sha(root / task / 'final_memory.json')
        all_rows.extend(rows(root / task / 'retrieval.jsonl'))
    rmap = {r['id']: r for r in all_rows}
    assert len(all_rows) == len(rmap) == len(qs) and set(rmap) == {q['id'] for q in qs}
    write_rows(root / 'retrieval.jsonl', [rmap[q['id']] for q in qs])


def reader_payload(q, context):
    return dict(record_id=q['id'], system=gallery.READER_SYSTEM,
        user_text=gallery_run.reader_text(q, context),
        query_images=[dict(path=path, sha256=sha(path)) for path in q['query_image_paths']])


def answers(output, env_file, limit=0):
    p = verify(output)
    qs = rows(output / 'inputs/questions.jsonl')
    retrieval = rows(output / 'gen_agents/retrieval.jsonl')
    rmap = {r['id']: r for r in retrieval}
    assert len(retrieval) == len(rmap) == len(qs)
    root = output / 'gen_agents/reader'
    model = transport.Model(root, env_file, p['expected_model'])
    manifest = [reader_payload(q, rmap[q['id']]['retrieved_context']) for q in qs]
    if (root / 'manifest.jsonl').exists():
        assert rows(root / 'manifest.jsonl') == manifest
    else:
        write_rows(root / 'manifest.jsonl', manifest)
    def one(payload):
        out = cached_completion(model, root / 'responses' / (obj(payload['record_id']) + '.json'),
                                payload, p, 'answer', payload['record_id'])
        return dict(id=payload['record_id'], **out, experiment_signature=p['experiment_signature'])
    selected = manifest[:limit] if limit else manifest
    with ThreadPoolExecutor(max_workers=p['api_workers']) as pool:
        futures = [pool.submit(one, v) for v in selected]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            dump(root / 'progress.json', dict(phase='answer', done=i, total=len(qs)))
            if i % 20 == 0 or i == len(selected):
                print(f'GPT-5 answers {i}/{len(qs)}', flush=True)
    if limit:
        return
    write_rows(root / 'responses.jsonl', [one(v) for v in manifest])
    freeze(root / 'complete.json', dict(count=len(qs), responses_sha256=sha(root / 'responses.jsonl'),
        manifest_sha256=sha(root / 'manifest.jsonl'), experiment_signature=p['experiment_signature']))


def analyze(output):
    p = verify(output)
    qs = rows(output / 'inputs/questions.jsonl')
    gold = {r['id']: r for r in rows(output / 'inputs/gold.jsonl')}
    observations = read(output / 'observations.json')
    cap_done = read(output / 'caption_source/complete.json')
    assert cap_done['observations_sha256'] == sha(output / 'observations.json')
    assert cap_done['captions_sha256'] == sha(output / 'caption_source/captions.jsonl')
    cap_rows = rows(output / 'caption_source/captions.jsonl')
    capmap = {r['id']: r for r in cap_rows}
    jobs = rows(output / 'caption_jobs.jsonl')
    assert len(capmap) == len(cap_rows) == len(jobs)
    for job in jobs:
        saved = read(output / 'caption_source/responses' / (job['id'] + '.json'))
        payload = dict(user_text=p['caption_prompt'], query_images=[dict(path=job['path'], sha256=job['sha256'])])
        assert saved['payload'] == payload
        assert saved['output']['request_sha256'] == request_sha256(wire_for(payload))
        assert saved['output']['returned_model'] == p['expected_model']
        assert capmap[job['id']]['caption'] == saved['output']['raw_response'].strip()
    root = output / 'gen_agents'
    notes, retrieval = {}, []
    for task in p['tasks']:
        done = read(root / task / 'complete.json')
        assert done['experiment_signature'] == p['experiment_signature']
        assert done['retrieval_sha256'] == sha(root / task / 'retrieval.jsonl')
        assert done['final_memory_sha256'] == sha(root / task / 'final_memory.json')
        notes[task] = read(root / task / 'final_memory.json')
        originals = [n for n in notes[task] if n.get('source') is False]
        by_id = {n['dialogue_id']: n for n in originals}
        assert len(originals) == len(by_id) == len(observations[task])
        for row in observations[task]:
            assert row['raw']['image_caption'] == [capmap[k]['caption'] for k in row['caption_ids']]
            assert by_id[row['id']]['text'] == round_text(row, row['speaker'], 'assistant')
        for n in notes[task]:
            if n.get('source') is not False:
                assert n['source'] and all(0 <= mid < n['counter_id'] for mid in n['source'])
        retrieval.extend(rows(root / task / 'retrieval.jsonl'))
    rmap = {r['id']: r for r in retrieval}
    answers_list = rows(root / 'reader/responses.jsonl')
    amap = {r['id']: r for r in answers_list}
    assert len(retrieval) == len(rmap) == len(answers_list) == len(amap) == len(gold) == len(qs) == 1711
    assert set(rmap) == set(amap) == set(gold) == {q['id'] for q in qs}
    scored = []
    for q in qs:
        r, a = rmap[q['id']], amap[q['id']]
        assert r['retrieval_query'] == q['retrieval_query']
        ids = r['retrieval_detail']['selected_memory_ids']
        assert len(ids) == len(set(ids)) <= 10
        chosen = [notes[q['task']][i] for i in ids]
        full = '\n' + '\n\n'.join(f'[{i}] ' + n['text'] for i, n in enumerate(chosen)) + '\n'
        expected_context = ' '.join(full.split(' ')[:4000])
        assert r['retrieved_context'] == expected_context, q['id']
        payload = reader_payload(q, expected_context)
        saved = read(root / 'reader/responses' / (obj(q['id']) + '.json'))
        assert saved['payload'] == payload and saved['output']['raw_response'] == a['raw_response']
        assert a['request_sha256'] == request_sha256(wire_for(payload))
        assert a['returned_model'] == p['expected_model'] and a['seed'] == 42
        assert a['finish_reason'] == 'stop' and not a['refusal'] and a['raw_response'].strip()
        assert r['experiment_signature'] == a['experiment_signature'] == p['experiment_signature']
        # Count only fully rendered original rounds, never reflection source links or partial final pages.
        sources = []
        for i, n in enumerate(chosen):
            block = f'[{i}] ' + n['text']
            if n.get('source') is False and block in expected_context:
                sources.append(q['task'] + '/' + n['dialogue_id'])
        target = gold[q['id']]
        gold_ids = set(target['gold_ids'])
        scored.append(dict(id=q['id'], task=q['task'], point=q['point'], answer=target['answer'],
            raw_response=a['raw_response'], **gallery.score_answer(a['raw_response'], target['answer']),
            context_count=len(ids), source_ids=sources, gold_ids=sorted(gold_ids),
            gold_hits=len(gold_ids & set(sources)), gold_count=len(gold_ids)))
    from scripts.evaluation.continue_deepseek_ga_format import instruction_for
    import ast
    calls = list((root / 'calls').glob('*.json'))
    call_hashes = {}
    for path in calls:
        c = read(path)
        req, out = c['request'], c['output']
        assert obj(req) == path.stem and req['experiment_signature'] == p['experiment_signature']
        assert req['kind'] == 'memory' and req['seed'] == 42
        assert out['request_sha256'] == request_sha256(transport.build_request(req['messages'], req['max_new_tokens'], req['temperature']))
        assert out['returned_model'] == p['expected_model'] and out['finish_reason'] == 'stop' and not out['refusal']
        instruction, kind = instruction_for(req['messages'][1:])
        assert req['messages'][0] == dict(role='system', content=instruction)
        raw = out['raw_response'].strip()
        assert (1 <= float(ast.literal_eval(raw)) <= 10) if kind == 'rating' else len([s for s in raw.splitlines() if s.strip()]) == 2
        call_hashes[str(path)] = sha(path)
    events = rows(root / 'call_events.jsonl')
    assert all(str(root / 'calls' / (e['key'] + '.json')) in call_hashes or
               (root / 'format_failures' / e['key']).exists() for e in events)
    def summary(items):
        valid = [v for v in items if v['gold_count']]
        return dict(questions=len(items), correct=sum(int(v['em']) for v in items),
            **{k: sum(v[k] for v in items) / len(items) for k in ('em', 'f1', 'bleu')},
            source_recall_micro=sum(v['gold_hits'] for v in valid) / sum(v['gold_count'] for v in valid) if valid else None,
            source_recall_macro=sum(v['gold_hits'] / v['gold_count'] for v in valid) / len(valid) if valid else None,
            context_counts=dict(Counter(v['context_count'] for v in items)))
    result = dict(status='complete', model=p['expected_model'], seed=42, overall=summary(scored),
        by_task={t: summary([r for r in scored if r['task'] == t]) for t in p['tasks']},
        by_type={t: summary([r for r in scored if r['point'] == t]) for t in sorted({q['point'] for q in qs})},
        captions=len(cap_rows), original_memories=sum(map(len, observations.values())),
        total_memories_with_reflections=sum(map(len, notes.values())), memory_calls=len(calls),
        format_retry_attempts=len(rows(root / 'format_retry_events.jsonl')) if (root / 'format_retry_events.jsonl').exists() else 0,
        experiment_signature=p['experiment_signature'])
    write_rows(output / 'scored_questions.jsonl', scored)
    with (output / 'scored_questions.csv').open('w') as f:
        fields = ['id', 'task', 'point', 'answer', 'raw_response', 'em', 'f1', 'bleu', 'context_count', 'gold_hits', 'gold_count']
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(scored)
    dump(output / 'final_results.json', result)
    artifacts = [output / 'protocol.json', output / 'observations.json', output / 'scored_questions.jsonl',
                 output / 'final_results.json', root / 'reader/responses.jsonl', root / 'reader/manifest.jsonl',
                 output / 'caption_source/captions.jsonl']
    dump(output / 'audit.json', dict(status='passed', questions=1711, original_rounds=3962,
        caption_request_and_image_hashes_verified=True, memory_originals_and_reflection_links_verified=True,
        native_context_reconstructed=True, reader_payload_and_image_hashes_verified=True,
        gold_used_only_in_analysis=True, memory_call_sha256=call_hashes,
        artifact_sha256={str(f): sha(f) for f in artifacts}))
    s = result['overall']
    body = ('<!doctype html><meta charset="utf-8"><title>Mem-Gallery · GPT-5 · Generative Agents</title>'
        '<style>body{font:17px system-ui;max-width:1000px;margin:40px auto;line-height:1.6}td,th{padding:12px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
        '<h1>Mem-Gallery · GPT-5 · Generative Agents</h1>'
        f'<p>20 个场景，3,962 轮历史，1,711 题；历史图片重新生成 GPT-5 caption；seed=42。实际模型：{html.escape(p["expected_model"])}。</p>'
        f'<p>EM：{s["correct"]}/1711（{s["em"]:.2%}）；F1：{s["f1"]:.2%}；BLEU：{s["bleu"]:.2%}。</p>'
        f'<p>原始来源召回：micro {s["source_recall_micro"]:.2%}，macro {s["source_recall_macro"]:.2%}。原生 K=10，不标为 Recall@5。</p>'
        '<p>题图官方 caption 用于检索，原始题图用于回答。重要性与反思均由 GPT-5 生成，MiniLM 编码及原生检索规则保留。'
        '历史 caption、记忆操作和回答模型均改变，因此与 Qwen 差异不能只归因于回答模型。</p>'
        '<p><a href="scored_questions.csv">逐题 CSV</a> · <a href="final_results.json">完整结果</a> · <a href="audit.json">审计</a></p>')
    (output / 'report.html').write_text(body)
    print(json.dumps(result['overall']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'captions', 'memory', 'answers', 'analyze', 'all'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--task')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = ['prepare', 'captions', 'memory', 'answers', 'analyze'] if args.stage == 'all' else [args.stage]
    try:
        for stage in stages:
            dump(output / 'execution_status.json', dict(status='running', stage=stage, pid=os.getpid(), updated=time.time()))
            if stage == 'prepare': prepare(output)
            elif stage == 'captions': captions(output, args.env_file, args.limit)
            elif stage == 'memory': memory_stage(output, args.env_file, args.task)
            elif stage == 'answers': answers(output, args.env_file, args.limit)
            else: analyze(output)
    except BaseException as exc:
        dump(output / 'execution_status.json', dict(status='stopped_before_completion', stage=stage,
            error=str(exc), updated=time.time(), results_available=False))
        raise
    dump(output / 'execution_status.json', dict(status='complete' if stages[-1] == 'analyze' else 'stage_complete',
        stage=stages[-1], updated=time.time(), results_available=stages[-1] == 'analyze'))


if __name__ == '__main__':
    main()
