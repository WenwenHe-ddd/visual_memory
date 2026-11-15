"""Open MemEye queries against the pinned CLIP and HippoMM GPT-5 memories.

No gold or MCQ options are read by retrieval or answering. HippoMM preserves
its summary fast path, visual-query rewrite, per-event fallback, and detailed
representative-image reader instead of substituting a static Top-5 retriever.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import gzip
import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]

from scripts.evaluation import mem_gallery_master_reader as runtime
from scripts.memory import clip_image_memory as clip_base
from scripts.memory import hippomm as hip_base, hippomm_core as hip_core
from scripts.memory import hippomm_original as hip_original
from scripts.memory import hippomm_api as hip_api
from scripts.evaluation import openai_gpt5_base_transport as gpt

OPEN = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
DEFAULT = ROOT / 'results/memeye_gpt5_open_20261105/visual_methods'
CLIP_SOURCE = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue'
HIPPO_SOURCE = ROOT / 'results/hippomm_api_memeye_20261104/gpt5'
MODEL = 'gpt-5-2025-08-07'
CLIP_API_SOURCE = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue_api/gpt-5'
ENV_FILE = Path('/storage/hewenwen/NTU Decentralized Agent/.env')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing frozen artifact differs: ' + str(path))
    path.write_text(text)


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing frozen artifact differs: ' + str(path))
    path.write_text(text)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def questions():
    values = read(OPEN)
    assert len(values) == len({q['id'] for q in values}) == 371
    assert not any(set(q) & {'answer', 'options', 'gold_ids', 'gold_choice'} for q in values)
    return values


def payload(messages):
    paths = [c['path'] for m in messages if isinstance(m['content'], list)
             for c in m['content'] if c['type'] == 'image']
    hashes = {path: sha(path) for path in paths}
    for message in messages:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'image':
                    part['sha256'] = hashes[part['path']]
    return {'reader_messages': messages, 'reader_image_paths': paths,
            'reader_image_hashes': hashes,
            'payload_sha256': runtime.digest({'messages': messages, 'image_hashes': hashes})}


def prepare_clip(output=DEFAULT / 'clip_image_dialogue', device='cpu'):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    qs = questions()
    protocol = json.loads((CLIP_SOURCE / 'protocol.json').read_text())
    source = Path(protocol['frozen_retrieval_source'])
    candidates = read(source / 'image_candidates.jsonl')
    frozen = read(CLIP_SOURCE / 'qa_manifest.jsonl')
    responses = {r['id']: r for r in read(CLIP_SOURCE / 'qa_responses.jsonl')}
    assert len(responses) == len(frozen) == 371
    old_api = {r['id']: r for r in read(CLIP_API_SOURCE / 'qa_responses.jsonl')}
    correct = sum(old_api[q['id']]['predicted_choice'] == q['gold_choice'] for q in frozen)
    assert correct == 207
    api_protocol = json.loads((CLIP_API_SOURCE / 'protocol.json').read_text())
    assert api_protocol['source'] == str(CLIP_SOURCE)
    assert api_protocol['settings']['max_completion_tokens'] == 2048
    dialogues = {}
    for row in frozen:
        for mid, text in row['dialogue_by_memory_id'].items():
            if mid in dialogues and dialogues[mid] != text:
                raise ValueError('Conflicting source dialogue: ' + mid)
            dialogues[mid] = text
    # Some historical rounds might never have been retrieved by MCQ queries.
    for filename in protocol['source_dialogue_files']:
        task = Path(filename).name.removesuffix('_Open.json')
        for session in json.loads(Path(filename).read_text())['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = task + '/' + turn['round']
                text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                if mid in dialogues and dialogues[mid] != text:
                    raise ValueError('Source dialogue differs from frozen MCQ snapshot')
                dialogues[mid] = text
    vectors_file = source / 'clip_embeddings.npz'
    with np.load(vectors_file) as archive:
        vectors = archive['image_vectors'].copy()
        assert archive['image_ids'].tolist() == [c['image_id'] for c in candidates]
    assert vectors.shape == (433, 512)
    for item in candidates:
        assert sha(item['path']) == item['image_id']
    checkpoint = ROOT / 'models/clip-vit-base-patch32'
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        checkpoint, config=config.text_config, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
    model.to(device=device, dtype=torch.float32).eval()
    texts = [q['question'] for q in qs]
    lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]
    query_vectors = []
    limit = config.text_config.max_position_embeddings
    for start in range(0, len(texts), 32):
        encoded = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                            max_length=limit, return_tensors='pt').to(device)
        with torch.inference_mode():
            value = model(**encoded, return_dict=True).text_embeds.double().cpu().numpy()
        query_vectors.extend(value / np.linalg.norm(value, axis=1, keepdims=True))
    system = ('You answer questions about remembered images. Use the retrieved historical '
              'images and their accompanying dialogue and any query image provided. '
              'Return a concise natural-language answer. If the evidence is insufficient, say so.')
    rows, traces = [], []
    frozen_by_id = {r['id']: r for r in frozen}
    unchanged_checks = 0
    for q, vector, length in zip(qs, query_vectors, lengths):
        retrieved = clip_base.rank_images(q, candidates, vectors, vector, 5)
        original = frozen_by_id[q['id']]
        if q['question'] == original['question']:
            assert [r['image_id'] for r in retrieved] == [r['image_id'] for r in original['retrieved_images']]
            unchanged_checks += 1
        content = [{'type': 'text', 'text': 'Retrieved historical images (ranked by relevance):'}]
        for i, item in enumerate(retrieved, 1):
            dialogue = '\n\n'.join(dialogues[mid] for mid in item['source_memory_ids'])
            content.extend([{'type': 'text', 'text': f'Historical image {i}:'},
                            {'type': 'image', 'path': item['path']},
                            {'type': 'text', 'text': f'Dialogue associated with historical image {i}:\n' + dialogue}])
        for i, path in enumerate(q['query_image_paths'], 1):
            content.extend([{'type': 'text', 'text': f'Question image {i} (part of the question):'},
                            {'type': 'image', 'path': path}])
        content.append({'type': 'text', 'text': 'Question: ' + q['question']
                        + '\nReturn a concise natural-language answer.'})
        ids = list(dict.fromkeys(mid for item in retrieved for mid in item['source_memory_ids']))
        trace = {'id': q['id'], 'task': q['task'], 'retrieval_query': q['question'],
                 'query_token_count': length, 'query_truncated': length > limit,
                 'retrieved_images': retrieved, 'top_memory_ids': ids}
        rows.append({**q, 'method_id': 'clip_image_dialogue', 'top_memory_ids': ids,
                     'retrieval_trace': trace, 'max_completion_tokens': 2048,
                     **payload([{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])})
        traces.append(trace)
    write(output / 'reader_manifest.jsonl', rows)
    write(output / 'retrieval.jsonl', traces)
    files = [OPEN, Path(__file__), Path(clip_base.__file__), vectors_file,
             source / 'image_candidates.jsonl', CLIP_SOURCE / 'protocol.json',
             CLIP_SOURCE / 'qa_manifest.jsonl', CLIP_SOURCE / 'qa_responses.jsonl',
             CLIP_API_SOURCE / 'protocol.json', CLIP_API_SOURCE / 'qa_responses.jsonl',
             CLIP_API_SOURCE / 'summary.json']
    files.extend(Path(p) for p in protocol['source_dialogue_files'])
    files.extend(p for p in checkpoint.iterdir() if p.is_file())
    dump(output / 'protocol.json', {'method_id': 'clip_image_dialogue', 'display_label': 'VisRAG',
        'implementation_note': 'Screenshot label; actual CLIP image + dialogue route, not official VisRAG.',
        'dataset': 'MemEye Open', 'answer_model': MODEL, 'questions': 371, 'top_k': 5,
        'source_mcq_correct': correct, 'source_mcq_root': str(CLIP_API_SOURCE),
        'query_policy': 'True Open question text only; no query caption, choices, or gold.',
        'memory_policy': 'Frozen 433 task/image candidates, 438 image-associated historical rounds.',
        'query_truncation': {'limit': limit, 'count': sum(n > limit for n in lengths)},
        'settings': api_protocol['settings'], 'image_policy': api_protocol['image_policy'], 'seed': 42,
        'source_sha256': {str(p): sha(p) for p in files},
        'artifact_sha256': {n: sha(output / n) for n in ('reader_manifest.jsonl', 'retrieval.jsonl')}})
    print(json.dumps({'method': 'clip_image_dialogue', 'prepared': len(rows), 'unchanged_question_rankings_checked': unchanged_checks,
                      'original_mcq_correct': correct, 'query_truncated': sum(n > limit for n in lengths)}), flush=True)


def prepare_hippomm(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    qs = questions()
    source = HIPPO_SOURCE / 'online'; memory = HIPPO_SOURCE / 'offline'
    original = read(source / 'per_question.jsonl')
    assert len(original) == 371 and sum(r['accuracy'] for r in original) == 190
    assert {r['id'] for r in original} == {q['id'] for q in qs}
    old = json.loads((source / 'protocol.json').read_text())['contract']
    assert old['model'] == MODEL and old['max_output_tokens'] == 2048
    assert old['top_k'] == 5 and old['confidence_threshold'] == .75
    pinned = json.loads((HIPPO_SOURCE / 'experiment_manifest.json').read_text())['contract']['implementation']
    for filename, expected in pinned.items():
        assert sha(filename) == expected, filename
    events = read(memory / 'events.jsonl')
    images = read(memory / 'representative_images.jsonl')
    assert len(events) == len(images) == 250
    captions = hip_api.caption_source('gpt5')[1]
    dialogues = {r['id']: r for r in read(memory / 'dialogues.jsonl')}
    emap = {e['event_id']: e for e in events}
    files = [OPEN, Path(__file__), source / 'protocol.json', source / 'summary.json',
             source / 'per_question.jsonl', source / 'qa_responses.jsonl',
             HIPPO_SOURCE / 'experiment_manifest.json']
    files += [Path(f) for f in pinned]
    files += [memory / n for n in ('memory.json', 'events.jsonl', 'dialogues.jsonl',
              'image_mapping.jsonl', 'visual_embeddings.npz', 'representative_images.jsonl',
              'caption_provenance.jsonl', 'build_contract.json')]
    capdir = Path(old['caption_source'])
    files += [capdir / 'captions.jsonl', capdir / 'completion_manifest.json']
    for img in images:
        path = memory / img['file']; event = emap[img['event_id']]
        assert event['caption'] == captions[sha(path)]['caption']
        summaries = []
        for call in (memory / 'llm_cache' / event['event_id']).glob('call_*.json'):
            row = json.loads(call.read_text())
            if row['request']['role'] == 'summary':
                assert row['request']['prompt'] == hip_core.summary_prompt(event, dialogues)
                assert row['response']['returned_model'] == MODEL
                assert row['response']['finish_reason'] == 'stop'
                assert row['response']['raw_response'].strip() == event['summary']
                summaries.append(call)
        assert len(summaries) == 1
        files += [path, *summaries]
    files += [p for p in (ROOT / 'models/clip-vit-base-patch32').iterdir() if p.is_file()]
    write(output / 'questions.jsonl', qs)
    dump(output / 'protocol.json', {
        'method_id': 'hippomm', 'display_label': 'HippoMM', 'dataset': 'MemEye Open',
        'answer_model': MODEL, 'source_mcq_correct': 190, 'source_mcq_root': str(source),
        'memory_dir': str(memory), 'questions': 371, 'top_k': 5,
        'max_completion_tokens': 2048, 'seed': 42, 'reasoning_effort': 'minimal',
        'temperature': 'omitted', 'image_detail': 'high', 'image_resize': False,
        'confidence_threshold': .75, 'fallback_threshold': .4, 'fallback_score': .6,
        'query_recovery': old['query_recovery'], 'api_attempts': 4,
        'policy': 'Native GPT memory and original fast/rewrite/caption-fallback/representative-image answer pipeline; true Open questions; no options or gold.',
        'source_sha256': {str(p): sha(p) for p in sorted(set(files))},
        'questions_sha256': sha(output / 'questions.jsonl')})
    print(json.dumps({'method': 'hippomm', 'prepared': 371, 'source_mcq_correct': 190}), flush=True)


class AuditedChat:
    """Source-identical GPT wires, plus durable exact-wire and every-attempt logs."""
    def __init__(self, directory, allow_api=True):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.allow_api = allow_api
        self.config = gpt.load_config(ENV_FILE) if allow_api else None
        self.last_call_id = None
        self.index = {}
        for path in sorted(self.directory.glob('call_*.json')):
            row = json.loads(path.read_text())
            key = runtime.digest(row['request'])
            if key in self.index:
                raise ValueError('Duplicate native API cache request')
            self.index[key] = path.stem

    def __call__(self, role, prompt, image_paths=()):
        request = {'role': role, 'prompt': prompt, 'image_paths': [str(p) for p in image_paths],
                   'backbone': 'gpt5', 'max_output_tokens': 2048}
        images = [{'path': str(p), 'sha256': sha(p)} for p in image_paths]
        body = gpt.build_request({'system': runtime.READER_SYSTEM, 'user_text': prompt,
                                  'query_images': images}, seed=42,
                                 max_output_tokens=2048, image_detail='high')
        key = runtime.digest(request)
        call_id = self.index.get(key, f'call_{len(self.index) + 1:04d}')
        path = self.directory / (call_id + '.json')
        wire_path = self.directory / (call_id + '.wire.json.gz')
        wire_hash = runtime.digest(body)
        if path.exists():
            row = json.loads(path.read_text())
            assert row['request'] == request and row['request_sha256'] == wire_hash
            assert row['wire_file_sha256'] == sha(wire_path)
            actual = json.loads(gzip.decompress(wire_path.read_bytes()))
            assert actual == body and runtime.digest(actual) == wire_hash
        else:
            if not self.allow_api:
                raise ValueError('Offline replay requires a cached native request')
            # Write request before transmission; credentials are never persisted.
            wire_bytes = gzip.compress(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False).encode(), mtime=0)
            if wire_path.exists() and wire_path.read_bytes() != wire_bytes:
                raise ValueError('Unfinished native request wire differs')
            wire_path.write_bytes(wire_bytes)
            response = None
            for attempt in range(1, 5):
                began = time.time()
                try:
                    response = gpt.chat_request(self.config, body, timeout=60)
                    append(self.directory / 'api_attempts.jsonl', {
                        'call_id': call_id, 'attempt': attempt, 'started_unix': began,
                        'status': 'success', 'request_sha256': wire_hash,
                        'response_id': response['response_id'],
                        'request_id': response.get('request_id')})
                    break
                except gpt.APIError as exc:
                    append(self.directory / 'api_attempts.jsonl', {
                        'call_id': call_id, 'attempt': attempt, 'started_unix': began,
                        'status': 'error', 'request_sha256': wire_hash,
                        'kind': exc.kind, 'retryable': exc.retryable,
                        'http_status': exc.http_status, 'code': exc.code})
                    if not exc.retryable or attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 30))
            row = {'call_id': call_id, 'request': request, 'response': response,
                   'request_sha256': wire_hash, 'wire_file_sha256': sha(wire_path),
                   'image_sha256': {r['path']: r['sha256'] for r in images}}
            dump(path, row); self.index[key] = call_id
        self.last_call_id = call_id
        response = row['response']
        if response['returned_model'] != MODEL:
            raise ValueError('Returned model differs from frozen GPT caption backbone')
        if response['finish_reason'] != 'stop' or response.get('refusal') or not response['raw_response'].strip():
            raise ValueError('Native GPT response did not finish normally; raw preserved: ' + str(path))
        return response['raw_response']


def load_hippo(output):
    import numpy as np
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen GPT source changed: ' + path)
    assert sha(output / 'questions.jsonl') == protocol['questions_sha256']
    qs = read(output / 'questions.jsonl'); assert qs == questions()
    memory = Path(protocol['memory_dir'])
    events = read(memory / 'events.jsonl')
    dialogues = {r['id']: r for r in read(memory / 'dialogues.jsonl')}
    images = {r['event_id']: r for r in read(memory / 'representative_images.jsonl')}
    with np.load(memory / 'visual_embeddings.npz', allow_pickle=False) as data:
        assert data['event_ids'].tolist() == [e['event_id'] for e in events]
        vectors = dict(zip(data['event_ids'].tolist(), data['vectors']))
    return protocol, qs, memory, events, vectors, dialogues, images


def native_answer(output, index, q, memory, events, vectors, dialogues, images, encode, allow_api=True):
    """The source core and rewrite recovery are invoked directly, not reimplemented."""
    emap = {e['event_id']: e for e in events}
    directory = Path(output) / 'llm_cache' / f'Q{index:04d}'
    chat = AuditedChat(directory, allow_api=allow_api)
    calls, recovery, image_order = [], [], []
    rewrite_count = 0
    def llm(role, prompt):
        nonlocal rewrite_count, image_order
        paths = q['query_image_paths'] if role in ('fast', 'answer') else []
        actual_role = role
        if role == 'answer':
            evidence = json.loads(prompt.rsplit('\n', 1)[-1])
            ids = [e['event_id'] for e in evidence['memories']]
            if ids:
                prompt, paths, image_order = hip_original.reader_payload(q,
                    {'route': 'detailed', 'selected_event_ids': ids}, emap, images, dialogues, memory)
                actual_role = 'answer_with_representatives'
        raw = chat(actual_role, prompt, paths)
        calls.append(chat.last_call_id)
        if role == 'rewrite':
            rewrite_count += 1
            if rewrite_count == 3:
                raw = hip_api.repaired_rewrite(chat, raw, hip_core.query_prompt(q), calls, recovery)
        return raw
    trace = hip_core.answer_question(q, events, vectors, dialogues, llm, encode)
    final = json.loads((directory / (calls[-1] + '.json')).read_text())
    out = final['response']; usage = out.get('usage') or {}
    trace.update(point=q['point'], question=q['question'], raw_response=trace['answer'],
                 answer_model=MODEL, backbone='gpt5', model=MODEL,
                 llm_calls=[str((directory / (c + '.json')).relative_to(output)) for c in calls],
                 query_format_recovery=recovery, reader_image_order=image_order,
                 historical_image_reads=sum(r['kind'] == 'historical_representative' for r in image_order),
                 hit_token_limit=out['finish_reason'] == 'length', input_tokens=usage.get('prompt_tokens'),
                 generated_tokens=usage.get('completion_tokens'), source_mcq_correct=190,
                 payload_sha256=final['request_sha256'], request_sha256=final['request_sha256'])
    return trace


def run_hippomm(output, workers=4, limit=0):
    output = Path(output)
    lock = (output / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _, qs, memory, events, vectors, dialogues, images = load_hippo(output)
    import torch
    torch.set_num_threads(8)
    encoder = hip_base.ClipText('cpu'); mutex = threading.Lock()
    def encode(query):
        with mutex:
            return encoder(query)
    path = output / 'responses.jsonl'
    old = read(path) if path.exists() else []
    done = {r['id'] for r in old}
    assert len(done) == len(old) and done <= {q['id'] for q in qs}
    pending = [(i, q) for i, q in enumerate(qs, 1) if q['id'] not in done]
    if limit:
        pending = pending[:limit]
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(native_answer, output, i, q, memory, events, vectors,
                               dialogues, images, encode): q['id'] for i, q in pending}
        for future in as_completed(futures):
            qid = futures[future]
            try:
                row = future.result()
                append(path, row); done.add(qid)
                print(json.dumps({'method': 'hippomm', 'completed': len(done), 'total': 371,
                                  'id': qid, 'route': row['route'], 'calls': len(row['llm_calls'])}), flush=True)
            except Exception as exc:
                error = {'id': qid, 'type': type(exc).__name__, 'error': str(exc)}
                errors.append(error); append(output / 'question_errors.jsonl', error)
                print(json.dumps(error), flush=True)
                if getattr(exc, 'kind', '') in ('auth', 'quota', 'model_mismatch', 'protocol'):
                    for pending_future in futures:
                        pending_future.cancel()
                    raise
    status = {'status': 'generated' if len(done) == 371 else 'partial',
              'questions': 371, 'answered': len(done), 'errors_this_pass': errors,
              'responses_sha256': sha(path) if path.exists() else None}
    (output / 'generation_status.json').write_text(json.dumps(status, indent=2) + '\n')
    if len(done) == 371:
        audit_hippomm(output)
        collect_hippomm(output)
    return status


def audit_hippomm(output):
    output = Path(output)
    protocol, qs, memory, events, vectors, dialogues, images = load_hippo(output)
    responses = {r['id']: r for r in read(output / 'responses.jsonl')}
    assert len(responses) == 371 and set(responses) == {q['id'] for q in qs}
    import torch
    torch.set_num_threads(8)
    encoder = hip_base.ClipText('cpu')
    calls = set(); recovery_ids = []; roles = {}
    for i, q in enumerate(qs, 1):
        replay = native_answer(output, i, q, memory, events, vectors, dialogues, images, encoder, allow_api=False)
        assert replay == responses[q['id']], 'Native replay differs: ' + q['id']
        if replay['query_format_recovery']:
            recovery_ids.append(q['id'])
        for name in replay['llm_calls']:
            if name in calls:
                continue
            calls.add(name)
            call = json.loads((output / name).read_text())
            req = call['request']; roles[req['role']] = roles.get(req['role'], 0) + 1
            successes = [a for a in read((output / name).parent / 'api_attempts.jsonl')
                         if a['call_id'] == call['call_id'] and a['status'] == 'success']
            assert len(successes) == 1 and successes[0]['response_id'] == call['response']['response_id']
            assert successes[0]['request_sha256'] == call['request_sha256']
    audit = {'status': 'complete', 'questions': 371, 'native_full_replay': True,
             'source_mcq_correct': 190, 'answer_model': MODEL, 'unique_api_calls': len(calls),
             'roles': roles, 'format_recovery_ids': recovery_ids,
             'responses_sha256': sha(output / 'responses.jsonl'),
             'source_sha256': {str(output / n): sha(output / n) for n in sorted(calls)},
             'original_memory_and_source_hashes_verified': True, 'gold_accessed': False}
    dump(output / 'validation_final.json', audit)
    dump(output / 'completion.json', {'status': 'complete', 'questions': 371, 'answered': 371,
        'responses_sha256': audit['responses_sha256'], 'format_recovery_ids': recovery_ids,
        'token_limit_hits': sum(r['hit_token_limit'] for r in responses.values()),
        'empty_answers': sum(not r['raw_response'] for r in responses.values()),
        'validation_sha256': sha(output / 'validation_final.json')})
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_sha256'}), flush=True)


def collect_hippomm(output):
    output = Path(output)
    seal = json.loads((output / 'completion.json').read_text())
    assert seal['status'] == 'complete' and seal['answered'] == 371
    assert sha(output / 'responses.jsonl') == seal['responses_sha256']
    assert sha(output / 'validation_final.json') == seal['validation_sha256']
    gold_path = OPEN.parent / 'gold.jsonl'
    gold = {r['id']: r for r in read(gold_path)}
    records = {r['id']: r for r in read(output / 'responses.jsonl')}
    assert len(gold) == len(records) == 371 and set(gold) == set(records)
    result = []
    for q in questions():
        r = records[q['id']]
        result.append({'id': q['id'], 'task': q['task'], 'point': q['point'],
            'question': q['question'], 'ground_truth': gold[q['id']]['answer'],
            'model_output': r['raw_response'], 'answer_model': MODEL,
            'source_record_id': q['id'], 'source_hit_token_limit': r['hit_token_limit'],
            'source_responses_sha256': seal['responses_sha256'],
            'source_protocol_sha256': sha(output / 'protocol.json'),
            'source_payload_sha256': r['payload_sha256'],
            'source_native_route': r['route'], 'source_format_recovery': bool(r['query_format_recovery'])})
    write(output / 'judge_inputs.jsonl', result)
    files = [output / n for n in ('protocol.json', 'responses.jsonl', 'completion.json',
              'validation_final.json', 'judge_inputs.jsonl', 'questions.jsonl')]
    files += [OPEN, gold_path, Path(__file__)]
    dump(output / 'collection_protocol.json', {'status': 'complete', 'method_id': 'hippomm',
        'display_label': 'HippoMM', 'dataset': 'MemEye Open', 'questions': 371,
        'answer_model': MODEL, 'reference_mcq_correct': 190, 'max_completion_tokens': 2048,
        'raw_final_answers_unchanged': True, 'gold_read_only_after_answers_sealed': True,
        'generic_reader_rerun': False, 'format_recovery_ids': seal['format_recovery_ids'],
        'source_sha256': {str(p): sha(p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare-clip', 'prepare-hippomm', 'run-hippomm', 'audit-hippomm', 'collect-hippomm'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    output = args.output or DEFAULT / ('clip_image_dialogue' if args.phase == 'prepare-clip' else 'hippomm')
    if args.phase == 'prepare-clip': prepare_clip(output)
    elif args.phase == 'prepare-hippomm': prepare_hippomm(output)
    elif args.phase == 'run-hippomm': run_hippomm(output, args.workers, args.limit)
    elif args.phase == 'audit-hippomm': audit_hippomm(output)
    else: collect_hippomm(output)


if __name__ == '__main__':
    main()

