"""Open MemEye queries against the pinned CLIP and HippoMM Qwen memories.

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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]

from scripts.evaluation import mem_gallery_master_reader as runtime
from scripts.memory import clip_image_memory as clip_base
from scripts.memory import hippomm as hip_base, hippomm_core as hip_core
from scripts.memory import hippomm_original as hip_original

OPEN = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
DEFAULT = ROOT / 'results/memeye_qwen_other_open_20261105/visual_methods'
CLIP_SOURCE = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue'
HIPPO_SOURCE = ROOT / 'results/hippomm_original_controlled_20261104/memeye'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'


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
    correct = sum(responses[q['id']]['predicted_choice'] == q['gold_choice'] for q in frozen)
    assert correct == 193
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
    for q, vector, length in zip(qs, query_vectors, lengths):
        retrieved = clip_base.rank_images(q, candidates, vectors, vector, 5)
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
                     'retrieval_trace': trace, 'max_new_tokens': 512,
                     'image_max_pixels': 1638400,
                     **payload([{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])})
        traces.append(trace)
    write(output / 'reader_manifest.jsonl', rows)
    write(output / 'retrieval.jsonl', traces)
    files = [OPEN, Path(__file__), Path(clip_base.__file__), vectors_file,
             source / 'image_candidates.jsonl', CLIP_SOURCE / 'protocol.json',
             CLIP_SOURCE / 'qa_manifest.jsonl', CLIP_SOURCE / 'qa_responses.jsonl']
    files.extend(Path(p) for p in protocol['source_dialogue_files'])
    files.extend(p for p in checkpoint.iterdir() if p.is_file())
    dump(output / 'protocol.json', {'method_id': 'clip_image_dialogue', 'display_label': 'VisRAG',
        'implementation_note': 'Screenshot label; actual CLIP image + dialogue route, not official VisRAG.',
        'dataset': 'MemEye Open', 'answer_model': MODEL, 'questions': 371, 'top_k': 5,
        'source_mcq_correct': correct, 'source_mcq_root': str(CLIP_SOURCE),
        'query_policy': 'True Open question text only; no query caption, choices, or gold.',
        'memory_policy': 'Frozen 433 task/image candidates, 438 image-associated historical rounds.',
        'query_truncation': {'limit': limit, 'count': sum(n > limit for n in lengths)},
        'max_new_tokens': 512, 'image_max_pixels': 1638400, 'seed': 42,
        'source_sha256': {str(p): sha(p) for p in files},
        'artifact_sha256': {n: sha(output / n) for n in ('reader_manifest.jsonl', 'retrieval.jsonl')}})
    print(json.dumps({'method': 'clip_image_dialogue', 'prepared': len(rows),
                      'original_mcq_correct': correct, 'query_truncated': sum(n > limit for n in lengths)}), flush=True)


def prepare_hippomm(output=DEFAULT / 'hippomm'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    qs = questions()
    source_online = HIPPO_SOURCE / 'online'
    source_memory = HIPPO_SOURCE / 'offline'
    # The standalone source per-question file records the exact source accuracy.
    source_rows = read(source_online / 'per_question.jsonl')
    assert len(source_rows) == 371 and sum(r['accuracy'] for r in source_rows) == 179
    original_qs = read(source_online / 'questions.jsonl')
    original_responses = read(source_online / 'qa_responses.jsonl')
    assert {r['id'] for r in original_qs} == {r['id'] for r in original_responses} == {q['id'] for q in qs}
    source_contract = json.loads((source_online / 'protocol.json').read_text())['contract']
    baseline_contract = json.loads((Path(source_contract['baseline_dir']) / 'online/protocol.json').read_text())['contract']
    events = read(source_memory / 'events.jsonl')
    images = read(source_memory / 'representative_images.jsonl')
    assert len(events) == len(images) == 250
    files = [OPEN, Path(__file__), Path(hip_base.__file__), Path(hip_core.__file__),
             Path(hip_original.__file__), Path(runtime.__file__),
             source_online / 'protocol.json', source_online / 'summary.json',
             source_online / 'questions.jsonl', source_online / 'qa_responses.jsonl',
             source_online / 'per_question.jsonl',
             Path(source_contract['baseline_dir']) / 'online/protocol.json']
    files += [source_memory / name for name in ('memory.json', 'events.jsonl', 'dialogues.jsonl',
              'image_mapping.jsonl', 'visual_embeddings.npz', 'representative_images.jsonl')]
    files += [source_memory / img['file'] for img in images]
    write(output / 'questions.jsonl', qs)
    dump(output / 'protocol.json', {'method_id': 'hippomm', 'dataset': 'MemEye Open',
        'answer_model': MODEL, 'questions': 371, 'source_mcq_correct': 179,
        'source_mcq_root': str(source_online), 'memory_dir': str(source_memory),
        'source_mcq_condition': source_contract['method'], 'top_k': 5,
        'confidence_threshold': baseline_contract['confidence_threshold'],
        'fallback_threshold': baseline_contract['fallback_threshold'],
        'fallback_score': baseline_contract['fallback_score'],
        'max_new_tokens': 1024, 'context_budget': 120000, 'seed': 42,
        'image_max_pixels': 1638400,
        'retrieval_policy': 'Fresh Open question; same-task summary fast path; 2-5 word rewrite; CLIP and per-event caption fallback; Top5.',
        'answer_policy': 'Fast natural answer when confidence > .75; otherwise selected original representative images and native event contexts; original query images in fast/final answer only.',
        'gold_options_access': False, 'source_sha256': {str(p): sha(p) for p in files},
        'questions_sha256': sha(output / 'questions.jsonl')})
    print(json.dumps({'method': 'hippomm', 'prepared': len(qs), 'original_mcq_correct': 179}), flush=True)


class CachedChat:
    """Request-keyed callback adapter; each raw model output is persisted."""
    def __init__(self, output, generate, protocol):
        self.output, self.generate, self.protocol = Path(output), generate, protocol
        self.directory = self.output / 'llm_cache'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def __call__(self, role, prompt, paths=()):
        messages = [{'role': 'system', 'content': runtime.READER_SYSTEM},
                    {'role': 'user', 'content': [{'type': 'image', 'path': str(p)} for p in paths]
                     + [{'type': 'text', 'text': prompt}]}]
        request = {'role': role, 'max_new_tokens': 1024, **payload(messages)}
        key = runtime.digest(request)
        path = self.directory / (key + '.json')
        if path.exists():
            row = json.loads(path.read_text())
            assert row['request'] == request
        else:
            result = self.generate(request['reader_messages'], max_new_tokens=1024)
            if result['input_tokens'] + 1024 > self.protocol['context_budget']:
                raise ValueError('HippoMM input exceeded its original context budget')
            row = {'request_sha256': key, 'request': request, 'response': result}
            dump(path, row)
        self.calls.append(key)
        # Raw answers are preserved, including an empty or token-limited final
        # response. Structured stages retain the original bounded validation.
        return row['response']['raw_response']


def run_hippomm_open(generate, output=DEFAULT / 'hippomm', limit=0):
    """generate(messages, max_new_tokens=1024) returns native Qwen response metadata."""
    import numpy as np
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen HippoMM source changed: ' + path)
    memory = Path(protocol['memory_dir'])
    qs = read(output / 'questions.jsonl')
    assert sha(output / 'questions.jsonl') == protocol['questions_sha256']
    events = read(memory / 'events.jsonl')
    event_map = {e['event_id']: e for e in events}
    dialogues = {r['id']: r for r in read(memory / 'dialogues.jsonl')}
    images = {r['event_id']: r for r in read(memory / 'representative_images.jsonl')}
    with np.load(memory / 'visual_embeddings.npz') as archive:
        vectors = {str(eid): vector.copy() for eid, vector in
                   zip(archive['event_ids'].tolist(), archive['vectors'])}
    encoder = hip_base.ClipText('cpu')
    chat = CachedChat(output, generate, protocol)
    path = output / 'responses.jsonl'
    old = read(path) if path.exists() else []
    done = {r['id'] for r in old}
    assert len(done) == len(old) and done <= {q['id'] for q in qs}
    pending = [q for q in qs if q['id'] not in done]
    if limit:
        pending = pending[:limit]
    for q in pending:
        started = time.perf_counter()
        chat.calls = []
        local = [e for e in events if e['task'] == q['task']]
        try:
            fast_raw = chat('fast', hip_core.fast_prompt(q, local), q['query_image_paths'])
            try:
                fast = hip_core.parse_fast(fast_raw, [e['event_id'] for e in local])
                fast_error = None
            except (ValueError, TypeError) as exc:
                fast = {'answer': None, 'confidence': 0., 'evidence_event_ids': []}
                fast_error = str(exc)
            trace = {'id': q['id'], 'task': q['task'], 'point': q['point'], 'question': q['question'],
                     'fast': fast, 'fast_parse_error': fast_error,
                     'fast_context_event_ids': [e['event_id'] for e in local],
                     'fast_context_round_ids': list(dict.fromkeys(mid for e in local for mid in e['source_round_ids']))}
            if fast['answer'] is not None and fast['confidence'] > protocol['confidence_threshold']:
                answer = fast['answer']
                trace.update(route='fast', selected_event_ids=fast['evidence_event_ids'],
                             context_round_ids=trace['fast_context_round_ids'],
                             historical_image_reads=0, reader_image_order=[])
            else:
                llm = lambda role, prompt: chat(role, prompt)
                query = hip_core.structured_call(llm, 'rewrite', hip_core.query_prompt(q), hip_core.parse_query)
                vector, encoding = encoder(query)
                ranked, audit = hip_core.detailed_search(q, local, vectors, vector, llm,
                    top_k=5, fallback_threshold=protocol['fallback_threshold'],
                    fallback_score=protocol['fallback_score'])
                selected = [r['event_id'] for r in ranked]
                trace.update(route='detailed', visual_query=query, query_encoding=encoding,
                             selected_event_ids=selected, detail_candidates=audit, ranked_candidates=ranked,
                             context_round_ids=list(dict.fromkeys(mid for eid in selected for mid in event_map[eid]['source_round_ids'])))
                if selected:
                    prompt, paths, order = hip_original.reader_payload(q, trace, event_map, images, dialogues, memory)
                    answer = chat('answer_with_representatives', prompt, paths).strip()
                    trace.update(historical_image_reads=len(selected), reader_image_order=order)
                else:
                    prompt = ('Answer using only the retrieved memory evidence. Original historical images are '
                              'unavailable; do not invent visual details missing from captions or dialogue. '
                              'Respect dates and source image IDs; visually redundant aliases are not identical '
                              'images. If evidence is absent, acknowledge it. \nReturn a concise answer.\n'
                              + json.dumps({'question': q['question'], 'memories': []}, ensure_ascii=False))
                    answer = chat('answer', prompt, q['query_image_paths']).strip()
                    trace.update(historical_image_reads=0, reader_image_order=[])
            cache = json.loads((chat.directory / (chat.calls[-1] + '.json')).read_text())
            trace.update(raw_response=answer, answer=answer, answer_model=MODEL,
                         input_tokens=cache['response']['input_tokens'],
                         generated_tokens=cache['response']['generated_tokens'],
                         hit_token_limit=cache['response']['hit_token_limit'],
                         payload_sha256=cache['request']['payload_sha256'],
                         llm_calls=chat.calls, source_mcq_correct=179,
                         elapsed_seconds=time.perf_counter() - started)
            append(path, trace)
            done.add(q['id'])
            print(json.dumps({'method': 'hippomm', 'completed': len(done), 'total': 371,
                              'id': q['id'], 'route': trace['route'], 'calls': len(chat.calls)}), flush=True)
        except (ValueError, TypeError) as exc:
            append(output / 'question_errors.jsonl', {'id': q['id'], 'error': str(exc), 'llm_calls': chat.calls})
            print(json.dumps({'method': 'hippomm', 'id': q['id'], 'error': str(exc)}), flush=True)
    records = read(path) if path.exists() else []
    status = {'status': 'complete' if len(records) == 371 else 'partial', 'questions': 371,
              'answered': len(records), 'responses_sha256': sha(path) if path.exists() else None,
              'empty_answers': sum(not r['raw_response'] for r in records),
              'token_limit_hits': sum(r['hit_token_limit'] for r in records)}
    (output / 'completion.json').write_text(json.dumps(status, indent=2) + '\n')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare-clip', 'prepare-hippomm'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.phase == 'prepare-clip':
        prepare_clip(args.output or DEFAULT / 'clip_image_dialogue', args.device)
    else:
        prepare_hippomm(args.output or DEFAULT / 'hippomm')


if __name__ == '__main__':
    main()
