"""Fresh MemGallery seed-1042 visual-method replication.

Keep the screenshot's CLIP-image/dialogue and paired HippoMM procedures. Only
deterministic CLIP features are imported; every Qwen output is generated inside
this run. This module never changes signed source files or seed-42 artifacts.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for _key, _value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                     'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ.setdefault(_key, _value)

import numpy as np
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation import mem_gallery_clip_image_dialogue as clip
from scripts.memory import hippomm as hip, hippomm_core as core
from scripts.memory import hippomm_original as original
from scripts.memory.clip_image_memory import rank_images

SEED = 1042
QUESTIONS = 1711
DEFAULT = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
SOURCE_INPUTS = ROOT / 'results/mem_gallery_master_answer_20261021/inputs'
SOURCE_CLIP = ROOT / 'results/mem_gallery_clip_image_dialogue_qwen_20261102'
SOURCE_HIP_BASE = ROOT / 'results/hippomm_short_ids_20261104/mem_gallery'
SOURCE_HIP = ROOT / 'results/hippomm_original_controlled_20261104/mem_gallery'
sha, digest, read, dump = common.sha, common.digest, common.read_lines, common.dump


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def frozen_rows(path, values):
    path = Path(path)
    data = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
    if path.exists() and path.read_text() != data:
        raise ValueError('Frozen rows changed: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)


def frozen_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Frozen JSON changed: ' + str(path))
    dump(path, value)


def out(args):
    return args.run_root / 'visual_methods' / args.method


def implementation():
    files = [Path(__file__), Path(common.__file__), Path(clip.__file__), Path(hip.__file__),
             Path(core.__file__), Path(original.__file__), Path(hip.inputs.__file__),
             ROOT / 'scripts/memory/clip_image_memory.py',
             ROOT / 'scripts/graph/visual_image_encoder.py']
    return {str(p.resolve()): sha(p) for p in files}


def source_audit(args):
    """Read-only source validation; no shared-input/model/CUDA dependency."""
    clip_p = json.loads((SOURCE_CLIP / 'protocol.json').read_text())
    clip_s = json.loads((SOURCE_CLIP / 'summary.json').read_text())
    hip_s = json.loads((SOURCE_HIP / 'online/summary.json').read_text())
    hip_p = json.loads((SOURCE_HIP_BASE / 'online/protocol.json').read_text())['contract']
    clip_q, clip_a = read(SOURCE_CLIP / 'qa_manifest.jsonl'), read(SOURCE_CLIP / 'qa_responses.jsonl')
    hip_q, hip_a = read(SOURCE_HIP / 'online/questions.jsonl'), read(SOURCE_HIP / 'online/qa_responses.jsonl')
    gold = common.by_id(read(SOURCE_INPUTS / 'gold.jsonl'))
    if len(gold) != QUESTIONS or len(clip_q) != QUESTIONS or len(hip_q) != QUESTIONS:
        raise ValueError('Seed-42 cohort differs')
    ca, ha = common.by_id(clip_a), common.by_id(hip_a)
    if set(ca) != set(gold) or set(ha) != set(gold):
        raise ValueError('Seed-42 answer IDs differ')
    clip_correct = sum(common.score_answer(ca[i]['raw_response'], gold[i]['answer'])['em'] for i in gold)
    hip_correct = sum(common.score_answer(ha[i]['answer'], gold[i]['answer'])['em'] for i in gold)
    if clip_correct != 699 or hip_correct != 681 or clip_s['overall']['exact_match_correct'] != 699:
        raise ValueError('Sources do not reproduce the screenshot')
    if abs(hip_s['em'] - 681 / QUESTIONS) > 1e-12:
        raise ValueError('HippoMM source EM differs')
    if (clip_p['seed'] != 42 or clip_p['do_sample'] or clip_p['max_new_tokens'] != 256
            or clip_p['batch_size'] != 2 or clip_p['max_padded_input_tokens'] != 24000
            or hip_p['max_new_tokens'] != 1024 or hip_p['top_k'] != 5
            or hip_p['confidence_threshold'] != .75 or hip_p['fallback_threshold'] != .4
            or hip_p['fallback_score'] != .6 or hip_p['structured_output_attempts'] != 3):
        raise ValueError('Native generation/retrieval settings differ')
    for name, details in clip_p['reader_model']['files'].items():
        if sha(common.MODEL / name) != details['sha256']:
            raise ValueError('Qwen source checkpoint changed: ' + name)
    source_files = [SOURCE_CLIP / n for n in ('protocol.json', 'summary.json', 'qa_manifest.jsonl',
                    'qa_responses.jsonl', 'clip_embeddings.npz', 'image_candidates.jsonl')]
    source_files += [SOURCE_HIP / 'online' / n for n in ('protocol.json', 'summary.json', 'questions.jsonl', 'qa_responses.jsonl')]
    source_files += [SOURCE_HIP_BASE / 'offline' / n for n in ('build_contract.json', 'events.jsonl',
                      'dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz', 'memory.json')]
    source_files += [SOURCE_HIP_BASE / 'online/protocol.json', SOURCE_INPUTS / 'gold.jsonl']
    value = {'status': 'complete', 'seed': SEED, 'reference_seed': 42, 'judge_seed': 42,
        'questions': QUESTIONS, 'method': args.method,
        'reference_clip': {'correct': 699, 'em': 699 / QUESTIONS, 'f1': clip_s['overall']['f1'],
                           'bleu_1': clip_s['overall']['bleu_1'], 'source': str(SOURCE_CLIP)},
        'reference_hippomm': {'correct': 681, 'em': 681 / QUESTIONS, 'f1': hip_s['f1'],
            'source': str(SOURCE_HIP / 'online'), 'fast_answers': hip_s['fast_answers_reused'],
            'nonempty_detail_answers': hip_s['detailed_answers_regenerated'],
            'empty_detail_answers': hip_s['empty_retrieval_answers_reused']},
        'source_sha256': {str(p): sha(p) for p in source_files},
        'model_provenance': clip_p['reader_model'],
        'implementation_sha256': implementation(), 'old_llm_outputs_imported': 0,
        'decoding_note': 'Native greedy decoding is retained. A seed change need not change an answer.'}
    frozen_json(out(args) / 'source42_audit.json', value)
    print(json.dumps({'phase': 'source-audit', 'method': args.method, 'status': 'complete'}), flush=True)
    return value


def verify_source(args):
    target = out(args) / 'source42_audit.json'
    saved = json.loads(target.read_text())
    if saved['implementation_sha256'] != implementation():
        raise ValueError('Visual implementation changed after preparation')
    for path, expected in saved['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Seed-42 source changed: ' + path)
    return saved


def new_inputs(args):
    marker = args.run_root / 'caption_methods/stages/shared-inputs.json'
    if not marker.is_file():
        raise FileNotFoundError('Shared-input stage is not ready: ' + str(marker))
    source = args.run_root / 'inputs'
    names = ('memories.jsonl', 'questions.jsonl', 'gold.jsonl', 'protocol.json', 'artifact_manifest.json')
    for name in names:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    hip.inputs.gallery_files(source, ['memories.jsonl', 'questions.jsonl', 'protocol.json'])
    memories, questions = read(source / 'memories.jsonl'), read(source / 'questions.jsonl')
    old_m = common.by_id(read(SOURCE_INPUTS / 'memories.jsonl'))
    old_q = common.by_id(read(SOURCE_INPUTS / 'questions.jsonl'))
    if len(memories) != 3962 or len(questions) != QUESTIONS:
        raise ValueError('Wrong MemGallery input coverage')
    for memory in memories:
        for key in ('id', 'task', 'round', 'session_id', 'timestamp', 'dialogue', 'is_visual',
                    'image_path', 'image_id', 'image_sha256', 'insertion_index'):
            if memory[key] != old_m[memory['id']][key]:
                raise ValueError('Historical source field differs: ' + key)
        if memory['is_visual'] and not memory['caption'].strip():
            raise ValueError('Missing fresh historical caption')
    for question in questions:
        for key in ('id', 'task', 'point', 'question', 'query_image_paths'):
            if question[key] != old_q[question['id']][key]:
                raise ValueError('Canonical question changed: ' + key)
    if sha(source / 'gold.jsonl') != sha(SOURCE_INPUTS / 'gold.jsonl'):
        raise ValueError('Official gold changed')
    return source, memories, questions, {str(source / n): sha(source / n) for n in names}


def prepare(args):
    verify_source(args)
    source, memories, questions, hashes = new_inputs(args)
    target = out(args)
    contract = {'dataset': 'MemGallery', 'method': args.method, 'questions': QUESTIONS,
        'seed': SEED, 'judge_seed': 42, 'do_sample': False, 'device': args.device,
        'model_repo': 'Qwen/Qwen3-VL-8B-Instruct', 'model_revision': common.MODEL_REVISION,
        'dtype': 'bfloat16', 'attention': 'sdpa', 'image_max_pixels': common.QUERY_IMAGE_MAX_PIXELS,
        'new_inputs_sha256': hashes, 'implementation_sha256': implementation(),
        'source_audit_sha256': sha(target / 'source42_audit.json'),
        'old_llm_answer_or_summary_cache_imported': False,
        'clip_policy': 'Reuse audited deterministic image/query embeddings; native stable image Top5.',
        'caption_policy': 'Use the fresh seed-1042 shared inputs, including historical captions.',
        'gold_policy': 'Gold hashes tracked; gold content parsed only during source-audit/analyze.'}
    common.ensure_protocol(target / 'stability_protocol.json', contract)
    if args.method == 'clip_image_dialogue':
        prepare_clip(args, memories, questions, source, hashes)
    else:
        prepare_hip(args, source)
    frozen_json(target / 'prepared.json', {'status': 'complete', 'seed': SEED,
        'method': args.method, 'questions': QUESTIONS, 'stability_protocol_sha256': sha(target / 'stability_protocol.json')})
    print(json.dumps({'phase': 'prepare', 'method': args.method, 'status': 'complete'}), flush=True)


def verify(args):
    verify_source(args)
    p = json.loads((out(args) / 'stability_protocol.json').read_text())
    if digest(p['contract']) != p['experiment_signature']:
        raise ValueError('Stability protocol changed')
    if p['contract']['device'] != args.device or p['contract']['seed'] != SEED:
        raise ValueError('Wrong visual runtime device/seed')
    for path, expected in p['contract']['new_inputs_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Fresh input changed: ' + path)
    return p


def prepare_clip(args, memories, questions, source, hashes):
    old = json.loads((SOURCE_CLIP / 'protocol.json').read_text())
    candidates = clip.candidates_from(memories)
    source_candidates = read(SOURCE_CLIP / 'image_candidates.jsonl')
    if candidates != source_candidates:
        raise ValueError('CLIP historical image/dialogue candidates changed')
    vectors = np.load(SOURCE_CLIP / 'clip_embeddings.npz', allow_pickle=False)
    if (list(vectors['image_ids']) != [c['image_id'] for c in candidates]
            or list(vectors['query_ids']) != [q['id'] for q in questions]):
        raise ValueError('Deterministic CLIP vector identity changed')
    old_manifest = common.by_id(read(SOURCE_CLIP / 'qa_manifest.jsonl'))
    manifest = []
    for q, vector in zip(questions, vectors['query_vectors']):
        q = {k: q[k] for k in ('id', 'task', 'point', 'question', 'query_image_paths')}
        selected = rank_images(q, candidates, vectors['image_vectors'], vector, 5)
        messages = clip.messages_for(q, selected)
        paths = [p['path'] for p in messages[1]['content'] if p['type'] == 'image']
        image_hashes = {p: sha(p) for p in paths}
        old_q = old_manifest[q['id']]
        item = dict(**q, retrieval_query=q['question'], query_token_count=old_q['query_token_count'],
            query_truncated=old_q['query_truncated'], retrieved_images=selected,
            top_memory_ids=list(dict.fromkeys(mid for c in selected for mid in c['source_memory_ids'])),
            reader_messages=messages, reader_image_paths=paths, reader_image_hashes=image_hashes,
            payload_sha256=digest(dict(messages=messages, image_hashes=image_hashes)))
        if item != old_q:
            raise ValueError('CLIP frozen retrieval or original multimodal payload changed: ' + q['id'])
        manifest.append(item)
    target = out(args)
    frozen_rows(target / 'qa_manifest.jsonl', manifest)
    frozen_rows(target / 'image_candidates.jsonl', candidates)
    # The original verification/analyzer consumes this exact protocol schema.
    protocol = copy.deepcopy(old)
    protocol.pop('experiment_signature')
    protocol.update(seed=SEED, device=args.device, source=str(source.resolve()),
        frozen_clip_features_source=str(SOURCE_CLIP / 'clip_embeddings.npz'),
        fresh_llm_answers=True, implementation_sha256=implementation())
    files = {str(target / 'qa_manifest.jsonl'): sha(target / 'qa_manifest.jsonl'),
             str(target / 'image_candidates.jsonl'): sha(target / 'image_candidates.jsonl'),
             str(SOURCE_CLIP / 'clip_embeddings.npz'): sha(SOURCE_CLIP / 'clip_embeddings.npz')}
    files.update(hashes)
    files.update({path: expected for q in manifest for path, expected in q['reader_image_hashes'].items()})
    protocol['input_hashes'] = files
    protocol['experiment_signature'] = digest(protocol)
    frozen_json(target / 'protocol.json', protocol)
    clip.verify(target)
    frozen_json(target / 'input_audit.json', {'status': 'complete', 'questions': QUESTIONS,
        'native_payloads_equal_seed42': QUESTIONS, 'old_answers_imported': 0,
        'source_embedding_sha256': sha(SOURCE_CLIP / 'clip_embeddings.npz'),
        'manifest_sha256': sha(target / 'qa_manifest.jsonl')})


def hip_settings(args):
    target = out(args)
    return SimpleNamespace(dataset='mem_gallery', task=None, question_ids=None, limit_questions=0,
        top_k=5, device=args.device, hippomm_gallery_inputs=args.run_root / 'inputs',
        hippomm_max_new_tokens=1024, hippomm_context_budget=120000,
        hippomm_duplicate_threshold=.9, hippomm_confidence_threshold=.75,
        hippomm_fallback_threshold=.4, hippomm_fallback_score=.6,
        hippomm_baseline_root=target / 'baseline', hippomm_stage='build')


def patch_native_seed():
    """In-process adapter only; original signed files remain byte-identical."""
    common.SEED = SEED
    random.seed(SEED)
    np.random.seed(SEED)
    base_contract = hip.contract_at
    base_hashes = hip.implementation_hashes

    def seeded_contract(path, contract):
        return base_contract(path, {**contract, 'seed': SEED, 'do_sample': False,
                                    'seed_wrapper_sha256': sha(__file__), 'judge_seed': 42})

    hip.contract_at = seeded_contract
    hip.implementation_hashes = lambda: {**base_hashes(), str(Path(__file__)): sha(__file__)}
    hip.LocalChat = SeededLocalChat


def prepare_hip(args, source):
    cfg = hip_settings(args)
    rounds, images, _ = hip.inputs.memory_inputs(cfg)
    cached = np.load(SOURCE_CLIP / 'clip_embeddings.npz', allow_pickle=False)
    by_hash = dict(zip(cached['image_ids'].tolist(), cached['image_vectors']))
    path_to_hash = {image['path']: image['image_sha256'] for image in images}
    events, vectors, mapping = hip.construct_events(rounds, images,
        lambda path: by_hash[path_to_hash[str(path)]], duplicate_threshold=.9)
    old = read(SOURCE_HIP_BASE / 'offline/events.jsonl')
    keys = ('event_id', 'task', 'representative_image', 'source_images', 'source_round_ids')
    if len(events) != 860 or [{k: e[k] for k in keys} for e in events] != [{k: e[k] for k in keys} for e in old]:
        raise ValueError('Native deterministic HippoMM event identity/provenance changed')
    old_mapping = read(SOURCE_HIP_BASE / 'offline/image_mapping.jsonl')
    mkeys = ('task', 'image_ref', 'event_id', 'retained', 'source_round_ids')
    if [{k: m[k] for k in mkeys} for m in mapping] != [{k: m[k] for k in mkeys} for m in old_mapping]:
        raise ValueError('Native duplicate filter membership changed')
    target = out(args) / 'baseline/mem_gallery/offline'
    target.mkdir(parents=True, exist_ok=True)
    hip.code_manifest(target)
    frozen_rows(target / 'unsummarized_events.jsonl', events)
    frozen_rows(target / 'dialogues.jsonl', rounds)
    frozen_rows(target / 'image_mapping.jsonl', mapping)
    vector_path = target / 'visual_embeddings.npz'
    ids, matrix = np.array([e['event_id'] for e in events]), np.stack([vectors[e['event_id']] for e in events]).astype(np.float32)
    if vector_path.exists():
        prior = np.load(vector_path, allow_pickle=False)
        if not np.array_equal(prior['event_ids'], ids) or not np.array_equal(prior['vectors'], matrix):
            raise ValueError('Run-local HippoMM vectors changed')
    else:
        np.savez(vector_path, event_ids=ids, vectors=matrix)
    hip.contract_at(target / 'build_contract.json', {'dataset': 'mem_gallery',
        'method': 'hippomm_image_clip_no_originals', 'duplicate_threshold': .9,
        'max_new_tokens': 1024, 'context_budget': 120000, 'events': len(events),
        'source': str(source), 'native_event_builder': str(Path(hip.__file__)),
        'fresh_summary_policy': 'All event summaries generated in seed-1042 run-local cache.'})
    frozen_json(out(args) / 'event_construction_audit.json', {'status': 'complete',
        'events': len(events), 'distinct_task_images': len(images), 'dialogue_rounds': len(rounds),
        'same_event_membership_as_seed42': True, 'fresh_caption_events': len(events),
        'new_caption_count_different_from_seed42': sum(e['caption'] != o['caption'] for e, o in zip(events, old)),
        'mapping_similarity_max_abs_delta': max(abs(m['similarity_to_representative'] - o['similarity_to_representative']) for m, o in zip(mapping, old_mapping)),
        'old_summaries_imported': 0, 'frozen_clip_source_sha256': sha(SOURCE_CLIP / 'clip_embeddings.npz')})


class SeededLocalChat:
    """Native HippoMM message/context policy, fresh seed-qualified durable calls."""

    def __init__(self, directory, contract, device, token_limit=1024, context_budget=120000):
        self.directory, self.device = Path(directory), device
        self.token_limit, self.context_budget = token_limit, context_budget
        self.runtime, self.last_call_id = None, None
        self.signature = digest({'contract': contract, 'seed': SEED, 'implementation': implementation()})
        self.cache_dir = self.directory / 'llm_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = {}
        attempts = self.directory / 'generation_attempts.jsonl'
        if attempts.exists():
            for record in read(attempts):
                if record['status'] != 'success':
                    continue
                path = self.cache_dir / (record['call_id'] + '.json')
                durable = {k: v for k, v in record.items() if k != 'status'}
                if not path.exists():
                    dump(path, durable)
                elif json.loads(path.read_text()) != durable:
                    raise ValueError('Durable native generation differs from cache')
        for path in sorted(self.cache_dir.glob('call_*.json')):
            row = json.loads(path.read_text())
            key = digest(row['request'])
            if (key in self.index or row['request']['seed'] != SEED
                    or row['request']['experiment_signature'] != self.signature
                    or row['request_sha256'] != key):
                raise ValueError('Duplicate/wrong-contract HippoMM call cache')
            self.index[key] = path.stem

    def __call__(self, role, prompt, image_paths=()):
        paths = [str(p) for p in image_paths]
        request = {'role': role, 'prompt': prompt, 'query_image_paths': paths,
            'image_sha256': {p: sha(p) for p in paths}, 'max_new_tokens': self.token_limit,
            'seed': SEED, 'do_sample': False, 'temperature': 0.,
            'experiment_signature': self.signature}
        key = digest(request)
        call_id = self.index.get(key, f'call_{len(self.index) + 1:06d}')
        path = self.cache_dir / (call_id + '.json')
        reused = path.exists()
        if reused:
            record = json.loads(path.read_text())
            if record['request'] != request:
                raise ValueError('Run-local native call differs')
        else:
            if self.runtime is None:
                common.SEED = SEED
                self.runtime = common.load_runtime(self.device)
                self.runtime[2].update(seed=SEED, do_sample=False, seed_wrapper_sha256=sha(__file__))
                dump(self.directory / ('generation_environment_' + str(os.getpid()) + '.json'), self.runtime[2])
            from PIL import Image
            processor, _, _ = self.runtime
            content = [{'type': 'image'} for _ in paths] + [{'type': 'text', 'text': prompt}]
            rendered = processor.apply_chat_template([
                {'role': 'system', 'content': common.READER_SYSTEM},
                {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            images = []
            try:
                for image_path in paths:
                    with Image.open(image_path) as image:
                        images.append(image.convert('RGB'))
                encoded = processor(text=[rendered], images=images or None, padding=True,
                                    truncation=False, return_tensors='pt')
                length = int(encoded['attention_mask'].sum())
                del encoded
            finally:
                for image in images:
                    image.close()
            if length + self.token_limit > self.context_budget:
                raise ValueError('Native HippoMM context budget exceeded; no truncation')
            started = time.time()
            try:
                common.SEED = SEED
                output = common.generate_batch(self.runtime, [{'system': common.READER_SYSTEM,
                    'user_text': prompt, 'query_image_paths': paths}], self.token_limit)[0]
            except BaseException as exc:
                append(self.directory / 'generation_attempts.jsonl', {'call_id': call_id,
                    'request_sha256': key, 'request': request, 'status': 'error',
                    'error': type(exc).__name__ + ': ' + str(exc), 'at_unix': started})
                raise
            record = {'call_id': call_id, 'request': request, 'response': output,
                      'request_sha256': key, 'at_unix': started}
            # The full completed response survives even if the cache rename fails.
            append(self.directory / 'generation_attempts.jsonl', {**record, 'status': 'success'})
            dump(path, record)
            self.index[key] = call_id
        self.last_call_id = call_id
        append(self.directory / 'llm_calls.jsonl', {'role': role, 'call_id': call_id,
            'request_sha256': key, 'reused': reused, 'input_tokens': record['response']['input_tokens'],
            'generated_tokens': record['response']['generated_tokens']})
        if record['response']['hit_token_limit']:
            raise ValueError('Native HippoMM response reached its token limit; raw output retained')
        return record['response']['raw_response']

    def close(self):
        self.runtime = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_hip(args):
    verify(args)
    cfg = hip_settings(args)
    target = out(args) / 'baseline/mem_gallery/offline'
    if (target / 'build_completion.json').exists():
        hip.verify_memory(target)
        return
    events = read(target / 'unsummarized_events.jsonl')
    rounds = read(target / 'dialogues.jsonl')
    dialogues = common.by_id(rounds)
    contract = json.loads((target / 'build_contract.json').read_text())['contract']
    chat = SeededLocalChat(target, contract, args.device, 1024, 120000)
    try:
        for index, event in enumerate(events, 1):
            event['summary'] = chat('summary', core.summary_prompt(event, dialogues)).strip()
            if not event['summary']:
                raise ValueError('Empty native event summary')
            dump(target / 'build_progress.json', {'events': index, 'total': len(events), 'seed': SEED})
            print(json.dumps({'stage': 'hippo-summary', 'completed': index, 'total': len(events)}), flush=True)
    finally:
        chat.close()
    frozen_rows(target / 'events.jsonl', events)
    memory = {'offline_method': 'hippomm', 'schema': hip.SCHEMA, 'dataset': 'mem_gallery',
        'counts': {'dialogue_rounds': len(rounds), 'image_rounds': sum(bool(r['images']) for r in rounds),
            'distinct_task_images': 1001, 'tasks': sorted({r['task'] for r in rounds}), 'events': len(events)},
        'clip_encoder': {'model': str(common.CLIP), 'feature_dim': 512,
                         'normalization': 'L2', 'dtype': 'float32'},
        'history_image_paths_saved': False, 'image_copies': 0,
        'text_only_policy': 'stored for provenance; no standalone event, no text-only retrieval candidates'}
    frozen_json(target / 'memory.json', memory)
    hip.verify_memory(target)
    frozen_json(target / 'build_completion.json', {'status': 'complete', 'seed': SEED,
        'events': len(events), 'events_sha256': sha(target / 'events.jsonl'),
        'memory_sha256': sha(target / 'memory.json'), 'all_summaries_generated_in_this_run': True})


def seeded_clip_generate(runtime, batch, protocol, directory):
    """Original CLIP reader batching, rendering and EOS rules; actual seed1042."""
    import torch
    from PIL import Image
    processor, model, _ = runtime
    rendered = [processor.apply_chat_template(q['reader_messages'], tokenize=False,
                add_generation_prompt=True) for q in batch]
    images = []
    for q in batch:
        for path in q['reader_image_paths']:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
    try:
        encoded = processor(text=rendered, images=images, padding=True, truncation=False,
                            return_tensors='pt')
    finally:
        for image in images:
            image.close()
    width = int(encoded['input_ids'].shape[1])
    if width + protocol['max_new_tokens'] > model.config.text_config.max_position_embeddings:
        raise ValueError('Reader would truncate')
    if width * len(batch) > protocol['max_padded_input_tokens'] and len(batch) > 1:
        del encoded
        return sum([seeded_clip_generate(runtime, [q], protocol, directory) for q in batch], [])
    if len(encoded['image_grid_thw']) != sum(len(q['reader_image_paths']) for q in batch):
        raise ValueError('Reader image count mismatch')
    lengths, grids = encoded['attention_mask'].sum(dim=1).tolist(), encoded['image_grid_thw'].tolist()
    request = {'ids': [q['id'] for q in batch], 'payload_sha256': [q['payload_sha256'] for q in batch],
               'seed': SEED, 'do_sample': False, 'cap': 256,
               'experiment_signature': protocol['experiment_signature']}
    key = digest(request)
    call_path = directory / 'calls' / (key + '.json')
    if call_path.exists():
        record = json.loads(call_path.read_text())
        if record['request'] != request:
            raise ValueError('CLIP generation cache request changed')
        return record['responses']
    encoded = encoded.to(next(model.parameters()).device)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=256,
                                       pad_token_id=processor.tokenizer.pad_token_id)
    except BaseException as exc:
        append(directory / 'generation_attempts.jsonl', {'status': 'error', 'request': request,
            'request_sha256': key, 'error': type(exc).__name__ + ': ' + str(exc)})
        raise
    elapsed = time.perf_counter() - started
    eos = processor.tokenizer.eos_token_id
    result, offset = [], 0
    for q, sequence, length in zip(batch, generated[:, width:].tolist(), lengths):
        actual = sequence[:sequence.index(eos) + 1] if eos in sequence else sequence
        raw = processor.decode(actual, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        count = len(q['reader_image_paths'])
        result.append(dict(id=q['id'], task=q['task'], point=q['point'], raw_response=raw,
            status='ok' if raw else 'empty_response', input_tokens=length, generated_tokens=len(actual),
            hit_token_limit=len(actual) >= 256 and eos not in actual, reader_image_count=count,
            image_grid_thw=grids[offset:offset + count],
            dialogue_round_count=sum(len(c['source_rounds']) for c in q['retrieved_images']),
            top_memory_ids=q['top_memory_ids'], batch_size=len(batch), batch_elapsed_seconds=elapsed,
            payload_sha256=q['payload_sha256'], experiment_signature=protocol['experiment_signature'],
            seed=SEED, request_sha256=key))
        offset += count
    record = {'request': request, 'request_sha256': key, 'responses': result}
    append(directory / 'generation_attempts.jsonl', {**record, 'status': 'success'})
    dump(call_path, record)
    return result


def run_clip(args):
    verify(args)
    target = out(args)
    protocol, manifest, previous = clip.verify(target)
    done = common.by_id(previous)
    canonical = common.by_id(manifest)
    # Complete a batch that was durably generated before a client/process exit.
    # Recover its original batch outputs before forming any new pending batches.
    attempts = target / 'generation_attempts.jsonl'
    if attempts.exists():
        for record in read(attempts):
            if record['status'] != 'success':
                continue
            durable = {k: v for k, v in record.items() if k != 'status'}
            path = target / 'calls' / (record['request_sha256'] + '.json')
            if not path.exists():
                dump(path, durable)
            elif json.loads(path.read_text()) != durable:
                raise ValueError('Durable CLIP generation differs from cache')
    for path in sorted((target / 'calls').glob('*.json')):
        record = json.loads(path.read_text())
        request = record['request']
        if (digest(request) != path.stem or request['seed'] != SEED
                or request['experiment_signature'] != protocol['experiment_signature']):
            raise ValueError('Recovered CLIP request provenance differs')
        for answer in record['responses']:
            if answer['payload_sha256'] != canonical[answer['id']]['payload_sha256']:
                raise ValueError('Recovered CLIP payload differs')
            if answer['id'] in done:
                if done[answer['id']] != answer:
                    raise ValueError('Conflicting durable CLIP answer')
            else:
                append(target / 'qa_responses.jsonl', answer)
                done[answer['id']] = answer
    for answer in previous:
        cache = json.loads((target / 'calls' / (answer['request_sha256'] + '.json')).read_text())
        if answer not in cache['responses'] or cache['request']['seed'] != SEED:
            raise ValueError('Fresh CLIP answer provenance differs')
    pending = [q for q in manifest if q['id'] not in done]
    if args.limit_run:
        pending = pending[:args.limit_run]
    if not pending:
        return
    common.SEED = SEED
    runtime = common.load_runtime(args.device)
    runtime[2].update(seed=SEED, do_sample=False, experiment_signature=protocol['experiment_signature'])
    dump(target / ('generation_environment_' + str(os.getpid()) + '.json'), runtime[2])
    try:
        for start in range(0, len(pending), 2):
            answers = seeded_clip_generate(runtime, pending[start:start + 2], protocol, target)
            for answer in answers:
                append(target / 'qa_responses.jsonl', answer)
                done[answer['id']] = answer
            dump(target / 'progress.json', {'completed': len(done), 'total': QUESTIONS, 'seed': SEED})
            print(json.dumps({'method': args.method, 'completed': len(done), 'total': QUESTIONS}), flush=True)
    finally:
        del runtime
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def run_hip(args):
    verify(args)
    if args.limit_run:
        raise ValueError('HippoMM native full-cohort pipeline does not accept --limit-run')
    cfg = hip_settings(args)
    cfg.memory_dir = out(args) / 'baseline/mem_gallery/offline'
    if not (cfg.memory_dir / 'build_completion.json').is_file():
        raise ValueError('HippoMM summaries must finish before querying')
    cfg.output = out(args) / 'baseline/mem_gallery/online'
    hip.prepare(cfg)
    hip.run(cfg)
    hip.analyze(cfg)
    cfg.output = out(args) / 'offline'
    original.build(cfg)
    cfg.memory_dir = cfg.output
    cfg.output = out(args) / 'online'
    original.prepare(cfg)
    original.run(cfg)
    original.analyze(cfg)


def audit_calls(directory):
    paths = sorted(Path(directory).glob('llm_cache/call_*.json'))
    attempts = read(Path(directory) / 'generation_attempts.jsonl') if (Path(directory) / 'generation_attempts.jsonl').exists() else []
    successful = {}
    for record in attempts:
        if record['status'] == 'success':
            if record['request_sha256'] in successful:
                raise ValueError('Duplicate completed native generation for one request')
            successful[record['request_sha256']] = {k: v for k, v in record.items() if k != 'status'}
    for path in paths:
        record = json.loads(path.read_text())
        request, response = record['request'], record['response']
        if (request['seed'] != SEED or request['do_sample'] or request['temperature'] != 0
                or record['request_sha256'] != digest(request)):
            raise ValueError('Wrong native call seed/metadata')
        if successful.get(record['request_sha256']) != record:
            raise ValueError('Native generation cache has no matching successful raw attempt')
        if response['hit_token_limit']:
            raise ValueError('Unresolved native generation failure')
        for image_path, expected in request['image_sha256'].items():
            if sha(image_path) != expected:
                raise ValueError('Native image bytes changed')
    return len(paths)


def analyze(args):
    verify(args)
    target = out(args)
    if args.method == 'clip_image_dialogue':
        clip.analyze(SimpleNamespace(output=target))
        responses = read(target / 'qa_responses.jsonl')
        for row in responses:
            cache = json.loads((target / 'calls' / (row['request_sha256'] + '.json')).read_text())
            if row not in cache['responses'] or cache['request']['seed'] != SEED:
                raise ValueError('CLIP answer cache audit failed')
        summary = json.loads((target / 'summary.json').read_text())
        summary.update(seed=SEED, judge_seed=42)
        dump(target / 'summary.json', summary)
        answer_path, question_path = target / 'qa_responses.jsonl', target / 'qa_manifest.jsonl'
        prediction_field = 'raw_response'
        native_call_count = len(list((target / 'calls').glob('*.json')))
    else:
        cfg = hip_settings(args)
        cfg.memory_dir, cfg.output = target / 'offline', target / 'online'
        original.analyze(cfg)
        native_call_count = sum(audit_calls(p) for p in [target / 'baseline/mem_gallery/offline',
                                target / 'baseline/mem_gallery/online', target / 'online'])
        responses = read(target / 'online/qa_responses.jsonl')
        baseline = common.by_id(read(target / 'baseline/mem_gallery/online/qa_responses.jsonl'))
        for row in responses:
            old = baseline[row['id']]
            original.check_trace(row, old)
            if row.get('baseline_llm_calls') != old['llm_calls']:
                raise ValueError('HippoMM baseline call provenance changed')
        gold = common.by_id(read(args.run_root / 'inputs/gold.jsonl'))
        detail = []
        for row in responses:
            q = gold[row['id']]
            detail.append({'id': row['id'], 'task': row['task'], 'prediction': row['answer'],
                'answer': q['answer'], **common.score_answer(row['answer'], q['answer'])})
        native = json.loads((target / 'online/summary.json').read_text())
        summary = {'status': 'complete', 'method': 'hippomm', 'seed': SEED, 'judge_seed': 42,
            'questions': QUESTIONS, 'overall': {'questions': QUESTIONS,
                'exact_match_correct': sum(int(r['em']) for r in detail),
                **{key: sum(r[key] for r in detail) / QUESTIONS for key in ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')}},
            'native_summary': native, 'fresh_only_reuse': True}
        frozen_rows(target / 'per_question.jsonl', detail)
        dump(target / 'summary.json', summary)
        answer_path, question_path = target / 'online/qa_responses.jsonl', target / 'online/questions.jsonl'
        prediction_field = 'answer'
    if len(responses) != QUESTIONS or len(common.by_id(responses)) != QUESTIONS:
        raise ValueError('Incomplete visual answers')
    questions = common.by_id(read(question_path))
    gold = common.by_id(read(args.run_root / 'inputs/gold.jsonl'))
    if set(questions) != set(gold) or set(common.by_id(responses)) != set(gold):
        raise ValueError('Visual answer/gold IDs differ')
    judge = [{'id': r['id'], 'task': questions[r['id']]['task'],
        'question': questions[r['id']]['question'], 'ground_truth': gold[r['id']]['answer'],
        'model_output': r[prediction_field], 'answer_model': 'Qwen/Qwen3-VL-8B-Instruct',
        'source_record_id': r['id'], 'source_hit_token_limit': r.get('hit_token_limit', False),
        'source_responses_sha256': sha(answer_path)} for r in responses]
    frozen_rows(target / 'judge_inputs.jsonl', judge)
    completion = {'status': 'complete', 'method': args.method, 'seed': SEED, 'judge_seed': 42,
        'questions': QUESTIONS, 'answered': len(responses), 'native_generation_cache_calls': native_call_count,
        'source_responses': str(answer_path), 'responses_sha256': sha(answer_path),
        'summary_sha256': sha(target / 'summary.json'),
        'judge_inputs_sha256': sha(target / 'judge_inputs.jsonl'),
        'old_llm_outputs_imported': 0, 'implementation_sha256': implementation()}
    normalized = args.run_root / 'evaluations' / args.method
    source_hashes = {str(answer_path): sha(answer_path), str(question_path): sha(question_path),
        str(target / 'summary.json'): sha(target / 'summary.json'),
        str(args.run_root / 'inputs/gold.jsonl'): sha(args.run_root / 'inputs/gold.jsonl'),
        str(target / 'stability_protocol.json'): sha(target / 'stability_protocol.json')}
    records = [{**r, 'point': questions[r['id']]['point'],
        'source_response_file': str(answer_path), 'source_response_sha256': sha(answer_path),
        'source_prediction_field': prediction_field,
        'source_raw_answer': r['model_output']} for r in judge]
    frozen_rows(normalized / 'records.jsonl', records)
    overall = summary['overall']
    metrics = {'status': 'complete', 'seed': SEED, 'questions': QUESTIONS,
        'method': args.method, 'correct': overall['exact_match_correct'],
        'em': overall['em'], 'f1': overall['f1'], 'bleu_1': overall['bleu_1'],
        'source_sha256': source_hashes, 'records_sha256': sha(normalized / 'records.jsonl')}
    if args.method == 'hippomm':
        metrics['native_routes'] = {k: summary['native_summary'][k] for k in
            ('fast_answers_reused', 'detailed_answers_regenerated', 'empty_retrieval_answers_reused')}
        metrics['native_routes']['source'] = 'Fresh seed-1042 baseline only; no seed-42 answer reuse.'
    frozen_json(normalized / 'metrics.json', metrics)
    completion.update(evaluation_metrics_sha256=sha(normalized / 'metrics.json'),
                      evaluation_records_sha256=sha(normalized / 'records.jsonl'))
    frozen_json(target / 'completion.json', completion)
    print(json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('clip_image_dialogue', 'hippomm'), required=True)
    parser.add_argument('--phase', choices=('source-audit', 'prepare', 'build', 'run', 'analyze'), required=True)
    parser.add_argument('--run-root', type=Path, default=DEFAULT)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--limit-run', type=int, default=0)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.limit_run < 0:
        parser.error('--limit-run must be nonnegative')
    target = out(args)
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        patch_native_seed()
        if args.phase == 'source-audit':
            source_audit(args)
        elif args.phase == 'prepare':
            prepare(args)
        elif args.phase == 'build':
            if args.method != 'hippomm':
                parser.error('build applies only to HippoMM event summaries')
            build_hip(args)
        elif args.phase == 'run':
            (run_clip if args.method == 'clip_image_dialogue' else run_hip)(args)
            if not args.limit_run:
                analyze(args)
        else:
            analyze(args)


if __name__ == '__main__':
    main()
