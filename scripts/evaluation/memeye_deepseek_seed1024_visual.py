"""Fresh full-flow DeepSeek Flash seed-1024 visual-method replication.

CLIP image/dialogue indexing is recomputed, and HippoMM re-encodes all historical
images, consolidates events and obtains fresh DeepSeek summaries before fresh
MCQ/Open retrieval and answering. Frozen original DeepSeek captions are shared.
All generation uses the isolated seed-1024 API transport; Qwen is never loaded.
Original three-attempt rewrites plus three three-word-array repairs are retained.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gc
import gzip
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import mem_gallery_master_reader as io
from scripts.evaluation import memeye_deepseek_seed1024_transport as api
from scripts.evaluation import memeye_deepseek_open_visual_methods as old_open
from scripts.evaluation import memeye_seed1024_visual as clip_helpers
from scripts.memory import hippomm as base, hippomm_api as hip
from scripts.memory import clip_image_dialogue as dialogue, clip_image_memory as clip

DEFAULT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/visual_methods'
CAPTIONS = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
NATURAL = ROOT / 'results/natural_observation'
OPEN = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
GOLD = OPEN.parent / 'gold.jsonl'
REFERENCE_HIP = ROOT / 'results/hippomm_api_memeye_20261104/deepseek'
REFERENCE_CLIP = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue_api/deepseek-v4.1-flash'
sha, digest, read, dump = io.sha, io.digest, io.read_lines, io.dump


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def frozen(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Existing frozen protocol differs: ' + str(path))
    dump(path, value)


def write(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
    if path.exists() and path.read_text() != text:
        raise ValueError('Existing frozen artifact differs: ' + str(path))
    path.write_text(text)


def cfg(args):
    return SimpleNamespace(dataset='memeye', natural_root=NATURAL, benchmark_manifest=BENCHMARK,
        task=None, question_ids=None, limit_questions=0, top_k=5, device=args.device,
        output_root=args.output_root / 'clip_full', clip_image_source=None,
        hippomm_backbone='deepseek', hippomm_api_workers=args.workers,
        hippomm_api_env_file=old_open.ENV_FILE, hippomm_max_new_tokens=2048,
        hippomm_context_budget=120000, hippomm_stage='build')


def patch(args):
    # The source audited chat persists exact original-image wires and every API
    # attempt. Change its transport at the process-local configuration boundary.
    old_open.gpt = api
    original_contract = base.contract_at
    def contract(path, value):
        if Path(path).resolve().is_relative_to(args.output_root):
            value = {**value, 'seed': 1024, 'judge_seed': 42,
                'seed_wrapper_sha256': sha(__file__), 'seed_transport_sha256': sha(api.__file__)}
        return original_contract(path, value)
    base.contract_at = contract
    original_manifest = hip.manifest
    def manifest(directory, backbone):
        if backbone != 'deepseek':
            raise ValueError('Only the original DeepSeek Flash condition is supported')
        return original_manifest(directory, backbone)
    hip.manifest = manifest
    hip.transport.APIChat = DeepChat
    old_open.AuditedChat = DeepChat


SOURCE_CHAT = old_open.AuditedChat
class DeepChat(SOURCE_CHAT):
    def __init__(self, directory, backbone='deepseek', env_file=None, token_limit=2048, allow_api=True):
        if backbone != 'deepseek' or token_limit != 2048:
            raise ValueError('HippoMM must preserve the original DeepSeek/2048 condition')
        super().__init__(directory, allow_api=allow_api)
    def __call__(self, role, prompt, image_paths=()):
        raw = super().__call__(role, prompt, image_paths)
        record = json.loads((self.directory / (self.last_call_id + '.json')).read_text())
        if record['response']['seed'] != 1024 or record['response']['returned_model'] != api.MODEL:
            raise ValueError('HippoMM actual provider seed/model differs')
        return raw


def source_files():
    files = [Path(__file__), Path(api.__file__), Path(hip.__file__), Path(old_open.__file__),
        Path(base.__file__), Path(base.core.__file__), Path(base.inputs.__file__),
        Path(dialogue.__file__), Path(clip.__file__), Path(clip_helpers.__file__),
        BENCHMARK, BENCHMARK.parent / 'protocol.json', OPEN,
        CAPTIONS / 'captions.jsonl', CAPTIONS / 'completion_manifest.json',
        CAPTIONS / 'caption_generation/fresh_responses.jsonl',
        REFERENCE_HIP / 'online/protocol.json', REFERENCE_HIP / 'online/summary.json',
        REFERENCE_CLIP / 'protocol.json', REFERENCE_CLIP / 'summary.json']
    files += [NATURAL / n for n in ('protocol.json', 'images.jsonl', 'memories.jsonl', 'captions.jsonl')]
    files += [ROOT / s['path'] for s in json.loads((NATURAL / 'protocol.json').read_text())['source_files']]
    return files


def verify(args):
    p = json.loads((args.output_root / 'stability_protocol.json').read_text())
    if digest(p['contract']) != p['experiment_signature']:
        raise ValueError('Stability protocol signature differs')
    for name, expected in p['contract']['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen stability source changed: ' + name)
    return p


def prepare(args):
    _, captions = hip.caption_source('deepseek')
    config = cfg(args)
    rounds, images, _ = base.inputs.memory_inputs(config)
    if len(rounds) != 848 or len(images) != 433 or len(captions) != 437:
        raise ValueError('Original complete DeepSeek input cohort differs')
    for image in images:
        if image['image_sha256'] not in captions or sha(image['path']) != image['image_sha256']:
            raise ValueError('Original image/DeepSeek caption coverage differs')
    ref_clip, ref_hip = (json.loads((p / 'summary.json').read_text()) for p in
                        (REFERENCE_CLIP, REFERENCE_HIP / 'online'))
    if round(ref_clip['accuracy'] * 371) != 212 or ref_hip['correct'] != 207:
        raise ValueError('Seed-42 DeepSeek reference scores differ')
    contract = dict(model=api.MODEL, seed=1024, judge_seed=42, questions=371,
        shared_caption_seed=42, shared_caption_model=api.MODEL,
        caption_policy='Exact original 437 DeepSeek captions; no new caption calls.',
        visual_policy='Fresh CLIP images, novelty filter and queries; fresh DeepSeek summaries/retrieval/answers.',
        qwen_generation=False, historical_answer_reuse=False,
        clip_mcq_max_tokens=16, open_max_tokens=2048, hippo_all_roles_max_tokens=2048,
        hippo_query_recovery='original 3 attempts then at most 3 three-word-array attempts',
        source_sha256={str(p.resolve()): sha(p) for p in source_files()})
    frozen(args.output_root / 'stability_protocol.json', dict(contract=contract, experiment_signature=digest(contract)))
    frozen(args.output_root / 'readiness.json', dict(status='ready_not_started', questions=371,
        historical_dialogue_rounds=848, distinct_task_images=433, frozen_captions=437,
        reference_clip_correct=212, reference_hippomm_correct=207, seed=1024,
        api_calls=0, model_loaded=False))
    print(json.dumps(dict(phase='prepare', status='ready_not_started', seed=1024,
        captions=437, images=433, dialogue_rounds=848, questions=371)), flush=True)


def build_hippomm(args, index_only=False):
    verify(args)
    config = cfg(args)
    output = args.output_root / 'hippomm/offline'
    hip.manifest(output, 'deepseek')
    if (output / 'memory.json').exists():
        hip.load_memory(output, 'deepseek'); return
    output.mkdir(parents=True, exist_ok=True)
    source, captions = hip.caption_source('deepseek')
    rounds, images, sources = base.inputs.memory_inputs(config)
    for image in images:
        image['caption'] = captions[image['image_sha256']]['caption']
    base.contract_at(output / 'build_contract.json', dict(method=hip.METHOD, backbone='deepseek',
        caption_source=str(source), max_output_tokens=2048, duplicate_threshold=.9,
        visual_construction='Fresh original-image CLIP encoding and original sequential novelty consolidation.',
        source_sha256=sources))
    index_path = output / 'index_manifest.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for name, expected in index['artifact_sha256'].items():
            if sha(name) != expected: raise ValueError('Fresh visual index changed')
        events = read(output / 'pending_events.jsonl')
        records = read(output / 'representative_images.jsonl')
        encoder_metadata = index['clip_encoder']
    else:
        from scripts.graph.visual_image_encoder import ImageEncoder
        import torch
        torch.set_num_threads(8); torch.manual_seed(1024)
        encoder = ImageEncoder(str(base.CLIP), device=args.device, local_files_only=True)
        events, vectors, mapping = base.construct_events(rounds, images, encoder.encode, duplicate_threshold=.9)
        encoder_metadata = encoder.metadata
        del encoder; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        import numpy as np
        by_ref = {i['image_ref']: i for i in images}
        records, provenance = [], []
        for event in events:
            image = by_ref[event['representative_image']]
            source_path = Path(image['path'])
            relative = Path('images') / (event['event_id'] + source_path.suffix.lower())
            hip.original.copy_exact(source_path, output / relative)
            records.append(dict(event_id=event['event_id'], task=event['task'], image_ref=image['image_ref'],
                                file=str(relative), bytes=source_path.stat().st_size))
            cap = captions[image['image_sha256']]
            provenance.append(dict(event_id=event['event_id'], image_ref=image['image_ref'],
                caption_model=cap['model'], seed=42, response_id=cap['response_id']))
        write(output / 'dialogues.jsonl', rounds)
        write(output / 'image_mapping.jsonl', mapping)
        write(output / 'representative_images.jsonl', records)
        write(output / 'caption_provenance.jsonl', provenance)
        np.savez(output / 'visual_embeddings.npz', event_ids=np.array([e['event_id'] for e in events]),
                 vectors=np.stack([vectors[e['event_id']] for e in events]).astype(np.float32))
        write(output / 'pending_events.jsonl', events)
        index_files = [output / n for n in ('pending_events.jsonl','dialogues.jsonl','image_mapping.jsonl',
            'representative_images.jsonl','caption_provenance.jsonl','visual_embeddings.npz')]
        frozen(output / 'index_manifest.json', dict(status='index_ready_summary_pending', seed=1024,
            events=len(events), historical_images=len(images), clip_encoder=encoder_metadata,
            artifact_sha256={str(p):sha(p) for p in index_files}))
    if index_only:
        print(json.dumps(dict(phase='index', method='hippomm', status='index_ready_summary_pending',
            events=len(events), images=len(images), api_calls=0)),flush=True)
        return
    dialogues = base.inputs.unique(rounds)
    def summarize(event):
        chat = DeepChat(output / 'llm_cache' / event['event_id'])
        return {**event, 'summary': chat('summary', base.core.summary_prompt(event, dialogues)).strip()}
    completed = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(summarize, e): e['event_id'] for e in events}
        for future in as_completed(jobs):
            event = future.result(); completed[event['event_id']] = event
            print(json.dumps(dict(stage='summary', done=len(completed), total=len(events))), flush=True)
    events = [completed[e['event_id']] for e in events]
    write(output / 'events.jsonl', events)
    files = (*hip.original.MEMORY_FILES, 'representative_images.jsonl', 'caption_provenance.jsonl')
    dump(output / 'memory.json', dict(method=hip.METHOD, backbone='deepseek', model=api.MODEL,
        events=len(events), dialogue_rounds=len(rounds), caption_source=str(source),
        original_image_bytes=sum(r['bytes'] for r in records),
        memory_artifact_bytes=sum((output / n).stat().st_size for n in files),
        max_output_tokens=2048, seed=1024, clip_encoder=encoder_metadata))
    hip.load_memory(output, 'deepseek')


def prepare_clip_mcq(args):
    # This helper only executes CLIP encoding and dialogue attachment. Its Qwen
    # runtime/run functions are never invoked; the original DeepSeek VisRAG
    # condition itself reused the same Qwen-independent CLIP index definition.
    config = cfg(args)
    target = clip_helpers.prepare_clip_mcq(args, config)
    protocol, manifest = dialogue.verify(target)
    return target, protocol, manifest


def image_messages(q):
    messages = copy.deepcopy(q['reader_messages'])
    for message in messages:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'image': part['sha256'] = q['reader_image_hashes'][part['path']]
    return messages


def api_call(config, directory, body, max_attempts):
    key = api.request_sha256(body)
    directory = directory / key; directory.mkdir(parents=True, exist_ok=True)
    wire_path, out_path = directory / 'wire.json.gz', directory / 'response.json'
    if out_path.exists():
        if json.loads(gzip.decompress(wire_path.read_bytes())) != body:
            raise ValueError('Run-local exact wire cache differs')
        result = json.loads(out_path.read_text())
        if result['seed'] != 1024 or result['request_sha256'] != key:
            raise ValueError('Cached provider seed/request differs')
        return result
    wire = gzip.compress(json.dumps(body, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(), mtime=0)
    if wire_path.exists() and wire_path.read_bytes() != wire:
        raise ValueError('Unfinished request bytes differ')
    wire_path.write_bytes(wire)
    log = directory / 'attempts.jsonl'
    prior = read(log) if log.exists() else []
    if prior and prior[-1]['status'] == 'ok':
        dump(out_path, prior[-1]['response']); return prior[-1]['response']
    if prior and (len(prior) >= max_attempts or not prior[-1]['retryable']):
        raise ValueError('Request exhausted original bounded transient retry policy')
    for attempt in range(len(prior) + 1, max_attempts + 1):
        try:
            response = api.chat_request(config, body, timeout=120)
            append(log, dict(attempt=attempt, status='ok', retryable=False, response=response))
            dump(out_path, response); return response
        except api.APIError as exc:
            append(log, dict(attempt=attempt, status='error', retryable=exc.retryable,
                kind=exc.kind, http_status=exc.http_status, code=exc.code, request_sha256=key))
            if not exc.retryable or attempt == max_attempts: raise
            time.sleep(min(2 ** attempt, 16))


def offline_inventory(args):
    """Reviewable local inputs and exact wire hashes; never transmits a request."""
    output=args.output_root/args.method
    if args.method=='hippomm':
        memory=output/'offline'
        events=read(memory/'pending_events.jsonl')
        dialogues=base.inputs.unique(read(memory/'dialogues.jsonl'))
        previews=[]
        for event in events:
            prompt=base.core.summary_prompt(event,dialogues)
            body=api.build_request(dict(system=io.READER_SYSTEM,user_text=prompt,query_images=[]),
                model=api.MODEL,max_output_tokens=2048,image_detail='high')
            previews.append(dict(event_id=event['event_id'],role='summary',seed=body['seed'],
                max_tokens=body['max_tokens'],wire_sha256=api.request_sha256(body),
                user_text=prompt,images=0))
        write(output/'summary_request_manifest.jsonl',previews)
        inventory=dict(method='hippomm',status='ready_for_review_not_transmitted',seed=1024,
            summary_requests=len(previews),mcq_questions=371,open_questions=371,
            historical_images=433,representative_images=len(events),dialogue_rounds=len(dialogues),
            captions='Frozen original DeepSeek seed42 captions',
            summary_request_manifest_sha256=sha(output/'summary_request_manifest.jsonl'),
            api_calls=0,caption_api_calls=0,qwen_generation=False,
            destination=api.DASHSCOPE_BASE_URL+'/chat/completions')
    else:
        source,_,mcq=prepare_clip_mcq(args)
        opened=read(output/'open/reader_manifest.jsonl')
        manifests={}
        for name,cohort,cap in [('mcq',mcq,16),('open',opened,2048)]:
            values=[]
            for q in cohort:
                body=api.build_message_request(image_messages(q),max_tokens=cap,temperature=0)
                values.append(dict(id=q['id'],seed=1024,max_tokens=cap,
                    wire_sha256=api.request_sha256(body),payload_sha256=q['payload_sha256'],
                    original_image_paths=q['reader_image_paths'],
                    original_image_sha256=q['reader_image_hashes']))
            path=output/(name+'_request_manifest.jsonl');write(path,values)
            manifests[name]=dict(questions=len(values),manifest=str(path),sha256=sha(path),max_tokens=cap)
        inventory=dict(method='clip_image_dialogue',status='ready_for_review_not_transmitted',seed=1024,
            requests=manifests,history_images_per_question=5,api_calls=0,qwen_generation=False,
            image_policy='Original PNG/JPEG bytes; high detail; unchanged native image/text interleaving.',
            destination=api.DASHSCOPE_BASE_URL+'/chat/completions')
    dump(output/'payload_inventory.json',inventory)
    print(json.dumps(dict(method=args.method,phase='inventory',status=inventory['status'],api_calls=0)),flush=True)


def clip_run(args, open_task=False):
    verify(args)
    if open_task:
        target = args.output_root / 'clip_image_dialogue/open'
        p, manifest = json.loads((target / 'protocol.json').read_text()), read(target / 'reader_manifest.jsonl')
        cap, attempts = 2048, 4
    else:
        source, original, manifest = prepare_clip_mcq(args)
        target = args.output_root / 'clip_image_dialogue/online'
        target.mkdir(parents=True, exist_ok=True)
        write(target / 'qa_manifest.jsonl', manifest)
        p = dict(method_id='clip_image_dialogue', answer_model=api.MODEL, seed=1024,
            max_tokens=16, questions=371, top_k=5, source_manifest=str(source / 'qa_manifest.jsonl'),
            source_signature=original['experiment_signature'],
            frozen_retrieval_source=original['frozen_retrieval_source'],
            image_policy='Original PNG/JPEG, high, unchanged order and bytes.')
        p['experiment_signature'] = digest(p)
        frozen(target / 'protocol.json', p)
        cap, attempts = 16, 3
    config = api.load_config(old_open.ENV_FILE)
    path = target / ('responses.jsonl' if open_task else 'qa_responses.jsonl')
    previous = read(path) if path.exists() else []
    done = base.inputs.unique(previous)
    if not set(done) <= {q['id'] for q in manifest}: raise ValueError('Foreign response ID')
    def answer(q):
        body = api.build_message_request(image_messages(q), max_tokens=cap, temperature=0)
        try:
            result = api_call(config, target / 'calls', body, attempts)
        except api.APIError as exc:
            if open_task or (exc.kind!='data_inspection_failed' and getattr(exc,'provider_error_code',None)!='data_inspection_failed'):
                raise
            # Match the original VisRAG MCQ policy: a real provider content
            # rejection is an explicitly incorrect evaluation record, not a
            # fabricated successful API response or an answer-quality retry.
            result=dict(status='provider_rejection',raw_response='',returned_model=None,
                request_sha256=api.request_sha256(body),seed=1024,hit_token_limit=False,
                error_type='data_inspection_failed',http_status=exc.http_status)
        row = {**result, **{k:q[k] for k in ('id','task')}, 'payload_sha256': q['payload_sha256'],
               'experiment_signature': p.get('experiment_signature',digest(p)), 'method_id':'clip_image_dialogue'}
        if open_task: row['point'] = q['point']
        else:
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', result['raw_response'].strip(), re.I)
            row['predicted_choice'] = match.group(1).upper() if match else None
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(answer, q) for q in manifest if q['id'] not in done]
        for future in as_completed(jobs):
            row = future.result(); append(path, row); done[row['id']] = row
            print(json.dumps(dict(method='clip_image_dialogue', open=open_task, done=len(done), total=371)), flush=True)
    if len(done) != 371: raise ValueError('Incomplete CLIP answers; no final metric')
    dump(target / 'completion.json', dict(status='complete', questions=371, answered=371,
        seed=1024, responses_sha256=sha(path), token_limit_hits=sum(r['hit_token_limit'] for r in done.values()),
        empty_answers=sum(not r['raw_response'] for r in done.values())))
    if open_task: collect(target, path)
    else: analyze_clip(args)


def analyze_clip(args):
    target = args.output_root / 'clip_image_dialogue/online'
    qs, answers = base.inputs.unique(read(target / 'qa_manifest.jsonl')), base.inputs.unique(read(target / 'qa_responses.jsonl'))
    if len(qs) != 371 or set(qs) != set(answers): raise ValueError('Incomplete CLIP answers')
    correct = sum(answers[k]['predicted_choice'] == q['gold_choice'] for k,q in qs.items())
    hits = sum(len(set(q['gold_ids']) & set(q['top_memory_ids'])) for q in qs.values())
    denominator = sum(len(set(q['gold_ids'])) for q in qs.values())
    summary = dict(status='complete', method='clip_image_dialogue', questions=371, correct=correct,
        accuracy=correct/371, seed=1024, context_source_micro_recall=hits/denominator,
        recall_numerator=hits, recall_denominator=denominator,
        format_errors=sum(r['predicted_choice'] is None for r in answers.values()))
    dump(target / 'summary.json', summary)


def open_prepare(args):
    verify(args)
    target = args.output_root / args.method / 'open'
    if (target / 'protocol.json').exists(): return
    qs = old_open.questions()
    if args.method == 'hippomm':
        memory = args.output_root / 'hippomm/offline'
        hip.load_memory(memory, 'deepseek')
        files = [Path(__file__), Path(api.__file__), Path(old_open.__file__), OPEN]
        files += [memory / n for n in ('memory.json','events.jsonl','dialogues.jsonl','image_mapping.jsonl',
                    'visual_embeddings.npz','representative_images.jsonl','caption_provenance.jsonl')]
        write(target / 'questions.jsonl', qs)
        frozen(target / 'protocol.json', dict(method_id='hippomm', answer_model=api.MODEL, questions=371,
            memory_dir=str(memory), seed=1024, evaluator_seed=42, max_completion_tokens=2048,
            top_k=5, confidence_threshold=.75, fallback_threshold=.4, fallback_score=.6,
            query_recovery='original 3 attempts then at most 3 three-word-array attempts', api_attempts=4,
            questions_sha256=sha(target/'questions.jsonl'), source_sha256={str(p):sha(p) for p in files}))
        return
    # The pure preparation helper preserves the source Open prompts and CPU
    # CLIP text precision. It loads no answer model and issues no API requests.
    original_verify = clip_helpers.verify_preflight
    clip_helpers.verify_preflight = lambda _args: verify(args)
    try: clip_helpers.open_prepare(args)
    finally: clip_helpers.verify_preflight = original_verify
    manifest = read(target / 'reader_manifest.jsonl')
    for row in manifest:
        row.pop('max_new_tokens', None); row['max_completion_tokens'] = 2048
    # Replace only still-unconsumed preparation metadata before freezing the
    # actual API run; native messages, retrieval vectors and order are intact.
    (target / 'reader_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    p = json.loads((target / 'protocol.json').read_text())
    p.update(answer_model=api.MODEL, seed=1024, max_new_tokens=2048,
             source_mcq_correct=212, source_mcq_correct_note='Original seed42 reference; no answer reuse.')
    p['artifact_sha256']['reader_manifest.jsonl'] = sha(target / 'reader_manifest.jsonl')
    p['source_sha256'][str(Path(api.__file__))] = sha(api.__file__)
    dump(target / 'protocol.json', p)


def collect(target, response_path):
    seal = json.loads((target / 'completion.json').read_text())
    values = base.inputs.unique(read(response_path))
    qs, gold = old_open.questions(), base.inputs.unique(read(GOLD))
    if seal['status'] != 'complete' or seal['responses_sha256'] != sha(response_path) or len(values) != 371:
        raise ValueError('All original Open answers must be sealed before gold is read')
    write(target / 'judge_inputs.jsonl', [dict(id=q['id'], task=q['task'], point=q['point'], question=q['question'],
        ground_truth=gold[q['id']]['answer'], model_output=values[q['id']]['raw_response'],
        answer_model=api.MODEL, source_record_id=q['id'], source_seed=1024,
        source_responses_sha256=sha(response_path), source_hit_token_limit=values[q['id']]['hit_token_limit']) for q in qs])


def hip_open_run(args):
    target = args.output_root / 'hippomm/open'
    p = json.loads((target / 'protocol.json').read_text())
    for name, expected in p['source_sha256'].items():
        if sha(name) != expected: raise ValueError('Open source changed')
    memory, qs = Path(p['memory_dir']), read(target/'questions.jsonl')
    _, events, vectors, dialogues, images = hip.load_memory(memory,'deepseek')
    encoder = base.ClipText('cpu'); mutex = threading.Lock()
    def encode(query):
        with mutex: return encoder(query)
    path = target/'responses.jsonl'
    done = base.inputs.unique(read(path)) if path.exists() else {}
    def answer(i,q):
        return old_open.native_answer(target,i,q,memory,events,vectors,dialogues,images,encode)
    errors=[]
    with ThreadPoolExecutor(max_workers=args.open_workers) as pool:
        jobs={pool.submit(answer,i,q):q['id'] for i,q in enumerate(qs,1) if q['id'] not in done}
        for future in as_completed(jobs):
            qid=jobs[future]
            try:
                row=future.result(); row['seed']=1024
                append(path,row);done[qid]=row
                print(json.dumps(dict(method='hippomm',open=True,done=len(done),total=371)),flush=True)
            except Exception as exc:
                error=dict(id=qid,type=type(exc).__name__,error=str(exc))
                errors.append(error);append(target/'question_errors.jsonl',error)
    status=dict(status='complete' if len(done)==371 else 'partial',questions=371,answered=len(done),
        seed=1024,responses_sha256=sha(path) if path.exists() else None,errors_this_pass=errors,
        token_limit_hits=sum(r['hit_token_limit'] for r in done.values()),empty_answers=sum(not r['raw_response'] for r in done.values()))
    dump(target/'completion.json',status)
    if len(done)!=371: raise ValueError('HippoMM Open partial; raw failures preserved and no synthetic success/score inserted')
    collect(target,path)


def run(args):
    verify(args)
    if args.method=='clip_image_dialogue': return clip_run(args)
    build_hippomm(args)
    config=cfg(args);config.memory_dir=args.output_root/'hippomm/offline';config.output=args.output_root/'hippomm/online'
    hip.prepare(config);hip.run(config);hip.analyze(config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=('clip_image_dialogue','hippomm'),required=True)
    parser.add_argument('--phase',choices=('prepare','index','inventory','run','analyze','open-prepare','open-run','all'),required=True)
    parser.add_argument('--output-root',type=Path,default=DEFAULT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--open-workers',type=int,default=8)
    args=parser.parse_args();args.output_root=args.output_root.resolve();args.output_root.mkdir(parents=True,exist_ok=True)
    patch(args)
    with (args.output_root/(args.method+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='prepare': prepare(args);return
        if args.phase=='index':
            verify(args)
            if args.method=='hippomm': build_hippomm(args,index_only=True)
            else:
                prepare_clip_mcq(args);open_prepare(args)
            offline_inventory(args)
            return
        if args.phase=='inventory': verify(args);offline_inventory(args);return
        if args.phase in ('run','all'):run(args)
        if args.phase=='analyze':
            if args.method=='clip_image_dialogue':analyze_clip(args)
            else:
                config=cfg(args);config.memory_dir=args.output_root/'hippomm/offline';config.output=args.output_root/'hippomm/online';hip.analyze(config)
        if args.phase in ('open-prepare','all'):open_prepare(args)
        if args.phase in ('open-run','all'):
            if args.method=='clip_image_dialogue':clip_run(args,True)
            else:hip_open_run(args)


if __name__=='__main__':main()
