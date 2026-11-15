"""Fresh MemEye seed-1024 VisRAG-label and HippoMM full-flow experiments.

The original signed implementations and seed-42 artifacts are read-only.  The
VisRAG table label denotes the repository's CLIP image-plus-dialogue method.
HippoMM preserves the original paired procedure: first construct/summarize and
answer with a fresh no-pixel baseline, then rerun its detailed representative-
image reader.  Only answers generated within this seed-1024 run are reused.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.evaluation import mem_gallery_master_reader as runtime
from scripts.evaluation import memeye_open_visual_methods as open_base
from scripts.memory import clip_image_memory as clip
from scripts.memory import clip_image_dialogue as dialogue
from scripts.memory import hippomm as hip, hippomm_original as hip_original

SEED = 1024
DEFAULT = ROOT / 'results/memeye_qwen_seed1024_full_20261105/visual_methods'
NATURAL = ROOT / 'results/natural_observation'
BENCHMARK = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
REFERENCE_CLIP = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue'
REFERENCE_HIP = ROOT / 'results/hippomm_original_controlled_20261104/memeye'
REFERENCE_HIP_BASE = ROOT / 'results/hippomm_short_ids_20261104/memeye'
OPEN = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
sha, digest, read, dump = runtime.sha, runtime.digest, runtime.read_lines, runtime.dump


def write(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values)
    if path.exists() and path.read_text() != text:
        raise ValueError('Frozen artifact changed: ' + str(path))
    path.write_text(text)


def frozen_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Frozen protocol changed: ' + str(path))
    dump(path, value)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def patch_runtime_seed():
    """Change the actual local runtime seed and signed config boundary together."""
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    runtime.SEED = SEED
    original = hip.contract_at

    def seeded_contract(path, contract):
        return original(path, {**contract, 'seed': SEED, 'do_sample': False,
            'seed_wrapper_sha256': sha(__file__), 'evaluator_seed': 42})

    hip.contract_at = seeded_contract
    load = runtime.load_runtime

    def seeded_load(device):
        loaded = load(device)
        loaded[2].update(seed=SEED, do_sample=False, seed_wrapper_sha256=sha(__file__))
        return loaded

    runtime.load_runtime = seeded_load


def settings(args):
    root = args.output_root.resolve()
    return SimpleNamespace(dataset='memeye', natural_root=NATURAL,
        benchmark_manifest=BENCHMARK, task=None, question_ids=None, limit_questions=0,
        top_k=5, device=args.device, output_root=root / 'clip_full',
        clip_image_source=None, hippomm_gallery_inputs=ROOT / 'results/mem_gallery_master_answer_20261021/inputs',
        hippomm_max_new_tokens=1024, hippomm_context_budget=120000,
        hippomm_duplicate_threshold=.9, hippomm_confidence_threshold=.75,
        hippomm_fallback_threshold=.4, hippomm_fallback_score=.6,
        hippomm_baseline_root=root / 'hippomm_baseline', hippomm_stage='build')


def source_contract(method):
    files = [Path(__file__), Path(runtime.__file__), Path(clip.__file__),
        Path(dialogue.__file__), Path(hip.__file__), Path(hip_original.__file__),
        Path(hip.core.__file__), Path(hip.inputs.__file__), Path(open_base.__file__),
        ROOT / 'scripts/graph/visual_image_encoder.py',
        ROOT / 'scripts/evaluation/mcq_reader.py', BENCHMARK,
        BENCHMARK.parent / 'protocol.json']
    files += [NATURAL / n for n in ('protocol.json', 'memories.jsonl', 'images.jsonl', 'captions.jsonl')]
    natural = json.loads((NATURAL / 'protocol.json').read_text())
    files += [ROOT / f['path'] for f in natural['source_files']]
    ref = REFERENCE_CLIP if method == 'clip_image_dialogue' else REFERENCE_HIP / 'online'
    files += [ref / 'protocol.json', ref / 'summary.json', ref / 'qa_responses.jsonl']
    if method == 'hippomm':
        files += [REFERENCE_HIP_BASE / 'offline/build_contract.json',
                  REFERENCE_HIP_BASE / 'online/protocol.json']
    return dict(method=method, seed=SEED, evaluator_seed=42, questions=371,
        do_sample=False, source_reference=str(ref), reference_seed42_correct=193 if method == 'clip_image_dialogue' else 179,
        caption_policy='Shared original Qwen caption bytes frozen; no caption regeneration.',
        run_policy='Fresh local memory construction, retrieval and answering; no seed-42 answer/cache import.',
        input_sha256={str(p.resolve()): sha(p) for p in files})


def verify_preflight(args):
    target = args.output_root / args.method
    saved = json.loads((target / 'stability_protocol.json').read_text())
    contract = saved['contract']
    if digest(contract) != saved['experiment_signature'] or contract != source_contract(args.method):
        raise ValueError('Seed/source stability protocol changed')
    return saved


def prepare(args):
    cfg = settings(args)
    contract = source_contract(args.method)
    source = Path(contract['source_reference'])
    summary = json.loads((source / 'summary.json').read_text())
    if summary['questions'] != 371 or round(summary['accuracy'] * 371) != contract['reference_seed42_correct']:
        raise ValueError('Reference does not reproduce the user table')
    questions = read(BENCHMARK)
    if len(questions) != 371 or len({q['id'] for q in questions}) != 371:
        raise ValueError('Expected 371 unique MCQ questions')
    rounds, images, _ = hip.inputs.memory_inputs(cfg)
    # Verify historical bytes without loading any model or touching old caches.
    for image in images:
        if sha(image['path']) != image['image_sha256']:
            raise ValueError('Original historical image changed')
    for q in questions:
        for path in q['query_image_paths']:
            if not Path(path).is_file():
                raise ValueError('Missing query image')
    target = args.output_root / args.method
    target.mkdir(parents=True, exist_ok=True)
    frozen_json(target / 'stability_protocol.json', dict(contract=contract, experiment_signature=digest(contract)))
    frozen_json(target / 'readiness.json', dict(status='ready_not_started', seed=SEED,
        questions=371, historical_dialogue_rounds=len(rounds), task_image_candidates=len(images),
        model_loaded=False, old_answer_caches_imported=0,
        stages=['fresh_memory', 'fresh_retrieval', 'fresh_mcq_answers', 'fresh_open_answers']))
    print(json.dumps(dict(method=args.method, phase='prepare', status='ready_not_started',
        questions=371, historical_dialogue_rounds=len(rounds), task_image_candidates=len(images))), flush=True)


def prepare_clip_mcq(args, cfg):
    target = dialogue.output_dir(cfg)
    if (target / 'protocol.json').exists():
        dialogue.verify(target)
        return target
    # clip.prepare inherits its seed from this small, fresh configuration file.
    src_protocol = json.loads((BENCHMARK.parent / 'protocol.json').read_text())
    baseline = {k: src_protocol[k] for k in ('reader_model', 'reader_model_path',
        'dtype', 'max_new_tokens', 'query_image_max_pixels', 'reader_system_prompt')}
    baseline['seed'] = SEED
    pinned = args.output_root / 'clip_full/inputs'
    frozen_json(pinned / 'protocol.json', baseline)
    write(pinned / 'qa_manifest.jsonl', read(BENCHMARK))
    cfg.benchmark_manifest = pinned / 'qa_manifest.jsonl'
    image_target = clip.output_dir(cfg)
    if not (image_target / 'protocol.json').exists():
        clip.prepare(cfg)
    protocol, source_rows = clip.verify(image_target)
    if protocol['seed'] != SEED:
        raise ValueError('CLIP runtime protocol seed mismatch')
    dialogues, sources = dialogue.source_dialogues(NATURAL, {q['task'] for q in source_rows})
    system = protocol['reader_system_prompt'].replace('retrieved historical images',
        'retrieved historical images and their accompanying dialogue')
    output = []
    for source in source_rows:
        row = copy.deepcopy(source)
        row['dialogue_by_memory_id'] = dialogue.attach_dialogues(row, dialogues)
        row['reader_messages'] = dialogue.reader_messages(row, system)
        row['source_payload_sha256'] = source['payload_sha256']
        row['payload_sha256'] = clip.digest(dict(messages=row['reader_messages'], image_hashes=row['reader_image_hashes']))
        output.append(row)
    write(target / 'qa_manifest.jsonl', output)
    protocol = copy.deepcopy(protocol)
    protocol.pop('experiment_signature')
    protocol.update(mode='clip_image_dialogue', reader_system_prompt=system,
        frozen_retrieval_source=str(image_target),
        frozen_retrieval_signature=json.loads((image_target / 'protocol.json').read_text())['experiment_signature'],
        source_dialogue_files=[str(p) for p in sources],
        dialogue_policy='all source rounds attached to each deduplicated image, in original order; exactly K images',
        answer_policy='fresh Top-K native historical images with exact corresponding User/Assistant dialogue; question/options and native query image; no captions')
    files = [Path(__file__), Path(dialogue.__file__), image_target / 'protocol.json',
        image_target / 'qa_manifest.jsonl', target / 'qa_manifest.jsonl', *sources]
    protocol['input_hashes'].update({str(p.resolve()): sha(p) for p in files})
    protocol['experiment_signature'] = clip.digest(protocol)
    frozen_json(target / 'protocol.json', protocol)
    dialogue.verify(target)
    return target


def run_mcq(args):
    verify_preflight(args)
    cfg = settings(args)
    if args.method == 'clip_image_dialogue':
        prepare_clip_mcq(args, cfg)
        dialogue.run(cfg)
        analyze(args)
        return
    cfg.output = cfg.hippomm_baseline_root / 'memeye/offline'
    hip.build(cfg)
    cfg.memory_dir = cfg.output
    cfg.output = cfg.hippomm_baseline_root / 'memeye/online'
    hip.prepare(cfg)
    hip.run(cfg)
    hip.analyze(cfg)  # Full coverage is required before representative replay.
    cfg.output = args.output_root / 'hippomm/offline'
    hip_original.build(cfg)
    cfg.memory_dir = cfg.output
    cfg.output = args.output_root / 'hippomm/online'
    hip_original.prepare(cfg)
    hip_original.run(cfg)
    hip_original.analyze(cfg)
    analyze(args)


def analyze(args):
    preflight = verify_preflight(args)
    cfg = settings(args)
    if args.method == 'hippomm':
        cfg.memory_dir = args.output_root / 'hippomm/offline'
        cfg.output = args.output_root / 'hippomm/online'
        hip_original.analyze(cfg)
        qs, responses = read(cfg.output / 'questions.jsonl'), read(cfg.output / 'qa_responses.jsonl')
        summary = json.loads((cfg.output / 'summary.json').read_text())
        baseline = cfg.hippomm_baseline_root / 'memeye/online'
        if not json.loads((baseline / 'completion_manifest.json').read_text())['status'] == 'complete':
            raise ValueError('Fresh baseline incomplete')
        for r in responses:
            for call in r.get('baseline_llm_calls', []):
                if not (baseline / 'llm_cache' / (call + '.json')).is_file():
                    raise ValueError('Missing fresh-seed baseline call')
    else:
        target = dialogue.output_dir(cfg)
        protocol, qs = dialogue.verify(target)
        responses = clip.checked_responses(target, protocol, qs)
        if len(responses) != 371:
            raise ValueError('Incomplete CLIP answers; final metrics withheld')
        by_id = {r['id']: r for r in responses}
        per_question = []
        for q in qs:
            r = by_id[q['id']]
            if r['reader_image_count'] != 5 + len(q['query_image_paths']) or r['dialogue_round_count'] != len(q['dialogue_by_memory_id']):
                raise ValueError('Actual CLIP image/dialogue payload count differs')
            hits = len(set(q['gold_ids']) & set(q['top_memory_ids']))
            per_question.append(dict(id=q['id'], task=q['task'], gold_choice=q['gold_choice'],
                predicted_choice=r['predicted_choice'], correct=r['predicted_choice'] == q['gold_choice'],
                context_gold_hits=hits, gold_count=len(set(q['gold_ids']))))
        gold_count = sum(r['gold_count'] for r in per_question)
        hits = sum(r['context_gold_hits'] for r in per_question)
        correct = sum(r['correct'] for r in per_question)
        summary = dict(method='clip_image_dialogue', questions=371, correct=correct, accuracy=correct / 371,
            context_source_micro_recall=hits / gold_count, recall_numerator=hits, recall_denominator=gold_count,
            format_errors=sum(r['status'] != 'ok' for r in responses),
            token_limit_hits=sum(r['hit_token_limit'] for r in responses), seed=SEED,
            experiment_signature=protocol['experiment_signature'])
        write(target / 'per_question.jsonl', per_question)
        dump(target / 'summary.json', summary)
        dump(target / 'completion_manifest.json', dict(status='complete', questions=371,
            artifacts=['qa_manifest.jsonl', 'qa_responses.jsonl', 'per_question.jsonl', 'summary.json']))
    if len(qs) != len(responses) or len(responses) != len({r['id'] for r in responses}) or {r['id'] for r in qs} != {r['id'] for r in responses}:
        raise ValueError('Final response ID coverage mismatch')
    audit = dict(status='complete', method=args.method, seed=SEED, evaluator_seed=42,
        questions=371, responses=371, duplicate_ids=0, missing_ids=[], seed42_answers_imported=0,
        stability_signature=preflight['experiment_signature'], summary=summary)
    dump(args.output_root / args.method / 'coverage_audit.json', audit)
    print(json.dumps(audit), flush=True)


def open_questions():
    values = read(OPEN)
    if len(values) != len({q['id'] for q in values}) or len(values) != 371:
        raise ValueError('Incomplete Open cohort')
    if any(set(q) & {'answer', 'options', 'gold_ids', 'gold_choice'} for q in values):
        raise ValueError('Open question cohort contains evaluation fields')
    return values


def open_prepare(args):
    verify_preflight(args)
    cfg = settings(args)
    target = args.output_root / args.method / 'open'
    if (target / 'protocol.json').exists():
        verify_open(target)
        return
    qs = open_questions()
    if args.method == 'hippomm':
        memory = args.output_root / 'hippomm/offline'
        hip_original.verify_memory(memory)
        images = read(memory / 'representative_images.jsonl')
        from scripts.evaluation import memeye_seed1024_open as shared_open
        files = [OPEN, Path(__file__), Path(shared_open.__file__), Path(open_base.__file__), Path(hip.__file__),
            Path(hip_original.__file__), Path(hip.core.__file__), Path(runtime.__file__)]
        files += [memory / n for n in ('memory.json', 'events.jsonl', 'dialogues.jsonl',
            'image_mapping.jsonl', 'visual_embeddings.npz', 'representative_images.jsonl')]
        files += [memory / i['file'] for i in images]
        write(target / 'questions.jsonl', qs)
        protocol = dict(method_id='hippomm', dataset='MemEye Open', answer_model=open_base.MODEL,
            questions=371, memory_dir=str(memory), seed=SEED, do_sample=False,
            confidence_threshold=.75, fallback_threshold=.4, fallback_score=.6,
            max_new_tokens=1024, context_budget=120000, image_max_pixels=1638400,
            top_k=5, source_mcq_correct=179, source_mcq_correct_note='Seed-42 reference only; no old answers reused.',
            source_sha256={str(p): sha(p) for p in files}, questions_sha256=sha(target / 'questions.jsonl'))
    else:
        import numpy as np
        import torch
        from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
        source = clip.output_dir(cfg)
        protocol, _ = clip.verify(source)
        candidates = read(source / 'image_candidates.jsonl')
        with np.load(source / 'clip_embeddings.npz') as data:
            vectors = data['image_vectors'].copy()
            if data['image_ids'].tolist() != [c['image_id'] for c in candidates]:
                raise ValueError('CLIP candidate/vector alignment mismatch')
        dialogues, source_dialogues = dialogue.source_dialogues(NATURAL, {q['task'] for q in qs})
        checkpoint = ROOT / 'models/clip-vit-base-patch32'
        torch.set_num_threads(8)
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model, loading = CLIPTextModelWithProjection.from_pretrained(checkpoint,
            config=config.text_config, local_files_only=True, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Incomplete CLIP text projection checkpoint')
        # The seed-42 Open adapter encoded text on CPU; retain that boundary.
        model.to(device='cpu', dtype=torch.float32).eval()
        limit = config.text_config.max_position_embeddings
        texts = [q['question'] for q in qs]
        lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]
        query_vectors = []
        for start in range(0, len(texts), 32):
            encoded = tokenizer(texts[start:start + 32], padding=True, truncation=True,
                max_length=limit, return_tensors='pt').to('cpu')
            with torch.inference_mode():
                value = model(**encoded, return_dict=True).text_embeds.double().cpu().numpy()
            query_vectors.extend(value / np.linalg.norm(value, axis=1, keepdims=True))
        system = ('You answer questions about remembered images. Use the retrieved historical '
            'images and their accompanying dialogue and any query image provided. '
            'Return a concise natural-language answer. If the evidence is insufficient, say so.')
        manifest, traces = [], []
        for q, vector, length in zip(qs, query_vectors, lengths):
            retrieved = clip.rank_images(q, candidates, vectors, vector, 5)
            content = [dict(type='text', text='Retrieved historical images (ranked by relevance):')]
            for i, item in enumerate(retrieved, 1):
                text = '\n\n'.join(dialogues[mid]['text'] for mid in item['source_memory_ids'])
                content += [dict(type='text', text=f'Historical image {i}:'),
                    dict(type='image', path=item['path']),
                    dict(type='text', text=f'Dialogue associated with historical image {i}:\n' + text)]
            for i, path in enumerate(q['query_image_paths'], 1):
                content += [dict(type='text', text=f'Question image {i} (part of the question):'), dict(type='image', path=path)]
            content.append(dict(type='text', text='Question: ' + q['question'] + '\nReturn a concise natural-language answer.'))
            ids = list(dict.fromkeys(mid for item in retrieved for mid in item['source_memory_ids']))
            trace = dict(id=q['id'], task=q['task'], retrieval_query=q['question'],
                query_token_count=length, query_truncated=length > limit, retrieved_images=retrieved, top_memory_ids=ids)
            manifest.append({**q, 'method_id': args.method, 'top_memory_ids': ids, 'retrieval_trace': trace,
                'max_new_tokens': 512, 'image_max_pixels': 1638400,
                **open_base.payload([dict(role='system', content=system), dict(role='user', content=content)])})
            traces.append(trace)
        write(target / 'reader_manifest.jsonl', manifest)
        write(target / 'retrieval.jsonl', traces)
        from scripts.evaluation import memeye_seed1024_open as shared_open
        files = [OPEN, Path(__file__), Path(shared_open.__file__), Path(open_base.__file__), Path(clip.__file__),
            source / 'protocol.json', source / 'clip_embeddings.npz', source / 'image_candidates.jsonl', *source_dialogues]
        files += [p for p in checkpoint.iterdir() if p.is_file()]
        protocol = dict(method_id=args.method, display_label='VisRAG', dataset='MemEye Open',
            answer_model=open_base.MODEL, questions=371, top_k=5, seed=SEED, do_sample=False,
            max_new_tokens=512, image_max_pixels=1638400,
            source_mcq_correct=193, source_mcq_correct_note='Seed-42 reference only; no old answers reused.',
            query_truncation=dict(limit=limit, count=sum(n > limit for n in lengths)),
            source_sha256={str(p): sha(p) for p in files},
            artifact_sha256={n: sha(target / n) for n in ('reader_manifest.jsonl', 'retrieval.jsonl')})
        del model
        gc.collect()
    protocol.update(evaluator_seed=42, fresh_seed1024_memory=True)
    frozen_json(target / 'protocol.json', protocol)
    verify_open(target)
    print(json.dumps(dict(method=args.method, phase='open-prepare', questions=371, seed=SEED)), flush=True)


def verify_open(target):
    p = json.loads((target / 'protocol.json').read_text())
    if p['seed'] != SEED or p['evaluator_seed'] != 42:
        raise ValueError('Open seed contract mismatch')
    for name, expected in p['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Open input changed: ' + name)
    for name, expected in p.get('artifact_sha256', {}).items():
        if sha(target / name) != expected:
            raise ValueError('Open payload changed')
    if 'questions_sha256' in p and sha(target / 'questions.jsonl') != p['questions_sha256']:
        raise ValueError('Open questions changed')
    return p


class OpenGenerator:
    """Shared native transport with both actual RNG boundaries set to 1024."""
    def __init__(self, device, output):
        from scripts.evaluation.memeye_seed1024_open import SeededMultimodalQwen
        self.model = SeededMultimodalQwen(device, seed=SEED)
        dump(output / 'generation_environment.json', self.model.environment)

    def __call__(self, messages, max_new_tokens):
        return self.model.generate([messages], max_new_tokens)[0]


def open_run(args):
    from scripts.evaluation import memeye_seed1024_open as shared
    verify_preflight(args)
    target = args.output_root / args.method / 'open'
    verify_open(target)
    if args.method == 'clip_image_dialogue':
        result = shared.run_manifest(target, args.method, args.device)
        if result['status'] != 'complete':
            raise ValueError('Incomplete CLIP Open answers')
        shared.collect(target)
        return
    generate = OpenGenerator(args.device, target)
    open_base.run_hippomm_open(generate, target)
    values = read(target / 'responses.jsonl') if (target / 'responses.jsonl').exists() else []
    ids = {q['id'] for q in open_questions()}
    if len(values) != len({r['id'] for r in values}) or not {r['id'] for r in values} <= ids:
        raise ValueError('Open response ID mismatch')
    complete = len(values) == 371 and {r['id'] for r in values} == ids
    status = dict(status='complete' if complete else 'partial', questions=371,
        answered=len(values), seed=SEED, evaluator_seed=42,
        missing_ids=sorted(ids - {r['id'] for r in values}),
        responses_sha256=sha(target / 'responses.jsonl') if values else None,
        token_limit_hits=sum(r['hit_token_limit'] for r in values),
        empty_answers=sum(not r['raw_response'] for r in values))
    dump(target / 'coverage_audit.json', status)
    if not complete:
        raise ValueError('Incomplete Open answers; inspect raw structured-stage errors')
    # The generic fixed-seed judge collector consumes this exact fresh output.
    write(target / 'qa_responses.jsonl', values)
    status['responses_sha256'] = sha(target / 'qa_responses.jsonl')
    dump(target / 'completion.json', status)
    shared.collect(target, hippo=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('clip_image_dialogue', 'hippomm'), required=True)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze', 'open-prepare', 'open-run'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    patch_runtime_seed()
    with (args.output_root / (args.method + '.lock')).open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'prepare': prepare, 'run': run_mcq, 'analyze': analyze,
         'open-prepare': open_prepare, 'open-run': open_run}[args.phase](args)


if __name__ == '__main__':
    main()
