"""Fresh seed-2026 DeepSeek VisRAG answers on the fixed MemEye cohort.

Only deterministic CLIP features and original image/dialogue data are shared.
MCQ/Open API caches and answers are new. Native reader functions are compiled
with explicitly identified seed nodes changed; output caps and byte units stay
unchanged. The five-grade Open evaluator retains its own seed 42.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.evaluation import memeye_deepseek_seed1024_visual as original
from scripts.evaluation import memeye_deepseek_seed2026_transport as api
from scripts.evaluation import memeye_deepseek_seed1024_open as judge_driver
from scripts.evaluation import memeye_seed1024_visual_metrics as memory_metrics
from scripts.memory import clip_image_memory as clip, clip_image_dialogue as dialogue

SEED = 2026
METHOD = 'clip_image_dialogue'
DEFAULT = ROOT / 'results/memeye_deepseek_seed2026_full_20261113'
REFERENCE = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue'
OPEN_REFERENCE = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/visual_methods/clip_image_dialogue/open'
BENCHMARK, OPEN, NATURAL = original.BENCHMARK, original.OPEN, original.NATURAL
read = original.read
digest = original.digest


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if load(path) != value:
            raise ValueError('Frozen seed2026 artifact differs: ' + str(path))
        return
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def frozen(path, value):
    dump(path, value, frozen=True)


def write(path, values):
    path = Path(path)
    data = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in values)
    if path.exists():
        if path.read_text() != data:
            raise ValueError('Frozen seed2026 rows differ: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def append(path, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def paths(args):
    visual = args.run_root / 'visual_methods'
    return SimpleNamespace(visual=visual, method=visual / METHOD,
        index=visual / 'clip_full/top5/clip_image',
        mcq_input=visual / 'clip_full/top5/clip_image_dialogue',
        online=visual / METHOD / 'online', opened=visual / METHOD / 'open',
        marker=args.run_root / 'methods' / METHOD,
        evaluations=args.run_root / 'evaluations' / METHOD)


def native_functions(args):
    """Change only seed keywords and a result['seed'] comparison in five functions."""
    selected = {'image_messages', 'api_call', 'clip_run', 'analyze_clip', 'collect'}
    source = ast.parse(Path(original.__file__).read_text())
    nodes = [copy.deepcopy(n) for n in source.body
             if isinstance(n, ast.FunctionDef) and n.name in selected]
    changes = []
    class Seed(ast.NodeTransformer):
        def visit_keyword(self, node):
            self.generic_visit(node)
            if node.arg in ('seed', 'source_seed') and isinstance(node.value, ast.Constant) and node.value.value == 1024:
                changes.append({'line': node.value.lineno, 'kind': 'keyword', 'name': node.arg})
                node.value = ast.copy_location(ast.Constant(SEED), node.value)
            return node
        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Subscript) and isinstance(node.left.slice, ast.Constant)
                    and node.left.slice.value == 'seed'):
                for i, value in enumerate(node.comparators):
                    if isinstance(value, ast.Constant) and value.value == 1024:
                        changes.append({'line': value.lineno, 'kind': 'seed_comparison'})
                        node.comparators[i] = ast.copy_location(ast.Constant(SEED), value)
            return node
    module = ast.fix_missing_locations(Seed().visit(ast.Module(body=nodes, type_ignores=[])))
    if len(changes) != 6:
        raise ValueError('Native reader seed-node inventory changed')
    if any(isinstance(n, ast.Constant) and type(n.value) is int and n.value == 1024 for n in ast.walk(module)):
        raise ValueError('An unidentified seed1024 constant remains in the selected native functions')
    namespace = dict(original.__dict__)
    namespace.update(api=api, append=append, dump=dump, frozen=frozen, write=write,
        old_open=SimpleNamespace(**(dict(original.old_open.__dict__) | {'ENV_FILE': args.env_file})),
        verify=lambda _: verify(args), prepare_clip_mcq=lambda _: prepared_mcq(args))
    exec(compile(module, str(Path(original.__file__)), 'exec'), namespace)
    native_api_call = namespace['api_call']
    def audited_api_call(config, directory, body, max_attempts):
        try:
            return native_api_call(config, directory, body, max_attempts)
        except api.APIError as exc:
            # Keep the real provider error separately; native retry behavior and
            # MCQ rejection accounting are unchanged.
            append(directory / api.request_sha256(body) / 'provider_error_evidence.jsonl',
                dict(request_sha256=api.request_sha256(body), seed=SEED, kind=exc.kind,
                     code=exc.code, provider_error_code=getattr(exc, 'provider_error_code', None),
                     http_status=exc.http_status, retryable=exc.retryable))
            raise
    namespace['api_call'] = audited_api_call
    return namespace, changes


def prepared_mcq(args):
    p = paths(args)
    return p.mcq_input, load(p.mcq_input / 'protocol.json'), read(p.mcq_input / 'qa_manifest.jsonl')


def verify(args):
    p = paths(args)
    protocol = load(p.method / 'stability_protocol.json')
    if protocol['seed'] != SEED or protocol['judge_seed'] != 42 or protocol['caption_seed'] != 42:
        raise ValueError('Answer/caption/Judge seeds differ')
    for name, expected in protocol['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen visual source changed: ' + name)
    for name, expected in protocol['artifact_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen deterministic retrieval payload changed: ' + name)
    return protocol


def open_vectors(questions):
    """Keep the native Open CLIP CPU/fp32, double-normalization and 77-token rule."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    checkpoint = ROOT / 'models/clip-vit-base-patch32'
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, loading = CLIPTextModelWithProjection.from_pretrained(checkpoint,
        config=config.text_config, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Native CLIP text/projection weights are incomplete')
    model.to(device='cpu', dtype=torch.float32).eval()
    limit = config.text_config.max_position_embeddings
    lengths = [len(tokenizer.encode(q['question'], truncation=False)) for q in questions]
    result = []
    for start in range(0, len(questions), 32):
        encoded = tokenizer([q['question'] for q in questions[start:start + 32]], padding=True,
            truncation=True, max_length=limit, return_tensors='pt').to('cpu')
        with torch.inference_mode():
            values = model(**encoded, return_dict=True).text_embeds.double().cpu().numpy()
        result.extend(values / np.linalg.norm(values, axis=1, keepdims=True))
    return np.asarray(result), lengths, limit


def prepare(args):
    import numpy as np
    p = paths(args)
    if (p.method / 'stability_protocol.json').exists():
        return verify(args)
    _, changes = native_functions(args)
    reference_protocol = load(REFERENCE / 'protocol.json')
    source = Path(reference_protocol['frozen_retrieval_source'])
    source_protocol = load(source / 'protocol.json')
    candidates = clip.image_candidates(read(NATURAL / 'memories.jsonl'),
        read(NATURAL / 'images.jsonl'), {q['task'] for q in read(BENCHMARK)})
    if candidates != read(source / 'image_candidates.jsonl') or len(candidates) != 433:
        raise ValueError('Raw historical images differ from the deterministic native CLIP index')
    with np.load(source / 'clip_embeddings.npz', allow_pickle=False) as archive:
        vectors, query_vectors = archive['image_vectors'].copy(), archive['query_vectors'].copy()
        image_ids, query_ids = archive['image_ids'].tolist(), archive['query_ids'].tolist()
    mcq, opened = read(BENCHMARK), original.old_open.questions()
    if (len(mcq) != 371 or len(opened) != 371 or len({q['id'] for q in mcq}) != 371
            or {q['id'] for q in mcq} != {q['id'] for q in opened}
            or query_ids != [q['id'] for q in mcq]
            or image_ids != [c['image_id'] for c in candidates] or vectors.shape != (433, 512)):
        raise ValueError('Native 371-question / 433-image cohort differs')
    dialogues, dialogue_files = dialogue.source_dialogues(NATURAL, {q['task'] for q in mcq})
    if len(dialogues) != 438:
        raise ValueError('Expected 438 exact image-associated dialogue rounds')
    p.index.mkdir(parents=True, exist_ok=True)
    for name in ('image_candidates.jsonl', 'clip_embeddings.npz'):
        shutil.copyfile(source / name, p.index / name)
        if sha(source / name) != sha(p.index / name):
            raise ValueError('Shared deterministic CLIP feature copy differs')
    mcq_reference = {q['id']: q for q in read(REFERENCE / 'qa_manifest.jsonl')}
    mcq_rows = []
    image_system = source_protocol['reader_system_prompt']
    system = image_system.replace('retrieved historical images',
        'retrieved historical images and their accompanying dialogue')
    for q, vector in zip(mcq, query_vectors):
        old = mcq_reference[q['id']]
        row = {k: q[k] for k in ('id', 'task', 'x', 'y', 'question', 'options',
            'gold_choice', 'gold_ids', 'query_image_paths')}
        row.update(retrieval_query=q['question'], query_token_count=old['query_token_count'],
            query_truncated=old['query_truncated'],
            retrieved_images=clip.rank_images(q, candidates, vectors, vector, 5))
        row['top_memory_ids'] = list(dict.fromkeys(mid for item in row['retrieved_images'] for mid in item['source_memory_ids']))
        row['dialogue_by_memory_id'] = dialogue.attach_dialogues(row, dialogues)
        image_only = clip.reader_messages(row, row['retrieved_images'], image_system)
        messages = dialogue.reader_messages(row, system)
        image_paths = [c['path'] for c in messages[1]['content'] if c['type'] == 'image']
        hashes = {path: sha(path) for path in image_paths}
        row.update(reader_messages=messages, reader_image_paths=image_paths, reader_image_hashes=hashes,
            source_payload_sha256=clip.digest(dict(messages=image_only, image_hashes=hashes)),
            payload_sha256=clip.digest(dict(messages=messages, image_hashes=hashes)))
        if row != old:
            raise ValueError('Reconstructed original MCQ retrieval/images/dialogue payload differs: ' + q['id'])
        mcq_rows.append(row)
    write(p.mcq_input / 'qa_manifest.jsonl', mcq_rows)
    source_view = {'seed': SEED, 'questions': 371, 'retrieval_top_k': 5,
        'frozen_retrieval_source': str(p.index), 'source_seed42_protocol_sha256': sha(REFERENCE / 'protocol.json'),
        'manifest_sha256': sha(p.mcq_input / 'qa_manifest.jsonl'), 'reader_system_prompt': system}
    source_view['experiment_signature'] = digest(source_view)
    frozen(p.mcq_input / 'protocol.json', source_view)
    open_queries, lengths, limit = open_vectors(opened)
    query_path = p.index / 'open_query_vectors.npz'
    np.savez(query_path, query_vectors=open_queries, query_ids=np.asarray([q['id'] for q in opened]))
    open_reference = {q['id']: q for q in read(OPEN_REFERENCE / 'reader_manifest.jsonl')}
    open_rows, traces = [], []
    open_system = ('You answer questions about remembered images. Use the retrieved historical '
        'images and their accompanying dialogue and any query image provided. '
        'Return a concise natural-language answer. If the evidence is insufficient, say so.')
    for q, vector, length in zip(opened, open_queries, lengths):
        retrieved = clip.rank_images(q, candidates, vectors, vector, 5)
        content = [{'type': 'text', 'text': 'Retrieved historical images (ranked by relevance):'}]
        for i, item in enumerate(retrieved, 1):
            text = '\n\n'.join(dialogues[mid]['text'] for mid in item['source_memory_ids'])
            content += [{'type': 'text', 'text': f'Historical image {i}:'},
                {'type': 'image', 'path': item['path']},
                {'type': 'text', 'text': f'Dialogue associated with historical image {i}:\n' + text}]
        for i, path in enumerate(q['query_image_paths'], 1):
            content += [{'type': 'text', 'text': f'Question image {i} (part of the question):'}, {'type': 'image', 'path': path}]
        content.append({'type': 'text', 'text': 'Question: ' + q['question'] + '\nReturn a concise natural-language answer.'})
        ids = list(dict.fromkeys(mid for item in retrieved for mid in item['source_memory_ids']))
        trace = dict(id=q['id'], task=q['task'], retrieval_query=q['question'], query_token_count=length,
            query_truncated=length > limit, retrieved_images=retrieved, top_memory_ids=ids)
        row = {**q, 'method_id': METHOD, 'top_memory_ids': ids, 'retrieval_trace': trace,
            'max_completion_tokens': 2048, 'image_max_pixels': 1638400,
            **original.old_open.payload([dict(role='system', content=open_system), dict(role='user', content=content)])}
        old = open_reference[q['id']]
        if any(row[key] != old[key] for key in ('reader_messages', 'reader_image_paths', 'reader_image_hashes', 'payload_sha256', 'top_memory_ids')):
            raise ValueError('Fresh CPU Open CLIP/context differs from the native payload: ' + q['id'])
        if trace != old['retrieval_trace']:
            raise ValueError('Fresh CPU Open ranking/score trace differs: ' + q['id'])
        open_rows.append(row); traces.append(trace)
    write(p.opened / 'reader_manifest.jsonl', open_rows)
    write(p.opened / 'retrieval.jsonl', traces)
    frozen(p.opened / 'protocol.json', dict(method_id=METHOD, dataset='MemEye Open', answer_model=api.MODEL,
        questions=371, top_k=5, seed=SEED, evaluator_seed=42, caption_seed=42,
        max_new_tokens=2048, max_completion_tokens=2048, temperature=0, do_sample=False,
        query_truncation=dict(limit=limit, count=sum(n > limit for n in lengths)),
        query_policy='Official Open question text only; no options, answer, image caption or query image in CLIP retrieval.',
        answer_policy='Exactly five original historical images with all associated raw dialogue, then original question images and question text.',
        reader_manifest_sha256=sha(p.opened / 'reader_manifest.jsonl')))
    code = [Path(__file__), Path(original.__file__), Path(api.__file__), Path(api.original.__file__),
        Path(clip.__file__), Path(dialogue.__file__), Path(original.old_open.__file__),
        Path(judge_driver.__file__), Path(memory_metrics.__file__)]
    inputs = [BENCHMARK, BENCHMARK.parent / 'protocol.json', OPEN, NATURAL / 'protocol.json',
        NATURAL / 'memories.jsonl', NATURAL / 'images.jsonl', source / 'image_candidates.jsonl',
        source / 'clip_embeddings.npz', REFERENCE / 'protocol.json', REFERENCE / 'qa_manifest.jsonl',
        OPEN_REFERENCE / 'reader_manifest.jsonl', *dialogue_files]
    inputs += [path for path in (ROOT / 'models/clip-vit-base-patch32').iterdir() if path.is_file()]
    image_paths = {path for q in mcq_rows + open_rows for path in q['reader_image_paths']}
    inputs += [Path(path) for path in image_paths]
    artifacts = [p.index / 'image_candidates.jsonl', p.index / 'clip_embeddings.npz', query_path,
        p.mcq_input / 'qa_manifest.jsonl', p.mcq_input / 'protocol.json',
        p.opened / 'reader_manifest.jsonl', p.opened / 'retrieval.jsonl', p.opened / 'protocol.json']
    wire = api.build_message_request(native_functions(args)[0]['image_messages'](mcq_rows[0]), max_tokens=16, temperature=0)
    if wire['seed'] != SEED or wire['max_tokens'] != 16 or wire['enable_thinking']:
        raise ValueError('Actual native MCQ wire settings changed')
    for wrong in (42, 1024):
        try:
            api.build_message_request([dict(role='user', content='preflight')], seed=wrong)
        except ValueError:
            pass
        else:
            raise ValueError('Transport accepted a wrong wire seed')
    protocol = dict(method=METHOD, seed=SEED, caption_seed=42, judge_seed=42, model=api.MODEL,
        questions=371, top_k=5, history_images=433, image_associated_dialogue_rounds=438,
        mcq_max_tokens=16, mcq_attempts=3, open_max_tokens=2048, open_attempts=4,
        source_sha256={str(path.resolve()): sha(path) for path in set(code + inputs)},
        artifact_sha256={str(path.resolve()): sha(path) for path in artifacts},
        old_llm_answers_or_memories_imported=0, caption_calls=0, api_calls_during_prepare=0,
        gpu_or_qwen_model_loaded=False, native_seed_ast_changes=changes,
        implementation_note='Screenshot VisRAG label means this controlled CLIP original-image plus raw-dialogue route.',
        deterministic_feature_policy='Original frozen CLIP history/MCQ query features; fresh original CPU Open query encoding and all rankings/payloads reconstructed.')
    frozen(p.method / 'stability_protocol.json', protocol)
    frozen(p.marker / 'preparation.json', dict(status='complete', seed=SEED, method=METHOD,
        questions=371, mcq_payloads_equal_reference=371, open_payloads_equal_reference=371,
        api_calls=0, protocol_sha256=sha(p.method / 'stability_protocol.json')))
    return protocol


def audit_native(args, target, open_task):
    namespace, _ = native_functions(args)
    cap = 2048 if open_task else 16
    manifest = read(target / ('reader_manifest.jsonl' if open_task else 'qa_manifest.jsonl'))
    response_path = target / ('responses.jsonl' if open_task else 'qa_responses.jsonl')
    answers = {row['id']: row for row in read(response_path)}
    if len(answers) != 371 or set(answers) != {q['id'] for q in manifest} or len(read(response_path)) != 371:
        raise ValueError('Native answers require all 371 unique canonical IDs')
    for q in manifest:
        row = answers[q['id']]
        request = api.build_message_request(namespace['image_messages'](q), max_tokens=cap, temperature=0)
        key = api.request_sha256(request)
        cache = target / 'calls' / key
        if (request['seed'] != SEED or row.get('seed') != SEED or row['request_sha256'] != key
                or row['payload_sha256'] != q['payload_sha256']
                or json.loads(gzip.decompress((cache / 'wire.json.gz').read_bytes())) != request):
            raise ValueError('Answer differs from its actual seed2026 image/text wire')
        attempts = read(cache / 'attempts.jsonl')
        if not 1 <= len(attempts) <= (4 if open_task else 3):
            raise ValueError('Original bounded API retry count changed')
        if row.get('status') == 'provider_rejection':
            evidence_path = cache / 'provider_error_evidence.jsonl'
            evidence = read(evidence_path) if evidence_path.exists() else []
            if (open_task or not attempts or attempts[-1]['status'] != 'error' or not evidence
                    or evidence[-1]['provider_error_code'] != 'data_inspection_failed'
                    or evidence[-1]['request_sha256'] != key):
                raise ValueError('Provider rejection lacks its original native error')
        else:
            response = load(cache / 'response.json')
            successes = [record for record in attempts if record['status'] == 'ok']
            if (not successes or successes[-1]['response'] != response or response['returned_model'] != api.MODEL
                    or response['seed'] != SEED or response['raw_response'] != row['raw_response']):
                raise ValueError('Answer lacks the original durable provider response')
    completion = load(target / 'completion.json')
    if completion['status'] != 'complete' or completion['responses_sha256'] != sha(response_path):
        raise ValueError('Native response completion seal differs')
    return manifest, answers


def audit(args):
    protocol = verify(args)
    p = paths(args)
    mcq, answers = audit_native(args, p.online, False)
    opened, open_answers = audit_native(args, p.opened, True)
    from scripts.evaluation import memeye_deepseek_open_judge as judge
    judge_root = p.opened / 'judge'
    judge_protocol, summary = load(judge_root / 'protocol.json'), load(judge_root / 'summary.json')
    if (judge_protocol['seed'] != 42 or summary['status'] != 'complete'
            or summary['questions_scored'] != 371 or summary['questions_missing'] != 0):
        raise ValueError('Fixed seed42 Judge is incomplete')
    requests, mapping = read(judge_root / 'requests.jsonl'), read(judge_root / 'question_map.jsonl')
    judge.load_completed_cache(judge_root, judge.shared.PROMPT.read_text(), judge.load_canonical()[0])
    scores = judge.shared.validated_results(judge_root, requests)
    if len(mapping) != 371 or {r['id'] for r in mapping} != {q['id'] for q in opened}:
        raise ValueError('Judge coverage differs from the fresh Open answers')
    inputs = {q['id']: q for q in read(p.opened / 'judge_inputs.jsonl')}
    by_question = {q['id']: q for q in opened}
    open_records = []
    for row in mapping:
        source = inputs[row['id']]
        score = scores[row['request_sha256']]['score']
        if (row['model_output'] != open_answers[row['id']]['raw_response']
                or source['model_output'] != row['model_output'] or source['source_seed'] != SEED
                or score not in judge.shared.GRADES):
            raise ValueError('Judge was not applied to the original fresh Open answer')
        open_records.append(dict(id=row['id'], task=row['task'], question=row['question'], point=row['point'],
            ground_truth=row['ground_truth'], model_output=row['model_output'], judge_score=score,
            source_record_id=row['id'], source_response_path=str(p.opened / 'responses.jsonl'),
            source_response_sha256=sha(p.opened / 'responses.jsonl'),
            source_judge_path=str(judge_root / 'question_map.jsonl'),
            source_judge_sha256=sha(judge_root / 'question_map.jsonl')))
    mean = sum(row['judge_score'] for row in open_records) / 371
    if abs(mean - summary['llm_judge']) > 1e-12:
        raise ValueError('Judge score does not reproduce from actual raw grades')
    mcq_records = [dict(id=q['id'], task=q['task'], question=q['question'],
        predicted_choice=answers[q['id']]['predicted_choice'], gold_choice=q['gold_choice'],
        correct=answers[q['id']]['predicted_choice'] == q['gold_choice'],
        raw_response=answers[q['id']]['raw_response'],
        native_outcome=('provider_rejection' if answers[q['id']].get('status') == 'provider_rejection'
                        else 'invalid_choice' if answers[q['id']]['predicted_choice'] is None else 'ok'),
        source_record_id=q['id'],
        source_response_path=str(p.online / 'qa_responses.jsonl'),
        source_response_sha256=sha(p.online / 'qa_responses.jsonl')) for q in mcq]
    correct = sum(row['correct'] for row in mcq_records)
    native_summary = load(p.online / 'summary.json')
    if correct != native_summary['correct'] or correct / 371 != native_summary['accuracy']:
        raise ValueError('Native MCQ accuracy does not reproduce')
    source_hashes = dict(protocol['source_sha256'])
    source_hashes.update(protocol['artifact_sha256'])
    native_paths = [p.online / 'qa_responses.jsonl', p.online / 'summary.json', p.online / 'protocol.json',
        p.online / 'completion.json', p.opened / 'responses.jsonl', p.opened / 'completion.json',
        p.opened / 'judge_inputs.jsonl', p.method / 'stability_protocol.json',
        p.method / 'metric_summary.json', judge_root / 'protocol.json', judge_root / 'summary.json',
        judge_root / 'question_map.jsonl', judge_root / 'requests.jsonl']
    native_paths += [judge_root / name for name in ('scores.jsonl', 'attempts.jsonl', 'reused_scores.jsonl') if (judge_root / name).exists()]
    source_hashes.update({str(path): sha(path) for path in native_paths})
    write(p.evaluations / 'mcq_records.jsonl', mcq_records)
    write(p.evaluations / 'open_records.jsonl', open_records)
    metric = load(p.method / 'metric_summary.json')
    metrics = dict(status='complete', method=METHOD, seed=SEED, caption_seed=42, judge_seed=42,
        model=api.MODEL, questions=371, correct=correct, accuracy=correct / 371,
        open_questions=371, judge_questions=371, llm_judge=mean,
        recall_at5_micro=metric['recall_at5_micro'], recall_numerator=metric['recall_numerator'],
        recall_denominator=metric['recall_denominator'], storage_bytes=metric['storage_bytes'], storage_mib=metric['storage_mib'],
        mcq_records_sha256=sha(p.evaluations / 'mcq_records.jsonl'),
        open_records_sha256=sha(p.evaluations / 'open_records.jsonl'), source_sha256=source_hashes)
    frozen(p.evaluations / 'metrics.json', metrics)
    outputs = {str(path): sha(path) for path in (p.evaluations / 'metrics.json', p.evaluations / 'mcq_records.jsonl', p.evaluations / 'open_records.jsonl')}
    frozen(p.marker / 'completion.json', dict(status='complete', method=METHOD, seed=SEED,
        questions=371, open_questions=371, judge_questions=371, output_sha256=outputs,
        native_payload_and_wire_audit=True, original_answer_imports=0,
        protocol_sha256=sha(p.method / 'stability_protocol.json')))
    return metrics


def run(args):
    verify(args)
    p = paths(args)
    native, _ = native_functions(args)
    cfg = SimpleNamespace(output_root=p.visual, workers=args.workers, device='cpu', method=METHOD)
    native['clip_run'](cfg, False)
    native['clip_run'](cfg, True)
    memory_metrics.clip_metrics(SimpleNamespace(visual_root=p.visual, clip_mcq_source=p.online))
    judge_driver.judge(p.opened, METHOD, 'VisRAG (CLIP original images + historical dialogue)',
        env_file=args.env_file, workers=args.workers)
    return audit(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=(METHOD,), default=METHOD)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit'), required=True)
    parser.add_argument('--run-root', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--env-file', type=Path, default=original.old_open.ENV_FILE)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if not args.run_root.is_relative_to(ROOT / 'results') or args.run_root == original.DEFAULT.parent:
        parser.error('Use a new local results root; signed seed1024 results are read-only')
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be in 1..8')
    p = paths(args); p.marker.mkdir(parents=True, exist_ok=True)
    with (p.marker / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare(args) if args.phase == 'prepare' else run(args) if args.phase == 'run' else audit(args)
        print(json.dumps({key: result.get(key) for key in ('status', 'method', 'seed', 'questions', 'correct', 'accuracy', 'llm_judge')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
