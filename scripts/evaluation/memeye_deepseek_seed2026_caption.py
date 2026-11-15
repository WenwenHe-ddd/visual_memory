#!/usr/bin/env python3
"""Isolated, resumable MemEye seed2026 caption baselines.

Only historical DeepSeek seed42 captions and the original MCQ option statements
are frozen. Ward memories, vector retrieval, Open hypotheses, MCQ/Open answers
are produced in this new run; the five-grade evaluator remains at seed42.
The two requested methods share a locked fresh construction/retrieval stage.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sys
import time
import types
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.memory import memeye_backbone_retrieval as native

SEED = 2026
MODEL = 'deepseek-v4.1-flash'
RUN = ROOT / 'results/memeye_deepseek_seed2026_full_20261113'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
TRANSPORT = 'scripts.evaluation.memeye_deepseek_seed2026_transport'
METHODS = {'semrag': ('Semantic RAG (caption baseline)', 'baseline_question_only'),
           'ours_1_ours_2': ('ours_1 + ours_2 (lambda=0.03, alpha=1)', 'master_answer_query_zscore')}
OLD_MAIN = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_ours.py'
OLD_OPEN = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_ours_open.py'
OLD_READER = ROOT / 'scripts/evaluation/memeye_deepseek_open_reader.py'
read, rows, sha, digest = native.read, native.lines, native.sha, native.digest


def dump(path, value):
    """Atomic, fsynced publication; concurrent methods never share temp names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact differs: ' + str(path))
    else:
        dump(path, value)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def clone(path, name, old_seed=1024):
    """Clone signed implementation in memory; signed source files stay untouched."""
    allowed_seed_lines = {
        OLD_MAIN: {40, 159, 313, 326, 397, 409},
        OLD_OPEN: {70, 94, 117, 130, 232, 256, 278, 295, 329, 331},
        OLD_READER: {150, 202},
    }
    original_tree = ast.parse(path.read_text())
    found = {n.lineno for n in ast.walk(original_tree) if isinstance(n, ast.Constant)
             and type(n.value) is int and n.value == old_seed}
    if found != allowed_seed_lines[path]:
        raise ValueError('Seed literal locations changed; review semantics before cloning: ' + str(path))
    class Seed(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == old_seed:
                return ast.copy_location(ast.Constant(SEED), node)
            if isinstance(node.value, str):
                value = node.value.replace(f'seed{old_seed}', f'seed{SEED}')
                if value != node.value:
                    return ast.copy_location(ast.Constant(value), node)
            return node
    tree = ast.fix_missing_locations(Seed().visit(original_tree))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = 'scripts.evaluation'
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def api_call(output, payload, cap, *, env_file=ENV):
    """Exact seed2026 wire cache with durable received-output attempt records."""
    import importlib
    api = importlib.import_module(TRANSPORT)
    request = api.build_request(payload, model=MODEL, max_output_tokens=cap, image_detail='high')
    if request.get('seed') != SEED or request.get('model') != MODEL or request.get('enable_thinking') is not False:
        raise ValueError('Actual transmitted seed/model/settings differ')
    key = api.request_sha256(request)
    output = Path(output)
    path = output / 'api_calls' / (key + '.json')
    attempt_dir = output / 'api_attempts' / key
    with lock(output / 'api_locks' / (key + '.lock')):
        if path.exists():
            saved = read(path)
            if saved['request'] != request or saved['request_sha256'] != key:
                raise ValueError('Exact-wire cache differs')
            return key, saved['response']
        previous = [read(p) for p in sorted(attempt_dir.glob('attempt-*.json'))]
        if previous and previous[-1]['status'] in ('ok', 'provider_input_rejected'):
            response = previous[-1]['response']
            freeze(path, {'request': request, 'request_sha256': key, 'response': response,
                          'attempts': previous, 'recovered_after_received_output': True})
            return key, response
        if previous and (len(previous) >= 4 or not previous[-1].get('retryable')):
            raise RuntimeError('Saved API failure exhausted original four-attempt policy: ' + key)
        config = api.load_config(env_file)
        for number in range(len(previous) + 1, 5):
            event = {'attempt': number, 'request_sha256': key, 'seed': SEED}
            try:
                response = api.chat_request(config, request, timeout=120)
                response = {**response, 'seed': SEED, 'status': 'ok'}
                event.update(status='ok' if response['returned_model'] == MODEL else 'model_mismatch',
                             response=response, retryable=False)
                freeze(attempt_dir / f'attempt-{number:02d}.json', event)
                if response['returned_model'] != MODEL:
                    raise ValueError('Provider returned another model')
            except api.APIError as error:
                kind = getattr(error, 'provider_error_code', None) or getattr(error, 'kind', type(error).__name__)
                rejected = kind in ('data_inspection_failed', 'input_inspection_failed',
                                    'provider_input_rejected', 'content_filter')
                if rejected:
                    response = {'status': 'provider_input_rejected', 'raw_response': None,
                                'returned_model': None, 'seed': SEED, 'error_kind': kind}
                    event.update(status='provider_input_rejected', response=response, retryable=False)
                else:
                    event.update(status='error', kind=kind, retryable=bool(getattr(error, 'retryable', False)))
                freeze(attempt_dir / f'attempt-{number:02d}.json', event)
                previous.append(event)
                if rejected:
                    break
                if not event['retryable'] or number == 4:
                    raise
                time.sleep(min(2 ** number, 8))
                continue
            previous.append(event)
            break
        freeze(path, {'request': request, 'request_sha256': key, 'response': response,
                      'attempts': previous, 'recovered_after_received_output': False})
        return key, response


@lru_cache(maxsize=4)
def runtime(run_root):
    import importlib
    api = importlib.import_module(TRANSPORT)
    if api.MODEL != MODEL or api.SEED != SEED:
        raise ValueError('Seed2026 transport is required')
    run_root = Path(run_root).resolve()
    main = clone(OLD_MAIN, 'memeye_deepseek_2026_caption_base')
    main.OUTPUT = run_root / 'ours'
    main.TRANSPORT = TRANSPORT
    main.api_call = api_call
    main.dump = dump
    main.freeze = freeze
    opened = clone(OLD_OPEN, 'memeye_deepseek_2026_caption_open')
    opened.main = main
    opened.OUTPUT = main.OUTPUT / 'open'
    opened.dump = dump
    reader = clone(OLD_READER, 'memeye_deepseek_2026_caption_reader', old_seed=42)
    def settings(model=MODEL):
        api.canonical_model(model)
        return dict(model=model, max_tokens=2048, temperature=0, seed=SEED,
                    enable_thinking=False, n=1, stream=False)
    reader.api = SimpleNamespace(MODEL=MODEL, settings=settings, chat=api.chat_request,
        load_config=api.load_config, APIError=api.APIError, vision=api.original, __file__=api.__file__)
    reader.text_api = api
    reader.RUN = run_root
    reader.code_files = lambda: [Path(__file__), Path(api.__file__), OLD_MAIN, OLD_OPEN, OLD_READER,
        Path(reader.common.__file__), Path(api.original.__file__),
        ROOT / 'scripts/evaluation/openai_chat_transport.py']
    return main, opened, reader, api


def prepare(run_root, method, device='cuda:0', workers=4):
    run_root = Path(run_root).resolve()
    main, opened, reader, api = runtime(str(run_root))
    with lock(run_root / 'ours/common.lock'):
        parent = main.prepare(main.OUTPUT, device=device, workers=workers)
        opened.prepare(opened.OUTPUT)
        code = [Path(__file__), OLD_MAIN, OLD_OPEN, OLD_READER, Path(api.__file__),
                Path(api.original.__file__), ROOT / 'scripts/evaluation/memeye_deepseek_open_judge.py',
                ROOT / 'scripts/evaluation/memeye_open_judge_engine.py',
                ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_open.py']
        contract = {'dataset': 'MemEye', 'method': method, 'label': METHODS[method][0],
            'model': MODEL, 'seed': SEED, 'caption_seed': 42, 'judge_seed': 42,
            'questions': 371, 'top_k': 5, 'lambda': .03, 'alpha': 1.,
            'device': device, 'workers': workers,
            'new_memory_and_retrieval': True, 'historical_llm_results_reused': False,
            'mcq_candidate_policy': 'Original fixed four option statements, shared between backbones and seeds.',
            'open_candidate_policy': 'Fresh seed2026 four hypotheses from Open question + frozen query image caption.',
            'caption_policy': 'Original DeepSeek seed42 image captions; no new caption calls.',
            'source_sha256': {str(p): sha(p) for p in [*code, main.OUTPUT / 'protocol.json',
                opened.OUTPUT / 'preparation_protocol.json']}}
        protocol = {'contract': contract, 'experiment_signature': digest(contract),
                    'source_run_id': parent['run_id']}
        freeze(run_root / 'methods' / method / 'protocol.json', protocol)
    return protocol


def verify_preparation(run_root, method):
    protocol = read(Path(run_root) / 'methods' / method / 'protocol.json')
    if digest(protocol['contract']) != protocol['experiment_signature']:
        raise ValueError('Method preparation signature differs')
    if (protocol['contract']['seed'], protocol['contract']['model']) != (SEED, MODEL):
        raise ValueError('Method seed/model differs')
    native.verify_hashes(protocol['contract']['source_sha256'])
    return protocol


def common(run_root, device, workers):
    main, opened, reader, api = runtime(str(run_root))
    marker = run_root / 'ours/common_completion.json'
    with lock(run_root / 'ours/common.lock'):
        if marker.exists():
            proof = read(marker)
            if proof['seed'] != SEED or proof['status'] != 'complete':
                raise ValueError('Shared completion belongs to another run')
            native.verify_hashes(proof['artifact_sha256'])
            return proof
        parent = read(main.OUTPUT / 'protocol.json')
        main.verify(parent)
        main.rebuild(main.OUTPUT, parent)
        main.retrieve(main.OUTPUT, parent)
        if not (opened.OUTPUT / 'candidate_validation.json').exists():
            opened.candidates(opened.OUTPUT, workers=workers)
        else:
            candidate_proof = read(opened.OUTPUT / 'candidate_validation.json')
            if (candidate_proof['seed'] != SEED or candidate_proof['candidates_sha256'] !=
                    sha(opened.OUTPUT / 'candidates.jsonl')):
                raise ValueError('Fresh Open hypotheses differ')
        opened.retrieve(opened.OUTPUT, device=device)
        paths = [main.OUTPUT / n for n in ('protocol.json', 'memory_manifest.json',
                 'retrieval_audit.json', 'retrieval_results.jsonl', 'reader/requests.jsonl')]
        paths += [opened.OUTPUT / n for n in ('preparation_protocol.json', 'candidate_generation_protocol.json',
                  'candidates.jsonl', 'candidate_validation.json', 'retrieval_validation.json')]
        for method in METHODS:
            paths += [opened.OUTPUT / method / n for n in ('protocol.json', 'reader_manifest.jsonl', 'retrieval.jsonl')]
        proof = {'status': 'complete', 'seed': SEED, 'caption_seed': 42,
                 'historical_graphs_vectors_candidates_reused': False, 'open_hypotheses': 1484,
                 'mcq_option_statements': 'Original frozen MCQ option statements, no LLM generation.',
                 'artifact_sha256': {str(p): sha(p) for p in paths}}
        dump(marker, proof)
        return proof


def mcq(run_root, method, workers):
    main, opened, reader, api = runtime(str(run_root))
    out = run_root / 'methods' / method
    condition = METHODS[method][1]
    requests = [r for r in rows(main.OUTPUT / 'reader/requests.jsonl') if r['condition'] == condition]
    if len(requests) != 371 or len(native.unique(requests, 'question_id')) != 371:
        raise ValueError('MCQ cohort is incomplete')
    assert len(requests) == 371 and len(native.unique(requests, 'question_id')) == 371
    native.write_lines(out / 'mcq/requests.jsonl', requests)
    base_protocol = read(main.OUTPUT / 'protocol.json')
    method_protocol = verify_preparation(run_root, method)
    contract = {'seed': SEED, 'model': MODEL, 'method': method, 'condition': condition,
        'questions': 371, 'max_tokens': 16, 'temperature': 0, 'enable_thinking': False,
        'source_signature': base_protocol['experiment_signature'],
        'method_signature': method_protocol['experiment_signature'],
        'source_sha256': {str(p): sha(p) for p in [out / 'mcq/requests.jsonl',
            main.OUTPUT / 'retrieval_results.jsonl', main.OUTPUT / 'retrieval_audit.json', Path(__file__)]}}
    freeze(out / 'mcq/protocol.json', {'contract': contract, 'experiment_signature': digest(contract)})
    results = {}
    def one(request):
        return api_call(main.OUTPUT, {k: request[k] for k in ('system', 'user_text', 'query_images')}, 16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, request): request['question_id'] for request in requests}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            if len(results) % 20 == 0 or len(results) == 371:
                print(json.dumps({'stage': 'mcq', 'method': method, 'completed': len(results), 'total': 371}), flush=True)
    gold = native.unique(rows(main.OUTPUT / 'inputs/gold.jsonl'))
    responses = []
    for request in requests:
        key, response = results[request['question_id']]
        raw = response.get('raw_response')
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
        choice = match.group(1).upper() if match else None
        responses.append({**response, 'id': request['question_id'], 'question_id': request['question_id'],
            'task': request['task'], 'method': method, 'condition': condition,
            'record_id': request['record_id'], 'predicted_choice': choice,
            'gold_choice': gold[request['question_id']]['gold_choice'],
            'correct': choice == gold[request['question_id']]['gold_choice'],
            'request_sha256': key, 'payload_sha256': request['payload_sha256'],
            'experiment_signature': digest(contract), 'historical_answer_reuse': False})
    native.write_lines(out / 'mcq/responses.jsonl', responses)
    correct = sum(r['correct'] for r in responses)
    report = {'status': 'complete', 'seed': SEED, 'method': method, 'questions': 371,
              'correct': correct, 'accuracy': correct / 371,
              'artifact_sha256': {str(p): sha(p) for p in [out / 'mcq/protocol.json', out / 'mcq/requests.jsonl',
                                                         out / 'mcq/responses.jsonl']}}
    dump(out / 'mcq/completion.json', report)
    return report


def open_answers(run_root, method, workers, reference_correct):
    main, opened, reader, api = runtime(str(run_root))
    out = opened.OUTPUT / method
    args = SimpleNamespace(root=out, method_id=method, reference_correct=reference_correct,
        cache=run_root / 'caption_answer_api_cache', workers=workers, env_file=ENV, limit=0)
    reader.run(args)
    reader.collect(args)
    return read(out / 'reader/completion.json')


def judge(run_root, method, workers):
    from scripts.evaluation import memeye_deepseek_open_judge as original
    out = run_root / 'ours/open' / method
    args = SimpleNamespace(input=out / 'judge_inputs.jsonl', output=out / 'judge',
        method_id=method, method_label=METHODS[method][0], source_protocol=out / 'protocol.json',
        cache_root=[], env_file=ENV, workers=workers, max_attempts=5, limit=0)
    protocol, requests, question_map = original.prepare(args)
    if protocol['seed'] != 42:
        raise ValueError('Judge must remain seed42')
    report = original.base.run(args, protocol, requests, question_map)
    original.publish_summary(args, protocol, report)
    if report['status'] != 'complete':
        raise ValueError('Judge coverage incomplete; original outputs retained')
    return report


def audit(run_root, method):
    run_root = Path(run_root).resolve()
    main, opened, reader, api = runtime(str(run_root))
    out = run_root / 'methods' / method
    preparation = verify_preparation(run_root, method)
    shared = read(main.OUTPUT / 'common_completion.json')
    native.verify_hashes(shared['artifact_sha256'])
    native.verify_hashes(read(main.OUTPUT / 'memory_manifest.json')['output_sha256'])
    candidate_protocol = read(opened.OUTPUT / 'candidate_generation_protocol.json')
    assert digest(candidate_protocol['contract']) == candidate_protocol['experiment_signature']
    assert candidate_protocol['contract']['seed'] == SEED
    native.verify_hashes(candidate_protocol['contract']['source_sha256'])
    sanitized = native.unique(rows(opened.OUTPUT / 'inputs/sanitized_candidate_inputs.jsonl'))
    generated = native.unique(rows(opened.OUTPUT / 'candidates.jsonl'))
    assert len(generated) == 371 and set(generated) == set(sanitized)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(opened.candidates_core.CLIP), local_files_only=True)
    for qid, hypothesis in generated.items():
        assert hypothesis['seed'] == SEED and not hypothesis['gold_or_history_used']
        assert hypothesis['source_input_sha256'] == digest(sanitized[qid])
        assert hypothesis['experiment_signature'] == candidate_protocol['experiment_signature']
        payload = {'system': opened.candidates_core.CANDIDATE_SYSTEM,
            'user_text': opened.candidates_core.candidate_user_text(sanitized[qid], hypothesis['format_attempt']),
            'query_images': []}
        request = api.build_request(payload, model=MODEL, max_output_tokens=2048, image_detail='high')
        key = api.request_sha256(request)
        cached = read(main.OUTPUT / 'api_calls' / (key + '.json'))
        assert key == hypothesis['request_sha256'] and cached['request'] == request
        assert request['seed'] == cached['response']['seed'] == SEED
        assert cached['response']['status'] == 'ok' and cached['response']['returned_model'] == MODEL
        parsed, lengths, fmt = opened.candidates_core.parse_candidates(
            cached['response']['raw_response'], tokenizer, return_format=True)
        assert parsed == hypothesis['candidate_statements'] and lengths == hypothesis['clip_token_lengths']
        assert fmt == hypothesis['response_format']
    report = read(out / 'mcq/completion.json')
    native.verify_hashes(report['artifact_sha256'])
    mcq_protocol = read(out / 'mcq/protocol.json')
    native.verify_hashes(mcq_protocol['contract']['source_sha256'])
    assert digest(mcq_protocol['contract']) == mcq_protocol['experiment_signature']
    responses = rows(out / 'mcq/responses.jsonl')
    expected = native.unique(rows(main.OUTPUT / 'inputs/gold.jsonl'))
    requests = native.unique(rows(out / 'mcq/requests.jsonl'), 'question_id')
    assert len(responses) == len(native.unique(responses)) == len(requests) == 371
    assert set(native.unique(responses)) == set(expected) == set(requests)
    correct = 0
    for response in responses:
        qid = response['id']
        source = requests[qid]
        body = api.build_request({k: source[k] for k in ('system', 'user_text', 'query_images')},
                                model=MODEL, max_output_tokens=16, image_detail='high')
        key = api.request_sha256(body)
        cache = read(main.OUTPUT / 'api_calls' / (key + '.json'))
        assert cache['request'] == body and body['seed'] == SEED
        assert response['seed'] == SEED and response['request_sha256'] == key
        assert response['payload_sha256'] == source['payload_sha256']
        assert cache['response'].get('raw_response') == response.get('raw_response')
        assert response['experiment_signature'] == mcq_protocol['experiment_signature']
        raw = response.get('raw_response')
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
        choice = match.group(1).upper() if match else None
        assert response['predicted_choice'] == choice and response['correct'] == (choice == expected[qid]['gold_choice'])
        correct += choice == expected[qid]['gold_choice']
    assert correct == report['correct'] and report['status'] == 'complete'
    open_root = opened.OUTPUT / method
    manifest = rows(open_root / 'reader_manifest.jsonl')
    open_protocol = read(open_root / 'reader/protocol.json')
    open_rows = reader.validate(open_root, open_protocol, manifest)
    seal = read(open_root / 'reader/completion.json')
    assert seal['status'] == 'complete' and seal['questions_answered'] == len(open_rows) == 371
    assert seal['responses_sha256'] == sha(open_root / 'reader/responses.jsonl')
    assert all(r['seed'] == SEED and r['returned_model'] == MODEL for r in open_rows)
    judged = read(open_root / 'judge/summary.json')
    judge_protocol = read(open_root / 'judge/protocol.json')
    assert judge_protocol['seed'] == 42 and judged['status'] == 'complete'
    assert judged['questions_scored'] == 371 and judged['questions_missing'] == 0
    native.verify_hashes(judged.get('artifact_sha256', {}), base=open_root / 'judge')
    scores = native.unique(rows(open_root / 'judge/per_question.jsonl'))
    reader_map = native.unique(open_rows)
    open_questions = native.unique(manifest)
    canonical = native.unique(rows(opened.CANONICAL_GOLD))
    assert set(scores) == set(reader_map) == set(open_questions) == set(expected)
    assert all(scores[qid]['model_output'] == reader_map[qid]['raw_response']
               and scores[qid]['question'] == open_questions[qid]['question']
               and scores[qid]['ground_truth'] == canonical[qid]['answer']
               and scores[qid]['score'] in (0, .25, .5, .75, 1) for qid in scores)
    assert abs(sum(r['score'] for r in scores.values()) / 371 - judged['llm_judge']) < 1e-12
    retrieval = read(main.OUTPUT / 'retrieval_audit.json')['metrics'][METHODS[method][1]]
    source_paths = [out / 'protocol.json', out / 'mcq/protocol.json', out / 'mcq/responses.jsonl',
        out / 'mcq/completion.json', main.OUTPUT / 'common_completion.json',
        open_root / 'reader/protocol.json', open_root / 'reader/responses.jsonl',
        open_root / 'reader/completion.json', open_root / 'judge/summary.json',
        open_root / 'judge/protocol.json', open_root / 'judge/per_question.jsonl']
    original_questions = native.unique(rows(main.OUTPUT / 'inputs/questions.jsonl'))
    mcq_records = [{**r, 'question': original_questions[r['id']]['question'],
        'native_outcome': (r['status'] if r['status'] != 'ok' else
            'invalid_response' if r['predicted_choice'] is None else 'ok'),
        'source_record_id': r['record_id'], 'source_file': str(out / 'mcq/responses.jsonl'),
        'source_sha256': sha(out / 'mcq/responses.jsonl')} for r in responses]
    open_records = [{**r, 'question': open_questions[r['id']]['question'],
        'ground_truth': canonical[r['id']]['answer'], 'model_output': r['raw_response'],
        'judge_score': scores[r['id']]['score'], 'judge_seed': 42,
        'source_record_id': r['id'], 'source_file': str(open_root / 'reader/responses.jsonl'),
        'source_sha256': sha(open_root / 'reader/responses.jsonl'),
        'judge_source_file': str(open_root / 'judge/per_question.jsonl'),
        'judge_source_sha256': sha(open_root / 'judge/per_question.jsonl')} for r in open_rows]
    native.write_lines(run_root / 'evaluations' / method / 'mcq_records.jsonl', mcq_records)
    native.write_lines(run_root / 'evaluations' / method / 'open_records.jsonl', open_records)
    record_paths = [run_root / 'evaluations' / method / n for n in ('mcq_records.jsonl', 'open_records.jsonl')]
    metrics = {'status': 'complete', 'passed': True, 'dataset': 'MemEye', 'method': method,
        'model': MODEL, 'seed': SEED, 'caption_seed': 42, 'judge_seed': 42, 'questions': 371,
        'correct': correct, 'accuracy': correct / 371, 'em': correct / 371,
        'gold_hits_top5': retrieval['gold_hits_top5'], 'gold_references': retrieval['gold_references'],
        'recall_at5_micro': retrieval['recall_at5_micro'], 'llm_judge': judged['llm_judge'],
        'open_questions': 371, 'judge_questions': 371, 'historical_answers_reused': 0,
        'source_sha256': {str(p): sha(p) for p in [*source_paths, *record_paths]},
        'method_experiment_signature': preparation['experiment_signature']}
    dump(run_root / 'evaluations' / method / 'metrics.json', metrics)
    completion = {**metrics, 'completion_time_utc': datetime.now(timezone.utc).isoformat(),
        'metrics_path': str(run_root / 'evaluations' / method / 'metrics.json'),
        'metrics_sha256': sha(run_root / 'evaluations' / method / 'metrics.json')}
    dump(out / 'completion.json', completion)
    print(json.dumps({'stage': 'audited', 'method': method, 'correct': correct,
                      'questions': 371, 'llm_judge': metrics['llm_judge']}), flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit'), required=True)
    parser.add_argument('--method', choices=tuple(METHODS), required=True)
    parser.add_argument('--run-root', type=Path, default=RUN)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--judge-workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.judge_workers < 1:
        parser.error('Worker counts must be positive')
    run_root = args.run_root.resolve()
    with lock(run_root / 'methods' / args.method / 'execution.lock'):
        if args.phase == 'prepare':
            result = prepare(run_root, args.method, args.device, args.workers)
        elif args.phase == 'audit':
            result = audit(run_root, args.method)
        else:
            prepare(run_root, args.method, args.device, args.workers)
            common(run_root, args.device, args.workers)
            scored = mcq(run_root, args.method, args.workers)
            open_answers(run_root, args.method, args.workers, scored['correct'])
            judge(run_root, args.method, args.judge_workers)
            result = audit(run_root, args.method)
        print(json.dumps({'phase': args.phase, 'method': args.method, 'seed': SEED,
                          'status': result.get('status', 'prepared')}), flush=True)


if __name__ == '__main__':
    main()
