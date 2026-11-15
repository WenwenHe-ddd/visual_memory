"""Reader-only API ablation of the frozen CLIP Top5 native-image/dialogue run.

Run with python3 -m scripts.evaluation.clip_image_dialogue_api --model gpt-5
--stage prepare|smoke|run|analyze. No caption generation or retrieval is performed.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import http.client
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

from scripts.memory import clip_image_dialogue as source_api
from scripts.evaluation import openai_gpt5_base_transport as gpt
from scripts.evaluation import deepseek_flash_seed42_transport as deepseek
from scripts.evaluation.gpt5_text_memory_transport import classify_http_error
from scripts.evaluation.openai_chat_transport import APIError, _NoRedirect

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue'
DEFAULT_OUTPUT = ROOT / 'results/visual_memory_pipeline/top5/clip_image_dialogue_api'
DEFAULT_ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
MODELS = ('gpt-5', 'deepseek-v4.1-flash')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def settings(model):
    common = dict(model=model, seed=42, n=1, stream=False)
    if model == 'gpt-5':
        return dict(**common, max_completion_tokens=2048, reasoning_effort='minimal', store=False)
    if model == deepseek.DEEPSEEK_MODEL:
        return dict(**common, max_tokens=16, temperature=0, enable_thinking=False)
    raise ValueError('Unsupported model')


def build_request(messages, image_hashes, model):
    """Accept only reader messages and image hashes; preserve text/image interleaving."""
    if (len(messages) != 2 or messages[0]['role'] != 'system'
            or messages[1]['role'] != 'user' or not isinstance(messages[0]['content'], str)
            or any(set(m) != {'role', 'content'} for m in messages)):
        raise ValueError('Expected frozen system/user messages')
    content, paths = [], []
    for part in messages[1]['content']:
        if part.get('type') == 'text' and set(part) == {'type', 'text'}:
            if not isinstance(part['text'], str):
                raise ValueError('Text must be a string')
            content.append(copy.deepcopy(part))
        elif part.get('type') == 'image' and set(part) == {'type', 'path'}:
            data = Path(part['path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != image_hashes[part['path']]:
                raise ValueError('Original image bytes changed')
            if data.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif data.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            else:
                raise ValueError('Unsupported original image format')
            content.append(dict(type='image_url', image_url=dict(detail='high',
                url='data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii'))))
            paths.append(part['path'])
        else:
            raise ValueError('Unexpected reader content')
    if set(paths) != set(image_hashes) or len(paths) not in (5, 6):
        raise ValueError('Expected five history images and optional question image')
    return dict(**settings(model), messages=[copy.deepcopy(messages[0]), dict(role='user', content=content)])


def config_for(model, env_file):
    return (gpt if model == 'gpt-5' else deepseek).load_config(env_file)


def chat(config, body, timeout=120):
    if {k: v for k, v in body.items() if k != 'messages'} != settings(config.model):
        raise ValueError('Wire settings changed')
    if config.model == 'gpt-5':
        if config.base_url != gpt.OFFICIAL_BASE_URL or config.credential_variable != 'OPENAI_API_KEY':
            raise ValueError('Invalid GPT-5 route')
    else:
        deepseek._validate_config(config)
    req = urllib.request.Request(config.base_url + '/chat/completions',
        data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'})
    started = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
            raw, request_id = response.read(), response.headers.get('x-request-id')
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(65536)
        finally:
            exc.close()
        error = classify_http_error(exc.code, raw)
        try:
            if json.loads(raw).get('error', {}).get('code') == 'data_inspection_failed':
                error.kind = 'data_inspection_failed'
        except (ValueError, AttributeError, TypeError):
            pass
        raise error from None
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        raise APIError('network', retryable=True, message='API network request failed') from None
    try:
        data = json.loads(raw)
        assert len(data['choices']) == 1
        choice = data['choices'][0]
        message = choice['message']
        text, refusal = message.get('content'), message.get('refusal')
        if text is None and refusal:
            text = ''
        assert isinstance(text, str) and isinstance(choice['finish_reason'], str)
        assert isinstance(data['id'], str) and data['id']
        returned = data['model']
        usage = data.get('usage') or {}
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    except (ValueError, KeyError, TypeError, AttributeError, AssertionError):
        raise APIError('protocol', message='Invalid API response') from None
    if not (gpt.valid_returned_model(returned) if config.model == 'gpt-5' else returned == config.model):
        raise APIError('model_mismatch', message='Returned model mismatch')
    if config.model != 'gpt-5' and (message.get('reasoning_content') or reasoning not in (None, 0)):
        raise APIError('unexpected_reasoning', message='Thinking was not disabled')
    return dict(raw_response=text, returned_model=returned, response_id=data['id'], request_id=request_id,
        system_fingerprint=data.get('system_fingerprint'), finish_reason=choice['finish_reason'],
        usage=usage, refusal=refusal, reasoning_tokens=reasoning, elapsed_seconds=time.monotonic()-started)


def prepare(args, target):
    if (target / 'protocol.json').exists():
        raise FileExistsError('Protocol exists; use smoke/run/analyze')
    source_protocol, questions = source_api.verify(args.source)
    if len(questions) != 371 or source_protocol['retrieval_top_k'] != 5:
        raise ValueError('Expected complete 371-question Top5 source')
    config = config_for(args.model, args.env_file)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'qa_manifest.jsonl').write_bytes((args.source / 'qa_manifest.jsonl').read_bytes())
    manifest = []
    for q in questions:
        body = build_request(q['reader_messages'], q['reader_image_hashes'], args.model)
        manifest.append(dict(id=q['id'], source_payload_sha256=q['payload_sha256'],
            request_sha256=digest(body), image_count=len(q['reader_image_paths'])))
    (target / 'request_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    inputs = [Path(__file__), args.source / 'protocol.json', args.source / 'qa_manifest.jsonl',
        args.source / 'qa_responses.jsonl', target / 'qa_manifest.jsonl', target / 'request_manifest.jsonl',
        Path(gpt.__file__), Path(deepseek.__file__),
        ROOT / 'scripts/evaluation/gpt5_text_memory_transport.py',
        ROOT / 'scripts/evaluation/openai_chat_transport.py']
    protocol = dict(model=args.model, api=config.public_dict(), settings=settings(args.model),
        source=str(args.source.resolve()), source_signature=source_protocol['experiment_signature'],
        questions=len(questions), top_k=5, workers=args.workers,
        source_qa_accuracy=read(args.source / 'summary.json')['accuracy'],
        policy='Exact frozen CLIP retrieval, interleaved native original images and associated dialogue; no captions',
        image_policy='Verified original PNG/JPEG bytes; image_detail=high; no local resizing',
        decoding_note='GPT-5 minimal reasoning, provider-default temperature, 2048 total completion tokens; DeepSeek thinking=false, temperature=0, 16 tokens',
        retry_policy='At most 3 attempts on transient failures only; no format retries; provider content rejection scored incorrect',
        input_hashes={str(p.resolve()): sha(p) for p in inputs})
    protocol['experiment_signature'] = digest(protocol)
    dump(target / 'protocol.json', protocol)
    print(json.dumps(dict(stage='prepared', model=args.model, questions=len(questions))), flush=True)


def verify(target):
    p = read(target / 'protocol.json')
    if digest({k:v for k,v in p.items() if k != 'experiment_signature'}) != p['experiment_signature']:
        raise ValueError('Protocol changed')
    for name, expected in p['input_hashes'].items():
        if sha(name) != expected:
            raise ValueError('Input changed: ' + name)
    qs, requests = rows(target / 'qa_manifest.jsonl'), rows(target / 'request_manifest.jsonl')
    if len(qs) != p['questions'] or [q['id'] for q in qs] != [r['id'] for r in requests]:
        raise ValueError('Request coverage mismatch')
    responses = rows(target / 'qa_responses.jsonl') if (target / 'qa_responses.jsonl').exists() else []
    if len({r['id'] for r in responses}) != len(responses):
        raise ValueError('Duplicate responses')
    expected = {r['id']:r for r in requests}
    for r in responses:
        if (r['experiment_signature'] != p['experiment_signature']
                or r['request_sha256'] != expected[r['id']]['request_sha256']):
            raise ValueError('Response provenance mismatch')
    return p, qs, expected, responses


def parse_choice(text):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', text.strip(), re.I)
    return match.group(1).upper() if match else None


def run(args, target, smoke=False):
    protocol, questions, expected, previous = verify(target)
    config = config_for(args.model, args.env_file)
    if config.public_dict() != protocol['api']:
        raise ValueError('API route changed')
    done = {r['id'] for r in previous}
    if smoke:
        questions = [next(q for q in questions if not q['query_image_paths']),
                     next(q for q in questions if q['query_image_paths'])]
    else:
        probe = read(target / 'smoke.json')
        if probe['status'] != 'passed' or probe['experiment_signature'] != protocol['experiment_signature']:
            raise ValueError('Successful 5/6-image smoke required')
    lock, stop = threading.Lock(), threading.Event()
    revisions = {r['returned_model'] for r in previous if r['status'] == 'ok'}

    def append(name, value):
        with lock:
            with (target / name).open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

    def one(q):
        if stop.is_set():
            return
        body = build_request(q['reader_messages'], q['reader_image_hashes'], args.model)
        key = digest(body)
        if key != expected[q['id']]['request_sha256']:
            stop.set()
            raise ValueError('Request changed')
        base = dict(id=q['id'], request_sha256=key, experiment_signature=protocol['experiment_signature'],
                    source_payload_sha256=q['payload_sha256'], reader_image_count=len(q['reader_image_paths']))
        for attempt in range(1, 4):
            try:
                out = chat(config, body)
                with lock:
                    if revisions and out['returned_model'] not in revisions:
                        raise APIError('model_mismatch', message='Model revision changed within run')
                    revisions.add(out['returned_model'])
                append('api_attempts.jsonl', dict(id=q['id'], attempt=attempt, status='ok', request_sha256=key,
                    response_id=out['response_id']))
                result = dict(**base, **out, status='ok', predicted_choice=parse_choice(out['raw_response']))
                append('qa_responses.jsonl', result)
                return result
            except APIError as exc:
                failure = dict(id=q['id'], attempt=attempt, status='error', error_type=exc.kind,
                    http_status=exc.http_status, code=exc.code, retryable=exc.retryable, request_sha256=key)
                append('api_attempts.jsonl', failure)
                if exc.kind == 'data_inspection_failed':
                    result = dict(**base, status='provider_rejection', raw_response='', predicted_choice=None,
                                  error_type=exc.kind, returned_model=None)
                    append('qa_responses.jsonl', result)
                    return result
                if not exc.retryable or attempt == 3:
                    stop.set()
                    raise
                time.sleep(2 * attempt)

    pending = [q for q in questions if q['id'] not in done]
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        futures = [pool.submit(one, q) for q in pending]
        for future in as_completed(futures):
            result = future.result()
            if result:
                done.add(result['id'])
                print(json.dumps(dict(model=args.model, completed=len(done), total=protocol['questions'],
                    id=result['id'], status=result['status'])), flush=True)
    if smoke:
        answers = {r['id']:r for r in rows(target / 'qa_responses.jsonl')}
        passed = all(answers[q['id']]['status'] == 'ok' and answers[q['id']]['predicted_choice']
                     and answers[q['id']]['finish_reason'] == 'stop' for q in questions)
        dump(target / 'smoke.json', dict(status='passed' if passed else 'failed',
            experiment_signature=protocol['experiment_signature'], ids=[q['id'] for q in questions],
            image_counts=[len(q['reader_image_paths']) for q in questions]))
        if not passed:
            raise ValueError('Smoke failed; inspect responses')


def analyze(target):
    p, questions, _, answers = verify(target)
    by_id = {a['id']:a for a in answers}
    if set(by_id) != {q['id'] for q in questions}:
        raise ValueError('Incomplete answers')
    correct, by_task, detail = 0, {}, []
    for q in questions:
        a = by_id[q['id']]
        ok = a['status'] == 'ok' and a['predicted_choice'] == q['gold_choice']
        correct += ok
        task = by_task.setdefault(q['task'], dict(questions=0, correct=0))
        task['questions'] += 1
        task['correct'] += ok
        detail.append(dict(id=q['id'], correct=ok, gold=q['gold_choice'], predicted=a['predicted_choice']))
    for task in by_task.values():
        task['accuracy'] = task['correct'] / task['questions']
    summary = dict(model=p['model'], questions=len(questions), correct=correct, accuracy=correct/len(questions),
        top_k=5, unchanged_retrieval=True, unchanged_dialogue_and_prompts=True,
        provider_rejections=sum(a['status'] != 'ok' for a in answers),
        format_errors=sum(a['status'] == 'ok' and a['predicted_choice'] is None for a in answers),
        token_limit_hits=sum(a.get('finish_reason') == 'length' for a in answers),
        returned_models=sorted({a['returned_model'] for a in answers if a['status'] == 'ok'}),
        usage={k:sum((a.get('usage') or {}).get(k, 0) for a in answers)
               for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        by_task=by_task, experiment_signature=p['experiment_signature'])
    dump(target / 'summary.json', summary)
    (target / 'per_question.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in detail))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--stage', choices=('prepare', 'smoke', 'run', 'analyze'), required=True)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    target = args.output_root / args.model
    if args.stage == 'prepare':
        prepare(args, target)
    elif args.stage == 'analyze':
        analyze(target)
    else:
        run(args, target, smoke=args.stage == 'smoke')


if __name__ == '__main__':
    main()
