"""Resume only the 12 failed HippoMM rewrites, with seed-42's exact recovery.

Native seed-1042 generation, retrieval and reader code is unchanged. Completed
rows remain a byte-identical prefix, and new calls are appended to native caches.
All amendment files live in the run's recovery/hippomm_20261113 directory.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memgallery_seed1042_visual as visual
from scripts.memory import hippomm as method, hippomm_core as core

DEFAULT = visual.DEFAULT
REFERENCE = ROOT / 'reports/hippomm_short_ids_20261104/recover_failed.py'
REFERENCE_SHA256 = '338b4731a76792853f51e635a0a30a02ff667e30aeee00a28a71afa64c4a125e'
POLICY = 'three_word_array_after_original_rewrite_attempts'
ORIGINAL_STRUCTURED_CALL = core.structured_call


def corrected_query(llm, role, prompt, parse, attempts=3):
    try:
        return ORIGINAL_STRUCTURED_CALL(llm, role, prompt, parse, attempts)
    except ValueError:
        if role != 'rewrite':
            raise
    correction = (
        'Generate a concise visual search cue from the original question below. '
        'Return exactly THREE separate English words as a JSON array in this object: '
        '{"words": ["word1", "word2", "word3"]}. Each element must be one word, '
        'with no whitespace. Focus on the visually identifiable objects or attributes '
        'explicitly mentioned in the question. Do not infer the answer or invent '
        'attributes. Do not return an explanation.\nOriginal request:\n' + prompt)
    for attempt in range(3):
        raw = llm('rewrite_format_recovery', correction)
        try:
            obj = core.parse_object(raw)
            words = obj.get('words')
            if (set(obj) != {'words'} or not isinstance(words, list) or len(words) != 3
                    or any(not isinstance(w, str) or not w.strip()
                           or any(c.isspace() for c in w) for w in words)):
                raise ValueError('Return exactly three nonempty single-word strings')
            return parse(json.dumps({'visual_query': ' '.join(words)}))
        except (TypeError, ValueError) as exc:
            if attempt == 2:
                raise ValueError('Additional query formatting failed: ' + str(exc)) from exc
            correction += '\nPrevious output: ' + raw + '\nValidation error: ' + str(exc)


def sha_prefix(path, size):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        left = size
        while left:
            block = stream.read(min(1024 * 1024, left))
            if not block:
                raise ValueError('Previously frozen file was truncated: ' + str(path))
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def directory(args):
    return args.run_root / 'recovery/hippomm_20261113'


def online(args):
    return args.run_root / 'visual_methods/hippomm/baseline/mem_gallery/online'


def freeze(path, value):
    visual.frozen_json(path, value)


def function_ast(path, name):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def equality_preflight():
    if visual.sha(REFERENCE) != REFERENCE_SHA256:
        raise ValueError('Seed-42 recovery implementation changed')
    if function_ast(__file__, 'corrected_query') != function_ast(REFERENCE, 'corrected_query'):
        raise ValueError('Recovery differs from the exact seed-42 implementation')
    # The native attempts run first, then exactly the existing array-format rule.
    calls = []
    def llm(role, prompt):
        calls.append((role, prompt))
        if role == 'rewrite':
            return '{"visual_query":"one two three four five six"}'
        return '{"words":["one","two","three"]}'
    query = corrected_query(llm, 'rewrite', 'ORIGINAL QUESTION ONLY', core.parse_query)
    if query != 'one two three' or [r for r, _ in calls] != ['rewrite'] * 3 + ['rewrite_format_recovery']:
        raise ValueError('Original attempts or correction trigger differs')
    native_calls = []
    def valid(role, prompt):
        native_calls.append(role)
        return '{"visual_query":"two words"}'
    if corrected_query(valid, 'rewrite', 'x', core.parse_query) != 'two words' or native_calls != ['rewrite']:
        raise ValueError('Already valid native query was changed')
    invalid_calls = []
    def invalid(role, prompt):
        invalid_calls.append(role)
        return ('{"visual_query":"too many words in this visual question"}'
                if role == 'rewrite' else '{"words":["two words","x","y"]}')
    try:
        corrected_query(invalid, 'rewrite', 'x', core.parse_query)
    except ValueError as exc:
        if not str(exc).startswith('Additional query formatting failed: '):
            raise
    else:
        raise ValueError('Malformed recovery was incorrectly accepted')
    if invalid_calls != ['rewrite'] * 3 + ['rewrite_format_recovery'] * 3:
        raise ValueError('Recovery no longer has exactly the seed42 retry bound')
    return {'status': 'passed', 'exact_seed42_function_ast': True,
            'original_attempts_before_recovery': 3, 'recovery_attempts_at_most': 3,
            'unchanged_valid_query': True, 'no_model_loaded': True}


def native_preflight(args):
    """Load the real signed native contracts and arrays, without any runtime."""
    checked = visual.verify(SimpleNamespace(run_root=args.run_root, method='hippomm', device=args.device))
    visual.patch_native_seed()
    cfg = visual.hip_settings(SimpleNamespace(run_root=args.run_root, method='hippomm', device=args.device))
    cfg.memory_dir = args.run_root / 'visual_methods/hippomm/baseline/mem_gallery/offline'
    cfg.output = online(args)
    _, questions, events, vectors, dialogues = method.load_run(cfg)
    responses = method.checked_responses(cfg.output, None, questions)
    missing = sorted({q['id'] for q in questions} - {r['id'] for r in responses})
    before = json.loads((directory(args) / 'before.json').read_text())['hippomm']
    if (len(questions) != 1711 or len(events) != 860 or len(vectors) != 860
            or not set(missing) <= set(before['missing_ids']) or len(before['missing_ids']) != 12):
        raise ValueError('Native contract/memory/recovery cohort does not match the planned experiment')
    return {'status': 'passed', 'questions': len(questions), 'events': len(events),
            'dialogue_rounds': len(dialogues), 'answered': len(responses), 'pending_ids': missing,
            'seed': checked['contract']['seed'], 'device': args.device,
            'native_contracts_loaded': True, 'no_model_loaded': True}


def verify_prefix(args, protocol):
    for path, record in protocol['append_only_prefixes'].items():
        if sha_prefix(path, record['size_bytes']) != record['sha256']:
            raise ValueError('Native pre-recovery prefix changed: ' + path)
    for path, expected in protocol['immutable_sha256'].items():
        if visual.sha(path) != expected:
            raise ValueError('Frozen recovery input changed: ' + path)


def prepare(args):
    root = directory(args)
    before_path = root / 'before.json'
    before = json.loads(before_path.read_text())
    frozen = before['hippomm']
    target = online(args)
    response_path = target / 'qa_responses.jsonl'
    old = visual.read(response_path)
    questions = visual.read(target / 'questions.jsonl')
    missing = sorted({r['id'] for r in questions} - {r['id'] for r in old})
    if len(old) != 1699 or len(questions) != 1711 or missing != frozen['missing_ids'] or len(missing) != 12:
        raise ValueError('Only the original 1699-complete, 12-missing cohort may prepare recovery')
    if (response_path.stat().st_size != frozen['native_response_file_before']['size_bytes']
            or visual.sha(response_path) != frozen['native_response_file_before']['sha256']):
        raise ValueError('Original answer prefix differs from independent before.json')
    preflight = equality_preflight()
    preflight['native_contracts'] = native_preflight(args)
    errors = {row['id']: row for row in visual.read(target / 'question_errors.jsonl')}
    cache_hashes, failures = {}, []
    for qid in missing:
        err = errors[qid]
        if err['error'] != 'rewrite: invalid output after 3 attempts: Visual query must contain 2-5 words':
            raise ValueError('This amendment only covers exhausted query word-count validation')
        rewrites = []
        for call_id in dict.fromkeys(err['llm_calls']):
            path = target / 'llm_cache' / (call_id + '.json')
            cache_hashes[str(path)] = visual.sha(path)
            row = json.loads(path.read_text())
            if row['request']['role'] == 'rewrite':
                raw = row['response']['raw_response']
                value = core.parse_object(raw)
                if set(value) != {'visual_query'} or not isinstance(value['visual_query'], str):
                    raise ValueError('Raw rewrite is not the expected well-formed object')
                count = len(value['visual_query'].strip().split())
                if 2 <= count <= 5 or row['response']['hit_token_limit']:
                    raise ValueError('Unexpected native rewrite failure')
                rewrites.append({'call_id': call_id, 'raw_response': raw, 'words': count,
                                 'cache_sha256': cache_hashes[str(path)]})
        failures.append({'id': qid, 'error': err['error'], 'invalid_rewrites': rewrites})
    prefixes = {}
    for name in ('qa_responses.jsonl', 'generation_attempts.jsonl', 'llm_calls.jsonl', 'question_errors.jsonl'):
        path = target / name
        prefixes[str(path)] = {'sha256': visual.sha(path), 'size_bytes': path.stat().st_size}
    cache_names = sorted(p.name for p in (target / 'llm_cache').glob('call_*.json'))
    if not cache_names:
        raise ValueError('Native cache is missing')
    source_paths = [Path(__file__), REFERENCE, before_path, target / 'questions.jsonl', target / 'protocol.json',
                    args.run_root / 'plan.json', args.run_root / 'visual_methods/hippomm/stability_protocol.json']
    immutable = {str(path): visual.sha(path) for path in source_paths}
    immutable.update(cache_hashes)
    protocol = {'schema': 1, 'status': 'prepared', 'method': 'hippomm', 'seed': 1042,
        'reference_seed': 42, 'judge_seed': 42, 'questions': 1711, 'previous_answers': 1699,
        'missing_ids': missing, 'policy': POLICY, 'native_seed_driver_sha256': visual.sha(visual.__file__),
        'reference_recovery_sha256': REFERENCE_SHA256, 'recovery_source_sha256': visual.sha(__file__),
        'append_only_prefixes': prefixes, 'immutable_sha256': immutable,
        'native_cache_files_before': len(cache_names),
        'native_max_call_number_before': max(int(n[5:-5]) for n in cache_names),
        'generation_settings': {'seed': 1042, 'temperature': 0., 'do_sample': False,
            'max_new_tokens': 1024, 'context_budget': 120000, 'top_k': 5,
            'confidence_threshold': .75, 'fallback_threshold': .4, 'fallback_score': .6},
        'old_successful_answers_regenerated': 0, 'new_seed42_outputs_imported': 0,
        'gold_or_answers_in_recovery_prompt': False, 'preflight': preflight}
    freeze(root / 'rewrite_failures.json', failures)
    freeze(root / 'rewrite_protocol.json', protocol)
    freeze(root / 'rewrite_prepare.json', {'status': 'complete', 'missing_questions': 12,
        'original_answers': 1699, 'protocol_sha256': visual.sha(root / 'rewrite_protocol.json'),
        'source_sha256': visual.sha(__file__), 'reference_ast_exact': True})
    print(json.dumps({'phase': 'prepare', 'status': 'complete', 'missing': len(missing)}), flush=True)


def audit(args):
    root = directory(args)
    protocol = json.loads((root / 'rewrite_protocol.json').read_text())
    verify_prefix(args, protocol)
    target = online(args)
    questions = visual.read(target / 'questions.jsonl')
    responses = visual.read(target / 'qa_responses.jsonl')
    if len(responses) != 1711 or len({r['id'] for r in responses}) != 1711:
        raise ValueError('Recovery is incomplete; no final marker written')
    if {r['id'] for r in responses} != {q['id'] for q in questions}:
        raise ValueError('Recovered IDs differ from the canonical cohort')
    recovered = responses[1699:]
    if sorted(r['id'] for r in recovered) != protocol['missing_ids']:
        raise ValueError('This amendment appended unexpected questions')
    newly_referenced = set()
    recovery_calls = []
    for row in recovered:
        if row.get('recovery_policy') != POLICY or row.get('recovery_amendment') != str(root / 'rewrite_protocol.json'):
            raise ValueError('Missing explicit recovery provenance')
        core.parse_query(json.dumps({'visual_query': row['visual_query']}))
        for cid in row['llm_calls']:
            call = json.loads((target / 'llm_cache' / (cid + '.json')).read_text())
            request, response = call['request'], call['response']
            if (request['seed'] != 1042 or request['do_sample'] or request['temperature'] != 0
                    or request['max_new_tokens'] != 1024 or response['hit_token_limit']
                    or visual.digest(request) != call['request_sha256']):
                raise ValueError('Recovery call differs from native seed1042 settings')
            if int(cid[5:]) > protocol['native_max_call_number_before']:
                newly_referenced.add(cid)
            if request['role'] == 'rewrite_format_recovery':
                if request['query_image_paths']:
                    raise ValueError('Query pixels entered rewrite recovery')
                recovery_calls.append({'id': row['id'], 'call_id': cid,
                    'request_sha256': call['request_sha256'], 'raw_response': response['raw_response']})
    attempts = visual.read(target / 'generation_attempts.jsonl')
    successful = {r['call_id']: r for r in attempts if r['status'] == 'success'}
    for cid in newly_referenced:
        cached = json.loads((target / 'llm_cache' / (cid + '.json')).read_text())
        if {k: v for k, v in successful[cid].items() if k != 'status'} != cached:
            raise ValueError('New call lacks its durable raw successful generation attempt')
    counts = {qid: sum(r['id'] == qid for r in recovery_calls) for qid in protocol['missing_ids']}
    if any(not 1 <= n <= 3 for n in counts.values()):
        raise ValueError('Recovery did not retain the seed42 bounded retry count')
    value = {'status': 'passed', 'questions': 1711, 'original_answers_unchanged': 1699,
        'recovered_questions': 12, 'recovered_ids': [r['id'] for r in recovered],
        'seed': 1042, 'seed42_recovery_function_ast_exact': True,
        'native_generation_settings_unchanged': True, 'original_prefixes_byte_identical': True,
        'format_recovery_calls': recovery_calls, 'newly_referenced_calls': len(newly_referenced),
        'answers_sha256': visual.sha(target / 'qa_responses.jsonl'),
        'protocol_sha256': visual.sha(root / 'rewrite_protocol.json')}
    freeze(root / 'rewrite_audit.json', value)
    freeze(root / 'recovery_completion.json', {'status': 'complete', 'method': 'hippomm',
        'seed': 1042, 'questions': 1711, 'previous_answers_unchanged': 1699, 'recovered_questions': 12,
        'native_baseline_answers_sha256': value['answers_sha256'],
        'audit_sha256': visual.sha(root / 'rewrite_audit.json'),
        'note': 'Baseline answers complete; controlled original-image answers and Judge still require their own completion gates.'})
    print(json.dumps({'phase': 'audit', 'status': 'passed', 'questions': 1711}), flush=True)


def run(args):
    root = directory(args)
    protocol = json.loads((root / 'rewrite_protocol.json').read_text())
    verify_prefix(args, protocol)
    equality_preflight()
    rows = visual.read(online(args) / 'qa_responses.jsonl')
    missing = {q['id'] for q in visual.read(online(args) / 'questions.jsonl')} - {r['id'] for r in rows}
    if not missing <= set(protocol['missing_ids']):
        raise ValueError('Attempt to regenerate an original successful answer')
    visual.verify(SimpleNamespace(run_root=args.run_root, method='hippomm', device=args.device))
    visual.patch_native_seed()
    core.structured_call = corrected_query
    original_append = method.append
    def append_with_recovery_note(path, row):
        if Path(path).name == 'qa_responses.jsonl':
            if row['id'] not in protocol['missing_ids']:
                raise ValueError('Attempt to append an out-of-scope recovered question')
            row = {**row, 'recovery_policy': POLICY,
                   'recovery_amendment': str(root / 'rewrite_protocol.json')}
            # Durable append; the original reader/retriever and raw calls are unchanged.
            return visual.append(path, row)
        return original_append(path, row)
    method.append = append_with_recovery_note
    cfg = visual.hip_settings(SimpleNamespace(run_root=args.run_root, method='hippomm', device=args.device))
    cfg.memory_dir = args.run_root / 'visual_methods/hippomm/baseline/mem_gallery/offline'
    cfg.output = online(args)
    try:
        method.run(cfg)
    finally:
        core.structured_call = ORIGINAL_STRUCTURED_CALL
        method.append = original_append
    audit(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'audit', 'preflight'), required=True)
    parser.add_argument('--run-root', type=Path, default=DEFAULT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.phase == 'preflight':
        proof = equality_preflight()
        proof['native_contracts'] = native_preflight(args)
        print(json.dumps(proof), flush=True)
        return
    directory(args).mkdir(parents=True, exist_ok=True)
    with (directory(args) / 'rewrite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            if (directory(args) / 'rewrite_protocol.json').is_file():
                protocol = json.loads((directory(args) / 'rewrite_protocol.json').read_text())
                verify_prefix(args, protocol)
                print(json.dumps({'phase': 'prepare', 'status': 'already_prepared'}), flush=True)
            else:
                prepare(args)
        elif args.phase == 'audit':
            audit(args)
        else:
            with (args.run_root / 'visual_methods/hippomm/run.lock').open('a') as native_lock:
                fcntl.flock(native_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run(args)


if __name__ == '__main__':
    main()
