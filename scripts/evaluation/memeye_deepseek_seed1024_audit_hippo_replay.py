"""Replay sealed native Hippo Open calls locally; all network is forbidden.

The prior independent wire audit supplies immutable cache hashes. This check
returns only those original raw cache outputs, reruns the original 3+3 rewrite
logic and native core, and recomputes CLIP queries on CPU. No LLM is loaded.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory import hippomm as base, hippomm_core as core, hippomm_api as hip, hippomm_original as original

OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def compare(actual, expected, label):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or not set(actual) <= set(expected):
            raise ValueError(label + ': missing trace field')
        for key, value in actual.items():
            compare(value, expected[key], label + '.' + key)
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(label + ': trace length differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, label + '[' + str(index) + ']')
    elif type(actual) is float:
        if not isinstance(expected, (int, float)) or not math.isclose(actual, expected, abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError(label + ': numeric trace differs')
    elif actual != expected:
        raise ValueError(label + ': trace value differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--wire-audit', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    root = output / 'visual_methods/hippomm/open'
    source_audit = args.wire_audit or output / 'audit/hippomm_open_judge_early.json'
    report_path = args.report or output / 'audit/hippomm_native_replay_early.json'
    audit = read(source_audit)
    if (audit['status'] != 'passed' or audit['expected_methods'] != 1 or
            audit['methods'][0]['method'] != 'hippomm' or audit['open_answers_verified'] != 371):
        raise ValueError('A completed independent Hippo wire audit is required')
    bound = audit['artifact_sha256']
    verified = {}
    stability_path = output / 'visual_methods/stability_protocol.json'
    stability = read(stability_path)
    if digest(stability['contract']) != stability['experiment_signature']:
        raise ValueError('Visual stability protocol signature changed')
    manifest_path = output / 'visual_methods/hippomm/experiment_manifest.json'
    manifest = read(manifest_path)['contract']
    driver = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_visual.py'
    transport_path = ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_transport.py'
    if (manifest['seed'] != 1024 or manifest['judge_seed'] != 42 or
            manifest['seed_wrapper_sha256'] != bound[str(driver)] or
            manifest['seed_transport_sha256'] != bound[str(transport_path)]):
        raise ValueError('Native implementation manifest is not bound to the audited seeded wrapper')
    native_sources = dict(stability['contract']['source_sha256'], **manifest['implementation'])
    verified[str(stability_path)] = sha(stability_path)
    verified[str(manifest_path)] = sha(manifest_path)
    def frozen(path):
        path = Path(path).resolve()
        if str(path) not in verified:
            actual = sha(path)
            if bound.get(str(path)) != actual:
                raise ValueError('Previously audited source or cache changed: ' + str(path))
            verified[str(path)] = actual
        return read(path)
    def frozen_rows(path):
        frozen_hash(path)
        return rows(path)
    def frozen_hash(path):
        path = Path(path).resolve()
        if str(path) not in verified:
            actual = sha(path)
            if (bound.get(str(path)) or native_sources.get(str(path))) != actual:
                raise ValueError('Previously audited source changed: ' + str(path))
            verified[str(path)] = actual
    for path in (driver, Path(base.__file__), Path(core.__file__), Path(hip.__file__), Path(original.__file__)):
        frozen_hash(path)
    tree = ast.parse(driver.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'hip_open_run')
    cpu_calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute) and node.func.attr == 'ClipText' and
        node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'cpu']
    if len(cpu_calls) != 1:
        raise ValueError('Pinned Open wrapper no longer explicitly constructs the CPU CLIP encoder')
    def deny(*unused, **unused_keywords):
        raise RuntimeError('Network forbidden during native Hippo replay')
    socket.socket.connect = deny
    urllib.request.OpenerDirector.open = deny
    import numpy as np
    import torch
    torch.set_num_threads(4)
    encoder = base.ClipText('cpu')
    encoded = {}
    def encode(query):
        if query not in encoded:
            encoded[query] = encoder(query)
        return encoded[query]
    protocol = frozen(root / 'protocol.json')
    memory = Path(protocol['memory_dir'])
    events = frozen_rows(memory / 'events.jsonl')
    dialogues = {r['id']: r for r in frozen_rows(memory / 'dialogues.jsonl')}
    images = {r['event_id']: r for r in frozen_rows(memory / 'representative_images.jsonl')}
    # The memory archive is bound by the Open source protocol, which was fully
    # verified in the wire audit. Bind it directly without rehashing any images.
    vectors_path = memory / 'visual_embeddings.npz'
    expected = protocol['source_sha256'][str(vectors_path)]
    if sha(vectors_path) != expected:
        raise ValueError('Native visual vectors changed')
    verified[str(vectors_path)] = expected
    with np.load(vectors_path, allow_pickle=False) as data:
        if data['event_ids'].tolist() != [e['event_id'] for e in events]:
            raise ValueError('Native visual vector event order differs')
        vectors = dict(zip(data['event_ids'].tolist(), data['vectors']))
    emap = {e['event_id']: e for e in events}
    questions = frozen_rows(root / 'questions.jsonl')
    saved = {r['id']: r for r in frozen_rows(root / 'responses.jsonl')}
    counts, replayed = Counter(), []
    for question in questions:
        answer = saved[question['id']]
        calls = [frozen(root / relative) for relative in answer['llm_calls']]
        cursor, rewrite_count = 0, 0
        replay_calls, recovery, image_order = [], [], []
        class CachedChat:
            last_call_id = None
            def __call__(self, role, prompt, image_paths=()):
                nonlocal cursor
                if cursor >= len(calls):
                    raise ValueError('Native replay requested an uncached additional call')
                call = calls[cursor]; request = call['request']
                if (request['role'] != role or request['prompt'] != prompt or
                        request['image_paths'] != [str(p) for p in image_paths]):
                    raise ValueError('Native replay request differs from the sealed cache: ' + question['id'])
                cursor += 1; self.last_call_id = call['call_id']; counts[role] += 1
                return call['response']['raw_response']
        chat = CachedChat()
        def llm(role, prompt):
            nonlocal rewrite_count, image_order
            paths = question['query_image_paths'] if role in ('fast', 'answer') else []
            actual_role = role
            if role == 'answer':
                evidence = json.loads(prompt.rsplit('\n', 1)[-1])
                ids = [e['event_id'] for e in evidence['memories']]
                if ids:
                    prompt, paths, image_order = original.reader_payload(question,
                        {'route': 'detailed', 'selected_event_ids': ids}, emap, images, dialogues, memory)
                    actual_role = 'answer_with_representatives'
            raw = chat(actual_role, prompt, paths)
            replay_calls.append(chat.last_call_id)
            if role == 'rewrite':
                rewrite_count += 1
                if rewrite_count == 3:
                    raw = hip.repaired_rewrite(chat, raw, core.query_prompt(question), replay_calls, recovery)
            return raw
        trace = core.answer_question(question, events, vectors, dialogues, llm, encode)
        compare(trace, answer, question['id'])
        compare(recovery, answer['query_format_recovery'], question['id'] + '.query_format_recovery')
        compare(image_order, answer['reader_image_order'], question['id'] + '.reader_image_order')
        if cursor != len(calls) or replay_calls != [call['call_id'] for call in calls]:
            raise ValueError('Native call sequence was not exactly consumed')
        if rewrite_count > 3 or sum(c['request']['role'] == 'rewrite_format_recovery' for c in calls) > 3:
            raise ValueError('Native 3+3 rewrite bound changed')
        replayed.append(question['id'])
        if len(replayed) % 25 == 0 or len(replayed) == 371:
            print(json.dumps(dict(replayed=len(replayed), total=371, cpu_queries=len(encoded))), flush=True)
    if len(replayed) != len(set(replayed)) or len(replayed) != 371:
        raise ValueError('Incomplete native replay')
    report = dict(status='passed', method='hippomm', native_questions_replayed=371,
        cpu_clip_queries_recomputed=len(encoded), native_call_roles=dict(counts),
        native_calls_replayed=sum(counts.values()), query_recovery_bound='3 original + at most 3 exact-three-word repairs',
        query_encoder_device='cpu', query_encoder_device_evidence='Pinned wrapper explicitly constructs ClipText(cpu); this audit independently recomputed queries on CPU.',
        original_native_trace_reproduced=True, max_float_tolerance=1e-6, real_api_calls=0,
        gpu_calls=0, source_wire_audit_sha256=sha(source_audit), audit_code_sha256=sha(__file__),
        verified_source_sha256=verified)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'verified_source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
