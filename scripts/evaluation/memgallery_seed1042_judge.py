"""Fixed-seed five-grade Judge with exact-wire reuse of verified seed42 scores."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memgallery_multi_method_judge as engine

original, transport = engine.original, engine.transport
PROMPT, GRADES, DEFAULT_CACHE = engine.PROMPT, engine.GRADES, engine.DEFAULT_CACHE
REQUIRED_FIELDS = engine.REQUIRED_FIELDS
sha, read_rows, dump, write_rows = engine.sha, engine.read_rows, engine.dump, engine.write_rows
normalize_answer, build_request, judge_settings = engine.normalize_answer, engine.build_request, engine.judge_settings
run, analyze = engine.run, engine.analyze
EXTRA_CACHE = ROOT / 'results/memgallery_qwen_all_methods_deepseek_judge_20261104'


def read(path):
    return json.loads(Path(path).read_text())


def verify_hashes(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise ValueError('Frozen seed42 Judge source changed: ' + str(path))


def load_baseline_cache(root, template):
    canonical, cached, source_hashes = engine.load_baseline_cache(root, template)
    folder = EXTRA_CACHE
    protocol, summary = read(folder / 'protocol.json'), read(folder / 'summary.json')
    if (summary.get('status') != 'complete' or summary.get('total_scored_method_questions') != 8 * 1711
            or protocol.get('answer_model') != 'Qwen/Qwen3-VL-8B-Instruct'
            or (folder / 'judge_prompt.txt').read_bytes() != PROMPT.read_bytes()):
        raise ValueError('Seed42 method Judge cache is incomplete or incompatible')
    for key, expected in judge_settings().items():
        if protocol.get(key) != expected:
            raise ValueError('Seed42 Judge settings differ: ' + key)
    for name, expected in summary['artifact_sha256'].items():
        if sha(folder / name) != expected:
            raise ValueError('Seed42 Judge artifact changed: ' + name)
    for field in ('source_sha256', 'baseline_cache_sha256', 'implementation_sha256'):
        verify_hashes(protocol[field])
        source_hashes.update(protocol[field])
    requests = read_rows(folder / 'requests.jsonl')
    request_map = {r['request_sha256']: r for r in requests}
    if len(request_map) != len(requests):
        raise ValueError('Duplicate source Judge request')
    for key, row in request_map.items():
        wire = build_request(row, template)
        if row['request'] != wire or transport.request_sha256(wire) != key:
            raise ValueError('Seed42 Judge wire does not reproduce')
    scores = engine.validated_results(folder, requests)
    if set(scores) != set(request_map):
        raise ValueError('Seed42 Judge cache lacks raw scores')
    attempts = read_rows(folder / 'attempts.jsonl')
    grouped = {}
    for attempt in attempts:
        key = attempt['request_sha256']; seq = grouped.setdefault(key, [])
        if key not in request_map or attempt['attempt'] != len(seq) + 1:
            raise ValueError('Seed42 Judge attempt sequence differs')
        seq.append(attempt)
    for key, score in scores.items():
        if score['score_origin'] == 'new_api':
            last = grouped.get(key, [{}])[-1]
            if (last.get('status') != 'ok' or last.get('raw_response') != score['raw_response']
                    or last.get('response_id') != score['response_id']):
                raise ValueError('Seed42 grade lacks its real successful attempt')
        elif key not in cached or any(score[k] != cached[key][k] for k in ('score', 'raw_response', 'response_id')):
            raise ValueError('Seed42 grade reuse differs from the original baseline')
    qmap = read_rows(folder / 'question_map.jsonl')
    if len(qmap) != 8 * 1711:
        raise ValueError('Seed42 Judge cohort differs')
    by_method, seen = {}, set()
    for row in qmap:
        identity = (row['method_id'], row['id'])
        if identity in seen or row['id'] not in canonical:
            raise ValueError('Invalid source method/question identity')
        seen.add(identity)
        if any(row[k] != canonical[row['id']][k] for k in ('task', 'point', 'question', 'ground_truth')):
            raise ValueError('Seed42 canonical question or gold differs')
        key = transport.request_sha256(build_request(row, template))
        if key != row['request_sha256'] or key not in scores:
            raise ValueError('Seed42 source question maps to a different Judge wire')
        by_method.setdefault(row['method_id'], []).append(scores[key]['score'])
    for method in summary['methods']:
        values = by_method[method['method_id']]
        if len(values) != 1711 or sum(values) != method['score_sum'] or sum(values) / 1711 != method['llm_judge']:
            raise ValueError('Seed42 per-method grade aggregate does not reproduce')
    file_names = ('protocol.json', 'summary.json', 'requests.jsonl', 'question_map.jsonl',
                  'scores.jsonl', 'reused_scores.jsonl', 'attempts.jsonl', 'judge_prompt.txt')
    source_hashes.update({str(folder / name): sha(folder / name) for name in file_names})
    for key, score in scores.items():
        if key in cached:
            if cached[key]['score'] != score['score']:
                raise ValueError('Conflicting grades for the same Judge wire')
            continue
        provenance = {'root': str(folder), 'source_request_sha256': key,
            'scores_path': str(folder / ('scores.jsonl' if score['score_origin'] == 'new_api' else 'reused_scores.jsonl')),
            'summary_sha256': sha(folder / 'summary.json'), 'response_id': score['response_id']}
        cached[key] = {**score, 'score_origin': 'baseline_cache', 'reuse_sources': [provenance]}
    return canonical, cached, source_hashes


def prepare(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    methods = manifest.get('methods')
    if not isinstance(methods, list) or not methods:
        raise ValueError('Manifest needs a nonempty methods list')
    if manifest.get('questions_per_method', 1711) != 1711:
        raise ValueError('This evaluation requires the complete 1711-question cohort')
    source_seed = manifest.get('source_seed', 1042)
    if source_seed not in (42, 1042) or any(m.get('source_seed', source_seed) != source_seed for m in methods):
        raise ValueError('Mismatched source answer seed')
    template = PROMPT.read_text()
    canonical, baseline, cache_hashes = load_baseline_cache(args.baseline_cache.resolve(), template)
    unique_requests, question_map, method_protocols, verified_sources = {}, [], [], {}
    seen_methods = set()
    for method in methods:
        mid = method.get('id', '')
        if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', mid) or mid in seen_methods):
            raise ValueError('Method IDs must be unique and safe directory names')
        seen_methods.add(mid)
        if not isinstance(method.get('label'), str) or not method['label'].strip():
            raise ValueError(f'Method {mid} needs a label')
        path = Path(method['records_path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        source_hashes = method.get('source_sha256')
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError(f'Method {mid} requires source_sha256 provenance')
        for source, expected_digest in source_hashes.items():
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            digest = verified_sources.setdefault(str(source_path), sha(source_path))
            if digest != expected_digest:
                raise ValueError(f'Source hash mismatch for {source_path}')
        records_digest = sha(path)
        if method.get('records_sha256', records_digest) != records_digest:
            raise ValueError(f'Frozen records hash mismatch for {mid}')
        records = read_rows(path)
        source_map = original.unique_map(records)
        if len(records) != 1711 or set(source_map) != set(canonical):
            raise ValueError(f'Method {mid} does not cover the canonical 1711 questions')
        correct = 0
        for row in records:
            if any(key not in row for key in REQUIRED_FIELDS):
                raise ValueError(f'Method {mid} record lacks required fields')
            if any(not isinstance(row[key], str) for key in REQUIRED_FIELDS if key != 'em'):
                raise ValueError(f'Method {mid} judge fields must contain original strings')
            canon = canonical[row['id']]
            if any(row[key] != canon[key] for key in ('task', 'point', 'question', 'ground_truth')):
                raise ValueError(f'Method {mid} changed the canonical question or gold answer')
            em = int(normalize_answer(row['model_output']) == normalize_answer(row['ground_truth']))
            if type(row['em']) not in (int, bool) or row['em'] != em:
                raise ValueError(f'Method {mid} per-question EM does not reproduce')
            correct += em
            request = build_request(row, template)
            digest = transport.request_sha256(request)
            origin = {'method_id': mid, 'id': row['id']}
            if digest in unique_requests:
                if unique_requests[digest]['request'] != request:
                    raise ValueError('Request hash collision')
                unique_requests[digest]['uses'].append(origin)
            else:
                unique_requests[digest] = {
                    'request_sha256': digest, 'request': request,
                    **{key: row[key] for key in ('question', 'ground_truth', 'model_output')},
                    'point': row['point'], 'uses': [origin]}
            # Retain original metadata, including pre-existing truncated reader outputs.
            question_map.append({**row, 'method_id': mid, 'request_sha256': digest})
        if correct != method.get('expected_em_correct'):
            raise ValueError(f'Method {mid}: recomputed EM={correct}/1711 differs from expected')
        if 'em' in method and abs(method['em'] - correct / 1711) > 1e-12:
            raise ValueError(f'Method {mid}: reported source EM differs')
        method_protocols.append({**method, 'records_path': str(path),
                                 'records_sha256': records_digest,
                                 'questions': 1711, 'em': correct / 1711})
    requests = list(unique_requests.values())
    reused = [baseline[digest] for digest in unique_requests if digest in baseline]
    code_paths = [Path(__file__), Path(engine.__file__), Path(original.__file__), Path(transport.__file__),
                  ROOT / 'scripts/evaluation/openai_chat_transport.py',
                  ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemGallery', 'answer_model': 'Qwen/Qwen3-VL-8B-Instruct',
        'answer_generation_seed': source_seed, 'comparison_seed': 42,
        'cache_policy': 'Identical question/gold/original-answer wire only; validated complete seed42 cohort and actual raw Judge attempts. Judge seed remains 42.',
        **judge_settings(), 'judge_model_identity_kind': 'provider_alias',
        'official_prompt_url': original.PROMPT_URL,
        'parser_difference_from_public_code': 'Keep all five grades exactly; no three-level remapping.',
        'aggregation': 'Per-method arithmetic mean over all 1711 questions; no missing-score imputation.',
        'deduplication': 'One score per exact serialized request hash; reused across identical inputs.',
        'original_answer_policy': 'Keep every frozen raw answer, including any source token-limit outputs.',
        'methods': method_protocols, 'method_count': len(method_protocols),
        'total_method_questions': len(question_map), 'unique_requests': len(requests),
        'baseline_reused_unique_requests': len(reused),
        'new_unique_requests_expected': len(requests) - len(reused),
        'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'source_sha256': verified_sources, 'baseline_cache_sha256': cache_hashes,
        'implementation_sha256': {str(path.resolve()): sha(path) for path in code_paths}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    frozen = {'requests.jsonl': requests, 'question_map.jsonl': question_map,
              'reused_scores.jsonl': reused}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        for filename, rows in frozen.items():
            if read_rows(args.output / filename) != rows:
                raise ValueError(f'Frozen artifact differs: {filename}')
        if (args.output / 'judge_prompt.txt').read_bytes() != PROMPT.read_bytes():
            raise ValueError('Frozen prompt differs')
        if (args.output / 'methods_manifest.json').read_bytes() != manifest_path.read_bytes():
            raise ValueError('Frozen methods manifest differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without protocol')
        for filename, rows in frozen.items():
            write_rows(args.output / filename, rows)
        (args.output / 'judge_prompt.txt').write_bytes(PROMPT.read_bytes())
        (args.output / 'methods_manifest.json').write_bytes(manifest_path.read_bytes())
        dump(protocol_path, protocol)
    return protocol, requests, question_map

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'analyze'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--env-file', type=Path,
                        default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=5,
                        help='Maximum total attempts per unique request, including earlier runs')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit new unique requests for a smoke run; 0 means all pending')
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error('workers/max-attempts must be positive; limit must be nonnegative')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, requests, question_map = prepare(args)
        if args.phase == 'run':
            report = run(args, protocol, requests, question_map)
            if report['status'] != 'complete' and not args.limit:
                raise SystemExit(2)
        else:
            analyze(args, protocol, requests, question_map)

if __name__ == '__main__':
    main()
