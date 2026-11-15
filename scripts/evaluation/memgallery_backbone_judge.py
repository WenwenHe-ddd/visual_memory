"""Judge frozen answers from an explicitly named backbone with five exact grades.

The API runner and reports reuse the audited multi-method evaluator. Preparation
accepts completed compatible judge caches, irrespective of the original answer
backbone: the exact wire request contains only question, gold, and prediction.
No retrieval or reader generation is performed here.
"""
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
from scripts.evaluation import memgallery_multi_method_judge as multi

original = multi.original
transport = multi.transport
sha, read_rows, dump, write_rows = multi.sha, multi.read_rows, multi.dump, multi.write_rows
DEFAULT_CACHES = (multi.DEFAULT_CACHE,
                  ROOT / 'results/memgallery_qwen_all_methods_deepseek_judge_20261104')


def verify_report(report, scores, count, mean_key='llm_judge'):
    """Recompute completed-cache aggregates from its parsed raw responses."""
    counts = Counter(row['score'] for row in scores)
    total = sum(row['score'] for row in scores)
    expected = {'status': 'complete', 'questions_scored': count, 'questions_missing': 0,
                'score_sum': total, mean_key: total / count,
                'score_distribution': {str(grade): counts[grade] for grade in multi.GRADES}}
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Completed cache summary does not reproduce: {key}')


def load_completed_cache(root, template, canonical):
    """Validate both historical cache layouts without modifying either source."""
    root = root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    for key, value in multi.judge_settings().items():
        if protocol.get(key) != value:
            raise ValueError(f'Incompatible cache protocol {root}: {key}')
    if (root / 'judge_prompt.txt').read_bytes() != multi.PROMPT.read_bytes():
        raise ValueError(f'Incompatible cache prompt: {root}')
    summary = json.loads((root / 'summary.json').read_text())
    if 'methods' not in protocol:
        # This validator verifies request reconstruction, complete 1711 coverage,
        # exact response metadata/raw JSON, and requests/scores summary hashes.
        sources, _, hashes = multi.load_baseline_cache(root, template)
        if set(sources) != set(canonical):
            raise ValueError('Canonical baseline cache IDs differ')
        scores_by_id = original.unique_map(read_rows(root / 'scores.jsonl'))
        verify_report(summary, list(scores_by_id.values()), 1711)
        entries = [(source['request_sha256'], scores_by_id[qid], {'id': qid})
                   for qid, source in sources.items()]
        return entries, hashes

    required = {'protocol.json', 'requests.jsonl', 'question_map.jsonl',
                'reused_scores.jsonl', 'scores.jsonl', 'attempts.jsonl'}
    artifacts = summary.get('artifact_sha256', {})
    if not required <= artifacts.keys():
        raise ValueError('Completed multi-method cache lacks required artifact hashes')
    hashes = {}
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        if path.parent != root or sha(path) != expected:
            raise ValueError(f'Completed cache artifact hash differs: {name}')
        hashes[str(path)] = expected
    for name in ('summary.json', 'judge_prompt.txt'):
        hashes[str(root / name)] = sha(root / name)
    requests = read_rows(root / 'requests.jsonl')
    request_map = original.unique_map(requests, 'request_sha256')
    for digest, row in request_map.items():
        rebuilt = multi.build_request(row, template)
        if row['request'] != rebuilt or transport.request_sha256(rebuilt) != digest:
            raise ValueError('Completed cache request does not reproduce from raw input')
    scores = multi.validated_results(root, requests)
    question_map = read_rows(root / 'question_map.jsonl')
    methods = original.unique_map(protocol['methods'])
    reports = original.unique_map(summary['methods'], 'method_id')
    if set(methods) != set(reports) or len(question_map) != 1711 * len(methods):
        raise ValueError('Completed cache method coverage differs')
    if (summary.get('status') != 'complete' or set(scores) != set(request_map) or
            summary.get('unique_requests_scored') != len(requests) or
            summary.get('unique_requests_missing') != 0 or
            summary.get('total_scored_method_questions') != len(question_map)):
        raise ValueError('Multi-method cache is not completely scored')
    for key, value in {'method_count': len(methods), 'unique_requests': len(requests),
                       'total_method_questions': len(question_map)}.items():
        if protocol.get(key) != value or summary.get(key) != value:
            raise ValueError(f'Completed cache coverage count differs: {key}')
    uses = {digest: [] for digest in request_map}
    for mid, method in methods.items():
        rows = [row for row in question_map if row['method_id'] == mid]
        by_id = original.unique_map(rows)
        if len(rows) != 1711 or set(by_id) != set(canonical):
            raise ValueError(f'Completed cache method lacks canonical coverage: {mid}')
        correct = 0
        for row in rows:
            if any(row[key] != canonical[row['id']][key]
                   for key in ('task', 'point', 'question', 'ground_truth')):
                raise ValueError('Completed cache changed canonical question/gold')
            em = int(multi.normalize_answer(row['model_output']) ==
                     multi.normalize_answer(row['ground_truth']))
            if row['em'] != em:
                raise ValueError('Completed cache per-question EM does not reproduce')
            correct += em
            request = multi.build_request(row, template)
            digest = transport.request_sha256(request)
            if row['request_sha256'] != digest or digest not in scores:
                raise ValueError('Completed cache question-to-request mapping differs')
            uses[digest].append({'method_id': mid, 'id': row['id']})
        if correct != method['expected_em_correct'] or correct != reports[mid]['source_em_correct']:
            raise ValueError('Completed cache original EM differs')
        verify_report(reports[mid], [scores[row['request_sha256']] for row in rows], 1711)
    for digest, expected_uses in uses.items():
        if not expected_uses or request_map[digest].get('uses') != expected_uses:
            raise ValueError('Completed cache request uses differ')
    entries = [(digest, score, {'uses': uses[digest]}) for digest, score in scores.items()]
    return entries, hashes


def collect_caches(roots, template, canonical):
    cached, hashes, sources = {}, {}, []
    for root in roots:
        root = root.resolve()
        entries, source_hashes = load_completed_cache(root, template, canonical)
        hashes.update(source_hashes)
        sources.append({'root': str(root), 'validated_score_rows': len(entries),
                        'unique_request_hashes': len({entry[0] for entry in entries}),
                        'summary_sha256': sha(root / 'summary.json')})
        for digest, response, location in entries:
            provenance = {'root': str(root), **location,
                          'summary_sha256': sha(root / 'summary.json'),
                          'response_id': response.get('response_id'),
                          'source_response': response}
            if digest in cached:
                if cached[digest]['score'] != response['score']:
                    raise ValueError(f'Conflicting grades for the same request hash: {digest}')
                cached[digest]['reuse_sources'].append(provenance)
            else:
                cached[digest] = {
                    **{key: value for key, value in response.items()
                       if key not in ('id', 'reuse_sources', 'score_origin')},
                    # Historical machine field required by the unchanged audited
                    # runner; source_root/reuse_sources define the actual source.
                    'score_origin': 'baseline_cache', 'source_root': str(root),
                    'reuse_kind': 'compatible_completed_judge_cache',
                    'reuse_sources': [provenance]}
    return cached, hashes, sources


def prepare(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    answer_model = manifest.get('answer_model')
    if not isinstance(answer_model, str) or not answer_model.strip():
        raise ValueError('Manifest must explicitly identify answer_model')
    methods = manifest.get('methods')
    if not isinstance(methods, list) or not methods or manifest.get('questions_per_method', 1711) != 1711:
        raise ValueError('Manifest requires methods and the complete 1711-question cohort')
    original.unique_map(methods)
    template = multi.PROMPT.read_text()
    canonical, _, canonical_hashes = multi.load_baseline_cache(args.baseline_cache.resolve(), template)
    cache_roots = args.cache_root if args.cache_root is not None else DEFAULT_CACHES
    cache_roots = list(dict.fromkeys(path.resolve() for path in cache_roots))
    cached, cache_hashes, cache_sources = collect_caches(cache_roots, template, canonical)
    requests, question_map, method_protocols, source_hashes = {}, [], [], {}

    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else manifest_path.parent / path).resolve()

    for method in methods:
        mid = method['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', mid) or not method.get('label', '').strip():
            raise ValueError('Methods need safe IDs and nonempty labels')
        if method.get('answer_model', answer_model) != answer_model:
            raise ValueError(f'Method {mid} answer model differs from the manifest')
        provenance = method.get('source_sha256')
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f'Method {mid} requires original source hashes')
        for source, expected in provenance.items():
            path = resolve(source)
            if str(path) not in source_hashes:
                source_hashes[str(path)] = sha(path)
            if source_hashes[str(path)] != expected:
                raise ValueError(f'Original source hash differs: {path}')
        path = resolve(method['records_path'])
        records_hash = sha(path)
        if method.get('records_sha256') != records_hash:
            raise ValueError(f'Method {mid} requires matching frozen records_sha256')
        records = read_rows(path)
        if len(records) != 1711 or set(original.unique_map(records)) != set(canonical):
            raise ValueError(f'Method {mid} lacks complete canonical IDs')
        correct = 0
        for row in records:
            if any(key not in row for key in multi.REQUIRED_FIELDS):
                raise ValueError(f'Method {mid} record lacks required fields')
            if any(not isinstance(row[key], str) for key in multi.REQUIRED_FIELDS if key != 'em'):
                raise ValueError('Frozen judge inputs must be original strings')
            if any(row[key] != canonical[row['id']][key]
                   for key in ('task', 'point', 'question', 'ground_truth')):
                raise ValueError(f'Method {mid} changed the canonical question/gold')
            em = int(multi.normalize_answer(row['model_output']) ==
                     multi.normalize_answer(row['ground_truth']))
            if type(row['em']) not in (int, bool) or row['em'] != em:
                raise ValueError(f'Method {mid} source EM does not reproduce')
            correct += em
            request = multi.build_request(row, template)
            digest = transport.request_sha256(request)
            use = {'method_id': mid, 'id': row['id']}
            if digest in requests:
                if requests[digest]['request'] != request:
                    raise ValueError('Request hash collision')
                requests[digest]['uses'].append(use)
            else:
                requests[digest] = {'request_sha256': digest, 'request': request,
                    **{key: row[key] for key in ('question', 'ground_truth', 'model_output')},
                    'point': row['point'], 'uses': [use]}
            question_map.append({**row, 'method_id': mid, 'request_sha256': digest})
        if correct != method.get('expected_em_correct'):
            raise ValueError(f'Method {mid}: original EM {correct}/1711 differs')
        if 'em' in method and abs(method['em'] - correct / 1711) > 1e-12:
            raise ValueError(f'Method {mid}: source EM fraction differs')
        method_protocols.append({**method, 'answer_model': answer_model,
                                 'records_path': str(path), 'records_sha256': records_hash,
                                 'questions': 1711, 'em': correct / 1711})
    reused = [cached[digest] for digest in requests if digest in cached]
    code_paths = [Path(__file__), Path(multi.__file__), Path(original.__file__),
                  Path(transport.__file__), ROOT / 'scripts/evaluation/openai_chat_transport.py',
                  ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemGallery', 'answer_model': answer_model,
        **multi.judge_settings(), 'judge_model_identity_kind': 'provider_alias',
        'official_prompt_url': original.PROMPT_URL,
        'parser_difference_from_public_code': 'Keep all five grades exactly; no three-level remapping.',
        'aggregation': 'Per-method arithmetic mean over all 1711 questions; no missing-score imputation.',
        'deduplication': 'One score per exact serialized request hash across methods and compatible caches.',
        'original_answer_policy': 'Keep frozen original backbone answers, including source token-limit outputs.',
        'cache_reuse_policy': 'Exact wire input equality; answer backbone is not a judge input. First compatible cache wins; conflicting grades fail.',
        'score_origin_baseline_cache_meaning': 'Compatible completed judge cache; actual sources are source_root and reuse_sources.',
        'methods': method_protocols, 'method_count': len(methods),
        'total_method_questions': len(question_map), 'unique_requests': len(requests),
        'baseline_reused_unique_requests': len(reused),
        'compatible_cache_reused_unique_requests': len(reused),
        'new_unique_requests_expected': len(requests) - len(reused),
        'compatible_cache_sources': cache_sources, 'compatible_cache_sha256': cache_hashes,
        'canonical_question_cache_root': str(args.baseline_cache.resolve()),
        'canonical_question_cache_sha256': canonical_hashes,
        'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
        'source_sha256': source_hashes,
        'implementation_sha256': {str(path.resolve()): sha(path) for path in code_paths}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    frozen = {'requests.jsonl': list(requests.values()), 'question_map.jsonl': question_map,
              'reused_scores.jsonl': reused}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        for filename, rows in frozen.items():
            if read_rows(args.output / filename) != rows:
                raise ValueError(f'Frozen artifact differs: {filename}')
        if ((args.output / 'judge_prompt.txt').read_bytes() != multi.PROMPT.read_bytes() or
                (args.output / 'methods_manifest.json').read_bytes() != manifest_path.read_bytes()):
            raise ValueError('Frozen prompt or methods manifest differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without protocol')
        for filename, rows in frozen.items():
            write_rows(args.output / filename, rows)
        (args.output / 'judge_prompt.txt').write_bytes(multi.PROMPT.read_bytes())
        (args.output / 'methods_manifest.json').write_bytes(manifest_path.read_bytes())
        dump(protocol_path, protocol)
    return protocol, list(requests.values()), question_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze'), required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-cache', type=Path, default=multi.DEFAULT_CACHE,
                        help='Validated canonical question/gold cache (not the answer backbone)')
    parser.add_argument('--cache-root', type=Path, action='append',
                        help='Repeatable ordered completed caches; defaults to baseline and Qwen all-methods')
    parser.add_argument('--env-file', type=Path,
                        default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error('workers/max-attempts must be positive; limit must be nonnegative')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, requests, question_map = prepare(args)
        if args.phase == 'run':
            result = multi.run(args, protocol, requests, question_map)
            if result['status'] != 'complete' and not args.limit:
                raise SystemExit(2)
        else:
            multi.analyze(args, protocol, requests, question_map)


if __name__ == '__main__':
    main()
