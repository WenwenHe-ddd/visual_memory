"""Five-grade judging of GPT-5 answers on the canonical MemEye Open cohort.

Reuse is limited to completed, independently validated 371-question caches with
the exact same judge prompt and settings. A result is reused only for an identical
wire request. Cache roots and file hashes are frozen on the first preparation.
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
from scripts.evaluation import memeye_open_judge_engine as base

shared, transport = base.shared, base.transport
sha, read_rows, dump, write_rows = base.sha, base.read_rows, base.dump, base.write_rows
COHORT_SIZE = 371
CANONICAL_ROOT = ROOT / 'results/memeye_qwen_semrag_open_20261105'
DEFAULT_CACHE = CANONICAL_ROOT / 'judge'
RESPONSE_FIELDS = frozenset('model_identity_kind resolved_model_revision reasoning_content_present '
    'enable_thinking seed raw_response returned_model requested_model response_id system_fingerprint '
    'finish_reason usage reasoning_tokens request_id elapsed_seconds refusal request_sha256 attempts '
    'retry_count score reasoning recovered_from_successful_attempt_log'.split())
RESERVED_INPUT = frozenset('request request_sha256 score reasoning score_origin judge_status '
    'returned_model requested_model reuse_sources source_root reuse_kind method_id method'.split())


def read(path):
    return json.loads(Path(path).read_text())


def unique(rows, key='id'):
    mapped = {row[key]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return mapped


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source or implementation changed: {path}')


def load_canonical():
    paths = {name: CANONICAL_ROOT / 'inputs' / name for name in ('questions.jsonl', 'gold.jsonl')}
    protocol = read(CANONICAL_ROOT / 'protocol.json')
    for name, path in paths.items():
        if sha(path) != protocol['artifact_sha256']['inputs/' + name]:
            raise ValueError('Canonical Open inputs differ from their frozen protocol')
    questions, gold = (unique(read_rows(paths[name])) for name in ('questions.jsonl', 'gold.jsonl'))
    if len(questions) != COHORT_SIZE or set(questions) != set(gold):
        raise ValueError('Canonical Open cohort is incomplete')
    canonical = {}
    for qid, q in questions.items():
        if gold[qid]['point'] != q['point']:
            raise ValueError('Canonical question/gold category differs')
        canonical[qid] = {key: q[key] for key in ('id', 'task', 'point', 'question')}
        canonical[qid]['ground_truth'] = gold[qid]['answer']
    hashes = {str(path): sha(path) for path in [*paths.values(), CANONICAL_ROOT / 'protocol.json']}
    return canonical, hashes


def validate_source_row(row, canonical, *, allow_request_hash=False, answer_model=None):
    if any(not isinstance(row.get(key), str) for key in base.FIELDS):
        raise ValueError('Judge fields must contain original strings')
    if any(not row[key].strip() for key in base.FIELDS if key != 'model_output'):
        raise ValueError('Only the frozen original answer may be empty')
    forbidden = RESERVED_INPUT - ({'request_sha256'} if allow_request_hash else set())
    if forbidden.intersection(row):
        raise ValueError('Input contains reserved judge result fields')
    reference = canonical.get(row['id'])
    if reference is None or any(row[key] != reference[key]
                               for key in ('task', 'point', 'question', 'ground_truth')):
        raise ValueError('Question ID, text, category, or gold differs from canonical MemEye Open')
    expected = answer_model or base.ANSWER_MODEL
    allowed = {'gpt-5', 'gpt-5-2025-08-07'} if expected in {'gpt-5', 'gpt-5-2025-08-07'} else {expected}
    if expected not in {'gpt-5', 'gpt-5-2025-08-07', 'Qwen/Qwen3-VL-8B-Instruct'}:
        raise ValueError('Unsupported completed-cache answer backbone')
    if row.get('answer_model', expected) not in allowed:
        raise ValueError('Source answer backbone differs from its experiment')
    em = int(shared.normalize_answer(row['model_output']) == shared.normalize_answer(row['ground_truth']))
    if 'em' in row and (type(row['em']) not in (bool, int) or row['em'] != em):
        raise ValueError('Internal original-answer EM does not reproduce')
    return em


def load_completed_cache(root, template, canonical):
    root = Path(root).resolve()
    protocol, report = read(root / 'protocol.json'), read(root / 'summary.json')
    for key, expected in shared.judge_settings().items():
        if protocol.get(key) != expected:
            raise ValueError(f'Incompatible completed cache settings: {root}: {key}')
    if (protocol.get('dataset') != 'MemEye' or protocol.get('task') != 'open_ended_answering'
            or protocol.get('questions') != COHORT_SIZE):
        raise ValueError('Cache is not the full MemEye Open experiment')
    if (root / 'judge_prompt.txt').read_bytes() != shared.PROMPT.read_bytes():
        raise ValueError('Cache judge prompt bytes differ')
    if report.get('status') != 'complete':
        raise ValueError('Only complete caches may be reused')
    artifacts = report.get('artifact_sha256', {})
    required = {'protocol.json', 'requests.jsonl', 'question_map.jsonl', 'reused_scores.jsonl'}
    required.update(name for name in ('scores.jsonl', 'attempts.jsonl') if (root / name).exists())
    if not required <= artifacts.keys():
        raise ValueError('Completed cache lacks required artifact hashes')
    hashes = {}
    for name, expected in artifacts.items():
        path = (root / name).resolve()
        if path.parent != root or sha(path) != expected:
            raise ValueError(f'Completed cache artifact differs: {name}')
        hashes[str(path)] = expected
    for field in ('source_sha256', 'implementation_sha256', 'canonical_source_sha256', 'cache_source_sha256'):
        source_hashes = protocol.get(field, {})
        verify_hashes(source_hashes)
        hashes.update(source_hashes)
    requests = read_rows(root / 'requests.jsonl')
    request_map = unique(requests, 'request_sha256')
    for digest, row in request_map.items():
        request = shared.build_request(row, template)
        if request != row.get('request') or transport.request_sha256(request) != digest:
            raise ValueError('Cached wire request does not reproduce')
    question_map = read_rows(root / 'question_map.jsonl')
    by_id = unique(question_map)
    if len(question_map) != COHORT_SIZE or set(by_id) != set(canonical):
        raise ValueError('Cache does not contain all 371 canonical questions')
    scores = shared.validated_results(root, requests)
    if set(scores) != set(request_map):
        raise ValueError('Completed cache has missing scores')
    uses = {digest: [] for digest in request_map}
    em_count, ordered_scores = 0, []
    for row in question_map:
        em_count += validate_source_row(row, canonical, allow_request_hash=True, answer_model=protocol['answer_model'])
        rebuilt = shared.build_request(row, template)
        digest = transport.request_sha256(rebuilt)
        if row.get('request_sha256') != digest or digest not in request_map:
            raise ValueError('Cached question-to-wire mapping differs')
        if request_map[digest]['request'] != rebuilt:
            raise ValueError('Cached source strings do not match their wire request')
        uses[digest].append(row['id'])
        ordered_scores.append(scores[digest])
    for digest, ids in uses.items():
        if not ids or request_map[digest].get('question_ids') != ids:
            raise ValueError('Cached request coverage does not reproduce')
    counts = Counter(row['score'] for row in ordered_scores)
    total = sum(row['score'] for row in ordered_scores)
    expected_report = {'status': 'complete', 'dataset': 'MemEye', 'task': 'open_ended_answering',
        'answer_model': protocol['answer_model'],
        'questions_expected': COHORT_SIZE, 'questions_scored': COHORT_SIZE, 'questions_missing': 0,
        'unique_requests': len(requests), 'unique_requests_scored': len(requests),
        'score_sum': total, 'llm_judge': total / COHORT_SIZE,
        'llm_judge_percent': 100 * total / COHORT_SIZE,
        'score_distribution': {str(grade): counts[grade] for grade in shared.GRADES},
        'source_open_em_correct': em_count, 'source_open_em': em_count / COHORT_SIZE,
        'missing_ids': [], 'judge_model': transport.DEEPSEEK_MODEL, 'grades': list(shared.GRADES)}
    for key, expected in expected_report.items():
        if report.get(key) != expected:
            raise ValueError(f'Completed cache aggregate does not reproduce: {key}')
    if protocol.get('unique_requests') != len(requests) or protocol.get('source_open_em_correct') != em_count:
        raise ValueError('Cache protocol counts do not reproduce')
    if report.get('method') != protocol.get('method'):
        raise ValueError('Cache method label differs between summary and protocol')
    attempts_path = root / 'attempts.jsonl'
    attempts = read_rows(attempts_path) if attempts_path.exists() else []
    previous = {}
    for attempt in attempts:
        digest = attempt.get('request_sha256')
        if digest not in request_map or attempt.get('attempt') != previous.get(digest, {}).get('attempt', 0) + 1:
            raise ValueError('Cache attempt sequence differs')
        previous[digest] = attempt
    for digest, score in scores.items():
        if score['score_origin'] == 'new_api':
            last = previous.get(digest, {})
            if last.get('status') != 'ok' or last.get('raw_response') != score['raw_response']:
                raise ValueError('Cached API score is not backed by its successful attempt')
    if (report.get('api_attempt_count') != len(attempts) or
            report.get('failed_api_attempt_count') != sum(r.get('status') != 'ok' for r in attempts)):
        raise ValueError('Cache API attempt totals differ')
    expected_rows = [{**source, **scores[source['request_sha256']], 'judge_status': 'ok'} for source in question_map]
    if read_rows(root / 'per_question.jsonl') != expected_rows:
        raise ValueError('Cache per-question export differs from frozen inputs and raw scores')
    for name in ('summary.json', 'judge_prompt.txt', 'per_question.jsonl', 'per_question.csv'):
        path = root / name
        if path.exists():
            hashes[str(path)] = sha(path)
    entries = [(digest, score, {'question_ids': uses[digest]}) for digest, score in scores.items()]
    return entries, hashes


def collect_caches(roots, template, canonical):
    cached, hashes, sources = {}, {}, []
    for root in roots:
        root = Path(root).resolve()
        entries, source_hashes = load_completed_cache(root, template, canonical)
        hashes.update(source_hashes)
        sources.append({'root': str(root), 'questions': COHORT_SIZE, 'unique_requests': len(entries),
                        'summary_sha256': sha(root / 'summary.json')})
        for digest, response, location in entries:
            provenance = {'root': str(root), **location,
                          'summary_sha256': sha(root / 'summary.json'),
                          'response_id': response.get('response_id'),
                          'source_score_origin': response['score_origin'],
                          'source_request_sha256': digest}
            if digest in cached:
                if cached[digest]['score'] != response['score']:
                    raise ValueError(f'Conflicting grades for identical wire request: {digest}')
                cached[digest]['reuse_sources'].append(provenance)
            else:
                cached[digest] = {key: value for key, value in response.items() if key in RESPONSE_FIELDS}
                cached[digest].update(score_origin='baseline_cache', source_root=str(root),
                    reuse_kind='compatible_completed_memeye_open_cache', reuse_sources=[provenance])
                shared.validate_score(cached[digest], digest)
    return cached, hashes, sources


def prepare(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.method_id):
        raise ValueError('method-id must be a safe nonempty identifier')
    if not isinstance(args.method_label, str) or not args.method_label.strip():
        raise ValueError('method-label must be nonempty')
    args.output = args.output.resolve()
    existing_path = args.output / 'protocol.json'
    previous = read(existing_path) if existing_path.exists() else None
    requested_roots = args.cache_root
    if requested_roots is None:
        requested_roots = previous['cache_roots'] if previous else [DEFAULT_CACHE]
    cache_roots = list(dict.fromkeys(Path(path).resolve() for path in requested_roots))
    if args.output in cache_roots:
        raise ValueError('Output cannot reuse itself as a cache')
    if previous and [str(path) for path in cache_roots] != previous['cache_roots']:
        raise ValueError('Cache roots are frozen; resume cannot add, remove or reorder caches')
    if previous:
        for field in ('source_sha256', 'implementation_sha256', 'canonical_source_sha256', 'cache_source_sha256'):
            verify_hashes(previous[field])
    canonical, canonical_hashes = load_canonical()
    source = args.input.resolve()
    records = read_rows(source)
    if len(records) != COHORT_SIZE or set(unique(records)) != set(canonical):
        raise ValueError('Input must contain all 371 unique canonical Open question IDs')
    template = shared.PROMPT.read_text()
    cached, cache_hashes, cache_sources = collect_caches(cache_roots, template, canonical)
    requests, question_map = {}, []
    for row in records:
        em = validate_source_row(row, canonical)
        request = shared.build_request(row, template)
        digest = transport.request_sha256(request)
        if digest in requests:
            if requests[digest]['request'] != request:
                raise ValueError('Wire request hash collision')
            requests[digest]['question_ids'].append(row['id'])
        else:
            requests[digest] = {'request_sha256': digest, 'request': request,
                **{key: row[key] for key in ('question', 'ground_truth', 'model_output')},
                'point': row['point'], 'question_ids': [row['id']]}
        question_map.append({**row, 'em': em, 'request_sha256': digest})
    reused = [cached[digest] for digest in requests if digest in cached]
    sources = [source]
    if args.source_protocol is not None:
        sources.append(args.source_protocol.resolve())
    code = [Path(__file__), Path(base.__file__), Path(base.original.__file__), Path(shared.__file__), Path(shared.original.__file__),
            Path(transport.__file__), ROOT / 'scripts/evaluation/openai_chat_transport.py',
            ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemEye', 'task': 'open_ended_answering', 'method_id': args.method_id,
        'method': args.method_label, 'answer_model': base.ANSWER_MODEL, 'questions': COHORT_SIZE,
        'accepted_source_answer_models': ['gpt-5', 'gpt-5-2025-08-07'],
        'expected_answer_revision': 'gpt-5-2025-08-07',
        'source_open_em_correct': sum(row['em'] for row in question_map),
        **shared.judge_settings(), 'judge_model_identity_kind': 'provider_alias',
        'official_prompt_url': shared.original.PROMPT_URL,
        'parser_difference_from_public_code': 'Keep all five grades exactly; no three-level remapping.',
        'aggregation': 'Arithmetic mean over all 371 questions; no missing-score imputation.',
        'original_answer_policy': 'Preserve original answers including empty and truncated outputs.',
        'deduplication': 'Exact wire request hash only; all 371 questions remain in the denominator.',
        'cross_dataset_score_cache_reuse': False, 'unique_requests': len(requests),
        'cached_unique_requests': len(reused), 'new_unique_requests': len(requests) - len(reused),
        'max_attempts_per_unique_request': args.max_attempts, 'input_path': str(source),
        'source_sha256': {str(path): sha(path) for path in sources},
        'implementation_sha256': {str(path.resolve()): sha(path) for path in code},
        'canonical_source_sha256': canonical_hashes, 'cache_source_sha256': cache_hashes,
        'cache_roots': [str(path) for path in cache_roots], 'completed_cache_sources': cache_sources,
        'cache_policy': 'Validate completed 371-question caches, exact prompt/settings/wire/raw grade, summary and artifact hashes. Reject conflicting grades. Freeze ordered cache roots and all hashes for resume.',
        'user_facing_summary': 'llm_judge_summary.json contains Judge and completeness only; EM retained solely for internal source checks.'}
    frozen = {'requests.jsonl': list(requests.values()), 'question_map.jsonl': question_map,
              'reused_scores.jsonl': reused}
    args.output.mkdir(parents=True, exist_ok=True)
    if previous:
        if previous != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        for name, values in frozen.items():
            if read_rows(args.output / name) != values:
                raise ValueError(f'Frozen judge artifact differs: {name}')
        if (args.output / 'judge_prompt.txt').read_bytes() != shared.PROMPT.read_bytes():
            raise ValueError('Frozen output prompt differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output is nonempty without a protocol')
        for name, values in frozen.items():
            write_rows(args.output / name, values)
        (args.output / 'judge_prompt.txt').write_bytes(shared.PROMPT.read_bytes())
        dump(existing_path, protocol)
    return protocol, list(requests.values()), question_map


def publish_summary(args, protocol, report):
    public = {'status': report['status'], 'dataset': 'MemEye Open', 'method_id': protocol['method_id'],
        'method': protocol['method'], 'answer_model': base.ANSWER_MODEL,
        'judge_model': transport.DEEPSEEK_MODEL, 'questions_expected': COHORT_SIZE,
        'questions_scored': report['questions_scored'], 'questions_missing': report['questions_missing'],
        'llm_judge': report['llm_judge'], 'score_sum': report['score_sum'],
        'score_distribution': report['score_distribution'], 'grades': list(shared.GRADES),
        'summary_sha256': sha(args.output / 'summary.json')}
    dump(args.output / 'llm_judge_summary.json', public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze'), required=True)
    parser.add_argument('--method-id', required=True)
    parser.add_argument('--method-label', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-protocol', type=Path)
    parser.add_argument('--cache-root', type=Path, action='append',
                        help='Completed 371-question judge cache; repeat for several. Frozen after prepare.')
    parser.add_argument('--env-file', type=Path, default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
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
        report = (base.run if args.phase == 'run' else base.analyze)(args, protocol, requests, question_map)
        publish_summary(args, protocol, report)
        if args.phase == 'run' and report['status'] != 'complete' and not args.limit:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
