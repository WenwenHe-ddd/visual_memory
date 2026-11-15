"""Score frozen MemGallery predictions with the official five-grade rubric.

This post-generation evaluation intentionally sends gold answers to the judge.
It never calls retrieval or the original reader. Each item is judged once
successfully, with bounded retries for failed attempts and resumable outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
from scripts.evaluation.mem_gallery_master_reader import normalize_answer_universal

SOURCE = ROOT / 'results/mem_gallery_master_answer_20261021'
OUTPUT = ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104'
PROMPT = Path(__file__).with_name('prompts') / 'memgallery_official_five_grade_judge.txt'
PROMPT_URL = 'https://raw.githubusercontent.com/YuanchenBei/Mem-Gallery/main/benchmark/memengine/evaluate/llm_judge.txt'
CONDITION = 'baseline_question_only'
GRADES = (0, 0.25, 0.5, 0.75, 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_map(rows, key='id'):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    temporary.replace(path)


def parse_grade(text):
    """Preserve all five grades. Never round, threshold, or infer from prose."""
    cleaned = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
    if fence:
        cleaned = fence.group(1)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError('Judge output must be a JSON object')
    score, reasoning = value.get('score'), value.get('reasoning')
    if type(score) not in (int, float) or score not in GRADES:
        raise ValueError('Judge score must be one of the five exact numeric grades')
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError('Judge reasoning must be nonempty text')
    return {'score': score, 'reasoning': reasoning}


def prepare(args):
    root = args.source.resolve()
    paths = [root / name for name in (
        'inputs/questions.jsonl', 'inputs/gold.jsonl', 'reader/responses.jsonl',
        'reader/qa_manifest.jsonl', 'reader/protocol.json', 'reader/summary.json')]
    questions = read_rows(paths[0])
    qmap = unique_map(questions)
    gold = unique_map(read_rows(paths[1]))
    responses = unique_map([r for r in read_rows(paths[2]) if r['condition'] == CONDITION])
    manifest = unique_map([r for r in read_rows(paths[3]) if r['condition'] == CONDITION])
    if not len(questions) == 1711 or not set(qmap) == set(gold) == set(responses) == set(manifest):
        raise ValueError('Expected identical coverage of all 1711 source questions')
    template = PROMPT.read_text()
    records = []
    for q in questions:
        qid = q['id']
        r, g, m = responses[qid], gold[qid], manifest[qid]
        if r['status'] != 'ok' or r['hit_token_limit']:
            raise ValueError('Source prediction is failed or truncated')
        if r['payload_sha256'] != m['payload_sha256'] or len(m['top5_memory_ids']) != 5:
            raise ValueError('Source prediction/Top5 manifest mismatch')
        if not q['point'] == g['point'] == r['point'] == m['point']:
            raise ValueError('Source question category mismatch')
        fields = {'question': q['question'], 'ground_truth': g['answer'], 'model_output': r['raw_response']}
        if any(not isinstance(value, str) for value in fields.values()):
            raise ValueError('Judge input fields must be strings')
        # One-pass replacement prevents placeholder-like text in answers from being modified.
        prompt = re.sub(r'\{\{(question|ground_truth|model_output)\}\}',
                        lambda match: fields[match.group(1)], template)
        request = transport.build_request({'user_text': prompt}, model=transport.DEEPSEEK_MODEL,
                                          max_output_tokens=1024)
        records.append({'id': qid, 'task': q['task'], 'point': q['point'], **fields,
            'source_record_id': r['record_id'], 'source_payload_sha256': r['payload_sha256'],
            'em': int(normalize_answer_universal(fields['model_output']) ==
                      normalize_answer_universal(fields['ground_truth'])),
            'request': request, 'request_sha256': transport.request_sha256(request)})
    em_correct = sum(row['em'] for row in records)
    if em_correct != 778:
        raise ValueError(f'Wrong source baseline: EM={em_correct}/1711, expected 778/1711')
    summary_em = json.loads(paths[-1].read_text())['overall'][CONDITION]['em']
    if abs(summary_em - em_correct / 1711) > 1e-12:
        raise ValueError('Recomputed EM differs from saved source summary')
    code_paths = [Path(__file__), Path(transport.__file__),
                  ROOT / 'scripts/evaluation/openai_chat_transport.py',
                  ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']
    protocol = {'dataset': 'MemGallery', 'method': 'Semantic RAG (caption baseline)',
        'answer_model': 'Qwen/Qwen3-VL-8B-Instruct', 'source_condition': CONDITION,
        'source_em_correct': 778, 'questions': 1711, 'source_em': 778 / 1711,
        'judge_model': transport.DEEPSEEK_MODEL, 'judge_model_identity_kind': 'provider_alias',
        'judge_base_url': transport.DASHSCOPE_BASE_URL, 'temperature': 0, 'seed': 42,
        'enable_thinking': False, 'max_tokens': 1024, 'grades': list(GRADES),
        'aggregation': 'Arithmetic mean across all 1711 questions; no missing-score imputation.',
        'judge_input_fields': ['question', 'ground_truth', 'model_output'],
        'images_context_captions_sent_to_judge': False,
        'official_prompt_url': PROMPT_URL, 'official_prompt_sha256': sha(PROMPT),
        'parser_difference_from_public_code': 'Keep 0.25 and 0.75 exactly; no three-level remapping.',
        'source_sha256': {str(p): sha(p) for p in paths},
        'implementation_sha256': {str(p.resolve()): sha(p) for p in code_paths}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing protocol differs; use a new output root')
        if read_rows(args.output / 'requests.jsonl') != records:
            raise ValueError('Frozen judge inputs differ')
        if (args.output / 'judge_prompt.txt').read_bytes() != PROMPT.read_bytes():
            raise ValueError('Frozen judge prompt differs')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output directory is nonempty without protocol')
        write_rows(args.output / 'requests.jsonl', records)
        (args.output / 'judge_prompt.txt').write_bytes(PROMPT.read_bytes())
        dump(protocol_path, protocol)
    return records


def validated_results(output, records):
    path = output / 'scores.jsonl'
    saved = unique_map(read_rows(path)) if path.exists() else {}
    expected = unique_map(records)
    if set(saved) - set(expected):
        raise ValueError('Unknown question IDs in saved scores')
    for qid, row in saved.items():
        source = expected[qid]
        if row['request_sha256'] != source['request_sha256']:
            raise ValueError('Cached response request hash mismatch')
        if row['returned_model'] != transport.DEEPSEEK_MODEL or row['finish_reason'] != 'stop':
            raise ValueError('Cached response model/finish reason mismatch')
        grade = parse_grade(row['raw_response'])
        if any(row[key] != grade[key] for key in grade):
            raise ValueError('Cached parsed grade differs from raw judge response')
    return saved


def analyze(args, records):
    saved = validated_results(args.output, records)
    rows = [{**{k: v for k, v in source.items() if k != 'request'}, **saved[source['id']]}
            for source in records if source['id'] in saved]
    complete = len(rows) == len(records)
    counts = Counter(row['score'] for row in rows)
    score_sum = sum(row['score'] for row in rows)
    def group(field):
        result = {}
        for key in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == key]
            result[str(key)] = {'questions': len(selected),
                'mean_score': sum(row['score'] for row in selected) / len(selected)}
        return result
    report = {'status': 'complete' if complete else 'partial', 'questions_expected': len(records),
        'questions_scored': len(rows), 'questions_missing': len(records) - len(rows),
        'judge_model': transport.DEEPSEEK_MODEL, 'source_em_correct': 778,
        'source_em': 778 / 1711, 'score_sum': score_sum,
        'llm_judge': score_sum / len(records) if complete else None,
        'llm_judge_percent': 100 * score_sum / len(records) if complete else None,
        'score_distribution': {str(g): counts[g] for g in GRADES},
        'by_question_type': group('point'), 'by_scene': group('task'), 'by_source_em': group('em'),
        'successful_request_usage': {key: sum((row.get('usage') or {}).get(key, 0) or 0 for row in rows)
                                     for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'missing_ids': [r['id'] for r in records if r['id'] not in saved],
        'requests_sha256': sha(args.output / 'requests.jsonl'),
        'scores_sha256': sha(args.output / 'scores.jsonl') if saved else None}
    dump(args.output / 'summary.json', report)
    fields = ['id', 'task', 'point', 'em', 'question', 'ground_truth', 'model_output',
              'score', 'reasoning', 'returned_model', 'response_id']
    with (args.output / 'per_question.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: report[key] for key in ('status', 'questions_scored', 'questions_missing',
                                                  'llm_judge', 'score_distribution')}, ensure_ascii=False), flush=True)
    return report


def run(args, records):
    config = transport.load_config(args.env_file)
    saved = validated_results(args.output, records)
    pending = [row for row in records if row['id'] not in saved]
    if args.limit:
        # Smoke tests cover different task types before filling remaining slots.
        seen, diversified, remainder = set(), [], []
        for row in pending:
            if row['point'] not in seen:
                diversified.append(row)
                seen.add(row['point'])
            else:
                remainder.append(row)
        pending = (diversified + remainder)[:args.limit]
    def judge(row):
        attempts = []
        for number in range(1, args.max_attempts + 1):
            audit = {'id': row['id'], 'request_sha256': row['request_sha256'],
                     'attempt': number, 'started_at_unix': time.time()}
            try:
                response = transport.chat_request(config, row['request'], timeout=60)
                audit.update(response)
                if response['finish_reason'] != 'stop' or response.get('refusal'):
                    raise ValueError('Judge response incomplete or refused')
                grade = parse_grade(response['raw_response'])
                audit['status'] = 'ok'
                attempts.append(audit)
                return {'id': row['id'], **response, **grade}, attempts
            except transport.APIError as error:
                audit.update(status='error', error_kind=error.kind,
                             provider_error_code=getattr(error, 'provider_error_code', None))
                retryable = error.retryable
            except (ValueError, TypeError) as error:
                audit.update(status='error', error_kind='invalid_judge_output',
                             error_class=type(error).__name__)
                retryable = True
            attempts.append(audit)
            if not retryable or number == args.max_attempts:
                break
            time.sleep(min(2 ** number, 16))
        return None, attempts
    print(f'Judging {len(pending)} pending items; {len(saved)} validated cached scores', flush=True)
    with (args.output / 'attempts.jsonl').open('a') as log, (args.output / 'scores.jsonl').open('a') as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(judge, row): row['id'] for row in pending}
            for count, future in enumerate(as_completed(futures), 1):
                result, attempts = future.result()
                for attempt in attempts:
                    log.write(json.dumps(attempt, ensure_ascii=False, allow_nan=False) + '\n')
                log.flush()
                if result is not None:
                    out.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    out.flush()
                if count % 50 == 0 or count == len(pending) or result is None:
                    print(f'Finished {count}/{len(pending)}; last_success={result is not None}', flush=True)
    return analyze(args, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run', 'analyze'], required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--env-file', type=Path, default=Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test; 0 means all remaining questions')
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.limit < 0:
        parser.error('workers/max-attempts must be positive; limit must be nonnegative')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / (args.output.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = prepare(args)
        if args.phase == 'prepare':
            print(f'Validated and froze {len(records)} requests; source EM=778/1711')
        elif args.phase == 'run':
            report = run(args, records)
            if report['status'] != 'complete' and not args.limit:
                raise SystemExit(2)
        else:
            analyze(args, records)


if __name__ == '__main__':
    main()
