"""Verify the complete GPT-5 MemEye Open answer/grade chain and export Judge only."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memgallery_multi_method_judge as judge

CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105'
DEFAULT_ROOT = ROOT / 'results/memeye_gpt5_open_20261105'
MODEL = 'gpt-5'
REVISION = 'gpt-5-2025-08-07'
METHOD_DIRS = {**{m: 'caption_methods/' + m for m in ('semrag', 'ours_1', 'ours_2', 'ours_1_ours_2')},
               **{m: 'text_methods/' + m for m in ('gen_agents', 'memoryos', 'simplemem', 'a_mem')},
               **{m: 'visual_methods/' + m for m in ('clip_image_dialogue', 'hippomm')}}
HASH_CACHE = {}
GRADE_STORES = {}


def read(path):
    return json.loads(Path(path).read_text())


def checked_hash(path, expected=None):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        HASH_CACHE[key] = judge.sha(path)
    actual = HASH_CACHE[key]
    if expected is not None and actual != expected:
        raise ValueError('Frozen file changed: ' + str(path))
    return actual


def verify_hashes(values, base=None):
    for name, expected in values.items():
        path = Path(name)
        if not path.is_absolute() and base is not None:
            path = Path(base) / path
        checked_hash(path, expected)


def grade_store(root, visiting=None):
    """Follow each reused grade back to its successful raw API attempt."""
    root = Path(root).resolve()
    if str(root) in GRADE_STORES:
        return GRADE_STORES[str(root)]
    visiting = set() if visiting is None else set(visiting)
    if str(root) in visiting:
        raise ValueError('Cyclic grade cache provenance')
    visiting.add(str(root))
    protocol, summary = read(root / 'protocol.json'), read(root / 'summary.json')
    if summary.get('status') != 'complete' or summary.get('questions_scored') != 371:
        raise ValueError('Grade cache must cover 371 complete Open questions')
    for k, v in judge.judge_settings().items():
        if protocol.get(k) != v:
            raise ValueError('Cache judge setting differs: ' + k)
    if (root / 'judge_prompt.txt').read_bytes() != judge.PROMPT.read_bytes():
        raise ValueError('Cached judge prompt differs')
    verify_hashes(summary['artifact_sha256'], root)
    for field in ('source_sha256', 'implementation_sha256', 'cache_source_sha256', 'canonical_source_sha256'):
        verify_hashes(protocol.get(field, {}))
    requests = judge.read_rows(root / 'requests.jsonl')
    request_map = {r['request_sha256']: r for r in requests}
    if len(request_map) != len(requests):
        raise ValueError('Duplicate cached Judge request')
    template = judge.PROMPT.read_text()
    for key, r in request_map.items():
        wire = judge.build_request(r, template)
        if wire != r['request'] or judge.transport.request_sha256(wire) != key:
            raise ValueError('Cached judge wire does not reproduce')
    scores = judge.validated_results(root, requests)
    if set(scores) != set(request_map):
        raise ValueError('Missing grade in completed cache')
    attempts = judge.read_rows(root / 'attempts.jsonl') if (root / 'attempts.jsonl').exists() else []
    grouped = {}
    for a in attempts:
        key = a['request_sha256']; seq = grouped.setdefault(key, [])
        if key not in request_map or a['attempt'] != len(seq) + 1:
            raise ValueError('Judge attempt sequence differs')
        seq.append(a)
    for key, score in scores.items():
        if score['score_origin'] == 'new_api':
            a = grouped.get(key, [{}])[-1]
            if (a.get('status') != 'ok' or a.get('raw_response') != score['raw_response']
                    or a.get('response_id') != score.get('response_id')):
                raise ValueError('Judge grade lacks its successful raw attempt')
        else:
            origins = score.get('reuse_sources', [])
            if not origins or Path(score['source_root']).resolve() != Path(origins[0]['root']).resolve():
                raise ValueError('Reused grade has no exact primary provenance')
            for i, origin in enumerate(origins):
                previous = Path(origin['root']).resolve()
                checked_hash(previous / 'summary.json', origin['summary_sha256'])
                old_scores, old_requests = grade_store(previous, visiting)
                if origin['source_request_sha256'] != key or key not in old_scores:
                    raise ValueError('Reused grade request provenance differs')
                prior = old_scores[key]
                if old_requests[key]['request'] != request_map[key]['request'] or prior['score'] != score['score']:
                    raise ValueError('Reused grade wire or grade differs')
                if origin.get('response_id') != prior.get('response_id'):
                    raise ValueError('Reused grade source response identity differs')
                if i == 0 and any(score.get(k) != prior.get(k) for k in ('raw_response', 'reasoning', 'response_id')):
                    raise ValueError('Reused raw grade differs from primary source')
    question_map = judge.read_rows(root / 'question_map.jsonl')
    if len(question_map) != len(unique(question_map)) or len(question_map) != 371:
        raise ValueError('Grade cache question cohort differs')
    total = sum(scores[r['request_sha256']]['score'] for r in question_map)
    distribution = {str(g): sum(scores[r['request_sha256']]['score'] == g for r in question_map) for g in judge.GRADES}
    if (summary['score_sum'] != total or summary['llm_judge'] != total / 371
            or summary['score_distribution'] != distribution):
        raise ValueError('Cached score aggregate differs')
    expected_rows = [{**r, **scores[r['request_sha256']], 'judge_status': 'ok'} for r in question_map]
    if judge.read_rows(root / 'per_question.jsonl') != expected_rows:
        raise ValueError('Cached per-question rows differ from raw grading records')
    GRADE_STORES[str(root)] = scores, request_map
    return scores, request_map


def generic_answers(root, method, questions):
    from scripts.evaluation import memeye_gpt5_open_reader as reader
    protocol = read(root / 'reader/protocol.json'); contract = protocol['contract']
    if reader.digest(contract) != protocol['experiment_signature']:
        raise ValueError('Reader signature mismatch')
    if (contract['method_id'] != method['id'] or contract['answer_model'] != MODEL
            or contract['expected_returned_model'] != REVISION
            or contract['reference_mcq_correct'] != method['source_mcq_correct_for_identification_only']):
        raise ValueError('Reader backbone/method/source mapping differs')
    if contract['settings'] != reader.api.settings(MODEL):
        raise ValueError('GPT reader settings differ')
    verify_hashes(contract['source_sha256']); verify_hashes(contract['image_sha256'])
    manifests = unique(judge.read_rows(root / 'reader_manifest.jsonl'))
    answers = unique(judge.read_rows(root / 'reader/responses.jsonl'))
    if set(manifests) != set(questions) or set(answers) != set(questions):
        raise ValueError('Reader/manifest coverage differs')
    provenance = {}
    for qid, q in questions.items():
        row, source = answers[qid], manifests[qid]
        if any(source[k] != q[k] for k in ('question', 'task', 'point')):
            raise ValueError('Reader uses a changed Open question')
        wire = reader.build_request(source)
        key = reader.digest(wire)
        original = reader.cached_output(Path(contract['cache_root']), wire)
        if any(row.get(k) != v for k, v in original.items()):
            raise ValueError('Original answer differs from durable raw GPT output')
        if (row['experiment_signature'] != protocol['experiment_signature'] or row['method_id'] != method['id']
                or row['answer_model'] != MODEL or row['returned_model'] != REVISION):
            raise ValueError('Original answer identity differs')
        response_path = Path(contract['cache_root']) / 'calls' / key / 'response.json'
        provenance[qid] = {'source_answer_api_request_sha256': key, 'source_answer_record': str(response_path),
            'source_answer_record_sha256': checked_hash(response_path), 'answer_model_revision': row['returned_model']}
    seal = read(root / 'reader/completion.json')
    checked_hash(root / 'reader/protocol.json', seal['protocol_sha256'])
    return answers, provenance


def hippo_answers(root, method, questions):
    from scripts.evaluation import memeye_gpt5_open_visual_methods as native
    from scripts.memory.hippomm_core import parse_fast
    collection = read(root / 'collection_protocol.json')
    validation = read(root / 'validation_final.json')
    protocol = read(root / 'protocol.json')
    if (collection['status'] != 'complete' or collection['answer_model'] != REVISION
            or collection['reference_mcq_correct'] != method['source_mcq_correct_for_identification_only']
            or validation['native_full_replay'] is not True or validation['questions'] != 371):
        raise ValueError('HippoMM lacks complete native-source validation')
    for p in (collection, validation, protocol): verify_hashes(p['source_sha256'])
    answers = unique(judge.read_rows(root / 'responses.jsonl'))
    if set(answers) != set(questions): raise ValueError('HippoMM answer cohort differs')
    checked_calls, provenance = {}, {}
    for qid, row in answers.items():
        if row['answer_model'] != REVISION or row['model'] != REVISION or row['question'] != questions[qid]['question']:
            raise ValueError('HippoMM answer identity differs')
        for name in row['llm_calls']:
            if name in checked_calls: continue
            path = root / name; call = read(path); request = call['request']; raw = call['response']
            wire_path = path.with_name(path.stem + '.wire.json.gz')
            checked_hash(wire_path, call['wire_file_sha256'])
            verify_hashes(call['image_sha256'])
            wire = json.loads(gzip.decompress(wire_path.read_bytes()))
            rebuilt = native.gpt.build_request({'system': native.runtime.READER_SYSTEM, 'user_text': request['prompt'],
                'query_images': [{'path': p, 'sha256': checked_hash(p)} for p in request['image_paths']]},
                seed=42, max_output_tokens=2048, image_detail='high')
            if wire != rebuilt or native.runtime.digest(wire) != call['request_sha256']:
                raise ValueError('HippoMM native wire differs')
            if raw['returned_model'] != REVISION or raw['requested_model'] != MODEL or raw['finish_reason'] != 'stop' or raw.get('refusal'):
                raise ValueError('HippoMM native response model or completion differs')
            successes = [a for a in judge.read_rows(path.parent / 'api_attempts.jsonl')
                         if a['call_id'] == call['call_id'] and a['status'] == 'success']
            if len(successes) != 1 or successes[0]['response_id'] != raw['response_id'] or successes[0]['request_sha256'] != call['request_sha256']:
                raise ValueError('HippoMM call lacks its successful raw attempt')
            checked_calls[name] = call
        final_name = row['llm_calls'][-1]; final = checked_calls[final_name]
        if row['route'] == 'fast':
            expected = parse_fast(final['response']['raw_response'], row['fast_context_event_ids'])['answer'].strip()
        elif row['route'] == 'detailed':
            expected = final['response']['raw_response'].strip()
        else: raise ValueError('Unknown HippoMM answer route')
        if expected != row['raw_response'] or expected != row['answer'] or row['payload_sha256'] != final['request_sha256']:
            raise ValueError('HippoMM final answer differs from original native response')
        provenance[qid] = {'source_answer_api_request_sha256': final['request_sha256'],
            'source_answer_record': str(root / final_name), 'source_answer_record_sha256': checked_hash(root / final_name),
            'answer_model_revision': REVISION}
    return answers, provenance


def unique(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate question IDs')
    return result


def verify_method(method, questions, gold):
    root = Path(method['root']).resolve()
    if method['id'] not in METHOD_DIRS or root != (DEFAULT_ROOT / METHOD_DIRS[method['id']]).resolve():
        raise ValueError('Method root does not identify this GPT-5 Open experiment')
    out = root / 'judge'
    protocol = json.loads((out / 'protocol.json').read_text())
    summary = json.loads((out / 'summary.json').read_text())
    if protocol.get('method_id') != method['id'] or protocol.get('method') != method['label']:
        raise ValueError('Manifest method ID/label does not match the frozen Judge protocol')
    if (protocol.get('answer_model') != MODEL
            or summary.get('answer_model') != MODEL
            or summary.get('method') != protocol.get('method')):
        raise ValueError('Method or GPT-5 answer backbone differs between protocol and summary')
    for key, value in judge.judge_settings().items():
        if protocol.get(key) != value:
            raise ValueError(f"{method['id']}: different Judge setting {key}")
    if (out / 'judge_prompt.txt').read_bytes() != judge.PROMPT.read_bytes():
        raise ValueError('Different Judge prompt')
    if summary.get('status') != 'complete' or summary.get('questions_scored') != 371:
        raise ValueError(f"{method['id']}: all 371 grades must be complete")
    for name, expected in summary['artifact_sha256'].items():
        if judge.sha(out / name) != expected:
            raise ValueError('Changed Judge artifact: ' + name)
    for field in ('source_sha256', 'implementation_sha256', 'cache_source_sha256',
                  'canonical_source_sha256'):
        for path, expected in protocol.get(field, {}).items():
            if judge.sha(Path(path)) != expected:
                raise ValueError('Changed source: ' + path)
    records = unique(judge.read_rows(root / 'judge_inputs.jsonl'))
    reader_dir = root if method['id'] == 'hippomm' else root / 'reader'
    responses_path = reader_dir / 'responses.jsonl'
    original_answers, answer_provenance = (hippo_answers if method['id'] == 'hippomm' else generic_answers)(root, method, questions)
    responses_sha = judge.sha(responses_path)
    seal = json.loads((reader_dir / 'completion.json').read_text())
    if (seal.get('status') != 'complete' or seal.get('responses_sha256') != responses_sha
            or len(original_answers) != 371 or set(original_answers) != set(questions)):
        raise ValueError('Original answers lack a valid complete seal')
    mapped = unique(judge.read_rows(out / 'question_map.jsonl'))
    per_question = unique(judge.read_rows(out / 'per_question.jsonl'))
    if not (len(records) == 371 and set(records) == set(mapped) == set(per_question) == set(questions)):
        raise ValueError('Incomplete canonical cohort')
    requests = judge.read_rows(out / 'requests.jsonl')
    request_map = {r['request_sha256']: r for r in requests}
    if len(request_map) != len(requests):
        raise ValueError('Duplicate request hashes')
    saved, validated_request_map = grade_store(out)
    if validated_request_map != request_map:
        raise ValueError('Validated Judge request map differs')
    if set(saved) != set(request_map):
        raise ValueError('Missing unique request grades')
    template = judge.PROMPT.read_text()
    output, used = [], set()
    for qid, canonical in questions.items():
        row = records[qid]
        if row.get('answer_model') not in (MODEL, REVISION):
            raise ValueError('Frozen answer was not generated by the GPT-5 backbone')
        original = original_answers[qid]
        if (row['model_output'] != original['raw_response']
                or row.get('source_responses_sha256') != responses_sha
                or row.get('source_hit_token_limit', False) != original['hit_token_limit']
                or row.get('source_record_id') != qid):
            raise ValueError('Judge input differs from original reader answer: ' + qid)
        source_key = 'source_payload_sha256' if method['id'] == 'hippomm' else 'source_api_request_sha256'
        if row[source_key] != answer_provenance[qid]['source_answer_api_request_sha256']:
            raise ValueError('Judge input answer wire provenance differs')
        if method['id'] != 'hippomm' and row.get('answer_model_revision') != REVISION:
            raise ValueError('Judge input answer revision differs')
        if any(row[k] != canonical[k] for k in ('question', 'task', 'point')):
            raise ValueError('Question changed: ' + qid)
        if row['ground_truth'] != gold[qid]['answer']:
            raise ValueError('Open gold changed: ' + qid)
        wire = judge.build_request(row, template)
        digest = judge.transport.request_sha256(wire)
        used.add(digest)
        if request_map[digest]['request'] != wire or mapped[qid]['request_sha256'] != digest:
            raise ValueError('Judge wire does not reproduce')
        for key in ('question', 'ground_truth', 'model_output', 'task', 'point'):
            if any(value[qid][key] != row[key] for value in (mapped, per_question)):
                raise ValueError('Frozen answer or metadata changed')
        score = saved[digest]
        parsed = judge.parse_grade(score['raw_response'])
        if per_question[qid]['judge_status'] != 'ok' or any(per_question[qid][k] != v for k, v in parsed.items()):
            raise ValueError('Per-question grade differs from original raw score')
        output.append({'method_id': method['id'], 'method': method['label'], 'id': qid,
            'task': row['task'], 'point': row['point'], 'question': row['question'],
            'ground_truth': row['ground_truth'], 'model_output': row['model_output'],
            'score': parsed['score'], 'reasoning': parsed['reasoning'],
            'source_hit_token_limit': row.get('source_hit_token_limit', False),
            'format_recovery': bool(original.get('query_format_recovery')),
            'request_sha256': digest, 'score_origin': score['score_origin'],
            'judge_raw_response': score['raw_response'], 'judge_response_id': score.get('response_id'),
            'judge_reuse_sources': score.get('reuse_sources', []),
            'source_original_responses_sha256': responses_sha, **answer_provenance[qid]})
    if used != set(request_map):
        raise ValueError('Unused requests')
    total = sum(row['score'] for row in output)
    counts = Counter(row['score'] for row in output)
    distribution = {str(g): counts[g] for g in judge.GRADES}
    if (summary['score_sum'] != total or summary['llm_judge'] != total / 371
            or summary['score_distribution'] != distribution):
        raise ValueError('Judge mean or distribution does not reproduce')
    recovered = [row['id'] for row in output if row['format_recovery']]
    note = method['protocol_note']
    if recovered:
        note += (' Native rewrite-format repairs were recorded using the original bounded GPT-5 '
                 'repair procedure; original question and 2-5 word constraint retained. IDs: '
                 + ', '.join(recovered))
    return {'id': method['id'], 'method': method['label'], 'questions': 371,
        'score_sum': total, 'llm_judge': total / 371, 'score_distribution': distribution,
        'token_limit_answers': sum(bool(r['source_hit_token_limit']) for r in output),
        'empty_answers': sum(not r['model_output'] for r in output),
        'root': str(root), 'protocol_note': note, 'format_recovery_ids': recovered,
        'judge_summary_sha256': judge.sha(out / 'summary.json')}, output


def workbook(path, summaries, detail):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = 'LLM Judge'
    ws.append(['MemEye Open · GPT-5 · 371 questions'])
    ws.merge_cells('A1:B1')
    ws.append(['Method', 'LLM Judge'])
    for row in summaries:
        ws.append([row['method'], row['llm_judge']])
        ws.cell(ws.max_row, 2).number_format = '0.0000'
    ws.append([])
    ws.append(['Judge', judge.transport.DEEPSEEK_MODEL])
    ws.append(['Grades', '0 / 0.25 / 0.5 / 0.75 / 1'])
    ws.append(['Aggregation', 'Sum of scores / 371; every question included.'])
    ws.column_dimensions['A'].width = 49
    ws.column_dimensions['B'].width = 55
    ws.freeze_panes = 'B3'
    ws.auto_filter.ref = f'A2:B{2+len(summaries)}'
    best = max(row['llm_judge'] for row in summaries)
    for i, row in enumerate(summaries, 3):
        if row['llm_judge'] == best:
            ws.cell(i, 2).font = Font(bold=True, color='166534')
            ws.cell(i, 2).fill = PatternFill('solid', fgColor='DCFCE7')
    columns = ['method', 'id', 'point', 'question', 'ground_truth', 'model_output',
               'score', 'reasoning', 'source_hit_token_limit', 'format_recovery', 'request_sha256',
               'judge_raw_response', 'score_origin', 'judge_response_id',
               'source_answer_api_request_sha256', 'source_answer_record', 'answer_model_revision']
    details = wb.create_sheet('逐题评分')
    details.append(columns)
    for row in detail:
        if any(isinstance(row.get(key), str) and len(row[key]) > 32767 for key in columns):
            raise ValueError('Original text exceeds Excel cell limit; refuse silent truncation')
        details.append([row.get(key) for key in columns])
        # Original strings must remain literal even if they start with '='.
        for cell in details[details.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = 's'
        details.cell(details.max_row, 7).number_format = '0.00'
    for i, width in enumerate([43, 48, 12, 65, 55, 75, 12, 90, 20, 20, 68, 95, 22, 55, 68, 100, 30], 1):
        details.column_dimensions[get_column_letter(i)].width = width
    details.freeze_panes = 'D2'
    details.auto_filter.ref = details.dimensions
    meta = wb.create_sheet('实验口径与来源')
    meta.append(['Method', 'Protocol / adaptation', 'Result directory', 'Judge summary SHA256'])
    for row in summaries:
        meta.append([row['method'], row['protocol_note'], row['root'], row['judge_summary_sha256']])
    for col, width in zip('ABCD', [48, 100, 105, 70]):
        meta.column_dimensions[col].width = width
    meta.freeze_panes = 'B2'
    for sheet in wb:
        header = 2 if sheet is ws else 1
        for cell in sheet[header]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='24476A')
        for row in sheet:
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
                if isinstance(cell.value, str):
                    cell.data_type = 's'
        sheet.sheet_view.showGridLines = False
    wb.save(path)
    reopened = load_workbook(path, data_only=True, read_only=True)
    if reopened['逐题评分'].max_row != len(detail) + 1:
        raise ValueError('Export lost detailed rows')
    for i, row in enumerate(summaries, 3):
        if abs(reopened['LLM Judge'].cell(i, 2).value - row['llm_judge']) > 1e-14:
            raise ValueError('Excel Judge value differs')
        if reopened['LLM Judge'].cell(i, 1).value != row['method']:
            raise ValueError('Excel method label differs')
    for cells, original in zip(reopened['逐题评分'].iter_rows(min_row=2, values_only=True), detail):
        expected = [original.get(key) for key in columns]
        if list(cells) != [None if v == '' else v for v in expected]:
            raise ValueError('Excel original answer/grade/provenance text differs')
    reopened.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_ROOT / 'methods_manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/memeye_gpt5_open_llm_judge_20261105')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if (manifest.get('answer_model') != MODEL or manifest.get('questions_per_method') != 371
            or manifest.get('expected_returned_model') != REVISION):
        raise ValueError('Manifest must identify the complete GPT-5 MemEye Open experiment')
    if len(manifest['methods']) != 10 or {m['id'] for m in manifest['methods']} != set(METHOD_DIRS):
        raise ValueError('Expected the ten screenshot methods')
    questions = unique(judge.read_rows(CANONICAL / 'inputs/questions.jsonl'))
    gold = unique(judge.read_rows(CANONICAL / 'inputs/gold.jsonl'))
    canonical_protocol = read(CANONICAL / 'protocol.json')
    for name in ('questions.jsonl', 'gold.jsonl'):
        checked_hash(CANONICAL / 'inputs' / name, canonical_protocol['artifact_sha256']['inputs/' + name])
    if len(questions) != 371 or set(questions) != set(gold):
        raise ValueError('Canonical Open question/gold coverage differs')
    summaries, detail = [], []
    for method in manifest['methods']:
        summary, rows = verify_method(method, questions, gold)
        summaries.append(summary)
        detail.extend(rows)
    if len(detail) != 3710:
        raise ValueError('Expected 3710 method-question scores')
    args.output.mkdir(parents=True, exist_ok=True)
    judge.dump(args.output / 'audit.json', {'status': 'complete', 'methods': summaries,
        'answer_model': MODEL, 'answer_model_revision': REVISION,
        'questions_per_method': 371, 'method_questions': len(detail),
        'manifest_sha256': judge.sha(args.manifest), 'script_sha256': judge.sha(Path(__file__)),
        'judge_settings': judge.judge_settings(), 'reported_metric': 'LLM Judge only',
        'original_answer_API_wire_and_raw_attempts_validated': True,
        'grade_raw_response_and_recursive_cache_provenance_validated': True,
        'complete_original_text_and_numeric_excel_roundtrip_validated': True})
    with (args.output / 'LLM_Judge.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Method', 'LLM Judge'])
        writer.writerows((row['method'], row['llm_judge']) for row in summaries)
    judge.write_rows(args.output / 'per_question.jsonl', detail)
    path = args.output / 'MemEye_GPT5_Open_LLM_Judge_20261105.xlsx'
    workbook(path, summaries, detail)
    exported_detail = judge.read_rows(args.output / 'per_question.jsonl')
    if exported_detail != detail:
        raise ValueError('JSON detail roundtrip differs')
    with (args.output / 'LLM_Judge.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.reader(stream))
    if records[0] != ['Method', 'LLM Judge'] or len(records) != 11:
        raise ValueError('CSV primary comparison shape differs')
    for record, row in zip(records[1:], summaries):
        if record[0] != row['method'] or float(record[1]) != row['llm_judge']:
            raise ValueError('CSV primary comparison differs')
    judge.dump(args.output / 'completion.json', {'status': 'complete', 'methods': 10,
        'questions_per_method': 371, 'artifact_sha256': {p.name: judge.sha(p)
            for p in args.output.iterdir() if p.is_file() and p.name != 'completion.json'}})
    print(json.dumps({'xlsx': str(path), 'scores': [{k: r[k] for k in ('method', 'llm_judge')}
        for r in summaries]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
