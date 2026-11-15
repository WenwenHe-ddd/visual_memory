"""Independent raw-grade, source-answer and Excel audit; never invokes a model."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/memeye_deepseek_open_20261106'
REPORT = ROOT / 'reports/memeye_deepseek_open_llm_judge_20261106'
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def by_id(values):
    found = {r['id']: r for r in values}
    assert len(found) == len(values)
    return found


def main():
    from openpyxl import load_workbook
    manifest = read(RUN / 'methods_manifest.json')
    assert manifest['answer_model'] == 'deepseek-v4.1-flash' and manifest['expected_returned_model'] == 'deepseek-v4.1-flash'
    complete = read(REPORT / 'completion.json')
    assert complete['status'] == 'complete' and complete['methods'] == 10 and complete['questions_per_method'] == 371
    for name, digest in complete['artifact_sha256'].items(): assert sha(REPORT / name) == digest
    canonical, gold = by_id(rows(CANONICAL / 'questions.jsonl')), by_id(rows(CANONICAL / 'gold.jsonl'))
    assert len(canonical) == len(gold) == 371 and canonical.keys() == gold.keys()
    detail = rows(REPORT / 'per_question.jsonl'); assert len(detail) == 3710
    report_audit = read(REPORT / 'audit.json')
    summaries = {r['id']: r for r in report_audit['methods']}
    calculated = []
    for m in manifest['methods']:
        root = Path(m['root'])
        if m['id'] == 'hippomm':
            name = read(root / 'collection_protocol.json').get('source_responses_file', 'responses.jsonl')
            assert name in ('responses.jsonl', 'evaluation_responses.jsonl')
            original_path = root / name
        else:
            original_path = root / ('reader/evaluation_responses.jsonl' if (root / 'reader/evaluation_completion.json').exists() else 'reader/responses.jsonl')
        original = by_id(rows(original_path))
        entries = by_id([r for r in detail if r['method_id'] == m['id']])
        per = by_id(rows(root / 'judge/per_question.jsonl'))
        scores = {}
        for name in ('scores.jsonl', 'reused_scores.jsonl'):
            p = root / 'judge' / name
            for r in rows(p) if p.exists() else []:
                assert r['request_sha256'] not in scores
                scores[r['request_sha256']] = r
        assert original.keys() == entries.keys() == per.keys() == canonical.keys()
        quarter_sum = 0; counts = Counter()
        for qid, q in canonical.items():
            d, g = entries[qid], per[qid]; s = scores[d['request_sha256']]
            assert d['method'] == m['label'] and d['question'] == q['question'] and d['ground_truth'] == gold[qid]['answer']
            assert d['model_output'] == original[qid]['raw_response'] == g['model_output']
            assert d['answer_model_revision'] is None
            if d['source_status'] == 'permanent_http_error':
                assert original[qid]['status'] == 'permanent_http_error'
                assert not d['model_output'] and d['answer_returned_model'] is None
                assert original[qid].get('returned_model') is None
                assert not original[qid].get('response_id') and not original[qid].get('usage')
                error_path = Path(d['source_answer_record'])
                assert sha(error_path) == d['source_answer_record_sha256']
                attempts = [a for a in rows(error_path) if a['request_sha256'] == d['source_answer_api_request_sha256']]
                assert attempts and not any(a['status'] in ('ok', 'success') for a in attempts)
                assert attempts[-1]['status'] == 'error' and attempts[-1]['http_status'] == 400 and attempts[-1]['retryable'] is False
            else:
                assert d['source_status'] == 'answered' and d['answer_returned_model'] == 'deepseek-v4.1-flash'
            assert d['judge_raw_response'] == s['raw_response']
            raw = s['raw_response'].strip()
            match = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
            if match: raw = match.group(1)
            parsed = json.loads(raw)
            grade = parsed['score']
            assert type(grade) in (int, float) and grade in (0, .25, .5, .75, 1)
            assert grade == d['score'] == g['score'] and parsed['reasoning'] == d['reasoning'] == g['reasoning']
            quarter_sum += int(grade * 4); counts[grade] += 1
        mean = quarter_sum / (4 * 371)
        s = summaries[m['id']]
        assert s['score_sum'] == quarter_sum / 4 and s['llm_judge'] == mean
        assert s['score_distribution'] == {str(v): counts[v] for v in (0, .25, .5, .75, 1)}
        calculated.append({'method_id': m['id'], 'method': m['label'], 'score_sum': quarter_sum / 4, 'llm_judge': mean})
    xlsx = REPORT / 'MemEye_DeepSeek_Open_LLM_Judge_20261106.xlsx'
    book = load_workbook(xlsx, data_only=False, read_only=True)
    assert book['LLM Judge']['A1'].value == 'MemEye Open · DeepSeek-v4.1-flash · 371 questions'
    metadata = {r[0]: r[1] for r in book['实验口径与来源'].iter_rows(values_only=True) if r[0]}
    assert metadata['Requested answer API alias'] == 'deepseek-v4.1-flash'
    assert metadata['Returned answer API identifier'] == 'deepseek-v4.1-flash'
    assert metadata['Immutable answer model revision'] == 'Unknown (provider alias)'
    assert [book['LLM Judge'].cell(2, c).value for c in (1, 2)] == ['Method', 'LLM Judge']
    for i, r in enumerate(calculated, 3):
        assert book['LLM Judge'].cell(i, 1).value == r['method']
        assert abs(book['LLM Judge'].cell(i, 2).value - r['llm_judge']) < 1e-14
    sheet = book['逐题评分']; assert sheet.max_row == 3711
    iterator = sheet.iter_rows(values_only=True); names = next(iterator)
    for values, expected in zip(iterator, detail):
        actual = dict(zip(names, values))
        assert actual == {key: None if expected[key] == '' else expected[key] for key in names}
    for sheet in book:
        for values in sheet:
            assert not any(c.data_type == 'f' for c in values), 'Unexpected executable Excel formula'
    book.close()
    with (REPORT / 'LLM_Judge.csv').open(encoding='utf-8-sig', newline='') as f: csv_rows = list(csv.reader(f))
    assert len(csv_rows) == 11 and csv_rows[0] == ['Method', 'LLM Judge']
    for r, expected in zip(csv_rows[1:], calculated):
        assert r[0] == expected['method'] and float(r[1]) == expected['llm_judge']
    proof = {'passed': True, 'methods': 10, 'original_answers_and_raw_grades_rechecked': 3710,
        'evaluation_records_and_raw_grades_rechecked': 3710,
        'successful_answer_records': sum(r['source_status'] == 'answered' for r in detail),
        'provider_rejection_records': sum(r['source_status'] == 'permanent_http_error' for r in detail),
        'provider_rejected_ids': [{'method_id': r['method_id'], 'id': r['id']} for r in detail if r['source_status'] == 'permanent_http_error'],
        'all_excel_detail_fields_rechecked': True, 'numeric_sums_recomputed_as_integer_quarters': True,
        'formula_cells': 0, 'reported_metric': 'LLM Judge only', 'scores': calculated,
        'report_completion_sha256': sha(REPORT / 'completion.json'), 'xlsx_sha256': sha(xlsx),
        'script_sha256': sha(__file__)}
    out = RUN / 'export_audits/final_report.json'; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(proof, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
