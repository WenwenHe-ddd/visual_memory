"""Independently compare the completed MemEye Open workbook with all raw score rows."""
from __future__ import annotations

from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/memeye_qwen_other_open_20261105'
REPORT = ROOT / 'reports/memeye_qwen_open_llm_judge_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(values):
    result = {value['id']: value for value in values}
    assert len(result) == len(values)
    return result


def main():
    manifest = read(RUN / 'methods_manifest.json')
    methods = manifest['methods']
    assert len(methods) == len({m['id'] for m in methods}) == 10
    assert len({m['label'] for m in methods}) == 10
    seal, audit = read(REPORT / 'completion.json'), read(REPORT / 'audit.json')
    assert seal['status'] == audit['status'] == 'complete'
    assert seal['methods'] == 10 and seal['questions_per_method'] == 371
    assert audit['method_questions'] == 3710 and audit['reported_metric'] == 'LLM Judge only'
    assert audit['manifest_sha256'] == sha(RUN / 'methods_manifest.json')
    for name, expected in seal['artifact_sha256'].items():
        assert sha(REPORT / name) == expected
    expected_rows, expected_summaries = [], []
    sources = [RUN / 'methods_manifest.json', REPORT / 'completion.json']
    for method in methods:
        root = Path(method['root'])
        p = root / 'judge/per_question.jsonl'
        scored, inputs = keyed(rows(p)), keyed(rows(root / 'judge_inputs.jsonl'))
        native = root if method['id'] == 'hippomm' else root / 'reader'
        answers = keyed(rows(native / 'responses.jsonl'))
        summary = read(root / 'judge/summary.json')
        assert summary['status'] == 'complete' and len(scored) == 371
        assert set(scored) == set(inputs) == set(answers)
        quarter_units = 0
        for qid, score in scored.items():
            inp, answer = inputs[qid], answers[qid]
            for field in ('question', 'ground_truth', 'model_output', 'point', 'task'):
                assert score[field] == inp[field]
            assert score['model_output'] == answer['raw_response']
            assert score['score'] in (0, .25, .5, .75, 1)
            quarter_units += round(score['score'] * 4)
            expected_rows.append([method['label'], qid, score['point'], score['question'],
                score['ground_truth'], score['model_output'], score['score'], score['reasoning'],
                answer['hit_token_limit'], bool(answer.get('format_recovery_dir')), score['request_sha256']])
        mean = float(Fraction(quarter_units, 4 * 371))
        assert mean == summary['llm_judge']
        expected_summaries.append([method['label'], mean])
        sources.extend([p, root / 'judge_inputs.jsonl', native / 'responses.jsonl', root / 'judge/summary.json'])
        if method['id'] != 'semrag':
            independent_path = RUN / 'audits' / (method['id'] + '_judge_independent_20261105.json')
            independent = read(independent_path)
            assert independent['passed'] and independent['questions_checked'] == 371
            assert independent['llm_judge'] == mean
            sources.append(independent_path)
    hip_audit_path = RUN / 'audits/hippomm_native_independent_20261105.json'
    assert read(hip_audit_path)['passed'] and read(hip_audit_path)['questions_checked'] == 371
    sources.append(hip_audit_path)
    xlsx = REPORT / 'MemEye_Qwen_Open_LLM_Judge_20261105.xlsx'
    with ZipFile(xlsx) as archive:
        assert archive.testzip() is None
    wb = load_workbook(xlsx, data_only=False, read_only=True)
    assert wb.sheetnames == ['LLM Judge', '逐题评分', '实验口径与来源']
    table = wb['LLM Judge']
    assert [table.cell(2, col).value for col in (1, 2)] == ['Method', 'LLM Judge']
    for offset, (label, mean) in enumerate(expected_summaries, 3):
        assert table.cell(offset, 1).value == label
        assert abs(table.cell(offset, 2).value - mean) < 1e-14
    details = wb['逐题评分']
    assert details.max_row == 3711 and details.max_column == 11
    actual_rows = list(details.iter_rows(min_row=2, values_only=True))
    actual = {(row[0], row[1]): list(row) for row in actual_rows}
    expected = {(row[0], row[1]): row for row in expected_rows}
    assert len(actual) == len(expected) == 3710 and actual.keys() == expected.keys()
    for key, row in expected.items():
        # An empty literal is represented by an empty cell in OOXML.
        normalized = [None if value == '' else value for value in row]
        assert actual[key] == normalized, key
    for sheet in wb:
        for row in sheet:
            assert all(cell.data_type != 'f' for cell in row), ('unexpected formula', sheet.title)
    meta = wb['实验口径与来源']
    assert meta.max_row == 11
    for offset, method in enumerate(methods, 2):
        assert meta.cell(offset, 1).value == method['label']
        assert meta.cell(offset, 3).value == method['root']
        assert meta.cell(offset, 4).value == sha(Path(method['root']) / 'judge/summary.json')
        if method['id'] == 'hippomm' and read(hip_audit_path)['format_recovery_questions']:
            assert 'format adaptation' in meta.cell(offset, 2).value
    wb.close()
    with (REPORT / 'LLM_Judge.csv').open(encoding='utf-8-sig', newline='') as stream:
        csv_rows = list(csv.reader(stream))
    assert csv_rows[0] == ['Method', 'LLM Judge'] and len(csv_rows) == 11
    for saved, (label, mean) in zip(csv_rows[1:], expected_summaries):
        assert saved[0] == label and float(saved[1]) == mean
    exported = rows(REPORT / 'per_question.jsonl')
    assert len(exported) == len({(r['method'], r['id']) for r in exported}) == 3710
    for saved in exported:
        check = [saved[key] for key in ('method', 'id', 'point', 'question', 'ground_truth',
                 'model_output', 'score', 'reasoning', 'source_hit_token_limit', 'format_recovery', 'request_sha256')]
        assert check == expected[(saved['method'], saved['id'])]
    assert Counter(row[0] for row in expected_rows) == {method['label']: 371 for method in methods}
    sources.extend([Path(__file__), hip_audit_path, xlsx, REPORT / 'audit.json', REPORT / 'per_question.jsonl', REPORT / 'LLM_Judge.csv'])
    result = {'passed': True, 'method_count': 10, 'questions_per_method': 371,
        'detailed_excel_cells_compared': 3710 * 11, 'question_rows': 3710,
        'original_answers_scores_reasons_and_source_mapping_preserved': True,
        'main_table_reports_only_llm_judge': True, 'unexpected_formulas': 0,
        'xlsx_zip_integrity_passed': True, 'csv_and_json_match_workbook': True,
        'scores': [{'method': label, 'llm_judge': value} for label, value in expected_summaries],
        'source_sha256': {str(path): sha(path) for path in sorted(set(sources))}}
    output = RUN / 'audits/final_workbook_independent_20261105.json'
    assert not output.exists(), 'Do not overwrite a completed signed audit'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
