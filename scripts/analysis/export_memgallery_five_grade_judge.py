"""Export completed MemGallery five-grade judge runs without changing any result.

The completed Semantic RAG baseline is always included. Add one or more frozen
multi-method evaluations with repeated --judge-root arguments. Partial roots,
missing questions, altered hashes, and conflicting scores are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104'
SOURCE_METRICS = ROOT / 'results/mem_gallery_master_answer_20261021/reader/summary.json'
DEFAULT_OUTPUT = ROOT / 'reports/memgallery_qwen_llm_judge_20261104/MemGallery_Qwen_五档LLM_Judge_20261104.xlsx'
GRADES = (0, 0.25, 0.5, 0.75, 1)
EXPECTED = 1711


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, field='id'):
    result = {row[field]: row for row in rows}
    require(len(result) == len(rows), f'Duplicate {field} in saved records')
    return result


def verify_hashes(hashes, base=None):
    for filename, expected in hashes.items():
        path = Path(filename)
        if not path.is_absolute():
            require(base is not None, f'Relative hash path without base: {path}')
            path = base / path
        require(sha(path) == expected, f'File hash changed: {path}')


def verify_grade(row):
    score = row.get('score')
    require(type(score) in (int, float) and score in GRADES, 'Invalid five-grade score')
    require(isinstance(row.get('reasoning'), str) and row['reasoning'].strip(), 'Missing judge reasoning')
    raw = row['raw_response'].strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
    parsed = json.loads(fence.group(1) if fence else raw)
    require(parsed.get('score') == score and parsed.get('reasoning') == row['reasoning'],
            'Saved grade/reasoning differs from the raw judge output')
    require(row.get('finish_reason') == 'stop' and not row.get('refusal'),
            'Judge response did not finish successfully')
    require(row.get('returned_model') == row.get('requested_model') == 'deepseek-v4.1-flash',
            'Unexpected judge model')


def verify_method(summary, rows):
    require(summary.get('status') == 'complete', 'Refusing to export an incomplete method')
    require(summary['questions_expected'] == summary['questions_scored'] == EXPECTED
            and summary['questions_missing'] == 0, 'Incomplete method coverage')
    require(len(rows) == len(unique(rows)) == EXPECTED, 'Expected 1711 unique method questions')
    for row in rows:
        verify_grade(row)
    correct = sum(row['em'] for row in rows)
    require(correct == summary['source_em_correct'], 'Source EM count differs')
    require(abs(correct / EXPECTED - summary['source_em']) < 1e-12, 'Source EM differs')
    total = sum(row['score'] for row in rows)
    require(total == summary['score_sum'], 'Score sum differs')
    require(abs(total / EXPECTED - summary['llm_judge']) < 1e-12, 'Judge mean differs')
    counts = Counter(row['score'] for row in rows)
    require({str(grade): counts[grade] for grade in GRADES} == summary['score_distribution'],
            'Grade distribution differs')


def load_baseline():
    summary = read_json(BASELINE / 'summary.json')
    protocol = read_json(BASELINE / 'protocol.json')
    verify_hashes({name: summary[name.removesuffix('.jsonl') + '_sha256']
                   for name in ('requests.jsonl', 'scores.jsonl')}, BASELINE)
    verify_hashes(protocol['source_sha256'])
    requests = read_rows(BASELINE / 'requests.jsonl')
    scores = unique(read_rows(BASELINE / 'scores.jsonl'))
    require(set(scores) == set(unique(requests)), 'Baseline question/score ID mismatch')
    rows = []
    for source in requests:
        score = scores[source['id']]
        require(source['request_sha256'] == score['request_sha256'], 'Baseline request hash mismatch')
        rows.append({**source, **score, 'method_id': 'baseline_question_only',
                     'score_origin': 'original_baseline_evaluation', 'judge_status': 'ok'})
    verify_method(summary, rows)
    metrics = read_json(SOURCE_METRICS)['overall']['baseline_question_only']
    require(abs(metrics['em'] - summary['source_em']) < 1e-12, 'Baseline source metrics differ')
    summary = {**summary, 'method_id': 'baseline_question_only',
               'label': 'Semantic RAG（caption 基线）', 'code_method': 'baseline_question_only',
               'f1': metrics['f1'], 'bleu_1': metrics['bleu_1'],
               'source_root': str(SOURCE_METRICS.parent.parent),
               'context': 'Qwen 原始 Semantic RAG Top5 回答；固定 778/1711。',
               'source_condition': 'baseline_question_only'}
    return {'summary': summary, 'rows': rows, 'root': BASELINE, 'protocol': protocol}


def load_multi(root, baseline):
    root = root.resolve()
    summary = read_json(root / 'summary.json')
    require(summary.get('status') == 'complete', f'Refusing incomplete judge root: {root}')
    require(summary['unique_requests_missing'] == 0, 'Missing unique judge requests')
    verify_hashes(summary['artifact_sha256'], root)
    protocol = read_json(root / 'protocol.json')
    for field in ('judge_model', 'judge_base_url', 'temperature', 'seed', 'enable_thinking',
                  'max_tokens', 'grades', 'official_prompt_sha256', 'judge_input_fields',
                  'images_context_captions_sent_to_judge'):
        require(protocol.get(field) == baseline['protocol'].get(field), f'Judge protocol differs: {field}')
    verify_hashes(protocol['source_sha256'])
    require((root / 'judge_prompt.txt').read_bytes() == (BASELINE / 'judge_prompt.txt').read_bytes(),
            'Judge prompt differs from the baseline')
    score_rows = []
    for filename in ('scores.jsonl', 'reused_scores.jsonl'):
        if (root / filename).exists():
            score_rows.extend(read_rows(root / filename))
    saved = unique(score_rows, 'request_sha256')
    require(len(saved) == summary['unique_requests_scored'] == summary['unique_requests'],
            'Unique request coverage differs')
    question_map = {(row['method_id'], row['id']): row for row in read_rows(root / 'question_map.jsonl')}
    expected_canonical = {row['id']: row for row in baseline['rows']}
    loaded = []
    for method in summary['methods']:
        mid = method['method_id']
        rows = read_rows(root / 'methods' / mid / 'per_question.jsonl')
        verify_method(method, rows)
        require(set(unique(rows)) == set(expected_canonical), f'Canonical IDs differ: {mid}')
        for row in rows:
            require(row.get('judge_status') == 'ok' and row['method_id'] == mid,
                    'Per-question status or method ID mismatch')
            original = question_map[(mid, row['id'])]
            require(all(row[key] == value for key, value in original.items()), 'Frozen input changed')
            score = saved[row['request_sha256']]
            require(all(row[key] == value for key, value in score.items()), 'Frozen score changed')
            canonical = expected_canonical[row['id']]
            require(all(row[key] == canonical[key] for key in ('task', 'point', 'question', 'ground_truth')),
                    'Method question or gold differs from baseline')
        loaded.append({'summary': method, 'rows': rows, 'root': root, 'protocol': protocol})
    require(len(loaded) == summary['method_count'], 'Root method count differs')
    return loaded


def append_text_safe(sheet, values):
    """Keep source text literally, including leading =, +, - and @ characters."""
    # max_row scans all existing cells; the append cursor avoids quadratic work
    # when exporting tens of thousands of long-text records.
    row_number = sheet._current_row + 1
    for column, value in enumerate(values, 1):
        cell = sheet.cell(row_number, column)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, str):
            require(len(value) <= 32767, 'Source text exceeds Excel cell length; refusing silent truncation')
            cell.value = value
            cell.data_type = 's'
            cell.number_format = '@'
        else:
            cell.value = value


def style_sheet(sheet, widths):
    sheet.freeze_panes = 'A2'
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 32
    for cell in sheet[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='244062')
        cell.alignment = Alignment(vertical='center', wrap_text=True)
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical='top', wrap_text=True)


def export(judge_roots, output):
    baseline = load_baseline()
    methods = [baseline]
    seen_roots = set()
    for root in judge_roots:
        require(root.resolve() not in seen_roots, 'Duplicate --judge-root')
        seen_roots.add(root.resolve())
        methods.extend(load_multi(root, baseline))
    mids = [item['summary']['method_id'] for item in methods]
    require(len(mids) == len(set(mids)), 'Duplicate method ID across judge roots')
    priority = {'baseline_question_only': 0, 'gen_agents': 10, 'memoryos': 11,
                'simplemem': 12, 'a_mem': 13, 'clip_image_dialogue': 14, 'hippomm': 15}
    methods.sort(key=lambda item: priority.get(item['summary']['method_id'], 5))
    book = Workbook()
    overall = book.active
    overall.title = '方法汇总'
    details = book.create_sheet('逐题评分')
    provenance = book.create_sheet('协议与来源')
    append_text_safe(overall, ['方法', '原EM正确数', '题数', 'EM', 'F1', 'BLEU-1', 'LLM Judge (0–1)',
                              '0分题数', '0.25分题数', '0.5分题数', '0.75分题数', '1分题数',
                              'code_method', 'source', '完整性', '原始上下文说明'])
    append_text_safe(details, ['method_id', '方法', 'ID', '场景', '题型', '原EM', '原问题', '标准答案',
                              '原回答', 'Judge分数', '评分理由', '评分来源', '原回答达到token上限',
                              'request_sha256', '原source_record_id', 'Judge结果目录'])
    append_text_safe(provenance, ['范围', '项目', '值'])
    global_notes = {
        '数据集与回答模型': 'MemGallery；Qwen3-VL-8B-Instruct；每方法固定1711题。',
        '统计定义': 'LLM Judge = 所有1711题五档得分的算术平均；不等于EM；不是0/1答对率。',
        '五档评分': '0 / 0.25 / 0.5 / 0.75 / 1；不进行官方当前代码的三档映射。',
        'Judge输入': '仅原始问题、标准答案、原模型回答；不额外发送图片、检索上下文或caption。',
        '复用策略': '完全相同的序列化Judge请求按SHA256复用同一真实评分；与旧SemRAG请求相同者复用其评分。',
        '冻结策略': '不重跑检索或原回答；原有回答达到token上限的记录仍原样评分，不删除或补答。',
        'VisRAG命名说明': '截图VisRAG对应clip_image_dialogue，即CLIP检索→原图＋历史对话；不是已核实的论文VisRAG复现。',
        'ours命名更正': '截图0.4553对应ours_2（baseline_question_only_zscore），不是ours_1＋ours_2；组合方法必须使用对应真实源回答。',
        '完整性策略': '仅导出status=complete且1711题全部有有效真实评分的方法；缺失/暂态分数拒绝导出。',
        '指标来源': '原EM/F1/BLEU-1来自冻结回答结果，LLM Judge来自对应逐题真实评分。',
        '文本安全': '问题、答案、理由及路径均保存为Excel文本，不作为公式执行。',
    }
    for key, value in global_notes.items():
        append_text_safe(provenance, ['全局', key, value])
    for item in methods:
        summary, rows, root, protocol = item['summary'], item['rows'], item['root'], item['protocol']
        mid, label = summary['method_id'], summary['label']
        code_method = summary.get('code_method', summary.get('source_condition', mid))
        append_text_safe(overall, [label, summary['source_em_correct'], EXPECTED,
            summary['source_em'], summary['f1'], summary['bleu_1'], summary['llm_judge'],
            *[summary['score_distribution'][str(grade)] for grade in GRADES],
            code_method, summary.get('source_root', ''), '完整：1711/1711', summary.get('context', '')])
        for row in rows:
            append_text_safe(details, [mid, label, row['id'], row['task'], row['point'], row['em'],
                row['question'], row['ground_truth'], row['model_output'], row['score'], row['reasoning'],
                row.get('score_origin', ''), row.get('source_hit_token_limit'), row['request_sha256'],
                row['source_record_id'], str(root)])
        for key in ('judge_model', 'judge_base_url', 'temperature', 'seed', 'enable_thinking',
                    'max_tokens', 'official_prompt_url', 'official_prompt_sha256', 'aggregation',
                    'deduplication', 'original_answer_policy'):
            if key in protocol:
                append_text_safe(provenance, [mid, key, protocol[key]])
        for key, value in [('judge_root', str(root)), ('summary_sha256', sha(root / 'summary.json')),
                           ('protocol_sha256', sha(root / 'protocol.json')),
                           ('source_condition', summary.get('source_condition', code_method)),
                           ('context', summary.get('context', '')),
                           ('source_hit_token_limit_marked_true', sum(row.get('source_hit_token_limit') is True for row in rows)),
                           ('source_hit_token_limit_field_missing', sum('source_hit_token_limit' not in row for row in rows)),
                           ('score_origin_question_counts', dict(Counter(row.get('score_origin', '') for row in rows)))]:
            append_text_safe(provenance, [mid, key, value])
        for path, digest in summary.get('source_sha256', protocol.get('source_sha256', {})).items():
            append_text_safe(provenance, [mid, 'source_sha256: ' + path, digest])
    style_sheet(overall, [45, 15, 12, 14, 14, 14, 20, 12, 13, 13, 13, 12, 38, 66, 23, 80])
    style_sheet(details, [26, 40, 58, 45, 10, 10, 80, 60, 100, 14, 100, 29, 24, 68, 70, 80])
    style_sheet(provenance, [28, 70, 110])
    for row in overall.iter_rows(min_row=2):
        for column in range(4, 8):
            row[column - 1].number_format = '0.000000'
        overall.row_dimensions[row[0].row].height = 48
    for row in details.iter_rows(min_row=2):
        row[9].number_format = '0.00'
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp.xlsx')
    book.save(temporary)
    with ZipFile(temporary) as archive:
        require(archive.testzip() is None, 'Invalid XLSX archive')
    check = load_workbook(temporary, read_only=True, data_only=False)
    require(check['方法汇总'].max_row == len(methods) + 1, 'Exported method count differs')
    require(check['逐题评分'].max_row == len(methods) * EXPECTED + 1, 'Exported question count differs')
    for sheet in check:
        require(all(cell.data_type != 'f' for row in sheet.iter_rows() for cell in row),
                'Unexpected formula in exported workbook')
    check.close()
    temporary.replace(output)
    print(json.dumps({'status': 'complete', 'methods': len(methods),
                      'questions_per_method': EXPECTED, 'output': str(output),
                      'sha256': sha(output)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge-root', action='append', type=Path, default=[],
                        help='Completed multi-method judge root; may be repeated')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.judge_root, args.output)


if __name__ == '__main__':
    main()
