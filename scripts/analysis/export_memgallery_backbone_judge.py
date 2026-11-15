"""Export one completed backbone's frozen MemGallery LLM-Judge evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis import export_memgallery_five_grade_judge as common


def export(root, output, expected_answer_model='gpt-5-2025-08-07'):
    root = root.resolve()
    # This baseline supplies canonical question/gold text and judge settings.
    # Its Qwen method is not appended to the requested backbone's workbook.
    canonical = common.load_baseline()
    methods = common.load_multi(root, canonical)
    protocol = common.read_json(root / 'protocol.json')
    model = protocol['answer_model']
    common.require(model == expected_answer_model,
                   f'Answer model differs: expected {expected_answer_model}, received {model}')
    model_label = {'gpt-5-2025-08-07': 'GPT-5',
                   'deepseek-v4.1-flash': 'DeepSeek Flash'}.get(model, model)
    method_protocols = common.unique(protocol['methods'])
    order = ['semantic_rag', 'ours1_ours2', 'gen_agents', 'memoryos',
             'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm']
    common.require({m['summary']['method_id'] for m in methods} == set(order),
                   'Expected the eight methods in the requested backbone screenshot')
    methods.sort(key=lambda item: order.index(item['summary']['method_id']))
    book = Workbook()
    summary_sheet = book.active
    summary_sheet.title = '方法汇总'
    detail_sheet = book.create_sheet('逐题评分')
    source_sheet = book.create_sheet('协议与来源')
    append = common.append_text_safe
    append(summary_sheet, ['方法', '回答模型', '原EM正确数', '题数', 'EM', 'F1', 'BLEU-1',
        'LLM Judge (0–1)', '0分', '0.25分', '0.5分', '0.75分', '1分',
        'code_method', '原结果目录', '上下文', '完整性'])
    append(detail_sheet, ['method_id', '方法', 'ID', '场景', '题型', '原EM', '原问题',
        '标准答案', f'{model_label}原回答', 'Judge分数', '评分理由', '评分来源',
        '原回答达到token上限', 'request_sha256', 'response_id', 'source_record_id'])
    append(source_sheet, ['范围', '项目', '值'])
    notes = {
        '回答模型': model, 'Judge模型': protocol['judge_model'],
        '统计定义': '每种方法1711题五档分数的算术平均；不是EM或二元正确率。',
        '五档': '0 / 0.25 / 0.5 / 0.75 / 1；保留0.25和0.75。',
        'Judge输入': f'原始问题、标准答案、冻结的{model_label}原回答；无图片或检索上下文。',
        '冻结策略': '不重新检索、不重新回答，原有输出截断记录也照原文评分。',
        '缓存策略': f'仅完整Judge请求JSON的SHA256完全一致时，复用同协议已有真实评分。缓存不替换{model_label}原回答。',
        '评分来源字段': 'baseline_cache表示兼容的历史评分缓存，具体来源见reuse_sources；new_api表示本次新调用。',
        'VisRAG标签': '沿用截图标签，实际对应CLIP检索→原图＋历史对话。',
        '完整性': '只导出全部1711题完整评分并通过原始JSON、源文件哈希、均值与分布校验的方法。',
    }
    for key, value in notes.items():
        append(source_sheet, ['全局', key, value])
    for key in ('temperature', 'seed', 'enable_thinking', 'max_tokens', 'judge_base_url',
                'official_prompt_url', 'official_prompt_sha256', 'aggregation',
                'deduplication', 'cache_policy'):
        if key in protocol:
            append(source_sheet, ['全局', key, protocol[key]])
    for item in methods:
        summary, rows = item['summary'], item['rows']
        mid, label = summary['method_id'], summary['label']
        common.require(summary.get('answer_model') == model,
                       f'Answer model provenance differs for {mid}')
        append(summary_sheet, [label, model, summary['source_em_correct'], 1711,
            summary['source_em'], summary['f1'], summary['bleu_1'], summary['llm_judge'],
            *[summary['score_distribution'][str(g)] for g in common.GRADES],
            summary.get('code_method', mid), summary.get('source_root', ''),
            summary.get('context', ''), '完整：1711/1711'])
        for row in rows:
            append(detail_sheet, [mid, label, row['id'], row['task'], row['point'], row['em'],
                row['question'], row['ground_truth'], row['model_output'], row['score'],
                row['reasoning'], row.get('score_origin'), row.get('source_hit_token_limit'),
                row['request_sha256'], row.get('response_id'), row['source_record_id']])
        for key, value in [('source_root', summary.get('source_root')),
                ('code_method', summary.get('code_method')),
                ('context', summary.get('context')),
                ('score_origin_counts', dict(Counter(r['score_origin'] for r in rows))),
                ('source_hit_token_limit_true', sum(r.get('source_hit_token_limit') is True for r in rows)),
                ('source_hit_token_limit_unknown', sum('source_hit_token_limit' not in r for r in rows))]:
            append(source_sheet, [mid, key, value])
        for filename, digest in summary['source_sha256'].items():
            append(source_sheet, [mid, 'source_sha256: ' + filename, digest])
        # Preserve the complete frozen method metadata, including caveats for
        # merged source runs, rather than dropping fields outside a fixed list.
        for key, value in method_protocols[mid].items():
            if key not in {'source_root', 'code_method', 'context', 'source_sha256'}:
                append(source_sheet, [mid, 'method_manifest: ' + key, value])
    append(source_sheet, ['全局', 'judge_root', str(root)])
    append(source_sheet, ['全局', 'protocol_sha256', common.sha(root / 'protocol.json')])
    append(source_sheet, ['全局', 'summary_sha256', common.sha(root / 'summary.json')])
    common.style_sheet(summary_sheet, [43, 25, 15, 12, 14, 14, 14, 20, 12, 12, 12, 12, 12, 42, 70, 70, 23])
    common.style_sheet(detail_sheet, [25, 43, 58, 44, 10, 10, 80, 60, 100, 14, 100, 24, 24, 68, 40, 65])
    common.style_sheet(source_sheet, [26, 76, 110])
    for row in summary_sheet.iter_rows(min_row=2):
        for cell in row[4:8]:
            cell.number_format = '0.000000'
        summary_sheet.row_dimensions[row[0].row].height = 48
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp.xlsx')
    book.save(temporary)
    with ZipFile(temporary) as archive:
        common.require(archive.testzip() is None, 'Invalid XLSX archive')
    check = load_workbook(temporary, read_only=True, data_only=False)
    common.require(check['方法汇总'].max_row == 9 and check['逐题评分'].max_row == 13689,
                   'Workbook coverage differs')
    for row, item in zip(check['方法汇总'].iter_rows(min_row=2, values_only=True), methods):
        summary = item['summary']
        expected = [summary['source_em_correct'], 1711, summary['source_em'],
                    summary['f1'], summary['bleu_1'], summary['llm_judge']]
        common.require(all(abs(a - b) < 1e-12 for a, b in zip(row[2:8], expected)),
                       'Workbook metrics differ from their source')
    common.require(check['逐题评分'].cell(1, 9).value == f'{model_label}原回答',
                   'Workbook original-answer model label differs')
    exported_metadata = {(scope, key): value for scope, key, value in
                         check['协议与来源'].iter_rows(min_row=2, values_only=True)}
    for mid, method in method_protocols.items():
        for key, value in method.items():
            if key in {'source_root', 'code_method', 'context', 'source_sha256'}:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if value == '':
                value = None  # XLSX represents empty string cells as empty.
            actual = exported_metadata[(mid, 'method_manifest: ' + key)]
            matches = (abs(actual - value) < 1e-12
                       if isinstance(value, float) and isinstance(actual, (int, float))
                       else actual == value)
            common.require(matches, f'Workbook method provenance differs: {mid}/{key}')
    for sheet in check:
        common.require(all(c.data_type != 'f' for row in sheet.iter_rows() for c in row),
                       'Unexpected formula in original answer text')
    check.close()
    temporary.replace(output)
    audit = {'passed': True, 'methods': 8, 'questions_per_method': 1711,
             'question_rows': 13688, 'answer_model': model,
             'xlsx_sha256': common.sha(output), 'output': str(output)}
    output.with_suffix('.validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge-root', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        help='Workbook path; required for answer models other than the default GPT-5 snapshot')
    parser.add_argument('--expected-answer-model', default='gpt-5-2025-08-07',
                        help='Required frozen answer-model identity; defaults to the GPT-5 snapshot')
    args = parser.parse_args()
    if args.output is None:
        if args.expected_answer_model != 'gpt-5-2025-08-07':
            parser.error('--output is required for a nondefault answer model')
        args.output = (ROOT / 'reports/memgallery_gpt5_llm_judge_20261104'
                       / 'MemGallery_GPT5_五档LLM_Judge_20261104.xlsx')
    export(args.judge_root, args.output, args.expected_answer_model)


if __name__ == '__main__':
    main()
