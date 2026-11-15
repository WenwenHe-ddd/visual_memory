#!/usr/bin/env python3
"""Status, strict export and independent audit of two-seed Mem-Gallery results.

No incomplete value is imputed. Two seeds provide an initial repeatability
check; greedy decoding means a changed RNG seed need not change model output.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis.prepare_memgallery_seed1042_reference import METHOD_IDS, LABELS

RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
OUTPUT = ROOT / 'reports/memgallery_qwen_seed1042_stability_20261106'
WORKBOOK_NAME = 'MemGallery_Qwen_seed42_vs1042_稳定性分析.xlsx'
METRICS = ('em', 'f1', 'bleu_1', 'llm_judge')
GRADES = (0, .25, .5, .75, 1)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def unique(rows):
    result = {row['id']: row for row in rows}
    if len(rows) != len(result):
        raise ValueError('Duplicate question IDs')
    return result


def judge_paths(run, mid, old=False):
    root = run / ('judge_seed42_ours_1' if old else 'judge/' + mid)
    return root, root / 'methods' / mid / 'summary.json', root / 'methods' / mid / 'per_question.jsonl'


def status(run=RUN):
    run = Path(run).resolve()
    reference = read(run / 'reference_seed42.json') if (run / 'reference_seed42.json').exists() else None
    seal_path = run / 'generation_completion.json'
    generation_sealed = seal_path.exists() and read(seal_path).get('status') == 'generation_complete'
    methods = []
    old_ours_judge = judge_paths(run, 'ours_1', old=True)[1]
    old_ours_complete = old_ours_judge.exists() and read(old_ours_judge).get('status') == 'complete'
    for mid, label in zip(METHOD_IDS, LABELS):
        mp = run / 'evaluations' / mid / 'metrics.json'
        rp = run / 'evaluations' / mid / 'records.jsonl'
        _, jp, pp = judge_paths(run, mid)
        m = read(mp) if mp.exists() else {}
        j = read(jp) if jp.exists() else {}
        metric_complete = m.get('status') == 'complete' and m.get('questions') == 1711 and rp.exists()
        judge_complete = (j.get('status') == 'complete' and j.get('questions_scored') == 1711
                          and j.get('questions_missing') == 0 and pp.exists())
        methods.append({'id': mid, 'label': label, 'metrics_complete': metric_complete,
                        'judge_complete': judge_complete,
                        **{k: m.get(k) if metric_complete else None for k in ('correct', 'em', 'f1', 'bleu_1')},
                        'llm_judge': j.get('llm_judge') if judge_complete else None})
    ready = bool(reference and generation_sealed and old_ours_complete and all(m['metrics_complete'] and m['judge_complete'] for m in methods))
    return {'status': 'ready' if ready else 'partial', 'ready_for_export': ready,
            'reference_ready': bool(reference), 'seed42_ours_1_judge_complete': old_ours_complete,
            'generation_complete_seal': bool(generation_sealed),
            'methods_expected': 9, 'questions_per_method': 1711,
            'methods_with_metrics': sum(m['metrics_complete'] for m in methods),
            'methods_with_judge': sum(m['judge_complete'] for m in methods), 'methods': methods}


def collect(run=RUN):
    from scripts.analysis import export_memgallery_five_grade_judge as common
    from scripts.evaluation import memgallery_multi_method_judge as wire
    from scripts.evaluation.mem_gallery_master_reader import score_answer, normalize_answer_universal
    run = Path(run).resolve()
    current_status = status(run)
    if not current_status['ready_for_export']:
        raise ValueError('All nine full1711-answer metrics and judge runs, including seed42 ours_1 judge, are required: '
                         + json.dumps(current_status, ensure_ascii=False))
    reference = read(run / 'reference_seed42.json')
    generation_seal = read(run / 'generation_completion.json')
    assert generation_seal['status'] == 'generation_complete'
    marker = read(run / 'reference_ready.json')
    assert reference['status'] == marker['status'] == 'complete'
    assert [r['id'] for r in reference['methods']] == list(METHOD_IDS)
    source_hashes = {}

    def pin(path, expected=None):
        path = str(Path(path).resolve())
        actual = source_hashes.get(path)
        if actual is None:
            actual = sha(path)
            source_hashes[path] = actual
        assert expected is None or expected == actual, f'Source hash changed: {path}'
        return actual

    for path, expected in marker['output_sha256'].items():
        pin(path, expected)
    canonical = unique(lines(run / 'inputs/questions.jsonl'))
    gold = unique(lines(run / 'inputs/gold.jsonl'))
    assert len(canonical) == len(gold) == 1711 and set(canonical) == set(gold)
    baseline = common.load_baseline()
    protocol_keys = ('judge_model', 'judge_base_url', 'temperature', 'seed', 'enable_thinking',
                     'max_tokens', 'grades', 'official_prompt_sha256', 'judge_input_fields',
                     'images_context_captions_sent_to_judge')
    old_ours_root = judge_paths(run, 'ours_1', old=True)[0]
    old_ours = common.load_multi(old_ours_root, baseline)
    assert len(old_ours) == 1 and old_ours[0]['summary']['method_id'] == 'ours_1'
    old_ours = old_ours[0]
    methods, details, statistics_rows = [], [], []
    old_ours_judge_rows = unique(old_ours['rows'])
    for ref in reference['methods']:
        mid = ref['id']
        old = unique(lines(ref['records_path']))
        metric_path, records_path = [run / 'evaluations' / mid / name for name in ('metrics.json', 'records.jsonl')]
        metric = read(metric_path)
        assert metric['seed'] == 1042 and metric['questions'] == 1711 and metric['status'] == 'complete'
        for path, expected in metric['source_sha256'].items():
            pin(path, expected)
        pin(records_path, metric.get('records_sha256'))
        pin(metric_path)
        new = unique(lines(records_path))
        assert set(old) == set(new) == set(canonical)
        for path, expected in ref['source_sha256'].items():
            pin(path, expected)
        judge_root, judge_summary, judge_records = judge_paths(run, mid)
        judged = common.load_multi(judge_root, baseline)
        assert len(judged) == 1 and judged[0]['summary']['method_id'] == mid
        judged = judged[0]
        jp = judged['protocol']
        assert all(jp.get(k) == baseline['protocol'].get(k) for k in protocol_keys)
        assert jp['seed'] == 42 and jp['temperature'] == 0 and jp['enable_thinking'] is False
        assert jp['answer_model'] == 'Qwen/Qwen3-VL-8B-Instruct'
        judge_rows = unique(judged['rows'])
        saved_requests = {r['request_sha256']: r['request'] for r in lines(judge_root / 'requests.jsonl')}
        template = (judge_root / 'judge_prompt.txt').read_text()
        assert template == (common.BASELINE / 'judge_prompt.txt').read_text()
        new_scores, old_scores = [], []
        for qid, question in canonical.items():
            previous, actual, graded = old[qid], new[qid], judge_rows[qid]
            for source in (previous, actual, graded):
                assert all(source[k] == v for k, v in {'task': question['task'], 'point': question['point'],
                    'question': question['question'], 'ground_truth': gold[qid]['answer']}.items()), (mid, qid)
            assert actual['model_output'] == graded['model_output']
            assert isinstance(actual['model_output'], str)
            expected_request = wire.build_request(actual, template)
            expected_digest = wire.transport.request_sha256(expected_request)
            assert graded['request_sha256'] == expected_digest
            assert saved_requests[expected_digest] == expected_request
            assert graded['score'] in GRADES and graded['seed'] == 42 and graded['enable_thinking'] is False
            raw = graded['raw_response'].strip()
            fenced = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
            parsed = json.loads(fenced.group(1) if fenced else raw)
            assert parsed['score'] == graded['score'] and parsed['reasoning'] == graded['reasoning']
            old_score = score_answer(previous['model_output'], gold[qid]['answer'])
            new_score = score_answer(actual['model_output'], gold[qid]['answer'])
            for key in ('em', 'f1', 'bleu_1'):
                assert abs(old_score[key] - previous[key]) < 1e-12
                if key in actual:
                    assert abs(new_score[key] - actual[key]) < 1e-12
            assert int(new_score['em']) == graded['em']
            if mid == 'ours_1':
                old_graded = old_ours_judge_rows[qid]
                assert old_graded['model_output'] == previous['model_output']
                old_score['llm_judge'] = old_graded['score']
            else:
                old_score['llm_judge'] = previous['llm_judge']
            assert old_score['llm_judge'] in GRADES
            new_score['llm_judge'] = graded['score']
            old_scores.append(old_score)
            new_scores.append(new_score)
            row = {'method': mid, 'label': ref['label'], 'id': qid, 'task': question['task'],
                'point': question['point'], 'question': question['question'], 'ground_truth': gold[qid]['answer'],
                'answer42': previous['model_output'], 'answer1042': actual['model_output'],
                'answer_changed': previous['model_output'] != actual['model_output'],
                'normalized_answer_changed': normalize_answer_universal(previous['model_output']) != normalize_answer_universal(actual['model_output']),
                'hit_token_limit42': previous.get('source_hit_token_limit'),
                'hit_token_limit1042': actual.get('source_hit_token_limit'),
                'source_record_id42': previous['source_record_id'],
                'source_record_id1042': actual.get('source_record_id', qid),
                'judge_request_sha2561042': expected_digest}
            for key in METRICS:
                row[key + '42'] = old_score[key]
                row[key + '1042'] = new_score[key]
                row[key + '_delta'] = new_score[key] - old_score[key]
            details.append(row)
        old_values = {k: sum(s[k] for s in old_scores) / 1711 for k in METRICS}
        new_values = {k: sum(s[k] for s in new_scores) / 1711 for k in METRICS}
        for key in ('em', 'f1', 'bleu_1'):
            assert abs(old_values[key] - ref[key]) < 1e-12
            assert abs(new_values[key] - metric[key]) < 1e-12, (mid, key)
        old_judge_value = old_ours['summary']['llm_judge'] if mid == 'ours_1' else ref['llm_judge']
        assert abs(old_values['llm_judge'] - old_judge_value) < 1e-12
        assert abs(new_values['llm_judge'] - judged['summary']['llm_judge']) < 1e-12
        assert sum(int(s['em']) for s in new_scores) == metric['correct']
        subset = details[-1711:]
        summary = {'id': mid, 'label': ref['label'], 'questions': 1711,
            'correct42': ref['correct'], 'correct1042': metric['correct'],
            'seed42': old_values, 'seed1042': new_values,
            'delta': {k: new_values[k] - old_values[k] for k in METRICS},
            'answer_changes': sum(r['answer_changed'] for r in subset),
            'normalized_answer_changes': sum(r['normalized_answer_changed'] for r in subset),
            'em_gained': sum(r['em_delta'] > 0 for r in subset),
            'em_lost': sum(r['em_delta'] < 0 for r in subset),
            'reference_records_path': ref['records_path'], 'new_records_path': str(records_path),
            'new_metrics_path': str(metric_path), 'new_judge_summary': str(judge_summary),
            'old_judge_summary': str(ref['judge_source_path'])}
        for key in METRICS:
            values = [old_values[key], new_values[key]]
            statistics_rows.append({'method': mid, 'label': ref['label'], 'metric': key,
                'seed42': values[0], 'seed1042': values[1], 'n': 2,
                'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
                'min': min(values), 'max': max(values), 'absolute_delta_pp': abs(values[1] - values[0]) * 100})
        methods.append(summary)
        for path in (judge_summary, judge_records, judge_root / 'protocol.json', judge_root / 'summary.json',
                     judge_root / 'requests.jsonl', judge_root / 'question_map.jsonl',
                     judge_root / 'scores.jsonl', judge_root / 'reused_scores.jsonl', judge_root / 'attempts.jsonl'):
            if path.exists():
                pin(path)
    for path in (run / 'reference_seed42.json', run / 'reference_ready.json', run / 'generation_completion.json', run / 'inputs/questions.jsonl',
                 run / 'inputs/gold.jsonl', old_ours_root / 'protocol.json', old_ours_root / 'summary.json',
                 old_ours_root / 'methods/ours_1/summary.json', old_ours_root / 'methods/ours_1/per_question.jsonl',
                 Path(__file__), ROOT / 'scripts/analysis/prepare_memgallery_seed1042_reference.py'):
        pin(path)
    assert len(methods) == 9 and len(details) == 15399 and len(statistics_rows) == 36
    return {'status': 'complete', 'seeds': [42, 1042], 'questions_per_method': 1711,
            'methods': methods, 'statistics': statistics_rows, 'paired_records': details,
            'source_sha256': source_hashes,
            'interpretation': 'Two seeds are an initial repeatability check, not strong evidence of robustness. Caption, candidate, final-reader and HippoMM stages retain their native greedy decoding. Text-memory construction and retrieval retain native per-call temperatures; positive temperatures use an independent categorical sampler with seed1042, implemented through logits processing even though model.generate receives do_sample=False.',
            'caption_policy': 'Fresh seed1042 generated image-only captions and method memories; official question-image captions fixed.',
            'combo_correction': reference['correction']}


SUMMARY_COLUMNS = ['方法', '代码命名', '题数', '正确42', '正确1042', 'EM42', 'EM1042', 'ΔEM(pp)',
    'F1_42', 'F1_1042', 'ΔF1(pp)', 'BLEU1_42', 'BLEU1_1042', 'ΔBLEU1(pp)',
    'Judge42', 'Judge1042', 'ΔJudge(pp)', '回答改变数', '标准化回答改变数', 'EM由错变对', 'EM由对变错']
DETAIL_KEYS = ['method', 'label', 'id', 'task', 'point', 'question', 'ground_truth', 'answer42', 'answer1042',
    'answer_changed', 'normalized_answer_changed', 'em42', 'em1042', 'em_delta', 'f142', 'f11042', 'f1_delta',
    'bleu_142', 'bleu_11042', 'bleu_1_delta', 'llm_judge42', 'llm_judge1042', 'llm_judge_delta',
    'hit_token_limit42', 'hit_token_limit1042', 'source_record_id42', 'source_record_id1042', 'judge_request_sha2561042']
STAT_KEYS = ['method', 'label', 'metric', 'seed42', 'seed1042', 'n', 'mean', 'sample_sd', 'min', 'max', 'absolute_delta_pp']


def summary_values(method):
    values = [method['label'], method['id'], method['questions'], method['correct42'], method['correct1042']]
    for key in METRICS:
        values.extend([method['seed42'][key], method['seed1042'][key], method['delta'][key] * 100])
    return [*values, method['answer_changes'], method['normalized_answer_changes'], method['em_gained'], method['em_lost']]


def write_workbook(data, path):
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill
    from scripts.analysis.export_memgallery_five_grade_judge import append_text_safe as append, style_sheet
    book = Workbook()
    summary = book.active
    summary.title = '两种子结果'
    append(summary, SUMMARY_COLUMNS)
    for method in data['methods']:
        append(summary, summary_values(method))
    stats = book.create_sheet('均值与样本SD')
    append(stats, STAT_KEYS)
    for row in data['statistics']:
        append(stats, [row[key] for key in STAT_KEYS])
    detail = book.create_sheet('逐题对照')
    append(detail, DETAIL_KEYS)
    for row in data['paired_records']:
        append(detail, [row[key] for key in DETAIL_KEYS])
    provenance = book.create_sheet('协议与来源')
    append(provenance, ['范围', '项目', '内容'])
    notes = {'回答模型': 'Qwen/Qwen3-VL-8B-Instruct; same frozen model revision as source protocols.',
        '回答及构建seed': '42 vs1042. Caption/candidate/final reader/HippoMM保留原贪心设置；文本记忆构建及检索保留原逐调用temperature，正temperature由独立categorical sampler按seed1042采样。该采样通过logits处理实现，不能仅根据model.generate(do_sample=False)判为贪心。',
        'Judge': 'DeepSeek-v4.1-flash; fixed seed42, temperature0, thinking=false; same five-grade prompt.',
        '初步重复性': data['interpretation'], 'caption与问题图': data['caption_policy'],
        '截图纠正': data['combo_correction'],
        '样本SD': 'n=2, sample SD = |metric1042 - metric42| / sqrt(2), denominator n-1. Means/SD describe two aggregate runs.',
        '差值': 'Seed1042 - seed42; summary differences are percentage points (100×difference).',
        '完整性': 'All9 methods ×1711 questions per seed. Missing results are never filled with0. Existing truncated raw answers retained.',
        'VisRAG命名': 'Screenshot label corresponds to this repository CLIP retrieval → original images + historical dialogue baseline.',
        'HippoMM参考': 'Seed42 uses the controlled original representative-image source,681/1711; native fast/detail route retained by seed1042 driver.',
        'ours_1旧Judge': 'Frozen reference contains null; fixed old seed42 answers are judged separately and that actual result is used here.'}
    for key, value in notes.items():
        append(provenance, ['全局', key, value])
    for method in data['methods']:
        for key in ('reference_records_path', 'new_records_path', 'new_metrics_path', 'old_judge_summary', 'new_judge_summary'):
            append(provenance, [method['id'], key, method[key]])
    for filename, digest in data['source_sha256'].items():
        append(provenance, ['文件校验', filename, digest])
    style_sheet(summary, [43, 24, 10, 12, 14] + [15] * 12 + [16] * 4)
    style_sheet(stats, [24, 43, 16, 16, 16, 8, 16, 18, 16, 16, 21])
    style_sheet(detail, [24, 43, 56, 44, 10, 85, 65, 100, 100] + [18] * (len(DETAIL_KEYS) - 9))
    style_sheet(provenance, [26, 86, 115])
    for row in summary.iter_rows(min_row=2):
        for index in range(5, 17):
            row[index].number_format = '+0.0000;-0.0000;0.0000' if index in (7, 10, 13, 16) else '0.000000'
        summary.row_dimensions[row[0].row].height = 42
    for row in stats.iter_rows(min_row=2):
        for index in (3, 4, 6, 7, 8, 9, 10):
            row[index].number_format = '0.000000'
    for row in detail.iter_rows(min_row=2):
        for index in range(11, 23):
            row[index].number_format = '0.000000'
        if row[9].value:
            row[9].fill = PatternFill('solid', fgColor='FFF2CC')
    temporary = path.with_name(path.stem + '.tmp.xlsx')
    book.save(temporary)
    with ZipFile(temporary) as archive:
        assert archive.testzip() is None
    temporary.replace(path)


def equivalent(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-12
    return actual == (None if expected == '' else expected)


def audit(run=RUN, output=OUTPUT, data=None):
    from openpyxl import load_workbook
    run, output = Path(run).resolve(), Path(output).resolve()
    # Re-read original answers and actual judge outputs, recompute all metrics;
    # no reliance on workbook values or earlier report aggregates.
    fresh = collect(run)
    stored = read(output / 'summary.json')
    assert stored['methods'] == fresh['methods'] and stored['statistics'] == fresh['statistics']
    assert stored['source_sha256'] == fresh['source_sha256']
    saved_details = lines(output / 'per_question.jsonl')
    assert saved_details == fresh['paired_records']
    workbook = output / WORKBOOK_NAME
    with ZipFile(workbook) as archive:
        assert archive.testzip() is None
    book = load_workbook(workbook, read_only=True, data_only=False)
    assert book.sheetnames == ['两种子结果', '均值与样本SD', '逐题对照', '协议与来源']
    assert book['两种子结果'].max_row == 10 and book['均值与样本SD'].max_row == 37
    assert book['逐题对照'].max_row == 15400
    for values, method in zip(book['两种子结果'].iter_rows(min_row=2, values_only=True), fresh['methods']):
        assert all(equivalent(a, b) for a, b in zip(values, summary_values(method)))
    for values, row in zip(book['均值与样本SD'].iter_rows(min_row=2, values_only=True), fresh['statistics']):
        assert all(equivalent(a, row[k]) for a, k in zip(values, STAT_KEYS))
        assert abs(row['sample_sd'] - abs(row['seed1042'] - row['seed42']) / math.sqrt(2)) < 1e-12
    text_rows = 0
    for cells, row in zip(book['逐题对照'].iter_rows(min_row=2), fresh['paired_records']):
        for cell, key in zip(cells, DETAIL_KEYS):
            assert equivalent(cell.value, row[key]), (row['method'], row['id'], key)
            assert cell.data_type != 'f', 'Original source text must not become an Excel formula'
        text_rows += 1
    assert text_rows == 15399
    book.close()
    report = {'status': 'passed', 'methods': 9, 'questions_per_method': 1711,
              'paired_questions': 15399, 'answer_records': 30798,
              'new_and_reference_metrics_recomputed': True, 'all_judge_original_raw_json_verified': True,
              'new_judge_full_wire_hashes_verified': True, 'all_sources_sha256_verified': True,
              'all_workbook_summary_statistics_and_detail_cells_verified': True,
              'no_missing_score_imputation': True, 'sample_sd_n': 2,
              'workbook_sha256': sha(workbook), 'script_sha256': sha(Path(__file__))}
    dump(output / 'audit.json', report)
    return report


def export(run=RUN, output=OUTPUT):
    run, output = Path(run).resolve(), Path(output).resolve()
    data = collect(run)
    output.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in data.items() if k != 'paired_records'}
    dump(output / 'summary.json', summary)
    (output / 'per_question.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in data['paired_records']))
    write_workbook(data, output / WORKBOOK_NAME)
    audited = audit(run, output)
    completion = {'status': 'complete', 'seeds': [42, 1042], 'methods': list(METHOD_IDS),
                  'questions_per_method': 1711, 'paired_questions': 15399, 'independent_audit_passed': audited['status'] == 'passed',
                  'completed_utc': datetime.now(timezone.utc).isoformat(),
                  'output_sha256': {str(path): sha(path) for path in (output / WORKBOOK_NAME,
                       output / 'summary.json', output / 'per_question.jsonl', output / 'audit.json')}}
    dump(output / 'completion.json', completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=('status', 'export', 'audit'), required=True)
    args = parser.parse_args()
    result = status(args.run) if args.phase == 'status' else export(args.run, args.output) if args.phase == 'export' else audit(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
