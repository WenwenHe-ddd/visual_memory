"""Export the completed 371-question MemEye Qwen Semantic RAG open evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from openpyxl import Workbook, load_workbook

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis import export_memgallery_five_grade_judge as common
from scripts.evaluation import memeye_open_five_grade_judge as judge
from scripts.evaluation import mem_gallery_master_reader as reader

DEFAULT_ROOT = ROOT / 'results/memeye_qwen_semrag_open_20261105'
DEFAULT_OUTPUT = (ROOT / 'reports/memeye_qwen_semrag_open_20261105'
                  / 'MemEye_Qwen_SemanticRAG_Open_五档LLM_Judge_20261105.xlsx')
N = 371


def load_and_validate(root):
    answer = common.read_json(root / 'answer_summary.json')
    summary = common.read_json(root / 'judge/summary.json')
    protocol = common.read_json(root / 'protocol.json')
    reader_protocol = common.read_json(root / 'reader/protocol.json')
    judge_protocol = common.read_json(root / 'judge/protocol.json')
    inputs = common.read_json(root / 'input_validation.json')
    seal = common.read_json(root / 'reader/completion.json')
    require = common.require
    require(answer['status'] == summary['status'] == seal['status'] == 'complete',
            'All 371 reader answers and judge grades must be complete before export')
    require(answer['questions'] == summary['questions_expected'] == summary['questions_scored'] == N
            and summary['questions_missing'] == 0 and seal['questions_answered'] == N,
            'Expected the complete 371-question cohort')
    require(answer['answer_model'] == summary['answer_model'] == judge.ANSWER_MODEL,
            'Wrong answer backbone')
    require(summary['judge_model'] == judge.transport.DEEPSEEK_MODEL, 'Wrong judge backbone')
    for key, expected in judge.shared.judge_settings().items():
        require(judge_protocol[key] == expected, f'Judge protocol differs: {key}')
    for record, directory in ((answer, root), (summary, root / 'judge'), (protocol, root)):
        common.verify_hashes(record['artifact_sha256'], directory)
    common.verify_hashes(protocol['source_sha256'])
    common.verify_hashes(judge_protocol['source_sha256'])
    common.verify_hashes(judge_protocol['implementation_sha256'])
    contract = reader_protocol['contract']
    common.verify_hashes(contract['source_sha256'])
    common.verify_hashes(contract['query_image_sha256'])
    require(reader.digest(contract) == reader_protocol['experiment_signature'],
            'Reader protocol signature changed')
    require(seal['responses_sha256'] == common.sha(root / 'reader/responses.jsonl'),
            'Sealed original answers changed')
    require(contract['model']['repo'] == judge.ANSWER_MODEL and contract['seed'] == 42
            and contract['do_sample'] is False and contract['batch_size'] == 1
            and contract['max_new_tokens'] == 512 and contract['top_k'] == 5,
            'Unexpected Qwen reader settings')
    require(protocol['memory_leaves'] == 438 and protocol['top_k'] == 5
            and inputs['passed'] and inputs['all_contexts_exactly_five_memories'],
            'Unexpected memory or retrieval settings')
    paths = {'scored': 'scored_questions.jsonl', 'judge': 'judge/per_question.jsonl',
        'manifest': 'reader_manifest.jsonl', 'responses': 'reader/responses.jsonl',
        'questions': 'inputs/questions.jsonl', 'gold': 'inputs/gold.jsonl',
        'judge_inputs': 'judge_inputs.jsonl'}
    maps = {name: common.unique(common.read_rows(root / filename)) for name, filename in paths.items()}
    expected = set(maps['questions'])
    require(len(expected) == N and all(set(values) == expected for values in maps.values()),
            'Original data, answers, and scores must have identical unique IDs')
    requests = common.unique(common.read_rows(root / 'judge/requests.jsonl'), 'request_sha256')
    scores = judge.shared.validated_results(root / 'judge', list(requests.values()))
    require(len(scores) == len(requests), 'Incomplete unique judge requests')
    template = judge.shared.PROMPT.read_text()
    for qid in maps['questions']:
        q, gold, source, scored, graded, manifest, ji = (
            maps[name][qid] for name in ('questions', 'gold', 'responses', 'scored', 'judge',
                                        'manifest', 'judge_inputs'))
        require(q['question'] == scored['question'] == graded['question'] == manifest['question']
                == ji['question'], f'Original question changed: {qid}')
        require(gold['answer'] == scored['answer'] == graded['ground_truth'] == ji['ground_truth'],
                f'Original reference answer changed: {qid}')
        require(source['raw_response'] == scored['raw_response'] == graded['model_output']
                == ji['model_output'], f'Original Qwen answer changed: {qid}')
        require(source['payload_sha256'] == manifest['payload_sha256'] == ji['source_payload_sha256']
                and source['top_memory_ids'] == manifest['top_memory_ids'] == scored['top_memory_ids'],
                f'Original reader evidence changed: {qid}')
        require(source['experiment_signature'] == reader_protocol['experiment_signature']
                and source['run_id'] == reader_protocol['run_id'], f'Wrong reader run: {qid}')
        require(len(manifest['top_memory_ids']) == len(set(manifest['top_memory_ids'])) == 5,
                f'Context does not contain five distinct memories: {qid}')
        computed = reader.score_answer(source['raw_response'], gold['answer'])
        require(all(abs(scored[key] - computed[key]) < 1e-12 for key in ('em', 'f1', 'bleu_1')),
                f'Answer metrics do not reproduce: {qid}')
        hits = len(set(gold['gold_ids']) & set(manifest['top_memory_ids']))
        require(scored['gold_hits'] == hits and scored['gold_count'] == len(gold['gold_ids']),
                f'Retrieval recall counts differ: {qid}')
        request = judge.shared.build_request(graded, template)
        digest = judge.transport.request_sha256(request)
        require(digest == graded['request_sha256'] and requests[digest]['request'] == request,
                f'Judge request does not reproduce: {qid}')
        judge.shared.validate_score(graded, digest)
        require(all(graded[key] == scores[digest][key] for key in ('score', 'reasoning', 'raw_response')),
                f'Per-question judge output differs from durable score: {qid}')
        require(graded['em'] == computed['em'] and graded['judge_status'] == 'ok',
                f'Judge row metric/status mismatch: {qid}')
    scored_rows = list(maps['scored'].values())
    graded_rows = list(maps['judge'].values())
    correct = sum(row['em'] for row in scored_rows)
    require(correct == answer['correct_em'] == summary['source_open_em_correct'], 'Open EM count differs')
    for metric in ('em', 'f1', 'bleu_1'):
        require(abs(answer[metric] - sum(row[metric] for row in scored_rows) / N) < 1e-12,
                f'Overall answer metric differs: {metric}')
    hits, refs = (sum(row[key] for row in scored_rows) for key in ('gold_hits', 'gold_count'))
    require(hits == answer['gold_hits_top5'] and refs == answer['gold_references']
            and abs(answer['recall_at5_micro'] - hits / refs) < 1e-12, 'Micro Recall@5 differs')
    total = sum(row['score'] for row in graded_rows)
    counts = Counter(row['score'] for row in graded_rows)
    require(total == summary['score_sum'] and abs(total / N - summary['llm_judge']) < 1e-12,
            'Judge mean differs from original grades')
    require({str(g): counts[g] for g in common.GRADES} == summary['score_distribution'],
            'Judge distribution differs')
    require(answer['reference_mcq']['correct'] == protocol['original_mcq_correct'] == 184
            and answer['reference_mcq']['questions'] == 371, 'Wrong historical MCQ reference')
    return answer, summary, protocol, reader_protocol, judge_protocol, inputs, maps


def export(root, output):
    root, output = root.resolve(), output.resolve()
    answer, summary, protocol, rp, jp, inputs, maps = load_and_validate(root)
    book = Workbook()
    front = book.active
    front.title = '汇总'
    details = book.create_sheet('逐题评分')
    sources = book.create_sheet('协议与来源')
    append = common.append_text_safe
    append(front, ['任务类型', '方法', '回答模型', '正确数', '题数', 'Open EM',
        'Accuracy（仅MCQ）', 'F1', 'BLEU-1', 'LLM Judge（0–1）', 'Recall@5（micro）',
        '0分', '0.25分', '0.5分', '0.75分', '1分', 'Top K', '说明'])
    open_summary_row = ['本次开放式生成', 'Semantic RAG（caption基线）', judge.ANSWER_MODEL,
        answer['correct_em'], N, answer['em'], None, answer['f1'], answer['bleu_1'],
        summary['llm_judge'], answer['recall_at5_micro'],
        *[summary['score_distribution'][str(g)] for g in common.GRADES], 5,
        '官方Open题目，不提供选项；原始问题图片＋检索得到的历史对话与完整caption回答。']
    append(front, open_summary_row)
    append(front, ['历史选择题（独立参考）', 'Semantic RAG（caption基线）', judge.ANSWER_MODEL,
        184, N, None, 184 / N, None, None, None, None, None, None, None, None, None, 5,
        '184/371是选择题Accuracy；不是本次开放式EM，题目文本及回答形式不同，不能直接比较。'])
    append(details, ['ID', '场景', '题型', '官方Open问题', '标准答案', 'Qwen原始回答',
        'EM', 'F1', 'BLEU-1', 'Judge分数', 'Judge评分理由', '命中线索数', '总线索数',
        'Top5记忆ID（按排名）', '回答使用的完整上下文', '原始问题图片路径',
        '实际检索输入', '回答达到token上限', 'Judge request_sha256', 'Judge response_id'])
    expected_detail_rows = []
    for qid, question in maps['questions'].items():
        scored, grade, manifest = (maps[name][qid] for name in ('scored', 'judge', 'manifest'))
        row = [qid, question['task'], question['point'], question['question'], scored['answer'],
            scored['raw_response'], scored['em'], scored['f1'], scored['bleu_1'], grade['score'],
            grade['reasoning'], scored['gold_hits'], scored['gold_count'], manifest['top_memory_ids'],
            manifest['context'], manifest['query_image_paths'], question['retrieval_query'],
            scored['hit_token_limit'], grade['request_sha256'], grade.get('response_id')]
        append(details, row)
        expected_detail_rows.append(row)
    append(sources, ['范围', '项目', '值'])
    notes = {
        '指标定义': 'LLM Judge为全部371题五档评分的算术平均；空回答也保留。',
        'EM/F1/BLEU-1': '沿用MemGallery文本归一化、词干F1与句子BLEU-1实现；不是选择题正确率。',
        'Recall@5': f'所有题命中的gold线索数/全部gold线索数={answer["gold_hits_top5"]}/{answer["gold_references"]}。',
        '回答输入': 'Top5历史对话＋完整Qwen图片caption＋官方Open问题＋原始问题图片；无选项。',
        '检索输入': 'Open问题文本＋已有Qwen问题图片caption（若有）；不使用选项或标准答案。',
        'Memory': '复用438条历史记忆，含原始对话及完整Qwen caption；不重新构建记忆。',
        'Open题目变化': f'{inputs["changed_question_text"]}题与MCQ问题文本不同，{inputs["unchanged_question_text"]}题相同。',
        '检索结果变化': f'{inputs["top5_changed_from_mcq"]}题的有序Top5与MCQ版不同；基于真实Open问题重新检索。',
        'Qwen设置': 'seed=42；贪心解码do_sample=false；batch_size=1；max_new_tokens=512。',
        'Judge设置': 'deepseek-v4.1-flash；temperature=0；seed=42；enable_thinking=false；max_tokens=1024。',
        'Judge输入': '仅问题、文本标准答案、Qwen生成答案；不输入检索上下文或图片。',
        '五档': '0 / 0.25 / 0.5 / 0.75 / 1；保留0.25和0.75，不重新映射。',
        '历史184/371': '原选择题Accuracy，单独保留作方法来源参考，不能当作本次开放式EM。',
    }
    for key, value in notes.items():
        append(sources, ['说明', key, value])
    for scope, record in [('输入协议', protocol), ('Reader协议', rp['contract']),
                          ('Judge协议', jp), ('回答汇总', answer), ('Judge汇总', summary)]:
        for key, value in record.items():
            if key.endswith('sha256') and isinstance(value, dict):
                for name, digest in value.items():
                    append(sources, [scope, key + ': ' + name, digest])
            elif key not in {'question_differences'}:
                append(sources, [scope, key, value])
    for relative in ('protocol.json', 'reader/protocol.json', 'reader/responses.jsonl',
                     'answer_summary.json', 'judge/summary.json', 'judge/per_question.jsonl',
                     'scored_questions.jsonl', 'reader_manifest.jsonl', 'input_validation.json'):
        append(sources, ['本次文件', str(root / relative), common.sha(root / relative)])
    common.style_sheet(front, [28, 35, 35, 12, 12, 15, 22, 15, 15, 23, 23, 10, 10, 10, 10, 10, 10, 85])
    common.style_sheet(details, [65, 45, 16, 80, 60, 100, 10, 15, 15, 15, 100, 15, 15, 90, 120, 100, 90, 23, 68, 42])
    common.style_sheet(sources, [24, 90, 120])
    for row in front.iter_rows(min_row=2):
        for cell in row[5:11]:
            cell.number_format = '0.000000'
        front.row_dimensions[row[0].row].height = 70
    for row in details.iter_rows(min_row=2):
        for cell in row[6:10]:
            cell.number_format = '0.000000'
        details.row_dimensions[row[0].row].height = 90
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.tmp.xlsx')
    book.save(temporary)
    with ZipFile(temporary) as archive:
        common.require(archive.testzip() is None, 'Invalid XLSX archive')
    checked = load_workbook(temporary, read_only=True, data_only=False)
    common.require(checked['汇总'].max_row == 3 and checked['逐题评分'].max_row == N + 1,
                   'Workbook row counts differ')

    def equivalent(actual, expected):
        if isinstance(expected, (dict, list)):
            expected = json.dumps(expected, ensure_ascii=False, sort_keys=True)
        if expected == '':
            expected = None
        if isinstance(expected, float):
            return isinstance(actual, (int, float)) and abs(actual - expected) < 1e-12
        return actual == expected

    exported_summary = list(checked['汇总'].iter_rows(min_row=2, max_row=2, values_only=True))[0]
    common.require(all(equivalent(a, b) for a, b in zip(exported_summary, open_summary_row)),
                   'Exported summary differs from its sources')
    for actual, expected in zip(checked['逐题评分'].iter_rows(min_row=2, values_only=True), expected_detail_rows):
        common.require(all(equivalent(a, b) for a, b in zip(actual, expected)),
                       f'Exported original text or metric changed: {expected[0]}')
    for sheet in checked:
        common.require(all(cell.data_type != 'f' for row in sheet.iter_rows() for cell in row),
                       'Unexpected formula in exported original text')
    checked.close()
    temporary.replace(output)
    validation = {'passed': True, 'question_rows': N, 'summary_rows': 2,
        'open_em_correct': answer['correct_em'], 'llm_judge': summary['llm_judge'],
        'score_sum': summary['score_sum'], 'score_distribution': summary['score_distribution'],
        'original_answer_text_verified': True, 'original_judge_output_verified': True,
        'all_exported_detail_cells_read_back': True, 'formula_count': 0,
        'output': str(output), 'xlsx_sha256': common.sha(output)}
    output.with_suffix('.validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.root, args.output)


if __name__ == '__main__':
    main()
