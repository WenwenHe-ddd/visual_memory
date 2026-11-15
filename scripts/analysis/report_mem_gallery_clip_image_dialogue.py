"""Compare completed Qwen Mem-Gallery image and caption routes on the same cohort."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from scripts.evaluation import mem_gallery_clip_image_dialogue as experiment


def main():
    root=experiment.OUTPUT
    experiment.analyze(SimpleNamespace(output=root))
    summary=json.loads((root/'summary.json').read_text())
    source=experiment.common.DEFAULT_OUTPUT/'reader'
    baseline=json.loads((source/'summary.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text())['contract']
    for name in ('questions.jsonl','memories.jsonl'):
        path=experiment.SOURCE/name
        assert protocol['source_sha256'][str(path)]==experiment.sha(path)
    qs=experiment.rows(root/'qa_manifest.jsonl')
    responses=experiment.rows(root/'qa_responses.jsonl')
    indexed=experiment.common.by_id(responses)
    assert set(indexed)=={q['id'] for q in qs} and len(responses)==1711
    for q in qs:
        r=indexed[q['id']]
        assert r['reader_image_count']==len(r['image_grid_thw'])==len(q['reader_image_paths'])
        assert r['dialogue_round_count']==sum(len(c['source_rounds']) for c in q['retrieved_images'])
        assert r['top_memory_ids']==q['top_memory_ids']
    stats=summary['overall']
    assert stats['questions']==1711 and stats['gold_references']==4109
    assert stats['nonempty_clue_questions']==1527
    prior=baseline['overall']['baseline_question_only']
    comparison=[dict(method='Caption baseline (BGE Top5)',questions=1711,em=prior['em'],f1=prior['f1'],
        recall_at5_micro=prior['recall_at_5_micro'],source=str(source/'summary.json')),
        dict(method='CLIP Top5 → 原图＋历史对话',questions=1711,em=stats['em'],f1=stats['f1'],
        recall_at5_micro=stats['recall_at5_micro'],source=str(root/'summary.json'))]
    out=experiment.ROOT/'reports/mem_gallery_clip_image_dialogue_qwen_20261102'
    out.mkdir(parents=True,exist_ok=True)
    detail=experiment.rows(root/'per_question.jsonl')
    assert sum(r['em'] for r in detail)==stats['exact_match_correct']
    for name,records in [('comparison.csv',comparison),('per_question.csv',detail)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    book=Workbook();sheet=book.active;sheet.title='结果对比'
    sheet.append(['方法','题数','EM','F1','Recall@5 micro','结果来源'])
    for row in comparison:
        sheet.append(list(row.values()))
        for c in (3,4,5):sheet.cell(sheet.max_row,c).number_format='0.00%'
    types=book.create_sheet('题型明细');types.append(['题型','题数','EM','F1','Recall@5 micro'])
    for name,row in summary['by_question_type'].items():
        types.append([name,row['questions'],row['em'],row['f1'],row['recall_at5_micro']])
        for c in (3,4,5):types.cell(types.max_row,c).number_format='0.00%'
    notes=book.create_sheet('实验口径');notes.append(['说明'])
    for text in [
        'Mem-Gallery全部1711题，Qwen3-VL-8B-Instruct，GPU0，BF16，greedy，max_new_tokens=256，Top5。EM是开放式答案规范化后的完全匹配率，不是选择题准确率。',
        'CLIP仅编码问题文本，按同场景历史原图相似度检索。无历史caption、问题图caption、候选答案或gold参与检索。问题原图只送入reader。',
        '1003个含图轮次按同场景图片内容去重为1001个候选；每张取回图包含所有对应源轮次的对话、日期、round和官方image ID。',
        '1224题输入5张历史图，487题另有1张问题原图。2959个纯文本轮次不在CLIP图片候选池内。',
        'Recall micro分母保留所有4109条gold线索，包括纯文本线索；184道无clue题保留在答案评分中，排除于有标注题macro recall分母。',
        'Caption对照为已完成BGE Top5实验，检索问题包含已有问题图caption，候选池也包含纯文本轮次；本表是两条完整路线的比较，不能将差异单独归因于原图/文本回答。',
        '沿用项目中冻结的官方规范化、EM、F1和BLEU实现；不额外运行LLM judge。',
    ]:notes.append([text])
    for s in book:
        s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
        for cell in s[1]:cell.fill=PatternFill('solid',fgColor='17365D');cell.font=Font(color='FFFFFF',bold=True)
        for row in s:
            for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='top')
        s.column_dimensions['A'].width=45
        for col in ('B','C','D','E'):s.column_dimensions[col].width=18
    sheet.column_dimensions['F'].width=90;notes.column_dimensions['A'].width=130
    for i in range(2,notes.max_row+1):notes.row_dimensions[i].height=45
    path=out/'MemGallery_Qwen_CLIP原图对话_结果对比_20261102.xlsx';book.save(path)
    reopened=load_workbook(path,data_only=True)
    assert abs(reopened.worksheets[0].cell(3,3).value-stats['em'])<1e-12
    audit=dict(status='passed',questions=1711,unique_complete_response_ids=True,actual_images_and_dialogue_counts_verified=True,
        em_recounted=True,source_cohort_matches_caption_baseline=True,responses_sha256=experiment.sha(root/'qa_responses.jsonl'))
    experiment.dump(root/'independent_output_audit.json',audit)
    experiment.dump(out/'comparison.json',dict(rows=comparison,validation=audit))
    print(path)


if __name__=='__main__':main()
