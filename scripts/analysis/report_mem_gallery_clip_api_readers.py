"""Audit and report three readers over identical frozen Mem-Gallery CLIP contexts."""
import csv
import json
from pathlib import Path
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment
from scripts.evaluation import mem_gallery_clip_image_dialogue_api as api
from scripts.evaluation import mem_gallery_clip_image_dialogue as qwen


def main():
    source_protocol,manifest,qwen_answers=qwen.verify(qwen.OUTPUT)
    source_by_id={q['id']:q for q in manifest}
    sources=[qwen.OUTPUT]+[api.DEFAULT_OUTPUT/m for m in api.MODELS]
    summaries=[]
    for root in sources:
        if root!=qwen.OUTPUT:api.analyze(root)
        summary=api.read(root/'summary.json')
        detail=api.rows(root/'per_question.jsonl')
        responses=api.rows(root/'qa_responses.jsonl')
        assert len(responses)==len(detail)==1711
        assert {r['id'] for r in responses}==set(source_by_id)
        assert len({r['id'] for r in responses})==1711
        assert sum(int(r['em']) for r in detail)==summary['overall']['exact_match_correct']
        assert sum(r['gold_hits'] for r in detail)==1458
        assert sum(r['gold_count'] for r in detail)==4109
        if root!=qwen.OUTPUT:
            p,qs,expected,verified=api.verify(root)
            assert p['source_signature']==source_protocol['experiment_signature']
            assert api.sha(root/'qa_manifest.jsonl')==api.sha(qwen.OUTPUT/'qa_manifest.jsonl')
            by_id={r['id']:r for r in responses}
            changed_images=set();changed_questions=[]
            cache=api.read(api.IMAGE_CACHE)
            for q in qs:
                original=source_by_id[q['id']];a=by_id[q['id']]
                assert q==original
                assert a['top_memory_ids']==q['top_memory_ids']
                assert a['source_payload_sha256']==q['payload_sha256']
                assert a['reader_image_count']==len(q['reader_image_paths'])
                body=api.build_request(q['reader_messages'],q['reader_image_hashes'],p['model'])
                assert api.digest(body)==expected[q['id']]['request_sha256']==a['request_sha256']
                for old,new in zip(q['reader_messages'][1]['content'],body['messages'][1]['content']):
                    if old['type']=='text':assert old==new
                changed=[h for h in q['reader_image_hashes'].values() if h in cache['entries']]
                if p['model']!='gpt-5' and changed:
                    changed_images.update(changed);changed_questions.append(q['id'])
            audit=dict(status='passed',questions=1711,original_source_signature=source_protocol['experiment_signature'],
                identical_frozen_manifest=True,all_request_hashes_rebuilt=True,all_context_ids_verified=True,
                source_payloads_verified=True,gold_hits=1458,gold_references=4109,
                reencoded_unique_images=len(changed_images),reencoded_image_question_count=len(changed_questions),
                reencoded_image_question_ids=changed_questions,
                responses_sha256=api.sha(root/'qa_responses.jsonl'))
            api.dump(root/'independent_output_audit.json',audit)
        summaries.append((root,summary,detail))
    out=qwen.ROOT/'reports/mem_gallery_clip_api_readers_20261102';out.mkdir(parents=True,exist_ok=True)
    comparison=[]
    for root,s,_ in summaries:
        v=s['overall']
        comparison.append(dict(model=s['model'],questions=v['questions'],top_k=5,
            exact_match_correct=v['exact_match_correct'],em=v['em'],f1=v['f1'],bleu_1=v['bleu_1'],recall_at5_micro=v['recall_at5_micro'],
            token_limit_hits=v['token_limit_hits'],empty_responses=v['empty_responses'],
            provider_rejections=v.get('provider_rejections',0),source=str(root/'summary.json')))
    api.dump(out/'comparison.json',comparison)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    book=Workbook();s=book.active;s.title='模型结果对比'
    s.append(['回答模型','题数','K','EM完全匹配题数','EM','F1','BLEU-1','Recall@5 micro','输出达到上限','空回答','接口内容拒绝','结果来源'])
    for r in comparison:
        s.append(list(r.values()))
        for c in (5,6,7,8):s.cell(s.max_row,c).number_format='0.00%'
    typ=book.create_sheet('题型结果');typ.append(['模型','题型','题数','EM','F1','BLEU-1','Recall@5 micro'])
    for _,su,_ in summaries:
        for t,v in su['by_question_type'].items():
            typ.append([su['model'],t,v['questions'],v['em'],v['f1'],v['bleu_1'],v['recall_at5_micro']])
            for c in (4,5,6,7):typ.cell(typ.max_row,c).number_format='0.00%'
    per=book.create_sheet('逐题结果');per.append(['ID','场景','题型','标准答案','Qwen回答','Qwen EM','GPT-5回答','GPT-5 EM','DeepSeek回答','DeepSeek EM','GPT-5状态','DeepSeek状态'])
    details=[{r['id']:r for r in d} for _,_,d in summaries]
    for q in manifest:
        d=[x[q['id']] for x in details]
        per.append([q['id'],q['task'],q['point'],d[0]['answer']]+[v for r in d for v in (r['prediction'],r['em'])]+[d[1]['status'],d[2]['status']])
    notes=book.create_sheet('实验口径');notes.append(['说明'])
    lines=[
      'Mem-Gallery全部1711题，20场景。EM为规范化答案完全匹配率，非选择题Accuracy；F1沿用冻结的官方实现。',
      'BLEU-1为规范化答案与标准答案的单词级匹配评分，包含长度惩罚；先逐题计算sentence BLEU-1，再对1711题取平均，以百分比显示。Recall@5单独衡量检索命中，不是回答评分。',
      '三模型使用同一CLIP ViT-B/32问题文本检索、同场景Top5图片、对应历史对话、日期、round、image ID和同一回答提示词。无caption参与检索或回答。',
      '1224题有5张历史图片，487题另加1张原始问题图片。输入不做文本截断。所有图像位置、文本块和取回ID均核对。',
      'Recall@5 micro固定1458/4109；分母保留纯文本线索。图片候选不覆盖2959个纯文本轮次。184道无clue题仍纳入答案评分。',
      'GPT-5实际模型为gpt-5-2025-08-07；DeepSeek使用既有接口实际模型名deepseek-v4.1-flash。实际返回型号以summary为准。',
      'Qwen greedy，最多256输出tokens；DeepSeek温度0、关闭思考、最多256输出tokens；GPT-5 minimal reasoning、默认温度，2048总completion tokens包含思考。这不是完全相同解码器或图像预处理的比较。',
      'GPT使用原始图像字节；DeepSeek超过单图Data URI限制的图片沿用已验证的等分辨率JPEG缓存，不裁切不缩放。CLIP始终使用原图，转换题号见independent_output_audit.json。',
      '每题只接受一次成功回答；仅瞬时网络/限流等错误最多3次重试，不根据答案或格式重试。接口内容拒绝保留分母。',
      '参考结果文件和protocol中冻结的请求hash；报告核验1711题ID、取回ID、上下文、请求重建hash以及答案计数。',
    ]
    for line in lines:notes.append([line])
    for sh in book:
        sh.freeze_panes='A2';sh.auto_filter.ref=sh.dimensions
        for c in sh[1]:c.fill=PatternFill('solid',fgColor='17365D');c.font=Font(bold=True,color='FFFFFF')
        for row in sh:
            for c in row:
                c.alignment=Alignment(wrap_text=True,vertical='top')
                if isinstance(c.value,str):c.data_type='s'
        for col in sh.columns:sh.column_dimensions[col[0].column_letter].width=22
    s.column_dimensions['A'].width=30;s.column_dimensions['L'].width=85
    for c in ('D','E','G','I'):per.column_dimensions[c].width=60
    per.column_dimensions['A'].width=65;notes.column_dimensions['A'].width=135
    for i in range(2,notes.max_row+1):notes.row_dimensions[i].height=45
    path=out/'MemGallery_CLIP原图对话_Qwen_GPT5_DeepSeek_20261102.xlsx';book.save(path)
    check=load_workbook(path,data_only=True)
    assert check['逐题结果'].max_row==1712
    for i,r in enumerate(comparison,2):
        assert check.worksheets[0].cell(i,4).value==r['exact_match_correct']
        assert abs(check.worksheets[0].cell(i,5).value-r['em'])<1e-12
        assert abs(check.worksheets[0].cell(i,7).value-r['bleu_1'])<1e-12
    print(json.dumps(comparison,ensure_ascii=False));print(path)

if __name__=='__main__':main()
