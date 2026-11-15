"""Recall of first five frozen GA blocks / MemoryOS pages; no reranking or QA."""
import csv
import hashlib
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results';OUT=ROOT/'reports/memgallery_prefix5_recall_20261103'
TRACKED={}

def track(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    TRACKED[str(p)]=h

def read(p):track(p);return json.loads(p.read_text())

def rows(p):
    track(p)
    with p.open() as f:
        for line in f:
            if line.strip():yield json.loads(line)

def user_text(o):
    parts=['Date: '+o['timestamp'],'Round: '+o['id'],o['user'].strip()]
    for i,c in zip(o['raw']['image_id'],o['raw']['image_caption']):parts.extend(['image:','image_id: '+i,'image_caption: '+c])
    return '\n'.join(parts)

def main():
    gold={g['id']:g for g in rows(R/'mem_gallery_master_answer_20261021/inputs/gold.jsonl')}
    assert len(gold)==1711 and sum(len(set(g['gold_ids'])) for g in gold.values())==4109
    specs=[
      ('Qwen3-VL-8B','Generative Agents','mem_gallery_qwen_seed42_text_methods_20261026_v2','gen_agents',1071),
      ('Qwen3-VL-8B','MemoryOS','mem_gallery_qwen_seed42_text_methods_20261026_v2','memoryos',2030),
      ('GPT-5','Generative Agents','mem_gallery_gpt5_gen_agents_20261028','gen_agents',378),
      ('GPT-5','MemoryOS','mem_gallery_gpt5_memoryos_20261030','memoryos',None),
      ('DeepSeek-v4.1-flash','Generative Agents','mem_gallery_deepseek_text_methods_20261031/gen_agents','memory',460),
      ('DeepSeek-v4.1-flash','MemoryOS','mem_gallery_deepseek_text_methods_20261031/memoryos','memory',1599)]
    detail=[];summaries=[]
    for model,method,rel,sub,expected_hits in specs:
        source=R/rel;root=source/sub;obs=read(source/'observations.json');maps={};notes={};blocks={}
        if method=='MemoryOS':
            for task,group in obs.items():
                mapping=defaultdict(list)
                for o in group:mapping[(user_text(o).strip(),o['assistant'].strip(),o['timestamp'].strip())].append(task+'/'+o['id'])
                assert all(len(v)==1 for v in mapping.values());maps[task]=mapping
        else:
            for task in obs:
                ns=read(root/task/'final_memory.json');notes[task]=ns;mapping=defaultdict(list)
                for n in ns:mapping[n['text'].strip()].append(n)
                blocks[task]=mapping
        current=[];ids=set()
        for r in rows(root/'retrieval.jsonl'):
            rid,task=r['id'],r['task'];assert rid not in ids;ids.add(rid)
            context=r['retrieved_context'];d=r['retrieval_detail'];sources=[];reflections=[];extras=0
            if method=='MemoryOS':
                pages=d['retrieved_pages'];assert 0<len(pages)<=10
                for i,p in enumerate(pages,1):
                    key=tuple(p[k].strip() for k in ('user_input','agent_response','timestamp'));matches=maps[task][key];assert len(matches)==1
                    rendered=f'[Retrieved Memory {i}]\nUser: '+p['user_input'].strip()+'\nAssistant: '+p['agent_response'].strip()
                    assert rendered in context
                    sources.append(matches[0]);reflections.append(False)
                extras=sum(len(d.get(k,[])) for k in ('retrieved_user_knowledge','retrieved_assistant_knowledge'))
            elif 'selected_memory_ids' in d:
                selected=d['selected_memory_ids'];assert len(selected)==len(set(selected))==10
                chosen=[notes[task][i] for i in selected]
                full='\n'+'\n\n'.join(f'[{i}] '+n['text'] for i,n in enumerate(chosen))+'\n'
                assert ' '.join(full.split(' ')[:4000])==context
                for i,n in enumerate(chosen):
                    rendered=f'[{i}] '+n['text'];complete=rendered in context
                    sources.append(task+'/'+n['dialogue_id'] if n.get('source') is False and complete else None)
                    reflections.append(n.get('source') is not False)
            else:
                boundaries=list(re.finditer(r'(?m)^\[(\d+)\] ',context));assert len(boundaries)==10
                assert [int(b.group(1)) for b in boundaries]==list(range(10))
                for i,b in enumerate(boundaries):
                    text=context[b.end():boundaries[i+1].start() if i+1<len(boundaries) else len(context)].strip()
                    candidates=blocks[task][text];assert candidates,(rid,i)
                    originals=[n for n in candidates if n.get('source') is False]
                    if originals:assert len(candidates)==len(originals)==1
                    sources.append(task+'/'+originals[0]['dialogue_id'] if originals else None)
                    reflections.append(not originals)
            first=sources[:5];gold_ids=set(gold[rid]['gold_ids']);original={s for s in sources if s};prefix={s for s in first if s}
            assert prefix<=original
            row=dict(model=model,method=method,id=rid,task=task,point=gold[rid]['point'],
                native_items=len(sources),prefix_items=len(first),prefix_reflections=sum(reflections[:5]),
                extra_knowledge_excluded=extras,first5_source_ids=first,native_source_ids=sources,
                gold_count=len(gold_ids),gold_hits5=len(prefix&gold_ids),gold_hits_native=len(original&gold_ids))
            current.append(row);detail.append(row)
        assert ids==set(gold)
        hits=sum(r['gold_hits5'] for r in current);native=sum(r['gold_hits_native'] for r in current)
        if expected_hits is None:
            expected_hits=read(source/'final_results.json')['overall']['gold_hits']
        assert native==expected_hits,(model,method,native,expected_hits)
        annotated=[r for r in current if r['gold_count']];assert len(annotated)==1527
        result=dict(model=model,method=method,questions=len(current),gold_questions=len(annotated),
            gold_hits5=hits,gold_links=4109,recall_at5_micro=hits/4109,
            recall_at5_macro=sum(r['gold_hits5']/r['gold_count'] for r in annotated)/len(annotated),
            hit_rate_at5=sum(r['gold_hits5']>0 for r in annotated)/len(annotated),
            native_source_hits=native,native_source_recall=native/4109,
            prefix_count_distribution=dict(Counter(r['prefix_items'] for r in current)),
            prefix_reflection_slots=sum(r['prefix_reflections'] for r in current),
            extra_knowledge_excluded=True,source=str(root/'retrieval.jsonl'))
        summaries.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    notes=[
      '对已保存顺序做前缀截取，不重新检索、不重排、不运行回答；原有EM/F1/BLEU属于完整上下文，不能标成Top5回答结果。',
      'GA取原始前5条记忆，反思条目照常占位置；没有精确原始轮次归属的反思不计source命中，不跳过反思补足5条原始轮次。',
      'MemoryOS只取retrieved_pages前5页，用户/助手长期知识排除。少于5页的题保留原数量，不补齐、不换题。',
      'Recall@5 micro=所有题前5条命中的唯一gold轮次总数/4109；macro仅对1527道有gold线索题平均；184道无gold题不进入macro或hit-rate分母。',
      '全部6组均核对1711个相同问题ID；GA重建渲染内容或逐块精确匹配快照；MemoryOS按原始user/assistant/timestamp精确匹配源轮次。',
      '完整上下文的source命中数重算后与此前审计一致。该指标是原始证据轮次覆盖率，不评估反思/长期知识的语义覆盖。',
    ]
    (OUT/'summary.json').write_text(json.dumps(dict(results=summaries,notes=notes,source_sha256=TRACKED),ensure_ascii=False,indent=2)+'\n')
    (OUT/'per_question.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in detail))
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    wb=Workbook();sh=wb.active;sh.title='Recall前5条'
    sh.append(['模型','方法','题数','命中线索数','总线索数','Recall@5 micro','Recall@5 macro','至少命中1条','原完整页/条召回率','少于5条题数','前5中反思槽位数'])
    for r in summaries:
        sh.append([r['model'],r['method'],1711,r['gold_hits5'],4109,r['recall_at5_micro'],r['recall_at5_macro'],r['hit_rate_at5'],r['native_source_recall'],sum(n for k,n in r['prefix_count_distribution'].items() if int(k)<5),r['prefix_reflection_slots']])
        for c in (6,7,8,9):sh.cell(sh.max_row,c).number_format='0.00%'
    ds=wb.create_sheet('逐题明细');ds.append(list(detail[0]))
    for r in detail:ds.append([json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for v in r.values()])
    ns=wb.create_sheet('统计口径');ns.append(['说明'])
    for note in notes:ns.append([note])
    for s in wb:
        s.freeze_panes='A2';s.auto_filter.ref=s.dimensions
        for col in s.columns:s.column_dimensions[col[0].column_letter].width=23
        for c in s[1]:c.fill=PatternFill('solid',fgColor='17365D');c.font=Font(bold=True,color='FFFFFF')
        for row in s:
            for c in row:c.alignment=Alignment(wrap_text=True,vertical='top')
    ds.column_dimensions['C'].width=70;ds.column_dimensions['J'].width=90;ds.column_dimensions['K'].width=90
    ns.column_dimensions['A'].width=140
    for i in range(2,ns.max_row+1):ns.row_dimensions[i].height=45
    path=OUT/'MemGallery_GA_MemoryOS_固定检索前5条_Recall_20261103.xlsx';wb.save(path)
    back=load_workbook(path,data_only=True);assert back['逐题明细'].max_row==10267
    for i,r in enumerate(summaries,2):assert abs(back.worksheets[0].cell(i,6).value-r['recall_at5_micro'])<1e-12
    for p,h in TRACKED.items():
        with Path(p).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
    print(path)

if __name__=='__main__':main()
