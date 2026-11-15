"""Audit frozen MemEye memory prefixes and export a joint MemEye/MemGallery workbook."""
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

from scripts.analysis.memgallery_prefix5_recall import ROOT, R, TRACKED, read, rows, track

OUT = ROOT / 'reports/memeye_memgallery_prefix5_recall_20261103'


def memoryos_helpers():
    path = ROOT / 'scripts/vendor_memeye_text/memoryos_adapter.py'
    track(path)
    names = {'_round_user_with_captions', '_format_retrieval_context'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    ns = {'Dict': dict, 'Any': object, 'List': list,
          'round_image_captions': lambda o: o['raw']['image_caption']}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['_round_user_with_captions'], ns['_format_retrieval_context']


def main():
    track(Path(__file__).resolve())
    gold = {q['id']: q for q in rows(R / 'mcq_top15_caption_qa/qa_manifest.jsonl')}
    assert len(gold) == 371 and all(q['gold_ids'] for q in gold.values())
    total = sum(len(set(q['gold_ids'])) for q in gold.values())
    assert total == 1041
    all_gold = set().union(*(set(q['gold_ids']) for q in gold.values()))
    prior = {(q['method'], q['question_id']): q for q in rows(
        ROOT / 'reports/text_memory_recall_20261025/per_question.jsonl')}
    user_text, format_context = memoryos_helpers()
    specs = [('Qwen3-VL-8B', 'visual_memory_pipeline'),
             ('GPT-5', 'memeye_gpt5_seed42_text_methods_20261025'),
             ('DeepSeek-v4.1-flash', 'memeye_deepseek_flash_seed42_text_methods_20261025')]
    details, summaries = [], []
    for model, base in specs:
        for method, folder in [('Generative Agents', 'gen_agents'), ('MemoryOS', 'memoryos')]:
            root = R / base / folder
            protocol = read(root / 'protocol.json')
            mapping = {}
            if method == 'MemoryOS':
                for task, observations in read(root / 'observations.json').items():
                    mapping[task] = defaultdict(list)
                    for o in observations:
                        key = (user_text(o).strip(), o['assistant'].strip(), o['timestamp'].strip())
                        mapping[task][key].append(task + '/' + o['id'])
            else:
                for task in protocol['tasks']:
                    mapping[task] = defaultdict(list)
                    for n in read(root / task / 'final_memory.json'):
                        mapping[task][n['text'].strip()].append(n)
            current, seen = [], set()
            for r in rows(root / 'qa_responses.jsonl'):
                rid, task = r['id'], r['task']
                assert rid not in seen
                seen.add(rid)
                q = gold[rid]
                for key in ('question', 'options', 'gold_choice', 'gold_ids', 'task'):
                    assert r[key] == q[key], (model, method, rid, key)
                assert r['experiment_signature'] == protocol['experiment_signature']
                context, d = r['retrieved_context'], r['retrieval_detail']
                sources, reflections, ambiguities = [], [], []
                extras = 0
                if method == 'MemoryOS':
                    assert format_context(d) == context
                    pages = d['retrieved_pages']
                    assert 0 <= len(pages) <= 10
                    for page in pages:
                        key = tuple(page[k].strip() for k in ('user_input', 'agent_response', 'timestamp'))
                        candidates = mapping[task][key]
                        assert candidates, (model, rid, page['page_id'])
                        if len(candidates) > 1:
                            assert not set(candidates) & all_gold
                        sources.append(candidates[0] if len(candidates) == 1 else None)
                        reflections.append(False)
                        ambiguities.append(candidates if len(candidates) > 1 else [])
                    extras = sum(len(d[k]) for k in ('retrieved_user_knowledge', 'retrieved_assistant_knowledge'))
                else:
                    boundaries = list(re.finditer(r'(?m)^\[(\d+)\] ', context))
                    assert [int(b.group(1)) for b in boundaries] == list(range(10))
                    for i, b in enumerate(boundaries):
                        text = context[b.end():boundaries[i+1].start() if i+1 < len(boundaries) else len(context)].strip()
                        found = mapping[task][text]
                        assert found, (model, rid, 'unmatched or truncated block', i)
                        original = [n for n in found if n.get('source') is False]
                        candidates = [task + '/' + n['dialogue_id'] for n in original]
                        if len(found) > 1 and original:
                            assert len(found) == len(original) and not set(candidates) & all_gold
                        sources.append(candidates[0] if len(found) == len(original) == 1 else None)
                        reflections.append(not original)
                        ambiguities.append(candidates if len(candidates) > 1 else [])
                first = sources[:5]
                native = set(s for s in sources if s)
                prefix = set(s for s in first if s)
                g = set(q['gold_ids'])
                assert prefix <= native
                if model == 'Qwen3-VL-8B':
                    previous = prior[(method, rid)]
                    assert native == set(previous['retrieved_round_ids'])
                    assert len(g & native) == previous['gold_hits']
                row = dict(dataset='MemEye', model=model, method=method, id=rid, task=task,
                           native_items=len(sources), prefix_items=len(first),
                           prefix_reflections=sum(reflections[:5]), extra_knowledge_excluded=extras,
                           first5_source_ids=first, native_source_ids=sources,
                           ambiguous_candidates_by_slot=ambiguities,
                           gold_ids=sorted(g), gold_count=len(g), gold_hits5=len(g & prefix),
                           gold_hits_native=len(g & native))
                current.append(row)
            assert seen == set(gold)
            hits = sum(r['gold_hits5'] for r in current)
            native_hits = sum(r['gold_hits_native'] for r in current)
            result = dict(dataset='MemEye', model=model, method=method, questions=len(current),
                          gold_questions=len(current), gold_links=total, gold_hits5=hits,
                          recall_at5_micro=hits/total,
                          recall_at5_macro=sum(r['gold_hits5']/r['gold_count'] for r in current)/len(current),
                          hit_rate_at5=sum(r['gold_hits5'] > 0 for r in current)/len(current),
                          native_source_hits=native_hits, native_source_recall=native_hits/total,
                          prefix_count_distribution=dict(Counter(r['prefix_items'] for r in current)),
                          prefix_reflection_slots=sum(r['prefix_reflections'] for r in current),
                          ambiguous_prefix_slots=sum(bool(a) for r in current for a in r['ambiguous_candidates_by_slot'][:5]),
                          source=str(root / 'qa_responses.jsonl'))
            summaries.append(result)
            details.extend(current)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    gallery_root = ROOT / 'reports/memgallery_prefix5_recall_20261103'
    gallery = read(gallery_root / 'summary.json')
    # Verify the saved Gallery audit still refers to unchanged retrieval inputs.
    for path, expected in gallery['source_sha256'].items():
        track(Path(path))
        assert TRACKED[path] == expected
    combined = summaries + [dict(dataset='MemGallery', **r) for r in gallery['results']]
    gallery_details = [dict(dataset='MemGallery', **r) for r in rows(gallery_root / 'per_question.jsonl')]
    notes = [
        '固定现有检索顺序，只取前5个条目；不重排、不补齐、不重新检索或回答。',
        'Generative Agents的反思记忆占用条目位置；没有唯一原始轮次归属的反思不计source命中，不跳过反思补取。',
        'MemoryOS只取retrieved_pages前5页；额外用户/助手长期知识不计入；不足5页保留实际数量。',
        'MemEye: 371题均有标注，共1041个gold轮次引用；MemGallery: 1711题，其中1527题有标注，共4109个gold轮次引用。',
        'Micro Recall@5=每题前5条命中的唯一gold轮次数之和 / gold轮次引用总数。Macro与至少命中1条比例只统计有标注题。',
        '重复原文无法唯一对应轮次时不计命中，并验证全部候选均非任何题的gold；候选保存在逐题明细。',
        '原有Accuracy/EM/F1/BLEU仍属于原完整上下文回答；这里仅重新统计检索Recall@5，不是Top5回答结果。',
        'MemEye Qwen完整上下文对应的原始轮次集合和命中数，逐题复核与之前审计一致；MemGallery复用上次审计并重新验证输入SHA256。',
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'per_question.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details+gallery_details))
    (OUT / 'summary.json').write_text(json.dumps(dict(results=combined, notes=notes, source_sha256=TRACKED), ensure_ascii=False, indent=2)+'\n')
    fields = list(dict.fromkeys(k for r in combined for k in r))
    with (OUT / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(combined)
    wb = Workbook(); wb.remove(wb.active)
    headers = ['数据集','模型','方法','题数','命中线索数','总线索数','Recall@5 micro','Recall@5 macro','至少命中1条','原完整页/条召回率','少于5条题数','前5中反思槽位数']
    for dataset in ('MemEye', 'MemGallery'):
        sh = wb.create_sheet(dataset+'汇总'); sh.append(headers)
        for r in combined:
            if r['dataset'] != dataset: continue
            sh.append([dataset,r['model'],r['method'],r['questions'],r['gold_hits5'],r['gold_links'],r['recall_at5_micro'],r['recall_at5_macro'],r['hit_rate_at5'],r['native_source_recall'],sum(v for k,v in r['prefix_count_distribution'].items() if int(k)<5),r['prefix_reflection_slots']])
            for c in range(7,11): sh.cell(sh.max_row,c).number_format='0.00%'
        data = details if dataset == 'MemEye' else gallery_details
        sh = wb.create_sheet(dataset+'逐题明细'); keys=list(data[0]); sh.append(keys)
        for r in data:
            sh.append([json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(dict,list)) else r[k] for k in keys])
    sh=wb.create_sheet('统计口径'); sh.append(['说明'])
    for n in notes: sh.append([n])
    for sh in wb:
        sh.freeze_panes='A2'; sh.auto_filter.ref=sh.dimensions
        for col in sh.columns: sh.column_dimensions[col[0].column_letter].width=24
        for c in sh[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='17365D')
        for row in sh:
            for c in row: c.alignment=Alignment(wrap_text=True,vertical='top')
    wb['统计口径'].column_dimensions['A'].width=140
    for i in range(2,len(notes)+2): wb['统计口径'].row_dimensions[i].height=45
    path=OUT/'MemEye_MemGallery_GA_MemoryOS_固定前5条_Recall_20261103.xlsx'; wb.save(path)
    check=load_workbook(path,data_only=True)
    assert check['MemEye逐题明细'].max_row==2227 and check['MemGallery逐题明细'].max_row==10267
    for dataset in ('MemEye','MemGallery'):
        for i,r in enumerate((r for r in combined if r['dataset']==dataset),2):
            assert abs(check[dataset+'汇总'].cell(i,7).value-r['recall_at5_micro'])<1e-12
    for p,h in TRACKED.items():
        with Path(p).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==h
    print(path)


if __name__ == '__main__':
    main()
