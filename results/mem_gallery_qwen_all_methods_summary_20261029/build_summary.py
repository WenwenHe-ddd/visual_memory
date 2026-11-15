import csv, hashlib, html, json
from pathlib import Path
from collections import defaultdict
ROOT=Path('/storage/hewenwen/VisualMemory')
OUT=Path(__file__).resolve().parent
metrics=('em','f1','bleu_1','bleu_2','bleu')
sources={}
rows=[]
detail=[]
expected=None
def read(p):
 p=ROOT/p; sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return [json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text())
def aggregate(items):
 valid=[r for r in items if r['gold_count']]
 return dict(questions=len(items),correct=sum(int(r['em']) for r in items),**{k:sum(r[k] for r in items)/len(items) for k in metrics},source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in valid)/len(valid) if valid else None,source_recall_micro=sum(r['gold_hits'] for r in valid)/sum(r['gold_count'] for r in valid) if valid else None,source_hit_rate=sum(r['gold_hits']>0 for r in valid)/len(valid) if valid else None,gold_questions=len(valid),gold_links=sum(r['gold_count'] for r in valid),token_limit_records=sum(r['hit_token_limit'] for r in items))
def add(name,group,context,items,source,stored):
 global expected
 ids={r['id'] for r in items}
 assert len(items)==len(ids)==1711,(name,len(items))
 signatures={r['id']:(r['answer'],tuple(sorted(r['gold_ids']))) for r in items}
 if expected is None: expected=signatures
 else: assert expected==signatures,name
 for r in items: assert r['gold_hits']==len(set(r['source_ids'])&set(r['gold_ids'])),(name,r['id'])
 a=aggregate(items)
 for k in metrics: assert abs(a[k]-stored[k])<1e-12,(name,k)
 rows.append(dict(method=name,group=group,context=context,**a,llm_judge=None,source=str(ROOT/source)))
 for field in ('point','task'):
  groups=defaultdict(list)
  for r in items: groups[r[field]].append(r)
  for k,v in sorted(groups.items()): detail.append(dict(method=name,group_by=field,group_value=k,**aggregate(v)))
base='results/mem_gallery_master_answer_20261021'
variants=[(base,'主要方法',''),('results/mem_gallery_qwen_zscore_l0p01_a1_20261027','参数实验',' + z-score (λ=0.01, α=1)'),('results/mem_gallery_qwen_zscore_l0p03_a1_20261027','参数实验',' + z-score (λ=0.03, α=1)'),('results/mem_gallery_qwen_zscore_l0p03_a0p3_20261028','参数实验',' + z-score (λ=0.03, α=0.3)')]
for root,group,suffix in variants:
 pairs=read(root+'/reader/paired_results.json'); summary=read(root+'/reader/summary.json')
 protocol=read(root+'/reader/protocol.json')['contract']
 assert protocol['model']['repo']=='Qwen/Qwen3-VL-8B-Instruct' and protocol['seed']==42
 for key,label in [('baseline_question_only','Natural（全量检索基线）'),('master_answer_query','Master-answer')]:
  condition=key+('+zscore' if suffix else '')
  items=[]
  for r in pairs:
   v=r[condition]; items.append(dict(id=r['id'],answer=r['answer'],gold_ids=r['gold_ids'],task=r['task'],point=r['point'],**v,gold_count=v['gold_clues'],source_ids=v['top5_memory_ids']))
  add(label+suffix,group,'Top5',items,root+'/reader/paired_results.json',summary['overall'][condition])
root='results/mem_gallery_qwen_seed42_text_methods_20261026_v2'
final=read(root+'/final_results.json')
for key,label,context in [('simplemem','SimpleMem','最多5条；8题为空'),('memoryos','MemoryOS','原生多层检索'),('a_mem','A-Mem','5条'),('gen_agents','Generative Agents','10条')]:
 p=read(root+'/'+key+'/protocol.json'); assert p['reader_model']=='Qwen3-VL-8B-Instruct' and p['seed']==42
 items=read(root+'/'+key+'/scored_questions.jsonl')
 add(label,'主要方法',context,items,root+'/'+key+'/scored_questions.jsonl',final['methods'][key]['overall'])
rows.sort(key=lambda r:(r['group']!='主要方法',-r['f1']))
for file,data in [('overall.csv',rows),('by_type_and_domain.csv',detail)]:
 with (OUT/file).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
notes=['模型：Qwen3-VL-8B-Instruct；seed=42；20个场景、1711道题。所有回答指标均以全部题目为分母。',
'汇总本地已完成的6种主要方法及6组z-score参数结果；不代表已复现Mem-Gallery论文全部方法。旧版未完成文本记忆实验、GPT-5结果及仅检索实验不混入问答比较。',
'历史图片使用缓存的Qwen caption，问题图片caption沿用官方数据；回答阶段使用原始问题图片。',
'LLM-as-a-Judge均未运行。BLEU列为逐题平滑BLEU-4；所有分数为0到1，网页按百分数展示。',
'证据召回率按各运行已记录的原始source_ids与gold_ids计算。仅1527道有证据标注的题参与，合计4109条证据引用；不能度量反思或压缩知识的全部语义覆盖。',
'各方法检索器、记忆表达及上下文预算不同，不能把本表解释成固定Top-K的纯检索消融。Natural/Master-answer为Top5；GA原生K=10；MemoryOS原生多层检索；SimpleMem和A-Mem最多5条。',
'MemoryOS有2条回答触及输出token上限，SimpleMem有1条，均保留在指标中。',
'从逐题已保存分数重新汇总；核对每组1711个唯一ID、相同标准答案与证据、源ID交集以及全部5项回答指标与历史汇总一致。本次未重新调用模型。']
(OUT/'summary.json').write_text(json.dumps(dict(model='Qwen3-VL-8B-Instruct',seed=42,notes=notes,results=rows),ensure_ascii=False,indent=2))
(OUT/'audit.json').write_text(json.dumps(dict(status='passed',configurations=len(rows),unique_questions_per_configuration=1711,records_checked=len(rows)*1711,identical_question_answer_gold_sets=True,stored_answer_metrics_reaggregated=True,source_intersections_verified=True,source_sha256=sources),indent=2))
cols=[('method','方法'),('context','上下文'),('correct','EM命中数'),('em','EM'),('f1','F1'),('bleu_1','BLEU-1'),('bleu_2','BLEU-2'),('bleu','BLEU-4'),('source_recall_macro','证据Recall宏'),('source_recall_micro','证据Recall微'),('source_hit_rate','证据Hit'),('token_limit_records','输出达上限')]
parts=['<!doctype html><meta charset="utf-8"><title>Mem-Gallery / Qwen 全方法结果</title><style>body{font:16px system-ui;margin:36px;color:#172b40}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f7}li{margin:8px 0}h1{font-size:26px}</style><h1>Mem-Gallery · Qwen3-VL-8B-Instruct</h1><p>1711道题 · 20个场景 · seed 42 · 本地实验汇总</p>']
for group in ['主要方法','参数实验']:
 parts.append('<h2>'+group+'</h2><table><tr>'+''.join('<th>'+label+'</th>' for _,label in cols)+'</tr>')
 for r in rows:
  if r['group']!=group:continue
  parts.append('<tr>'+''.join('<td>'+html.escape(f'{r[k]*100:.2f}%' if k in (*metrics,'source_recall_macro','source_recall_micro','source_hit_rate') else str(r[k]))+'</td>' for k,_ in cols)+'</tr>')
 parts.append('</table>')
parts.append('<h2>口径与核验</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in notes)+'</ul>')
parts.append('<p><a href="overall.csv">总表 CSV</a> · <a href="by_type_and_domain.csv">按题型与场景 CSV</a> · <a href="audit.json">核验与来源哈希</a></p>')
(OUT/'report.html').write_text(''.join(parts))
for r in rows:print(r['method'],r['correct'],*[f'{r[k]*100:.2f}' for k in ['em','f1','bleu_1','source_recall_macro']])
print('Audit passed:',len(rows),'configurations,',len(rows)*1711,'records')
