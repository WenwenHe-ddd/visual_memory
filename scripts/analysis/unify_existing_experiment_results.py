#!/usr/bin/env python3
"""Build a numeric snapshot without changing running experiments or their reports."""
import json, hashlib, math, statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT=Path(__file__).resolve().parents[2]
GRID=ROOT/'results/required_grid_seed42_1024_2026_20261113'
OUT=ROOT/'reports/unified_existing_results_20261114'
OUT.mkdir(exist_ok=True)
SOURCES={}
def read(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 b=p.read_bytes();SOURCES[str(p)]=hashlib.sha256(b).hexdigest()
 return json.loads(b)
def canon(s):
 s=s.lower().replace('＋','+').replace(' ','')
 if s in ['semantic_rag','semrag'] or 'semanticrag' in s:return 'semrag'
 if 'ours1_ours2' in s or 'ours_1_ours_2' in s or ('ours_1' in s and ('ours_2' in s or '+' in s)):return 'ours_1_ours_2'
 if 'ours_1' in s:return 'ours_1'
 if 'ours_2' in s or s=='ours2_zscore':return 'ours_2'
 if 'generative' in s or s=='gen_agents':return 'gen_agents'
 if 'memoryos' in s:return 'memoryos'
 if 'simplemem' in s:return 'simplemem'
 if 'a-mem' in s or s=='a_mem':return 'a_mem'
 if 'hippo' in s:return 'hippomm'
 if 'clip' in s or 'visrag' in s:return 'clip_image_dialogue'
 raise ValueError(s)
def model(s):
 if 'qwen' in s.lower():return 'Qwen3-VL-8B'
 if 'deepseek' in s.lower():return 'DeepSeek Flash'
 if s.lower() in ['gpt5','gpt-5']:return 'GPT-5'
 raise ValueError(s)
INV=read('reports/experiment_inventory_20261113/inventory.json')
rows={}
for x in INV['main_method_records']:
 r=dict(x);r['sources']=list(r.pop('source_paths'));r['metric_sources']={};r['judge_seed']=42
 rows[(r['dataset'],r['backbone'],r['seed'],r['method_id'])]=r

def put(key,vals,src):
 if key not in rows:return
 r=rows[key]
 for k,v in vals.items():
  if v is not None:r[k]=v;r['metric_sources'][k]=str((ROOT/src).resolve())
 if str((ROOT/src).resolve()) not in r['sources']:r['sources'].append(str((ROOT/src).resolve()))

# These paired audits identify the exact original MCQ protocols and seed-specific storage.
for bb,src in [('Qwen3-VL-8B','results/memeye_qwen_seed1024_full_20261105/comparison.json'),('DeepSeek Flash','results/memeye_deepseek_seed1024_full_20261106/comparison.json')]:
 for x in read(src)['methods']:
  mid=canon(x.get('id',x.get('method')))
  for seed in [42,1024]:
   pre=f'seed{seed}_';rec=x.get(pre+'recall_at5_micro',x.get(pre+'recall'))
   vals={'correct':x[pre+'correct'],'accuracy':x[pre+'accuracy'],'llm_judge':x.get(pre+'llm_judge',x.get(pre+'judge')),'storage_mib':x.get(pre+'storage_mib')}
   if mid=='hippomm':vals['context_source_recall']=x.get(pre+'context_source_micro_recall',rec)
   else:vals['recall_at5_micro']=rec
   put(('MemEye',bb,seed,mid),vals,src)
src='results/memeye_deepseek_seed2026_full_20261113/comparison.json'
for x in read(src)['methods']:
 put(('MemEye','DeepSeek Flash',2026,canon(x['method'])),{'correct':x['seed2026_correct'],'accuracy':x['seed2026_accuracy'],'llm_judge':x['seed2026_llm_judge']},src)
# The formal GPT-5 protocol is 217/371 and 222/371; the storage report's old
# 219/371 caption variant must not overwrite this benchmark identification.
src='results/memeye_gpt5_open_20261105/methods_manifest.json'
for x in read(src)['methods']:
 c=x['source_mcq_correct_for_identification_only'];put(('MemEye','GPT-5',42,x['id']),{'correct':c,'accuracy':c/371},src)
for bb,tag in [('Qwen3-VL-8B','qwen'),('GPT-5','gpt5'),('DeepSeek Flash','deepseek')]:
 src=f'reports/memeye_{tag}_open_llm_judge_'+('20261106' if tag=='deepseek' else '20261105')+'/audit.json'
 for x in read(src)['methods']:put(('MemEye',bb,42,canon(x['id'])),{'llm_judge':x['llm_judge']},src)

storage_rows=[]
for dataset,src,bb in [
 ('MemEye','reports/memory_storage_20261101/storage_summary.json','Qwen3-VL-8B'),
 ('MemEye','reports/memory_storage_gpt5_memeye_20261102/storage_summary.json','GPT-5'),
 ('MemEye','reports/memory_storage_deepseek_memeye_20261102/storage_summary.json','DeepSeek Flash'),
 ('MemGallery','reports/memgallery_storage_20261103/storage_summary.json',None)]:
 for x in read(src)['rows']:
  b=bb or model(x['model']);m=canon(x['method']);key=(dataset,b,42,m)
  storage_rows.append([dataset,b,42,x['method'],x.get('storage_bytes'),x.get('storage_mib'),x.get('correct'),x.get('accuracy',x.get('em')),x.get('note'),str(ROOT/src)])
  vals={}
  # Use storage in the main table only when its recorded QA outcome matches.
  if dataset=='MemEye':
   if key in rows and rows[key].get('correct')==x.get('correct'):
    vals['storage_mib']=x.get('storage_mib')
    if m!='hippomm':vals['recall_at5_micro']=x.get('recall_at5_micro')
   elif key in rows:
    rows[key]['note']+='；历史Storage对应不同QA版本，未并入主列，见Storage原始统计'
  else:vals={k:x.get(k) for k in ['em','f1','bleu_1','storage_mib']}
  put(key,vals,src)

# MemGallery Qwen paired audit provides full precision and source-linked seed42/1042 metrics.
src='reports/memgallery_qwen_seed1042_stability_20261106/summary.json'
for x in read(src)['methods']:
 for seed in [42,1042]:
  vals=dict(x[f'seed{seed}']);vals['correct']=x[f'correct{seed}'];put(('MemGallery','Qwen3-VL-8B',seed,x['id']),vals,src)
for bb,src in [
 ('Qwen3-VL-8B','results/memgallery_qwen_all_methods_deepseek_judge_20261104/summary.json'),
 ('GPT-5','results/memgallery_gpt5_deepseek_judge_20261104/summary.json'),
 ('DeepSeek Flash','results/memgallery_deepseek_flash_llm_judge_20261105/summary.json')]:
 for x in read(src)['methods']:
  vals={k:x.get(k) for k in ['em','f1','bleu_1','llm_judge']};vals['correct']=x.get('source_em_correct',x.get('expected_em_correct'))
  key=('MemGallery',bb,42,canon(x['id']));put(key,vals,src)
  if bb=='DeepSeek Flash' and canon(x['id'])=='a_mem' and key in rows:
   rows[key]['note']+='；复合恢复来源，包含一次获准的memory evolution提示修改；original_prompt_baseline=false'
src='reports/memgallery_prefix5_recall_20261103/summary.json'
for x in read(src)['results']:
 key=('MemGallery',model(x['model']),42,canon(x['method']))
 put(key,{'recall_at5_micro':x['recall_at5_micro'],'recall_note':'原检索顺序前5条；回答仍为原完整上下文'},src)

# Add only newly completed jobs with their method-level completion proof.
plan=read(GRID/'plan.json');progress=read(GRID/'progress.json')
for job in plan['jobs']:
 state=progress['jobs'][job['id']]['status']
 if state!='complete':continue
 src=Path(job['method_root'])/'completion.json';d=read(src)
 assert d['status']=='complete' and d['questions']==job['questions']
 for family in ['source_sha256','artifact_sha256']:
  for p,h in d.get(family,{}).items():
   assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, ('changed completion source',p)
 key=('MemEye' if job['dataset']=='memeye' else 'MemGallery',model(job['backbone']),job['seed'],job['method'])
 assert key not in rows
 labels={r['method_id']:r['method'] for r in rows.values()}
 rows[key]={'dataset':key[0],'backbone':key[1],'seed':key[2],'method_id':key[3],'method':labels[key[3]],'questions':d['questions'],'qa_status':'complete','judge_status':'complete','judge_seed':42,'note':'新三种子网格；完成标记及来源SHA256已核验','sources':[str(src)],'metric_sources':{}}
 put(key,{k:d.get(k) for k in ['correct','accuracy','em','f1','bleu_1','llm_judge']},src)
 if 'em_correct' in d:put(key,{'correct':d['em_correct']},src)

for key,r in rows.items():
 if r['dataset']=='MemGallery' and r.get('correct') is None:r['correct']=round(r['em']*r['questions'])
 if r['method_id']=='simplemem':r['recall_note']='压缩记忆按来源映射；保守归因，非原生ID检索'
 if r['method_id']=='hippomm':r['recall_note']='context_source_recall为实际上下文来源覆盖率，不能当作Recall@5'
 r.setdefault('context',{'gen_agents':'Top10','memoryos':'原生动态上下文','hippomm':'原生多路动态上下文','simplemem':'最终最多5条（保留原生空上下文）','a_mem':'最终5条'}.get(r['method_id'],'Top5'))
 assert isinstance(r.get('correct'),int),('missing correct',key)
 metric='accuracy' if r['dataset']=='MemEye' else 'em'
 assert abs(r[metric]-r['correct']/r['questions'])<1e-12,(key,'numerator mismatch')
 if r['judge_status']=='complete':assert 0<=r['llm_judge']<=1,(key,'missing judge')
 for v in ['em','accuracy','f1','bleu_1','llm_judge','recall_at5_micro','context_source_recall']:
  if r.get(v) is not None:assert math.isfinite(r[v]) and 0<=r[v]<=1,(key,v)

order={m:i for i,m in enumerate(['semrag','ours_1','ours_2','ours_1_ours_2','gen_agents','memoryos','simplemem','a_mem','clip_image_dialogue','hippomm'])}
records=sorted(rows.values(),key=lambda r:(r['dataset'],['Qwen3-VL-8B','GPT-5','DeepSeek Flash'].index(r['backbone']),r['seed'],order[r['method_id']]))
wb=Workbook();wb.remove(wb.active)
def sheet(name,headers,data,percent=(),widths=None):
 ws=wb.create_sheet(name);ws.append(headers)
 for row in data:ws.append(row)
 ws.freeze_panes='E2';ws.auto_filter.ref=ws.dimensions
 for c in ws[1]:c.fill=PatternFill('solid',fgColor='203864');c.font=Font(name='Calibri',color='FFFFFF',bold=True);c.alignment=Alignment(wrap_text=True,vertical='center')
 ws.row_dimensions[1].height=32
 for i,h in enumerate(headers,1):
  ws.column_dimensions[get_column_letter(i)].width=(widths or {}).get(h,18 if i>4 else 23)
 for i,row in enumerate(ws.iter_rows(min_row=2),2):
  for c in row:
   c.font=Font(name='Calibri',size=11);c.alignment=Alignment(vertical='top',wrap_text=True)
   if i%2==0:c.fill=PatternFill('solid',fgColor='EDF2F8')
   if c.column in percent and isinstance(c.value,(int,float)):c.number_format='0.00%'
   elif isinstance(c.value,float):c.number_format='0.000000'
   if c.value in ['pending','running','failed']:c.font=Font(name='Calibri',color={'pending':'996600','running':'0070C0','failed':'C00000'}[c.value],bold=True)
  ws.row_dimensions[i].height=32
 ws.sheet_view.zoomScale=85
 ws.print_options.horizontalCentered=True
 return ws
stamp=datetime.now(timezone.utc).isoformat()
notes=[
 ['快照时间UTC',stamp],['范围','MemEye / MemGallery；Qwen3-VL-8B、GPT-5、DeepSeek Flash；包含42/1024/2026及历史1042，按实际已有记录列出'],
 ['已有QA结果组数',len(records)],['QA及Judge均完整组数',sum(r['judge_status']=='complete' for r in records)],
 ['MemEye','371道选择题Accuracy；LLM Judge来自另行生成的开放式回答，不评估选项字母'],
 ['MemGallery','1711道开放式题；EM、F1、BLEU-1、LLM Judge'],
 ['显示口径','EM、Accuracy、Recall以百分比显示；F1、BLEU-1、Judge为0至1；原始精度保存在数值单元格中'],
 ['Judge','DeepSeek实际API标识deepseek-v4.1-flash；固定seed42；五档0/.25/.5/.75/1'],
 ['缺失值','空白表示该指标没有可直接对齐的已核验结果，不能当作0；进行中/失败不进入已完成表'],
 ['Seed','1024与1042分开；稳定性表仅统计实际已完成seed，不代表三种子网格已经跑完'],
 ['Graph超参数','ours_2及ours_1＋ours_2主表均λ=0.03、α=1；正向方差等其他条件在历史扩展页分开'],
 ['Storage','MiB=bytes/1048576；历史持久化记忆库文件，非模型权重、显存或单次上下文；各方法具体计数边界见原始统计备注'],
 ['Storage对齐','MemEye GPT旧Storage caption结果与正式217/222协议不同，主表对应单元格留空，原始Storage数值保留在独立页'],
 ['Recall','GA/MemoryOS的前5条Recall不能把其完整上下文回答结果称为Top5回答；HippoMM上下文来源覆盖率单列'],
 ['历史扩展','额外backbone与消融保留独立页；数字字段以原始JSON键路径保留，避免误并为正式协议'],
 ['网格状态','待完成网格是当前快照；实验继续运行，本文件不会自动更新']]
sheet('说明与口径',['项目','内容'],notes,widths={'项目':28,'内容':115})
headers=['Dataset','Backbone','Seed','Method','代码命名','题数','正确数','Accuracy / EM','F1','BLEU-1','LLM Judge','Recall@5 micro','上下文来源覆盖率','Storage MiB','回答上下文','QA状态','Judge状态','Recall口径','备注']
def values(r):return [r['dataset'],r['backbone'],r['seed'],r['method'],r['method_id'],r['questions'],r['correct'],r.get('accuracy',r.get('em')),r.get('f1'),r.get('bleu_1'),r.get('llm_judge'),r.get('recall_at5_micro'),r.get('context_source_recall'),r.get('storage_mib'),r['context'],r['qa_status'],r['judge_status'],r.get('recall_note'),r.get('note')]
for name,rs in [('全部已有结果',records),('MemEye结果',[r for r in records if r['dataset']=='MemEye']),('MemGallery结果',[r for r in records if r['dataset']=='MemGallery'])]:
 sheet(name,headers,[values(r) for r in rs],percent=[8,12,13],widths={'Method':46,'代码命名':26,'回答上下文':33,'Recall口径':55,'备注':68})
for dataset in ['MemEye','MemGallery']:
 grouped=defaultdict(list)
 for r in records:
  if r['dataset']==dataset:grouped[(r['backbone'],r['method_id'])].append(r)
 data=[]
 for (bb,mid),rs in grouped.items():
  lookup={r['seed']:r for r in rs};v=lambda s,k:lookup.get(s,{}).get(k)
  metric='accuracy' if dataset=='MemEye' else 'em';ss=[r[metric] for r in rs if r['seed'] in [42,1024,2026]]
  data.append([bb,rs[0]['method'],mid,*[v(s,metric) for s in [42,1024,2026,1042]],*[v(s,'llm_judge') for s in [42,1024,2026,1042]],len(ss),statistics.mean(ss),statistics.stdev(ss) if len(ss)>1 else None,','.join(str(s) for s in [42,1024,2026] if s in lookup)])
 sheet(dataset+'跨Seed',['Backbone','Method','代码命名','42 ACC/EM','1024 ACC/EM','2026 ACC/EM','历史1042 ACC/EM','42 Judge','1024 Judge','2026 Judge','历史1042 Judge','主三Seed已完成数','已有主Seed均值','已有主Seed样本SD','参与统计Seeds'],data,percent=[4,5,6,7,13,14],widths={'Method':46,'代码命名':26})
statuses=[]
for j in plan['jobs']:
 s=progress['jobs'][j['id']];statuses.append([j['dataset'],j['backbone'],j['seed'],j['method'],s['status'],j['questions'],j['method_root'],s.get('log'),s.get('exit_code')])
sheet('144组网格状态',['Dataset','Backbone','Seed','Method','当前状态','题数','结果目录','运行日志','退出码'],statuses,widths={'结果目录':90,'运行日志':90})
sheet('Storage原始统计',['Dataset','Backbone','Seed','Method','Bytes','MiB','来源QA正确数','来源QA ACC或EM','计数边界备注','来源文件'],storage_rows,percent=[8],widths={'Method':45,'计数边界备注':100,'来源文件':100})
evidence=[]
for r in records:
 for metric,src in r['metric_sources'].items():evidence.append([r['dataset'],r['backbone'],r['seed'],r['method_id'],metric,r.get(metric),src])
sheet('逐指标出处',['Dataset','Backbone','Seed','Method','指标','原始数值','来源文件'],evidence,widths={'来源文件':120})
# Preserve older studies separately. Each scalar is keyed by its exact JSON path.
extra_meta=[];extra_numeric=[]
for kind,entries in [('额外Backbone',INV['extra_backbones']),('历史方法及消融',INV['extra_methods_and_ablations'])]:
 for x in entries:
  src=x[-1];data=read(src)
  if kind=='额外Backbone':extra_meta.append([kind,x[0],x[1],x[2],x[3],x[4],x[5],x[6],str(ROOT/src)])
  else:extra_meta.append([kind,x[0],x[1],None,x[2],x[3],x[4],x[5],str(ROOT/src)])
  def walk(v,path='$'):
   if isinstance(v,dict):
    for k,z in v.items():
     if k not in ['source_sha256','artifact_sha256','output_sha256','input_hashes']:walk(z,path+'.'+k)
   elif isinstance(v,list):
    for i,z in enumerate(v):walk(z,f'{path}[{i}]')
   elif isinstance(v,(int,float)) and not isinstance(v,bool):extra_numeric.append([kind,x[0],str(x[1]),str(x[-2]),path,v,str(ROOT/src)])
  walk(data)
# Metadata entries have different column layouts; store original record as JSON too.
sheet('历史扩展索引',['类别','Dataset','Backbone','实际模型标识','Seed','方法或实验','状态','协议或备注','来源文件'],[r+['']*(9-len(r)) for r in extra_meta],widths={'协议或备注':90,'来源文件':100,'方法或实验':45})
sheet('历史扩展原始数值',['类别','Dataset','Backbone','实验备注','原始JSON字段路径','数值','来源文件'],extra_numeric,widths={'实验备注':65,'原始JSON字段路径':90,'来源文件':100})
sheet('来源文件SHA256',['来源文件','SHA256'],sorted(SOURCES.items()),widths={'来源文件':130,'SHA256':70})
path=OUT/'VisualMemory_已有实验结果统一汇总_20261114.xlsx'
wb.save(path)
# Reopen the actual workbook to ensure every main result is numeric and no row was lost.
check=load_workbook(path,data_only=True);assert check['全部已有结果'].max_row==len(records)+1
for a,b in zip(check['全部已有结果'].iter_rows(min_row=2,values_only=True),records):
 assert a[6]==b['correct'] and abs(a[7]-b['correct']/b['questions'])<1e-12
assert not any(c.data_type=='f' for ws in check for row in ws for c in row)
report={'created_at_utc':stamp,'qa_result_count':len(records),'judge_complete_count':sum(r['judge_status']=='complete' for r in records),'grid_status_counts':dict(Counter(s['status'] for s in progress['jobs'].values())),'cohort_counts':dict(Counter(f"{r['dataset']} / {r['backbone']} / {r['seed']}" for r in records)),'xlsx':str(path),'xlsx_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':SOURCES,'validation':'all primary QA numerators/denominators consistent; completed new artifact hashes verified; reopened workbook values verified'}
(OUT/'snapshot.json').write_text(json.dumps({'report':report,'results':records},ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
