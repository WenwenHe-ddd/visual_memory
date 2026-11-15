"""Publish the live requested experiment coverage without changing source results."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
OUT=c.ROOT/'reports/required_seed_grid_20261113'
NAMES=dict(reused_complete='已有完整结果',complete='补跑已完成',running='正在运行',pending='已排队',failed='失败待处理')

def report():
    from openpyxl import Workbook
    from openpyxl.styles import Font,PatternFill,Alignment
    from openpyxl.utils import get_column_letter
    plan=c.read(c.RUN/'plan.json');state=c.read(c.RUN/'progress.json');rows=[];counts=Counter()
    for job in plan['jobs']:
        status=state['jobs'].get(job['id'],{}).get('status',job['status']);counts[status]+=1
        path=Path(job['method_root'])/'completion.json'
        result=c.read(path) if status=='complete' else {}
        rows.append(dict(dataset=job['dataset'],backbone=job['backbone'],seed=job['seed'],method=job['method'],
            method_label=c.LABELS[job['method']],status=NAMES[status],questions=job['questions'],
            new_mcq_correct=result.get('correct'),new_accuracy=result.get('accuracy'),new_gallery_em=result.get('em'),
            new_gallery_f1=result.get('f1'),new_gallery_bleu1=result.get('bleu_1'),new_llm_judge=result.get('llm_judge'),
            evidence=str(path) if result else ' | '.join(job['existing_source_paths']),
            source_note=job['existing_note'],
            run_root=job['run_root'] if job['status']!='reused_complete' else '',
            log=state['jobs'].get(job['id'],{}).get('log','')))
    OUT.mkdir(parents=True,exist_ok=True)
    temp=OUT/'实验进度.csv.tmp'
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(OUT/'实验进度.csv')
    book=Workbook();matrix=book.active;matrix.title='144组覆盖情况'
    matrix.append(['Dataset','Backbone','Seed',*[c.LABELS[m] for m in c.METHODS],'完成数/8'])
    for ds in plan['dataset']:
        for bb in plan['backbone']:
            for seed in plan['seeds']:
                group={r['method']:r for r in rows if (r['dataset'],r['backbone'],r['seed'])==(ds,bb,seed)}
                matrix.append([ds,bb,seed,*[group[m]['status'] for m in c.METHODS],
                    sum(group[m]['status'] in ('已有完整结果','补跑已完成') for m in c.METHODS)])
    detail=book.create_sheet('逐组进度与出处');detail.append(list(rows[0]))
    for row in rows:detail.append(list(row.values()))
    note=book.create_sheet('统计口径')
    lines=[['更新时间 UTC',c.now()],['目标',144],['保留的已有完整结果',counts['reused_complete']],
        ['补跑已完成',counts['complete']],['正在运行',counts['running']],['排队',counts['pending']],['失败待处理',counts['failed']],
        ['并行限制','全局最多两个完整方法任务；包含记忆构建、检索、回答、Judge和审计。'],
        ['种子','42、1024、2026；旧1042不算1024。'],
        ['caption控制','新实验固定各方法对应backbone的原始缓存caption；重新构建记忆、检索、生成回答。'],
        ['Judge','DeepSeek五档评分，seed固定42。'],
        ['上下文','GA Top10；MemoryOS原生策略；SimpleMem官方最终5条（保留原生空上下文）；A-Mem最终5条；VisRAG原图5张；HippoMM原生动态路径。'],
        ['VisRAG','用户重复列出的VisRAG合并为同一种方法。'],
        ['数字字段','new_开头的指标只填本次补跑且完成审计的结果；已有结果保留原文件出处，空白不表示0。'],
        ['已有结果协议','沿用各结果原始协议与修复记录。MemGallery/DeepSeek/seed42/A-Mem为既有复合recovery来源，含一次已授权提示修订；original_prompt_baseline=false。详细出处见source_note。'],
        ['注意','任务已排队不等于已完成；失败项保留原始诊断，不填补答案或分数。'],
        ['运行目录',str(c.RUN)],['后台服务','visualmemory-required-grid-42-1024-2026-20261113.service']]
    for row in lines:note.append(row)
    colors={'已有完整结果':'D9EAD3','补跑已完成':'B6D7A8','正在运行':'FFF2CC','已排队':'E8EDF3','失败待处理':'F4CCCC'}
    for sheet in book:
        sheet.freeze_panes='D2' if sheet==matrix else 'A2';sheet.auto_filter.ref=sheet.dimensions
        for cell in sheet[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='244062')
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment=Alignment(vertical='top',wrap_text=True)
                if cell.value in colors:cell.fill=PatternFill('solid',fgColor=colors[cell.value])
        for col in range(1,sheet.max_column+1):sheet.column_dimensions[get_column_letter(col)].width=22 if col<4 else 28
    matrix.column_dimensions['L'].width=12;note.column_dimensions['B'].width=115
    for col in ('I','J','K','L','M'):
        for cell in detail[col][1:]:cell.number_format='0.0000%'
    temp=OUT/'VisualMemory_三种子实验进度.tmp.xlsx';book.save(temp);temp.replace(OUT/'VisualMemory_三种子实验进度.xlsx')
    c.dump(OUT/'coverage_summary.json',dict(updated_at=c.now(),total=144,counts=dict(counts),active=state['active'],all_missing_cells_registered=True))
    return counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        with c.lock(OUT/'report.lock'):
            print(dict(report()),flush=True)
        if not a.watch or c.read(c.RUN/'progress.json')['status']!='running':break
        time.sleep(30)
