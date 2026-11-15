"""Collect audited full Gallery A-Mem runs; optionally await their user services."""
import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/mem_gallery_amem_backbones_20261104'

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
    while not all((OUTPUT/b/'audit.json').exists() for b in ('gpt5','deepseek')):
        if not args.wait:raise RuntimeError('Both experiments must complete and pass audit first')
        for b in ('gpt5','deepseek'):
            if (OUTPUT/b/'audit.json').exists():continue
            status=subprocess.run(['systemctl','--user','is-active',f'memgallery-amem-{b}-20261104.service'],capture_output=True,text=True).stdout.strip()
            if status not in ('active','activating'):raise RuntimeError(f'{b} stopped before final audit; inspect its run.log')
        time.sleep(30)
    data=[]
    for b in ('gpt5','deepseek'):
        audit=json.loads((OUTPUT/b/'audit.json').read_text());assert audit['status']=='passed'
        d=json.loads((OUTPUT/b/'final_results.json').read_text());assert d['status']=='complete' and d['overall']['questions']==1711
        recovery_path=OUTPUT/b/'budget_recovery_audit.json'
        recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        if (OUTPUT/b/'budget_recovery_amendment.json').exists():
            assert recovery and recovery['status']=='passed'
        data.append(dict(model=d['model'],**d['overall'],memory_calls=d['memory_calls'],structured_format_validation_failures=d['structured_format_validation_failures'],budget_recovered_calls=recovery['recovered_calls'] if recovery else 0))
    with (OUTPUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (OUTPUT/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    cols=['model','questions','correct','em','f1','bleu_1','bleu_2','bleu','source_recall_macro','source_recall_micro','provider_rejections']
    percent=set(cols[3:10])
    text='<!doctype html><meta charset="utf-8"><title>Mem-Gallery A-Mem 模型对比</title><style>body{font:16px system-ui;margin:36px;color:#172b40}table{border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:right}</style><h1>Mem-Gallery · A-Mem · GPT-5 / DeepSeek</h1><p>1711题；3962轮原始对话；主检索K5，关联扩展去重后最终5条完整记忆；seed42；MiniLM。</p><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'
    for row in data:text+='<tr>'+''.join('<td>'+html.escape(f'{row[k]*100:.2f}%' if k in percent else str(row[k]))+'</td>' for k in cols)+'</tr>'
    text+='</table><p>bleu 为 BLEU-4。证据召回率只统计有证据标注的题。两个模型各自使用其历史caption；DeepSeek保留此前2张图片的caption拒绝记录。LLM-Judge未运行。模型、caption及API策略的变化均包含在对比中，不能视为只更换最终回答模型的消融。</p><p>记忆输出默认上限2048 token；启用恢复的运行仅在输出截断时以相同输入增加到4096/8192/16384，上限调整和实际恢复次数另行记录在 budget_recovery_amendment.json、budget_recovery_audit.json 及 CSV 中。</p><p><a href="comparison.csv">CSV</a> · <a href="gpt5/audit.json">GPT-5核验</a> · <a href="deepseek/audit.json">DeepSeek核验</a></p>'
    (OUTPUT/'report.html').write_text(text)
    print(json.dumps(data,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
