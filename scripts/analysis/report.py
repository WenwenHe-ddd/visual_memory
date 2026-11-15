"""Export editable vector figures, a Chinese HTML report, and audit examples."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/natural_observation'
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def number(x,d=3):return '—' if x is None else f'{x:.{d}f}'
def esc(x):return html.escape(str(x))
def save(fig,name):
    for ext in ('png','svg','pdf'):fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def scatter(ax,rows,load,title,colors):
    for task,color in colors.items():
        subset=[r for r in rows if r['task']==task and r[load] is not None and r['competitor_count']>0]
        ax.scatter([r[load] for r in subset],[r['mean_gold_rank_fraction'] for r in subset],s=18,color=color,alpha=.58,label=task.replace('_',' '))
    ax.set(title=title,xlabel='Mean other memories sharing a query cue',ylabel='Mean gold rank fraction (higher = worse)',ylim=(-.03,1.03))
    ax.grid(alpha=.15)

def main():
    s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'data_audit.json').read_text())
    config=json.loads((OUT/'protocol.json').read_text())
    details=[json.loads(x) for x in (OUT/'retrieval_details.jsonl').read_text().splitlines()]
    rows=[d['metrics'] for d in details];groups={r['group']:r for r in s['groups']}
    tasks=sorted(set(r['task'] for r in rows));colors={t:plt.cm.tab10(i) for i,t in enumerate(tasks)}
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    scatter(axes[0],rows,'all_cue_load','All primary query cues',colors)
    scatter(axes[1],rows,'supported_cue_load','Cues present in at least one gold caption',colors)
    for ax,key in zip(axes,['all_cue_load','supported_cue_load']):
        a=s['associations'][key]
        ax.text(.03,.98,f"Spearman = {number(a['spearman'])}\nAdjusted = {number(a['partial_rank_correlation'])}",transform=ax.transAxes,va='top',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.15),ncol=2,fontsize=8)
    fig.suptitle('MemEye natural observation: cue sharing and caption retrieval')
    fig.tight_layout();save(fig,'overall_association')

    fig,axes=plt.subplots(3,4,figsize=(15,10),sharey=True)
    for yi,y in enumerate(['Y1','Y2','Y3']):
        for xi,x in enumerate(['X1','X2','X3','X4']):
            subset=[r for r in rows if r['x']==x and r['y']==y];g=groups[x+'/'+y]
            title=f"{x} / {y}  n={g['n']}  rho={number(g['rho_all_load_rank'],2)}"
            if g['n']<10:title+=' *'
            scatter(axes[yi,xi],subset,'all_cue_load',title,colors)
            if xi:axes[yi,xi].set_ylabel('')
            if yi<2:axes[yi,xi].set_xlabel('')
    fig.suptitle('X = visual granularity; Y = memory reasoning\n* Fewer than 10 questions: exploratory; colors denote task',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'xy_associations')

    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,field,title,cmap,limits in zip(axes,['n','mean_recall_at_5','rho_all_load_rank'],
             ['Question count','Annotated-evidence Recall@5','Sharing vs rank: Spearman'],['Blues','Blues','coolwarm'],[(0,max(g['n'] for k,g in groups.items() if '/' in k)),(0,1),(-1,1)]):
        a=np.array([[groups[x+'/'+y][field] if groups[x+'/'+y][field] is not None else np.nan for x in ['X1','X2','X3','X4']] for y in ['Y1','Y2','Y3']])
        im=ax.imshow(a,cmap=cmap,vmin=limits[0],vmax=limits[1])
        for iy in range(3):
            for ix in range(4):
                label=str(int(a[iy,ix])) if field=='n' else f'{a[iy,ix]:.2f}'
                if field=='rho_all_load_rank':
                    n=groups[f'X{ix+1}/Y{iy+1}']['n_correlation_eligible']
                    label+=f'\n(n={n})'+('*' if n<10 else '')
                color='white' if field!='rho_all_load_rank' and a[iy,ix]>limits[1]*.65 else 'black'
                ax.text(ix,iy,label,ha='center',va='center',fontsize=10,color=color)
        ax.set(xticks=range(4),xticklabels=['X1','X2','X3','X4'],yticks=range(3),yticklabels=['Y1','Y2','Y3'],title=title)
        fig.colorbar(im,ax=ax,shrink=.75)
    fig.text(.5,-.02,'Positive correlation = more cue sharing, worse rank.  * Fewer than 10 observations.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'xy_heatmaps')

    allg=groups['all'];a=s['associations']['all_cue_load'];b=s['associations']['supported_cue_load'];qa=s['audit']
    def ci(v):return '—' if v is None else f"[{v['low']:.3f}, {v['high']:.3f}]"
    assoc_rows=''.join(f"<tr><td>{name} (n={v['n']})</td><td>{number(v['spearman'])}</td><td>{ci(v['cluster_bootstrap_95ci'])}</td><td>{number(v['partial_rank_correlation'])}</td><td>{ci(v['partial_cluster_bootstrap_95ci'])}</td></tr>" for name,v in [('所有主要查询线索',a),('目标 caption 中出现的主要查询线索',b)])
    group_rows=''.join(f"<tr><td>{esc(g['group'])}</td><td>{g['n']}</td><td>{g['n_with_primary_cue']}</td><td>{number(g['mean_recall_at_5'])}</td><td>{number(g['rho_all_load_rank'])}</td><td>{number(g['rho_supported_load_rank'])}</td></tr>" for g in s['groups'])
    # Seeded, X/Y-stratified examples; select before looking at retrieval outcomes.
    rng=np.random.default_rng(42);examples=[]
    for x in ['X1','X2','X3','X4']:
        for y in ['Y1','Y2','Y3']:
            ds=sorted([d for d in details if d['x']==x and d['y']==y],key=lambda d:d['id'])
            for i in sorted(rng.choice(len(ds),size=min(2,len(ds)),replace=False)):
                examples.append(ds[i])
    (OUT/'audit_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    example_html=''.join('<details><summary>'+esc(d['id'])+' · '+d['x']+'/'+d['y']+'</summary><p>'+esc(d['question'])+'</p><p><b>提取线索：</b>'+esc(', '.join(c['category']+': '+c['cue']+' (其他记忆 '+str(c['other_matches'])+')' for c in d['cues']))+'</p>'+''.join('<p><b>证据 '+esc(g['id'])+'，排名 '+str(g['midrank'])+'</b><br>'+esc(g['caption'])+'</p>' for g in d['gold'])+'</details>' for d in examples)
    if a['partial_rank_correlation'] is not None and a['partial_rank_correlation']>0:
        finding='调整任务、X/Y、证据数、线索数、目标 caption 长度及线索覆盖率后，相关方向仍为共享较多对应排名较差。需结合区间和分组一致性判断证据强弱。'
    else:
        finding='调整任务、X/Y、证据数、线索数、目标 caption 长度及线索覆盖率后，没有得到共享较多对应排名较差的正相关方向。当前结果不足以支持所提出的整体趋势。'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MemEye 线索共享自然观察</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#243044}}h1,h2{{color:#15283e}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.note{{background:#eef3f8;padding:16px;border-radius:8px}}details{{border:1px solid #ddd;padding:12px;margin:10px 0}}code{{word-break:break-all}}</style>
<h1>MemEye：共享检索线索与正确证据排名的自然观察（探索性首轮）</h1>
<p>本地 Qwen3-VL-8B-Instruct 普通图像 caption + BGE-small-en-v1.5。共 {len(rows)} 道原始开放题、438 个含图历史轮次；按 X1–X4、Y1–Y3 和 12 个 X×Y 格子分析。</p>
<p class="note">整体未调整相关较弱（ρ={a['spearman']:.3f}），不同 X×Y 分组方向不一致。{finding} 这是单一 caption 模型与检索器上的自然相关性观察；尚不能据此建立因果机制，也没有评测最终回答质量。自动线索抽取存在下述质量问题，当前统计应视为探索性结果。</p>
<h2>整体结果</h2><p>标注证据 Recall@5：{allg['mean_recall_at_5']:.3f}；Recall@10：{allg['mean_recall_at_10']:.3f}。{allg['n_with_primary_cue']} 道题有可统计的场景／人物／物体线索，{allg['n_with_supported_cue']} 道题至少有一个此类线索出现在目标 caption 中。</p>
<table><tr><th>统计线索范围</th><th>Spearman ρ</th><th>按任务重采样 95% 区间</th><th>调整后秩相关</th><th>调整后区间</th></tr>{assoc_rows}</table>
<p>正相关代表共享较多时正确证据排名较差；负相关代表共享较多时排名较好。仅有 8 个任务历史，聚类重采样区间及调整分析均需谨慎解释，不报告独立样本假设下的显著性。</p>
<img src="figures/overall_association.png"><h2>X×Y 结果</h2><img src="figures/xy_heatmaps.png"><img src="figures/xy_associations.png">
<h2>实验定义</h2><p>使用固定版本的 8 个原始任务开放题，不重复计入选择题、旋转选项或拼接任务。每个含图轮次为一条记忆，检索文本仅由新生成的普通 caption 组成。问题仅检索自己所属任务的完整历史。数据没有提问时间字段；QA 的 session_id 不作为时间截止点。</p>
<p>8 道题附带查询图片，将同一普通 caption 流程生成的描述附加至问题；另报告纯文本查询子集。所有 caption 只看到图片。提取查询线索的模型只看到问题及其附图描述，不看到答案、目标证据或 X/Y 标签。</p>
<p>主要线索为场景、人物、物体。对每个线索统计其他记忆中的匹配数量，排除该题全部标注证据；每道题对这些计数取均值。采用独立于 BGE 相似度的规范化词面短语匹配，包含固定的词形变化和少量同义词映射。该指标是可审计的词面线索共享代理，不能覆盖所有语义等价或隐含线索。</p>
<p>“目标支持的线索”只要求至少一条目标 caption 中含该线索；没有这种线索时记录为缺失，不当作零负载。线索覆盖率不等于答题细节保留率。对每条标注证据记录排名，主相关指标为平均相对排名 (rank−1)/(库大小−1)。多证据题的 Recall@k 是找回的标注证据比例；全部命中也不意味着已完成关系关联或演化综合。</p>
<p>有 {qa['questions_without_competitors']} 道题将其任务内全部记忆列为证据，没有非目标记忆可用于定义竞争；这些题保留在检索指标中，未纳入共享数量相关分析。主相关分析 n={a['n']}，目标支持线索分析 n={b['n']}。</p>
<p>相同得分（余弦分数保留六位小数）使用平均排名；Recall@k 取并列排序的期望，避免相同图片因文件顺序产生虚假差异。</p>
<h2>已发现的线索质量问题</h2><p>模型助理检查分层抽样后发现：Social Chat Q39 把 work-related、personal 等修饰词标成物体；Health Q62 把 upload 标成场景并漏掉 snack；一些题提取 yellow cards，另一些题拆成 cards 与 yellow。复合短语在 caption 中的词面命中率可能比物体头词低，因此会影响共享计数与目标线索覆盖率。当前保留原始自动输出和统计，未根据相关系数方向修改线索。这不属于独立人工验证，后续应先统一线索粒度，再检验结果能否保持。</p>
<h2>分组数值</h2><table><tr><th>分组</th><th>题数</th><th>含主要线索题数</th><th>Recall@5</th><th>全部线索 ρ</th><th>目标支持线索 ρ</th></tr>{group_rows}</table>
<h2>运行与数据检查</h2><ul><li>生成 {qa['captions']} 份按图像内容哈希去重的 caption；处理 {qa['queries']} 道题。</li><li>caption 超长截断：{qa['caption_token_limit_hits']}；线索解析错误：{qa['cue_parse_errors']}；线索超长截断：{qa['cue_token_limit_hits']}。</li><li>拒绝不符合原文片段或格式的线索：{qa['rejected_nonliteral_or_schema_cues']}；无主要线索题：{qa['questions_without_primary_cues']}。</li><li>目标存在并列排名的题：{qa['questions_with_tied_gold_ranks']}；存在相同图像内容替代证据的题：{qa['questions_with_duplicate_image_gold_alternatives']}。</li><li>BGE 输入最长：记忆 {qa['memory_max_bge_tokens']} tokens，问题 {qa['query_max_bge_tokens']} tokens；上限 {qa['bge_max_seq_length']}，未截断。</li><li>下载版本 X/Y 数量与论文 v1 表格不同，统计以 protocol.json 固定版本的实际 JSON 标签为准。</li><li>线索由模型自动提取，附 24 个按 X×Y 分层抽取的样本供检查；尚未进行独立人工标注验证。</li></ul>
<h2>分层抽查样本</h2>{example_html}
<h2>可复核文件</h2><p><a href="per_question.csv">逐题结果</a> · <a href="per_cue.csv">逐线索计数及匹配记忆 ID</a> · <a href="group_summary.csv">分组统计</a> · <a href="protocol.json">完整协议、提示词与版本</a> · <a href="summary.json">统计结果与审计</a> · <a href="retrieval_details.jsonl">完整排序与证据 caption</a></p>
<p>图形同时提供 PNG、可编辑 SVG 和 PDF，位于 figures/。数据版本：<code>{config['provenance']['dataset']['revision']}</code>；模型版本：<code>{config['provenance']['caption_model']['revision']}</code>。</p></html>'''
    (OUT/'report.html').write_text(doc)
    print('Wrote',OUT/'report.html')

if __name__=='__main__':main()
