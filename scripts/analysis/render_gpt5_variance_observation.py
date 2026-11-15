"""Render audited numeric observations with original images and editable CSVs."""
from pathlib import Path
import base64
import csv
import html
import json
import os
import shutil
import sys
import textwrap
os.environ.setdefault('MPLCONFIGDIR','/tmp/visualmemory-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/gpt5_variance_observation_20261108'
SOURCE=ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def h(s):return html.escape(str(s))
def short(mid):return mid.split('/')[-1]


def main():
    summary=read(OUT/'numeric_summary.json');sel=summary['selected_master']
    members=read(OUT/'largest_members.json');mm={m['id']:m for m in members}
    coords=read(OUT/'largest_coordinates.json')
    sp=[r for r in csv.DictReader((OUT/'spectra.csv').open(encoding='utf-8-sig')) if r['domain']==sel['domain'] and r['master_id']==sel['master_id']]
    variance=np.array([float(r['variance']) for r in sp]);retention=np.array([float(r['white_retention']) for r in sp]);rank=len(sp)
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    colors={'high':'#2457A6','middle':'#7195C6','low':'#B7CAE2'}
    fig,axs=plt.subplots(1,2,figsize=(13.8,4.6))
    pcs=np.arange(1,rank+1)
    axs[0].bar(pcs,variance/variance.sum()*100,color=[colors[r['band']] for r in sp]);axs[0].set_xlabel('Principal component (descending variance)');axs[0].set_ylabel('Variance share (×100)')
    axs[1].plot(pcs,retention,color='#2457A6',linewidth=2,label='Whitening')
    axs[1].plot(pcs,2-retention,color='#DB7D31',linewidth=2,label='Positive variance')
    axs[1].set_xlabel('Principal component (descending variance)');axs[1].set_ylabel('Multiplier on score contribution');axs[1].legend(frameon=False)
    for ax in axs:ax.set_xticks([1,10,20,30,40,47]);ax.grid(False)
    fig.tight_layout();save(fig,'largest_spectrum')
    fig,axs=plt.subplots(1,2,figsize=(13.8,4.8))
    for ax,key,hard,label in [(axs[0],'all_pairs','all_original_top5_distractors','All eligible masters'),
                              (axs[1],'largest_master','largest_original_top5_distractors','Largest Health master')]:
        for offset,source,color,legend in [(-.18,key,'#2457A6','All same-master non-targets'),(.18,hard,'#DB7D31','Original Top-5 non-targets')]:
            vals=[summary['metrics'][source][b]['supports_target_fraction_question_macro']*100 for b in ['high','middle','low']]
            bars=ax.bar(np.arange(3)+offset,vals,.34,color=color,label=legend)
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.7,f'{v:.2f}',ha='center',fontsize=11,weight='bold')
        ax.set_xticks(range(3),['High variance\n(first 20%)','Middle variance\n(middle 60%)','Low variance\n(last 20%)'])
        ax.set_ylabel('Target-supporting fraction (×100)');ax.set_ylim(0,95);ax.set_xlabel(label);ax.legend(frameon=False,fontsize=10,loc='upper right')
    fig.tight_layout();save(fig,'band_target_support')
    pairs=lines(OUT/'largest_pairs_with_contributions.jsonl')
    eligible=[r for r in pairs if r['distractor_in_original_top5'] and r['raw_margin']<0<r['white_margin']]
    eligible.sort(key=lambda r:(r['question_id'],r['target_id'],r['distractor_id']))
    example=dict(eligible[0]);qmap={q['id']:q for q in lines(SOURCE/'inputs/source_questions.jsonl')}
    rawmap={r['question_id']:r for r in lines(SOURCE/'retrieval_results.jsonl')}
    q=qmap[example['question_id']];example.update(question=q['question'],options=q['options'],
        master_query=next(g['augmented_query'] for g in rawmap[q['id']]['masters'] if g['master_id']==sel['master_id']),
        selection='Post-hoc illustrative example: lexicographically first original-Top5 same-master non-target pair with raw margin < 0 and white intermediate margin > 0; not used for aggregate selection.')
    dump(OUT/'illustrative_rescue.json',example)
    examples={**read(OUT/'examples.json'),'illustrative_rescue':example}
    for name,row in examples.items():
        c=np.array(row['contributions']);fig,ax=plt.subplots(figsize=(14,4.5))
        ax.bar(pcs,c,color=np.where(c>=0,'#2457A6','#DB7D31'),label='Original PC contribution')
        ax.scatter(pcs,c*retention,color='#20252B',s=15,label='After ridge whitening',zorder=3)
        ax.axhline(0,color='#777777',linewidth=.7);ax.set_xlabel('Principal component (descending variance)')
        ax.set_ylabel('Target − non-target score contribution');ax.set_xticks([1,10,20,30,40,47])
        ax.legend(handles=[Patch(facecolor='#2457A6',label='Supports target'),Patch(facecolor='#DB7D31',label='Supports non-target'),
                           Line2D([0],[0],marker='o',linestyle='',color='#20252B',label='After ridge whitening')],frameon=False)
        fig.tight_layout();save(fig,name+'_contributions')
    assets=OUT/'images';assets.mkdir(exist_ok=True)
    for m in members:
        name=short(m['id']).replace(':','_')+Path(m['image_path']).suffix
        shutil.copy2(m['image_path'],assets/name);m['display_path']='images/'+name
    extremes=[]
    for pc in [1,2,3,45,46,47]:
        sortedrows=sorted(coords,key=lambda r:r['coordinates'][pc-1]);chosen=sortedrows[:2]+sortedrows[-2:]
        fig,axs=plt.subplots(1,4,figsize=(17,4.6))
        for ax,row in zip(axs,chosen):
            m=mm[row['memory_id']];ax.imshow(Image.open(m['image_path']));ax.axis('off')
            ax.set_title(f'{short(m["id"])}\nPC{pc}: {row["coordinates"][pc-1]:+.4f}',fontsize=11)
        fig.tight_layout();fig.savefig(OUT/f'pc{pc:02d}_extremes.png',dpi=180,bbox_inches='tight');plt.close(fig)
        extremes.append({'pc':pc,'variance_share':float(variance[pc-1]/variance.sum()),
            'members':[{'id':r['memory_id'],'coordinate':r['coordinates'][pc-1]} for r in chosen]})
    dump(OUT/'pc_extremes.json',extremes)
    for label,row in examples.items():
        fig,axs=plt.subplots(1,2,figsize=(11,5.7))
        for ax,key,role in [(axs[0],'target_id','Gold target'),(axs[1],'distractor_id','Non-target')]:
            m=mm[row[key]];ax.imshow(Image.open(m['image_path']));ax.axis('off');ax.set_title(role+' — '+short(m['id']),fontsize=14)
        fig.tight_layout();fig.savefig(OUT/(label+'_images.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT-5 MemEye 方差方向观察</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#222}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}details{border:1px solid #ddd;padding:10px;margin:10px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}article{margin:25px 0;padding-bottom:20px;border-bottom:1px solid #ccc}.note{background:#f2f5f9;padding:15px}</style>',
        '<h1>GPT-5 / MemEye：高方差方向在支持谁？</h1>',
        '<p>固定原有 GPT-5 caption、master、查询、BGE 编码设置。λ=0.03，α=1。仅做本地观察，不调用回答模型，不修改记忆库。</p>',
        '<p>预先按簇大小选择：Health / master_000070，48 条记忆、47 个主成分。高方差 PC1–10，中间 PC11–37，低方差 PC38–47。原图与完整对话/caption 均可展开查看。</p>',
        '<p class="note">正贡献表示目标相对同簇非目标更匹配查询；负贡献表示非目标更匹配。主成分来自 BGE 对话＋caption，而非图像像素。“支持目标”不等于已经识别出图像细节语义。</p>',
        '<h2>全量观察与实际 Top-5 干扰项</h2><img src="band_target_support.png">',
        '<table><tr><th>范围</th><th>有效题数 / 配对数</th><th>高方差支持目标</th><th>中方差支持目标</th><th>低方差支持目标</th></tr>']
    for key,label in [('all_pairs','全部簇，所有同簇非目标'),('all_original_top5_distractors','全部簇，原 Top-5 非目标'),
                      ('largest_master','最大 Health 簇，所有非目标'),('largest_original_top5_distractors','最大 Health 簇，原 Top-5 非目标')]:
        m=summary['metrics'][key];values=[label,f"{m['band_eligible_questions']} / {m['band_eligible_pairs']}"]+[f"{m[b]['supports_target_fraction_question_macro']*100:.2f}%" for b in ['high','middle','low']]
        body.append('<tr>'+''.join('<td>'+h(v)+'</td>' for v in values)+'</tr>')
    body+=['</table><p>上表先在每道题内平均配对，再对题目平均。数值不是直接将所有配对混在一起。有效秩小于 5 的 master 不进入三段比较。总体配对共 17,240 个、347 题；分段比较涉及 16,957 个、262 题。原 Top-5 干扰项由原始分数选出，天然属于难例，因此不能将条件样本中的现象推广为所有高方差方向都无效。</p>',
        '<h2>方差与抑制强度</h2><img src="largest_spectrum.png">',
        '<h2>同簇排序：中间白化分数与最终赋权分数</h2><table><tr><th>范围</th><th>配对数</th><th>原始目标领先</th><th>白化中间分数</th><th>PV 中间分数</th><th>zscore 最终分数</th><th>PV 最终分数</th></tr>']
    for key,label in [('all_original_top5_distractors','全部簇，原 Top-5 非目标'),('largest_original_top5_distractors','最大 Health 簇，原 Top-5 非目标')]:
        m=summary['metrics'][key];vals=[label,m['pairs']]+[m[k] for k in ['raw_target_ahead_pairs','white_target_ahead_pairs','pv_target_ahead_pairs','zscore_final_target_ahead_pairs','pv_final_target_ahead_pairs']]
        body.append('<tr>'+''.join('<td>'+h(v)+'</td>' for v in vals)+'</tr>')
    body+=['</table><p>上表为配对计数，不是 QA 正确题数或 Recall。白化中间分数的改善，经过组内标准化、tanh 和原始分数融合后会减弱；不同 master 合并排序的影响不由此表单独解释。</p>',
           '<h2>具体目标与非目标：同时展示帮助与损害</h2>']
    for name,row in examples.items():
        body+=['<article><h3>'+h(name)+'：'+h(row['question_id'])+'</h3><p>'+h(row['question'])+'</p><details><summary>实际 master 查询及分数</summary><pre>'+h(row['master_query'])+'</pre><pre>'+h(json.dumps({k:row[k] for k in ['raw_margin','white_margin','pv_margin','zscore_final_margin','pv_final_margin','high_contribution','middle_contribution','low_contribution']},indent=2))+'</pre></details>',
               '<img src="'+name+'_images.png"><img src="'+name+'_contributions.png">']
        if 'selection' in row:body.append('<p>'+h(row['selection'])+'</p>')
        body.append('<div class="pair">')
        for key in ['target_id','distractor_id']:
            m=mm[row[key]];body.append('<details><summary>'+h(short(m['id']))+'：完整对话＋caption</summary><pre>'+h(m['retrieval_text'])+'</pre></details>')
        body.append('</div></article>')
    body+=['<h2>主成分两端的原图与文本</h2><p>每个方向展示坐标最小两条、最大两条；方向符号固定为绝对值最大载荷为正。极端成员只能辅助解释，不能给主成分赋予唯一语义。</p>']
    for e in extremes:
        body.append('<article><h3>PC'+str(e['pc'])+f"，方差占比 {e['variance_share']*100:.2f}%</h3><img src=\"pc{e['pc']:02d}_extremes.png\">")
        for entry in e['members']:
            m=mm[entry['id']];body.append('<details><summary>'+h(short(m['id']))+'</summary><img src="'+m['display_path']+'"><pre>'+h(m['retrieval_text'])+'</pre></details>')
        body.append('</article>')
    notes=OUT/'visual_observations.json'
    if notes.exists():body.append('<h2>原图观察备注</h2><pre>'+h(json.dumps(read(notes),ensure_ascii=False,indent=2))+'</pre>')
    body+=['<h2>数据与复核</h2><p><a href="observation_plan.json">事先固定的观察口径</a> · <a href="numeric_summary.json">全量统计与来源</a> · <a href="pairs.csv">17,240 个配对</a> · <a href="pc_by_question.csv">逐题主成分统计</a> · <a href="spectra.csv">全部 master 方差谱</a></p>',
        '<p>结论边界：这项观察检验的是方差方向对目标识别的作用。确认“图像细节更多位于低方差方向”，还需要与 PCA 无关的细节标注或保持其他因素不变的图像/文本干预。</p></html>']
    (OUT/'report.html').write_text('\n'.join(body))
    dump(OUT/'render_manifest.json',{'plots':['band_target_support.png','largest_spectrum.png',*[n+'_contributions.png' for n in examples]],
                                   'original_images_copied':48,'reader_calls':0,'plan_selection_preserved':True})
    print(json.dumps({'report':str(OUT/'report.html'),'examples':list(examples),'extreme_pcs':[e['pc'] for e in extremes]}))


if __name__=='__main__':main()
