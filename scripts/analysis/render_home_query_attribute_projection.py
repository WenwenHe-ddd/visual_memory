"""Render the Home/Q10 real-query attribute experiment without altering inputs."""
from pathlib import Path
import csv
import hashlib
import html
import json
import os
import shutil

os.environ.setdefault('MPLCONFIGDIR','/tmp/visualmemory-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import Workbook
from openpyxl.chart import BarChart,Reference
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font,PatternFill

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/home_q10_query_attribute_projection_20261113'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    summary=read(OUT/'numeric_summary.json');plan=read(OUT/'plan.json');stimuli=read(OUT/'stimuli.json')
    members=read(OUT/'members.json');info=read(OUT/'question.json');data=np.load(OUT/'analysis_vectors.npz')
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.labelweight':'bold','pdf.fonttype':42,
                        'axes.spines.top':False,'axes.spines.right':False})
    cmap=LinearSegmentedColormap.from_list('score_change',['#CB6A27','#FFFFFF','#2457A6'])
    # No row normalization: both panels and every PC use the same absolute score scale.
    values=np.concatenate([data['contributions'],data['outside'][:,:,None],data['total'][:,:,None]],axis=2)*100
    vmax=float(abs(values).max())
    fig=plt.figure(figsize=(17.5,10))
    gs=fig.add_gridspec(1,3,width_ratios=[6,6,.2],left=.24,right=.95,bottom=.12,top=.93,wspace=.12)
    query_labels=['Raw question','Frozen master query (selected statement B)']
    pcs=[f'PC{j+1}\n{100*summary["variance_shares"][j]:.2f}' for j in range(4)]+['Outside\n4 PCs','Total\nchange']
    for k in range(2):
        ax=fig.add_subplot(gs[0,k]);im=ax.imshow(values[k],cmap=cmap,vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set_yticks(range(len(stimuli)),[s['label'] for s in stimuli] if k==0 else [])
        ax.set_xticks(range(6),pcs);ax.xaxis.tick_top();ax.set_xlabel(query_labels[k],labelpad=12)
        for line in [1.5,3.5,5.5,7.5,9.5,11.5,13.5]:ax.axhline(line,color='#999999',lw=.5)
        for line in [3.5,4.5]:ax.axvline(line,color='#333333',lw=.8)
        for i in range(len(stimuli)):
            for j in range(6):
                v=float(values[k,i,j]);ax.text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=10,
                                              color='white' if abs(v)>.65*vmax else '#182331')
        if k==0:
            for label,s in zip(ax.get_yticklabels(),stimuli):
                if s['role'].startswith('answer_') and s['role']!='answer_invariant_visual':label.set_color('#2457A6');label.set_weight('bold')
                elif s['role']=='episode_locator':label.set_color('#87631F')
    cb=fig.colorbar(im,cax=fig.add_subplot(gs[0,2]));cb.set_label('Change in query similarity (×100)',weight='bold')
    save(fig,'query_attribute_heatmap')
    # Original, unedited target minus the fixed original top-ranked distractor.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    retention=.03/(data['eigenvalues']+.03)
    for i,ax in enumerate(axs):
        c=data['original_pair'][i]*100;pcs_i=np.arange(1,5)
        ax.bar(pcs_i,c,color=np.where(c>=0,'#2457A6','#D7742D'),width=.6)
        ax.scatter(pcs_i,c*retention,c='#20252B',s=35,label='After ridge whitening',zorder=3)
        for j,v in enumerate(c):ax.annotate(f'{v:+.2f}',(j+1,v),xytext=(0,6 if v>=0 else -15),textcoords='offset points',ha='center',weight='bold',fontsize=11)
        ax.axhline(0,c='#555555',lw=.8);ax.set_xticks(pcs_i,[f'PC{j+1}\n{100*summary["variance_shares"][j]:.2f}' for j in range(4)])
        ax.set_xlabel(query_labels[i]);ax.legend(frameon=False,fontsize=10)
        ax.set_ylim(float(data['original_pair'].min()*100)-.65,float(data['original_pair'].max()*100)+.65)
    axs[0].set_ylabel('Target − distractor contribution (×100)')
    fig.tight_layout();save(fig,'original_pair_contributions')
    # Keep a compact comparison of the total changes without inventing a semantic score.
    fig,ax=plt.subplots(figsize=(11,8.5));y=np.arange(len(stimuli))
    ax.barh(y-.18,data['total'][0]*100,height=.34,color='#2457A6',label='Raw question')
    ax.barh(y+.18,data['total'][1]*100,height=.34,color='#D7742D',label='Frozen master query')
    ax.set_yticks(y,[s['label'] for s in stimuli]);ax.invert_yaxis();ax.axvline(0,c='#666666',lw=.8)
    ax.set_xlabel('Total change in query similarity (×100)');ax.legend(frameon=False)
    fig.tight_layout();save(fig,'total_score_changes')
    (OUT/'images').mkdir(exist_ok=True)
    for m in members:
        name=m['id'].split('/')[-1].replace(':','_')+Path(m['image_path']).suffix
        shutil.copy2(m['image_path'],OUT/'images'/name);m['display_image']='images/'+name
    mm={m['id']:m for m in members}
    wb=Workbook();ws=wb.active;ws.title='Readme'
    for row in [
        ['Home/Q10 真实 query 属性观察','GPT-5 caption；原 master_000124；5 条记忆，4 PCs'],
        ['原题',plan['queries']['raw_query']],['实际 master query',plan['queries']['frozen_master_query']],
        ['重要事实','原 master 选择 B：两张桌都没有显示器；正确选项 C：靠窗桌。保留原查询，不使用标准答案修正。'],
        ['方法','Delta c_j=(q dot u_j)*(u_j dot (e_edit-e_original));另列原子空间外贡献。'],
        ['正负含义','正=提高与当前 query 的相似度，负=降低；正值不等于正确。'],
        ['角色','答案事实/归属关系、定位事件、答案不变的视觉属性、表达对照；编码前按原题及原图标注。'],
        ['范围','1 题、1 条目标记忆、16 个分析副本、2 个真实查询；没有运行回答模型或改写记忆。'],
        ['PCA','按原 master 全部 5 条对话+caption 拟合；本题恰好与原 master_answer_query 的全局 Top-5 成员集合相同，顺序不同。修改后不重算 PCA。'],
        ['白化','原始目标/干扰项分差展示 lambda=0.03 alpha=1；扰动白化变化是固定原基底诊断。'],
        ['局限','合成文本干预，不是改动图像；反事实会改变正确答案，不按旧 gold 打分。']]:ws.append(row)
    for name,file in [('Edits','stimuli.csv'),('Contributions','query_attribute_contributions.csv'),
                      ('OriginalPair','original_pair_contributions.csv'),('OriginalScores','original_member_scores.csv'),
                      ('Energy','attribute_energy.csv'),('Spectrum','spectrum.csv')]:
        ws=wb.create_sheet(name)
        rows=list(csv.reader((OUT/file).open(encoding='utf-8-sig')))
        for i,row in enumerate(rows):
            converted=[]
            for value in row:
                try: converted.append(float(value) if i else value)
                except ValueError: converted.append(value)
            ws.append(converted)
        ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
    for k,name in enumerate(['RawHeatmap','MasterHeatmap']):
        ws=wb.create_sheet(name);ws.append(['Attribute','PC1','PC2','PC3','PC4','Outside','Total'])
        for s,row in zip(stimuli,values[k]):ws.append([s['label'],*map(float,row)])
        ws.conditional_formatting.add('B2:G17',ColorScaleRule(start_type='num',start_value=-vmax,start_color='CB6A27',
          mid_type='num',mid_value=0,mid_color='FFFFFF',end_type='num',end_value=vmax,end_color='2457A6'))
        for row in ws.iter_rows(min_row=2,min_col=2,max_col=7):
            for cell in row:cell.number_format='+0.00;-0.00;0.00'
    ws=wb.create_sheet('ScoreChangeChart');ws.append(['Edit','Raw question','Frozen master query'])
    for i,s in enumerate(stimuli):ws.append([s['label'],float(data['total'][0,i]*100),float(data['total'][1,i]*100)])
    chart=BarChart();chart.type='bar';chart.grouping='clustered';chart.overlap=0
    chart.add_data(Reference(ws,min_col=2,max_col=3,min_row=1,max_row=17),titles_from_data=True)
    chart.set_categories(Reference(ws,min_col=1,min_row=2,max_row=17));chart.width=28;chart.height=20
    chart.x_axis.title='Query similarity change (x100)';ws.add_chart(chart,'E2')
    ws=wb.create_sheet('Figures')
    for name,anchor,width in [('query_attribute_heatmap.png','A1',1250),('original_pair_contributions.png','A55',1000)]:
        image=ExcelImage(OUT/name);ratio=width/image.width;image.width=width;image.height*=ratio;ws.add_image(image,anchor)
    for ws in wb:
        ws.column_dimensions['A'].width=42;ws.column_dimensions['B'].width=105 if ws.title=='Readme' else 24
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='2457A6')
    wb.save(OUT/'Home_Q10_query_attribute_projection.xlsx')
    h=lambda x:html.escape(str(x))
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 真实查询属性贡献</title>',
      '<style>body{font:16px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#172434}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#f2f5fa;padding:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}details{margin:12px 0;border:1px solid #ddd;padding:10px}</style>',
      '<h1>Home/Q10：真实问题参与的属性贡献观察</h1>',
      '<p><b>原问题：</b>'+h(plan['queries']['raw_query'])+'</p><pre>'+h(json.dumps(info['question']['options'],ensure_ascii=False,indent=2))+'</pre>',
      '<p>正确答案 C：The window desk。原图可确认显示器在靠窗桌，右侧长桌只有合上的笔记本电脑。</p>',
      '<p class="note"><b>原实验的实际 master 查询：</b><br>'+h(plan['queries']['frozen_master_query']).replace('\n','<br>')+'<br>其中选中的陈述 B 与正确答案 C 不符。本观察保留这个输入，同时展示原始问题输入。没有使用正确答案修改检索查询。</p>',
      '<div class="pair">']
    for mid,role in [(plan['target_id'],'原始目标'),(plan['distractor_id'],'原始排名第一的同簇干扰项')]:
        m=mm[mid];body+=['<div><h3>'+role+'：'+h(mid.split('/')[-1])+'</h3><img src="'+m['display_image']+'"><pre>'+h(m['dialogue'])+'</pre></div>']
    body+=['</div><h2>原始目标与干扰项的分差分解</h2><img src="original_pair_contributions.png">',
      '<p>柱高 c_j=(q·u_j)[u_j·(e_target−e_distractor)]。正值支持原目标，负值支持原干扰项；黑点为 lambda=0.03 的白化中间贡献。横轴 PC 下方数字是原簇方差比例乘 100。</p>',
      '<table><tr><th>查询</th><th>原始分差</th><th>白化中间分差</th><th>最终 zscore 辅助赋权分差</th></tr>']
    for r in summary['original_pair']:
        body.append('<tr>'+''.join('<td>'+h(r[k] if k=='query' else f'{r[k]:+.6f}')+'</td>' for k in ['query','raw_margin','white_margin','zscore_final_margin_in_this_master'])+'</tr>')
    body+=['</table><p>原始问题列用同一 master 的原基底诊断，不代表 baseline_question_only 的全量簇白化方案。实际 master 查询列已与历史 lambda=0.03、alpha=1 的五条叶子原始/白化/最终分数复核。</p>',
      '<p class="note">这条样本中，实际 master 查询的 PC1 支持干扰项，PC2–4 支持目标。白化中间分差缩小，但仍未反超；最终辅助赋权分差也未改善。PC2 本身具有较大方差且支持目标，因此不能把高方差统称为无关。各方向对分差的作用不等于已识别其语义。</p>',
      '<h2>属性修改怎样改变真实查询的匹配分数</h2><img src="query_attribute_heatmap.png">',
      '<p>每行是一项明确的改动。左侧使用原始问题，右侧使用原实验的 master 查询。蓝色增加当前查询的相似度，橙色降低；颜色与正负不代表语义正确性。每格为 Delta c_j=(q·u_j)[u_j·(e_edit−e_original)]，乘 100 展示；前四列加 Outside 等于 Total。两幅图共享色阶，不做逐行归一化。</p>',
      '<p>按原 master 的全部五条记忆拟合，所以最多四个非零主成分。本题这个成员集合恰好与原 master_answer_query 的全局 Top-5 相同，顺序不同；选择规则来自原 master 成员表。计划中的 Not the global retrieval Top-5 指不按 Top-5 选取拟合样本，并非断言两集合不同。Outside 是修改产生的新变化在原四维子空间之外的<b>有符号查询贡献</b>，与此前图中的平方投影能量不是同一指标。</p>',
      '<h2>预先标注的角色</h2><ul><li>删除显示器、将显示器从靠窗桌移到右桌：改变答案事实或归属关系。</li><li>把“布局测试”改为“一周实际使用后”：改变定位记忆的事件线索，原图中的显示器归属不变。</li><li>显示器颜色、墙色、地毯颜色：保持本题答案不变的视觉属性对照；不能因此假定它们对记忆身份或检索完全无用。</li><li>caption 开头与助手确认句：保持事实和用户事件不变的表达对照。</li></ul>',
      '<p>角色依据实际问题、原图和对话，在编码前固定；没有依据投影结果反推标签。所有改动都是分析副本，原图及记忆未修改。删掉或转移显示器会改变反事实答案，所以不沿用原 gold 计算准确率。</p>',
      '<h2>总匹配分数变化</h2><img src="total_score_changes.png">',
      '<p class="note">例如两种“移除显示器”的表述：v1 是 holding only a mouse pad；v2 是 holding a mouse pad but no monitor。两者在原始问题下的总分变化分别为 −0.006657、+0.002422；实际 master 查询下分别为 −0.003693、+0.006901。同一语义改变不同写法会产生不同方向的响应，说明不能将单次词替换响应直接等同于语义相关性。</p>',
      '<h2>完整原记忆与分析副本</h2>']
    for m in members:
        body.append('<details><summary>'+h(m['id'])+'：原簇成员</summary><img src="'+m['display_image']+'"><pre>'+h(m['retrieval_text'])+'</pre></details>')
    for s in stimuli:
        body.append('<details><summary>'+h(s['id'])+' · '+h(s['label'])+'</summary><p>'+h(s['semantic_rationale'])+'</p><pre>'+h(json.dumps(s['replacements'],ensure_ascii=False,indent=2))+'</pre><pre>'+h(s['edited_text'])+'</pre></details>')
    body+=['<h2>数据与复核</h2><p><a href="Home_Q10_query_attribute_projection.xlsx">Excel：数据、热力图、可编辑柱状图</a> · <a href="query_attribute_contributions.csv">逐项贡献 CSV</a> · <a href="plan.json">编码前固定的设计</a> · <a href="numeric_summary.json">数值结果</a> · <a href="independent_audit.json">独立复核</a></p>',
      '<p>单题单目标、16 个文字反事实、2 个真实查询。没有重新回答或外部 API 调用，不作为准确率或跨样本语义结论。</p></html>']
    (OUT/'report.html').write_text('\n'.join(body),encoding='utf-8')
    interpretation={'status':'single_question_exploratory',
      'real_query_used':True,'actual_master_statement_is_wrong':True,
      'observations':['实际 master 查询下 PC1 对原目标/干扰项分差为负，PC2–4 为正。',
                      'PC2 的方差也较大且支持目标，不能将高方差方向整体称作无关。',
                      '原始白化中间分差改善但没有反超，最终 zscore 辅助赋权分差没有改善。',
                      '同一语义改动的两种表述可能导致相反的匹配变化；保持答案不变的属性/表达也会改变分数。',
                      '这些文字反事实不支持一个稳定的高方差无关、低方差细节分离结论。'],
      'limitations':['单题、单目标、16 个合成文字改动；没有重新回答。',
                     '正贡献只是当前 query 相似度上升，不等于事实正确。',
                     '对子空间外贡献完整保留；它不是第五个低方差 PC。']}
    (OUT/'interpretation.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2)+'\n')
    meta={'status':'rendered','script_sha256':sha(__file__),'original_images_copied':5,
          'figures':['query_attribute_heatmap','original_pair_contributions','total_score_changes'],
          'workbook':'Home_Q10_query_attribute_projection.xlsx'}
    (OUT/'render_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__':main()
