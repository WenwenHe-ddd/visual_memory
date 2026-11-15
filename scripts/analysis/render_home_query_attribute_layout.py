"""Presentation-only revision: exact edits left, query above, caption below.

Reads the completed experiment; never re-encodes inputs or overwrites it.
"""
from pathlib import Path
import hashlib
import html
import json
import os
import textwrap
import zipfile

os.environ.setdefault('MPLCONFIGDIR','/tmp/visualmemory-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.formatting.rule import ColorScaleRule

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'reports/home_q10_query_attribute_projection_20261113'
OUT=ROOT/'reports/home_q10_query_attribute_layout_20261113'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def wrap(s,width):return textwrap.fill(s,width=width,break_long_words=False,break_on_hyphens=False)
def edit_label(s,wrapped=False):
    pairs=['"'+r['old']+'" → "'+r['new']+'"' for r in s['replacements']]
    return '\n'.join(wrap(x,83) if wrapped else x for x in pairs)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion_manifest.json').exists():raise FileExistsError('Completed layout exists; use a new version.')
    original=read(SOURCE/'completion_manifest.json')
    for p,h in original['output_sha256'].items():assert sha(p)==h,p
    plan=read(SOURCE/'plan.json');summary=read(SOURCE/'numeric_summary.json')
    stimuli=read(SOURCE/'stimuli.json');members=read(SOURCE/'members.json')
    target=next(m for m in members if m['id']==plan['target_id'])
    data=np.load(SOURCE/'analysis_vectors.npz')
    values=np.concatenate([data['contributions'],data['outside'][:,:,None],data['total'][:,:,None]],axis=2)*100
    assert values.shape==(2,16,6)
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelweight':'bold',
                         'axes.linewidth':1.1,'pdf.fonttype':42,'svg.fonttype':'none'})
    cmap=LinearSegmentedColormap.from_list('query_delta',['#CB6A27','#FFFFFF','#2457A6'])
    vmax=float(abs(values).max());labels=[edit_label(s,True) for s in stimuli]
    heights=np.array([max(.47,.25*len(label.splitlines())+.12) for label in labels])
    edges=np.r_[0,np.cumsum(heights)];centers=(edges[:-1]+edges[1:])/2
    heat_height=float(edges[-1]);bottom=4.0;top=2.45;height=bottom+heat_height+top
    qtexts=[plan['queries']['raw_query'],plan['queries']['frozen_master_query']]
    qheads=['Query','Query + candidate answer (original master selection)']
    cell_headers=[f'PC{j+1}\n{100*summary["variance_shares"][j]:.2f}' for j in range(4)]+['Outside\n4 PCs','Total\nchange']
    all_checks=[]

    def draw(name,panels):
        left=.35;label_width=9.8;gap=.22;heat_width=5.1;panel_gap=.38;bar_gap=.55;bar_width=.22;right=.9
        plot_start=left+label_width+gap
        width=plot_start+len(panels)*heat_width+(len(panels)-1)*panel_gap+bar_gap+bar_width+right
        fig=plt.figure(figsize=(width,height))
        lax=fig.add_axes([left/width,bottom/height,label_width/width,heat_height/height])
        lax.set_xlim(0,1);lax.set_ylim(edges[-1],0);lax.axis('off')
        text_artists=[]
        for i,label in enumerate(labels):
            t=lax.text(0,centers[i],label,fontsize=14,ha='left',va='center',linespacing=1.18,color='#172333')
            text_artists.append(t)
        fig.text(left/width,(bottom+heat_height+.28)/height,'Raw content → Modified content',fontsize=16,weight='bold',ha='left',va='bottom')
        query_boxes=[]
        for panel_index,k in enumerate(panels):
            x=plot_start+panel_index*(heat_width+panel_gap)
            ax=fig.add_axes([x/width,bottom/height,heat_width/width,heat_height/height])
            im=ax.pcolormesh(np.arange(7),edges,values[k],cmap=cmap,vmin=-vmax,vmax=vmax,shading='flat',rasterized=False)
            ax.set_ylim(edges[-1],0);ax.set_xlim(0,6);ax.set_yticks([])
            ax.set_xticks(np.arange(6)+.5,cell_headers);ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=8,labelsize=13)
            for tick in ax.get_xticklabels():tick.set_weight('bold')
            for j in [4,5]:ax.axvline(j,color='#555555',lw=.8)
            for i in range(16):
                for j in range(6):
                    value=float(values[k,i,j]);label=f'{value:+.2f}'
                    ax.text(j+.5,centers[i],label,ha='center',va='center',fontsize=14,
                            weight='bold',color='white' if abs(value)>.65*vmax else '#172333')
            # A dedicated query box above the PC labels, with the complete actual input.
            head=fig.text(x/width,(bottom+heat_height+top-.15)/height,qheads[k],
                          fontsize=14,weight='bold',ha='left',va='top',color='#2457A6')
            query_text='\n'.join(wrap(part,55) for part in qtexts[k].splitlines())
            qt=fig.text(x/width,(bottom+heat_height+top-.53)/height,query_text,
                        fontsize=15,ha='left',va='top',linespacing=1.15)
            query_boxes.append((qt,x))
        cax=fig.add_axes([(plot_start+len(panels)*heat_width+(len(panels)-1)*panel_gap+bar_gap)/width,
                          bottom/height,bar_width/width,heat_height/height])
        cb=fig.colorbar(im,cax=cax);cb.set_label('Change in query similarity (×100)',fontsize=15,weight='bold')
        # Full caption and dialogue, so every displayed edit can be located.
        caption_width=int((width-left-.4)*8.4)
        caption=wrap(target['caption'],caption_width)
        fig.text(left/width,3.61/height,'Original caption — '+target['id'].split('/')[-1],fontsize=16,weight='bold',va='top')
        caption_artist=fig.text(left/width,3.27/height,caption,fontsize=15,ha='left',va='top',linespacing=1.18)
        dialogue_y=3.27-(len(caption.splitlines())*.25)-.23
        dialogue_artist=fig.text(left/width,dialogue_y/height,'Original dialogue:\n'+target['dialogue'],
                               fontsize=13,ha='left',va='top',linespacing=1.18)
        footer='Positive = higher query similarity; negative = lower. PC header numbers = variance share ×100. Two replacements in one row apply together.'
        if 1 in panels:footer+=' Original candidate B says neither desk; ground-truth answer is C (window desk).'
        fig.text(left/width,.18/height,wrap(footer,int((width-left-.4)*10)),fontsize=11,color='#465260',va='bottom')
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        issues=[]
        for i,t in enumerate(text_artists):
            bb=t.get_window_extent(renderer);row_top=lax.transData.transform((0,edges[i]))[1];row_bottom=lax.transData.transform((0,edges[i+1]))[1]
            if bb.y0<row_bottom-1 or bb.y1>row_top+1:issues.append(f'row {i+1} text exceeds row height')
            if bb.x1>lax.get_window_extent(renderer).x1+1:issues.append(f'row {i+1} label exceeds left column')
        for qt,x in query_boxes:
            bb=qt.get_window_extent(renderer)
            limit=fig.transFigure.transform((0,(bottom+heat_height+.69)/height))[1]
            if bb.y0<limit:issues.append('query overlaps PC column labels')
        if caption_artist.get_window_extent(renderer).y0<dialogue_artist.get_window_extent(renderer).y1:issues.append('caption overlaps dialogue')
        if dialogue_artist.get_window_extent(renderer).y0<fig.transFigure.transform((0,.65/height))[1]:issues.append('dialogue overlaps footer')
        assert not issues,issues
        for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
        all_checks.append({'figure':name,'panels':panels,'exact_edit_rows':16,'text_overflow_issues':issues,'caption_complete':True})

    draw('query_attribute_heatmap_with_text',[0,1])
    draw('raw_query_heatmap_with_text',[0])
    draw('master_query_heatmap_with_text',[1])
    payload={'source_root':str(SOURCE),'queries':plan['queries'],'original_caption':target['caption'],
             'original_dialogue':target['dialogue'],'rows':[{'id':s['id'],'label':edit_label(s),'replacements':s['replacements']} for s in stimuli],
             'numeric_values_x100':values.tolist(),'color_limits':[-vmax,vmax]}
    dump(OUT/'figure_content.json',payload)

    # Update only the presentation of the two workbook heatmaps, preserving all numeric source sheets.
    wb=load_workbook(SOURCE/'Home_Q10_query_attribute_projection.xlsx')
    for k,name in enumerate(['RawHeatmap','MasterHeatmap']):
        position=wb.sheetnames.index(name);del wb[name];ws=wb.create_sheet(name,position)
        ws.merge_cells('A1:G1');ws['A1']=qheads[k];ws['A1'].font=Font(name='Times New Roman',bold=True,size=15,color='2457A6');ws.row_dimensions[1].height=24
        ws.merge_cells('A2:G2');ws['A2']=qtexts[k];ws['A2'].alignment=Alignment(wrap_text=True,vertical='top');ws['A2'].font=Font(name='Times New Roman',size=13);ws.row_dimensions[2].height=72 if k else 40
        if k:
            ws.merge_cells('A3:G3');ws['A3']='原实验候选陈述 B：两桌都无显示器；正确答案 C：靠窗桌。本图保留原查询。';ws.row_dimensions[3].height=22
        for col,label in enumerate(['Raw content → Modified content','PC1','PC2','PC3','PC4','Outside','Total'],1):
            cell=ws.cell(5,col,label);cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='2457A6')
        for i,s in enumerate(stimuli,6):
            ws.cell(i,1,edit_label(s));ws.cell(i,1).alignment=Alignment(wrap_text=True,vertical='center')
            ws.cell(i,1).font=Font(name='Times New Roman',size=12)
            ws.row_dimensions[i].height=max(36,18*len(labels[i-6].splitlines())+8)
            for j,value in enumerate(values[k,i-6],2):ws.cell(i,j,float(value)).number_format='+0.00;-0.00;0.00'
        ws.conditional_formatting.add('B6:G21',ColorScaleRule(start_type='num',start_value=-vmax,start_color='CB6A27',
            mid_type='num',mid_value=0,mid_color='FFFFFF',end_type='num',end_value=vmax,end_color='2457A6'))
        ws.merge_cells('A23:G23');ws['A23']='Original caption — '+target['id'].split('/')[-1];ws['A23'].font=Font(bold=True,size=14)
        ws.merge_cells('A24:G24');ws['A24']=target['caption'];ws['A24'].alignment=Alignment(wrap_text=True,vertical='top');ws.row_dimensions[24].height=115
        ws.merge_cells('A26:G26');ws['A26']='Original dialogue';ws['A26'].font=Font(bold=True,size=14)
        ws.merge_cells('A27:G27');ws['A27']=target['dialogue'];ws['A27'].alignment=Alignment(wrap_text=True,vertical='top');ws.row_dimensions[27].height=40
        ws.column_dimensions['A'].width=110
        for column in 'BCDEFG':ws.column_dimensions[column].width=12
        ws.freeze_panes='B6';ws.sheet_view.zoomScale=75
    figures=wb['Figures'];figures._images.clear()
    for filename,anchor,width in [('query_attribute_heatmap_with_text.png','A1',1650)]:
        im=ExcelImage(OUT/filename);ratio=width/im.width;im.width=width;im.height*=ratio;figures.add_image(im,anchor)
    for name,filename in [('RawFigure','raw_query_heatmap_with_text.png'),('MasterFigure','master_query_heatmap_with_text.png')]:
        ws=wb.create_sheet(name);im=ExcelImage(OUT/filename);ratio=1250/im.width;im.width=1250;im.height*=ratio;ws.add_image(im,'A1')
    wb['Readme'].append(['展示调整','热力图上方为实际 query，下方为完整原 caption；行标签为精确原文→修改后文本。实验数据未变。'])
    destination=OUT/'Home_Q10_query_attribute_projection_with_text.xlsx';wb.save(destination)
    h=lambda s:html.escape(str(s))
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 属性热力图：完整文本</title>',
          '<style>body{font:16px/1.7 system-ui;color:#182435;margin:24px}img{width:100%;height:auto}main{max-width:1800px;margin:auto}a{color:#2457A6}details{margin:20px 0}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ddd}</style><main>',
          '<p>Home/Q10：仅调整展示。上方是实际查询，左侧是原文 → 修改后文本，下方是完整原 caption 及对话。分数、主成分、修改清单均沿用已完成实验。</p>',
          '<img src="query_attribute_heatmap_with_text.png">',
          '<p><a href="query_attribute_heatmap_with_text.pdf">合并版 PDF</a> · <a href="query_attribute_heatmap_with_text.svg">合并版 SVG</a> · <a href="Home_Q10_query_attribute_projection_with_text.xlsx">Excel：更新后的图与可编辑数据</a></p>',
          '<details><summary>单独查看原始 query</summary><img src="raw_query_heatmap_with_text.png"></details>',
          '<details><summary>单独查看 query + candidate answer</summary><img src="master_query_heatmap_with_text.png"></details>',
          '<p>右栏保留原 master 错误选择的陈述 B（两桌均无显示器），不是正确答案 C（靠窗桌）。正值表示相似度升高，不代表语义正确。一行有两处替换时，两处一起生效。</p>',
          '<p><a href="../home_q10_query_attribute_projection_20261113/report.html">原实验完整报告</a> · <a href="figure_content.json">图中文字与数据</a> · <a href="layout_audit.json">展示核验</a></p></main></html>']
    (OUT/'report.html').write_text('\n'.join(body),encoding='utf-8')

    # Presentation audit: numerical sheets identical, actual strings unabridged, original files unchanged.
    new=load_workbook(destination);old=load_workbook(SOURCE/'Home_Q10_query_attribute_projection.xlsx')
    preserved=['Edits','Contributions','OriginalPair','OriginalScores','Energy','Spectrum','ScoreChangeChart']
    for name in preserved:
        assert list(new[name].values)==list(old[name].values),name
    for k,name in enumerate(['RawHeatmap','MasterHeatmap']):
        assert new[name]['A2'].value==qtexts[k] and new[name]['A24'].value==target['caption'] and new[name]['A27'].value==target['dialogue']
        for i,s in enumerate(stimuli,6):
            assert new[name].cell(i,1).value==edit_label(s)
            np.testing.assert_allclose([new[name].cell(i,j).value for j in range(2,8)],values[k,i-6],atol=1e-12,rtol=0)
    assert len(new['ScoreChangeChart']._charts)==1
    assert sum(len(ws._images) for ws in new)==3
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    for p,h0 in original['output_sha256'].items():assert sha(p)==h0,p
    audit={'status':'passed','layout_checks':all_checks,'original_files_unchanged':len(original['output_sha256']),
           'numeric_sheets_identical':preserved,'exact_edit_rows':16,'complete_query_inputs':2,
           'full_caption_and_dialogue_below':True,'embedded_figures':3,'native_charts_preserved':1,
           'new_model_calls':0,'new_retrieval_runs':0,'source_completion_manifest_sha256':sha(SOURCE/'completion_manifest.json')}
    dump(OUT/'layout_audit.json',audit)
    dump(OUT/'completion_manifest.json',{'status':'complete','source_root':str(SOURCE),'source_manifest_sha256':sha(SOURCE/'completion_manifest.json'),
         'renderer_sha256':sha(__file__),'outputs':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['completion_manifest.json','render.log']}})
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
