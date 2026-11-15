"""Final alpha-ablation figure with separated labels; signed run unchanged."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/visualmemory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

BASE=Path(__file__).resolve().parents[2]/'results/memeye_qwen_positive_variance_alpha_ablation_l0p03_20261107'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary=json.loads((BASE/'summary.json').read_text());rows=summary['ablation'];overall=summary['overall']
    alphas=[r['alpha'] for r in rows]
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    for ax,metric,label in zip(axes,['accuracy','recall_at_5'],['Answer accuracy (×100)','Recall@5, micro (×100)']):
        for prefix,method,color,name in zip(['baseline','master'],['baseline_question_only','master_answer_query'],
            ['#E9B187','#A94B13'],['Caption baseline','Caption visual master']):
            values=[r[prefix+'_'+metric] for r in rows]
            other=[r[('master' if prefix=='baseline' else 'baseline')+'_'+metric] for r in rows]
            ax.plot(alphas,values,'o-',color=color,label=name+' + PV',linewidth=2)
            control=overall[method]['accuracy' if metric=='accuracy' else 'recall_at_5_micro']
            ax.axhline(control,color=color,linestyle='--',alpha=.65,label=name+' (unweighted)')
            for x,y,neighbor in zip(alphas,values,other):
                offset=-17 if y<neighbor and abs(y-neighbor)<.01 else 9
                ax.annotate(f'{y*100:.2f}',(x,y),xytext=(0,offset),textcoords='offset points',
                            ha='center',color=color,fontsize=11,weight='bold')
        ax.set_xscale('log');ax.set_xticks(alphas,[format(v,'g') for v in alphas]);ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v*100:.2f}'))
        ax.set_xlabel('Alpha (log scale)',fontsize=17);ax.set_ylabel(label,fontsize=17)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.005),frameon=False,prop={'size':13,'weight':'bold'})
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.29,wspace=.35)
    outputs=[]
    for extension in ['png','pdf','svg']:
        path=BASE/f'alpha_ablation_final.{extension}'
        fig.savefig(path,dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15);outputs.append(path)
    plt.close(fig)
    report=BASE/'report_final.html'
    report.write_text((BASE/'report.html').read_text().replace('alpha_ablation.png','alpha_ablation_final.png'));outputs.append(report)
    sources=[Path(__file__),BASE/'summary.json',BASE/'report.html',BASE/'completion_manifest.json',BASE/'independent_audit.json']
    manifest={'description':'Presentation-only render separating nearby point labels; all signed experiment artifacts unchanged.',
        'source_sha256':{str(p):sha(p) for p in sources},'artifact_sha256':{str(p):sha(p) for p in outputs}}
    (BASE/'presentation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
