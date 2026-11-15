"""Presentation-only render with labels separated from curves; signed data unchanged."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/memgallery_qwen_pv_original_answers_alpha_ablation_l0p03_20261107'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((BASE / 'summary.json').read_text())
    table, overall, alphas = summary['ablation'], summary['overall'], summary['alphas']
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, metric, label in zip(axes, ['em', 'recall_at_5'], ['Exact match (×100)', 'Recall@5, micro (×100)']):
        for prefix, method, color, name in zip(['baseline', 'master'],
                ['baseline_question_only', 'master_answer_query'],
                ['#E4B27C', '#A85B13'], ['Caption baseline', 'Caption visual master']):
            values = [r[prefix + '_' + metric] for r in table]
            ax.plot(alphas, values, 'o-', color=color, label=name + ' + PV', linewidth=2)
            control = overall[method]['em' if metric == 'em' else 'recall_at_5_micro']
            ax.axhline(control, color=color, linestyle='--', alpha=.65, label=name + ' (unweighted)')
            for x, y in zip(alphas, values):
                offset = -18 if prefix == 'baseline' else 10
                if metric == 'em' and x == 1.:
                    offset = 10 if prefix == 'baseline' else -18
                ax.annotate(f'{y*100:.2f}', (x, y), xytext=(0, offset),
                    textcoords='offset points', ha='center', color=color, fontsize=12, weight='bold',
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .35}, zorder=5)
        ax.set_xscale('log'); ax.set_xticks(alphas, [format(a, 'g') for a in alphas]); ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v*100:.2f}'))
        ax.set_xlabel('Alpha (log scale)', fontsize=17); ax.set_ylabel(label, fontsize=17)
        for tick in ax.get_xticklabels() + ax.get_yticklabels(): tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, .005),
               frameon=False, prop={'size': 13, 'weight': 'bold'})
    fig.subplots_adjust(left=.08, right=.98, top=.95, bottom=.28, wspace=.32)
    files = []
    for ext in ['png', 'pdf', 'svg']:
        target = BASE / f'alpha_ablation_final.{ext}'
        fig.savefig(target, dpi=220, facecolor='white', bbox_inches='tight', pad_inches=.15)
        files.append(target)
    plt.close(fig)
    report = BASE / 'report_final.html'
    report.write_text((BASE / 'report.html').read_text().replace('alpha_ablation.png', 'alpha_ablation_final.png'))
    files.append(report)
    sources = [Path(__file__), BASE / 'summary.json', BASE / 'report.html',
               BASE / 'completion_manifest.json', BASE / 'independent_audit.json']
    manifest = {'description': 'Presentation only: separate data labels from curves; no experiment results changed.',
        'source_sha256': {str(f): sha(f) for f in sources}, 'artifact_sha256': {str(f): sha(f) for f in files}}
    (BASE / 'presentation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
