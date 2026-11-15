"""Render the user-supplied ablations; run with python3 plot_ablation.py."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-semrag-vamo')
import csv
import json
import re
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import FormatStrFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
FONT_DIR = Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman')
for font in FONT_DIR.glob('*.TTF'):
    font_manager.fontManager.addfont(str(font))
FONT = font_manager.findfont(font_manager.FontProperties(family='Times New Roman', weight='bold'), fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.weight': 'bold',
    'font.size': 16, 'axes.labelsize': 18, 'axes.labelweight': 'bold',
    'axes.linewidth': 1.8, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'xtick.major.width': 1.8, 'ytick.major.width': 1.8,
    'xtick.major.size': 6, 'ytick.major.size': 6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.unicode_minus': True, 'axes.grid': False,
    'hatch.linewidth': 0.9,
    'savefig.facecolor': 'white',
})

DATA = [
    {'name': 'lambda', 'xlabel': 'λ (α = 1)', 'fixed': 'α=1',
     'colors': ['#A8CEE5', '#17679D'],
     'rows': [
         ('−', '0.49595687', '0.52560647'),
         ('0.001', '0.50404313', '0.52291105'),
         ('0.003', '0.49595687', '0.53369272'),
         ('0.01', '0.50673854', '0.54177898'),
         ('0.03', '0.49865229', '0.53638814'),
         ('0.1', '0.49326146', '0.54447439'),
     ]},
    {'name': 'alpha', 'xlabel': 'α (λ = 0.03)', 'fixed': 'λ=0.03',
     'colors': ['#F5C4A1', '#D46119'],
     'rows': [
         ('−', '0.49595687', '0.52560647'),
         ('0.1', '0.49326146', '0.52021563'),
         ('0.3', '0.49595687', '0.53369272'),
         ('0.6', '0.49595687', '0.53638814'),
         ('1', '0.49865229', '0.53638814'),
         ('2', '0.51212938', '0.52021563'),
     ]},
]

def scaled(value):
    return Decimal(value) * 100

def label(value):
    return str(scaled(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def border_color(fill):
    return to_hex(tuple(channel * 0.72 for channel in to_rgb(fill)))

def draw(ax, panel):
    rows = panel['rows']
    x = np.arange(len(rows))
    width = 0.36
    for method_index, method in enumerate(['SemRAG', 'VAMO']):
        values = [float(scaled(row[method_index + 1])) for row in rows]
        bars = ax.bar(x + (method_index - 0.5) * 0.46, values,
                      width=width, color=panel['colors'][method_index],
                      edgecolor=border_color(panel['colors'][method_index]), linewidth=3.0,
                      label=method)
        # Independent texture encoding, consistent across panels and legend.
        for bar in bars:
            bar.set_hatch(['//', '..'][method_index])
            bar.set_hatchcolor(['#444444', '#FFFFFF'][method_index])
        ax.bar_label(bars, labels=[label(row[method_index + 1]) for row in rows],
                     padding=5, fontsize=16, fontweight='bold')
    ax.set_xticks(x, [row[0] for row in rows])
    ax.set_xlim(-0.62, len(rows) - 0.38)
    ax.set_ylim(48, 56)
    ax.set_yticks([48, 50, 52, 54, 56])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel(panel['xlabel'], labelpad=10)
    ax.set_ylabel('Score (×100)', labelpad=12)
    ax.tick_params(axis='both', direction='out', pad=6)
    ax.spines[['top', 'right']].set_visible(True)
    ax.set_axisbelow(True)
    ax.grid(False, axis='x')
    ax.grid(True, axis='y', color='#D9D9D9', linestyle='--', linewidth=1.8)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=2,
              frameon=False, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 20},
              handlelength=1.5, handletextpad=0.6, columnspacing=2.0, borderaxespad=0)

def save(fig, name):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=400, bbox_inches='tight', pad_inches=0.16)
    plt.close(fig)

def main():
    with (OUT / 'plot_data.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['ablation', 'fixed_parameter', 'parameter_value', 'method', 'raw_score', 'score_x100', 'display_label'])
        for panel in DATA:
            for row in panel['rows']:
                for index, method in enumerate(['SemRAG', 'VAMO']):
                    value = row[index + 1]
                    writer.writerow([panel['name'], panel['fixed'], row[0], method, value, str(scaled(value)), label(value)])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 12))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.07, top=0.95, hspace=0.28)
    for ax, panel in zip(axes, DATA):
        draw(ax, panel)
    fig.savefig(OUT / 'ablation_preview.png', dpi=130, bbox_inches='tight', pad_inches=0.16)
    save(fig, 'ablation_alpha_lambda_patterns')
    for panel in DATA:
        fig, ax = plt.subplots(figsize=(10.5, 6))
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.88)
        draw(ax, panel)
        save(fig, f"ablation_{panel['name']}")
    pdf_fonts = sorted({v.decode() for v in re.findall(rb'/BaseFont\s*/([^\s/]+)', (OUT / 'ablation_alpha_lambda_patterns.pdf').read_bytes())})
    assert pdf_fonts and all('TimesNewRoman' in name for name in pdf_fonts), pdf_fonts
    assert sum(len(p['rows']) * 2 for p in DATA) == 24
    metadata = {
        'source': 'User-provided table; no data inferred or recomputed from experiments.',
        'font_path': FONT, 'pdf_base_fonts': pdf_fonts,
        'tick_and_data_label_font_pt': 16, 'axis_label_font_pt': 18, 'legend_font_pt': 20,
        'all_text_weight': 'bold', 'axis_linewidth_pt': 1.8,
        'scale': 100, 'display_decimals': 2, 'percent_sign': False,
        'metric_name': 'Score (metric name not provided)',
        'baseline': 'The user-provided dash row is displayed as its own category.',
        'x_scale': 'Categorical, equal spacing; lambda=1.0 excluded at user request.',
        'y_limits': [48, 56], 'y_ticks': [48, 50, 52, 54, 56],
        'grid': {'axis': 'y', 'color': '#D9D9D9', 'linestyle': '--', 'linewidth_pt': 1.8, 'behind_bars': True},
        'titles': False, 'data_count': 24,
        'patterns': {'SemRAG': 'dark diagonal lines //', 'VAMO': 'white dots ..'},
        'hatch_linewidth_pt': 0.9,
        'axes_frame': 'All four spines visible, 1.8 pt.',
        'combined_hspace': 0.28,
        'colors': {p['name']: dict(zip(['SemRAG', 'VAMO'], p['colors'])) for p in DATA},
        'bar_edge_linewidth_pt': 3.0,
        'combined_figsize_inches': [10.5, 12],
        'single_figsize_inches': [10.5, 6],
        'bar_edge_colors': {p['name']: dict(zip(['SemRAG', 'VAMO'], [border_color(c) for c in p['colors']])) for p in DATA},
    }
    (OUT / 'plot_settings.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
