#!/usr/bin/env python3
"""Accuracy-only PCA dimension plot with requested Times New Roman styling."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cvm_dimension_plot_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/memeye_qwen_cvm_pca_dimensions_1to10_zscore_l0p03_a1_20261114'
FONT = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
font_manager.fontManager.addfont(FONT)
assert font_manager.FontProperties(fname=FONT).get_name() == 'Times New Roman'
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 20,
    'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})
source = OUT / 'summary.json'
rows = sorted(json.loads(source.read_text())['conditions'], key=lambda r: r['dimension'])
x = [r['dimension'] for r in rows]
y = [100*r['accuracy'] for r in rows]
assert x == list(range(1, 11))
fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
baseline = next(100*r['accuracy'] for r in rows if r['dimension'] == 2)
ax.axhline(baseline, color='#888888', linestyle='--', linewidth=2, zorder=1, label='2D reference')
ax.plot(x, y, color='#245a81', linewidth=2, marker='o', markersize=7)
for i, (dimension, value) in enumerate(zip(x, y)):
    below = any(neighbor > value for neighbor in y[max(0, i-1):i+2])
    ax.annotate(f'{value:.2f}', (dimension, value), xytext=(0, -12 if below else 10),
                textcoords='offset points', ha='center', va='top' if below else 'bottom', fontsize=20)
ax.set(xlabel='PCA Dimension', ylabel='Accuracy', xticks=x,
       xlim=(.5, 10.5), ylim=(53.15, 54.10), yticks=[53.2, 53.4, 53.6, 53.8, 54.0])
ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=20)
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.grid(False)
legend = ax.legend(loc='lower right', frameon=False, fontsize=20)
fig.canvas.draw()
texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts, *legend.get_texts()]
assert all(t.get_fontsize() == 20 for t in texts)
assert all(font_manager.findfont(t.get_fontproperties(), fallback_to_default=False) == FONT for t in texts)
assert all(s.get_linewidth() == 2 for s in ax.spines.values())
assert not ax.get_title() and [t.get_text() for t in legend.get_texts()] == ['2D reference']
outputs = []
for suffix in ('png', 'pdf', 'svg'):
    path = OUT / f'accuracy_1to10_tnr20.{suffix}'
    fig.savefig(path, dpi=300, facecolor='white')
    outputs.append(path)
plt.close(fig)
manifest = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'font': 'Times New Roman', 'font_file': FONT, 'font_size_pt': 20,
            'spine_width_pt': 2, 'tick_width_pt': 2, 'line_width_pt': 2,
            'reference_line': {'dimension': 2, 'value': baseline, 'style': '--', 'width_pt': 2},
            'point_labels': [f'{v:.2f}' for v in y], 'display_percent_symbol': False,
            'title': None, 'legend': {'label': '2D reference', 'location': 'lower right', 'frame': False}, 'grid': False,
            'artifact_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
(OUT / 'accuracy_1to10_tnr20_style.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'font_file': FONT, 'outputs': [str(p) for p in outputs]}))
