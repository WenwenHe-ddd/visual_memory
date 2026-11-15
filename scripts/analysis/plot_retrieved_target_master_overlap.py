#!/usr/bin/env python3
"""Plot the frozen 138-question, retrieved-target answer-failure cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/baseline_failure_masters_20261022/qa_failure_set/retrieved_target_master_overlap'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=Path('/tmp/visualmemory-plot-fonts/NotoSansCJKsc-Regular.otf'))
    parser.add_argument('--output', type=Path, default=SOURCE / 'figures')
    args = parser.parse_args()
    summary = json.loads((SOURCE / 'summary.json').read_text())
    with (SOURCE / 'all_138_questions.csv').open(encoding='utf-8-sig') as handle:
        records = list(csv.DictReader(handle))
    counts = [sum(r['any_same_master'] == 'True' for r in records),
              sum(r['any_same_master'] == 'False' and int(r['distractor_count']) > 0 for r in records),
              sum(int(r['distractor_count']) == 0 for r in records)]
    assert counts == [summary['same_master_questions'], summary['with_distractors_but_no_same_master'],
                      summary['without_distractors']] == [112, 23, 3]
    n = len(records)
    assert sum(counts) == n == summary['eligible_questions'] == 138
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'pdf.fonttype': 42, 'figure.facecolor': 'white'})
    fig = plt.figure(figsize=(12.8, 7.8))
    ink, muted, accent = '#182E39', '#647681', '#177E78'
    colors = [accent, '#ABBBC6', '#E8EDF0']
    fig.text(.055, .925, '多数“命中目标仍答错”的题目含同 master 干扰项', fontsize=25, color=ink)
    fig.text(.055, .867, 'MemEye  |  原始 Qwen caption  |  全量检索 Top-5  |  失败题子集 N = 138',
             fontsize=13, color=muted)
    ax = fig.add_axes([.015, .18, .52, .64])
    wedges, _ = ax.pie(counts, startangle=92, counterclock=False, colors=colors,
                      wedgeprops={'linewidth': 3, 'edgecolor': 'white'})
    for i in [0, 1]:
        angle = np.deg2rad((wedges[i].theta1 + wedges[i].theta2) / 2)
        radius = .53 if i == 0 else .69
        ax.text(radius * np.cos(angle), radius * np.sin(angle),
                f'{counts[i] / n:.1%}\n{counts[i]} 道', ha='center', va='center',
                color='white' if i == 0 else ink, fontsize=24 if i == 0 else 16, linespacing=1.5)
    angle = np.deg2rad((wedges[2].theta1 + wedges[2].theta2) / 2)
    ax.annotate('3 道 · 2.2%', xy=(.95 * np.cos(angle), .95 * np.sin(angle)),
                xytext=(-.45, 1.16), ha='center', va='center', fontsize=12, color=muted,
                arrowprops={'arrowstyle': '-', 'color': '#9BAAB3', 'linewidth': 1.2,
                            'connectionstyle': 'angle,angleA=0,angleB=90,rad=4'})
    ax.set(aspect='equal', xlim=(-1.2, 1.2), ylim=(-1.12, 1.28))
    box = FancyBboxPatch((.55, .49), .39, .30, transform=fig.transFigure,
                         boxstyle='round,pad=0.015,rounding_size=0.018',
                         facecolor='#EDF7F5', edgecolor='none')
    fig.add_artist(box)
    fig.text(.575, .735, '存在同 master 的目标—干扰项配对', color=accent, fontsize=17)
    fig.text(.575, .62, '112 / 138', color=accent, fontsize=42)
    fig.text(.575, .54, '81.16% 的该类失败题', color=ink, fontsize=19)
    fig.text(.565, .425, '●', color=colors[1], fontsize=20)
    fig.text(.595, .425, '目标与干扰项均不同 master', fontsize=15, color=ink)
    fig.text(.595, .382, '23 道  ·  16.67%', fontsize=14, color=muted)
    fig.text(.565, .312, '●', color='#CDD7DE', fontsize=20)
    fig.text(.595, .312, 'Top-5 全为目标，无干扰项', fontsize=15, color=ink)
    fig.text(.595, .269, '3 道  ·  2.17%', fontsize=14, color=muted)
    fig.add_artist(plt.Line2D([.055, .945], [.16, .16], transform=fig.transFigure,
                             color='#DCE4E8', linewidth=1))
    fig.text(.055, .113, '分母：187 道回答失败题中，Top-5 至少包含一条目标记忆的 138 题；另 49 题未命中目标，未纳入本图。',
             fontsize=11, color=muted)
    fig.text(.055, .068, '同 master：Top-5 内至少一条目标记忆与一条非目标记忆属于同一 caption_visual_master 簇；每题计一次。',
             fontsize=11, color=muted)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / 'same_master_failed_questions_pie'
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=240, facecolor='white')
    plt.close(fig)
    with (args.output / 'chart_data.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['category', 'question_count', 'denominator', 'percentage'])
        for name, count in zip(['同 master 目标与干扰项', '目标与干扰项均不同 master', 'Top-5 无干扰项'], counts):
            writer.writerow([name, count, n, 100 * count / n])
    provenance = {'summary_sha256': hashlib.sha256((SOURCE / 'summary.json').read_bytes()).hexdigest(),
                  'per_question_sha256': hashlib.sha256((SOURCE / 'all_138_questions.csv').read_bytes()).hexdigest(),
                  'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'font_sha256': hashlib.sha256(args.font.read_bytes()).hexdigest(),
                  'font_source': 'https://github.com/notofonts/noto-cjk',
                  'denominator': n, 'counts': counts, 'source_counts_recomputed': True}
    (args.output / 'chart_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'png': str(stem.with_suffix('.png')), 'counts': counts, 'denominator': n}))


if __name__ == '__main__':
    main()
