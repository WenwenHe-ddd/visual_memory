#!/usr/bin/env python3
"""Draw memory cards -> group score correction -> global ranking.

Scores are a synthetic illustration, not experiment results. SVG/PDF embed
Chinese glyphs; all labels and layout remain editable in this source.
"""
from pathlib import Path
import argparse
import os
from math import tanh

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory_method2_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon

OUT = Path(__file__).resolve().parent
INK, MUTED = '#23334B', '#6F7D90'
BLUE, ORANGE, PURPLE, GREEN = '#4285CB', '#DC9B48', '#8961C4', '#268974'
COLORS = {'A': BLUE, 'B': BLUE, 'C': ORANGE, 'D': ORANGE}


def configure_font(font_path=None):
    candidates = [font_path, os.environ.get('VISUALMEMORY_CJK_FONT'),
                  '/tmp/visualmemory-plot-fonts/NotoSansCJKsc-Regular.otf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams['font.family'] = [family, 'DejaVu Sans']
            return
    raise SystemExit('Chinese font required: pass --font /path/to/chinese-font.otf')


def main(font_path=None):
    configure_font(font_path)
    plt.rcParams.update({'svg.fonttype': 'path', 'pdf.fonttype': 42,
                         'font.size': 13, 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(15, 6.4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 15), ylim=(0, 6.4))
    ax.axis('off')

    def text(x, y, value, size=14, color=INK, align='center', **kwargs):
        ax.text(x, y, value, fontsize=size, color=color, ha=align,
                va='center', **kwargs)

    def rect(x, y, w, h, fill='white', edge='none', radius=.12):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=fill, edgecolor=edge, linewidth=1.2))

    def arrow(a, b, color='#A5B1C2', width=2.5, scale=18):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=scale,
                                    linewidth=width, color=color))

    def step(x, number, title):
        ax.add_patch(Circle((x, 4.87), .17, color=INK))
        text(x, 4.87, str(number), 12, 'white')
        text(x + .33, 4.87, title, 19, align='left', weight='bold')

    def memory_card(x, y, letter):
        color = COLORS[letter]
        rect(x + .04, y - .045, 1.16, .83, '#E4EAF2', radius=.09)
        rect(x, y, 1.16, .83, 'white', '#DFE5ED', radius=.09)
        rect(x, y, .07, .83, color, radius=.025)
        text(x + .26, y + .59, letter, 16, color, weight='bold')
        for i, length in enumerate((.52, .39)):
            ax.plot([x + .48, x + .48 + length], [y + .32 - i * .13] * 2,
                    color='#CAD4E0', linewidth=2.6, solid_capstyle='round')

    text(.55, 5.86, 'Method 2 · 白化与 z-score', 27, align='left', weight='bold')
    text(.57, 5.36, '在原始匹配分数上作组内修正，再选出 Top-K 记忆', 14, MUTED, align='left')
    step(.73, 1, '已有查询与记忆')
    step(5.55, 2, '组内调整分数')
    step(10.64, 3, '全局重新排序')

    # Existing query construction is deliberately abstracted away.
    rect(1.27, 3.94, 2.25, .57, '#EEF3F9', radius=.16)
    ax.add_patch(Polygon([(1.64, 3.95), (1.81, 3.95), (1.64, 3.80)],
                         facecolor='#EEF3F9', edgecolor='none'))
    text(2.395, 4.225, '已有检索 query', 14)
    for y, label, members, fill, color in [
        (2.55, '组 1', 'AB', '#EEF5FD', BLUE),
        (1.07, '组 2', 'CD', '#FFF6E9', ORANGE),
    ]:
        rect(.55, y, 3.65, 1.25, fill, radius=.16)
        text(1.0, y + .63, label, 14, color)
        for j, letter in enumerate(members):
            memory_card(1.47 + j * 1.32, y + .21, letter)
    arrow((4.39, 2.63), (5.03, 2.63))

    # For a non-tied two-leaf group, white z-scores are +/-1.
    # Each pair is realizable with x=[r,sqrt(1-r*r)], q=[1,0].
    # Alpha=1; delta=std_pop(raw)*tanh(z). No dataset is being reported.
    raw = {'A': .70, 'B': .66, 'C': .68, 'D': .25}
    corrected = {}
    for high, low in (('A', 'B'), ('C', 'D')):
        delta = (raw[high] - raw[low]) / 2 * tanh(1)
        corrected[high], corrected[low] = raw[high] + delta, raw[low] - delta

    rect(5.23, 1.06, 4.31, 3.42, '#FAF7FD', radius=.2)
    text(7.385, 4.16, 'Ridge 白化 + z-score', 15, PURPLE, weight='bold')
    text(6.615, 3.67, '原始', 12, MUTED)
    text(8.665, 3.67, '修正后', 12, PURPLE)
    row_y = {'A': 3.18, 'B': 2.66, 'C': 1.98, 'D': 1.46}
    for letter in 'ABCD':
        y, color = row_y[letter], COLORS[letter]
        text(5.62, y, letter, 16, color, weight='bold')
        rect(5.94, y - .105, 1.35, .21, '#ECEAF0', radius=.07)
        rect(5.94, y - .105, raw[letter] * 1.35, .21, '#B9C2CF', radius=.07)
        arrow((7.49, y), (7.84, y), '#B6A2D1', width=1.3, scale=11)
        rect(7.99, y - .105, 1.35, .21, '#ECEAF0', radius=.07)
        rect(7.99, y - .105, corrected[letter] * 1.35, .21, color, radius=.07)
    ax.plot([5.49, 9.25], [2.34, 2.34], color='#E4DDED', linewidth=1)
    text(7.385, .70, '每组全部记忆分别计算', 13, MUTED)
    arrow((9.72, 2.63), (10.37, 2.63))

    text(12.30, 4.20, 'Top-K（示意 K = 3）', 14, GREEN)
    ranked = sorted(corrected, key=lambda key: -corrected[key])
    for rank, letter in enumerate(ranked[:3], 1):
        y, color = 3.15 - (rank - 1) * .94, COLORS[letter]
        text(10.83, y + .34, str(rank), 22, GREEN, weight='bold')
        rect(11.20, y, 2.99, .70, '#F2F8F6', '#DCEBE5', radius=.12)
        rect(11.20, y, .075, .70, color, radius=.02)
        text(11.56, y + .35, letter, 19, color, weight='bold')
        for i, length in enumerate((1.55, 1.12)):
            ax.plot([11.98, 11.98 + length], [y + .43 - i * .18] * 2,
                    color='#BDCFC9', linewidth=3, solid_capstyle='round')
    text(12.30, .70, '合并各组，取分数最高的记忆', 13, MUTED)
    text(2.375, .70, '保留原有文本与分组', 13, MUTED)
    ax.plot([.55, 14.20], [.40, .40], color='#E5EAF0', linewidth=1)
    text(.57, .17, '卡片颜色表示所属组；条形与排名仅作流程示意。', 10, MUTED, align='left')

    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'framework.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', help='Path to a Chinese OTF/TTF/TTC font')
    main(parser.parse_args().font)
