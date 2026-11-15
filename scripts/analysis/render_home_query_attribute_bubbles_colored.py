"""Presentation revision: 2 pt coordinate axes, Times New Roman 16 pt.

Uses the completed bubble report without re-encoding or changing its data.
"""
from pathlib import Path
from copy import copy
import json
import textwrap

from render_home_query_attribute_bubbles import (
    ROOT, POSITIVE, NEGATIVE, TEXT, read, sha, dump, np, plt,
    font_manager, load_workbook, ExcelImage,
)
from matplotlib.text import Text

SOURCE = ROOT / 'reports/home_q10_query_attribute_bubbles_20261114'
OUT = ROOT / 'reports/home_q10_query_attribute_bubbles_colored_20261114'
FONT_SIZE = 16
AXIS_WIDTH = 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'completion_manifest.json').exists():
        raise FileExistsError('Completed report exists; use a new output version.')
    source_manifest = read(SOURCE / 'completion_manifest.json')
    manifest_hash = sha(SOURCE / 'completion_manifest.json')

    def check_source():
        assert sha(SOURCE / 'completion_manifest.json') == manifest_hash
        for name, digest in source_manifest['outputs'].items():
            assert sha(SOURCE / name) == digest, name

    check_source()
    payload = read(SOURCE / 'figure_content.json')
    stimuli = read(Path(payload['source_root']) / 'stimuli.json')
    roles = {s['id']: s['role'] for s in stimuli}
    def label_color(role):
        if role in ['answer_fact', 'answer_relation']:
            return '#2457A6'
        if role == 'episode_locator':
            return '#87631F'
        assert role in ['answer_invariant_visual', 'expression_control'], role
        return '#000000'
    label_colors = [label_color(roles[row['id']]) for row in payload['rows']]
    summary = read(Path(payload['source_root']) / 'numeric_summary.json')
    values = np.array(payload['numeric_values_x100'])
    areas = np.array(payload['bubble_areas_points_squared'])
    encoding = payload['bubble_encoding']
    assert values.shape == areas.shape == (2, 16, 6)
    labels = ['\n'.join(textwrap.fill('"' + r['old'] + '" → "' + r['new'] + '"',
                                     width=83, break_long_words=False, break_on_hyphens=False)
                        for r in row['replacements']) for row in payload['rows']]
    headers = [f'PC{j+1}\n{100*summary["variance_shares"][j]:.2f}' for j in range(4)] + ['Outside\n4 PCs', 'Total\nchange']
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': FONT_SIZE,
                         'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
                         'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
                         'axes.titlesize': FONT_SIZE, 'axes.labelweight': 'bold',
                         'axes.linewidth': AXIS_WIDTH, 'xtick.major.width': AXIS_WIDTH,
                         'ytick.major.width': AXIS_WIDTH, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    heights = np.array([max(.8, .29 * len(label.splitlines()) + .16) for label in labels])
    edges = np.r_[0, np.cumsum(heights)]
    centers = (edges[:-1] + edges[1:]) / 2
    plot_height = float(edges[-1])
    bottom, top = .92, 1.48
    height = bottom + plot_height + top
    checks = []

    def draw(name, panels):
        label_width, panel_width, panel_gap, right = 11.2, 6.3, .65, .35
        width = label_width + len(panels) * panel_width + (len(panels) - 1) * panel_gap + right
        fig = plt.figure(figsize=(width, height))
        axes, number_artists = [], []
        # Keep brief panel identifiers only; the query, caption, and dialogue
        # text blocks from the previous figure are deliberately not rendered.
        fig.text(.2 / width, (bottom + plot_height + .18) / height,
                 'Raw content → Modified content', va='bottom', weight='bold', color=TEXT)
        legend = fig.add_axes([.2 / width, (bottom + plot_height + .64) / height,
                              (width - .55) / width, .75 / height])
        legend.set_xlim(0, width - .55)
        legend.set_ylim(0, .75)
        legend.axis('off')
        legend.scatter([.16, 4.4], [.48, .48], s=105, c=[POSITIVE, NEGATIVE], edgecolors='none')
        legend.text(.4, .48, 'Higher similarity', va='center', weight='bold')
        legend.text(4.64, .48, 'Lower similarity', va='center', weight='bold')
        legend.text(8.35, .48, 'Magnitude (|Δ| ×100)', va='center', weight='bold')
        refs = np.array([.1, .5, 1., 2.5])
        positions = np.array([11.1, 12.3, 13.5, 14.7])
        legend.scatter(positions, np.full(4, .48),
                       s=encoding['max_area_points_squared'] * refs / encoding['max_abs_delta_x100'],
                       c='#728095', edgecolors='none')
        for x, value in zip(positions, refs):
            legend.text(x, .03, f'{value:.2f}', va='bottom', ha='center')
        for pi, k in enumerate(panels):
            x = label_width + pi * (panel_width + panel_gap)
            ax = fig.add_axes([x / width, bottom / height, panel_width / width, plot_height / height])
            axes.append(ax)
            ax.set_xlim(0, 6)
            ax.set_ylim(edges[-1], 0)
            ax.set_xticks(np.arange(6) + .5, headers)
            ax.set_yticks(centers, labels if pi == 0 else [''] * len(labels))
            ax.tick_params(axis='both', width=AXIS_WIDTH, length=6, pad=10, direction='out')
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            for tick, color in zip(ax.get_yticklabels(), label_colors):
                tick.set_linespacing(1.18)
                tick.set_color(color)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(AXIS_WIDTH)
                spine.set_color('#000000')
            ax.grid(False)
            ax.axvspan(5, 6, color='#F3F5F8', zorder=0)
            for j in [4, 5]:
                ax.axvline(j, color='#C1C8D2', lw=.7, zorder=0)
            xx, yy = np.meshgrid(np.arange(6) + .5, centers - .12)
            dots = ax.scatter(xx.ravel(), yy.ravel(), s=areas[k].ravel(),
                              c=np.where(values[k].ravel() >= 0, POSITIVE, NEGATIVE),
                              edgecolors='none', zorder=2)
            np.testing.assert_array_equal(dots.get_sizes(), areas[k].ravel())
            for i in range(16):
                for j in range(6):
                    artist = ax.text(j + .5, centers[i] + .22, f'{values[k, i, j]:+.2f}',
                                     ha='center', va='center', weight='bold', color=TEXT)
                    number_artists.append((ax, i, j, artist, areas[k, i, j]))
            panel_name = '(a) Raw-query projection' if k == 0 else '(b) Master-query projection'
            ax.set_title(panel_name, pad=14, weight='bold', color=TEXT)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        issues = []
        for ax in axes:
            assert [t.get_color() for t in ax.get_yticklabels()] == label_colors
            assert all(s.get_visible() and s.get_linewidth() == AXIS_WIDTH for s in ax.spines.values())
            for axis in [ax.xaxis, ax.yaxis]:
                assert all(t.tick1line.get_markeredgewidth() == AXIS_WIDTH for t in axis.get_major_ticks())
            for i, artist in enumerate(ax.get_yticklabels()):
                if not artist.get_text():
                    continue
                bb = artist.get_window_extent(renderer)
                row_top = ax.transData.transform((0, edges[i]))[1]
                row_bottom = ax.transData.transform((0, edges[i + 1]))[1]
                if bb.x0 < 0 or bb.y0 < row_bottom - 1 or bb.y1 > row_top + 1:
                    issues.append(f'row {i+1}: edit label overflow')
        for ax, i, j, artist, area in number_artists:
            bb = artist.get_window_extent(renderer)
            dot_y = ax.transData.transform((j + .5, centers[i] - .12))[1]
            radius = np.sqrt(area) / 2 * fig.dpi / 72
            row_top = ax.transData.transform((0, edges[i]))[1]
            row_bottom = ax.transData.transform((0, edges[i + 1]))[1]
            if dot_y - radius < bb.y1 - 1 or bb.y0 < row_bottom - 1 or dot_y + radius > row_top + 1:
                issues.append(f'row {i+1}, col {j+1}: bubble or number overflow')
        texts = [t for t in fig.findobj(Text) if t.get_text() and t.get_visible()]
        assert all(t.get_fontsize() == FONT_SIZE for t in texts)
        assert all(t.get_fontfamily() == ['Times New Roman'] for t in texts)
        forbidden = [payload['queries']['raw_query'], payload['queries']['frozen_master_query'],
                     payload['original_caption'], payload['original_dialogue']]
        rendered = '\n'.join(t.get_text() for t in texts)
        assert all(text not in rendered for text in forbidden)
        assert not issues, issues
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight', pad_inches=.15)
        plt.close(fig)
        checks.append({'figure': name, 'panels': panels, 'font': 'Times New Roman', 'font_size_pt': FONT_SIZE,
                       'axis_width_pt': AXIS_WIDTH, 'tick_width_pt': AXIS_WIDTH, 'layout_issues': issues})

    draw('query_attribute_bubbles_axes16', [0, 1])
    draw('raw_query_bubbles_axes16', [0])
    draw('master_query_bubbles_axes16', [1])
    original_book = SOURCE / 'Home_Q10_query_attribute_bubbles.xlsx'
    wb = load_workbook(original_book)
    for name in ['RawBubbleData', 'MasterBubbleData']:
        for i, color in enumerate(label_colors, 6):
            font = copy(wb[name].cell(i, 1).font)
            font.color = color.lstrip('#')
            wb[name].cell(i, 1).font = font
    for sheet, filename, display_width in [
        ('Figures', 'query_attribute_bubbles_axes16.png', 1650),
        ('RawFigure', 'raw_query_bubbles_axes16.png', 1250),
        ('MasterFigure', 'master_query_bubbles_axes16.png', 1250),
    ]:
        ws = wb[sheet]
        ws._images.clear()
        im = ExcelImage(OUT / filename)
        im.height *= display_width / im.width
        im.width = display_width
        ws.add_image(im, 'A1')
    wb['Readme'].append(['最新绘图格式', '坐标轴与刻度线 2 pt；Times New Roman 16 pt；图中去除 query、caption 和 dialogue 文本块。原始数据不变。'])
    wb['Readme'].append(['左侧文字颜色', '蓝色：答案事实或关系变化；棕黄色：事件阶段变化；黑色：答案事实不变。颜色为人工语义分类，与气泡正负颜色含义不同。'])
    destination = OUT / 'Home_Q10_query_attribute_bubbles_axes16.xlsx'
    wb.save(destination)
    new, old = load_workbook(destination), load_workbook(original_book)
    preserved = [name for name in old.sheetnames if name != 'Readme']
    for name in preserved:
        assert list(new[name].values) == list(old[name].values), name
    assert sum(len(ws._images) for ws in new) == 3
    assert len(new['ScoreChangeChart']._charts) == 1
    payload['presentation_source_root'] = str(SOURCE)
    payload['row_label_colors'] = [{'id': row['id'], 'role': roles[row['id']], 'color': color}
                                   for row, color in zip(payload['rows'], label_colors)]
    payload['presentation_style'] = {'font': 'Times New Roman', 'font_size_pt': 16, 'axis_width_pt': 2,
                                     'query_text_shown': False, 'original_caption_shown': False,
                                     'original_dialogue_shown': False, 'edit_labels_shown': True}
    dump(OUT / 'figure_content.json', payload)
    (OUT / 'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 气泡图</title>
<style>body{margin:24px;font:16px/1.6 system-ui}img{width:100%;height:auto}a{color:#2457A6}</style>
<p>2 pt 坐标轴与刻度；Times New Roman 16 pt。图中去除 query、原始 caption 及 dialogue 文本块，保留原文→修改后文本标签；实验数值、气泡面积和颜色保持不变。</p>
<p>左侧文字颜色：<span style="color:#2457A6">蓝色＝答案事实或关系变化</span>；<span style="color:#87631F">棕黄色＝事件阶段变化</span>；黑色＝答案事实不变。这是人工语义分类，与气泡颜色含义不同。</p>
<img src="query_attribute_bubbles_axes16.png" alt="带坐标系的 Home Q10 气泡图">
<p><a href="query_attribute_bubbles_axes16.pdf">PDF</a> · <a href="query_attribute_bubbles_axes16.svg">SVG</a> · <a href="Home_Q10_query_attribute_bubbles_axes16.xlsx">Excel</a></p>
<p>(a) 使用原始 query 的投影；(b) 使用原 master query 的投影。气泡面积表示贡献变化绝对值；蓝色相似度增加，橙色相似度减少。数值为贡献变化 ×100；PC 列头数字为方差占比 ×100。</p></html>''', encoding='utf-8')
    check_source()
    audit = {'status': 'passed', 'layout_checks': checks, 'numeric_values_unchanged': True,
             'row_label_colors': payload['row_label_colors'],
             'bubble_areas_unchanged': True, 'colors_unchanged': True, 'source_files_unchanged': len(source_manifest['outputs']),
             'workbook_data_sheets_unchanged': preserved, 'embedded_images': 3, 'native_charts_preserved': 1,
             'query_text_shown': False, 'caption_text_shown': False, 'dialogue_text_shown': False,
             'new_model_calls': 0, 'new_retrieval_runs': 0}
    dump(OUT / 'layout_audit.json', audit)
    dump(OUT / 'completion_manifest.json', {'status': 'complete', 'source_root': str(SOURCE),
         'source_manifest_sha256': manifest_hash, 'renderer_sha256': sha(__file__),
         'outputs': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'completion_manifest.json'}})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
