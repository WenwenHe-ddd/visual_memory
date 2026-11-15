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
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter

SOURCE = ROOT / 'reports/home_q10_query_attribute_heatmap_rdbu18_20261114'
OUT = ROOT / 'reports/home_q10_query_attribute_heatmap_values18_20261114'
FONT_SIZE = 18
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
    # Restore the previously used reference palette without changing the scale.
    cmap = plt.get_cmap('RdBu_r')
    vmin, vmax = float(values.min()), float(values.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
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
    heights = np.array([max(.8, .33 * len(label.splitlines()) + .16) for label in labels])
    edges = np.r_[0, np.cumsum(heights)]
    centers = (edges[:-1] + edges[1:]) / 2
    plot_height = float(edges[-1])
    bottom, top = 1.0, 2.0
    height = bottom + plot_height + top
    checks = []

    def draw(name, panels):
        label_width, panel_width, panel_gap, right = 12.6, 6.3, .65, .35
        width = label_width + len(panels) * panel_width + (len(panels) - 1) * panel_gap + right
        fig = plt.figure(figsize=(width, height))
        axes = []
        # Keep brief panel identifiers only; the query, caption, and dialogue
        # text blocks from the previous figure are deliberately not rendered.
        edit_heading = fig.text((label_width / 2) / width, (bottom + plot_height + .18) / height,
                 'Raw content → Modified content', ha='center', va='bottom', weight='bold', color=TEXT)
        # A single shared colorbar centered over the heatmap panels.
        plot_span = len(panels) * panel_width + (len(panels) - 1) * panel_gap
        legend_width = min(6.0, plot_span - .3)
        legend_left = label_width + (plot_span - legend_width) / 2
        cax = fig.add_axes([legend_left / width, (bottom + plot_height + 1.15) / height,
                            legend_width / width, .18 / height])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation='horizontal', ticks=[vmin, 0, vmax])
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cb.ax.xaxis.set_label_position('top')
        cb.set_label('Score change (×100)', fontsize=FONT_SIZE, weight='bold', labelpad=6)
        cb.ax.tick_params(width=2, length=4, labelsize=FONT_SIZE, pad=4)
        cb.outline.set_linewidth(2)
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
            mesh = ax.pcolormesh(np.arange(7), edges, values[k], cmap=cmap, norm=norm,
                                 shading='flat', edgecolors='none', rasterized=False)
            np.testing.assert_array_equal(np.asarray(mesh.get_array()).reshape(16, 6), values[k])
            for i in range(16):
                for j in range(6):
                    value = float(values[k, i, j])
                    rgb = np.array(cmap(norm(value))[:3])
                    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
                    luminance = float(linear @ np.array([.2126, .7152, .0722]))
                    text_color = '#FFFFFF' if luminance < .179 else '#000000'
                    ax.text(j + .5, centers[i], f'{value:+.2f}', ha='center', va='center',
                            fontsize=FONT_SIZE, weight='bold', color=text_color)
            for j in [4, 5]:
                ax.axvline(j, color='#C1C8D2', lw=.7, zorder=0)
            panel_name = '(a) Raw-query projection' if k == 0 else '(b) Master-query projection'
            ax.set_title(panel_name, pad=14, weight='bold', color=TEXT)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Center over the actual text extents, rather than the blank margin.
        label_boxes = [t.get_window_extent(renderer) for t in axes[0].get_yticklabels()]
        text_center = (min(bb.x0 for bb in label_boxes) + max(bb.x1 for bb in label_boxes)) / 2
        edit_heading.set_x(fig.transFigure.inverted().transform((text_center, 0))[0])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        heading_box = edit_heading.get_window_extent(renderer)
        assert abs((heading_box.x0 + heading_box.x1) / 2 - text_center) < 1
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
        for ax, k in zip(axes, panels):
            assert len(ax.texts) == 96
            for index, artist in enumerate(ax.texts):
                i, j = divmod(index, 6)
                assert artist.get_text() == f'{values[k, i, j]:+.2f}'
                bb = artist.get_window_extent(renderer)
                lower = ax.transData.transform((j, edges[i + 1]))
                upper = ax.transData.transform((j + 1, edges[i]))
                if bb.x0 < lower[0] or bb.x1 > upper[0] or bb.y0 < lower[1] or bb.y1 > upper[1]:
                    issues.append(f'row {i+1}, column {j+1}: numeric label overflow')
        assert abs((legend_left + legend_width / 2) - (label_width + plot_span / 2)) < 1e-12
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

    draw('query_attribute_heatmap18', [0, 1])
    draw('raw_query_heatmap18', [0])
    draw('master_query_heatmap18', [1])
    original_book = SOURCE / 'Home_Q10_query_attribute_heatmap18.xlsx'
    wb = load_workbook(original_book)
    for name in ['RawBubbleData', 'MasterBubbleData']:
        for i, color in enumerate(label_colors, 6):
            font = copy(wb[name].cell(i, 1).font)
            font.color = color.lstrip('#')
            wb[name].cell(i, 1).font = font
    for sheet, filename, display_width in [
        ('Figures', 'query_attribute_heatmap18.png', 1650),
        ('RawFigure', 'raw_query_heatmap18.png', 1250),
        ('MasterFigure', 'master_query_heatmap18.png', 1250),
    ]:
        ws = wb[sheet]
        ws._images.clear()
        im = ExcelImage(OUT / filename)
        im.height *= display_width / im.width
        im.width = display_width
        ws.add_image(im, 'A1')
    wb['Readme'].append(['最新绘图格式', '热力图格内添加带正负号、两位小数的数值（贡献变化×100），18 pt 新罗马；根据背景深浅选择黑/白文字。原色标、布局、2 pt 坐标轴及完整数据不变。'])
    wb['Readme'].append(['左侧文字颜色', '蓝色：答案事实或关系变化；棕黄色：事件阶段变化；黑色：答案事实不变。颜色为人工语义分类，与气泡正负颜色含义不同。'])
    wb['Readme'].append(['最新热力图配色', f'RdBu_r：负值蓝色、零附近白色、正值红色。共用原范围 [{vmin:.8f}, {vmax:.8f}]（分数变化×100）；负、正区间分别线性映射，文字分类颜色不变。'])
    destination = OUT / 'Home_Q10_query_attribute_heatmap18.xlsx'
    wb.save(destination)
    new, old = load_workbook(destination), load_workbook(original_book)
    preserved = [name for name in old.sheetnames if name != 'Readme']
    for name in preserved:
        assert list(new[name].values) == list(old[name].values), name
    assert sum(len(ws._images) for ws in new) == 3
    assert len(new['ScoreChangeChart']._charts) == 1
    payload['presentation_source_root'] = str(SOURCE)
    payload['bubble_encoding'].pop('positive_color', None)
    payload['bubble_encoding'].pop('negative_color', None)
    payload['bubble_encoding']['color'] = {'colormap': cmap.name,
        'reference_image': '/storage/hewenwen/.codex/attachments/f288c580-6580-4cc0-a8f7-4c4e0233f914/codex-clipboard-c71bd4c6-7afa-4be8-8aa2-771e408e7048.png',
        'normalization': 'TwoSlopeNorm',
        'vmin': vmin, 'vcenter': 0, 'vmax': vmax, 'units': 'score change x100',
        'shared_across_all_panels': True, 'negative': 'blue', 'zero': 'near-white', 'positive': 'red'}
    payload['row_label_colors'] = [{'id': row['id'], 'role': roles[row['id']], 'color': color}
                                   for row, color in zip(payload['rows'], label_colors)]
    payload['presentation_style'] = {'font': 'Times New Roman', 'font_size_pt': FONT_SIZE, 'axis_width_pt': 2,
                                     'chart_type': 'heatmap', 'edit_heading_centered_over_text': True,
                                     'size_legend_shown': False,
                                     'cell_value_labels_shown': True, 'cell_value_format': '+.2f', 'legends_centered_above_plot': True,
                                     'query_text_shown': False, 'original_caption_shown': False,
                                     'original_dialogue_shown': False, 'edit_labels_shown': True}
    dump(OUT / 'figure_content.json', payload)
    (OUT / 'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 气泡图</title>
<style>body{margin:24px;font:16px/1.6 system-ui}img{width:100%;height:auto}a{color:#2457A6}</style>
<p>热力图格内已填入数值（贡献变化×100，带正负号、两位小数），深色背景用白字、浅色背景用黑字。保留 RdBu_r 蓝—白—红渐变、原归一化、文本分类颜色、Times New Roman 18 pt 及 2 pt 坐标轴。完整数值保存在 Excel 的 RawBubbleData、MasterBubbleData 和 Contributions 工作表中。</p>
<p>左侧文字颜色：<span style="color:#2457A6">蓝色＝答案事实或关系变化</span>；<span style="color:#87631F">棕黄色＝事件阶段变化</span>；黑色＝答案事实不变。这是人工语义分类，与气泡颜色含义不同。</p>
<img src="query_attribute_heatmap18.png" alt="Home Q10 热力图">
<p><a href="query_attribute_heatmap18.pdf">PDF</a> · <a href="query_attribute_heatmap18.svg">SVG</a> · <a href="Home_Q10_query_attribute_heatmap18.xlsx">Excel</a></p>
<p>(a) 使用原始 query 的投影；(b) 使用原 master query 的投影。颜色表示贡献变化 ×100：正值红色、负值蓝色、零附近白色；PC 列头数字为方差占比 ×100。</p></html>''', encoding='utf-8')
    check_source()
    audit = {'status': 'passed', 'layout_checks': checks, 'numeric_values_unchanged': True,
             'chart_type': 'heatmap', 'edit_heading_centered_over_text': True, 'size_legend_shown': False,
             'cell_numeric_labels_shown': True, 'numeric_labels_per_panel': 96, 'legends_centered_above_plot': True,
             'row_label_colors': payload['row_label_colors'],
             'bubble_areas_unchanged': True, 'row_label_colors_unchanged': True,
             'bubble_color_mapping': payload['bubble_encoding']['color'],
             'source_files_unchanged': len(source_manifest['outputs']),
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
