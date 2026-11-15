"""Render the frozen Home/Q10 projection experiment as signed bubble charts.

Presentation only: no encoding, model requests, or retrieval; preserve source files.
"""
from pathlib import Path
import hashlib
import json
import os
import textwrap
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'reports/home_q10_query_attribute_layout_20261113'
OUT = ROOT / 'reports/home_q10_query_attribute_bubbles_20261114'
POSITIVE = '#2457A6'
NEGATIVE = '#CB6A27'
TEXT = '#172333'
MAX_AREA = 660.0


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def wrap(text, width):
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'completion_manifest.json').exists():
        raise FileExistsError('Completed bubble report exists; use a new version.')
    source_manifest = read(SOURCE / 'completion_manifest.json')
    source_manifest_hash = sha(SOURCE / 'completion_manifest.json')

    def check_source():
        assert sha(SOURCE / 'completion_manifest.json') == source_manifest_hash
        for filename, digest in source_manifest['outputs'].items():
            assert sha(SOURCE / filename) == digest, filename

    check_source()
    payload = read(SOURCE / 'figure_content.json')
    summary = read(Path(payload['source_root']) / 'numeric_summary.json')
    values = np.array(payload['numeric_values_x100'], dtype=float)
    assert values.shape == (2, 16, 6) and np.isfinite(values).all()
    # PC components and the outside-span component reconstruct total score change.
    np.testing.assert_allclose(values[:, :, :5].sum(axis=2), values[:, :, 5], atol=1e-10, rtol=0)
    vmax = float(abs(values).max())
    areas = MAX_AREA * abs(values) / vmax
    labels = ['\n'.join(wrap('"' + r['old'] + '" → "' + r['new'] + '"', 83)
                        for r in row['replacements']) for row in payload['rows']]
    qtexts = [payload['queries']['raw_query'], payload['queries']['frozen_master_query']]
    qheads = ['Query', 'Query + candidate answer (original master selection)']
    headers = [f'PC{j+1}\n{100*summary["variance_shares"][j]:.2f}' for j in range(4)] + ['Outside\n4 PCs', 'Total\nchange']
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14,
                         'axes.labelweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    heights = np.array([max(.64, .25 * len(label.splitlines()) + .12) for label in labels])
    edges = np.r_[0, np.cumsum(heights)]
    centers = (edges[:-1] + edges[1:]) / 2
    plot_height = float(edges[-1])
    bottom, top = 4.0, 2.45
    height = bottom + plot_height + top
    checks = []

    def draw(name, panels):
        left, label_width, gap, panel_width, panel_gap, right = .35, 9.8, .22, 5.1, .48, .35
        plot_start = left + label_width + gap
        width = plot_start + len(panels) * panel_width + (len(panels) - 1) * panel_gap + right
        fig = plt.figure(figsize=(width, height))
        lax = fig.add_axes([left / width, bottom / height, label_width / width, plot_height / height])
        lax.set_xlim(0, 1)
        lax.set_ylim(edges[-1], 0)
        lax.axis('off')
        label_artists = [lax.text(0, centers[i], label, fontsize=14, ha='left', va='center',
                                  linespacing=1.18, color=TEXT) for i, label in enumerate(labels)]
        fig.text(left / width, (bottom + plot_height + .28) / height, 'Raw content → Modified content',
                 fontsize=16, weight='bold', ha='left', va='bottom')

        # One common area scale for all panels and standalone exports. No minimum
        # bubble size and no marker stroke, so small effects are not inflated.
        legend = fig.add_axes([left / width, (bottom + plot_height + .88) / height, 9.5 / width, 1.40 / height])
        legend.set_xlim(0, 9.5)
        legend.set_ylim(0, 1.4)
        legend.axis('off')
        legend.scatter([.18, 4.30], [1.18, 1.18], s=95, c=[POSITIVE, NEGATIVE], edgecolors='none')
        legend.text(.40, 1.18, 'Positive: higher similarity', va='center', fontsize=14, weight='bold', color=TEXT)
        legend.text(4.52, 1.18, 'Negative: lower similarity', va='center', fontsize=14, weight='bold', color=TEXT)
        legend.text(0, .72, 'Magnitude\n(|Δ| ×100)', va='center', fontsize=13, weight='bold')
        reference = np.array([.1, .5, 1., 2.5])
        positions = np.array([1.9, 3.0, 4.1, 5.2])
        legend.scatter(positions, np.full(4, .72), s=MAX_AREA * reference / vmax, c='#728095', edgecolors='none')
        for x, value in zip(positions, reference):
            legend.text(x, .39, f'{value:.2f}', ha='center', va='center', fontsize=12)
        legend.text(0, .03, 'Area is proportional to |Δ similarity|. Labels: signed Δ ×100.', fontsize=13, va='bottom')

        query_artists, number_artists = [], []
        for pi, k in enumerate(panels):
            x = plot_start + pi * (panel_width + panel_gap)
            ax = fig.add_axes([x / width, bottom / height, panel_width / width, plot_height / height])
            ax.set_ylim(edges[-1], 0)
            ax.set_xlim(0, 6)
            ax.set_yticks([])
            ax.set_xticks(np.arange(6) + .5, headers)
            ax.xaxis.tick_top()
            ax.tick_params(axis='x', length=0, pad=8, labelsize=13)
            for tick in ax.get_xticklabels():
                tick.set_weight('bold')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.axvspan(5, 6, color='#F3F5F8', zorder=0)
            for j in [4, 5]:
                ax.axvline(j, color='#C1C8D2', lw=.7, zorder=0)
            xx, yy = np.meshgrid(np.arange(6) + .5, centers - .09)
            colors = np.where(values[k].ravel() >= 0, POSITIVE, NEGATIVE)
            dots = ax.scatter(xx.ravel(), yy.ravel(), s=areas[k].ravel(), c=colors, edgecolors='none', zorder=2)
            np.testing.assert_array_equal(dots.get_sizes(), areas[k].ravel())
            for i in range(16):
                for j in range(6):
                    t = ax.text(j + .5, centers[i] + .205, f'{values[k, i, j]:+.2f}',
                                ha='center', va='center', fontsize=11.5, weight='bold', color=TEXT)
                    number_artists.append((ax, i, j, t, areas[k, i, j]))
            fig.text(x / width, (bottom + plot_height + top - .15) / height, qheads[k],
                     fontsize=14, weight='bold', ha='left', va='top', color=POSITIVE)
            qt = fig.text(x / width, (bottom + plot_height + top - .53) / height,
                          '\n'.join(wrap(part, 55) for part in qtexts[k].splitlines()),
                          fontsize=15, ha='left', va='top', linespacing=1.15)
            query_artists.append(qt)

        caption = wrap(payload['original_caption'], int((width - left - .4) * 8.4))
        fig.text(left / width, 3.61 / height, 'Original caption — HOME_S7:R6', fontsize=16, weight='bold', va='top')
        caption_artist = fig.text(left / width, 3.27 / height, caption, fontsize=15, ha='left', va='top', linespacing=1.18)
        dialogue_y = 3.27 - len(caption.splitlines()) * .25 - .23
        dialogue_artist = fig.text(left / width, dialogue_y / height,
                                   'Original dialogue:\n' + payload['original_dialogue'],
                                   fontsize=13, ha='left', va='top', linespacing=1.18)
        footer = ('Positive = higher query similarity, not necessarily a correct answer. PC headers = variance share ×100. '
                  'Values rounded to two decimals. Two replacements in one row apply together.')
        if 1 in panels:
            footer += ' Original candidate B says neither desk; ground-truth answer is C (window desk).'
        footer_artist = fig.text(left / width, .18 / height, wrap(footer, int((width - left - .4) * 10)),
                                fontsize=11, color='#465260', va='bottom')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        issues = []
        for i, t in enumerate(label_artists):
            bb = t.get_window_extent(renderer)
            row_top = lax.transData.transform((0, edges[i]))[1]
            row_bottom = lax.transData.transform((0, edges[i + 1]))[1]
            if bb.y0 < row_bottom - 1 or bb.y1 > row_top + 1:
                issues.append(f'row {i+1}: label exceeds row height')
            if bb.x1 > lax.get_window_extent(renderer).x1 + 1:
                issues.append(f'row {i+1}: label exceeds left column')
        for qt in query_artists:
            if qt.get_window_extent(renderer).y0 < fig.transFigure.transform((0, (bottom + plot_height + .69) / height))[1]:
                issues.append('query overlaps PC column labels')
        for ax, i, j, t, area in number_artists:
            bb = t.get_window_extent(renderer)
            dot_center = ax.transData.transform((j + .5, centers[i] - .09))
            radius = np.sqrt(area) / 2 * fig.dpi / 72
            row_top = ax.transData.transform((0, edges[i]))[1]
            row_bottom = ax.transData.transform((0, edges[i + 1]))[1]
            if dot_center[1] - radius < bb.y1 - 1:
                issues.append(f'row {i+1}, col {j+1}: bubble overlaps number')
            if bb.y0 < row_bottom - 1 or dot_center[1] + radius > row_top + 1:
                issues.append(f'row {i+1}, col {j+1}: bubble or number exceeds row')
        if caption_artist.get_window_extent(renderer).y0 < dialogue_artist.get_window_extent(renderer).y1:
            issues.append('caption overlaps dialogue')
        if dialogue_artist.get_window_extent(renderer).y0 < footer_artist.get_window_extent(renderer).y1 + 8:
            issues.append('dialogue overlaps footer')
        assert not issues, issues
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=160, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
        checks.append({'figure': name, 'panels': panels, 'exact_edit_rows': 16, 'layout_issues': issues})

    draw('query_attribute_bubbles_with_text', [0, 1])
    draw('raw_query_bubbles_with_text', [0])
    draw('master_query_bubbles_with_text', [1])

    # All original editable numerical data and the native score-change bar chart
    # remain; only figure images and the presentation-specific sheets change.
    source_workbook = SOURCE / 'Home_Q10_query_attribute_projection_with_text.xlsx'
    wb = load_workbook(source_workbook)
    for old_name, new_name in [('RawHeatmap', 'RawBubbleData'), ('MasterHeatmap', 'MasterBubbleData')]:
        wb[old_name].title = new_name
        ws = wb[new_name]
        for rule_range in list(ws.conditional_formatting):
            del ws.conditional_formatting[str(rule_range.sqref)]
    for name, filename, display_width in [
        ('Figures', 'query_attribute_bubbles_with_text.png', 1650),
        ('RawFigure', 'raw_query_bubbles_with_text.png', 1250),
        ('MasterFigure', 'master_query_bubbles_with_text.png', 1250),
    ]:
        ws = wb[name]
        ws._images.clear()
        im = ExcelImage(OUT / filename)
        ratio = display_width / im.width
        im.width = display_width
        im.height *= ratio
        ws.add_image(im, 'A1')
    wb['Readme'].append(['气泡图', '面积正比于变化绝对值；蓝色=相似度增加，橙色=相似度减少。两种查询共用面积尺度；所有数值沿用原实验。'])
    wb['Readme'].append(['保留布局', '上方实际 query / query + 原候选陈述；左侧精确原文→修改后文本；下方完整 caption 和对话。'])
    destination = OUT / 'Home_Q10_query_attribute_bubbles.xlsx'
    wb.save(destination)
    new, old = load_workbook(destination), load_workbook(source_workbook)
    preserved = ['Edits', 'Contributions', 'OriginalPair', 'OriginalScores', 'Energy', 'Spectrum', 'ScoreChangeChart']
    for name in preserved:
        assert list(new[name].values) == list(old[name].values), name
    for k, (old_name, new_name) in enumerate([('RawHeatmap', 'RawBubbleData'), ('MasterHeatmap', 'MasterBubbleData')]):
        assert list(new[new_name].values) == list(old[old_name].values)
        assert new[new_name]['A2'].value == qtexts[k]
        assert new[new_name]['A24'].value == payload['original_caption']
        assert new[new_name]['A27'].value == payload['original_dialogue']
        for i, row in enumerate(payload['rows'], 6):
            assert new[new_name].cell(i, 1).value == row['label']
            np.testing.assert_allclose([new[new_name].cell(i, j).value for j in range(2, 8)], values[k, i - 6], atol=1e-12, rtol=0)
    assert len(new['ScoreChangeChart']._charts) == 1
    assert sum(len(ws._images) for ws in new) == 3
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    payload['presentation_source_root'] = str(SOURCE)
    payload['bubble_encoding'] = {'area': 'max_area * abs(delta_x100) / max_abs_delta_x100',
                                   'max_area_points_squared': MAX_AREA, 'max_abs_delta_x100': vmax,
                                   'positive_color': POSITIVE, 'negative_color': NEGATIVE,
                                   'shared_scale': True, 'minimum_area': 0, 'marker_edge_width': 0}
    payload['bubble_areas_points_squared'] = areas.tolist()
    dump(OUT / 'figure_content.json', payload)
    body = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Home Q10 属性投影气泡图</title>
<style>body{font:16px/1.7 system-ui;color:#182435;margin:24px}img{width:100%;height:auto}main{max-width:1800px;margin:auto}a{color:#2457A6}details{margin:20px 0}</style><main>
<p>Home/Q10：将热力图改为气泡图，实验数据保持不变。气泡面积与变化绝对值成正比；蓝色表示相似度增加，橙色表示减少。两种 query 共用面积尺度。</p>
<img src="query_attribute_bubbles_with_text.png" alt="两种实际查询的属性投影气泡图">
<p><a href="query_attribute_bubbles_with_text.pdf">PDF</a> · <a href="query_attribute_bubbles_with_text.svg">SVG</a> · <a href="Home_Q10_query_attribute_bubbles.xlsx">Excel：图与可编辑数据</a></p>
<details><summary>单独查看原始 query</summary><img src="raw_query_bubbles_with_text.png" alt="原始 query 气泡图"></details>
<details><summary>单独查看 query + candidate answer</summary><img src="master_query_bubbles_with_text.png" alt="原 master 查询气泡图"></details>
<p>上方是完整实际查询，左侧是精确原文 → 修改后文本，下方是完整原 caption 和对话。PC 列头为方差占比 ×100；气泡下方标注为带符号分数变化 ×100，保留两位小数。一行两处替换时，两处同时生效。</p>
<p>右栏保留原 master 选择的错误陈述 B（两桌均无显示器）；正确答案为 C（靠窗桌）。相似度增加不等于答案更正确。</p>
<p><a href="figure_content.json">完整图中文字与数值</a> · <a href="layout_audit.json">核验记录</a> · <a href="../home_q10_query_attribute_layout_20261113/report.html">此前热力图</a></p></main></html>'''
    (OUT / 'report.html').write_text(body, encoding='utf-8')
    check_source()
    audit = {'status': 'passed', 'layout_checks': checks, 'original_files_unchanged': len(source_manifest['outputs']),
             'numeric_sheets_identical': preserved + ['RawBubbleData', 'MasterBubbleData'],
             'numeric_values_unchanged': True, 'exact_edit_rows': 16, 'complete_query_inputs': 2,
             'full_caption_and_dialogue_below': True, 'shared_bubble_area_scale': True,
             'area_proportional_to_absolute_change': True, 'bubble_number_overlap_issues': 0,
             'embedded_figures': 3, 'native_charts_preserved': 1, 'new_model_calls': 0, 'new_retrieval_runs': 0,
             'source_completion_manifest_sha256': source_manifest_hash}
    dump(OUT / 'layout_audit.json', audit)
    dump(OUT / 'completion_manifest.json', {'status': 'complete', 'source_root': str(SOURCE),
         'source_manifest_sha256': source_manifest_hash, 'renderer_sha256': sha(__file__),
         'outputs': {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['completion_manifest.json', 'render.log']}})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
