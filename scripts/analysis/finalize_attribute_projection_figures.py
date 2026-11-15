"""Finalize plot layout and evidence notes without changing frozen probe inputs."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/gpt5_attribute_projection_20261113'


def main():
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 13,
                         'axes.labelweight': 'bold', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    summary = json.loads((OUT / 'numeric_summary.json').read_text())
    rows = summary['attributes']
    matrix = np.array([[r[f'PC{j+1:02d}']*100 for j in range(47)] for r in rows])
    outside = np.array([r['outside']*100 for r in rows])
    v = np.load(OUT / 'analysis_vectors.npz')['eigenvalues']
    labels = ['Meal Variability [requested]', 'Evening Drift [requested]',
              'Time In Range [other]', 'Morning Response [other]', 'Late Window [other]',
              'Theme: dark to light', 'Caption verb paraphrase', 'Assistant acknowledgment']
    fig = plt.figure(figsize=(18.5, 7.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[47, 1.1, 5], height_ratios=[1, 5],
                          left=.24, right=.97, top=.98, bottom=.14, wspace=.17, hspace=.16)
    top = fig.add_subplot(gs[0, 0])
    top.bar(range(47), 100*v/v.sum(), color=['#2457A6']*10+['#789BC6']*27+['#C1D1E7']*10)
    top.set_xlim(-.5, 46.5); top.set_xticks([]); top.set_yticks([0, 25, 50])
    top.set_ylim(0, 52); top.set_ylabel('Variance\nshare (×100)', fontsize=11)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=float(matrix.max()))
    ax.set_yticks(range(8), labels)
    ax.set_xticks([0, 9, 19, 29, 36, 46], [1, 10, 20, 30, 37, 47])
    ax.set_xlabel('Original principal component (descending variance)')
    for boundary in [9.5, 36.5]: ax.axvline(boundary, color='#333333', lw=1)
    for label in ax.get_yticklabels()[:2]: label.set_color('#2457A6'); label.set_weight('bold')
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=11)
    cb.set_label('Edit energy in one PC (×100)', fontsize=11, weight='bold')
    outside_ax = fig.add_subplot(gs[1, 2])
    outside_ax.imshow(outside[:, None], aspect='auto', cmap='Blues', vmin=0, vmax=100)
    outside_ax.set_yticks([]); outside_ax.set_xticks([0], ['Outside\n47 PCs'])
    for i, value in enumerate(outside):
        outside_ax.text(0, i, f'{value:.2f}', ha='center', va='center', weight='bold',
                        color='white' if value > 60 else 'black', fontsize=12)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'attribute_pc_heatmap.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    workbook = OUT / 'Attribute_projection_heatmap.xlsx'
    wb = load_workbook(workbook)
    ws = wb['Figures']; ws._images.clear()
    for file, anchor, width in [('attribute_pc_heatmap.png', 'A1', 1250), ('attribute_band_heatmap.png', 'A40', 950)]:
        image = Image(OUT / file); ratio = width/image.width; image.width = width; image.height *= ratio
        ws.add_image(image, anchor)
    wb.save(workbook)
    notes = {
        'status': 'exploratory_pilot_not_supporting_simple_high_irrelevant_low_detail_claim',
        'requested_numeric_mean': {k: sum(r[k] for r in rows[:2])/2 for k in ['high', 'middle', 'low', 'outside']},
        'other_numeric_mean': {k: sum(r[k] for r in rows[2:5])/3 for k in ['high', 'middle', 'low', 'outside']},
        'interpretation': [
            '所问数值改动与其他数值改动的分布接近：并未出现所问数值明显集中在低方差段的现象。',
            'caption 动词与助手应答改动在高方差段上占比高于所问数值改动，但这只支持部分表达变化投向前列方向。',
            '所问数值改动约 69.51% 的能量落在原簇 47 维子空间之外。该能量不等于 PC38–47，不应重新归一化后隐藏。',
            '原图没有被编辑。该实验检验 BGE 对合成文字改动的响应，不能直接判定视觉编码器是否识别细节。',
            '主题词改动不等同于真实图像布局或整类页面变化；这个探针没有单独操纵布局。',
            '当前结果不支持简单的高方差等于无关、低方差等于细节叙事；也不足以排除其他簇、其他自然细节对照中的条件性作用。',
        ],
        'followup': '优先寻找同一布局、同一对话与模板下的真实记忆配对，再分析实际数字差异；保持原始基底，避免只依据合成词替换推断图像语义。',
        'figure_finalizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / 'interpretation.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2)+'\n')
    report = (OUT / 'report.html').read_text()
    marker = '<h2>主成分投影</h2>'
    block = ('<h2>这一版观察到了什么</h2><p><b>未支持“高方差无关、低方差为关键细节”的简单划分。</b>'
             '所问数值改动在高/低方差段分别占 10.99 / 4.73；其他数值为 9.67 / 5.02（均为能量比例乘 100）。'
             '所问数值约 69.51 的能量在原簇子空间外。表达改动在高方差段更突出，但不能由此把全部高方差方向视为无关。</p>'
             '<p>主题词替换没有操纵整页布局；合成文字探针也不等于自然图像变化。详细解释见 <a href="interpretation.json">观察结论及边界</a>。</p>')
    assert marker in report
    (OUT / 'report.html').write_text(report.replace(marker, block+marker, 1))
    print(json.dumps({'status': 'figures_finalized', 'notes': notes}, ensure_ascii=False))


if __name__ == '__main__': main()
