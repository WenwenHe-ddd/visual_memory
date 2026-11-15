"""Render each observed scenario and question type as a separate figure."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'results/.matplotlib'))

TYPES = {
    'VS': ('Visual search', '视觉搜索'),
    'TTL': ('Test-time learning', '测试时学习'),
    'VR': ('Visual reasoning', '视觉推理'),
    'CD': ('Conflict detection', '冲突检测'),
    'KR': ('Knowledge resolution', '知识更新'),
}
SCENARIOS = {
    'AI_Robotics_Automation_Future_Tech': ('AI / robotics / future tech', 'AI、机器人与未来科技'),
    'Academic_Animal_Pet_Research_Life': ('Animal and pet research', '动物与宠物研究'),
    'Architecture_Art_Culture_Exhibition_Technology': ('Architecture / art / exhibitions', '建筑、艺术与展览'),
    'Astronomy_Physics_Scientific_Experiments_Cosmology': ('Astronomy / physics', '天文学与物理'),
    'Baking_Dessert_Daily_Life_Skill': ('Baking / desserts', '烘焙与甜点'),
    'Dog_Behavior_Research_Academic_Life': ('Dog behavior research', '犬类行为研究'),
    'Education_Career_Research_Lifestyle': ('Education / career', '教育与职业'),
    'Entrepreneurship_Blockchain_Economics_Logistics_Nature': ('Entrepreneurship / economics', '创业与经济'),
    'Fashion_Personal_Care_Lifestyle_Shopping': ('Fashion / personal care / shopping', '时尚、护理与购物'),
    'Global_Travel_Cultural_Sightseeing': ('Global travel / sightseeing', '全球旅行与文化观光'),
    'Global_Travel_Sustainable_Fashion_Design': ('Travel / sustainable fashion', '旅行与可持续时尚'),
    'Home_Health_Lifestyle_Product': ('Home / health / lifestyle', '家居、健康与生活'),
    'Landscape_Travel_Architecture_Nature': ('Landscape / travel / nature', '风景、旅行与自然'),
    'Music_Dance_Theater_Performance_Learning': ('Music / dance / theater', '音乐、舞蹈与戏剧'),
    'Nature_Economics_Programming_Student_Life': ('Nature / economics / programming', '自然、经济与编程'),
    'Parenting_Commuting_Hobbies_Travel_Gear': ('Parenting / commuting / travel gear', '育儿、通勤与旅行装备'),
    'Python_Botany_AI_Student_Life': ('Python / botany / AI', 'Python、植物学与 AI'),
    'Real_Estate_Home_Decor_DIY_Lifestyle': ('Real estate / home decor / DIY', '房产、家装与 DIY'),
    'Technology_Ethics_Future_Society': ('Technology / ethics / society', '科技、伦理与社会'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results/mem_gallery_distractor_injection/v1/summary.json')
    args = parser.parse_args()
    source = args.source.resolve()
    summary = json.loads(source.read_text())
    out = source.parent/'individual_figures'
    out.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 11})
    index = []
    for category, key, names in [('question_type', 'by_type', TYPES), ('scenario', 'by_scenario', SCENARIOS)]:
        directory = out/category
        directory.mkdir(exist_ok=True)
        for name, (english, chinese) in names.items():
            rows = [r for r in summary[key] if r['group'] == name]
            if not rows:
                continue
            assert len(rows) == 6
            by_condition = {c: sorted([r for r in rows if r['condition'] == c], key=lambda r: r['N'])
                            for c in ['same_scenario', 'other_scenarios']}
            for values in by_condition.values():
                assert [r['N'] for r in values] == [0, 10, 20]
                assert len({(r['targets'], r['questions']) for r in values}) == 1
            same, other = by_condition['same_scenario'], by_condition['other_scenarios']
            questions, targets = rows[0]['questions'], rows[0]['targets']
            fig, ax = plt.subplots(figsize=(7.2, 4.9))
            for values, color, marker, line, label in [
                (same, '#b96537', 'o', '-', 'Same-scenario distractors'),
                (other, '#287d9b', 'D', '--', 'Other-scenario distractors')]:
                ax.plot([r['N'] for r in values], [r['recall_at_5'] for r in values],
                        color=color, marker=marker, linestyle=line, linewidth=2, markersize=7, label=label)
            for a, b in zip(same, other):
                x, av, bv = a['N'], a['recall_at_5'], b['recall_at_5']
                if abs(av-bv) < 1e-12:
                    ax.annotate(f'{av:.2%}', (x, av), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10)
                else:
                    # Labels separate vertically even for very small differences.
                    for value, color, high in [(av, '#b96537', av>bv), (bv, '#287d9b', bv>av)]:
                        ax.annotate(f'{value:.2%}', (x, value), xytext=(0, 10 if high else -18),
                                    textcoords='offset points', ha='center', color=color, fontsize=10)
            ax.set_xlim(-1.5, 22.5); ax.set_ylim(-.04, 1.12)
            ax.set_xticks([0, 10, 20]); ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel('Number of distractors N'); ax.set_ylabel('Target Recall@5')
            ax.grid(alpha=.2); ax.set_axisbelow(True)
            prefix = name+' | ' if category == 'question_type' else ''
            small = ' | Small sample' if questions < 10 else ''
            ax.set_title(prefix+english+f'\n{questions} questions / {targets} targets'+small+'\n'+summary.get('caption_label', 'Official supplied captions'), fontsize=11, pad=12)
            # Keep legend off the data curves, including very small KR samples.
            ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncol=1, frameon=False, fontsize=10)
            gap = 100*(same[-1]['recall_at_5']-other[-1]['recall_at_5'])
            fig.text(.5, .035, f'N=20: same minus other = {gap:+.2f} percentage points', ha='center', fontsize=10)
            fig.text(.5, .005, 'Mean of 5 nested random draws per target. N=0 is target-only.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .10, 1, 1))
            paths = {}
            for extension in ['png', 'svg', 'pdf']:
                path = directory/(name+'.'+extension)
                fig.savefig(path, dpi=170, bbox_inches='tight')
                paths[extension] = str(path.relative_to(out))
            plt.close(fig)
            index.append({'category': category, 'group': name, 'label_zh': chinese, 'questions': questions,
                          'targets': targets, 'small_sample': questions<10, 'N20_gap_pp': gap, 'paths': paths})
    sections = []
    for category, heading in [('question_type', '按题型：每种题型一张图'), ('scenario', '按场景：每个场景一张图')]:
        parts = [f'<section id="{category}"><h2>{heading}</h2>']
        for item in [x for x in index if x['category'] == category]:
            paths = item['paths']; label = html.escape(item['label_zh'])
            small = ' · 小样本，请谨慎解释' if item['small_sample'] else ''
            opened = ' open' if category == 'question_type' else ''
            parts.append(f'<details{opened}><summary>{label} · {item["questions"]} 题 / {item["targets"]} 个目标{small}</summary>'
                         f'<img loading="lazy" src="{paths["png"]}" alt="{label}的干扰实验曲线">'
                         f'<p><a href="{paths["png"]}">PNG</a> · <a href="{paths["svg"]}">SVG</a> · <a href="{paths["pdf"]}">PDF</a></p></details>')
        sections.append(''.join(parts)+'</section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mem-Gallery 干扰实验：单独图表</title><style>body{font:16px/1.7 system-ui;max-width:950px;margin:30px auto;padding:18px}img{width:100%;max-width:780px}details{border-bottom:1px solid #ddd;padding:14px 0}summary{cursor:pointer}a{color:#176480}</style>
<h1>Mem-Gallery 干扰实验：单独图表</h1>
<p>由已完成实验的统计结果直接绘图，未重新运行或改变实验。所有图使用统一的 0–100% 刻度；折线上标出精确数值。
橙色实线为同场景干扰，蓝色虚线为其他场景干扰。每次小记忆库为一个目标加 N 个干扰；图中没有置信区间。</p>
<p>主样本只有 19 个场景和 5 种题型。Home Repair 场景，以及 FR、TR、MR、AR 题型没有纳入本轮主样本，不能将其视为 0% 召回。
CD 只有 3 题，KR 只有 2 题，单独标记为小样本。</p>
<nav><a href="#question_type">按题型查看</a> · <a href="#scenario">按场景查看</a> · <a href="../report.html">原始实验报告</a></nav>'''+''.join(sections)+'</html>'
    page = page.replace('<h1>', '<p>Caption 来源：'+html.escape(summary.get('caption_label', 'Official supplied captions'))+'</p><h1>', 1)
    (out/'index.html').write_text(page)
    (out/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    manifest = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scenario_figures': sum(r['category']=='scenario' for r in index),
                'question_type_figures': sum(r['category']=='question_type' for r in index),
                'artifacts': {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k!='artifacts'}, indent=2))


if __name__ == '__main__':
    main()
