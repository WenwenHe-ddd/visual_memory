#!/usr/bin/env python3
"""Export the completed, paired PCA/Random-2D Qwen ablation."""
import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/memeye_random2d_report_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'reports/memeye_qwen_random2d_ablation_20261105'
SHORT = {'Brand_Memory_Test': 'Brand', 'Card_Playlog_Test': 'Card',
         'Cartoon_Entertainment_Companion': 'Cartoon',
         'Home_Renovation_Interior_Design': 'Home',
         'Multi-Scene_Visual_Case_Archive_Assistant': 'Multi-Scene',
         'Outdoor_Navigation_Route_Memory_Assistant': 'Outdoor',
         'Personal_Health_Dashboard_Assistant': 'Health', 'Social_Chat_Memory_Test': 'Social'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(base, seeds):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Select at least two distinct nonnegative seeds for sample SD.')
    conditions = ['pca_2d', *[f'random_2d_seed_{s}' for s in seeds]]
    seed_text = ', '.join(map(str, seeds))
    retrieval = read(base / 'retrieval_summary.json')
    reader = read(base / 'reader/summary.json')
    offline = read(base / 'offline/manifest.json')
    assert retrieval['completed'] and offline['completed'] and reader['status'] == 'complete'
    assert all(c in retrieval['conditions'] and c in reader['overall'] for c in conditions)
    assert reader['historical_answers_reused'] == reader['format_errors'] == reader['token_limit_hits'] == 0
    verified_sources = {}
    for filename in ('retrieval_protocol.json', 'offline/frozen_inputs.json', 'reader/protocol.json'):
        for path, expected in read(base / filename)['source_sha256'].items():
            assert sha(Path(path)) == expected, path
            verified_sources[path] = expected
    selected_answers = [json.loads(line) for line in (base / 'reader/per_question.jsonl').read_text().splitlines()
                        if json.loads(line)['condition'] in conditions]
    reference_ids = {r['question_id'] for r in selected_answers if r['condition'] == 'pca_2d'}
    for condition in conditions:
        answers = [r for r in selected_answers if r['condition'] == condition]
        assert len(answers) == len({r['question_id'] for r in answers}) == 371
        assert {r['question_id'] for r in answers} == reference_ids
        assert sum(r['correct'] for r in answers) == reader['overall'][condition]['correct']
    unique_payloads = len({r['payload_sha256'] for r in selected_answers})
    write_csv(base / 'selected_reader_per_question.csv', selected_answers)
    rows = []
    for condition in conditions:
        r = retrieval['conditions'][condition]
        a = reader['overall'][condition]
        assert r['questions'] == a['questions'] == 371 and r['gold_references'] == 1041
        rows.append({'condition': condition, 'seed': '' if condition == 'pca_2d' else condition.rsplit('_', 1)[1],
                     'questions': 371, 'gold_references': 1041, 'retrieved_gold': r['gold_hits_top5'],
                     'recall_at_5_percent': r['recall_at_5'] * 100, 'qa_correct': a['correct'],
                     'qa_accuracy_percent': a['accuracy'] * 100,
                     'recall_delta_pp_vs_pca': 100 * (r['recall_at_5'] - retrieval['conditions']['pca_2d']['recall_at_5']),
                     'accuracy_delta_pp_vs_pca': 100 * (a['accuracy'] - reader['overall']['pca_2d']['accuracy'])})
    write_csv(base / 'comparison.csv', rows)
    domain_rows = []
    for domain in SHORT:
        recalls = [retrieval['by_domain'][c][domain]['recall_at_5'] * 100 for c in conditions[1:]]
        accuracies = [reader['by_domain'][c][domain]['accuracy'] * 100 for c in conditions[1:]]
        domain_rows.append({'domain': domain, 'short_domain': SHORT[domain],
            'questions': reader['by_domain']['pca_2d'][domain]['questions'],
            'pca_recall_percent': retrieval['by_domain']['pca_2d'][domain]['recall_at_5'] * 100,
            'random_recall_mean_percent': statistics.mean(recalls), 'random_recall_sd_pp': statistics.stdev(recalls),
            'pca_accuracy_percent': reader['by_domain']['pca_2d'][domain]['accuracy'] * 100,
            'random_accuracy_mean_percent': statistics.mean(accuracies), 'random_accuracy_sd_pp': statistics.stdev(accuracies)})
    write_csv(base / 'domain_comparison.csv', domain_rows)
    recalls = [r['recall_at_5_percent'] for r in rows[1:]]
    accuracies = [r['qa_accuracy_percent'] for r in rows[1:]]
    summary = {'completed': True, 'dataset': 'MemEye', 'domains': 8, 'memory_leaves': 438,
        'masters_per_condition': 88, 'questions_per_condition': 371, 'conditions': len(conditions),
        'caption_and_reader_backbone': 'Qwen3-VL-8B-Instruct', 'online_weighting': 'none',
        'pca_2d': rows[0],
        'random_2d': {'seeds': seeds,
            'recall_mean_percent': statistics.mean(recalls),
            'recall_sd_percentage_points': statistics.stdev(recalls),
            'recall_range_percent': [min(recalls), max(recalls)],
            'accuracy_mean_percent': statistics.mean(accuracies),
            'accuracy_sd_percentage_points': statistics.stdev(accuracies),
            'accuracy_range_percent': [min(accuracies), max(accuracies)],
            'mean_correct': statistics.mean(r['qa_correct'] for r in rows[1:])},
        'unique_reader_payloads': unique_payloads, 'logical_reader_records': len(selected_answers),
        'report_selection': {'conditions': conditions,
            'selection_rule': 'First three seeds (0, 1, 2) by user-requested reduction; no selection by score.' if seeds == [0, 1, 2] else 'Explicit --seeds argument.',
            'model_rerun_for_report': False,
            'source_run_conditions': reader['conditions'],
            'source_run_logical_records': reader['logical_records'],
            'source_run_fresh_reader_inferences': reader['fresh_model_inferences'],
            'source_run_preserved': True},
        'historical_answers_reused': 0, 'format_errors': 0, 'token_limit_hits': 0,
        'dynamic_split_steps_per_condition': sum(d['conditions']['pca_2d']['split_steps'] for d in offline['domains']),
        'image_forwards_shared_across_conditions': offline['image_forwards_all_conditions_shared'],
        'original_images_or_full_leaf_visuals_retained': False,
        'pca_historical_ordered_top5_parity': retrieval['pca_replay_ordered_top5_matches_historical'],
        'std_definition': f'Sample standard deviation across {len(seeds)} selected random basis seeds ({seed_text}); not a confidence interval over questions',
        'interpretation': 'Comparable downstream results in this fixed MemEye/Qwen protocol; not evidence of statistically superior performance or equivalence in other settings.'}
    dump(base / 'summary.json', summary)
    # Zero-baseline bars show the magnitude of both metrics without exaggerating
    # the small differences. The random error bars are sample SD across seeds.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    entries = [
        ('Recall@5', rows[0]['recall_at_5_percent'], summary['random_2d']['recall_mean_percent'], summary['random_2d']['recall_sd_percentage_points']),
        ('QA accuracy', rows[0]['qa_accuracy_percent'], summary['random_2d']['accuracy_mean_percent'], summary['random_2d']['accuracy_sd_percentage_points'])]
    for ax, (title, pca, mean, sd) in zip(axes, entries):
        ax.bar([0, 1], [pca, mean], width=.56, color=['#3274A1', '#3C9A77'],
               yerr=[0, sd], capsize=5, error_kw={'elinewidth': 1.2})
        ax.text(0, pca + 3, f'{pca:.2f}%', ha='center')
        ax.text(1, mean + 3, f'{mean:.2f} ± {sd:.2f}%', ha='center')
        ax.set_xticks([0, 1], ['PCA-2D', 'Random-2D'])
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percent')
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('MemEye | Qwen3-VL-8B-Instruct | Basis ablation', fontsize=14)
    fig.text(.5, .035, f'371 questions · 8 domains · Random-2D: mean ± sample SD over seeds {seed_text}',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .94))
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(base / f'comparison.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    text = [
        'MemEye / Qwen3-VL-8B-Instruct：PCA-2D 与 Random-2D 基底消融', '',
        '范围：8 个领域，438 条图片记忆，每条件 88 个 master，371 道题（8 道带图题）。',
        '主配置：caption_visual_master + multi_path + online_weighting=none。',
        '仅改变离线固定二维基底。原始 Qwen caption、题图 caption、CLIP/BGE、到达顺序、候选陈述和回答提示固定。',
        f'PCA 沿用各领域的外域校准基底；Random 采用高斯矩阵 reduced QR，本报告使用 seeds {seed_text}，每个 seed 的基底跨领域共享。',
        '按用户要求将报告缩减为前三个种子（0、1、2），未按得分挑选；直接使用已完成的对应结果，无需重新推理。' if seeds == [0, 1, 2] else '报告使用 --seeds 指定的已有结果，无需重新推理。',
        '逐张图像到达时在 GPU0 编码一次，供各组状态使用后释放。全部历史前缀的 caption 分簇均一致；每条件发生 75 个拆分步骤。',
        '没有保存原图或完整叶视觉向量，没有重编码旧图；每叶仅保留两个坐标。',
        '在线沿用已核验的固定文本分数，仅用各组实际 master 重新选择候选陈述和 Top5。',
        '原始消融回答推理配置：BF16、SDPA、greedy、seed42、max_new_tokens16、batch1、整模型 GPU0。',
        f'本报告包含 {len(conditions)} 个条件、{len(selected_answers):,} 条逻辑记录、{unique_payloads} 个不同完整 reader payload。',
        f'原始运行的 {reader["conditions"]} 条件、{reader["logical_records"]:,} 条记录保留供溯源；未选种子不参与本报告任何统计。报告汇总不触发新推理。', '',
        f'总体结果（± 为 {len(seeds)} 个随机基底种子的样本标准差）',
        f"PCA-2D: Recall@5={rows[0]['recall_at_5_percent']:.4f}% (500/1041); QA={rows[0]['qa_accuracy_percent']:.4f}% (195/371).",
        f"Random-2D: Recall@5={summary['random_2d']['recall_mean_percent']:.4f} ± {summary['random_2d']['recall_sd_percentage_points']:.4f}%; QA={summary['random_2d']['accuracy_mean_percent']:.4f} ± {summary['random_2d']['accuracy_sd_percentage_points']:.4f}%.",
        'Random 各种子的 QA 正确数为：' + ', '.join(str(r['qa_correct']) for r in rows[1:]),
        '', '逐领域：PCA Recall | Random Recall mean±SD | PCA ACC | Random ACC mean±SD（%）']
    for d in domain_rows:
        text.append(f"{d['short_domain']}: {d['pca_recall_percent']:.2f} | {d['random_recall_mean_percent']:.2f}±{d['random_recall_sd_pp']:.2f} | {d['pca_accuracy_percent']:.2f} | {d['random_accuracy_mean_percent']:.2f}±{d['random_accuracy_sd_pp']:.2f}")
    text += ['', '结论：在当前 MemEye/Qwen 设置中，随机正交基底获得了与外域 PCA 接近的下游效果；本实验支持在此配置下移除外部校准图片依赖。平均提高很小，不据此声称显著优于 PCA，也不将随机种子波动当作统计等价检验。',
        '', '验证：PCA 重建检索与历史 371 题 Top5 顺序全一致；fresh PCA 正确数仍为195。所有完整回答格式有效、无 token 截断。',
        '历史漂移：旧 QA 路径移动及历史检索脚本 hash 变化已记录；当前 QA 逐字段一致，实际冻结数据 hash 和历史检索重算均通过。',
        'GPU/CPU 编码浮点差已记录；主对比使用同次 GPU 图像编码和同一 GPU reader 配置。',
        '', '复现入口：',
        'python3 scripts/analysis/build_memeye_random2d_ablation.py --device cuda:0',
        'python3 scripts/analysis/retrieve_memeye_random2d_ablation.py',
        'python3 scripts/analysis/evaluate_memeye_random2d_reader.py --device cuda:0 --stage all',
        f'python3 scripts/analysis/report_memeye_random2d_ablation.py --seeds {" ".join(map(str, seeds))}',
        '前三个脚本对应已完成的原始完整运行；最后一步只汇总指定种子。',
        '复现完整运行时各脚本 --output/--source 指向新的统一目录，避免覆盖已经签名的实验。']
    (base / 'report.txt').write_text('\n'.join(text) + '\n')
    artifacts = ['summary.json', 'comparison.csv', 'domain_comparison.csv', 'comparison.png', 'comparison.pdf',
                 'comparison.svg', 'report.txt', 'selected_reader_per_question.csv',
                 'reader/summary.json', 'reader/responses.jsonl', 'retrieval_results.jsonl']
    dump(base / 'completion_manifest.json', {'completed': True, 'source_files_verified': len(verified_sources),
        'all_stages_complete': True, 'selected_seeds': seeds,
        'new_artifact_sha256': {p: sha(base / p) for p in artifacts},
        'report_script_sha256': sha(Path(__file__)), 'independent_offline_audit': 'offline/validation.json',
        'reader_input_audit': 'reader/validation_input.json', 'reader_output_audit': 'reader/validation_output.json'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds included in the report (default: 0 1 2).')
    args = parser.parse_args()
    run(args.output.resolve(), args.seeds)
