#!/usr/bin/env python3
"""Decompose the frozen Home/Q10 retrieval margin in its Top-5 PCA basis."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = 'Home_Renovation_Interior_Design'
QID = DOMAIN + '/Q10'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain' / DOMAIN
RUN = ROOT / 'reports/all_master_answer_query_20261021'
FAILURES = ROOT / 'reports/baseline_failure_masters_20261022/qa_failure_set/failures.jsonl'
OUTPUT = ROOT / 'reports/home_q10_top5_pca_contributions_20261022'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--font', type=Path, default=Path('/tmp/visualmemory-plot-fonts/NotoSansCJKsc-Regular.otf'))
    args = parser.parse_args()
    frozen = read(RUN / 'protocol.json')
    sources = {}
    for name in ['inputs.json', 'features.npz', 'memory/graph.json', 'protocol.json']:
        path = INPUTS / name
        sources[str(path)] = sha(path)
        assert sources[str(path)] == frozen['source_sha256'][str(path)], f'Changed source: {path}'
    for path in [FAILURES, RUN / 'protocol.json', Path(__file__).resolve()]:
        sources[str(path)] = sha(path)

    inputs, graph = read(INPUTS / 'inputs.json'), read(INPUTS / 'memory/graph.json')
    cases = [json.loads(line) for line in FAILURES.read_text().splitlines() if line.strip()]
    matches = [case for case in cases if case['question_id'] == QID]
    assert len(matches) == 1
    case = matches[0]
    assert case['gold_choice'] != case['predicted_choice'] and case['reader_status'] == 'ok'
    qi = next(i for i, question in enumerate(inputs['questions']) if question['id'] == QID)
    question = inputs['questions'][qi]
    assert case['gold_ids'] == question['gold_ids'] and case['question'] == question['question']
    with np.load(INPUTS / 'features.npz', allow_pickle=False) as features:
        query = features['bge_query'][qi].copy()
        leaves = features['bge_leaf'].copy()
    scores = leaves @ query
    order = np.argsort(-scores, kind='stable')[:5]
    selected = [inputs['leaves'][int(i)] for i in order]
    assert [r['memory_id'] for r in selected] == [r['memory_id'] for r in case['top5']]
    score_error = float(np.max(np.abs(scores[order] - [r['score'] for r in case['top5']])))
    assert score_error < 1e-12

    # Freeze the comparison by rank before examining the component contributions.
    target_positions = [i for i, row in enumerate(selected) if row['memory_id'] in question['gold_ids']]
    assert len(target_positions) == 1
    target_pos = target_positions[0]
    target = selected[target_pos]
    distractor_pos = next(i for i, row in enumerate(selected)
                          if row['memory_id'] not in question['gold_ids']
                          and row['master_id'] == target['master_id'])
    distractor = selected[distractor_pos]
    assert target['filename'] == 'D7_IMG_005.jpg' and distractor['filename'] == 'D13_IMG_001.jpg'
    master = next(m for m in graph['masters'] if m['id'] == target['master_id'])
    assert graph['method'] == 'caption_ward_silhouette_visual'
    assert master['n'] == len(master['member_ids']) == 6
    assert all(row['master_id'] == master['id'] and row['image_id'] in master['member_ids'] for row in selected)
    nodes = {node['id']: node for node in graph['image_nodes']}
    for row in selected:
        node = nodes[row['image_id']]
        assert row['master_id'] == node['master_id']
        assert row['caption'] == node['caption'] and row['dialogue'] == node['dialogue']
        assert row['retrieval_text'] == row['dialogue'] + '\nMemory description: ' + row['caption']

    X = leaves[order]
    assert abs(np.linalg.norm(query) - 1) < 1e-12
    assert np.max(np.abs(np.linalg.norm(X, axis=1) - 1)) < 1e-12
    mean = X.mean(axis=0)
    centered = X - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = max(centered.shape) * np.finfo(centered.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    assert rank == len(X) - 1 == 4
    components = vt[:rank].copy()
    # A reproducible sign convention; each contribution is itself sign invariant.
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    eigenvalues = singular[:rank] ** 2 / (len(X) - 1)
    variance_ratios = eigenvalues / (singular ** 2 / (len(X) - 1)).sum()
    delta = X[target_pos] - X[distractor_pos]
    # Center X to fit PCA. Keep the original q so sum(c_i) equals the cosine margin.
    query_projection = components @ query
    delta_projection = components @ delta
    contributions = query_projection * delta_projection
    margin = float(query @ delta)
    reconstruction_error = float(np.linalg.norm(delta - components.T @ delta_projection))
    orthogonality_error = float(np.max(np.abs(components @ components.T - np.eye(rank))))
    margin_error = abs(float(contributions.sum()) - margin)
    assert reconstruction_error < 1e-12 and orthogonality_error < 1e-12 and margin_error < 1e-12
    assert abs(margin - float(scores[order[target_pos]] - scores[order[distractor_pos]])) < 1e-12
    np.testing.assert_allclose((-components @ query) * (-components @ delta), contributions, rtol=0, atol=1e-15)

    # Independently recover the PCA directions from the 5-by-5 Gram matrix.
    gram_values, gram_vectors = np.linalg.eigh(centered @ centered.T / (len(X) - 1))
    descending = np.argsort(-gram_values)[:rank]
    gram_components = (gram_vectors[:, descending].T @ centered) / np.sqrt(
        (len(X) - 1) * gram_values[descending, None])
    gram_contributions = (gram_components @ query) * (gram_components @ delta)
    gram_error = float(np.max(np.abs(gram_contributions - contributions)))
    np.testing.assert_allclose(gram_values[descending], eigenvalues, rtol=1e-12, atol=1e-14)
    assert gram_error < 1e-12

    rows = [{'component': f'PC{i + 1}', 'contribution_label': f'c_{i + 1}',
             'eigenvalue': float(eigenvalues[i]), 'explained_variance_ratio': float(variance_ratios[i]),
             'query_projection': float(query_projection[i]), 'target_minus_distractor_projection': float(delta_projection[i]),
             'contribution': float(contributions[i]),
             'supports': 'target' if contributions[i] > 0 else 'distractor' if contributions[i] < 0 else 'neutral'}
            for i in range(rank)]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'contributions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output / 'decomposition.npz', query=query, top5_vectors=X,
                        source_leaf_indices=order, pca_mean=mean, components=components,
                        singular_values=singular, eigenvalues=eigenvalues, variance_ratios=variance_ratios,
                        query_projection=query_projection, delta_projection=delta_projection,
                        contributions=contributions)
    summary = {'question_id': QID, 'question': question['question'],
               'gold_answer': case['gold_answer'], 'predicted_answer': case['predicted_answer'],
               'master_id': master['id'], 'master_size': master['n'], 'pca_sample_count': len(X), 'pca_rank': rank,
               'target': {**target, 'rank': target_pos + 1, 'score': float(scores[order[target_pos]])},
               'distractor': {**distractor, 'rank': distractor_pos + 1, 'score': float(scores[order[distractor_pos]])},
               'top5': [{**row, 'rank': i + 1, 'score': float(scores[order[i]]), 'is_gold': i == target_pos}
                        for i, row in enumerate(selected)],
               'excluded_master_member_ids': sorted(set(master['member_ids']) - {r['image_id'] for r in selected}),
               'components': rows, 'target_minus_distractor_score': margin,
               'contribution_sum': float(contributions.sum()),
               'positive_contribution_sum': float(contributions[contributions > 0].sum()),
               'negative_contribution_sum': float(contributions[contributions < 0].sum()),
               'validation': {'source_hashes_match_frozen_protocol': True, 'full_scan_top5_matches_failure_case': True,
                              'max_top5_score_error': score_error, 'delta_reconstruction_l2_error': reconstruction_error,
                              'orthogonality_max_abs_error': orthogonality_error, 'margin_abs_error': margin_error,
                              'independent_gram_contribution_max_abs_error': gram_error, 'sign_invariance_checked': True}}
    write_json(args.output / 'summary.json', summary)
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=args.font).get_name(),
                         'axes.unicode_minus': False, 'svg.fonttype': 'path', 'pdf.fonttype': 42,
                         'figure.facecolor': 'white'})
    ink, muted, positive, negative = '#182E39', '#647681', '#177E78', '#D76542'
    fig = plt.figure(figsize=(12.8, 8.1))
    fig.text(.075, .93, 'Q10：目标与同 master 干扰项的检索分差贡献', fontsize=23, color=ink)
    fig.text(.075, .879, 'Top-5 PCA  |  BGE 对话 + 原始 Qwen caption  |  5 条记忆，4 个主成分', fontsize=13, color=muted)
    ax = fig.add_axes([.11, .32, .83, .46])
    positions = np.arange(rank)
    heights = contributions * 1000
    colors = [positive if c >= 0 else negative for c in contributions]
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E8EDF0', linewidth=.8)
    ax.axhline(0, color='#607782', linewidth=1.1)
    ax.bar(positions, heights, width=.58, color=colors, zorder=3)
    for x, height, color in zip(positions, heights, colors):
        ax.text(x, height + (.19 if height >= 0 else -.20), f'{height:+.3f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=17, color=color)
    ax.set_xticks(positions, [f'$c_{i + 1}$ / PC{i + 1}\n方差占比 {variance_ratios[i]:.2%}' for i in range(rank)])
    ax.tick_params(axis='x', labelsize=13, pad=12, length=0, colors=ink)
    ax.tick_params(axis='y', labelsize=12, length=0, colors=muted)
    ax.set_ylim(-7.65, 2.65)
    ax.set_xlim(-.65, 3.65)
    ax.set_yticks([-6, -4, -2, 0, 2])
    ax.set_ylabel(r'对检索分差的贡献 $c_i$（$\times 10^{-3}$）', fontsize=14, color=ink, labelpad=13)
    ax.set_xlabel('主成分按 Top-5 内方差从大到小排列', fontsize=12, color=muted, labelpad=13)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#DCE4E8')
    ax.legend(handles=[Patch(facecolor=positive, label='正值：支持目标'),
                       Patch(facecolor=negative, label='负值：支持干扰项')],
              loc='lower right', bbox_to_anchor=(1, 1.025), ncol=2, frameon=False, fontsize=12)
    fig.add_artist(plt.Line2D([.075, .94], [.185, .185], transform=fig.transFigure, color='#DCE4E8', linewidth=1))
    fig.text(.075, .143, '目标：D7_IMG_005（Top-2，0.704623）', fontsize=12, color=positive)
    fig.text(.515, .143, '干扰项：D13_IMG_001（Top-1，0.715550）', fontsize=12, color=negative)
    fig.text(.075, .092, f'贡献之和 = {margin:+.6f} = 目标得分 − 干扰项得分，净支持干扰项。', fontsize=15, color=ink)
    fig.text(.075, .044, 'Top-5 均属于 master_000132；该 master 共 6 条记忆，PCA 仅使用检索到的 5 条。此图分解检索分差。',
             fontsize=10.5, color=muted)
    stem = args.output / 'q10_top5_pca_contributions'
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=240, facecolor='white')
    plt.close(fig)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'question_id': QID,
                'formula': 'c_i = (q^T v_i) * (v_i^T (x_target - x_distractor))',
                'pca_scope': 'Only the baseline Top-5 retrieved memories; exclude the sixth master member.',
                'pca_fit': 'SVD of feature-centered original normalized BGE dialogue+fullcaption vectors; sample covariance / (k-1).',
                'preprocessing': 'No whitening, standardization, or renormalization after centering; original query is not centered.',
                'component_order': 'Descending eigenvalue; no reordering by contribution magnitude.',
                'component_sign': 'Largest absolute loading positive; c_i is invariant to direction sign.',
                'target_policy': 'The unique gold memory retrieved in this case.',
                'distractor_policy': 'Highest-ranked non-gold retrieved memory sharing the target master, selected before PCA.',
                'text_encoder': read(INPUTS / 'protocol.json')['text_encoder'],
                'bge_encoding': read(INPUTS / 'protocol.json')['bge_encoding'],
                'source_sha256': sources, 'font_sha256': sha(args.font),
                'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                'interpretation': 'Exact retrieval similarity margin decomposition for this one pair; not causal attribution of the reader answer.',
                'validation': summary['validation'],
                'output_sha256': {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != 'protocol.json'}}
    write_json(args.output / 'protocol.json', protocol)
    print(json.dumps({'output': str(args.output), 'components': rows, 'margin': margin,
                      'validation': summary['validation']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
