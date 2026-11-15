"""Plot every centered PC of the largest frozen Qwen MemEye master.

Reuses the original dialogue+caption BGE vectors, verifies them against the
195/371 experiment protocol, and checks the graph against the later z-score
memory store. Produces the exact score-equivalent symmetric ridge spectrum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
from scripts.memory import ridge_whitening

INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021/protocol.json'
SEARCH = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024/protocol.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ridge', type=float, default=.03)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/memeye_largest_master_whitening_20261103')
    args = parser.parse_args()
    if not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error('Ridge must be finite and positive.')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty output directory.')
    reference, search = read(REFERENCE), read(SEARCH)
    stores = Path(search['memory_path'])
    hashes = {str(p): sha(p) for p in [REFERENCE, SEARCH]}

    def verify(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        hashes[str(path)] = actual

    clusters, graphs = [], {}
    for domain in sorted(search['master_counts']):
        graph_path = stores / domain / 'caption_visual_master/graph.json'
        verify(graph_path, search['source_sha256'][str(graph_path)])
        old_graph_path = INPUTS / domain / 'memory/graph.json'
        verify(old_graph_path, reference['source_sha256'][str(old_graph_path)])
        assert graph_path.read_bytes() == old_graph_path.read_bytes()
        graph = read(graph_path)
        assert len({n['id'] for n in graph['image_nodes']}) == len(graph['image_nodes'])
        counts = {}
        for node in graph['image_nodes']:
            counts[node['master_id']] = counts.get(node['master_id'], 0) + 1
        assert len(counts) == search['master_counts'][domain]
        clusters.extend({'domain': domain, 'master_id': master, 'size': size}
                        for master, size in counts.items())
        graphs[domain] = graph
    clusters.sort(key=lambda row: (-row['size'], row['domain'], row['master_id']))
    assert len(clusters) == search['masters']
    assert sum(c['size'] for c in clusters) == search['memory_leaves']
    selected = clusters[0]
    print(json.dumps({'selected': selected, 'masters_scanned': len(clusters)}), flush=True)

    domain_path = INPUTS / selected['domain']
    for filename in ['inputs.json', 'features.npz', 'protocol.json']:
        path = domain_path / filename
        verify(path, reference['source_sha256'][str(path)])
    leaves = read(domain_path / 'inputs.json')['leaves']
    store_leaves_path = stores / selected['domain'] / 'caption_visual_master/leaves.json'
    verify(store_leaves_path, search['source_sha256'][str(store_leaves_path)])
    store_leaves = read(store_leaves_path)
    assert len(leaves) == len(store_leaves)
    for original, stored in zip(leaves, store_leaves):
        assert all(original[key] == stored[key] for key in ('image_id', 'memory_id', 'dialogue', 'caption'))
        assert original['retrieval_text'] == original['dialogue'] + '\nMemory description: ' + original['caption']
    nodes = {n['id']: n for n in graphs[selected['domain']]['image_nodes']}
    assert set(nodes) == {leaf['image_id'] for leaf in leaves}
    for leaf in leaves:
        node = nodes[leaf['image_id']]
        assert all(leaf[key] == node[key] for key in ('master_id', 'caption', 'dialogue'))
    indices = [i for i, leaf in enumerate(leaves) if leaf['master_id'] == selected['master_id']]
    members = [leaves[i] for i in indices]
    assert len(members) == selected['size']
    with np.load(domain_path / 'features.npz', allow_pickle=False) as features:
        vectors = features['bge_leaf'][indices]
        queries = features['bge_query'].copy()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12)
    components, variance = ridge_whitening.fit(vectors)
    sigma = np.linalg.svd(vectors - vectors.mean(0), compute_uv=False)[:len(variance)]
    scale = np.sqrt(args.ridge / (variance + args.ridge))

    def transform(matrix):
        return matrix + ((matrix @ components.T) * (scale - 1)) @ components

    transformed = transform(vectors)
    sigma_after = sigma * scale
    variance_after = variance * scale**2
    actual_sigma = np.linalg.svd(transformed - transformed.mean(0), compute_uv=False)[:len(variance)]
    np.testing.assert_allclose(actual_sigma, sigma_after, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sigma**2 / (len(vectors) - 1), variance, atol=1e-12)
    expected = np.column_stack([ridge_whitening.score(q, vectors, components, variance, args.ridge)
                                for q in queries])
    actual = transformed @ transform(queries).T
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    share, share_after = variance / variance.sum(), variance_after / variance_after.sum()
    positive = variance > 0
    rows = [dict(pc=j + 1, singular_value_before=float(sigma[j]), singular_value_after=float(sigma_after[j]),
                 variance_before=float(variance[j]), variance_after=float(variance_after[j]),
                 variance_share_before=float(share[j]), variance_share_after=float(share_after[j]),
                 embedding_scale=float(scale[j]), score_direction_retention=float(scale[j]**2))
            for j in range(len(variance))]
    output.mkdir(parents=True, exist_ok=True)
    table(output / 'spectrum.csv', rows)
    table(output / 'all_master_sizes.csv', clusters)
    dump(output / 'selected_members.json', members)
    plot(output, selected, rows, args.ridge)
    for path, expected_hash in hashes.items():
        assert sha(path) == expected_hash
    metadata = {
        'dataset': 'MemEye', 'caption_source': 'original Qwen captions from the 195/371 master_answer_query experiment',
        'selection': selected, 'selection_rule': 'maximum leaf count across all masters; domain/master lexicographic tie break',
        'largest_size_tie_count': sum(c['size'] == selected['size'] for c in clusters),
        'domains_scanned': len(graphs), 'masters_scanned': len(clusters), 'total_leaves': sum(c['size'] for c in clusters),
        'lambda': args.ridge, 'alpha': 'not applicable to the whitening spectrum',
        'pc_count': len(variance), 'numerical_rank': int(positive.sum()),
        'embedding_source': str(domain_path / 'features.npz') + ':bge_leaf',
        'embedding_reencoded': False, 'embedding_normalized_before': True, 'embedding_normalized_after': False,
        'text_encoder': read(domain_path / 'protocol.json')['text_encoder'],
        'leaf_text': 'dialogue + newline + Memory description: + full caption',
        'transform': 'W = [lambda (C + lambda I)^(-1)]^(1/2); s(q,x) = (Wq)^T(Wx)',
        'singular_value_after': 'sigma_j * sqrt(lambda / (variance_j + lambda))',
        'variance_after': 'lambda * variance_j / (variance_j + lambda)',
        'first_pc_variance_share_before': float(share[0]), 'first_pc_variance_share_after': float(share_after[0]),
        'top5_pc_variance_share_before': float(share[:5].sum()), 'top5_pc_variance_share_after': float(share_after[:5].sum()),
        'largest_smallest_positive_variance_ratio_before': float(variance[positive].max()/variance[positive].min()),
        'largest_smallest_positive_variance_ratio_after': float(variance_after[positive].max()/variance_after[positive].min()),
        'audit': {'source_hashes_unchanged': True, 'original_and_zscore_graphs_identical': True,
                  'score_equivalence_max_abs_error': float(np.max(np.abs(expected-actual))),
                  'spectrum_max_abs_error': float(np.max(np.abs(sigma_after-actual_sigma))),
                  'original_query_vectors_used_only_for_numeric_check': len(queries)},
        'reader_called': False, 'memory_modified': False, 'new_vectors_persisted': False,
        'source_sha256': hashes,
        'code_sha256': {str(p): sha(p) for p in [Path(__file__), Path(ridge_whitening.__file__)]},
        'artifact_sha256': {str(p): sha(p) for p in sorted(output.iterdir())},
        'invocation': sys.argv,
    }
    dump(output / 'provenance.json', metadata)
    print(json.dumps({k: v for k,v in metadata.items() if k in (
        'selection', 'pc_count', 'numerical_rank', 'first_pc_variance_share_before', 'first_pc_variance_share_after',
        'top5_pc_variance_share_before', 'top5_pc_variance_share_after', 'largest_smallest_positive_variance_ratio_before',
        'largest_smallest_positive_variance_ratio_after', 'audit')}, indent=2), flush=True)


def plot(output, selected, rows, ridge):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.6), sharex=True)
    fig.subplots_adjust(left=.075, right=.985, top=.785, bottom=.17, hspace=.40)
    fig.suptitle('Ridge whitening in the largest MemEye master', y=.965, fontsize=21, weight='bold')
    fig.text(.5, .91, f"{selected['domain']}  |  {selected['master_id']}  |  "
             f"{selected['size']} memories / {len(rows)} PCs", ha='center', fontsize=12)
    fig.text(.5, .87, rf"Largest of 88 masters across 8 domains  |  $\lambda={ridge:g}$  |  Original Qwen memory",
             ha='center', color='#4B5563', fontsize=11)
    x = np.arange(len(rows))
    for ax, keys, title, ylabel, percent in [
        (axes[0], ('singular_value_before', 'singular_value_after'), 'A   Singular values', 'Singular value', False),
        (axes[1], ('variance_share_before', 'variance_share_after'), 'B   Variance distribution', 'Share of total variance', True),
    ]:
        before, after = [np.array([r[k] for r in rows]) for k in keys]
        handles = [ax.bar(x-.19, before, .37, color='#466FA3', zorder=3),
                   ax.bar(x+.19, after, .37, color='#DB8455', zorder=3)]
        ax.set_ylim(0, before.max()*1.20)
        ax.set_xlim(-.7, len(rows)-.3)
        ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', color='#E5E7EB', zorder=0)
        if percent:
            ax.yaxis.set_major_locator(MultipleLocator(.02))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            note = f"PC1: {before[0]:.1%} → {after[0]:.1%}     |     Top 5 PCs: {before[:5].sum():.1%} → {after[:5].sum():.1%}"
        else:
            note = f"PC1: {before[0]:.3f} → {after[0]:.3f}     |     PC{len(rows)}: {before[-1]:.3f} → {after[-1]:.3f}"
        ax.text(.985, .91, note, transform=ax.transAxes, ha='right', va='top', fontsize=11, color='#374151')
    axes[1].set_xticks(x, [str(r['pc']) for r in rows], fontsize=9)
    axes[0].tick_params(axis='x', labelbottom=True)
    axes[1].set_xlabel('Principal component (ordered by original variance)', labelpad=11)
    fig.legend(handles, ['Before whitening', 'After ridge whitening'], loc='upper center',
               bbox_to_anchor=(.5, .845), ncol=2, frameon=False, fontsize=12)
    fig.text(.075, .075, r'Exact score-equivalent spectrum:  $\sigma_j^{\mathrm{after}}'
             r'=\sigma_j\sqrt{\lambda/(v_j+\lambda)}$,  $v_j=\sigma_j^2/(k-1)$.', fontsize=12)
    fig.text(.075, .035, 'Frozen BGE dialogue + full-caption vectors; all 44 cluster members; '
             'no L2 renormalization after whitening. Alpha does not affect this spectrum.', fontsize=10, color='#4B5563')
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(output / f'whitening_spectrum.{extension}', dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
