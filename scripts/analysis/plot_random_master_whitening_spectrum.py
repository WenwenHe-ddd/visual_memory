"""Plot the score-equivalent ridge-whitening spectrum of one random master.

Uses frozen Mem-Gallery dialogue+caption texts and the production BGE/SVD code.
No QA calls, graph changes, or persisted embedding vectors. The symmetric
transform has directional scale sqrt(lambda / (variance + lambda)); the scale
lambda / (variance + lambda) belongs to the bilinear SCORE, not singular values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/visualmemory-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
from scripts.memory.caption_visual_master import encode_text, load_text_encoder
from scripts.memory import ridge_whitening


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n')


def table(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/mem_gallery_master_answer_20261021')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/mem_gallery_master_whitening_spectrum_20261103')
    parser.add_argument('--ridge', type=float, default=.03)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-size', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not np.isfinite(args.ridge) or args.ridge <= 0 or args.min_size < 3:
        parser.error('Require ridge > 0 and min-size >= 3.')
    output, source = args.output.resolve(), args.source.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new, empty output directory to preserve previous artifacts.')

    offline = source / 'offline'
    protocol_path = offline / 'protocol.json'
    protocol = read(protocol_path)
    memory_path = offline / 'memories.jsonl'
    graph_paths = sorted((offline / 'memory').glob('*/graph.json'))
    # Verify every graph used to define the sampling population, and the texts.
    input_hashes = {str(p): sha(p) for p in [protocol_path, memory_path, *graph_paths]}
    for p in [memory_path, *graph_paths]:
        assert input_hashes[str(p)] == protocol['artifact_sha256'][str(p)], str(p)
    graphs = {p.parent.name: read(p) for p in graph_paths}
    eligible = []
    for domain, graph in sorted(graphs.items()):
        counts = {}
        for node in graph['image_nodes']:
            counts[node['master_id']] = counts.get(node['master_id'], 0) + 1
        for master, count in sorted(counts.items()):
            if count >= args.min_size:
                eligible.append({'domain': domain, 'master_id': master, 'size': count})
    if not eligible:
        raise ValueError('No eligible masters.')
    chosen_index = random.Random(args.seed).randrange(len(eligible))
    selected = eligible[chosen_index]
    print(json.dumps({'selected': selected, 'eligible': len(eligible)}, ensure_ascii=False), flush=True)

    memories = [json.loads(line) for line in memory_path.read_text().splitlines()]
    domain_rows = [row for row in memories if row['task'] == selected['domain']]
    nodes = {n['id']: n for n in graphs[selected['domain']]['image_nodes']}
    indices = [i for i, row in enumerate(domain_rows)
               if row['is_visual'] and nodes[row['node_id']]['master_id'] == selected['master_id']]
    members = [domain_rows[i] for i in indices]
    assert len(members) == selected['size']
    for row in members:
        node = nodes[row['node_id']]
        assert all(row[key] == node[key] for key in ('caption', 'dialogue'))
        assert row['retrieval_text'] == row['dialogue'] + '\nMemory description: ' + row['caption']

    import torch
    torch.set_num_threads(4)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device is unavailable.')
    encoder = load_text_encoder(protocol['text_encoder'], args.device)
    # Match the original domain pool and batch order before selecting the master.
    domain_vectors = encode_text(encoder, [row['retrieval_text'] for row in domain_rows])
    vectors = domain_vectors[indices]
    assert vectors.shape == (len(members), protocol['text_encoder']['dimension'])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12)
    components, variance = ridge_whitening.fit(vectors)
    centered = vectors - vectors.mean(axis=0)
    sigma = np.linalg.svd(centered, compute_uv=False)[:len(variance)]
    scale = np.sqrt(args.ridge / (variance + args.ridge))

    def transform(matrix):
        # W = I + V diag(sqrt(lambda/(v+lambda)) - 1) V^T.
        return matrix + ((matrix @ components.T) * (scale - 1)) @ components

    transformed = transform(vectors)
    centered_after = transformed - transformed.mean(axis=0)
    sigma_after = sigma * scale
    observed_after = np.linalg.svd(centered_after, compute_uv=False)[:len(variance)]
    variance_after = variance * scale**2
    np.testing.assert_allclose(observed_after, sigma_after, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sigma**2 / (len(members) - 1), variance, atol=1e-12)
    # Check actual production score equivalence with normalized synthetic queries.
    # These queries are only a numerical check, not retrieval/evaluation evidence.
    checks = np.random.default_rng(args.seed).normal(size=(16, vectors.shape[1]))
    checks /= np.linalg.norm(checks, axis=1, keepdims=True)
    expected = np.column_stack([
        ridge_whitening.score(q, vectors, components, variance, args.ridge) for q in checks
    ])
    observed = transformed @ transform(checks).T
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
    positive = variance > 0
    if not positive.any():
        raise ValueError('Randomly selected master has zero variance; do not silently resample.')
    share_before = variance / variance.sum()
    share_after = variance_after / variance_after.sum()
    rows = [dict(
        pc=j + 1, singular_value_before=float(sigma[j]), singular_value_after=float(sigma_after[j]),
        variance_before=float(variance[j]), variance_after=float(variance_after[j]),
        variance_share_before=float(share_before[j]), variance_share_after=float(share_after[j]),
        embedding_scale=float(scale[j]), score_direction_retention=float(scale[j]**2),
    ) for j in range(len(variance))]
    output.mkdir(parents=True, exist_ok=True)
    table(output / 'spectrum.csv', rows)
    table(output / 'eligible_masters.csv', eligible)
    dump(output / 'selected_members.json', members)
    metadata = {
        'dataset': 'Mem-Gallery', 'source': str(source), 'selection': selected,
        'sampling': {'seed': args.seed, 'rng': 'random.Random', 'eligible_count': len(eligible),
                     'chosen_zero_based_index': chosen_index, 'minimum_cluster_size': args.min_size,
                     'ordering': 'domain, then master_id, lexicographic',
                     'rule': 'uniform over all eligible masters; sampled before encoding or inspecting spectra'},
        'lambda': args.ridge, 'alpha': 'not applicable to the whitening spectrum',
        'text_encoder': protocol['text_encoder'], 'device': args.device,
        'leaf_text': 'dialogue + newline + Memory description: + full caption; no query prefix',
        'encoded_domain_count': len(domain_rows), 'bge_batch_size': 16,
        'l2_normalized_before': True, 'l2_normalized_after': False,
        'pc_count': len(variance), 'numerical_rank': int(positive.sum()),
        'transform': 'W = [lambda (C + lambda I)^(-1)]^(1/2); s(q,x) = (Wq)^T (Wx)',
        'singular_value_after': 'sigma_j * sqrt(lambda / (variance_j + lambda))',
        'variance_after': 'lambda * variance_j / (variance_j + lambda)',
        'convention': 'score-equivalent symmetric ridge transform; sqrt(lambda) times conventional ridge whitening',
        'centered_for_covariance_only': True, 'vectors_persisted': False, 'reader_called': False,
        'first_pc_variance_share_before': float(share_before[0]),
        'first_pc_variance_share_after': float(share_after[0]),
        'largest_to_smallest_positive_variance_before': float(variance[positive].max() / variance[positive].min()),
        'largest_to_smallest_positive_variance_after': float(variance_after[positive].max() / variance_after[positive].min()),
        'audit': {'score_equivalence_max_abs_error': float(np.max(np.abs(observed - expected))),
                  'spectrum_formula_max_abs_error': float(np.max(np.abs(observed_after - sigma_after))),
                  'synthetic_queries_for_numerical_check_only': len(checks)},
        'input_sha256': input_hashes,
        'code_sha256': {str(p): sha(p) for p in [Path(__file__), Path(ridge_whitening.__file__),
                                               ROOT / 'scripts/memory/caption_visual_master.py']},
        'software': {'numpy': np.__version__, 'torch': torch.__version__},
        'invocation': sys.argv,
    }
    plot(output, selected, rows, args.ridge, args.seed)
    for path, expected_hash in input_hashes.items():
        assert sha(path) == expected_hash, f'Input changed: {path}'
    metadata['audit']['input_hashes_unchanged'] = True
    metadata['artifact_sha256'] = {str(p): sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    dump(output / 'provenance.json', metadata)
    print(json.dumps({'output': str(output), 'spectrum': rows, 'audit': metadata['audit']}, indent=2), flush=True)


def plot(output, selected, rows, ridge, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.7))
    fig.subplots_adjust(left=.08, right=.98, top=.69, bottom=.23, wspace=.27)
    fig.suptitle('Ridge whitening within a random memory cluster', fontsize=18, y=.96, weight='bold')
    fig.text(.5, .88, f"Mem-Gallery | {selected['domain']}", ha='center', fontsize=9.5, color='#4B5563')
    fig.text(.5, .83, f"{selected['master_id']}  |  {selected['size']} memories  |  "
             rf"$\lambda={ridge:g}$  |  sampling seed {seed}", ha='center', fontsize=11)
    colors = ['#466FA3', '#DB8455']
    positions = np.arange(len(rows))
    handles = None
    for ax, (before_key, after_key, ylabel, title, percent) in zip(axes, [
        ('singular_value_before', 'singular_value_after', 'Singular value', 'A   Singular values', False),
        ('variance_share_before', 'variance_share_after', 'Share of total variance', 'B   Variance distribution', True),
    ]):
        before = np.array([r[before_key] for r in rows])
        after = np.array([r[after_key] for r in rows])
        bars_before = ax.bar(positions - .18, before, width=.34, color=colors[0], label='Before whitening', zorder=3)
        bars_after = ax.bar(positions + .18, after, width=.34, color=colors[1], label='After ridge whitening', zorder=3)
        handles = [bars_before, bars_after]
        for bars in (bars_before, bars_after):
            labels = [f'{b.get_height():.1%}' if percent else f'{b.get_height():.3f}' for b in bars]
            ax.bar_label(bars, labels=labels, padding=5, fontsize=10)
        ax.set_xticks(positions, [f'PC {r["pc"]}' for r in rows])
        ax.set_xlabel('Components ordered by original variance', labelpad=10, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=12, pad=13, weight='bold')
        ax.set_ylim(0, max(before.max(), after.max()) * 1.27)
        ax.grid(axis='y', color='#E5E7EB', linewidth=.8, zorder=0)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(handles, ['Before whitening', 'After ridge whitening'], loc='upper center',
               bbox_to_anchor=(.5, .798), ncol=2, frameon=False, fontsize=11)
    fig.text(.08, .105, r'Exact score-equivalent spectrum:  $\sigma_j^{\mathrm{after}}'
             r'=\sigma_j\sqrt{\lambda/(v_j+\lambda)}$,  '
             r'$v_j=\sigma_j^2/(k-1)$.', fontsize=11)
    fig.text(.08, .057, 'BGE dialogue + full caption; all cluster members; no L2 renormalization after whitening. '
             'The z-score step follows this transform.', fontsize=9, color='#4B5563')
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'whitening_spectrum.{extension}', dpi=220, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
