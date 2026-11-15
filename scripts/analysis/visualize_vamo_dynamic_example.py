#!/usr/bin/env python3
"""Decode one image from its own historical VAMO master/2-D snapshots.

This is the existing raw VAE-latent adaptation, not a CLIP-to-image decoder.
The streaming phase receives only each new image; evaluation references are
opened after the entire replay and never enter a memory update.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vamo_dynamic_mpl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HELPERS = ROOT / 'reports/brand_vae_vamo_20261105/run_experiment.py'
SWEEP = ROOT / 'reports/other_domains_vae_vamo_20261105'

import numpy as np
import torch
from scripts.graph.caption_ward_visual import update_visual_partition


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rebuild(snapshot, basis):
    mean = np.load(snapshot / 'master_sum.npy') / read(snapshot / 'metadata.json')['n']
    coords = np.load(snapshot / 'target_coords.npy')
    return mean + basis @ (coords - basis.T @ mean)


def run(output, device, domain, target, split_step, merge_step, selection_note):
    old_root = SWEEP / domain
    stages = {split_step - 1: 'before_split', split_step: 'after_split',
              merge_step - 1: 'before_merge', merge_step: 'after_merge'}
    assert len(stages) == 4 and split_step < merge_step, 'Use separate chronological event pairs.'
    completed = read(old_root / 'completed.json')
    assert completed['completed']
    partition_path = Path(completed['source_audit'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; historical experiments remain unchanged.')
    spec = importlib.util.spec_from_file_location('frozen_vae_protocol', HELPERS)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.DEVICE = device
    previous_protocol = read(SWEEP / 'protocol.json')
    for name, expected in previous_protocol['model_sha256'].items():
        assert sha(old.MODEL / name) == expected, name
    for path, expected in previous_protocol['source_sha256'].items():
        assert sha(path) == expected, path
    source_domain = old.SOURCE / domain
    arrivals = read(source_domain / 'input_manifest.json')['arrivals']
    partitions = lines(partition_path)
    assert len(arrivals) == len(partitions) == completed['n']
    prior_audits = lines(old_root / 'replay_audit.jsonl')
    lookup = {r['image_id']: r for r in arrivals}
    target_index = next(i for i, r in enumerate(arrivals) if r['image_id'] == target)
    basis = np.load(old_root / 'memory/basis.npy')
    assert basis.shape == (4096, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12, rtol=0)
    sources = [Path(__file__), HELPERS, SWEEP / 'protocol.json', SWEEP / 'run_domains.py',
               old_root / 'memory/basis.npy', old_root / 'memory/master_sums.npy', old_root / 'memory/leaf_coords.npy',
               old_root / 'memory/membership.json', old_root / 'replay_audit.jsonl',
               source_domain / 'input_manifest.json', partition_path,
               ROOT / 'scripts/graph/caption_ward_visual.py', old_root / 'image_metadata.json',
               old_root / 'evaluation_only/full_latents.npy']
    source_hashes = {str(p): sha(p) for p in sources}
    output.mkdir(parents=True)
    (output / 'snapshots').mkdir()
    (output / 'images').mkdir()
    np.save(output / 'basis.npy', basis)

    def membership(step):
        return next(c for c in partitions[step - 1]['clusters'] if target in c['member_ids'])

    before, after = membership(split_step - 1), membership(split_step)
    assert set(after['member_ids']) < set(before['member_ids'])
    assert after['master_id'] not in prior_audits[split_step - 1]['merged_master_ids']
    assert before['master_id'] in prior_audits[split_step - 1]['split_master_ids']
    assert not set(after['member_ids']) & set(prior_audits[split_step - 1]['new_image_ids'])
    merged = membership(merge_step)
    overlaps = [r for r in prior_audits[merge_step - 1]['source_overlaps'] if r['new_master_id'] == merged['master_id']]
    assert len(overlaps) >= 2 and all(r['count'] == r['old_n'] for r in overlaps)
    assert not set(merged['member_ids']) & set(prior_audits[merge_step - 1]['new_image_ids'])
    selection = {'image_id': target, 'filename': lookup[target]['filename'],
                 'selection': selection_note, 'domain': domain,
                 'split_step': split_step, 'merge_step': merge_step, 'snapshots': stages,
                 'intervening_updates': 'All arrivals are replayed. Two distinct immediate before/after pairs are shown.',
                 'split_new_arrival': arrivals[split_step - 1]['filename'], 'merge_new_arrival': arrivals[merge_step - 1]['filename']}
    dump(output / 'selection.json', selection)
    model = old.model_load()
    masters, coords, next_index = {}, {}, 1
    total = np.zeros(4096)
    audits, timeline, snapshots = [], [], []
    with torch.inference_mode():
        for step, (row, frozen) in enumerate(zip(arrivals, partitions), 1):
            assert row['image_id'] == frozen['image_id']
            image, _ = old.prepare(row)
            visual = model.encode(image[None].to(device)).latent_dist.mode()[0].cpu().double().numpy().reshape(-1)
            assert visual.shape == (4096,)
            total += visual
            masters, audit, next_index = update_visual_partition(
                masters, coords, frozen['clusters_partition'], {row['image_id']: visual}, basis, next_index)
            coords[row['image_id']] = basis.T @ visual
            assert audit['clusters'] == prior_audits[step - 1]['clusters']
            np.testing.assert_allclose(sum(m['visual_sum'] for m in masters.values()), total, atol=1e-9, rtol=0)
            assert audit['max_projection_consistency_l2_error'] < 1e-9
            current = next((m for m in masters.values() if target in m['member_ids']), None)
            audits.append({'step': step, **audit})
            if current is None:
                del visual, image
                continue
            meta = {'step': step, 'target_image_id': target, 'target_filename': lookup[target]['filename'],
                    'master_id': current['id'], 'n': current['n'], 'member_ids': current['member_ids'],
                    'member_filenames': [lookup[mid]['filename'] for mid in current['member_ids']],
                    'approximate': bool(current['approximate']), 'new_arrival': row['filename']}
            timeline.append(meta)
            if step in stages:
                folder = output / 'snapshots' / stages[step]
                folder.mkdir()
                np.save(folder / 'master_sum.npy', current['visual_sum'])
                np.save(folder / 'target_coords.npy', coords[target])
                dump(folder / 'metadata.json', meta)
                snapshots.append(meta)
            del visual, image
            print(json.dumps({'stage': 'replay', 'step': step, 'target_master': current['id'], 'n': current['n']}), flush=True)
    saved = read(old_root / 'memory/membership.json')
    final_sums = np.stack([masters[mid]['visual_sum'] for mid in saved['master_ids']])
    final_coords = np.stack([coords[mid] for mid in saved['leaf_ids']])
    sum_error = float(np.max(np.abs(final_sums - np.load(old_root / 'memory/master_sums.npy'))))
    coords_error = float(np.max(np.abs(final_coords - np.load(old_root / 'memory/leaf_coords.npy'))))
    assert sum_error < 1e-6 and coords_error < 1e-6, (sum_error, coords_error)
    dump(output / 'timeline.json', timeline)
    (output / 'replay_audit.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in audits))

    # Evaluation-only references are opened after replay completion.
    reference = np.load(old_root / 'evaluation_only/full_latents.npy', mmap_mode='r')[target_index].reshape(-1).astype(np.float64)
    original, metadata = old.prepare(lookup[target])
    x0, y0, x1, y1 = metadata['content_box']
    rebuilt = {name: rebuild(output / 'snapshots' / name, basis) for name in stages.values()}
    np.testing.assert_allclose(rebuilt['before_split'], rebuilt['after_split'], atol=1e-10, rtol=0)
    import lpips
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    metric = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    with torch.inference_mode():
        tensors = {'original': original[None].to(device)}
        for name, vector in {'vae_direct': reference, **rebuilt}.items():
            tensors[name] = model.decode(torch.tensor(vector.reshape(1, 4, 32, 32), dtype=torch.float32,
                                                     device=device)).sample.clamp(-1, 1)
        metrics = []
        for name, tensor in tensors.items():
            old.output_tensor(tensor[0], output / 'images' / f'{name}.png')
            if name == 'original':
                continue
            prediction = ((tensor[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()
            truth = ((tensors['original'][0] + 1) / 2).permute(1, 2, 0).cpu().numpy()
            kwargs = dict(data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
            vector = reference if name == 'vae_direct' else rebuilt[name]
            meta = next((m for m in snapshots if stages[m['step']] == name), None)
            metrics.append({'stage': name, 'step': None if meta is None else meta['step'],
                            'master_id': None if meta is None else meta['master_id'],
                            'cluster_size': None if meta is None else meta['n'],
                            'latent_cosine': float(vector @ reference / (np.linalg.norm(vector) * np.linalg.norm(reference))),
                            'latent_relative_l2': float(np.linalg.norm(vector - reference) / np.linalg.norm(reference)),
                            'content_ssim': float(structural_similarity(truth[y0:y1, x0:x1], prediction[y0:y1, x0:x1], **kwargs)),
                            'content_psnr': float(peak_signal_noise_ratio(truth[y0:y1, x0:x1], prediction[y0:y1, x0:x1], data_range=1)),
                            'content_lpips': float(metric(tensor[:, :, y0:y1, x0:x1], tensors['original'][:, :, y0:y1, x0:x1]).item())})
    transitions = []
    for label, left, right in [('split', 'before_split', 'after_split'), ('merge', 'before_merge', 'after_merge')]:
        a, b = rebuilt[left], rebuilt[right]
        pa, pb = tensors[left].cpu().numpy(), tensors[right].cpu().numpy()
        transitions.append({'event': label, 'before': left, 'after': right,
                            'latent_max_abs_change': float(np.max(np.abs(a - b))),
                            'latent_l2_change': float(np.linalg.norm(a - b)),
                            'latent_cosine_between_stages': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                            'decoded_float_max_abs_change': float(np.max(np.abs(pa - pb))),
                            'decoded_png_identical': sha(output / 'images' / f'{left}.png') == sha(output / 'images' / f'{right}.png')})
    csv_write(output / 'metrics.csv', metrics)
    dump(output / 'transitions.json', transitions)
    source_unchanged = all(sha(path) == expected for path, expected in source_hashes.items())
    assert source_unchanged
    validation = {'passed': True, 'completed': True, 'target_image_id': target,
                  'representation': 'Raw 4096D VAE posterior mean; not the 512D CLIP retrieval representation',
                  'leaf_coordinates': 2, 'image_arrival_forwards': len(arrivals),
                  'old_images_reencoded_during_updates': 0, 'old_full_latents_used_for_updates': False,
                  'evaluation_references_opened_after_stream': True,
                  'historical_prefix_partitions_matched': len(arrivals),
                  'final_master_sum_max_abs_error': sum_error, 'final_coords_max_abs_error': coords_error,
                  'source_files_unchanged': source_unchanged, 'source_sha256': source_hashes,
                  'device': device, 'model': previous_protocol['model'],
                  'split_png_identical': transitions[0]['decoded_png_identical']}
    draw_figure(output, metrics, selection)
    report = ['同一张图片的 VAMO 拆分与合并重建', '', f'目标：{target} / {lookup[target]["filename"]}',
              '沿用 sd-vae-ft-mse 的 4096 维原始 VAE latent 适配版；不是将 CLIP embedding 直接输入 VAE。',
              '固定原实验二维 PCA 基底、caption 分簇轨迹、预处理和确定性 VAE posterior mean。',
              f'回放 {len(arrivals)} 次新图到达，每图在到达时编码一次；旧图完整 latent 不参与更新。',
              '各时点图像仅由该时点 master sum/count、该叶子两坐标和共享基底重建后解码。',
              f'原图与 VAE 直解只是评估参照。第 {split_step-1}/{split_step} 和 {merge_step-1}/{merge_step} 步是各事件紧邻的前后状态。', '', '目标簇：']
    for m in snapshots:
        report.append(f"t={m['step']}: {m['master_id']}, n={m['n']}: " + ', '.join(m['member_filenames']))
    report += ['', f'拆分：第{split_step}步只取原簇子集，无其它簇或新图混入目标子簇。',
               f'合并：第{merge_step}步将{len(overlaps)}个完整旧簇相加，无本次新图混入目标合并簇。',
               '中间的所有流式更新均实际执行；图中分别展示两个事件各自紧邻的前后状态。',
               '重建公式：v_hat_i = mu_C + U (z_i - U^T mu_C)。',
               '纯拆分中，未观测残差按人数分配，每成员的残差均值不变，因此旧叶子重建不变。',
               '合并后各源簇残差均值按人数加权；重建可能改变，但不保证恢复画质提高。', '',
               '量化指标（content_* 均排除灰色padding；同一目标原图作参照）：',
               *[f"{m['stage']}: cosine={m['latent_cosine']:.6f}, SSIM={m['content_ssim']:.6f}, LPIPS={m['content_lpips']:.6f}" for m in metrics],
               '', '该单图仅用于说明动态更新行为，不代表整个数据集的平均恢复质量。']
    (output / 'report.txt').write_text('\n'.join(report) + '\n')
    validation['output_sha256'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    dump(output / 'validation.json', validation)
    print(json.dumps({'completed': True, 'metrics': metrics, 'transitions': transitions}, indent=2), flush=True)


def draw_figure(output, metrics, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    entries = [('original', 'Original (256 x 256)'), ('vae_direct', 'VAE direct reconstruction')]
    for name in ('before_split', 'after_split', 'before_merge', 'after_merge'):
        m = next(m for m in metrics if m['stage'] == name)
        entries.append((name, f"{name.replace('_', ' ').capitalize()} | t={m['step']} | {m['cluster_size']} members"))
    scores = {m['stage']: m for m in metrics}
    fig, axes = plt.subplots(3, 2, figsize=(9, 12.4))
    for ax, (name, title) in zip(axes.flat, entries):
        with Image.open(output / 'images' / f'{name}.png') as image:
            ax.imshow(np.asarray(image), interpolation='nearest')
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if name in scores:
            s = scores[name]
            ax.set_xlabel(f"Latent cosine: {s['latent_cosine']:.3f} | SSIM: {s['content_ssim']:.3f}", fontsize=10)
    fig.suptitle(f"One image through VAMO split and merge\n{selection['filename']} | fixed 2-D coordinates + changing master", fontsize=15, y=.99)
    fig.text(.5, .014, f"VAE-latent adaptation (4096D) | Same decoder and preprocessing\nSplit: t{selection['split_step']-1} / t{selection['split_step']}. Merge: t{selection['merge_step']-1} / t{selection['merge_step']}; all intervening updates replayed.", ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95), h_pad=2.0)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'comparison.{ext}', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--split-step', type=int, required=True)
    parser.add_argument('--merge-step', type=int, required=True)
    parser.add_argument('--selection-note', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.output.resolve(), args.device, args.domain, args.image_id, args.split_step, args.merge_step, args.selection_note)
