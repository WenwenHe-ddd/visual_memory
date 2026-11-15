#!/usr/bin/env python3
"""Decode shortlisted historical split/merge reconstructions for case selection.

This is explicitly a post-hoc qualitative screen. Cached reference latents do
not replace the fresh chronological replay of the finally selected example.
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
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vamo_merge_screen_mpl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def load_helpers(device):
    path = ROOT / 'reports/brand_vae_vamo_20261105/run_experiment.py'
    spec = importlib.util.spec_from_file_location('frozen_vae_helpers', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.DEVICE = device
    protocol = helper.read(path.parent / 'protocol.json')
    for name, expected in protocol['model_sha256'].items():
        assert helper.sha(helper.MODEL / name) == expected
    return helper


@torch.inference_mode()
def main(screen, device):
    import lpips
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    helper = load_helpers(device)
    candidates = helper.read(screen / 'shortlist.json')
    vectors = np.load(screen / 'shortlist_vectors.npz')
    output = screen / 'decoded'
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), screen / 'shortlist.json', screen / 'shortlist_vectors.npz']
    source_hashes = {str(p): helper.sha(p) for p in sources}
    model = helper.model_load()
    perceptual = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    streams = {}
    results = []
    for candidate in candidates:
        cid, domain = candidate['candidate_id'], candidate['domain']
        if domain not in streams:
            rows = helper.read(helper.SOURCE / domain / 'input_manifest.json')['arrivals']
            streams[domain] = {row['image_id']: row for row in rows}
        row = streams[domain][candidate['image_id']]
        original, meta = helper.prepare(row)
        original = original[None].to(device)
        x0, y0, x1, y1 = meta['content_box']
        truth = ((original[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        directory = output / cid
        directory.mkdir()
        helper.output_tensor(original[0], directory / 'original.png')
        scores = {}
        for stage in ('after_split', 'after_merge'):
            latent = vectors[f'{cid}__{stage}']
            tensor = torch.tensor(latent.reshape(1, 4, 32, 32), dtype=torch.float32, device=device)
            image = model.decode(tensor).sample.clamp(-1, 1)
            helper.output_tensor(image[0], directory / f'{stage}.png')
            prediction = ((image[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
            scores[stage] = {
                'ssim': float(structural_similarity(truth, prediction, data_range=1, channel_axis=2,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                'psnr': float(peak_signal_noise_ratio(truth, prediction, data_range=1)),
                'lpips': float(perceptual(image[:, :, y0:y1, x0:x1], original[:, :, y0:y1, x0:x1]).item()),
            }
        result = dict(candidate)
        for stage, score in scores.items():
            result.update({f'{stage}_{key}': value for key, value in score.items()})
        result['ssim_gain'] = scores['after_merge']['ssim'] - scores['after_split']['ssim']
        result['lpips_improvement'] = scores['after_split']['lpips'] - scores['after_merge']['lpips']
        result['both_image_metrics_improved'] = result['ssim_gain'] > 0 and result['lpips_improvement'] > 0
        results.append(result)
        print(json.dumps(result), flush=True)

    helper.write(output / 'metrics.json', results)
    columns = list(dict.fromkeys(key for row in results for key in row))
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    ordered = sorted(results, key=lambda row: (row['both_image_metrics_improved'], row['lpips_improvement']), reverse=True)
    helper.write(output / 'ranked_candidates.json', ordered)
    for page, start in enumerate(range(0, len(ordered), 5), 1):
        items = ordered[start:start + 5]
        fig, axes = plt.subplots(len(items), 3, figsize=(11, len(items) * 3.4), squeeze=False)
        for row_axes, item in zip(axes, items):
            for ax, stage in zip(row_axes, ('original', 'after_split', 'after_merge')):
                with Image.open(output / item['candidate_id'] / f'{stage}.png') as image:
                    ax.imshow(np.asarray(image), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if stage == 'original':
                    ax.set_title(f"{item['candidate_id']} | {item['filename']}", fontsize=10)
                else:
                    step = item['split_step'] if stage == 'after_split' else item['merge_step']
                    ax.set_title(f"{stage.replace('_', ' ')} | t={step}", fontsize=10)
                    ax.set_xlabel(f"SSIM {item[stage + '_ssim']:.3f} | LPIPS {item[stage + '_lpips']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f'review_page_{page:02d}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    helper.write(output / 'validation.json', {
        'completed': True, 'candidate_count': len(results),
        'both_image_metrics_improved': sum(row['both_image_metrics_improved'] for row in results),
        'selection': 'Post-hoc improvement-case screen; not an aggregate performance estimate.',
        'device': device, 'source_sha256': source_hashes,
        'source_files_unchanged': all(helper.sha(path) == value for path, value in source_hashes.items()),
        'output_sha256': {str(path.relative_to(output)): helper.sha(path) for path in output.rglob('*') if path.is_file()},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    main(args.screen.resolve(), args.device)
