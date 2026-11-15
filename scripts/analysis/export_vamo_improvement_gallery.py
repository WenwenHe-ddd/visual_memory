#!/usr/bin/env python3
"""Verify selected examples by fresh streaming replay, export triptychs and raw originals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vamo_gallery_mpl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.graph.caption_ward_visual import update_visual_partition


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers(device):
    path = ROOT / 'reports/brand_vae_vamo_20261105/run_experiment.py'
    spec = importlib.util.spec_from_file_location('frozen_gallery_vae_helpers', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.DEVICE = device
    protocol = read(path.parent / 'protocol.json')
    for name, value in protocol['model_sha256'].items():
        assert helper.sha(helper.MODEL / name) == value
    return helper


@torch.inference_mode()
def run(screen, selected_path, output, device):
    assert not output.exists(), 'Use a new output root.'
    selected = read(selected_path)
    assert len(selected) == 24
    assert len({sha(row['source_path']) for row in selected}) == 24, 'Need 24 distinct source images.'
    for row in selected:
        assert row['both_image_metrics_improved'] and row['last_merge_both_improved']
    output.mkdir(parents=True)
    for folder in ('originals', 'triptychs', 'panels', 'snapshots', 'preview_pages'):
        (output / folder).mkdir()
    sources = {str(path): sha(path) for path in [Path(__file__), selected_path,
        screen / 'shortlist.json', screen / 'shortlist_vectors.npz', screen / 'decoded/metrics.json',
        ROOT / 'scripts/graph/caption_ward_visual.py', ROOT / 'reports/brand_vae_vamo_20261105/run_experiment.py']}
    dump(output / 'selection.json', selected)
    helper = load_helpers(device)
    model = helper.model_load()
    grouped = {}
    for index, row in enumerate(selected, 1):
        grouped.setdefault(row['domain'], []).append((index, row))
    reconstructed = {}
    domain_checks = []
    for domain, cases in grouped.items():
        old = Path(cases[0][1]['old_experiment'])
        manifest = helper.SOURCE / domain / 'input_manifest.json'
        arrivals = read(manifest)['arrivals']
        partitions_path = Path(cases[0][1]['partition_audit'])
        partitions = [json.loads(line) for line in partitions_path.read_text().splitlines() if line.strip()]
        prior_path = old / 'replay_audit.jsonl'
        prior = [json.loads(line) for line in prior_path.read_text().splitlines() if line.strip()]
        assert len(arrivals) == len(partitions) == len(prior)
        basis_path = old / 'memory/basis.npy'
        basis = np.load(basis_path)
        for path in (manifest, partitions_path, prior_path, basis_path, old / 'memory/membership.json',
                     old / 'memory/master_sums.npy', old / 'memory/leaf_coords.npy'):
            sources[str(path)] = sha(path)
        masters, coords, next_index = {}, {}, 1
        total = np.zeros(4096)
        for step, (arrival, frozen, saved) in enumerate(zip(arrivals, partitions, prior), 1):
            assert arrival['image_id'] == frozen['image_id']
            image, _ = helper.prepare(arrival)
            visual = model.encode(image[None].to(device)).latent_dist.mode()[0].cpu().double().numpy().reshape(-1)
            total += visual
            masters, audit, next_index = update_visual_partition(masters, coords, frozen['clusters_partition'],
                {arrival['image_id']: visual}, basis, next_index)
            coords[arrival['image_id']] = basis.T @ visual
            assert audit['clusters'] == saved['clusters']
            np.testing.assert_allclose(sum(m['visual_sum'] for m in masters.values()), total, atol=1e-8, rtol=0)
            for index, row in cases:
                for stage, requested_step in [('after_split', row['split_step']),
                                               ('before_merge', row['merge_step'] - 1),
                                               ('after_merge', row['merge_step'])]:
                    if step != requested_step:
                        continue
                    master = next(m for m in masters.values() if row['image_id'] in m['member_ids'])
                    folder = output / 'snapshots' / f'{index:02d}' / stage
                    folder.mkdir(parents=True)
                    np.save(folder / 'master_sum.npy', master['visual_sum'])
                    np.save(folder / 'target_coords.npy', coords[row['image_id']])
                    np.save(folder / 'basis.npy', basis)
                    dump(folder / 'metadata.json', {'step': step, 'master_id': master['id'], 'n': master['n'],
                         'image_id': row['image_id'], 'member_ids': master['member_ids']})
                    mean = np.load(folder / 'master_sum.npy') / master['n']
                    z = np.load(folder / 'target_coords.npy')
                    reconstructed[(index, stage)] = mean + basis @ (z - basis.T @ mean)
            del visual, image
        saved = read(old / 'memory/membership.json')
        sums = np.stack([masters[mid]['visual_sum'] for mid in saved['master_ids']])
        leaf_coords = np.stack([coords[mid] for mid in saved['leaf_ids']])
        sum_error = float(np.max(np.abs(sums - np.load(old / 'memory/master_sums.npy'))))
        coords_error = float(np.max(np.abs(leaf_coords - np.load(old / 'memory/leaf_coords.npy'))))
        assert sum_error < 1e-6 and coords_error < 1e-6
        domain_checks.append({'domain': domain, 'fresh_arrival_encodes': len(arrivals),
            'prefix_partitions_matched': len(arrivals), 'final_sum_max_abs': sum_error, 'final_coords_max_abs': coords_error})
        print(json.dumps({'replay_completed': domain, 'arrivals': len(arrivals)}), flush=True)

    # All fresh streams are finished before loading full evaluation references.
    import lpips
    from skimage.metrics import structural_similarity
    perceptual = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    results = []
    max_screen_error = 0.0
    screened = np.load(screen / 'shortlist_vectors.npz')
    for index, row in enumerate(selected, 1):
        folder = output / 'panels' / f'{index:02d}'
        folder.mkdir()
        manifest = read(helper.SOURCE / row['domain'] / 'input_manifest.json')['arrivals']
        arrival = next(a for a in manifest if a['image_id'] == row['image_id'])
        original, meta = helper.prepare(arrival)
        original = original[None].to(device)
        helper.output_tensor(original[0], folder / 'original.png')
        filename = f"{index:02d}__{row['domain']}__{row['filename']}"
        destination = output / 'originals' / filename
        shutil.copyfile(row['source_path'], destination)
        assert sha(destination) == arrival['sha256']
        sources[row['source_path']] = arrival['sha256']
        x0, y0, x1, y1 = meta['content_box']
        truth = ((original[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        result = dict(row, index=index, original_archive_filename=filename, original_sha256=arrival['sha256'])
        for stage in ('after_split', 'before_merge', 'after_merge'):
            vector = reconstructed[(index, stage)]
            error = float(np.max(np.abs(vector - screened[f"{row['candidate_id']}__{stage}"])))
            max_screen_error = max(max_screen_error, error)
            assert error < 1e-6
            image = model.decode(torch.tensor(vector.reshape(1, 4, 32, 32), dtype=torch.float32, device=device)).sample.clamp(-1, 1)
            prediction = ((image[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
            ssim = float(structural_similarity(truth, prediction, data_range=1, channel_axis=2,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
            distance = float(perceptual(image[:, :, y0:y1, x0:x1], original[:, :, y0:y1, x0:x1]).item())
            assert abs(ssim - row[f'{stage}_ssim']) < 1e-4
            assert abs(distance - row[f'{stage}_lpips']) < 1e-4
            result[f'{stage}_ssim'] = ssim
            result[f'{stage}_lpips'] = distance
            if stage != 'before_merge':
                helper.output_tensor(image[0], folder / f'{stage}.png')
        results.append(result)
        print(json.dumps({'example_completed': index, 'filename': row['filename']}), flush=True)
    dump(output / 'metrics.json', results)
    columns = ['index','domain','image_id','filename','split_step','merge_step','split_n','merge_n',
        'unchanged_between','merge_has_new_image','merge_has_partial_source','after_split_ssim',
        'before_merge_ssim','after_merge_ssim','after_split_lpips','before_merge_lpips','after_merge_lpips',
        'ssim_gain','lpips_improvement','last_merge_ssim_gain','last_merge_lpips_improvement',
        'original_archive_filename','original_sha256']
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    draw(output, results)
    zip_path = output / 'original_images_24.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((output / 'originals').iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == 24
        for result in results:
            assert hashlib.sha256(archive.read(result['original_archive_filename'])).hexdigest() == result['original_sha256']
    stable = sum(row['unchanged_between'] for row in results)
    (output / 'README.txt').write_text(
        '24张不同原图，72个展示面板：原图、split之后、merge之后。\n'
        'original_images_24.zip仅收录24张未缩放、未重编码的原始图片；与源文件SHA256完全一致。\n'
        'triptychs.pdf包含24页三联图；preview_pages每页4组，共6页。panels共72个256px预处理/解码面板。\n'
        '这是用户要求的按改善效果筛选的案例集，不是平均性能评估。改善程度有差异。\n'
        '每张图片的after_merge相对after_split，以及紧邻before_merge，均SSIM上升且LPIPS下降。\n'
        f'{stable}个案例从split后至merge前保持表征不变；其余{24-stable}个包含中间更新，不能将整个时间区间的改善单独归因于最后合并。\n'
        '包含新图片加入或部分旧簇迁移的真实更新事件已在metrics.csv中单独标记。\n'
        '固定原VAE、二维基底与caption分簇轨迹，按新图到达顺序重放；旧图完整表征不参与更新。\n'
        '这是4096维原始VAE latent版VAMO，不是将512维CLIP表征直接输入VAE。\n')
    assert all(sha(path) == expected for path, expected in sources.items())
    dump(output / 'validation.json', {'passed': True, 'completed': True, 'selected_photos': 24,
        'unique_original_hashes': len({row['original_sha256'] for row in results}),
        'panel_count': len(list((output / 'panels').glob('*/*.png'))), 'zip_entries': 24,
        'zip_original_bytes_match_sources': True, 'stable_interval_examples': stable,
        'old_full_latents_used_in_updates': False, 'fresh_replay': domain_checks,
        'max_fresh_screen_vector_difference': max_screen_error, 'source_sha256': sources,
        'output_sha256': {str(path.relative_to(output)): sha(path) for path in output.rglob('*') if path.is_file()}})


def draw(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'svg.fonttype':'none'})

    def panels(axes, row):
        for ax, stage in zip(axes, ('original','after_split','after_merge')):
            with Image.open(output / 'panels' / f"{row['index']:02d}" / f'{stage}.png') as image:
                ax.imshow(np.asarray(image), interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if stage == 'original':
                ax.set_title(f"{row['index']:02d} | Original | {row['filename']}", fontsize=10)
            else:
                step = row['split_step'] if stage == 'after_split' else row['merge_step']
                n = row['split_n'] if stage == 'after_split' else row['merge_n']
                ax.set_title(f"{stage.replace('_',' ').capitalize()} | t={step} | n={n}", fontsize=10)
                ax.set_xlabel(f"SSIM {row[stage+'_ssim']:.3f} | LPIPS {row[stage+'_lpips']:.3f}", fontsize=9)
    with PdfPages(output / 'triptychs.pdf') as pdf:
        for row in rows:
            fig, axes = plt.subplots(1,3,figsize=(12,4.3))
            panels(axes, row); fig.tight_layout()
            fig.savefig(output / 'triptychs' / f"{row['index']:02d}.png", dpi=180, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    with PdfPages(output / 'overview.pdf') as pdf:
        for start in range(0,len(rows),4):
            fig, axes = plt.subplots(4,3,figsize=(11,14))
            for axrow, row in zip(axes,rows[start:start+4]): panels(axrow,row)
            fig.tight_layout(h_pad=2)
            fig.savefig(output / 'preview_pages' / f'page_{start//4+1:02d}.png', dpi=130, bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    run(args.screen.resolve(),args.selected.resolve(),args.output.resolve(),args.device)
