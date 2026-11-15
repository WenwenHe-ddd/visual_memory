#!/usr/bin/env python3
"""Fresh chronological replay of a post-hoc split/merge improvement example.

Exactly three panels are drawn: original, after split, after merge. The memory
update sees only the arriving image's raw VAE latent, prior master aggregates,
and prior two-dimensional coordinates. Evaluation references and screened
vectors are loaded only after the complete stream has been replayed.
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
os.environ.setdefault('MPLCONFIGDIR', '/tmp/vamo_selected_triptych_mpl')
ROOT = Path(__file__).resolve().parents[2]
BRAND = ROOT / 'reports/brand_vae_vamo_20261105'
SWEEP = ROOT / 'reports/other_domains_vae_vamo_20261105'
HELPERS = BRAND / 'run_experiment.py'
CORE = ROOT / 'scripts/graph/caption_ward_visual.py'
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.graph.caption_ward_visual import update_visual_partition


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose(screen, candidate_id):
    """Resolve identity only from saved screening records, never CLI overrides."""
    shortlisted = [row for row in read(screen / 'shortlist.json') if row['candidate_id'] == candidate_id]
    decoded = [row for row in read(screen / 'decoded/metrics.json') if row['candidate_id'] == candidate_id]
    assert len(shortlisted) == len(decoded) == 1, 'Candidate must occur once in both screening records.'
    candidate = decoded[0]
    for key, value in shortlisted[0].items():
        assert candidate[key] == value, (key, 'shortlist/decoded mismatch')
    assert candidate['unchanged_between'], 'Intervening updates would confound the merge comparison.'
    assert candidate['after_merge_ssim'] > candidate['after_split_ssim']
    assert candidate['after_merge_lpips'] < candidate['after_split_lpips']
    assert 1 < candidate['split_step'] < candidate['merge_step']
    return candidate


def load_inputs(screen, candidate_id, device):
    candidate = choose(screen, candidate_id)
    domain = candidate['domain']
    old_root = BRAND if domain == 'Brand_Memory_Test' else SWEEP / domain
    assert old_root.resolve() == Path(candidate['old_experiment']).resolve()
    protocol_path = BRAND / 'protocol.json' if domain == 'Brand_Memory_Test' else SWEEP / 'protocol.json'
    protocol = read(protocol_path)
    spec = importlib.util.spec_from_file_location('frozen_selected_vae_helpers', HELPERS)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.DEVICE = device
    for name, expected in protocol['model_sha256'].items():
        assert sha(helper.MODEL / name) == expected, name
    for path, expected in protocol['source_sha256'].items():
        assert sha(path) == expected, path
    source_domain = helper.SOURCE / domain
    if domain == 'Brand_Memory_Test':
        assert read(BRAND / 'validation.json')['passed']
        partition_path = source_domain / 'insertion_audit.jsonl'
    else:
        completed = read(old_root / 'completed.json')
        assert completed['completed']
        partition_path = Path(completed['source_audit'])
    assert partition_path.resolve() == Path(candidate['partition_audit']).resolve()
    arrivals = read(source_domain / 'input_manifest.json')['arrivals']
    partitions = lines(partition_path)
    prior_audits = lines(old_root / 'replay_audit.jsonl')
    assert len(arrivals) == len(partitions) == len(prior_audits)
    lookup = {row['image_id']: row for row in arrivals}
    target = candidate['image_id']
    index = next(i for i, row in enumerate(arrivals) if row['image_id'] == target)
    assert index == candidate['target_index']
    assert lookup[target]['filename'] == candidate['filename']
    basis = np.load(old_root / 'memory/basis.npy')
    assert basis.shape == (4096, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12, rtol=0)

    def membership(step):
        return next(c for c in partitions[step - 1]['clusters'] if target in c['member_ids'])

    split, merge = candidate['split_step'], candidate['merge_step']
    before_split, after_split = membership(split - 1), membership(split)
    assert set(after_split['member_ids']) < set(before_split['member_ids'])
    assert before_split['master_id'] in prior_audits[split - 1]['split_master_ids']
    assert after_split['master_id'] not in prior_audits[split - 1]['merged_master_ids']
    assert not set(after_split['member_ids']) & set(prior_audits[split - 1]['new_image_ids'])
    after_merge = membership(merge)
    overlaps = [r for r in prior_audits[merge - 1]['source_overlaps'] if r['new_master_id'] == after_merge['master_id']]
    assert len(overlaps) >= 2
    assert after_merge['master_id'] in prior_audits[merge - 1]['merged_master_ids']
    merge_new_ids = sorted(set(after_merge['member_ids']) & set(prior_audits[merge - 1]['new_image_ids']))
    merge_event = {'source_overlaps': overlaps, 'merge_has_new_image': bool(merge_new_ids),
                   'merge_new_ids': merge_new_ids,
                   'merge_has_partial_source': any(r['count'] < r['old_n'] for r in overlaps)}
    for key in ('merge_has_new_image', 'merge_new_ids', 'merge_has_partial_source'):
        if key in candidate:
            assert candidate[key] == merge_event[key]
    candidate['verified_merge_event'] = merge_event
    assert after_split['member_ids'] == candidate['split_members']
    assert after_merge['member_ids'] == candidate['merge_members']
    sources = [Path(__file__), HELPERS, CORE, protocol_path,
               old_root / 'memory/basis.npy', old_root / 'memory/master_sums.npy',
               old_root / 'memory/leaf_coords.npy', old_root / 'memory/membership.json',
               old_root / 'replay_audit.jsonl', source_domain / 'input_manifest.json', partition_path,
               screen / 'shortlist.json', screen / 'shortlist_vectors.npz', screen / 'decoded/metrics.json']
    source_hashes = {str(p): sha(p) for p in sources}
    return candidate, helper, old_root, protocol, arrivals, partitions, prior_audits, lookup, basis, source_hashes


def rebuild(snapshot, basis):
    metadata = read(snapshot / 'metadata.json')
    mean = np.load(snapshot / 'master_sum.npy') / metadata['n']
    coords = np.load(snapshot / 'target_coords.npy')
    return mean + basis @ (coords - basis.T @ mean)


def draw(output, metrics, candidate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    scores = {row['stage']: row for row in metrics}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, stage in zip(axes, ('original', 'after_split', 'after_merge')):
        with Image.open(output / 'images' / f'{stage}.png') as image:
            ax.imshow(np.asarray(image), interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if stage == 'original':
            ax.set_title('Original', fontsize=14, pad=12)
        else:
            row = scores[stage]
            ax.set_title(f"{stage.replace('_', ' ').capitalize()} | t={row['step']} | n={row['cluster_size']}", fontsize=14, pad=12)
            ax.set_xlabel(f"SSIM {row['content_ssim']:.3f} | LPIPS {row['content_lpips']:.3f}", fontsize=11, labelpad=10)
    fig.suptitle(candidate['filename'], fontsize=16, y=1.01)
    fig.tight_layout(pad=1.0, w_pad=1.0)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'comparison.{suffix}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


@torch.inference_mode()
def run(screen, candidate_id, output, device, check_only=False):
    loaded = load_inputs(screen, candidate_id, device)
    candidate, helper, old_root, protocol, arrivals, partitions, prior_audits, lookup, basis, source_hashes = loaded
    if check_only:
        print(json.dumps({'preflight_passed': True, 'candidate': candidate, 'arrivals': len(arrivals)}, indent=2))
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; no historical output is overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'snapshots').mkdir()
    (output / 'images').mkdir()
    np.save(output / 'basis.npy', basis)
    target = candidate['image_id']
    # A step can have two roles when the split and merge occur consecutively.
    stages = {'after_split': candidate['split_step'], 'before_merge': candidate['merge_step'] - 1,
              'after_merge': candidate['merge_step']}
    note = ('Post-hoc qualitative improvement example selected from actual frozen stream events '
            'using decoded SSIM/LPIPS and visual inspection. Latent cosine improvement is not '
            'required. This selected example is not an estimate of average performance.')
    dump(output / 'selection.json', {**candidate, 'selection_note': note, 'screen': str(screen), 'snapshot_steps': stages})
    model = helper.model_load()
    masters, coords, next_index = {}, {}, 1
    total = np.zeros(4096)
    audits, timeline = [], []
    for step, (row, frozen, previous) in enumerate(zip(arrivals, partitions, prior_audits), 1):
        assert row['image_id'] == frozen['image_id']
        image, _ = helper.prepare(row)
        # No array of earlier per-image latents exists in this replay.
        visual = model.encode(image[None].to(device)).latent_dist.mode()[0].cpu().double().numpy().reshape(-1)
        assert visual.shape == (4096,)
        total += visual
        masters, audit, next_index = update_visual_partition(
            masters, coords, frozen['clusters_partition'], {row['image_id']: visual}, basis, next_index)
        coords[row['image_id']] = basis.T @ visual
        assert audit['clusters'] == previous['clusters']
        assert {key: tuple(value['member_ids']) for key, value in masters.items()} == {
            c['master_id']: tuple(c['member_ids']) for c in frozen['clusters']}
        np.testing.assert_allclose(sum(m['visual_sum'] for m in masters.values()), total, atol=1e-8, rtol=0)
        assert audit['max_projection_consistency_l2_error'] < 1e-8
        audits.append({'step': step, **audit})
        current = next((m for m in masters.values() if target in m['member_ids']), None)
        if current is not None:
            meta = {'step': step, 'target_image_id': target, 'target_filename': candidate['filename'],
                    'master_id': current['id'], 'n': current['n'], 'member_ids': current['member_ids'],
                    'member_filenames': [lookup[mid]['filename'] for mid in current['member_ids']],
                    'approximate': bool(current['approximate']), 'new_arrival': row['filename']}
            timeline.append(meta)
            for name, saved_step in stages.items():
                if saved_step == step:
                    folder = output / 'snapshots' / name
                    folder.mkdir()
                    np.save(folder / 'master_sum.npy', current['visual_sum'])
                    np.save(folder / 'target_coords.npy', coords[target])
                    dump(folder / 'metadata.json', meta)
        del visual, image
        print(json.dumps({'stage': 'replay', 'step': step, 'total': len(arrivals)}), flush=True)
    saved = read(old_root / 'memory/membership.json')
    sums = np.stack([masters[mid]['visual_sum'] for mid in saved['master_ids']])
    leaf_coords = np.stack([coords[mid] for mid in saved['leaf_ids']])
    sum_error = float(np.max(np.abs(sums - np.load(old_root / 'memory/master_sums.npy'))))
    coord_error = float(np.max(np.abs(leaf_coords - np.load(old_root / 'memory/leaf_coords.npy'))))
    assert sum_error < 1e-6 and coord_error < 1e-6, (sum_error, coord_error)
    dump(output / 'timeline.json', timeline)
    (output / 'replay_audit.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in audits))
    rebuilt = {name: rebuild(output / 'snapshots' / name, basis) for name in stages}
    stable_error = float(np.max(np.abs(rebuilt['after_split'] - rebuilt['before_merge'])))
    assert stable_error < 1e-9

    # The stream has finished. Only now open evaluation references and screen vectors.
    reference_path = old_root / 'evaluation_only/full_latents.npy'
    source_hashes[str(reference_path)] = sha(reference_path)
    reference = np.load(reference_path, mmap_mode='r')[candidate['target_index']].reshape(-1).astype(np.float64)
    assert reference.shape == (4096,)  # Brand stores N x 4 x 32 x 32, other domains N x 4096.
    with np.load(screen / 'shortlist_vectors.npz') as screened:
        screen_error = max(float(np.max(np.abs(rebuilt[name] - screened[f'{candidate_id}__{name}'])))
                           for name in ('after_split', 'after_merge'))
    assert screen_error < 1e-6, screen_error
    original, metadata = helper.prepare(lookup[target])
    x0, y0, x1, y1 = metadata['content_box']
    original = original[None].to(device)
    helper.output_tensor(original[0], output / 'images/original.png')
    truth = ((original[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
    import lpips
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    perceptual = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval()
    metrics = []
    for name in ('after_split', 'after_merge'):
        vector = rebuilt[name]
        image = model.decode(torch.tensor(vector.reshape(1, 4, 32, 32), dtype=torch.float32, device=device)).sample.clamp(-1, 1)
        helper.output_tensor(image[0], output / 'images' / f'{name}.png')
        prediction = ((image[0] + 1) / 2).permute(1, 2, 0).cpu().numpy()[y0:y1, x0:x1]
        meta = read(output / 'snapshots' / name / 'metadata.json')
        metrics.append({'stage': name, 'step': meta['step'], 'master_id': meta['master_id'], 'cluster_size': meta['n'],
                        'latent_cosine': float(vector @ reference / (np.linalg.norm(vector) * np.linalg.norm(reference))),
                        'latent_relative_l2': float(np.linalg.norm(vector - reference) / np.linalg.norm(reference)),
                        'content_ssim': float(structural_similarity(truth, prediction, data_range=1, channel_axis=2,
                            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
                        'content_psnr': float(peak_signal_noise_ratio(truth, prediction, data_range=1)),
                        'content_lpips': float(perceptual(image[:, :, y0:y1, x0:x1], original[:, :, y0:y1, x0:x1]).item())})
        for key in ('ssim', 'lpips', 'psnr'):
            assert abs(metrics[-1][f'content_{key}'] - candidate[f'{name}_{key}']) < 1e-4
    split, merge = metrics
    assert merge['content_ssim'] > split['content_ssim']
    assert merge['content_lpips'] < split['content_lpips']
    with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    dump(output / 'metrics.json', metrics)
    draw(output, metrics, candidate)
    report = [f"目标：{candidate['domain']} / {candidate['filename']}",
              '三联图：原图、拆分后、合并后；每张图均不做锐化、美化或生成式补全。',
              '沿用固定的 sd-vae-ft-mse、256px 等比例灰边预处理和4096维原始latent版VAMO。',
              '不是把原版512维CLIP表征直接输入VAE。',
              f'新图到达时各编码一次，共{len(arrivals)}次；旧图完整latent不参与记忆更新。',
              '从真实历史时点的master sum/count和叶子二维坐标重建，然后由原VAE解码。',
              f"拆分发生在t={candidate['split_step']}，合并发生在t={candidate['merge_step']}。",
              f'拆分后至合并前重建最大变化={stable_error:.3g}，故变化来自合并发生的该次更新。',
              f"该次更新是否纳入新到图片：{candidate['verified_merge_event']['merge_has_new_image']}；是否含部分旧簇迁移：{candidate['verified_merge_event']['merge_has_partial_source']}。",
              '如含新图或部分旧簇迁移，该变化代表整个真实更新事件，不能单独归因于纯整簇合并。',
              '这是按恢复效果事后选取的改善案例，不代表平均效果，也不保证每次合并改善。',
              '仅要求SSIM上升、LPIPS下降；latent余弦相似度不必上升。', '']
    for name in stages:
        meta = read(output / 'snapshots' / name / 'metadata.json')
        report.append(f"{name}, t={meta['step']}, n={meta['n']}, {meta['master_id']}: " + ', '.join(meta['member_filenames']))
    report.extend(['', '指标相对同一预处理原图计算，排除灰色padding：'])
    report.extend(f"{row['stage']}: SSIM={row['content_ssim']:.6f}, LPIPS={row['content_lpips']:.6f}, latent cosine={row['latent_cosine']:.6f}" for row in metrics)
    (output / 'report.txt').write_text('\n'.join(report) + '\n')
    assert all(sha(path) == value for path, value in source_hashes.items())
    validation = {'passed': True, 'completed': True, 'candidate_id': candidate_id, 'target_image_id': target,
                  'selection': note, 'panel_count': 3, 'panels': ['original', 'after_split', 'after_merge'],
                  'representation': 'Raw 4096D VAE posterior mode; not the 512D CLIP retrieval representation',
                  'model': protocol.get('model', protocol.get('model_source')), 'device': device,
                  'leaf_coordinates': 2, 'image_arrival_forwards': len(arrivals),
                  'old_images_reencoded_during_updates': 0, 'old_full_latents_used_for_updates': False,
                  'evaluation_references_opened_after_stream': True,
                  'historical_prefix_partitions_matched': len(arrivals),
                  'final_master_sum_max_abs_error': sum_error, 'final_coords_max_abs_error': coord_error,
                  'split_to_before_merge_max_abs': stable_error,
                  'fresh_replay_screen_latent_max_abs': screen_error,
                  'ssim_gain': merge['content_ssim'] - split['content_ssim'],
                  'lpips_improvement': split['content_lpips'] - merge['content_lpips'],
                  'latent_cosine_gain': merge['latent_cosine'] - split['latent_cosine'],
                  'merge_event': candidate['verified_merge_event'],
                  'source_files_unchanged': True, 'source_sha256': source_hashes,
                  'model_sha256': protocol['model_sha256'],
                  'output_sha256': {str(path.relative_to(output)): sha(path) for path in output.rglob('*') if path.is_file()}}
    dump(output / 'validation.json', validation)
    print(json.dumps({'completed': True, 'output': str(output), 'metrics': metrics}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-only', action='store_true', help='Read-only source and event validation; no model or output.')
    args = parser.parse_args()
    run(args.screen.resolve(), args.candidate_id, args.output.resolve(), args.device, args.check_only)
