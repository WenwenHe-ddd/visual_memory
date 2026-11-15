"""Read frozen Brand memory; compute independent image-reference scores only."""
from pathlib import Path
import os
import sys
import json
import csv
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
from scripts.graph.visual_image_encoder import ImageEncoder

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(4)
    source = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
    domain = source / 'Brand_Memory_Test'
    manifest = json.loads((source / 'manifest.json').read_text())
    record = next(r for r in manifest['domains'] if r['domain'] == domain.name)
    tracked = {}
    for name in ['memory/graph.json', 'memory/basis.npy', 'memory/master_sums.npy', 'input_manifest.json', 'protocol.json']:
        tracked[str(domain / name)] = sha(domain / name)
        assert tracked[str(domain / name)] == record['sha256'][name], name
    graph = json.loads((domain / 'memory/graph.json').read_text())
    protocol = json.loads((domain / 'protocol.json').read_text())
    arrivals = json.loads((domain / 'input_manifest.json').read_text())['arrivals']
    arrival_by_id = {r['image_id']: r for r in arrivals}
    nodes = {r['id']: r for r in graph['image_nodes']}
    basis = np.load(domain / 'memory/basis.npy')
    sums = np.load(domain / 'memory/master_sums.npy')
    assert basis.shape == (512, 2)
    assert np.max(np.abs(basis.T @ basis - np.eye(2))) < 1e-10
    meta = protocol['image_encoder']
    encoder = ImageEncoder(meta['model_name'], revision=meta['revision'], device='cpu', local_files_only=True)
    assert encoder.metadata == meta, 'Encoder or preprocessing differs from frozen source'
    rows, master_rows = [], []
    projection_errors = []
    for k, master in enumerate(graph['masters'], 1):
        total = sums[master['visual_sum_ref']['row']]
        center = total / master['n']
        reconstructed_sum = np.zeros(512)
        current_rows = []
        # Keep actual member order. Do not sort by measured similarity.
        for j, image_id in enumerate(master['member_ids'], 1):
            node, original = nodes[image_id], arrival_by_id[image_id]
            assert node['master_id'] == master['id']
            z = np.asarray(node['visual_coords'])
            rebuilt = center + basis @ (z - basis.T @ center)
            reconstructed_sum += rebuilt
            path = Path(original['source_path'])
            assert sha(path) == original['sha256'], path
            reference = np.asarray(encoder.encode(path))
            reference /= np.max(np.abs(reference))
            reference /= np.linalg.norm(reference)
            projection_error = float(np.max(np.abs(basis.T @ reference - z)))
            projection_errors.append(projection_error)
            assert projection_error < 1e-6, (image_id, projection_error)
            cosine = float(np.dot(rebuilt, reference) / (np.linalg.norm(rebuilt) * np.linalg.norm(reference)))
            assert -1.00000001 <= cosine <= 1.00000001
            cosine = float(np.clip(cosine, -1, 1))
            row = dict(master_label=f'M{k}', master_id=master['id'], master_n=master['n'],
                       master_approximate=master['approximate'], member_order=j, image_id=image_id,
                       filename=original['filename'], image_sha256=original['sha256'],
                       cosine_similarity=cosine, reconstruction_norm=float(np.linalg.norm(rebuilt)),
                       reference_projection_max_abs_error=projection_error)
            rows.append(row)
            current_rows.append(row)
            del reference, rebuilt
        assert np.max(np.abs(reconstructed_sum - total)) < 1e-10
        values = [r['cosine_similarity'] for r in current_rows]
        master_rows.append(dict(master_label=f'M{k}', master_id=master['id'], n=master['n'],
                                approximate=master['approximate'], mean_cosine=float(np.mean(values)),
                                min_cosine=min(values), max_cosine=max(values)))
        print(f"M{k}: n={master['n']}, cosine mean={np.mean(values):.6f}, min={min(values):.6f}", flush=True)
    assert len(rows) == len(nodes) == 30 and len(master_rows) == 8
    assert len({r['image_id'] for r in rows}) == len(nodes)
    for filename, data in [('memory_similarity.csv', rows), ('master_summary.csv', master_rows)]:
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for path, expected in tracked.items():
        assert sha(path) == expected, 'Source changed during evaluation'
    values = np.array([r['cosine_similarity'] for r in rows])
    audit = dict(source=str(domain), domain='Brand', n_memories=30, n_masters=8,
                 reconstruction='vhat_i = muhat_m + U @ (z_i - U.T @ muhat_m)',
                 reference='Independent evaluation-only reencoding of original images with exact frozen CLIP encoder/preprocessing',
                 original_image_reads_for_evaluation=30, source_memory_modified=False,
                 full_reference_vectors_saved=False, reconstructed_vectors_saved=False,
                 model_metadata=encoder.metadata, source_sha256=tracked,
                 max_reference_projection_error=max(projection_errors),
                 mean_cosine=float(values.mean()), median_cosine=float(np.median(values)),
                 min_cosine=float(values.min()), max_cosine=float(values.max()),
                 rendering_language='R', aggregation='one heatmap cell per memory; means across memories',
                 order='saved master order, then saved member order; no similarity-based sorting')
    (OUT / 'evaluation_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['mean_cosine', 'median_cosine', 'min_cosine', 'max_cosine', 'max_reference_projection_error']}, indent=2))

if __name__ == '__main__':
    main()
