#!/usr/bin/env python3
"""Offline cache enumeration for image-metric screening; no GPU/models.

The existing evaluation latent cache supplies one current arrival vector per
update. This is explicitly a screening replay, not a fresh streaming experiment.
References are also used for candidate scoring, never in the update function.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, json, csv, hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.graph.caption_ward_visual import update_visual_partition

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
SWEEP = ROOT / 'reports/other_domains_vae_vamo_20261105'
BRAND = ROOT / 'reports/brand_vae_vamo_20261105'

def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def rebuild(master, z, basis):
    mean = master['visual_sum'] / master['n']
    return mean + basis @ (z - basis.T @ mean)
def cosine(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def main():
    assert not (OUT / 'all_candidates.json').exists(), 'Do not overwrite a completed screen.'
    domains = [(p.name, p, Path(read(p / 'completed.json')['source_audit']))
               for p in sorted(SWEEP.iterdir()) if (p / 'completed.json').exists()]
    domains += [('Brand_Memory_Test', BRAND, SOURCE / 'Brand_Memory_Test/insertion_audit.jsonl')]
    protocol = {
        'purpose': 'Explicit outcome-based illustrative example selection requested by user; not an unbiased evaluation.',
        'representation': 'Existing 4096D raw VAE latent adaptation with domain-specific fixed PCA-2D basis.',
        'screen_input': 'Existing evaluation_only/full_latents.npy, one current arrival row supplied per update. No GPU or fresh encoding in this screening stage.',
        'updates': 'Same update_visual_partition helper; only prior masters, prior 2D coordinates, frozen caption partition and current arrival vector.',
        'events': 'Pure subset split followed at least one arrival later by a merge of at least two complete old masters, with no current arriving image in the target merge.',
        'stable_interval': 'Every reconstruction after split and before merge must remain within absolute tolerance 1e-9 of the split reconstruction.',
        'shortlist': 'All candidates with stable interval; no latent cosine or L2 prefilter. Deterministic sort by domain, target, split step and merge step. All stable vectors exported for downstream image metric evaluation.',
        'final_confirmation': 'Shortlist is for decoder inspection; chosen example must be independently replayed by encoding current arrivals again.',
        'domains': [name for name, _, _ in domains],
    }
    dump(OUT / 'selection_protocol.json', protocol)
    candidates, candidate_vectors, validations, sources = [], {}, [], [Path(__file__), ROOT / 'scripts/graph/caption_ward_visual.py']
    for domain, old, partition_path in domains:
        manifest_path = SOURCE / domain / 'input_manifest.json'
        arrivals = read(manifest_path)['arrivals']
        frozen = lines(partition_path)
        prior_audits = lines(old / 'replay_audit.jsonl')
        basis_path = old / 'memory/basis.npy'
        latent_path = old / 'evaluation_only/full_latents.npy'
        basis = np.load(basis_path)
        cache = np.load(latent_path, mmap_mode='r')
        assert len(arrivals) == len(frozen) == len(prior_audits) == len(cache)
        assert basis.shape == (4096, 2)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12, rtol=0)
        source_files = [manifest_path, partition_path, basis_path, latent_path, old / 'replay_audit.jsonl',
                        old / 'memory/membership.json', old / 'memory/master_sums.npy', old / 'memory/leaf_coords.npy']
        sources.extend(source_files)
        lookup = {r['image_id']: r for r in arrivals}
        indices = {r['image_id']: i for i, r in enumerate(arrivals)}
        masters, coords, next_index = {}, {}, 1
        active_splits = {}
        domain_candidate_start = len(candidates)
        split_events = 0
        for step, (row, part) in enumerate(zip(arrivals, frozen), 1):
            assert row['image_id'] == part['image_id']
            previous = masters
            current_arrival = np.array(cache[step - 1], dtype=np.float64).reshape(-1)
            masters, audit, next_index = update_visual_partition(
                previous, coords, part['clusters_partition'], {row['image_id']: current_arrival}, basis, next_index)
            coords[row['image_id']] = basis.T @ current_arrival
            assert audit['clusters'] == prior_audits[step - 1]['clusters']
            overlaps = {}
            for overlap in audit['source_overlaps']:
                overlaps.setdefault(overlap['new_master_id'], []).append(overlap)
            current_vectors = {}
            for mid, master in masters.items():
                ov = overlaps.get(mid, [])
                contains_new = row['image_id'] in master['member_ids']
                is_merge = len(ov) >= 2 and all(o['count'] == o['old_n'] for o in ov) and not contains_new
                is_split = len(ov) == 1 and ov[0]['count'] < ov[0]['old_n'] and not contains_new
                if is_split:
                    split_events += 1
                for target in master['member_ids']:
                    vector = rebuild(master, coords[target], basis)
                    current_vectors[target] = vector
                    if is_merge:
                        prior_owner = next(m for m in previous.values() if target in m['member_ids'])
                        prior_vector = rebuild(prior_owner, coords[target], basis)
                        reference = np.array(cache[indices[target]], dtype=np.float64).reshape(-1)
                        for split in active_splits.get(target, []):
                            if step < split['step'] + 1:
                                continue
                            a = split['vector']
                            c0, c1 = cosine(a, reference), cosine(vector, reference)
                            e0 = float(np.linalg.norm(a - reference) / np.linalg.norm(reference))
                            e1 = float(np.linalg.norm(vector - reference) / np.linalg.norm(reference))
                            stable = split['unchanged'] and bool(np.max(np.abs(a - prior_vector)) < 1e-9)
                            cid = f'candidate_{len(candidates)+1:05d}'
                            item = dict(candidate_id=cid, domain=domain, image_id=target,
                                        filename=lookup[target]['filename'], source_path=lookup[target]['source_path'],
                                        target_index=indices[target], split_step=split['step'], merge_step=step,
                                        split_n=split['n'], merge_n=master['n'], before_split_n=split['old_n'],
                                        before_merge_n=prior_owner['n'], split_master=split['master_id'], merge_master=mid,
                                        cos_split=c0, cos_merge=c1, gain=c1-c0,
                                        relative_l2_split=e0, relative_l2_merge=e1, relative_l2_gain=e0-e1,
                                        unchanged_between=stable, split_members=split['member_ids'],
                                        merge_members=list(master['member_ids']),
                                        split_member_filenames=[lookup[k]['filename'] for k in split['member_ids']],
                                        merge_member_filenames=[lookup[k]['filename'] for k in master['member_ids']],
                                        merge_source_sizes=[o['old_n'] for o in ov],
                                        split_to_before_merge_max_abs=float(np.max(np.abs(a - prior_vector))),
                                        old_experiment=str(old), partition_audit=str(partition_path))
                            candidates.append(item)
                            candidate_vectors[cid] = dict(after_split=a.copy(), after_merge=vector.copy(), reference=reference.copy())
                    if is_split:
                        active_splits.setdefault(target, []).append(dict(step=step, vector=vector.copy(), unchanged=True,
                            n=master['n'], old_n=ov[0]['old_n'], master_id=mid, member_ids=list(master['member_ids'])))
            for target, splits in active_splits.items():
                for split in splits:
                    if np.max(np.abs(current_vectors[target] - split['vector'])) >= 1e-9:
                        split['unchanged'] = False
            del current_arrival
        saved = read(old / 'memory/membership.json')
        got_sums = np.stack([masters[mid]['visual_sum'] for mid in saved['master_ids']])
        got_coords = np.stack([coords[mid] for mid in saved['leaf_ids']])
        sum_error = float(np.max(np.abs(got_sums - np.load(old / 'memory/master_sums.npy'))))
        coords_error = float(np.max(np.abs(got_coords - np.load(old / 'memory/leaf_coords.npy'))))
        assert sum_error < 1e-8 and coords_error < 1e-8
        validation = dict(domain=domain, arrivals=len(arrivals), partition_prefixes_matched=len(arrivals),
                          target_child_split_events=split_events, candidates=len(candidates)-domain_candidate_start,
                          final_sum_max_abs_error=sum_error, final_coords_max_abs_error=coords_error)
        validations.append(validation)
        print(json.dumps(validation), flush=True)
    eligible = [c for c in candidates if c['unchanged_between']]
    eligible.sort(key=lambda c: (c['domain'], c['image_id'], c['split_step'], c['merge_step']))
    shortlist = eligible
    dump(OUT / 'all_candidates.json', candidates)
    dump(OUT / 'stable_candidates.json', eligible)
    dump(OUT / 'shortlist.json', shortlist)
    arrays = {}
    for item in shortlist:
        for stage, vector in candidate_vectors[item['candidate_id']].items():
            arrays[f"{item['candidate_id']}__{stage}"] = vector
    np.savez_compressed(OUT / 'shortlist_vectors.npz', **arrays)
    dump(OUT / 'validation.json', dict(passed=True, completed=True, cached_replay=True,
         full_latent_cache_used_for_screening=True, fresh_streaming_claim=False, old_full_vectors_used_by_update=False,
         domains=validations, all_candidates=len(candidates), stable_candidates=len(eligible), shortlist_count=len(shortlist),
         source_sha256={str(p):sha(p) for p in sources}))
    print(json.dumps(dict(completed=True, all_candidates=len(candidates), stable_candidates=len(eligible),
                          shortlist_count=len(shortlist), shortlist=shortlist[:5]), ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
