#!/usr/bin/env python3
"""CPU cache screening of real pure/mixed split followed by later merge.

The evaluation cache supplies only each CURRENT arrival vector to the update.
It is also used outside the update for illustrative example selection. This is
not a fresh streaming replay or an unbiased aggregate quality evaluation.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys
import json
import hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.graph.caption_ward_visual import update_visual_partition

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
SWEEP = ROOT / 'reports/other_domains_vae_vamo_20261105'
BRAND = ROOT / 'reports/brand_vae_vamo_20261105'
PREVIOUS = ROOT / 'reports/vamo_24_triptychs_screen_20261114/all_candidates.json'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rebuild(master, z, basis):
    mean = master['visual_sum'] / master['n']
    return mean + basis @ (z - basis.T @ mean)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    assert not (OUT / 'all_candidates.json').exists(), 'Refusing to overwrite a completed screen.'
    old_candidates = read(PREVIOUS)
    old_keys = {(r['domain'], r['image_id'], r['split_step'], r['merge_step']): r['candidate_id']
                for r in old_candidates}
    domains = [(p.name, p, Path(read(p / 'completed.json')['source_audit']))
               for p in sorted(SWEEP.iterdir()) if (p / 'completed.json').exists()]
    domains += [('Brand_Memory_Test', BRAND, SOURCE / 'Brand_Memory_Test/insertion_audit.jsonl')]
    protocol = {
        'purpose': 'Outcome-selected illustrative candidates requested by user, not unbiased quality evaluation.',
        'representation': 'Existing 4096D raw VAE latent adaptation; unchanged domain fixed PCA-2D basis.',
        'screen_input': 'Existing evaluation_only/full_latents.npy; CPU cached replay, NOT fresh encoding.',
        'updates': 'Prior masters, prior 2D coordinates, frozen caption partition, CURRENT arrival vector only.',
        'split': 'Target old owner is in audit.split_master_ids and its target-destination overlap count < old_n. Target destination may contain other source masters and/or the current new arrival; flags identify this mixed redistribution.',
        'merge': 'At least two old source masters contribute to target destination; split_step < merge_step. Partial transfers and a new arrival are allowed and flagged.',
        'timing': 'Export actual after-split, immediately-before-merge and after-merge states. Intervening changes are explicitly recorded.',
        'state_storage': 'One vector per target+step; candidate records reference array keys in state_vectors.npz. Reference vectors are separate evaluation-only entries.',
        'reuse': 'previous_candidate_id maps identical event pairs from earlier pure-split screening.',
        'domains': [name for name, _, _ in domains],
    }
    dump(OUT / 'selection_protocol.json', protocol)
    candidates, states, state_metadata, validations = [], {}, {}, []
    sources = [Path(__file__), ROOT / 'scripts/graph/caption_ward_visual.py', PREVIOUS]

    for di, (domain, old, partition_path) in enumerate(domains):
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
        sources.extend([manifest_path, partition_path, basis_path, latent_path, old / 'replay_audit.jsonl',
                        old / 'memory/membership.json', old / 'memory/master_sums.npy', old / 'memory/leaf_coords.npy'])
        lookup = {r['image_id']: r for r in arrivals}
        indices = {r['image_id']: i for i, r in enumerate(arrivals)}

        def save_state(target, step, vector, master_id=None, reference=False):
            suffix = 'reference' if reference else f't{step:04d}'
            state_id = f'd{di:02d}_i{indices[target]:04d}_{suffix}'
            if state_id in states:
                assert np.array_equal(states[state_id], vector), state_id
            else:
                states[state_id] = vector.copy()
                state_metadata[state_id] = dict(domain=domain, image_id=target, filename=lookup[target]['filename'],
                    target_index=indices[target], step=None if reference else step,
                    master_id=master_id, kind='evaluation_reference' if reference else 'historical_reconstruction')
            return state_id

        masters, coords, next_index, active_splits = {}, {}, 1, {}
        domain_start = len(candidates)
        split_target_events = mixed_split_target_events = 0
        previous_vectors = {}
        max_conservation_error = 0.0
        max_projection_error = 0.0
        for step, (row, part) in enumerate(zip(arrivals, frozen), 1):
            assert row['image_id'] == part['image_id']
            previous = masters
            owner = {target: mid for mid, master in previous.items() for target in master['member_ids']}
            current_arrival = np.array(cache[step - 1], dtype=np.float64).reshape(-1)
            masters, audit, next_index = update_visual_partition(
                previous, coords, part['clusters_partition'], {row['image_id']: current_arrival}, basis, next_index)
            coords[row['image_id']] = basis.T @ current_arrival
            assert audit['clusters'] == prior_audits[step - 1]['clusters']
            max_conservation_error = max(max_conservation_error, audit['global_conservation_max_abs_error'])
            max_projection_error = max(max_projection_error, audit['max_projection_consistency_l2_error'])
            overlaps = {}
            for overlap in audit['source_overlaps']:
                overlaps.setdefault(overlap['new_master_id'], []).append(overlap)
            current_vectors = {}
            for mid, master in masters.items():
                ov = overlaps.get(mid, [])
                by_source = {o['old_master_id']: o for o in ov}
                contains_new = row['image_id'] in master['member_ids']
                is_merge = len(ov) >= 2
                for target in master['member_ids']:
                    vector = rebuild(master, coords[target], basis)
                    current_vectors[target] = vector
                    if is_merge and target in active_splits:
                        prior_owner_id = owner[target]
                        prior_owner = previous[prior_owner_id]
                        prior_vector = previous_vectors[target]
                        np.testing.assert_array_equal(prior_vector, rebuild(prior_owner, coords[target], basis))
                        reference = np.array(cache[indices[target]], dtype=np.float64).reshape(-1)
                        after_merge_state = save_state(target, step, vector, mid)
                        before_merge_state = save_state(target, step - 1, prior_vector, prior_owner_id)
                        reference_state = save_state(target, None, reference, reference=True)
                        for split in active_splits[target]:
                            assert split['step'] < step
                            a = states[split['state_id']]
                            c0, c1 = cosine(a, reference), cosine(vector, reference)
                            cb = cosine(prior_vector, reference)
                            e0 = float(np.linalg.norm(a - reference) / np.linalg.norm(reference))
                            e1 = float(np.linalg.norm(vector - reference) / np.linalg.norm(reference))
                            stable = split['unchanged'] and bool(np.max(np.abs(a - prior_vector)) < 1e-9)
                            cid = f'candidate_{len(candidates)+1:05d}'
                            previous_id = old_keys.get((domain, target, split['step'], step))
                            candidates.append(dict(
                                candidate_id=cid, domain=domain, image_id=target, filename=lookup[target]['filename'],
                                source_path=lookup[target]['source_path'], target_index=indices[target],
                                split_step=split['step'], merge_step=step, split_n=split['n'], merge_n=master['n'],
                                before_split_n=split['old_n'], before_merge_n=prior_owner['n'],
                                split_master=split['master_id'], before_merge_master=prior_owner_id, merge_master=mid,
                                cos_split=c0, cos_before_merge=cb, cos_merge=c1, gain=c1-c0, immediate_merge_gain=c1-cb,
                                relative_l2_split=e0, relative_l2_merge=e1, relative_l2_gain=e0-e1,
                                unchanged_between=stable, split_members=split['member_ids'], merge_members=list(master['member_ids']),
                                before_merge_members=list(prior_owner['member_ids']),
                                split_member_filenames=[lookup[k]['filename'] for k in split['member_ids']],
                                merge_member_filenames=[lookup[k]['filename'] for k in master['member_ids']],
                                split_is_mixed=split['mixed'], split_has_new=split['has_new'],
                                split_has_new_image=split['has_new'], split_new_ids=split['new_ids'],
                                split_sources=split['sources'], target_old_owner=split['old_owner'],
                                split_target_source_count=split['source_count'],
                                merge_source_sizes=[o['old_n'] for o in ov], merge_source_counts=[o['count'] for o in ov],
                                merge_source_master_ids=[o['old_master_id'] for o in ov],
                                merge_has_new_image=contains_new, merge_new_ids=[k for k in audit['new_image_ids'] if k in master['member_ids']],
                                merge_has_partial_source=any(o['count'] < o['old_n'] for o in ov),
                                split_to_before_merge_max_abs=float(np.max(np.abs(a - prior_vector))),
                                old_experiment=str(old), partition_audit=str(partition_path), previous_candidate_id=previous_id,
                                newly_enumerated=previous_id is None, after_split_state=split['state_id'],
                                before_merge_state=before_merge_state, after_merge_state=after_merge_state, reference_state=reference_state,
                            ))
                    old_owner = owner.get(target)
                    target_overlap = by_source.get(old_owner)
                    is_target_split = (old_owner in audit['split_master_ids'] and target_overlap is not None
                                       and target_overlap['count'] < target_overlap['old_n'])
                    if is_target_split:
                        split_target_events += 1
                        mixed = len(ov) > 1 or contains_new
                        mixed_split_target_events += int(mixed)
                        state_id = save_state(target, step, vector, mid)
                        active_splits.setdefault(target, []).append(dict(
                            step=step, state_id=state_id, unchanged=True, n=master['n'], old_n=target_overlap['old_n'],
                            master_id=mid, member_ids=list(master['member_ids']), mixed=mixed, has_new=contains_new,
                            new_ids=[k for k in audit['new_image_ids'] if k in master['member_ids']], sources=ov,
                            old_owner=old_owner, source_count=target_overlap['count']))
            for target, splits in active_splits.items():
                for split in splits:
                    if split['unchanged'] and np.max(np.abs(current_vectors[target] - states[split['state_id']])) >= 1e-9:
                        split['unchanged'] = False
            previous_vectors = current_vectors
            del current_arrival
        saved = read(old / 'memory/membership.json')
        got_sums = np.stack([masters[mid]['visual_sum'] for mid in saved['master_ids']])
        got_coords = np.stack([coords[mid] for mid in saved['leaf_ids']])
        sum_error = float(np.max(np.abs(got_sums - np.load(old / 'memory/master_sums.npy'))))
        coords_error = float(np.max(np.abs(got_coords - np.load(old / 'memory/leaf_coords.npy'))))
        assert sum_error < 1e-8 and coords_error < 1e-8
        dc = candidates[domain_start:]
        validation = dict(domain=domain, arrivals=len(arrivals), partition_prefixes_matched=len(arrivals),
                          target_split_events=split_target_events, mixed_target_split_events=mixed_split_target_events,
                          candidates=len(dc), new_candidates=sum(c['newly_enumerated'] for c in dc),
                          unique_targets=len({c['image_id'] for c in dc}),
                          final_sum_max_abs_error=sum_error, final_coords_max_abs_error=coords_error,
                          max_conservation_error=max_conservation_error, max_projection_error=max_projection_error)
        validations.append(validation)
        print(json.dumps(validation), flush=True)
    used_states = {c[k] for c in candidates for k in ('after_split_state', 'before_merge_state', 'after_merge_state', 'reference_state')}
    # Drop split snapshots that were never followed by a qualifying later merge.
    states = {k: v for k, v in states.items() if k in used_states}
    state_metadata = {k: v for k, v in state_metadata.items() if k in used_states}
    recovered_old_ids = {c['previous_candidate_id'] for c in candidates if c['previous_candidate_id']}
    assert recovered_old_ids == {c['candidate_id'] for c in old_candidates}, 'Must preserve all old pure-split pairs.'
    new_candidates = [c for c in candidates if c['newly_enumerated']]
    eligible = [c for c in candidates if c['unchanged_between']]
    dump(OUT / 'all_candidates.json', candidates)
    dump(OUT / 'new_candidates.json', new_candidates)
    dump(OUT / 'stable_candidates.json', eligible)
    dump(OUT / 'state_metadata.json', state_metadata)
    np.savez_compressed(OUT / 'state_vectors.npz', **states)
    validation = dict(passed=True, completed=True, cached_replay=True,
        full_latent_cache_used_for_screening=True, fresh_streaming_claim=False, old_full_vectors_used_by_update=False,
        domains=validations, all_candidates=len(candidates), previously_enumerated_candidates=len(recovered_old_ids),
        newly_enumerated_candidates=len(new_candidates), stable_candidates=len(eligible),
        unique_targets=len({(c['domain'], c['image_id']) for c in candidates}),
        unique_new_targets=len({(c['domain'], c['image_id']) for c in new_candidates}),
        newly_covered_targets=len({(c['domain'], c['image_id']) for c in candidates} - {(c['domain'], c['image_id']) for c in old_candidates}),
        unique_states=len(states), reconstruction_states=sum(m['kind'] == 'historical_reconstruction' for m in state_metadata.values()),
        vector_bytes=sum(v.nbytes for v in states.values()), source_sha256={str(p): sha(p) for p in sources})
    dump(OUT / 'validation.json', validation)
    print(json.dumps({k: v for k, v in validation.items() if k not in ('source_sha256', 'domains')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
