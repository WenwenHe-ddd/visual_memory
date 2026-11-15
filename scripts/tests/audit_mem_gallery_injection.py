"""Audit all persisted draws, exact tie calculations, and paired denominators."""
import collections
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'.vendor'))
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def csvread(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    out = Path(sys.argv[1]).resolve()
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['status'] == 'complete'
    for p, digest in protocol['input_hashes'].items():
        assert sha(ROOT/p) == digest, p
    artifacts = json.loads((out/'artifact_manifest.json').read_text())
    for p, digest in artifacts.items():
        assert sha(out/p) == digest, p
    nat = Path(protocol.get('natural_dir', ROOT/'results/mem_gallery_natural_observation/v1'))
    mem, queries, golds = [jl(nat/(name+'.jsonl')) for name in ('memories', 'queries', 'gold')]
    mi = {r['id']: i for i, r in enumerate(mem)}
    qi = {r['id']: i for i, r in enumerate(queries)}
    gold_by_q = {r['id']: r for r in golds}
    vec = np.load(nat/'embeddings.npz')
    m, q = vec['memory'].astype(float), vec['query'].astype(float)
    scores, similarities = q@m.T, m@m.T
    catalog = jl(out/'memory_catalog.jsonl')
    keys = {r['id']: r['image_sha256'] for r in catalog}
    pools = {r['question_id']: r for r in jl(out/'sampling_pools.jsonl')}
    plans = jl(out/'sampling_plans.jsonl')
    expected_samples = {g['id']+'|'+mid for g in golds if g['primary_cohort'] for mid in g['all_clue_ids']}
    assert len(expected_samples) == protocol['cohort']['targets']
    for qid, pool in pools.items():
        gold = gold_by_q[qid]
        assert gold['primary_cohort'] and set(pool['all_target_ids']) == set(gold['all_clue_ids'])
        forbidden = {keys[mid] for mid in gold['all_clue_ids']} | {pool['query_image_sha256']}
        assert forbidden == set(pool['excluded_image_sha256'])
        for condition in protocol['conditions']:
            representatives = {}
            for r in mem:
                if (r['task'] == pool['task']) != (condition == 'same_scenario') or keys[r['id']] in forbidden:
                    continue
                representatives.setdefault(keys[r['id']], r['id'])
            assert list(representatives.values()) == pool[condition]
            assert len(pool[condition]) >= max(protocol['N_levels'])
    plan_map = {}
    for plan in plans:
        key = (plan['sample_id'], plan['condition'], plan['repetition'])
        assert key not in plan_map and plan['sample_id'] in expected_samples
        seed = hashlib.sha256(f"42|{plan['sample_id']}|{plan['condition']}|{plan['repetition']}".encode()).hexdigest()[:16]
        assert seed == plan['seed_hex']
        pool = pools[plan['question_id']][plan['condition']]
        chosen = np.random.default_rng(int(seed, 16)).permutation([mi[mid] for mid in pool])[:max(protocol['N_levels'])]
        assert [mem[i]['id'] for i in chosen] == plan['ordered_distractor_ids']
        assert len({keys[mid] for mid in plan['ordered_distractor_ids']}) == max(protocol['N_levels'])
        plan_map[key] = plan
    expected_plans = len(expected_samples)*len(protocol['conditions'])*protocol['repetitions']
    assert len(plan_map) == expected_plans
    runs = csvread(out/'runs.csv')
    buckets = collections.defaultdict(list)
    seen = set()
    for row in runs:
        n, rep = int(row['N']), int(row['repetition'])
        key = (row['sample_id'], row['condition'], rep)
        assert (*key, n) not in seen
        seen.add((*key, n))
        plan = plan_map[key]
        assert row['target_id'] == plan['target_id'] and row['question_id'] == plan['question_id']
        inds = [mi[mid] for mid in plan['ordered_distractor_ids'][:n]]
        target, query = mi[row['target_id']], qi[row['question_id']]
        rounded = np.round(scores[query, inds], 6)
        score = round(float(scores[query, target]), 6)
        better, equal = int(sum(rounded>score)), int(sum(rounded==score))
        possible_ranks = list(range(1+better, 2+better+equal))
        expected_recall = sum(r<=5 for r in possible_ranks)/len(possible_ranks)
        assert abs(expected_recall-float(row['recall_at_5'])) < 1e-12
        assert abs(float(np.mean(possible_ranks))-float(row['target_expected_rank'])) < 1e-12
        assert int(row['memory_count']) == n+1
        if n:
            assert abs(float(np.mean(similarities[target, inds]))-float(row['target_distractor_cosine'])) < 1e-12
        buckets[(row['sample_id'], n, row['condition'])].append(row)
    assert len(runs) == expected_plans*len(protocol['N_levels'])
    sample_rows = csvread(out/'sample_means.csv')
    for row in sample_rows:
        bucket = buckets[(row['sample_id'], int(row['N']), row['condition'])]
        assert len(bucket) == protocol['repetitions']
        assert abs(np.mean([float(r['recall_at_5']) for r in bucket])-float(row['recall_at_5'])) < 1e-12
    for n in protocol['N_levels']:
        for condition in protocol['conditions']:
            assert {r['sample_id'] for r in sample_rows if int(r['N']) == n and r['condition'] == condition} == expected_samples
    summary = json.loads((out/'summary.json').read_text())
    for aggregate in summary['overall']:
        selected = [r for r in sample_rows if int(r['N']) == aggregate['N'] and r['condition'] == aggregate['condition']]
        assert abs(np.mean([float(r['recall_at_5']) for r in selected])-aggregate['recall_at_5']) < 1e-12
    # Validate finite-pool analytic expectations against exhaustive small-pool draws.
    path = ROOT/'scripts/analysis/mem_gallery_distractor_injection.py'
    spec = importlib.util.spec_from_file_location('injection', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cases = 0
    for size, better, equal in [(8, 0, 0), (8, 8, 0), (8, 2, 3), (8, 0, 8), (8, 5, 2)]:
        values = [1]*better+[0]*equal+[-1]*(size-better-equal)
        for n in (0, 3, 5, 8):
            probabilities = []
            for subset in itertools.combinations(values, n):
                b, e = sum(v>0 for v in subset), sum(v==0 for v in subset)
                probabilities.append(sum(r<=5 for r in range(b+1, b+e+2))/(e+1))
            assert abs(module.exact_recall(size, better, equal, n)-float(np.mean(probabilities))) < 1e-12
            cases += 1
    audit = {'status': 'passed', 'audited_pools': len(pools), 'audited_sampling_plans': len(plans),
             'audited_runs': len(runs), 'fixed_target_samples': len(expected_samples),
             'same_cohort_at_every_N_and_condition': True, 'random_seeds_reproduced': True,
             'all_gold_and_query_image_bytes_excluded': True, 'unique_image_hashes_within_every_draw': True,
             'rank_and_similarity_recomputed': True, 'analytic_exhaustive_test_cases': cases,
             'input_hashes_verified': len(protocol['input_hashes']), 'script_sha256': sha(Path(__file__))}
    (out/'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    artifacts['independent_audit.json'] = sha(out/'independent_audit.json')
    (out/'artifact_manifest.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
