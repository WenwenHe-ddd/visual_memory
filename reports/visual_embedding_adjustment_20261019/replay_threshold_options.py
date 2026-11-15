"""Counterfactual scalar-Gram replay; no image embeddings are stored or loaded.

These are geometry diagnostics, not labeled cluster-quality measurements.
Original production code and graph files are only read.
"""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path('/storage/hewenwen/VisualMemory')
OUT = Path(__file__).parent
PAIR_FILE = ROOT / 'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv'
SOURCE = ROOT / 'results/visual_embedding_graph'
TASKS = {}
for path in sorted(SOURCE.glob('*/graph.json')):
    graph = json.loads(path.read_text())
    nodes = graph['image_nodes']
    index = {node['id']: i for i, node in enumerate(nodes)}
    gram = [[float(i == j) for j in range(len(nodes))] for i in range(len(nodes))]
    members = {edge['image_id']: edge['master_id'] for edge in graph['edges']}
    TASKS[path.parent.name] = dict(nodes=nodes, index=index, gram=gram,
        original=[members[node['id']] for node in nodes],
        hashes=[hashlib.sha256(Path(node['image_path']).read_bytes()).hexdigest() for node in nodes])
with PAIR_FILE.open() as stream:
    for row in csv.DictReader(stream):
        task = TASKS[row['task']]
        i, j = task['index'][row['earlier_id']], task['index'][row['later_id']]
        task['gram'][i][j] = task['gram'][j][i] = float(row['cosine'])


def replay(task, rule, floor, margin=0.0):
    gram = task['gram']
    clusters, assignments, decisions = [], [], []
    for i in range(len(gram)):
        options = []
        for k, members in enumerate(clusters):
            n = len(members)
            r2 = max(0.0, sum(gram[a][b] for a in members for b in members) / (n*n))
            r = math.sqrt(r2)
            mean_pair = sum(gram[i][j] for j in members)/n
            center_cosine = min(1.0, max(-1.0, mean_pair/r)) if r else -1.0
            if rule == 'fixed':
                threshold = floor
            elif rule == 'center_margin':
                threshold = floor if n == 1 else max(floor, r-margin)
            elif rule == 'pairwise_mean':
                q = (n*r2-1)/(n-1) if n > 1 else None
                threshold = floor if n == 1 else max(floor, (q-margin)/r)
            else:
                raise ValueError(rule)
            options.append(dict(master=k, center_cosine=center_cosine,
                                threshold=threshold, eligible=center_cosine+1e-8>=threshold))
        # Compare only the closest center; never fall back to a farther eligible cluster.
        closest = max(options, key=lambda row: row['center_cosine']) if options else None
        if closest is not None and closest['eligible']:
            chosen = closest['master']
            clusters[chosen].append(i)
        else:
            chosen = len(clusters)
            clusters.append([i])
        assignments.append(chosen)
        decisions.append(dict(id=task['nodes'][i]['id'], master=f'M{chosen+1}',
                              closest_before=closest))
    return assignments, decisions


def metrics(task, assignment):
    counts = Counter(assignment)
    within, high_total, high_split = [], 0, 0
    for i in range(len(assignment)):
        for j in range(i):
            score = task['gram'][i][j]
            if assignment[i] == assignment[j]:
                within.append(score)
            if score >= .99:
                high_total += 1
                high_split += assignment[i] != assignment[j]
    grouped = {}
    for digest, master in zip(task['hashes'], assignment):
        grouped.setdefault(digest, []).append(master)
    return dict(masters=len(counts), singles=sum(n==1 for n in counts.values()),
                largest=max(counts.values()), within_pairs=len(within),
                within_pairs_lt_08=sum(s<.8 for s in within),
                min_within_pair=min(within,default=None),
                pairs_ge_099=high_total, split_pairs_ge_099=high_split,
                duplicate_hash_groups_split=sum(len(set(v))>1 for v in grouped.values() if len(v)>1))


def evaluate(name, rule=None, floor=None, margin=0.0):
    stats, examples = {}, {}
    for task_name, task in TASKS.items():
        if rule is None:
            assignment, decisions = task['original'], []
        else:
            assignment, decisions = replay(task, rule, floor, margin)
        stats[task_name] = metrics(task, assignment)
        if task_name == 'Home_Renovation_Interior_Design':
            examples['home_first_three'] = decisions[:3] if decisions else assignment[:3]
            examples['home_R2_R4_same'] = assignment[0] == assignment[2]
            examples['home_R2_R3_same'] = assignment[0] == assignment[1]
        if task_name == 'Personal_Health_Dashboard_Assistant':
            chosen = [i for i, node in enumerate(task['nodes'])
                      if node['id'].split('/')[-1] in ('HEALTH_S1:R2#image1','HEALTH_S1:R3#image1','HEALTH_S1:R4#image1','HEALTH_S1:R6#image1')]
            examples['health'] = [dict(id=task['nodes'][i]['id'], master=assignment[i]) for i in chosen]
    totals = {key:sum(row[key] for row in stats.values()) for key in (
        'masters','singles','within_pairs','within_pairs_lt_08','pairs_ge_099',
        'split_pairs_ge_099','duplicate_hash_groups_split')}
    totals['largest_cluster'] = max(row['largest'] for row in stats.values())
    report = dict(name=name, rule=rule, center_cosine_floor=floor, margin=margin,
                  totals=totals, examples=examples, tasks=stats)
    return report


reports = [evaluate('original_actual_graph')]
for floor in (.8,.85,.9):
    reports.append(evaluate(f'fixed_{floor}', 'fixed', floor))
    for margin in (0.,.03,.05):
        reports.append(evaluate(f'center_floor_{floor}_margin_{margin}', 'center_margin', floor, margin))
    for margin in (0.,.03):
        reports.append(evaluate(f'pairwise_floor_{floor}_margin_{margin}', 'pairwise_mean', floor, margin))
payload = dict(image_occurrences=sum(len(t['nodes']) for t in TASKS.values()),
    source_pair_file=str(PAIR_FILE), source_pair_sha256=hashlib.sha256(PAIR_FILE.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    caveats=['Counterfactual replay from scalar pairwise cosine matrix; no per-image embedding persistence.',
             'Original baseline comes from actual graph membership, not reconstructed floating-point decisions.',
             'All variants select closest center only and use >= with 1e-8 tolerance.',
             'All variants initialize every new singleton with a fixed floor, replacing peer-dependent initialization.',
             'Pairwise rule uses equivalent center threshold max(floor,(q-margin)/r); floor remains a center-cosine floor.',
             'Split high-cosine pairs and low-cosine within-cluster pairs are diagnostics, not semantic accuracy.',
             'Parameters are illustrative sensitivity checks, not calibrated or held-out optimized.'],
    results=reports)
(OUT/'threshold_replay.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
for report in reports:
    print(report['name'], json.dumps(report['totals']),
          'home_same_R2_R4=',report['examples']['home_R2_R4_same'],
          'home_same_R2_R3=',report['examples']['home_R2_R3_same'])
