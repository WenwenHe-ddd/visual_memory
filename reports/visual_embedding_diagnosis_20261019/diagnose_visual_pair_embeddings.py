import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.graph.visual_embedding_graph import VisualEmbeddingGraph
from scripts.graph.visual_embedding_storage import load_graph

OUT = ROOT / 'reports/visual_embedding_diagnosis_20261019'
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = ROOT / 'results/visual_embedding_graph'
model = ImageEncoder(str(ROOT / 'models/clip-vit-base-patch32'),
                     revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
                     device='cpu', local_files_only=True)
all_pairs = []
all_prior = []
stats = {}
for graph_path in sorted(GRAPH.glob('*/graph.json')):
    task = graph_path.parent.name
    original = load_graph(graph_path).to_dict()
    nodes = original['image_nodes']
    membership = {e['image_id']: e['master_id'] for e in original['edges']}
    logs = [json.loads(s) for s in (graph_path.parent/'write_log.jsonl').read_text().splitlines()]
    vectors = []  # Temporary task-local audit state; never persisted.
    replay = VisualEmbeddingGraph()
    max_candidate_error = 0.0
    max_threshold_error = 0.0
    decisions_match = True
    for node, old in zip(nodes, logs):
        vector = model.encode(node['image_path'])
        vectors.append(vector)
        current = replay.insert(node['id'], vector, dialogue=node['dialogue'],
                                caption=node['caption'], timestamp=node['timestamp'],
                                image_path=node['image_path'])
        decisions_match &= current['master_id'] == old['master_id'] and current['new_master'] == old['new_master']
        for a, b in zip(current['candidates'], old['candidates']):
            max_candidate_error = max(max_candidate_error, abs(a['similarity']-b['similarity']))
        max_threshold_error = max(max_threshold_error, abs(current['threshold_after']-old['threshold_after']))
    matrix = np.asarray(vectors, dtype=np.float64)
    sims = np.clip(matrix @ matrix.T, -1., 1.)
    within, across = [], []
    for i in range(len(nodes)):
        for j in range(i):
            same = membership[nodes[i]['id']] == membership[nodes[j]['id']]
            score = float(sims[i,j])
            (within if same else across).append(score)
            all_pairs.append(dict(task=task, earlier_id=nodes[j]['id'], later_id=nodes[i]['id'],
                                  earlier_image=nodes[j]['image_path'], later_image=nodes[i]['image_path'],
                                  earlier_master=membership[nodes[j]['id']], later_master=membership[nodes[i]['id']],
                                  same_cluster=same, cosine=score))
        if i:
            nearest = int(np.argmax(sims[i,:i]))
            target_master = membership[nodes[nearest]['id']]
            candidate = next(c for c in logs[i]['candidates'] if c['master_id'] == target_master)
            chosen = next((c for c in logs[i]['candidates'] if c['master_id'] == logs[i]['master_id']), None)
            all_prior.append(dict(task=task, image_id=nodes[i]['id'], image=nodes[i]['image_path'],
                nearest_prior_id=nodes[nearest]['id'], nearest_prior_image=nodes[nearest]['image_path'],
                nearest_prior_cosine=float(sims[i,nearest]), nearest_prior_master=target_master,
                assigned_master=logs[i]['master_id'], new_master=logs[i]['new_master'],
                nearest_prior_master_score=candidate['similarity'], nearest_prior_master_threshold=candidate['threshold'],
                nearest_prior_master_eligible=candidate['eligible'],
                assigned_score=chosen['similarity'] if chosen else None,
                assigned_threshold=chosen['threshold'] if chosen else None,
                same_cluster_as_nearest_prior=membership[nodes[i]['id']] == target_master))
    final_error=max((max(abs(a-b) for a,b in zip(m['visual_mean'],n['visual_mean']))
                     for m,n in zip(replay.masters, original['masters'])),default=0.)
    def distribution(values):
        return dict(count=len(values), min=float(np.min(values)), median=float(np.median(values)),
                    max=float(np.max(values))) if values else dict(count=0)
    stats[task]=dict(images=len(nodes), masters=len(original['masters']),
                    all_assignment_decisions_match=bool(decisions_match),
                    max_candidate_similarity_error=max_candidate_error,
                    max_threshold_error=max_threshold_error, max_final_master_vector_error=final_error,
                    within_cluster_pair_cosines=distribution(within),
                    cross_cluster_pair_cosines=distribution(across))
    del matrix, vectors, sims, replay, vector
    print(task, json.dumps(stats[task]), flush=True)

def write_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write_csv('pairwise_cosines.csv', all_pairs)
write_csv('nearest_prior_audit.csv', all_prior)
report=dict(tasks=stats, pair_count=len(all_pairs), replayed_image_count=sum(s['images'] for s in stats.values()),
    same_cluster_pairs=sum(p['same_cluster'] for p in all_pairs),
    cross_cluster_pairs=sum(not p['same_cluster'] for p in all_pairs),
    nearest_prior_different_cluster=sum(not p['same_cluster_as_nearest_prior'] for p in all_prior),
    nearest_prior_compared_count=len(all_prior),
    high_similarity_split_pairs={str(t):sum(not p['same_cluster'] and p['cosine']>=t for p in all_pairs) for t in [.90,.95,.98,.99,.999999]},
    weakest_within_cluster_pairs=sorted([p for p in all_pairs if p['same_cluster']],key=lambda p:p['cosine'])[:20],
    strongest_cross_cluster_pairs=sorted([p for p in all_pairs if not p['same_cluster']],key=lambda p:-p['cosine'])[:30],
    nearest_prior_split_examples=sorted([p for p in all_prior if not p['same_cluster_as_nearest_prior']],key=lambda p:-p['nearest_prior_cosine'])[:30],
    image_embeddings_persisted=False,
    interpretation='Cosine and cluster consistency diagnostics; visual similarity requires inspection, no semantic ground-truth labels.',
    source_run=str(GRAPH),audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'pairwise_diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(OUT/'diagnose_visual_pair_embeddings.py').write_bytes(Path(__file__).read_bytes())
print('FINISHED',json.dumps({k:v for k,v in report.items() if k not in ('tasks','weakest_within_cluster_pairs','strongest_cross_cluster_pairs','nearest_prior_split_examples')}),flush=True)
