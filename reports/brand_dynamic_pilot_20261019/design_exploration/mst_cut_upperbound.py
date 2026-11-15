"""Oracle pair-distance proposal only; NOT a compliant streaming-memory implementation."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import silhouette_score,adjusted_rand_score
ROOT=Path('/storage/hewenwen/VisualMemory');OUT=ROOT/'reports/brand_dynamic_pilot_20261019/design_exploration'
rows=[r for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/log_diagnosis.csv')) if r['task']=='Brand_Memory_Test'];ids={r['image_id']:i for i,r in enumerate(rows)};names=[Path(r['image_path']).name for r in rows];g=np.eye(len(rows))
for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv')):
 if r['task']=='Brand_Memory_Test':a,b=ids[r['earlier_id']],ids[r['later_id']];g[a,b]=g[b,a]=float(r['cosine'])
cases=json.load(open(ROOT/'reports/brand_dynamic_pilot_20261019/evaluation/evaluation_findings.json'))['visual_cases']
result={}
for method in ['otsu','largest_gap']:
 logs=[]
 for n in range(1,31):
  # Only visible prefix used; existing distances are oracle here, not deployable stored state.
  edges=minimum_spanning_tree(1-g[:n,:n]).tocoo();edges=sorted(zip(edges.data,edges.row,edges.col));w=np.array([float(e[0]) for e in edges]);threshold=None
  if len(w)>=2 and w[-1]>w[0]:
   candidates=[]
   for split in range(1,len(w)):
    if w[split]==w[split-1]:continue
    s=split*(len(w)-split)*(w[:split].mean()-w[split:].mean())**2 if method=='otsu' else w[split]-w[split-1]
    candidates.append((s,split))
   _,split=max(candidates,key=lambda x:x[0]);threshold=float((w[split-1]+w[split])/2)
  labels=list(range(n))
  def root(x):
   while labels[x]!=x:x=labels[x]
   return x
  # N<3 provisional singletons. Constant weights explicitly uncut once N>=3.
  for weight,a,b in edges:
   if n>=3 and (threshold is None or weight<=threshold):labels[root(a)]=root(b)
  lab=[root(i) for i in range(n)];k=len(set(lab));clusters=[[names[i] for i in range(n) if lab[i]==j] for j in sorted(set(lab))]
  logs.append(dict(n=n,k=k,cut_distance=threshold,cut_cosine=None if threshold is None else 1-threshold,members=clusters))
 pairs=[]
 for c in cases:
  a,b=ids[c['left_id']],ids[c['right_id']];pairs.append(dict(left=c['left_image'],right=c['right_image'],relation=c['relation_evaluation_only'],same=bool(lab[a]==lab[b])))
 result[method]=dict(k=k,clusters=clusters,cut_distance=threshold,silhouette=silhouette_score(1-g,lab,metric='precomputed'),close_hits=sum(x['same'] for x in pairs if x['relation']=='close'),close_total=sum(x['relation']=='close' for x in pairs),different_separated=sum(not x['same'] for x in pairs if x['relation'] in ['different','different_same_brand']),different_total=sum(x['relation'] in ['different','different_same_brand'] for x in pairs),case_results=pairs,prefix_log=logs,limitation='Oracle-prefix Gram used for proposal evaluation; no claim this script satisfies retained-state constraints')
(OUT/'mst_cut_upperbound.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:{k:v for k,v in r.items() if k not in ['prefix_log','case_results']} for m,r in result.items()},indent=2))
