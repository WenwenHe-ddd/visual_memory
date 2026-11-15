"""Final exploratory comparison. Oracle prefix distances; no threshold sweep or label fitting."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import silhouette_score
ROOT=Path('/storage/hewenwen/VisualMemory');OUT=ROOT/'reports/brand_dynamic_pilot_20261019/design_exploration'
rows=[r for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/log_diagnosis.csv')) if r['task']=='Brand_Memory_Test'];ids={r['image_id']:i for i,r in enumerate(rows)};names=[Path(r['image_path']).name for r in rows];g=np.eye(len(rows))
for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv')):
 if r['task']=='Brand_Memory_Test':a,b=ids[r['earlier_id']],ids[r['later_id']];g[a,b]=g[b,a]=float(r['cosine'])
cases=json.load(open(ROOT/'reports/brand_dynamic_pilot_20261019/evaluation/evaluation_findings.json'))['visual_cases']
def otsu(vals):
 w=np.sort(np.asarray(vals,dtype=float));finite=w[np.isfinite(w)]
 if len(finite)<2 or finite[-1]==finite[0]:return None
 best=None
 for cut in range(1,len(finite)):
  if finite[cut]==finite[cut-1]:continue
  variance=cut*(len(finite)-cut)*(finite[:cut].mean()-finite[cut:].mean())**2
  if best is None or variance>best[0]:best=(float(variance),float((finite[cut-1]+finite[cut])/2))
 return best[1] if best else None
result={}
for method in ['mst_local_scale_log_otsu','nn_local_scale_log_otsu']:
 logs=[]
 for n in range(1,len(rows)+1):
  dist=1-g[:n,:n];work=dist.copy();np.fill_diagonal(work,np.inf);nearest=work.argmin(1);radii=work[np.arange(n),nearest]
  if method.startswith('mst'):
   tree=minimum_spanning_tree(dist).tocoo();edges=[(int(a),int(b),float(w)) for a,b,w in zip(tree.row,tree.col,tree.data)]
  else:
   pairs=set(tuple(sorted((int(a),int(b)))) for a,b in enumerate(nearest) if a!=b);edges=[(a,b,float(dist[a,b])) for a,b in sorted(pairs)]
  transformed=[]
  for a,b,d in edges:
   denom=min(radii[a],radii[b])
   # Zero-distance observations must remain contracted; nonzero edges from such a point have infinite ratio.
   value=0. if d==0 else np.inf if denom==0 else float(np.log(d/denom))
   transformed.append(value)
  cut=otsu(transformed);labels=list(range(n));kept=[];cut_edges=[]
  def root(a):
   while labels[a]!=a:a=labels[a]
   return a
  for (a,b,d),value in zip(edges,transformed):
   keep=n>=3 and np.isfinite(value) and (cut is None or value<=cut)
   (kept if keep else cut_edges).append(dict(a=names[a],b=names[b],cosine=1-d,log_local_ratio=value))
   if keep:labels[root(a)]=root(b)
  lab=[root(i) for i in range(n)];k=len(set(lab));clusters=[[names[i] for i in range(n) if lab[i]==j] for j in sorted(set(lab))]
  logs.append(dict(n=n,k=k,cut_log_ratio=cut,cut_ratio=None if cut is None else float(np.exp(cut)),members=clusters,kept=kept,cut=cut_edges))
 pairs=[]
 for c in cases:
  a,b=ids[c['left_id']],ids[c['right_id']];pairs.append(dict(left=c['left_image'],right=c['right_image'],relation=c['relation_evaluation_only'],same=bool(lab[a]==lab[b])))
 result[method]=dict(k=k,clusters=clusters,cut_log_ratio=cut,cut_ratio=None if cut is None else float(np.exp(cut)),silhouette=silhouette_score(1-g,lab,metric='precomputed') if 1<k<len(rows) else None,close_hits=sum(x['same'] for x in pairs if x['relation']=='close'),close_total=sum(x['relation']=='close' for x in pairs),different_separated=sum(not x['same'] for x in pairs if x['relation'] in ['different','different_same_brand']),different_total=sum(x['relation'] in ['different','different_same_brand'] for x in pairs),case_results=pairs,prefix_log=logs,limitation='Oracle-prefix Gram used for proposal evaluation; not compliant retained-state streaming implementation')
(OUT/'local_scale_upperbound.json').write_text(json.dumps(result,indent=2));print(json.dumps({m:{k:v for k,v in r.items() if k not in ['prefix_log','case_results']} for m,r in result.items()},indent=2))
