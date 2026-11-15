"""Oracle prefix-distance nearest-neighbor components; proposal only."""
exec(open('reports/brand_dynamic_pilot_20261019/design_exploration/mst_cut_upperbound.py').read().split("result={}\nfor method")[0])
result={};logs=[]
for n in range(1,31):
 labels=list(range(n));work=g[:n,:n].copy();np.fill_diagonal(work,-np.inf)
 def root(a):
  while labels[a]!=a:a=labels[a]
  return a
 if n>1:
  nearest=work.argmax(1)
  for a,b in enumerate(nearest):labels[root(a)]=root(int(b))
 lab=[root(i) for i in range(n)];k=len(set(lab));clusters=[[names[i] for i in range(n) if lab[i]==j] for j in sorted(set(lab))];logs.append(dict(n=n,k=k,members=clusters))
pairs=[]
for c in cases:
 a,b=ids[c['left_id']],ids[c['right_id']];pairs.append(dict(left=c['left_image'],right=c['right_image'],relation=c['relation_evaluation_only'],same=bool(lab[a]==lab[b])))
result=dict(k=k,clusters=clusters,silhouette=silhouette_score(1-g,lab,metric='precomputed'),close_hits=sum(x['same'] for x in pairs if x['relation']=='close'),close_total=sum(x['relation']=='close' for x in pairs),different_separated=sum(not x['same'] for x in pairs if x['relation'] in ['different','different_same_brand']),different_total=sum(x['relation'] in ['different','different_same_brand'] for x in pairs),case_results=pairs,prefix_log=logs,limitation='Oracle-prefix Gram used for proposal evaluation; no claim this script satisfies retained-state constraints')
(OUT/'nearest_neighbor_upperbound.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['prefix_log']},indent=2))
