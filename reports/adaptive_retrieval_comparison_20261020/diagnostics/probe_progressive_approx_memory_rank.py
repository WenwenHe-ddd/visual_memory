import json,sys,time,types
from pathlib import Path
ROOT=Path('/storage/hewenwen/VisualMemory');sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
import numpy as np
from scripts.graph.progressive_text_bounds import ProgressiveTextBounds
p=ROOT/'reports/home_online_routing_dialogue_caption_20261020';inputs=json.loads((p/'inputs.json').read_text());old=json.loads((p/'results.json').read_text())['questions']
f=np.load(ROOT/'reports/adaptive_retrieval_comparison_20261020/home_features.npz');queries=f['bge_query'];vectors=f['bge_leaf']
li={r['image_id']:i for i,r in enumerate(inputs['leaves'])};mi={m['id']:i for i,m in enumerate(inputs['masters'])};groups=[[li[k] for k in m['member_ids']] for m in inputs['masters']];orders=[[mi[m['master_id']] for m in q['masters']] for q in old]
index=ProgressiveTextBounds.fit(vectors,groups);x=index.vectors;m=index.mean;b=index.basis;z=index.coordinates;n,d=x.shape;r=index.rank
q=queries/np.linalg.norm(queries,axis=1)[:,None];true=q@x.T;oracle=np.argsort(-true,axis=1,kind='stable')[:,:5]
neighbor_scores=x@x.T;np.fill_diagonal(neighbor_scores,-np.inf);neighbors=np.argsort(-neighbor_scores,axis=1,kind='stable')[:,:5]
mean_bias=(x-m)@m
policies=[]
s=index.fit_singular_values[:r];gaps=(s[:-1]-s[1:])/s[:-1];gap_rank=int(np.argmax(gaps))+1
variance=s*s;fraction=variance/variance.sum();expected=np.array([np.sum(1/np.arange(j+1,r+1))/r for j in range(r)])
pass_bs=fraction>expected;broken_rank=0
for ok in pass_bs:
 if not ok:break
 broken_rank+=1
for variant in ('raw_query_projection','centered_query_plus_exact_mean_bias'):
 memory_proj=x@b.T if variant=='raw_query_projection' else (x-m)@b.T
 query_proj=q@b.T if variant=='raw_query_projection' else (q-m)@b.T
 bias=np.zeros(n) if variant=='raw_query_projection' else mean_bias
 neighbor_first=None
 for k in range(1,r+1):
  approximation=memory_proj[:,:k]@z[:,:k].T+bias[None,:];np.fill_diagonal(approximation,-np.inf)
  ranks=np.argsort(-approximation,axis=1,kind='stable')[:,:5]
  if np.array_equal(ranks,neighbors):neighbor_first=k;break
 for rule,k in [('ordered_memory_top5_fidelity',neighbor_first),('largest_adjacent_relative_singular_gap',gap_rank),('broken_stick_leading_components',broken_rank)]:
  approx=query_proj[:,:k]@z[:,:k].T+bias[None,:]
  candidates=np.argsort(-approx,axis=1,kind='stable')[:,:5]
  predicted=np.array([sorted(ids,key=lambda i:(-true[j,i],i)) for j,ids in enumerate(candidates)])
  exact=np.all(predicted==oracle,axis=1);overlap=np.array([len(set(a)&set(c))/5 for a,c in zip(predicted,oracle)])
  mask=np.array([not row['id'].endswith('/Q10') for row in inputs['questions']])
  extra_mean=d if variant=='centered_query_plus_exact_mean_bias' else 0
  # q-m projection only uses subtraction; no q.mean is needed for ranking in this approximation.
  extra_mean=0
  products=d*k+n*k+5*d
  policies.append({'variant':variant,'rule':rule,'rank':k,'all_exact':int(exact.sum()),'all_n':len(q),'all_overlap':float(overlap.mean()),'new_exact':int(exact[mask].sum()),'new_n':int(mask.sum()),'new_overlap':float(overlap[mask].mean()),'coordinate_products':products,'full_products':n*d,'cost_ratio':products/(n*d),'per_question':[{'id':row['id'],'exact':bool(exact[j]),'overlap':float(overlap[j]),'selected':predicted[j].tolist(),'oracle':oracle[j].tolist()} for j,row in enumerate(inputs['questions'])]})
source=(ROOT/'scripts/graph/progressive_text_bounds.py').read_text()
replacements={
 'center = mean_score+partial[indices]':'center = mean_score+partial[indices]+self.mean_centered_bias[indices]',
 '1-s+self.orthogonality_error*s+margin':'self.centered_query_norm_sq-s+self.orthogonality_error*s+margin',
 'mean_score = float(self.mean @ q)':'mean_score = float(self.mean @ q)\n            self.centered_query_norm_sq = 1 + self.mean_squared_norm - 2*mean_score',
 'component = float(self.basis[components] @ q)':'component = float(self.basis[components] @ (q-self.mean))',
}
for before,after in replacements.items():
 assert source.count(before)==1,(before,source.count(before))
 source=source.replace(before,after)
mod=types.ModuleType('temporary_mean_centered_progressive');sys.modules[mod.__name__]=mod;exec(compile(source,'/tmp/temporary_mean_centered_progressive.py','exec'),mod.__dict__)
tight=mod.ProgressiveTextBounds.fit(vectors,groups);tight.mean_centered_bias=(tight.vectors-tight.mean)@tight.mean;tight.mean_squared_norm=float(tight.mean @ tight.mean)
runs=[]
for j,(query,order) in enumerate(zip(queries,orders)):
 t=time.perf_counter();result=tight.search(query,order,5,record_trace=False);result['seconds']=time.perf_counter()-t;result['exact']=result['selected_indices']==oracle[j].tolist();runs.append(result)
energy=1-np.sum((q@b.T)**2,axis=1)
summary={'memory_rank':r,'n':n,'dimension':d,'query_memory_span_outside_energy':{'mean':float(energy.mean()),'min':float(energy.min()),'max':float(energy.max())},'approximation_policies':policies,'temporary_centered_query_strict':{'exact_matches':sum(a['exact'] for a in runs),'mean_components':float(np.mean([a['projected_components'] for a in runs])),'mean_exact_leaves':float(np.mean([a['exact_score_count'] for a in runs])),'mean_coordinate_products':float(np.mean([a['cost']['total_scalar_products'] for a in runs])),'full_coordinate_products':n*d,'mean_ms':1000*float(np.mean([a['seconds'] for a in runs])),'runtime_note':'no-trace single pass, provisional diagnostic only','mean_bias_extra_bytes':mean_bias.nbytes}}
Path('/tmp/progressive_memory_rank_probe_results.json').write_text(json.dumps(summary,indent=2))
small={**summary,'approximation_policies':[{k:v for k,v in a.items() if k!='per_question'} for a in policies]};print(json.dumps(small,indent=2))
