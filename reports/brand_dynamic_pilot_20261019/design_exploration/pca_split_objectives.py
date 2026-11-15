"""Prefix-only local PCA2 split proposals. Approximate, not final recommendation."""
import sys,csv,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/storage/hewenwen/VisualMemory');sys.path.insert(0,str(ROOT))
from scripts.graph.visual_image_encoder import ImageEncoder
OUT=ROOT/'reports/brand_dynamic_pilot_20261019/design_exploration'
rows=[r for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/log_diagnosis.csv')) if r['task']=='Brand_Memory_Test']
torch.set_num_threads(4)
enc=ImageEncoder(str(ROOT/'models/clip-vit-base-patch32'),device='cpu',local_files_only=True)
def score(c,metric):
 n=sum(len(x['ids']) for x in c);k=len(c);ns=np.array([len(x['ids']) for x in c]);mus=np.stack([x['mean'] for x in c]);ss=np.einsum('ij,ij->i',mus,mus)
 if metric=='ch':
  total=ns@mus;s=ns@ss;b=max(0,s-total@total/n);w=max(0,n-s)
  return b*(n-k)/(w*(k-1)) if 1<k<n and w>1e-13 else 0
 if k<2:return 0.
 cross=mus@mus.T;np.fill_diagonal(cross,-np.inf);bs=1-cross.max(axis=1);v=0.
 for j,x in enumerate(c):
  if ns[j]<2:continue
  a=ns[j]/(ns[j]-1)*(1-ss[j]);b=bs[j];v+=ns[j]*(b-a)/max(a,b,1e-15)
 return v/n
def fit(ids,x,mean=None):
 mean=x.mean(axis=0) if mean is None else mean;r=x-mean;u,s,vt=np.linalg.svd(r,full_matrices=False);rank=min(2,len(ids)-1)
 return dict(ids=ids,mean=mean,basis=vt[:rank].T,coords=u[:,:rank]*s[:rank])
def reconstruct(c):return c['mean']+c['coords']@c['basis'].T
def merge(a,b):
 ids=a['ids']+b['ids'];mean=(len(a['ids'])*a['mean']+len(b['ids'])*b['mean'])/len(ids)
 return fit(ids,np.concatenate([reconstruct(a),reconstruct(b)]),mean)
states={m:[] for m in ['proxy','ch']};logs={m:[] for m in states};enc_s=0.;times={m:0. for m in states};splitcount={m:0 for m in states}
for i,row in enumerate(rows):
 t=time.perf_counter();z=np.array(enc.encode(row['image_path']));enc_s+=time.perf_counter()-t
 for metric,c in states.items():
  t=time.perf_counter();single=fit([i],z[None,:]);aug=c+[single];best=aug;v=score(best,metric)
  for j in range(len(c)):
   cand=[x for k,x in enumerate(c) if k!=j]+[merge(c[j],single)];q=score(cand,metric)
   if q>v+1e-12:best,v=cand,q
  # Best single split along either retained axis, score uses approximate child means.
  baseline=best;winner=best
  for j,x in enumerate(baseline):
   n=len(x['ids'])
   if n<2:continue
   xrec=reconstruct(x)
   for axis in range(x['coords'].shape[1]):
    order=x['coords'][:,axis].argsort()
    for cut in range(1,n):
     ia,ib=order[:cut],order[cut:];a=fit([x['ids'][k] for k in ia],xrec[ia]);b=fit([x['ids'][k] for k in ib],xrec[ib]);cand=[p for k,p in enumerate(baseline) if k!=j]+[a,b];q=score(cand,metric)
     if q>v+1e-12:winner,v=cand,q
  if winner is not baseline:splitcount[metric]+=1
  states[metric]=winner;times[metric]+=time.perf_counter()-t
  logs[metric].append(dict(prefix=i+1,k=len(winner),score=v,members=[[rows[j]['image_path'].split('/')[-1] for j in x['ids']] for x in winner]))
 del z
# Full scalar Gram enters only evaluation after all streams completed.
gram=np.eye(len(rows));ids={r['image_id']:i for i,r in enumerate(rows)}
for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv')):
 if r['task']=='Brand_Memory_Test':a,b=ids[r['earlier_id']],ids[r['later_id']];gram[a,b]=gram[b,a]=float(r['cosine'])
from sklearn.metrics import silhouette_score,adjusted_rand_score
truth=[Path(r['image_path']).stem.rsplit('_',1)[0] for r in rows];result={}
for m,c in states.items():
 labels=np.zeros(len(rows),dtype=int)
 for k,x in enumerate(c):labels[x['ids']]=k
 tp=fp=fn=0
 for a in range(len(rows)):
  for b in range(a+1,len(rows)):
   p,t=labels[a]==labels[b],truth[a]==truth[b];tp+=p and t;fp+=p and not t;fn+=t and not p
 result[m]=dict(clusters=logs[m][-1]['members'],k=len(c),silhouette=silhouette_score(1-gram,labels,metric='precomputed'),ari_brandproxy=adjusted_rand_score(truth,labels),pair_precision=tp/(tp+fp) if tp+fp else 0,pair_recall=tp/(tp+fn),seconds=times[m],splits=splitcount[m],prefix_log=logs[m])
result['encoder_seconds']=enc_s
(OUT/'pca_split_objectives.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='prefix_log'} if isinstance(v,dict) else v for k,v in result.items()},indent=2))
