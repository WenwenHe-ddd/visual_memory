"""Prefix-only insertion objectives. No per-image vector cache; Gram only final evaluation."""
import sys,csv,json,time,copy
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
 n=sum(x['n'] for x in c);k=len(c)
 total=sum((x['sum'] for x in c),np.zeros(512));s=sum(np.dot(x['sum'],x['sum'])/x['n'] for x in c)
 b=max(0,s-np.dot(total,total)/n);w=max(0,n-s)
 metric=metric.split('_')[0]
 if metric=='proxy':
  if k<2:return 0.
  means=np.stack([x['sum']/x['n'] for x in c]);cross=means@means.T;np.fill_diagonal(cross,-np.inf)
  bs=1-cross.max(axis=1);v=0.
  for j,x in enumerate(c):
   if x['n']<2:continue
   a=x['n']/(x['n']-1)*(1-float(means[j]@means[j]));b=bs[j]
   v+=x['n']*(b-a)/max(a,b,1e-15)
  return v/n
 if metric=='ch':return b*(n-k)/(w*(k-1)) if 1<k<n and w>1e-13 else 0
 if metric=='cu':return b/(n*k)
def merge(c,a,b):
 return [x for i,x in enumerate(c) if i not in (a,b)]+[dict(n=c[a]['n']+c[b]['n'],sum=c[a]['sum']+c[b]['sum'],ids=c[a]['ids']+c[b]['ids'])]
states={m:[] for m in ['ch_append','ch_merge','cu_merge','proxy_append','proxy_merge']};logs={m:[] for m in states};enc_s=0.;times={m:0. for m in states}
for i,row in enumerate(rows):
 t=time.perf_counter();z=np.array(enc.encode(row['image_path']));enc_s+=time.perf_counter()-t
 for metric,c in states.items():
  t=time.perf_counter();aug=c+[dict(n=1,sum=z.copy(),ids=[i])];best=aug;v=score(best,metric)
  # Include old-old merge as well: all candidates use master sums, no stored old images.
  for a in range(len(aug)):
   for b in range(a+1,len(aug)):
    if metric.endswith('append') and b!=len(aug)-1:continue
    cand=merge(aug,a,b);q=score(cand,metric)
    if q>v+1e-12:best,v=cand,q
  states[metric]=best;times[metric]+=time.perf_counter()-t
  logs[metric].append(dict(prefix=i+1,k=len(best),score=v,members=[[rows[j]['image_path'].split('/')[-1] for j in x['ids']] for x in best]))
 del z
# Full scalar Gram enters only evaluation after all streams completed.
gram=np.eye(len(rows));ids={r['image_id']:i for i,r in enumerate(rows)}
for r in csv.DictReader(open(ROOT/'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv')):
 if r['task']=='Brand_Memory_Test':a,b=ids[r['earlier_id']],ids[r['later_id']];gram[a,b]=gram[b,a]=float(r['cosine'])
from sklearn.metrics import silhouette_score,adjusted_rand_score
truth=[Path(r['image_path']).stem.rsplit('_',1)[0] for r in rows]
result={}
for m,c in states.items():
 labels=np.zeros(len(rows),dtype=int)
 for k,x in enumerate(c):labels[x['ids']]=k
 tp=fp=fn=0
 for a in range(len(rows)):
  for b in range(a+1,len(rows)):
   p,t=labels[a]==labels[b],truth[a]==truth[b];tp+=p and t;fp+=p and not t;fn+=t and not p
 result[m]=dict(clusters=logs[m][-1]['members'],k=len(c),silhouette=silhouette_score(1-gram,labels,metric='precomputed'),ari_brandproxy=adjusted_rand_score(truth,labels),pair_precision=tp/(tp+fp) if tp+fp else 0,pair_recall=tp/(tp+fn),seconds=times[m],prefix_log=logs[m])
result['encoder_seconds']=enc_s
(OUT/'summary_objectives_v2.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='prefix_log'} if isinstance(v,dict) else v for k,v in result.items()},indent=2))
