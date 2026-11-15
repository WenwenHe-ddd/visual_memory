"""Independent post-hoc artifact audit; never encodes images or changes pilot decisions."""
from __future__ import annotations
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path('/storage/hewenwen/VisualMemory')
BASE=ROOT/'reports/brand_dynamic_pilot_20261019'
OUT=BASE/'validation'
METHODS=('verified_local_scale_nn','local_scale_nn','uncut_nn')
SOURCE=ROOT/'results/visual_embedding_graph/Brand_Memory_Test/graph.json'
PAIR=ROOT/'reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv'
TOL=1e-10  # Audit numerical tolerance only; never an admission parameter.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def summary(values):
 values=np.asarray(values,dtype=float)
 return {'count':len(values),'max_abs':float(np.max(np.abs(values))) if len(values) else 0.,'mean_abs':float(np.mean(np.abs(values))) if len(values) else 0.,'rmse':float(np.sqrt(np.mean(values**2))) if len(values) else 0.}
def memberships_labels(groups,n):
 result=[None]*n
 for k,group in enumerate(groups):
  for i in group:result[int(i)]=k
 return result
def coassignment(a,b):
 total=len(a)*(len(a)-1)//2;bad=[]
 for i in range(len(a)):
  for j in range(i+1,len(a)):
   if (a[i]==a[j])!=(b[i]==b[j]):bad.append([i,j])
 return {'pair_count':total,'disagreement_count':len(bad),'agreement':(total-len(bad))/total if total else 1.,'disagreements':bad}
source=load(SOURCE);nodes=source['image_nodes'];ids={x['id']:i for i,x in enumerate(nodes)};names=[Path(x['image_path']).name for x in nodes];nameindex={name:i for i,name in enumerate(names)};n=len(nodes);d=np.zeros((n,n));seen=set()
for r in csv.DictReader(PAIR.open()):
 if r['task']=='Brand_Memory_Test':
  a,b=ids[r['earlier_id']],ids[r['later_id']];d[a,b]=d[b,a]=1-float(r['cosine']);seen.add(tuple(sorted((a,b))))
assert len(seen)==n*(n-1)//2
oracle=load(BASE/'design_exploration/local_scale_upperbound.json')['nn_local_scale_log_otsu']
oracle_labels=memberships_labels([[nameindex[x] for x in group] for group in oracle['clusters']],n)
oracle_prefix={x['n']:memberships_labels([[nameindex[v] for v in group] for group in x['members']],x['n']) for x in oracle['prefix_log']}
results={};all_errors=[]
for method in METHODS:
 folder=BASE/method;graph=load(folder/'graph.json');run=load(folder/'run_summary.json');evaluation=load(folder/'evaluation.json');calls=load(folder/'encoder_calls.json');logs=[json.loads(s) for s in (folder/'admission_log.jsonl').read_text().splitlines()];arrays=np.load(folder/'visual_state.npz',allow_pickle=False)
 errors=[];warnings=[]
 def check(condition,message):
  if not bool(condition):errors.append(message)
 check(len(logs)==n,'admission prefix count does not equal image count')
 check(len(graph['image_nodes'])==n,'final image count mismatch')
 check(graph['stream_indices']=={str(i):x['id'] for i,x in enumerate(nodes)},'source order mismatch')
 check(run['source_graph_sha256']==sha(SOURCE),'source graph hash mismatch')
 check(graph['vector_storage']['sha256']==sha(folder/graph['vector_storage']['path']),'vector file hash mismatch')
 code_hash_matches={name:(value==sha(BASE/name)) for name,value in run['code_sha256'].items()}
 for name,ok in code_hash_matches.items():check(ok,f'code hash changed: {name}')
 payload={field:0 for field in ('id','image_id','image_path','dialogue','caption','timestamp')};metadata_nonempty=Counter()
 for old,new in zip(nodes,graph['image_nodes']):
  for field in payload:
   if old.get(field)==new.get(field):payload[field]+=1
   else:errors.append(f'payload changed: {old["id"]} {field}')
  for field in ('caption','timestamp','dialogue'):
   metadata_nonempty[field]+=bool(new.get(field))
  check(sha(new['image_path'])==run['image_sha256'][new['id']],f'image content changed: {new["id"]}')
 prefix_results=[];all_edge_errors=[];exact_claim_errors=[];future_violations=[]
 for pos,log in enumerate(logs,1):
  p=log['prefix_count'];check(p==pos,f'prefix sequence mismatch: {p} versus {pos}');check(log['arriving_id']==p-1,f'arrival mismatch prefix {p}')
  members=[int(i) for group in log['memberships'] for i in group];check(sorted(members)==list(range(p)),f'prefix {p} membership coverage mismatch')
  if any(i<0 or i>=p for i in members):future_violations.append({'prefix':p,'type':'membership'})
  edge_errors=[];exact_errors=[];edge_future=[]
  for e in log['edges']:
   a,b,measured=int(e['a']),int(e['b']),int(e['measured_at_prefix'])
   if a>=p or b>=p or a<0 or b<0 or measured>p or measured<=max(a,b):edge_future.append(e)
   delta=float(e['distance'])-float(d[a,b]);edge_errors.append(delta)
   if e['exact_up_to_roundoff']:exact_errors.append(delta)
  future_violations.extend({'prefix':p,'type':'edge','edge':e} for e in edge_future)
  all_edge_errors.extend(edge_errors);exact_claim_errors.extend(exact_errors)
  labels=memberships_labels(log['memberships'],p)
  prefix_results.append({'prefix':p,'edge_distance_error':summary(edge_errors),'exact_flag_edge_error':summary(exact_errors),'oracle_local_scale_coassignment':coassignment(labels,oracle_prefix[p])})
 check(not future_violations,'future observation in membership/edge provenance')
 last_visible=0;enc_future=[];arrivals=[]
 for c in calls:
  vis=int(c['visible_count']);index=int(c['image_index'])
  if not (0<=index<vis<=n) or vis<last_visible:enc_future.append(c)
  last_visible=vis
  if c['purpose']=='arrival':
   arrivals.append(index)
   if index!=vis-1:enc_future.append(c)
 check(not enc_future,'encoder call accesses unseen image or time goes backwards')
 check(arrivals==list(range(n)),'arrival encoder order or coverage incorrect')
 purpose_counts=dict(Counter(c['purpose'] for c in calls));stream_purpose_counts={k:v for k,v in purpose_counts.items() if k!='posthoc_geometry_audit'};check(stream_purpose_counts==run['runtime']['encode_calls'],'streaming encoder purpose counts mismatch');check(all(c['visible_count']==n for c in calls if c['purpose']=='posthoc_geometry_audit'),'posthoc audit encoder occurred before stream completion')
 final_nn=[];misses=[];nn_errors=[];regrets=[]
 for si,e in graph['nearest_neighbors'].items():
  i=int(si);j=int(e['neighbor']);true_values=d[i].copy();true_values[i]=np.inf;best=float(true_values.min());regret=float(d[i,j]-best);delta=float(e['distance']-d[i,j]);nn_errors.append(delta);regrets.append(regret)
  tied=[int(x) for x in np.flatnonzero(np.abs(true_values-best)<=TOL)];hit=j in tied
  if not hit:misses.append({'index':i,'image':names[i],'selected':j,'selected_image':names[j],'true_nearest_indices':tied,'true_nearest_images':[names[x] for x in tied],'distance_regret':regret})
  final_nn.append({'index':i,'neighbor':j,'distance_error':delta,'true_nearest_hit':hit,'distance_regret':regret,'exact_up_to_roundoff':e['exact_up_to_roundoff']})
 check(len(final_nn)==n,'final nearest-neighbor coverage incomplete')
 check(summary(exact_claim_errors)['max_abs']<=TOL,'an edge marked exact has excessive error')
 # Array schema, dimensions, references and sufficient-statistic norm audits.
 categories=Counter();dtypes=Counter();array_records=[];allowed=[];sum_members=[];mean_norm_errors=[];basis_orth_errors=[];coord_center_errors=[];edge_members={e['image_id']:e for e in graph['edges']};reference_error=[]
 for m in graph['masters']:
  mid=m['id'];rank=m['rank'];members=m['member_indices'];count=m['member_count'];sum_members.extend(members);check(rank<=2,f'rank above two: {mid}');check(count==len(members),f'count mismatch: {mid}')
  expected={'mean':(512,),'basis':(512,rank),'coordinates':(count,rank),'singular_values':(rank,)}
  for suffix,shape in expected.items():
   key=f'{mid}_{suffix}';allowed.append(key);a=arrays[key];check(a.shape==shape,f'array shape mismatch: {key}');check(np.isfinite(a).all(),f'nonfinite values: {key}');categories[suffix]+=a.nbytes;dtypes[str(a.dtype)]+=a.nbytes;array_records.append({'key':key,'shape':list(a.shape),'dtype':str(a.dtype),'elements':int(a.size),'bytes':int(a.nbytes)})
  mu=arrays[f'{mid}_mean'];basis=arrays[f'{mid}_basis'];coords=arrays[f'{mid}_coordinates'];expected_mu_norm=float(np.mean(1-d[np.ix_(members,members)]));mean_norm_errors.append(float(mu@mu-expected_mu_norm));basis_orth_errors.append(float(np.max(np.abs(basis.T@basis-np.eye(rank)))) if rank else 0.);coord_center_errors.append(float(np.max(np.abs(coords.sum(axis=0)))) if rank else 0.)
  for row,index in enumerate(members):
   e=edge_members[nodes[index]['id']];ref=e['residual_coordinates_ref'];check(e['master_id']==mid and ref['key']==f'{mid}_coordinates' and ref['row']==row and ref['path']=='visual_state.npz',f'coordinate row reference mismatch: {mid}:{row}')
 check(sorted(allowed)==sorted(arrays.files),'unexpected arrays potentially include retained image embeddings')
 check(sorted(sum_members)==list(range(n)),'final master membership coverage mismatch')
 check(summary(mean_norm_errors)['max_abs']<=TOL,'master mean squared norm inconsistent with exact scalar evidence')
 check(summary(basis_orth_errors)['max_abs']<=TOL,'basis not orthonormal')
 check(summary(coord_center_errors)['max_abs']<=TOL,'coordinates not centered')
 numeric=sum(categories.values());nearest_bytes=len(final_nn)*np.dtype('float64').itemsize
 check(numeric==run['storage']['sketch_array_bytes'],'reported numeric sketch bytes mismatch');check(nearest_bytes==run['storage']['nearest_distance_scalar_bytes'],'reported NN scalar bytes mismatch')
 graphlabels=memberships_labels([m['member_indices'] for m in graph['masters']],n);finalco=coassignment(graphlabels,oracle_labels)
 check(graphlabels==memberships_labels([m['member_indices'] for m in graph['masters']],n),'labels internal consistency')
 check(not evaluation['errors'],'posthoc evaluation reports errors')
 check(evaluation['data']['pairwise_cosines_sha256']==sha(PAIR),'evaluation distance source changed')
 files={p.name:p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()};json_bytes=sum(v for k,v in files.items() if k.endswith('.json'));jsonl_bytes=sum(v for k,v in files.items() if k.endswith('.jsonl'))
 if misses:warnings.append('Exact stored edge distance does not imply exact global nearest-neighbor selection; compressed candidate filtering can omit better neighbors.')
 if numeric+nearest_bytes>n*512*4:warnings.append('Current float64 compressed numeric state is larger than float32 individual embeddings; compare matched dtype before claiming savings.')
 warnings.append('Artifact prefix audit validates logged accesses and indices; it is not a system-call proof of all process reads.')
 results[method]={'errors':errors,'warnings':warnings,'payload':{'fields_equal_to_source_counts':payload,'nonempty_counts':dict(metadata_nonempty),'image_hashes_verified':n},'provenance':{'code_hash_matches':code_hash_matches,'source_graph_hash_verified':run['source_graph_sha256']==sha(SOURCE),'vector_hash_verified':graph['vector_storage']['sha256']==sha(folder/'visual_state.npz')},'prefix':{'count':len(logs),'future_violations':future_violations,'encoder_future_violations':enc_future,'encoder_call_counts_including_posthoc':purpose_counts,'encoder_call_counts_streaming_only':stream_purpose_counts,'stored_edges_over_all_prefixes':summary(all_edge_errors),'edges_claimed_exact_over_all_prefixes':summary(exact_claim_errors),'per_prefix':prefix_results},'final_nearest_neighbors':{'count':len(final_nn),'distance_error':summary(nn_errors),'true_nearest_hits':len(final_nn)-len(misses),'true_nearest_recall':(len(final_nn)-len(misses))/len(final_nn),'distance_regret':summary(regrets),'misses':misses,'edges':final_nn},'final_oracle_local_scale_coassignment':finalco,'array_integrity':{'master_mean_squared_norm_error':summary(mean_norm_errors),'basis_orthonormality_error':summary(basis_orth_errors),'coordinate_centering_error':summary(coord_center_errors),'allowed_only_master_mean_basis_coordinates_singular_values':sorted(allowed)==sorted(arrays.files)},'storage':{'array_bytes_by_category':dict(categories),'array_bytes_by_dtype':dict(dtypes),'sketch_array_bytes':numeric,'nearest_distances_float64_equivalent_bytes':nearest_bytes,'numeric_geometry_bytes_including_nn_distances':numeric+nearest_bytes,'nearest_neighbor_indices_int64_equivalent_bytes':len(final_nn)*8,'note':'NN distances/indices are serialized as JSON; typed equivalents exclude Python dict/object overhead. IDs, scalar sketch metadata and array references are included in JSON file bytes, not counted twice as ndarray bytes.','full_per_image_float64_comparator_bytes':n*512*8,'full_per_image_float32_comparator_bytes':n*512*4,'same_dtype_float64_geometry_ratio':(numeric+nearest_bytes)/(n*512*8),'npz_archive_bytes':files['visual_state.npz'],'json_files_bytes':json_bytes,'jsonl_admission_history_bytes':jsonl_bytes,'all_files_bytes':sum(files.values()),'file_bytes':files,'array_records':array_records}}
 all_errors.extend(f'{method}: {e}' for e in errors)
# Independently map each alternative order's final labels back to original image IDs.
order_results={}
primary=load(BASE/'verified_local_scale_nn/graph.json')
primary_labels=memberships_labels([m['member_indices'] for m in primary['masters']],n)
for order_name in ('reverse','shuffle_20261019','shuffle_20261020'):
 folder=OUT/order_name;g=load(folder/'graph.json');order_labels=[None]*n;stream_ids=g['stream_indices']
 for k,m in enumerate(g['masters']):
  for local_index in m['member_indices']:order_labels[ids[stream_ids[str(local_index)]]]=k
 compare=coassignment(primary_labels,order_labels)
 ocalls=load(folder/'encoder_calls.json');ologs=[json.loads(x) for x in (folder/'admission_log.jsonl').read_text().splitlines()]
 future=[]
 for c in ocalls:
  if not 0<=c['image_index']<c['visible_count']<=n:future.append(c)
 for log in ologs:
  p=log['prefix_count']
  if sorted(i for group in log['memberships'] for i in group)!=list(range(p)):future.append({'membership_prefix':p})
  for edge in log['edges']:
   if not (0<=edge['a']<p and 0<=edge['b']<p and max(edge['a'],edge['b'])<edge['measured_at_prefix']<=p):future.append({'edge_prefix':p,'edge':edge})
 order_results[order_name]={'coassignment_to_original_order':compare,'future_violations':future,'actual_input_ids':[stream_ids[str(i)] for i in range(n)]}
 if compare['disagreement_count'] or future:all_errors.append(f'order validation mismatch: {order_name}')
shortlogs=[json.loads(x) for x in (OUT/'prefix_12/admission_log.jsonl').read_text().splitlines()]
fulllogs=[json.loads(x) for x in (BASE/'verified_local_scale_nn/admission_log.jsonl').read_text().splitlines()]
short_mismatches=[]
for short,full in zip(shortlogs,fulllogs):
 for field in ('prefix_count','arriving_id','memberships','edges'):
  if short[field]!=full[field]:short_mismatches.append({'prefix':short['prefix_count'],'field':field})
if len(shortlogs)!=12 or short_mismatches:all_errors.append('truncated prefix replay mismatch')
prefix_replay={'independently_compared_prefixes':len(shortlogs),'fields':['prefix_count','arriving_id','memberships','edges'],'mismatches':short_mismatches,'scope':'Reads actual saved truncated-run logs and compares each full-run prefix; no future image data fed to this audit.'}
report={'schema_version':1,'purpose':'Independent read-only posthoc artifact validation, without model calls or changes to admissions','numerical_audit_tolerance':TOL,'source_graph_sha256':sha(SOURCE),'pairwise_cosines_sha256':sha(PAIR),'pairwise_count':len(seen),'oracle_source_sha256':sha(BASE/'design_exploration/local_scale_upperbound.json'),'methods':results,'independent_order_coassignment_audit':order_results,'independent_truncated_prefix_replay_audit':prefix_replay,'errors':all_errors,'limitations':['The 30-image audit is not independent human clustering ground truth.','Final agreement with oracle does not establish exact nearest-neighbor search or prefix-wide equality.','Master means are checked against scalar norm identities here; full-vector directional correctness requires the separate reconstruction audit.']}
(OUT/'artifact_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({'errors':all_errors,'methods':{m:{'NN_recall':v['final_nearest_neighbors']['true_nearest_recall'],'edge_error':v['final_nearest_neighbors']['distance_error'],'oracle_pair_agreement':v['final_oracle_local_scale_coassignment']['agreement'],'geometry_bytes':v['storage']['numeric_geometry_bytes_including_nn_distances'],'payload':v['payload']} for m,v in results.items()}},indent=2))
if all_errors:raise SystemExit(1)
