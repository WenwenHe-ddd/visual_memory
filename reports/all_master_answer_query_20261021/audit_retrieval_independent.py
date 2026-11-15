from pathlib import Path
import json,hashlib,datetime
import numpy as np
ROOT=Path('/storage/hewenwen/VisualMemory')
OUT=ROOT/'reports/all_master_answer_query_20261021'
SRC=ROOT/'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(l)for l in Path(p).read_text().splitlines()if l.strip()]
def norm(a):
 a=np.asarray(a,dtype=np.float64);return a/np.linalg.norm(a,axis=-1,keepdims=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def same(a,b):assert np.allclose(a,b,rtol=0,atol=1e-12),np.max(np.abs(np.asarray(a)-np.asarray(b)))
results=lines(OUT/'retrieval_results.jsonl');R={r['question_id']:r for r in results}
assert len(R)==len(results)==371
protocol=read(OUT/'protocol.json');sources=protocol['source_sha256']
assert all(sha(p)==h for p,h in sources.items())
candidates={r['id']:r['candidate_statements']for r in lines(OUT/'candidates.jsonl')}
assert set(candidates)==set(R)
summary=read(OUT/'retrieval_summary.json')
checks=[];baseline_matches=0;total_leaves=0;total_masters=0;leafscores=0;masterchoices=0;imgq=0
baselinehits=0;expandedhits=0;goldtotal=0;unchanged=0;unchangedsets=0;overlap=0
maxcliperr=0.;maxbgeerr=0.;maxmeanerr=0.
for domain in sorted(SRC.iterdir()):
 if not(domain/'inputs.json').exists():continue
 inp=read(domain/'inputs.json');g=read(domain/'memory/graph.json')
 f=np.load(domain/'features.npz',allow_pickle=False);sums=np.load(domain/'memory/master_sums.npy',allow_pickle=False)
 a=np.load(OUT/'domains'/domain.name/'score_matrices.npz',allow_pickle=False)
 leaves=inp['leaves'];masters=g['masters'];questions=inp['questions']
 total_leaves+=len(leaves);total_masters+=len(masters)
 membership={node:mi for mi,m in enumerate(masters)for node in m['member_ids']}
 assert len(membership)==len(leaves)
 lm=np.array([membership[l['image_id']]for l in leaves])
 assert np.array_equal(lm,a['leaf_master_indices'])
 assert [m['id']for m in masters]==[m['id']for m in inp['masters']]
 for leaf,node in zip(leaves,g['image_nodes']):
  assert leaf['image_id']==node['id']
  assert leaf['master_id']==masters[membership[node['id']]]['id']
  assert leaf['retrieval_text']==node['dialogue']+'\nMemory description: '+node['caption']
 means=[]
 for m in masters:
  ref=m['visual_mean_ref'];assert ref['path']=='master_sums.npy' and abs(ref['scale']-1/m['n'])<1e-15
  assert m['n']==len(m['member_ids']);means.append(sums[ref['row']]/m['n'])
 means=norm(means)
 expectedclip=np.stack([means@v.T for v in a['clip_candidate_vectors']])
 same(expectedclip,a['clip_master_candidate']);maxcliperr=max(maxcliperr,float(np.max(np.abs(expectedclip-a['clip_master_candidate']))))
 choices=np.argmax(expectedclip,axis=-1)
 assert np.array_equal(choices,a['selected_choice_indices'])
 expectedbge=np.stack([np.vstack([f['bge_query'][qi],a['bge_augmented_query_vectors'][qi]])@f['bge_leaf'].T for qi in range(len(questions))])
 same(expectedbge,a['bge_all_query_leaf']);maxbgeerr=max(maxbgeerr,float(np.max(np.abs(expectedbge-a['bge_all_query_leaf']))))
 selected=np.stack([expectedbge[qi,1+choices[qi,lm],np.arange(len(leaves))]for qi in range(len(questions))])
 same(selected,a['selected_leaf_scores'])
 reference={r['id']:r['selected_memory_ids']for r in read(SRC.parent/'all_empirical'/domain.name/'runs.json')if r['L']==5 and r['method']=='full_scan'}
 bh=eh=gt=0
 for qi,q in enumerate(questions):
  r=R[q['id']];assert r['candidate_statements']==candidates[q['id']]
  assert r['question']==q['question'] and r['options']==q['options'] and r['baseline_retrieval_query']==q['retrieval_query']
  assert r['query_image_paths']==q['query_image_paths']
  if q['query_image_paths']:
   imgq+=1
   for caption in q['query_image_captions']:assert caption in r['baseline_retrieval_query']
  for mi,m in enumerate(r['masters']):
   assert m['master_id']==masters[mi]['id'] and m['member_ids']==masters[mi]['member_ids']
   same([m['clip_candidate_scores'][k]for k in 'ABCD'],expectedclip[qi,mi])
   ch='ABCD'[choices[qi,mi]];assert m['selected_choice']==ch
   assert m['selected_statement']==candidates[q['id']][ch]
   assert m['augmented_query']==q['retrieval_query']+'\nCandidate answer: '+candidates[q['id']][ch]
  gold=set(q['gold_ids']);assert r['evaluation']=={'gold_choice':q['gold_choice'],'gold_ids':q['gold_ids']}
  assert gold<={l['memory_id']for l in leaves};gt+=len(gold)
  rankings=[]
  for ci,scores in enumerate([expectedbge[qi,0],selected[qi]]):
   order=np.argsort(-scores,kind='stable');rank=np.empty(len(order),dtype=int);rank[order]=np.arange(1,len(order)+1)
   c=r['conditions'][ci];assert c['name']==['baseline_question_only','master_answer_query'][ci]
   idx=order[:5].tolist();ids=[leaves[i]['memory_id']for i in idx];rankings.append(ids)
   assert idx==c['top5_leaf_indices'] and ids==c['top5_memory_ids'];same(c['top5_scores'],scores[idx])
   texts=[leaves[i]['retrieval_text']for i in idx]
   assert [x['reader_text']for x in c['reader_memories']]==texts
   assert c['reader_context']=='\n\n'.join(f'Memory {j}: {t}'for j,t in enumerate(texts,1))
   hit=len(set(ids)&gold);assert c['gold_hits_top5']==hit
   assert c['gold_rank']==min(rank[i]for i,l in enumerate(leaves)if l['memory_id']in gold)
   for i,l in enumerate(r['leaves']):
    assert l['memory_id']==leaves[i]['memory_id'] and l['master_id']==leaves[i]['master_id']
    assert l['master_choice']=='ABCD'[choices[qi,lm[i]]]
    scorefield,rankfield=('baseline_score','baseline_rank')if ci==0 else ('expanded_score','expanded_rank')
    same(l[scorefield],scores[i]);assert l[rankfield]==rank[i]
   if ci==0:bh+=hit
   else:eh+=hit
  assert rankings[0]==reference[q['id']];baseline_matches+=1
  assert r['top5_same_order']==(rankings[0]==rankings[1])
  unchanged+=rankings[0]==rankings[1];unchangedsets+=set(rankings[0])==set(rankings[1]);overlap+=len(set(rankings[0])&set(rankings[1]))/5
 baselinehits+=bh;expandedhits+=eh;goldtotal+=gt;leafscores+=len(questions)*len(leaves);masterchoices+=len(questions)*len(masters)
 ds=summary['by_domain'][domain.name]
 assert ds['gold_references']==gt and ds['baseline_question_only']['gold_hits_top5']==bh and ds['master_answer_query']['gold_hits_top5']==eh
 checks.append({'domain':domain.name,'questions':len(questions),'leaves':len(leaves),'masters':len(masters),'all_scores_ranks_queries_evidence_match':True})
assert (total_leaves,total_masters,imgq,goldtotal)==(438,88,8,1041)
s=summary['overall'];assert s['questions']==371 and s['gold_references']==1041
assert s['baseline_question_only']['gold_hits_top5']==baselinehits and s['master_answer_query']['gold_hits_top5']==expandedhits
assert s['identical_ordered_top5']==unchanged and s['identical_top5_set']==unchangedsets
same(s['mean_top5_overlap'],overlap/371)
old=read(ROOT/'reports/home_q10_master_answer_query_20261021/retrieval_results.json');new=R['Home_Renovation_Interior_Design/Q10']
assert old['candidate_statements']==new['candidate_statements']
assert [c['top5_memory_ids']for c in old['conditions']]==[c['top5_memory_ids']for c in new['conditions']]
assert [m['selected_choice']for m in old['masters']]==[m['selected_choice']for m in new['masters']]
report={'auditor':'all_master_answer_protocol','status':'passed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'questions':371,'domains':checks,'leaf_count':total_leaves,'master_count':total_masters,'master_choices_recomputed':masterchoices,'leaf_scores_recomputed_per_condition':leafscores,'source_hashes_verified':len(sources),'source_files_unchanged':True,'actual_saved_sum_over_n_centers_verified':True,'max_clip_matrix_error':maxcliperr,'max_bge_matrix_error':maxbgeerr,'all_argmax_and_own_master_scores_verified':True,'all_stable_global_rankings_verified':True,'baseline_matches_frozen_full_scan':baseline_matches,'image_queries_preserving_original_caption':imgq,'reader_evidence_original_dialogue_and_full_caption':True,'q10_candidates_choices_top5_reproduced':True,'baseline_gold_hits':baselinehits,'expanded_gold_hits':expandedhits,'gold_reference_denominator':goldtotal,'candidate_sha256':sha(OUT/'candidates.jsonl'),'retrieval_results_sha256':sha(OUT/'retrieval_results.jsonl'),'audit_scope':'All frozen vectors and score/rank matrices independently recomputed; encoders not re-run. Reader generation audited separately. Candidate paraphrases frozen before scoring; no raw old image or per-leaf visual used.'}
(OUT/'validation_independent.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
