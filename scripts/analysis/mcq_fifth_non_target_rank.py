"""Find the original retrieval rank of the fifth non-target for each MCQ."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np

NAT = ROOT / 'results/natural_observation'
DIA = ROOT / 'results/dialogue_only_retrieval'
OUT = ROOT / 'results/mcq_fifth_non_target_rank'


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csvout(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stats(group, rows):
    valid = [r for r in rows if r['fifth_non_target_rank'] is not None]
    ranks = np.array([r['fifth_non_target_rank'] for r in valid])
    return {'group':group, 'questions':len(rows), 'valid_questions':len(valid),
            'unavailable_questions':len(rows)-len(valid),
            'mean_rank':float(ranks.mean()), 'median_rank':float(np.median(ranks)),
            'p90_rank':float(np.percentile(ranks,90)), 'p95_rank':float(np.percentile(ranks,95)),
            'min_rank':int(ranks.min()), 'max_rank':int(ranks.max()),
            'rank_5_count':int(np.sum(ranks==5)), 'rank_6_to_10_count':int(np.sum((ranks>=6)&(ranks<=10))),
            'rank_over_10_count':int(np.sum(ranks>10)), 'rank_at_most_10_fraction':float(np.mean(ranks<=10)),
            'mean_rank_under_random_tie_order':float(np.mean([r['fifth_rank_expected_in_tie'] for r in valid])),
            'rank_can_change_under_tie_order':sum(r['fifth_rank_min_in_tie']!=r['fifth_rank_max_in_tie'] for r in valid)}


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8); torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((NAT/'protocol.json').read_text())
    memories = read(NAT/'memories.jsonl'); open_queries = read(NAT/'queries.jsonl')
    old_gold = {r['id']:r for r in read(NAT/'gold.jsonl')}
    images = {r['id']:r for r in read(NAT/'images.jsonl')}
    queries=[]; source_files=[]; history_differences=[]
    for item in config['source_files']:
        open_path=ROOT/item['path']; path=open_path.with_name(open_path.name.replace('_Open.json','.json'))
        assert sha(open_path)==item['sha256']
        source_files.append({'path':str(path.relative_to(ROOT)), 'sha256':sha(path)})
        source=json.loads(path.read_text()); open_source=json.loads(open_path.read_text()); task=path.stem
        mcq_rounds={r['round']:r for s in source['multi_session_dialogues'] for r in s['dialogues']}
        open_rounds={r['round']:r for s in open_source['multi_session_dialogues'] for r in s['dialogues']}
        assert set(mcq_rounds)==set(open_rounds)
        for rid, turn in mcq_rounds.items():
            assert turn.get('input_image')==open_rounds[rid].get('input_image')
            if turn!=open_rounds[rid]:
                history_differences.append({'id':task+'/'+rid,
                    'different_fields':[k for k in set(turn)|set(open_rounds[rid]) if turn.get(k)!=open_rounds[rid].get(k)]})
        for q in source['human-annotated QAs']:
            qid=task+'/'+q['question_id']; gids=list(dict.fromkeys(task+'/'+rid for rid in q['clue']))
            assert gids==old_gold[qid]['clue_ids']
            queries.append({'id':qid, 'task':task, 'question':q['question'],
                            'gold_ids':gids, 'x':q['point'][0][0], 'y':q['point'][1][0],
                            'has_query_image':bool(q.get('question_image'))})
    assert len(queries)==371
    assert [q['id'] for q in queries]==[q['id'] for q in open_queries]
    cache=np.load(NAT/'embeddings.npz'); previous=np.load(DIA/'embeddings.npz')
    assert list(cache['memory_ids'])==[m['id'] for m in memories]
    assert list(previous['query_ids'])==[q['id'] for q in queries]
    caption_emb=cache['memory']; query_emb=previous['query'].copy()
    assert np.allclose(np.linalg.norm(caption_emb,axis=1),1,atol=1e-6)
    indexed_previous={r['id']:r['query_text'] for r in read(DIA/'queries.jsonl')}
    changed=[i for i,q in enumerate(queries) if q['question']!=indexed_previous[q['id']]]
    model=SentenceTransformer(config['retriever_model_path'],device='cpu',local_files_only=True)
    lengths=[len(model.tokenizer.encode(config['query_prefix']+q['question'])) for q in queries]
    assert max(lengths)<=model.max_seq_length
    if changed:
        print(f'Encoding {len(changed)} changed MCQ question texts; reusing {len(queries)-len(changed)} identical query vectors.',flush=True)
        query_emb[changed]=model.encode([config['query_prefix']+queries[i]['question'] for i in changed],
                                        batch_size=32,normalize_embeddings=True,show_progress_bar=True)
    assert np.allclose(np.linalg.norm(query_emb,axis=1),1,atol=1e-6)
    # Image content provenance must match the caption index, including MCQ files.
    for mid in memories:
        for image_id in mid['images']:
            assert sha(Path(images[image_id]['path']))==image_id
    source_paths=[NAT/name for name in ['protocol.json','memories.jsonl','embeddings.npz','gold.jsonl','images.jsonl']]
    source_paths += [DIA/name for name in ['embeddings.npz','queries.jsonl']]
    input_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    task_indices={t:[i for i,m in enumerate(memories) if m['task']==t] for t in sorted({q['task'] for q in queries})}
    rows=[];details=[]
    for qi,q in enumerate(queries):
        inds=task_indices[q['task']]; gold=set(q['gold_ids'])
        scores=caption_emb[inds]@query_emb[qi]; rounded=np.round(scores.astype(float),6)
        order=np.argsort(-rounded,kind='stable')
        local_others=[int(j) for j in order if memories[inds[j]]['id'] not in gold]
        selected=local_others[:5]
        rank_map={int(j):rank+1 for rank,j in enumerate(order)}
        positions=[rank_map[j] for j in selected]
        available=len(selected)==5
        row={'id':q['id'], 'task':q['task'], 'x':q['x'], 'y':q['y'], 'question':q['question'],
             'has_query_image':q['has_query_image'], 'memory_count':len(inds), 'gold_count':len(gold),
             'non_target_count':len(local_others), 'status':'available' if available else 'fewer_than_five_non_targets',
             'fifth_non_target_rank':max(positions) if available else None,
             'fifth_non_target_id':memories[inds[selected[-1]]]['id'] if available else '',
             'first_five_non_target_ids':'|'.join(memories[inds[j]]['id'] for j in selected),
             'first_five_non_target_ranks':'|'.join(str(p) for p in positions),
             'fifth_rank_min_in_tie':None, 'fifth_rank_max_in_tie':None, 'fifth_rank_expected_in_tie':None,
             'fifth_score_tie_group_size':None, 'fifth_score_tie_group_gold_count':None}
        if available:
            cutoff=rounded[selected[-1]]; tied=rounded==cutoff
            above=rounded>cutoff; tie_total=int(tied.sum())
            tie_others=sum(bool(tied[j]) for j in local_others)
            tie_gold=tie_total-tie_others
            above_others=sum(bool(above[j]) for j in local_others)
            needed=5-above_others; base=int(above.sum())
            row.update({'fifth_rank_min_in_tie':base+needed,
                        'fifth_rank_max_in_tie':base+needed+tie_gold,
                        'fifth_rank_expected_in_tie':base+needed*(tie_total+1)/(tie_others+1),
                        'fifth_score_tie_group_size':tie_total,
                        'fifth_score_tie_group_gold_count':tie_gold})
            assert base+needed<=row['fifth_non_target_rank']<=base+needed+tie_gold
            assert row['fifth_non_target_rank']==5+sum(memories[inds[j]]['id'] in gold for j in order[:max(positions)])
        detail={'id':q['id'],'question':q['question'],'gold_ids':q['gold_ids'],
                'first_five_non_targets':[{'id':memories[inds[j]]['id'],'original_rank':rank_map[j],
                                          'query_score':float(scores[j])} for j in selected],
                'fifth_rank':row['fifth_non_target_rank'],
                'ranking':[{'id':memories[inds[int(j)]]['id'],'rank':rank+1,'score':float(scores[j]),
                            'is_target':memories[inds[int(j)]]['id'] in gold} for rank,j in enumerate(order)]}
        rows.append(row);details.append(detail)
    overall=stats('all',rows)
    per_task=[stats(t,[r for r in rows if r['task']==t]) for t in task_indices]
    by_y=[stats(y,[r for r in rows if r['y']==y]) for y in ['Y1','Y2','Y3']]
    distribution=[{'fifth_non_target_rank':rank,'questions':sum(r['fifth_non_target_rank']==rank for r in rows),
                   'fraction_of_available_questions':sum(r['fifth_non_target_rank']==rank for r in rows)/overall['valid_questions']}
                  for rank in sorted({r['fifth_non_target_rank'] for r in rows if r['fifth_non_target_rank'] is not None})]
    summary={'overall':overall,'by_task':per_task,'by_y':by_y,'rank_distribution':distribution,
             'unavailable':[{'id':r['id'],'gold_count':r['gold_count'],'non_target_count':r['non_target_count']} for r in rows if r['fifth_non_target_rank'] is None],
             'audit':{'mcq_questions':len(queries),'memory_rounds':len(memories),'query_texts_reencoded':len(changed),
                      'query_vectors_reused':len(queries)-len(changed),'query_max_bge_tokens':max(lengths),
                      'truncated_queries':0,'mcq_clue_matches_prior_open_clue':True,
                      'all_mcq_image_paths_match_open_history':True,
                      'mcq_vs_open_nonimage_history_differences':history_differences}}
    protocol={'study':'Original rank of fifth non-target caption retrieval for original MemEye MCQs',
              'dataset':config['provenance']['dataset'],'source_files':source_files,
              'query':'raw MCQ question text plus fixed BGE prefix; no options, answer, clue, or query-image caption',
              'query_prefix':config['query_prefix'],'retriever':config['retriever'],'retriever_revision':config['retriever_revision'],
              'indexed_memory':'same 438 original image-bearing rounds; reuse ordinary Qwen caption vectors',
              'retrieval_scope':'all image-bearing rounds in the same original task, across all sessions',
              'target':'all original MCQ clue rounds, not lexical cues',
              'procedure':'sort all candidates by query-to-caption cosine, remove annotated targets from selection, take the first five non-targets, report maximum of their ORIGINAL ranks',
              'ties':'scores rounded to six decimals; stable source-memory ordering inside ties; retain minimum, maximum, and expected fifth rank under random tie permutations',
              'unavailable_policy':'retain all 371 questions; exclude only questions with fewer than five non-targets from rank summaries; more than five target rounds is allowed',
              'execution':'local CPU BGE only; no model answers, no QA success/failure split, no VLM generation, no data changes',
              'input_hashes':input_hashes}
    for p in source_paths: assert sha(p)==input_hashes[str(p.relative_to(ROOT))]
    for item in source_files: assert sha(ROOT/item['path'])==item['sha256']
    dump('protocol.json',protocol);dump('summary.json',summary)
    csvout('per_question.csv',rows);csvout('rank_distribution.csv',distribution)
    csvout('task_summary.csv',per_task);csvout('y_summary.csv',by_y)
    (OUT/'mcq_queries.jsonl').write_text(''.join(json.dumps({k:q[k] for k in ['id','task','question','has_query_image']},ensure_ascii=False)+'\n' for q in queries))
    (OUT/'retrieval_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in details))
    np.savez_compressed(OUT/'query_embeddings.npz',query=query_emb,query_ids=np.array([q['id'] for q in queries]))
    dump('artifact_manifest.json',{'source_files_unchanged':True,'outputs':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
