"""Transfer all five Qwen-eligible shared parameter pairs to frozen GPT-5 memory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
for key,value in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():
    os.environ.setdefault(key,value)
import numpy as np
from scripts.evaluation import memeye_gpt5_reader as api
from scripts.evaluation import memeye_gpt5_base_captions as captions
from scripts.memory.online_zscore import ZScoreWeighting
read,lines,sha,digest,dump=api.read,api.lines,api.sha,api.digest,api.dump
DEFAULT_OUTPUT=ROOT/'results/memeye_gpt5_qwen_five_validation_20261024'
SEARCH=ROOT/'results/memeye_qwen_zscore_joint_search_20261024'
FIRST=ROOT/'results/memeye_gpt5_qwen_selected_validation_20261024'
RAW=('baseline_question_only','master_answer_query')
# Fixed order from the already completed Qwen ranking, including every eligible pair.
PAIRS=((.01,1.),(.1,.6),(.03,1.),(.01,.6),(.003,.3))

def condition(method,ridge,alpha):
    return method+'__l'+format(ridge,'g').replace('.','p')+'_a'+format(alpha,'g').replace('.','p')

SPECS={m:{'method':m,'lambda':None,'alpha':0.} for m in RAW}
SPECS.update({condition(m,l,a):{'method':m,'lambda':l,'alpha':a} for l,a in PAIRS for m in RAW})
CONDITIONS=tuple(SPECS)
WEIGHTED=tuple(n for n in CONDITIONS if n not in RAW)
from scripts.analysis.retrieve_memeye_gpt5_base_lambda0p1 import verify_source


def eligible_pairs(search=SEARCH):
    search=Path(search).resolve();summary=read(search/'summary.json');proof=read(search/'independent_audit.json')
    assert proof['passed']
    hashes={str(search/n):h for n,h in proof['artifact_sha256'].items()}
    hashes[str(search/'independent_audit.json')]=sha(search/'independent_audit.json')
    selection=read(search/'selected_parameters.json');hashes.update(selection['source_sha256'])
    candidates=[r for r in summary['search_results'] if r['passes_all_three']]
    candidates.sort(key=lambda r:(-r['master_correct'],-r['baseline_correct'],r['alpha'],r['lambda']))
    assert tuple((r['lambda'],r['alpha']) for r in candidates)==PAIRS and len(summary['search_results'])==25
    b0,m0=(summary['overall'][n]['correct'] for n in RAW)
    for r in summary['search_results']:
        assert r['passes_all_three']==(r['baseline_correct']>b0 and r['master_correct']>m0 and r['master_correct']>r['baseline_correct'])
    first=read(FIRST/'independent_audit.json');assert first['passed'] and first['validation_passed'] is False
    hashes.update({str(FIRST/n):h for n,h in first['artifact_sha256'].items()})
    hashes[str(FIRST/'independent_audit.json')]=sha(FIRST/'independent_audit.json')
    for p,h in hashes.items():assert sha(p)==h,p
    return candidates,hashes


def checked_weighting(query, vectors, ridge, alpha):
    result = ZScoreWeighting(ridge, alpha).apply(query, vectors)
    # Independent dual ridge solve, without the production SVD implementation.
    centered = vectors - vectors.mean(axis=0)
    if len(vectors) == 1:
        check = vectors @ query
    else:
        adjusted = query - centered.T @ np.linalg.solve(
            centered @ centered.T + (len(vectors) - 1) * ridge * np.eye(len(vectors)),
            centered @ query)
        check = vectors @ adjusted
    np.testing.assert_allclose(result['white'], check, atol=2e-12, rtol=0)
    return result, float(np.max(np.abs(result['white'] - check)))


def retrieve(output, source, device, search=SEARCH):
    eligible, selection_hashes=eligible_pairs(search)
    output, source = Path(output).resolve(), Path(source).resolve()
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        p = read(marker)
        assert p['memory_path'] == str(source) and p['condition_specs']==SPECS
        for filename, expected in {**p['source_sha256'], **p['output_sha256']}.items():
            assert sha(filename) == expected, filename
        return p
    assert not (output / 'reader/protocol.json').exists(), 'Do not rebuild a signed API experiment.'
    proof = verify_source(source)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    source_inputs = read(source / 'inputs/protocol.json')
    memories, questions, model_name = original.attach_captions(
        lines(source / 'inputs/source_memories.jsonl'), lines(source / 'inputs/source_questions.jsonl'),
        lines(source / 'captions.jsonl'))
    assert model_name == proof['caption_model']
    prior_weighted=api.indexed(lines(source/'zscore/retrieval_results.jsonl'),'question_id')
    old = api.indexed(lines(source / 'retrieval_results.jsonl'), 'question_id')
    args = SimpleNamespace(device=device)
    original.configure_device(args)
    bge = original.load_text_encoder(source_inputs['text_encoder'], args.device)
    clip, tokenizer, limit = original.load_clip_text_encoder(source_inputs['image_encoder'], args.device)
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'source_validation.json', proof)
    results, encoding, domains = [], [], sorted({m['task'] for m in memories})
    group_checks = leaf_checks = rank_checks = 0
    max_raw_error = max_inverse_error = 0.
    master_counts = {}
    for domain in domains:
        ms = [m for m in memories if m['task'] == domain]
        qs = [q for q in questions if q['task'] == domain]
        graph = read(source / 'domains' / domain / 'memory/graph.json')
        sums = np.load(source / 'domains' / domain / 'memory/master_sums.npy', allow_pickle=False)
        master_counts[domain] = len(graph['masters'])
        e, einfo = encode_complete(bge, [m['retrieval_text'] for m in ms],
                                  ids=[m['id'] for m in ms], field='dialogue+caption')
        query_texts, query_ids, statements = [], [], []
        for q in qs:
            candidates = [q['candidate_statements'][k] for k in 'ABCD']
            statements.extend(candidates)
            query_texts.extend([q['retrieval_query']] + [q['retrieval_query'] + '\nCandidate answer: ' + s for s in candidates])
            query_ids.extend(q['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D'))
        qv, qinfo = encode_complete(bge, query_texts, source_inputs['query_prefix'], ids=query_ids, field='query')
        cv, cinfo = original._clip_candidates(clip, tokenizer, limit, statements)
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        for qi, question in enumerate(qs):
            raw = original.rank_question(question, ms, graph, sums, qv[qi], e, cv[qi])
            prior = old[question['id']]
            for key in ('question', 'options', 'query_images', 'query_image_captions', 'baseline_retrieval_query', 'candidate_statements'):
                assert raw[key] == prior[key], (question['id'], key)
            for a, b in zip(raw['masters'], prior['masters']):
                for key in ('master_id', 'member_ids', 'selected_choice', 'selected_statement', 'augmented_query'):
                    assert a[key] == b[key], (question['id'], key)
                np.testing.assert_allclose(list(a['clip_candidate_scores'].values()), list(b['clip_candidate_scores'].values()), atol=2e-6, rtol=0)
            for a, b in zip(raw['conditions'], prior['conditions']):
                assert a['full_rank_memory_ids'] == b['full_rank_memory_ids'], (question['id'], a['name'], 'raw ranks')
                assert a['reader_context'] == b['reader_context']
                rank_checks += 1
            for field in ('baseline_score', 'expanded_score'):
                err = float(np.max(np.abs(np.array([l[field] for l in raw['leaves']]) - [l[field] for l in prior['leaves']])))
                assert err < 2e-6, (question['id'], field, err)
                max_raw_error = max(max_raw_error, err)
            conditions = []
            for name in WEIGHTED:
                spec=SPECS[name];ridge,alpha=spec['lambda'],spec['alpha'];method=spec['method']
                base_name=method+'_zscore';raw_field='baseline_score' if method==RAW[0] else 'expanded_score'
                groups = ([{'master_id': 'all_memory', 'selected_choice': None, 'selected_statement': None,
                            'augmented_query': question['retrieval_query']}]
                          if method == RAW[0] else raw['masters'])
                leaves = [None] * len(ms)
                group_rows = []
                for g in groups:
                    indices = list(range(len(ms))) if method == RAW[0] else [i for i, l in enumerate(raw['leaves']) if l['master_id'] == g['master_id']]
                    query_index = 0 if method == RAW[0] else 1 + 'ABCD'.index(g['selected_choice'])
                    scores, error = checked_weighting(qv[qi, query_index], e[indices], ridge, alpha)
                    max_inverse_error = max(max_inverse_error, error)
                    np.testing.assert_allclose(scores['raw'], [raw['leaves'][i][raw_field] for i in indices], atol=2e-15, rtol=0)
                    group_rows.append({'group_id': g['master_id'], 'size': len(indices), 'query': g['augmented_query'],
                                       'selected_choice': g['selected_choice'], 'selected_statement': g['selected_statement'],
                                       **{k: scores[k] for k in ('raw_std', 'white_std', 'raw_status', 'z_status')}})
                    for j, i in enumerate(indices):
                        leaves[i] = {'memory_id': ms[i]['id'], 'leaf_index': i, 'group_id': g['master_id'],
                                     **{k: float(scores[k][j]) for k in ('raw', 'white', 'z', 'correction', 'weighted')}}
                    group_checks += 1
                    leaf_checks += len(indices)
                prior_condition=next(c for c in prior_weighted[question['id']]['conditions'] if c['name']==base_name)
                assert len(group_rows)==len(prior_condition['groups']) and len(leaves)==len(prior_condition['leaves'])
                for a,b in zip(group_rows,prior_condition['groups']):
                    for key in ('group_id','size','query','selected_choice','selected_statement','raw_std','raw_status'):assert a[key]==b[key],(question['id'],name,key)
                for a,b in zip(leaves,prior_condition['leaves']):
                    for key in ('memory_id','leaf_index','group_id','raw'):assert a[key]==b[key],(question['id'],name,key)
                if (ridge,alpha)==(.01,1.):
                    for a,b in zip(leaves,prior_condition['leaves']):
                        for key in ('raw','white','z','correction','weighted'):assert a[key]==b[key],(question['id'],name,key)
                order = sorted(range(len(ms)), key=lambda i: -leaves[i]['weighted'])
                conditions.append({'name': name, 'groups': group_rows, 'leaves': leaves,
                                   'top5_memory_ids': [ms[i]['id'] for i in order[:5]],
                                   'top5_scores': [leaves[i]['weighted'] for i in order[:5]],
                                   'full_rank_memory_ids': [ms[i]['id'] for i in order]})
            conditions.extend({k:c[k] for k in ('name','top5_memory_ids','top5_scores','full_rank_memory_ids')} for c in raw['conditions'])
            conditions.sort(key=lambda c:CONDITIONS.index(c['name']))
            results.append({'question_id': question['id'], 'task': domain, 'conditions': conditions})
        encoding.append({'domain': domain, 'leaf': einfo, 'query': qinfo, 'candidate': cinfo})
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'completed_questions': len(results),
                          'raw_rankings_verified': rank_checks, 'masters': len(graph['masters'])}), flush=True)
    assert len(results) == 371 and rank_checks == 742
    api.write_lines(output / 'retrieval_results.jsonl', results)
    sources = {**proof['source_sha256'], **selection_hashes}
    for p in (Path(__file__), ROOT / 'scripts/analysis/retrieve_memeye_gpt5_base_lambda0p1.py', Path(original.__file__), ROOT / 'scripts/memory/memeye_backbone_long_text.py',
              ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py'):
        sources[str(p)] = sha(p)
    for filename, expected in sources.items():
        assert sha(filename) == expected, filename
    p = {'status': 'complete', 'memory_path': str(source), 'caption_model': proof['caption_model'], 'device': args.device,
         'parameter_pairs': eligible, 'qwen_search_path': str(Path(search).resolve()), 'condition_specs': SPECS, 'conditions': list(CONDITIONS), 'questions': 371, 'memory_leaves': 438,
         'unweighted_full_rankings_equal': rank_checks, 'max_raw_score_difference': max_raw_error,
         'independent_inverse_ridge_groups': group_checks, 'group_leaf_scores_checked': leaf_checks,
         'max_inverse_ridge_difference': max_inverse_error, 'master_counts': master_counts, 'encoding': encoding,
         'all_raw_scores_and_groups_exactly_equal': True, 'lambda0p01_alpha1_scores_exactly_equal': True, 'caption_regenerated': False,
         'vectors_persisted': False, 'memory_modified': False, 'old_images_reencoded': False,
         'gold_loaded_for_ranking': False, 'source_sha256': sources,
         'output_sha256': {str(output / n): sha(output / n) for n in ('retrieval_results.jsonl', 'source_validation.json')}}
    dump(marker, p)
    return p



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--memory',type=Path,default=captions.DEFAULT_OUTPUT)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--qwen-search',type=Path,default=SEARCH)
    a=p.parse_args();result=retrieve(a.output,a.memory,a.device,a.qwen_search)
    print(json.dumps({k:result[k] for k in ('status','questions','unweighted_full_rankings_equal','max_raw_score_difference','independent_inverse_ridge_groups','max_inverse_ridge_difference')}),flush=True)

if __name__=='__main__':main()
