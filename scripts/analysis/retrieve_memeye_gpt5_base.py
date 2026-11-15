"""Rebuild fresh GPT-5 masters and evaluate raw plus online z-score retrieval."""
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
NAMES=('baseline_question_only_zscore','master_answer_query_zscore')


def verify_source(source):
    root=captions.verify(source)
    captions.verify_captions(source)
    ret=read(source/'retrieval_protocol.json')
    assert ret['source_run_id']==root['run_id'] and ret['source_experiment_signature']==root['experiment_signature']
    hashes={**ret['source_sha256'],**ret['output_sha256']}
    hashes.update({str(source/name):sha(source/name) for name in ('protocol.json','retrieval_protocol.json','captions.jsonl','caption_generation/fresh_responses.jsonl')})
    for name,value in hashes.items():assert sha(name)==value,name
    return {'passed':True,'caption_model':root['contract']['model'],'source_sha256':hashes,'source_run_id':root['run_id']}

def checked_weighting(query, vectors):
    result = ZScoreWeighting(.01, 1.).apply(query, vectors)
    # Independent dual ridge solve, without the production SVD implementation.
    centered = vectors - vectors.mean(axis=0)
    if len(vectors) == 1:
        check = vectors @ query
    else:
        adjusted = query - centered.T @ np.linalg.solve(
            centered @ centered.T + (len(vectors) - 1) * .01 * np.eye(len(vectors)),
            centered @ query)
        check = vectors @ adjusted
    np.testing.assert_allclose(result['white'], check, atol=2e-12, rtol=0)
    return result, float(np.max(np.abs(result['white'] - check)))


def retrieve(output, source, device):
    output, source = Path(output).resolve(), Path(source).resolve()
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        p = read(marker)
        assert p['memory_path'] == str(source)
        for filename, expected in {**p['source_sha256'], **p['output_sha256']}.items():
            assert sha(filename) == expected, filename
        return p
    assert not (output / 'protocol.json').exists(), 'Do not rebuild a signed API experiment.'
    proof = verify_source(source)
    from scripts.memory import memeye_backbone_retrieval as original
    from scripts.memory.memeye_backbone_long_text import encode_complete
    source_inputs = read(source / 'inputs/protocol.json')
    memories, questions, model_name = original.attach_captions(
        lines(source / 'inputs/source_memories.jsonl'), lines(source / 'inputs/source_questions.jsonl'),
        lines(source / 'captions.jsonl'))
    assert model_name == proof['caption_model']
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
            for name, raw_field in zip(NAMES, ('baseline_score', 'expanded_score')):
                groups = ([{'master_id': 'all_memory', 'selected_choice': None, 'selected_statement': None,
                            'augmented_query': question['retrieval_query']}]
                          if name == NAMES[0] else raw['masters'])
                leaves = [None] * len(ms)
                group_rows = []
                for g in groups:
                    indices = list(range(len(ms))) if name == NAMES[0] else [i for i, l in enumerate(raw['leaves']) if l['master_id'] == g['master_id']]
                    query_index = 0 if name == NAMES[0] else 1 + 'ABCD'.index(g['selected_choice'])
                    scores, error = checked_weighting(qv[qi, query_index], e[indices])
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
                order = sorted(range(len(ms)), key=lambda i: -leaves[i]['weighted'])
                conditions.append({'name': name, 'groups': group_rows, 'leaves': leaves,
                                   'top5_memory_ids': [ms[i]['id'] for i in order[:5]],
                                   'top5_scores': [leaves[i]['weighted'] for i in order[:5]],
                                   'full_rank_memory_ids': [ms[i]['id'] for i in order]})
            results.append({'question_id': question['id'], 'task': domain, 'conditions': conditions})
        encoding.append({'domain': domain, 'leaf': einfo, 'query': qinfo, 'candidate': cinfo})
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'completed_questions': len(results),
                          'raw_rankings_verified': rank_checks, 'masters': len(graph['masters'])}), flush=True)
    assert len(results) == 371 and rank_checks == 742
    api.write_lines(output / 'retrieval_results.jsonl', results)
    sources = dict(proof['source_sha256'])
    for p in (Path(__file__), Path(original.__file__), ROOT / 'scripts/memory/memeye_backbone_long_text.py',
              ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py'):
        sources[str(p)] = sha(p)
    for filename, expected in sources.items():
        assert sha(filename) == expected, filename
    p = {'status': 'complete', 'memory_path': str(source), 'caption_model': proof['caption_model'], 'device': args.device,
         'lambda': .01, 'alpha': 1., 'questions': 371, 'memory_leaves': 438,
         'unweighted_full_rankings_equal': rank_checks, 'max_raw_score_difference': max_raw_error,
         'independent_inverse_ridge_groups': group_checks, 'group_leaf_scores_checked': leaf_checks,
         'max_inverse_ridge_difference': max_inverse_error, 'master_counts': master_counts, 'encoding': encoding,
         'vectors_persisted': False, 'memory_modified': False, 'old_images_reencoded': False,
         'gold_loaded_for_ranking': False, 'source_sha256': sources,
         'output_sha256': {str(output / n): sha(output / n) for n in ('retrieval_results.jsonl', 'source_validation.json')}}
    dump(marker, p)
    return p



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=captions.DEFAULT_OUTPUT)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();output=a.output.resolve()
    captions.verify_captions(output)
    if not (output/'retrieval_protocol.json').exists():
        from scripts.memory.memeye_backbone_retrieval import rebuild_and_retrieve
        try:
            rebuild_and_retrieve(output,device=a.device)
        except ValueError as error:
            if ' exceeds BGE token limit ' not in str(error):raise
            captions.freeze(output/'long_text_transition.json', {
                'reason':str(error),
                'action':'Original BGE 512-token encoding limit with truncation audit; preserve complete captions and reader evidence.',
                'caption_generation_changed':False,'memory_construction_changed':False,'reader_evidence_changed':False})
            from scripts.memory.memeye_backbone_long_text import complete_retrieval
            complete_retrieval(output,device=a.device)
    proof=verify_source(output)
    result=retrieve(output/'zscore',output,a.device)
    print(json.dumps({'stage':'all_retrieval_complete','model':proof['caption_model'],
        'unweighted_rankings_verified':result['unweighted_full_rankings_equal'],'master_counts':result['master_counts']}),flush=True)


if __name__=='__main__':main()
