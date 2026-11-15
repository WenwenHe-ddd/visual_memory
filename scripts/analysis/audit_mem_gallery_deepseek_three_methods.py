"""Audit shared DeepSeek hypotheses, fresh masters, paired rankings and actual answers."""
import argparse
from collections import Counter
import csv
import html
import math
from pathlib import Path

import numpy as np
from scripts.evaluation import mem_gallery_deepseek_three_methods as run
from scripts.analysis.audit_mem_gallery_gpt5_baseline import summary
from scripts.graph.caption_ward_visual import choose_caption_partition

read, rows, sha, obj, dump = run.read, run.rows, run.sha, run.obj, run.dump


def independent_weights(vectors, query, ridge, alpha):
    raw = vectors @ query
    if len(vectors) == 1:
        return raw, raw.copy()
    centered = vectors - vectors.mean(axis=0)
    adjusted = query - centered.T @ np.linalg.solve(
        centered @ centered.T + (len(vectors)-1)*ridge*np.eye(len(vectors)), centered @ query)
    white = vectors @ adjusted
    residual = white-white.mean()
    std = float(np.sqrt(np.mean(residual**2)))
    tol = 32*np.finfo(float).eps*max(1., float(np.abs(white).max()))
    raw_tol = 32*np.finfo(float).eps*max(1., float(np.abs(raw).max()))
    if std <= tol or raw.std() <= raw_tol:
        return white, raw.copy()
    z = np.array([1., -1.])*np.sign(white[0]-white[1]) if len(raw) == 2 else residual/std
    return white, raw + alpha*raw.std()*np.tanh(z)


def validate_model(out, p):
    if out.get('status') == 'provider_rejection':
        assert out['provider_error_code'] == 'data_inspection_failed' and out['http_status'] == 400
        assert out['raw_response'] == '' and out['returned_model'] is None and out['response_id'] is None
        assert out['finish_reason'] == 'provider_rejection' and out['refusal'] == 'provider_data_inspection_failed'
        assert out['requested_model'] == p['expected_model'] and out['seed'] == 42
        return
    assert out['requested_model'] == out['returned_model'] == p['expected_model']
    assert out['seed'] == 42 and out['temperature'] == 0 and out['enable_thinking'] is False
    assert out['model_identity_kind'] == 'provider_alias' and out['resolved_model_revision'] is None
    assert out['reasoning_tokens'] in (None,0) and out['reasoning_content_present'] is False
    assert out['finish_reason'] == 'stop' and not out['refusal'] and out['raw_response'].strip()


def audit_captions(output, p):
    jobs = rows(output/'caption_jobs.jsonl')
    caps = rows(output/'caption_source/captions.jsonl')
    api = rows(output/'caption_source/api_responses.jsonl')
    cmap = {c['id']:c for c in caps}
    amap = {r['id']:r for r in api}
    assert len(jobs) == len(caps) == len(cmap) == len(api) == len(amap) == 979
    for job in jobs:
        assert sha(job['path']) == job['sha256'] == job['id']
        payload = run.caption_payload(job, p)
        saved = read(output/'caption_source/responses'/(job['id']+'.json'))
        assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
        out = saved['output']; validate_model(out, p)
        assert out['request_sha256'] == run.transport.request_sha256(run.images.wire_for(payload, p['caption_max_tokens']))
        assert amap[job['id']] == dict(stage='caption', id=job['id'], output=out)
        assert cmap[job['id']] == dict(id=job['id'], caption=out['raw_response'].strip(), **out)
    obs = read(output/'observations.json')
    source = read(output/'observations_without_captions.json')
    occurrences = 0
    for task, group in source.items():
        for r in group:
            r['raw']['image_caption'] = [cmap[k]['caption'] for k in r['caption_ids']]
            occurrences += len(r['caption_ids'])
        assert obs[task] == group
    assert occurrences == 1003
    rejected = [c['id'] for c in caps if c.get('status') == 'provider_rejection']
    dump(output/'caption_audit.json', dict(status='passed', unique_images=979, occurrences=1003, rejected_images=len(rejected), rejected_image_ids=rejected,
        exact_requests_and_image_encoding_provenance_verified=True, captions_joined_by_image_sha256=True,
        source_dialogues_and_metadata_preserved=True))


def audit_reused_responses(output, p):
    manifest = read(output/'reuse_manifest.json')
    assert manifest['experiment_signature'] == p['experiment_signature']
    assert sha(manifest['source_protocol_path']) == manifest['source_protocol_sha256']
    for entry in manifest['records']:
        assert sha(entry['source_path']) == entry['source_sha256']
        original = read(entry['source_path'])
        dest = read(entry['destination_path'])
        assert original['experiment_signature'] == manifest['source_signature']
        assert dest == dict(original, experiment_signature=p['experiment_signature'])
        out = dest['output']; validate_model(out, p)
        assert run.transport.request_sha256(run.images.wire_for(dest['payload'], 2048)) == out['request_sha256']
    for entry in manifest['rejections']:
        assert sha(entry['source_path']) == entry['source_sha256']
        failures=rows(entry['source_path'])
        assert entry['attempt'] in failures and entry['attempt']['provider_error_code']=='data_inspection_failed'
        saved=read(entry['destination_path']); validate_model(saved['output'], p)
        assert saved['output']['request_sha256'] == entry['attempt']['request_sha256']
        assert run.transport.request_sha256(run.images.wire_for(saved['payload'],2048)) == saved['output']['request_sha256']


def audit(output):
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    p = run.verify(output)
    dump(output/'image_encoding_audit.json', run.images.audit())
    audit_reused_responses(output, p)
    artifacts = {}
    stage_markers = {}
    for name in ('caption_source/complete.json', 'candidates_complete.json', 'offline_complete.json', 'retrieval_complete.json', 'reader/complete.json'):
        marker = run.check_marker(output, name, p)
        assert marker, name
        artifacts.update(marker['artifact_sha256'])
        artifacts[str(output / name)] = sha(output / name)
        stage_markers[name] = marker
    audit_captions(output, p)
    qs, memories = rows(output / 'inputs/questions.jsonl'), rows(output / 'memories.jsonl')
    assert memories == run.base.memory_records(read(output / 'observations.json'))
    qmap, mmap = {q['id']: q for q in qs}, {m['id']: m for m in memories}
    candidates = rows(output / 'candidates.jsonl')
    cmap = {c['id']: c for c in candidates}
    assert len(candidates) == len(cmap) == len(qmap) == 1711 and set(cmap) == set(qmap)
    tokenizer = AutoTokenizer.from_pretrained(p['image_encoder']['model_name'], local_files_only=True)
    candidate_calls = rows(output / 'candidate_generation/api_responses.jsonl')
    c_api = {c['output']['response_id']: c for c in candidate_calls}
    assert len(c_api) == len(candidate_calls)
    for q in qs:
        c = cmap[q['id']]
        assert c['experiment_signature'] == p['experiment_signature']
        for attempt in range(1, c['accepted_attempt']+1):
            path = output / 'candidate_generation/responses' / (obj([q['id'], attempt])+'.json')
            saved = read(path)
            payload = run.candidate_payload(q, attempt)
            assert saved['payload'] == payload and saved['experiment_signature'] == p['experiment_signature']
            out = saved['output']
            assert out['request_sha256'] == run.ga.request_sha256(run.images.wire_for(payload, p['candidate_max_tokens']))
            assert c_api[out['response_id']] == dict(stage='candidate', id=q['id'], output=out)
            assert out['returned_model'] == p['expected_model'] and out['seed'] == 42
            assert out['finish_reason'] == 'stop' and not out['refusal']
            validate_model(out, p)
            if attempt < c['accepted_attempt']:
                try: run.ga.gallery.parse_candidates(out['raw_response'], tokenizer, return_format=True)
                except ValueError: pass
                else: raise AssertionError('Previously rejected candidate parses as valid')
            if attempt == c['accepted_attempt']:
                statements, lengths, fmt = run.ga.gallery.parse_candidates(out['raw_response'], tokenizer, return_format=True)
                assert statements == c['candidate_statements'] and lengths == c['clip_token_lengths'] and fmt == c['response_format']
    retrieved = rows(output / 'retrieval.jsonl')
    rmap = {r['id']: r for r in retrieved}
    diagnostics = {r['id']: r for r in rows(output / 'weighting_diagnostics.jsonl')}
    assert len(retrieved) == len(rmap) == 1711 and set(rmap) == set(qmap) == set(diagnostics)
    encoder = run.base.bge.load_text_encoder(p['text_encoder'], 'cpu')
    clip, tokenizer, limit = run.base.bge.load_clip_text_encoder(p['image_encoder'], 'cpu')
    max_dual_error, group_count, master_counts = 0., 0, {}
    for task in p['tasks']:
        ms, selected = [m for m in memories if m['task'] == task], [q for q in qs if q['task'] == task]
        dest = output / 'offline' / task
        graph, sums = read(dest / 'graph.json'), np.load(dest / 'master_sums.npy', allow_pickle=False)
        basis = np.load(dest / 'basis.npy', allow_pickle=False)
        assert np.array_equal(basis, np.load(run.original.BASIS, allow_pickle=False))
        masters, nodes = graph['masters'], {n['id']: n for n in graph['image_nodes']}
        visual = [m for m in ms if m['is_visual']]
        assert set(nodes) == {m['id']+'#image1' for m in visual}
        assert graph['leaf_visual_dim'] == 2 and not graph['text_embedding_cache']['persisted']
        cv = np.stack([run.base.bge.encode_text(encoder, [m['caption']])[0] for m in visual])
        choice = choose_caption_partition(cv)
        expected_groups = {frozenset(visual[i]['id']+'#image1' for i in group) for group in choice['partition']}
        assert expected_groups == {frozenset(m['member_ids']) for m in masters}
        insertions = rows(dest / 'insertion_audit.jsonl')
        assert [a['image_id'] for a in insertions] == [m['id']+'#image1' for m in visual]
        assert [a['n_images'] for a in insertions] == list(range(1, len(visual)+1))
        assert all(a['global_conservation_max_abs_error'] < 1e-10 and a['max_projection_consistency_l2_error'] < 1e-10 for a in insertions)
        for m in visual:
            n = nodes[m['id']+'#image1']
            assert all(n[k] == m[k] for k in ('caption', 'dialogue', 'timestamp'))
            assert len(n['visual_coords']) == 2
        for m in masters:
            assert m['n'] == len(m['member_ids'])
            assert all(nodes[n]['master_id'] == m['id'] for n in m['member_ids'])
            expected = np.sum([nodes[n]['visual_coords'] for n in m['member_ids']], axis=0)
            np.testing.assert_allclose(basis.T @ sums[m['visual_sum_ref']['row']], expected, atol=1e-10, rtol=0)
        master_counts[task] = len(masters)
        centers = run.original.normalize([sums[m['visual_mean_ref']['row']]/m['n'] for m in masters])
        with np.load(output / 'embeddings' / (task+'.npz'), allow_pickle=False) as saved:
            vectors, queries, candidate_vectors = saved['memory_vectors'], saved['query_vectors'], saved['candidate_vectors']
            assert saved['memory_ids'].tolist() == [m['id'] for m in ms]
            assert saved['question_ids'].tolist() == [q['id'] for q in selected]
        np.testing.assert_allclose(vectors, run.base.bge.encode_text(encoder, [m['retrieval_text'] for m in ms]), atol=1e-7, rtol=0)
        np.testing.assert_allclose(queries[:, 0], run.base.bge.encode_text(encoder, [q['retrieval_query'] for q in selected], p['query_prefix']), atol=1e-7, rtol=0)
        text = [q['retrieval_query']+'\nCandidate answer: '+h for q in selected for h in cmap[q['id']]['candidate_statements']]
        np.testing.assert_allclose(queries[:, 1:], run.base.bge.encode_text(encoder, text, p['query_prefix']).reshape(len(selected),4,-1), atol=1e-7, rtol=0)
        statements = [h for q in selected for h in cmap[q['id']]['candidate_statements']]
        np.testing.assert_allclose(candidate_vectors, run.clip_vectors(clip, tokenizer, limit, statements).reshape(len(selected),4,-1), atol=1e-7, rtol=0)
        groups = [[i for i,m in enumerate(ms) if m['is_visual'] and nodes[m['id']+'#image1']['master_id'] == master['id']] for master in masters]
        for qi, q in enumerate(selected):
            r = rmap[q['id']]
            assert r['task'] == task and r['retrieval_query'] == q['retrieval_query'] and r['candidate_count'] == len(ms)
            assert r['experiment_signature'] == p['experiment_signature']
            scores = centers @ candidate_vectors[qi].T
            choices = np.argmax(scores, axis=1)
            raw = vectors @ queries[qi, 0]; weighted = raw.copy()
            for mi, indices in enumerate(groups):
                c = r['master_choices'][mi]
                assert c['master_id'] == masters[mi]['id'] and c['candidate_index'] == int(choices[mi])
                np.testing.assert_allclose(c['clip_scores'], scores[mi], atol=1e-12, rtol=0)
                query = queries[qi, 1+choices[mi]]
                raw[indices] = vectors[indices] @ query
                white, weighted[indices] = independent_weights(vectors[indices], query, p['lambda'], p['alpha'])
                stored = diagnostics[q['id']]['groups'][mi]
                assert stored['indices'] == indices and stored['query_index'] == 1+choices[mi]
                err = float(np.max(np.abs(white-stored['white'])))
                assert err < 2e-11
                max_dual_error = max(max_dual_error, err); group_count += 1
            for name, values, c in zip(run.CONDITIONS, (vectors @ queries[qi, 0], raw, weighted), r['conditions']):
                order = sorted(range(len(ms)), key=lambda i: (-float(values[i]), i))
                ids = [ms[i]['id'] for i in order]
                assert c['name'] == name and c['ranking_memory_ids'] == ids and c['source_ids'] == ids[:5]
                np.testing.assert_allclose(c['ranking_scores'], values[order], atol=2e-11, rtol=0)
                assert c['retrieved_context'] == run.base.context_for(ids[:5], mmap)
        print(f'Audited master retrieval: {task}', flush=True)
    dump(output / 'retrieval_audit.json', dict(status='passed', full_rankings_verified=5133,
        caption_partitions_recomputed=True, clip_choices_recomputed=True, independent_ridge_groups=group_count,
        max_dual_ridge_error=max_dual_error, master_counts=master_counts, text_only_baseline_scores_preserved=True))
    # Reconstruct the exact outbound reader request for every condition/question.
    manifest, answers = rows(output / 'reader/manifest.jsonl'), rows(output / 'reader/responses.jsonl')
    lookup = {(m['condition'], m['payload']['record_id']): m for m in manifest}
    amap = {(a['condition'], a['id']): a for a in answers}
    expected = {(c,qid) for c in run.CONDITIONS for qid in qmap}
    assert len(manifest) == len(answers) == len(lookup) == len(amap) == 5133
    assert set(lookup) == set(amap) == expected
    api = rows(output / 'reader/api_responses.jsonl')
    by_response = {a['output']['request_sha256']: a['output'] for a in api}
    assert len(api) == len(by_response) == len({m['key'] for m in manifest})
    evaluated = []
    for q in qs:
        for c in rmap[q['id']]['conditions']:
            m, a = lookup[c['name'], q['id']], amap[c['name'], q['id']]
            payload = run.ga.reader_payload(q, c['retrieved_context'])
            assert m['payload'] == payload
            assert m['key'] == run.ga.request_sha256(run.images.wire_for(payload, p['max_new_tokens']))
            saved = read(output / 'reader/responses' / (m['key']+'.json'))
            assert run.ga.request_sha256(run.images.wire_for(saved['payload'], p['max_new_tokens'])) == m['key']
            out = saved['output']
            assert a == dict(id=q['id'], condition=c['name'], **out, experiment_signature=p['experiment_signature'])
            assert out == by_response[out['request_sha256']] and out['request_sha256'] == m['key']
            validate_model(out, p)
            evaluated.append(dict(id=q['id'], task=q['task'], point=q['point'], condition=c['name'],
                raw_response=out['raw_response'], provider_rejected=out.get('status')=='provider_rejection', source_ids=c['source_ids'], context_count=5, request_sha256=m['key']))
    # Answer labels are first opened here, after generation and payload audits.
    gold = {g['id']:g for g in rows(output / 'inputs/gold.jsonl')}
    assert set(gold) == set(qmap)
    for r in evaluated:
        g = gold[r['id']]; gids = set(g['gold_ids'])
        assert g['answer'].strip()
        r.update(answer=g['answer'], **run.ga.gallery.score_answer(r['raw_response'], g['answer']),
            gold_ids=sorted(gids), gold_count=len(gids), gold_hits=len(gids & set(r['source_ids'])))
        pt, gt = (run.ga.gallery.normalize_answer_universal(s).split() for s in (r['raw_response'],g['answer']))
        hits = sum((Counter(pt)&Counter(gt)).values())
        bleu = hits/len(pt)*min(1.,math.exp(1.-len(gt)/len(pt))) if pt and gt and hits else 0.
        assert abs(bleu-r['bleu_1']) < 1e-12
    for r in evaluated:
        if r['provider_rejected']: assert all(r[k] == 0 for k in ('em','f1','bleu_1','bleu_2','bleu'))
    result = dict(status='complete', model=p['expected_model'], seed=42, **{'lambda':p['lambda'],'alpha':p['alpha']},
        results={c:dict(**summary([r for r in evaluated if r['condition']==c]), provider_rejected_answers=sum(r['provider_rejected'] for r in evaluated if r['condition']==c)) for c in run.CONDITIONS},
        by_type={c:{t:summary([r for r in evaluated if r['condition']==c and r['point']==t]) for t in sorted({q['point'] for q in qs})} for c in run.CONDITIONS},
        by_task={c:{t:summary([r for r in evaluated if r['condition']==c and r['task']==t]) for t in p['tasks']} for c in run.CONDITIONS},
        experiment_signature=p['experiment_signature'])
    run.write_rows(output / 'scored_questions.jsonl', evaluated)
    with (output / 'scored_questions.csv').open('w') as f:
        fields=['condition','id','task','point','answer','raw_response','em','f1','bleu_1','bleu_2','bleu','gold_hits','gold_count']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(evaluated)
    dump(output / 'final_results.json',result)
    comparison=[dict(method=c,**result['results'][c]) for c in run.CONDITIONS]
    dump(output / 'comparison.json',dict(model=p['expected_model'],questions=1711,top_k=5,methods=comparison,
        **{'lambda':p['lambda'],'alpha':p['alpha']}, identical_paired_requests_shared=stage_markers['reader/complete.json']['shared_requests'],
        baseline_policy='All three methods use fresh DeepSeek captions and fresh responses; identical memories, original queries and reader prompts.'))
    table=''.join('<tr><td>'+html.escape(r['method'])+'</td><td>'+str(r['correct'])+'/1711</td>'+''.join(f'<td>{100*r[k]:.2f}%</td>' for k in ('em','f1','bleu_1','source_recall_micro'))+'</tr>' for r in comparison)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>DeepSeek master comparison</title><style>body{font:16px system-ui;margin:36px}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        f'<h1>Mem-Gallery · DeepSeek · Top5</h1><p>1711 questions. z-score lambda={p["lambda"]}, alpha={p["alpha"]}. Identical DeepSeek history captions and official question captions. Twelve oversized source files use recorded JPEG compression at the original resolution; all other image bytes are unchanged. Content-inspection failures are retained: empty image captions, or zero-scored answers; no rejected request is resubmitted and no question is dropped.</p>'
        '<table><tr><th>Method</th><th>Exact matches</th><th>EM</th><th>F1</th><th>BLEU-1</th><th>Recall@5</th></tr>'+table+'</table>'
        '<p>Fresh DeepSeek hypotheses and caption-based masters are shared by both methods; only within-master BGE scoring changes. Text-only memories retain original query scores. All evidence is original dialogue/captions.</p>'
        '<p>Exact same paired requests share one fresh response; no previous answers reused for the three methods. The Natural baseline also receives fresh DeepSeek captions and answers.</p>'
        '<p><a href="scored_questions.csv">Per-question results</a> · <a href="audit.json">Audit</a> · <a href="protocol.json">Protocol</a></p>')
    usage={}
    for stage, folder in [('caption','caption_source'),('candidate','candidate_generation'),('answer','reader')]:
        outs=[c['output'] for c in rows(output/folder/'api_responses.jsonl')]
        attempts=rows(output/folder/'api_attempts.jsonl')
        for out in outs: validate_model(out, p)
        usage[stage]=dict(calls=len(outs),attempts=dict(Counter(a['status'] for a in attempts)),
            finish_reasons=dict(Counter(o['finish_reason'] for o in outs)),refusals=sum(bool(o['refusal']) for o in outs),
            input_tokens=sum(o['input_tokens'] for o in outs),output_tokens=sum(o['generated_tokens'] for o in outs))
    for name in ('image_encoding_audit.json','reuse_manifest.json','caption_audit.json','protocol.json','final_results.json','retrieval_audit.json','scored_questions.jsonl','scored_questions.csv','comparison.json','report.html'):
        artifacts[str(output/name)]=sha(output/name)
    dump(output/'audit.json',dict(status='passed',questions_per_method=1711,methods=3,
        actual_payloads_verified=True,query_image_provenance_and_encoded_bytes_verified=True,independent_bleu1_checked=True,
        independent_ridge_groups=group_count,max_dual_ridge_error=max_dual_error,
        api_usage=usage,candidate_format_retries=stage_markers['candidates_complete.json']['format_retries'],
        artifacts_sha256=artifacts,experiment_signature=p['experiment_signature']))
    artifacts[str(output/'audit.json')]=sha(output/'audit.json')
    dump(output/'completion_manifest.json',dict(status='complete',artifacts_sha256=artifacts,experiment_signature=p['experiment_signature']))
    print(result['results'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=run.OUTPUT)
    audit(parser.parse_args().output.resolve())
