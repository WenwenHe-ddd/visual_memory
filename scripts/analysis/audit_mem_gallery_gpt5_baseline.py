"""Reconstruct Natural retrieval, actual GPT-5 payloads and scores before reporting."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import math
from pathlib import Path

import numpy as np
from scripts.evaluation import mem_gallery_gpt5_baseline as run

read, rows, sha, obj, dump = run.read, run.rows, run.sha, run.obj, run.dump


def summary(items):
    gold = [r for r in items if r['gold_count']]
    return dict(questions=len(items), correct=sum(int(r['em']) for r in items),
        **{k: sum(r[k] for r in items) / len(items) for k in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu')},
        gold_questions=len(gold), gold_hits=sum(r['gold_hits'] for r in gold),
        gold_links=sum(r['gold_count'] for r in gold),
        source_recall_micro=sum(r['gold_hits'] for r in gold) / sum(r['gold_count'] for r in gold) if gold else None,
        source_recall_macro=sum(r['gold_hits'] / r['gold_count'] for r in gold) / len(gold) if gold else None,
        context_counts=dict(Counter(r['context_count'] for r in items)))


def audit_retrieval(output, p):
    import torch
    torch.set_num_threads(4)
    done = read(output / 'retrieval_complete.json')
    assert done['count'] == 1711 and done['truncated_inputs'] == 0
    assert done['experiment_signature'] == p['experiment_signature']
    for path, digest in done['artifact_sha256'].items():
        assert sha(path) == digest
    qs, memories = rows(output / 'inputs/questions.jsonl'), rows(output / 'memories.jsonl')
    assert memories == run.memory_records(read(output / 'observations.json'))
    retrieved = rows(output / 'retrieval.jsonl')
    rmap, mmap = {r['id']: r for r in retrieved}, {r['id']: r for r in memories}
    assert len(rmap) == len(retrieved) == len(qs) == 1711
    assert set(rmap) == {q['id'] for q in qs}
    encoder = run.bge.load_text_encoder(p['text_encoder'], 'cpu')
    max_error = 0.
    for task in p['tasks']:
        ms, selected = [m for m in memories if m['task'] == task], [q for q in qs if q['task'] == task]
        vectors = run.bge.encode_text(encoder, [m['retrieval_text'] for m in ms])
        queries = run.bge.encode_text(encoder, [q['retrieval_query'] for q in selected], p['query_prefix'])
        with np.load(output / 'embeddings' / (task + '.npz'), allow_pickle=False) as saved:
            assert saved['memory_ids'].tolist() == [m['id'] for m in ms]
            assert saved['question_ids'].tolist() == [q['id'] for q in selected]
            assert np.allclose(vectors, saved['memory_vectors'], rtol=0, atol=1e-7)
            assert np.allclose(queries, saved['query_vectors'], rtol=0, atol=1e-7)
        for q, vector in zip(selected, queries):
            r = rmap[q['id']]
            scores = vectors @ vector
            # Independent stable Python sorting also checks ties against source order.
            order = sorted(range(len(ms)), key=lambda i: (-float(scores[i]), i))
            ranked = [ms[i]['id'] for i in order]
            error = float(np.max(np.abs(scores[order] - r['ranking_scores'])))
            assert error < 1e-7
            max_error = max(max_error, error)
            assert r['task'] == q['task'] and r['candidate_count'] == len(ms)
            assert r['ranking_memory_ids'] == ranked and r['source_ids'] == ranked[:5]
            assert r['retrieved_context'] == run.context_for(ranked[:5], mmap)
            assert r['retrieval_query'] == q['retrieval_query']
            assert r['experiment_signature'] == p['experiment_signature']
    dump(output / 'retrieval_audit.json', dict(status='passed', reencoded_memories=len(memories),
        reencoded_queries=len(qs), all_full_rankings_verified=True, all_contexts_verified=True,
        maximum_score_error=max_error, max_embedding_tokens=done['max_embedding_tokens'], truncated_inputs=0))
    return qs, rmap, dict(done['artifact_sha256'])


def audit(output):
    p = run.verify(output)
    qs, retrieval, artifacts = audit_retrieval(output, p)
    root = output / 'reader'
    done = read(root / 'complete.json')
    assert done['count'] == 1711 and done['experiment_signature'] == p['experiment_signature']
    for key in ('responses', 'manifest'):
        assert done[key + '_sha256'] == sha(root / (key + '.jsonl'))
    answers, manifest = rows(root / 'responses.jsonl'), rows(root / 'manifest.jsonl')
    amap, mmap = {a['id']: a for a in answers}, {m['record_id']: m for m in manifest}
    assert len(answers) == len(manifest) == len(amap) == len(mmap) == 1711
    assert set(amap) == set(mmap) == set(retrieval)
    calls, attempts = rows(root / 'api_responses.jsonl'), rows(root / 'api_attempts.jsonl')
    by_response = {c['output']['response_id']: c for c in calls}
    assert len(by_response) == len(calls) == 1711
    assert len({a['response_id'] for a in answers}) == 1711
    evaluated = []
    for q in qs:
        r, a = retrieval[q['id']], amap[q['id']]
        payload = run.ga.reader_payload(q, r['retrieved_context'])
        saved_path = root / 'responses' / (obj(q['id']) + '.json')
        saved = read(saved_path)
        artifacts[str(saved_path)] = sha(saved_path)
        assert saved['payload'] == payload == mmap[q['id']]
        out = saved['output']
        assert a == dict(id=q['id'], **out, experiment_signature=p['experiment_signature'])
        assert saved['experiment_signature'] == p['experiment_signature']
        assert out['request_sha256'] == run.ga.request_sha256(run.images.wire_for(payload, p['max_new_tokens']))
        assert out['requested_model'] == 'gpt-5' and out['returned_model'] == p['expected_model']
        assert out['seed'] == 42 and out['reasoning_effort'] == 'minimal'
        assert out['finish_reason'] == 'stop' and not out['refusal'] and out['raw_response'].strip()
        assert by_response[out['response_id']] == dict(stage='answer', id=q['id'], output=out)
        evaluated.append(dict(id=q['id'], task=q['task'], point=q['point'],
            raw_response=out['raw_response'], source_ids=r['source_ids'], context_count=5))
    # Gold labels are first opened only after retrieval/payload/generation checks.
    gold = {g['id']: g for g in rows(output / 'inputs/gold.jsonl')}
    assert set(gold) == set(retrieval)
    for r in evaluated:
        g = gold[r['id']]
        r.update(answer=g['answer'], **run.ga.gallery.score_answer(r['raw_response'], g['answer']))
        gids = set(g['gold_ids'])
        r.update(gold_ids=sorted(gids), gold_count=len(gids), gold_hits=len(gids & set(r['source_ids'])))
        # Independent unigram BLEU calculation (clipped precision and brevity penalty).
        pt = run.ga.gallery.normalize_answer_universal(r['raw_response']).split()
        gt = run.ga.gallery.normalize_answer_universal(g['answer']).split()
        hits = sum((Counter(pt) & Counter(gt)).values())
        bleu1 = hits / len(pt) * min(1., math.exp(1. - len(gt) / len(pt))) if pt and gt and hits else 0.
        assert abs(bleu1 - r['bleu_1']) < 1e-12
    result = dict(status='complete', method='Natural baseline', model=p['expected_model'], seed=42,
        overall=summary(evaluated),
        by_task={t: summary([r for r in evaluated if r['task'] == t]) for t in p['tasks']},
        by_type={t: summary([r for r in evaluated if r['point'] == t]) for t in sorted({r['point'] for r in evaluated})},
        experiment_signature=p['experiment_signature'],
        metric_note='Official normalized EM, stemmed token F1, sentence BLEU1/2/4 macro means; original-source Recall@5.')
    run.write_rows(output / 'scored_questions.jsonl', evaluated)
    with (output / 'scored_questions.csv').open('w') as f:
        fields = ['id', 'task', 'point', 'answer', 'raw_response', 'em', 'f1', 'bleu_1', 'bleu_2', 'bleu', 'gold_hits', 'gold_count']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(evaluated)
    dump(output / 'final_results.json', result)
    comparison = [dict(method='Natural baseline', **result['overall'], result_root=str(output))]
    for label, source in [('Generative Agents', run.SOURCE), ('MemoryOS', run.shared.OUTPUT)]:
        assert sha(source / 'inputs/questions.jsonl') == sha(output / 'inputs/questions.jsonl')
        assert sha(source / 'inputs/gold.jsonl') == sha(output / 'inputs/gold.jsonl')
        assert sha(source / 'observations.json') == sha(output / 'observations.json')
        audit_record = read(source / 'audit.json')
        recorded = audit_record.get('artifacts_sha256', audit_record.get('artifact_sha256'))
        assert recorded[str(source / 'scored_questions.jsonl')] == sha(source / 'scored_questions.jsonl')
        old_rows = rows(source / 'scored_questions.jsonl')
        assert len(old_rows) == 1711 and {r['id'] for r in old_rows} == set(gold)
        comparison.append(dict(method=label, **summary(old_rows), result_root=str(source)))
    dump(output / 'comparison.json', dict(model=p['expected_model'], questions=1711, methods=comparison,
        caption_policy=p['caption_policy'],
        limitation='Native context budgets differ: Natural Top5, GA Top10 with native word cap, MemoryOS pages plus generated knowledge.'))
    table = ''.join('<tr><td>' + html.escape(r['method']) + '</td><td>' + str(r['correct']) + '/1711</td>' +
        ''.join(f'<td>{r[k]*100:.2f}%</td>' for k in ('em', 'f1', 'bleu_1', 'source_recall_micro')) + '</tr>' for r in comparison)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Mem-Gallery GPT-5 baseline</title>'
        '<style>body{font:16px system-ui;margin:36px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}</style>'
        '<h1>Mem-Gallery · GPT-5</h1><p>1711 questions; shared GPT-5 history captions, official question captions and original question image bytes.</p>'
        '<table><tr><th>Method</th><th>Exact matches</th><th>EM</th><th>F1</th><th>BLEU-1</th><th>Original-source recall</th></tr>' + table + '</table>'
        '<p>Natural: BGE full scan, Top5 original rounds, no z-score, hypotheses or reflection. GA and MemoryOS results are reused after verification.</p>'
        '<p>Context budgets differ. Recall uses only original retrieved rounds; generated reflections/knowledge are not credited. BLEU is a mean over sentences.</p>'
        '<p><a href="scored_questions.csv">Per-question scores</a> · <a href="audit.json">Audit</a> · <a href="protocol.json">Protocol</a></p>')
    for path in [output / name for name in ('protocol.json', 'memories.jsonl', 'retrieval_audit.json',
        'final_results.json', 'scored_questions.jsonl', 'scored_questions.csv', 'comparison.json', 'report.html')]:
        artifacts[str(path)] = sha(path)
    for path in [root / name for name in ('complete.json', 'manifest.jsonl', 'responses.jsonl', 'api_responses.jsonl', 'api_attempts.jsonl')]:
        artifacts[str(path)] = sha(path)
    outs = [c['output'] for c in calls]
    audit_result = dict(status='passed', questions=1711, source_rounds=3962,
        actual_payloads_verified=True, original_image_bytes_verified=True, full_rankings_recomputed=True,
        independent_bleu1_recalculation=True, api_calls=len(calls),
        api_attempts=dict(Counter(a['status'] for a in attempts)),
        finish_reasons=dict(Counter(o['finish_reason'] for o in outs)),
        input_tokens=sum(o['input_tokens'] for o in outs), output_tokens=sum(o['generated_tokens'] for o in outs),
        refusals=sum(bool(o['refusal']) for o in outs), artifacts_sha256=artifacts,
        experiment_signature=p['experiment_signature'])
    dump(output / 'audit.json', audit_result)
    dump(output / 'execution_status.json', dict(status='complete', results_available=True,
        updated_at_utc=datetime.now(timezone.utc).isoformat()))
    artifacts[str(output / 'audit.json')] = sha(output / 'audit.json')
    dump(output / 'completion_manifest.json', dict(status='complete', artifacts_sha256=artifacts,
        experiment_signature=p['experiment_signature']))
    print(result['overall'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    audit(parser.parse_args().output.resolve())
