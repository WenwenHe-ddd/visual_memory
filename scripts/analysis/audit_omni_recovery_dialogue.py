"""Full original Omni audit plus explicit, bounded recovery-amendment verification."""
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path


def analyze(output):
    from scripts.evaluation import mem_gallery_qwen_omni as run
    p = run.verify(output)
    from scripts.evaluation.omni_qwen_recovery_dialogue import verify_amendment, correction_prompt
    from scripts.evaluation.omni_qwen_recovery_20261105 import retry_message
    amendment = verify_amendment(output)
    qlist = run.rows(output / 'inputs/questions.jsonl')
    questions = {q['id']: q for q in qlist}
    gold = {r['id']: r for r in run.rows(output / 'inputs/gold.jsonl')}
    retrieved = run.rows(output / 'retrieval.jsonl')
    answers = run.rows(output / 'answers.jsonl')
    assert len(retrieved) == len(answers) == len(questions) == p['questions']
    assert {r['id'] for r in retrieved} == {r['id'] for r in answers} == set(questions)
    amap = {a['id']: a for a in answers}
    all_sources, all_maus, ingestion = {}, {}, []
    for task in p['tasks']:
        dest = output / 'memory' / task
        complete = run.read(dest / 'complete.json')
        assert complete['experiment_signature'] == p['experiment_signature']
        for path, digest in complete['artifact_sha256'].items():
            assert run.sha(path) == digest, path
        sources = run.read(dest / 'runtime/sources.json')
        maus = run.read(dest / 'runtime/final_memory.json')
        assert set(sources) == {m['id'] for m in maus}
        all_sources.update(sources)
        all_maus.update({m['id']: m for m in maus})
        records = run.read(dest / 'runtime/ingest.json')
        assert all(r['success'] or r['skipped'] for r in records)
        assert sum(r['source']['modality'] == 'text' for r in records) == complete['rounds']
        for mau in maus:
            assert len(mau['embedding']) == 512
            if sources[mau['id']]['modality'] == 'visual':
                assert mau['raw_pointer'] and mau['modality_type'] == 'visual'
        ingestion.extend(records)
    scored, visual_queries = [], 0
    for record in retrieved:
        identity = record['id']
        q, answer, detail = questions[identity], amap[identity], record['retrieval_detail']
        assert record['retrieval_query'] == q['retrieval_query']
        assert record['experiment_signature'] == answer['experiment_signature'] == p['experiment_signature']
        ids = detail['context_memory_ids']
        assert len(ids) == len(set(ids)) <= 5 and set(ids) <= set(all_sources)
        # Recompute rankings without calling the adapter's fusion function.
        weights, ties = {}, {}
        for channel, entries in detail['channels'].items():
            assert len(entries) <= {'semantic': 10, 'keyword': 5}[channel]
            seen = set()
            for rank, entry in enumerate(entries, 1):
                key = entry['id']
                assert key in all_sources
                if key in seen:
                    continue
                seen.add(key)
                ties.setdefault(key, len(ties))
                weights[key] = weights.get(key, 0.) + 1. / (60 + rank)
        assert sorted(weights, key=lambda k: (-weights[k], ties[k]))[:5] == ids
        assert detail['source_ids'] == [all_sources[k]['round_id'] for k in ids]
        assert all(all_sources[k]['task'] == q['task'] for k in ids)
        assert [i['id'] for i in detail['context_items']] == ids
        for item in detail['context_items']:
            mau = all_maus[item['id']]
            assert item['summary'] == mau['summary']
            assert item.get('details') == mau.get('details')
        expected_visual = [k for k in ids if all_sources[k]['modality'] == 'visual']
        assert [i['mau_id'] for i in detail['history_images']] == expected_visual
        for image in detail['history_images']:
            assert run.sha(image['path']) == image['sha256']
            assert image['image_id'] == all_sources[image['mau_id']]['image_id']
        parts = answer['messages'][1]['content']
        actual_images = [part for part in parts if part['type'] == 'image']
        expected_images = [dict(path=i['path'], sha256=i['sha256']) for i in detail['history_images']]
        expected_images += [dict(path=f, sha256=run.sha(f)) for f in q['query_image_paths']]
        assert [dict(path=i['path'], sha256=i['sha256']) for i in actual_images] == expected_images
        assert answer['messages'][0] == dict(role='system', content=p['reader_system_prompt'])
        assert parts[-1]['text'] == run.base.reader_text(q, record['retrieved_context'])
        call = run.read(output / 'memory/calls' / (answer['output']['call_key'] + '.json'))
        assert call['request']['messages'] == answer['messages']
        assert call['request']['max_new_tokens'] == 256 and not call['request']['do_sample']
        assert all(call['output'][k] == v for k, v in answer['output'].items() if k != 'call_key')
        assert not answer['output']['hit_token_limit']
        score = run.base.gallery.score_answer(answer['output']['raw_response'], gold[identity]['answer'])
        target = set(gold[identity].get('gold_ids', []))
        hits = len(target & set(detail['source_ids']))
        scored.append(dict(id=identity, task=q['task'], point=q['point'], answer=gold[identity]['answer'],
            raw_response=answer['output']['raw_response'], **score, context_count=len(ids),
            history_image_count=len(expected_visual), gold_sources=len(target), source_hits=hits))
        visual_queries += bool(expected_visual)
    calls = list((output / 'memory/calls').glob('*.json'))
    recoveries_path = output / 'memory/transport_recoveries.jsonl'
    recoveries = run.rows(recoveries_path) if recoveries_path.exists() else []
    rejected = set()
    for recovery in recoveries:
        keys = recovery['attempt_keys']
        assert 2 <= len(keys) <= 7 and recovery['status'] == 'recovered'
        if len(keys) > 3:
            assert recovery['recovery_amendment'] == '20261105'
        assert keys[-1] == recovery['final_key'] and len(recovery['causes']) == len(keys) - 1
        original = run.read(output / 'memory/calls' / (keys[0] + '.json'))['request']
        for index, key in enumerate(keys):
            call = run.read(output / 'memory/calls' / (key + '.json'))
            request, out = call['request'], call['output']
            assert request['max_new_tokens'] <= 16384
            dialogue = index == 6 and recovery['causes'][index - 1] == 'format' and request['response_format']
            if dialogue:
                assert recovery['dialogue_amendment'] == '20261105-dialogue'
                first_output = run.read(output / 'memory/calls' / (keys[0] + '.json'))['output']
                assert request['messages'] == original['messages'] + [
                    dict(role='assistant', content=first_output['raw_response']),
                    dict(role='user', content=correction_prompt(first_output, request['response_format']))]
            elif index >= 3 and recovery['causes'][index - 1] != 'length':
                previous = run.read(output / 'memory/calls' / (keys[index - 1] + '.json'))
                expected = retry_message(index, previous['output'], request['response_format'])
                assert sum(m == dict(role='system', content=expected) for m in request['messages']) == 1
            assert request['response_format'] == original['response_format']
            strip_retry = lambda messages: [m for m in messages if not (m['role'] == 'system' and
                isinstance(m['content'], str) and m['content'].startswith('Format retry '))]
            if not dialogue:
                assert strip_retry(request['messages']) == strip_retry(original['messages'])
            for field in ('temperature', 'do_sample', 'seed', 'kind', 'experiment_signature'):
                assert request[field] == original[field]
            invalid = out['hit_token_limit'] or not out['raw_response'].strip() or out.get('structured_output_valid') is False
            assert bool(invalid) == (index < len(keys) - 1)
            if invalid:
                rejected.add(key)
    image_calls, call_kinds, truncated = 0, Counter(), 0
    for path in calls:
        call = run.read(path)
        assert run.obj(call['request']) == path.stem
        assert call['request']['experiment_signature'] == p['experiment_signature']
        invalid = not call['output']['raw_response'].strip() or call['output']['hit_token_limit'] or call['output'].get('structured_output_valid') is False
        assert bool(invalid) == (path.stem in rejected)
        call_kinds[call['request']['kind']] += 1
        for message in call['request']['messages']:
            for part in message['content'] if isinstance(message['content'], list) else []:
                if part['type'] == 'image':
                    assert run.sha(part['path']) == part['sha256']
                    if call['request']['kind'] == 'memory':
                        image_calls += 1
    def stats(items):
        n = len(items)
        return dict(questions=n, correct=sum(int(r['em']) for r in items),
            **{k: sum(r[k] for r in items) / n for k in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu')})
    denominator = sum(s['gold_sources'] for s in scored)
    hits = sum(s['source_hits'] for s in scored)
    result = dict(status='pilot_complete' if p['pilot'] else 'complete', method=p['method'],
        model=p['reader_model'], overall=stats(scored),
        by_type={t: stats([r for r in scored if r['point'] == t]) for t in sorted({r['point'] for r in scored})},
        recall_at_5=dict(hits=hits, gold_sources=denominator, value=hits / denominator if denominator else None,
            definition='Saved-order final five MAUs mapped to unique original source rounds; source coverage, not visual fact correctness.'),
        context_counts=dict(Counter(r['context_count'] for r in scored)),
        questions_with_history_pixels=visual_queries,
        ingestion=dict(Counter(r['source']['modality'] + ('_accepted' if r['success'] else '_skipped') for r in ingestion)),
        differences_from_text=p['differences_from_text'],
        recovery_amendment=dict(signature=amendment['signature'], policy=amendment['policy'],
            path=str(output / 'recovery_dialogue_amendment.json')))
    run.write_rows(output / 'scored_questions.jsonl', scored)
    with (output / 'scored_questions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0])); writer.writeheader(); writer.writerows(scored)
    run.dump(output / 'final_results.json', result)
    run.dump(output / 'audit.json', dict(status='passed', questions=len(scored),
        call_counts=dict(call_kinds), memory_image_calls=image_calls, memory_recoveries=len(recoveries),
        discarded_generation_attempts=len(rejected), text_controls_unchanged=True,
        recovery_amendment_verified=True, original_cached_calls_unchanged=amendment['original_calls_preserved'],
        source_labels_only_opened_after_inference=True, rrf_recomputed=True,
        actual_history_image_payloads_verified=True, all_final_generations_complete=True))
    run.dump(output / 'execution.json', dict(status=result['status'], completed_at=datetime.now(timezone.utc).isoformat()))
    body = '<!doctype html><meta charset="utf-8"><title>Omni-SimpleMem Qwen</title><h1>Omni-SimpleMem 0.3.0 / Qwen3-VL-8B</h1>'
    body += '<pre>' + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre>'
    (output / 'report.html').write_text(body)
    artifacts = {str(path): run.sha(path) for path in output.rglob('*') if path.is_file()
                 and path.name not in ('completion_manifest.json', 'native.log', 'run.log')}
    run.freeze(output / 'completion_manifest.json', dict(status=result['status'],
        experiment_signature=p['experiment_signature'], artifact_sha256=artifacts))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
