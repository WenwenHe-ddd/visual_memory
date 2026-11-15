"""Audit actual MemoryOS states/contexts/API payloads before Gallery scoring."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from scripts.evaluation import mem_gallery_gpt5_memoryos as run

read, rows, sha, obj, dump = run.read, run.rows, run.sha, run.obj, run.dump


def page_key(page):
    return tuple(page[k].strip() for k in ('user_input', 'agent_response', 'timestamp'))


def summarize(items):
    gold = [r for r in items if r['gold_count']]
    return dict(questions=len(items), correct=sum(int(r['em']) for r in items),
        **{k: sum(r[k] for r in items) / len(items) for k in ('em', 'f1', 'bleu')},
        gold_questions=len(gold), gold_links=sum(r['gold_count'] for r in gold),
        gold_hits=sum(r['gold_hits'] for r in gold),
        source_recall_micro=sum(r['gold_hits'] for r in gold) / sum(r['gold_count'] for r in gold) if gold else None,
        source_recall_macro=sum(r['gold_hits'] / r['gold_count'] for r in gold) / len(gold) if gold else None,
        context_counts=dict(Counter(r['context_count'] for r in items)),
        extra_knowledge_counts=dict(Counter(r['extra_knowledge_count'] for r in items)))


def audit(output):
    p = run.verify(output)
    root = output / 'memoryos'
    complete = read(root / 'reader/complete.json')
    assert complete['count'] == p['questions'] == 1711
    assert complete['experiment_signature'] == p['experiment_signature']
    for key in ('responses', 'manifest'):
        assert complete[key + '_sha256'] == sha(root / 'reader' / (key + '.jsonl'))
    questions = rows(output / 'inputs/questions.jsonl')
    observations = read(output / 'observations.json')
    mappings, state_counts = {}, {}
    artifacts = {}
    def tracked(path):
        artifacts[str(path)] = sha(path)
        return read(path)
    for task, original in observations.items():
        dest = root / task
        done = tracked(dest / 'complete.json')
        stats = tracked(dest / 'memory_stats.json')
        assert done['experiment_signature'] == p['experiment_signature']
        assert done['stored_rounds'] == stats['stored_rounds'] == len(original)
        assert done['retrieval_sha256'] == sha(dest / 'retrieval.jsonl')
        artifacts[str(dest / 'retrieval.jsonl')] = done['retrieval_sha256']
        for path, digest in done['state_sha256'].items():
            assert sha(path) == digest
            artifacts[path] = digest
        mapping = defaultdict(list)
        for row in original:
            mapping[(run.user_text(row).strip(), row['assistant'].strip(), row['timestamp'].strip())].append(task + '/' + row['id'])
        assert all(len(v) == 1 for v in mapping.values())
        mappings[task] = mapping
        users = dest / 'runtime/users/memorybench_user'
        short = read(users / 'short_term.json')
        middle = read(users / 'mid_term.json')
        pages = [page for session in middle['sessions'].values() for page in session['details']]
        covered = set()
        for page in [*short, *pages]:
            assert len(mapping.get(page_key(page), [])) == 1, (task, page.get('page_id'))
            covered.update(mapping[page_key(page)])
        assert covered == {task + '/' + row['id'] for row in original}, task
        state_counts[task] = dict(source_rounds=len(original), covered_rounds=len(covered),
            short_term_pages=len(short), mid_term_pages=len(pages), mid_term_segments=len(middle['sessions']))
    retrieved = rows(root / 'retrieval.jsonl')
    answers = rows(root / 'reader/responses.jsonl')
    manifest = rows(root / 'reader/manifest.jsonl')
    rmap, amap, mmap = ({r['id'] if 'id' in r else r['record_id']: r for r in records}
                       for records in (retrieved, answers, manifest))
    assert len(retrieved) == len(answers) == len(manifest) == len(rmap) == len(amap) == len(mmap) == 1711
    assert set(rmap) == set(amap) == set(mmap) == {q['id'] for q in questions}
    helper = run.native.helpers('memoryos')
    evaluated = []
    for q in questions:
        r, a = rmap[q['id']], amap[q['id']]
        assert r['task'] == q['task'] and r['retrieval_query'] == q['retrieval_query']
        detail = r['retrieval_detail']
        assert helper['_format_retrieval_context'](detail) == r['retrieved_context']
        source_ids = set()
        for page in detail['retrieved_pages']:
            matches = mappings[q['task']].get(page_key(page), [])
            assert len(matches) == 1, q['id']
            source_ids.update(matches)
        payload = run.ga.reader_payload(q, r['retrieved_context'])
        saved = tracked(root / 'reader/responses' / (obj(q['id']) + '.json'))
        assert saved['payload'] == payload == mmap[q['id']]
        assert saved['output']['raw_response'] == a['raw_response']
        assert a['request_sha256'] == run.ga.request_sha256(run.images.wire_for(payload, p['max_new_tokens']))
        assert a['returned_model'] == p['expected_model'] and a['seed'] == 42
        assert a['finish_reason'] == 'stop' and not a['refusal'] and a['raw_response'].strip()
        assert r['experiment_signature'] == a['experiment_signature'] == saved['experiment_signature'] == p['experiment_signature']
        evaluated.append(dict(id=q['id'], task=q['task'], point=q['point'], raw_response=a['raw_response'],
            source_ids=sorted(source_ids), context_count=len(detail['retrieved_pages']),
            extra_knowledge_count=sum(len(detail[k]) for k in ('retrieved_user_knowledge', 'retrieved_assistant_knowledge'))))
    api_outputs = rows(root / 'api_responses.jsonl')
    memory_api = {r['output']['response_id']: r for r in api_outputs}
    call_paths = list((root / 'calls').glob('*.json'))
    for path in call_paths:
        c = tracked(path)
        request, out = c['request'], c['output']
        assert obj(request) == path.stem and request['experiment_signature'] == p['experiment_signature']
        assert request['kind'] == 'memory' and request['seed'] == 42
        wire = run.transport.build_request(request['messages'], request['max_new_tokens'], request['temperature'])
        assert out['request_sha256'] == run.ga.request_sha256(wire)
        assert out['returned_model'] == p['expected_model'] and out['finish_reason'] == 'stop'
        assert not out['refusal'] and out['raw_response'].strip()
        response = memory_api[out['response_id']]
        assert response['request'] == wire and response['output']['raw_response'] == out['raw_response']
    events = rows(root / 'call_events.jsonl')
    assert all((root / 'calls' / (event['key'] + '.json')).is_file() for event in events)
    usage, attempts, ids = {}, {}, []
    for stage, folder in [('memory', root), ('answer', root / 'reader')]:
        records = rows(folder / 'api_responses.jsonl')
        outs = [r['output'] for r in records]
        assert all(r['stage'] == stage for r in records)
        assert all(o['requested_model'] == 'gpt-5' and o['returned_model'] == p['expected_model'] for o in outs)
        assert all(o['seed'] == 42 and o['reasoning_effort'] == 'minimal' for o in outs)
        ids.extend(o['response_id'] for o in outs)
        usage[stage] = dict(calls=len(outs), input_tokens=sum(o['input_tokens'] for o in outs),
            completion_tokens=sum(o['generated_tokens'] for o in outs),
            reasoning_tokens=sum(o.get('reasoning_tokens') or 0 for o in outs),
            finish_reasons=dict(Counter(o['finish_reason'] for o in outs)),
            refusals=sum(bool(o['refusal']) for o in outs))
        attempts[stage] = dict(Counter(r['status'] for r in rows(folder / 'api_attempts.jsonl')))
        if stage == 'answer':
            by_response = {o['response_id']: o for o in outs}
            for a in answers:
                assert a['raw_response'] == by_response[a['response_id']]['raw_response']
                assert a['request_sha256'] == by_response[a['response_id']]['request_sha256']
    assert len(ids) == len(set(ids))
    # All generation/context checks above are independent of answer labels.
    gold = {r['id']: r for r in rows(output / 'inputs/gold.jsonl')}
    assert set(gold) == set(rmap)
    for r in evaluated:
        target = gold[r['id']]
        r.update(answer=target['answer'], **run.ga.gallery.score_answer(r['raw_response'], target['answer']))
        gold_ids = set(target['gold_ids'])
        r.update(gold_ids=sorted(gold_ids), gold_count=len(gold_ids), gold_hits=len(gold_ids & set(r['source_ids'])))
    result = dict(status='complete', method='MemoryOS', model=p['expected_model'], seed=42,
        overall=summarize(evaluated),
        by_task={t: summarize([r for r in evaluated if r['task'] == t]) for t in p['tasks']},
        by_type={t: summarize([r for r in evaluated if r['point'] == t]) for t in sorted({r['point'] for r in evaluated})},
        experiment_signature=p['experiment_signature'],
        metric_note='Official normalized open-answer EM/F1/BLEU; source recall credits only original retrieved pages, not generated knowledge.')
    run.write_rows(output / 'scored_questions.jsonl', evaluated)
    with (output / 'scored_questions.csv').open('w') as f:
        fields = ['id', 'task', 'point', 'answer', 'raw_response', 'em', 'f1', 'bleu', 'context_count', 'extra_knowledge_count', 'gold_hits', 'gold_count']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(evaluated)
    dump(output / 'final_results.json', result)
    log = output / 'memory.log'
    warnings = [line for line in log.read_text(errors='replace').splitlines() if
        re.search(r'^(Warning:|Error |Traceback|JSON parsing error|.*: Error|.*: An unexpected error)', line)] if log.exists() else []
    normalizations = rows(root / 'summary_normalization_events.jsonl') if (root / 'summary_normalization_events.jsonl').exists() else []
    dump(output / 'runtime_diagnostics.json', dict(warning_lines=warnings, summary_normalizations=normalizations,
        interpretation='Native parse fallbacks and summary normalization are disclosed; they are not API failure counts.'))
    for path in [output / 'protocol.json', output / 'observations.json', output / 'scored_questions.jsonl',
                 output / 'final_results.json', root / 'retrieval.jsonl', root / 'reader/manifest.jsonl',
                 root / 'reader/responses.jsonl', output / 'runtime_diagnostics.json', Path(__file__)]:
        artifacts[str(path)] = sha(path)
    report = dict(status='passed', questions=1711, source_rounds=3962, scenarios=20,
        all_original_rounds_present_in_native_states=True, native_context_reconstructed=True,
        all_retrieved_pages_mapped_to_original_sources=True, original_query_image_bytes_preserved=True,
        reader_payloads_verified=True, memory_calls=len(call_paths), source_state_counts=state_counts,
        api_usage=usage, api_attempts=attempts, summary_normalization_count=len(normalizations),
        warning_line_count=len(warnings), artifacts_sha256=artifacts,
        experiment_signature=p['experiment_signature'])
    dump(output / 'audit.json', report)
    comparison = []
    for label, summary, source in [('Generative Agents', read(Path(p['ga_control_root']) / 'final_results.json')['overall'], p['ga_control_root']),
                                   ('MemoryOS', result['overall'], str(output))]:
        comparison.append(dict(method=label, **summary, result_root=source))
    dump(output / 'comparison.json', dict(model=p['expected_model'], seed=42, methods=comparison,
        caption_policy=p['caption_policy'], query_policy=p['query_policy'], reader_policy=p['reader_policy'],
        context_policy='Native GA Top10 plus native word limit; native MemoryOS pages and knowledge. Not a fixed Top5 comparison.',
        ga_result_reused=True, memoryos_new_run=True))
    table = ''.join(f'<tr><td>{html.escape(r["method"])}</td><td>{r["correct"]}/{r["questions"]}</td>'
        f'<td>{100*r["em"]:.2f}%</td><td>{100*r["f1"]:.2f}%</td><td>{100*r["source_recall_micro"]:.2f}%</td></tr>' for r in comparison)
    (output / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Mem-Gallery GPT-5</title>'
        '<style>body{font:16px system-ui;margin:36px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}</style>'
        '<h1>Mem-Gallery · GPT-5</h1><p>1711 open-answer questions; identical GPT-5 history captions, official question captions and query-image pixels.</p>'
        '<table><tr><th>Method</th><th>Exact matches</th><th>EM</th><th>F1</th><th>Source recall (native context)</th></tr>'+table+'</table>'
        '<p>Generative Agents reuses the verified completed run. MemoryOS is newly constructed and answered. Native context sizes differ.</p>'
        '<p><a href="comparison.json">Comparison</a> · <a href="audit.json">Audit</a> · <a href="scored_questions.csv">Per-question scores</a></p>')
    dump(output / 'execution_status.json', dict(status='complete', stage='analyze', results_available=True,
        updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='complete', overall=result['overall'], warnings=len(warnings), summary_normalizations=len(normalizations))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=run.OUTPUT)
    audit(parser.parse_args().output.resolve())
