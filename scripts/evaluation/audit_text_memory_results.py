"""Independent provenance/payload checks and paired MemEye result comparison."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from scripts.evaluation import mcq_reader
from scripts.memory.reflexion_memory import ROOT, sha, hash_object, read_rows, write_json
from scripts.memory.text_memory_baselines import METHODS, verify_vendor, retrieval_query, reader_messages


def method_root(output_root, method):
    aligned = output_root / 'top5' / method
    return aligned if method == 'a_mem' and (aligned / 'protocol.json').exists() else output_root / method


def audit(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    snapshot_path = root / 'source_snapshot/manifest.json'
    snapshots = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    for name, digest in protocol['input_hashes'].items():
        if name in snapshots:
            assert snapshots[name]['sha256'] == digest
            assert sha(Path(snapshots[name]['snapshot'])) == digest, name
        else:
            assert sha(Path(name)) == digest, name
    questions = read_rows(root / 'questions.jsonl')
    observations = json.loads((root / 'observations.json').read_text())
    rows = read_rows(root / 'qa_responses.jsonl')
    isolation_path = root / 'storage_isolation_audit.json'
    isolation = json.loads(isolation_path.read_text()) if isolation_path.exists() else None
    if isolation:
        assert isolation['status'] == 'passed' and isolation['new_llm_calls'] == 0
        assert isolation['qa_response_sha256'] == sha(root / 'qa_responses.jsonl')
        for group in ('source_hashes', 'recovered_graph_sha256', 'native_save_snapshot_sha256'):
            for path, digest in isolation[group].items():
                assert sha(Path(path)) == digest, path
        assert isolation['replay_events_sha256'] == sha(root / 'graph_recovery_transport/call_events.jsonl')
    assert hash_object(questions) == protocol['prepared_questions_sha256']
    assert hash_object(observations) == protocol['observations_sha256']
    assert [r['id'] for r in rows] == [q['id'] for q in questions]
    assert len(rows) == len({r['id'] for r in rows})
    total_rounds = 0
    for task in protocol['tasks']:
        stats = json.loads((root / task / 'memory_stats.json').read_text())
        assert stats['stored_rounds'] == len(observations[task])
        total_rounds += stats['stored_rounds']
    for q, row in zip(questions, rows):
        for key in ('question', 'options', 'gold_choice', 'gold_ids', 'query_image_captions'):
            assert row[key] == q[key], (q['id'], key)
        assert row['retrieval_query'] == retrieval_query(q)
        assert row['reader_messages_sha256'] == hash_object(reader_messages(q, row['retrieved_context'], protocol))
        call = json.loads((root / 'calls' / (row['call_key'] + '.json')).read_text())
        assert call['request']['messages'] == reader_messages(q, row['retrieved_context'], protocol)
        assert call['request']['kind'] == 'answer'
        assert call['output']['raw_response'] == row['raw_response']
        assert row['correct'] == (row['gold_choice'] == row['predicted_choice'])
    calls = [json.loads(p.read_text()) for p in (root / 'calls').glob('*.json')]
    for path in (root / 'calls').glob('*.json'):
        assert path.stem == hash_object(json.loads(path.read_text())['request'])
    for call in calls:
        assert call['request']['experiment_signature'] == protocol['experiment_signature']
        assert all(isinstance(m['content'], str) for m in call['request']['messages'])
    structured = [c for c in calls if c['request']['response_format']]
    events = read_rows(root / 'call_events.jsonl')
    text = (root / 'run.log').read_text() if (root / 'run.log').exists() else ''
    # Keep failures visible, including those upstream catches and converts to fallbacks.
    keyword_matches = [line for line in text.splitlines() if re.search(r'error|failed|fallback|could not', line, re.I)
                and 'Qwen3VLVideoProcessorInitKwargs' not in line and not line.startswith('{"mode":')]
    # A-Mem prints full memory contents, including phrases like "spelling error".
    # Retain raw matches, but identify diagnostics by its actual logging prefixes.
    warnings = keyword_matches
    if protocol['mode'] == 'a_mem':
        warnings = [line for line in keyword_matches if re.match(
            r'^(JSON parsing error|Error analyzing content|SGLang completion error|LiteLLM completion error|Traceback|(?:WARNING|ERROR)[: ])', line)]
    result = {'status': 'passed', 'questions': len(rows), 'stored_rounds': total_rounds,
              'verified_source_hashes': len(protocol['input_hashes']), 'unique_llm_calls': len(calls),
              'structured_calls': len(structured),
              'invalid_structured_calls': sum(c['output'].get('structured_output_valid') is False for c in structured),
              'memory_generation_token_limit_hits': sum(c['output']['hit_token_limit'] for c in calls if c['request']['kind'] == 'memory'),
              'token_limit_hits_by_budget': dict(Counter(str(c['request']['max_new_tokens']) for c in calls if c['output']['hit_token_limit'])),
              'requested_temperatures': dict(Counter(str(e['requested_temperature']) for e in events)),
              'request_events': len(events), 'cached_request_events': sum(e['reused'] for e in events),
              'upstream_warning_lines': warnings,
              'log_keyword_matches': keyword_matches,
              'note': 'Provenance/payload pass does not imply upstream formatting/fallback success; inspect counts and warnings.'}
    if protocol.get('memory_replay'):
        manifest = json.loads((root / 'memory_source/replay_manifest.json').read_text())
        imported = {c['source_key']: c for c in manifest['calls']}
        for call in calls:
            if call['request']['kind'] == 'answer':
                assert 'replay_provenance' not in call
                continue
            provenance = call['replay_provenance']
            original = {**call['request'], 'experiment_signature': manifest['source_experiment_signature']}
            key = hash_object(original)
            assert key == provenance['source_key']
            assert provenance['source_sha256'] == imported[key]['source_sha256']
            assert hash_object(call['output']) == imported[key]['output_sha256']
        assert all(e['reused'] for e in events if e['kind'] == 'memory')
        assert all(not e['reused'] for e in events if e['kind'] == 'answer')
        for task in protocol['tasks']:
            before = json.loads((root / 'memory_source' / task / 'final_memory.json').read_text())
            after = json.loads((root / task / 'final_memory.json').read_text())
            assert list(before) == list(after)
            for mid in before:
                assert {k: v for k, v in before[mid].items() if k != 'last_accessed'} == {
                    k: v for k, v in after[mid].items() if k != 'last_accessed'}, (task, mid)
        result['memory_replay'] = {'source_nodes_preserved': total_rounds,
                                   'replayed_memory_and_keyword_calls': len(imported),
                                   'fresh_answer_calls': sum(c['request']['kind'] == 'answer' for c in calls),
                                   'note': 'Memory diagnostics below describe replayed original construction, not fresh memory generation.'}
    diagnostics = {}
    for task in protocol['tasks']:
        directory = root / task
        info = {'input_rounds': len(observations[task])}
        if protocol['mode'] == 'gen_agents':
            memories = json.loads((directory / 'final_memory.json').read_text())
            info.update(final_memories=len(memories), reflection_memories=sum(bool(m.get('source')) for m in memories))
        elif protocol['mode'] == 'a_mem':
            memories = json.loads((directory / 'final_memory.json').read_text())
            if protocol.get('context_memory_limit') is not None:
                limit = protocol['context_memory_limit']
                for row in (r for r in rows if r['task'] == task):
                    detail = row['retrieval_detail']
                    ids = detail['context_memory_ids']
                    assert len(ids) == len(set(ids)) == detail['context_memory_count'] == limit
                    assert detail['context_memory_limit'] == limit
                    assert detail['primary_top_k'] == protocol['retrieval_top_k']
                    expected = ''.join('talk start time:' + memories[mid]['timestamp']
                        + 'memory content: ' + memories[mid]['content']
                        + 'memory context: ' + memories[mid]['context']
                        + 'memory keywords: ' + str(memories[mid]['keywords'])
                        + 'memory tags: ' + str(memories[mid]['tags']) + '\n' for mid in ids)
                    assert row['retrieved_context'] == expected, row['id']
                info['verified_context_memory_limit'] = limit
            links = [link for m in memories.values() for link in m['links']]
            # Mirror upstream neighbor lookup: string keys and integer positions.
            # Numeric strings are not automatically converted to positions.
            resolved = sum((link in memories) if isinstance(link, str) else
                           (isinstance(link, int) and -len(memories) <= link < len(memories))
                           for link in links)
            info.update(final_memories=len(memories), notes_with_links=sum(bool(m['links']) for m in memories.values()),
                        link_count=len(links), resolved_links=resolved, unresolved_links=len(links)-resolved,
                        link_types=dict(Counter(type(link).__name__ for link in links)),
                        notes_without_keywords=sum(not m['keywords'] for m in memories.values()))
        elif protocol['mode'] == 'memoryos':
            runtime = directory / 'runtime'
            short = json.loads(next(runtime.rglob('short_term.json')).read_text())
            mid = json.loads(next(runtime.rglob('mid_term.json')).read_text())
            long_user = json.loads(next(runtime.rglob('long_term_user.json')).read_text())
            long_assistant = json.loads(next(runtime.rglob('long_term_assistant.json')).read_text())
            info.update(short_term_pages=len(short), mid_term_segments=len(mid['sessions']),
                        mid_term_pages=sum(len(s.get('details', [])) for s in mid['sessions'].values()),
                        user_knowledge=len(long_user.get('knowledge_base', [])),
                        assistant_knowledge=len(long_assistant.get('assistant_knowledge', [])))
        else:
            maus = [m for path in (directory / 'runtime').rglob('mau_*.jsonl') for m in read_rows(path)]
            unique_maus = len({m['id'] for m in maus})
            assert unique_maus == len(observations[task]), (task, 'missing SM(T) memory units')
            graph_path = directory / 'runtime/knowledge_graph'
            graph_origin = 'original per-task saved graph'
            if not graph_path.exists():
                assert isolation is not None, 'legacy shared graph requires an isolation/reconstruction audit'
                graph_path = root / 'graph_recovery' / task
                graph_origin = 'reconstructed from exact cached outputs; see storage_isolation_audit.json'
            info.update(serialized_memory_units=len(maus), unique_memory_units=unique_maus,
                        graph_entities=len(read_rows(graph_path / 'entities.jsonl')),
                        graph_relations=len(read_rows(graph_path / 'relations.jsonl')),
                        graph_snapshot_origin=graph_origin)
        subset = [r for r in rows if r['task'] == task]
        info.update(empty_retrieval_contexts=sum(not r['retrieved_context'].strip() for r in subset),
                    mean_answer_input_tokens=sum(r['input_tokens'] for r in subset)/len(subset))
        diagnostics[task] = info
    result['memory_diagnostics'] = diagnostics
    write_json(root / 'audit.json', result)
    return result


def compare(output_root):
    baseline_path = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    reference_protocol = json.loads((baseline_path.parent / 'protocol.json').read_text())
    qs = read_rows(baseline_path)
    byid = {q['id']: q for q in qs}
    sources = {
        'FC(T)': output_root / 'fc_t', 'Reflexion (gold feedback)': output_root / 'reflexion',
        'Natural Top5': ROOT / 'results/mcq_top5_10_ablation/top5/caption',
        'Natural Top15': ROOT / 'results/mcq_top15_caption_qa',
        **{m: method_root(output_root, m) for m in METHODS}}
    table, aligned = [], {}
    for name, source in sources.items():
        response_path = source / 'qa_responses.jsonl'
        if not response_path.exists():
            continue
        rows = {r['id']: r for r in read_rows(response_path)}
        assert len(rows) == len(qs) and rows.keys() == byid.keys(), name
        method_protocol = json.loads((source / 'protocol.json').read_text())
        for key in ('reader_model', 'reader_model_path', 'reader_system_prompt', 'seed', 'do_sample', 'max_new_tokens'):
            assert method_protocol[key] == reference_protocol[key], (name, key)
        manifest_path = source / ('questions.jsonl' if name in METHODS else 'qa_manifest.jsonl')
        manifest = {r['id']: r for r in read_rows(manifest_path)}
        for q in qs:
            for key in ('question','options','gold_choice'):
                assert manifest[q['id']][key] == q[key], (name, q['id'], key)
        values = [int(rows[q['id']]['predicted_choice'] == q['gold_choice']) for q in qs]
        aligned[name] = values
        cells = {}
        for y in ('Y1','Y2','Y3'):
            for x in ('X1','X2','X3','X4'):
                idx = [i for i,q in enumerate(qs) if q['x'] == x and q['y'] == y]
                cells[f'{y}/{x}'] = sum(values[i] for i in idx)/len(idx)
        fc = aligned.get('FC(T)')
        internal_audit = json.loads((source / 'audit.json').read_text()) if name in METHODS else {}
        table.append({'method':name,'n':len(qs),'correct':sum(values),'accuracy':sum(values)/len(qs),
            'retrieval_top_k': method_protocol.get('retrieval_top_k', method_protocol.get('method_defaults', {}).get('retrieve_k')),
            'macro_12_cells':sum(cells.values())/12,'by_cell':cells,
            'format_errors':sum(rows[q['id']]['predicted_choice'] not in ('A','B','C','D') for q in qs),
            'wins_vs_fc':sum(a and not b for a,b in zip(values,fc)) if fc else None,
            'losses_vs_fc':sum(b and not a for a,b in zip(values,fc)) if fc else None,
            'memory_token_limit_hits': internal_audit.get('memory_generation_token_limit_hits'),
            'invalid_structured_calls': internal_audit.get('invalid_structured_calls'),
            'internal_warning_count': len(internal_audit['upstream_warning_lines']) if internal_audit else None,
            'response_sha256':sha(response_path),'manifest_sha256':sha(manifest_path)})
    result = {'questions':len(qs),'comparison':table,'interpretation':
        'Shared Qwen/371 questions/fixed options; memory input, prompts, retrieval backbone, query image policy and Reflexion gold feedback differ. Not Table 2 reproduction.'}
    write_json(output_root / 'text_memory_comparison.json', result)
    with (output_root / 'text_memory_comparison.csv').open('w') as f:
        fields = ['method','retrieval_top_k','n','correct','accuracy','macro_12_cells','format_errors','wins_vs_fc','losses_vs_fc',
                  'memory_token_limit_hits','invalid_structured_calls','internal_warning_count']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(table)
    with (output_root / 'text_memory_paired_questions.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','task','x','y',*aligned])
        writer.writerows([[q['id'],q['task'],q['x'],q['y'],*[v[i] for v in aligned.values()]] for i,q in enumerate(qs)])
    print(json.dumps([{k:r[k] for k in ('method','correct','accuracy','macro_12_cells')} for r in table],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/visual_memory_pipeline')
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    verify_vendor()
    if args.method:
        print(json.dumps(audit(args.output_root / args.method), indent=2))
    else:
        for method in METHODS:
            path = method_root(args.output_root, method) / 'summary.json'
            if path.exists() and json.loads(path.read_text())['status'] == 'complete':
                audit(path.parent)
        compare(args.output_root)
