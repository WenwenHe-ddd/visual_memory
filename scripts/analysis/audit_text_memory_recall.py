"""Audit gold source-round recall of four saved MemEye answer contexts.

Count identifiable original dialogue rounds, not inferred factual coverage of
reflections/knowledge summaries. Ambiguous repeated text is tolerated only when
none of its candidate rounds is a benchmark gold round; its candidates remain
in the audit but are never optimistically credited.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from statistics import mean

from scripts.memory import text_memory_baselines as baseline
from scripts.memory.reflexion_memory import ROOT, read_rows, write_json, write_rows, sha, hash_object

METHODS = {
    'Generative Agents': 'gen_agents', 'MemoryOS': 'memoryos',
    'SimpleMem (context cap 5)': 'top5/simplemem_context5',
    'A-Mem (context 5)': 'top5/a_mem_context5',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/visual_memory_pipeline')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/text_memory_recall_20261025')
    args = parser.parse_args()
    canonical_path = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    canonical = {q['id']: q for q in read_rows(canonical_path)}
    assert len(canonical) == 371 and all(q['gold_ids'] for q in canonical.values())
    all_gold = set().union(*(set(q['gold_ids']) for q in canonical.values()))
    hashes = {str(canonical_path): sha(canonical_path)}
    all_rows, summaries, diagnostics = [], [], {}
    def tracked_json(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    def tracked_rows(path):
        hashes[str(path)] = sha(path)
        return read_rows(path)
    for name, subdir in METHODS.items():
        root = args.results / subdir
        protocol = tracked_json(root / 'protocol.json')
        rows = tracked_rows(root / 'qa_responses.jsonl')
        assert len(rows) == len({r['id'] for r in rows}) == 371
        assert {r['id'] for r in rows} == set(canonical)
        notes, maps = {}, {}
        stats = Counter()
        mode = protocol['mode']
        if mode in ('a_mem', 'gen_agents'):
            notes = {task: tracked_json(root / task / 'final_memory.json') for task in protocol['tasks']}
        if mode == 'memoryos':
            helper = baseline.helpers(mode)
            observations = tracked_json(root / 'observations.json')
            for task, turns in observations.items():
                mapping = defaultdict(list)
                for turn in turns:
                    key = (helper['_round_user_with_captions'](turn).strip(), turn['assistant'].strip(), turn['timestamp'].strip())
                    mapping[key].append(task + '/' + turn['id'])
                maps[task] = mapping
        if mode == 'simplemem':
            from scripts.evaluation.simplemem_context_ablation import formatter
            simple_render = formatter()
        evaluated = []
        for row in rows:
            q = canonical[row['id']]
            for key in ('question', 'options', 'gold_choice', 'gold_ids', 'task'):
                assert row[key] == q[key], (name, row['id'], key)
            assert row['experiment_signature'] == protocol['experiment_signature']
            messages = baseline.reader_messages(row, row['retrieved_context'], protocol)
            assert hash_object(messages) == row['reader_messages_sha256']
            call_path = root / 'calls' / (row['call_key'] + '.json')
            call = tracked_json(call_path)
            assert hash_object(call['request']) == row['call_key']
            assert call['request']['messages'] == messages and call['request']['kind'] == 'answer'
            assert call['request']['experiment_signature'] == protocol['experiment_signature']
            assert call['output']['raw_response'] == row['raw_response']
            ids, ambiguous = [], []
            detail = row['retrieval_detail']
            reflection_count = knowledge_count = 0
            if mode == 'a_mem':
                selected = detail['context_memory_ids']
                assert len(selected) == len(set(selected)) == 5
                def block(n):
                    return ('talk start time:' + n['timestamp'] + 'memory content: ' + n['content']
                            + 'memory context: ' + n['context'] + 'memory keywords: ' + str(n['keywords'])
                            + 'memory tags: ' + str(n['tags']) + '\n')
                assert ''.join(block(notes[row['task']][mid]) for mid in selected) == row['retrieved_context']
                ids = [row['task'] + '/' + mid for mid in selected]
                count = len(selected)
            elif mode == 'simplemem':
                assert simple_render(detail)[0] == row['retrieved_context']
                items = detail['items']
                assert len(items) in (0, 5) and len({i['id'] for i in items}) == len(items)
                for item in items:
                    round_ids = [tag.removeprefix('round_id:') for tag in item['tags'] if tag.startswith('round_id:')]
                    assert len(round_ids) == 1
                    ids.append(row['task'] + '/' + round_ids[0])
                count = len(items)
            elif mode == 'gen_agents':
                context = row['retrieved_context']
                boundaries = list(re.finditer(r'(?m)^\[(\d+)\] ', context))
                assert len(boundaries) == 10
                for i, boundary in enumerate(boundaries):
                    assert int(boundary.group(1)) == i
                    text = context[boundary.end():boundaries[i+1].start() if i+1<len(boundaries) else len(context)].strip()
                    found = [n for n in notes[row['task']] if n['text'].strip() == text]
                    assert found, (name, row['id'], 'unmatched/truncated block', i)
                    if len(found) > 1:
                        assert all(n.get('source') is False for n in found)
                        candidates = [row['task'] + '/' + n['dialogue_id'] for n in found]
                        assert not set(candidates) & all_gold, (row['id'], 'ambiguous gold mapping')
                        ambiguous.append(candidates)
                    elif found[0].get('source') is False:
                        ids.append(row['task'] + '/' + found[0]['dialogue_id'])
                    else:
                        reflection_count += 1
                count = len(boundaries)
            else:
                assert helper['_format_retrieval_context'](detail) == row['retrieved_context']
                for page in detail['retrieved_pages']:
                    key = (page['user_input'].strip(), page['agent_response'].strip(), page['timestamp'].strip())
                    candidates = maps[row['task']].get(key, [])
                    assert candidates, (row['id'], page['page_id'], 'unmatched page')
                    if len(candidates) > 1:
                        assert not set(candidates) & all_gold, (row['id'], 'ambiguous gold mapping')
                        ambiguous.append(candidates)
                    else:
                        ids.append(candidates[0])
                count = len(detail['retrieved_pages'])
                knowledge_count = sum(len(detail[k]) for k in ('retrieved_user_knowledge', 'retrieved_assistant_knowledge'))
            unique = set(ids)
            gold = set(q['gold_ids'])
            hits = unique & gold
            metrics = {'method': name, 'question_id': row['id'], 'task': row['task'],
                'context_memory_entries': count, 'extra_knowledge_entries': knowledge_count,
                'reflection_entries': reflection_count, 'ambiguous_non_gold_blocks': len(ambiguous),
                'gold_count': len(gold), 'gold_hits': len(hits), 'recall': len(hits)/len(gold),
                'any_hit': bool(hits), 'all_hit': gold <= unique,
                'qa_correct': row['predicted_choice'] == q['gold_choice'],
                'retrieved_round_ids': sorted(unique), 'gold_ids': sorted(gold), 'hit_ids': sorted(hits),
                'ambiguous_non_gold_candidates': ambiguous}
            evaluated.append(metrics)
            stats.update({'reflection_blocks_not_credited': reflection_count,
                          'knowledge_blocks_not_credited': knowledge_count,
                          'ambiguous_non_gold_blocks': len(ambiguous)})
        def summarize(group):
            n = len(group)
            return {'questions': n, 'gold_references': sum(r['gold_count'] for r in group),
                'gold_hits': sum(r['gold_hits'] for r in group),
                'recall_micro': sum(r['gold_hits'] for r in group)/sum(r['gold_count'] for r in group),
                'recall_macro': mean(r['recall'] for r in group),
                'questions_any_hit': sum(r['any_hit'] for r in group), 'any_hit_rate': mean(r['any_hit'] for r in group),
                'questions_all_hit': sum(r['all_hit'] for r in group), 'all_hit_rate': mean(r['all_hit'] for r in group),
                'qa_correct': sum(r['qa_correct'] for r in group), 'qa_accuracy': mean(r['qa_correct'] for r in group),
                'context_entries_min': min(r['context_memory_entries'] for r in group),
                'context_entries_max': max(r['context_memory_entries'] for r in group),
                'context_entries_mean': mean(r['context_memory_entries'] for r in group)}
        summary = {'method': name, **summarize(evaluated), 'source': str(root),
                   'by_task': {task: summarize([r for r in evaluated if r['task'] == task]) for task in protocol['tasks']}}
        summaries.append(summary)
        diagnostics[name] = {**dict(stats), 'context_entry_counts': dict(Counter(r['context_memory_entries'] for r in evaluated)),
                              'empty_context_question_ids': [r['question_id'] for r in evaluated if not r['context_memory_entries']]}
        all_rows.extend(evaluated)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / 'per_question.jsonl', all_rows)
    for filename, rows in [('recall_summary.csv', [{k:v for k,v in s.items() if k!='by_task'} for s in summaries]), ('per_question.csv', all_rows)]:
        with (args.output / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else v for k,v in row.items()})
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    write_json(args.output / 'audit.json', {'status': 'passed',
        'metric_unit': 'gold source dialogue-round identity, not semantic fact retention',
        'primary_metric': 'micro recall = total gold round hits / total gold round references',
        'secondary_metric': 'macro recall = mean per-question hit_count/gold_count',
        'reflection_and_knowledge_provenance_not_automatically_credited': True,
        'ambiguous_text_matches_all_verified_non_gold': True,
        'GA_and_MemoryOS_not_restricted_to_top5': True,
        'gold_used_only_after_context_mapping': True,
        'comparisons': summaries, 'diagnostics': diagnostics, 'input_hashes': hashes})
    print(json.dumps([{k:v for k,v in s.items() if k!='by_task'} for s in summaries], indent=2))


if __name__ == '__main__':
    main()
