"""Post-hoc diagnosis from frozen rankings/answers; no model or API calls.

The fixed-group-quota counterfactual measures retrieval only. It does not
estimate QA accuracy, and is not a replacement for a matched reader experiment.
"""
import collections
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import mem_gallery_zscore as run


def main():
    root, source = run.DEFAULT_OUTPUT, run.SOURCE
    run.verify(root)
    completion = run.read(root / 'completion_manifest.json')
    run.verify_hashes(completion['artifacts_sha256'])
    output = root / 'master_drop_diagnosis'
    output.mkdir(exist_ok=True)
    paths = [source / 'reader/paired_results.json', root / 'reader/paired_results.json',
             root / 'weighting_diagnostics.jsonl', root / 'inputs/questions.jsonl',
             root / 'inputs/memories.jsonl', Path(__file__).resolve()]
    hashes = {str(p): run.sha(p) for p in paths}
    before = {r['id']: r for r in run.read(paths[0])}
    after = {r['id']: r for r in run.read(paths[1])}
    questions = {r['id']: r for r in run.lines(paths[3])}
    memories = {r['id']: r for r in run.lines(paths[4])}
    assert len(before) == len(after) == len(questions) == 1711
    graph_maps = {}
    for task in {m['task'] for m in memories.values()}:
        graph_path = source / 'offline/memory' / task / 'graph.json'
        hashes[str(graph_path)] = run.sha(graph_path)
        graph_maps[task] = {n['id']: n['master_id'] for n in run.read(graph_path)['image_nodes']}
    scope_counts = collections.defaultdict(collections.Counter)
    hit_counts = collections.defaultdict(collections.Counter)
    type_counts = collections.defaultdict(collections.Counter)
    totals = collections.Counter()
    rows, cases, ratios = [], [], []
    for d in run.lines(paths[2]):
        qid = d['id']; q = questions[qid]
        old = before[qid]['master_answer_query']
        new = after[qid]['master_answer_query+zscore']
        ids = d['memory_ids']; indices = {mid: i for i, mid in enumerate(ids)}
        gold = set(after[qid]['gold_ids'])
        old_set, new_set = set(old['top5_memory_ids']), set(new['top5_memory_ids'])
        lost_gold, gained_gold = gold & (old_set-new_set), gold & (new_set-old_set)
        scope = ('same_order' if old['top5_memory_ids'] == new['top5_memory_ids']
                 else 'order_only' if old_set == new_set else 'set_changed')
        outcome = ('lost' if old['em'] > new['em'] else 'gained' if old['em'] < new['em']
                   else 'both_right' if old['em'] else 'both_wrong')
        hits = ('fewer_gold' if new['gold_hits'] < old['gold_hits'] else
                'more_gold' if new['gold_hits'] > old['gold_hits'] else 'equal_gold')
        scope_counts[scope][outcome] += 1
        hit_counts[hits][outcome] += 1
        type_counts[q['point']][outcome] += 1
        totals['old_visual_top5'] += sum(memories[i]['is_visual'] for i in old_set)
        totals['new_visual_top5'] += sum(memories[i]['is_visual'] for i in new_set)
        for mid in gold:
            kind = 'visual' if memories[mid]['is_visual'] else 'text'
            totals[kind+'_gold_total'] += 1
            totals[kind+'_old_hits'] += mid in old_set
            totals[kind+'_new_hits'] += mid in new_set
        for mid in lost_gold:
            totals['removed_gold_'+('visual' if memories[mid]['is_visual'] else 'text')] += 1
        for mid in gained_gold:
            totals['added_gold_'+('visual' if memories[mid]['is_visual'] else 'text')] += 1
        if outcome == 'lost':
            totals['em_regressions_with_some_gold_removed'] += bool(lost_gold)
            totals['em_regressions_lost_all_gold'] += bool(old['gold_hits'] and not new['gold_hits'])
            totals['em_regressions_still_contain_gold'] += new['gold_hits'] > 0
            totals['em_regressions_no_gold_before_or_after'] += not (old['gold_hits'] or new['gold_hits'])

        stats = d['conditions']['master_answer_query+zscore']
        raw, weighted = np.array(stats['raw']), np.array(stats['weighted'])
        group = {i: graph_maps[q['task']][memories[mid]['node_id']] if memories[mid]['is_visual'] else '__text_only__'
                 for i, mid in enumerate(ids)}
        original = [indices[mid] for mid in old['top5_memory_ids']]
        current = [indices[mid] for mid in new['top5_memory_ids']]
        quota = collections.Counter(group[i] for i in original)
        selected = []
        for key, count in quota.items():
            candidates = [i for i in range(len(ids)) if group[i] == key]
            selected.extend(sorted(candidates, key=lambda i: (-weighted[i], i))[:count])
        selected = sorted(selected, key=lambda i: (-weighted[i], i))
        quota_hits = len(gold & {ids[i] for i in selected})
        totals['quota_counterfactual_gold_hits'] += quota_hits
        totals['original_gold_hits'] += old['gold_hits']
        totals['weighted_gold_hits'] += new['gold_hits']
        totals['quota_counterfactual_changed_sets'] += set(selected) != set(original)
        totals['global_weighting_changed_group_quotas'] += quota != collections.Counter(group[i] for i in current)
        removed, added = set(original)-set(current), set(current)-set(original)
        for i in removed:
            totals['removed_'+('visual' if memories[ids[i]]['is_visual'] else 'text')] += 1
        for i in added:
            totals['added_'+('visual' if memories[ids[i]]['is_visual'] else 'text')] += 1
        for i in removed:
            for j in added:
                margin = raw[i]-raw[j]
                correction_gap = (weighted[j]-raw[j])-(weighted[i]-raw[i])
                assert margin >= -1e-12 and correction_gap >= margin-1e-12
                totals['boundary_pairs'] += 1
                totals['same_group_boundary_pairs' if group[i] == group[j] else 'cross_group_boundary_pairs'] += 1
                if margin > 1e-12:
                    ratios.append(correction_gap/margin)
        row = {'id': qid, 'point': q['point'], 'scope': scope, 'outcome': outcome,
               'old_em': old['em'], 'new_em': new['em'], 'old_f1': old['f1'], 'new_f1': new['f1'],
               'old_gold_hits': old['gold_hits'], 'new_gold_hits': new['gold_hits'],
               'quota_counterfactual_gold_hits': quota_hits, 'gold_count': len(gold),
               'removed_gold_count': len(lost_gold), 'added_gold_count': len(gained_gold)}
        rows.append(row)
        if outcome in ('lost', 'gained'):
            raw_ranks = {i: rank for rank, i in enumerate(np.argsort(-raw, kind='stable'), 1)}
            new_ranks = {i: rank for rank, i in enumerate(np.argsort(-weighted, kind='stable'), 1)}
            cases.append({**row, 'question': q['question'], 'answer': after[qid]['answer'],
                          'old_answer': old['raw_response'], 'new_answer': new['raw_response'],
                          'old_top5': old['top5_memory_ids'], 'new_top5': new['top5_memory_ids'],
                          'gold_ids': sorted(gold),
                          'changed_memories': [dict(memory_id=ids[i], is_visual=memories[ids[i]]['is_visual'],
                              master_id=group[i], is_gold=ids[i] in gold, raw=float(raw[i]),
                              correction=float(weighted[i]-raw[i]), weighted=float(weighted[i]),
                              old_rank=raw_ranks[i], new_rank=new_ranks[i], reader_text=memories[ids[i]]['reader_text'])
                              for i in sorted(removed | added)]})
    assert sum(r['old_em'] for r in rows) == 798
    assert sum(r['new_em'] for r in rows) == 756
    total_gold = sum(r['gold_count'] for r in rows)
    assert total_gold == 4109
    summary = {'questions': 1711, 'lambda': .03, 'alpha': 1., 'net_correct_change': -42,
               'em_changes_by_top5_scope': dict(scope_counts), 'em_changes_by_gold_hits': dict(hit_counts),
               'em_changes_by_question_type': dict(type_counts), 'counts': totals,
               'retrieval_only_counterfactual': {
                   'definition': 'Preserve each original Top-5 group quota (each visual master, plus unchanged text-only pool); use weighted ranking within each group.',
                   'original_recall_at_5_micro': totals['original_gold_hits']/total_gold,
                   'quota_preserved_recall_at_5_micro': totals['quota_counterfactual_gold_hits']/total_gold,
                   'global_weighted_recall_at_5_micro': totals['weighted_gold_hits']/total_gold,
                   'qa_rerun': False, 'caution': 'This is a diagnostic intervention on retrieval only, not a causal decomposition of QA errors.'},
               'boundary_pair_correction_gap_to_raw_gap_quantiles': dict(zip(['median', 'p90'], np.quantile(ratios, [.5,.9]).tolist())),
               'em_caution': 'EM regressions include semantically correct answers with extra wording. Report EM flips, not verified factual-error counts.',
               'source_sha256': hashes}
    run.verify_hashes(hashes)
    run.dump(output / 'summary.json', summary)
    run.dump(output / 'all_em_flip_cases.json', cases)
    with (output / 'per_question.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
