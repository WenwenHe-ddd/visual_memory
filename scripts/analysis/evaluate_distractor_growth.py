"""Add matched, nested random distractors to fixed Brand gold memories."""

import csv
import json
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/object_relation_memory'
PREFIX = OUT / 'Brand_distractor_growth'
VARIANTS = ('natural_caption', 'objects_and_relations',
            'objects_and_relations_no_headers', 'relations_only')
QUERY_SETS = ('mcq_question_only', 'open_question_only')
SEED, TRIALS, MAX_DISTRACTORS = 42, 1000, 20


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choose(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def hypergeom_cdf(total, successes, draws, threshold):
    return sum(choose(successes, k) * choose(total - successes, draws - k)
               for k in range(max(0, draws - (total - successes)), min(draws, successes, threshold) + 1)) / comb(total, draws)


def exact_query_curve(row):
    """Uniform subsets: exact expectations, with stable ranking and raw margins."""
    golds = [r for r in row['ranking'] if r['is_gold']]
    negatives = [r for r in row['ranking'] if not r['is_gold']]
    total, count = len(negatives), len(golds)
    negative_scores = sorted(r['score'] for r in negatives)
    curves = []
    for n in range(MAX_DISTRACTORS + 1):
        hits, ranks, wins = [], [], []
        for target in golds:
            gold_before = sum(r['rank'] < target['rank'] for r in golds)
            neg_before = sum(r['rank'] < target['rank'] for r in negatives)
            hits.append(hypergeom_cdf(total, neg_before, n, 4 - gold_before))
            ranks.append(1 + gold_before + n * neg_before / total)
            not_lower = sum(r['score'] >= target['score'] for r in negatives)
            wins.append(choose(total - not_lower, n) / comb(total, n))
        expected_max = (sum(score * comb(j, n - 1) for j, score in enumerate(negative_scores)
                            if j >= n - 1) / comb(total, n)) if n else None
        curves.append({
            'distractors': n, 'expected_top5_hits': sum(hits),
            'recall_at_5': sum(hits) / count, 'mean_gold_rank': float(np.mean(ranks)),
            'expected_top_distractor_score': expected_max,
            'margin': float(np.mean([r['score'] for r in golds])) - expected_max if n else None,
            'expected_gold_above_all_distractors': sum(wins) if n else None,
        })
    return curves


def describe_trials(values):
    return {'mean': float(np.mean(values)), 'p05': float(np.quantile(values, .05)),
            'p95': float(np.quantile(values, .95)),
            'standard_error_of_sampling_mean': float(np.std(values, ddof=1) / sqrt(TRIALS))}


def save_plot(curves, ceiling):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), layout='constrained')
    labels = ('Natural caption', 'Objects + relations (original headers)',
              'Objects + relations (no headers)', 'Relations only')
    colors = ('#2675A5', '#D16B22', '#8C572A', '#808080')
    styles = ('-', '-', '--', ':')
    panels = [('recall_at_5_micro', 100, 'Gold memories retrieved in Top 5 (%)'),
              ('gold_above_all_distractors_micro', 100, 'Gold memories above every distractor (%)'),
              ('margin_macro_questions', 1, 'Mean gold score - highest distractor score')]
    for vi, (variant, label, color, style) in enumerate(zip(VARIANTS, labels, colors, styles)):
        rows = curves['mcq_question_only'][variant]
        for ax, (metric, factor, ylabel) in zip(axes, panels):
            selected = [r for r in rows if r[metric] is not None]
            x = [r['distractors'] for r in selected]
            ax.plot(x, [r[metric] * factor for r in selected], color=color,
                    linestyle=style, linewidth=2 if vi < 2 else 1.6, label=label)
            if vi < 2:
                ax.fill_between(x, [r['sampling'][metric]['p05'] * factor for r in selected],
                                [r['sampling'][metric]['p95'] * factor for r in selected],
                                color=color, alpha=.10, linewidth=0)
            ax.set(xlabel='Added distractor memories per question', ylabel=ylabel)
            ax.set_xticks([0, 1, 5, 10, 15, 20])
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0].axhline(ceiling * 100, color='#999999', linestyle=':', linewidth=1)
    axes[0].text(.4, ceiling * 100 + 1, 'Gold-only Top5 ceiling: 94.4%', fontsize=9, color='#555555')
    axes[0].set_ylim(45, 100)
    axes[0].set_title('Recall@5', loc='left')
    axes[1].set_ylim(25, 100)
    axes[1].set_title('Resistance to the strongest distractor', loc='left')
    axes[2].axhline(0, color='#555555', linewidth=1)
    axes[2].set_title('Gold advantage (question average)', loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Brand | Fixed 25 questions / 108 query-gold pairs | All gold memories retained\n'
                 'Lines: exact uniform-subset expectations. Bands: 5th-95th percentiles over 1,000 shared addition orders.', fontsize=12)
    fig.savefig(PREFIX.with_suffix('.png'), dpi=180, facecolor='white')
    plt.close(fig)


def main():
    source = OUT / 'Brand_retrieval_details.jsonl'
    previous_path = OUT / 'Brand_retrieval_comparison.json'
    previous = json.loads(previous_path.read_text())
    rows = read_jsonl(source)
    base = [r for r in rows if r['query_set'] == QUERY_SETS[0] and r['variant'] == VARIANTS[0]
            and r['top_distractor'] is not None]
    query_ids = [r['id'] for r in base]
    memory_ids = [r['memory_id'] for r in previous['memory_inputs']]
    memory_images = [r['image'] for r in previous['memory_inputs']]
    mid_to_index = {mid: i for i, mid in enumerate(memory_ids)}
    gold_sets = {r['id']: set(r['gold_ids']) for r in base}
    counts = np.array([len(gold_sets[qid]) for qid in query_ids])
    target_count, question_count = int(sum(counts)), len(query_ids)
    assert (question_count, target_count, len(memory_ids)) == (25, 108, 30)
    assert min(30 - counts) == MAX_DISTRACTORS

    # Verify provenance against current inputs before reusing independent BGE scores.
    mcq = {r['id']: r for r in read_jsonl(ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl')}
    opened = {r['id']: r for r in read_jsonl(ROOT / 'results/natural_observation/queries.jsonl')}
    natural_gold = {r['id']: r['clue_ids'] for r in read_jsonl(ROOT / 'results/natural_observation/gold.jsonl')}
    extraction = {r['cache_name']: r for r in read_jsonl(OUT / 'extractions.jsonl')}
    for m in previous['memory_inputs']:
        e = extraction[m['image']]
        texts = (e['caption'], 'Objects: ' + ', '.join(e['objects']) + '\nRelations:\n' + '\n'.join(e['relations']),
                 ', '.join(e['objects']) + '\n' + '\n'.join(e['relations']), '\n'.join(e['relations']))
        assert [m['texts'][v] for v in VARIANTS] == list(texts)
    for r in rows:
        q = (mcq if r['query_set'] == QUERY_SETS[0] else opened)[r['id']]
        assert r['question'] == q['question']
        assert r['query_input'] == previous['query_prefix'] + q['question']
        assert r['gold_ids'] == natural_gold[r['id']]
        assert set(t['memory_id'] for t in r['ranking']) == set(memory_ids)
        assert [t['rank'] for t in r['ranking']] == list(range(1, 31))
        assert all(t['is_gold'] == (t['memory_id'] in r['gold_ids']) for t in r['ranking'])

    rng = np.random.default_rng(SEED)
    orders = np.full((question_count, TRIALS, 29), -1, dtype=np.int16)
    for qi, qid in enumerate(query_ids):
        negatives = [mid_to_index[mid] for mid in memory_ids if mid not in gold_sets[qid]]
        for trial in range(TRIALS):
            orders[qi, trial, :len(negatives)] = rng.permutation(negatives)
        assert all(set(order[:len(negatives)]) == set(negatives) for order in orders[qi])

    metrics = ('recall_at_5_micro', 'recall_at_5_macro_questions', 'mean_gold_rank_macro_questions',
               'gold_above_all_distractors_micro', 'margin_macro_questions')
    archive = {'distractor_orders': orders, 'query_ids': np.array(query_ids),
               'memory_ids': np.array(memory_ids), 'image_names': np.array(memory_images),
               'gold_counts': counts}
    curves, per_question, full_pool = {}, {}, {}
    validation_errors, recall_errors = [], []
    for query_set in QUERY_SETS:
        curves[query_set], per_question[query_set], full_pool[query_set] = {}, {}, {}
        for variant in VARIANTS:
            selected = {r['id']: r for r in rows if r['query_set'] == query_set and r['variant'] == variant}
            accumulated = {metric: np.zeros((TRIALS, MAX_DISTRACTORS + 1)) for metric in metrics}
            exact_queries, full_hits = [], 0
            for qi, qid in enumerate(query_ids):
                row = selected[qid]
                assert set(row['gold_ids']) == gold_sets[qid]
                exact = exact_query_curve(row)
                exact_queries.append({'id': qid, 'question': row['question'],
                    'gold_images': row['gold_images'], 'available_distractors': 30 - len(row['gold_ids']),
                    'curve': exact})
                golds = [r for r in row['ranking'] if r['is_gold']]
                negatives = [r for r in row['ranking'] if not r['is_gold']]
                ranks = np.empty(30, dtype=int)
                scores = np.empty(30)
                for r in row['ranking']:
                    idx = mid_to_index[r['memory_id']]
                    ranks[idx], scores[idx] = r['rank'], r['score']
                gold_ranks = np.array([r['rank'] for r in golds])
                gold_scores = np.array([r['score'] for r in golds])
                gold_only_ranks = np.array([1 + sum(s['rank'] < r['rank'] for s in golds) for r in golds])
                order = orders[qi, :, :MAX_DISTRACTORS]
                added_above = np.cumsum(ranks[order, None] < gold_ranks, axis=1)
                added_above = np.concatenate([np.zeros((TRIALS, 1, len(golds)), dtype=int), added_above], axis=1)
                target_ranks = gold_only_ranks + added_above
                hits = np.sum(target_ranks <= 5, axis=2)
                assert np.all(np.diff(hits, axis=1) <= 0)
                accumulated['recall_at_5_micro'] += hits / target_count
                accumulated['recall_at_5_macro_questions'] += hits / len(golds) / question_count
                accumulated['mean_gold_rank_macro_questions'] += target_ranks.mean(axis=2) / question_count
                max_distractor = np.maximum.accumulate(scores[order], axis=1)
                accumulated['gold_above_all_distractors_micro'][:, 1:] += (
                    np.sum(gold_scores > max_distractor[:, :, None], axis=2) / target_count)
                accumulated['margin_macro_questions'][:, 1:] += (gold_scores.mean() - max_distractor) / question_count
                full_rank = gold_only_ranks + np.sum(
                    ranks[orders[qi, :, :len(negatives)], None] < gold_ranks, axis=1)
                assert np.all(full_rank == gold_ranks)
                full_hits += int(np.sum(gold_ranks <= 5))
            accumulated['gold_above_all_distractors_micro'][:, 0] = np.nan
            accumulated['margin_macro_questions'][:, 0] = np.nan
            output_rows = []
            for n in range(MAX_DISTRACTORS + 1):
                exact = [q['curve'][n] for q in exact_queries]
                record = {
                    'distractors': n, 'questions': question_count, 'query_gold_pairs': target_count,
                    'expected_top5_hits': sum(q['expected_top5_hits'] for q in exact),
                    'recall_at_5_micro': sum(q['expected_top5_hits'] for q in exact) / target_count,
                    'recall_at_5_macro_questions': float(np.mean([q['recall_at_5'] for q in exact])),
                    'mean_gold_rank_macro_questions': float(np.mean([q['mean_gold_rank'] for q in exact])),
                    'gold_above_all_distractors_micro': sum(q['expected_gold_above_all_distractors'] for q in exact) / target_count if n else None,
                    'margin_macro_questions': float(np.mean([q['margin'] for q in exact])) if n else None,
                    'sampling': {},
                }
                for metric in metrics:
                    if record[metric] is None:
                        record['sampling'][metric] = None
                        continue
                    sampling = describe_trials(accumulated[metric][:, n])
                    record['sampling'][metric] = sampling
                    error = abs(record[metric] - sampling['mean'])
                    validation_errors.append(error / max(sampling['standard_error_of_sampling_mean'], 1e-12))
                    assert error < 6 * sampling['standard_error_of_sampling_mean'] + 1e-10
                    if metric == 'recall_at_5_micro':
                        recall_errors.append(error)
                output_rows.append(record)
            curves[query_set][variant] = output_rows
            per_question[query_set][variant] = exact_queries
            saved_full = previous['summaries'][query_set]['with_distractors'][variant]
            assert full_hits == saved_full['recall']['5']['hits']
            full_pool[query_set][variant] = {'distractor_count': 'all available: 20 to 29 depending on question',
                'recall_at_5_micro': full_hits / target_count,
                'top5_hits': full_hits, 'query_gold_pairs': target_count,
                'margin_macro_questions': saved_full['margin_macro_questions']}
            for metric, values in accumulated.items():
                archive[query_set + '__' + variant + '__' + metric] = values

    paired = {}
    for query_set in QUERY_SETS:
        paired[query_set] = {}
        for variant in VARIANTS[1:]:
            paired[query_set][variant] = []
            for n in range(MAX_DISTRACTORS + 1):
                differences = {}
                for metric in ('recall_at_5_micro', 'margin_macro_questions'):
                    baseline = curves[query_set][VARIANTS[0]][n][metric]
                    if baseline is None:
                        differences[metric] = None
                    else:
                        diff = archive[query_set + '__' + variant + '__' + metric][:, n] - archive[query_set + '__' + VARIANTS[0] + '__' + metric][:, n]
                        differences[metric] = {'exact_difference': curves[query_set][variant][n][metric] - baseline,
                                               'sampling': describe_trials(diff)}
                paired[query_set][variant].append({'distractors': n, **differences})
    ceiling = sum(min(5, g) for g in counts) / target_count
    report = {
        'task': 'Brand_Memory_Test', 'experiment': 'Fixed-query retrieval under increasing random distractor counts',
        'primary_query_set': QUERY_SETS[0], 'sensitivity_query_set': QUERY_SETS[1],
        'questions': question_count, 'query_gold_pairs': target_count,
        'excluded_all_gold_questions': previous['excluded_from_competition'],
        'counts': list(range(MAX_DISTRACTORS + 1)), 'trials': TRIALS, 'seed': SEED,
        'sampling': 'NumPy default_rng PCG64; uniform permutations of non-golds in natural source memory order; prefix subsets nested within trial; identical subsets across all representations and query versions',
        'candidate_pool': 'All gold memories for each question plus exactly n sampled non-golds; no replacement or synthetic memories',
        'gold_definition': previous['gold_definition'],
        'gold_only_top5_ceiling': float(ceiling), 'gold_only_hits': int(sum(min(5, g) for g in counts)),
        'ceiling_explanation': 'Q16 has 10 golds and Q20 has 6: Top5 cannot retrieve every gold even without distractors',
        'score_reuse': 'Reuse verified BGE cosine scores: each question and memory is encoded independently of the candidate pool; subset rankings exactly filter complete rankings',
        'retriever': previous['retriever'], 'query_prefix': previous['query_prefix'],
        'no_new_model_generation': True, 'sources': [str(source), str(previous_path)],
        'exact_expectations': 'Gold rank = 1 + golds ranked above + Hypergeom(D, non-golds ranked above, n); Top5 iff rank <= 5. Expected maximum distractor score uses finite-sample order statistics.',
        'uncertainty': '5th-95th percentiles over 1000 random addition orders conditional on these fixed questions and memories; not population confidence intervals',
        'zero_distractors': 'Margin and above-all-distractors proportion are undefined and stored as null',
        'scope': 'Candidate-count effect on fixed-score retrieval; does not vary shared cue strength, measure reader accuracy, or establish a general causal cue-overload mechanism',
        'validation': {'input_texts_and_questions_match_current_sources': True,
            'identical_candidate_sets_across_variants': True, 'all_sampled_orders_valid': True,
            'nested_addition_ranks_monotone': True, 'full_pool_target_ranks_match_saved_rankings': True,
            'all_full_pool_recalls_match_previous_analysis': True,
            'maximum_sampling_mean_error_in_standard_errors': max(validation_errors),
            'maximum_recall_sampling_mean_error': max(recall_errors)},
        'curves': curves, 'paired_changes_vs_natural': paired,
        'full_pool_reference': full_pool, 'per_question_curves': per_question,
    }
    PREFIX.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(PREFIX.with_suffix('.npz'), **archive)
    with PREFIX.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_set', 'variant', 'distractors', 'questions', 'query_gold_pairs',
            'expected_top5_hits', *metrics, 'recall_sampling_p05', 'recall_sampling_p95'])
        writer.writeheader()
        for query_set in QUERY_SETS:
            for variant in VARIANTS:
                for r in curves[query_set][variant]:
                    writer.writerow({'query_set': query_set, 'variant': variant,
                        **{k: v for k, v in r.items() if k != 'sampling'},
                        'recall_sampling_p05': r['sampling']['recall_at_5_micro']['p05'],
                        'recall_sampling_p95': r['sampling']['recall_at_5_micro']['p95']})
    save_plot(curves, ceiling)
    print(json.dumps({'validation': report['validation'], 'primary_selected_counts': {
        v: [{k: r[k] for k in ['distractors', 'recall_at_5_micro', 'margin_macro_questions', 'gold_above_all_distractors_micro']}
            for r in curves[QUERY_SETS[0]][v] if r['distractors'] in [0, 1, 5, 10, 15, 20]] for v in VARIANTS}}, indent=2))


if __name__ == '__main__':
    main()
