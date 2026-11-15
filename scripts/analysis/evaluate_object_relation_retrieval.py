"""Compare fixed Brand queries against caption and object/relation memories."""

import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.visual_memory_pipeline import rank_task

OUT = ROOT / 'results/object_relation_memory'
NATURAL = ROOT / 'results/natural_observation'
TASK = 'Brand_Memory_Test'
VARIANTS = ('natural_caption', 'objects_and_relations',
            'objects_and_relations_no_headers', 'relations_only')
LABELS = ('Natural caption', 'Objects + relations\n(original headers)',
          'Objects + relations\n(no headers)', 'Relations only')
COLORS = ('#2874A6', '#92989B', '#D17A22', '#37816C')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def avg(values):
    return float(np.mean(values))


def summarize(rows):
    targets = [t for row in rows for t in row['targets']]
    competing = [row for row in rows if row['top_distractor'] is not None]
    margins = [t['margin'] for row in competing for t in row['targets']]
    result = {
        'questions': len(rows), 'query_gold_pairs': len(targets),
        'mean_gold_rank_macro_questions': avg([r['mean_gold_rank'] for r in rows]),
        'mean_gold_rank_micro_targets': avg([t['rank'] for t in targets]),
        'mean_gold_score_macro_questions': avg([r['mean_gold_score'] for r in rows]),
        'questions_with_distractors': len(competing),
        'margin_macro_questions': avg([r['mean_target_margin'] for r in competing]),
        'margin_micro_targets': avg(margins),
        'mean_top_distractor_score_macro_questions': avg([
            r['top_distractor']['score'] for r in competing]),
        'targets_above_strongest_distractor': sum(m > 0 for m in margins),
        'targets_with_distractors': len(margins),
        'gold_vs_distractor_pairwise_win_rate_macro_questions': avg([
            r['pairwise_win_rate'] for r in competing]),
        'recall': {},
    }
    for k in (1, 5, 10):
        hits = sum(t['rank'] <= k for t in targets)
        result['recall'][str(k)] = {
            'hits': hits, 'targets': len(targets), 'micro': hits / len(targets),
            'macro': avg([r['recall'][str(k)] for r in rows]),
            'any_gold_questions': sum(any(t['rank'] <= k for t in r['targets']) for r in rows),
            'all_gold_questions': sum(all(t['rank'] <= k for t in r['targets']) for r in rows),
        }
    return result


def plot_results(details, summaries):
    rows = {v: [r for r in details if r['query_set'] == 'mcq_question_only'
                and r['variant'] == v and r['top_distractor'] is not None] for v in VARIANTS}
    summary = summaries['mcq_question_only']['with_distractors']
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), layout='constrained')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    x = np.arange(len(VARIANTS))
    margins = [summary[v]['margin_macro_questions'] for v in VARIANTS]
    axes[0].bar(x, margins, color=COLORS, width=.65)
    axes[0].axhline(0, color='#333333', linewidth=1)
    for i, value in enumerate(margins):
        axes[0].text(i, value - .002, f'{value:+.4f}', ha='center', va='top')
    axes[0].set_ylim(min(margins) * 1.3, .012)
    axes[0].set_ylabel('Mean gold score - highest distractor score')
    axes[0].set_title('Gold advantage (higher is better)\nMean within question, then across 25 questions', fontsize=11)
    recall = [summary[v]['recall']['5'] for v in VARIANTS]
    axes[1].bar(x, [r['micro'] * 100 for r in recall], color=COLORS, width=.65)
    for i, r in enumerate(recall):
        axes[1].text(i, r['micro'] * 100 + 1.5,
                     f"{r['micro']:.1%}\n{r['hits']}/{r['targets']}", ha='center')
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Gold memories retrieved in Top 5 (%)')
    axes[1].set_title('Recall@5 (higher is better)\n108 query-gold pairs in the same 25 questions', fontsize=11)
    for ax in axes[:2]:
        ax.set_xticks(x, LABELS, rotation=20, ha='right', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    baseline = rows['natural_caption']
    changed = rows['objects_and_relations_no_headers']
    a = np.array([r['mean_target_margin'] for r in baseline])
    b = np.array([r['mean_target_margin'] for r in changed])
    assert [r['id'] for r in baseline] == [r['id'] for r in changed]
    low, high = min(a.min(), b.min()) - .015, max(a.max(), b.max()) + .015
    axes[2].plot([low, high], [low, high], '--', color='#999999', linewidth=1)
    axes[2].axhline(0, color='#BBBBBB', linewidth=.8)
    axes[2].axvline(0, color='#BBBBBB', linewidth=.8)
    axes[2].scatter(a, b, c=np.where(b > a, '#37816C', '#BA4B44'), s=36)
    for old, new, row in zip(a, b, baseline):
        axes[2].annotate(row['id'].split('/')[-1], (old, new),
                         xytext=(4, 3), textcoords='offset points', fontsize=7)
    axes[2].set(xlim=(low, high), ylim=(low, high),
                xlabel='Natural caption: gold advantage',
                ylabel='Objects + relations, no headers: gold advantage')
    axes[2].set_title(f'Question-level comparison\nAbove diagonal = improved ({int((b > a).sum())}/25)', fontsize=11)
    axes[2].set_aspect('equal', adjustable='box')
    fig.suptitle('Brand | Fixed questions and 30 candidate memories | BGE-small-en-v1.5', fontsize=15)
    fig.savefig(OUT / 'Brand_retrieval_comparison.png', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    protocol = json.loads((NATURAL / 'protocol.json').read_text())
    prior = json.loads((OUT / 'Brand_similarity_diagnosis.json').read_text())
    memories = [r for r in read_jsonl(NATURAL / 'memories.jsonl') if r['task'] == TASK]
    captions = {r['id']: r['caption'] for r in read_jsonl(NATURAL / 'captions.jsonl')}
    extraction = {r['image_id']: r for r in read_jsonl(OUT / 'extractions.jsonl')}
    assert len(memories) == 30 and all(len(m['images']) == 1 for m in memories)
    ids = [m['id'] for m in memories]
    order = {mid: i for i, mid in enumerate(ids)}
    inputs = []
    for m in memories:
        e = extraction[m['images'][0]]
        assert e['caption'] == captions[m['images'][0]]
        inputs.append({'memory_id': m['id'], 'image': e['cache_name'], 'texts': {
            'natural_caption': e['caption'],
            'objects_and_relations': 'Objects: ' + ', '.join(e['objects']) + '\nRelations:\n' + '\n'.join(e['relations']),
            'objects_and_relations_no_headers': ', '.join(e['objects']) + '\n' + '\n'.join(e['relations']),
            'relations_only': '\n'.join(e['relations']),
        }})
    labels = {r['memory_id']: r['image'] for r in inputs}
    assert len(set(labels.values())) == 30
    for v in VARIANTS:
        old = {r['image']: r['text'] for r in prior['variants'][v]['inputs']}
        assert all(r['texts'][v] == old[r['image']] for r in inputs)

    mcq_path = ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl'
    mcq = [r for r in read_jsonl(mcq_path) if r['task'] == TASK]
    opened = [r for r in read_jsonl(NATURAL / 'queries.jsonl') if r['task'] == TASK]
    gold = {r['id']: r['clue_ids'] for r in read_jsonl(NATURAL / 'gold.jsonl') if r['task'] == TASK}
    assert len(mcq) == len(opened) == 29
    assert [q['id'] for q in mcq] == [q['id'] for q in opened]
    for q in mcq:
        assert q['gold_ids'] == gold[q['id']] and not q['query_image_paths']
    assert all(not q['images'] for q in opened)
    assert all(len(set(g)) == len(g) and set(g) <= set(ids) for g in gold.values())
    queries = {'mcq_question_only': mcq, 'open_question_only': opened}
    prefix = protocol['query_prefix']
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    model = SentenceTransformer(protocol['retriever_model_path'], device='cpu', local_files_only=True)
    texts = [r['texts'][v] for v in VARIANTS for r in inputs]
    texts += [prefix + q['question'] for questions in queries.values() for q in questions]
    lengths = [len(model.tokenizer.encode(t, truncation=False)) for t in texts]
    assert max(lengths) <= model.max_seq_length, 'Do not silently truncate retrieval inputs'
    vectors = model.encode(texts, batch_size=16, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    details = []
    for qi, (query_set, questions) in enumerate(queries.items()):
        for vi, variant in enumerate(VARIANTS):
            memory_vectors = {mid: vectors[vi * 30 + i] for i, mid in enumerate(ids)}
            for j, q in enumerate(questions):
                qvec = vectors[len(VARIANTS) * 30 + qi * 29 + j].copy()
                ranking = rank_task(qvec, ids, memory_vectors, order, 30)
                ranking = [{'memory_id': mid, 'image': labels[mid], 'rank': rank,
                            'score': score, 'is_gold': mid in gold[q['id']]}
                           for rank, (mid, score) in enumerate(ranking, 1)]
                targets = [dict(r) for r in ranking if r['is_gold']]
                distractors = [r for r in ranking if not r['is_gold']]
                top = max(distractors, key=lambda r: r['score']) if distractors else None
                for t in targets:
                    t['margin'] = t['score'] - top['score'] if top else None
                pairwise = [float(t['score'] > d['score']) + .5 * float(t['score'] == d['score'])
                            for t in targets for d in distractors]
                details.append({
                    'query_set': query_set, 'variant': variant, 'id': q['id'],
                    'question': q['question'], 'query_input': prefix + q['question'],
                    'gold_ids': gold[q['id']], 'gold_images': [labels[mid] for mid in gold[q['id']]],
                    'targets': targets, 'top_distractor': top, 'ranking': ranking,
                    'mean_gold_rank': avg([t['rank'] for t in targets]),
                    'mean_gold_score': avg([t['score'] for t in targets]),
                    'mean_target_margin': avg([t['margin'] for t in targets]) if top else None,
                    'pairwise_win_rate': avg(pairwise) if pairwise else None,
                    'recall': {str(k): sum(t['rank'] <= k for t in targets) / len(targets) for k in (1, 5, 10)},
                })

    old_path = ROOT / 'results/mcq_fifth_non_target_rank/retrieval_details.jsonl'
    old = {r['id']: r for r in read_jsonl(old_path) if r['id'].startswith(TASK + '/')}
    base = [r for r in details if r['query_set'] == 'mcq_question_only' and r['variant'] == 'natural_caption']
    errors = []
    for row, q in zip(base, mcq):
        previous = old[row['id']]
        assert row['question'] == previous['question'] and row['gold_ids'] == previous['gold_ids']
        assert [r['memory_id'] for r in row['ranking']] == [r['id'] for r in previous['ranking']]
        assert [r['memory_id'] for r in row['ranking'][:15]] == q['top15_memory_ids']
        errors.extend(abs(new['score'] - saved['score']) for new, saved in zip(row['ranking'], previous['ranking']))
    assert max(errors) <= 1e-6, f'Natural baseline score mismatch: {max(errors)}'

    summaries, comparisons = {}, {}
    for query_set in queries:
        summaries[query_set] = {}
        comparisons[query_set] = {}
        for subset in ('all_questions', 'with_distractors'):
            summaries[query_set][subset] = {}
            for v in VARIANTS:
                selected = [r for r in details if r['query_set'] == query_set and r['variant'] == v
                            and (subset == 'all_questions' or r['top_distractor'] is not None)]
                summaries[query_set][subset][v] = summarize(selected)
        base = [r for r in details if r['query_set'] == query_set and r['variant'] == 'natural_caption'
                and r['top_distractor'] is not None]
        for v in VARIANTS[1:]:
            changed = [r for r in details if r['query_set'] == query_set and r['variant'] == v
                       and r['top_distractor'] is not None]
            target_changes, question_changes = [], []
            for before, after in zip(base, changed):
                assert before['id'] == after['id']
                question_changes.append({'id': before['id'], 'question': before['question'],
                    'margin_change': after['mean_target_margin'] - before['mean_target_margin'],
                    'recall_at_5_change': after['recall']['5'] - before['recall']['5']})
                by_id = {t['memory_id']: t for t in after['targets']}
                for old_target in before['targets']:
                    new_target = by_id[old_target['memory_id']]
                    target_changes.append({'id': before['id'], 'question': before['question'],
                        'image': old_target['image'], 'before': old_target, 'after': new_target,
                        'before_top_distractor': before['top_distractor'],
                        'after_top_distractor': after['top_distractor']})
            comparisons[query_set][v] = {
                'questions_margin_improved': sum(r['margin_change'] > 1e-6 for r in question_changes),
                'questions_margin_worsened': sum(r['margin_change'] < -1e-6 for r in question_changes),
                'top5_gains': sum(r['before']['rank'] > 5 >= r['after']['rank'] for r in target_changes),
                'top5_losses': sum(r['before']['rank'] <= 5 < r['after']['rank'] for r in target_changes),
                'question_changes': question_changes,
                'target_changes': target_changes,
            }
    excluded = [q['id'] for q in mcq if set(gold[q['id']]) == set(ids)]
    report = {
        'task': TASK, 'primary_query_set': 'mcq_question_only',
        'sensitivity_query_set': 'open_question_only',
        'candidate_count': 30, 'questions': 29, 'query_gold_pairs': 228,
        'competition_questions': 25, 'competition_query_gold_pairs': 108,
        'excluded_from_competition': excluded,
        'gold_definition': 'Original dataset clue memories; all clues excluded from distractors',
        'margin_definition': 'Each gold score minus highest non-gold score; first average golds within question, then questions',
        'retriever': protocol['retriever'], 'model_path': protocol['retriever_model_path'],
        'query_prefix': prefix, 'memory_prefix': '', 'device': 'cpu', 'threads': 4,
        'seed': protocol['seed'], 'embedding_dimension': int(vectors.shape[1]),
        'normalized': True, 'max_seq_length': model.max_seq_length,
        'max_input_tokens': max(lengths), 'truncated_inputs': 0,
        'ranking': 'Current pipeline rank_task: descending cosine rounded to 6 decimals, source memory order breaks ties',
        'controls': 'Same question embeddings and 30 candidates for every variant; no options, answers, filenames, timestamps, or gold information encoded',
        'limitations': 'Retrieval only, not answer accuracy; no time fields in either memory representation; annotated clues need not be the only semantically relevant memories',
        'sources': [str(mcq_path), str(NATURAL / 'queries.jsonl'), str(NATURAL / 'gold.jsonl'),
                    str(NATURAL / 'memories.jsonl'), str(NATURAL / 'captions.jsonl'),
                    str(NATURAL / 'protocol.json'), str(OUT / 'extractions.jsonl'),
                    str(OUT / 'Brand_similarity_diagnosis.json')],
        'natural_baseline_replay': {'source': str(old_path), 'questions': 29, 'scores': len(errors),
            'identical_full_rankings': True, 'identical_top15_to_pipeline': True,
            'max_absolute_score_error': max(errors), 'tolerance': 1e-6},
        'memory_inputs': inputs, 'summaries': summaries, 'paired_changes': comparisons,
    }
    assert len(excluded) == 4
    assert summaries['mcq_question_only']['with_distractors']['natural_caption']['query_gold_pairs'] == 108
    (OUT / 'Brand_retrieval_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'Brand_retrieval_details.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in details))
    plot_results(details, summaries)
    print(json.dumps({'baseline_replay': report['natural_baseline_replay'], 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
