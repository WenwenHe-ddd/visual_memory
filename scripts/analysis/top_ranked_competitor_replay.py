"""Replay fixed top-ranked non-target images against three retrieval methods.

Entry: python3 scripts/analysis/top_ranked_competitor_replay.py
Competitors are frozen by the original caption full-task ranking, excluding
all annotated targets and duplicate image content. They are then inserted in
rank order into a target-plus-competitors small memory for paired evaluation.
"""
from __future__ import annotations

import csv
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))

import numpy as np

from scripts.analysis.random_task_injection import NAMES, single_target_metric
from scripts.analysis.random_task_injection_visual_master import (
    read_jsonl, digest, write_csv, score_graph,
)
from scripts.analysis.random_task_injection_visual_master_zscore import score_graph as score_zscore
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master import load_clip_text_encoder, load_text_encoder, encode_text, normalize

NAT = ROOT / 'results/natural_observation'
MCQ = ROOT / 'results/mcq_fifth_non_target_rank'
FROZEN = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
MASTER = ROOT / 'reports/all_master_answer_query_20261021'
SOURCE = ROOT / 'results/random_task_injection'
OUT = ROOT / 'results/top_ranked_competitor_replay'
LEVELS = (0, 2, 5, 10)


def build_plans(details, memories, cohort):
    image_hash = {r['id']: r['images'][0] for r in memories}
    tasks = {r['id']: r['task'] for r in memories}
    plans = []
    excluded = []
    for detail in details:
        qid = detail['id']
        if qid not in cohort:
            continue
        gold = set(detail['gold_ids'])
        excluded_hashes = {image_hash[mid] for mid in gold}
        ranked = sorted(detail['ranking'], key=lambda r: int(r['rank']))
        if any(tasks[row['id']] != tasks[next(iter(gold))] for row in ranked):
            raise ValueError(f'Cross-task ranked result: {qid}')
        if [int(row['rank']) for row in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError(f'Incomplete ranking: {qid}')
        competitors = []
        seen = set(excluded_hashes)
        for row in ranked:
            mid = row['id']
            if mid in gold or image_hash[mid] in seen:
                continue
            competitors.append({'id': mid, 'original_rank': int(row['rank']),
                                'original_query_score': float(row['score'])})
            seen.add(image_hash[mid])
            if len(competitors) == max(LEVELS):
                break
        if len(competitors) != max(LEVELS):
            excluded.append({'question_id': qid, 'task': tasks[next(iter(gold))],
                             'target_samples': len(cohort[qid]),
                             'available_distinct_non_targets': len(competitors),
                             'required': max(LEVELS)})
            continue
        for target in sorted(cohort[qid]):
            if target not in gold:
                raise ValueError(f'Cohort target is not annotated gold: {qid} {target}')
            plans.append({'sample_id': qid + '|' + target, 'question_id': qid,
                          'task': tasks[target], 'target_id': target,
                          'gold_count': len(gold), 'competitors': competitors})
    if len(plans) + sum(row['target_samples'] for row in excluded) != 878 or len({p['question_id'] for p in plans}) + len(excluded) != 366:
        raise ValueError('Frozen source 878-target / 366-question cohort changed')
    return plans, excluded


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'caption': '#2e8052', 'visual_master': '#7656a8', 'visual_master_zscore': '#c65875'}
    labels = {'caption': 'Original caption', 'visual_master': 'Caption visual master',
              'visual_master_zscore': 'Visual master + z-score'}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    for ax, (task, title) in zip(axes.flat, NAMES.items()):
        subset = [r for r in rows if r['task'] == task]
        for method in colors:
            points = sorted([r for r in subset if r['method'] == method], key=lambda r: int(r['N']))
            ax.plot([r['N'] for r in points], [r['recall_at_5'] for r in points],
                    '-o', color=colors[method], lw=2, ms=5, label=labels[method])
        ax.set_title(f"{title} · {subset[0]['targets']} targets")
        ax.set_xticks(LEVELS)
        ax.set_ylim(0, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2)
        ax.set_xlabel('Top-ranked non-target images N')
    axes[0, 0].set_ylabel('Target Recall@5')
    axes[1, 0].set_ylabel('Target Recall@5')
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False)
    fig.suptitle(f'Fixed original-caption Top-{max(LEVELS)} non-target competitors', fontsize=14)
    fig.text(.5, .052, 'All annotated targets and duplicate image contents excluded. '
             'Same target and ranked competitors for all methods; small-memory replay.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .94))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'by_task.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def paired_bootstrap(rows):
    """Question-cluster bootstrap of paired visual-minus-caption differences."""
    by_sample = {(r['sample_id'], int(r['N']), r['method']): float(r['recall_at_5']) for r in rows}
    question_samples = {}
    for row in rows:
        question_samples.setdefault(row['question_id'], set()).add(row['sample_id'])
    qids = sorted(question_samples)
    rng = np.random.default_rng(20261030)
    output = []
    for n in (level for level in LEVELS if level >= 5):
        for method in ('visual_master', 'visual_master_zscore'):
            per_question = []
            counts = []
            wins = losses = ties = 0
            for qid in qids:
                differences = [by_sample[sid, n, method] - by_sample[sid, n, 'caption']
                               for sid in question_samples[qid]]
                wins += sum(value > 0 for value in differences)
                losses += sum(value < 0 for value in differences)
                ties += sum(value == 0 for value in differences)
                per_question.append(sum(differences))
                counts.append(len(differences))
            per_question = np.asarray(per_question, dtype=float)
            counts = np.asarray(counts, dtype=int)
            indices = rng.integers(0, len(qids), size=(3000, len(qids)))
            draws = per_question[indices].sum(axis=1) / counts[indices].sum(axis=1)
            low, high = np.percentile(draws, (2.5, 97.5))
            output.append({'N': n, 'method': method, 'delta_recall_at_5': float(per_question.sum() / counts.sum()),
                           'bootstrap_low_95': float(low), 'bootstrap_high_95': float(high),
                           'wins': wins, 'losses': losses, 'ties': ties,
                           'question_clusters': len(qids)})
    write_csv(OUT / 'paired_deltas.csv', output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = {}
    for row in read_jsonl(SOURCE / 'sampling_plans.jsonl'):
        if row['condition'] == 'same_task':
            cohort.setdefault(row['question_id'], set()).add(row['target_id'])
    memories = read_jsonl(NAT / 'memories.jsonl')
    details = read_jsonl(MCQ / 'retrieval_details.jsonl')
    plans, excluded = build_plans(details, memories, cohort)
    if excluded:
        write_csv(OUT / 'excluded_questions.csv', excluded)
    with (OUT / 'competitor_plans.jsonl').open('w') as f:
        for plan in plans:
            f.write(json.dumps(plan) + '\n')
    source_vectors = np.load(NAT / 'embeddings.npz')
    if list(source_vectors['memory_ids']) != [row['id'] for row in memories]:
        raise ValueError('Caption vector order changed')
    leaf_vectors = {row['id']: source_vectors['memory'][i] for i, row in enumerate(memories)}
    captions = {row['id']: row['caption'] for row in read_jsonl(NAT / 'captions.jsonl')}
    caption_text = {row['id']: captions[row['images'][0]] for row in memories}
    paths = {row['id']: Path(row['path']) for row in read_jsonl(NAT / 'images.jsonl')}
    by_mid = {row['id']: row for row in memories}
    needed = {plan['target_id'] for plan in plans} | {row['id'] for plan in plans for row in plan['competitors']}
    queries = {row['id']: row for row in read_jsonl(MCQ / 'mcq_queries.jsonl')}
    candidates = {row['id']: row for row in read_jsonl(MASTER / 'candidates.jsonl')}
    baseline_vectors = np.load(MCQ / 'query_embeddings.npz')
    query_ids = list(baseline_vectors['query_ids'])
    baseline_query = {qid: baseline_vectors['query'][i] for i, qid in enumerate(query_ids)}
    if not {p['question_id'] for p in plans} <= set(baseline_query):
        raise ValueError('Missing question-only query vector')

    protocol = json.loads((FROZEN / 'Home_Renovation_Interior_Design/protocol.json').read_text())
    import torch
    torch.set_num_threads(4)
    text_model = load_text_encoder(protocol['text_encoder'], 'cpu')
    question_vectors = {}
    for qid in sorted(cohort):
        q = queries[qid]
        c = candidates[qid]
        if c['question'] != q['question'] or set(c['candidate_statements']) != set('ABCD'):
            raise ValueError(f'Candidate mismatch: {qid}')
        question_vectors[qid] = encode_text(text_model,
            [q['question'] + '\nCandidate answer: ' + c['candidate_statements'][key] for key in 'ABCD'],
            protocol['query_prefix'])
    del text_model
    clip_model, tokenizer, token_limit = load_clip_text_encoder(protocol['image_encoder'], 'cpu')
    candidate_vectors = {}
    with torch.inference_mode():
        for qid in sorted(cohort):
            c = candidates[qid]
            encoded = tokenizer([c['candidate_statements'][key] for key in 'ABCD'],
                                padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > token_limit:
                raise ValueError(f'CLIP candidate too long: {qid}')
            candidate_vectors[qid] = normalize(clip_model(**encoded).text_embeds.double().numpy())
    del clip_model
    encoder = ImageEncoder(protocol['image_encoder']['model_name'],
                           revision=protocol['image_encoder']['revision'], device='cpu', local_files_only=True)
    image_vectors = {}
    for index, mid in enumerate(sorted(needed), 1):
        path = paths[by_mid[mid]['images'][0]]
        if digest(path) != by_mid[mid]['images'][0]:
            raise ValueError(f'Image changed: {mid}')
        image_vectors[mid] = encoder.encode(path)
        if index % 50 == 0:
            print(f'Encoded {index}/{len(needed)} images', flush=True)
    del encoder

    bases = {task: np.load(FROZEN / task / 'memory/basis.npy', allow_pickle=False) for task in NAMES}
    rows = []
    for index, plan in enumerate(plans, 1):
        target = plan['target_id']
        qid = plan['question_id']
        ordered = [row['id'] for row in plan['competitors']]
        graph = CaptionWardVisualMemory(bases[plan['task']])
        current = []
        for n in range(max(LEVELS) + 1):
            mid = target if n == 0 else ordered[n-1]
            graph.add(mid, caption_text[mid], leaf_vectors[mid], image_vectors[mid])
            current.append(mid)
            if n not in LEVELS:
                continue
            base_scores = [float(baseline_query[qid] @ leaf_vectors[item]) for item in current]
            base = single_target_metric(base_scores[0], base_scores[1:])
            visual = score_graph(graph, current, leaf_vectors, question_vectors[qid], candidate_vectors[qid], target)
            weighted = score_zscore(graph, current, leaf_vectors, question_vectors[qid], candidate_vectors[qid], target)
            if abs(visual['target_score'] - weighted['raw_target_score']) > 1e-10:
                raise ValueError('Visual raw score and z-score replay differ')
            for method, metric in (('caption', base), ('visual_master', visual), ('visual_master_zscore', weighted)):
                rows.append({'sample_id': plan['sample_id'], 'question_id': qid, 'task': plan['task'],
                             'target_id': target, 'N': n, 'method': method,
                             'recall_at_5': metric['recall_at_5'],
                             'target_rank_min': metric['target_rank_min'],
                             'higher_scoring_distractors': metric['higher_scoring_distractors']})
        if index % 100 == 0:
            print(f'Evaluated {index}/{len(plans)} targets', flush=True)
    write_csv(OUT / 'runs.csv', rows)
    paired_bootstrap(rows)
    means = []
    for task in ('all', *NAMES):
        selected = [row for row in rows if task == 'all' or row['task'] == task]
        for n in LEVELS:
            for method in ('caption', 'visual_master', 'visual_master_zscore'):
                subset = [row for row in selected if row['N'] == n and row['method'] == method]
                means.append({'task': task, 'N': n, 'method': method, 'targets': len(subset),
                              'questions': len({row['question_id'] for row in subset}),
                              'recall_at_5': float(np.mean([row['recall_at_5'] for row in subset]))})
    write_csv(OUT / 'group_means.csv', means)
    plot([row for row in means if row['task'] != 'all'])
    (OUT / 'protocol.json').write_text(json.dumps({
        'selection': f'Scan original same-task caption full ranking; exclude every annotated gold ID and all duplicate gold/competitor image SHA256 contents; take first {max(LEVELS)} remaining distinct images',
        'top_k_plus_n': 'Top-(k+number of annotated targets) suffices before image deduplication; scan farther for duplicate content',
        'source_cohort': '878 original target samples / 366 questions from random_task_injection',
        'evaluated_targets': len(plans), 'evaluated_questions': len({p['question_id'] for p in plans}),
        'excluded_for_k': len(excluded),
        'levels': LEVELS, 'insertion_order': 'target, then competitors in original caption rank order',
        'reader_called': False, 'device': 'cpu',
        'limitation': 'Baseline-selected hard competitors; small-memory replay is not full-history retrieval. Visual methods also use candidate statements while caption uses question-only query.',
        'input_hashes': {str(path.relative_to(ROOT)): digest(path) for path in (
            MCQ / 'retrieval_details.jsonl', MCQ / 'query_embeddings.npz',
            NAT / 'embeddings.npz', SOURCE / 'sampling_plans.jsonl')},
        'zscore': {'ridge_lambda': 0.03, 'alpha': 1.0},
    }, indent=2) + '\n')
    print('Done:', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-competitors', type=int, choices=(10, 25), default=10)
    args = parser.parse_args()
    if args.max_competitors == 25:
        LEVELS = (0, 2, 5, 10, 15, 20, 25)
        OUT = ROOT / 'results/top_ranked_competitor_replay_25'
    main()
