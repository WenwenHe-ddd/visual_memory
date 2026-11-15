"""Exploratory paired Recall@5 curves by caption similarity pressure.

Usage: python3 scripts/analysis/plot_similarity_pressure_strata.py
This is a post-hoc representation-similarity proxy, not a cue-overload label.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
SOURCE = ROOT / 'results/random_task_injection'
VISUAL = ROOT / 'results/random_task_injection_visual_master'
ZSCORE = ROOT / 'results/random_task_injection_visual_master_zscore'
OUT = ROOT / 'results/similarity_pressure_strata'
METHODS = {'caption': SOURCE, 'visual_master': VISUAL, 'visual_master_zscore': ZSCORE}
STRATA = ('Low', 'Medium', 'High')
LEVELS = (0, 10, 20)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {method: read_rows(folder / 'runs.csv') for method, folder in METHODS.items()}
    def key(row):
        return row['sample_id'], int(row['repetition']), int(row['N'])

    tables = {}
    for method, rows in raw.items():
        selected = [row for row in rows if row['condition'] == 'same_task' and int(row['N']) in LEVELS]
        table = {key(row): row for row in selected}
        if len(table) != len(selected):
            raise ValueError(f'Duplicate trial key: {method}')
        tables[method] = table
    keys = set(tables['caption'])
    if any(set(table) != keys for table in tables.values()):
        raise ValueError('Paired trial keys differ across methods')
    if len(keys) != 878 * 5 * 3:
        raise ValueError(f'Unexpected paired cohort: {len(keys)}')

    samples = defaultdict(list)
    for (sid, rep, n), row in tables['caption'].items():
        if n == 20:
            samples[sid].append(row)
    sample_info = {}
    by_task = defaultdict(list)
    for sid, rows in samples.items():
        if len(rows) != 5 or len({row['task'] for row in rows}) != 1:
            raise ValueError(f'Incomplete N=20 similarity samples: {sid}')
        value = float(np.mean([float(row['mean_target_distractor_cosine']) for row in rows]))
        row = rows[0]
        sample_info[sid] = {'task': row['task'], 'question_id': row['question_id'],
                            'target_id': row['target_id'], 'similarity': value}
        by_task[row['task']].append(sid)
    for task, sids in by_task.items():
        ordered = sorted(sids, key=lambda sid: (sample_info[sid]['similarity'], sid))
        for index, sid in enumerate(ordered):
            sample_info[sid]['stratum'] = STRATA[min(3 * index // len(ordered), 2)]

    strata_rows = []
    for sid in sorted(sample_info):
        data = sample_info[sid]
        strata_rows.append({'sample_id': sid, 'question_id': data['question_id'],
                            'target_id': data['target_id'], 'task': data['task'],
                            'stratum': data['stratum'],
                            'mean_target_distractor_cosine_N20': data['similarity']})
    write_rows(OUT / 'sample_strata.csv', strata_rows)

    # One sample contributes its five-draw mean at each N. This keeps the
    # original target-level weighting and fixed cohort across N.
    sample_scores = {}
    for method, table in tables.items():
        for sid in samples:
            for n in LEVELS:
                values = [float(table[(sid, rep, n)]['recall_at_5']) for rep in range(5)]
                sample_scores[(method, sid, n)] = float(np.mean(values))
    curves = []
    for stratum in STRATA:
        sids = [sid for sid, row in sample_info.items() if row['stratum'] == stratum]
        for n in LEVELS:
            for method in METHODS:
                curves.append({'stratum': stratum, 'N': n, 'method': method,
                               'targets': len(sids),
                               'questions': len({sample_info[sid]['question_id'] for sid in sids}),
                               'mean_recall_at_5': float(np.mean([sample_scores[(method, sid, n)] for sid in sids]))})
    write_rows(OUT / 'curves.csv', curves)

    # Cluster bootstrap resamples questions; multiple targets and draws from a
    # question stay together. These are descriptive post-hoc intervals.
    rng = np.random.default_rng(20261030)
    by_question = defaultdict(list)
    for sid, row in sample_info.items():
        by_question[row['question_id']].append(sid)
    qids = sorted(by_question)
    comparisons = []
    for method in ('visual_master', 'visual_master_zscore'):
        for n in (10, 20):
            sample_delta = {sid: sample_scores[(method, sid, n)] - sample_scores[('caption', sid, n)] for sid in samples}
            def statistic(question_ids):
                grouped = {name: [] for name in STRATA}
                for qid in question_ids:
                    for sid in by_question[qid]:
                        grouped[sample_info[sid]['stratum']].append(sample_delta[sid])
                means = {name: float(np.mean(grouped[name])) for name in STRATA}
                means['High_minus_Low'] = means['High'] - means['Low']
                return means
            point = statistic(qids)
            draws = {name: [] for name in (*STRATA, 'High_minus_Low')}
            for _ in range(2000):
                sampled = rng.choice(qids, size=len(qids), replace=True)
                result = statistic(sampled)
                for name in draws:
                    draws[name].append(result[name])
            for name in draws:
                low, high = np.percentile(draws[name], (2.5, 97.5))
                comparisons.append({'method': method, 'N': n, 'stratum': name,
                                    'delta_recall_at_5': point[name],
                                    'bootstrap_low_95': float(low),
                                    'bootstrap_high_95': float(high),
                                    'bootstrap_clusters': len(qids)})
    write_rows(OUT / 'paired_deltas.csv', comparisons)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = {'caption': '#2e8052', 'visual_master': '#7656a8', 'visual_master_zscore': '#c65875'}
    labels = {'caption': 'Original caption', 'visual_master': 'Caption visual master',
              'visual_master_zscore': 'Visual master + z-score'}
    markers = {'caption': 'o', 'visual_master': 's', 'visual_master_zscore': '^'}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=True)
    for ax, stratum in zip(axes, STRATA):
        selected = [sid for sid, row in sample_info.items() if row['stratum'] == stratum]
        for method in METHODS:
            points = [next(row for row in curves if row['stratum'] == stratum and row['method'] == method and row['N'] == n)
                      for n in LEVELS]
            ax.plot(LEVELS, [row['mean_recall_at_5'] for row in points],
                    color=colors[method], marker=markers[method], linewidth=2.3,
                    markersize=6, label=labels[method])
        ax.set_title(f'{stratum} similarity pressure\n{len(selected)} targets / '
                     f'{len({sample_info[sid]["question_id"] for sid in selected})} questions\n'
                     'within-task tertile')
        ax.set_xticks(LEVELS)
        ax.set_xlabel('Same-task distractors N')
        ax.set_ylim(0, 1.04)
        ax.grid(axis='y', alpha=.22)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Target Recall@5')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', frameon=False, ncol=3)
    fig.suptitle('Exploratory stratification by target–distractor caption similarity', fontsize=14)
    fig.text(.5, .075, 'Strata fixed from mean cosine at N=20 over 5 draws, split into tertiles within each task. '
             'Similarity is a proxy, not an annotation of shared question cues.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .13, 1, .93))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'paired_curves.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

    protocol = {'analysis': 'Exploratory similarity-pressure strata, not independently annotated cue overload',
                'stratification': 'Within-task sample-level tertiles of five-draw mean target-distractor BGE caption cosine at N=20',
                'fixed_cohort': '878 target samples, 366 questions, five draws, N=0/10/20',
                'same_task_only': True,
                'paired_trial_keys_identical': True,
                'limitation': 'Similarity may measure shared semantics unrelated to the query; visual-master uses candidate statements while caption uses question-only retrieval; this analysis is descriptive, not causal.',
                'bootstrap': '2000 question-cluster resamples; each question keeps its targets and their five-draw means',
                'input_sha256': {str(folder.relative_to(ROOT) / 'runs.csv'): digest(folder / 'runs.csv')
                                 for folder in METHODS.values()}}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print('Wrote', OUT)
    for row in comparisons:
        if row['N'] == 20:
            print(row)


if __name__ == '__main__':
    main()
