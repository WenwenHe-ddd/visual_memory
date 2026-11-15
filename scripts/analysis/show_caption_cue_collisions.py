"""Auditable caption-cue collision examples from frozen MemEye retrieval.

Run: python3 scripts/analysis/show_caption_cue_collisions.py
The two cases are illustrative: Q4 is a failed Home retrieval with a target
caption containing the combined cue; Q10 was discussed previously and is a
successful retrieval despite shared cues. Terms are checked only in frozen
captions, not inferred from pixels. No model annotation or answer reading.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
NAT = ROOT / 'results/natural_observation'
RANK = ROOT / 'results/mcq_fifth_non_target_rank'
OUT = ROOT / 'results/caption_cue_collision_examples'

CASES = [
    {
        'question_id': 'Home_Renovation_Interior_Design/Q4',
        'short_name': 'Home Q4 · target rank 8',
        'cues': [
            ('cabinet', r'\bcabinet\b'),
            ('sample', r'\bsamples?\b'),
            ('floor', r'\bfloor\b'),
            ('gold or brass', r'\b(?:gold|brass)\b'),
        ],
        'joint': ('cabinet AND gold/brass', ('cabinet', 'gold or brass')),
    },
    {
        'question_id': 'Home_Renovation_Interior_Design/Q10',
        'short_name': 'Home Q10 · target rank 3',
        'cues': [
            ('desk', r'\bdesks?\b'),
            ('monitor', r'\bmonitors?\b'),
            ('window', r'\bwindows?\b'),
        ],
        'joint': ('desk AND monitor', ('desk', 'monitor')),
    },
]


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def content_words(text):
    return {word for word in re.findall(r'[a-z]+', text.lower())
            if len(word) > 2 and word not in ENGLISH_STOP_WORDS}


def audit_all_questions(rankings, memories, captions):
    """Conservative exact-word proxy; no inference of semantic equivalence."""
    words = {memory['id']: content_words(captions[memory['images'][0]]) for memory in memories}
    rows = []
    for qid, q in sorted(rankings.items()):
        target_words = set().union(*(words[mid] for mid in q['gold_ids']))
        cues = sorted(content_words(q['question']) & target_words)
        competitors = [item['id'] for item in q['ranking'][:15] if not item['is_target']]
        counts = {word: sum(word in words[mid] for mid in competitors) for word in cues}
        most_shared = max(cues, key=lambda word: (counts[word], word)) if cues else ''
        target_rank = min(item['rank'] for item in q['ranking'] if item['is_target'])
        rows.append({'question_id': qid, 'task': qid.split('/')[0],
                     'gold_count': len(q['gold_ids']), 'first_target_rank': target_rank,
                     'target_in_top5': int(target_rank <= 5),
                     'query_target_exact_content_words': len(cues),
                     'most_repeated_word': most_shared,
                     'top15_non_target_count': len(competitors),
                     'non_targets_with_most_repeated_word': counts.get(most_shared, 0),
                     'full_shared_words': ','.join(cues)})
    write_csv(OUT / 'all_questions_lexical_proxy.csv', rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    memories = jsonl(NAT / 'memories.jsonl')
    captions = {row['id']: row['caption'] for row in jsonl(NAT / 'captions.jsonl')}
    images = {row['id']: row['path'] for row in jsonl(NAT / 'images.jsonl')}
    by_mid = {row['id']: row for row in memories}
    rankings = {row['id']: row for row in jsonl(RANK / 'retrieval_details.jsonl')}
    all_question_rows = audit_all_questions(rankings, memories, captions)
    matching_rows = []
    summary = []
    evidence = []
    for spec in CASES:
        q = rankings[spec['question_id']]
        task = spec['question_id'].split('/')[0]
        ranked = q['ranking'][:15]
        target_ranks = [row['rank'] for row in ranked if row['is_target']]
        if len(target_ranks) != 1:
            raise ValueError(f'Expected one target in Top-15: {spec["question_id"]}')
        if len(ranked) != 15 or [row['rank'] for row in ranked] != list(range(1, 16)):
            raise ValueError(f'Ranking changed: {spec["question_id"]}')

        matches = {}
        for memory in memories:
            if memory['task'] != task:
                continue
            text = captions[memory['images'][0]]
            hits = {name: bool(re.search(pattern, text, flags=re.I)) for name, pattern in spec['cues']}
            joint_name, parts = spec['joint']
            hits[joint_name] = all(hits[part] for part in parts)
            matches[memory['id']] = hits

        labels = [name for name, _ in spec['cues']] + [spec['joint'][0]]
        for label in labels:
            full_ids = [mid for mid, hits in matches.items() if hits[label]]
            non_target_top = [row for row in ranked if not row['is_target'] and matches[row['id']][label]]
            summary.append({'question_id': q['id'], 'question': q['question'],
                            'target_rank': target_ranks[0], 'cue': label,
                            'full_task_memories': len(matches), 'full_task_cue_hits': len(full_ids),
                            'top_15_non_target_cue_hits': len(non_target_top),
                            'top_15_non_target_ranks': ','.join(str(row['rank']) for row in non_target_top),
                            'target_caption_has_cue': int(matches[q['gold_ids'][0]][label])})
        for row in ranked:
            mid = row['id']
            memory = by_mid[mid]
            caption = captions[memory['images'][0]]
            evidence.append({'question_id': q['id'], 'rank': row['rank'], 'memory_id': mid,
                             'is_target': row['is_target'], 'query_score': row['score'],
                             'image_path': images[memory['images'][0]], 'caption': caption,
                             'cue_hits': matches[mid]})
            for label in labels:
                matching_rows.append({'question_id': q['id'], 'rank': row['rank'],
                                      'memory_id': mid, 'is_target': int(row['is_target']),
                                      'cue': label, 'caption_match': int(matches[mid][label])})
    write_csv(OUT / 'cue_counts.csv', summary)
    write_csv(OUT / 'top15_matrix.csv', matching_rows)
    with (OUT / 'caption_evidence.jsonl').open('w') as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 6.9), gridspec_kw={'height_ratios': [1.1, 1]})
    lookup = {(row['question_id'], int(row['rank']), row['cue']): row for row in matching_rows}
    for ax, spec in zip(axes, CASES):
        qid = spec['question_id']
        q = rankings[qid]
        labels = [name for name, _ in spec['cues']] + [spec['joint'][0]]
        matrix = []
        for label in labels:
            line = []
            for rank in range(1, 16):
                cell = lookup[qid, rank, label]
                line.append(2 if cell['caption_match'] and cell['is_target'] else
                            1 if cell['caption_match'] else 0)
            matrix.append(line)
        cmap = ListedColormap(['#f1f1ee', '#dc8f62', '#2f7d7b'])
        ax.imshow(matrix, aspect='auto', cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3))
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(15), [str(i) for i in range(1, 16)])
        ax.set_xlabel('Original caption retrieval rank (Top-15)')
        ax.set_title(f'{spec["short_name"]}  |  {q["question"]}', loc='left', fontsize=11, pad=12)
        ax.set_xticks(np.arange(-.5, 15, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.3)
        ax.tick_params(which='minor', bottom=False, left=False)
        target_rank = next(row['rank'] for row in q['ranking'] if row['is_target'])
        ax.axvline(target_rank - 1.5, color='#145c5b', lw=1.7)
        ax.axvline(target_rank - .5, color='#145c5b', lw=1.7)
        for y, label in enumerate(labels):
            count_row = next(row for row in summary if row['question_id'] == qid and row['cue'] == label)
            ax.text(15.18, y, f'{count_row["full_task_cue_hits"]}/{count_row["full_task_memories"]}', va='center', fontsize=9)
        ax.set_xlim(-.5, 16.9)
        ax.text(15.18, -.65, 'In full task', fontsize=9, clip_on=False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#dc8f62', label='Non-target caption contains cue'),
                        Patch(color='#2f7d7b', label='Target caption contains cue'),
                        Patch(color='#f1f1ee', label='Cue absent in caption')],
               loc='lower center', frameon=False, ncol=3)
    fig.suptitle('Repeated caption cues among actual Top-15 retrieval results', fontsize=15)
    fig.text(.5, .058, 'Exact caption-text matches. The joint cue is the conjunction of the rows above; '
             'colors do not verify pixels or causal retrieval effects.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .935), h_pad=1.5)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'cue_matrix.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    task_names = {
        'Brand_Memory_Test': 'Brand', 'Card_Playlog_Test': 'Card',
        'Cartoon_Entertainment_Companion': 'Cartoon',
        'Home_Renovation_Interior_Design': 'Home',
        'Multi-Scene_Visual_Case_Archive_Assistant': 'Cross-scene',
        'Outdoor_Navigation_Route_Memory_Assistant': 'Outdoor',
        'Personal_Health_Dashboard_Assistant': 'Health',
        'Social_Chat_Memory_Test': 'Social',
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    values = []
    for task, name in task_names.items():
        question_rows = [row for row in all_question_rows if row['task'] == task]
        numerator = sum(row['non_targets_with_most_repeated_word'] >= 5 for row in question_rows)
        values.append((name, numerator, len(question_rows)))
    ys = np.arange(len(values))
    ax.barh(ys, [a / b for _, a, b in values], color='#dc8f62', height=.62)
    ax.set_yticks(ys, [name for name, _, _ in values])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6), [f'{int(t*100)}%' for t in np.linspace(0, 1, 6)])
    ax.set_xlabel('Questions with a repeated query–target word in at least 5 non-target Top-15 captions')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    for y, (_, numerator, denominator) in zip(ys, values):
        ax.text(numerator / denominator + .015, y, f'{numerator}/{denominator}', va='center', fontsize=10)
    fig.suptitle('Exact-word repetition in original caption retrieval', fontsize=14)
    fig.text(.5, .035, '371 original questions. Select the most repeated exact content word shared by question and gold caption. '
             'This lexical proxy does not annotate semantic cues or establish retrieval causality.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .085, 1, .94))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'lexical_repetition_by_task.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    case_html = []
    for spec in CASES:
        q = rankings[spec['question_id']]
        rows = [row for row in evidence if row['question_id'] == q['id']]
        cards = []
        for row in rows:
            hits = ', '.join(name for name, present in row['cue_hits'].items() if present) or 'none'
            cards.append(f'<article class="{"target" if row["is_target"] else ""}">'
                         f'<h3>Rank {row["rank"]} {"TARGET" if row["is_target"] else "non-target"} · '
                         f'{html.escape(row["memory_id"])} · score {row["query_score"]:.4f}</h3>'
                         f'<p><b>Caption cue hits:</b> {html.escape(hits)}</p>'
                         f'<p><b>Source image:</b> <code>{html.escape(row["image_path"])}</code></p>'
                         f'<p>{html.escape(row["caption"])}</p></article>')
        case_html.append(f'<section><h2>{html.escape(q["id"])}: {html.escape(q["question"])}</h2>'
                         + ''.join(cards) + '</section>')
    (OUT / 'evidence.html').write_text('<!doctype html><meta charset="utf-8"><title>Caption cue evidence</title>'
        '<style>body{font:16px/1.5 Arial;max-width:1100px;margin:36px auto;color:#222}'
        'article{padding:12px 18px;border:1px solid #ddd;margin:12px 0;border-radius:8px}'
        'article.target{border:3px solid #2f7d7b;background:#edf7f5}'
        'h3{font-size:15px;margin:0}code{overflow-wrap:anywhere}</style>'
        '<h1>Caption cue collision: complete Top-15 evidence</h1>'
        '<p>Frozen Qwen captions, BGE same-task retrieval ranking. Exact text matching only; '
        'the labels are not image annotations.</p>' + ''.join(case_html), encoding='utf-8')
    (OUT / 'protocol.json').write_text(json.dumps({
        'scope': 'two illustrative Home questions; Q4 is the first failed Home query in original order, Q10 was the previously discussed case',
        'source': 'frozen natural Qwen captions and original same-task BGE Top-15 ranking',
        'cue_rules': CASES, 'full_task_memory_count': sum(row['task'] == 'Home_Renovation_Interior_Design' for row in memories),
        'limitations': 'Exact text matches, including gold/brass as one explicitly declared cue; shows repeated cues and ranking, not a causal estimate or general prevalence; Q10 target is retrieved, Q4 target is missed at Top-5.',
        'all_questions_lexical_proxy': 'For each of 371 questions, lowercased exact English content words shared by question and any gold caption; take the word occurring in the most non-target Top-15 captions. Standard scikit-learn English stopwords; no task-specific vocabulary. This is a lexical repetition proxy, not semantic cue annotation or evidence of causal retrieval harm.',
        'all_questions_with_any_exact_query_target_word': sum(row['query_target_exact_content_words'] > 0 for row in all_question_rows),
        'all_questions_with_word_in_at_least_five_non_targets': sum(row['non_targets_with_most_repeated_word'] >= 5 for row in all_question_rows),
        'input_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            NAT / 'memories.jsonl', NAT / 'captions.jsonl', NAT / 'images.jsonl',
            RANK / 'retrieval_details.jsonl')},
    }, ensure_ascii=False, indent=2) + '\n')
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
