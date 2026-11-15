"""Paired same-task distractor injection with a freshly built caption_visual_master.

The target, distractor IDs, draw order and caption-only leaf vectors are frozen
by random_task_injection. Only the retrieval organization changes. No reader
is called and individual CLIP image vectors are never written to disk.
"""
from __future__ import annotations

import csv
import hashlib
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
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master import load_clip_text_encoder, load_text_encoder, encode_text, normalize

SOURCE = ROOT / 'results/random_task_injection'
NAT = ROOT / 'results/natural_observation'
MCQ = ROOT / 'results/mcq_fifth_non_target_rank'
FROZEN = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
MASTER = ROOT / 'reports/all_master_answer_query_20261021'
OUT = ROOT / 'results/random_task_injection_visual_master'


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_graph(memory, ids, leaf_vectors, query_vectors, candidate_vectors, target_id):
    masters = list(memory.masters.values())
    centers = normalize([m['visual_sum'] / m['n'] for m in masters])
    choices = np.argmax(centers @ candidate_vectors.T, axis=1)
    by_master = {m['id']: int(choice) for m, choice in zip(masters, choices)}
    scores = np.array([
        float(query_vectors[by_master[memory.image_nodes[mid]['master_id']]] @ leaf_vectors[mid])
        for mid in ids
    ])
    target_score = scores[ids.index(target_id)]
    distractor_scores = np.delete(scores, ids.index(target_id))
    result = single_target_metric(target_score, distractor_scores)
    result.update(target_score=float(target_score), master_count=len(masters),
                  target_master_size=memory.masters[memory.image_nodes[target_id]['master_id']]['n'])
    return result


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    baseline = list(csv.DictReader((SOURCE / 'group_means.csv').open(encoding='utf-8-sig')))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    names = [('same_task', '#2e8052', 'Original caption · same task'),
             ('other_tasks', '#cfba23', 'Original caption · other tasks')]
    for ax, (task, title) in zip(axes.flat, NAMES.items()):
        for condition, color, label in names:
            points = sorted((r for r in baseline if r['task'] == task and r['condition'] == condition),
                            key=lambda r: int(r['N']))
            ax.plot([int(r['N']) for r in points], [float(r['mean_recall_at_5']) for r in points],
                    '-o', color=color, lw=2.1, ms=5, label=label)
        points = sorted((r for r in rows if r['task'] == task), key=lambda r: int(r['N']))
        ax.plot([int(r['N']) for r in points], [float(r['mean_recall_at_5']) for r in points],
                '-s', color='#7455a9', lw=2.1, ms=5, label='Visual master · same task')
        ax.set_title(f"{title}\n{points[0]['target_samples']} targets / {points[0]['questions']} questions")
        ax.set_xticks([int(r['N']) for r in points]); ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2); ax.set_xlabel('Distractors N')
    axes[0, 0].set_ylabel('Target Recall@5'); axes[1, 0].set_ylabel('Target Recall@5')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Controlled distractor injection: paired visual-master comparison', fontsize=14)
    fig.text(.5, .055, 'Same targets, same five draws and same caption-only leaf vectors; visual masters rebuilt for each sampled memory.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .95))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'by_task_injection_paired.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((SOURCE / 'protocol.json').read_text())
    if source_protocol['cohort'] != 'fixed 878 target samples / 366 questions at all main N=0,10,20 and both conditions':
        raise ValueError('Source cohort changed')
    memories = read_jsonl(NAT / 'memories.jsonl')
    captions = {r['id']: r['caption'] for r in read_jsonl(NAT / 'captions.jsonl')}
    image_paths = {r['id']: Path(r['path']) for r in read_jsonl(NAT / 'images.jsonl')}
    queries = {r['id']: r for r in read_jsonl(MCQ / 'mcq_queries.jsonl')}
    plans = [r for r in read_jsonl(SOURCE / 'sampling_plans.jsonl') if r['condition'] == 'same_task']
    candidates = {r['id']: r for r in read_jsonl(MASTER / 'candidates.jsonl')}
    if len(plans) != 878 * 5 or len({r['question_id'] for r in plans}) != 366 or not {r['question_id'] for r in plans} <= set(queries):
        raise ValueError('Sampling plan mismatch')
    source_vectors = np.load(NAT / 'embeddings.npz')
    if list(source_vectors['memory_ids']) != [r['id'] for r in memories]:
        raise ValueError('Caption embedding order changed')
    leaf_vectors = {r['id']: source_vectors['memory'][i] for i, r in enumerate(memories)}
    caption_text = {r['id']: captions[r['images'][0]] for r in memories}
    all_needed = {r['target_id'] for r in plans} | {mid for r in plans for mid in r['ordered_distractor_ids']}
    image_by_mid = {r['id']: image_paths[r['images'][0]] for r in memories if r['id'] in all_needed}
    if set(image_by_mid) != all_needed:
        raise ValueError('Some sampled images are missing')

    protocol = json.loads((FROZEN / 'Home_Renovation_Interior_Design/protocol.json').read_text())
    import torch
    torch.set_num_threads(4)
    text_model = load_text_encoder(protocol['text_encoder'], 'cpu')
    question_vectors = {}
    for qid, q in queries.items():
        c = candidates[qid]
        if c['question'] != q['question'] or set(c['candidate_statements']) != set('ABCD'):
            raise ValueError(f'Candidate mismatch: {qid}')
        question_vectors[qid] = encode_text(text_model,
            [q['question'] + '\nCandidate answer: ' + c['candidate_statements'][k] for k in 'ABCD'],
            protocol['query_prefix'])
    del text_model
    clip_model, tokenizer, token_limit = load_clip_text_encoder(protocol['image_encoder'], 'cpu')
    candidate_vectors = {}
    with torch.inference_mode():
        for qid, c in candidates.items():
            if qid not in queries:
                continue
            encoded = tokenizer([c['candidate_statements'][k] for k in 'ABCD'],
                                padding=True, truncation=False, return_tensors='pt')
            if encoded['input_ids'].shape[1] > token_limit:
                raise ValueError(f'CLIP candidate too long: {qid}')
            candidate_vectors[qid] = normalize(clip_model(**encoded).text_embeds.double().numpy())
    del clip_model
    encoder = ImageEncoder(protocol['image_encoder']['model_name'],
                           revision=protocol['image_encoder']['revision'], device='cpu', local_files_only=True)
    image_vectors = {}
    for i, mid in enumerate(sorted(all_needed), 1):
        path = image_by_mid[mid]
        if digest(path) != next(r['images'][0] for r in memories if r['id'] == mid):
            raise ValueError(f'Image changed: {mid}')
        image_vectors[mid] = encoder.encode(path)
        if i % 50 == 0:
            print(f'Encoded {i}/{len(all_needed)} images', flush=True)
    del encoder
    print(f'Encoded {len(all_needed)} images; evaluating {len(plans)} paired draws', flush=True)

    bases = {task: np.load(FROZEN / task / 'memory/basis.npy', allow_pickle=False) for task in NAMES}
    by_mid = {r['id']: r for r in memories}
    rows = []
    for i, plan in enumerate(plans, 1):
        target = plan['target_id']; qid = plan['question_id']; task = by_mid[target]['task']
        levels = source_protocol['task_levels'][task]
        ordered = plan['ordered_distractor_ids']
        if len(ordered) < max(levels) or any(by_mid[mid]['task'] != task for mid in ordered):
            raise ValueError('Draw differs from original same-task plan')
        memory = CaptionWardVisualMemory(bases[task])
        current = []
        for n in range(max(levels) + 1):
            mid = target if n == 0 else ordered[n - 1]
            memory.add(mid, caption_text[mid], leaf_vectors[mid], image_vectors[mid])
            current.append(mid)
            if n not in levels:
                continue
            metric = score_graph(memory, current, leaf_vectors, question_vectors[qid],
                                 candidate_vectors[qid], target)
            rows.append({'sample_id': plan['sample_id'], 'question_id': qid, 'task': task,
                         'target_id': target, 'condition': 'same_task', 'N': n,
                         'repetition': plan['repetition'], 'memory_count': n + 1, **metric})
        if i % 100 == 0:
            print(f'Evaluated {i}/{len(plans)} draws', flush=True)
    write_csv(OUT / 'runs.csv', rows)
    groups = []
    for task in NAMES:
        for n in source_protocol['task_levels'][task]:
            selected = [r for r in rows if r['task'] == task and r['N'] == n]
            groups.append({'task': task, 'N': n, 'condition': 'same_task',
                           'target_samples': len(selected) // 5,
                           'questions': len({r['question_id'] for r in selected}),
                           'mean_recall_at_5': float(np.mean([r['recall_at_5'] for r in selected])),
                           'mean_master_count': float(np.mean([r['master_count'] for r in selected]))})
    write_csv(OUT / 'group_means.csv', groups)
    (OUT / 'protocol.json').write_text(json.dumps({
        'source_sampling_plans_sha256': digest(SOURCE / 'sampling_plans.jsonl'),
        'source_group_means_sha256': digest(SOURCE / 'group_means.csv'),
        'paired_samples': len(plans), 'targets': 878, 'questions': 366,
        'method': 'CaptionWardVisualMemory.add, freshly rebuilt for target plus each sampled distractor prefix',
        'leaf_text': 'the exact caption-only text and cached BGE vector used by original injection',
        'query_text': 'original question plus frozen candidate statement; BGE query prefix',
        'visual_input': 'original image re-encoded with frozen CLIP vision checkpoint; vectors RAM only',
        'master_choice': 'CLIP cosine of each current master visual mean to four candidate statements',
        'ranking': 'per-leaf BGE cosine to the augmented query selected by its current master',
        'comparison_limit': 'The two original curves use question-only retrieval; new curve changes query routing and graph organization together.',
        'reader_called': False, 'device': 'cpu',
        'source_protocol_sha256': digest(SOURCE / 'protocol.json'),
        'script_sha256': digest(Path(__file__)),
    }, indent=2) + '\n')
    plot(groups)
    print('Done:', OUT, flush=True)


if __name__ == '__main__':
    main()
