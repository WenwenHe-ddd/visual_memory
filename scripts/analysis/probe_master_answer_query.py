#!/usr/bin/env python3
"""Q10: CLIP master answer hypothesis -> query expansion -> all-leaf BGE Top5.

The hypothesis expands the QUERY only. Every leaf keeps its original dialogue
and full caption. No leaf distances, reconstructed visual vectors, image pixels,
score fusion, or manually selected similarity threshold enters this experiment.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np

from scripts.analysis.probe_caption_visual_online import TOKENIZER_HASHES, normalize, read_json, sha, write_json

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design'
DEFAULT_OUTPUT = ROOT / 'reports/home_q10_master_answer_query_20261021'
QID = 'Home_Renovation_Interior_Design/Q10'
# Fixed from the question and its four options before any model score is seen.
CANDIDATES = {
    'A': 'The window desk has a monitor, and the longer desk along the right wall has a monitor.',
    'B': 'The window desk has no monitor, and the longer desk along the right wall has no monitor.',
    'C': 'The window desk has a monitor, and the longer desk along the right wall has no monitor.',
    'D': 'The window desk has no monitor, and the longer desk along the right wall has a monitor.',
}


def csv_write(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(output):
    result = read_json(output / 'retrieval_results.json')
    esc = lambda v: html.escape(str(v))
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
             '<title>Q10 · master 答案扩展 query</title><style>',
             'body{font:15px/1.65 system-ui;max-width:1250px;margin:32px auto;padding:0 18px;color:#18283b}',
             'table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border-bottom:1px solid #d5dce5;padding:9px;text-align:left}',
             'th{background:#edf2f8}pre{white-space:pre-wrap;background:#f3f6fa;padding:14px}summary{cursor:pointer}',
             '.note{background:#fff5d9;padding:14px}.gold{background:#e7f5eb}small{color:#607086}</style>',
             '<h1>Q10：master 选择候选答案，再扩展问题 query</h1>',
             f'<p>{esc(result["question"])}</p>',
             '<p>25 个 master 各自用保存的 CLIP 视觉均值选择四个候选描述中余弦最高的一项。',
             '该描述追加到原问题，BGE 只检索本簇的原始对话＋完整 caption；汇总全部 89 个叶子后直接取 Top5。',
             '本轮不使用距离、不改变叶子证据、不融合视觉分数。</p>',
             '<p class="note">扩展内容是候选假设，不是观察事实。不同 master 的 BGE 分数来自不同 query；',
             '按本轮定义直接排序，分数不代表校准后的概率。单题结果不代表总体效果。</p>',
             '<h2>预先固定的候选描述</h2><table><tr><th>选项</th><th>原选项</th><th>CLIP 候选文本</th></tr>']
    for choice, statement in CANDIDATES.items():
        parts.append(f'<tr><td>{choice}</td><td>{esc(result["options"][choice])}</td><td>{esc(statement)}</td></tr>')
    parts.append('</table><h2>检索与回答对照</h2>')
    reader_path = output / 'reader/summary.json'
    if reader_path.exists():
        parts.append('<p>下面保留 reader 原始结果，比较只使用相同模型下新生成的回答。</p><pre>' +
                     esc(json.dumps(read_json(reader_path), ensure_ascii=False, indent=2)) + '</pre>')
    else:
        parts.append('<p>检索已完成，回答模型验证待写入。</p>')
    for condition in result['conditions']:
        parts.append(f'<h3>{esc(condition["name"])}</h3><p>目标在全库的名次：{condition["gold_rank"]}</p>')
        parts.append('<table><tr><th>名次</th><th>图片</th><th>Master</th><th>共享假设</th><th>BGE 余弦</th><th>目标</th></tr>')
        for row in condition['top5']:
            parts.append(f'<tr class="{"gold" if row["is_gold"] else ""}"><td>{row["rank"]}</td><td>{esc(row["filename"])}</td>'
                         f'<td>{esc(row["master_id"])}</td><td>{esc(row["master_choice"])}</td><td>{row["score"]:.8f}</td>'
                         f'<td>{"是" if row["is_gold"] else ""}</td></tr>')
        parts.append('</table>')
        for leaf in condition['reader_memories']:
            parts.append(f'<details><summary>{esc(leaf["filename"])} · 原始对话和 caption</summary>'
                         f'<pre>{esc(leaf["reader_text"])}</pre></details>')
    parts.append('<h2>全部 master 的视觉选择</h2><table><tr><th>Master</th><th>成员数</th><th>A</th><th>B</th><th>C</th><th>D</th><th>选择</th><th>前两名差值</th></tr>')
    for row in result['masters']:
        scores = ''.join(f'<td>{row["clip_candidate_scores"][k]:.6f}</td>' for k in CANDIDATES)
        parts.append(f'<tr><td>{esc(row["master_id"])}</td><td>{row["n"]}</td>{scores}<td>{row["selected_choice"]}</td>'
                     f'<td>{row["top_two_margin"]:.8f}</td></tr>')
    parts.append('</table><h2>逐簇 query 和逐叶得分</h2>')
    for master in result['masters']:
        parts.append(f'<details><summary>{esc(master["master_id"])} · {master["selected_choice"]} · {master["n"]} 个叶子</summary>'
                     f'<pre>{esc(master["augmented_query"])}</pre><table><tr><th>图片</th><th>原query得分</th><th>扩展query得分</th><th>新名次</th></tr>')
        for leaf in result['leaves']:
            if leaf['master_id'] == master['master_id']:
                parts.append(f'<tr><td>{esc(leaf["filename"])}</td><td>{leaf["baseline_score"]:.8f}</td>'
                             f'<td>{leaf["expanded_score"]:.8f}</td><td>{leaf["expanded_rank"]}</td></tr>')
        parts.append('</table></details>')
    parts.append('<p><a href="protocol.json">实验口径</a> · <a href="master_choices.csv">25 簇选择</a> · '
                 '<a href="all_scores.csv">89 条检索分数</a> · <a href="validation_independent.json">独立核验</a></p></html>')
    (output / 'report.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.render_only:
        render(output)
        return
    if (output / 'retrieval_results.json').exists():
        raise FileExistsError('Use a fresh output directory.')
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    inputs, source_protocol = read_json(SOURCE / 'inputs.json'), read_json(SOURCE / 'protocol.json')
    graph = read_json(SOURCE / 'memory/graph.json')
    features = np.load(SOURCE / 'features.npz', allow_pickle=False)
    questions = inputs['questions']
    qi = next(i for i, q in enumerate(questions) if q['id'] == QID)
    question = questions[qi]
    assert not question['query_image_paths']
    assert question['options'] == {'A': 'Both desks', 'B': 'Neither desk', 'C': 'The window desk',
                                  'D': 'The longer desk on the right wall'}
    leaves, masters = inputs['leaves'], inputs['masters']
    assert (len(leaves), len(masters)) == (89, 25)
    assert [m['id'] for m in masters] == [m['id'] for m in graph['masters']]
    assert [leaf['image_id'] for leaf in leaves] == [node['id'] for node in graph['image_nodes']]
    for leaf, node in zip(leaves, graph['image_nodes']):
        assert leaf['retrieval_text'] == node['dialogue'] + '\nMemory description: ' + node['caption']
    source_paths = [SOURCE / name for name in ('inputs.json', 'protocol.json', 'features.npz',
                                             'memory/graph.json', 'memory/master_sums.npy', 'memory/basis.npy')]
    source_paths.append(Path(__file__))
    hashes = {str(path): sha(path) for path in source_paths}
    queries = [question['question'] + '\nCandidate answer: ' + text for text in CANDIDATES.values()]
    write_json(output / 'candidates.json', {
        'question': question['question'], 'options': question['options'],
        'candidate_statements': CANDIDATES, 'augmented_queries': dict(zip(CANDIDATES, queries)),
        'construction': 'Parallel has/no-monitor statements from question and options only; fixed before encoding.',
        'gold_or_memory_text_used_to_construct_candidates': False,
    })
    protocol = {
        'question_id': QID, 'n_masters': 25, 'n_leaves': 89, 'top_k': 5,
        'master_visual_state': 'Actual saved approximate mean, not oracle or per-image reconstruction.',
        'master_choice': 'argmax CLIP cosine among A/B/C/D; ties keep option order; no threshold.',
        'query_expansion': 'original question + newline + Candidate answer: + selected statement',
        'leaf_retrieval_text': 'Original dialogue + newline + Memory description: + full caption',
        'global_ranking': 'Raw BGE cosine from each leaf own master query, descending; ties by original leaf index.',
        'global_score_caveat': 'Different masters can use different queries; no probability calibration claimed.',
        'leaf_distance_used': False, 'visual_2d_coordinates_used': False,
        'old_image_reencoding': False, 'leaf_visual_embeddings_saved': False,
        'options_used': True, 'gold_used_for_scoring_or_candidate_selection': False,
        'reader_policy': 'Original question/options and original dialogue/full-caption evidence only; expanded query and hypotheses are retrieval-only.',
        'candidate_variants_tried': 1, 'query_prefix': source_protocol['query_prefix'],
        'source_sha256': hashes, 'candidates_sha256': sha(output / 'candidates.json'),
        'image_encoder': source_protocol['image_encoder'], 'text_encoder': source_protocol['text_encoder'],
    }
    write_json(output / 'protocol.json', protocol)

    import torch
    import transformers
    import sentence_transformers
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    clip_meta = source_protocol['image_encoder']
    clip_path = Path(clip_meta['model_name'])
    for name, expected in {**clip_meta['files_sha256'], **TOKENIZER_HASHES}.items():
        assert sha(clip_path / name) == expected, name
    config = AutoConfig.from_pretrained(str(clip_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    clip_lengths = [len(x) for x in tokenizer(list(CANDIDATES.values()), truncation=False)['input_ids']]
    assert max(clip_lengths) <= config.text_config.max_position_embeddings
    clip_model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(clip_path), config=config.text_config, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs'))
    assert all(k.startswith(('vision_model.', 'visual_projection.')) or k == 'logit_scale'
               for k in loading.get('unexpected_keys', []))
    clip_model.to(device='cpu', dtype=torch.float32).eval()
    encoded = tokenizer(list(CANDIDATES.values()), padding=True, return_tensors='pt', truncation=False)
    with torch.inference_mode():
        clip_candidates = normalize(clip_model(**encoded, return_dict=True).text_embeds.double().numpy())
    sums = np.load(SOURCE / 'memory/master_sums.npy', allow_pickle=False)
    means = []
    for master in graph['masters']:
        ref = master['visual_mean_ref']
        assert ref['path'] == 'master_sums.npy' and abs(ref['scale'] - 1/master['n']) < 1e-15
        means.append(sums[ref['row']] * ref['scale'])
    clip_scores = normalize(means) @ clip_candidates.T
    selected_choices = np.argmax(clip_scores, axis=1)
    del clip_model

    text_meta = source_protocol['text_encoder']
    text_path = Path(text_meta['model_path'])
    for name, expected in text_meta['files_sha256'].items():
        assert sha(text_path / name) == expected, name
    text_model = SentenceTransformer(str(text_path), device='cpu', local_files_only=True)
    query_inputs = [source_protocol['query_prefix'] + query for query in queries]
    query_lengths = [len(x) for x in text_model.tokenizer(query_inputs, truncation=False)['input_ids']]
    assert max(query_lengths) <= text_model.max_seq_length
    augmented_vectors = normalize(text_model.encode(query_inputs, batch_size=16,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt=''))
    del text_model
    # Cached canonical question and leaf vectors keep the full-scan control exact.
    baseline_query, leaf_vectors = features['bge_query'][qi], features['bge_leaf']
    scores_by_query = np.vstack((baseline_query, augmented_vectors)) @ leaf_vectors.T
    baseline_scores = scores_by_query[0]
    master_lookup = {m['id']: i for i, m in enumerate(masters)}
    leaf_master = np.array([master_lookup[leaf['master_id']] for leaf in leaves])
    expanded_scores = scores_by_query[1 + selected_choices[leaf_master], np.arange(len(leaves))]
    baseline_order = np.argsort(-baseline_scores, kind='stable')
    expanded_order = np.argsort(-expanded_scores, kind='stable')
    baseline_ranks = np.empty(len(leaves), dtype=int); baseline_ranks[baseline_order] = np.arange(1, 90)
    expanded_ranks = np.empty(len(leaves), dtype=int); expanded_ranks[expanded_order] = np.arange(1, 90)

    # Gold labels enter evaluation only after all scores and orderings exist.
    gold = set(question['gold_ids'])
    letters = list(CANDIDATES)
    master_rows = []
    for mi, master in enumerate(masters):
        choice = letters[selected_choices[mi]]
        descending = np.sort(clip_scores[mi])[::-1]
        master_rows.append({
            'master_id': master['id'], 'n': master['n'], 'member_ids': master['member_ids'],
            'clip_candidate_scores': dict(zip(letters, clip_scores[mi].tolist())),
            'selected_choice': choice, 'selected_statement': CANDIDATES[choice],
            'top_two_margin': float(descending[0] - descending[1]),
            'augmented_query': queries[selected_choices[mi]],
            'contains_gold': any(leaves[i]['memory_id'] in gold for i in np.flatnonzero(leaf_master == mi)),
        })
    leaf_rows = []
    for i, leaf in enumerate(leaves):
        leaf_rows.append({
            'leaf_index': i, 'image_id': leaf['image_id'], 'memory_id': leaf['memory_id'],
            'filename': leaf['filename'], 'master_id': leaf['master_id'],
            'master_choice': letters[selected_choices[leaf_master[i]]],
            'baseline_score': float(baseline_scores[i]), 'expanded_score': float(expanded_scores[i]),
            'baseline_rank': int(baseline_ranks[i]), 'expanded_rank': int(expanded_ranks[i]),
            'is_gold': leaf['memory_id'] in gold,
        })
    conditions = []
    for name, order, scores, ranks in (
        ('baseline_question_only', baseline_order, baseline_scores, baseline_ranks),
        ('master_answer_query', expanded_order, expanded_scores, expanded_ranks),
    ):
        selected = order[:5].tolist()
        memories = [{'memory_id': leaves[i]['memory_id'], 'filename': leaves[i]['filename'],
                     'reader_text': leaves[i]['retrieval_text']} for i in selected]
        conditions.append({
            'name': name, 'top5_memory_ids': [leaves[i]['memory_id'] for i in selected],
            'top5_scores': scores[selected].tolist(), 'top5_leaf_indices': selected,
            'top5': [{**leaf_rows[i], 'rank': k, 'score': float(scores[i])} for k, i in enumerate(selected, 1)],
            'reader_memories': memories,
            'reader_context': '\n\n'.join(f'Memory {k}: {row["reader_text"]}' for k, row in enumerate(memories, 1)),
            'gold_rank': min(int(ranks[i]) for i, leaf in enumerate(leaves) if leaf['memory_id'] in gold),
            'gold_hits_top5': sum(leaves[i]['memory_id'] in gold for i in selected),
        })
    result = {
        'question_id': QID, 'question': question['question'], 'options': question['options'],
        'query_image_paths': question['query_image_paths'], 'candidate_statements': CANDIDATES,
        'evaluation': {'gold_choice': question['gold_choice'], 'gold_ids': question['gold_ids']},
        'conditions': conditions, 'masters': master_rows, 'leaves': leaf_rows,
        'master_choice_counts': {choice: sum(r['selected_choice'] == choice for r in master_rows) for choice in letters},
        'top5_same_order': conditions[0]['top5_memory_ids'] == conditions[1]['top5_memory_ids'],
    }
    np.savez(output / 'score_matrices.npz', clip_candidate_vectors=clip_candidates,
             clip_master_candidate=clip_scores, selected_choice_indices=selected_choices,
             bge_augmented_query_vectors=augmented_vectors, bge_all_query_leaf=scores_by_query,
             selected_leaf_scores=expanded_scores, leaf_master_indices=leaf_master)
    write_json(output / 'retrieval_results.json', result)
    csv_write(output / 'all_scores.csv', leaf_rows)
    csv_write(output / 'master_choices.csv', [{
        'master_id': row['master_id'], 'n': row['n'], **row['clip_candidate_scores'],
        'selected_choice': row['selected_choice'], 'top_two_margin': row['top_two_margin'],
        'contains_gold': row['contains_gold'], 'augmented_query': row['augmented_query'],
    } for row in master_rows])
    protocol.update(
        clip_candidate_token_counts=clip_lengths, clip_token_limit=config.text_config.max_position_embeddings,
        bge_augmented_query_token_counts=query_lengths, bge_token_limit=text_meta['max_seq_length'],
        truncated_inputs=0, clip_candidate_batch=4, bge_augmented_query_batch=4,
        cached_baseline_and_leaf_vectors=True, baseline_source_question_index=qi,
        versions={'torch': torch.__version__, 'transformers': transformers.__version__,
                  'sentence_transformers': sentence_transformers.__version__, 'numpy': np.__version__},
        elapsed_seconds=time.perf_counter() - start,
    )
    assert all(sha(path) == expected for path, expected in hashes.items())
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'validation.json', {
        'passed': True, 'masters': len(master_rows), 'leaves': len(leaf_rows),
        'all_master_argmax_verified': True, 'all_leaf_scores_select_own_master_query': True,
        'source_hashes_unchanged': True, 'no_token_truncation': True,
        'candidate_hypotheses_not_added_to_reader': True,
    })
    render(output)
    print(json.dumps({'output': str(output), 'choice_counts': result['master_choice_counts'],
                      'top5': [{k: c[k] for k in ('name', 'top5_memory_ids', 'top5_scores', 'gold_rank')}
                               for c in conditions]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
