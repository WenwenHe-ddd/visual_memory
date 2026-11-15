#!/usr/bin/env python3
"""Frozen all-domain test: master visual answer choice expands its members' query.

Candidate statements are authored from questions/options, then frozen before any
similarity is calculated. Gold labels are used only after retrieval. All reader
evidence remains the exact original dialogue and full caption.
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

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
OUTPUT = ROOT / 'reports/all_master_answer_query_20261021'
ORIGINAL_QA = ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'
LETTERS = 'ABCD'
CONDITIONS = ('baseline_question_only', 'master_answer_query')


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def csv_write(path, rows):
    with Path(path).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    denominator = sum(len(row['evaluation']['gold_ids']) for row in rows)
    result = {'questions': len(rows), 'gold_references': denominator}
    for index, name in enumerate(CONDITIONS):
        conditions = [row['conditions'][index] for row in rows]
        hits = sum(row['gold_hits_top5'] for row in conditions)
        result[name] = {
            'gold_hits_top5': hits, 'gold_recall': hits / denominator,
            'questions_with_gold_hit': sum(row['gold_hits_top5'] > 0 for row in conditions),
            'mean_first_gold_rank': float(np.mean([row['gold_rank'] for row in conditions])),
        }
    result['identical_ordered_top5'] = sum(row['top5_same_order'] for row in rows)
    result['identical_top5_set'] = sum(set(row['conditions'][0]['top5_memory_ids']) ==
                                       set(row['conditions'][1]['top5_memory_ids']) for row in rows)
    result['mean_top5_overlap'] = float(np.mean([
        len(set(row['conditions'][0]['top5_memory_ids']) & set(row['conditions'][1]['top5_memory_ids'])) / 5
        for row in rows]))
    return result


def run(output):
    if (output / 'retrieval_results.jsonl').exists():
        raise FileExistsError('Use a fresh directory; existing scored results must not be overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / 'candidates.jsonl'
    candidates = read_lines(candidate_path)
    candidate_by_id = {row['id']: row for row in candidates}
    assert len(candidate_by_id) == len(candidates) == 371
    original = {row['id']: row for row in read_lines(ORIGINAL_QA)}
    baseline_reference = {}
    domains = {}
    sources = [candidate_path, ORIGINAL_QA, Path(__file__)]
    all_questions, all_statements, all_queries = [], [], []
    for directory in sorted(SOURCE.iterdir()):
        if not (directory / 'inputs.json').exists():
            continue
        inputs = read_json(directory / 'inputs.json')
        graph = read_json(directory / 'memory/graph.json')
        protocol = read_json(directory / 'protocol.json')
        full_scan_path = SOURCE.parent / 'all_empirical' / directory.name / 'runs.json'
        for row in read_json(full_scan_path):
            if row['L'] == 5 and row['method'] == 'full_scan':
                assert row['id'] not in baseline_reference
                baseline_reference[row['id']] = row['selected_memory_ids']
        with np.load(directory / 'features.npz', allow_pickle=False) as archive:
            features = {name: archive[name] for name in archive.files}
        assert [r['id'] for r in graph['masters']] == [r['id'] for r in inputs['masters']]
        assert [r['id'] for r in graph['image_nodes']] == [r['image_id'] for r in inputs['leaves']]
        for leaf, node in zip(inputs['leaves'], graph['image_nodes']):
            assert leaf['retrieval_text'] == node['dialogue'] + '\nMemory description: ' + node['caption']
        sums = np.load(directory / 'memory/master_sums.npy', allow_pickle=False)
        centers = []
        for master in graph['masters']:
            ref = master['visual_mean_ref']
            assert ref['path'] == 'master_sums.npy'
            assert abs(ref['scale'] - 1 / master['n']) < 1e-15
            centers.append(sums[ref['row']] * ref['scale'])
        sources += [directory / name for name in (
            'inputs.json', 'protocol.json', 'features.npz', 'memory/graph.json', 'memory/master_sums.npy')]
        sources.append(full_scan_path)
        domains[directory.name] = (inputs, graph, protocol, features, normalize(centers))
        for question in inputs['questions']:
            row = candidate_by_id[question['id']]
            statements = row['candidate_statements']
            assert set(statements) == set(LETTERS)
            assert all(isinstance(statements[k], str) and statements[k].strip() for k in LETTERS)
            for field in ('question', 'options', 'query_image_paths'):
                assert question[field] == original[question['id']][field], (question['id'], field)
            all_questions.append((directory.name, question['id']))
            all_statements.extend(statements[k] for k in LETTERS)
            all_queries.extend(question['retrieval_query'] + '\nCandidate answer: ' + statements[k]
                               for k in LETTERS)
    assert len(all_questions) == 371 and len(domains) == 8
    assert set(candidate_by_id) == {qid for _, qid in all_questions} == set(original)
    assert set(baseline_reference) == set(original)
    source_hashes = {str(path): sha(path) for path in sources}
    reference = next(iter(domains.values()))[2]
    for inputs, graph, protocol, features, centers in domains.values():
        assert protocol['image_encoder'] == reference['image_encoder']
        assert protocol['query_prefix'] == reference['query_prefix']
        assert protocol['text_encoder']['model_path'] == reference['text_encoder']['model_path']
        assert protocol['text_encoder']['files_sha256'] == reference['text_encoder']['files_sha256']
    protocol = {
        'design': 'All 371 questions, paired original full-scan vs master-selected candidate query expansion.',
        'domains': 8, 'questions': 371, 'top_k': 5,
        'candidate_construction': 'One frozen assistant-authored statement for each option, using only question and options; Q10 uses its already frozen four statements.',
        'candidate_variants_scored': 1,
        'master_choice': 'Highest CLIP cosine of actual saved normalized master visual mean with four candidate statements; ties use ABCD order.',
        'query_expansion': 'Complete baseline retrieval_query (including cached query-image caption when present) + newline + Candidate answer: + selected statement.',
        'leaf_text': 'Original dialogue + newline + Memory description: + full caption; unchanged.',
        'global_ranking': 'Pool raw BGE cosine for each leaf using only its own master selected query; stable ties by original leaf index.',
        'manual_similarity_thresholds': 0, 'score_fusion_weights': 0,
        'distance_used': False, 'leaf_2d_used': False, 'old_image_reencoding': False,
        'individual_image_embeddings_used_or_saved': False,
        'query_image_policy': 'Preserve cached caption in BGE query and original pixels in reader. No extra query-image feature is added to CLIP candidate selection.',
        'reader_policy': 'Original question/options, original query image channel, original dialogue/full captions only. Candidate hypotheses are retrieval-only.',
        'baseline_reference': 'Frozen adaptive all_empirical full_scan L=5, using original dialogue+full-caption leaf vectors.',
        'historical_qa_manifest_role': 'Question/options/query images/labels only. Its earlier dialogue+unique-caption retrieval order is not this baseline.',
        'source_sha256': source_hashes, 'query_prefix': reference['query_prefix'],
        'image_encoder': reference['image_encoder'], 'text_encoder': reference['text_encoder'],
        'limitations': ['BGE scores across masters can arise from different queries and are not calibrated probabilities.',
                       'Candidate options are available at retrieval time; this experiment does not test open-ended answer generation.',
                       'Candidate wording is frozen for this evaluation; no automatic production candidate generator is evaluated.'],
    }
    write_json(output / 'protocol.json', protocol)
    print(json.dumps({'stage': 'frozen', 'questions': 371, 'candidate_statements': len(all_statements)}), flush=True)

    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    started = time.perf_counter()
    clip_meta = reference['image_encoder']
    clip_path = Path(clip_meta['model_name'])
    for name, expected in {**clip_meta['files_sha256'], **TOKENIZER_HASHES}.items():
        assert sha(clip_path / name) == expected, name
    config = AutoConfig.from_pretrained(str(clip_path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(clip_path), local_files_only=True)
    clip_lengths = [len(row) for row in tokenizer(all_statements, truncation=False)['input_ids']]
    assert max(clip_lengths) <= config.text_config.max_position_embeddings, max(clip_lengths)
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(clip_path), config=config.text_config, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs'))
    assert all(key.startswith(('vision_model.', 'visual_projection.')) or key == 'logit_scale'
               for key in loading.get('unexpected_keys', []))
    model.to(device='cpu', dtype=torch.float32).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(all_statements), 32):
            encoded = tokenizer(all_statements[start:start + 32], padding=True, truncation=False, return_tensors='pt')
            vectors.append(model(**encoded, return_dict=True).text_embeds.double().numpy())
    clip_vectors = normalize(np.concatenate(vectors))
    del model
    print(json.dumps({'stage': 'clip_encoded', 'max_tokens': max(clip_lengths)}), flush=True)
    text_meta = reference['text_encoder']
    text_path = Path(text_meta['model_path'])
    for name, expected in text_meta['files_sha256'].items():
        assert sha(text_path / name) == expected, name
    model = SentenceTransformer(str(text_path), device='cpu', local_files_only=True)
    query_inputs = [reference['query_prefix'] + query for query in all_queries]
    query_lengths = [len(row) for row in model.tokenizer(query_inputs, truncation=False)['input_ids']]
    assert max(query_lengths) <= model.max_seq_length, max(query_lengths)
    bge_vectors = normalize(model.encode(query_inputs, batch_size=16, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, prompt=''))
    del model
    print(json.dumps({'stage': 'bge_encoded', 'max_tokens': max(query_lengths)}), flush=True)
    offset = 0
    results, csv_rows = [], []
    baseline_matches = 0
    q10_check = None
    for domain, (inputs, graph, domain_protocol, features, centers) in domains.items():
        leaves, masters, questions = inputs['leaves'], inputs['masters'], inputs['questions']
        master_lookup = {master['id']: index for index, master in enumerate(masters)}
        leaf_master = np.asarray([master_lookup[leaf['master_id']] for leaf in leaves])
        assert sorted(mid for master in masters for mid in master['member_ids']) == sorted(leaf['image_id'] for leaf in leaves)
        clip_domain = clip_vectors[offset:offset + 4 * len(questions)].reshape(len(questions), 4, -1)
        bge_domain = bge_vectors[offset:offset + 4 * len(questions)].reshape(len(questions), 4, -1)
        clip_scores = np.einsum('md,qcd->qmc', centers, clip_domain)
        choices = np.argmax(clip_scores, axis=2)
        all_scores = np.concatenate((features['bge_query'][:, None, :], bge_domain), axis=1) @ features['bge_leaf'].T
        selected = np.stack([all_scores[qi, 1 + choices[qi, leaf_master], np.arange(len(leaves))]
                             for qi in range(len(questions))])
        domain_output = output / 'domains' / domain
        domain_output.mkdir(parents=True, exist_ok=True)
        np.savez(domain_output / 'score_matrices.npz',
            clip_candidate_vectors=clip_domain, clip_master_candidate=clip_scores,
            selected_choice_indices=choices, bge_augmented_query_vectors=bge_domain,
            bge_all_query_leaf=all_scores, selected_leaf_scores=selected, leaf_master_indices=leaf_master)
        for qi, question in enumerate(questions):
            statements = candidate_by_id[question['id']]['candidate_statements']
            baseline_scores, expanded_scores = all_scores[qi, 0], selected[qi]
            baseline_order = np.argsort(-baseline_scores, kind='stable')
            expanded_order = np.argsort(-expanded_scores, kind='stable')
            orders = (baseline_order, expanded_order)
            scores = (baseline_scores, expanded_scores)
            ranks = []
            for order in orders:
                rank = np.empty(len(leaves), dtype=int)
                rank[order] = np.arange(1, len(leaves) + 1)
                ranks.append(rank)
            # All candidate choices and score orderings are complete before labels are used.
            gold = set(question['gold_ids'])
            assert len(gold) == len(question['gold_ids']) and gold <= {leaf['memory_id'] for leaf in leaves}
            leaf_rows = [{
                'leaf_index': i, 'image_id': leaf['image_id'], 'memory_id': leaf['memory_id'],
                'filename': leaf['filename'], 'master_id': leaf['master_id'],
                'master_choice': LETTERS[choices[qi, leaf_master[i]]],
                'baseline_score': float(baseline_scores[i]), 'expanded_score': float(expanded_scores[i]),
                'baseline_rank': int(ranks[0][i]), 'expanded_rank': int(ranks[1][i]),
                'is_gold': leaf['memory_id'] in gold,
            } for i, leaf in enumerate(leaves)]
            conditions = []
            for name, order, score, rank in zip(CONDITIONS, orders, scores, ranks):
                indices = order[:5].tolist()
                memories = [{'memory_id': leaves[i]['memory_id'], 'filename': leaves[i]['filename'],
                             'reader_text': leaves[i]['retrieval_text']} for i in indices]
                conditions.append({
                    'name': name, 'top5_memory_ids': [leaves[i]['memory_id'] for i in indices],
                    'top5_scores': score[indices].tolist(), 'top5_leaf_indices': indices,
                    'top5': [{**leaf_rows[i], 'rank': k, 'score': float(score[i])} for k, i in enumerate(indices, 1)],
                    'reader_memories': memories,
                    'reader_context': '\n\n'.join(f'Memory {k}: {row["reader_text"]}' for k, row in enumerate(memories, 1)),
                    'gold_rank': min(int(rank[i]) for i, leaf in enumerate(leaves) if leaf['memory_id'] in gold),
                    'gold_hits_top5': sum(leaves[i]['memory_id'] in gold for i in indices),
                })
            baseline_matches += conditions[0]['top5_memory_ids'] == baseline_reference[question['id']]
            master_rows = []
            for mi, master in enumerate(masters):
                choice = LETTERS[choices[qi, mi]]
                sorted_scores = np.sort(clip_scores[qi, mi])[::-1]
                master_rows.append({
                    'master_id': master['id'], 'n': master['n'], 'member_ids': master['member_ids'],
                    'clip_candidate_scores': dict(zip(LETTERS, clip_scores[qi, mi].tolist())),
                    'selected_choice': choice, 'selected_statement': statements[choice],
                    'top_two_margin': float(sorted_scores[0] - sorted_scores[1]),
                    'augmented_query': question['retrieval_query'] + '\nCandidate answer: ' + statements[choice],
                    'contains_gold': any(leaves[i]['memory_id'] in gold for i in np.flatnonzero(leaf_master == mi)),
                })
            result = {
                'question_id': question['id'], 'task': domain, 'question': question['question'],
                'options': question['options'], 'query_image_paths': question['query_image_paths'],
                'baseline_retrieval_query': question['retrieval_query'],
                'evaluation': {'gold_choice': question['gold_choice'], 'gold_ids': question['gold_ids']},
                'candidate_statements': statements, 'conditions': conditions,
                'masters': master_rows, 'leaves': leaf_rows,
                'master_choice_counts': {k: sum(row['selected_choice'] == k for row in master_rows) for k in LETTERS},
                'top5_same_order': conditions[0]['top5_memory_ids'] == conditions[1]['top5_memory_ids'],
            }
            if question['id'] == 'Home_Renovation_Interior_Design/Q10':
                prior = read_json(ROOT / 'reports/home_q10_master_answer_query_20261021/retrieval_results.json')
                assert statements == prior['candidate_statements']
                assert [c['top5_memory_ids'] for c in conditions] == [c['top5_memory_ids'] for c in prior['conditions']]
                assert [m['selected_choice'] for m in master_rows] == [m['selected_choice'] for m in prior['masters']]
                q10_check = {'same_candidates': True, 'same_master_choices': True, 'same_ordered_top5': True,
                             'max_score_difference': max(abs(x-y) for a,b in zip(conditions,prior['conditions'])
                                                         for x,y in zip(a['top5_scores'],b['top5_scores']))}
            results.append(result)
            csv_rows.append({
                'id': question['id'], 'domain': domain, 'query_has_image': bool(question['query_image_paths']),
                'gold_count': len(gold), 'baseline_hits': conditions[0]['gold_hits_top5'],
                'expanded_hits': conditions[1]['gold_hits_top5'], 'baseline_first_gold_rank': conditions[0]['gold_rank'],
                'expanded_first_gold_rank': conditions[1]['gold_rank'], 'same_ordered_top5': result['top5_same_order'],
                'top5_overlap': len(set(conditions[0]['top5_memory_ids']) & set(conditions[1]['top5_memory_ids'])) / 5,
                'baseline_top5': '|'.join(conditions[0]['top5_memory_ids']),
                'expanded_top5': '|'.join(conditions[1]['top5_memory_ids']),
            })
        offset += 4 * len(questions)
        print(json.dumps({'stage': 'domain_ranked', 'domain': domain, 'questions': len(questions)}), flush=True)
    assert offset == len(all_statements) and len(results) == 371 and q10_check
    assert baseline_matches == 371, baseline_matches
    assert all(sha(path) == expected for path, expected in source_hashes.items())
    summary = {'overall': metrics(results),
               'by_domain': {domain: metrics([row for row in results if row['task'] == domain]) for domain in domains},
               'by_query_image': {name: metrics([row for row in results if bool(row['query_image_paths']) == flag])
                                  for name, flag in [('text_only', False), ('with_image', True)]}}
    protocol.update(clip_candidate_max_tokens=max(clip_lengths), clip_token_limit=77,
                    bge_query_max_tokens=max(query_lengths), bge_token_limit=text_meta['max_seq_length'],
                    truncated_inputs=0, candidate_token_counts=clip_lengths,
                    baseline_and_leaf_vectors_reused=True, baseline_top5_matches_frozen_full_scan=baseline_matches,
                    elapsed_seconds=time.perf_counter() - started,
                    versions={'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__})
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'retrieval_summary.json', summary)
    write_json(output / 'validation_retrieval.json', {
        'passed': True, 'questions': len(results), 'domains': len(domains),
        'unique_question_ids': len({row['question_id'] for row in results}),
        'baseline_top5_matches_frozen_full_scan': baseline_matches, 'source_hashes_unchanged': True,
        'leaf_texts_unchanged': True, 'all_leaf_scores_use_own_master_query': True,
        'gold_labels_used_after_retrieval': True, 'no_truncation': True, 'q10_reproduction': q10_check,
    })
    csv_write(output / 'retrieval_per_question.csv', csv_rows)
    with (output / 'retrieval_results.jsonl').open('w') as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    render(output)
    print(json.dumps({'stage': 'complete', 'summary': summary['overall']}), flush=True)


def render(output):
    rows = read_lines(output / 'retrieval_results.jsonl')
    summary = read_json(output / 'retrieval_summary.json')
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>全部问题 · master 答案扩展 query</title>',
             '<style>body{font:15px/1.6 system-ui;color:#233247;max-width:1300px;margin:30px auto;padding:0 20px}',
             'table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}',
             'th{background:#edf2f8}pre{white-space:pre-wrap;background:#f5f7fa;padding:12px}summary{cursor:pointer}',
             '.note{background:#fff2cf;padding:15px}details{margin:8px 0}</style>',
             '<h1>全部 371 题：master 候选答案扩展 query</h1>',
             '<p>8 领域、438 条记忆、88 个 master。每个 master 用 CLIP 选择候选陈述；',
             '将该陈述追加到该题完整检索 query，检索本簇所有原始对话＋完整 caption，再对全领域叶子取 Top5。</p>',
             '<p class="note">候选仅由问题及选项改写，固定后一次评分；Q10 沿用既有四句。无距离项、手动相似度阈值或融合权重。',
             '候选仅参与检索，不进入回答证据。8 道带图题保留原有 caption 检索输入和 reader 原图输入。</p>',
             '<h2>检索结果</h2><table><tr><th>领域</th><th>题数</th><th>原检索 gold 命中</th><th>扩展后 gold 命中</th><th>分母</th><th>Top5 顺序不变</th></tr>']
    for domain, stat in [*summary['by_domain'].items(), ('全部', summary['overall'])]:
        a, b = stat[CONDITIONS[0]], stat[CONDITIONS[1]]
        parts.append(f'<tr><td>{esc(domain)}</td><td>{stat["questions"]}</td><td>{a["gold_hits_top5"]}</td>'
                     f'<td>{b["gold_hits_top5"]}</td><td>{stat["gold_references"]}</td><td>{stat["identical_ordered_top5"]}</td></tr>')
    parts.append('</table><h2>新生成的配对回答</h2>')
    reader_summary = output / 'reader/summary.json'
    if reader_summary.exists():
        reader = read_json(reader_summary)
        parts.append('<table><tr><th>领域</th><th>题数</th><th>原检索正确</th><th>扩展后正确</th><th>准确率变化</th><th>错→对</th><th>对→错</th></tr>')
        for domain, stat in [*reader['by_domain'].items(), ('全部', reader['overall'])]:
            parts.append(f'<tr><td>{esc(domain)}</td><td>{stat["questions"]}</td>'
                         f'<td>{stat["baseline_correct"]} ({100*stat["baseline_accuracy"]:.2f}%)</td>'
                         f'<td>{stat["expanded_correct"]} ({100*stat["expanded_accuracy"]:.2f}%)</td>'
                         f'<td>{stat["delta_percentage_points"]:+.2f} 个百分点</td>'
                         f'<td>{stat["wrong_to_correct"]}</td><td>{stat["correct_to_wrong"]}</td></tr>')
        parts.append(f'</table><p>本轮新推理 {reader["fresh_model_inferences"]} 次；'
                     f'完全相同输入共享 {reader["same_run_shared_records"]} 条记录。'
                     f'格式错误 {reader["format_errors"]}；达到输出长度上限 {reader["token_limit_hits"]}。</p>')
        parts.append('<details><summary>完整回答统计与图文子集</summary><pre>' +
                     esc(json.dumps(reader, ensure_ascii=False, indent=2)) + '</pre></details>')
    else:
        parts.append('<p>回答测试尚未完成；上述检索指标不是回答准确率。</p>')
    responses_path = output / 'reader/responses.jsonl'
    responses = {(row['question_id'], row['condition']): row for row in read_lines(responses_path)} if responses_path.exists() else {}
    parts.append('<h2>逐题查看</h2>')
    for row in rows:
        a, b = row['conditions']
        answer_info = []
        for condition in CONDITIONS:
            response = responses.get((row['question_id'], condition))
            if response:
                answer_info.append(str(response['predicted_choice']))
        answers = ' · reader ' + ' → '.join(answer_info) if answer_info else ''
        parts.append(f'<details><summary>{esc(row["question_id"])} · 首个目标 {a["gold_rank"]} → {b["gold_rank"]}'
                     f' · gold {esc(row["evaluation"]["gold_choice"])}{esc(answers)}</summary><p>{esc(row["question"])}</p>')
        for key in LETTERS:
            parts.append(f'<p><b>{key}: {esc(row["options"][key])}</b><br>候选：{esc(row["candidate_statements"][key])}</p>')
        for condition in row['conditions']:
            parts.append(f'<h3>{esc(condition["name"])}</h3><table><tr><th>排名</th><th>图片</th><th>Master</th><th>候选</th><th>相似度</th><th>目标</th></tr>')
            for leaf in condition['top5']:
                parts.append(f'<tr><td>{leaf["rank"]}</td><td>{esc(leaf["filename"])}</td><td>{esc(leaf["master_id"])}</td>'
                             f'<td>{leaf["master_choice"]}</td><td>{leaf["score"]:.6f}</td><td>{"是" if leaf["is_gold"] else ""}</td></tr>')
            parts.append('</table>')
            for memory in condition['reader_memories']:
                parts.append(f'<details><summary>{esc(memory["filename"])} · 原对话和 caption</summary><pre>{esc(memory["reader_text"])}</pre></details>')
        parts.append('<details><summary>各 master 的四个候选分数及选择</summary><table><tr><th>Master</th><th>A</th><th>B</th><th>C</th><th>D</th><th>选择</th></tr>')
        for master in row['masters']:
            cells = ''.join(f'<td>{master["clip_candidate_scores"][k]:.6f}</td>' for k in LETTERS)
            parts.append(f'<tr><td>{esc(master["master_id"])}</td>{cells}<td>{master["selected_choice"]}</td></tr>')
        parts.append('</table></details></details>')
    parts.append('<p><a href="protocol.json">实验设置</a> · <a href="retrieval_per_question.csv">逐题检索 CSV</a> · '
                 '<a href="reader/summary.json">回答汇总</a></p></html>')
    (output / 'report.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    render(args.output) if args.render_only else run(args.output)
