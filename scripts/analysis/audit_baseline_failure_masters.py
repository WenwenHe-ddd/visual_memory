#!/usr/bin/env python3
"""Audit frozen MemEye full-scan failures against final caption-Ward masters.

No model calls or memory rebuilds: recompute full-scan scores from frozen BGE
vectors, validate every rank and final graph membership, then join gold labels.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'reports/all_master_answer_query_20261021'
DEFAULT_INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
LABELS = {
    'top1_wrong': 'Top-1 为非目标',
    'zero_hit_top5': 'Top-5 完全未命中目标',
    'incomplete_top5': 'Top-5 未召回全部目标',
    'reader_wrong': '最终回答错误（单独统计）',
}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), f'Duplicate {key}'
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    assert rows
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'ratio': numerator / denominator if denominator else None}


def fmt(value):
    if not value['denominator']:
        return '0/0（无适用样本）'
    return f"{value['numerator']}/{value['denominator']} = {100 * value['ratio']:.2f}%"


def aggregate(cases):
    out = {'questions': len(cases), 'gold_references': sum(c['gold_count'] for c in cases),
           'gold_hits_top5': sum(c['gold_hits_top5'] for c in cases),
           'questions_with_more_than_5_gold': sum(c['gold_count'] > 5 for c in cases)}
    for label in LABELS:
        subset = [c for c in cases if c[label]]
        # For incomplete retrieval, relate distractors specifically to MISSED
        # targets. Other case definitions use all gold targets.
        key = 'any_distractor_shares_missing_target_master' if label == 'incomplete_top5' else 'any_distractor_shares_target_master'
        if label == 'top1_wrong':
            key = 'top1_shares_target_master'
        out[label] = {
            'failures': len(subset), 'failure_rate': fraction(len(subset), len(cases)),
            'same_master_cases': fraction(sum(c[key] for c in subset), len(subset)),
            'with_top5_distractors': sum(bool(c['distractor_count']) for c in subset),
            'any_top5_distractor_vs_any_target': fraction(sum(c['any_distractor_shares_target_master'] for c in subset), len(subset)),
            'all_top5_distractors_share_a_target_master': fraction(sum(c['all_distractors_share_target_master'] for c in subset), len(subset)),
        }
    missed = sum(c['missing_count'] for c in cases)
    out['missed_targets_with_same_master_distractor'] = fraction(
        sum(c['missing_targets_with_same_master_distractor'] for c in cases), missed)
    out['missed_target_distractor_pairs_same_master'] = fraction(
        sum(c['same_master_missed_pairs'] for c in cases),
        sum(c['missed_pairs'] for c in cases))
    out['incomplete_top5_gold_count_at_most_5'] = fraction(
        sum(c['incomplete_top5'] and c['any_distractor_shares_missing_target_master']
            for c in cases if c['gold_count'] <= 5),
        sum(c['incomplete_top5'] for c in cases if c['gold_count'] <= 5))
    out['reader_wrong_with_all_gold_retrieved'] = sum(c['reader_wrong'] and not c['incomplete_top5'] for c in cases)
    return out


def run(args):
    run_dir, input_dir, output = args.run.resolve(), args.inputs.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing audit: {output}')
    protocol = read(run_dir / 'protocol.json')
    recorded_hashes = protocol['source_sha256']
    tracked = {}

    def track(path, frozen=False):
        path = Path(path).resolve()
        digest = sha(path)
        if frozen:
            assert str(path) in recorded_hashes, f'No original hash: {path}'
            assert digest == recorded_hashes[str(path)], f'Frozen source changed: {path}'
        tracked[str(path)] = digest

    retrieval = unique(lines(run_dir / 'retrieval_results.jsonl'), 'question_id')
    original = unique(lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'), 'id')
    assert set(retrieval) == set(original) and len(retrieval) == 371
    manifests = unique(lines(run_dir / 'reader/qa_manifest.jsonl'), 'record_id')
    responses = unique(lines(run_dir / 'reader/responses.jsonl'), 'record_id')
    assert set(manifests) == set(responses) and len(manifests) == 742
    reader_protocol = read(run_dir / 'reader/protocol.json')
    for path in [run_dir / 'protocol.json', run_dir / 'retrieval_results.jsonl',
                 run_dir / 'retrieval_summary.json', run_dir / 'reader/qa_manifest.jsonl',
                 run_dir / 'reader/responses.jsonl', run_dir / 'reader/protocol.json',
                 ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl', Path(__file__)]:
        track(path)

    cases, details, target_pairs, domain_checks = [], [], [], []
    total_nodes = total_masters = total_scores = 0
    maximum_score_error = 0.0
    for domain in sorted(input_dir.iterdir()):
        if not (domain / 'inputs.json').exists():
            continue
        for path in [domain / 'inputs.json', domain / 'protocol.json',
                     domain / 'memory/graph.json', domain / 'features.npz']:
            track(path, frozen=True)
        inputs, graph = read(domain / 'inputs.json'), read(domain / 'memory/graph.json')
        assert graph['method'] == 'caption_ward_silhouette_visual'
        leaves, nodes, masters = inputs['leaves'], graph['image_nodes'], graph['masters']
        assert [x['image_id'] for x in leaves] == [x['id'] for x in nodes]
        unique(leaves, 'memory_id')  # This dataset has one image per memory.
        master_by_id = unique(masters, 'id')
        membership = {}
        for master in masters:
            assert master['n'] == len(master['member_ids'])
            for node_id in master['member_ids']:
                assert node_id not in membership
                membership[node_id] = master['id']
        assert set(membership) == {n['id'] for n in nodes}
        for leaf, node in zip(leaves, nodes):
            assert leaf['master_id'] == node['master_id'] == membership[node['id']]
            assert leaf['caption'] == node['caption'] and leaf['dialogue'] == node['dialogue']
            assert leaf['retrieval_text'] == node['dialogue'] + '\nMemory description: ' + node['caption']
        with np.load(domain / 'features.npz', allow_pickle=False) as features:
            scores = features['bge_query'] @ features['bge_leaf'].T
        full_scan_path = input_dir.parent / 'all_empirical' / domain.name / 'runs.json'
        track(full_scan_path)
        full_scan = unique([r for r in read(full_scan_path) if r['method'] == 'full_scan' and r['L'] == 5], 'id')
        total_nodes += len(nodes)
        total_masters += len(masters)
        total_scores += scores.size
        for qi, question in enumerate(inputs['questions']):
            qid = question['id']
            saved, source = retrieval[qid], original[qid]
            for key in ['question', 'options', 'gold_ids', 'gold_choice', 'query_image_paths']:
                assert question[key] == source[key]
            assert saved['question'] == question['question'] and saved['options'] == question['options']
            assert saved['evaluation']['gold_ids'] == question['gold_ids']
            assert saved['baseline_retrieval_query'] == question['retrieval_query']
            order = np.argsort(-scores[qi], kind='stable')
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            assert len(saved['leaves']) == len(leaves)
            for i, (leaf, scored) in enumerate(zip(leaves, saved['leaves'])):
                assert scored['memory_id'] == leaf['memory_id']
                assert scored['master_id'] == leaf['master_id']
                assert scored['baseline_rank'] == int(ranks[i])
                error = abs(scored['baseline_score'] - float(scores[qi, i]))
                maximum_score_error = max(maximum_score_error, error)
                assert error < 1e-12
            condition = next(c for c in saved['conditions'] if c['name'] == 'baseline_question_only')
            top_ids = [leaves[i]['memory_id'] for i in order[:5]]
            assert top_ids == condition['top5_memory_ids'] == full_scan[qid]['selected_memory_ids']
            gold = set(question['gold_ids'])
            assert gold and len(gold) == len(question['gold_ids'])
            assert gold <= {leaf['memory_id'] for leaf in leaves}
            hit, missing = gold & set(top_ids), gold - set(top_ids)
            assert len(hit) == condition['gold_hits_top5']

            def leaf_detail(i):
                leaf = leaves[i]
                return {'memory_id': leaf['memory_id'], 'image_id': leaf['image_id'],
                        'filename': leaf['filename'], 'master_id': leaf['master_id'],
                        'qualified_master_id': domain.name + '/' + leaf['master_id'],
                        'master_size': master_by_id[leaf['master_id']]['n'],
                        'rank': int(ranks[i]), 'score': float(scores[qi, i]),
                        'is_gold': leaf['memory_id'] in gold,
                        'caption': leaf['caption'], 'dialogue': leaf['dialogue']}

            targets = [leaf_detail(i) for i in order if leaves[i]['memory_id'] in gold]
            top = [leaf_detail(i) for i in order[:5]]
            distractors = [leaf for leaf in top if not leaf['is_gold']]
            missed_targets = [leaf for leaf in targets if leaf['memory_id'] in missing]
            target_masters = {leaf['master_id'] for leaf in targets}
            missing_masters = {leaf['master_id'] for leaf in missed_targets}
            distractor_masters = {leaf['master_id'] for leaf in distractors}

            record_id = qid + '::baseline_question_only'
            manifest, response = manifests[record_id], responses[record_id]
            assert manifest['top_memory_ids'] == top_ids and manifest['gold_ids'] == question['gold_ids']
            assert response['payload_sha256'] == manifest['payload_sha256']
            assert response['experiment_signature'] == reader_protocol['experiment_signature']
            assert response['condition'] == 'baseline_question_only' and response['question_id'] == qid
            assert response['gold_choice'] == question['gold_choice']
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', response['raw_response'].strip(), re.I)
            assert match and response['status'] == 'ok' and not response['hit_token_limit']
            answer = match.group(1).upper()
            assert answer == response['predicted_choice']
            assert response['correct'] == (answer == question['gold_choice'])
            assert not response['historical_response_reuse']
            pairs = []
            for target in targets:
                for distractor in distractors:
                    pair = {'question_id': qid, 'domain': domain.name,
                            'target_id': target['memory_id'], 'target_filename': target['filename'],
                            'target_rank': target['rank'], 'target_master': target['qualified_master_id'],
                            'target_missed_top5': target['rank'] > 5,
                            'distractor_id': distractor['memory_id'], 'distractor_filename': distractor['filename'],
                            'distractor_rank': distractor['rank'], 'distractor_master': distractor['qualified_master_id'],
                            'distractor_outranks_target': distractor['rank'] < target['rank'],
                            'same_master': target['master_id'] == distractor['master_id']}
                    pairs.append(pair)
            target_pairs.extend(pairs)
            case = {'question_id': qid, 'domain': domain.name, 'question': question['question'],
                    'gold_count': len(gold), 'gold_hits_top5': len(hit), 'missing_count': len(missing),
                    'distractor_count': len(distractors), 'top1_wrong': not top[0]['is_gold'],
                    'zero_hit_top5': not hit, 'incomplete_top5': bool(missing),
                    'reader_wrong': answer != question['gold_choice'],
                    'gold_choice': question['gold_choice'], 'baseline_answer': answer,
                    'top1_shares_target_master': not top[0]['is_gold'] and top[0]['master_id'] in target_masters,
                    'any_distractor_shares_target_master': bool(distractor_masters & target_masters),
                    'any_distractor_shares_missing_target_master': bool(distractor_masters & missing_masters),
                    'all_distractors_share_target_master': bool(distractors) and distractor_masters <= target_masters,
                    'missing_targets_with_same_master_distractor': sum(t['master_id'] in distractor_masters for t in missed_targets),
                    'same_master_missed_pairs': sum(p['same_master'] and p['target_missed_top5'] for p in pairs),
                    'missed_pairs': sum(p['target_missed_top5'] for p in pairs),
                    'gold_ids': sorted(gold), 'missing_gold_ids': sorted(missing), 'top5_ids': top_ids,
                    'gold_master_ids': [t['qualified_master_id'] for t in targets],
                    'distractor_master_ids': [d['qualified_master_id'] for d in distractors]}
            cases.append(case)
            details.append({**case, 'options': question['options'], 'targets': targets,
                            'top5': top, 'same_master_pairs': [p for p in pairs if p['same_master']]})
        domain_checks.append({'domain': domain.name, 'questions': len(inputs['questions']),
                              'memories': len(nodes), 'masters': len(masters),
                              'partition_and_full_scan_verified': True})
    assert len(cases) == len(unique(cases, 'question_id')) == 371
    assert total_nodes == 438 and total_masters == 88
    overall = aggregate(cases)
    saved_summary = read(run_dir / 'retrieval_summary.json')['overall']
    assert overall['gold_references'] == saved_summary['gold_references']
    assert overall['gold_hits_top5'] == saved_summary['baseline_question_only']['gold_hits_top5']
    assert overall['questions'] - overall['zero_hit_top5']['failures'] == saved_summary['baseline_question_only']['questions_with_gold_hit']
    summary = {'dataset': 'MemEye', 'caption_source': 'original frozen Qwen3-VL-8B-Instruct captions',
               'baseline': 'full_scan / baseline_question_only; BGE dialogue + full caption; top_k=5',
               'master_source': 'final caption_ward_silhouette_visual graph used by caption_visual_master',
               'domain_count': len(domain_checks), 'memories': total_nodes, 'masters': total_masters,
               'overall': overall,
               'by_domain': {domain: aggregate([c for c in cases if c['domain'] == domain])
                             for domain in sorted({c['domain'] for c in cases})}}
    audit = {'passed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'new_model_calls': 0, 'memory_rebuilt': False,
             'used_frozen_input_hashes_verified': 4 * len(domain_checks),
             'baseline_scores_recomputed': total_scores, 'max_absolute_score_error': maximum_score_error,
             'all_371_full_rankings_verified': True, 'all_371_top5_match_original_full_scan': True,
             'all_master_memberships_checked_both_directions': True,
             'missing_gold_or_unmapped_memories': 0, 'reader_records_checked': len(cases),
             'reader_status_and_length_errors': 0, 'domains': domain_checks,
             'source_sha256': tracked,
             'scope': 'Recompute scores from frozen embeddings; no encoder rerun. Validate stored reader output/payload references; no fresh reader inference.'}
    output.mkdir(parents=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'audit.json', audit)
    csv_write(output / 'per_question.csv', cases)
    csv_write(output / 'retrieval_failures.csv', [c for c in cases if c['incomplete_top5']])
    csv_write(output / 'reader_failures.csv', [c for c in cases if c['reader_wrong']])
    csv_write(output / 'target_distractor_pairs.csv', target_pairs)
    csv_write(output / 'missed_target_distractor_pairs.csv', [p for p in target_pairs if p['target_missed_top5']])
    with (output / 'cases.jsonl').open('w') as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + '\n')
    render_report(output, summary, details)
    print(json.dumps({'output': str(output), 'overall': overall, 'audit_passed': True}, ensure_ascii=False))


def render_report(output, summary, details):
    s = summary['overall']
    text = [
        '# MemEye 全量检索基线：失败案例与同 master 统计', '',
        '使用原始 Qwen caption、8 个领域、371 道题、438 条图像记忆、88 个最终 master。'
        '复用已构建完成的 CaptionWardVisualMemory；逐领域全量扫描 BGE dialogue + full caption，取 Top-5。', '',
        '**干扰项定义：检索返回的非 gold 记忆，不是 MCQ 错误选项。master ID 在领域内有效，比较时附带领域。**', '',
        '| 失败口径 | 失败题数 / 全部题数 | 同 master 判定 | 符合题数 / 失败题数 |',
        '|---|---:|---|---:|',
    ]
    definitions = {
        'top1_wrong': 'Top-1 干扰项与至少一个目标同 master',
        'zero_hit_top5': '至少一个 Top-5 干扰项与至少一个目标同 master',
        'incomplete_top5': '至少一个 Top-5 干扰项与至少一个**漏检目标**同 master',
        'reader_wrong': '至少一个 Top-5 干扰项与至少一个目标同 master',
    }
    for key, label in LABELS.items():
        text.append(f"| {label} | {fmt(s[key]['failure_rate'])} | {definitions[key]} | {fmt(s[key]['same_master_cases'])} |")
    text += ['', '各行是不同口径，彼此重叠，不能相加。问题级比例每题只计一次；多目标问题的“至少一个”是存在性判定。', '',
             f"- 目标引用召回：{s['gold_hits_top5']}/{s['gold_references']}；漏检 {s['gold_references'] - s['gold_hits_top5']} 个目标引用。",
             f"- 漏检目标中，至少有一个同 master 的 Top-5 干扰项：{fmt(s['missed_targets_with_same_master_distractor'])}。",
             f"- 全部“漏检目标 × Top-5 干扰项”配对中，同 master 的配对：{fmt(s['missed_target_distractor_pairs_same_master'])}。每个问题内做笛卡尔积，多个目标会重复计入同一干扰项。",
             f"- 共有 {s['questions_with_more_than_5_gold']} 题的 gold 数超过 5，Top-5 无法覆盖全部目标。限定 gold 数 ≤ 5 后，漏检题的同 master 比例为 {fmt(s['incomplete_top5_gold_count_at_most_5'])}。",
             f"- 最终回答错误但全部目标已在 Top-5 内的题数：{s['reader_wrong_with_all_gold_retrieved']}。回答错误不能直接归因于漏检。", '',
             '## 分领域（Top-5 完全未命中目标）', '',
             '| 领域 | 题数 | 完全漏检题数 | 至少一个同 master 干扰项 / 完全漏检题数 |',
             '|---|---:|---:|---:|']
    for domain, stats in summary['by_domain'].items():
        text.append(f"| {domain} | {stats['questions']} | {stats['zero_hit_top5']['failures']} | {fmt(stats['zero_hit_top5']['same_master_cases'])} |")
    text += ['', '## 可复核文件', '',
             '- `per_question.csv`：371 题的全部标志、目标及干扰项 master。',
             '- `retrieval_failures.csv`：所有 Top-5 未覆盖全部目标的题。',
             '- `reader_failures.csv`：最终回答错误的题，单独列出。',
             '- `missed_target_distractor_pairs.csv`：每个漏检目标与每个 Top-5 非目标的配对。',
             '- `target_distractor_pairs.csv`：全部目标与 Top-5 非目标的配对，含目标是否漏检。',
             '- `cases.jsonl`：完整问题、目标/干扰项原 caption、dialogue、rank、score、master。',
             '- `audit.json`：本次使用的源文件 SHA-256、完整排名及 master 双向归属核验。', '',
             '同 master 仅表示最终 caption 聚类共簇；不证明干扰导致了检索失败。所有统计来自既有固定输入，本次未调用模型或改变聚类。', '']
    (output / 'report.md').write_text('\n'.join(text))
    table_rows = ''.join(f'<tr><td>{html.escape(LABELS[k])}</td><td>{fmt(s[k]["failure_rate"])}</td>'
                         f'<td>{html.escape(definitions[k].replace("**", ""))}</td><td>{fmt(s[k]["same_master_cases"])}</td></tr>' for k in LABELS)
    cards = []
    for d in details:
        def leaf_rows(leaves):
            return ''.join('<tr><td>' + str(x['rank']) + '</td><td>' + html.escape(x['memory_id']) +
                           '<br>' + html.escape(x['filename']) + '</td><td>' + html.escape(x['master_id']) +
                           f"<br>簇大小 {x['master_size']}</td><td>{x['score']:.6f}</td><td>" +
                           html.escape(x['caption']) + '</td></tr>' for x in leaves)
        flags = ' · '.join(label for key, label in LABELS.items() if d[key]) or '四种口径均成功'
        cards.append(f'<details class="case" data-zero="{int(d["zero_hit_top5"])}" '
                     f'data-missing="{int(d["incomplete_top5"])}" data-reader="{int(d["reader_wrong"])}" '
                     f'data-topone="{int(d["top1_wrong"])}">'
                     f'<summary>{html.escape(d["question_id"])} · {flags} · 目标命中 {d["gold_hits_top5"]}/{d["gold_count"]}</summary>'
                     f'<p>{html.escape(d["question"])}</p><p>标准答案 {d["gold_choice"]}；基线回答 {d["baseline_answer"]}。'
                     f'漏检目标与 Top-5 干扰项同 master 配对：{d["same_master_missed_pairs"]}/{d["missed_pairs"]}。</p>'
                     '<h3>所有目标（含漏检目标的完整排名）</h3><table><tr><th>排名</th><th>记忆 / 图片</th><th>Master</th><th>BGE</th><th>Caption</th></tr>' +
                     leaf_rows(d['targets']) + '</table><h3>Top-5 非目标干扰项</h3><table><tr><th>排名</th><th>记忆 / 图片</th><th>Master</th><th>BGE</th><th>Caption</th></tr>' +
                     leaf_rows([x for x in d['top5'] if not x['is_gold']]) + '</table></details>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemEye 失败案例同 master 审计</title>
<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:35px auto;padding:0 24px;color:#17263a;background:#f6f8fb}table{border-collapse:collapse;width:100%;background:white;margin:16px 0;font-size:14px}th,td{border:1px solid #dce2ea;padding:10px;text-align:left;vertical-align:top}th{background:#e9eff6}details{background:white;border:1px solid #dce2ea;border-radius:8px;margin:10px 0;padding:14px}summary{cursor:pointer;font-weight:600}input,select{padding:10px;margin:8px;font-size:16px}input{width:40%}p{line-height:1.6}a{color:#1758a6}</style>
<h1>MemEye 全量检索：失败案例与同 master</h1><p>原始 Qwen caption · 371 题 · 438 条图像记忆 · 88 个最终 master · BGE 全量 Top-5。干扰项指非 gold 检索记忆。各失败口径重叠，不能相加。</p>
<table><tr><th>失败口径</th><th>失败题 / 全部题</th><th>同 master 判定</th><th>同 master 题 / 失败题</th></tr>''' + table_rows + '''</table>
<p><a href="report.md">完整统计口径</a> · <a href="retrieval_failures.csv">下载检索失败案例</a> · <a href="per_question.csv">全部逐题结果</a> · <a href="missed_target_distractor_pairs.csv">目标与干扰项配对</a> · <a href="audit.json">来源核验</a></p>
<p>同 master 是共簇统计，不能证明因果；多个目标时按“至少一对”计算题级比例。Top-5 未覆盖全部目标的指标含 gold 数大于 5 的问题，完整报告给出 gold 数 ≤ 5 的补充统计。</p>
<select id="filter"><option value="zero">Top-5 完全漏检</option><option value="missing">Top-5 未覆盖全部目标</option><option value="topone">Top-1 非目标</option><option value="reader">回答错误</option><option value="all">全部问题</option></select><input id="search" placeholder="搜索题号、问题、图片或 master"><span id="count"></span>
''' + '\n'.join(cards) + '''<script>const f=document.querySelector('#filter'),q=document.querySelector('#search'),cards=[...document.querySelectorAll('.case')];function update(){let n=0;for(const c of cards){let show=(f.value==='all'||c.dataset[f.value]==='1')&&c.textContent.toLowerCase().includes(q.value.toLowerCase());c.hidden=!show;n+=show;}document.querySelector('#count').textContent=n+' 个案例';}f.onchange=update;q.oninput=update;update();</script></html>'''
    (output / 'index.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/baseline_failure_masters_20261022')
    run(parser.parse_args())
