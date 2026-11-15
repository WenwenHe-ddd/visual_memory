#!/usr/bin/env python3
"""Freeze answer-error cases for the original Qwen full-scan Top-5 baseline."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'reports/all_master_answer_query_20261021'
AUDIT = ROOT / 'reports/baseline_failure_masters_20261022'
OUT = AUDIT / 'qa_failure_set'
CONDITION = 'baseline_question_only'


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = [RUN / 'reader/qa_manifest.jsonl', RUN / 'reader/responses.jsonl',
               RUN / 'reader/protocol.json', RUN / 'retrieval_results.jsonl', AUDIT / 'cases.jsonl',
               AUDIT / 'audit.json', Path(__file__).resolve()]
    source_hashes = {str(p): sha(p) for p in sources}
    previous_audit = json.loads((AUDIT / 'audit.json').read_text())
    assert previous_audit['passed'] and previous_audit['all_371_top5_match_original_full_scan']
    for path, digest in previous_audit['source_sha256'].items():
        assert sha(Path(path)) == digest, f'Source changed since ranking audit: {path}'
    manifests = [m for m in read_lines(sources[0]) if m['condition'] == CONDITION]
    responses = [r for r in read_lines(sources[1]) if r['condition'] == CONDITION]
    by_response = {r['record_id']: r for r in responses}
    cases = {c['question_id']: c for c in read_lines(AUDIT / 'cases.jsonl')}
    retrieval = {r['question_id']: r for r in read_lines(RUN / 'retrieval_results.jsonl')}
    settings = json.loads((RUN / 'reader/protocol.json').read_text())
    assert len(manifests) == len(responses) == len(by_response) == len(cases) == 371
    assert {m['record_id'] for m in manifests} == set(by_response)
    assert {m['id'] for m in manifests} == set(cases) == set(retrieval)
    failures, successes, rows = [], [], []
    all_by_domain, failed_by_domain, coverage = Counter(), Counter(), Counter()
    for m in manifests:
        r, c = by_response[m['record_id']], cases[m['id']]
        ranked = next(v for v in retrieval[m['id']]['conditions'] if v['name'] == CONDITION)
        assert len(m['top_memory_ids']) == len(set(m['top_memory_ids'])) == 5
        assert m['top_memory_ids'] == ranked['top5_memory_ids'] == [v['memory_id'] for v in c['top5']]
        assert m['gold_ids'] == retrieval[m['id']]['evaluation']['gold_ids']
        assert r['payload_sha256'] == m['payload_sha256']
        assert r['experiment_signature'] == settings['experiment_signature']
        assert r['gold_choice'] == m['gold_choice'] == c['gold_choice']
        assert r['status'] == 'ok' and not r['hit_token_limit']
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', r['raw_response'].strip(), re.I)
        assert match and match.group(1).upper() == r['predicted_choice']
        correct = r['predicted_choice'] == m['gold_choice']
        assert correct == r['correct'] == (not c['reader_wrong'])
        all_by_domain[m['task']] += 1
        if correct:
            successes.append(m['id'])
            continue
        hits = len(set(m['top_memory_ids']) & set(m['gold_ids']))
        group = 'none' if hits == 0 else 'all' if hits == len(m['gold_ids']) else 'partial'
        coverage[group] += 1
        failed_by_domain[m['task']] += 1
        detail = {
            'question_id': m['id'], 'domain': m['task'], 'question': m['question'],
            'options': m['options'], 'gold_choice': m['gold_choice'],
            'gold_answer': m['options'][m['gold_choice']], 'predicted_choice': r['predicted_choice'],
            'predicted_answer': m['options'][r['predicted_choice']],
            'gold_ids': m['gold_ids'], 'gold_count': len(m['gold_ids']), 'gold_hits_top5': hits,
            'gold_coverage': group, 'top5': c['top5'], 'targets': c['targets'],
            'record_id': m['record_id'], 'payload_sha256': m['payload_sha256'],
            'raw_response': r['raw_response'], 'reader_status': r['status'],
        }
        failures.append(detail)
        rows.append({k: detail[k] for k in ['question_id', 'domain', 'question', 'options',
                     'gold_choice', 'gold_answer', 'predicted_choice', 'predicted_answer',
                     'gold_count', 'gold_hits_top5', 'gold_coverage', 'gold_ids']} |
                    {f'top{i}_id': item['memory_id'] for i, item in enumerate(c['top5'], 1)} |
                    {f'top{i}_is_gold': item['is_gold'] for i, item in enumerate(c['top5'], 1)})
    assert len(failures) == 187 and len(successes) == 184
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'condition': CONDITION,
               'baseline': 'Original Qwen captions; full-scan BGE dialogue+caption; Top-5; Qwen reader',
               'failure_definition': 'predicted_choice != gold_choice', 'questions': len(manifests),
               'failures': len(failures), 'successes': len(successes),
               'failure_rate': len(failures) / len(manifests),
               'failed_cases_gold_coverage': dict(coverage),
               'by_domain': {d: {'questions': n, 'failures': failed_by_domain[d]}
                             for d, n in sorted(all_by_domain.items())},
               'failure_question_ids': [c['question_id'] for c in failures],
               'success_question_ids': successes,
               'checks': {'unique_ids_complete': True, 'top5_matches_frozen_full_scan': True,
                          'correctness_recomputed_from_raw_answer': True, 'format_errors': 0,
                          'token_limit_errors': 0, 'new_inference_calls': 0},
               'source_sha256': source_hashes}
    OUT.mkdir()
    write_json(OUT / 'summary.json', summary)
    with (OUT / 'failures.jsonl').open('w') as f:
        for c in failures:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    with (OUT / 'failures.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)
    render(failures, summary)
    assert len(read_lines(OUT / 'failures.jsonl')) == 187
    assert (OUT / 'index.html').read_text().count('class="case"') == 187
    assert all(sha(Path(p)) == digest for p, digest in source_hashes.items())
    print(json.dumps({k: summary[k] for k in ['questions', 'failures', 'successes',
                     'failure_rate', 'failed_cases_gold_coverage', 'by_domain']}, ensure_ascii=False))
    print(json.dumps([{'id': c['question_id'], 'question': c['question'],
                      'gold': c['gold_answer'], 'prediction': c['predicted_answer'],
                      'hits': c['gold_hits_top5'], 'targets': c['gold_count']}
                     for c in failures[:3]], ensure_ascii=False))


def render(cases, summary):
    esc = html.escape
    cards = []
    names = {'none': '目标全部缺失', 'partial': '目标部分命中', 'all': '目标全部命中'}
    for c in cases:
        memories = []
        for m in c['top5']:
            memories.append(f'<tr><td>{m["rank"]}</td><td>{esc(m["memory_id"])}<br>{esc(m["filename"])}</td>'
                            f'<td>{"目标" if m["is_gold"] else "非目标"}</td><td>{m["score"]:.6f}</td>'
                            f'<td>{esc(m["caption"])}<details><summary>原始对话</summary><pre>{esc(m["dialogue"])}</pre></details></td></tr>')
        options = ''.join(f'<li><b>{k}.</b> {esc(v)}</li>' for k, v in c['options'].items())
        cards.append(f'<details class="case" data-coverage="{c["gold_coverage"]}"><summary>{esc(c["question_id"])} · '
                     f'标准 {c["gold_choice"]} → 模型 {c["predicted_choice"]} · 目标命中 {c["gold_hits_top5"]}/{c["gold_count"]}</summary>'
                     f'<h3>{esc(c["question"])}</h3><ul>{options}</ul><p>标准答案：{esc(c["gold_answer"])}<br>'
                     f'模型回答：{esc(c["predicted_answer"])}</p><p>目标 ID：{esc(", ".join(c["gold_ids"]))}</p>'
                     '<table><tr><th>排名</th><th>记忆 / 图片</th><th>目标标注</th><th>检索分数</th><th>原始 caption 与对话</th></tr>' +
                     ''.join(memories) + '</table></details>')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全量检索 Top-5 基线：187 个回答失败案例</title>
<style>body{font-family:system-ui,sans-serif;max-width:1350px;margin:35px auto;padding:0 24px;background:#f6f8fb;color:#18283b}p{line-height:1.65}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dbe3ec;padding:10px;text-align:left;vertical-align:top}th{background:#edf2f8}details.case{background:white;border:1px solid #dbe3ec;border-radius:8px;margin:10px 0;padding:16px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font-family:inherit}input,select{padding:10px;font-size:16px;margin:10px 10px 10px 0}input{width:45%}</style>
<h1>全量检索 Top-5 基线：失败案例集合</h1><p>MemEye 原始 Qwen caption · 全量 BGE 检索取 Top-5 · Qwen 回答。371 题中答对 184 题，答错 187 题（50.40%）。<br><b>纳入规则只有一个：模型预测选项 ≠ 标准选项。</b>每题固定保留实际传给回答模型的五条记忆和顺序。</p>
''' + '<p>187 个回答错误案例中：' + '；'.join(f'{names[k]} {summary["failed_cases_gold_coverage"].get(k, 0)} 题' for k in ['none','partial','all']) + '''。这些是失败集合内部的描述，不改变纳入规则。</p>
<p><a href="failures.csv">下载 187 题明细 CSV</a> · <a href="failures.jsonl">完整内容 JSONL</a> · <a href="summary.json">集合 ID、统计与来源校验</a></p>
<select id="coverage"><option value="all_cases">全部 187 个失败案例</option><option value="none">目标全部缺失</option><option value="partial">目标部分命中</option><option value="all">目标全部命中</option></select><input id="search" placeholder="搜索题号、问题、选项或记忆"><span id="count"></span>
''' + '\n'.join(cards) + '''<script>const q=document.querySelector('#search'),f=document.querySelector('#coverage'),cards=[...document.querySelectorAll('.case')];function update(){let n=0;for(const c of cards){const show=(f.value==='all_cases'||c.dataset.coverage===f.value)&&c.textContent.toLowerCase().includes(q.value.toLowerCase());c.hidden=!show;n+=show;}document.querySelector('#count').textContent=n+' 个案例';}q.oninput=update;f.onchange=update;update();</script></html>'''
    (OUT / 'index.html').write_text(page)


if __name__ == '__main__':
    main()
