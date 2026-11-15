"""Read-only diagnosis of existing master_answer_query ablations; no model calls."""
import csv
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/memeye_master_zscore_diagnosis_20261024'
SOURCES = {}
RAW = 'master_answer_query'
WEIGHTED = RAW + '_zscore'


def read(path):
    path = ROOT / path
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return ([json.loads(line) for line in data.splitlines() if line]
            if path.suffix == '.jsonl' else json.loads(data))


def index(rows, key='question_id'):
    result = {r[key]: r for r in rows}
    assert len(result) == len(rows)
    return result


def subset(rows):
    wins = sum(not r['raw_correct'] and r['weighted_correct'] for r in rows)
    losses = sum(r['raw_correct'] and not r['weighted_correct'] for r in rows)
    n = wins + losses
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)
    return dict(questions=len(rows), raw_correct=sum(r['raw_correct'] for r in rows),
                weighted_correct=sum(r['weighted_correct'] for r in rows),
                wins=wins, losses=losses, net=wins-losses, mcnemar_exact_p=p,
                answer_disagreements=sum(r['raw_answer'] != r['weighted_answer'] for r in rows))


def main():
    qwen_root = 'reports/master_answer_query_zscore_20261023/reader'
    qwen = read(f'{qwen_root}/paired_results.json')
    manifests = index(read(f'{qwen_root}/qa_manifest.jsonl'), 'record_id')
    gold = {r['id']: r for r in manifests.values()}
    assert len(qwen) == len(gold) == 371
    assert sum(len(r['gold_ids']) for r in gold.values()) == 1041
    rows = []

    def add(run, lam, qid, old_correct, new_correct, old_answer, new_answer,
            old_top, new_top, same_input, same_request=None):
        targets = set(gold[qid]['gold_ids'])
        old_hits, new_hits = len(targets & set(old_top)), len(targets & set(new_top))
        rows.append(dict(run=run, ridge=lam, alpha=1., question_id=qid,
                         raw_correct=bool(old_correct), weighted_correct=bool(new_correct),
                         raw_answer=old_answer, weighted_answer=new_answer,
                         same_input=same_input, same_api_request=same_request,
                         same_top5_order=old_top == new_top,
                         same_top5_set=set(old_top) == set(new_top),
                         raw_gold_hits=old_hits, weighted_gold_hits=new_hits,
                         gold_hits_delta=new_hits-old_hits,
                         removed_gold=sorted((targets & set(old_top))-set(new_top)),
                         added_gold=sorted((targets & set(new_top))-set(old_top)),
                         raw_top5=old_top, weighted_top5=new_top))

    for r in qwen:
        add('qwen_caption_qwen_reader', .01, r['question_id'], r['baseline_correct'],
            r['zscore_correct'], r['baseline_answer'], r['zscore_answer'],
            r['baseline_top5'], r['zscore_top5'], r['same_payload'])

    old_root = 'results/memeye_reader_ablation_20261021/gpt-5.4-2026-03-05_none'
    new_root = 'results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024'
    old_pairs = index(read(f'{old_root}/paired_results.json'))
    old_req = index(read(f'{old_root}/inputs/requests.jsonl'), 'record_id')
    new_req = index(read(f'{new_root}/inputs/requests.jsonl'), 'record_id')
    old_resp = index(read(f'{old_root}/responses.jsonl'), 'record_id')
    new_resp = index(read(f'{new_root}/responses.jsonl'), 'record_id')
    frozen_reader_inputs_checked = 0
    for r in read(f'{new_root}/paired_results.json'):
        qid = r['question_id']
        a, b = old_req[qid+'::'+RAW], new_req[qid+'::'+WEIGHTED]
        for req, condition in [(a, RAW), (b, WEIGHTED)]:
            ref = manifests[qid+'::'+condition]
            assert all(req[k] == ref[k] for k in ['reader_text', 'query_images', 'top_memory_ids'])
            frozen_reader_inputs_checked += 1
        old = old_pairs[qid]
        add('qwen_caption_gpt54_reader', .01, qid, old['expanded_correct'],
            r[WEIGHTED+'_correct'], old['expanded_answer'], r[WEIGHTED+'_answer'],
            a['top_memory_ids'], b['top_memory_ids'], a['payload_sha256'] == b['payload_sha256'],
            old_resp[a['record_id']]['request_sha256'] == new_resp[b['record_id']]['request_sha256'])

    full = 'results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05'
    old_ret = index(read(f'{full}/retrieval_results.jsonl'), 'id')
    old_req = index(read(f'{full}/reader/requests.jsonl'), 'record_id')
    full_gold = index(read(f'{full}/inputs/evaluation.jsonl'), 'id')
    assert set(full_gold) == set(gold)
    for qid in gold:
        assert gold[qid]['gold_choice'] == full_gold[qid]['gold_choice']
        assert set(gold[qid]['gold_ids']) == set(full_gold[qid]['gold_ids'])
    for label, lam in [('0p001', .001), ('0p01', .01), ('0p1', .1)]:
        root = f'results/memeye_gpt54_caption_zscore_lambda{label}_alpha1_20261024'
        requests = index(read(f'{root}/inputs/requests.jsonl'), 'record_id')
        responses = index(read(f'{root}/responses.jsonl'), 'record_id')
        for r in read(f'{root}/paired_results.json'):
            qid = r['question_id']
            a, b = old_req[qid+'::'+RAW], requests[qid+'::'+WEIGHTED]
            same = a['payload_sha256'] == b['payload_sha256']
            assert same == r[RAW+'_identical_historical_payload']
            old_top = next(c['top5_memory_ids'] for c in old_ret[qid]['conditions'] if c['name'] == RAW)
            add('gpt54_caption_gpt54_reader', lam, qid, r[RAW+'_historical_correct'],
                r[WEIGHTED+'_correct'], r[RAW+'_historical_answer'], r[WEIGHTED+'_answer'],
                old_top, r[WEIGHTED+'_top5'], same,
                a['request_sha256'] == responses[b['record_id']]['request_sha256'])

    summaries = []
    for run, lam in dict.fromkeys((r['run'], r['ridge']) for r in rows):
        rr = [r for r in rows if (r['run'], r['ridge']) == (run, lam)]
        assert len(rr) == len({r['question_id'] for r in rr}) == 371
        summaries.append(dict(run=run, ridge=lam, alpha=1., overall=subset(rr),
            identical_input=subset([r for r in rr if r['same_input']]),
            changed_input=subset([r for r in rr if not r['same_input']]),
            identical_api_requests=sum(r['same_api_request'] is True for r in rr),
            raw_gold_hits=sum(r['raw_gold_hits'] for r in rr),
            weighted_gold_hits=sum(r['weighted_gold_hits'] for r in rr), gold_references=1041,
            raw_any_gold=sum(r['raw_gold_hits'] > 0 for r in rr),
            weighted_any_gold=sum(r['weighted_gold_hits'] > 0 for r in rr),
            same_top5_order=sum(r['same_top5_order'] for r in rr),
            same_top5_set=sum(r['same_top5_set'] for r in rr),
            gains_by_gold_hit_change={str(delta):sum(not r['raw_correct'] and r['weighted_correct'] and
                ((r['gold_hits_delta'] > 0)-(r['gold_hits_delta'] < 0)) == delta for r in rr) for delta in [-1,0,1]},
            losses_by_gold_hit_change={str(delta):sum(r['raw_correct'] and not r['weighted_correct'] and
                ((r['gold_hits_delta'] > 0)-(r['gold_hits_delta'] < 0)) == delta for r in rr) for delta in [-1,0,1]}))
    result = dict(scope='Existing single-run master_answer_query results; no new inference',
        frozen_qwen_vs_gpt_reader_inputs_verified=frozen_reader_inputs_checked,
        summaries=summaries, source_sha256=SOURCES,
        limits=['Same-input differences cannot be caused by changed retrieval.',
                'Changed-input differences mix retrieval effects with reader variability.',
                'Identical-input subgroup is not a randomized estimate for the changed-input subgroup.',
                'Do not reuse answers post hoc and call that a new measured experiment.',
                'Gold-reference recall and final answer accuracy measure different outcomes.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'per_question.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
    parts = ['<!doctype html><meta charset="utf-8"><title>master z-score 逐题诊断</title>',
             '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>',
             '<h1>master_answer_query：检索变化与回答波动</h1>',
             '<p>所有结果均为现有单次实验，α=1；题目和目标集合一致。此诊断没有重新调用模型。</p>',
             '<table><tr><th>Caption / 回答模型</th><th>λ</th><th>答对数：原始 → 加权</th><th>同输入题数 / 净变化</th><th>不同输入净变化</th><th>目标命中数 / 1041</th></tr>']
    for s in summaries:
        a,b,c = s['overall'],s['identical_input'],s['changed_input']
        parts.append(f'<tr><td>{html.escape(s["run"])}</td><td>{s["ridge"]}</td><td>{a["raw_correct"]} → {a["weighted_correct"]}</td><td>{b["questions"]} / {b["net"]:+d}</td><td>{c["net"]:+d}</td><td>{s["raw_gold_hits"]} → {s["weighted_gold_hits"]}</td></tr>')
    parts += ['</table><p>λ=0.01 的 GPT caption + GPT 回答实验净减少 4 题，恰好来自同输入题组；不同输入组 18 题改善、18 题变差。这不证明检索无影响，只说明总分差不能直接解释为检索退步。</p>',
              '<p>Qwen 的准确率提升也伴随目标记忆命中总数下降。两者不能等同。白化与 z-score 没有使用目标标签，因此不保证凸显的差异正是回答所需信息。</p>',
              '<p>分组是描述性分析，不能用同输入组估计其余题目的波动，也不能据此修正已报告的实测总分。</p>',
              '<p><a href="per_question.csv">逐题明细</a> · <a href="diagnosis.json">统计与来源哈希</a></p>']
    (OUT/'report.html').write_text('\n'.join(parts)+'\n')
    print(json.dumps({'summaries':summaries,'frozen_inputs_checked':frozen_reader_inputs_checked,'output':str(OUT)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
