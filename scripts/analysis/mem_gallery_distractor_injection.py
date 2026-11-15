"""Paired random distractor injection using pinned Mem-Gallery caption vectors."""
import argparse
import collections
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
import numpy as np
from scipy.stats import hypergeom

LEVELS = (0, 10, 20)
CONDITIONS = ('same_scenario', 'other_scenarios')
REPEATS = 5
SEED = 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))


def csvout(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(target_score, others):
    target = round(float(target_score), 6)
    scores = np.round(np.asarray(others, dtype=float), 6)
    better, tied = int(sum(scores > target)), int(sum(scores == target))
    return {'recall_at_5': float(np.clip((5-better)/(1+tied), 0, 1)),
            'higher_scoring_distractors': better, 'equal_scoring_distractors': tied,
            'target_expected_rank': 1+better+tied/2}


def exact_recall(pool_size, better, equal, n):
    """Uniform sampling without replacement, including random ordering of ties."""
    if n == 0:
        return 1.0
    if equal == 0:
        return float(hypergeom.cdf(4, pool_size, better, n))
    total = 0.0
    for b in range(min(4, better, n)+1):
        probability_b = float(hypergeom.pmf(b, pool_size, better, n))
        remaining = n-b
        if not probability_b or remaining > pool_size-better:
            continue
        for t in range(min(equal, remaining)+1):
            probability_t = float(hypergeom.pmf(t, pool_size-better, equal, remaining))
            total += probability_b*probability_t*min(1.0, (5-b)/(1+t))
    return total


def summarize(label, samples):
    rows = []
    for n in LEVELS:
        for condition in CONDITIONS:
            selected = [r for r in samples if r['N'] == n and r['condition'] == condition]
            if not selected:
                continue
            rows.append({'group': label, 'N': n, 'condition': condition,
                         'targets': len(selected), 'questions': len({r['question_id'] for r in selected}),
                         'recall_at_5': float(np.mean([r['recall_at_5'] for r in selected])),
                         'exact_expected_recall_at_5': float(np.mean([r['exact_expected_recall_at_5'] for r in selected])),
                         'mean_target_rank': float(np.mean([r['target_expected_rank'] for r in selected])),
                         'target_distractor_cosine': float(np.mean([r['target_distractor_cosine'] for r in selected])) if n else None,
                         'distractor_query_cosine': float(np.mean([r['distractor_query_cosine'] for r in selected])) if n else None})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-dir', type=Path, default=ROOT/'results/mem_gallery_natural_observation/v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/mem_gallery_distractor_injection/v1')
    args = parser.parse_args()
    nat, out = args.natural_dir.resolve(), args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT/'data/Mem-Gallery'
    inputs = [nat/name for name in ('protocol.json', 'memories.jsonl', 'queries.jsonl', 'gold.jsonl', 'embeddings.npz', 'summary.json')]
    inputs += [base/'source_manifest.json', base/'image_manifest.json']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    natural = json.loads((nat/'protocol.json').read_text())
    assert natural['status'] == 'complete'
    for source in natural['dataset']['files']:
        assert sha(ROOT/source['path']) == source['sha256'], source['path']
    images = json.loads((base/'image_manifest.json').read_text())
    assert images['revision'] == natural['dataset']['revision'] and images['status'] == 'complete'
    image_by_path = {str((base/r['path']).resolve()): r for r in images['files']}
    mem, queries, golds = [read(nat/(name+'.jsonl')) for name in ('memories', 'queries', 'gold')]
    mi = {r['id']: i for i, r in enumerate(mem)}
    assert len(mi) == len(mem) and all(len(r['images']) == 1 for r in mem)
    paths = [(base/'data/dialog'/r['images'][0]).resolve() for r in mem]
    keys = [image_by_path[str(p)]['sha256'] for p in paths]
    for p in set(paths):
        assert sha(p) == image_by_path[str(p)]['sha256'], p
    vec = np.load(nat/'embeddings.npz')
    assert list(vec['memory_ids']) == [r['id'] for r in mem]
    assert list(vec['query_ids']) == [r['id'] for r in queries]
    emb, qemb = vec['memory'], vec['query']
    assert np.isfinite(emb).all() and np.isfinite(qemb).all()
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(qemb, axis=1), 1, atol=1e-6)
    # Float64 products on the identical saved float32 vectors prevent arithmetic-path drift.
    query_scores = np.asarray(qemb, dtype=float) @ np.asarray(emb, dtype=float).T
    caption_scores = np.asarray(emb, dtype=float) @ np.asarray(emb, dtype=float).T
    catalog = [{**r, 'image_path': str(paths[i]), 'image_sha256': keys[i]} for i, r in enumerate(mem)]
    pools, samples, exclusions = [], [], []
    query_copy_questions = []
    for qi, (q, gold) in enumerate(zip(queries, golds)):
        assert q['id'] == gold['id']
        if not gold['primary_cohort']:
            continue
        qpath = (base/'data/dialog'/q['question_image']).resolve()
        query_hash = image_by_path[str(qpath)]['sha256']
        assert sha(qpath) == query_hash
        target_hashes = {keys[mi[mid]] for mid in gold['all_clue_ids']}
        if any(k == query_hash and k not in target_hashes for k in keys):
            query_copy_questions.append(q['id'])
        excluded_hashes = target_hashes | {query_hash}
        pool = {}
        for condition in CONDITIONS:
            representatives = {}
            for i, row in enumerate(mem):
                same = row['task'] == q['task']
                if same != (condition == 'same_scenario') or keys[i] in excluded_hashes:
                    continue
                representatives.setdefault(keys[i], i)
            pool[condition] = list(representatives.values())
        pools.append({'question_id': q['id'], 'task': q['task'], 'all_target_ids': gold['all_clue_ids'],
                      'excluded_image_sha256': sorted(excluded_hashes), 'query_image_sha256': query_hash,
                      **{c: [mem[i]['id'] for i in pool[c]] for c in CONDITIONS}})
        if any(len(pool[c]) < max(LEVELS) for c in CONDITIONS):
            exclusions.append({'question_id': q['id'], 'targets': len(gold['all_clue_ids']),
                               'same_pool': len(pool['same_scenario']), 'other_pool': len(pool['other_scenarios'])})
            continue
        for mid in gold['all_clue_ids']:
            samples.append({'sample_id': q['id']+'|'+mid, 'question_id': q['id'], 'task': q['task'],
                            'point': q['point'], 'target_id': mid, 'query_index': qi,
                            'target_index': mi[mid], 'pools': pool})
    if exclusions:
        csvout(out/'excluded_questions.csv', exclusions)
        raise ValueError('Main cohort cannot be retained at every N; inspect pools before changing protocol')
    assert len(samples) == json.loads((nat/'summary.json').read_text())['primary']['targets']
    protocol = {'status': 'prepared', 'dataset_revision': images['revision'], 'retriever': natural['retriever'],
                'natural_dir': str(nat), 'caption_label': natural.get('caption_label', 'Official supplied captions'),
                'retriever_revision': natural['retriever_revision'], 'seed': SEED, 'repetitions': REPEATS,
                'N_levels': LEVELS, 'conditions': CONDITIONS, 'input_hashes': hashes, 'script_sha256': sha(Path(__file__)),
                'cohort': {'questions': len(pools), 'targets': len(samples), 'scenarios': len({s['task'] for s in samples})},
                'query': 'unchanged cached question + selected QA caption + BGE query prefix',
                'memory': 'unchanged cached full selected history caption; one fixed target plus N distractors',
                'other_gold_policy': 'all other clue rounds omitted from the small memory, never counted as distractors',
                'dedup': 'unique image-byte SHA256 within each condition; first source-order representative caption',
                'excluded_content': 'all question clue images plus query image, including exact-byte copies',
                'sampling': 'uniform without replacement; per-target condition repetition SHA256 seed; nested prefixes',
                'seed_rule': 'first 16 hex digits of SHA256(42|sample_id|condition|repetition)',
                'scores': 'float64 dot products of unchanged saved normalized float32 vectors; rounded to 6 decimals for ranks',
                'ties': 'expected target rank and Top-5 inclusion under uniform tie ordering',
                'aggregation': 'mean over five repetitions, then target-equal; scenario and question-type results also reported',
                'analytic_check': 'exact finite-pool expected Recall@5 from hypergeometric sampling, including ties',
                'interpretation': 'manipulates distractor count and scenario source; not isolated semantic cue overlap or answer accuracy',
                'N0': 'target-only library; Recall@5=1 by construction, not a model result',
                'new_caption_calls': 0, 'new_embedding_calls': 0, 'answer_generation_calls': 0}
    dump(out/'protocol.json', protocol)
    jsonl(out/'sampling_pools.jsonl', pools)
    jsonl(out/'memory_catalog.jsonl', catalog)
    print('Prepared fixed cohort', protocol['cohort'], flush=True)
    rows, sample_rows, plans = [], [], []
    for s in samples:
        qi, target = s['query_index'], s['target_index']
        scores = query_scores[qi]
        base_fields = {k: s[k] for k in ('sample_id', 'question_id', 'task', 'point', 'target_id')}
        for condition in CONDITIONS:
            pool = s['pools'][condition]
            pool_metric = metric(scores[target], scores[pool])
            exact = {n: exact_recall(len(pool), pool_metric['higher_scoring_distractors'],
                                    pool_metric['equal_scoring_distractors'], n) for n in LEVELS}
            current = []
            for repetition in range(REPEATS):
                seed_hex = hashlib.sha256(f'{SEED}|{s["sample_id"]}|{condition}|{repetition}'.encode()).hexdigest()[:16]
                ordered = np.random.default_rng(int(seed_hex, 16)).permutation(pool)[:max(LEVELS)].tolist()
                assert len(set(keys[i] for i in ordered)) == max(LEVELS)
                plans.append({**base_fields, 'condition': condition, 'repetition': repetition, 'seed_hex': seed_hex,
                              'ordered_distractor_ids': [mem[i]['id'] for i in ordered]})
                last = 1.0
                for n in LEVELS:
                    selected = ordered[:n]
                    result = metric(scores[target], scores[selected])
                    assert result['recall_at_5'] <= last+1e-12
                    last = result['recall_at_5']
                    row = {**base_fields, 'condition': condition, 'N': n, 'repetition': repetition,
                           'memory_count': n+1, 'pool_size': len(pool), 'target_query_score': float(scores[target]),
                           'target_distractor_cosine': float(np.mean(caption_scores[target, selected])) if n else None,
                           'distractor_query_cosine': float(np.mean(scores[selected])) if n else None,
                           'exact_expected_recall_at_5': exact[n], **result}
                    rows.append(row)
                    current.append(row)
            for n in LEVELS:
                selected = [r for r in current if r['N'] == n]
                sample_rows.append({**base_fields, 'condition': condition, 'N': n, 'repetitions': REPEATS,
                                    'recall_at_5': float(np.mean([r['recall_at_5'] for r in selected])),
                                    'exact_expected_recall_at_5': exact[n],
                                    'target_expected_rank': float(np.mean([r['target_expected_rank'] for r in selected])),
                                    'target_distractor_cosine': float(np.mean([r['target_distractor_cosine'] for r in selected])) if n else None,
                                    'distractor_query_cosine': float(np.mean([r['distractor_query_cosine'] for r in selected])) if n else None})
    overall = summarize('all', sample_rows)
    groups = [r for task in sorted({s['task'] for s in samples}) for r in summarize(task, [s for s in sample_rows if s['task'] == task])]
    by_type = [r for point in sorted({s['point'] for s in samples}) for r in summarize(point, [s for s in sample_rows if s['point'] == point])]
    lookup = {(r['sample_id'], r['N'], r['condition']): r for r in sample_rows}
    paired = []
    for s in samples:
        for n in LEVELS:
            a, b = [lookup[(s['sample_id'], n, c)] for c in CONDITIONS]
            paired.append({k: s[k] for k in ('sample_id', 'question_id', 'task', 'point', 'target_id')} |
                          {'N': n, 'same_minus_other_recall': a['recall_at_5']-b['recall_at_5'],
                           'exact_same_minus_other_recall': a['exact_expected_recall_at_5']-b['exact_expected_recall_at_5'],
                           'same_minus_other_similarity': a['target_distractor_cosine']-b['target_distractor_cosine'] if n else None})
    duplicate_groups = collections.defaultdict(list)
    for i, key in enumerate(keys):
        duplicate_groups[key].append(i)
    audit = {'input_images_hash_verified': len(set(paths)), 'memory_rounds': len(mem), 'unique_history_image_bytes': len(set(keys)),
             'duplicate_image_groups': sum(len(v)>1 for v in duplicate_groups.values()),
             'duplicate_image_groups_with_caption_variation': sum(len({mem[i]['caption'] for i in v})>1 for v in duplicate_groups.values()),
             'query_copy_exclusion_questions': len(query_copy_questions), 'query_copy_exclusion_question_ids': query_copy_questions,
             'cohort_exclusions': len(exclusions), 'runs': len(rows), 'sampling_plans': len(plans),
             'fixed_target_count_at_each_condition_and_N': len(samples),
             'min_same_scenario_pool': min(len(p['same_scenario']) for p in pools),
             'min_other_scenario_pool': min(len(p['other_scenarios']) for p in pools),
             'runs_with_score_ties': sum(r['equal_scoring_distractors']>0 for r in rows)}
    summary = {'caption_label': protocol['caption_label'], 'cohort': protocol['cohort'], 'overall': overall, 'by_scenario': groups, 'by_type': by_type,
               'paired': [{'N': n, 'target_equal_gap': float(np.mean([r['same_minus_other_recall'] for r in paired if r['N'] == n])),
                           'exact_target_equal_gap': float(np.mean([r['exact_same_minus_other_recall'] for r in paired if r['N'] == n]))}
                          for n in LEVELS], 'audit': audit}
    for name, data in [('runs', rows), ('sample_means', sample_rows), ('group_means', [*overall, *groups]),
                       ('type_means', by_type), ('paired_effects', paired)]:
        csvout(out/(name+'.csv'), data)
    jsonl(out/'sampling_plans.jsonl', plans)
    dump(out/'summary.json', summary)
    dump(out/'audit.json', audit)
    make_report(out, summary)
    for p in inputs:
        assert sha(p) == hashes[str(p.relative_to(ROOT))], p
    protocol['status'] = 'complete'
    protocol['environment'] = {'numpy': np.__version__}
    dump(out/'protocol.json', protocol)
    dump(out/'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'})
    print(json.dumps({'cohort': summary['cohort'], 'overall': overall, 'paired': summary['paired'], 'audit': audit}, indent=2), flush=True)


def make_report(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    styles = [('same_scenario', '#b96537', 'o', 'Same-scenario'), ('other_scenarios', '#287d9b', 'D', 'Other-scenario')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for condition, color, marker, label in styles:
        rows = [r for r in summary['overall'] if r['condition'] == condition]
        axes[0].plot([r['N'] for r in rows], [r['recall_at_5'] for r in rows], color=color, marker=marker, label=label+' (5 draws)')
        axes[0].plot([r['N'] for r in rows], [r['exact_expected_recall_at_5'] for r in rows], color=color, linestyle=':', alpha=.7, label=label+' (exact expectation)')
        rows = [r for r in rows if r['N']]
        axes[1].plot([r['N'] for r in rows], [r['target_distractor_cosine'] for r in rows], color=color, marker=marker, label=label)
    axes[0].set_ylabel('Target Recall@5'); axes[0].yaxis.set_major_formatter(PercentFormatter(1)); axes[0].set_ylim(0, 1.04)
    axes[1].set_ylabel('Mean target-to-distractor caption cosine'); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Number of distractors N'); ax.set_xticks(LEVELS); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Fixed target + N unique distractors'); axes[1].set_title('Measured similarity (not assumed by source)')
    fig.suptitle(f"Mem-Gallery: {summary.get('caption_label', 'Official supplied captions')}\n{summary['cohort']['questions']} questions / {summary['cohort']['targets']} targets")
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out/('overall_injection.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    tasks = sorted({r['group'] for r in summary['by_scenario']})
    fig, axes = plt.subplots(5, 4, figsize=(15, 15))
    for ax, task in zip(axes.flat, tasks):
        for condition, color, marker, label in styles:
            rows = [r for r in summary['by_scenario'] if r['group'] == task and r['condition'] == condition]
            ax.plot([r['N'] for r in rows], [r['recall_at_5'] for r in rows], color=color, marker=marker, label=label)
        ax.set_title(task.replace('_', ' ')[:47]+'\n'+f"{rows[0]['targets']} targets", fontsize=9)
        ax.set_xticks(LEVELS); ax.set_ylim(0, 1.04); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(alpha=.2)
        ax.set_xlabel('Distractors N'); ax.set_ylabel('Recall@5')
    for ax in list(axes.flat)[len(tasks):]:
        ax.set_visible(False)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', ncol=2)
    fig.suptitle('Same fixed cohort at every N and both source conditions')
    fig.tight_layout(rect=(0, .035, 1, .97))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out/('by_scenario_injection.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    def table(rows):
        content = '<table><tr><th>组别</th><th>N</th><th>来源</th><th>目标数</th><th>Recall@5</th><th>精确抽样期望</th><th>目标与干扰相似度</th></tr>'
        for r in rows:
            sim = '—' if r['target_distractor_cosine'] is None else f"{r['target_distractor_cosine']:.4f}"
            content += f"<tr><td>{html.escape(r['group'])}</td><td>{r['N']}</td><td>{r['condition']}</td><td>{r['targets']}</td><td>{r['recall_at_5']:.2%}</td><td>{r['exact_expected_recall_at_5']:.2%}</td><td>{sim}</td></tr>"
        return content+'</table>'
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Mem-Gallery 干扰实验</title>
<style>body{font:16px/1.7 system-ui;max-width:1250px;margin:30px auto;padding:20px}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style>
<h1>Mem-Gallery：同场景 / 其他场景随机干扰</h1>
<p>固定上一轮主样本。每个题目 × clue 目标单独建立小记忆库，保留一个目标并加入 N=0、10、20 个干扰；不加入同题其他目标。
全部 clue 图片和问题原图的相同字节副本均从候选池排除；每种条件内按图片 SHA256 去重，保留源顺序中首条 caption。
每个样本和来源独立抽样 5 次，同次抽样的不同 N 为同一排列的前缀。两种条件的样本、数量、query、caption 和向量完全相同，仅干扰来源不同。</p>
<p>查询和记忆向量直接复用自然观测缓存，没有重新生成 caption、重新编码或调用回答模型。先对五次抽样平均，再按目标等权汇总。
虚线是同一有限候选池下随机不放回抽样的数学期望，不是置信区间。N=0 时只有目标，100% 是构造结果。</p>
<img src="overall_injection.png" alt="干扰数量与目标召回、caption相似度">'''+table(summary['overall'])+'''
<h2>分场景</h2><img src="by_scenario_injection.png" alt="各场景干扰实验曲线">
<h2>分题型</h2>'''+table(summary['by_type'])+'''
<p>此实验操纵数量与来源。添加更多候选在固定向量检索中不会提高目标排名；应比较同量干扰下的来源差异，不能仅凭下降曲线证明线索过载。
同场景不是人工确认的高线索重叠标签；未标注为 clue 的图片也不保证语义无关。结果仅描述此 caption 检索配置，不是端到端作答准确率。</p>
<p><a href="protocol.json">协议</a> · <a href="summary.json">完整结果</a> · <a href="runs.csv">逐次抽样</a> ·
<a href="paired_effects.csv">逐目标配对差值</a> · <a href="sampling_plans.jsonl">抽样图片 ID</a> ·
<a href="memory_catalog.jsonl">caption 与本地原图</a></p></html>'''
    text = text.replace('<h1>', '<p>Caption 来源：'+html.escape(summary.get('caption_label', 'Official supplied captions'))+'</p><h1>', 1)
    (out/'report.html').write_text(text)


if __name__ == '__main__':
    main()
