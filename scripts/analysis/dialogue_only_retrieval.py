"""Dialogue-only evidence retrieval, paired with the pinned caption baseline."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import collections
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))

import numpy as np
from scripts.analysis.analyze import rank_metrics

NAT = ROOT / 'results/natural_observation'
OUT = ROOT / 'results/dialogue_only_retrieval'


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(name, values):
    (OUT / name).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def csvout(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def any_gold_hit(scores, gold, k=5):
    rounded = np.round(scores.astype(float), 6)
    best = rounded[gold].max()
    better = int(np.sum(rounded > best))
    tied = int(np.sum(rounded == best))
    tied_gold = int(np.sum(rounded[gold] == best))
    taken = max(0, min(tied, k - better))
    return 1 - math.comb(tied - tied_gold, taken) / math.comb(tied, taken) if taken <= tied - tied_gold else 1.


def boundary_groups(scores, ids, gold_ids, k=5):
    """Include every tied item at the boundary rather than arbitrary first IDs."""
    rounded = np.round(scores.astype(float), 6)
    cutoff = sorted(rounded, reverse=True)[min(k, len(ids)) - 1]
    result = []
    for score in sorted(set(rounded[rounded >= cutoff]), reverse=True):
        positions = np.flatnonzero(rounded == score)
        better = int(np.sum(rounded > score))
        slots = max(0, min(len(positions), k - better))
        result.append({'rounded_score': float(score), 'group_size': len(positions),
                       'available_top5_slots': slots, 'selection_probability_per_item': slots / len(positions),
                       'memory_ids': [ids[j] for j in positions],
                       'gold_ids': [ids[j] for j in positions if ids[j] in gold_ids]})
    return result


def mean(rows, field):
    return float(np.mean([r[field] for r in rows]))


def group_summary(label, rows):
    result = {'group': label, 'questions': len(rows)}
    for arm in ('caption_matched', 'dialogue_matched', 'dialogue_all_rounds'):
        for metric in ('recall_at_1', 'recall_at_5', 'recall_at_10', 'all_annotated_hit_at_5', 'any_gold_hit_at_5', 'random_recall_at_5'):
            result[f'{arm}_{metric}'] = mean(rows, f'{arm}_{metric}')
        result[f'{arm}_questions_with_gold_ties'] = sum(r[f'{arm}_gold_rank_ties'] > 0 for r in rows)
    result['dialogue_minus_caption_recall_at_5'] = mean(rows, 'dialogue_minus_caption_recall_at_5')
    return result


def cluster_intervals(rows):
    """Resample task histories jointly for the paired comparison."""
    tasks = sorted({r['task'] for r in rows})
    fields = ['dialogue_matched_recall_at_5', 'caption_matched_recall_at_5', 'dialogue_minus_caption_recall_at_5']
    counts = np.array([sum(r['task'] == t for r in rows) for t in tasks])
    sums = np.array([[sum(r[field] for r in rows if r['task'] == t) for field in fields] for t in tasks])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(tasks), size=(5000, len(tasks)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1, keepdims=True)
    return {field: {'estimate': mean(rows, field), 'low': float(np.percentile(estimates[:, i], 2.5)),
                    'high': float(np.percentile(estimates[:, i], 97.5)), 'task_clusters': len(tasks),
                    'bootstrap_replicates': 5000} for i, field in enumerate(fields)}


def report(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    arms = ['caption_matched', 'dialogue_matched', 'dialogue_all_rounds']
    colors = ['#3477ae', '#d27636', '#7f8995']
    overall = summary['groups'][0]
    task_rows = [r for r in summary['groups'] if r['group'].startswith('task:')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={'width_ratios': [1, 2.2]})
    values = [overall[f'{a}_recall_at_5'] for a in arms]
    bars = axes[0].bar(range(3), values, color=colors)
    axes[0].set(xticks=range(3), xticklabels=['Caption\n438 rounds', 'Dialogue\n438 rounds', 'Dialogue\n848 rounds'],
                ylim=(0, 1), title='All questions (n=371)', ylabel='Annotated-evidence Recall@5')
    axes[0].axhline(overall['dialogue_matched_random_recall_at_5'], color='#555', ls=':', label='Random: matched corpus')
    axes[0].legend(fontsize=8)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x()+bar.get_width()/2, value+.02, f'{value:.1%}', ha='center', fontsize=11)
    names = {'Brand_Memory_Test':'Brand', 'Card_Playlog_Test':'Card', 'Cartoon_Entertainment_Companion':'Cartoon',
             'Home_Renovation_Interior_Design':'Home', 'Multi-Scene_Visual_Case_Archive_Assistant':'CrossScene',
             'Outdoor_Navigation_Route_Memory_Assistant':'Outdoor', 'Personal_Health_Dashboard_Assistant':'Health',
             'Social_Chat_Memory_Test':'Social'}
    xx = np.arange(len(task_rows))
    for i, (arm, color) in enumerate(zip(arms, colors)):
        axes[1].bar(xx+(i-1)*.25, [r[f'{arm}_recall_at_5'] for r in task_rows], width=.25,
                    color=color, label=arm.replace('_', ' '))
    axes[1].plot(xx, [r['dialogue_matched_random_recall_at_5'] for r in task_rows], 'k_', ms=15, label='Random: matched corpus')
    axes[1].set(xticks=xx, xticklabels=[names[r['group'][5:]]+f"\nq={r['questions']}" for r in task_rows],
                ylim=(0, 1), title='By task; same question text and original gold clues')
    axes[1].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('MemEye: original dialogue utterances versus image captions')
    fig.text(.5, .015, 'Primary comparison: identical 438 image-bearing rounds. 848-round dialogue retrieval is a supplementary corpus change. Ties use expected recall.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT/f'recall_comparison.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    def pct(value):return f'{value:.1%}'
    tables = ''.join('<tr><td>'+html.escape(r['group'])+'</td><td>'+str(r['questions'])+'</td>'+''.join(
        '<td>'+pct(r[f'{a}_recall_at_5'])+'</td>' for a in arms)+'<td>'+pct(r['dialogue_matched_random_recall_at_5'])+'</td></tr>' for r in summary['groups'])
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MemEye 仅对话文本检索</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:20px}}img{{max-width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{border:1px solid #ddd;padding:7px}}</style>
<h1>MemEye：只用用户与助手发言检索</h1>
<p>371 道原始开放题；不修改题目、答案、clue、历史对话。检索器沿用 BGE-small-en-v1.5。主对照固定同一批 438 个含图轮次，将索引中的普通图像 caption 替换为该轮 user 与 assistant 原文。额外报告全部 848 个对话轮次的文本检索。</p>
<p>查询只含原题文字，8 道附图问题也不附加图像 caption。363 道无附图题使用原有完全相同的查询向量；8 道重新编码纯文字查询。主对照不索引日期、ID、人物档案、图片内容、原数据 image_caption 或答案；文字发言本身若提及时间或人物则自然保留。</p>
<p><b>主对照 Recall@5：对话 {pct(overall['dialogue_matched_recall_at_5'])}，caption {pct(overall['caption_matched_recall_at_5'])}；随机排序参考 {pct(overall['dialogue_matched_random_recall_at_5'])}。</b>完整 848 轮对话结果为 {pct(overall['dialogue_all_rounds_recall_at_5'])}。完整轮次增加了候选数量，不能将该变化完全归因于文本表示。</p>
<img src="recall_comparison.png">
<p>Recall@5 是每题 Top-5 找回的全部标注证据比例，再对题目取平均。多证据题不改成“任意一条命中”；另提供全部标注证据命中和任意命中指标。分数保留六位小数，相同得分按随机排列的期望计算，避免 Social Chat 等重复发言产生人为命中。Top-5 边界并列组完整保存在 retrieval_details.jsonl。</p>
<table><tr><th>分组</th><th>题数</th><th>caption，438轮</th><th>对话，438轮</th><th>对话，848轮</th><th>随机，438轮</th></tr>{tables}</table>
<p>仅对话检索表现高说明这一表示足以定位标注证据；该比较不能直接证明其他包含图像的系统主要依靠对话锚点。表现低也不能排除跨轮次组织或上下文窗口能够帮助定位。当前评测单位是单轮文本，未评测最终回答质量。</p>
<p><a href="per_question.csv">逐题对照</a> · <a href="group_summary.csv">分组统计</a> · <a href="memory_texts.jsonl">实际索引发言</a> · <a href="retrieval_details.jsonl">正确记忆排名与边界并列组</a> · <a href="summary.json">汇总与审计</a> · <a href="protocol.json">实验协议</a></p></html>'''
    (OUT/'report.html').write_text(doc)


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8);torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((NAT/'protocol.json').read_text())
    for item in config['source_files']:
        assert file_hash(ROOT/item['path']) == item['sha256'], item['path']
    matched = read(NAT/'memories.jsonl');queries = read(NAT/'queries.jsonl');gold = {r['id']:r for r in read(NAT/'gold.jsonl')}
    original_files = {ROOT/r['path'] for r in config['source_files']}
    all_rounds = []
    for path in sorted(original_files):
        task = path.stem.removesuffix('_Open');source = json.loads(path.read_text())
        for session in source['multi_session_dialogues']:
            for turn in session['dialogues']:
                assert isinstance(turn['user'], str) and isinstance(turn['assistant'], str)
                all_rounds.append({'id':task+'/'+turn['round'], 'task':task, 'session_id':session['session_id'],
                                   'user':turn['user'], 'assistant':turn['assistant'],
                                   'indexed_text':turn['user']+'\n'+turn['assistant'],
                                   'has_image':bool(turn.get('input_image'))})
    round_map = {r['id']:r for r in all_rounds}
    assert len(round_map) == len(all_rounds) == 848 and len(matched) == 438 and len(queries) == 371
    assert {r['id'] for r in all_rounds if r['has_image']} == {r['id'] for r in matched}
    assert all(g in round_map for r in gold.values() for g in r['clue_ids'])
    model = SentenceTransformer(config['retriever_model_path'], device='cpu', local_files_only=True)
    texts = list(dict.fromkeys(r['indexed_text'] for r in all_rounds));text_positions = {t:i for i,t in enumerate(texts)}
    lengths = [len(model.tokenizer.encode(t)) for t in texts]
    query_lengths = [len(model.tokenizer.encode(config['query_prefix']+q['question'])) for q in queries]
    assert max(lengths+query_lengths) <= model.max_seq_length, 'Unexpected truncation'
    print(f'Encoding {len(texts)} distinct dialogue texts for {len(all_rounds)} original rounds.', flush=True)
    unique_emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    dialogue_emb = np.array([unique_emb[text_positions[r['indexed_text']]] for r in all_rounds])
    cache = np.load(NAT/'embeddings.npz')
    assert list(cache['memory_ids']) == [r['id'] for r in matched]
    assert list(cache['query_ids']) == [q['id'] for q in queries]
    qemb = cache['query'].copy()
    sources = {r['id']:r['source_text'] for r in read(NAT/'query_cues.jsonl')}
    changed = []
    for i, q in enumerate(queries):
        if q['images']:changed.append(i)
        else:assert sources[q['id']] == q['question']
    if changed:
        qemb[changed] = model.encode([config['query_prefix']+queries[i]['question'] for i in changed],
                                    batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    assert len(changed) == 8
    all_pos = {r['id']:i for i,r in enumerate(all_rounds)}
    matched_dialogue_emb = dialogue_emb[[all_pos[r['id']] for r in matched]]
    arms = {'caption_matched':(matched, cache['memory']), 'dialogue_matched':(matched, matched_dialogue_emb),
            'dialogue_all_rounds':(all_rounds, dialogue_emb)}
    task_indices = {arm:{task:[i for i,r in enumerate(memories) if r['task']==task] for task in sorted({q['task'] for q in queries})}
                    for arm,(memories,_) in arms.items()}
    rows = [];details = []
    for qi, q in enumerate(queries):
        g = gold[q['id']]
        row = {'id':q['id'], 'task':q['task'], 'x':g['x'], 'y':g['y'], 'has_query_image':bool(q['images']), 'gold_count':len(g['clue_ids'])}
        detail = {'id':q['id'], 'question':q['question'], 'gold_ids':g['clue_ids'], 'arms':{}}
        for arm,(memories,emb) in arms.items():
            inds = task_indices[arm][q['task']];ids = [memories[i]['id'] for i in inds]
            local_gold = [ids.index(mid) for mid in g['clue_ids']]
            scores = emb[inds] @ qemb[qi]
            metrics, ranks = rank_metrics(scores, local_gold)
            for key, value in metrics.items():row[arm+'_'+key] = value
            row[arm+'_memory_count'] = len(inds)
            row[arm+'_random_recall_at_5'] = min(5/len(inds), 1.)
            row[arm+'_any_gold_hit_at_5'] = any_gold_hit(scores, local_gold)
            detail['arms'][arm] = {'gold':[{'id':ids[j], 'midrank':float(ranks[j]), 'score':float(scores[j]),
                                                  'dialogue_text':round_map[ids[j]]['indexed_text']} for j in local_gold],
                                   'top5_score_groups':boundary_groups(scores, ids, set(g['clue_ids'])),
                                   'metrics':metrics}
        row['dialogue_minus_caption_recall_at_5'] = row['dialogue_matched_recall_at_5']-row['caption_matched_recall_at_5']
        rows.append(row);details.append(detail)
    groups = [group_summary('all', rows), group_summary('text_only_questions', [r for r in rows if not r['has_query_image']]),
              group_summary('image_questions_text_only', [r for r in rows if r['has_query_image']])]
    for field in ('task','x','y'):
        for value in sorted({r[field] for r in rows}):
            groups.append(group_summary(field+':'+value, [r for r in rows if r[field]==value]))
    for x in ('X1','X2','X3','X4'):
        for y in ('Y1','Y2','Y3'):
            subset = [r for r in rows if r['x']==x and r['y']==y]
            groups.append(group_summary(x+'/'+y, subset))
    baseline = json.loads((NAT/'summary.json').read_text())
    prior_text_only = next(r for r in baseline['groups'] if r['group']=='text_only_queries')
    assert abs(groups[1]['caption_matched_recall_at_5']-prior_text_only['mean_recall_at_5']) < 1e-12
    hashes = {name:file_hash(NAT/name) for name in ['protocol.json','memories.jsonl','queries.jsonl','gold.jsonl','embeddings.npz','query_cues.jsonl','summary.json']}
    counts = collections.Counter(r['indexed_text'] for r in all_rounds)
    audit = {'questions':len(rows), 'matched_rounds':len(matched), 'all_rounds':len(all_rounds),
             'unique_dialogue_texts':len(texts), 'largest_identical_dialogue_group':max(counts.values()),
             'memory_max_bge_tokens':max(lengths), 'query_max_bge_tokens':max(query_lengths),
             'bge_max_seq_length':model.max_seq_length, 'truncated_inputs':0,
             'reencoded_image_question_queries':len(changed), 'caption_baseline_363_text_only_reproduced':True,
             'baseline_input_hashes':hashes}
    protocol = {'study':'Dialogue-only retrieval ablation on unchanged MemEye histories',
                'provenance':config['provenance'], 'retriever':config['retriever'], 'retriever_revision':config['retriever_revision'],
                'query_prefix':config['query_prefix'], 'query':'original Open question text only; no query-image captions',
                'indexed_dialogue':'user + newline + assistant from the same original round; no surrounding rounds or extra metadata',
                'primary_corpus':'same 438 image-bearing rounds as caption baseline',
                'supplementary_corpus':'all 848 original rounds; increased candidate count is explicitly separate',
                'retrieval_scope':config['retrieval_scope'], 'gold':'all original clue rounds; no necessity relabeling',
                'primary_metric':'macro mean over questions of the fraction of annotated gold rounds retrieved in Top-5',
                'ties':'scores rounded to six decimals; expected recall under random ordering inside each tie group',
                'random_reference':'min(5/task corpus size,1), macro-averaged using the same question weights',
                'uncertainty':'paired whole-task bootstrap, 8 source tasks, 5000 draws, seed 42',
                'execution':'local BGE on CPU, offline, 8 threads, no VLM generation or data mutation',
                'source_files':config['source_files'], 'input_hashes':hashes}
    summary = {'groups':groups, 'paired_cluster_bootstrap':cluster_intervals(rows), 'audit':audit}
    dump('protocol.json', protocol);dump('summary.json', summary)
    jsonl('memory_texts.jsonl', all_rounds);jsonl('queries.jsonl', [{'id':q['id'], 'query_text':q['question']} for q in queries])
    jsonl('retrieval_details.jsonl', details);csvout('per_question.csv', rows);csvout('group_summary.csv', groups)
    np.savez_compressed(OUT/'embeddings.npz', dialogue_memory=dialogue_emb, query=qemb,
                        memory_ids=np.array([r['id'] for r in all_rounds]), query_ids=np.array([q['id'] for q in queries]))
    report(summary)
    for name, digest in hashes.items():assert file_hash(NAT/name) == digest
    dump('artifact_manifest.json', {p.name:file_hash(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps({'overall':groups[0], 'by_y':[g for g in groups if g['group'].startswith('y:')], 'audit':audit}, ensure_ascii=False, indent=2), flush=True)


if __name__=='__main__':
    main()
