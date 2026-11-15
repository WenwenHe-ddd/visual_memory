"""Experiment 1: full-task caption retrieval, with one sample per clue round."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))
import numpy as np

NAT = ROOT / 'results/natural_observation'
MCQ = ROOT / 'results/mcq_fifth_non_target_rank'
EXP2 = ROOT / 'results/random_task_injection'
OUT = ROOT / 'results/full_history_target_retrieval'
TASK_NAMES = {
    'Brand_Memory_Test': 'Brand', 'Card_Playlog_Test': 'Card',
    'Cartoon_Entertainment_Companion': 'Cartoon',
    'Home_Renovation_Interior_Design': 'Home',
    'Multi-Scene_Visual_Case_Archive_Assistant': 'Cross-scene',
    'Outdoor_Navigation_Route_Memory_Assistant': 'Outdoor',
    'Personal_Health_Dashboard_Assistant': 'Health',
    'Social_Chat_Memory_Test': 'Social',
}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jsonl(name, rows):
    (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))


def summarize(group, rows):
    valid = [row for row in rows if row['included_in_similarity_analysis']]
    hit = [row['competitor_similarity'] for row in valid if row['retrieved_at_5']]
    miss = [row['competitor_similarity'] for row in valid if not row['retrieved_at_5']]
    hm = float(np.mean(hit)) if hit else None
    mm = float(np.mean(miss)) if miss else None
    return {
        'group': group, 'all_target_samples': len(rows),
        'all_questions': len({row['question_id'] for row in rows}),
        'all_retrieved_samples': sum(row['retrieved_at_5'] for row in rows),
        'all_target_recall_at_5': float(np.mean([row['retrieved_at_5'] for row in rows])),
        'eligible_target_samples': len(valid),
        'eligible_questions': len({row['question_id'] for row in valid}),
        'excluded_target_samples': len(rows) - len(valid),
        'retrieved_n': len(hit), 'missed_n': len(miss),
        'retrieved_mean': hm, 'missed_mean': mm,
        'missed_minus_retrieved': mm - hm if hm is not None and mm is not None else None,
    }


def figures(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 11})

    def bars(ax, group):
        for x, arm, color, label in [
            (0, 'retrieved', '#2a7e9b', 'Retrieved (Top-5)'),
            (1, 'missed', '#d17b45', 'Missed (outside Top-5)'),
        ]:
            value = group[arm + '_mean']
            n = group[arm + '_n']
            if value is not None:
                ax.bar(x, value, color=color, width=.65, label=label)
                ax.text(x, value + .028, f'{value:.3f}\nn={n}', ha='center', va='bottom', fontsize=11)
        ax.set_xticks([0, 1], ['Retrieved', 'Missed'])
        ax.set_ylim(0, 1.115)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(5.2, 5.7))
    bars(ax, groups[0])
    ax.set_ylabel('Mean target-to-non-key caption cosine similarity')
    ax.set_title('Full-history retrieval: pooled targets')
    fig.text(.5, .015, 'One sample = one annotated target round.\nFive non-keys selected from the full query ranking.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'overall_retrieval_similarity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for ax, group in zip(axes.flat, groups[1:]):
        bars(ax, group)
        ax.set_ylim(0, 1.23)
        ax.set_title(f"{TASK_NAMES[group['group']]}\n{group['eligible_target_samples']} targets / {group['eligible_questions']} questions")
        ax.set_ylabel('Mean caption cosine similarity')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(.5, 0))
    fig.suptitle('Full-history caption retrieval: target similarity by retrieval outcome', fontsize=17)
    fig.text(.5, .06, 'Retrieved = this target in Top-5. Five competitors exclude ALL annotated targets of the question.\n'
             'Each target has equal weight; the original complete task history is retained.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, .94))
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'by_task_retrieval_similarity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def report(groups, audit):
    cells = []
    for group in groups:
        name = 'Overall' if group['group'] == 'all' else TASK_NAMES[group['group']]
        cells.append('<tr><td>' + name + '</td>' + ''.join(
            '<td>' + (f'{group[key]:.4f}' if 'mean' in key or key == 'missed_minus_retrieved' else str(group[key])) + '</td>'
            for key in ['retrieved_n', 'missed_n', 'retrieved_mean', 'missed_mean', 'missed_minus_retrieved']) + '</tr>')
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>实验一：完整历史中的逐目标检索对比</title>
<style>body{font:16px/1.7 system-ui;max-width:1300px;margin:24px auto;padding:20px}img{max-width:100%}.overall{width:470px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}code{background:#f3f3f3}li{margin:8px 0}</style>
<h1>实验一：完整图片历史，按每个目标的检索结果分组</h1>
<p>本页采用用户最新修订：成功／失败指该条原始 clue 记忆是否进入查询 Top-5。分析单位是“题目 × 标注目标 round”，每个目标权重相同；多线索题拆成多个样本，同一道题的目标可以分属两组。</p>
<h2>计算方式</h2><ol>
<li>每个任务的完整含图 round 历史构成记忆库，共 438 条。仅使用普通 Qwen3-VL-8B-Instruct 图片 caption 的 BGE-small-en-v1.5 归一化向量；完整历史保留重复图片对应的原始 round。</li>
<li>查询使用 MemEye 原始选择题 question 文本和固定 BGE 查询前缀。按查询与 caption 的余弦相似度排列所属任务的全部记忆。</li>
<li>每条原始 clue 为一个目标。该目标原始排名 ≤ 5 即成功，否则失败。其他同题目标是否进入 Top-5 不影响它的分组。</li>
<li>在完整查询排名中排除该题全部 clue，然后取最靠前的五条非关键记忆。五条可排在原始 Top-5 之后，成功组与失败组使用完全相同的选择规则。</li>
<li>每个目标与这五条 caption 分别计算余弦，再取五个数的平均。最后在成功、失败目标样本内直接求均值，不再先平均回题目。</li>
</ol>
<p>371 道题共有 1,041 个目标，426 个进入 Top-5、615 个未进入。5 道题没有非关键记忆，因此它们的 163 个目标只保留检索结果。相似度图使用其余 366 道题的 878 个目标：401 个成功、477 个失败。这 878 个“题目 × 目标”与实验二的主样本集合完全相同。</p>
<h2>原始总体结果</h2><img class="overall" src="overall_retrieval_similarity.png">
<p>总体成功均值 0.7907，失败均值 0.8391，失败高 0.0484。总体直接汇总全部目标，不按任务重新加权。</p>
<h2>分任务结果</h2><img src="by_task_retrieval_similarity.png">
<table><tr><th>任务</th><th>成功目标数</th><th>失败目标数</th><th>成功均值</th><th>失败均值</th><th>失败－成功</th></tr>''' + ''.join(cells) + '''</table>
<p>8 个任务中，Health 的失败均值高于成功均值，其余 7 个任务相反。总体柱状图与任务内对比同时保留；各任务的相似度水平和两组样本构成不同。</p>
<h2>审查与解释</h2>
<p>原始数据、caption 和向量均未修改。排名复用已保存的完整 MCQ 查询排序，并重新核对每个任务的完整记忆集合、原始 clue、得分和所有选中的非关键 ID。查询排序沿用六位小数与原始记忆顺序处理并列；没有目标在 Top-5 边界出现分组不确定。NON_KEY_TIE_COUNT 道题在非关键第五位与第六位存在并列，沿用已保存的稳定顺序，恰好选五条。</p>
<p>这些是描述性均值，目标样本共享问题和记忆库，不能视为独立观测。这里测量的是 BGE caption 语义相似度；总体相关性不能给出“多少失败由过载造成”的因果比例。</p>
<p><a href="per_target.csv">逐目标结果与原始排名</a> · <a href="per_question.csv">逐题目标覆盖与非关键条目排名</a> · <a href="caption_audit.jsonl">问题、目标、五个非关键 caption 与图片路径</a> · <a href="excluded_questions.csv">无可比非关键条目的题目</a> · <a href="group_summary.csv">分组均值</a> · <a href="protocol.json">协议与输入哈希</a></p>
<p><a href="overall_retrieval_similarity.svg">总体 SVG</a> · <a href="overall_retrieval_similarity.pdf">总体 PDF</a> · <a href="by_task_retrieval_similarity.svg">分任务 SVG</a> · <a href="by_task_retrieval_similarity.pdf">分任务 PDF</a></p>
<p>对应干预实验：<a href="../random_task_injection/report.html">实验二：同任务／其他任务干扰注入</a>。旧分析保留：<a href="../mcq_top15_caption_qa/report.html">Top-15 选择题正确性分组</a>，其结果不作为本版实验一。</p></html>'''
    text = text.replace('NON_KEY_TIE_COUNT', str(len(audit['non_key_fifth_boundary_tie_question_ids'])))
    (OUT / 'report.html').write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((MCQ / 'protocol.json').read_text())
    nat_cfg = json.loads((NAT / 'protocol.json').read_text())
    source_paths = [ROOT / item['path'] for item in cfg['source_files']]
    for path, item in zip(source_paths, cfg['source_files']):
        assert sha(path) == item['sha256'], path
    for relative_path, expected in cfg['input_hashes'].items():
        assert sha(ROOT / relative_path) == expected, relative_path
    input_paths = source_paths + [NAT / name for name in ['protocol.json', 'memories.jsonl', 'captions.jsonl', 'images.jsonl', 'embeddings.npz']]
    input_paths += [MCQ / name for name in ['protocol.json', 'mcq_queries.jsonl', 'query_embeddings.npz', 'retrieval_details.jsonl']]
    input_paths += [EXP2 / 'sample_means.csv']
    before = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}

    memories = read(NAT / 'memories.jsonl')
    captions = {row['id']: row['caption'] for row in read(NAT / 'captions.jsonl')}
    images = {row['id']: row for row in read(NAT / 'images.jsonl')}
    queries = read(MCQ / 'mcq_queries.jsonl')
    saved = {row['id']: row for row in read(MCQ / 'retrieval_details.jsonl')}
    memory_cache = np.load(NAT / 'embeddings.npz')
    query_cache = np.load(MCQ / 'query_embeddings.npz')
    assert list(memory_cache['memory_ids']) == [row['id'] for row in memories]
    assert list(query_cache['query_ids']) == [row['id'] for row in queries]
    emb, qemb = memory_cache['memory'], query_cache['query']
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(qemb, axis=1), 1, atol=1e-6)
    mi = {row['id']: i for i, row in enumerate(memories)}
    task_indices = {task: [i for i, row in enumerate(memories) if row['task'] == task] for task in TASK_NAMES}

    original = {}
    for path in source_paths:
        data = json.loads(path.read_text())
        rounds = {row['round']: row for session in data['multi_session_dialogues'] for row in session['dialogues']}
        image_rounds = {path.stem + '/' + rid for rid, row in rounds.items() if row.get('input_image')}
        assert image_rounds == {memories[i]['id'] for i in task_indices[path.stem]}
        for question in data['human-annotated QAs']:
            qid = path.stem + '/' + question['question_id']
            gold = list(dict.fromkeys(path.stem + '/' + rid for rid in question['clue']))
            assert len(gold) == len(question['clue']), qid
            original[qid] = {'question': question['question'], 'gold_ids': gold,
                             'x': question['point'][0][0], 'y': question['point'][1][0],
                             'has_query_image': bool(question.get('question_image'))}
    assert set(original) == set(saved) == {row['id'] for row in queries}
    for image_id in {image_id for row in memories for image_id in row['images']}:
        assert sha(Path(images[image_id]['path'])) == image_id

    def memory_detail(mid):
        memory = memories[mi[mid]]
        return {'id': mid, 'caption': ' '.join(captions[image_id] for image_id in memory['images']),
                'image_paths': [images[image_id]['path'] for image_id in memory['images']]}

    rows, questions, details, excluded = [], [], [], []
    non_key_ties, target_ties = [], []
    for qi, query in enumerate(queries):
        qid = query['id']
        raw, retrieval = original[qid], saved[qid]
        assert raw['question'] == query['question'] == retrieval['question']
        assert raw['gold_ids'] == retrieval['gold_ids']
        gold = set(raw['gold_ids'])
        ranking = retrieval['ranking']
        inds = task_indices[query['task']]
        assert len(ranking) == len(inds)
        assert {row['id'] for row in ranking} == {memories[i]['id'] for i in inds}
        assert [row['rank'] for row in ranking] == list(range(1, len(inds) + 1))
        scores = emb[inds] @ qemb[qi]
        rounded = np.round(scores.astype(float), 6)
        order = np.argsort(-rounded, kind='stable')
        assert [row['id'] for row in ranking] == [memories[inds[int(j)]]['id'] for j in order]
        assert np.allclose([row['score'] for row in ranking], scores[order], rtol=0, atol=1e-7)
        assert all(row['is_target'] == (row['id'] in gold) for row in ranking)
        rank_map = {row['id']: row for row in ranking}
        non_keys = [row for row in ranking if row['id'] not in gold]
        selected = non_keys[:5]
        assert [row['id'] for row in selected] == [row['id'] for row in retrieval['first_five_non_targets']]
        eligible = len(selected) == 5
        reason = '' if eligible else 'fewer_than_five_non_key_rounds'
        cutoff = rounded[order[4]]
        tied_ids = [memories[inds[j]]['id'] for j in range(len(inds)) if rounded[j] == cutoff]
        if len(tied_ids) > 1 and sum(rounded > cutoff) < 5 < sum(rounded >= cutoff) and gold.intersection(tied_ids):
            target_ties.append(qid)
        non_key_boundary_tie = False
        if eligible and len(non_keys) > 5:
            non_key_boundary_tie = round(non_keys[4]['score'], 6) == round(non_keys[5]['score'], 6)
            if non_key_boundary_tie:
                non_key_ties.append(qid)

        question_detail = {'question_id': qid, 'task': query['task'], 'question': query['question'],
                           'gold_ids': raw['gold_ids'], 'targets': [], 'non_key_top5': []}
        for row in selected:
            question_detail['non_key_top5'].append({**memory_detail(row['id']),
                'original_query_rank': row['rank'], 'query_caption_cosine': row['score']})
        for mid in raw['gold_ids']:
            target = rank_map[mid]
            sims = [float(emb[mi[mid]] @ emb[mi[row['id']]]) for row in selected] if eligible else []
            similarity = float(np.mean(sims, dtype=np.float64)) if eligible else None
            row = {
                'sample_id': qid + '|' + mid, 'question_id': qid, 'task': query['task'],
                'x': raw['x'], 'y': raw['y'], 'question': query['question'],
                'has_query_image': raw['has_query_image'], 'target_id': mid,
                'target_query_rank': target['rank'], 'target_query_cosine': target['score'],
                'retrieved_at_5': target['rank'] <= 5,
                'outcome': 'retrieved' if target['rank'] <= 5 else 'missed',
                'competitor_similarity': similarity, 'included_in_similarity_analysis': eligible,
                'exclusion_reason': reason, 'memory_round_count': len(ranking),
                'question_gold_count': len(gold), 'non_key_round_count': len(non_keys),
                'all_question_gold_ids': '|'.join(raw['gold_ids']),
                'five_non_key_ids': '|'.join(row['id'] for row in selected),
                'five_non_key_original_ranks': '|'.join(str(row['rank']) for row in selected),
                'fifth_non_key_original_rank': selected[-1]['rank'] if eligible else None,
                'target_to_non_key_cosines': json.dumps(sims),
                'non_key_selection_boundary_tie': non_key_boundary_tie,
                'target_caption': memory_detail(mid)['caption'],
                'target_image_paths': '|'.join(memory_detail(mid)['image_paths']),
            }
            assert not gold.intersection(row['id'] for row in selected)
            rows.append(row)
            question_detail['targets'].append({**memory_detail(mid), 'sample_id': row['sample_id'],
                'original_query_rank': target['rank'], 'retrieved_at_5': row['retrieved_at_5'],
                'competitor_similarity': similarity, 'target_to_non_key_cosines': sims,
                'included_in_similarity_analysis': eligible, 'exclusion_reason': reason})
        qrow = {'question_id': qid, 'task': query['task'], 'x': raw['x'], 'y': raw['y'],
                'question': query['question'], 'memory_round_count': len(ranking),
                'gold_count': len(gold), 'retrieved_target_count': sum(rank_map[mid]['rank'] <= 5 for mid in gold),
                'non_key_round_count': len(non_keys), 'included_in_similarity_analysis': eligible,
                'exclusion_reason': reason, 'five_non_key_ids': '|'.join(row['id'] for row in selected),
                'five_non_key_original_ranks': '|'.join(str(row['rank']) for row in selected),
                'fifth_non_key_original_rank': selected[-1]['rank'] if eligible else None}
        questions.append(qrow)
        details.append(question_detail)
        if not eligible:
            excluded.append(qrow)
    assert not target_ties, target_ties
    assert len({row['sample_id'] for row in rows}) == len(rows)
    with (EXP2 / 'sample_means.csv').open(encoding='utf-8-sig', newline='') as f:
        exp2_ids = {row['sample_id'] for row in csv.DictReader(f)}
    valid_ids = {row['sample_id'] for row in rows if row['included_in_similarity_analysis']}
    assert valid_ids == exp2_ids
    groups = [summarize('all', rows)] + [summarize(task, [row for row in rows if row['task'] == task]) for task in TASK_NAMES]
    strata = [summarize(label, [row for row in rows if row[axis] == label]) for axis, levels in [('x', ['X1', 'X2', 'X3', 'X4']), ('y', ['Y1', 'Y2', 'Y3'])] for label in levels]
    audit = {'all_questions': len(queries), 'all_memory_rounds': len(memories),
             'memory_counts_by_task': {task: len(inds) for task, inds in task_indices.items()},
             'all_target_samples': len(rows), 'eligible_questions': sum(row['included_in_similarity_analysis'] for row in questions),
             'eligible_target_samples': len(valid_ids), 'excluded_questions': len(excluded),
             'excluded_target_samples': len(rows) - len(valid_ids),
             'target_top5_boundary_tie_questions': target_ties,
             'non_key_fifth_boundary_tie_question_ids': non_key_ties,
             'eligible_sample_ids_match_experiment_2': True,
             'all_full_task_rankings_reproduced': True, 'all_original_mcq_clues_verified': True,
             'all_original_image_contents_verified': True, 'source_files_unchanged': True,
             'model_inference_calls': 0,
             'within_task_missed_lower': sum(group['missed_minus_retrieved'] < 0 for group in groups[1:])}
    protocol = {
        'study': 'Experiment 1: full-history target retrieval and target-to-non-key caption similarity',
        'status': 'complete', 'dataset': cfg['dataset'], 'source_files': cfg['source_files'],
        'caption_model': nat_cfg['provenance']['caption_model'], 'caption_prompt': nat_cfg['caption_prompt'],
        'retriever': cfg['retriever'], 'retriever_revision': cfg['retriever_revision'],
        'query': 'original MCQ question text plus the fixed BGE prefix', 'query_prefix': cfg['query_prefix'],
        'query_images': 'not encoded in this question-text-only retrieval',
        'memory': 'complete original image-bearing round history within each task; ordinary caption embeddings only',
        'memory_duplicate_policy': 'retain every original round, including repeated image content',
        'analysis_unit': 'one (question_id, original annotated clue round_id) pair',
        'retrieval_top_k': 5, 'success': 'this specific target has original query rank <= 5',
        'failure': 'this specific target has original query rank > 5',
        'non_keys': 'exclude ALL original clue rounds for the question; take the first FIVE remaining entries from the full query ranking, regardless of original rank',
        'similarity': 'mean of five cosine similarities between the normalized BGE target caption vector and the selected non-key caption vectors',
        'aggregation': 'equal-weight mean of individual target samples within each retrieval outcome; no within-question averaging',
        'overall': 'raw pooling of eligible target samples from all eight tasks, without task reweighting',
        'exclusions': 'fewer than five non-key rounds; retain their target retrieval outcomes in the audit but omit similarity means',
        'more_than_five_gold_rounds': 'retained; each original clue is evaluated separately',
        'ties': 'reuse full saved MCQ ranking: six-decimal query cosines, stable source memory order; select exactly five non-keys; no annotated target has ambiguous Top-5 boundary membership',
        'interpretation': 'descriptive association with retrieval outcome; shared queries and histories; no causal attribution fraction or significance claim',
        'execution': 'offline reuse and verification of caption vectors and complete MCQ rankings; zero model inference',
        'experiment_2_sample_cohort': 'same eligible question-target pairs; Experiment 1 uses full histories, Experiment 2 uses target + N sampled distractors',
        'input_hashes': before, 'analysis_script': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
    }
    for relative_path, expected in before.items():
        assert sha(ROOT / relative_path) == expected, relative_path
    dump('protocol.json', protocol)
    dump('summary.json', {'overall': groups[0], 'by_task': groups[1:], 'by_x_y': strata, 'audit': audit})
    csvout('per_target.csv', rows)
    csvout('per_question.csv', questions)
    csvout('excluded_questions.csv', excluded)
    csvout('group_summary.csv', groups)
    csvout('x_y_summary.csv', strata)
    jsonl('caption_audit.jsonl', details)
    figures(groups)
    report(groups, audit)
    dump('artifact_manifest.json', {'source_files_unchanged': True, 'outputs': {
        path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != 'artifact_manifest.json'}})
    print(json.dumps({'overall': groups[0], 'by_task': groups[1:], 'audit': audit}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
