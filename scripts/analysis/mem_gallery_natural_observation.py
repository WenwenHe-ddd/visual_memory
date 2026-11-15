"""Mem-Gallery caption-only natural observation, using supplied captions verbatim.

This ports the full-history target-level analysis, not the official full benchmark.
No Qwen calls, answer generation, or image decoding are needed.
"""
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
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/.matplotlib'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def csvout(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                          for k, v in row.items()} for row in rows)


def prepare(manifest, caption_dir=None):
    memories, queries, golds, coverage = [], [], [], []
    counts = collections.Counter()
    for source in manifest['files']:
        path = ROOT / source['path']
        assert sha(path) == source['sha256'], path
        data = json.loads(path.read_text())
        caption_field = 'image_caption'
        if caption_dir is not None:
            import copy
            derived = json.loads((caption_dir/'dialog_with_qwen_caption'/path.name).read_text())
            stripped = copy.deepcopy(derived)
            caption_field = 'image_caption_qwen3_vl_8b_instruct'
            for session in stripped['multi_session_dialogues']:
                for row in session['dialogues']:
                    row.pop(caption_field, None)
            for qa in stripped['human-annotated QAs']:
                qa.pop(caption_field, None)
            assert stripped == data, path
            data = derived
        task = path.stem
        rounds = {}
        for session in data['multi_session_dialogues']:
            counts['sessions'] += 1
            for row in session['dialogues']:
                rid = task + '/' + row['round']
                assert rid not in rounds, rid
                rounds[rid] = row
                counts['rounds'] += 1
                images = row.get('input_image') or []
                if not images:
                    continue
                caps = row.get(caption_field) or []
                assert len(images) == len(caps) and all(isinstance(c, str) and c.strip() for c in caps), rid
                memories.append({'id': rid, 'task': task, 'date': session['date'],
                                 'images': images, 'image_ids': row.get('image_id', []),
                                 'caption': ' '.join(caps), 'source_captions': caps})
        visual_ids = {m['id'] for m in memories if m['task'] == task}
        for index, qa in enumerate(data['human-annotated QAs']):
            qid = task + f'/QA_{index:04d}'
            has_image = bool(qa.get('question_image'))
            caption = qa.get(caption_field) if has_image else None
            assert not has_image or isinstance(caption, str) and caption.strip(), qid
            # Match the official textual-memory query construction exactly.
            text = qa['question']
            if caption:
                text += "\nquestion's image:" + '\nimage_caption: ' + caption
            queries.append({'id': qid, 'task': task, 'qa_index_0_based': index,
                            'point': qa['point'], 'question': qa['question'],
                            'question_image': qa.get('question_image'),
                            'provided_caption': caption, 'retrieval_text': text})
            gids = [task + '/' + r for r in qa['clue']]
            assert len(set(gids)) == len(gids) and set(gids) <= set(rounds), qid
            vids = [g for g in gids if g in visual_ids]
            missing = [g for g in gids if g not in visual_ids]
            primary = has_image and bool(gids) and not missing
            reason = ('no_annotated_clue' if not gids else 'text_only_question' if not has_image
                      else 'contains_nonvisual_clue' if missing else '')
            golds.append({'id': qid, 'all_clue_ids': gids, 'visual_clue_ids': vids,
                          'nonvisual_clue_ids': missing, 'primary_cohort': primary})
            coverage.append({'question_id': qid, 'task': task, 'point': qa['point'],
                             'has_query_image': has_image, 'provided_query_caption': bool(caption),
                             'all_clue_count': len(gids), 'visual_clue_count': len(vids),
                             'nonvisual_clue_count': len(missing), 'primary_cohort': primary,
                             'primary_exclusion_reason': reason, 'nonvisual_clue_ids': missing})
    counts.update(scenarios=len(manifest['files']), questions=len(queries), image_memory_rounds=len(memories))
    assert len({m['id'] for m in memories}) == len(memories)
    assert len({q['id'] for q in queries}) == len(queries)
    return memories, queries, golds, coverage, dict(counts)


def summarize(label, rows):
    import numpy as np
    eligible = [r for r in rows if r['included_in_similarity_analysis']]
    hit = [r['competitor_similarity'] for r in eligible if r['retrieved_at_5']]
    miss = [r['competitor_similarity'] for r in eligible if not r['retrieved_at_5']]
    hm, mm = (float(np.mean(x)) if x else None for x in (hit, miss))
    return {'group': label, 'questions': len({r['question_id'] for r in rows}), 'targets': len(rows),
            'retrieved_targets': sum(r['retrieved_at_5'] for r in rows),
            'target_recall_at_5': sum(r['retrieved_at_5'] for r in rows) / len(rows) if rows else None,
            'eligible_targets': len(eligible), 'retrieved_n': len(hit), 'missed_n': len(miss),
            'retrieved_mean': hm, 'missed_mean': mm,
            'missed_minus_retrieved': mm - hm if hm is not None and mm is not None else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/mem_gallery_natural_observation/v1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--caption-dir', type=Path, help='Verified Qwen caption cache; omit for official captions')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory to preserve previous runs')
    out.mkdir(parents=True, exist_ok=True)
    nat = json.loads((ROOT / 'results/natural_observation/protocol.json').read_text())
    manifest_path = ROOT / 'data/Mem-Gallery/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    caption_dir = args.caption_dir.resolve() if args.caption_dir else None
    caption_inputs = {}
    if caption_dir:
        assert json.loads((caption_dir/'independent_audit.json').read_text())['status'] == 'passed'
        cache_manifest = json.loads((caption_dir/'artifact_manifest.json').read_text())
        for p in [caption_dir/'protocol.json', caption_dir/'independent_audit.json',
                  *sorted((caption_dir/'dialog_with_qwen_caption').glob('*.json'))]:
            digest = sha(p)
            assert digest == cache_manifest[str(p.relative_to(caption_dir))], p
            caption_inputs[str(p.relative_to(ROOT))] = digest
    caption_label = 'Qwen3-VL-8B-Instruct captions' if caption_dir else 'Official supplied captions'
    memories, queries, golds, coverage, counts = prepare(manifest, caption_dir)
    protocol = {'study': 'Mem-Gallery natural observation of caption competition',
                'status': 'prepared', 'dataset': manifest, 'source_manifest_sha256': sha(manifest_path),
                'script_sha256': sha(Path(__file__)), 'retriever': nat['retriever'],
                'retriever_revision': nat['retriever_revision'], 'model_path': nat['retriever_model_path'],
                'query_prefix': nat['query_prefix'], 'seed': nat['seed'], 'device': args.device,
                'query_policy': 'question + official literal separator + qa.image_caption when question_image exists',
                'caption_policy': 'Supplied historical and QA captions, verbatim; no caption generation',
                'memory_unit': 'one image-bearing round; concatenate full captions in source order',
                'index_content': 'captions only, without dialogue/date/ID/answer/clue',
                'scope': 'complete same-scenario image-bearing history, retaining original rounds',
                'primary_cohort': 'image-bearing questions with nonempty clues all present in image memory',
                'supplementary_cohort': 'all visual clue targets, explicitly excluding nonvisual clues from denominator',
                'outcome': 'each annotated visual target independently succeeds if original query rank <= 5',
                'competitors': 'first five in full query ranking after excluding ALL clues of the question',
                'statistic': 'mean caption cosine from target to five competitors; then target-equal group means',
                'ties': 'round query scores to 6 decimals; stable source order; record boundary ambiguity',
                'missing_competitors': 'retain retrieval outcome, exclude similarity if fewer than five nonkeys',
                'interpretation': 'descriptive association, not causation or answer accuracy; not official NaiveRAG reproduction',
                'official_difference': 'official textual memory includes dialogue text; this ports our caption-only observation',
                'counts': counts}
    protocol.update(caption_label=caption_label, caption_dir=str(caption_dir) if caption_dir else None,
                    caption_input_hashes=caption_inputs)
    if caption_dir:
        protocol['query_policy'] = 'question + official literal separator + cached Qwen question-image caption'
        protocol['caption_policy'] = 'Cached Qwen captions for BOTH history and question images; original fields preserved'
    dump(out / 'protocol.json', protocol)
    for name, rows in [('memories', memories), ('queries', queries), ('gold', golds)]:
        jsonl(out / (name + '.jsonl'), rows)
    csvout(out / 'question_coverage.csv', coverage)
    print(json.dumps(counts), flush=True)

    import numpy as np
    import torch
    import sentence_transformers
    torch.set_num_threads(8)
    torch.manual_seed(nat['seed'])
    np.random.seed(nat['seed'])
    model = sentence_transformers.SentenceTransformer(nat['retriever_model_path'], device=args.device, local_files_only=True)
    texts = [m['caption'] for m in memories] + [nat['query_prefix'] + q['retrieval_text'] for q in queries]
    lengths = [len(model.tokenizer.encode(t, truncation=False)) for t in texts]
    over = [i for i, n in enumerate(lengths) if n > model.max_seq_length]
    if over:
        dump(out / 'overlength_inputs.json', [{'index': i, 'tokens': lengths[i]} for i in over])
        raise ValueError('Refusing silent embedding truncation')
    print('Encoding', len(texts), 'texts; maximum tokens', max(lengths), flush=True)
    vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    assert vectors.shape[0] == len(texts) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    emb, qemb = vectors[:len(memories)], vectors[len(memories):]
    np.savez_compressed(out / 'embeddings.npz', memory=emb, query=qemb,
                        memory_ids=[m['id'] for m in memories], query_ids=[q['id'] for q in queries])
    mi = {m['id']: i for i, m in enumerate(memories)}
    task_indices = {t: [i for i, m in enumerate(memories) if m['task'] == t] for t in sorted({m['task'] for m in memories})}
    rows, question_rows, details = [], [], []
    for qi, (query, gold) in enumerate(zip(queries, golds)):
        inds = np.array(task_indices[query['task']])
        scores = emb[inds] @ qemb[qi]
        rounded = np.round(scores.astype(float), 6)
        order = np.argsort(-rounded, kind='stable')
        ranked = inds[order].tolist()
        rank = {memories[j]['id']: r + 1 for r, j in enumerate(ranked)}
        allgold = set(gold['all_clue_ids'])
        nonkeys = [j for j in ranked if memories[j]['id'] not in allgold]
        competitors = nonkeys[:5]
        eligible = len(competitors) == 5
        cutoff = rounded[order[min(4, len(order)-1)]]
        tied = {memories[int(inds[j])]['id'] for j in range(len(inds)) if rounded[j] == cutoff}
        straddles = int(sum(rounded > cutoff)) < 5 < int(sum(rounded >= cutoff))
        nonkey_tie = len(nonkeys) > 5 and round(float(emb[nonkeys[4]] @ qemb[qi]), 6) == round(float(emb[nonkeys[5]] @ qemb[qi]), 6)
        for mid in gold['visual_clue_ids']:
            sims = [float(emb[mi[mid]] @ emb[j]) for j in competitors] if eligible else []
            rows.append({'sample_id': query['id'] + '|' + mid, 'question_id': query['id'],
                         'task': query['task'], 'point': query['point'],
                         'has_query_image': bool(query['question_image']), 'primary_cohort': gold['primary_cohort'],
                         'target_id': mid, 'target_query_rank': rank[mid], 'retrieved_at_5': rank[mid] <= 5,
                         'target_top5_boundary_ambiguous': straddles and mid in tied,
                         'included_in_similarity_analysis': eligible,
                         'competitor_similarity': float(np.mean(sims)) if eligible else None,
                         'target_to_competitor_cosines': sims,
                         'competitor_ids': [memories[j]['id'] for j in competitors],
                         'competitor_original_ranks': [rank[memories[j]['id']] for j in competitors],
                         'nonkey_boundary_tie': nonkey_tie, 'memory_count': len(inds),
                         'all_clue_count': len(allgold), 'visual_clue_count': len(gold['visual_clue_ids'])})
        question_rows.append({**coverage[qi], 'memory_count': len(inds),
                              'visual_targets_retrieved_at_5': sum(rank[mid] <= 5 for mid in gold['visual_clue_ids']),
                              'enough_competitors': eligible, 'nonkey_boundary_tie': nonkey_tie})
        details.append({'question_id': query['id'], 'all_clue_ids': gold['all_clue_ids'],
                        'ranking': [{'id': memories[j]['id'], 'rank': r+1, 'score': float(emb[j] @ qemb[qi]),
                                     'is_target': memories[j]['id'] in allgold} for r, j in enumerate(ranked)],
                        'competitor_ids': [memories[j]['id'] for j in competitors]})
    primary = [r for r in rows if r['primary_cohort']]
    overall = summarize('primary_image_questions_all_clues_visual', primary)
    groups = [summarize(t, [r for r in primary if r['task'] == t]) for t in task_indices]
    by_point = [summarize(p, [r for r in primary if r['point'] == p]) for p in sorted({q['point'] for q in queries})]
    supplemental = summarize('all_available_visual_clue_targets', rows)
    audit = {'unique_sample_ids': len({r['sample_id'] for r in rows}) == len(rows),
             'primary_questions': sum(c['primary_cohort'] for c in coverage),
             'image_questions': sum(c['has_query_image'] for c in coverage),
             'image_questions_with_caption': sum(c['has_query_image'] and c['provided_query_caption'] for c in coverage),
             'nonvisual_clues_excluded': sum(c['nonvisual_clue_count'] for c in coverage),
             'no_clue_questions': sum(c['all_clue_count'] == 0 for c in coverage),
             'primary_boundary_ambiguous_targets': sum(r['target_top5_boundary_ambiguous'] for r in primary),
             'primary_nonkey_boundary_tie_questions': len({r['question_id'] for r in primary if r['nonkey_boundary_tie']}),
             'truncated_inputs': 0, 'max_memory_tokens': max(lengths[:len(memories)]),
             'max_query_tokens': max(lengths[len(memories):]), 'model_max_seq_length': model.max_seq_length,
             'qwen_generation_calls': 0, 'answer_generation_calls': 0,
             'per_scenario_positive_gap': sum(g['missed_minus_retrieved'] is not None and g['missed_minus_retrieved'] > 0 for g in groups),
             'per_scenario_negative_gap': sum(g['missed_minus_retrieved'] is not None and g['missed_minus_retrieved'] < 0 for g in groups),
             'per_scenario_comparable': sum(g['missed_minus_retrieved'] is not None for g in groups)}
    valid_gaps = [g['missed_minus_retrieved'] for g in groups if g['missed_minus_retrieved'] is not None]
    summary = {'primary': overall, 'by_scenario': groups, 'by_question_type': by_point,
               'supplemental': supplemental, 'scenario_macro_gap': float(np.mean(valid_gaps)) if valid_gaps else None,
               'primary_excluding_boundary_ties': summarize('exclude_top5_and_competitor_boundary_ties',
                    [r for r in primary if not r['target_top5_boundary_ambiguous'] and not r['nonkey_boundary_tie']]), 'audit': audit}
    csvout(out / 'per_target.csv', rows)
    csvout(out / 'per_question.csv', question_rows)
    csvout(out / 'group_summary.csv', [overall, *groups])
    csvout(out / 'type_summary.csv', by_point)
    jsonl(out / 'retrieval_details.jsonl', details)
    dump(out / 'summary.json', summary)
    dump(out / 'audit.json', audit)
    protocol.update(status='complete', environment={'torch': torch.__version__, 'numpy': np.__version__,
                    'sentence_transformers': sentence_transformers.__version__}, audit=audit)
    model_files = [p for p in Path(nat['retriever_model_path']).rglob('*') if p.is_file()]
    protocol['model_file_hashes'] = {str(p.relative_to(nat['retriever_model_path'])): sha(p) for p in model_files}
    assert all(sha(ROOT / s['path']) == s['sha256'] for s in manifest['files'])
    for p, digest in caption_inputs.items():
        assert sha(ROOT/p) == digest, p
    summary['caption_label'] = caption_label
    dump(out / 'summary.json', summary)
    make_report(out, summary)
    dump(out / 'protocol.json', protocol)
    dump(out / 'artifact_manifest.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def make_report(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    main = summary['primary']
    groups = [g for g in summary['by_scenario'] if g['missed_minus_retrieved'] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1, 2.3]})
    ax = axes[0]
    for i, (label, field, color) in enumerate([('Retrieved', 'retrieved', '#287d9b'), ('Missed', 'missed', '#b96537')]):
        value = main[field + '_mean']
        if value is not None:
            ax.bar(i, value, color=color, width=.6)
            ax.text(i, value+.015, f"{value:.4f}\nn={main[field+'_n']}", ha='center')
    ax.set_xticks([0, 1], ['Retrieved', 'Missed'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean target-to-competitor caption cosine')
    ax.set_title(f"Pooled targets: {main['questions']} questions\nTop-5 target recall: {main['target_recall_at_5']:.1%}")
    ax = axes[1]
    short = {'AI_Robotics_Automation_Future_Tech':'AI / robotics', 'Academic_Animal_Pet_Research_Life':'Animal research',
             'Architecture_Art_Culture_Exhibition_Technology':'Architecture / art', 'Astronomy_Physics_Scientific_Experiments_Cosmology':'Astronomy',
             'Baking_Dessert_Daily_Life_Skill':'Baking', 'Dog_Behavior_Research_Academic_Life':'Dog behavior',
             'Education_Career_Research_Lifestyle':'Education', 'Entrepreneurship_Blockchain_Economics_Logistics_Nature':'Entrepreneurship',
             'Fashion_Personal_Care_Lifestyle_Shopping':'Fashion / shopping', 'Global_Travel_Cultural_Sightseeing':'Travel / sightseeing',
             'Global_Travel_Sustainable_Fashion_Design':'Travel / fashion', 'Home_Health_Lifestyle_Product':'Home / health',
             'Home_Repair_Maintenance_Cleaning':'Home repair', 'Landscape_Travel_Architecture_Nature':'Landscape / travel',
             'Music_Dance_Theater_Performance_Learning':'Music / performance', 'Nature_Economics_Programming_Student_Life':'Nature / economics',
             'Parenting_Commuting_Hobbies_Travel_Gear':'Parenting', 'Python_Botany_AI_Student_Life':'Python / botany',
             'Real_Estate_Home_Decor_DIY_Lifestyle':'Home decor', 'Technology_Ethics_Future_Society':'Technology / ethics'}
    ys = list(range(len(groups)))
    gaps = [g['missed_minus_retrieved'] for g in groups]
    ax.barh(ys, gaps, color=['#b96537' if v > 0 else '#287d9b' for v in gaps])
    ax.set_yticks(ys, [f"{short[g['group']]} ({g['retrieved_n']}/{g['missed_n']})" for g in groups])
    ax.invert_yaxis()
    ax.axvline(0, color='#777777', linewidth=.8)
    ax.set_xlabel('Missed minus retrieved: mean caption cosine')
    ax.set_title('Within-scenario differences (retrieved / missed targets)')
    caption_label = summary.get('caption_label', 'Official supplied captions')
    fig.suptitle('Mem-Gallery: '+caption_label+', full-history natural observation')
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / ('observation.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    def fmt(v):
        return 'NA' if v is None else f'{v:.4f}'
    table = ''.join('<tr><td>'+html.escape(g['group'])+'</td>'+''.join('<td>'+fmt(g[k])+'</td>' for k in
                    ['retrieved_n','missed_n','retrieved_mean','missed_mean','missed_minus_retrieved'])+'</tr>' for g in summary['by_scenario'])
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Mem-Gallery 自然观测</title>
<style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:30px auto;padding:20px}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccc}}table{{border-collapse:collapse}}</style>
<h1>Mem-Gallery：{html.escape(caption_label)} 的线索竞争自然观测</h1>
<p>主样本为带问题图片、全部 clue 均指向含图轮次的问题。历史库是同场景完整含图轮次；每条 clue 单独判定 Top-5 成败。
问题使用原始 question 与本轮 caption，历史索引仅使用同来源图片 caption；本轮读取缓存、不重新生成 caption。BGE-small-en-v1.5、查询前缀和 Top-5 沿用既有观察。</p>
<p>从完整查询排名排除该题所有 clue，再取前五个非目标。计算目标与这五条 caption 的平均余弦，最后在成功/失败组内按目标等权平均。
“非目标”只表示未被 clue 标注，不保证语义上完全无关。</p>
<p>主样本 {main['questions']} 题、{main['targets']} 个目标；成功 {main['retrieved_n']}，失败 {main['missed_n']}。
成功均值 {fmt(main['retrieved_mean'])}；失败均值 {fmt(main['missed_mean'])}；差值 {fmt(main['missed_minus_retrieved'])}。</p>
<img src="observation.png" alt="总体均值与各场景失败减成功差值">
<p>场景等权差值 {fmt(summary['scenario_macro_gap'])}。各场景内部结果需要与总体同时看；不能将总体相关性解释为过载因果效应。
此处只评价证据检索，不评价回答正确率；不等同于官方 NaiveRAG（官方索引还含对话文本）。不同数据集 caption 来源不同，也不能直接归因于数据集难度。</p>
<table><tr><th>场景</th><th>成功数</th><th>失败数</th><th>成功均值</th><th>失败均值</th><th>失败－成功</th></tr>{table}</table>
<p>完整审计：<a href="protocol.json">协议与哈希</a> · <a href="summary.json">统计结果</a> · <a href="question_coverage.csv">所有问题与排除原因</a> ·
<a href="per_target.csv">逐目标结果</a> · <a href="retrieval_details.jsonl">完整检索排名</a> · <a href="memories.jsonl">原始 caption 索引</a> · <a href="queries.jsonl">实际检索输入</a>。</p></html>'''
    (out / 'report.html').write_text(report)


if __name__ == '__main__':
    main()
