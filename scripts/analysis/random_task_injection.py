"""Controlled random same-task versus other-task image-caption injection."""

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
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import numpy as np

NAT=ROOT/'results/natural_observation'
MCQ=ROOT/'results/mcq_fifth_non_target_rank'
OUT=ROOT/'results/random_task_injection'
NAMES={'Brand_Memory_Test':'Brand','Card_Playlog_Test':'Card','Cartoon_Entertainment_Companion':'Cartoon',
       'Home_Renovation_Interior_Design':'Home','Multi-Scene_Visual_Case_Archive_Assistant':'Cross-scene',
       'Outdoor_Navigation_Route_Memory_Assistant':'Outdoor','Personal_Health_Dashboard_Assistant':'Health',
       'Social_Chat_Memory_Test':'Social'}


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(name,rows):(OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def single_target_metric(target_score,distractor_scores):
    target=float(np.round(float(target_score),6));other=np.round(np.asarray(distractor_scores,dtype=float),6)
    better=int(np.sum(other>target));ties=int(np.sum(other==target))+1
    return {'recall_at_5':float(np.clip((5-better)/ties,0,1)),
            'target_rank_min':better+1,'target_rank_max':better+ties,
            'target_midrank':better+(ties+1)/2,'higher_scoring_distractors':better,'target_tie_group_size':ties}


def grouped_means(sample_rows):
    result=[]
    for task in ['all']+list(NAMES):
        subset=[r for r in sample_rows if (task=='all' and r['N']<=20) or r['task']==task]
        for n in sorted({r['N'] for r in subset}):
            for condition in ['same_task','other_tasks']:
                selected=[r for r in subset if r['N']==n and r['condition']==condition]
                sims=[r['mean_target_distractor_cosine'] for r in selected if r['mean_target_distractor_cosine'] is not None]
                result.append({'task':task,'N':n,'condition':condition,'target_samples':len(selected),
                    'questions':len({r['question_id'] for r in selected}),
                    'mean_recall_at_5':float(np.mean([r['mean_recall_at_5'] for r in selected])),
                    'mean_target_distractor_cosine':float(np.mean(sims)) if sims else None})
    return result


def figures(group_rows,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    styles=[('same_task','#d17b45','o','-','Same-task distractors'),
            ('other_tasks','#2a7e9b','D','--','Other-task distractors')]
    def lines(ax,task):
        for condition,color,marker,ls,label in styles:
            selected=sorted([r for r in group_rows if r['task']==task and r['condition']==condition],key=lambda r:r['N'])
            ax.plot([r['N'] for r in selected],[r['mean_recall_at_5'] for r in selected],color=color,
                    marker=marker,ls=ls,lw=2.2,ms=7,label=label)
        ax.set_ylim(0,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
        ax.set_xlabel('Number of injected distractors N');ax.set_ylabel('Target Recall@5')
    fig,ax=plt.subplots(figsize=(7.4,5.4));lines(ax,'all');ax.set_xticks([0,10,20])
    ax.set_title(f'Controlled distractor injection: {cohort["target_samples"]} target samples\n{cohort["questions"]} MCQs; mean of 5 random draws per sample')
    ax.legend();fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('overall_injection.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8.2))
    for ax,task in zip(axes.flat,NAMES):
        lines(ax,task);n=next(r['target_samples'] for r in group_rows if r['task']==task)
        qs=next(r['questions'] for r in group_rows if r['task']==task)
        levels=sorted({r['N'] for r in group_rows if r['task']==task});ax.set_xticks(levels)
        ax.set_title(f'{NAMES[task]}\n{n} targets / {qs} questions');ax.set_xlabel('Distractors N')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Same-task versus other-task distractor injection by task',fontsize=15)
    fig.text(.5,.052,'Each curve uses a fixed target cohort across N. N=40 is added only where all targets have enough same-task distinct images.\nOne sample = one original annotated target round; each small memory contains that target plus N distractors.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95))
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('by_task_injection.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    natural=json.loads((NAT/'protocol.json').read_text());mcq_protocol=json.loads((MCQ/'protocol.json').read_text())
    inputs=[NAT/name for name in ['protocol.json','memories.jsonl','captions.jsonl','images.jsonl','embeddings.npz']]
    inputs += [MCQ/name for name in ['protocol.json','query_embeddings.npz','mcq_queries.jsonl','retrieval_details.jsonl']]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    for source in mcq_protocol['source_files']:assert sha(ROOT/source['path'])==source['sha256']
    memories=read(NAT/'memories.jsonl');mpos={r['id']:i for i,r in enumerate(memories)}
    captions={r['id']:r['caption'] for r in read(NAT/'captions.jsonl')}
    image_data={r['id']:r for r in read(NAT/'images.jsonl')}
    queries=read(MCQ/'mcq_queries.jsonl');gold={r['id']:r['gold_ids'] for r in read(MCQ/'retrieval_details.jsonl')}
    memory_cache=np.load(NAT/'embeddings.npz');query_cache=np.load(MCQ/'query_embeddings.npz')
    assert list(memory_cache['memory_ids'])==[r['id'] for r in memories]
    assert list(query_cache['query_ids'])==[r['id'] for r in queries]
    emb=memory_cache['memory'];qemb=query_cache['query']
    assert np.allclose(np.linalg.norm(emb,axis=1),1,atol=1e-6)
    assert np.allclose(np.linalg.norm(qemb,axis=1),1,atol=1e-6)
    assert all(len(m['images'])==1 for m in memories)
    image_key=[m['images'][0] for m in memories]
    representatives={}
    for i,key in enumerate(image_key):representatives.setdefault(key,i)
    for key in representatives:assert sha(Path(image_data[key]['path']))==key
    unique_indices=list(representatives.values())
    pools=[];samples=[];excluded=[]
    for qi,q in enumerate(queries):
        gids=gold[q['id']];gimage={image_key[mpos[mid]] for mid in gids}
        same=[i for i in unique_indices if memories[i]['task']==q['task'] and image_key[i] not in gimage]
        other=[i for i in unique_indices if memories[i]['task']!=q['task'] and image_key[i] not in gimage]
        pools.append({'question_id':q['id'],'task':q['task'],'target_ids':gids,
                      'same_task_ids':[memories[i]['id'] for i in same],
                      'other_task_ids':[memories[i]['id'] for i in other]})
        if len(same)<20 or len(other)<20:
            excluded.append({'question_id':q['id'],'task':q['task'],'target_samples':len(gids),
                             'same_task_pool_size':len(same),'other_task_pool_size':len(other),
                             'reason':'not enough distinct non-target images for both conditions at N=20'})
            continue
        for mid in gids:
            samples.append({'sample_id':q['id']+'|'+mid,'question_id':q['id'],'task':q['task'],
                            'target_id':mid,'query_index':qi,'target_index':mpos[mid],
                            'same_task_pool':same,'other_task_pool':other,'all_target_ids':gids})
    assert len(samples)==878 and len({s['question_id'] for s in samples})==366
    levels={task:[0,10,20]+([40] if all(len(s['same_task_pool'])>=40 and len(s['other_task_pool'])>=40 for s in samples if s['task']==task) else []) for task in NAMES}
    rows=[];plans=[];sample_rows=[]
    for s in samples:
        q=queries[s['query_index']];qi=s['query_index'];target=s['target_index']
        # Compute all scores with the same vector arithmetic for both conditions.
        query_scores=emb@qemb[qi];target_score=float(query_scores[target])
        for condition,pool in [('same_task',s['same_task_pool']),('other_tasks',s['other_task_pool'])]:
            maximum=max(levels[s['task']]);current=[]
            for repetition in range(5):
                seed_hex=hashlib.sha256(('42|'+s['sample_id']+'|'+condition+'|'+str(repetition)).encode()).hexdigest()[:16]
                rng=np.random.default_rng(int(seed_hex,16));ordered=rng.permutation(pool)[:maximum].tolist()
                assert len(set(ordered))==maximum
                assert len({image_key[i] for i in ordered})==maximum
                assert not set(memories[i]['id'] for i in ordered)&set(s['all_target_ids'])
                assert not {image_key[i] for i in ordered}&{image_key[mpos[mid]] for mid in s['all_target_ids']}
                assert all((memories[i]['task']==s['task'])==(condition=='same_task') for i in ordered)
                plans.append({'sample_id':s['sample_id'],'question_id':s['question_id'],'target_id':s['target_id'],
                              'condition':condition,'repetition':repetition,'seed_hex':seed_hex,
                              'ordered_distractor_ids':[memories[i]['id'] for i in ordered]})
                last=1.
                for n in levels[s['task']]:
                    selected=ordered[:n];scores=query_scores[selected]
                    metric=single_target_metric(target_score,scores)
                    assert metric['recall_at_5']<=last+1e-12
                    last=metric['recall_at_5']
                    if n==0:assert metric['recall_at_5']==1.
                    sim=float(np.mean(emb[selected]@emb[target])) if n else None
                    row={'sample_id':s['sample_id'],'question_id':s['question_id'],'task':s['task'],'target_id':s['target_id'],
                         'condition':condition,'N':n,'repetition':repetition,'memory_count':n+1,
                         'target_query_score':target_score,'mean_target_distractor_cosine':sim,
                         'mean_distractor_query_score':float(scores.mean()) if n else None,**metric}
                    rows.append(row);current.append(row)
            for n in levels[s['task']]:
                chosen=[r for r in current if r['N']==n];assert len(chosen)==5
                sr={k:s[k] for k in ['sample_id','question_id','task','target_id']}
                sr.update({'condition':condition,'N':n,'repetitions':5,
                           'mean_recall_at_5':float(np.mean([r['recall_at_5'] for r in chosen])),
                           'min_recall_at_5':min(r['recall_at_5'] for r in chosen),
                           'max_recall_at_5':max(r['recall_at_5'] for r in chosen),
                           'mean_target_distractor_cosine':float(np.mean([r['mean_target_distractor_cosine'] for r in chosen])) if n else None})
                sample_rows.append(sr)
    groups=grouped_means(sample_rows)
    repetition_means=[]
    for task in ['all']+list(NAMES):
        rr=[r for r in rows if (task=='all' and r['N']<=20) or r['task']==task]
        for n in sorted({r['N'] for r in rr}):
            for condition in ['same_task','other_tasks']:
                for rep in range(5):
                    sub=[r for r in rr if r['N']==n and r['condition']==condition and r['repetition']==rep]
                    repetition_means.append({'task':task,'N':n,'condition':condition,'repetition':rep,
                                              'target_samples':len(sub),'mean_recall_at_5':float(np.mean([r['recall_at_5'] for r in sub]))})
    cohort={'questions':len({s['question_id'] for s in samples}),'target_samples':len(samples)}
    audit={'all_questions':len(queries),'all_annotated_target_samples':sum(len(g) for g in gold.values()),
           'main_questions':cohort['questions'],'main_target_samples':len(samples),
           'excluded_questions':len(excluded),'excluded_target_samples':sum(r['target_samples'] for r in excluded),
           'memory_rounds':len(memories),'unique_memory_images':len(unique_indices),'duplicate_round_entries_removed_from_sampling':len(memories)-len(unique_indices),
           'raw_runs':len(rows),'distinct_samples':len(samples),'repetitions':5,'sampling_plans':len(plans),
           'source_model_calls':0,'all_N0_recall_is_one':True,'all_non_target_images_exclude_all_question_gold_images':True,
           'within_draw_increasing_N_never_improves_recall':True}
    task_counts=[]
    for task in NAMES:
        sub=[s for s in samples if s['task']==task]
        task_counts.append({'task':task,'questions':len({s['question_id'] for s in sub}),'target_samples':len(sub),
                            'memory_rounds':sum(m['task']==task for m in memories),
                            'unique_images':sum(memories[i]['task']==task for i in unique_indices),
                            'min_same_task_pool':min(len(s['same_task_pool']) for s in sub),'levels':levels[task]})
    summary={'overall':[r for r in groups if r['task']=='all'],'by_task':[r for r in groups if r['task']!='all'],
             'cohort':cohort,'task_counts':task_counts,'excluded':excluded,'audit':audit}
    protocol={'study':'Experiment 2: controlled random same-task / other-task caption distractor injection',
              'dataset':mcq_protocol['dataset'],'source_files':mcq_protocol['source_files'],
              'retriever':mcq_protocol['retriever'],'retriever_revision':mcq_protocol['retriever_revision'],
              'query':'same original MCQ question text and cached BGE query vectors; no options, answers or query-image captions',
              'memory_unit':'one ordinary image caption; original annotated round retained as target sample ID',
              'analysis_unit':'one original annotated target round for one question; every target sample has equal weight',
              'small_memory':'one fixed target plus exactly N distractors; no background memories or other gold rounds',
              'same_task_pool':'distinct image contents from original target task, excluding all question clue images and their duplicate copies',
              'other_task_pool':'distinct images pooled uniformly from the other seven tasks; exclude all question clue images',
              'deduplication':'sample distinct image SHA256 contents, not repeated copies; cached caption vectors unchanged',
              'cohort':'fixed 878 target samples / 366 questions at all main N=0,10,20 and both conditions',
              'repeats':5,'seed':42,'seed_rule':'per target sample, condition and repetition SHA256 seed, first 16 hex digits',
              'sampling':'without replacement within each draw; N levels use prefixes of one random permutation per repetition, so smaller sets are nested uniform random samples',
              'task_levels':levels,'N40_policy':'only add N40 to a task when every primary target sample has enough candidates in both conditions; preserve cohort across all levels',
              'metric':'single-target Recall@5 in target + N distractors; average over five draws per target, then equally over target samples',
              'ties':'cosine rounded to six decimals; probability of target inclusion under random ordering inside a tie group',
              'similarity_check':'record mean target-to-distractor cosine using the same normalized caption embeddings',
              'interpretation':'manipulates distractor count and task source; same-task is a source condition, not a guaranteed high-overlap condition; no QA inference or claim that similarity alone was manipulated',
              'execution':'offline cached-vector analysis; no caption generation, image synthesis, history augmentation or human relabeling',
              'input_hashes':source_hashes}
    for p in inputs:assert sha(p)==source_hashes[str(p.relative_to(ROOT))]
    for source in mcq_protocol['source_files']:assert sha(ROOT/source['path'])==source['sha256']
    dump('protocol.json',protocol);dump('summary.json',summary);jsonl('sampling_pools.jsonl',pools);jsonl('sampling_plans.jsonl',plans)
    # Include every annotated target ID, including repeated-image rounds.
    jsonl('memory_catalog.jsonl',[{'id':memories[i]['id'],'task':memories[i]['task'],'image_sha256':image_key[i],
        'caption':captions[image_key[i]],'image_path':image_data[image_key[i]]['path'],
        'is_sampling_representative':representatives[image_key[i]]==i,
        'sampling_representative_id':memories[representatives[image_key[i]]]['id']} for i in range(len(memories))])
    csvout('runs.csv',rows);csvout('sample_means.csv',sample_rows);csvout('group_means.csv',groups)
    csvout('repetition_means.csv',repetition_means);csvout('excluded_questions.csv',excluded)
    figures(groups,cohort)
    def table(values):
        t='<table><tr><th>任务</th><th>N</th><th>来源</th><th>目标样本数</th><th>Recall@5</th><th>目标－干扰 caption 相似度</th></tr>'
        for r in values:
            sim='—' if r['mean_target_distractor_cosine'] is None else f'{r["mean_target_distractor_cosine"]:.4f}'
            t+=f'<tr><td>{html.escape(r["task"])}</td><td>{r["N"]}</td><td>{r["condition"]}</td><td>{r["target_samples"]}</td><td>{r["mean_recall_at_5"]:.1%}</td><td>{sim}</td></tr>'
        return t+'</table>'
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验二：随机任务来源干扰注入</title>
<style>body{font:16px/1.7 system-ui;max-width:1300px;margin:24px auto;padding:20px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:7px}</style>
<h1>实验二：同任务 / 其他任务的受控干扰注入</h1>
<p>沿用原始选择题题干、Qwen3-VL 普通图片 caption 和 BGE-small-en-v1.5 缓存向量。每个原始 clue 目标 round 为一个样本，构造目标＋N 个干扰的小记忆库。两种来源都排除该题所有目标图片及其重复副本，干扰按不同图片内容不放回抽样。每个样本、每种来源做五次随机抽样，同一次抽样的 N=10/20/40 使用同一随机排列的前缀，边际上仍是均匀随机样本。</p>
<p>371 题共有 1041 个目标样本。5 题同任务候选全被标为目标，没有可用同任务干扰；剔除后固定 366 题、878 个样本用于主图全部 N=0、10、20。Card 有 30 个记忆 round，但图片内容去重后为 27 张，仍足够全部主样本做 N20。N40 仅用于所有主样本都足量的 Cartoon、Home、Cross-scene、Health，Social 不切换样本范围来添加 N40。</p>
<p>指标为小库中单目标进入 Top-5 的概率。分数保留六位小数，并列组使用随机排序期望。先对五次抽样平均，再对目标样本平均，不按题目或任务重新加权。N0 必然为 100%。此次是检索实验，没有选择题作答阶段。</p>
<h2>总图</h2><img src="overall_injection.png">'''+table(summary['overall'])+'''
<h2>八任务分图</h2><img src="by_task_injection.png">'''+table(summary['by_task'])+'''
<p>这里操纵的是数量与任务来源。同任务 caption 是否更像目标，可看同时记录的目标－干扰相似度；同任务还可能更符合查询主题。因此曲线差异直接支持来源条件的效果，不足以单独归因为相似度本身的因果作用。</p>
<h2>审计</h2><pre>'''+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'''</pre>
<p><a href="sample_means.csv">逐目标五次平均</a> · <a href="runs.csv">全部抽样结果</a> · <a href="sampling_plans.jsonl">实际抽样 ID</a> · <a href="memory_catalog.jsonl">caption 与原图路径</a> · <a href="excluded_questions.csv">未纳入问题</a> · <a href="protocol.json">实验设置</a></p></html>'''
    (OUT/'report.html').write_text(report)
    dump('artifact_manifest.json',{'source_files_unchanged':True,
        'outputs':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
