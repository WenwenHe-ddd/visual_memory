"""Length-matched, nested, paired controlled injection using real-image memories."""

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
import random
from pathlib import Path

import numpy as np
from scripts.analysis.analyze import rank_metrics
from scripts.analysis.injection_design import ROOT,NAT,OUT,read,write,anchor_signature
from scripts.analysis.injection_candidates import seed,load_data,cue_gate,question_reviews,valid_pair_ids

ARMS=['high','low','unrelated']
LEVELS=[0,5,10,20,50]

def triples(pools,random_seed):
    """Find disjoint <=5-token-range triples, without seeing retrieval scores."""
    rng=random.Random(random_seed)
    highs=list(pools['high']);rng.shuffle(highs)
    def options(h):
        return sum(abs(l['tokens']-h['tokens'])<=5 for l in pools['low'])*sum(abs(u['tokens']-h['tokens'])<=5 for u in pools['unrelated'])
    highs.sort(key=options)
    low={r['id']:r for r in pools['low']};unrel={r['id']:r for r in pools['unrelated']};result=[]
    for h in highs:
        choices=[]
        for l in low.values():
            if abs(l['tokens']-h['tokens'])>5:continue
            for u in unrel.values():
                ts=[h['tokens'],l['tokens'],u['tokens']]
                if max(ts)-min(ts)<=5:choices.append((max(ts)-min(ts),rng.random(),l,u))
        if not choices:continue
        _,_,l,u=min(choices,key=lambda x:(x[0],x[1]))
        result.append({'high':h['id'],'low':l['id'],'unrelated':u['id'],
                       'token_lengths':{'high':h['tokens'],'low':l['tokens'],'unrelated':u['tokens']}})
        del low[l['id']];del unrel[u['id']]
    return result

def bootstrap(values,field):
    if not values:return None
    groups=collections.defaultdict(list)
    for r in values:groups[r['task']].append(r[field])
    tasks=sorted(groups)
    point=float(np.mean([r[field] for r in values]))
    result={'estimate':point,'questions':len(values),'task_clusters':len(tasks),'low':None,'high':None}
    if len(tasks)<2:return result
    rng=np.random.default_rng(42);means=[]
    for _ in range(2000):
        vals=[v for t in rng.choice(tasks,len(tasks),replace=True) for v in groups[t]]
        means.append(np.mean(vals))
    result.update(low=float(np.percentile(means,2.5)),high=float(np.percentile(means,97.5)))
    return result

def csvout(name,rows):
    if not rows:
        path=OUT/name
        header=path.read_text().splitlines()[0] if path.exists() and path.stat().st_size else ''
        path.write_text(header+'\n' if header else '')
        return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for f,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((NAT/f).read_bytes()).hexdigest()==digest,('Changed source',f)
    pools=read(OUT/'pools.jsonl');targets={t['id']:t for t in read(OUT/'targets.jsonl')}
    visual=read(OUT/'high_visual_judgments.jsonl');text=read(OUT/'high_text_judgments.jsonl')
    human=read(OUT/'human_review_decisions.jsonl')
    valid=valid_pair_ids(visual,text,human)
    mems,_=load_data();anchors={r['id']:r for r in read(OUT/'anchors.jsonl')};reviews=question_reviews()
    plans=[];availability=[]
    for p in pools:
        gate=cue_gate(targets[p['id']],anchors.get(p['id']),mems,reviews.get(p['id']))
        if gate['passed'] and p.get('anchor_signature')!=anchor_signature():
            gate={**gate,'passed':False,'reason':'Candidate pools predate complete locator-cue extraction; regenerate pools.'}
        filtered={**p['pools'],'high':[m for m in p['pools']['high'] if p['id']+'||'+m['id'] in valid]}
        alternatives=[triples(filtered,seed(p['id'])+i) for i in range(10)] if p['ready_for_real_min5'] and gate['passed'] else [[]]
        matched=max(alternatives,key=lambda ts:(len(ts),-sum(max(t['token_lengths'].values())-min(t['token_lengths'].values()) for t in ts)))
        availability.append({'id':p['id'],'task':p['task'],'x':p['x'],'y':p['y'],'high_lexical':len(p['pools']['high']),
                 'high_verified':len(filtered['high']),'low_available':len(filtered['low']),'unrelated_available':len(filtered['unrelated']),
                 'matched_triplets':len(matched),'reference_tokens':p['reference_tokens'],
                 'cue_gate_passed':gate['passed'],'cue_gate_reason':gate['reason'],
                 **{f'complete_at_{n}':len(matched)>=n for n in LEVELS[1:]}})
        if len(matched)>=5:plans.append({'id':p['id'],'task':p['task'],'x':p['x'],'y':p['y'],'base_ids':p['base_ids'],'triplets':matched})
    # Freeze matched candidate IDs before loading any retrieval scores.
    (OUT/'matched_plans.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in plans))
    csvout('availability.csv',availability)
    embs=np.load(NAT/'embeddings.npz');mmap={s:i for i,s in enumerate(embs['memory_ids'])};qmap={s:i for i,s in enumerate(embs['query_ids'])}
    outcomes=[];detail=[]
    for p in plans:
        qid=p['id'];t=targets[qid];scorevec=embs['memory']@embs['query'][qmap[qid]]
        base=p['base_ids'];gold_positions=[base.index(g) for g in t['clue_ids']]
        baseline,_=rank_metrics(scorevec[[mmap[i] for i in base]],gold_positions)
        for random_seed in protocol['seeds']:
            order=list(p['triplets']);random.Random(seed(qid)+random_seed).shuffle(order)
            for arm in ARMS:
                for n in LEVELS:
                    if n>len(order):continue
                    injected=[tr[arm] for tr in order[:n]];ids=base+injected
                    assert len(ids)==len(set(ids))
                    metrics,_=rank_metrics(scorevec[[mmap[i] for i in ids]],gold_positions)
                    row={'id':qid,'task':p['task'],'x':p['x'],'y':p['y'],'seed':random_seed,'condition':arm,'n':n,
                         'mean_gold_rank':metrics['mean_gold_rank'],'rank_degradation':metrics['mean_gold_rank']-baseline['mean_gold_rank'],
                         'recall_at_5':metrics['recall_at_5'],'recall5_change':metrics['recall_at_5']-baseline['recall_at_5'],
                         'recall_at_10':metrics['recall_at_10'],'expected_first_gold_rank':metrics['first_gold_rank']}
                    outcomes.append(row);detail.append({**row,'base_ids':base,'injection_ids':injected})
    csvout('runs.csv',outcomes)
    (OUT/'run_items.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
    grouped=collections.defaultdict(list)
    for r in outcomes:grouped[r['id'],r['condition'],r['n']].append(r)
    qmeans=[]
    for (qid,arm,n),rs in grouped.items():
        qmeans.append({'id':qid,'task':rs[0]['task'],'x':rs[0]['x'],'y':rs[0]['y'],'condition':arm,'n':n,
                      **{k:float(np.mean([r[k] for r in rs])) for k in ['rank_degradation','recall_at_5','recall5_change','recall_at_10','expected_first_gold_rank']}})
    csvout('question_means.csv',qmeans)
    qm={(r['id'],r['condition'],r['n']):r for r in qmeans};contrasts=[];summary=[]
    for n in LEVELS[1:]:
        ids=[p['id'] for p in plans if len(p['triplets'])>=n]
        for qid in ids:
            hi=qm[qid,'high',n];lo=qm[qid,'low',n];un=qm[qid,'unrelated',n]
            contrasts.append({'id':qid,'task':hi['task'],'x':hi['x'],'y':hi['y'],'n':n,
                 'high_minus_low':hi['rank_degradation']-lo['rank_degradation'],
                 'high_minus_unrelated':hi['rank_degradation']-un['rank_degradation'],
                 'high_low_interaction_per_item':(hi['rank_degradation']-lo['rank_degradation'])/n,
                 'high_unrelated_interaction_per_item':(hi['rank_degradation']-un['rank_degradation'])/n})
        cs=[r for r in contrasts if r['n']==n]
        summary.append({'n':n,'eligible_questions':len(ids),
             'rank_degradation':{arm:bootstrap([r for r in qmeans if r['n']==n and r['condition']==arm],'rank_degradation') for arm in ARMS},
             'interaction_high_low':bootstrap(cs,'high_minus_low'),
             'interaction_high_unrelated':bootstrap(cs,'high_minus_unrelated')})
    csvout('paired_interactions.csv',contrasts)
    real_summary={'status':'no_current_validated_results' if not plans else 'provisional_validated_results',
        'requested_levels':LEVELS,'plans':len(plans),'levels':summary,
        'complete_at_50':any(len(p['triplets'])>=50 for p in plans),
        'synthetic_used':0,'full_primary_experiment_complete':False,
        'cell_counts':dict(collections.Counter(p['x']+'/'+p['y'] for p in plans)),
        'task_counts':dict(collections.Counter(p['task'] for p in plans)),
        'length_match_max_triplet_spread':max([max(t['token_lengths'].values())-min(t['token_lengths'].values()) for p in plans for t in p['triplets']] or [0]),
        'text_judged':len(read(OUT/'high_text_judgments.jsonl')),
        'text_accepted':sum(r['accepted'] for r in read(OUT/'high_text_judgments.jsonl')),
        'visual_judged':len(visual),'visual_accepted':sum(r['accepted'] for r in visual),
        'current_validated_high_pairs':len(valid),'human_excluded_pairs':sum(r.get('decision')=='exclude' for r in human),
        'cue_gate_passed_questions':sum(r['cue_gate_passed'] for r in availability),
        'withdrawals':protocol.get('withdrawals',[]),
        'limitations':['Main 0/5/10/20/50 design remains incomplete wherever real pools lack matched distractors.',
                       'Q65 legacy rank and margin results are withdrawn: person/event/image-order locators were omitted from caption-only memories and candidate selection.',
                       'Legacy anchor and candidate annotations require regeneration under the full-question cue policy; no eligible result is not a measured zero effect.',
                       'VLM candidate checking is automated, not independent human annotation.',
                       'Low-overlap and unrelated are operational lexical/source-task controls, not full semantic human judgments.',
                       'Data-dependent missing conditions are not filled with duplicates, zeroes or unreviewed synthetic items.']}
    write(OUT/'real_results.json',real_summary)
    print(json.dumps(real_summary,indent=2))
    report(real_summary,qmeans,plans)

def report(s,qmeans,plans):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    counts={r['n']:r['eligible_questions'] for r in s['levels']}
    supported=[n for n,v in counts.items() if v]
    if supported:
        maxn=max(supported);ids={p['id'] for p in plans if len(p['triplets'])>=maxn}
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        colors={'high':'#c33a32','low':'#2876b9','unrelated':'#39865c'}
        for arm in ARMS:
            levels=[n for n in LEVELS if n<=maxn]
            ys=[np.mean([r['rank_degradation'] for r in qmeans if r['id'] in ids and r['condition']==arm and r['n']==n]) for n in levels]
            axes[0].plot(levels,ys,'o-',label=arm,color=colors[arm])
        axes[0].set(title=f'Same question cohort: n={len(ids)}, max injection={maxn}',xlabel='Injected distractors',ylabel='Mean gold rank increase (worse)')
        axes[0].legend();axes[0].grid(alpha=.2)
        for shift,key,label,color in [(-.12,'interaction_high_low','High - low','#8446ad'),(.12,'interaction_high_unrelated','High - unrelated','#b76e20')]:
            for i,level in enumerate(s['levels']):
                v=level[key]
                if v is None:continue
                x=i+shift;y=v['estimate']
                if v['low'] is not None:axes[1].errorbar(x,y,yerr=[[max(0,y-v['low'])],[max(0,v['high']-y)]],fmt='o',color=color,capsize=4,label=label if i==0 else None)
                else:axes[1].plot(x,y,'o',color=color,label=label if i==0 else None)
        axes[1].axhline(0,color='gray',ls='--');axes[1].set(xticks=range(4),xticklabels=[f'{n}\nq={counts[n]}' for n in [5,10,20,50]],xlabel='Injected distractors (paired cohort varies)',ylabel='Paired difference in rank increase',title='Overlap x count contrasts')
        axes[1].legend();fig.tight_layout()
        for ext in ['png','svg','pdf']:fig.savefig(OUT/f'real_injection.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    else:
        fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
        ax.text(.5,.62,'No current validated experimental results',ha='center',va='center',fontsize=18,weight='bold')
        ax.text(.5,.34,'Q65 withdrawn: event and image-order cues were omitted.\nFull-question cue validation is pending for the remaining questions.',ha='center',va='center',fontsize=12)
        for ext in ['png','svg','pdf']:fig.savefig(OUT/f'real_injection.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    def fmt(v):return '缺失' if v is None else f"{v['estimate']:.3f} (题数 {v['questions']}, 任务数 {v['task_clusters']})"
    tab=''.join('<tr><td>'+str(r['n'])+'</td><td>'+str(r['eligible_questions'])+'</td>'+''.join('<td>'+fmt(r['rank_degradation'][a])+'</td>' for a in ARMS)+'<td>'+fmt(r['interaction_high_low'])+'</td><td>'+fmt(r['interaction_high_unrelated'])+'</td></tr>' for r in s['levels'])
    body=f'''<!doctype html><meta charset="utf-8"><title>受控干扰注入：原始图像部分</title><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;padding:20px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd}}img{{max-width:100%}}</style><h1>受控干扰注入：原始图像部分</h1>
<p><b>Q65 旧结果已撤回，当前完整配对题数为 {s['plans']}。</b>旧筛选只使用 room，遗漏 Hannah、搬入当天、第一张图片；普通 caption 也没有保留这些定位信息。旧排名与分数间隔不能作为该题完整线索过载实验的结果。<a href="withdrawn_q65_20261006/withdrawal.json">撤回记录与历史输出</a>。其余题目需要按完整线索规则重新验证；当前无有效结果不表示效应为零。</p>
<p>沿用固定 Qwen3-VL 普通 caption 和 BGE-small-en-v1.5。每题固定全部标注证据与 5 条无关背景记忆，再注入独立候选。所有实验条件共享同一问题、查询线索、原始答案和基底库。此处是受控检索集合，不能直接与原始完整 MemEye 库的 Recall 数值比较。</p>
<p>长度：各候选 BGE token 长度在目标证据 caption 中位长度 ±10% 内，三条件逐槽长度差最多 5 tokens。高重叠候选先检查 caption 中不同关键值，再用原始图片进行 VLM 复核。低重叠为同任务零线索词面匹配，无关为其他任务零线索词面匹配。这些仍是自动操作定义，未做独立人工语义标注。</p>
<p>历史自动核验记录：caption {s['text_judged']} 对，通过 {s['text_accepted']} 对；原图 {s['visual_judged']} 对，通过 {s['visual_accepted']} 对。这些旧记录保留用于审计。应用完整线索版本与人工排除后，当前有效高重叠候选 {s['current_validated_high_pairs']} 对。</p>
<table><tr><th>注入条数</th><th>完整配对题数</th><th>高重叠排名退化</th><th>低重叠排名退化</th><th>无关排名退化</th><th>高−低</th><th>高−无关</th></tr>{tab}</table>
{('<img src="real_injection.png">' if supported else '')}
<p>左图在最大可用数量处固定同一题目集合，避免把样本更换画成剂量效应；右图每个数量内部作配对比较，不同数量可能有不同题目。置信区间按任务重采样；任务数不足时不提供区间。低重叠／无关“持平”不能用不显著代替；50 条条件缺失时无法验证预设的持平界限。</p>
<p><a href="protocol.json">完整协议</a> · <a href="availability.csv">371 道题的候选和配对数量</a> · <a href="question_means.csv">逐题结果</a> · <a href="paired_interactions.csv">逐题交互对比</a> · <a href="real_results.json">统计结果</a> · <a href="run_items.jsonl">每次实际注入的条目</a></p>'''
    (OUT/'real_report.html').write_text(body)

if __name__=='__main__':main()
