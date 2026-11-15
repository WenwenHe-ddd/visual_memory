"""Matched positive-variance lambda ablation on all Qwen Mem-Gallery questions.

Frozen original captions/graphs/queries; alpha=1 and five fixed lambdas. Both
unweighted controls and ten weighted conditions share fresh reader inference.
No original signed source or result is edited. Gold is opened only in analysis.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.vendor'), str(ROOT)]
for key,value in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                  'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                  'MPLCONFIGDIR':'/tmp/visualmemory-matplotlib'}.items():
    os.environ.setdefault(key,value)
import numpy as np
from scripts.memory import mem_gallery_master_answer as original
from scripts.memory.online_positive_variance import PositiveVarianceWeighting
from scripts.evaluation import mem_gallery_master_reader as reader

SOURCE = original.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT/'results/memgallery_qwen_positive_variance_lambda_ablation_20261107'
LAMBDAS = (.001,.003,.01,.03,.1)
ALPHA = 1.
METHODS = original.CONDITIONS
read, lines, dump, write_lines = original.read, original.lines, original.dump, original.write_lines
sha, verify_hashes = original.sha, original.verify_hashes


def condition(method,ridge):
    return method if ridge is None else method+'+positive_variance__l'+format(ridge,'g').replace('.','p')+'_a1'


SPECS = {m:{'method':m,'lambda':None,'alpha':0.} for m in METHODS}
SPECS.update({condition(m,ridge):{'method':m,'lambda':ridge,'alpha':ALPHA}
              for ridge in LAMBDAS for m in METHODS})


def score_group(query,vectors,ridge):
    """Use the exact MemEye production component with a covariance oracle."""
    result = PositiveVarianceWeighting(ridge,ALPHA).apply(query,vectors)
    centered = vectors-vectors.mean(0)
    expected = result['raw'].copy()
    if len(vectors)>1:
        expected += vectors@centered.T@np.linalg.solve(
            centered@centered.T+(len(vectors)-1)*ridge*np.eye(len(vectors)),centered@query)
    error=float(np.max(np.abs(expected-result['positive_variance'])))
    assert error<2e-12,error
    return result,error


def annotations():
    result={}
    for name,spec in SPECS.items():
        baseline=spec['method']==METHODS[0]
        weighted=spec['lambda'] is not None
        result[name]={
            '1. 候选记忆结构':('同场景所有对话记忆作为一个簇，含视觉和纯文本记忆。' if baseline else
                '同一全量记忆池；视觉记忆按原 caption_visual_master 分簇，纯文本记忆不属于 master。'),
            '2. 查询内容':('原始问题，加数据集提供的问题图片 caption（如有）。' if baseline else
                '各 master 使用原始问题/问题图 caption，加冻结的 CLIP 选中完整候选陈述；纯文本记忆使用原始查询。'),
            '3. 检索方式':{'weighting':'positive_variance' if weighted else 'none','lambda':spec['lambda'],
                'alpha':ALPHA if weighted else None,'top_k':5,'svd_term_sign':'+' if weighted else None,
                'intermediate':'h=raw+sum[v/(v+lambda)]*(q dot u)*(e dot u)' if weighted else None,
                'final':'raw+alpha*population_std(raw)*tanh(population_z(h))' if weighted else 'raw cosine',
                'group':'full same-scenario pool' if baseline else 'each visual master; text-only scores unchanged',
                'merge':'stable global Top-5'},
            '4. 编码方式':'原 BAAI/bge-small-en-v1.5，384维 L2；对话+完整 Qwen caption；原 query prefix；禁止截断。向量仅在本轮评测内存中共享，不写回记忆库。',
            '5. 回答方式':'原 Qwen3-VL-8B-Instruct，seed42，greedy，bf16 SDPA，256输出token，文本batch16/图像batch4；原始Top-5记忆及问题原图；无候选陈述或标准答案。官方归一化EM、F1评分。'}
    return result


def prepare(base,device):
    path=base/'retrieval_protocol.json'
    if path.exists():
        p=verify(base);assert p['condition_specs']==SPECS
        return p
    offline=read(SOURCE/'offline/protocol.json');reference=read(SOURCE/'retrieval_protocol.json')
    sources={**offline['source_sha256'],**offline['artifact_sha256'],**reference['source_sha256']}
    verify_hashes(sources)
    assert sha(SOURCE/'retrieval_results.jsonl')==reference['results_sha256']
    input_protocol=read(SOURCE/'inputs/protocol.json')
    input_hashes={str(ROOT/p):v for p,v in input_protocol['source_sha256'].items()}
    verify_hashes(input_hashes);sources.update(input_hashes)
    tracked=[Path(__file__),Path(reader.__file__),ROOT/'scripts/memory/online_positive_variance.py',
        ROOT/'scripts/memory/ridge_whitening.py',ROOT/'scripts/memory/online_zscore.py',
        SOURCE/'offline/protocol.json',SOURCE/'retrieval_protocol.json',SOURCE/'retrieval_results.jsonl',
        SOURCE/'candidates.jsonl',SOURCE/'reader/protocol.json',SOURCE/'reader/qa_manifest.jsonl',
        SOURCE/'reader/responses.jsonl',SOURCE/'reader/paired_results.json',SOURCE/'reader/summary.json']
    sources.update({str(p):sha(p) for p in tracked})
    base.mkdir(parents=True,exist_ok=True);(base/'inputs').mkdir(exist_ok=True)
    for name in ['questions.jsonl','memories.jsonl','gold.jsonl','protocol.json']:
        source,target=SOURCE/'inputs'/name,base/'inputs'/name
        if target.exists():assert sha(source)==sha(target)
        else:shutil.copyfile(source,target)
        sources.update({str(source):sha(source),str(target):sha(target)})
    questions=lines(base/'inputs/questions.jsonl');records=lines(SOURCE/'offline/memories.jsonl')
    canonical={m['id']:m for m in lines(base/'inputs/memories.jsonl')}
    assert len(questions)==1711 and len(records)==3962
    assert sum(bool(q['query_image_paths']) for q in questions)==487
    assert all(not {'answer','gold_ids','clue','gold_choice'}&set(q) for q in questions)
    assert all(m['retrieval_text']==canonical[m['id']]['retrieval_text'] and
               m['reader_text']==canonical[m['id']]['reader_text'] for m in records)
    candidates={r['id']:r['candidate_statements'] for r in lines(SOURCE/'candidates.jsonl')}
    reference_rows={r['id']:r for r in lines(SOURCE/'retrieval_results.jsonl')}
    assert set(candidates)==set(reference_rows)=={q['id'] for q in questions}
    from types import SimpleNamespace
    original.configure_device(SimpleNamespace(device=device))
    encoder=original.load_text_encoder(offline['text_encoder'],device)
    results=[];raw_checks=group_checks=full_rank_changes=0
    max_raw_error=max_oracle_error=0.
    for domain in dict.fromkeys(m['task'] for m in records):
        memories=[m for m in records if m['task']==domain]
        qs=[q for q in questions if q['task']==domain]
        graph=read(SOURCE/'offline/memory'/domain/'graph.json')
        nodes={n['id']:n for n in graph['image_nodes']};groups={m['id']:[] for m in graph['masters']}
        for i,m in enumerate(memories):
            if m['is_visual']:
                node=nodes[m['node_id']]
                assert (node['caption'],node['dialogue'])==(m['caption'],m['dialogue'])
                groups[node['master_id']].append(i)
        visual_indices={i for ix in groups.values() for i in ix}
        assert visual_indices=={i for i,m in enumerate(memories) if m['is_visual']}
        vectors=original.encode_text(encoder,[m['retrieval_text'] for m in memories])
        query_texts=[]
        for q in qs:
            c=candidates[q['id']]
            if isinstance(c,dict):c=[c[k] for k in 'ABCD']
            assert len(c)==len(set(c))==4
            query_texts.extend([q['retrieval_query'],*[q['retrieval_query']+'\nCandidate answer: '+s for s in c]])
        queries=original.encode_text(encoder,query_texts,offline['query_prefix']).reshape(len(qs),5,-1)
        domain_details=[]
        for qi,q in enumerate(qs):
            old=reference_rows[q['id']]
            choices={m['master_id']:m['candidate_index'] for m in old['master_choices']}
            assert set(choices)==set(groups)
            plans={METHODS[0]:[('all',list(range(len(memories))),0)],
                   METHODS[1]:[(mid,ix,1+choices[mid]) for mid,ix in groups.items()]}
            row={'id':q['id'],'task':domain,'candidate_count':len(memories),
                 'master_choices':old['master_choices'],'conditions':[]}
            details={'id':q['id'],'task':domain,'memory_ids':[m['id'] for m in memories],
                     'text_only_indices':[i for i,m in enumerate(memories) if not m['is_visual']],
                     'methods':{}}
            for method in METHODS:
                raw=vectors@queries[qi,0]
                arrays={str(ridge):{key:raw.copy() if key in ['weighted','enhanced'] else np.zeros(len(raw))
                    for key in ['weighted','enhanced','z','correction']} for ridge in LAMBDAS}
                group_info=[]
                for gid,indices,query_index in plans[method]:
                    ix=np.array(indices);x=vectors[ix];query=queries[qi,query_index]
                    for ridge in LAMBDAS:
                        stats,error=score_group(query,x,ridge)
                        group_checks+=1;max_oracle_error=max(max_oracle_error,error)
                        raw[ix]=stats['raw']
                        for key,field in [('weighted','weighted'),('enhanced','positive_variance'),('z','z'),('correction','correction')]:
                            arrays[str(ridge)][key][ix]=stats[field]
                        if ridge==LAMBDAS[0]:
                            group_info.append({'id':gid,'indices':indices,'query_index':query_index,
                                'query':query_texts[qi*5+query_index],'raw_std':stats['raw_std'],
                                'raw_status':stats['raw_status']})
                old_condition=next(c for c in old['conditions'] if c['name']==method)
                raw_order=np.argsort(-raw,kind='stable');raw_ids=[memories[i]['id'] for i in raw_order]
                assert raw_ids[:5]==old_condition['top5_memory_ids'],(q['id'],method)
                error=float(np.max(abs(raw[raw_order[:5]]-old_condition['top5_scores'])))
                assert error<2e-6,error
                max_raw_error=max(max_raw_error,error);raw_checks+=1
                full_rank_changes+=raw_ids!=old_condition['ranking_memory_ids']
                for ridge in [None,*LAMBDAS]:
                    values=raw if ridge is None else arrays[str(ridge)]['weighted']
                    order=np.argsort(-values,kind='stable')[:5]
                    row['conditions'].append({'name':condition(method,ridge),
                        'top5_memory_ids':[memories[i]['id'] for i in order],
                        'top5_scores':values[order].tolist()})
                details['methods'][method]={'raw':raw.tolist(),'groups':group_info,
                    'weighting':{l:{k:v.tolist() for k,v in arr.items()} for l,arr in arrays.items()}}
            results.append(row);domain_details.append(details)
        path=base/'diagnostics'/f'{domain}.jsonl';write_lines(path,domain_details)
        print(json.dumps({'stage':'positive_variance_retrieval','domain':domain,'completed':len(results),
                          'total':1711,'raw_top5_matches':raw_checks}),flush=True)
    del encoder,vectors,queries
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    assert len(results)==1711 and raw_checks==3422
    write_lines(base/'retrieval_results.jsonl',results);verify_hashes(sources)
    artifacts=[base/'retrieval_results.jsonl',*sorted((base/'diagnostics').glob('*.jsonl'))]
    p={'dataset':'Mem-Gallery','questions':1711,'scenarios':20,'memory_rounds':3962,
       'visual_rounds':1003,'text_only_rounds':2959,'masters':sum(d['masters'] for d in offline['domain_summaries']),
       'scheme':'positive_variance','lambdas':list(LAMBDAS),'alpha':ALPHA,'condition_specs':SPECS,
       'method_annotations':annotations(),'source_run':str(SOURCE),'device':device,
       'caption_graph_candidates_unchanged':True,'gold_read_for_retrieval':False,
       'vectors_persisted':False,'encoding_reuse':'Transient per-scenario vectors shared across questions and lambdas, as in original Gallery evaluation.',
       'raw_top5_matches':raw_checks,'raw_full_rank_changes':full_rank_changes,
       'max_raw_top5_score_error':max_raw_error,'independent_covariance_checks':group_checks,
       'max_independent_covariance_error':max_oracle_error,'source_sha256':sources,
       'artifact_sha256':{str(p):sha(p) for p in artifacts}}
    dump(base/'retrieval_protocol.json',p)
    return p


def verify(base):
    p=read(base/'retrieval_protocol.json')
    assert p['condition_specs']==SPECS
    verify_hashes(p['source_sha256']);verify_hashes(p['artifact_sha256'])
    return p


def run_reader(base,device):
    p=verify(base)
    previous=read(SOURCE/'reader/protocol.json')['contract']
    reader.CONDITIONS=tuple(SPECS)
    original_ensure=reader.ensure_protocol
    def ensure(path,contract):
        for key in ['model','system_prompt','task_instructions','official_prompt_revision','seed','do_sample',
                    'max_new_tokens','batch_size_text','batch_size_images','dtype','query_image_max_pixels',
                    'input_truncation','candidate_expansion_in_reader']:
            assert contract[key]==previous[key],key
        contract.update(condition_specs=SPECS,method_annotations=annotations(),
            inference_order='All twelve conditions in declared order on even questions, reversed on odd questions; grouped by image presence.',
            parameter_policy='Fixed alpha=1; lambda=.001/.003/.01/.03/.1; two fresh unweighted controls; one shared same-run payload cache.')
        contract['source_sha256'].update(p['source_sha256'])
        contract['source_sha256'][str(base/'retrieval_protocol.json')]=sha(base/'retrieval_protocol.json')
        return original_ensure(path,contract)
    reader.ensure_protocol=ensure
    try:
        result=reader.run_reader(base,device=device,batch_size=16,image_batch_size=4)
    finally:
        reader.ensure_protocol=original_ensure
    verify(base)
    return result


def analyze(base):
    protocol=verify(base);reader.CONDITIONS=tuple(SPECS)
    scored=reader.analyze(base)
    pairs=read(base/'reader/paired_results.json')
    old={r['id']:r for r in read(SOURCE/'reader/paired_results.json')}
    overall={}
    for name in SPECS:
        values=[p[name] for p in pairs]
        metrics=scored['overall'][name]
        correct=sum(int(v['em']) for v in values)
        overall[name]={**metrics,'correct':correct,'questions':len(values),
            'gold_hits':sum(v['gold_hits'] for v in values),'gold_links':sum(v['gold_clues'] for v in values)}
        assert overall[name]['gold_links']==4109 and metrics['nonempty_clue_questions']==1527
    table=[];changes={}
    for ridge in LAMBDAS:
        row={'lambda':ridge,'alpha':ALPHA}
        for prefix,method in zip(['baseline','master'],METHODS):
            name=condition(method,ridge);metrics=overall[name]
            wins=[];losses=[];changed=[]
            for p in pairs:
                a,b=p[method],p[name]
                if a['payload_sha256']==b['payload_sha256']:assert a['raw_response']==b['raw_response']
                else:changed.append(p['id'])
                if b['em']>a['em']:wins.append(p['id'])
                if b['em']<a['em']:losses.append(p['id'])
            row.update({prefix+'_correct':metrics['correct'],prefix+'_em':metrics['em'],prefix+'_f1':metrics['f1'],
                prefix+'_gain':metrics['correct']-overall[method]['correct'],
                prefix+'_recall_at_5':metrics['recall_at_5_micro'],prefix+'_gold_hits':metrics['gold_hits'],
                prefix+'_wrong_to_right':len(wins),prefix+'_right_to_wrong':len(losses),
                prefix+'_changed_payloads':len(changed)})
            changes[name]={'wrong_to_right':wins,'right_to_wrong':losses,'changed_payloads':changed}
        table.append(row)
    historical={}
    for method in METHODS:
        assert all(p[method]['payload_sha256']==old[p['id']][method]['payload_sha256'] for p in pairs)
        historical[method]={'old_correct':sum(int(p[method]['em']) for p in old.values()),
            'new_correct':overall[method]['correct'],
            'answer_changed_ids':[p['id'] for p in pairs if p[method]['raw_response']!=old[p['id']][method]['raw_response']],
            'em_gained':[p['id'] for p in pairs if p[method]['em']>old[p['id']][method]['em']],
            'em_lost':[p['id'] for p in pairs if p[method]['em']<old[p['id']][method]['em']]}
    summary={'status':'complete','dataset':'Mem-Gallery','model':'Qwen3-VL-8B-Instruct','scheme':'positive_variance',
        'seed':42,'alpha':ALPHA,'lambdas':list(LAMBDAS),'questions':1711,'conditions':12,
        'overall':overall,'ablation':table,'paired_changes':changes,'historical_controls':historical,
        'reader_validation':read(base/'reader/validation_output.json'),'method_annotations':annotations(),
        'metric_policy':scored['metric_policy'],'recall_definition':'Micro Recall@5: target round hits / 4109 links from 1527 nonempty-clue questions. All 1711 questions enter EM/F1.',
        'interpretation':'Descriptive ablation on the same benchmark; not held-out parameter validation.'}
    dump(base/'summary.json',summary)
    for name,rows in [('ablation.csv',table),('metrics.csv',[{'condition':n,**v} for n,v in overall.items()])]:
        with (base/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    wide=[]
    for p in pairs:
        r={k:p[k] for k in ['id','task','point','answer']}
        for name in SPECS:
            r.update({name+'_answer':p[name]['raw_response'],name+'_em':p[name]['em'],name+'_f1':p[name]['f1'],
                      name+'_top5':' | '.join(p[name]['top5_memory_ids'])})
        wide.append(r)
    with (base/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(wide[0]));w.writeheader();w.writerows(wide)
    plot(base,table,overall)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>MemGallery Positive Variance</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>MemGallery：Positive Variance λ 消融</h1><p>Qwen3-VL-8B-Instruct；α=1，seed=42，1711题。保留原始 caption、master 图、候选陈述、查询和开放式回答设置。两个未赋权对照与十个赋权条件在本轮重新回答，同一完整请求共享一次本轮推理，不复用历史答案。</p>',
        '<p>正向SVD项：h=raw+Σ[v/(v+λ)](q·u)(e·u)；最终 raw+α·std(raw)·tanh(z(h))。master方案的纯文本记忆保持原始分数。</p>',
        f'<p>本轮对照：baseline {overall[METHODS[0]]["correct"]}/1711；master {overall[METHODS[1]]["correct"]}/1711。历史对照778/1711和798/1711，逐题差异记录在summary.json。</p>',
        '<img src="lambda_ablation.png"><table><tr><th>λ</th><th>Baseline EM</th><th>Master EM</th><th>Baseline F1</th><th>Master F1</th><th>Baseline Recall@5</th><th>Master Recall@5</th></tr>']
    for r in table:
        body.append(f'<tr><td>{r["lambda"]}</td><td>{r["baseline_correct"]}/1711</td><td>{r["master_correct"]}/1711</td><td>{r["baseline_f1"]:.2%}</td><td>{r["master_f1"]:.2%}</td><td>{r["baseline_recall_at_5"]:.2%}</td><td>{r["master_recall_at_5"]:.2%}</td></tr>')
    body+=['</table><p>EM 是官方归一化精确匹配，不是选择题准确率或LLM裁判分数。Recall@5分母为4109条目标关联；184道无clue题仍计入EM/F1。</p>',
           '<p><a href="ablation.csv">消融CSV</a> · <a href="paired_results.csv">逐题结果</a> · <a href="summary.json">完整统计</a></p></html>']
    (base/'report.html').write_text('\n'.join(body))
    files=[base/n for n in ['retrieval_protocol.json','retrieval_results.jsonl','summary.json','ablation.csv','metrics.csv',
        'paired_results.csv','report.html','lambda_ablation.png','lambda_ablation.pdf','lambda_ablation.svg',
        'reader/protocol.json','reader/qa_manifest.jsonl','reader/responses.jsonl','reader/validation_output.json',
        'reader/summary.json','reader/paired_results.json']]
    files+=list((base/'reader').glob('generation_environment_*.json'))
    dump(base/'completion_manifest.json',{'status':'complete','questions_per_condition':1711,'conditions':12,
        'artifact_sha256':{str(p):sha(p) for p in files}})
    print(json.dumps({'stage':'complete','ablation':table,'reader':summary['reader_validation']}),flush=True)
    return summary


def plot(base,table,overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.2))
    for ax,metric,label in zip(axes,['em','recall_at_5'],['Exact match (×100)','Recall@5, micro (×100)']):
        for prefix,method,color,name in zip(['baseline','master'],METHODS,['#87B3D9','#184B7D'],['Caption baseline','Caption visual master']):
            values=[r[prefix+'_'+metric] for r in table]
            ax.plot(LAMBDAS,values,'o-',color=color,label=name+' + PV',linewidth=2)
            control=overall[method]['em' if metric=='em' else 'recall_at_5_micro']
            ax.axhline(control,color=color,linestyle='--',alpha=.65,label=name+' (unweighted)')
            for x,y in zip(LAMBDAS,values):ax.annotate(f'{y*100:.2f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',color=color,fontsize=12,weight='bold')
        ax.set_xscale('log');ax.set_xticks(LAMBDAS,[format(v,'g') for v in LAMBDAS]);ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v*100:.2f}'))
        ax.set_xlabel('Lambda (log scale)',fontsize=17);ax.set_ylabel(label,fontsize=17)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.005),frameon=False,prop={'size':13,'weight':'bold'})
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.29,wspace=.35)
    for ext in ['png','svg','pdf']:fig.savefig(base/f'lambda_ablation.{ext}',dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--stage',choices=['prepare','reader','analyze','all'],default='all')
    args=parser.parse_args();base=args.output.resolve()
    if args.stage in ['prepare','all']:prepare(base,args.device)
    if args.stage in ['reader','all']:run_reader(base,args.device)
    if args.stage in ['analyze','all']:analyze(base)


if __name__=='__main__':main()
