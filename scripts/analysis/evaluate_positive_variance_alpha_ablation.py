"""Qwen MemEye: fixed positive-variance lambda=.03, seven alpha values.

Reuse the signed lambda=.03 raw scores and unit corrections, then recompute
rankings. Both unweighted controls and all fourteen weighted conditions receive
fresh joint Qwen inference, with identical payloads shared only within this run.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
for key,value in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                 'MPLCONFIGDIR':'/tmp/visualmemory-matplotlib'}.items():
    os.environ.setdefault(key,value)
import numpy as np
from scripts.analysis import evaluate_positive_variance_lambda_ablation as previous
single=previous.single
read,lines,dump,sha,digest=previous.read,previous.lines,previous.dump,previous.sha,previous.digest
check_hashes,write_lines=previous.check_hashes,previous.write_lines
SOURCE=previous.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'results/memeye_qwen_positive_variance_alpha_ablation_l0p03_20261107'
RIDGE=.03
ALPHAS=(.1,.3,.6,1.,3.,6.,10.)
METHODS=previous.METHODS


def condition(method,alpha):
    return method if alpha is None else method+'+positive_variance__l0p03_a'+format(alpha,'g').replace('.','p')


SPECS={method:{'method':method,'lambda':None,'alpha':0.} for method in METHODS}
SPECS.update({condition(method,a):{'method':method,'lambda':RIDGE,'alpha':a} for a in ALPHAS for method in METHODS})
N_RECORDS=371*len(SPECS)


def retrieve(base):
    path=base/'retrieval_protocol.json'
    if path.exists():
        p=read(path);assert p['condition_specs']==SPECS
        check_hashes(p['source_sha256']);check_hashes(p['output_sha256']);return p
    original=read(SOURCE/'retrieval_protocol.json')
    completion=read(SOURCE/'completion_manifest.json')
    check_hashes(original['source_sha256']);check_hashes(original['output_sha256'])
    check_hashes(completion['artifact_sha256'])
    assert read(SOURCE/'independent_audit.json')['passed']
    sources={**original['source_sha256'],**original['output_sha256']}
    for file in [Path(__file__),SOURCE/'retrieval_protocol.json',SOURCE/'completion_manifest.json',
                 SOURCE/'independent_audit.json',SOURCE/'reader/protocol.json',SOURCE/'reader/qa_manifest.jsonl',
                 SOURCE/'reader/responses.jsonl',SOURCE/'summary.json']:
        sources[str(file)]=sha(file)
    output=[];group_checks=raw_checks=alpha1_checks=0;max_correction_error=0.
    for row in lines(SOURCE/'retrieval_results.jsonl'):
        current={'question_id':row['question_id'],'domain':row['domain'],'conditions':{},'methods':{}}
        for method in METHODS:
            data=row['methods'][method];ids=data['memory_ids'];raw=np.array(data['raw'])
            stats=data['weighting']['0.03'];unit=np.array(stats['correction'])
            enhanced=np.array(stats['enhanced']);z=np.array(stats['z'])
            assert np.isfinite(raw).all() and np.isfinite(unit).all()
            assert sorted(i for g in data['groups'] for i in g['indices'])==list(range(len(raw)))
            for group in data['groups']:
                ix=group['indices'];std=float(np.std(raw[ix]));expected=np.zeros(len(ix))
                if len(ix)>1 and group['raw_status']!='zero_variance':expected=std*np.tanh(z[ix])
                error=float(np.max(abs(unit[ix]-expected)));assert error<1e-12
                max_correction_error=max(max_correction_error,error);group_checks+=1
                assert max(abs(unit[ix]))<=std+1e-12
            assert np.array_equal(raw+unit,np.array(stats['weighted']))
            for alpha in [None,*ALPHAS]:
                values=raw if alpha is None else raw+alpha*unit
                order=np.argsort(-values,kind='stable')[:5];top=[ids[i] for i in order]
                name=condition(method,alpha)
                if alpha in (None,1.):
                    assert top==row['conditions'][name]['top5_memory_ids']
                    assert np.array_equal(values[order],np.array(row['conditions'][name]['top5_scores']))
                    if alpha is None:raw_checks+=1
                    else:alpha1_checks+=1
                current['conditions'][name]={'top5_memory_ids':top,'top5_scores':values[order].tolist()}
            current['methods'][method]={'memory_ids':ids,'raw':raw.tolist(),'unit_correction':unit.tolist(),
                'enhanced':enhanced.tolist(),'z':z.tolist(),'groups':data['groups']}
        output.append(current)
    assert len(output)==371 and raw_checks==alpha1_checks==742
    base.mkdir(parents=True,exist_ok=True);write_lines(base/'retrieval_results.jsonl',output)
    check_hashes(sources)
    p={'dataset':'MemEye','scheme':'positive_variance','lambda':RIDGE,'alphas':list(ALPHAS),
       'condition_specs':SPECS,'questions':371,'domains':8,'source_run':str(SOURCE),
       'raw_top5_matches':raw_checks,'alpha1_top5_matches':alpha1_checks,
       'unit_correction_group_checks':group_checks,'max_unit_correction_error':max_correction_error,
       'retrieval_execution':'Reuse frozen verified lambda=.03 raw/enhanced/z scores and unit correction. Recompute raw+alpha*unit_correction and stable global Top-5 for each requested alpha.',
       'intermediate_score':original['intermediate_score'],'final_score':original['final_score'],
       'caption_regenerated':False,'memory_modified':False,'gold_used_for_ranking':False,
       'vectors_persisted':False,'source_sha256':sources,
       'output_sha256':{str(base/'retrieval_results.jsonl'):sha(base/'retrieval_results.jsonl')}}
    dump(path,p)
    print(json.dumps({'stage':'retrieval_complete','records':N_RECORDS,'raw_matches':raw_checks,
                      'alpha1_matches':alpha1_checks,'groups':group_checks}),flush=True)
    return p


def prepare(base,device):
    out=base/'reader';path=out/'protocol.json'
    if path.exists():
        p=read(path);single.verify(p);assert p['condition_specs']==SPECS;return p
    method=read(base/'retrieval_protocol.json');check_hashes(method['source_sha256']);check_hashes(method['output_sha256'])
    prior=read(SOURCE/'reader/protocol.json');signed=dict(prior)
    signature=signed.pop('experiment_signature');assert digest(signed)==signature
    settings=prior['reader_settings'];assert settings['seed']==42 and not settings['do_sample']
    questions={q['id']:q for q in lines(single.ORIGINAL/'qa_manifest.jsonl')}
    memories={m['id']:m['reader_text'] for m in lines(single.ORIGINAL/'retrieval_memories.jsonl')}
    old={r['record_id']:r for r in lines(SOURCE/'reader/qa_manifest.jsonl')}
    manifest=[];images={};anchor_matches=0
    for qi,row in enumerate(lines(base/'retrieval_results.jsonl')):
        q=questions[row['question_id']];batch=[]
        for name in q['query_image_paths']:images[name]={'path':name,'sha256':sha(name)}
        query_images=[images[name] for name in q['query_image_paths']]
        for name,spec in SPECS.items():
            result=row['conditions'][name];ids=result['top5_memory_ids']
            assert len(ids)==len(set(ids))==5 and all(mid.split('/',1)[0]==q['task'] for mid in ids)
            text=single.reader_text(q,ids,memories)
            payload={'system':settings['reader_system_prompt'],'user_text':text,'query_images':query_images}
            if spec['alpha'] in (0.,1.):
                assert digest(payload)==old[q['id']+'::'+name]['payload_sha256'];anchor_matches+=1
            batch.append({**{k:q[k] for k in ['id','task','question','options','gold_choice','gold_ids','query_image_paths']},
                'record_id':q['id']+'::'+name,'condition':name,'top_memory_ids':ids,
                'top_query_scores':result['top5_scores'],'reader_text':text,'query_images':query_images,
                'payload_sha256':digest(payload)})
        offset=qi%len(SPECS);manifest.extend(batch[offset:]+batch[:offset])
    assert len(manifest)==N_RECORDS and anchor_matches==1484 and len(images)==6
    out.mkdir(parents=True,exist_ok=True);write_lines(out/'qa_manifest.jsonl',manifest)
    check_hashes(prior['model_source_sha256']);weights=copy.deepcopy(prior['weight_files'])
    for weight in weights:
        assert sha(weight['path'])==weight['sha256']
        stat=Path(weight['path']).stat();weight.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    annotations={}
    for name,spec in SPECS.items():
        old_name=spec['method'] if spec['lambda'] is None else condition(spec['method'],1.)
        annotations[name]=copy.deepcopy(prior['method_annotations'][old_name])
        for domain,description in annotations[name].items():
            description['scope']=name
            description['3. 检索方式'].update({'lambda':spec['lambda'],'alpha':spec['alpha'] if spec['lambda'] is not None else None})
    sources={**method['source_sha256'],**method['output_sha256']}
    for file in [base/'retrieval_protocol.json',out/'qa_manifest.jsonl']:sources[str(file)]=sha(file)
    p={'design':'Seven fixed positive-variance alphas, lambda=.03, two retrieval methods, two fresh unweighted controls.',
       'experiment_run_id':str(uuid.uuid4()),'condition_names':list(SPECS),'condition_specs':SPECS,
       'lambda':RIDGE,'alphas':list(ALPHAS),'questions':371,'logical_conditions':N_RECORDS,
       'unique_reader_payloads':len({r['payload_sha256'] for r in manifest}),'reader_settings':settings,
       'placement':{'gpu_index':int(device.split(':')[1]),'minimum_free_gib':24},'attention_implementation':'sdpa',
       'query_images':list(images.values()),'image_questions':8,'unique_query_images':6,
       'historical_response_reuse':False,'deduplication':'Same full payload shares one fresh answer within this joint run only.',
       'source_sha256':sources,'model_source_sha256':prior['model_source_sha256'],'weight_files':weights,
       'method_annotations':annotations}
    p['experiment_signature']=digest(p);dump(path,p);single.verify(p)
    dump(out/'validation_input.json',{'passed':True,'records':N_RECORDS,'anchor_payload_matches':anchor_matches,
        'unique_payloads':p['unique_reader_payloads'],'model_weight_hashes_verified':True})
    print(json.dumps({'stage':'reader_prepared','records':N_RECORDS,'unique_payloads':p['unique_reader_payloads']}),flush=True)
    return p


def analyze(out):
    out=Path(out);base=out.parent;p=read(out/'protocol.json');single.verify(p)
    signed=dict(p);signature=signed.pop('experiment_signature');assert digest(signed)==signature
    manifest={r['record_id']:r for r in lines(out/'qa_manifest.jsonl')}
    responses=lines(out/'responses.jsonl');by_id={r['record_id']:r for r in responses}
    assert len(manifest)==len(responses)==len(by_id)==N_RECORDS and set(manifest)==set(by_id)
    fresh={};shared=0
    for r in responses:
        source=manifest[r['record_id']]
        match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',r['raw_response'].strip(),re.I)
        choice=match.group(1).upper() if match else None
        assert r['predicted_choice']==choice and r['correct']==(choice==source['gold_choice'])
        assert r['payload_sha256']==source['payload_sha256'] and r['experiment_signature']==signature
        assert r['run_id']==p['experiment_run_id'] and not r['historical_response_reuse']
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh;fresh[r['payload_sha256']]=r
        else:
            origin=by_id[r['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            assert all(r[k]==origin[k] for k in ['payload_sha256','raw_response','generated_token_ids','execution_id']);shared+=1
    assert len(fresh)==p['unique_reader_payloads'] and len(fresh)+shared==N_RECORDS
    original=lines(single.ORIGINAL/'qa_manifest.jsonl');ids=[q['id'] for q in original]
    overall={};domain_metrics={};annotations=copy.deepcopy(p['method_annotations'])
    for name in SPECS:
        for domain in [None,*sorted({q['task'] for q in original})]:
            selected=[manifest[q['id']+'::'+name] for q in original if domain is None or q['task']==domain]
            hits=[len(set(r['top_memory_ids'])&set(r['gold_ids'])) for r in selected]
            totals=[len(set(r['gold_ids'])) for r in selected]
            correct=sum(by_id[r['record_id']]['correct'] for r in selected)
            metrics={'correct':correct,'questions':len(selected),'accuracy':correct/len(selected),
                'gold_hits_top5':sum(hits),'gold_references':sum(totals),'recall_at_5_micro':sum(hits)/sum(totals),
                'recall_at_5_macro':float(np.mean([h/t for h,t in zip(hits,totals) if t])),
                'questions_with_any_gold':sum(h>0 for h in hits),'hit_at_5':sum(h>0 for h in hits)/len(selected)}
            if domain is None:assert sum(totals)==1041;overall[name]=metrics
            else:
                domain_metrics.setdefault(domain,{})[name]=metrics
                annotations[name][domain]['5. 回答方式']={**p['reader_settings'],'reader_called':True,
                    'correct':correct,'questions':len(selected),'measured_qa_accuracy':metrics['accuracy'],
                    'candidate_statements_in_reader':False,'note':'Fresh joint run; identical payloads share one same-run answer.'}
    old={r['record_id']:r for r in lines(SOURCE/'reader/responses.jsonl')}
    anchors={name:[qid for qid in ids if by_id[qid+'::'+name]['predicted_choice']!=old[qid+'::'+name]['predicted_choice']]
             for method in METHODS for name in [method,condition(method,1.)]}
    table=[];paired={}
    for alpha in ALPHAS:
        row={'lambda':RIDGE,'alpha':alpha}
        for prefix,method in zip(['baseline','master'],METHODS):
            name=condition(method,alpha);metrics=overall[name];wins=[];losses=[];changed=[]
            for qid in ids:
                a,b=by_id[qid+'::'+method],by_id[qid+'::'+name]
                if not a['correct'] and b['correct']:wins.append(qid)
                if a['correct'] and not b['correct']:losses.append(qid)
                if a['payload_sha256']!=b['payload_sha256']:changed.append(qid)
                else:assert a['raw_response']==b['raw_response']
            row.update({prefix+'_correct':metrics['correct'],prefix+'_accuracy':metrics['accuracy'],
                prefix+'_gain':metrics['correct']-overall[method]['correct'],
                prefix+'_recall_at_5':metrics['recall_at_5_micro'],prefix+'_gold_hits':metrics['gold_hits_top5'],
                prefix+'_wrong_to_right':len(wins),prefix+'_right_to_wrong':len(losses),prefix+'_changed_payloads':len(changed)})
            paired[name]={'wrong_to_right':wins,'right_to_wrong':losses,'changed_payload_questions':changed}
        table.append(row)
    summary={'status':'complete','scheme':'positive_variance','lambda':RIDGE,'alphas':list(ALPHAS),
        'overall':overall,'ablation':table,'per_domain':domain_metrics,'paired_comparisons':paired,
        'anchor_answer_changes':anchors,'logical_records':N_RECORDS,'fresh_inferences':len(fresh),'shared_records':shared,
        'format_errors':sum(r['status']!='ok' for r in responses),'empty_responses':sum(not r['raw_response'].strip() for r in responses),
        'token_limit_hits':sum(r['hit_token_limit'] for r in responses),'historical_answers_reused':0,
        'method_annotations':annotations,'reader_settings':p['reader_settings'],
        'recall_definition':'Micro Recall@5 = total gold hits / 1041 references; macro also reported.',
        'interpretation':'Descriptive alpha ablation on the same 371 benchmark questions, not held-out validation.'}
    dump(base/'summary.json',summary)
    wide=[]
    for q in original:
        row={'question_id':q['id'],'domain':q['task'],'gold_choice':q['gold_choice']}
        for name in SPECS:
            answer=by_id[q['id']+'::'+name];record=manifest[answer['record_id']]
            row.update({name+'_answer':answer['predicted_choice'],name+'_correct':answer['correct'],
                        name+'_top5':' | '.join(record['top_memory_ids'])})
        wide.append(row)
    for filename,rows in [('ablation.csv',table),('metrics.csv',[{'condition':n,**v} for n,v in overall.items()]),('paired_results.csv',wide)]:
        with (base/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot(base,table,overall)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Positive Variance α 消融</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
        '<h1>Qwen MemEye：Positive Variance α 消融</h1><p>λ=0.03；α=0.1、0.3、0.6、1、3、6、10；371题，seed=42。冻结caption、记忆图、候选陈述和查询，复用已核验λ=0.03的原始分数与单位修正项，按各α重新排序。两个未赋权对照与十四个赋权条件共同重新回答，不复用历史答案。</p>',
        '<p>h=raw+Σ[v/(v+λ)](q·u)(e·u)；最终 raw+α·std(raw)·tanh(z(h))。</p>',
        f'<p>本轮未赋权对照：baseline {overall[METHODS[0]]["correct"]}/371；master {overall[METHODS[1]]["correct"]}/371。</p>',
        '<img src="alpha_ablation.png"><table><tr><th>α</th><th>Baseline 答对</th><th>Master 答对</th><th>Baseline Recall@5</th><th>Master Recall@5</th></tr>']
    for r in table:body.append(f'<tr><td>{r["alpha"]}</td><td>{r["baseline_correct"]}/371</td><td>{r["master_correct"]}/371</td><td>{r["baseline_recall_at_5"]:.2%}</td><td>{r["master_recall_at_5"]:.2%}</td></tr>')
    body+=['</table><p>Recall@5 分母为1041条目标关联。当前测试集上的消融不等同于独立泛化验证。</p><p><a href="ablation.csv">消融CSV</a> · <a href="summary.json">完整统计及五项方法标注</a> · <a href="paired_results.csv">逐题结果</a></p></html>']
    (base/'report.html').write_text('\n'.join(body));check_hashes(read(base/'retrieval_protocol.json')['source_sha256'])
    audit={'passed':True,'records':N_RECORDS,'conditions':len(SPECS),'questions_per_condition':371,
        'fresh_inferences':len(fresh),'shared_records':shared,'same_payload_same_answer':True,
        'input_hashes_unchanged':True,'format_errors':summary['format_errors'],'empty_responses':summary['empty_responses'],
        'token_limit_hits':summary['token_limit_hits'],'anchor_answer_changes':{k:len(v) for k,v in anchors.items()}}
    dump(base/'final_audit.json',audit)
    artifacts=[base/n for n in ['retrieval_protocol.json','retrieval_results.jsonl','summary.json','ablation.csv','metrics.csv',
        'paired_results.csv','report.html','final_audit.json','alpha_ablation.png','alpha_ablation.svg','alpha_ablation.pdf']]
    artifacts += [out/n for n in ['protocol.json','qa_manifest.jsonl','responses.jsonl','validation_input.json','generation_environment.json']]
    dump(base/'completion_manifest.json',{'status':'complete','experiment_signature':signature,
        'artifact_sha256':{str(path):sha(path) for path in artifacts}})
    print(json.dumps({'stage':'complete','ablation':table,'audit':audit}),flush=True)
    return summary


def plot(base,table,overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import FuncFormatter
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.2))
    for ax,metric,label in zip(axes,['accuracy','recall_at_5'],['Answer accuracy (×100)','Recall@5, micro (×100)']):
        for prefix,method,color,name in zip(['baseline','master'],METHODS,['#E9B187','#A94B13'],['Caption baseline','Caption visual master']):
            values=[r[prefix+'_'+metric] for r in table]
            ax.plot(ALPHAS,values,'o-',color=color,label=name+' + PV',linewidth=2)
            control=overall[method]['accuracy' if metric=='accuracy' else 'recall_at_5_micro']
            ax.axhline(control,color=color,linestyle='--',alpha=.65,label=name+' (unweighted)')
            for x,y in zip(ALPHAS,values):ax.annotate(f'{y*100:.2f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',color=color,fontsize=11,weight='bold')
        ax.set_xscale('log');ax.set_xticks(ALPHAS,[format(v,'g') for v in ALPHAS]);ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v*100:.2f}'))
        ax.set_xlabel('Alpha (log scale)',fontsize=17);ax.set_ylabel(label,fontsize=17)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.005),frameon=False,prop={'size':13,'weight':'bold'})
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.29,wspace=.35)
    for ext in ['png','svg','pdf']:fig.savefig(base/f'alpha_ablation.{ext}',dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--stage',choices=['retrieve','prepare','run','analyze','all'],default='all')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();base=args.output.resolve()
    if args.stage in ['retrieve','all']:retrieve(base)
    if args.stage in ['prepare','all']:prepare(base,args.device)
    if args.stage in ['run','all']:single.run_reader(base/'reader',analyze,args.resume)
    if args.stage=='analyze':analyze(base/'reader')


if __name__=='__main__':main()
