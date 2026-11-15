"""Matched Qwen MemEye lambda ablation of the production positive_variance component.

Five fixed lambdas, alpha=1, both retrieval strategies and two fresh controls.
All conditions share one same-run reader payload cache. The previous lambda=.03
run is a read-only numerical/answer reference, never a source of cached answers.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import html
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.vendor'), str(ROOT)]
for key,value in {'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                  'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                  'MPLCONFIGDIR':'/tmp/visualmemory-matplotlib'}.items():
    os.environ.setdefault(key,value)
import numpy as np
from scripts.analysis import evaluate_memeye_positive_variance as single
from scripts.memory import online_retrieval as online
from scripts.memory.online_positive_variance import PositiveVarianceWeighting

read, lines, dump, sha, digest = single.read, single.lines, single.dump, single.sha, single.digest
check_hashes, write_lines = single.check_hashes, single.write_lines
ANCHOR = single.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT/'results/memeye_qwen_positive_variance_lambda_ablation_20261107'
LAMBDAS = (.001,.003,.01,.03,.1)
ALPHA = 1.
METHODS = single.METHODS


def condition(method,ridge):
    return method if ridge is None else method+'+positive_variance__l'+format(ridge,'g').replace('.','p')+'_a1'


SPECS = {method:{'method':method,'lambda':None,'alpha':0.} for method in METHODS}
SPECS.update({condition(method,ridge):{'method':method,'lambda':ridge,'alpha':ALPHA}
              for ridge in LAMBDAS for method in METHODS})


def retrieve(base,device):
    path=base/'retrieval_protocol.json'
    if path.exists():
        p=read(path);assert p['condition_specs']==SPECS
        check_hashes(p['source_sha256']);check_hashes(p['output_sha256'])
        return p
    anchor=read(ANCHOR/'retrieval_protocol.json')
    check_hashes(anchor['source_sha256']);check_hashes(anchor['output_sha256'])
    previous={r['question_id']:r for r in lines(ANCHOR/'retrieval_results.jsonl')}
    search=read(single.SEARCH);stores=Path(search['memory_path'])
    sources={**anchor['source_sha256'],**anchor['output_sha256']}
    sources.update({str(p):sha(p) for p in [Path(__file__),ANCHOR/'retrieval_protocol.json']})
    original=lines(single.ORIGINAL/'qa_manifest.jsonl')
    canonical={r['id']:r['reader_text'] for r in lines(single.ORIGINAL/'retrieval_memories.jsonl')}
    results={};raw_checks=anchor_checks=group_checks=0
    max_raw_error=max_anchor_error=max_anchor_z_error=max_oracle_error=0.
    base.mkdir(parents=True,exist_ok=True)
    online.configure_device(SimpleNamespace(device=device))
    for domain in sorted(search['master_counts']):
        src=single.INPUTS/domain/'inputs.json'
        assert sha(src)==anchor['source_sha256'][str(src)]
        # Strip evaluation labels before memory planning, encoding, and ranking.
        questions=[{k:v for k,v in q.items() if k not in ('gold_ids','gold_choice')}
                   for q in read(src)['questions']]
        args=SimpleNamespace(mode='caption_visual_master',memory_dir=stores/domain/'caption_visual_master',
                             candidates=single.REFERENCE/'candidates.jsonl',retrieval='multi_path',device=device)
        memory,metadata,paths,_=online.load_memory(args,questions)
        natural,_,natural_paths,_=online.load_memory(SimpleNamespace(mode='natural',memory_dir=stores/domain/'natural'),questions)
        assert memory.texts==natural.texts
        assert all(text==canonical[leaf['memory_id']] for text,leaf in zip(memory.texts,memory.leaves))
        for p in [*paths,*natural_paths]:
            assert sha(p)==anchor['source_sha256'][str(p)]
        old_details={method:{q['question_id']:q for q in read(ANCHOR/'domains'/domain/method/'retrieval_results.json')['questions']}
                     for method in METHODS}
        encoder=online.load_text_encoder(metadata['text_encoder'],device)
        ids=[leaf['memory_id'] for leaf in memory.leaves]
        for question in questions:
            qid=question['id'];item={'question_id':qid,'domain':domain,'conditions':{},'methods':{}}
            # Only reuse this transient leaf encoding within the current question.
            vectors=online.encode_memory_text(memory,encoder)
            for method,obj,strategy in zip(METHODS,[natural,memory],[online.FullScanRetrieval(),online.MultiPathRetrieval()]):
                texts,groups=strategy.plan(question,obj)
                queries=online.encode_text(encoder,texts,metadata['query_prefix'])
                prior=old_details[method][qid]
                assert ids==[leaf['memory_id'] for leaf in prior['leaf_scores']]
                raw=np.empty(len(ids));group_rows=[]
                stats={str(ridge):{key:np.empty(len(ids)) for key in ['enhanced','z','correction','weighted']}
                       for ridge in LAMBDAS}
                for group in groups:
                    ix=np.asarray(group.indices);x=vectors[ix];q=queries[group.query_index]
                    old_group=next(g for g in prior['groups'] if g['group_id']==group.id)
                    assert texts[group.query_index]==old_group['query'] and len(ix)==old_group['size']
                    if method==METHODS[1]:
                        assert all(group.details[key]==old_group[key] for key in ['choice','selected_statement'])
                    centered=x-x.mean(0)
                    for ridge in LAMBDAS:
                        result=PositiveVarianceWeighting(ridge,ALPHA).apply(q,x)
                        # Independent dual covariance oracle verifies the POSITIVE
                        # SVD sign for every real group and every tested lambda.
                        enhanced=result['raw'].copy()
                        if len(x)>1:
                            enhanced+=x@centered.T@np.linalg.solve(
                                centered@centered.T+(len(x)-1)*ridge*np.eye(len(x)),centered@q)
                        error=float(np.max(np.abs(enhanced-result['positive_variance'])))
                        assert error<2e-12,(qid,group.id,ridge,error)
                        max_oracle_error=max(max_oracle_error,error);group_checks+=1
                        if ridge==LAMBDAS[0]:
                            raw[ix]=result['raw']
                            group_rows.append({'id':group.id,'indices':list(group.indices),'size':len(ix),
                                               'query':texts[group.query_index],**group.details,
                                               'raw_std':result['raw_std'],'raw_status':result['raw_status']})
                        for key,field in [('enhanced','positive_variance'),('z','z'),('correction','correction'),('weighted','weighted')]:
                            stats[str(ridge)][key][ix]=result[field]
                raw_order=np.argsort(-raw,kind='stable')
                assert [ids[i] for i in raw_order[:5]]==previous[qid]['conditions'][method]['top5_memory_ids']
                raw_checks+=1
                error=max(abs(raw[i]-leaf['raw_score']) for i,leaf in enumerate(prior['leaf_scores']))
                assert error<2e-6;max_raw_error=max(max_raw_error,error)
                anchor_stats=stats['0.03']
                for key,field in [('enhanced','positive_variance_score'),('correction','correction'),('weighted','weighted_score'),('z','positive_variance_z')]:
                    error=max(abs(anchor_stats[key][i]-leaf[field]) for i,leaf in enumerate(prior['leaf_scores']))
                    assert error<(2e-5 if key=='z' else 2e-6),(qid,key,error)
                    if key=='z':max_anchor_z_error=max(max_anchor_z_error,error)
                    else:max_anchor_error=max(max_anchor_error,error)
                for ridge in [None,*LAMBDAS]:
                    values=raw if ridge is None else stats[str(ridge)]['weighted']
                    order=np.argsort(-values,kind='stable')[:5]
                    top=[ids[i] for i in order]
                    if ridge==.03:
                        assert top==previous[qid]['conditions'][method+'+positive_variance']['top5_memory_ids']
                        anchor_checks+=1
                    item['conditions'][condition(method,ridge)]={'top5_memory_ids':top,'top5_scores':[float(values[i]) for i in order]}
                item['methods'][method]={'memory_ids':ids,'raw':raw.tolist(),'groups':group_rows,
                                          'weighting':{key:{k:v.tolist() for k,v in value.items()} for key,value in stats.items()}}
            results[qid]=item
        print(json.dumps({'stage':'retrieval_domain','domain':domain,'questions':len(results),
                          'raw_matches':raw_checks,'lambda_0p03_matches':anchor_checks}),flush=True)
        del encoder,memory,natural,vectors,queries
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    assert len(results)==371 and raw_checks==anchor_checks==742
    write_lines(base/'retrieval_results.jsonl',[results[q['id']] for q in original])
    check_hashes(sources)
    p={'scheme':'positive_variance','lambdas':list(LAMBDAS),'alpha':ALPHA,'condition_specs':SPECS,
       'questions':371,'domains':8,'memory_leaves':438,'masters':88,
       'raw_top5_matches':raw_checks,'lambda_0p03_top5_matches':anchor_checks,
       'max_raw_score_error':max_raw_error,'max_lambda_0p03_score_error':max_anchor_error,
       'max_lambda_0p03_z_error':max_anchor_z_error,'independent_positive_covariance_groups':group_checks,
       'max_independent_covariance_error':max_oracle_error,'device':device,
       'source_sha256':sources,'output_sha256':{str(base/'retrieval_results.jsonl'):sha(base/'retrieval_results.jsonl')},
       'encoding_reuse':'Within one question only; same transient leaf vectors and method-specific queries across lambdas.',
       'vectors_persisted':False,'memory_modified':False,'caption_regenerated':False,'gold_used_for_ranking':False,
       'intermediate_score':anchor['intermediate_score'],'final_score':anchor['final_score']}
    dump(path,p)
    return p


def prepare(base,device):
    out=base/'reader';path=out/'protocol.json'
    if path.exists():
        p=read(path);single.verify(p);assert p['condition_specs']==SPECS
        return p
    method=read(base/'retrieval_protocol.json');check_hashes(method['source_sha256']);check_hashes(method['output_sha256'])
    previous=read(ANCHOR/'reader/protocol.json');signed=dict(previous)
    signature=signed.pop('experiment_signature');assert digest(signed)==signature
    settings=previous['reader_settings']
    assert settings['seed']==42 and settings['do_sample'] is False
    original={q['id']:q for q in lines(single.ORIGINAL/'qa_manifest.jsonl')}
    memories={m['id']:m['reader_text'] for m in lines(single.ORIGINAL/'retrieval_memories.jsonl')}
    old={r['record_id']:r for r in lines(ANCHOR/'reader/qa_manifest.jsonl')}
    manifest=[];images={};anchor_matches=0
    for qi,item in enumerate(lines(base/'retrieval_results.jsonl')):
        q=original[item['question_id']];batch=[]
        for name in q['query_image_paths']:images[name]={'path':name,'sha256':sha(name)}
        query_images=[images[name] for name in q['query_image_paths']]
        for name,spec in SPECS.items():
            result=item['conditions'][name];ids=result['top5_memory_ids']
            assert len(ids)==len(set(ids))==5 and all(mid.split('/',1)[0]==q['task'] for mid in ids)
            text=single.reader_text(q,ids,memories)
            payload={'system':settings['reader_system_prompt'],'user_text':text,'query_images':query_images}
            if spec['lambda'] in (None,.03):
                old_name=spec['method']+('' if spec['lambda'] is None else '+positive_variance')
                assert digest(payload)==old[q['id']+'::'+old_name]['payload_sha256']
                anchor_matches+=1
            batch.append({**{k:q[k] for k in ['id','task','question','options','gold_choice','gold_ids','query_image_paths']},
                          'record_id':q['id']+'::'+name,'condition':name,'top_memory_ids':ids,
                          'top_query_scores':result['top5_scores'],'reader_text':text,'query_images':query_images,
                          'payload_sha256':digest(payload)})
        offset=qi%len(SPECS);manifest.extend(batch[offset:]+batch[:offset])
    assert len(manifest)==371*12 and anchor_matches==1484 and len(images)==6
    out.mkdir(parents=True,exist_ok=True);write_lines(out/'qa_manifest.jsonl',manifest)
    check_hashes(previous['model_source_sha256'])
    weights=copy.deepcopy(previous['weight_files'])
    for weight in weights:
        assert sha(weight['path'])==weight['sha256']
        stat=Path(weight['path']).stat();weight.update(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    annotations={}
    for name,spec in SPECS.items():
        old_name=spec['method']+('' if spec['lambda'] is None else '+positive_variance')
        annotations[name]=copy.deepcopy(previous['method_annotations'][old_name])
        for domain,description in annotations[name].items():
            description['scope']=name
            description['3. 检索方式'].update({'lambda':spec['lambda'],'alpha':spec['alpha'] if spec['lambda'] is not None else None})
    sources={**method['source_sha256'],**method['output_sha256']}
    for file in [base/'retrieval_protocol.json',out/'qa_manifest.jsonl',ANCHOR/'reader/protocol.json',
                 ANCHOR/'reader/qa_manifest.jsonl',ANCHOR/'reader/responses.jsonl',ANCHOR/'summary.json']:
        sources[str(file)]=sha(file)
    p={'design':'Five fixed positive-variance lambdas, alpha=1, both retrieval methods, two fresh unweighted controls.',
       'experiment_run_id':str(uuid.uuid4()),'condition_names':list(SPECS),'condition_specs':SPECS,
       'lambdas':list(LAMBDAS),'alpha':ALPHA,'questions':371,'logical_conditions':len(manifest),
       'unique_reader_payloads':len({r['payload_sha256'] for r in manifest}),'reader_settings':settings,
       'placement':{'gpu_index':int(device.split(':')[1]),'minimum_free_gib':24},'attention_implementation':'sdpa',
       'query_images':list(images.values()),'image_questions':8,'unique_query_images':6,
       'historical_response_reuse':False,'deduplication':'Same full payload shares one fresh answer within this joint run only.',
       'source_sha256':sources,'model_source_sha256':previous['model_source_sha256'],'weight_files':weights,
       'method_annotations':annotations}
    p['experiment_signature']=digest(p);dump(path,p);single.verify(p)
    dump(out/'validation_input.json',{'passed':True,'records':len(manifest),'anchor_payload_matches':anchor_matches,
                                     'unique_payloads':p['unique_reader_payloads'],'model_weight_hashes_verified':True})
    print(json.dumps({'stage':'reader_prepared','records':len(manifest),'unique_payloads':p['unique_reader_payloads']}),flush=True)
    return p


def analyze(out):
    out=Path(out);base=out.parent;p=read(out/'protocol.json');single.verify(p)
    signed=dict(p);signature=signed.pop('experiment_signature');assert digest(signed)==signature
    manifest={r['record_id']:r for r in lines(out/'qa_manifest.jsonl')}
    responses=lines(out/'responses.jsonl');by_id={r['record_id']:r for r in responses}
    assert len(manifest)==len(responses)==len(by_id)==371*12 and set(manifest)==set(by_id)
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
            assert all(r[k]==origin[k] for k in ['payload_sha256','raw_response','generated_token_ids','execution_id'])
            shared+=1
    assert len(fresh)==p['unique_reader_payloads'] and len(fresh)+shared==371*12
    original=lines(single.ORIGINAL/'qa_manifest.jsonl');ids=[q['id'] for q in original]
    overall={};domain_metrics={};annotations=copy.deepcopy(p['method_annotations'])
    for name in SPECS:
        for domain in [None,*sorted({q['task'] for q in original})]:
            chosen=[manifest[q['id']+'::'+name] for q in original if domain is None or q['task']==domain]
            hits=[len(set(r['top_memory_ids'])&set(r['gold_ids'])) for r in chosen]
            totals=[len(set(r['gold_ids'])) for r in chosen]
            correct=sum(by_id[r['record_id']]['correct'] for r in chosen)
            value={'correct':correct,'questions':len(chosen),'accuracy':correct/len(chosen),
                   'gold_hits_top5':sum(hits),'gold_references':sum(totals),'recall_at_5_micro':sum(hits)/sum(totals),
                   'recall_at_5_macro':float(np.mean([h/t for h,t in zip(hits,totals) if t])),
                   'questions_with_any_gold':sum(h>0 for h in hits),'hit_at_5':sum(h>0 for h in hits)/len(chosen)}
            if domain is None:
                assert sum(totals)==1041;overall[name]=value
            else:
                domain_metrics.setdefault(domain,{})[name]=value
                annotations[name][domain]['5. 回答方式']={**p['reader_settings'],'reader_called':True,
                    'correct':correct,'questions':len(chosen),'measured_qa_accuracy':correct/len(chosen),
                    'candidate_statements_in_reader':False,'note':'Fresh joint run; identical payloads share one same-run answer.'}
    old={r['record_id']:r for r in lines(ANCHOR/'reader/responses.jsonl')}
    anchor_differences={}
    for method in METHODS:
        for ridge in [None,.03]:
            name=condition(method,ridge);old_name=method+('' if ridge is None else '+positive_variance')
            anchor_differences[name]=[qid for qid in ids if by_id[qid+'::'+name]['predicted_choice']!=old[qid+'::'+old_name]['predicted_choice']]
    table=[];paired={}
    for ridge in LAMBDAS:
        row={'lambda':ridge,'alpha':ALPHA}
        for prefix,method in zip(['baseline','master'],METHODS):
            name=condition(method,ridge);metrics=overall[name]
            wins=[];losses=[];changed=[]
            for qid in ids:
                a,b=by_id[qid+'::'+method],by_id[qid+'::'+name]
                if not a['correct'] and b['correct']:wins.append(qid)
                if a['correct'] and not b['correct']:losses.append(qid)
                if a['payload_sha256']!=b['payload_sha256']:changed.append(qid)
                else:assert a['raw_response']==b['raw_response']
            row.update({prefix+'_correct':metrics['correct'],prefix+'_accuracy':metrics['accuracy'],
                        prefix+'_gain':metrics['correct']-overall[method]['correct'],
                        prefix+'_recall_at_5':metrics['recall_at_5_micro'],prefix+'_gold_hits':metrics['gold_hits_top5'],
                        prefix+'_wrong_to_right':len(wins),prefix+'_right_to_wrong':len(losses),
                        prefix+'_changed_payloads':len(changed)})
            paired[name]={'wrong_to_right':wins,'right_to_wrong':losses,'changed_payload_questions':changed}
        table.append(row)
    summary={'status':'complete','scheme':'positive_variance','lambdas':list(LAMBDAS),'alpha':ALPHA,
             'overall':overall,'ablation':table,'per_domain':domain_metrics,'paired_comparisons':paired,
             'anchor_answer_changes':anchor_differences,'logical_records':len(responses),'fresh_inferences':len(fresh),
             'shared_records':shared,'format_errors':sum(r['status']!='ok' for r in responses),
             'empty_responses':sum(not r['raw_response'].strip() for r in responses),
             'token_limit_hits':sum(r['hit_token_limit'] for r in responses),'historical_answers_reused':0,
             'method_annotations':annotations,'reader_settings':p['reader_settings'],
             'recall_definition':'Micro Recall@5 = total gold hits / 1041 references; macro also reported in overall.',
             'interpretation':'Descriptive lambda ablation on the same 371 benchmark questions, not held-out parameter validation.'}
    dump(base/'summary.json',summary)
    for filename,records in [('ablation.csv',table),('metrics.csv',[{'condition':name,**value} for name,value in overall.items()])]:
        with (base/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    paired_rows=[]
    for q in original:
        row={'question_id':q['id'],'domain':q['task'],'gold_choice':q['gold_choice']}
        for name in SPECS:
            answer=by_id[q['id']+'::'+name];record=manifest[answer['record_id']]
            row[name+'_answer']=answer['predicted_choice'];row[name+'_correct']=answer['correct']
            row[name+'_top5']=' | '.join(record['top_memory_ids'])
        paired_rows.append(row)
    with (base/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    plot(base,table,overall)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Positive Variance λ 消融</title>',
          '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style>',
          '<h1>Qwen MemEye：Positive Variance λ 消融</h1><p>α=1；λ=0.001、0.003、0.01、0.03、0.1；371 题。冻结原始 Qwen caption、memory、候选陈述及 reader 设置，seed=42。两个未赋权对照和十个赋权条件共同重新回答；不复用历史答案。</p>',
          '<p>SVD 正向增强：h=raw+Σ[v/(v+λ)](q·u)(e·u)；最终仍为 raw+α·std(raw)·tanh(z(h))。</p>',
          '<p>本轮未赋权对照：baseline '+str(overall[METHODS[0]]['correct'])+'/371；master '+str(overall[METHODS[1]]['correct'])+'/371。</p>',
          '<img src="lambda_ablation.png"><table><tr><th>λ</th><th>Baseline 答对</th><th>Master 答对</th><th>Baseline Recall@5</th><th>Master Recall@5</th></tr>']
    for r in table:body.append(f'<tr><td>{r["lambda"]}</td><td>{r["baseline_correct"]}/371</td><td>{r["master_correct"]}/371</td><td>{r["baseline_recall_at_5"]:.2%}</td><td>{r["master_recall_at_5"]:.2%}</td></tr>')
    body+=['</table><p>Recall@5 为命中目标记忆数 / 1041；当前测试集上的消融不等同于独立测试集泛化验证。</p>',
           '<p><a href="ablation.csv">消融表 CSV</a> · <a href="summary.json">完整统计与五项方法标注</a> · <a href="paired_results.csv">逐题结果</a></p></html>']
    (base/'report.html').write_text('\n'.join(body))
    check_hashes(read(base/'retrieval_protocol.json')['source_sha256'])
    audit={'passed':True,'records':len(responses),'conditions':len(SPECS),'questions_per_condition':371,
           'unique_payloads':len(fresh),'shared_records':shared,'input_hashes_unchanged':True,
           'same_payload_same_answer':True,'response_ids_and_signatures_verified':True,
           'format_errors':summary['format_errors'],'empty_responses':summary['empty_responses'],
           'token_limit_hits':summary['token_limit_hits'],'anchor_answer_changes':{k:len(v) for k,v in anchor_differences.items()}}
    dump(base/'final_audit.json',audit)
    artifacts=[base/n for n in ['retrieval_protocol.json','retrieval_results.jsonl','summary.json','ablation.csv',
        'metrics.csv','paired_results.csv','report.html','final_audit.json','lambda_ablation.png','lambda_ablation.svg','lambda_ablation.pdf']]
    artifacts += [out/n for n in ['protocol.json','qa_manifest.jsonl','responses.jsonl','validation_input.json','generation_environment.json']]
    dump(base/'completion_manifest.json',{'status':'complete','experiment_signature':signature,
                                         'artifact_sha256':{str(path):sha(path) for path in artifacts}})
    print(json.dumps({'stage':'complete','ablation':table,'audit':audit}),flush=True)
    return summary


def plot(base,table,overall):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib import font_manager
    for font in Path('/storage/hewenwen/.local/share/fonts/TimesNewRoman').glob('*.TTF'):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelweight':'bold',
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    for ax,metric,title in zip(axes,['accuracy','recall_at_5'],['Answer accuracy','Target-memory Recall@5 (micro)']):
        for prefix,method,color,label in zip(['baseline','master'],METHODS,['#87B3D9','#184B7D'],['Caption baseline','Caption visual master']):
            values=[r[prefix+'_'+metric] for r in table]
            ax.plot(LAMBDAS,values,'o-',color=color,label=label+' + PV',linewidth=2)
            control=overall[method]['accuracy' if metric=='accuracy' else 'recall_at_5_micro']
            ax.axhline(control,color=color,linestyle='--',alpha=.65,label=label+' (unweighted)')
            for x,y,r in zip(LAMBDAS,values,table):
                ax.annotate(f'{y*100:.2f}', (x,y),xytext=(0,9),textcoords='offset points',ha='center',color=color,fontsize=12,weight='bold')
        ax.set_xscale('log');ax.set_xticks(LAMBDAS,[format(v,'g') for v in LAMBDAS]);ax.minorticks_off()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value*100:.2f}'))
        ax.set_xlabel('Lambda (log scale)',fontsize=17);ax.set_ylabel(title+' (×100)',fontsize=17)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontweight('bold')
        ax.margins(y=.3)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,.005),frameon=False,prop={'size':13,'weight':'bold'})
    fig.subplots_adjust(left=.085,right=.98,top=.92,bottom=.29,wspace=.35)
    for ext in ['png','svg','pdf']:fig.savefig(base/f'lambda_ablation.{ext}',dpi=220,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--stage',choices=['retrieve','prepare','run','analyze','all'],default='all')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();base=args.output.resolve()
    if args.stage in ['retrieve','all']:retrieve(base,args.device)
    if args.stage in ['prepare','all']:prepare(base,args.device)
    if args.stage in ['run','all']:single.run_reader(base/'reader',analyze,args.resume)
    if args.stage=='analyze':analyze(base/'reader')


if __name__=='__main__':main()
