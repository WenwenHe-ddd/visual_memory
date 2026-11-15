"""Postprocess completed DeepSeek answers without modifying the signed runner.

Evaluation labels deliberately lack task metadata. Resolve domains from source
questions; no API calls or input/response changes are made by this continuation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.vendor'), str(ROOT)]
from scripts.analysis.memeye_deepseek_flash_zscore import (
    OUTPUT, MODEL, CONDITIONS, BASELINE, WEIGHTED, RIDGE, ALPHA, verify_reader,
    validate_response, read, lines, indexed, sha, dump, choice)


def summarize(output=OUTPUT):
    output=Path(output).resolve();folder=output/'reader'
    p, requests=verify_reader(output);c=p['contract'];source=Path(c['memory_path'])
    gold=indexed(lines(folder/'gold.jsonl'),'id')
    questions=indexed(lines(source/'inputs/source_questions.jsonl'),'id')
    fresh_rows=lines(folder/'fresh_responses.jsonl');fresh=indexed(fresh_rows,'request_sha256')
    logical=indexed(lines(folder/'responses.jsonl'),'record_id')
    assert set(gold)==set(questions) and len(gold)==371 and len(logical)==742
    representatives={}
    for row in requests:representatives.setdefault(row['request_sha256'],row)
    assert set(representatives)==set(fresh) and len(fresh)==c['unique_requests']
    for h,ans in fresh.items():validate_response(ans,representatives[h],p,'reader')
    previous=lines(source/'reader/fresh_responses.jsonl')
    assert not {r['response_id'] for r in previous}&{r['response_id'] for r in fresh_rows}
    assert len({r['response_id'] for r in fresh_rows})==len(fresh_rows)
    attempts=lines(folder/'api_attempts.jsonl')
    assert Counter(a['request_sha256'] for a in attempts if a['status']=='ok')==dict.fromkeys(fresh,1)
    per_question=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':questions[qid]['task'],'gold_choice':g['gold_choice'],'gold_count':len(g['gold_ids'])}
        for name in CONDITIONS:
            ans=logical[qid+'::'+name];pred=choice(ans['raw_response']);origin=fresh[ans['request_sha256']]
            assert pred is not None and pred==ans['predicted_choice']
            assert ans['correct']==(pred==g['gold_choice'])
            for k in ('raw_response','response_id','payload_sha256','request_sha256','run_id','experiment_signature','execution_id'):
                assert ans[k]==origin[k]
            is_fresh=origin['record_id']==ans['record_id']
            assert ans['fresh_inference_executed_for_this_record']==is_fresh
            assert ans['same_run_shared_inference_record_id']==(None if is_fresh else origin['record_id'])
            row.update({name+'_answer':pred,name+'_correct':ans['correct'],
                        name+'_gold_hits':len(set(ans['top_memory_ids'])&set(g['gold_ids'])),name+'_top5':ans['top_memory_ids']})
        per_question.append(row)
    assert sum(r['gold_count'] for r in per_question)==1041

    def metrics(rows,name):
        correct=sum(r[name+'_correct'] for r in rows);hits=sum(r[name+'_gold_hits'] for r in rows)
        total=sum(r['gold_count'] for r in rows)
        return {'correct':correct,'questions':len(rows),'accuracy':correct/len(rows),
                'gold_hits_top5':hits,'gold_references':total,'recall_at_5_micro':hits/total,
                'recall_at_5_macro':sum(r[name+'_gold_hits']/r['gold_count'] for r in rows)/len(rows),
                'questions_with_any_gold':sum(r[name+'_gold_hits']>0 for r in rows)}
    overall={n:metrics(per_question,n) for n in CONDITIONS}
    by_domain={domain:{n:metrics([r for r in per_question if r['domain']==domain],n) for n in CONDITIONS}
               for domain in sorted({r['domain'] for r in per_question})}
    before=indexed(previous,'request_sha256')
    repeat_changes=[r['question_id'] for r in requests if r['condition']==BASELINE and
                    choice(before[r['request_sha256']]['raw_response'])!=logical[r['record_id']]['predicted_choice']]
    wins=sum(not r[BASELINE+'_correct'] and r[WEIGHTED+'_correct'] for r in per_question)
    losses=sum(r[BASELINE+'_correct'] and not r[WEIGHTED+'_correct'] for r in per_question)
    annotations=copy.deepcopy(c['method_annotations'])
    for n in CONDITIONS:annotations[n]['5. 回答方式'].update(overall[n])
    summary={'status':'complete','model':MODEL,'caption_model':MODEL,'provider':c['provider'],
             'lambda':RIDGE,'alpha':ALPHA,'caption_source':str(source),'captions_regenerated':False,
             'memory_leaves':438,'masters':85,'overall':overall,'by_domain':by_domain,
             'comparison':{'wrong_to_right':wins,'right_to_wrong':losses,'net_correct':wins-losses},
             'historical_baseline_correct':read(source/'summary.json')['overall']['baseline_correct'],
             'baseline_changed_answers_vs_historical':repeat_changes,'fresh_requests':len(fresh),'logical_records':742,
             'same_run_shared_records':742-len(fresh),'failed_attempts':sum(a['status']!='ok' for a in attempts),
             'format_errors':0,'refusals':0,'length_stops':0,'historical_answers_reused':False,
             'method_annotations':annotations,'generation_parameters':{k:c[k] for k in ('temperature','seed','enable_thinking','reader_max_output_tokens')},
             'usage':{k:sum((r.get('usage') or {}).get(k,0) for r in fresh_rows) for k in ('prompt_tokens','completion_tokens','total_tokens')},
             'postprocessing':{'script':str(Path(__file__).resolve()),'script_sha256':sha(__file__),
                 'reason':'The signed runner tried reading task from evaluation labels; labels intentionally contain no domain.',
                 'domain_source':str(source/'inputs/source_questions.jsonl'),
                 'signed_runner_modified':False,'requests_modified':False,'responses_modified':False,'extra_api_calls':0},
             'limitations':['The provider alias does not establish a fixed weight revision.',
                 'Only the requested baseline and master+zscore conditions were freshly answered; historical unweighted master is not a new matched control.']}
    dump(output/'summary.json',summary);dump(output/'paired_results.json',per_question)
    with (output/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(per_question[0]));w.writeheader();w.writerows(per_question)
    table=[{'condition':name,**r} for name,r in overall.items()]
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    audit={'passed':True,'questions':371,'gold_references':1041,'records':742,'fresh_requests':len(fresh),
           'same_run_shared_records':742-len(fresh),'full_request_hashes_reconstructed':742,
           'historical_answer_reuse':False,'format_errors':0,'refusals':0,'length_stops':0,
           'postprocessor_sha256':sha(__file__),
           'artifact_sha256':{n:sha(output/n) for n in ('reader/protocol.json','reader/requests.jsonl','reader/gold.jsonl',
               'reader/fresh_responses.jsonl','reader/responses.jsonl','reader/api_attempts.jsonl','summary.json','paired_results.json',
               'retrieval_protocol.json','retrieval_results.jsonl','source_validation.json','independent_input_audit.json')}}
    dump(output/'independent_audit.json',audit)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek Flash MemEye</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
          '<h1>DeepSeek V4.1 Flash：baseline 与 master + zscore</h1>',
          '<p>DashScope International，deepseek-v4.1-flash。复用已核验的 437 张去重图片 caption、438 条记忆、85 个 master。全部 371 题重新回答；未加权 baseline 对比 master_answer_query + zscore（λ=0.03，α=1）。</p>',
          '<table><tr><th>方法</th><th>QA</th><th>Recall@5 micro</th><th>Recall@5 macro</th></tr>']
    for name,r in overall.items():
        body.append(f'<tr><td>{name}</td><td>{r["correct"]}/371 ({r["accuracy"]:.4%})</td><td>{r["gold_hits_top5"]}/1041 ({r["recall_at_5_micro"]:.4%})</td><td>{r["recall_at_5_macro"]:.4%}</td></tr>')
    body+=['</table>',f'<p>本轮 {len(fresh)} 次新推理，{742-len(fresh)} 条同请求共享。历史 baseline 为 {summary["historical_baseline_correct"]}/371；相同请求跨轮有 {len(repeat_changes)} 题答案改变。历史未加权 master 不作为本轮同批对照。</p>',
           '<p>temperature=0，enable_thinking=false，回答预算 16 tokens；seed 沿用已有 DeepSeek 协议，不发送。接口为无日期别名，实际权重版本未由服务返回确认。</p>',
           '<p><a href="metrics.csv">结果 CSV</a> · <a href="summary.json">统计和五项标注</a> · <a href="paired_results.csv">逐题 CSV</a> · <a href="external_audit.json">独立核验</a> · <a href="source_validation.json">Caption 来源核验</a></p>',
           '<p>所有调用完成后，仅修正本地汇总中 domain 字段的读取位置；原始请求、回答及已签名执行代码保持不变，没有额外 API 调用。</p>',
           '<pre>'+html.escape(json.dumps(annotations,ensure_ascii=False,indent=2))+'</pre></html>']
    (output/'report.html').write_text('\n'.join(body))
    dump(output/'execution_status.json',{'status':'complete','fresh_completed':len(fresh),'unique_requests':len(fresh),'logical_records':742})
    print(json.dumps({'status':'complete','overall':overall,'historical_baseline_answer_changes':len(repeat_changes),'fresh_requests':len(fresh)}),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    summarize(parser.parse_args().output)
