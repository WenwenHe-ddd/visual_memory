"""Offline raw-answer recount and input-control comparison for GPT-5 lambda ablations."""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.analysis.audit_memeye_gpt5_base import read,lines,sha,digest,indexed,choice,verify_hashes,comparison
SOURCE=ROOT/'results/memeye_gpt5_base_seed42_full_20261024'
SMALL=ROOT/'results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024'
DEFAULT=ROOT/'results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024'
NAMES=('baseline_question_only_zscore','master_answer_query_zscore')

def compare(output=DEFAULT):
    output=Path(output).resolve();roots={'0.001':SMALL,'0.01':SOURCE,'0.1':output};data={};hashes={}
    reference=read(SOURCE/'reader/protocol.json')['contract']
    gold=indexed(lines(SOURCE/'reader/inputs/gold.jsonl'),'id');assert len(gold)==371
    expected_ids={qid+'::'+n for qid in gold for n in NAMES}
    for label,path in roots.items():
        protocol=read(path/'reader/protocol.json');c=protocol['contract'];audit=read(path/'independent_audit.json')
        assert digest(c)==protocol['experiment_signature'] and audit['passed'] and (c['lambda'],c['alpha'])==(float(label),1.)
        verify_hashes(c['source_sha256']);verify_hashes({str(path/n):v for n,v in audit['artifact_sha256'].items()})
        assert indexed(lines(path/'reader/inputs/gold.jsonl'),'id')==gold
        for k in ('model','requested_model','expected_returned_model','caption_model','seed','caption_seed','temperature','temperature_wire_policy','reasoning_effort',
                  'max_completion_tokens','image_detail','base_url','api_config','reader_system_prompt','memory_path','top_k','workers'):
            assert c[k]==reference[k],(label,k)
        ret_dir=path/'zscore' if label=='0.01' else path
        r=read(ret_dir/'retrieval_protocol.json');verify_hashes(r['source_sha256']);verify_hashes(r['output_sha256'])
        assert r['memory_path']==str(SOURCE) and r['caption_model']==reference['caption_model'] and r['lambda']==float(label)
        selected=lambda file:indexed([row for row in lines(path/file) if row.get('condition',row.get('record_id','').split('::')[-1]) in NAMES],'record_id')
        requests=selected('reader/inputs/requests.jsonl');answers=selected('reader/responses.jsonl');manifest=selected('reader/api_request_manifest.jsonl')
        assert set(requests)==set(answers)==set(manifest)==expected_ids
        retrieval=indexed(lines(ret_dir/'retrieval_results.jsonl'),'question_id');assert set(retrieval)==set(gold)
        for rid,answer in answers.items():
            assert answer['returned_model']==reference['expected_returned_model'] and answer['requested_model']==reference['model']
            assert answer['top_memory_ids']==requests[rid]['top_memory_ids'] and answer['request_sha256']==manifest[rid]['request_sha256']
        data[label]={'contract':c,'requests':requests,'answers':answers,'manifest':manifest,'retrieval':retrieval,'summary':read(path/'summary.json')}
        for n in ('reader/protocol.json','reader/inputs/requests.jsonl','reader/inputs/gold.jsonl','reader/responses.jsonl','reader/api_request_manifest.jsonl','summary.json','independent_audit.json'):
            hashes[str(path/n)]=sha(path/n)
        for n in ('retrieval_protocol.json','retrieval_results.jsonl'):hashes[str(ret_dir/n)]=sha(ret_dir/n)
    controls={}
    for label in ('0.001','0.01'):
        groups=leaves=0
        for qid in gold:
            ca=indexed(data[label]['retrieval'][qid]['conditions'],'name');cb=indexed(data['0.1']['retrieval'][qid]['conditions'],'name')
            for n in NAMES:
                a,b=ca[n],cb[n];assert len(a['groups'])==len(b['groups']) and len(a['leaves'])==len(b['leaves'])
                for x,y in zip(a['groups'],b['groups']):
                    for key in ('group_id','size','query','selected_choice','selected_statement','raw_std','raw_status'):assert x[key]==y[key],(label,qid,n,key)
                    groups+=1
                for x,y in zip(a['leaves'],b['leaves']):
                    for key in ('memory_id','leaf_index','group_id','raw'):assert x[key]==y[key],(label,qid,n,key)
                    leaves+=1
        controls[label]={'passed':True,'raw_scores_exactly_equal':leaves,'queries_and_groups_equal':groups,'same_caption_memory_and_reader_settings':True}
    pairs=[]
    for qid,g in gold.items():
        row={'question_id':qid,'domain':g['task'],'gold_choice':g['gold_choice']}
        for n in NAMES:
            for label,d in data.items():
                tag=n+'_lambda'+label.replace('.','p');rid=qid+'::'+n;pred=choice(d['answers'][rid]['raw_response'])
                row.update({tag+'_answer':pred,tag+'_correct':pred==g['gold_choice'],tag+'_top5':d['requests'][rid]['top_memory_ids'],tag+'_request_sha256':d['manifest'][rid]['request_sha256']})
        pairs.append(row)
    results={};metrics={}
    for label,d in data.items():
        results[label]={}
        for n in NAMES:
            tag=n+'_lambda'+label.replace('.','p');correct=sum(r[tag+'_correct'] for r in pairs)
            results[label][n]={'correct':correct,'questions':371,'accuracy':correct/371}
            assert correct==d['summary']['overall'][n]['correct']
    assert [results['0.001'][n]['correct'] for n in NAMES]==[214,217]
    assert [results['0.01'][n]['correct'] for n in NAMES]==[221,221]
    for label in ('0.001','0.01'):
        metrics[label]={}
        for n in NAMES:
            old=n+'_lambda'+label.replace('.','p');new=n+'_lambda0p1';m=comparison(pairs,old,new)
            same=[r for r in pairs if r[old+'_request_sha256']==r[new+'_request_sha256']]
            m.update(identical_request_questions=len(same),changed_request_questions=371-len(same),
                identical_request_answer_disagreements=[r['question_id'] for r in same if r[old+'_answer']!=r[new+'_answer']],
                identical_request_net_correct=sum(r[new+'_correct']-r[old+'_correct'] for r in same))
            m['changed_request_net_correct']=m['net_correct']-m['identical_request_net_correct'];metrics[label][n]=m
            if label=='0.01':assert m==data['0.1']['summary']['lambda_comparisons'][n]
    result={'status':'complete','model':reference['expected_returned_model'],'seed':42,'alpha':1.,'current_lambda':.1,'questions':371,
        'memory_path':str(SOURCE),'results':results,'comparisons':metrics,'controls':controls,
        'interpretation':'All three settings share frozen GPT-5 captions and memory. Each setting was answered in a separate API run. Identical-request answer changes are reported separately; single-run differences are not wholly attributable to lambda.'}
    (output/'lambda_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (output/'lambda_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT-5 三个 λ 对比</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>GPT-5：固定 α=1，λ=0.001 / 0.01 / 0.1</h1><p>全部 371 题；同一批 caption、记忆、master、查询与原始分数，回答 seed=42、minimal 推理、不传 temperature、2048 token 总预算。λ=0.1 为本轮新回答，其余为已核验历史结果。</p>',
        '<table><tr><th>方案</th><th>λ=0.001</th><th>λ=0.01</th><th>λ=0.1（本轮）</th></tr>']
    for n in NAMES:body.append('<tr><td>'+n+'</td>'+''.join(f'<td>{results[label][n]["correct"]}/371（{results[label][n]["accuracy"]:.4%}）</td>' for label in roots)+'</tr>')
    body+=['</table><p>下方逐题统计单列了相同完整请求下的答案波动；单轮分差不能全部归因于 λ。</p><pre>'+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>',
        '<p><a href="report.html">本轮详情与五项标注</a> · <a href="lambda_comparison.csv">三组逐题 CSV</a> · <a href="lambda_comparison.json">对比统计</a> · <a href="lambda_comparison_audit.json">对比核验</a></p></html>']
    (output/'lambda_comparison.html').write_text('\n'.join(body))
    hashes[str(Path(__file__).resolve())]=sha(__file__);verify_hashes(hashes)
    proof={'passed':True,'all_371_ids_and_targets_equal':True,'independent_raw_answer_recount':True,'controls':controls,'source_sha256':hashes,
        'artifact_sha256':{n:sha(output/n) for n in ('lambda_comparison.json','lambda_comparison.csv','lambda_comparison.html')},'results':results}
    (output/'lambda_comparison_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stage':'three_lambda_comparison','passed':True,'results':results,'comparisons':metrics}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    a=p.parse_args();compare(a.output)
