"""Audit and replay original/new batches for unweighted MemGallery control drift."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from scripts.evaluation import mem_gallery_master_reader as reader

OLD=ROOT/'results/mem_gallery_master_answer_20261021'
NEW=ROOT/'results/memgallery_qwen_positive_variance_lambda_ablation_20261107'
OUTPUT=NEW/'control_drift_diagnosis'
METHODS=('baseline_question_only','master_answer_query')


def read(path):return json.loads(Path(path).read_text())


def inspect_run(root):
    p=read(root/'reader/protocol.json');contract=p['contract']
    assert reader.digest(contract)==p['experiment_signature']
    for path,expected in contract['source_sha256'].items():assert reader.sha(path)==expected,path
    manifest=reader.read_lines(root/'reader/qa_manifest.jsonl')
    assert reader.digest(manifest)==contract['manifest_sha256']
    responses=reader.read_lines(root/'reader/responses.jsonl')
    mmap=reader.by_id(manifest,'record_id');rmap=reader.by_id(responses,'record_id')
    assert set(mmap)==set(rmap)
    assert len({r['execution_id'] for r in responses})==1,'Batch reconstruction assumes one uninterrupted execution'
    unique={}
    for m in manifest:unique.setdefault(m['payload_sha256'],m)
    batches=[];payload_batch={};planned=[]
    for has_image in (False,True):
        selected=[m for m in unique.values() if bool(m['query_image_paths'])==has_image]
        size=contract['batch_size_images' if has_image else 'batch_size_text']
        for start in range(0,len(selected),size):
            items=selected[start:start+size];index=len(batches)
            token_width=max(rmap[m['record_id']]['input_tokens'] for m in items)
            record={'batch_index':index,'has_image':has_image,'size':len(items),
                'max_input_tokens':token_width,'payloads':[m['payload_sha256'] for m in items],
                'record_ids':[m['record_id'] for m in items]}
            batches.append(record)
            for position,m in enumerate(items):
                r=rmap[m['record_id']]
                assert r['fresh_inference_executed_for_this_record'] and r['batch_size']==len(items)
                payload_batch[m['payload_sha256']]={'batch_index':index,'position':position}
                planned.append(m['record_id'])
    assert planned==[r['record_id'] for r in responses if r['fresh_inference_executed_for_this_record']]
    return {'protocol':p,'manifest':mmap,'responses':rmap,'batches':batches,'payload_batch':payload_batch}


def audit():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    old,new=inspect_run(OLD),inspect_run(NEW)
    previous=read(OLD/'reader/paired_results.json');current=read(NEW/'reader/paired_results.json')
    pa,pb=reader.by_id(previous),reader.by_id(current)
    fields=['model','system_prompt','task_instructions','official_prompt_revision','seed','do_sample',
        'max_new_tokens','batch_size_text','batch_size_images','dtype','query_image_max_pixels','input_truncation']
    for key in fields:assert old['protocol']['contract'][key]==new['protocol']['contract'][key],key
    cases=[];changes=[];summary={};all_same_payloads=0
    for method in METHODS:
        counts=Counter()
        for qid in pa:
            rid=qid+'::'+method;a,b=old['responses'][rid],new['responses'][rid]
            ma,mb=old['manifest'][rid],new['manifest'][rid]
            for field in ['payload_sha256','reader_text','query_images','top5_memory_ids']:
                assert ma[field]==mb[field],(rid,field)
            all_same_payloads+=1
            aem=reader.score_answer(a['raw_response'],pa[qid]['answer'])['em']
            bem=reader.score_answer(b['raw_response'],pa[qid]['answer'])['em']
            assert aem==pa[qid][method]['em'] and bem==pb[qid][method]['em']
            assert pa[qid]['answer']==pb[qid]['answer']
            counts['old_correct']+=int(aem);counts['new_correct']+=int(bem)
            counts['wrong_to_right']+=bem>aem;counts['right_to_wrong']+=bem<aem
            oa=old['payload_batch'][ma['payload_sha256']];nb=new['payload_batch'][mb['payload_sha256']]
            old_batch=old['batches'][oa['batch_index']];new_batch=new['batches'][nb['batch_index']]
            composition_changed=old_batch['payloads']!=new_batch['payloads']
            counts['changed_batch_composition']+=composition_changed
            counts['changed_batch_position']+=oa['position']!=nb['position']
            counts['changed_padded_input_width']+=old_batch['max_input_tokens']!=new_batch['max_input_tokens']
            if a['raw_response']!=b['raw_response']:
                counts['changed_answer_text']+=1
                record={'record_id':rid,'method':method,'question_id':qid,'gold':pa[qid]['answer'],
                    'old_answer':a['raw_response'],'new_answer':b['raw_response'],'old_em':int(aem),'new_em':int(bem),
                    'same_payload':True,'batch_composition_changed':composition_changed,
                    'old_batch':old_batch['batch_index'],'new_batch':new_batch['batch_index'],
                    'old_position':oa['position'],'new_position':nb['position'],
                    'old_padded_tokens':old_batch['max_input_tokens'],'new_padded_tokens':new_batch['max_input_tokens'],
                    'query_has_image':bool(ma['query_image_paths']),'payload_sha256':ma['payload_sha256']}
                changes.append(record)
                if aem!=bem:cases.append(record)
        summary[method]=dict(counts)
    assert all_same_payloads==3422 and len(cases)==8
    sources=[Path(__file__),Path(reader.__file__)]
    sources += [root/'reader'/name for root in [OLD,NEW] for name in ['protocol.json','qa_manifest.jsonl','responses.jsonl','paired_results.json']]
    result={'status':'audited','settings_equal':fields,'controls_have_no_positive_variance':True,
        'identical_control_payloads':all_same_payloads,'summary':summary,'em_changed_cases':cases,
        'all_changed_answers':changes,'source_sha256':{str(p):reader.sha(p) for p in sources}}
    reader.dump(OUTPUT/'audit.json',result)
    for filename,rows in [('em_changes.csv',cases),('answer_changes.csv',changes)]:
        with (OUTPUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for label,data in [('old',old),('new',new)]:reader.dump(OUTPUT/(label+'_batch_plan.json'),data['batches'])
    print(json.dumps({'stage':'audit','identical_payloads':all_same_payloads,'summary':summary,'em_cases':len(cases)}),flush=True)
    return result,old,new


def replay(device):
    result,old,new=audit()
    cases=result['em_changed_cases'];runs={'old':old,'new':new}
    required={label:sorted({c[label+'_batch'] for c in cases}) for label in runs}
    plan=[]
    for label,indices in required.items():
        for index in indices:plan.append((label,index,1))
    # Repeat one old batch to distinguish the observed batch effect from a
    # merely unrepeated regeneration difference.
    plan.append(('old',required['old'][0],2))
    reader.dump(OUTPUT/'replay_plan.json',{'device':device,'batches':plan,'seed':42,
        'do_sample':False,'model':old['protocol']['contract']['model'],'no_gold_in_generation':True})
    runtime=reader.load_runtime(device);reader.dump(OUTPUT/'replay_environment.json',runtime[2])
    replayed=[];target_records=[]
    for label,index,repeat in plan:
        data=runs[label];batch=data['batches'][index]
        items=[data['manifest'][rid] for rid in batch['record_ids']]
        payloads=[{'system':data['protocol']['contract']['system_prompt'],'user_text':m['reader_text'],
                   'query_image_paths':m['query_image_paths']} for m in items]
        outputs=reader.generate_batch(runtime,payloads,256)
        for m,answer in zip(items,outputs):
            record={'source_run':label,'batch_index':index,'repeat':repeat,'record_id':m['record_id'],
                'payload_sha256':m['payload_sha256'],**answer,
                'matches_saved_answer':answer['raw_response']==data['responses'][m['record_id']]['raw_response']}
            replayed.append(record)
            for case in cases:
                if case['payload_sha256']==m['payload_sha256']:
                    target_records.append({'case_record_id':case['record_id'],**record,
                        'saved_answer':case[label+'_answer'],'em':reader.score_answer(answer['raw_response'],case['gold'])['em'],
                        'saved_em':case[label+'_em']})
        reader.dump(OUTPUT/'replay_responses.json',replayed);reader.dump(OUTPUT/'replay_targets.json',target_records)
        print(json.dumps({'stage':'batch_replayed','source':label,'batch':index,'repeat':repeat,
            'size':len(items),'matches_saved':sum(r['matches_saved_answer'] for r in replayed[-len(items):])}),flush=True)
    for path,expected in result['source_sha256'].items():assert reader.sha(path)==expected,path
    by_case={case['record_id']:{label:next(r for r in target_records if r['case_record_id']==case['record_id'] and r['source_run']==label and r['repeat']==1)
             for label in ['old','new']} for case in cases}
    report={'status':'complete','batches_replayed':len(plan),'requests_replayed':len(replayed),
        'all_replayed_answers_match_saved':all(r['matches_saved_answer'] for r in replayed),
        'all_em_changed_cases_reproduce_both_original_answers':all(r[label]['raw_response']==r[label]['saved_answer'] for r in by_case.values() for label in ['old','new']),
        'all_em_changed_cases_reproduce_both_em_values':all(r[label]['em']==r[label]['saved_em'] for r in by_case.values() for label in ['old','new']),
        'repeated_old_batch_matches_first':all(r['raw_response']==next(x['raw_response'] for x in replayed if x['source_run']=='old' and x['batch_index']==r['batch_index'] and x['repeat']==1 and x['record_id']==r['record_id']) for r in replayed if r['repeat']==2),
        'cases':by_case,'source_sha256':result['source_sha256'],
        'interpretation':'Replay holds model, seed, prompt, generation function and device fixed and switches only the reconstructed batch surrounding the target request. This tests batch-dependent output, not the exact floating-point kernel responsible.'}
    reader.dump(OUTPUT/'replay_summary.json',report)
    files=[p for p in OUTPUT.iterdir() if p.suffix in ['.json','.csv'] and p.name!='completion_manifest.json']
    reader.dump(OUTPUT/'completion_manifest.json',{'status':'complete','artifact_sha256':{str(p):reader.sha(p) for p in files}})
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','source_sha256']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','replay'],default='replay');parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.stage=='audit':audit()
    else:replay(args.device)
