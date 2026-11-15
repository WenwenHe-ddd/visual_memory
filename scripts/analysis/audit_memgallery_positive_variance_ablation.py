"""Independent score, payload, provenance and EM/F1/recall audit for Gallery PV."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.vendor'))
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):result.update(chunk)
    return result.hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check(hashes):
    for path,expected in hashes.items():assert sha(path)==expected,path
def normalize(value):
    value=re.sub(r'(?<=\d)\.(?=\d)','DOTPLACEHOLDER',str(value).lower())
    value=value.replace('_','UNDERSCOREPLACEHOLDER')
    value=re.sub(r'\b(a|an|the|and)\b',' ',value)
    value=''.join(c if c not in string.punctuation else ' ' for c in value)
    return ' '.join(value.replace('DOTPLACEHOLDER','.').replace('UNDERSCOREPLACEHOLDER','_').split())


def audit_retrieval(base):
    p=read(base/'retrieval_protocol.json');source=Path(p['source_run'])
    check(p['source_sha256']);check(p['artifact_sha256'])
    specs=p['condition_specs'];assert len(specs)==12
    assert p['lambdas']==[.001,.003,.01,.03,.1] and p['alpha']==1.
    questions={q['id']:q for q in lines(base/'inputs/questions.jsonl')}
    memories={m['id']:m for m in lines(base/'inputs/memories.jsonl')}
    candidates={q['id']:q['candidate_statements'] for q in lines(source/'candidates.jsonl')}
    previous={q['id']:q for q in lines(source/'retrieval_results.jsonl')}
    rows=lines(base/'retrieval_results.jsonl');ranked={q['id']:q for q in rows}
    assert len(questions)==len(rows)==len(ranked)==1711 and set(questions)==set(ranked)
    assert len(memories)==3962
    score_checks=ranking_checks=raw_checks=0
    max_z_error=max_correction_error=0.
    covered_questions=set()
    for path in sorted((base/'diagnostics').glob('*.jsonl')):
        graph=read(source/'offline/memory'/path.stem/'graph.json')
        nodes={n['id']:n for n in graph['image_nodes']}
        for detail in lines(path):
            qid=detail['id'];assert qid not in covered_questions;covered_questions.add(qid)
            q=questions[qid];ids=detail['memory_ids'];assert detail['task']==q['task']==path.stem
            assert len(ids)==len(set(ids)) and all(memories[mid]['task']==q['task'] for mid in ids)
            assert set(ids)=={mid for mid,m in memories.items() if m['task']==q['task']}
            assert not {'answer','gold_ids','clue','gold_choice'}&set(q)
            text_indices=[i for i,mid in enumerate(ids) if not memories[mid]['is_visual']]
            assert detail['text_only_indices']==text_indices
            choices={r['master_id']:r['candidate_index'] for r in previous[qid]['master_choices']}
            assert ranked[qid]['master_choices']==previous[qid]['master_choices']
            statements=candidates[qid]
            if isinstance(statements,dict):statements=[statements[k] for k in 'ABCD']
            actual={r['name']:r for r in ranked[qid]['conditions']}
            for method,data in detail['methods'].items():
                baseline=method=='baseline_question_only'
                raw=np.array(data['raw']);assert len(raw)==len(ids) and np.isfinite(raw).all()
                covered=[i for group in data['groups'] for i in group['indices']]
                assert len(covered)==len(set(covered))
                assert sorted(covered)==(list(range(len(ids))) if baseline else [i for i in range(len(ids)) if i not in text_indices])
                assert len(data['groups'])==(1 if baseline else len(graph['masters']))
                for group in data['groups']:
                    ix=group['indices']
                    if baseline:expected_query=q['retrieval_query'];assert group['query_index']==0
                    else:
                        choice=choices[group['id']]
                        assert group['query_index']==1+choice
                        expected_query=q['retrieval_query']+'\nCandidate answer: '+statements[choice]
                        assert all(nodes[memories[ids[i]]['node_id']]['master_id']==group['id'] for i in ix)
                    assert group['query']==expected_query
                    raw_std=float(np.std(raw[ix]));assert abs(raw_std-group['raw_std'])<1e-12
                    for ridge,stats in data['weighting'].items():
                        enhanced=np.array(stats['enhanced'])[ix];z=np.array(stats['z'])[ix]
                        correction=np.array(stats['correction'])[ix];weighted=np.array(stats['weighted'])[ix]
                        std=float(np.std(enhanced));tol=32*np.finfo(float).eps*max(1.,max(abs(enhanced)))
                        expected_z=enhanced.copy() if len(ix)==1 else (np.zeros(len(ix)) if std<=tol else (enhanced-enhanced.mean())/std)
                        error=float(max(abs(z-expected_z)));assert error<1e-10,error
                        max_z_error=max(max_z_error,error)
                        expected=np.zeros(len(ix)) if len(ix)==1 or group['raw_status']=='zero_variance' else raw_std*np.tanh(expected_z)
                        error=float(max(abs(correction-expected)));assert error<1e-12,error
                        max_correction_error=max(max_correction_error,error)
                        assert max(abs(correction))<=raw_std+1e-12
                        assert np.allclose(weighted,raw[ix]+correction,rtol=0,atol=1e-12)
                        score_checks+=1
                if not baseline:
                    for stats in data['weighting'].values():
                        assert np.array_equal(np.array(stats['weighted'])[text_indices],raw[text_indices])
                        assert np.array_equal(np.array(stats['correction'])[text_indices],np.zeros(len(text_indices)))
                for name,spec in specs.items():
                    if spec['method']!=method:continue
                    values=raw if spec['lambda'] is None else np.array(data['weighting'][str(spec['lambda'])]['weighted'])
                    assert np.isfinite(values).all()
                    indices=sorted(range(len(ids)),key=lambda i:(-values[i],i))[:5]
                    top=[ids[i] for i in indices]
                    assert actual[name]['top5_memory_ids']==top
                    assert np.allclose(actual[name]['top5_scores'],values[indices],rtol=0,atol=1e-12)
                    ranking_checks+=1
                    if spec['lambda'] is None:
                        old=next(c for c in previous[qid]['conditions'] if c['name']==method)
                        assert old['top5_memory_ids']==top;raw_checks+=1
    assert len(covered_questions)==1711 and ranking_checks==20532 and raw_checks==3422
    assert score_checks==p['independent_covariance_checks']
    result={'passed':True,'questions':1711,'conditions':12,'verified_rankings':ranking_checks,
        'raw_top5_match_original':raw_checks,'group_lambda_checks':score_checks,'max_z_error':max_z_error,
        'max_correction_error':max_correction_error,'text_only_fallback_unchanged':True,
        'original_master_membership_and_choices_verified':True,'gold_opened':False}
    (base/'retrieval_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def audit_final(base):
    from nltk.stem import PorterStemmer
    retrieval_audit=audit_retrieval(base)
    completion=read(base/'completion_manifest.json');check(completion['artifact_sha256'])
    p=read(base/'reader/protocol.json');contract=p['contract'];assert digest(contract)==p['experiment_signature']
    check(contract['source_sha256'])
    assert contract['seed']==42 and contract['do_sample'] is False and not contract['historical_response_reuse']
    assert not contract['candidate_expansion_in_reader'] and not contract['gold_loaded_before_analysis']
    assert len(contract['condition_specs'])==12
    source=Path(read(base/'retrieval_protocol.json')['source_run'])
    old_contract=read(source/'reader/protocol.json')['contract']
    for key in ['model','system_prompt','task_instructions','seed','do_sample','max_new_tokens','batch_size_text','batch_size_images',
                'dtype','query_image_max_pixels','input_truncation']:
        assert contract[key]==old_contract[key],key
    for name,meta in contract['model']['files'].items():
        file=Path(contract['model']['path'])/name
        assert file.stat().st_size==meta['bytes']
    manifest=lines(base/'reader/qa_manifest.jsonl');assert digest(manifest)==contract['manifest_sha256']
    responses=lines(base/'reader/responses.jsonl')
    mindex={m['record_id']:m for m in manifest};rindex={r['record_id']:r for r in responses}
    assert len(manifest)==len(responses)==len(mindex)==len(rindex)==20532 and set(mindex)==set(rindex)
    questions={q['id']:q for q in lines(base/'inputs/questions.jsonl')}
    memories={m['id']:m for m in lines(base/'inputs/memories.jsonl')}
    targets={q['id']:q for q in lines(base/'inputs/gold.jsonl')}
    ranked={q['id']:q for q in lines(base/'retrieval_results.jsonl')}
    pairs={q['id']:q for q in read(base/'reader/paired_results.json')}
    summary=read(base/'summary.json');assert len(pairs)==len(targets)==1711
    images={};fresh={};counts=Counter();correct=Counter();hits=Counter();totals=Counter();f1s=Counter()
    stemmer=PorterStemmer()
    for rid,m in mindex.items():
        q=questions[m['id']];gold=targets[m['id']];answer=rindex[rid];condition=m['condition']
        ranking=next(c for c in ranked[m['id']]['conditions'] if c['name']==condition)
        assert m['top5_memory_ids']==answer['top5_memory_ids']==ranking['top5_memory_ids']
        text='The retrieved memory contents are as follows:\n\n'
        text+='\n\n'.join(f'Memory {i}: {memories[mid]["reader_text"]}' for i,mid in enumerate(m['top5_memory_ids'],1))
        text+='\n\nPlease answer the question about the conversation using the retrieved memories. Return only the answer content, concisely but completely. For dates or times, provide a specific date or time whenever possible.\n\nQuestion:\n'+q['question']
        if instruction:=contract['task_instructions'].get(q['point'].upper()):text+='\n\n'+instruction
        assert text==m['reader_text']
        assert m['query_image_paths']==q['query_image_paths']
        expected_images=[]
        for path in q['query_image_paths']:
            if path not in images:images[path]={'path':path,'sha256':sha(path)}
            expected_images.append(images[path])
        assert m['query_images']==expected_images
        payload={'system':contract['system_prompt'],'user_text':text,'query_images':expected_images}
        assert digest(payload)==m['payload_sha256']==answer['payload_sha256']
        assert answer['experiment_signature']==p['experiment_signature'] and answer['run_id']==p['run_id']
        assert not {'gold_ids','answer','gold_choice'}&set(m)
        if answer['fresh_inference_executed_for_this_record']:
            assert answer['payload_sha256'] not in fresh;fresh[answer['payload_sha256']]=answer
        else:
            origin=rindex[answer['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            for key in ['payload_sha256','raw_response','execution_id','generated_tokens','input_tokens','hit_token_limit']:
                assert answer[key]==origin[key]
        predicted,expected=normalize(answer['raw_response']),normalize(gold['answer'])
        pt,gt=predicted.split(),expected.split()
        common=sum((Counter(stemmer.stem(t) for t in pt)&Counter(stemmer.stem(t) for t in gt)).values())
        f1=2*common/(len(pt)+len(gt)) if common else 0.
        scored=pairs[m['id']][condition]
        assert scored['em']==float(predicted==expected) and abs(scored['f1']-f1)<1e-15
        hit=len(set(gold['gold_ids'])&set(m['top5_memory_ids']));total=len(set(gold['gold_ids']))
        assert scored['gold_hits']==hit and scored['gold_clues']==total
        counts[condition]+=1;correct[condition]+=predicted==expected;hits[condition]+=hit;totals[condition]+=total;f1s[condition]+=f1
    assert len(fresh)==contract['unique_payloads']
    for name,metrics in summary['overall'].items():
        assert counts[name]==metrics['questions']==1711
        assert totals[name]==metrics['gold_links']==4109
        assert correct[name]==metrics['correct'] and hits[name]==metrics['gold_hits']
        assert correct[name]/1711==metrics['em'] and hits[name]/4109==metrics['recall_at_5_micro']
        assert abs(f1s[name]/1711-metrics['f1'])<1e-14
    result={'passed':True,'retrieval':retrieval_audit,'logical_records':20532,'conditions':12,
        'questions_per_condition':1711,'fresh_inferences':len(fresh),'shared_records':20532-len(fresh),
        'reconstructed_payloads':20532,'query_images_verified':len(images),
        'empty_responses':sum(not r['raw_response'] for r in responses),
        'token_limit_records':sum(r['hit_token_limit'] for r in responses),
        'independent_em_f1_and_recall_verified':True,'source_and_artifact_hashes_verified':True,
        'audit_script_sha256':sha(__file__),'completion_manifest_sha256':sha(base/'completion_manifest.json')}
    (base/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/memgallery_qwen_positive_variance_lambda_ablation_20261107')
    parser.add_argument('--stage',choices=['retrieval','final'],default='final')
    args=parser.parse_args()
    print(json.dumps(audit_retrieval(args.output) if args.stage=='retrieval' else audit_final(args.output),indent=2))
