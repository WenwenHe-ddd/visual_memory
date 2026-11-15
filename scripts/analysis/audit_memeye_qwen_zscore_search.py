"""Independent score, payload, fresh-answer and shared-parameter selection audit."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from scripts.analysis.audit_master_answer_query_zscore import read, rows, sha, digest
BASE = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
REF = ROOT / 'reports/all_master_answer_query_20261021'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
METHODS = ('baseline_question_only', 'master_answer_query')


def audit(base=BASE, inputs_only=False):
    base = Path(base).resolve(); out = base / 'reader'
    p = read(out / 'protocol.json'); signed = dict(p); sig = signed.pop('experiment_signature')
    assert sig == digest(signed)
    rp = read(base / 'protocol.json')
    for hashes in (p['source_sha256'], p['model_source_sha256'], rp['output_sha256']):
        for path, expected in hashes.items(): assert sha(path) == expected, path
    specs = p['condition_specs']; settings = p['reader_settings']
    assert specs == rp['condition_specs'] and p['selection_rule'] == rp['selection_rule']
    assert len(specs) == 52 and rp['lambdas'] == [.001,.003,.01,.03,.1] and rp['alphas'] == [.1,.3,.6,1.,2.]
    assert settings == read(REF / 'reader/protocol.json')['reader_settings']
    original = {q['id']: q for q in rows(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in rows(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl')}
    ret = {r['question_id']: r for r in rows(base / 'retrieval_results.jsonl')}
    manifest_list = rows(out / 'qa_manifest.jsonl'); manifest = {r['record_id']: r for r in manifest_list}
    old = {r['record_id']: r for r in rows(REF / 'reader/qa_manifest.jsonl')}
    historical = {r['question_id']: r for r in rows(REF / 'retrieval_results.jsonl')}
    assert len(ret) == len(original) == 371 and set(ret) == set(original)
    assert len(manifest_list) == len(manifest) == 19292
    assert Counter(r['condition'] for r in manifest_list) == dict.fromkeys(specs, 371)
    frozen_hashes = read(REF / 'protocol.json')['source_sha256']
    feature_hashes = {}; max_error = 0.; group_checks = 0
    for domain in sorted({q['task'] for q in original.values()}):
        src = read(INPUTS / domain / 'inputs.json')
        f = INPUTS / domain / 'features.npz'; feature_hashes[str(f)] = sha(f)
        assert sha(f) == frozen_hashes[str(f)]
        with np.load(f) as data:
            x = data['bge_leaf'].astype(float); full = data['bge_query'].astype(float)
        f = REF / 'domains' / domain / 'score_matrices.npz'; feature_hashes[str(f)] = sha(f)
        with np.load(f) as data:
            aug = data['bge_augmented_query_vectors'].astype(float); choices = data['selected_choice_indices']
        x /= np.linalg.norm(x,axis=1,keepdims=True); full /= np.linalg.norm(full,axis=1,keepdims=True); aug /= np.linalg.norm(aug,axis=2,keepdims=True)
        for method in METHODS:
            groups = [(None, np.arange(len(x)), full)] if method == METHODS[0] else [
                (m['id'], np.array([i for i,l in enumerate(src['leaves']) if l['master_id']==m['id']]), aug[np.arange(len(full)),choices[:,mi]])
                for mi,m in enumerate(src['masters'])]
            for gid, ix, qq in groups:
                xx=x[ix]; center=xx-xx.mean(0); raw=qq@xx.T
                for ridge in rp['lambdas']:
                    # Dual covariance solve, independent of the SVD implementation.
                    adjusted = qq if len(xx)==1 else qq-(center.T@np.linalg.solve(center@center.T+(len(xx)-1)*ridge*np.eye(len(xx)), center@qq.T)).T
                    white=adjusted@xx.T; key=format(ridge,'g').replace('.','p')
                    for qi,q in enumerate(src['questions']):
                        r=ret[q['id']]['methods'][method]
                        assert r['memory_ids']==[l['memory_id'] for l in src['leaves']]
                        g=next(g for g in r['groups'] if g['indices']==ix.tolist())
                        if method==METHODS[0]: assert g['query']==q['retrieval_query']
                        else:
                            prior=next(m for m in historical[q['id']]['masters'] if m['master_id']==gid)
                            assert g['id']==gid and g['query']==prior['augmented_query'] and g['choice']==prior['selected_choice']
                            assert g['selected_statement']==prior['selected_statement']
                            assert g['choice']==max('ABCD',key=lambda c:g['clip_statement_scores'][c])
                        for observed, expected in ((np.array(r['raw'])[ix],raw[qi]),(np.array(r['weighting'][key]['white'])[ix],white[qi])):
                            err=float(np.max(abs(observed-expected))); max_error=max(max_error,err)
                            np.testing.assert_allclose(observed,expected,rtol=0,atol=2e-6)
                        group_checks+=1
    hits=Counter(); baseline_matches=0
    for qid,item in ret.items():
        q=original[qid]
        for method,r in item['methods'].items():
            raw=np.array(r['raw']); ids=r['memory_ids']
            for ridge in rp['lambdas']:
                key=format(ridge,'g').replace('.','p'); w=r['weighting'][key]
                white=np.array(w['white']); z=np.array(w['z']); unit=np.array(w['unit_correction'])
                for g in r['groups']:
                    ix=np.array(g['indices']); rr=raw[ix]; ww=white[ix]
                    assert abs(rr.std()-g['raw_std'])<1e-14
                    tol=32*np.finfo(float).eps*max(1.,float(abs(ww).max()))
                    rt=32*np.finfo(float).eps*max(1.,float(abs(rr).max()))
                    ez=ww if len(ix)==1 else np.zeros(len(ix)) if ww.std()<=tol else (ww-ww.mean())/ww.std()
                    allowance=max(1e-12,64*np.finfo(float).eps*max(1.,float(abs(ww).max()))/max(ww.std(),1e-10))
                    np.testing.assert_allclose(z[ix],ez,atol=allowance,rtol=0)
                    correction=np.zeros(len(ix)) if len(ix)==1 or ww.std()<=tol or rr.std()<=rt else rr.std()*np.tanh(z[ix])
                    np.testing.assert_allclose(unit[ix],correction,rtol=0,atol=1e-14)
            for name,spec in specs.items():
                if spec['method']!=method: continue
                values=raw.copy()
                if spec['alpha']:
                    key=format(spec['lambda'],'g').replace('.','p'); w=r['weighting'][key]
                    for g in r['groups']:
                        ix=np.array(g['indices']); corr=spec['alpha']*g['raw_std']*np.tanh(np.array(w['z'])[ix])
                        values[ix]+=np.where(np.array(w['unit_correction'])[ix]==0,0,corr)
                rank=np.argsort(-values,kind='stable'); rr=item['conditions'][name]
                assert rr['full_rank_memory_ids']==[ids[i] for i in rank]
                selected=[ids[i] for i in rank[:5]]; assert rr['top5_memory_ids']==selected
                np.testing.assert_array_equal(rr['top5_scores'],values[rank[:5]])
                source=manifest[qid+'::'+name]
                text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
                text+='\n\n'.join(f'Memory {i}: {memories[mid]}' for i,mid in enumerate(selected,1))
                text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(k+'. '+q['options'][k] for k in 'ABCD')+'\n\nAnswer with one letter only.'
                images=[{'path':path,'sha256':sha(path)} for path in q['query_image_paths']]
                for k in ('id','task','question','options','gold_choice','gold_ids','query_image_paths'): assert source[k]==q[k]
                assert source['reader_text']==text and source['top_memory_ids']==selected and source['query_images']==images
                assert source['payload_sha256']==digest({'system':settings['reader_system_prompt'],'user_text':text,'query_images':images})
                hits[name]+=len(set(selected)&set(q['gold_ids']))
                if name in METHODS:
                    assert old[source['record_id']]['payload_sha256']==source['payload_sha256']; baseline_matches+=1
    proof={'passed':True,'questions':371,'logical_records':19292,'unique_payloads':len({r['payload_sha256'] for r in manifest_list}),
           'independent_dual_ridge_groups':group_checks,'max_frozen_vector_score_error':max_error,'feature_sha256':feature_hashes,
           'rankings_and_reader_payloads_verified':19292,'historical_baseline_payload_matches':baseline_matches,'audit_script_sha256':sha(__file__)}
    assert proof['unique_payloads']==p['unique_reader_payloads'] and baseline_matches==742
    if inputs_only:
        (base/'independent_input_audit.json').write_text(json.dumps(proof,indent=2)+'\n'); print(json.dumps(proof),flush=True); return proof
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(settings['reader_model_path'],local_files_only=True)
    response_list=rows(out/'responses.jsonl'); responses={r['record_id']:r for r in response_list}
    assert len(responses)==len(response_list)==19292 and set(responses)==set(manifest)
    counts=Counter(); fresh={}; shared=formats=limits=0
    for r in response_list:
        source=manifest[r['record_id']]
        assert r['experiment_signature']==sig and r['run_id']==p['experiment_run_id'] and not r['historical_response_reuse']
        assert r['payload_sha256']==source['payload_sha256'] and r['condition']==source['condition']
        raw=processor.batch_decode([r['generated_token_ids']],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',raw,re.I); pred=match.group(1).upper() if match else None
        assert r['raw_response']==raw and r['predicted_choice']==pred and r['correct']==(pred==source['gold_choice'])
        assert r['status']==('ok' if match else 'format_error'); formats+=not bool(match); limits+=r['hit_token_limit']
        counts[r['condition']]+=r['correct']
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh; fresh[r['payload_sha256']]=r
            content=[{'type':'image'} for _ in source['query_image_paths']]+[{'type':'text','text':source['reader_text']}]
            rendered=processor.apply_chat_template([{'role':'system','content':settings['reader_system_prompt']},{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            assert hashlib.sha256(rendered.encode()).hexdigest()==r['rendered_prompt_sha256']
        else:
            origin=responses[r['same_run_shared_inference_record_id']]; assert origin['fresh_inference_executed_for_this_record']
            for k in ('payload_sha256','generated_token_ids','raw_response','execution_id','rendered_prompt_sha256'): assert origin[k]==r[k]
            shared+=1
    assert len(fresh)==proof['unique_payloads'] and len(fresh)+shared==19292
    summary=read(base/'summary.json'); selection=read(base/'selected_parameters.json')
    for n in specs:
        assert counts[n]==summary['overall'][n]['correct'] and hits[n]==summary['overall'][n]['gold_hits_top5']
    candidates=[]
    for row in summary['search_results']:
        names=[next(n for n,s in specs.items() if s=={'method':m,'lambda':row['lambda'],'alpha':row['alpha']}) for m in METHODS]
        b,m=(counts[n] for n in names)
        assert (b,m)==(row['baseline_correct'],row['master_correct'])
        eligible=b>counts[METHODS[0]] and m>counts[METHODS[1]] and m>b
        assert eligible==row['passes_all_three']
        if eligible:candidates.append(row)
    winner=min(candidates,key=lambda r:(-r['master_correct'],-r['baseline_correct'],r['alpha'],r['lambda'])) if candidates else None
    assert winner==summary['selected']==selection['selected'] and not selection['gpt5_results_used_for_selection']
    for path,h in selection['source_sha256'].items():assert sha(path)==h
    assert formats==summary['format_errors'] and limits==summary['token_limit_hits']
    proof.update(correct=dict(counts),fresh_inferences=len(fresh),shared_records=shared,format_errors=formats,token_limit_hits=limits,selected=winner,
                 artifact_sha256={n:sha(base/n) for n in ('summary.json','selected_parameters.json','reader/responses.jsonl','reader/protocol.json','protocol.json','retrieval_results.jsonl')})
    (base/'independent_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE);p.add_argument('--inputs-only',action='store_true');a=p.parse_args();audit(a.output,a.inputs_only)
