"""Independent alpha-scaling, rankings, payload and metric audit."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
import numpy as np
from scripts.analysis.audit_positive_variance_lambda_ablation import read,lines,sha,digest,zscore


def audit(base):
    completion=read(base/'completion_manifest.json');p=read(base/'reader/protocol.json')
    retrieval_protocol=read(base/'retrieval_protocol.json');summary=read(base/'summary.json')
    hashes={**completion['artifact_sha256'],**p['source_sha256'],**p['model_source_sha256'],**retrieval_protocol['source_sha256']}
    for name,expected in hashes.items():assert sha(name)==expected,name
    unsigned=dict(p);signature=unsigned.pop('experiment_signature');assert digest(unsigned)==signature
    assert p['lambda']==.03 and p['alphas']==[.1,.3,.6,1.,3.,6.,10.]
    assert p['reader_settings']['seed']==42 and not p['reader_settings']['do_sample']
    specs=p['condition_specs'];assert len(specs)==16
    source=Path(retrieval_protocol['source_run'])
    prior={r['question_id']:r for r in lines(source/'retrieval_results.jsonl')}
    rows=lines(base/'retrieval_results.jsonl');assert len(rows)==len({r['question_id'] for r in rows})==371
    ranks={};groups_checked=0;production_rank_checks=0
    max_z_error=max_correction_error=max_production_roundoff=0.
    for row in rows:
        old=prior[row['question_id']];assert row['domain']==old['domain']
        for method,data in row['methods'].items():
            previous=old['methods'][method];stats=previous['weighting']['0.03']
            ids=data['memory_ids'];raw=np.array(data['raw']);unit=np.array(data['unit_correction'])
            assert ids==previous['memory_ids'] and data['groups']==previous['groups']
            assert data['raw']==previous['raw'] and data['unit_correction']==stats['correction']
            assert data['enhanced']==stats['enhanced'] and data['z']==stats['z']
            assert sorted(i for g in data['groups'] for i in g['indices'])==list(range(len(ids)))
            for group in data['groups']:
                ix=group['indices'];std=float(np.std(raw[ix]));assert abs(std-group['raw_std'])<1e-12
                expected_z=zscore(np.array(data['enhanced'])[ix]);stored_z=np.array(data['z'])[ix]
                error=float(max(abs(expected_z-stored_z)));assert error<1e-10
                max_z_error=max(max_z_error,error)
                expected=np.zeros(len(ix)) if len(ix)==1 or group['raw_status']=='zero_variance' else std*np.tanh(expected_z)
                error=float(max(abs(unit[ix]-expected)));assert error<1e-12
                max_correction_error=max(max_correction_error,error)
                for alpha in p['alphas']:assert max(abs(alpha*unit[ix]))<=alpha*std+1e-12
                groups_checked+=1
            for name,spec in specs.items():
                if spec['method']!=method:continue
                scores=raw+spec['alpha']*unit
                order=sorted(range(len(ids)),key=lambda i:(-scores[i],i))[:5]
                top=[ids[i] for i in order];actual=row['conditions'][name]
                assert actual['top5_memory_ids']==top
                assert np.allclose(actual['top5_scores'],scores[order],rtol=0,atol=1e-12)
                if spec['lambda'] is not None:
                    # Check exact production multiplication order too: alpha
                    # times raw_std, then tanh(z), versus alpha times cached c.
                    direct=raw.copy();zs=np.array(data['z'])
                    for group in data['groups']:
                        ix=group['indices']
                        if len(ix)>1 and group['raw_status']!='zero_variance':
                            direct[ix]+=spec['alpha']*float(np.std(raw[ix]))*np.tanh(zs[ix])
                    err=float(max(abs(direct-scores)));assert err<1e-12
                    max_production_roundoff=max(max_production_roundoff,err)
                    direct_order=sorted(range(len(ids)),key=lambda i:(-direct[i],i))[:5]
                    assert [ids[i] for i in direct_order]==top
                    production_rank_checks+=1
                if spec['alpha'] in (0.,1.):assert actual==old['conditions'][name]
                ranks[row['question_id']+'::'+name]=top
    assert groups_checked==retrieval_protocol['unit_correction_group_checks']==4233
    qa=lines(base/'reader/qa_manifest.jsonl');answers=lines(base/'reader/responses.jsonl')
    manifest={r['record_id']:r for r in qa};responses={r['record_id']:r for r in answers}
    assert len(qa)==len(answers)==len(manifest)==len(responses)==len(ranks)==5936
    assert set(manifest)==set(responses)==set(ranks)
    original={q['id']:q for q in lines(ROOT/'data/independent_visual_inputs/qa_manifest.jsonl')}
    memories={m['id']:m['reader_text'] for m in lines(ROOT/'data/independent_visual_inputs/retrieval_memories.jsonl')}
    fresh={};counts=Counter();correct=Counter();hits=Counter();totals=Counter()
    for rid,m in manifest.items():
        q=original[m['id']];r=responses[rid]
        assert m['top_memory_ids']==ranks[rid]
        assert all(m[k]==q[k] for k in ['gold_choice','gold_ids','question','options','query_image_paths'])
        text='Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        text+='\n\n'.join(f'Memory {i}: {memories[mid]}' for i,mid in enumerate(ranks[rid],1))
        text+='\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'
        text+='\n'.join(f'{c}. {q["options"][c]}' for c in 'ABCD')+'\n\nAnswer with one letter only.'
        assert text==m['reader_text']
        images=[{'path':path,'sha256':sha(path)} for path in q['query_image_paths']]
        assert images==m['query_images']
        payload={'system':p['reader_settings']['reader_system_prompt'],'user_text':text,'query_images':images}
        assert digest(payload)==m['payload_sha256']==r['payload_sha256']
        assert r['experiment_signature']==signature and r['run_id']==p['experiment_run_id']
        assert not r['historical_response_reuse']
        match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',r['raw_response'].strip(),re.I)
        assert match and r['status']=='ok' and not r['hit_token_limit']
        choice=match.group(1).upper();is_correct=choice==q['gold_choice']
        assert r['predicted_choice']==choice and r['correct']==is_correct
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh;fresh[r['payload_sha256']]=r
        else:
            origin=responses[r['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            assert all(r[k]==origin[k] for k in ['payload_sha256','raw_response','generated_token_ids','execution_id'])
        name=m['condition'];counts[name]+=1;correct[name]+=is_correct
        hits[name]+=len(set(m['top_memory_ids'])&set(q['gold_ids']));totals[name]+=len(set(q['gold_ids']))
    assert len(fresh)==p['unique_reader_payloads']==summary['fresh_inferences']
    for name in specs:
        metrics=summary['overall'][name]
        assert counts[name]==metrics['questions']==371 and totals[name]==metrics['gold_references']==1041
        assert correct[name]==metrics['correct'] and hits[name]==metrics['gold_hits_top5']
        assert correct[name]/371==metrics['accuracy'] and hits[name]/1041==metrics['recall_at_5_micro']
    old_answers={r['record_id']:r for r in lines(source/'reader/responses.jsonl')}
    for name,changed in summary['anchor_answer_changes'].items():
        expected=[q for q in original if responses[q+'::'+name]['predicted_choice']!=old_answers[q+'::'+name]['predicted_choice']]
        assert changed==expected
    result={'passed':True,'conditions':16,'questions_per_condition':371,'records':5936,
        'verified_file_hashes':len(hashes),'verified_rankings':len(ranks),'verified_groups':groups_checked,
        'raw_and_unit_corrections_match_signed_source':True,'alpha_correction_bounds_verified':True,
        'production_arithmetic_rankings_verified':production_rank_checks,'max_production_roundoff':max_production_roundoff,
        'max_z_error':max_z_error,'max_correction_error':max_correction_error,
        'fresh_inferences':len(fresh),'shared_records':5936-len(fresh),
        'reconstructed_payloads':5936,'format_errors':0,'token_limit_hits':0,
        'independent_answer_accuracy_and_recall_verified':True,'anchor_answer_changes':{k:len(v) for k,v in summary['anchor_answer_changes'].items()},
        'audit_script_sha256':sha(__file__),'audit_helpers_sha256':sha(Path(__file__).with_name('audit_positive_variance_lambda_ablation.py')),
        'completion_manifest_sha256':sha(base/'completion_manifest.json')}
    (base/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/memeye_qwen_positive_variance_alpha_ablation_l0p03_20261107')
    audit(parser.parse_args().output)
