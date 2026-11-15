"""Continue the authorized one-call purpose-statement variant after a valid probe.

Only the known rejected evolution call has an added system statement. Its actual
probe response is reused, not regenerated. All original results stay untouched.
"""
from pathlib import Path
from scripts.evaluation import retry_amem_deepseek_astronomy as astronomy

prior = astronomy.prior
run = astronomy.run
OUTPUT = run.OUTPUT/'deepseek_astronomy_purpose_20261105'


def prepare():
    p = run.gpt.verify(OUTPUT)
    original = run.read(OUTPUT/'original_failed_request.json')
    variant = run.read(OUTPUT/'purpose_probe_request.json')
    response = run.read(OUTPUT/'purpose_probe_response.json')
    result = run.read(OUTPUT/'purpose_probe_result.json')
    assert result['status']=='passed'
    assert variant['messages'][1:]==original['messages']
    assert variant['messages'][0]==dict(role='system',content=result['statement'])
    assert {k:v for k,v in variant.items() if k!='messages'}=={k:v for k,v in original.items() if k!='messages'}
    assert prior.isolated.api.request_sha256(original)=='6d871726a0aec3f2fdde910f7e3d2f60cfbba7a019743de09ed66a10388e5cb3'
    assert response['returned_model']==p['expected_model'] and response['finish_reason']=='stop'
    schema=run.json.loads(original['messages'][0]['content'].split('Follow this JSON schema exactly: ',1)[1])
    import jsonschema
    jsonschema.validate(run.json.loads(response['raw_response']),schema)
    request=dict(messages=original['messages'],max_new_tokens=1000,kind='memory',do_sample=True,
        temperature=original['temperature'],seed=42,experiment_signature=p['experiment_signature'],
        response_format=dict(type='json_schema',json_schema=dict(name='response',schema=schema,strict=True)))
    key=run.obj(request)
    output=dict(response,structured_output_valid=True,purpose_variant=dict(
        original_wire_sha256=prior.isolated.api.request_sha256(original),
        initial_variant_wire_sha256=prior.isolated.api.request_sha256(variant),
        statement=result['statement']))
    run.freeze(OUTPUT/'memory/calls'/(key+'.json'),dict(request=request,output=output))
    amendment_path=OUTPUT/'purpose_variant_amendment.json'
    if amendment_path.exists():
        amendment=run.read(amendment_path)
        assert amendment['script_sha256']==run.sha(__file__)
        for f,h in amendment['artifact_sha256'].items():
            assert run.sha(f)==h
        return p,key
    # Keep the provider's actual requests/responses and any cap escalation evidence.
    for filename in ('api_responses.jsonl','budget_recovery_events.jsonl'):
        source=OUTPUT/'probe'/filename
        if source.exists():
            with (OUTPUT/'memory'/filename).open('a') as dest:
                dest.write(source.read_text())
    files=[OUTPUT/'original_failed_request.json',OUTPUT/'purpose_probe_request.json',
        OUTPUT/'purpose_probe_response.json',OUTPUT/'purpose_probe_result.json',
        OUTPUT/'protocol.json',OUTPUT/'observations.json',OUTPUT/'inputs/questions.jsonl',OUTPUT/'inputs/gold.jsonl',
        *sorted((OUTPUT/'memory/calls').glob('*.json'))]
    run.freeze(amendment_path,dict(user_authorization='Add truthful evaluation-purpose statement to the failed call, test it, and continue if successful.',
        script_sha256=run.sha(__file__),runner_sha256=run.sha(prior.__file__),
        original_wire_sha256=prior.isolated.api.request_sha256(original),
        variant_initial_wire_sha256=prior.isolated.api.request_sha256(variant),
        logical_call_key=key,statement=result['statement'],
        policy='Only this one evolution call uses the additional system statement; retain original logical cache key with explicit actual-wire provenance. All other memory and reader requests retain original prompts.',
        prior_1657_results_unchanged=True,artifact_sha256={str(f):run.sha(f) for f in files}))
    return p,key


def audit_variant(p,key):
    rows=run.rows(OUTPUT/'memory/api_responses.jsonl')
    api={(r['output']['request_sha256'],r['output']['response_id']):r for r in rows if r.get('request')}
    changed=[]
    for f in (OUTPUT/'memory/calls').glob('*.json'):
        c=run.read(f);req,out=c['request'],c['output']
        assert run.obj(req)==f.stem and req['experiment_signature']==p['experiment_signature']
        messages=req['messages']
        if out.get('purpose_variant'):
            assert f.stem==key
            changed.append(key)
            messages=[dict(role='system',content=out['purpose_variant']['statement']),*messages]
        cap=out.get('budget_recovery',{}).get('successful_cap',max(2048,req['max_new_tokens']))
        expected=prior.isolated.api.build_request(messages,cap,req['temperature'])
        assert prior.isolated.api.request_sha256(expected)==out['request_sha256']
        actual=api[(out['request_sha256'],out['response_id'])]
        assert actual['request']==expected and actual['output']['raw_response']==out['raw_response']
        assert out['returned_model']==p['expected_model'] and out['finish_reason']=='stop'
    assert changed==[key]
    events=run.rows(OUTPUT/'memory/call_events.jsonl')
    assert any(e['key']==key and e['reused'] for e in events)
    run.dump(OUTPUT/'purpose_variant_audit.json',dict(status='passed',modified_calls=1,
        logical_call_key=key,probe_response_reused=True,all_memory_wire_requests_verified=True))


def main():
    p,key=prepare()
    prior.OUTPUT=OUTPUT;prior.TASKS={astronomy.TASK}
    run.dump(OUTPUT/'execution_status.json',dict(status='running',phase='memory',prompt_variant=True))
    errors=prior.memory_stage(p)
    completed=[astronomy.TASK] if (OUTPUT/'memory'/astronomy.TASK/'complete.json').exists() else []
    audit_variant(p,key)
    if completed:
        run.dump(OUTPUT/'execution_status.json',dict(status='running',phase='answers',prompt_variant=True))
        prior.isolated.partial_answers(OUTPUT,p,completed)
    astronomy.OUTPUT=OUTPUT
    astronomy.combine(p,completed)
    result=run.read(OUTPUT/'combined_results.json')
    result.update(prompt_variant='Evaluation-purpose system statement added to one Astronomy memory-evolution call.',
        variant_call_key=key,original_prompt_baseline=False)
    run.dump(OUTPUT/'combined_results.json',result)
    audit=run.read(OUTPUT/'combined_audit.json')
    audit.update(result_sha256=run.sha(OUTPUT/'combined_results.json'),
        purpose_variant_audit_sha256=run.sha(OUTPUT/'purpose_variant_audit.json'),original_prompt_baseline=False)
    run.dump(OUTPUT/'combined_audit.json',audit)
    run.dump(OUTPUT/'execution_status.json',dict(status='finished',prompt_variant=True,
        completed_tasks=completed,failed_tasks=[e['task'] for e in errors],full_dataset_score_available=bool(completed)))


if __name__=='__main__':
    main()
