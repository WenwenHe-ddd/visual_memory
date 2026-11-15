"""Preserve same-task responses and recorded content rejections without resubmission."""
from scripts.evaluation import mem_gallery_deepseek_three_methods as run
from scripts.evaluation import mem_gallery_deepseek_gallery_limits as previous
source,output=previous.OUTPUT,run.OUTPUT
old,p=previous.verify(source),run.verify(output)
records,rejections=[],[]
for folder in ('caption_source','candidate_generation'):
    calls=run.rows(source/folder/'api_responses.jsonl')
    attempts=run.rows(source/folder/'api_attempts.jsonl')
    kept=set()
    for path in sorted((source/folder/'responses').glob('*.json')):
        saved=run.read(path);assert saved['experiment_signature']==old['experiment_signature']
        wire=run.images.wire_for(saved['payload'],2048);out=saved['output']
        assert run.transport.request_sha256(wire)==out['request_sha256']
        assert out['finish_reason']=='stop' and out['returned_model']==p['expected_model'] and not out['refusal']
        dest=output/folder/'responses'/path.name
        run.freeze(dest,dict(saved,experiment_signature=p['experiment_signature']))
        kept.add(out['response_id'])
        records.append(dict(source_path=str(path),source_sha256=run.sha(path),destination_path=str(dest),request_sha256=out['request_sha256']))
    imported_calls=[r for r in calls if r['output']['response_id'] in kept]
    imported_attempts=[r for r in attempts if r.get('response_id') in kept]
    if folder=='caption_source':
        jobs={j['id']:j for j in run.rows(output/'caption_jobs.jsonl')}
        for failure in attempts:
            if failure.get('provider_error_code')!='data_inspection_failed':continue
            payload=run.caption_payload(jobs[failure['id']],p)
            key=run.transport.request_sha256(run.images.wire_for(payload,2048))
            assert key==failure['request_sha256']
            out=run.transport.rejection_outcome(key)
            dest=output/folder/'responses'/(failure['id']+'.json')
            run.freeze(dest,dict(payload=payload,output=out,experiment_signature=p['experiment_signature']))
            imported_calls.append(dict(stage='caption',id=failure['id'],output=out))
            imported_attempts.append(failure)
            src=source/folder/'api_attempts.jsonl'
            rejections.append(dict(source_path=str(src),source_sha256=run.sha(src),attempt=failure,destination_path=str(dest)))
    run.write_rows(output/folder/'api_responses.jsonl',imported_calls)
    run.write_rows(output/folder/'api_attempts.jsonl',imported_attempts)
    print(folder,'retained completions',len(kept),flush=True)
run.freeze(output/'reuse_manifest.json',dict(source_protocol_path=str(source/'protocol.json'),source_protocol_sha256=run.sha(source/'protocol.json'),
    source_signature=old['experiment_signature'],experiment_signature=p['experiment_signature'],records=records,rejections=rejections,
    initial_failures_path=str(source/'caption_source/api_attempts.jsonl'),initial_failures_sha256=run.sha(source/'caption_source/api_attempts.jsonl'),
    ancestor_reuse_manifest=str(source/'reuse_manifest.json'),ancestor_reuse_manifest_sha256=run.sha(source/'reuse_manifest.json'),
    policy='Exact-wire same-task successes retained; existing provider rejections recorded as failures and not resubmitted.'))
print('prior rejections retained without retry',len(rejections),flush=True)
