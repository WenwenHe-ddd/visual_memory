"""Reuse identical requests from this task's initial size-limited attempt."""
from pathlib import Path
from scripts.evaluation import mem_gallery_deepseek_gallery_limits as run
from scripts.evaluation import mem_gallery_deepseek_caption_visual_master as initial

source, output = initial.OUTPUT, run.OUTPUT
old, p = initial.verify(source), run.verify(output)
records=[]
for folder in ('caption_source','candidate_generation'):
    calls=run.rows(source/folder/'api_responses.jsonl')
    attempts=run.rows(source/folder/'api_attempts.jsonl')
    retained=set()
    for path in sorted((source/folder/'responses').glob('*.json')):
        saved=run.read(path)
        assert saved['experiment_signature']==old['experiment_signature']
        wire=run.images.wire_for(saved['payload'],2048)
        out=saved['output']
        if run.transport.request_sha256(wire)!=out['request_sha256']: continue
        assert out['returned_model']==p['expected_model'] and out['finish_reason']=='stop' and not out['refusal']
        dest=output/folder/'responses'/path.name
        run.freeze(dest,dict(saved,experiment_signature=p['experiment_signature']))
        retained.add(out['response_id'])
        records.append(dict(source_path=str(path),source_sha256=run.sha(path),destination_path=str(dest),request_sha256=out['request_sha256']))
    run.write_rows(output/folder/'api_responses.jsonl',[r for r in calls if r['output']['response_id'] in retained])
    run.write_rows(output/folder/'api_attempts.jsonl',[r for r in attempts if r.get('response_id') in retained])
    print(folder,'identical responses retained',len(retained),flush=True)
run.freeze(output/'reuse_manifest.json',dict(source_protocol_path=str(source/'protocol.json'),source_protocol_sha256=run.sha(source/'protocol.json'),
    source_signature=old['experiment_signature'],experiment_signature=p['experiment_signature'],records=records,
    initial_failures_path=str(source/'caption_source/api_attempts.jsonl'),initial_failures_sha256=run.sha(source/'caption_source/api_attempts.jsonl'),
    policy='Same task and exact wire hash; unchanged generation inputs only; initial files retained verbatim.'))
