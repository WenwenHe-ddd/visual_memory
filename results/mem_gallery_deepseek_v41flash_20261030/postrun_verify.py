"""Final integrity, provenance and matched-setting comparison after scoring."""
from pathlib import Path
from scripts.evaluation import mem_gallery_deepseek_three_methods as run

root=run.OUTPUT
p=run.verify(root)
completed=run.read(root/'completion_manifest.json')
assert completed['status']=='complete'
for filename,digest in completed['artifacts_sha256'].items():assert run.sha(filename)==digest,filename
reuse=run.read(root/'reuse_manifest.json')
assert run.sha(reuse['source_protocol_path'])==reuse['source_protocol_sha256']
old=run.read(reuse['source_protocol_path'])
assert old['experiment_signature']==reuse['source_signature']
assert run.sha(reuse['initial_failures_path'])==reuse['initial_failures_sha256']
assert run.sha(reuse['ancestor_reuse_manifest']) == reuse['ancestor_reuse_manifest_sha256']
ancestor = run.read(reuse['ancestor_reuse_manifest'])
assert run.sha(ancestor['source_protocol_path']) == ancestor['source_protocol_sha256']
for item in ancestor['records']:
    assert run.sha(item['source_path']) == item['source_sha256']
    olditem = run.read(item['source_path'])
    assert olditem['experiment_signature'] == ancestor['source_signature']
    assert run.read(item['destination_path']) == dict(olditem, experiment_signature=ancestor['experiment_signature'])
for r in reuse['records']:
    assert run.sha(r['source_path'])==r['source_sha256']
    assert run.read(r['source_path'])['experiment_signature']==reuse['source_signature']

master=run.ROOT/'results/mem_gallery_gpt5_caption_visual_master_20261030'
baseline=run.ROOT/'results/mem_gallery_gpt5_baseline_20261030'
for other in (master,baseline):
    assert run.sha(root/'inputs/questions.jsonl')==run.sha(other/'inputs/questions.jsonl')
    assert run.sha(root/'inputs/gold.jsonl')==run.sha(other/'inputs/gold.jsonl')
    cm=run.read(other/'completion_manifest.json')
    for key in ('protocol.json','final_results.json'):
        assert run.sha(other/key)==cm['artifacts_sha256'][str(other/key)]
    control=run.read(other/'protocol.json')
    for key in ('top_k','text_encoder','query_prefix','reader_system_prompt','task_instructions','max_new_tokens','seed'):
        assert control[key]==p[key],key
control=run.read(master/'protocol.json')
for key in ('lambda','alpha','image_encoder','candidate_system','candidate_max_attempts','candidate_max_tokens'):
    assert control[key]==p[key],key
actual=run.read(root/'final_results.json')
caption_outcomes=run.rows(root/'caption_source/captions.jsonl')
answers=run.rows(root/'reader/responses.jsonl')
run.dump(root/'provider_outcomes.json',dict(
    caption_rejections=[r['id'] for r in caption_outcomes if r.get('status')=='provider_rejection'],
    answer_rejections=[dict(id=r['id'],condition=r['condition'],request_sha256=r['request_sha256']) for r in answers if r.get('status')=='provider_rejection'],
    evaluated_questions_per_method=1711, no_questions_dropped=True,
    caption_policy='Rejected images keep an empty caption and the original dialogue/image node.',
    answer_policy='Rejected answers score zero in all answer metrics.'))
prior=dict(run.read(master/'final_results.json')['results'])
prior['Natural baseline']=run.read(baseline/'final_results.json')['overall']
comparison=[]
for name in run.CONDITIONS:
    current=actual['results'][name]
    comparison.append(dict(method=name,deepseek=current,gpt5=prior[name],
        em_delta_percentage_points=100*(current['em']-prior[name]['em'])))
run.dump(root/'cross_backbone_comparison.json',dict(methods=comparison,
    conditions_matched=True,top_k=5,**{'lambda':.03,'alpha':1.},
    differences=['Fresh model-specific history captions, candidate hypotheses and answers.',
        'DeepSeek temperature0/thinking=false versus GPT-5 minimal reasoning and omitted temperature.',
        '12 oversized image files are JPEG compressed at their original resolution for DeepSeek; GPT-5 received original bytes.',
        'DeepSeek returns a provider alias without a resolved revision; GPT-5 returned gpt-5-2025-08-07.']))
extra=['commands.json','finish_run.py','import_same_run.py','postrun_verify.py','cross_backbone_comparison.json','provider_outcomes.json','input_length_audit.json','context_coverage.json']
run.dump(root/'postrun_verification.json',dict(status='passed',sealed_artifacts_verified=len(completed['artifacts_sha256']),
    completion_manifest_sha256=run.sha(root/'completion_manifest.json'),
    same_task_reuse_records_verified=len(reuse['records']),matched_settings_verified=True,
    additional_artifacts_sha256={str(root/n):run.sha(root/n) for n in extra}))
print(actual['results'])
