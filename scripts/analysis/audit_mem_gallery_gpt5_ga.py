"""Additional end-to-end API-response and artifact audit for the Gallery GPT-5 run."""
from collections import Counter
import json
from pathlib import Path

from scripts.evaluation import mem_gallery_gpt5_image_formats as experiment


def main():
    base = experiment.run
    root = base.DEFAULT_OUTPUT
    p = experiment.verify(root)
    assert base.read(root / 'execution_status.json')['status'] == 'complete'
    audit = base.read(root / 'audit.json')
    assert audit['status'] == 'passed'
    for path, digest in audit['artifact_sha256'].items():
        assert base.sha(path) == digest
    for path, digest in audit['memory_call_sha256'].items():
        assert base.sha(path) == digest
    complete = base.read(root / 'gen_agents/reader/complete.json')
    assert complete['count'] == 1711 and complete['experiment_signature'] == p['experiment_signature']
    for name in ['responses', 'manifest']:
        assert complete[name + '_sha256'] == base.sha(root / 'gen_agents/reader' / (name + '.jsonl'))
    cap_done = base.read(root / 'caption_source/complete.json')
    assert cap_done['count'] == 979 and cap_done['experiment_signature'] == p['experiment_signature']
    images = base.rows(root / 'caption_jobs.jsonl')
    assert len({v['id'] for v in images}) == len(images) == 979
    qs = base.rows(root / 'inputs/questions.jsonl')
    answers = base.rows(root / 'gen_agents/reader/responses.jsonl')
    assert len(answers) == len({v['id'] for v in answers}) == 1711
    assert {v['id'] for v in qs} == {v['id'] for v in answers}
    observations = base.read(root / 'observations.json')
    assert len(observations) == 20 and sum(map(len, observations.values())) == 3962
    assert sum(len(v['raw']['image_id']) for group in observations.values() for v in group) == 1003
    all_response_ids, usage, attempts = [], {}, {}
    roots = dict(caption=root / 'caption_source', memory=root / 'gen_agents', answer=root / 'gen_agents/reader')
    for stage, folder in roots.items():
        raw = base.rows(folder / 'api_responses.jsonl')
        outputs = [v['output'] for v in raw]
        assert all(v['stage'] == stage for v in raw)
        assert all(v['requested_model'] == 'gpt-5' and v['returned_model'] == p['expected_model'] for v in outputs)
        assert all(v['seed'] == 42 and v['reasoning_effort'] == 'minimal' for v in outputs)
        assert all(v['response_id'] and v['request_id'] for v in outputs)
        all_response_ids.extend(v['response_id'] for v in outputs)
        usage[stage] = dict(api_responses=len(outputs),
            input_tokens=sum(v['input_tokens'] for v in outputs),
            cached_input_tokens=sum((v['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) for v in outputs),
            completion_tokens=sum(v['generated_tokens'] for v in outputs),
            reasoning_tokens=sum(v.get('reasoning_tokens') or 0 for v in outputs),
            finish_reasons=dict(Counter(v['finish_reason'] for v in outputs)),
            refusals=sum(bool(v['refusal']) for v in outputs))
        attempts[stage] = dict(Counter(v['status'] for v in base.rows(folder / 'api_attempts.jsonl')))
        if stage in ('caption', 'answer'):
            accepted = (base.rows(root / 'caption_source/captions.jsonl') if stage == 'caption' else answers)
            by_response = {v['response_id']: v for v in outputs}
            assert len(accepted) == (979 if stage == 'caption' else 1711)
            for accepted_row in accepted:
                o = by_response[accepted_row['response_id']]
                assert accepted_row['raw_response'] == o['raw_response']
                assert accepted_row['request_sha256'] == o['request_sha256']
                assert o['finish_reason'] == 'stop' and not o['refusal'] and o['raw_response'].strip()
    assert len(all_response_ids) == len(set(all_response_ids)), 'Duplicate API response identity across stages'
    result = dict(status='passed', questions=1711, source_rounds=3962, scenarios=20,
        unique_history_captions=979, history_image_references=1003, model=p['expected_model'],
        original_query_image_bytes_preserved=True, no_qwen_history_captions_in_final_observations=True,
        usage_by_stage=usage, attempts_by_stage=attempts, unique_api_response_ids=len(all_response_ids),
        experiment_signature=p['experiment_signature'], audit_script_sha256=base.sha(Path(__file__)),
        primary_audit_sha256=base.sha(root / 'audit.json'),
        image_format_amendment_sha256=base.sha(root / 'image_format_amendment.json'))
    base.dump(root / 'independent_audit.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
