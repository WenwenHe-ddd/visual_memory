"""Count the archived GPT-5 empty answer as a failure, without regeneration."""
from scripts.evaluation import mem_gallery_amem_api as run


def main():
    root = run.OUTPUT / 'gpt5'
    p = run.gpt.verify(root)
    run.configure(p)
    invalid = list((root / 'reader/invalid').glob('*.json'))
    assert len(invalid) == 1
    source = invalid[0]
    saved = run.read(source)
    out = saved['output']
    assert saved['experiment_signature'] == p['experiment_signature']
    assert out['returned_model'] == p['expected_model']
    assert out['finish_reason'] == 'stop' and not out.get('refusal')
    assert not out['raw_response'].strip()
    run.freeze(root / 'empty_answer_amendment.json', dict(
        script_sha256=run.sha(__file__), base_protocol_sha256=run.sha(root / 'protocol.json'),
        original_response_path=str(source), original_response_sha256=run.sha(source),
        record_id=saved['payload']['record_id'],
        policy='Preserve original empty response and include as zero; no regeneration or fabricated completion.'))
    run.freeze(root / 'reader/responses' / source.name, saved)
    run.runner.answers(root, run.base.DEFAULT_ENV)
    run.analyze(root)
    scored = {r['id']: r for r in run.rows(root / 'scored_questions.jsonl')}
    result = scored[saved['payload']['record_id']]
    assert all(result[k] == 0 for k in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu'))
    run.dump(root / 'empty_answer_audit.json', dict(status='passed', original_response_preserved=True,
        regenerated=False, record_id=result['id'], all_answer_metrics_zero=True))


if __name__ == '__main__':
    main()
