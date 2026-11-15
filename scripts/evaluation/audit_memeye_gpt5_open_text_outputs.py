"""Independently audit GPT native calls, immutable memories and final contexts."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from pathlib import Path
import tempfile
from scripts.evaluation import memeye_gpt5_open_text_retrieval as run


def audit(method):
    root = run.OUT / method
    p = run.read(root/'protocol.json')
    ready = run.read(root/'ready_for_reader.json')
    assert ready['reader_manifest_sha256'] == run.base.sha(root/'reader_manifest.jsonl')
    source = Path(p['open_evaluation']['source_root'])
    for filename, digest in p['open_evaluation']['source_hashes'].items():
        assert run.base.sha(Path(filename)) == digest, filename
    api_outputs = {r['output']['response_id']: r for r in run.base.read_rows(root/'api_responses.jsonl')} if (root/'api_responses.jsonl').exists() else {}
    calls = list((root/'calls').glob('*.json'))
    response_ids, tokens = set(), Counter()
    for filename in calls:
        c = run.read(filename)
        req, out = c['request'], c['output']
        wire = run.api.build_request(req['messages'], req['max_new_tokens'], req['temperature'])
        item = api_outputs[out['response_id']]
        assert item['request'] == wire
        assert all(out[k] == v for k,v in item['output'].items())
        assert set(out) - set(item['output']) <= {'structured_output_valid', 'structured_output_error'}
        assert out['returned_model'] == p['caption_model'] == 'gpt-5-2025-08-07'
        assert wire['seed'] == 42 and wire['reasoning_effort'] == 'minimal' and 'temperature' not in wire
        assert not out.get('refusal') and not out['hit_token_limit']
        assert out['input_tokens'] + wire['max_completion_tokens'] <= p['context_budget'] == 128000
        response_ids.add(out['response_id'])
        tokens['prompt_tokens'] += out['input_tokens']
        tokens['completion_tokens'] += out['generated_tokens']
        tokens['reasoning_tokens'] += out.get('reasoning_tokens') or 0
    # Every successful fresh request must be preserved by the native broker.
    assert set(api_outputs) == response_ids
    rows = run.base.read_rows(root/'retrieval.jsonl')
    assert len(rows) == 371
    restoration = {}
    if method in ('a_mem', 'gen_agents'):
        for task in p['tasks']:
            old, new = run.read(source/task/'final_memory.json'), run.read(root/task/'restored_memory.json')
            assert (run.without_access(old) == run.without_access(new)) if method == 'a_mem' else old == new
            restoration[task] = len(new)
    if method == 'simplemem':
        run.official.bootstrap()
        from simplemem.core.answer_generator import AnswerGenerator
        from simplemem.core.models.memory_entry import MemoryEntry
        formatter = object.__new__(AnswerGenerator)
        entries = {task: {e['entry_id']: e for e in run.read(source/task/'final_memory.json')} for task in p['tasks']}
        for row in rows:
            detail = row['retrieval_detail']
            old = entries[row['task']]
            selected = detail['context_memory_ids']
            assert all(item['id'] in old and {k:v for k,v in item.items() if k!='id'} == old[item['id']] for item in detail['items'])
            assert formatter._format_contexts([MemoryEntry(**old[i]) for i in selected]) == row['retrieved_context']
        from scripts.analysis.audit_simplemem_fusion_ranks import audit as ranks_audit
        with tempfile.TemporaryDirectory(prefix='gpt_open_simplemem_audit_') as directory:
            temporary = Path(directory)
            run.base.write_json(temporary/'protocol.json', p)
            for row in rows:
                run.base.write_json(temporary/'retrieval_checkpoints'/(run.base.hash_object(row['id'])+'.json'), row)
            ranks = ranks_audit(temporary)
        assert ranks['audit_passed'] and ranks['complete'] and ranks['questions'] == 371
        ranks.pop('checkpoint_hashes')
        ranks['source_retrieval_sha256'] = run.base.sha(root/'retrieval.jsonl')
        run.base.write_json(root/'independent_rank_audit.json', ranks)
    log = root/'retrieval_run.log'
    lines = log.read_text().splitlines()
    warnings = [line for line in lines if line.startswith('Error during keyword search:')]
    attempts = run.base.read_rows(root/'api_attempts.jsonl') if (root/'api_attempts.jsonl').exists() else []
    report = {'status': 'passed', 'method': method, 'questions': 371,
        'native_final_contexts_from_frozen_memory': True, 'frozen_restoration_sizes': restoration,
        'successful_native_api_calls': len(calls), 'all_calls_matched_actual_wire_and_response': True,
        'returned_model': p['caption_model'], 'usage': dict(tokens),
        'api_attempt_count': len(attempts), 'api_failed_attempts': sum(r['status'] != 'ok' for r in attempts),
        'recoverable_native_keyword_warnings': warnings,
        'warning_policy': 'Preserve native caught lexical-search errors and continuation through other channels; no algorithm repair.',
        'retrieval_sha256': run.base.sha(root/'retrieval.jsonl'),
        'reader_manifest_sha256': run.base.sha(root/'reader_manifest.jsonl'),
        'audit_script_sha256': run.base.sha(Path(__file__))}
    run.base.write_json(root/'independent_output_audit.json', report)
    print(method, 'independent output audit passed', len(calls), 'fresh calls', len(warnings), 'native keyword warnings')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=tuple(run.SOURCES), required=True)
    audit(parser.parse_args().method)
