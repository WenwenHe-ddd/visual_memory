"""Independent audit of original-cache reuse, new-only inference, and metrics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.analysis.audit_memgallery_positive_variance_ablation import (
    audit_retrieval, read, lines, sha, digest, check, normalize)

DEFAULT = ROOT / 'results/memgallery_qwen_pv_original_answers_lambda_ablation_20261107'


def audit(base):
    from nltk.stem import PorterStemmer
    retrieval = audit_retrieval(base)
    protocol = read(base / 'reader/protocol.json')
    contract = protocol['contract']
    assert digest(contract) == protocol['experiment_signature']
    check(contract['source_sha256'])
    assert contract['historical_response_reuse'] and contract['seed'] == 42
    assert contract['do_sample'] is False and not contract['gold_loaded_before_analysis']
    assert not contract['candidate_expansion_in_reader']
    original = Path(read(base / 'retrieval_protocol.json')['source_run'])
    old_protocol = read(original / 'reader/protocol.json')
    for key in ['model', 'system_prompt', 'task_instructions', 'official_prompt_revision', 'seed',
                'do_sample', 'max_new_tokens', 'batch_size_text', 'batch_size_images', 'dtype',
                'query_image_max_pixels', 'input_truncation']:
        assert contract[key] == old_protocol['contract'][key], key
    old_manifest = {m['record_id']: m for m in lines(original / 'reader/qa_manifest.jsonl')}
    old_responses = {r['record_id']: r for r in lines(original / 'reader/responses.jsonl')}
    old_cache = {}
    for r in old_responses.values():
        assert r['payload_sha256'] == old_manifest[r['record_id']]['payload_sha256']
        h = r['payload_sha256']
        if h in old_cache: assert old_cache[h] == r['raw_response']
        old_cache[h] = r['raw_response']
    manifest = lines(base / 'reader/qa_manifest.jsonl')
    assert digest(manifest) == contract['manifest_sha256']
    responses = lines(base / 'reader/responses.jsonl')
    mindex = {m['record_id']: m for m in manifest}
    rindex = {r['record_id']: r for r in responses}
    assert len(manifest) == len(mindex) == len(responses) == len(rindex) == 20532
    assert set(mindex) == set(rindex)
    plan = read(base / 'reader/inference_plan.json')
    assert sha(base / 'reader/inference_plan.json') == contract['batch_plan_sha256']
    assert sha(base / 'reader/reuse_manifest.jsonl') == contract['reuse_manifest_sha256']
    assignments = {r['record_id']: r for r in lines(base / 'reader/reuse_manifest.jsonl')}
    assert set(assignments) == set(mindex)
    fresh = {}
    for batch in plan['batches']:
        record = read(base / 'reader/batches' / f'{batch["batch_index"]:04d}.json')
        assert record['plan'] == batch
        assert record['experiment_signature'] == protocol['experiment_signature']
        assert len(record['responses']) == len(batch['payloads']) == len(batch['record_ids'])
        assert len(batch['record_ids']) <= (4 if batch['has_image'] else 16)
        for rid, h, r in zip(batch['record_ids'], batch['payloads'], record['responses']):
            assert rid == r['record_id'] and h == r['payload_sha256'] == mindex[rid]['payload_sha256']
            assert h not in old_cache and h not in fresh
            assert mindex[rid]['condition'] == batch['owner_condition']
            assert contract['condition_specs'][mindex[rid]['condition']]['lambda'] is not None
            assert bool(mindex[rid]['query_image_paths']) == batch['has_image']
            assert r['batch_size'] == len(batch['record_ids']) and r['batch_index'] == batch['batch_index']
            fresh[h] = r
    assert len(fresh) == contract['new_unique_payloads'] == plan['new_unique_payloads'] == 1516
    questions = {q['id']: q for q in lines(base / 'inputs/questions.jsonl')}
    memories = {m['id']: m for m in lines(base / 'inputs/memories.jsonl')}
    gold = {g['id']: g for g in lines(base / 'inputs/gold.jsonl')}
    ranked = {q['id']: {c['name']: c for c in q['conditions']} for q in lines(base / 'retrieval_results.jsonl')}
    pairs = {p['id']: p for p in read(base / 'reader/paired_results.json')}
    summary = read(base / 'summary.json')
    assert set(questions) == set(gold) == set(pairs) and len(questions) == 1711
    counts, correct, hits, totals, f1s, reuse = [Counter() for _ in range(6)]
    image_hashes, same_input_answers = {}, {}
    stemmer = PorterStemmer()
    control_count = 0
    same_input_checks = 0
    provenance_fields = ['raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit',
                         'execution_id', 'batch_size', 'batch_elapsed_seconds']
    for rid, m in mindex.items():
        r = rindex[rid]
        q = questions[m['id']]
        condition = m['condition']
        method = contract['condition_specs'][condition]['method']
        assert not {'gold_ids', 'answer', 'gold_choice', 'clue'} & set(m)
        text = 'The retrieved memory contents are as follows:\n\n'
        text += '\n\n'.join(f'Memory {i}: {memories[mid]["reader_text"]}' for i, mid in enumerate(m['top5_memory_ids'], 1))
        text += '\n\nPlease answer the question about the conversation using the retrieved memories. Return only the answer content, concisely but completely. For dates or times, provide a specific date or time whenever possible.\n\nQuestion:\n' + q['question']
        if instruction := contract['task_instructions'].get(q['point'].upper()): text += '\n\n' + instruction
        assert m['reader_text'] == text
        assert m['query_image_paths'] == q['query_image_paths']
        images = []
        for path in q['query_image_paths']:
            if path not in image_hashes: image_hashes[path] = sha(path)
            images.append({'path': path, 'sha256': image_hashes[path]})
        assert images == m['query_images']
        h = digest({'system': contract['system_prompt'], 'user_text': text, 'query_images': images})
        assert h == m['payload_sha256'] == r['payload_sha256']
        assert m['top5_memory_ids'] == r['top5_memory_ids'] == ranked[m['id']][condition]['top5_memory_ids']
        assert r['experiment_signature'] == protocol['experiment_signature'] and r['run_id'] == protocol['run_id']
        assert r['historical_response_reused'] == (h in old_cache)
        assignment = assignments[rid]
        assert assignment['payload_sha256'] == h
        assert assignment['origin'] == ('original' if h in old_cache else 'new')
        if h in old_cache:
            origin = old_responses[r['historical_record_id']]
            assert h == origin['payload_sha256'] and r['historical_run_id'] == old_protocol['run_id']
            assert not r['fresh_inference_executed_for_this_record']
            assert r['same_run_shared_inference_record_id'] is None
            assert assignment['source_record_id'] == origin['record_id']
            reuse[condition] += 1
        else:
            origin = fresh[h]
            assert r['historical_record_id'] is None and r['historical_run_id'] is None
            assert r['fresh_inference_executed_for_this_record'] == (rid == origin['record_id'])
            assert r['same_run_shared_inference_record_id'] == (None if rid == origin['record_id'] else origin['record_id'])
            assert assignment['source_record_id'] == origin['record_id']
            assert r['inference_batch_index'] == origin['batch_index']
        for field in provenance_fields: assert r[field] == origin[field], (rid, field)
        raw_rid = m['id'] + '::' + method
        same = h == old_manifest[raw_rid]['payload_sha256']
        assert r['same_as_own_unweighted_input'] == assignment['same_as_own_unweighted_input'] == same
        if same:
            assert r['raw_response'] == old_responses[raw_rid]['raw_response']
            same_input_checks += 1
        if contract['condition_specs'][condition]['lambda'] is None:
            assert same and r['historical_response_reused']
            assert r['raw_response'] == old_responses[rid]['raw_response']
            control_count += 1
        if h in same_input_answers: assert r['raw_response'] == same_input_answers[h]
        same_input_answers[h] = r['raw_response']
        predicted, target = normalize(r['raw_response']), normalize(gold[m['id']]['answer'])
        pt, gt = predicted.split(), target.split()
        common = sum((Counter(stemmer.stem(t) for t in pt) & Counter(stemmer.stem(t) for t in gt)).values())
        f1 = 2 * common / (len(pt) + len(gt)) if common else 0.
        em = int(predicted == target)
        hit = len(set(gold[m['id']]['gold_ids']) & set(m['top5_memory_ids']))
        total = len(set(gold[m['id']]['gold_ids']))
        scored = pairs[m['id']][condition]
        assert scored['em'] == em and abs(scored['f1'] - f1) < 1e-15
        assert scored['gold_hits'] == hit and scored['gold_clues'] == total
        counts[condition] += 1; correct[condition] += em; hits[condition] += hit
        totals[condition] += total; f1s[condition] += f1
    assert control_count == 3422
    assert correct['baseline_question_only'] == 778 and correct['master_answer_query'] == 798
    for name, metrics in summary['overall'].items():
        assert counts[name] == metrics['questions'] == 1711
        assert correct[name] == metrics['correct'] and totals[name] == metrics['gold_links'] == 4109
        assert hits[name] == metrics['gold_hits'] and correct[name] / 1711 == metrics['em']
        assert hits[name] / 4109 == metrics['recall_at_5_micro']
        assert abs(f1s[name] / 1711 - metrics['f1']) < 1e-14
        assert reuse[name] == metrics['original_answer_reused']
    result = {'passed': True, 'retrieval': retrieval, 'logical_records': len(responses),
        'control_records_identical_to_original': control_count, 'control_model_calls': 0,
        'original_answer_reused_records': sum(reuse.values()), 'fresh_unique_payloads': len(fresh),
        'same_run_shared_records': sum(r['same_run_shared_inference_record_id'] is not None for r in responses),
        'same_as_own_unweighted_input_answer_checks': same_input_checks,
        'all_complete_payloads_reconstructed': True, 'all_identical_payload_answers_identical': True,
        'original_model_and_settings_unchanged': True, 'new_inputs_only_in_generation': True,
        'independent_em_f1_recall_verified': True,
        'empty_responses': sum(not r['raw_response'] for r in responses),
        'token_limit_records': sum(r['hit_token_limit'] for r in responses),
        'audit_script_sha256': sha(__file__)}
    (base / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    check(contract['source_sha256'])
    artifacts = [f for f in base.rglob('*') if f.is_file() and f.name != 'completion_manifest.json'
                 and f.suffix in ['.json', '.jsonl', '.csv', '.html', '.png', '.pdf', '.svg']
                 and 'diagnostics' not in f.relative_to(base).parts]
    completion = {'status': 'complete', 'conditions': 12, 'questions_per_condition': 1711,
                  'independent_audit_passed': True, 'control_model_calls': 0,
                  'artifact_sha256': {str(f): sha(f) for f in artifacts}}
    (base / 'completion_manifest.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    audit(parser.parse_args().output.resolve())
