"""Independently audit stored ablation scores, payloads, answers and metrics."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def zscore(values):
    if len(values) == 1:
        return values.copy()
    centered = values - np.mean(values)
    sd = np.sqrt(np.mean(centered ** 2))
    tolerance = 32 * np.finfo(float).eps * max(1., max(abs(values)))
    if sd <= tolerance:
        return np.zeros_like(values)
    return centered / sd


def audit(base):
    completion = read(base / 'completion_manifest.json')
    protocol = read(base / 'reader/protocol.json')
    retrieval_protocol = read(base / 'retrieval_protocol.json')
    summary = read(base / 'summary.json')
    signatures = dict(completion['artifact_sha256'])
    signatures.update(protocol['source_sha256'])
    signatures.update(protocol['model_source_sha256'])
    signatures.update(retrieval_protocol['source_sha256'])
    for file, expected in signatures.items():
        assert sha(file) == expected, file
    unsigned = dict(protocol)
    experiment_signature = unsigned.pop('experiment_signature')
    assert digest(unsigned) == experiment_signature
    assert protocol['reader_settings']['seed'] == 42
    assert protocol['reader_settings']['do_sample'] is False
    assert protocol['lambdas'] == [.001, .003, .01, .03, .1] and protocol['alpha'] == 1.
    specs = protocol['condition_specs']
    assert len(specs) == 12
    retrieval = lines(base / 'retrieval_results.jsonl')
    qa = lines(base / 'reader/qa_manifest.jsonl')
    responses = lines(base / 'reader/responses.jsonl')
    original = {q['id']:q for q in lines(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')}
    memories = {m['id']:m['reader_text'] for m in lines(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl')}
    assert len(retrieval) == len(original) == 371
    assert len({r['question_id'] for r in retrieval}) == 371
    ranks = {}
    score_checks = 0
    max_z_error = max_correction_error = 0.
    for row in retrieval:
        assert row['question_id'] in original
        assert row['domain'] == original[row['question_id']]['task']
        for method, data in row['methods'].items():
            ids = data['memory_ids']
            raw = np.array(data['raw'])
            assert len(raw) == len(set(ids))
            assert sorted(i for group in data['groups'] for i in group['indices']) == list(range(len(raw)))
            for group in data['groups']:
                ix = group['indices']
                assert len(ix) == group['size']
                if method == 'baseline_question_only':
                    assert len(data['groups']) == 1
                raw_std = float(np.std(raw[ix]))
                assert abs(raw_std - group['raw_std']) < 1e-12
                for ridge, values in data['weighting'].items():
                    enhanced = np.array(values['enhanced'])[ix]
                    stored_z = np.array(values['z'])[ix]
                    correction = np.array(values['correction'])[ix]
                    weighted = np.array(values['weighted'])[ix]
                    expected_z = zscore(enhanced)
                    err = float(np.max(abs(stored_z - expected_z)))
                    assert err < 1e-10, (row['question_id'], method, ridge, err)
                    max_z_error = max(max_z_error, err)
                    expected_correction = np.zeros(len(ix))
                    if len(ix) > 1 and group['raw_status'] != 'zero_variance':
                        expected_correction = raw_std * np.tanh(expected_z)
                    err = float(np.max(abs(correction - expected_correction)))
                    assert err < 1e-12
                    max_correction_error = max(max_correction_error, err)
                    assert np.max(abs(correction)) <= raw_std + 1e-12
                    assert np.allclose(weighted, raw[ix] + correction, rtol=0, atol=1e-12)
                    score_checks += 1
            for name, spec in specs.items():
                if spec['method'] != method:
                    continue
                scores = raw if spec['lambda'] is None else np.array(data['weighting'][str(spec['lambda'])]['weighted'])
                order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))[:5]
                expected_ids = [ids[i] for i in order]
                actual = row['conditions'][name]
                assert actual['top5_memory_ids'] == expected_ids
                assert np.allclose(actual['top5_scores'], scores[order], rtol=0, atol=1e-12)
                ranks[row['question_id'] + '::' + name] = expected_ids
    assert score_checks == retrieval_protocol['independent_positive_covariance_groups']
    manifest = {r['record_id']:r for r in qa}
    answered = {r['record_id']:r for r in responses}
    assert len(qa) == len(responses) == len(manifest) == len(answered) == len(ranks) == 4452
    assert set(manifest) == set(answered) == set(ranks)
    fresh = {}
    correct = Counter()
    hits = Counter()
    totals = Counter()
    counts = Counter()
    for record_id, record in manifest.items():
        question = original[record['id']]
        assert record['top_memory_ids'] == ranks[record_id]
        assert all(record[k] == question[k] for k in ['gold_choice','gold_ids','question','options','query_image_paths'])
        text = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        text += '\n\n'.join(f'Memory {i}: {memories[mid]}' for i,mid in enumerate(ranks[record_id], 1))
        text += '\n\nQuestion:\n' + question['question'] + '\n\nOptions:\n'
        text += '\n'.join(f'{choice}. {question["options"][choice]}' for choice in 'ABCD')
        text += '\n\nAnswer with one letter only.'
        assert text == record['reader_text'], record_id
        images = [{'path':path,'sha256':sha(path)} for path in question['query_image_paths']]
        assert images == record['query_images']
        payload = {'system':protocol['reader_settings']['reader_system_prompt'],'user_text':text,'query_images':images}
        assert digest(payload) == record['payload_sha256']
        response = answered[record_id]
        assert response['payload_sha256'] == record['payload_sha256']
        assert response['experiment_signature'] == experiment_signature
        assert response['run_id'] == protocol['experiment_run_id']
        assert not response['historical_response_reuse']
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',response['raw_response'].strip(),re.I)
        assert match and response['status'] == 'ok' and not response['hit_token_limit']
        predicted = match.group(1).upper()
        is_correct = predicted == question['gold_choice']
        assert response['predicted_choice'] == predicted and response['correct'] == is_correct
        if response['fresh_inference_executed_for_this_record']:
            assert response['payload_sha256'] not in fresh
            fresh[response['payload_sha256']] = response
        else:
            source = answered[response['same_run_shared_inference_record_id']]
            assert source['fresh_inference_executed_for_this_record']
            for key in ['payload_sha256','raw_response','generated_token_ids','execution_id']:
                assert response[key] == source[key]
        name = record['condition']
        counts[name] += 1
        correct[name] += is_correct
        hits[name] += len(set(record['top_memory_ids']) & set(question['gold_ids']))
        totals[name] += len(set(question['gold_ids']))
    assert len(fresh) == protocol['unique_reader_payloads'] == summary['fresh_inferences']
    for name in specs:
        metrics = summary['overall'][name]
        assert counts[name] == metrics['questions'] == 371
        assert totals[name] == metrics['gold_references'] == 1041
        assert correct[name] == metrics['correct'] and hits[name] == metrics['gold_hits_top5']
        assert correct[name] / 371 == metrics['accuracy']
        assert hits[name] / 1041 == metrics['recall_at_5_micro']
    audit_result = {'passed':True,'conditions':12,'questions_per_condition':371,'records':4452,
        'verified_file_hashes':len(signatures),'verified_rankings':len(ranks),
        'verified_group_lambda_corrections':score_checks,'max_z_error':max_z_error,
        'max_correction_error':max_correction_error,'fresh_inferences':len(fresh),
        'shared_records':4452-len(fresh),'payloads_reconstructed_from_canonical_inputs':4452,
        'format_errors':0,'token_limit_hits':0,'raw_answer_correctness_recomputed':True,
        'recall_from_gold_ids_recomputed':True,'audit_script_sha256':sha(__file__),
        'completion_manifest_sha256':sha(base/'completion_manifest.json')}
    (base/'independent_audit.json').write_text(json.dumps(audit_result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(audit_result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/memeye_qwen_positive_variance_lambda_ablation_20261107')
    audit(parser.parse_args().output)
