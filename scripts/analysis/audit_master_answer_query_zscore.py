"""Independently audit master-wise z weighting, reader inputs, and decoded answers."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import numpy as np

CONTROL, TREATMENT = 'master_answer_query', 'master_answer_query_zscore'
NAMES = (CONTROL, TREATMENT)
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021'
INPUTS = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/master_answer_query_zscore_20261023')
    base = parser.parse_args().output.resolve()
    method = read(base / 'protocol.json'); protocol = read(base / 'reader/protocol.json')
    signed = dict(protocol); signature = signed.pop('experiment_signature')
    assert digest(signed) == signature
    source_hashes = {**method['source_sha256'], **protocol['source_sha256'], **protocol['model_source_sha256']}
    for path, expected in source_hashes.items():
        assert sha(path) == expected, path
    assert (method['lambda'], method['alpha'], method['top_k']) == (.01, 1., 5)
    settings = protocol['reader_settings']
    assert settings == read(REFERENCE / 'reader/protocol.json')['reader_settings']
    assert not settings['do_sample'] and settings['seed'] == 42 and settings['max_new_tokens'] == 16
    original = {q['id']: q for q in rows(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl')}
    memories = {r['id']: r['reader_text'] for r in rows(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl')}
    retrieval = rows(base / 'retrieval_results.jsonl')
    manifest = rows(base / 'reader/qa_manifest.jsonl'); indexed = {r['record_id']: r for r in manifest}
    previous = {r['record_id']: r for r in rows(REFERENCE / 'reader/qa_manifest.jsonl')}
    assert len(retrieval) == len(original) == 371 and {r['question_id'] for r in retrieval} == set(original)
    assert len(manifest) == len(indexed) == 742
    assert Counter(r['condition'] for r in manifest) == Counter(dict.fromkeys(NAMES, 371))

    # Reconstruct per-master scores via covariance inversion on historical
    # vectors, independently of the online SVD implementation.
    frozen_hashes = read(REFERENCE / 'protocol.json')['source_sha256']
    historical = {r['question_id']: r for r in rows(REFERENCE / 'retrieval_results.jsonl')}
    reconstructions = {}; feature_hashes = {}; source_leaves = {}; source_queries = {}
    for domain in sorted({q['task'] for q in original.values()}):
        directory = INPUTS / domain; source = read(directory / 'inputs.json')
        path = directory / 'features.npz'; feature_hashes[str(path)] = sha(path)
        assert feature_hashes[str(path)] == frozen_hashes[str(path)]
        augmented = REFERENCE / 'domains' / domain / 'score_matrices.npz'
        feature_hashes[str(augmented)] = sha(augmented)
        with np.load(path) as features:
            x = features['bge_leaf'].astype(np.float64)
        with np.load(augmented) as features:
            q = features['bge_augmented_query_vectors'].astype(np.float64)
            choices = features['selected_choice_indices'].copy()
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=2, keepdims=True)
        raw = np.empty((len(q), len(x))); white = np.empty_like(raw)
        for mi, master in enumerate(source['masters']):
            indices = np.array([i for i, leaf in enumerate(source['leaves']) if leaf['master_id'] == master['id']])
            xx = x[indices]; qq = q[np.arange(len(q)), choices[:, mi]]
            assert len(indices) == master['n']
            r = qq @ xx.T
            if len(xx) == 1:
                w = r.copy()
            else:
                centered = xx - xx.mean(0); covariance = centered.T @ centered / (len(xx) - 1)
                w = (xx @ np.linalg.solve(covariance + .01 * np.eye(x.shape[1]), .01 * qq.T)).T
            raw[:, indices] = r; white[:, indices] = w
            for qi, question in enumerate(source['questions']):
                old = next(m for m in historical[question['id']]['masters'] if m['master_id'] == master['id'])
                assert 'ABCD'[choices[qi, mi]] == old['selected_choice']
        for i, question in enumerate(source['questions']):
            qid = question['id']; reconstructions[qid] = (raw[i], white[i])
            source_queries[qid] = question['retrieval_query']
            np.testing.assert_allclose(raw[i], [l['expanded_score'] for l in historical[qid]['leaves']], rtol=0, atol=1e-12)
        source_leaves[domain] = source['leaves']
    assert set(reconstructions) == set(original)

    hits = Counter(); max_errors = Counter(); candidates_checked = groups_checked = 0; size_counts = Counter()
    for item in retrieval:
        qid = item['question_id']; q = original[qid]; leaves = item['leaf_scores']
        domain_leaves = source_leaves[q['task']]; prior = historical[qid]
        assert [v['memory_id'] for v in leaves] == [v['memory_id'] for v in domain_leaves]
        assert len(leaves) == len({v['memory_id'] for v in leaves})
        assert item['evaluation'] == {k: q[k] for k in ('gold_ids', 'gold_choice')}
        assert item['retrieval_query'] == source_queries[qid]
        prior_masters = {m['master_id']: m for m in prior['masters']}
        assert {g['group_id'] for g in item['groups']} == set(prior_masters)
        raw, white, z, correction, weighted = [np.array([v[key] for v in leaves]) for key in
                                             ('raw_score', 'white_score', 'white_z', 'correction', 'weighted_score')]
        for key, a, b in zip(('raw', 'white'), (raw, white), reconstructions[qid]):
            error = float(np.max(abs(a - b))); max_errors[key] = max(max_errors[key], error)
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-6)
        for group in item['groups']:
            old = prior_masters[group['group_id']]
            indices = np.array([i for i, leaf in enumerate(leaves) if leaf['group_id'] == group['group_id']])
            assert len(indices) == group['size'] == old['n']
            assert {leaves[i]['image_id'] for i in indices} == set(old['member_ids'])
            assert group['query'] == old['augmented_query'] and group['choice'] == old['selected_choice']
            assert group['selected_statement'] == old['selected_statement']
            assert group['choice'] == max('ABCD', key=lambda key: group['clip_statement_scores'][key])
            r, w = raw[indices], white[indices]
            assert abs(r.std() - group['raw_std']) < 1e-14
            assert abs(w.std() - group['white_std']) < 1e-14
            expected_correction = np.zeros(len(indices))
            if len(indices) == 1:
                expected_z = w.copy(); assert group['z_status'] == 'singleton_raw'
            else:
                tolerance = 32 * np.finfo(float).eps * max(1., float(abs(w).max()))
                if w.std() <= tolerance:
                    expected_z = np.zeros(len(w)); assert group['z_status'] == 'zero_variance'
                elif len(indices) == 2:
                    expected_z = np.array([1., -1.]) * np.sign(w[0] - w[1])
                else:
                    shifted = w - w[0]; centered = shifted - shifted.mean()
                    expected_z = centered / np.sqrt(np.mean(centered**2))
                raw_tolerance = 32 * np.finfo(float).eps * max(1., float(abs(r).max()))
                if r.std() > raw_tolerance and w.std() > tolerance:
                    expected_correction = r.std() * np.tanh(expected_z)
            allowance = max(1e-12, 64*np.finfo(float).eps*max(1., float(abs(w).max()))/max(w.std(), 1e-10))
            np.testing.assert_allclose(z[indices], expected_z, rtol=0, atol=allowance)
            np.testing.assert_allclose(correction[indices], expected_correction, rtol=0, atol=1e-13)
            groups_checked += 1; size_counts[str(len(indices))] += 1
        np.testing.assert_allclose(weighted, raw + correction, rtol=0, atol=1e-15)
        candidates_checked += len(leaves)
        for c, name, scores in zip(item['conditions'], NAMES, (raw, weighted)):
            assert c['name'] == name
            ids = [leaves[i]['memory_id'] for i in np.argsort(-scores, kind='stable')[:5]]
            assert ids == c['top5_memory_ids'] and len(set(ids)) == 5
            assert c['gold_hits_top5'] == len(set(ids) & set(q['gold_ids']))
            hits[name] += c['gold_hits_top5']
            for evidence, mid in zip(c['top5'], ids):
                assert evidence['memory_id'] == mid and evidence['reader_text'] == memories[mid]
            text = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
            text += '\n\n'.join(f'Memory {i}: {memories[mid]}' for i, mid in enumerate(ids, 1))
            text += '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
            text += '\n'.join(k + '. ' + q['options'][k] for k in 'ABCD') + '\n\nAnswer with one letter only.'
            images = [{'path': path, 'sha256': sha(path)} for path in q['query_image_paths']]
            payload = {'system': settings['reader_system_prompt'], 'user_text': text, 'query_images': images}
            source = indexed[qid + '::' + name]
            for key in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths'):
                assert source[key] == q[key]
            assert source['top_memory_ids'] == ids and source['reader_text'] == text
            assert source['query_images'] == images and source['payload_sha256'] == digest(payload)
            if name == CONTROL:
                assert previous[source['record_id']]['payload_sha256'] == digest(payload)
    assert groups_checked == 3862
    for path, expected in feature_hashes.items(): assert sha(path) == expected

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(settings['reader_model_path'], local_files_only=True)
    responses = rows(base / 'reader/responses.jsonl'); by_id = {r['record_id']: r for r in responses}
    assert len(responses) == len(by_id) == 742 and set(by_id) == set(indexed)
    correct = Counter(); fresh = set(); shared = 0
    for r in responses:
        source = indexed[r['record_id']]; q = original[r['question_id']]
        assert r['run_id'] == protocol['experiment_run_id'] and r['experiment_signature'] == signature
        assert r['payload_sha256'] == source['payload_sha256'] and not r['historical_response_reuse']
        raw = processor.batch_decode([r['generated_token_ids']], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw, re.I)
        assert raw == r['raw_response'] and match and r['status'] == 'ok' and not r['hit_token_limit']
        choice = match.group(1).upper()
        assert choice == r['predicted_choice'] and r['gold_choice'] == q['gold_choice']
        assert r['correct'] == (choice == q['gold_choice']); correct[r['condition']] += r['correct']
        content = [{'type': 'image'} for _ in source['query_image_paths']]
        content.append({'type': 'text', 'text': source['reader_text']})
        rendered = processor.apply_chat_template([{'role': 'system', 'content': settings['reader_system_prompt']},
            {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        assert hashlib.sha256(rendered.encode()).hexdigest() == r['rendered_prompt_sha256']
        assert r['query_image_count'] == len(source['query_image_paths']) and r['batch_size'] == 1
        if r['fresh_inference_executed_for_this_record']:
            assert r['payload_sha256'] not in fresh; fresh.add(r['payload_sha256'])
        else:
            origin = by_id[r['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            for key in ('payload_sha256', 'generated_token_ids', 'raw_response', 'execution_id'):
                assert r[key] == origin[key]
            shared += 1
    assert len(fresh) == protocol['unique_reader_payloads'] and len(fresh) + shared == 742
    historical = [r for r in rows(REFERENCE / 'reader/responses.jsonl') if r['condition'] == CONTROL]
    assert len(historical) == 371
    assert sum(r['predicted_choice'] == original[r['question_id']]['gold_choice'] for r in historical) == 195
    assert all(r['predicted_choice'] == by_id[r['record_id']]['predicted_choice'] for r in historical)
    summary = read(base / 'reader/summary.json')
    for name in NAMES:
        assert correct[name] == summary['overall']['conditions'][name]['correct']
        assert hits[name] == summary['overall']['conditions'][name]['gold_hits_top5']
    environment = read(base / 'reader/generation_environment.json')
    assert environment['parameter_devices'] == ['cuda:1'] and environment['input_device'] == 'cuda:1'
    assert environment['dtype'] == 'torch.bfloat16'
    assert environment['processor_size'] == {'shortest_edge': 65536, 'longest_edge': settings['query_image_max_pixels']}
    assert sum(bool(r['query_image_paths']) for r in manifest) == 16
    assert not list(base.rglob('*.npy')) and not list(base.rglob('*.npz'))
    audit = {'passed': True, 'questions': 371, 'logical_records': 742, 'candidate_scores_checked': candidates_checked,
        'master_memberships_and_queries_verified': True, 'master_question_groups': groups_checked, 'group_size_counts': dict(size_counts), 'query_text_and_reader_evidence_unchanged': True,
        'independent_whitening': 'Frozen CPU BGE queries and dialogue-caption vectors; .01 * (master covariance + .01 I)^(-1); audit only.',
        'max_cpu_frozen_vs_online_gpu_score_errors': dict(max_errors), 'independent_feature_sha256': feature_hashes,
        'z_correction_and_top5_reconstructed': True, 'historical_baseline_payload_and_answer_matches': 371,
        'raw_token_decodings_and_rendered_prompts_verified': 742, 'correct': dict(correct), 'gold_hits_top5': dict(hits),
        'fresh_inferences': len(fresh), 'same_run_shared_records': shared, 'native_image_records': 16,
        'historical_response_reuse': False, 'format_errors': 0, 'token_limit_hits': 0,
        'device': 'cuda:1', 'audit_script_sha256': sha(__file__)}
    (base / 'reader/independent_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == '__main__':
    main()
