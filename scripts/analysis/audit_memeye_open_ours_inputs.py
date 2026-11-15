"""Independently check frozen Open input payloads and dual-form z-score ranks."""
from pathlib import Path
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_ours_inputs import (
    DEFAULT, BASELINE, FROZEN, STORES, SPECS, SYSTEM, sha, read, rows, dump, digest, payload_hash,
)


def independent_weight(raw, vectors):
    centered = vectors - vectors.mean(0)
    if len(raw) == 1:
        return raw.copy()
    dual = centered @ centered.T + (len(raw) - 1) * .03 * np.eye(len(raw))
    white = raw - (vectors @ centered.T) @ np.linalg.solve(dual, raw - raw.mean())
    # This experiment's nondegenerate groups follow population normalization.
    raw_tolerance = 32 * np.finfo(float).eps * max(1., float(np.abs(raw).max()))
    white_tolerance = 32 * np.finfo(float).eps * max(1., float(np.abs(white).max()))
    if raw.std() <= raw_tolerance or white.std() <= white_tolerance:
        return raw.copy()
    z = (np.array([1., -1.]) * np.sign(white[0] - white[1]) if len(raw) == 2
         else (white - white.mean()) / white.std())
    return raw + raw.std() * np.tanh(z)


def main():
    questions = {r['id']: r for r in rows(DEFAULT / 'inputs/questions.jsonl')}
    baseline = {r['id']: r for r in rows(BASELINE / 'retrieval.jsonl')}
    candidates = {r['id']: r for r in rows(DEFAULT / 'candidates.jsonl')}
    audits = {r['id']: r for r in rows(DEFAULT / 'master_query_audit.jsonl')}
    expected_text = {r['id']: r['reader_text'] for r in rows(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl')}
    assert len(questions) == len(candidates) == len(audits) == 371
    candidate_protocol = read(DEFAULT / 'candidate_generation_protocol.json')
    assert digest(candidate_protocol['contract']) == candidate_protocol['experiment_signature']
    for section in (candidate_protocol['contract']['source_sha256'],
                    candidate_protocol['contract']['transport']['source_sha256']):
        for path, expected in section.items():
            assert sha(path) == expected, path
    sanitized = {r['id']: r for r in rows(DEFAULT / 'inputs/sanitized_candidate_inputs.jsonl')}
    for qid, q in questions.items():
        assert sanitized[qid] == {'id': qid, 'question': q['question'],
                                  'query_image_caption': '\n'.join(q['query_image_captions'])}
    attempts = rows(DEFAULT / 'candidate_attempts.jsonl')
    service_signature = candidate_protocol['contract']['transport']['service_signature']
    call_hashes, position = {}, 0
    while position < len(attempts):
        batch_size = attempts[position]['batch_size']
        batch = attempts[position:position + batch_size]
        assert len(batch) == batch_size
        assert len({r['attempt'] for r in batch}) == 1
        conversations = []
        for attempt in batch:
            text = common.candidate_user_text(sanitized[attempt['id']], attempt['attempt'])
            assert attempt['prompt_sha256'] == digest({'system': common.CANDIDATE_SYSTEM, 'user_text': text})
            conversations.append([{'role': 'system', 'content': common.CANDIDATE_SYSTEM},
                {'role': 'user', 'content': [{'type': 'text', 'text': text}]}])
        request = {'conversations': conversations, 'cap': 384, 'sampling_temperature': 0.,
                   'service_signature': service_signature}
        path = DEFAULT.parent / 'qwen_service/calls' / (digest(request) + '.json')
        call = read(path)
        assert call['request'] == request and call['request_sha256'] == digest(request)
        assert len(call['outputs']) == batch_size
        for expected, actual in zip(call['outputs'], batch):
            assert all(actual[key] == expected[key] for key in ('raw_response', 'input_tokens',
                'generated_tokens', 'hit_token_limit', 'batch_size'))
        call_hashes[str(path)] = sha(path)
        position += batch_size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(common.CLIP), local_files_only=True)
    successes = {r['id']: r for r in attempts if r['validation_error'] is None}
    assert set(successes) == set(candidates) == set(questions)
    for qid, candidate in candidates.items():
        success = successes[qid]
        statements, lengths = common.parse_candidates(success['raw_response'], tokenizer)
        assert candidate['candidate_statements'] == statements
        assert candidate['clip_token_lengths'] == lengths
        assert candidate['prompt_sha256'] == success['prompt_sha256']
        assert not success['hit_token_limit']
    counts, max_error = {}, 0.
    for method, spec in SPECS.items():
        output = DEFAULT / method
        protocol = read(output / 'protocol.json')
        for path, expected in protocol['source_sha256'].items():
            assert sha(path) == expected, path
        for path, expected in protocol['artifact_sha256'].items():
            assert sha(output / path) == expected, path
        retrieval = {r['id']: r for r in rows(output / 'retrieval.jsonl')}
        manifest = {r['id']: r for r in rows(output / 'reader_manifest.jsonl')}
        assert set(retrieval) == set(manifest) == set(questions)
        assert protocol['original_mcq_correct'] == spec['mcq_correct']
        checked = 0
        for task in dict.fromkeys(q['task'] for q in questions.values()):
            leaf = read(STORES / task / 'caption_visual_master/leaves.json')
            ids = [r['memory_id'] for r in leaf]
            with np.load(FROZEN / task / 'features.npz') as f:
                vectors = f['bge_leaf'].copy()
            for qid, q in questions.items():
                if q['task'] != task:
                    continue
                if method == 'ours_2':
                    raw_map = dict(zip(baseline[qid]['full_rank_memory_ids'], baseline[qid]['full_rank_scores']))
                    raw = np.array([raw_map[mid] for mid in ids])
                    scores = independent_weight(raw, vectors)
                else:
                    scores = np.empty(len(ids))
                    seen = []
                    for group in audits[qid]['groups']:
                        index = np.array([ids.index(mid) for mid in group['memory_ids']])
                        seen.extend(index)
                        raw = np.array(group['raw'])
                        weighted = independent_weight(raw, vectors[index])
                        max_error = max(max_error, float(np.max(np.abs(weighted - group['weighted']))))
                        assert np.max(np.abs(weighted - group['weighted'])) < 2e-12
                        assert group['candidate_index'] == int(np.argmax(group['clip_statement_scores']))
                        assert group['selected_statement'] == candidates[qid]['candidate_statements'][group['candidate_index']]
                        scores[index] = raw if method == 'ours_1' else weighted
                    assert sorted(seen) == list(range(len(ids)))
                order = np.argsort(-scores, kind='stable')
                r, m = retrieval[qid], manifest[qid]
                assert r['full_rank_memory_ids'] == [ids[i] for i in order]
                error = float(np.max(np.abs(scores[order] - r['full_rank_scores'])))
                max_error = max(max_error, error)
                assert error < 2e-6, (qid, error)
                assert r['top_memory_ids'] == r['full_rank_memory_ids'][:5] == m['top_memory_ids']
                assert len(m['top_memory_ids']) == len(set(m['top_memory_ids'])) == 5
                context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                    f'Memory {i}: {expected_text[mid]}' for i, mid in enumerate(m['top_memory_ids'], 1))
                assert m['context'] == context
                assert m['user_text'] == context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
                assert m['system'] == SYSTEM and m['question'] == q['question']
                assert m['query_image_paths'] == q['query_image_paths']
                assert m['payload_sha256'] == payload_hash(m['system'], m['user_text'], m['query_image_paths'])
                assert not set(m) & {'options', 'gold_ids', 'ground_truth', 'answer', 'clue', 'gold_choice'}
                checked += 1
        assert checked == 371
        counts[method] = checked
    result = {'passed': True, 'questions_checked': counts,
              'independent_dual_ridge_score_max_error': max_error,
              'source_and_artifact_hashes_verified': True,
              'exact_source_context_texts_verified': True,
              'all_payload_hashes_verified': True, 'all_contexts_top5': True,
              'candidate_selection_matches_clip_argmax': True,
              'candidate_questions_only_inputs_verified': True,
              'candidate_attempts': len(attempts),
              'candidate_generation_rpc_calls': len(call_hashes),
              'candidate_rpc_source_sha256': call_hashes,
              'judge_gold_not_in_model_input': True, 'audit_script_sha256': sha(Path(__file__))}
    dump(DEFAULT / 'independent_input_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'candidate_rpc_source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
