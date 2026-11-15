"""Independent input and ridge audit of the four DeepSeek Flash MemEye Open methods."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import memeye_deepseek_open_caption_inputs as src


def independent_scores(q, e):
    raw = e @ q
    x = e - e.mean(axis=0)
    wq = q if len(e) == 1 else q - x.T @ np.linalg.solve(x @ x.T + (len(e) - 1) * .03 * np.eye(len(e)), x @ q)
    white = e @ wq
    if len(e) == 1 or raw.std() <= np.finfo(float).eps * max(1., float(np.max(np.abs(raw)))) or white.std() <= np.finfo(float).eps * max(1., float(np.max(np.abs(white)))):
        return raw, raw
    return raw, raw + raw.std() * np.tanh((white - white.mean()) / white.std())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=src.DEFAULT)
    p.add_argument('--family', choices=['base', 'ours'], default='base'); args = p.parse_args(); root = args.root.resolve()
    prepare = src.read(root / 'preparation_protocol.json'); src.native.verify_hashes(prepare['source_sha256'])
    qs = src.rows(root / 'inputs/questions.jsonl'); canonical = src.native.unique(src.rows(src.CANONICAL))
    memories, old_questions, _ = src.native.attach_captions(src.rows(src.MEMORY / 'inputs/source_memories.jsonl'),
        src.rows(src.MEMORY / 'inputs/source_questions.jsonl'), src.rows(src.MEMORY / 'captions.jsonl'))
    old = src.native.unique(old_questions); memory = src.native.unique(memories)
    for q in qs:
        assert all(q[k] == canonical[q['id']][k] for k in ['id', 'task', 'point', 'question', 'query_image_paths'])
        assert q['query_image_captions'] == old[q['id']]['query_image_captions']
        assert not set(q) & {'answer', 'options', 'gold', 'gold_ids', 'gold_choice', 'clue'}
        for image in old[q['id']]['query_images']: assert src.sha(image['path']) == image['sha256']
    methods = ['semrag', 'ours_2'] if args.family == 'base' else ['ours_1', 'ours_1_ours_2']
    rankings = {m: src.native.unique(src.rows(root / m / 'retrieval.jsonl')) for m in methods}
    manifests = {m: src.native.unique(src.rows(root / m / 'reader_manifest.jsonl')) for m in methods}
    expected_ids = set(canonical)
    for method in methods:
        assert set(rankings[method]) == set(manifests[method]) == expected_ids
        protocol = src.read(root / method / 'protocol.json')
        src.native.verify_hashes(protocol['source_sha256'])
        src.native.verify_hashes(protocol['artifact_sha256'], root / method)
        assert protocol['original_mcq_correct'] == src.SPECS[method]['mcq_correct']
        assert protocol['caption_model'] == src.MODEL
    encoding = src.read(root / 'encoding_validation.json'); src.native.verify_hashes(encoding['artifact_sha256'])
    fresh_calls = []
    if args.family == 'ours':
        from transformers import AutoTokenizer
        from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
        candidates = src.native.unique(src.rows(root / 'candidates.jsonl'))
        sanitized = src.native.unique(src.rows(root / 'inputs/sanitized_candidate_inputs.jsonl'))
        assert set(candidates) == expected_ids
        tok = AutoTokenizer.from_pretrained(str(src.common.CLIP), local_files_only=True)
        candidate_protocol = src.read(root / 'candidate_generation_protocol.json')
        src.native.verify_hashes(candidate_protocol['contract']['source_sha256'])
        for q in qs:
            c = candidates[q['id']]; ci = sanitized[q['id']]
            assert ci == {'id': q['id'], 'question': q['question'], 'query_image_caption': '\n'.join(q['query_image_captions'])}
            assert c['source_input_sha256'] == src.digest(ci)
            path = root / 'candidate_calls' / (c['request_sha256'] + '.json')
            call = src.read(path)
            expected_wire = transport.build_request({'system': src.common.CANDIDATE_SYSTEM,
                'user_text': src.common.candidate_user_text(ci, c['attempt']), 'query_images': []}, model=src.MODEL, max_output_tokens=2048, image_detail='high')
            assert call['request'] == expected_wire and src.digest(expected_wire) == c['request_sha256']
            r = call['response']; assert r['returned_model'] == src.MODEL and r['finish_reason'] == 'stop' and not r.get('refusal')
            values, lengths = src.common.parse_candidates(r['raw_response'], tok)
            assert values == c['candidate_statements'] and lengths == c['clip_token_lengths']
            fresh_calls.append(str(path))
        bgeinfo = src.read(src.MEMORY / 'inputs/protocol.json')
        bge = src.native.load_text_encoder(bgeinfo['text_encoder'], 'cpu')
        clip, tokenizer, limit = src.native.load_clip_text_encoder(bgeinfo['image_encoder'], 'cpu')
        saved_groups = src.native.unique(src.rows(root / 'master_query_audit.jsonl'))
    checked = 0; maxscoreerror = 0.; changed = {m: 0 for m in methods}
    original = src.original_retrieval()
    for d in encoding['domains']:
        task = d['task']; selected = [q for q in qs if q['task'] == task]
        ids = d['memory_ids']
        with np.load(root / 'embeddings' / (task + '.npz')) as data:
            e = data['memory_vectors'].copy(); qv = data['open_query_vectors'].copy()
        if args.family == 'ours':
            texts = [q['retrieval_query'] + '\nCandidate answer: ' + s for q in selected for s in candidates[q['id']]['candidate_statements']]
            aug, _ = src.encode_complete(bge, texts, bgeinfo['query_prefix'], field='independent Open audit')
            aug = aug.reshape(len(selected), 4, -1)
            cv, _ = src.native._clip_candidates(clip, tokenizer, limit, [s for q in selected for s in candidates[q['id']]['candidate_statements']])
            cv = cv.reshape(len(selected), 4, -1)
            graph = src.read(src.MEMORY / 'domains' / task / 'memory/graph.json')
            sums = np.load(src.MEMORY / 'domains' / task / 'memory/master_sums.npy')
            centers = np.asarray([sums[m['visual_mean_ref']['row']] * m['visual_mean_ref']['scale'] for m in graph['masters']])
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        for qi, q in enumerate(selected):
            if args.family == 'base':
                raw, weighted = independent_scores(qv[qi], e)
                scores = {'semrag': raw, 'ours_2': weighted}
            else:
                scores = {m: np.empty(len(e)) for m in methods}
                similarity = centers @ cv[qi].T
                for mi, group in enumerate(graph['masters']):
                    indices = [i for i, n in enumerate(graph['image_nodes']) if n['master_id'] == group['id']]
                    choice = int(np.argmax(similarity[mi]))
                    audit = saved_groups[q['id']]['groups'][mi]
                    assert choice == audit['candidate_index']
                    assert candidates[q['id']]['candidate_statements'][choice] == audit['selected_statement']
                    raw, weighted = independent_scores(aug[qi, choice], e[indices])
                    scores['ours_1'][indices] = raw; scores['ours_1_ours_2'][indices] = weighted
            for method in methods:
                actual = rankings[method][q['id']]; manifest = manifests[method][q['id']]
                order = np.argsort(-scores[method], kind='stable'); ranked = [ids[i] for i in order]
                assert actual['full_rank_memory_ids'] == ranked and actual['top_memory_ids'] == ranked[:5]
                error = float(np.max(np.abs(np.asarray(actual['full_rank_scores']) - scores[method][order])))
                assert error < 2e-12; maxscoreerror = max(maxscoreerror, error)
                top = actual['top_memory_ids']; assert len(top) == len(set(top)) == 5
                context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                    f'Memory {rank}: {memory[mid]["retrieval_text"]}' for rank, mid in enumerate(top, 1))
                user = context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
                assert manifest['context'] == context and manifest['user_text'] == user and manifest['system'] == src.SYSTEM
                assert manifest['query_image_paths'] == q['query_image_paths']
                assert manifest['payload_sha256'] == src.payload_hash(src.SYSTEM, user, q['query_image_paths'])
                oldr = next(c for c in original[q['id']]['conditions'] if c['name'] == src.SPECS[method]['original_condition'])
                changed[method] += oldr['top5_memory_ids'] != top
                checked += 1
    result = {'passed': True, 'family': args.family, 'methods': methods, 'questions_per_method': 371,
        'rankings_and_reader_contexts_checked': checked, 'max_independent_score_error': maxscoreerror,
        'canonical_Open_questions_exact': True, 'DeepSeek_query_caption_verified': True, 'original_query_image_hashes_verified': True,
        'candidate_wires_validated': len(fresh_calls), 'source_revision': src.MODEL, 'gold_options_not_in_generation': True,
        'changed_ordered_Top5_from_source_MCQ': changed, 'audit_script_sha256': src.sha(__file__),
        'artifact_sha256': {str(root / m / name): src.sha(root / m / name) for m in methods for name in ['protocol.json', 'retrieval.jsonl', 'reader_manifest.jsonl']}}
    src.freeze(root / ('independent_' + args.family + '_audit.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
