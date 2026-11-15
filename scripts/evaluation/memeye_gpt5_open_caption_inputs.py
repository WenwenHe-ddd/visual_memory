"""Frozen GPT-5 caption methods on official, answer-free MemEye Open questions."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for k, v in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
             'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(k, v)
import numpy as np
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM, payload_hash
from scripts.memory import memeye_backbone_retrieval as native
from scripts.memory.memeye_backbone_long_text import encode_complete, POLICY
from scripts.memory.online_zscore import ZScoreWeighting

DEFAULT = ROOT / 'results/memeye_gpt5_open_20261105/caption_methods'
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/questions.jsonl'
SOURCE = ROOT / 'results/memeye_gpt5_qwen_five_validation_20261024'
MEMORY = ROOT / 'results/memeye_gpt5_base_seed42_full_20261024'
MODEL = 'gpt-5-2025-08-07'
SPECS = {
    'semrag': {'original_condition': 'baseline_question_only', 'mcq_correct': 217, 'lambda': None, 'alpha': 0.},
    'ours_1': {'original_condition': 'master_answer_query', 'mcq_correct': 221, 'lambda': None, 'alpha': 0.},
    'ours_2': {'original_condition': 'baseline_question_only__l0p03_a1', 'mcq_correct': 217, 'lambda': .03, 'alpha': 1.},
    'ours_1_ours_2': {'original_condition': 'master_answer_query__l0p03_a1', 'mcq_correct': 222, 'lambda': .03, 'alpha': 1.},
}
read, rows, sha, dump, digest = native.read, native.lines, native.sha, native.dump, native.digest


def freeze(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Frozen JSON changed: {path}'
    else:
        dump(path, value)


def checked_weighting(query, vectors):
    out = ZScoreWeighting(.03, 1.).apply(query, vectors)
    centered = vectors - vectors.mean(axis=0)
    adjusted = query if len(vectors) == 1 else query - centered.T @ np.linalg.solve(
        centered @ centered.T + (len(vectors) - 1) * .03 * np.eye(len(vectors)), centered @ query)
    err = float(np.max(np.abs(vectors @ adjusted - out['white'])))
    assert err < 3e-12, err
    return out, err


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'preparation_protocol.json'
    if marker.exists():
        p = read(marker)
        native.verify_hashes(p['source_sha256']); native.verify_hashes(p['artifact_sha256'])
        return p
    canonical = rows(CANONICAL)
    assert len(canonical) == len(native.unique(canonical)) == 371
    summary, old_answers = read(SOURCE / 'summary.json'), rows(SOURCE / 'reader/responses.jsonl')
    gold = native.unique(rows(SOURCE / 'reader/inputs/gold.jsonl'))
    assert summary['returned_model'] == summary['caption_model'] == MODEL
    identity = {}
    for method, spec in SPECS.items():
        selected = [r for r in old_answers if r['condition'] == spec['original_condition']]
        assert len(selected) == 371
        correct = sum(r['predicted_choice'] == gold[r['id']]['gold_choice'] for r in selected)
        assert correct == spec['mcq_correct'] == summary['overall'][spec['original_condition']]['correct']
        assert all(r['returned_model'] == MODEL and r['status'] == 'ok' and r['finish_reason'] == 'stop' for r in selected)
        identity[method] = {'source_root': str(SOURCE), **spec, 'questions': 371,
            'source_recall_at5_micro': summary['overall'][spec['original_condition']]['gold_hits_top5'] / 1041}
    memories, old_qs, model = native.attach_captions(rows(MEMORY / 'inputs/source_memories.jsonl'),
        rows(MEMORY / 'inputs/source_questions.jsonl'), rows(MEMORY / 'captions.jsonl'))
    assert model == MODEL and len(memories) == 438
    old_map = native.unique(old_qs)
    questions = []
    for q in canonical:
        old = old_map[q['id']]
        assert old['task'] == q['task'] and old['query_image_paths'] == q['query_image_paths']
        caps = old['query_image_captions']
        query = q['question'] + (native.QUERY_IMAGE_SEPARATOR + ' '.join(caps) if caps else '')
        questions.append({k: q[k] for k in ('id', 'task', 'point', 'question', 'query_image_paths')} |
            {'query_image_captions': caps, 'retrieval_query': query})
    assert sum(bool(q['query_image_paths']) for q in questions) == 8
    sanitized = [{'id': q['id'], 'question': q['question'], 'query_image_caption': '\n'.join(q['query_image_captions'])}
                 for q in questions]
    sources = [Path(__file__), CANONICAL, SOURCE / 'summary.json', SOURCE / 'reader/responses.jsonl',
        SOURCE / 'reader/inputs/gold.jsonl', SOURCE / 'reader/protocol.json', SOURCE / 'retrieval_protocol.json',
        SOURCE / 'retrieval_results.jsonl', SOURCE / 'independent_audit.json', MEMORY / 'protocol.json',
        MEMORY / 'inputs/protocol.json', MEMORY / 'inputs/source_memories.jsonl', MEMORY / 'inputs/source_questions.jsonl',
        MEMORY / 'captions.jsonl', MEMORY / 'caption_generation/fresh_responses.jsonl',
        MEMORY / 'retrieval_protocol.json', MEMORY / 'retrieval_results.jsonl', MEMORY / 'offline/memories.jsonl',
        Path(native.__file__), ROOT / 'scripts/memory/memeye_backbone_long_text.py',
        ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py',
        ROOT / 'scripts/memory/caption_visual_master.py', Path(common.__file__),
        ROOT / 'scripts/evaluation/memeye_open_semrag_inputs.py']
    old_proof = read(SOURCE / 'independent_audit.json'); assert old_proof['passed']
    for name, expected in old_proof['artifact_sha256'].items():
        path = SOURCE / name
        assert sha(path) == expected, str(path)
    masters = 0
    for task in dict.fromkeys(q['task'] for q in questions):
        directory = MEMORY / 'domains' / task
        build = read(directory / 'build_protocol.json')
        native.verify_hashes(build['artifact_sha256'])
        sources += [directory / 'build_protocol.json', directory / 'memory/graph.json', directory / 'memory/master_sums.npy']
        masters += len(read(directory / 'memory/graph.json')['masters'])
    assert masters == 72
    artifact_paths = [root / 'inputs/questions.jsonl', root / 'inputs/sanitized_candidate_inputs.jsonl', root / 'source_mapping.json']
    native.write_lines(artifact_paths[0], questions); native.write_lines(artifact_paths[1], sanitized)
    freeze(artifact_paths[2], identity)
    protocol = {'dataset': 'MemEye', 'evaluation_type': 'open_ended', 'questions': 371, 'memory_leaves': 438,
        'masters': 72, 'top_k': 5, 'caption_model': MODEL, 'requested_model': 'gpt-5', 'source_mcq': identity,
        'source_memory_root': str(MEMORY), 'source_mcq_root': str(SOURCE),
        'query_policy': 'Official Open question plus cached GPT-5 question-image caption using original separator. No Qwen captions.',
        'candidate_policy': 'Four fresh GPT-5 hypotheses from Open question and cached GPT question-image caption only; shared by ours_1 and ours_1_ours_2. No MCQ options, gold, history or retrieved evidence.',
        'reader_policy': 'Top5 original dialogue plus complete GPT-5 captions; Open question and original query images. Candidates are not reader evidence.',
        'unchanged_question_text_count': sum(q['question'] == old_map[q['id']]['question'] for q in questions),
        'source_sha256': {str(p): sha(p) for p in sources},
        'artifact_sha256': {str(p): sha(p) for p in artifact_paths}}
    freeze(marker, protocol)
    print(json.dumps({'stage': 'prepare', 'questions': 371, 'source_mcq': identity}), flush=True)
    return protocol


def encode(root, device):
    import torch
    torch.set_num_threads(4)
    preparation = prepare(root)
    marker = root / 'encoding_validation.json'
    if marker.exists():
        out = read(marker); native.verify_hashes(out['artifact_sha256']); return out
    info = read(MEMORY / 'inputs/protocol.json')
    memories, old_qs, _ = native.attach_captions(rows(MEMORY / 'inputs/source_memories.jsonl'),
        rows(MEMORY / 'inputs/source_questions.jsonl'), rows(MEMORY / 'captions.jsonl'))
    new_qs = rows(root / 'inputs/questions.jsonl')
    raw_ref = native.unique(rows(MEMORY / 'retrieval_results.jsonl'))
    weighted_ref = native.unique(rows(SOURCE / 'retrieval_results.jsonl'), 'question_id')
    bge = native.load_text_encoder(info['text_encoder'], device)
    clip, tokenizer, limit = native.load_clip_text_encoder(info['image_encoder'], device)
    assert int(bge.max_seq_length) == 512
    checks = 0; max_error = 0.; max_dual = 0.; reports = []; paths = []
    for task in dict.fromkeys(q['task'] for q in new_qs):
        ms = [m for m in memories if m['task'] == task]
        oq = [q for q in old_qs if q['task'] == task]
        nq = [q for q in new_qs if q['task'] == task]
        assert [q['id'] for q in oq] == [q['id'] for q in nq]
        graph = read(MEMORY / 'domains' / task / 'memory/graph.json')
        sums = np.load(MEMORY / 'domains' / task / 'memory/master_sums.npy', allow_pickle=False)
        vectors, einfo = encode_complete(bge, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='dialogue+caption')
        oldtexts = [t for q in oq for t in [q['retrieval_query'], *[q['retrieval_query'] + '\nCandidate answer: ' + q['candidate_statements'][k] for k in 'ABCD']]]
        oldvectors, qinfo = encode_complete(bge, oldtexts, info['query_prefix'], field='MCQ verification queries')
        oldvectors = oldvectors.reshape(len(oq), 5, -1)
        oldclip, cinfo = native._clip_candidates(clip, tokenizer, limit, [q['candidate_statements'][k] for q in oq for k in 'ABCD'])
        oldclip = oldclip.reshape(len(oq), 4, -1)
        for index, q in enumerate(oq):
            result = native.rank_question(q, ms, graph, sums, oldvectors[index], vectors, oldclip[index])
            source = raw_ref[q['id']]
            for actual, expected in zip(result['conditions'], source['conditions']):
                assert actual['full_rank_memory_ids'] == expected['full_rank_memory_ids'], (q['id'], actual['name'])
                assert actual['reader_context'] == expected['reader_context']
                checks += 1
            for field in ('baseline_score', 'expanded_score'):
                error = max(abs(a[field] - b[field]) for a, b in zip(result['leaves'], source['leaves']))
                assert error < 2e-6, (q['id'], field, error)
                max_error = max(max_error, error)
            selected = {m['master_id']: 'ABCD'.index(m['selected_choice']) for m in result['masters']}
            values = np.empty(len(ms))
            for group in graph['masters']:
                indices = [i for i, n in enumerate(graph['image_nodes']) if n['master_id'] == group['id']]
                adjusted, error = checked_weighting(oldvectors[index, 1 + selected[group['id']]], vectors[indices])
                max_dual = max(max_dual, error); values[indices] = adjusted['weighted']
            global_scores, error = checked_weighting(oldvectors[index, 0], vectors)
            max_dual = max(max_dual, error)
            for name, scores in [('baseline_question_only__l0p03_a1', global_scores['weighted']), ('master_answer_query__l0p03_a1', values)]:
                expected = next(c for c in weighted_ref[q['id']]['conditions'] if c['name'] == name)
                order = np.argsort(-scores, kind='stable')
                assert [ms[i]['id'] for i in order] == expected['full_rank_memory_ids'], (q['id'], name)
                checks += 1
        # Reuse source-reproduced vectors for textually identical questions. Fresh Open-only queries otherwise.
        changed = [i for i, (o, n) in enumerate(zip(oq, nq)) if o['retrieval_query'] != n['retrieval_query']]
        fresh, freshinfo = encode_complete(bge, [nq[i]['retrieval_query'] for i in changed], info['query_prefix'],
            ids=[nq[i]['id'] for i in changed], field='Open query') if changed else ([], {})
        openvectors = oldvectors[:, 0].copy()
        for i, vector in zip(changed, fresh): openvectors[i] = vector
        directory = root / 'embeddings'; directory.mkdir(exist_ok=True)
        path = directory / (task + '.npz')
        assert not path.exists(), 'Unsealed encoding files require inspection before resume'
        np.savez(path, memory_vectors=vectors, old_query_vectors=oldvectors, open_query_vectors=openvectors)
        paths.append(path)
        reports.append({'task': task, 'leaves': len(ms), 'masters': len(graph['masters']), 'leaf': einfo,
            'old_query': qinfo, 'old_clip': cinfo, 'open_query': freshinfo, 'newly_encoded_open_queries': len(changed),
            'memory_ids': [m['id'] for m in ms], 'question_ids': [q['id'] for q in nq]})
        print(json.dumps({'stage': 'encode_and_reproduce_source', 'task': task, 'verified_rankings': checks}), flush=True)
    assert checks == 1484
    native.verify_hashes(preparation['source_sha256'])
    result = {'passed': True, 'source_mcq_rankings_reproduced': checks, 'max_source_score_error': max_error,
        'max_independent_dual_ridge_error': max_dual, 'encoding_policy': POLICY, 'device': device,
        'memory_rebuilt': False, 'old_images_reencoded': False, 'source_vectors_persisted': False,
        'vector_policy': 'Re-encode frozen GPT-5 texts using identical frozen BGE weights and original limit; verify all original rankings and persist once for this Open run.',
        'domains': reports, 'artifact_sha256': {str(p): sha(p) for p in paths}}
    freeze(marker, result)
    return result


def retrieve(root, family, device):
    import torch
    torch.set_num_threads(4)
    preparation = prepare(root); encoding = encode(root, device)
    questions = rows(root / 'inputs/questions.jsonl')
    memories, _, _ = native.attach_captions(rows(MEMORY / 'inputs/source_memories.jsonl'),
        rows(MEMORY / 'inputs/source_questions.jsonl'), rows(MEMORY / 'captions.jsonl'))
    info = read(MEMORY / 'inputs/protocol.json')
    methods = ['semrag', 'ours_2'] if family == 'base' else ['ours_1', 'ours_1_ours_2']
    sources = dict(preparation['source_sha256'])
    sources.update({str(p): sha(p) for p in (root / 'preparation_protocol.json', root / 'encoding_validation.json')})
    sources.update(encoding['artifact_sha256'])
    if family == 'ours':
        proof = read(root / 'candidate_validation.json'); assert proof['passed'] and proof['questions'] == 371
        assert proof['candidates_sha256'] == sha(root / 'candidates.jsonl')
        candidates = native.unique(rows(root / 'candidates.jsonl'))
        sanitized = native.unique(rows(root / 'inputs/sanitized_candidate_inputs.jsonl'))
        for qid, c in candidates.items():
            assert c['source_input_sha256'] == digest(sanitized[qid]) and c['gold_or_history_used'] is False
        sources.update({str(p): sha(p) for p in (root / 'candidates.jsonl', root / 'candidate_validation.json', root / 'candidate_generation_protocol.json')})
        bge = native.load_text_encoder(info['text_encoder'], device)
        clip, tokenizer, limit = native.load_clip_text_encoder(info['image_encoder'], device)
    outputs = {m: [] for m in methods}; manifests = {m: [] for m in methods}; audits = []; extra_encodings = []
    max_dual = 0.
    for domain_info in encoding['domains']:
        task = domain_info['task']; ms = [m for m in memories if m['task'] == task]; qs = [q for q in questions if q['task'] == task]
        ids = [m['id'] for m in ms]
        assert ids == domain_info['memory_ids'] and [q['id'] for q in qs] == domain_info['question_ids']
        with np.load(root / 'embeddings' / (task + '.npz')) as f:
            vectors = f['memory_vectors'].copy(); base = f['open_query_vectors'].copy()
        if family == 'ours':
            texts = [q['retrieval_query'] + '\nCandidate answer: ' + s for q in qs for s in candidates[q['id']]['candidate_statements']]
            expanded, extra = encode_complete(bge, texts, info['query_prefix'], field='Open candidate-augmented query')
            expanded = expanded.reshape(len(qs), 4, -1)
            clipvectors, cinfo = native._clip_candidates(clip, tokenizer, limit, [s for q in qs for s in candidates[q['id']]['candidate_statements']])
            clipvectors = clipvectors.reshape(len(qs), 4, -1)
            graph = read(MEMORY / 'domains' / task / 'memory/graph.json')
            sums = np.load(MEMORY / 'domains' / task / 'memory/master_sums.npy', allow_pickle=False)
            centers = native.normalize([sums[m['visual_mean_ref']['row']] * m['visual_mean_ref']['scale'] for m in graph['masters']])
            groups = [[i for i, node in enumerate(graph['image_nodes']) if node['master_id'] == m['id']] for m in graph['masters']]
            assert sorted(i for g in groups for i in g) == list(range(len(ms)))
            extra_encodings.append({'task': task, 'bge': extra, 'clip': cinfo})
        for qi, q in enumerate(qs):
            if family == 'base':
                weighted, error = checked_weighting(base[qi], vectors); max_dual = max(max_dual, error)
                scores = {'semrag': weighted['raw'], 'ours_2': weighted['weighted']}
            else:
                similarities = centers @ clipvectors[qi].T; selected = np.argmax(similarities, axis=1)
                scores = {m: np.empty(len(ids)) for m in methods}; groups_audit = []
                for mi, (master, indices) in enumerate(zip(graph['masters'], groups)):
                    weighted, error = checked_weighting(expanded[qi, selected[mi]], vectors[indices]); max_dual = max(max_dual, error)
                    scores['ours_1'][indices] = weighted['raw']; scores['ours_1_ours_2'][indices] = weighted['weighted']
                    groups_audit.append({'master_id': master['id'], 'memory_ids': [ids[i] for i in indices],
                        'candidate_index': int(selected[mi]), 'selected_statement': candidates[q['id']]['candidate_statements'][selected[mi]],
                        'clip_statement_scores': similarities[mi].tolist(), 'raw': weighted['raw'].tolist(), 'weighted': weighted['weighted'].tolist()})
                audits.append({'id': q['id'], 'task': task, 'groups': groups_audit})
            for method, values in scores.items():
                order = np.argsort(-values, kind='stable'); top = order[:5]; topids = [ids[i] for i in top]
                assert len(topids) == len(set(topids)) == 5
                context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                    f'Memory {rank}: {ms[i]["retrieval_text"]}' for rank, i in enumerate(top, 1))
                user = context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
                outputs[method].append({'id': q['id'], 'task': task, 'retrieval_query': q['retrieval_query'], 'top_memory_ids': topids,
                    'top_scores': [float(values[i]) for i in top], 'full_rank_memory_ids': [ids[i] for i in order], 'full_rank_scores': [float(values[i]) for i in order]})
                manifests[method].append({'id': q['id'], 'task': task, 'point': q['point'], 'question': q['question'],
                    'top_memory_ids': topids, 'context': context, 'query_image_paths': q['query_image_paths'],
                    'system': SYSTEM, 'user_text': user, 'payload_sha256': payload_hash(SYSTEM, user, q['query_image_paths'])})
    native.verify_hashes(sources)
    if audits: native.write_lines(root / 'master_query_audit.jsonl', audits)
    for method in methods:
        out = root / method; spec = SPECS[method]
        native.write_lines(out / 'inputs/questions.jsonl', questions)
        native.write_lines(out / 'retrieval.jsonl', outputs[method]); native.write_lines(out / 'reader_manifest.jsonl', manifests[method])
        assert len(outputs[method]) == len(manifests[method]) == 371
        artifacts = [out / 'inputs/questions.jsonl', out / 'retrieval.jsonl', out / 'reader_manifest.jsonl']
        p = {'dataset': 'MemEye', 'evaluation_type': 'open_ended', 'stage': 'GPT5 Open caption inputs',
            'method': method, 'method_id': method, **spec, 'questions': 371, 'original_source_root': str(SOURCE),
            'original_mcq_correct': spec['mcq_correct'], 'original_mcq_questions': 371, 'caption_model': MODEL,
            'requested_model': 'gpt-5', 'expected_returned_model': MODEL, 'memory_leaves': 438, 'masters': 72, 'top_k': 5,
            'query_policy': preparation['query_policy'], 'reader_policy': preparation['reader_policy'],
            'open_adaptation': preparation['candidate_policy'] if family == 'ours' else 'True official Open query without options or generated candidates.',
            'weighting_group': 'within visual master' if family == 'ours' else 'entire task memory domain',
            'retrieval_device': device, 'max_independent_dual_ridge_error': max_dual,
            'extra_encoding_audits': extra_encodings, 'source_sha256': sources,
            'artifact_sha256': {str(p.relative_to(out)): sha(p) for p in artifacts}}
        freeze(out / 'protocol.json', p)
        freeze(out / 'ready_for_reader.json', {'questions': 371, 'method_id': method,
            'reader_manifest_sha256': sha(out / 'reader_manifest.jsonl'), 'protocol_sha256': sha(out / 'protocol.json')})
    print(json.dumps({'stage': 'retrieval_complete', 'methods': methods, 'questions_per_method': 371, 'max_dual_error': max_dual}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['prepare', 'encode', 'retrieve'], required=True)
    p.add_argument('--root', type=Path, default=DEFAULT); p.add_argument('--device', default='cpu'); p.add_argument('--family', choices=['base', 'ours'], default='base')
    args = p.parse_args(); root = args.root.resolve()
    if args.phase == 'prepare': prepare(root)
    elif args.phase == 'encode': encode(root, args.device)
    else: retrieve(root, args.family, args.device)


if __name__ == '__main__': main()
