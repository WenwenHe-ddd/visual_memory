#!/usr/bin/env python3
"""DeepSeek seed1024 Open candidates, fresh retrieval and fresh answers.

Prepare is offline. Candidates and answers transmit dataset material and are
separate phases so the orchestrator can defer them until export approval.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gc
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import memeye_deepseek_seed1024_ours as main
from scripts.evaluation import mem_gallery_master_reader as candidates_core
from scripts.evaluation import memeye_deepseek_seed1024_open as shared_reader
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM, payload_hash
from scripts.memory.online_zscore import ZScoreWeighting

OUTPUT = main.OUTPUT / 'open'
CANONICAL_GOLD = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs/gold.jsonl'
METHODS = ('semrag', 'ours_1', 'ours_2', 'ours_1_ours_2')
read, rows, sha, digest, dump, write_lines = main.read, main.lines, main.sha, main.digest, main.dump, main.write_lines


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    marker = output / 'preparation_protocol.json'
    if marker.exists():
        protocol = read(marker)
        assert digest(protocol['contract']) == protocol['experiment_signature']
        main.native.verify_hashes(protocol['contract']['source_sha256'])
        return protocol
    parent = read(output.parent / 'protocol.json')
    main.verify(parent)
    questions = rows(main.OPEN_SOURCE / 'inputs/questions.jsonl')
    assert len(questions) == len(main.native.unique(questions)) == 371
    assert all(not set(q) & {'options', 'gold_choice', 'gold_ids', 'clue', 'answer'} for q in questions)
    old_mcq = main.native.unique(rows(output.parent / 'inputs/questions.jsonl'))
    for q in questions:
        mcq = old_mcq[q['id']]
        assert q['query_image_captions'] == mcq['query_image_captions']
        assert q['query_image_paths'] == mcq['query_image_paths']
        expected = q['question'] + (main.native.QUERY_IMAGE_SEPARATOR + ' '.join(q['query_image_captions'])
                                    if q['query_image_captions'] else '')
        assert q['retrieval_query'] == expected
    sanitized = [{'id': q['id'], 'question': q['question'],
                  'query_image_caption': '\n'.join(q['query_image_captions'])} for q in questions]
    write_lines(output / 'inputs/questions.jsonl', questions)
    write_lines(output / 'inputs/gold.jsonl', rows(CANONICAL_GOLD))
    write_lines(output / 'inputs/sanitized_candidate_inputs.jsonl', sanitized)
    api = main.transport()
    paths = [Path(__file__), Path(main.__file__), Path(api.__file__), Path(candidates_core.__file__),
        Path(shared_reader.__file__), ROOT / 'scripts/evaluation/memeye_deepseek_open_reader.py',
        output.parent / 'protocol.json', output.parent / 'inputs/memories.jsonl',
        main.OPEN_SOURCE / 'inputs/questions.jsonl', CANONICAL_GOLD,
        ROOT / 'scripts/evaluation/memeye_open_semrag_inputs.py']
    paths.extend(output / 'inputs' / n for n in ('questions.jsonl', 'gold.jsonl', 'sanitized_candidate_inputs.jsonl'))
    for method in METHODS:
        path = main.OPEN_SOURCE / method / 'retrieval.jsonl'
        if path.exists():
            paths.append(path)
    contract = {'dataset': 'MemEye Open', 'model': main.MODEL, 'seed': 1024, 'judge_seed': 42,
        'caption_model': main.MODEL, 'caption_seed': 42, 'questions': 371,
        'methods': list(METHODS), 'top_k': 5, 'lambda': .03, 'alpha': 1.,
        'changed_open_questions_vs_mcq': sum(q['question'] != old_mcq[q['id']]['question'] for q in questions),
        'candidate_policy': 'Four fresh DeepSeek hypotheses; original prompt/parser, only Open question and cached DeepSeek query-image caption. No MCQ options, history or gold.',
        'candidate_max_tokens': 2048, 'answer_max_tokens': 2048,
        'frozen_caption_source': str(main.SOURCE / 'captions.jsonl'),
        'new_memory_source': str(output.parent / 'domains'),
        'historical_candidates_or_answers_reused': False,
        'reference_mcq_policy': 'Before Open answer execution, require complete new seed1024 MCQ results with 371 unique questions in each condition. Regrade raw responses and use the actual per-method correct counts; no numeric default represents unknown results.',
        'source_sha256': {str(p): sha(p) for p in paths}}
    protocol = {'contract': contract, 'experiment_signature': digest(contract), 'run_id': parent['run_id']}
    dump(marker, protocol)
    print(json.dumps({'stage': 'open_prepared', 'questions': 371,
                      'changed_open_questions': contract['changed_open_questions_vs_mcq'], 'api_calls': 0}), flush=True)
    return protocol


def candidates(output=OUTPUT, workers=4):
    output = Path(output).resolve()
    preparation = prepare(output)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(candidates_core.CLIP), local_files_only=True)
    source = rows(output / 'inputs/sanitized_candidate_inputs.jsonl')
    contract = {'model': main.MODEL, 'seed': 1024, 'temperature': 0, 'enable_thinking': False,
        'questions': 371, 'max_tokens': 2048, 'max_format_attempts': 3,
        'system_prompt': candidates_core.CANDIDATE_SYSTEM, 'parser': 'Original MemGallery exact-four JSON/line parser with CLIP77-token validation.',
        'source_sha256': {str(p): sha(p) for p in (output / 'preparation_protocol.json',
            output / 'inputs/sanitized_candidate_inputs.jsonl', Path(__file__), Path(candidates_core.__file__))}}
    main.freeze(output / 'candidate_generation_protocol.json', {'contract': contract, 'experiment_signature': digest(contract)})
    def one(row):
        for attempt in range(1, 4):
            payload = {'system': candidates_core.CANDIDATE_SYSTEM,
                       'user_text': candidates_core.candidate_user_text(row, attempt), 'query_images': []}
            key, response = main.api_call(output.parent, payload, 2048)
            error = None
            try:
                if response['status'] != 'ok' or response.get('finish_reason') == 'length':
                    raise ValueError('Candidate response rejected or token-limited')
                statements, lengths, fmt = candidates_core.parse_candidates(response['raw_response'], tokenizer, return_format=True)
            except (ValueError, TypeError) as exc:
                error = str(exc)
            dump(output / 'candidate_attempts' / (row['id'].replace('/', '__') + f'__{attempt}.json'),
                 {'id': row['id'], 'attempt': attempt, 'request_sha256': key, 'response': response, 'validation_error': error})
            if error is None:
                return {'id': row['id'], 'candidate_statements': statements, 'clip_token_lengths': lengths,
                    'response_format': fmt, 'request_sha256': key, 'source_input_sha256': digest(row),
                    'seed': 1024, 'gold_or_history_used': False, 'format_attempt': attempt,
                    'experiment_signature': digest(contract)}
        raise RuntimeError('No valid four-hypothesis response after original-format retries: ' + row['id'])
    done = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, q): q['id'] for q in source}
        for future in as_completed(futures):
            result = future.result()
            done[result['id']] = result
            if len(done) % 20 == 0 or len(done) == 371:
                print(json.dumps({'stage': 'open_candidates', 'completed': len(done), 'total': 371}), flush=True)
    values = [done[q['id']] for q in source]
    write_lines(output / 'candidates.jsonl', values)
    proof = {'passed': True, 'questions': 371, 'candidates': 1484, 'seed': 1024,
        'candidates_sha256': sha(output / 'candidates.jsonl'),
        'source_preparation_signature': preparation['experiment_signature'],
        'all_request_bodies_seed1024': True, 'historical_candidate_reuse': False}
    dump(output / 'candidate_validation.json', proof)
    return proof


def rank(question, memories, graph, sums, qv, leaves, cv):
    centers = main.native.normalize([sums[m['visual_mean_ref']['row']] * m['visual_mean_ref']['scale'] for m in graph['masters']])
    similarities = centers @ cv.T
    choices = np.argmax(similarities, axis=1)
    scores = {'semrag': leaves @ qv[0], 'ours_2': ZScoreWeighting(.03, 1.).apply(qv[0], leaves)['weighted'],
              'ours_1': np.empty(len(memories)), 'ours_1_ours_2': np.empty(len(memories))}
    audits = []
    for mi, group in enumerate(graph['masters']):
        indices = np.array([i for i, n in enumerate(graph['image_nodes']) if n['master_id'] == group['id']], dtype=int)
        adjusted = ZScoreWeighting(.03, 1.).apply(qv[1 + choices[mi]], leaves[indices])
        scores['ours_1'][indices] = adjusted['raw']
        scores['ours_1_ours_2'][indices] = adjusted['weighted']
        audits.append({'master_id': group['id'], 'member_ids': group['member_ids'],
            'candidate_index': int(choices[mi]), 'clip_scores': similarities[mi].tolist(),
            'raw': adjusted['raw'].tolist(), 'weighted': adjusted['weighted'].tolist()})
    return scores, audits


def retrieve(output=OUTPUT, device='cuda:0'):
    output = Path(output).resolve()
    prep = prepare(output)
    if (output / 'retrieval_validation.json').exists():
        proof = read(output / 'retrieval_validation.json')
        main.native.verify_hashes(proof['artifact_sha256'])
        return proof
    memories_index = read(output.parent / 'memory_manifest.json')
    main.native.verify_hashes(memories_index['output_sha256'])
    candidate_validation = read(output / 'candidate_validation.json')
    assert candidate_validation['passed'] and candidate_validation['candidates_sha256'] == sha(output / 'candidates.jsonl')
    generated = main.native.unique(rows(output / 'candidates.jsonl'))
    sanitized = main.native.unique(rows(output / 'inputs/sanitized_candidate_inputs.jsonl'))
    assert all(generated[qid]['source_input_sha256'] == digest(q) for qid, q in sanitized.items())
    main.seed_all()
    info = read(output.parent / 'inputs/encoding.json')
    memories = rows(output.parent / 'inputs/memories.jsonl')
    questions = rows(output / 'inputs/questions.jsonl')
    bge = main.native.load_text_encoder(info['text_encoder'], device)
    clip, tokenizer, limit = main.native.load_clip_text_encoder(info['image_encoder'], device)
    results = {m: [] for m in METHODS}
    manifests = {m: [] for m in METHODS}
    group_audits, encoding = [], []
    for task in sorted({m['task'] for m in memories}):
        ms = [m for m in memories if m['task'] == task]
        qs = [q for q in questions if q['task'] == task]
        graph = read(output.parent / 'domains' / task / 'memory/graph.json')
        sums = np.load(output.parent / 'domains' / task / 'memory/master_sums.npy', allow_pickle=False)
        leaves, leaf_info = main.encode_complete(bge, [m['retrieval_text'] for m in ms], ids=[m['id'] for m in ms], field='dialogue+caption')
        text = [t for q in qs for t in [q['retrieval_query'],
            *[q['retrieval_query'] + '\nCandidate answer: ' + s for s in generated[q['id']]['candidate_statements']]]]
        qv, query_info = main.encode_complete(bge, text, info['query_prefix'], field='Open queries')
        cv, candidate_info = main.native._clip_candidates(clip, tokenizer, limit,
            [s for q in qs for s in generated[q['id']]['candidate_statements']])
        qv, cv = qv.reshape(len(qs), 5, -1), cv.reshape(len(qs), 4, -1)
        for qi, q in enumerate(qs):
            scores, audit = rank(q, ms, graph, sums, qv[qi], leaves, cv[qi])
            group_audits.append({'id': q['id'], 'groups': audit})
            for method, values in scores.items():
                order = np.argsort(-values, kind='stable')
                top = order[:5]
                ids = [ms[i]['id'] for i in top]
                assert len(ids) == len(set(ids)) == 5
                context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                    f'Memory {j}: {ms[i]["retrieval_text"]}' for j, i in enumerate(top, 1))
                user_text = context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
                results[method].append({'id': q['id'], 'task': task, 'retrieval_query': q['retrieval_query'],
                    'top_memory_ids': ids, 'top_scores': [float(values[i]) for i in top],
                    'full_rank_memory_ids': [ms[i]['id'] for i in order], 'full_rank_scores': [float(values[i]) for i in order]})
                manifests[method].append({'id': q['id'], 'task': task, 'point': q['point'],
                    'question': q['question'], 'query_image_paths': q['query_image_paths'], 'top_memory_ids': ids,
                    'system': SYSTEM, 'context': context, 'user_text': user_text,
                    'payload_sha256': payload_hash(SYSTEM, user_text, q['query_image_paths'])})
        encoding.append({'task': task, 'leaf': leaf_info, 'query': query_info, 'candidate': candidate_info})
        print(json.dumps({'stage': 'open_retrieve', 'task': task, 'completed': len(results['semrag'])}), flush=True)
    del bge, clip
    gc.collect()
    gold = main.native.unique(rows(output / 'inputs/gold.jsonl'))
    sources = {**prep['contract']['source_sha256'], **memories_index['output_sha256']}
    sources.update({str(p): sha(p) for p in (output.parent / 'memory_manifest.json',
        output / 'candidates.jsonl', output / 'candidate_generation_protocol.json', output / 'candidate_validation.json')})
    reports, hashes = {}, {}
    for method in METHODS:
        out = output / method
        retrieved = main.native.unique(results[method])
        readers = main.native.unique(manifests[method])
        values = [retrieved[q['id']] for q in questions]
        write_lines(out / 'retrieval.jsonl', values)
        write_lines(out / 'reader_manifest.jsonl', [readers[q['id']] for q in questions])
        old = main.native.unique(rows(main.OPEN_SOURCE / method / 'retrieval.jsonl'))
        hits = sum(len(set(v['top_memory_ids']) & set(gold[v['id']]['gold_ids'])) for v in values)
        reports[method] = {'questions': 371, 'gold_hits_top5': hits, 'gold_references': 1041,
            'recall_at5_micro': hits / 1041,
            'changed_top5_order_vs_seed42': sum(v['top_memory_ids'] != old[v['id']]['top_memory_ids'] for v in values),
            'changed_top5_set_vs_seed42': sum(set(v['top_memory_ids']) != set(old[v['id']]['top_memory_ids']) for v in values)}
        contract = {'dataset': 'MemEye Open', 'method': method, 'model': main.MODEL,
            'seed': 1024, 'caption_seed': 42, 'judge_seed': 42, 'questions': 371,
            'source_sha256': sources, 'metrics': reports[method],
            'new_memory_built': True, 'fresh_candidates': True, 'historical_answers_reused': False}
        dump(out / 'protocol.json', {'contract': contract, 'experiment_signature': digest(contract)})
        hashes.update({str(p): sha(p) for p in (out / 'protocol.json', out / 'retrieval.jsonl', out / 'reader_manifest.jsonl')})
    write_lines(output / 'master_query_audit.jsonl', group_audits)
    proof = {'passed': True, 'questions_per_method': 371, 'methods': reports, 'encoding': encoding,
        'fresh_hypotheses': 1484, 'gold_or_mcq_options_used_for_retrieval': False,
        'historical_vectors_or_graphs_reused': False, 'artifact_sha256': hashes}
    dump(output / 'retrieval_validation.json', proof)
    return proof


def validate_mcq_reference(root):
    """Resolve per-method metadata only from the complete, verified new MCQ run."""
    root = Path(root).resolve()
    required = [root / name for name in ('protocol.json', 'summary.json', 'independent_audit.json',
                'reader/responses.jsonl', 'inputs/questions.jsonl', 'inputs/gold.jsonl')]
    if any(not path.is_file() for path in required):
        raise RuntimeError('Complete seed1024 MCQ results are required before starting Open answers; reference correct counts are unknown.')
    protocol = read(root / 'protocol.json')
    main.verify(protocol)
    summary = read(root / 'summary.json')
    audit = read(root / 'independent_audit.json')
    if (summary.get('status') != 'complete' or summary.get('seed') != 1024
            or summary.get('caption_seed') != 42 or summary.get('model') != main.MODEL
            or not audit.get('passed')):
        raise ValueError('MCQ reference must be a completed, audited new DeepSeek seed1024 run.')
    main.native.verify_hashes(audit['artifact_sha256'])
    questions = main.native.unique(rows(root / 'inputs/questions.jsonl'))
    gold = main.native.unique(rows(root / 'inputs/gold.jsonl'))
    responses = rows(root / 'reader/responses.jsonl')
    if len(questions) != 371 or set(gold) != set(questions):
        raise ValueError('MCQ reference must cover the complete unique 371-question cohort.')
    if len(responses) != 1484 or len(main.native.unique(responses, 'record_id')) != 1484:
        raise ValueError('MCQ reference must contain four complete 371-record conditions.')
    if {r['condition'] for r in responses} != set(main.CONDITIONS):
        raise ValueError('MCQ reference condition set differs.')
    counts = {}
    for method, condition in zip(METHODS, main.CONDITIONS):
        group = [r for r in responses if r['condition'] == condition]
        if len(group) != 371 or {r['question_id'] for r in group} != set(questions):
            raise ValueError('MCQ reference coverage differs for ' + condition)
        correct = 0
        for r in group:
            if (r['experiment_signature'] != protocol['experiment_signature']
                    or r['run_id'] != protocol['run_id'] or r['seed'] != 1024
                    or r.get('historical_answer_reuse') is not False):
                raise ValueError('MCQ reference response belongs to another run.')
            if r['record_id'] != r['question_id'] + '::' + condition:
                raise ValueError('MCQ reference record identity differs.')
            raw = r.get('raw_response')
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
            choice = match.group(1).upper() if match else None
            expected = gold[r['question_id']]['gold_choice']
            if (r['predicted_choice'] != choice or r['gold_choice'] != expected
                    or r['correct'] != (choice == expected)):
                raise ValueError('MCQ reference raw-response grading differs.')
            correct += choice == expected
        measured = summary['overall'][condition]
        if measured.get('questions') != 371 or measured.get('correct') != correct:
            raise ValueError('MCQ summary reference count differs for ' + condition)
        counts[method] = {'condition': condition, 'questions': 371, 'correct': int(correct)}
    return {'passed': True, 'seed': 1024, 'model': main.MODEL,
        'source_run_id': protocol['run_id'], 'source_experiment_signature': protocol['experiment_signature'],
        'per_method': counts, 'raw_responses_regraded': 1484,
        'source_sha256': {str(path): sha(path) for path in required}}


def answers(output=OUTPUT, workers=4):
    output = Path(output).resolve()
    reference = validate_mcq_reference(output.parent)
    main.freeze(output / 'reference_mcq_validation.json', reference)
    retrieval = read(output / 'retrieval_validation.json')
    main.native.verify_hashes(retrieval['artifact_sha256'])
    for method in METHODS:
        out = output / method
        reference_correct = reference['per_method'][method]['correct']
        shared_reader.run_reader(out, method, reference_correct=reference_correct, workers=workers)
        shared_reader.run_reader(out, method, reference_correct=reference_correct, workers=workers, phase='collect')
    return analyze(output)


def analyze(output=OUTPUT):
    output = Path(output).resolve()
    retrieval = read(output / 'retrieval_validation.json')
    main.native.verify_hashes(retrieval['artifact_sha256'])
    summaries = copy.deepcopy(retrieval['methods'])
    for method in METHODS:
        out = output / method
        status = read(out / 'reader/completion.json')
        assert status['status'] == 'complete' and status['questions_answered'] == 371
        assert status['responses_sha256'] == sha(out / 'reader/responses.jsonl')
        responses = rows(out / 'reader/responses.jsonl')
        assert len(responses) == len(main.native.unique(responses)) == 371
        reader = shared_reader.reader_module()
        reader.validate(out, read(out / 'reader/protocol.json'), rows(out / 'reader_manifest.jsonl'))
        assert all(r['seed'] == 1024 and r['returned_model'] == main.MODEL for r in responses)
        summaries[method]['open_answers'] = status
    result = {'status': 'open_answers_complete', 'seed': 1024, 'caption_seed': 42,
              'judge_seed': 42, 'model': main.MODEL, 'methods': summaries, 'judge_status': 'pending separate fixed-seed42 judging'}
    dump(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'candidates', 'retrieve', 'answers', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.output)
    elif args.phase == 'candidates':
        candidates(args.output, args.workers)
    elif args.phase == 'retrieve':
        retrieve(args.output, args.device)
    elif args.phase == 'answers':
        answers(args.output, args.workers)
    else:
        analyze(args.output)
