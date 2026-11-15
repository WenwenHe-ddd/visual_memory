"""Independent source, payload, coverage and metric audit; no model or API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT = ROOT / 'results/memeye_qwen_semrag_open_20261105'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(values, key='id'):
    result = {row[key]: row for row in values}
    assert len(result) == len(values), 'Duplicate IDs'
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize(value):
    text = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', value.lower())
    text = text.replace('_', 'UNDERSCOREPLACEHOLDER')
    text = re.sub(r'\b(a|an|the|and)\b', ' ', text)
    text = ''.join(c if c not in string.punctuation else ' ' for c in text)
    return ' '.join(text.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_').split())


def verify_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, f'Hash mismatch: {path}'
    return len(mapping)


def audit_inputs(root):
    protocol = read(root / 'protocol.json')
    source_count = verify_hashes(protocol['source_sha256'])
    artifact_count = verify_hashes({str(root / name): value
                                  for name, value in protocol['artifact_sha256'].items()})
    old = index(rows(ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'))
    memories = index(rows(ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl'))
    assert len(old) == 371 and len(memories) == 438
    questions = index(rows(root / 'inputs/questions.jsonl'))
    gold = index(rows(root / 'inputs/gold.jsonl'))
    retrieval = index(rows(root / 'retrieval.jsonl'))
    readers = index(rows(root / 'reader_manifest.jsonl'))
    assert set(old) == set(questions) == set(gold) == set(retrieval) == set(readers)
    canonical = {}
    for task in sorted({q['task'] for q in old.values()}):
        raw = read(ROOT / 'data/MemEye/data/dialog' / f'{task}_Open.json')['human-annotated QAs']
        for q in raw:
            qid = task + '/' + q['question_id']
            assert qid not in canonical
            canonical[qid] = q
    assert set(canonical) == set(old)
    changed, different_gold, image_cases, same_rank = [], [], [], 0
    forbidden = {'answer', 'ground_truth', 'gold_ids', 'gold_choice', 'options', 'clue'}
    context_header = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
    for qid, q in questions.items():
        actual, historical, target, retrieved, reader = canonical[qid], old[qid], gold[qid], retrieval[qid], readers[qid]
        task = qid.split('/', 1)[0]
        assert q['question'] == actual['question'] == reader['question']
        point = '/'.join(str(v) for axis in actual['point'] for v in (axis if isinstance(axis, list) else [axis]))
        assert q['point'] == target['point'] == reader['point'] == point
        assert target['answer'] == actual['answer']
        assert target['gold_ids'] == [task + '/' + item for item in actual['clue']]
        assert set(target['gold_ids']) == set(historical['gold_ids'])
        paths = actual.get('question_image') or []
        if isinstance(paths, str):
            paths = [paths]
        paths = [str((ROOT / 'data/MemEye/data/image' / p).resolve()) for p in paths]
        assert paths == q['query_image_paths'] == reader['query_image_paths'] == historical['query_image_paths']
        assert q['query_image_captions'] == historical['query_image_captions']
        suffix = "\nquestion's image:\n" + '\n'.join('image_caption: ' + s for s in q['query_image_captions']) if paths else ''
        assert q['retrieval_query'] == retrieved['retrieval_query'] == q['question'] + suffix
        assert not forbidden.intersection(q) and not forbidden.intersection(reader)
        top = retrieved['top_memory_ids']
        assert len(top) == len(set(top)) == 5 and reader['top_memory_ids'] == top
        assert all(mid in memories and memories[mid]['task'] == task for mid in top)
        expected = context_header + '\n\n'.join(f'Memory {i}: {memories[mid]["reader_text"]}' for i, mid in enumerate(top, 1))
        assert reader['context'] == expected
        assert reader['user_text'] == expected + '\n\nQuestion:\n' + actual['question'] + '\n\nAnswer concisely.'
        assert reader['system'] == protocol['reader_system']
        assert '\nOptions:\n' not in reader['user_text'] and 'one letter only' not in reader['user_text']
        payload = {'system': reader['system'], 'user_text': reader['user_text'],
                   'query_images': [{'path': p, 'sha256': sha(p)} for p in paths]}
        assert reader['payload_sha256'] == digest(payload)
        full_ids, full_scores = retrieved['full_rank_memory_ids'], retrieved['full_rank_scores']
        assert set(full_ids) == {mid for mid, m in memories.items() if m['task'] == task}
        assert len(full_ids) == len(set(full_ids)) == len(full_scores)
        assert all(np.isfinite(full_scores)) and all(a >= b for a, b in zip(full_scores, full_scores[1:]))
        assert full_ids[:5] == top and full_scores[:5] == retrieved['top_scores']
        if q['question'] != historical['question']:
            changed.append(qid)
            assert retrieved['query_vector_origin'] == 'new_open_question_encoding'
        else:
            assert retrieved['query_vector_origin'] == 'unchanged_query_frozen_vector'
            assert top == retrieved['mcq_top_memory_ids']
            same_rank += 1
        if actual['answer'] != historical['options'][historical['gold_choice']]:
            different_gold.append(qid)
        if paths:
            image_cases.append(qid)
    original = Path(protocol['original_source_root'])
    old_reader = index([r for r in rows(original / 'reader/qa_manifest.jsonl') if r['condition'] == 'baseline_question_only'])
    old_responses = [r for r in rows(original / 'reader/responses.jsonl') if r['condition'] == 'baseline_question_only']
    assert len(old_responses) == 371
    correct = 0
    for response in old_responses:
        qid = response.get('id', response.get('question_id'))
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', response['raw_response'].strip(), re.I)
        choice = match.group(1).upper() if match else None
        assert response['correct'] == (choice == old[qid]['gold_choice'])
        correct += response['correct']
    assert correct == 184
    frozen = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
    unchanged_replayed, old_replayed, max_error = 0, 0, 0.
    for task in sorted({q['task'] for q in questions.values()}):
        data = read(frozen / task / 'inputs.json')
        memory_ids = [r['memory_id'] for r in data['leaves']]
        with np.load(frozen / task / 'features.npz') as arrays:
            scores = arrays['bge_query'] @ arrays['bge_leaf'].T
        for i, old_question in enumerate(data['questions']):
            qid = old_question['id']
            order = np.argsort(-scores[i], kind='stable')
            old_top = [memory_ids[j] for j in order[:5]]
            assert old_top == old_reader[qid]['top_memory_ids'] == retrieval[qid]['mcq_top_memory_ids']
            old_replayed += 1
            if qid not in changed:
                r = retrieval[qid]
                assert [memory_ids[j] for j in order] == r['full_rank_memory_ids']
                error = float(np.max(np.abs(scores[i, order] - r['full_rank_scores'])))
                assert error < 1e-12
                max_error = max(max_error, error)
                unchanged_replayed += 1
    assert (len(changed), len(different_gold), len(image_cases), same_rank) == (234, 26, 8, 137)
    assert all(q.startswith('Outdoor_') for q in different_gold)
    assert old_replayed == 371 and unchanged_replayed == 137
    return {'passed': True, 'source_hashes_checked': source_count, 'artifact_hashes_checked': artifact_count,
        'questions': 371, 'memories': 438, 'raw_open_questions_and_answers_verified': 371,
        'reader_payloads_reconstructed_without_gold_or_options': 371, 'changed_open_questions': len(changed),
        'different_open_gold_texts': len(different_gold), 'native_query_image_questions': len(image_cases),
        'unique_query_image_files': len({p for q in questions.values() for p in q['query_image_paths']}),
        'original_mcq_correct_reproduced': correct, 'original_mcq_top5_cosine_replayed': old_replayed,
        'unchanged_open_queries_full_cosine_replayed': unchanged_replayed,
        'unchanged_open_queries_max_score_error': max_error,
        'all_open_full_rank_order_and_top5_verified': 371,
        'new_open_embedding_audit_boundary': '234 freshly encoded query vectors were not persisted; their encoding is code-reviewed, not independently re-encoded by this model-free audit.',
        'top5_changed_from_mcq': sum(r['top_memory_ids'] != r['mcq_top_memory_ids'] for r in retrieval.values())}


def audit_reader(root):
    protocol = read(root / 'reader/protocol.json')
    contract = protocol['contract']
    assert digest(contract) == protocol['experiment_signature']
    verify_hashes(contract['source_sha256'])
    verify_hashes(contract['query_image_sha256'])
    assert sha(root / 'inputs/gold.jsonl') == contract['gold_file_sha256']
    assert contract['seed'] == 42 and contract['do_sample'] is False
    assert contract['dtype'] == 'bfloat16' and contract['attention_implementation'] == 'sdpa'
    assert contract['batch_size'] == 1 and contract['max_new_tokens'] == 512
    assert contract['query_image_max_pixels'] == 1638400 and contract['input_truncation'] is False
    assert contract['model']['revision'] == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    manifest, gold = index(rows(root / 'reader_manifest.jsonl')), index(rows(root / 'inputs/gold.jsonl'))
    responses = index(rows(root / 'reader/responses.jsonl'))
    assert set(responses) == set(manifest) == set(gold)
    complete = read(root / 'reader/completion.json')
    assert complete['status'] == 'complete' and complete['responses_sha256'] == sha(root / 'reader/responses.jsonl')
    summary, scored = read(root / 'answer_summary.json'), index(rows(root / 'scored_questions.jsonl'))
    judge = index(rows(root / 'judge_inputs.jsonl'))
    assert set(scored) == set(judge) == set(gold)
    verify_hashes({str(root / p): v for p, v in summary['artifact_sha256'].items()})
    em, hits, refs, cap, empty = 0, 0, 0, 0, 0
    for qid, response in responses.items():
        item, target, score, judge_input = manifest[qid], gold[qid], scored[qid], judge[qid]
        assert response['experiment_signature'] == protocol['experiment_signature'] and response['run_id'] == protocol['run_id']
        assert response['payload_sha256'] == item['payload_sha256'] and response['top_memory_ids'] == item['top_memory_ids']
        assert response['historical_response_reuse'] is False and response['batch_size'] == 1
        assert response['answer_model'] == judge_input['answer_model'] == 'Qwen/Qwen3-VL-8B-Instruct'
        assert score['raw_response'] == judge_input['model_output'] == response['raw_response']
        assert score['answer'] == judge_input['ground_truth'] == target['answer']
        assert score['question'] == judge_input['question'] == item['question']
        assert judge_input['source_responses_sha256'] == complete['responses_sha256']
        correct = normalize(response['raw_response']) == normalize(target['answer'])
        assert score['em'] == judge_input['em'] == int(correct)
        gold_hits = len(set(item['top_memory_ids']) & set(target['gold_ids']))
        assert score['gold_hits'] == gold_hits and score['gold_count'] == len(target['gold_ids'])
        em += correct; hits += gold_hits; refs += len(target['gold_ids'])
        cap += response['hit_token_limit']; empty += not response['raw_response']
        environment = read(root / 'reader' / f'environment_{response["execution_id"]}.json')
        assert environment['dtype'] == 'torch.bfloat16' and environment['device'] == contract['device']
        assert response['input_tokens'] + 512 <= environment['context_limit']
    assert summary['questions'] == 371 and summary['correct_em'] == em and summary['em'] == em / 371
    assert summary['gold_hits_top5'] == hits and summary['gold_references'] == refs and summary['recall_at5_micro'] == hits / refs
    assert summary['token_limit_hits'] == cap and summary['empty_answers'] == empty
    return {'passed': True, 'responses': 371, 'sealed_sha256': complete['responses_sha256'],
        'open_em_correct': em, 'open_em': em / 371, 'recall_hits': hits, 'recall_references': refs,
        'recall_at5_micro': hits / refs, 'token_limit_hits': cap, 'empty_answers': empty,
        'token_limit_ids': [qid for qid, response in responses.items() if response['hit_token_limit']],
        'empty_answer_ids': [qid for qid, response in responses.items() if not response['raw_response']],
        'weight_provenance_boundary': 'Reader prepare hashed every model file against the pinned baseline; this final audit checks signed metadata and reader runtime, without rehashing 17GB of weights.'}


def audit_judge(root):
    from scripts.evaluation import memgallery_multi_method_judge as shared
    out = root / 'judge'
    protocol, summary = read(out / 'protocol.json'), read(out / 'summary.json')
    verify_hashes(protocol['source_sha256']); verify_hashes(protocol['implementation_sha256'])
    verify_hashes({str(out / p): v for p, v in summary['artifact_sha256'].items()})
    source, mapped, final = index(rows(root / 'judge_inputs.jsonl')), index(rows(out / 'question_map.jsonl')), index(rows(out / 'per_question.jsonl'))
    assert set(source) == set(mapped) == set(final) and len(final) == 371
    requests = rows(out / 'requests.jsonl'); request_map = index(requests, 'request_sha256')
    scores = shared.validated_results(out, requests)
    assert set(scores) == set(request_map)
    template = shared.PROMPT.read_text()
    assert (out / 'judge_prompt.txt').read_text() == template
    for key, value in shared.judge_settings().items():
        assert protocol[key] == value
    for qid, original in source.items():
        for field, value in original.items():
            assert mapped[qid][field] == value and final[qid][field] == value
        wire = shared.build_request(original, template)
        h = shared.transport.request_sha256(wire)
        assert mapped[qid]['request_sha256'] == h and request_map[h]['request'] == wire
        assert final[qid]['score'] == scores[h]['score'] and final[qid]['reasoning'] == scores[h]['reasoning']
        assert final[qid]['judge_status'] == 'ok'
    count = Counter(r['score'] for r in final.values()); total = sum(r['score'] for r in final.values())
    assert summary['status'] == 'complete' and summary['questions_scored'] == 371
    assert summary['score_sum'] == total and summary['llm_judge'] == total / 371
    assert summary['score_distribution'] == {str(grade): count[grade] for grade in shared.GRADES}
    attempts = rows(out / 'attempts.jsonl')
    assert summary['api_attempt_count'] == len(attempts)
    assert summary['failed_api_attempt_count'] == sum(r['status'] != 'ok' for r in attempts)
    return {'passed': True, 'questions_scored': 371, 'unique_requests': len(requests),
        'llm_judge': total / 371, 'score_sum': total, 'grade_distribution': summary['score_distribution'],
        'api_attempts': len(attempts), 'failed_api_attempts': summary['failed_api_attempt_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--stage', choices=('inputs', 'reader', 'final'), default='inputs')
    args = parser.parse_args()
    report = {'passed': True, 'stage': args.stage, 'inputs': audit_inputs(args.root)}
    if args.stage in ('reader', 'final'):
        report['reader'] = audit_reader(args.root)
    if args.stage == 'final':
        report['judge'] = audit_judge(args.root)
    report['auditor_sha256'] = sha(__file__)
    path = args.root / f'independent_audit_{args.stage}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
