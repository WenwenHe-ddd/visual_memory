"""Freeze the two screenshot-matched DeepSeek caption conditions for judging."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import normalize_answer_universal as normalize_answer

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/mem_gallery_deepseek_v41flash_20261030'
MODEL = 'deepseek-v4.1-flash'

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def index(items):
    result = {row['id']: row for row in items}
    assert len(result) == len(items) == 1711
    return result

def main():
    canonical = index(rows(ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104/requests.jsonl'))
    questions = index(rows(SOURCE / 'inputs/questions.jsonl'))
    gold = index(rows(SOURCE / 'inputs/gold.jsonl'))
    responses = rows(SOURCE / 'reader/responses.jsonl')
    scored = rows(SOURCE / 'scored_questions.jsonl')
    manifests = rows(SOURCE / 'reader/manifest.jsonl')
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    final = json.loads((SOURCE / 'final_results.json').read_text())
    assert final['status'] == 'complete' and final['model'] == MODEL
    assert protocol['expected_model'] == MODEL
    assert protocol['lambda'] == .03 and protocol['alpha'] == 1
    assert protocol['top_k'] == 5
    files = ['inputs/questions.jsonl', 'inputs/gold.jsonl', 'reader/responses.jsonl',
             'reader/manifest.jsonl', 'scored_questions.jsonl', 'final_results.json',
             'protocol.json', 'audit.json', 'postrun_verification.json', 'completion_manifest.json']
    source_hashes = {str(SOURCE / name): sha(SOURCE / name) for name in files}
    methods = []
    for mid, label, condition, expected in [
        ('semantic_rag', 'Semantic RAG（caption 基线）', 'Natural baseline', 770),
        ('ours1_ours2', 'ours_1＋ours_2（λ=0.03，α=1）', 'caption_visual_master+zscore', 785),
    ]:
        selected = index([row for row in responses if row['condition'] == condition])
        scores = index([row for row in scored if row['condition'] == condition])
        requests = {row['payload']['record_id']: row for row in manifests if row['condition'] == condition}
        assert set(selected) == set(scores) == set(requests) == set(canonical)
        frozen = []
        for qid, q in questions.items():
            response, metric, request, c = selected[qid], scores[qid], requests[qid], canonical[qid]
            assert all(q[key] == c[key] for key in ('task', 'point', 'question'))
            assert gold[qid]['answer'] == c['ground_truth'] == metric['answer']
            assert response['raw_response'] == metric['raw_response']
            assert response['request_sha256'] == metric['request_sha256'] == request['key']
            assert response['requested_model'] == response['returned_model'] == MODEL
            assert response['finish_reason'] == 'stop' and response['hit_token_limit'] is False
            assert metric['context_count'] == 5
            em = int(normalize_answer(response['raw_response']) == normalize_answer(gold[qid]['answer']))
            assert em == metric['em']
            frozen.append({'id': qid, 'task': q['task'], 'point': q['point'],
                'question': q['question'], 'ground_truth': gold[qid]['answer'],
                'model_output': response['raw_response'], 'em': em,
                'source_record_id': qid, 'source_condition': condition,
                'answer_model': MODEL, 'source_hit_token_limit': response['hit_token_limit'],
                'source_finish_reason': response['finish_reason'],
                'source_returned_model': response['returned_model'],
                'source_response_id': response['response_id'],
                'source_request_sha256': response['request_sha256']})
        report = final['results'][condition]
        assert sum(row['em'] for row in frozen) == report['correct'] == expected
        for key in ('em', 'f1', 'bleu_1'):
            assert abs(sum(row[key] for row in scores.values()) / 1711 - report[key]) < 1e-12
        path = OUT / (mid + '.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in frozen))
        methods.append({'id': mid, 'label': label, 'display_label': label,
            'code_method': condition, 'source_condition': condition,
            'source_root': str(SOURCE), 'answer_model': MODEL, 'expected_em_correct': expected,
            'records_path': str(path), 'records_sha256': sha(path), 'source_sha256': source_hashes,
            **{key: report[key] for key in ('em', 'f1', 'bleu_1')},
            'context': 'Top5 original dialogue rounds + full DeepSeek captions; original question images in answer input',
            'parameters': {'seed': 42, 'top_k': 5, **({'lambda': .03, 'alpha': 1} if mid == 'ours1_ours2' else {})},
            'source_token_limit_records': 0})
    (OUT / 'caption_methods_manifest.json').write_text(json.dumps({'answer_model': MODEL,
        'questions_per_method': 1711, 'methods': methods}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({m['id']: {k: m[k] for k in ('expected_em_correct', 'em', 'f1', 'bleu_1')} for m in methods}, indent=2))

if __name__ == '__main__':
    main()
