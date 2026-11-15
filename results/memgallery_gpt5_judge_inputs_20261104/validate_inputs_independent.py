"""Independent source-to-frozen-input validation; never calls an API."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import normalize_answer_universal, score_answer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(records):
    result = {row['id']: row for row in records}
    assert len(result) == len(records) == 1711
    return result


def main():
    manifest_path = HERE / 'methods_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    canonical_path = ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104/requests.jsonl'
    canonical = keyed(rows(canonical_path))
    prompt_path = ROOT / 'scripts/evaluation/prompts/memgallery_official_five_grade_judge.txt'
    prompt = prompt_path.read_text()
    expectations = dict(zip(
        ['semantic_rag', 'ours1_ours2', 'gen_agents', 'memoryos', 'simplemem',
         'a_mem', 'clip_image_dialogue', 'hippomm'],
        [741, 760, 376, 676, 575, 679, 687, 501]))
    assert {m['id'] for m in manifest['methods']} == set(expectations)
    hashes, requests, reports, empty_answers = {}, {}, [], []
    for method in manifest['methods']:
        mid = method['id']
        for name, expected in method['source_sha256'].items():
            path = Path(name)
            assert path.is_absolute()
            actual = sha(path)
            assert actual == expected, name
            hashes[name] = actual
        input_path = Path(method['records_path'])
        assert sha(input_path) == method['records_sha256']
        frozen = keyed(rows(input_path))
        assert set(frozen) == set(canonical)
        source = Path(method['source_root'])
        if mid in ('clip_image_dialogue', 'hippomm'):
            response_path = source / 'qa_responses.jsonl'
            scored_path = source / 'per_question.jsonl'
            question_path = (source / 'questions.jsonl' if mid == 'hippomm' else
                             ROOT / 'results/mem_gallery_master_answer_20261021/inputs/questions.jsonl')
            gold_path = ROOT / 'results/mem_gallery_master_answer_20261021/inputs/gold.jsonl'
        else:
            sub = {'gen_agents': 'gen_agents/reader', 'memoryos': 'memoryos/reader'}.get(mid, 'reader')
            response_path = source / sub / 'responses.jsonl'
            scored_path = source / 'scored_questions.jsonl'
            question_path = source / 'inputs/questions.jsonl'
            gold_path = source / 'inputs/gold.jsonl'
        for path in [response_path, scored_path, question_path, gold_path]:
            assert str(path) in method['source_sha256']
        response_rows, scored_rows = rows(response_path), rows(scored_path)
        if mid == 'ours1_ours2':
            condition = 'caption_visual_master+zscore'
            assert method['source_condition'] == condition
            response_rows = [r for r in response_rows if r['condition'] == condition]
            scored_rows = [r for r in scored_rows if r['condition'] == condition]
            protocol = json.loads((source / 'protocol.json').read_text())
            assert protocol['lambda'] == .03 and protocol['alpha'] == 1
        response_map, scored_map = keyed(response_rows), keyed(scored_rows)
        question_map, gold_map = keyed(rows(question_path)), keyed(rows(gold_path))
        for mapping in [response_map, scored_map, question_map, gold_map]:
            assert set(mapping) == set(canonical)
        metrics, models = [], Counter()
        token_limit_known = token_limit_true = 0
        for qid, frozen_row in frozen.items():
            canon, response, scored, q, gold = (canonical[qid], response_map[qid],
                                               scored_map[qid], question_map[qid], gold_map[qid])
            for key in ['task', 'point', 'question']:
                assert frozen_row[key] == canon[key] == q[key]
            assert frozen_row['ground_truth'] == canon['ground_truth'] == gold['answer']
            answer = response['answer'] if mid == 'hippomm' else response['raw_response']
            assert frozen_row['model_output'] == answer
            if 'raw_response' in scored:
                assert answer == scored['raw_response']
            if 'prediction' in scored:
                assert answer == scored['prediction']
            model = response['model'] if mid == 'hippomm' else response['returned_model']
            assert model == method['answer_model'] == 'gpt-5-2025-08-07'
            for key in ['answer_model', 'source_returned_model']:
                if key in frozen_row:
                    assert frozen_row[key] == model
            models[model] += 1
            expected_source_id = qid + '::caption_visual_master+zscore' if mid == 'ours1_ours2' else qid
            assert frozen_row['source_record_id'] == expected_source_id
            recomputed = score_answer(answer, gold['answer'])
            for metric in ['em', 'f1', 'bleu_1']:
                assert abs(recomputed[metric] - scored[metric]) < 1e-12
            assert type(frozen_row['em']) is int and frozen_row['em'] == int(recomputed['em'])
            if 'source_hit_token_limit' in frozen_row:
                actual_limit = response.get('hit_token_limit', scored.get('hit_token_limit'))
                assert isinstance(actual_limit, bool)
                assert frozen_row['source_hit_token_limit'] is actual_limit
                token_limit_known += 1
                token_limit_true += actual_limit
            if not answer.strip():
                empty_answers.append({'method': mid, 'id': qid,
                                      'original_response_preserved': True,
                                      'source': str(response_path)})
            metrics.append(recomputed)
            text = re.sub(r'\{\{(question|ground_truth|model_output)\}\}',
                          lambda match: frozen_row[match.group(1)], prompt)
            request = {'model': 'deepseek-v4.1-flash',
                       'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
                       'temperature': 0, 'seed': 42, 'max_tokens': 1024, 'n': 1,
                       'stream': False, 'enable_thinking': False}
            serialized = json.dumps(request, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()
            digest = hashlib.sha256(serialized).hexdigest()
            if digest in requests:
                assert requests[digest] == serialized
            requests[digest] = serialized
        means = {name: sum(r[name] for r in metrics) / 1711 for name in ['em', 'f1', 'bleu_1']}
        correct = sum(int(r['em']) for r in metrics)
        assert correct == expectations[mid] == method['expected_em_correct']
        for key, val in means.items():
            assert abs(val - method[key]) < 1e-12
        reports.append({'method': mid, 'questions': 1711, 'correct': correct, **means,
                        'unique_ids': 1711, 'model_counts': dict(models),
                        'source_token_limit_records': token_limit_true,
                        'source_token_limit_known_records': token_limit_known,
                        'canonical_and_original_question_gold_exact': True,
                        'prediction_matches_original_source_exactly': True,
                        'per_question_em_f1_bleu1_recomputed': True,
                        'records_sha256': sha(input_path)})
    assert len(requests) == 7839
    assert len(empty_answers) == 1 and empty_answers[0]['method'] == 'a_mem'
    report = {'status': 'passed', 'api_calls': 0, 'method_count': 8,
              'questions_per_method': 1711, 'total_method_questions': 13688,
              'unique_requests': len(requests), 'verified_unique_source_files': len(hashes),
              'source_sha256': hashes, 'manifest_sha256': sha(manifest_path),
              'canonical_sha256': sha(canonical_path), 'prompt_sha256': sha(prompt_path),
              'validator_sha256': sha(Path(__file__)), 'methods': reports,
              'original_empty_answers_preserved': empty_answers,
              'judge_request_sha256_sorted': sorted(requests)}
    out = HERE / 'input_validation.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'api_calls', 'method_count',
                'total_method_questions', 'unique_requests', 'verified_unique_source_files']}, indent=2))


if __name__ == '__main__':
    main()
