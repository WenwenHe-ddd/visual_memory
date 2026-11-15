"""Independent, read-only source/metric/wire-input review of all eight methods."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import score_answer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(data, count=1711):
    out = {row['id']: row for row in data}
    assert len(out) == len(data) and (count is None or len(out) == count)
    return out


def main():
    manifest_path = HERE / 'methods_manifest.json'
    manifest = read(manifest_path)
    run = ROOT / 'results/memgallery_deepseek_flash_llm_judge_20261105'
    protocol = read(run / 'protocol.json')
    assert protocol['manifest_sha256'] == sha(manifest_path)
    assert (run / 'methods_manifest.json').read_bytes() == manifest_path.read_bytes()
    canonical_dir = ROOT / 'results/mem_gallery_master_answer_20261021/inputs'
    canonical_q = keyed(rows(canonical_dir / 'questions.jsonl'))
    canonical_g = keyed(rows(canonical_dir / 'gold.jsonl'))
    assert set(canonical_q) == set(canonical_g)
    expected = dict(zip(['semantic_rag', 'ours1_ours2', 'gen_agents', 'memoryos',
                         'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm'],
                        [770, 785, 374, 622, 644, 678, 656, 427]))
    assert {m['id'] for m in manifest['methods']} == set(expected)
    assert manifest['answer_model'] == protocol['answer_model'] == 'deepseek-v4.1-flash'
    reports, hash_checks, all_rows = [], {}, {}
    response_file_cache = {}
    for m in manifest['methods']:
        mid = m['id']
        for filename, digest in m['source_sha256'].items():
            path = Path(filename)
            assert path.is_absolute() and sha(path) == digest, path
            hash_checks[filename] = digest
        path = Path(m['records_path'])
        assert sha(path) == m['records_sha256']
        frozen = keyed(rows(path))
        assert set(frozen) == set(canonical_q)
        source = Path(m['source_root'])
        if mid == 'a_mem':
            responses = {}
            for filename in {r['source_response_file'] for r in frozen.values()}:
                assert filename in m['source_sha256']
                part = keyed(rows(Path(filename)), None)
                assert not set(part) & set(responses)
                responses.update(part)
                response_file_cache[filename] = part
            scores = keyed(rows(source / 'combined_scored_questions.jsonl'))
            source_questions, source_gold = canonical_q, canonical_g
            assert len(responses) == 1711
            assert m['source_variant']['original_prompt_baseline'] is False
            assert m['source_variant']['modified_memory_calls'] == 1
        elif mid in ('clip_image_dialogue', 'hippomm'):
            responses = keyed(rows(source / 'qa_responses.jsonl'))
            scores = keyed(rows(source / 'per_question.jsonl'))
            qfile = source / ('qa_manifest.jsonl' if mid == 'clip_image_dialogue' else 'questions.jsonl')
            source_questions, source_gold = keyed(rows(qfile)), canonical_g
        else:
            response_list, score_list = rows(source / 'reader/responses.jsonl'), rows(source / 'scored_questions.jsonl')
            if mid in ('semantic_rag', 'ours1_ours2'):
                response_list = [r for r in response_list if r['condition'] == m['source_condition']]
                score_list = [r for r in score_list if r['condition'] == m['source_condition']]
                source_protocol = read(source / 'protocol.json')
                assert source_protocol['lambda'] == .03 and source_protocol['alpha'] == 1
            responses, scores = keyed(response_list), keyed(score_list)
            source_questions = keyed(rows(source / 'inputs/questions.jsonl'))
            source_gold = keyed(rows(source / 'inputs/gold.jsonl'))
        assert set(responses) == set(scores) == set(source_questions) == set(source_gold) == set(canonical_q)
        empty, model_counts, status_counts, metric_rows = [], Counter(), Counter(), []
        for qid, record in frozen.items():
            question, gold = canonical_q[qid], canonical_g[qid]
            source_response, score = responses[qid], scores[qid]
            for key in ['task', 'point', 'question']:
                assert record[key] == question[key] == source_questions[qid][key], (mid, qid, key)
            assert record['ground_truth'] == gold['answer'] == source_gold[qid]['answer']
            answer = source_response['answer'] if mid == 'hippomm' else source_response['raw_response']
            assert record['model_output'] == answer
            for key in ['raw_response', 'prediction']:
                if key in score:
                    assert score[key] == answer
            status = source_response.get('status', 'ok')
            status_counts[status] += 1
            model = source_response.get('model') if mid == 'hippomm' else source_response.get('returned_model')
            if model is None:
                assert mid == 'clip_image_dialogue' and status == 'provider_rejection' and answer == ''
                assert read(source / 'protocol.json')['model'] == 'deepseek-v4.1-flash'
            else:
                assert model == m['answer_model'] == record.get('answer_model', m['answer_model']) == 'deepseek-v4.1-flash'
            model_counts[str(model)] += 1
            if not answer.strip():
                empty.append(qid)
            if mid == 'a_mem':
                assert response_file_cache[record['source_response_file']][qid] == source_response
            recomputed = score_answer(answer, gold['answer'])
            for metric in ['em', 'f1', 'bleu_1']:
                assert abs(recomputed[metric] - score[metric]) < 1e-12, (mid, qid, metric)
            assert type(record['em']) is int and record['em'] == recomputed['em']
            metric_rows.append(recomputed)
            all_rows[(mid, qid)] = record
        correct = sum(r['em'] for r in metric_rows)
        means = {name: sum(r[name] for r in metric_rows) / 1711 for name in ['em', 'f1', 'bleu_1']}
        assert correct == expected[mid] == m['expected_em_correct']
        assert all(abs(means[k] - m[k]) < 1e-12 for k in means)
        assert len(empty) == {'clip_image_dialogue': 13, 'hippomm': 14}.get(mid, 0)
        reports.append({'method': mid, 'questions': 1711, 'correct': correct, **means,
                        'canonical_ids_and_question_gold_types_exact': True,
                        'all_original_answers_identical': True,
                        'all_per_question_em_f1_bleu1_recomputed': True,
                        'model_counts': dict(model_counts), 'source_status_counts': dict(status_counts),
                        'original_empty_answers_preserved': empty,
                        'source_prompt_variant': m.get('source_variant'),
                        'records_sha256': sha(path)})
    parity_keys = ['judge_model', 'judge_base_url', 'temperature', 'seed', 'enable_thinking',
                   'max_tokens', 'grades', 'official_prompt_sha256', 'judge_input_fields',
                   'images_context_captions_sent_to_judge', 'official_prompt_url',
                   'parser_difference_from_public_code', 'aggregation']
    parity = []
    for previous in ['memgallery_gpt5_deepseek_judge_20261104', 'memgallery_qwen_all_methods_deepseek_judge_20261104']:
        previous_root = ROOT / 'results' / previous
        previous_protocol = read(previous_root / 'protocol.json')
        for key in parity_keys:
            assert protocol[key] == previous_protocol[key], (previous, key)
        assert (run / 'judge_prompt.txt').read_bytes() == (previous_root / 'judge_prompt.txt').read_bytes()
        parity.append({'root': str(previous_root), 'matching_fields': parity_keys, 'prompt_identical': True,
                       'previous_deduplication_description': previous_protocol['deduplication'],
                       'current_deduplication_description': protocol['deduplication']})
    prompt_path = ROOT / 'scripts/evaluation/prompts/memgallery_official_five_grade_judge.txt'
    assert sha(prompt_path) == protocol['official_prompt_sha256']
    prompt = prompt_path.read_text()
    actual_requests = {row['request_sha256']: row for row in rows(run / 'requests.jsonl')}
    qmap = rows(run / 'question_map.jsonl')
    assert len(qmap) == len(all_rows) == 13688
    seen, reconstructed = set(), {}
    for row in qmap:
        key = (row['method_id'], row['id'])
        assert key not in seen
        seen.add(key)
        original = all_rows[key]
        assert {k: v for k, v in row.items() if k not in ['method_id', 'request_sha256']} == original
        text = re.sub(r'\{\{(question|ground_truth|model_output)\}\}', lambda match: original[match.group(1)], prompt)
        wire = {'model': 'deepseek-v4.1-flash',
                'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}],
                'temperature': 0, 'seed': 42, 'max_tokens': 1024, 'n': 1, 'stream': False, 'enable_thinking': False}
        serialized = json.dumps(wire, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        digest = hashlib.sha256(serialized).hexdigest()
        assert digest == row['request_sha256'] and actual_requests[digest]['request'] == wire
        reconstructed[digest] = wire
        assert {'method_id': row['method_id'], 'id': row['id']} in actual_requests[digest]['uses']
    assert len(reconstructed) == len(actual_requests) == protocol['unique_requests'] == 8257
    assert sum(len(r['uses']) for r in actual_requests.values()) == 13688
    result = {'status': 'passed', 'api_calls': 0, 'method_count': 8, 'total_method_questions': 13688,
              'unique_judge_wire_requests': 8257, 'verified_source_files': len(hash_checks),
              'source_sha256': hash_checks, 'methods': reports, 'protocol_parity': parity,
              'every_wire_request_independently_reconstructed': True,
              'only_original_question_gold_answer_sent_to_judge': True,
              'all_original_empty_answers_preserved': 27,
              'manifest_sha256': sha(manifest_path), 'judge_protocol_sha256': sha(run / 'protocol.json'),
              'review_script_sha256': sha(Path(__file__))}
    out = HERE / 'independent_input_review.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'api_calls', 'method_count',
                     'total_method_questions', 'unique_judge_wire_requests', 'verified_source_files',
                     'all_original_empty_answers_preserved']}, indent=2))


if __name__ == '__main__':
    main()
