"""Freeze the existing paired MemEye reader payloads for a reader-only ablation.

The historical protocol is retained verbatim. Some upstream files were moved or
edited after that run; those differences are reported, not silently forgiven.
The exact frozen payloads are reconstructed against the current source fields.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READER = ROOT / 'reports/all_master_answer_query_20261021/reader'
RELOCATED = ROOT / 'data/independent_visual_inputs'
CONDITIONS = ('baseline_question_only', 'master_answer_query')
REQUEST_FIELDS = ('record_id', 'id', 'task', 'condition', 'reader_text',
                  'query_images', 'query_image_paths', 'top_memory_ids', 'payload_sha256')


def sha(path: Path | str) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hash(path, expected):
    require(sha(path) == expected, f'Input hash mismatch: {path}')


def request_from_row(row, original, memories, system):
    """Validate evidence and return an explicit gold-free request allowlist."""
    for key in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths'):
        require(row[key] == original[key], f'Original question field differs: {row["record_id"]}/{key}')
    ids = row['top_memory_ids']
    require(len(ids) == len(set(ids)) == 5, 'Expected five distinct ordered memories')
    require(all(mid.split('/', 1)[0] == row['task'] for mid in ids), 'Memory domain mismatch')
    expected = (
        'Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
        + '\n\n'.join(f'Memory {rank}: {memories[mid]}' for rank, mid in enumerate(ids, 1))
        + '\n\nQuestion:\n' + original['question'] + '\n\nOptions:\n'
        + '\n'.join(f'{key}. {original["options"][key]}' for key in 'ABCD')
        + '\n\nAnswer with one letter only.'
    )
    require(row['reader_text'] == expected, f'Original reader evidence differs: {row["record_id"]}')
    require([image['path'] for image in row['query_images']] == original['query_image_paths'],
            'Original query image order differs')
    for image in row['query_images']:
        verify_hash(image['path'], image['sha256'])
    payload = {'system': system, 'user_text': expected, 'query_images': row['query_images']}
    require(digest(payload) == row['payload_sha256'], f'Reader payload hash mismatch: {row["record_id"]}')
    return {key: row[key] for key in REQUEST_FIELDS}


def paired_summary(pairs):
    n = len(pairs)
    baseline = sum(row['baseline_correct'] for row in pairs)
    expanded = sum(row['expanded_correct'] for row in pairs)
    wins = sum(not row['baseline_correct'] and row['expanded_correct'] for row in pairs)
    losses = sum(row['baseline_correct'] and not row['expanded_correct'] for row in pairs)
    discordant = wins + losses
    pvalue = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
                  / 2**discordant) if discordant else 1.)
    return {
        'questions': n, 'baseline_correct': baseline, 'expanded_correct': expanded,
        'baseline_accuracy': baseline / n if n else None,
        'expanded_accuracy': expanded / n if n else None,
        'delta_percentage_points': 100 * (expanded - baseline) / n if n else None,
        'wrong_to_correct': wins, 'correct_to_wrong': losses,
        'answer_changed': sum(row['baseline_answer'] != row['expanded_answer'] for row in pairs),
        'same_reader_payload': sum(row['same_reader_payload'] for row in pairs),
        'same_top5_order': sum(row['same_top5_order'] for row in pairs),
        'same_top5_set': sum(row['same_top5_set'] for row in pairs),
        'mcnemar_exact_two_sided_p': pvalue,
    }


def validate_responses(rows, responses, protocol):
    indexed = {row['record_id']: row for row in rows}
    answers = {row['record_id']: row for row in responses}
    require(len(answers) == len(responses) == len(indexed) == len(rows)
            and set(answers) == set(indexed), 'Qwen reference coverage or duplicate mismatch')
    for response in responses:
        original = indexed[response['record_id']]
        require(response['question_id'] == original['id'] and response['condition'] == original['condition'],
                'Qwen reference question or condition mismatch')
        require(response['run_id'] == protocol['experiment_run_id']
                and response['experiment_signature'] == protocol['experiment_signature'],
                'Qwen reference run signature mismatch')
        require(response['payload_sha256'] == original['payload_sha256'], 'Qwen reference payload mismatch')
        require(response['gold_choice'] == original['gold_choice'], 'Qwen reference label mismatch')
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', response['raw_response'].strip(), re.I)
        choice = match.group(1).upper() if match else None
        require(choice == response['predicted_choice'] and response['correct'] == (choice == original['gold_choice']),
                'Qwen reference answer scoring mismatch')
        require(match is not None and response['status'] == 'ok' and not response['hit_token_limit'],
                'Qwen reference has an invalid or truncated answer')
        require(response['historical_response_reuse'] is False, 'Historical Qwen response reuse')
        require(response['query_image_count'] == len(original['query_images']) and response['batch_size'] == 1,
                'Qwen image channel or batch mismatch')
        if response['fresh_inference_executed_for_this_record']:
            require(response['same_run_shared_inference_record_id'] is None, 'Fresh response has a shared source')
        else:
            shared = answers[response['same_run_shared_inference_record_id']]
            require(shared['fresh_inference_executed_for_this_record'] is True, 'Invalid Qwen shared inference source')
            for key in ('payload_sha256', 'rendered_prompt_sha256', 'raw_response', 'generated_token_ids', 'execution_id'):
                require(response[key] == shared[key], f'Qwen shared inference mismatch: {key}')
    fresh = [row for row in responses if row['fresh_inference_executed_for_this_record']]
    require(len(fresh) == len({row['payload_sha256'] for row in fresh})
            == len({row['payload_sha256'] for row in rows}), 'Qwen fresh inference coverage differs')
    return answers


def commit_artifacts(output, artifacts):
    """Check every existing file before writing anything; never overwrite drift."""
    for name, content in artifacts.items():
        path = output / name
        if path.exists():
            require(path.read_bytes() == content, f'Refusing to overwrite changed input: {path}')
    if (output / 'responses.jsonl').exists():
        require(all((output / name).exists() for name in artifacts),
                'Refusing to create missing inputs after responses exist')
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        path = output / name
        if not path.exists():
            path.write_bytes(content)


def prepare_inputs(output: Path, source_reader: Path = DEFAULT_READER) -> dict:
    output, source_reader = Path(output).resolve(), Path(source_reader).resolve()
    base = source_reader.parent
    source_paths = [source_reader / name for name in (
        'protocol.json', 'qa_manifest.jsonl', 'pairs.json', 'responses.jsonl', 'summary.json')]
    source_paths += [base / name for name in ('protocol.json', 'retrieval_results.jsonl',
                                             'candidates.jsonl', 'candidate_generation_protocol.json')]
    source_paths += [RELOCATED / name for name in ('qa_manifest.jsonl', 'protocol.json', 'retrieval_memories.jsonl')]
    source_paths.append(Path(__file__).resolve())
    source_hashes = {str(path): sha(path) for path in source_paths}
    protocol, root_protocol = read(source_reader / 'protocol.json'), read(base / 'protocol.json')
    unsigned = {key: value for key, value in protocol.items() if key != 'experiment_signature'}
    require(digest(unsigned) == protocol['experiment_signature'], 'Frozen reader protocol signature mismatch')
    for path in (source_reader / 'qa_manifest.jsonl', source_reader / 'pairs.json', base / 'retrieval_results.jsonl'):
        verify_hash(path, protocol['source_sha256'][str(path)])
    verify_hash(base / 'candidates.jsonl', root_protocol['source_sha256'][str(base / 'candidates.jsonl')])
    candidate_protocol = read(base / 'candidate_generation_protocol.json')
    verify_hash(base / 'candidates.jsonl', candidate_protocol['candidates_sha256'])
    settings = protocol['reader_settings']
    system = settings['reader_system_prompt']
    require(system == read(RELOCATED / 'protocol.json')['reader_system_prompt'], 'Relocated source system prompt mismatch')
    original_rows = lines(RELOCATED / 'qa_manifest.jsonl')
    originals = {row['id']: row for row in original_rows}
    memory_rows = lines(RELOCATED / 'retrieval_memories.jsonl')
    memories = {row['id']: row['reader_text'] for row in memory_rows}
    rows, source_pairs = lines(source_reader / 'qa_manifest.jsonl'), read(source_reader / 'pairs.json')
    retrieval_rows = lines(base / 'retrieval_results.jsonl')
    retrieval = {row['question_id']: row for row in retrieval_rows}
    candidates = lines(base / 'candidates.jsonl')
    require(len(originals) == len(original_rows) == len(retrieval) == len(retrieval_rows) == len(source_pairs) == 371,
            'Expected the complete 371-question frozen experiment')
    require(len(rows) == 742 and len(memories) == len(memory_rows) == 438, 'MemEye input count mismatch')
    require({pair['question_id'] for pair in source_pairs} == set(originals), 'Frozen pair question coverage mismatch')
    require({row['id'] for row in candidates} == set(originals) and len(candidates) == 371,
            'Frozen candidate coverage mismatch')
    require(Counter(row['condition'] for row in rows) == dict.fromkeys(CONDITIONS, 371), 'Condition coverage mismatch')
    requests = []
    for row in rows:
        require(row['record_id'] == f'{row["id"]}::{row["condition"]}', 'Invalid condition record ID')
        request = request_from_row(row, originals[row['id']], memories, system)
        result = retrieval[row['id']]
        require(all(result[key] == originals[row['id']][key] for key in ('question', 'options', 'query_image_paths')),
                'Retrieval question fields mismatch')
        condition = next(item for item in result['conditions'] if item['name'] == row['condition'])
        require(condition['top5_memory_ids'] == row['top_memory_ids']
                and condition['top5_scores'] == row['top_query_scores'], 'Frozen Top5 differs')
        require([item['reader_text'] for item in condition['reader_memories']]
                == [memories[mid] for mid in row['top_memory_ids']], 'Retrieval evidence mismatch')
        requests.append(request)
    indexed = {row['record_id']: row for row in rows}
    require(len(indexed) == 742 and set(originals) == {row['id'] for row in rows}, 'Question coverage mismatch')
    responses = lines(source_reader / 'responses.jsonl')
    answers = validate_responses(rows, responses, protocol)
    pairs, scored = [], []
    for pair in source_pairs:
        qid = pair['question_id']
        first, second = [indexed[f'{qid}::{name}'] for name in CONDITIONS]
        require(pair['conditions'] == [first, second], 'Frozen pair payload mismatch')
        small = {'question_id': qid, 'domain': first['task'], 'query_has_image': bool(first['query_images']),
                 'same_reader_payload': first['payload_sha256'] == second['payload_sha256'],
                 'same_top5_order': first['top_memory_ids'] == second['top_memory_ids'],
                 'same_top5_set': set(first['top_memory_ids']) == set(second['top_memory_ids'])}
        require(all(pair[key] == value for key, value in small.items()), 'Frozen pair comparison mismatch')
        pairs.append(small)
        a, b = answers[first['record_id']], answers[second['record_id']]
        scored.append({**small, 'baseline_answer': a['predicted_choice'], 'expanded_answer': b['predicted_choice'],
                       'baseline_correct': a['correct'], 'expanded_correct': b['correct']})
    summary = read(source_reader / 'summary.json')
    require(summary['overall'] == paired_summary(scored), 'Historical summary does not match Qwen responses')
    for domain in {row['domain'] for row in scored}:
        require(summary['by_domain'][domain] == paired_summary([row for row in scored if row['domain'] == domain]),
                f'Historical per-domain summary mismatch: {domain}')
    unique_images = {image['path'] for row in rows for image in row['query_images']}
    source_hashes.update({path: sha(path) for path in sorted(unique_images)})
    counts = {'questions': 371, 'domains': len({row['task'] for row in rows}), 'logical_conditions': 742,
              'unique_reader_payloads': len({row['payload_sha256'] for row in rows}),
              'same_payload_pairs': sum(row['same_reader_payload'] for row in pairs),
              'query_image_questions': sum(row['query_has_image'] for row in pairs),
              'unique_query_image_files': len(unique_images), 'memory_nodes': 438,
              'qwen_fresh_inferences': sum(row['fresh_inference_executed_for_this_record'] for row in responses),
              'qwen_same_run_shared_records': sum(not row['fresh_inference_executed_for_this_record'] for row in responses)}
    require((counts['domains'], counts['unique_reader_payloads'], counts['same_payload_pairs'],
             counts['query_image_questions'], counts['unique_query_image_files'], counts['qwen_fresh_inferences'],
             counts['qwen_same_run_shared_records']) == (8, 719, 23, 8, 6, 719, 23), 'Frozen experiment counts differ')
    require((summary['overall']['baseline_correct'], summary['overall']['expanded_correct']) == (184, 195),
            'Frozen Qwen reference accuracy differs')
    for key, expected in {'logical_conditions': 742, 'fresh_model_inferences': 719,
                          'same_run_shared_records': 23, 'historical_response_reuse': False,
                          'format_errors': 0, 'token_limit_hits': 0,
                          'generation_executions': len({row['execution_id'] for row in responses})}.items():
        require(summary[key] == expected, f'Qwen reference provenance summary mismatch: {key}')
    for name in CONDITIONS:
        selected = [row for row in responses if row['condition'] == name]
        require(summary['conditions'][name] == {'questions': 371, 'correct': sum(row['correct'] for row in selected),
                                                'format_errors': 0, 'token_limit_hits': 0},
                f'Qwen condition summary mismatch: {name}')
    for name, value in (('text_only', False), ('with_query_image', True)):
        require(summary['by_query_modality'][name]
                == paired_summary([row for row in scored if row['query_has_image'] == value]),
                f'Qwen query modality summary mismatch: {name}')
    drift = []
    for path, expected in protocol['source_sha256'].items():
        current = sha(path) if Path(path).is_file() else None
        if current != expected:
            drift.append({'path': path, 'historical_sha256': expected, 'current_sha256': current,
                          'status': 'missing' if current is None else 'changed'})
    audit = {
        'passed': True, 'counts': counts, 'direct_frozen_artifact_hashes_verified': True,
        'all_reader_payload_hashes_recomputed': True, 'all_question_image_hashes_verified': True,
        'original_question_options_labels_and_top5_verified': True,
        'reader_text_rebuilt_from_current_original_dialogue_and_full_caption': True,
        'candidate_statements_not_in_reader': True, 'requests_have_no_gold_fields': True,
        'qwen_answers_reparsed_and_rescored': True, 'qwen_historical_response_reuse': False,
        'historical_upstream_source_drift': drift,
        'historical_upstream_all_hashes_match': not drift,
        'provenance_note': 'Historical upstream paths and script bytes have changed. The historical protocol is untouched; '
                           'exact frozen reader payloads and original evidence fields were independently reconstructed.',
        'memory_pool': '438 image-bearing memory leaves, complete same-domain pool; Top5 frozen.',
        'query_image_policy': 'Original query image bytes; no historical memory images are sent or copied.',
        'candidate_policy': 'Existing assistant-authored question-and-options hypotheses remain frozen; retrieval only.',
        'source_sha256': source_hashes,
    }
    response_keys = ('record_id', 'condition', 'payload_sha256', 'raw_response', 'predicted_choice', 'gold_choice',
                     'correct', 'fresh_inference_executed_for_this_record',
                     'same_run_shared_inference_record_id', 'historical_response_reuse')
    reference = {'reader_model': settings['reader_model'], 'reader_settings': settings,
                 'experiment_signature': protocol['experiment_signature'],
                 'experiment_run_id': protocol['experiment_run_id'], 'summary': summary,
                 'responses': [{**{key: row[key] for key in response_keys}, 'id': row['question_id']} for row in responses]}
    encoded = lambda value: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    jsonl = lambda values: ''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n' for value in values).encode()
    artifacts = {
        'requests.jsonl': jsonl(requests),
        'gold.jsonl': jsonl([{key: row[key] for key in ('id', 'gold_choice', 'gold_ids')} for row in original_rows]),
        'pairs.json': encoded(pairs), 'qwen_reference.json': encoded(reference), 'input_audit.json': encoded(audit),
    }
    metadata = {'reader_system_prompt': system, 'reader_settings': settings, 'counts': counts,
                'source_sha256': source_hashes,
                'artifact_sha256': {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}}
    artifacts['input_metadata.json'] = encoded(metadata)
    require(all(sha(path) == expected for path, expected in source_hashes.items()), 'Source changed during input preparation')
    commit_artifacts(output, artifacts)
    return metadata
