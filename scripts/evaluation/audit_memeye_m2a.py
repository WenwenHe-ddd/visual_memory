"""Independent artifact/coverage/metric audit for a completed M2A MemEye pilot."""
import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def audit(out):
    protocol = read(out / 'protocol.json')
    manifest = read(out / 'completion_manifest.json')
    labels = read(out / 'evaluation_labels.json')
    observations = read(out / 'observations.json')
    queries = read(out / 'retrieval_queries.json')
    answers = rows(out / 'responses.jsonl')
    construction = rows(out / 'construction.jsonl')
    calls = rows(out / 'llm_calls.jsonl')
    before, after = read(out / 'semantic_before_qa.json'), read(out / 'semantic_after_qa.json')
    raw_before, raw_after = read(out / 'raw_before_qa.json'), read(out / 'raw_after_qa.json')
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(manifest['protocol_signature'] == protocol['signature'], 'Protocol signature mismatch')
    unsigned = dict(protocol)
    unsigned.pop('signature')
    computed = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    check(computed == protocol['signature'], 'Protocol signature does not match its contents')
    for name, expected in manifest['artifacts'].items():
        check(sha(out / name) == expected, 'Changed artifact: ' + name)
    for name, expected in protocol['sources'].items():
        check(sha(name) == expected, 'Changed input source: ' + name)
    for name, expected in protocol['prepared_sha256'].items():
        check(sha(out / name) == expected, 'Changed prepared input: ' + name)
    for name, expected in protocol['implementation_sha256'].items():
        snapshot = out / 'code_snapshot' / name
        check(snapshot.exists() and sha(snapshot) == expected, 'Code snapshot missing or changed: ' + name)
    if protocol.get('memory_reused_from'):
        source = Path(protocol['memory_reused_from'])
        for name, expected in protocol['frozen_memory_inputs'].items():
            check(sha(source / name) == expected, 'Frozen source changed: ' + name)
        check(before == read(source / 'semantic_before_qa.json'), 'Reused semantic memory differs from source')
        check(raw_before == read(source / 'raw_before_qa.json'), 'Reused raw memory differs from source')
    expected_ids = [q['id'] for q in labels]
    check([a['id'] for a in answers] == expected_ids, 'QA ID coverage/order mismatch')
    check(len(expected_ids) == len(set(expected_ids)), 'Duplicate QA ID')
    check([o['id'] for o in observations] == [r['id'] for r in construction], 'Construction coverage mismatch')
    check(queries == [{'id': q['id'], 'question': q['question'], 'images': q['query_image_paths']}
                      for q in labels], 'Retrieval input differs from answer-free projection')
    check(before == after, 'Semantic memories changed during QA')
    check(raw_before == raw_after, 'Raw memories changed during QA')
    raw_ids = {r['msg_id'] for r in raw_before}
    source_ids = {r['raw_id'] for r in construction}
    unknown_evidence, unsupported = [], []
    for m in before:
        ev = json.loads(m['evidence_ids'])
        ids = {i for a,b in ev for i in range(a,b+1)}
        if not ids <= raw_ids or any(a > b for a,b in ev):
            unknown_evidence.append(m['id'])
        if not ids & source_ids:
            unsupported.append(m['id'])
        if m['image_path']:
            check(Path(m['image_path']).is_file(), 'Missing memory image: ' + m['image_path'])
    check(not unknown_evidence, 'Semantic evidence refers to invalid raw IDs')
    check(all(c['status'] == 'ok' and not c['hit_token_limit'] for c in calls), 'LLM errors or truncation')
    if protocol.get('llm_backend') == 'deepseek_api':
        import gzip
        from scripts.memory.m2a_local import safe_content
        from scripts.memory.m2a_deepseek import provider
        for call in calls:
            check(call['requested_model'] == call['returned_model'] == protocol['model']
                  and call['seed'] == 42 and call['enable_thinking'] is False,
                  'DeepSeek provider/model/seed mismatch')
            cache = Path(call['cache_directory'])
            wire = json.loads(gzip.decompress((cache/'request.json.gz').read_bytes()))
            check(provider.request_sha256(wire) == call['request_sha256']
                  and safe_content(wire) == call['request'], 'DeepSeek transmitted request differs from log')
            check(wire['model'] == protocol['model'] and wire['seed'] == 42
                  and wire['temperature'] == 0 and wire['enable_thinking'] is False,
                  'DeepSeek wire settings differ')
            saved = read(cache/'response.json')
            check(saved['raw_response'] == call['raw_response'] and saved['tool_calls'] == call['tool_calls']
                  and saved['response_id'] == call['response_id'], 'DeepSeek cached response differs from call log')
    if protocol.get('llm_backend') == 'gpt5_api':
        import gzip
        from scripts.memory.m2a_local import safe_content
        from scripts.memory.m2a_gpt5 import provider
        for call in calls:
            check(call['requested_model'] == protocol['model'] == 'gpt-5'
                  and provider.valid_returned_model(call['returned_model'])
                  and call['seed'] == 42 and call['reasoning_effort'] == 'minimal',
                  'GPT-5 model or settings mismatch')
            cache = Path(call['cache_directory'])
            wire = json.loads(gzip.decompress((cache/'request.json.gz').read_bytes()))
            check(provider.request_sha256(wire) == call['request_sha256']
                  and safe_content(wire) == call['request'], 'GPT-5 wire differs from log')
            check(wire['model'] == 'gpt-5' and wire['seed'] == 42
                  and wire['reasoning_effort'] == 'minimal' and 'temperature' not in wire
                  and wire['max_completion_tokens'] >= 2048 and wire['store'] is False,
                  'GPT-5 wire settings differ')
            saved = read(cache/'response.json')
            check(all(saved[k] == call[k] for k in
                      ('raw_response','tool_calls','response_id','returned_model','usage')),
                  'GPT-5 cached response differs from log')
    check(all(a['search_calls'] > 0 for a in answers), 'Some questions did not search memory')
    if 'max_search_calls_per_question' in protocol:
        from collections import Counter
        search_counts = Counter(s['sample_id'] for s in rows(out / 'searches.jsonl')
                                if s['stage'] == 'retrieval')
        limit = protocol['max_search_calls_per_question']
        check(limit == 1 + protocol['max_additional_search_rounds'] and limit >= 1,
              'Invalid per-question search budget')
        check(all(a['search_calls'] == search_counts[a['id']] <= limit for a in answers),
              'Search counts differ from log or exceed the per-question budget')
    if protocol.get('answer_format') == 'open':
        from scripts.evaluation.m2a_inputs import gallery_reader
        scorer = gallery_reader().score_answer
        scores = [scorer(a['raw_answer'], q['answer']) for a, q in zip(answers, labels)]
        correct = sum(int(s['em']) for s in scores)
        check(all(a['correct'] == bool(s['em']) and all(abs(a[k] - s[k]) < 1e-12 for k in s)
                  for a, s in zip(answers, scores)), 'Open-answer metrics differ from recomputation')
    else:
        correct = sum(a['predicted_choice'] == q['gold_choice'] for a,q in zip(answers,labels))
        check(all(a['correct'] == (a['predicted_choice'] == q['gold_choice'])
                  for a,q in zip(answers,labels)), 'Saved correctness differs from recomputed metric')
    summary = read(out / 'summary.json')
    check(summary['correct'] == correct and summary['completed_questions'] == len(labels),
          'Summary numerator/denominator mismatch')
    check(all(a['status'] == 'ok' for a in answers), 'Reader format errors')
    if protocol.get('construction_mode') == 'single_pass':
        build_calls = [c for c in calls if c['stage'] == 'construction']
        groups = [(key, list(group)) for key, group in groupby(build_calls, key=lambda c: c['sample_id'])]
        max_attempts = protocol.get('extraction_max_attempts', 1)
        check(max_attempts in (1, 2, 3), 'Invalid extraction attempt limit')
        check([key for key, group in groups] == [o['id'] for o in observations],
              'Construction call coverage/order differs from original rounds')
        check(all(1 <= len(group) <= max_attempts and record['calls'] == len(group)
                  for (_, group), record in zip(groups, construction)), 'Extraction attempt counts differ')
        retry_path = out / 'extraction_retries.jsonl'
        retries = rows(retry_path) if retry_path.exists() else []
        rejected = []
        from scripts.memory.m2a_single_pass import parse_extraction
        for (key, group), obs in zip(groups, observations):
            for attempt, call in enumerate(group[:-1], 1):
                try:
                    parse_extraction(call['raw_response'], bool(obs['image']))
                except ValueError:
                    pass
                else:
                    check(False, 'Valid extraction was unnecessarily retried: ' + key)
                rejected.append((key, attempt, call['raw_response']))
        check([(r['id'], r['attempt'], r['raw_response']) for r in retries] == rejected
              and all(r['will_retry'] for r in retries), 'Extraction retry logs differ from rejected calls')
        check(all(not c['tools'] and not c['tool_calls'] for c in build_calls), 'Construction used tools')
        check(len(before) == len(raw_before) == len(observations), 'Not exactly one raw and semantic memory per round')
        memories = {m['id']: m for m in before}
        raw_by_id = {r['msg_id']: r for r in raw_before}
        extractions = rows(out / 'extractions.jsonl')
        check([e['id'] for e in extractions] == [o['id'] for o in observations], 'Extraction coverage mismatch')
        for obs, record, extraction, (_, group) in zip(observations, construction, extractions, groups):
            call = group[-1]
            check(record.get('extraction_attempts', 1) == extraction.get('extraction_attempts', 1) == len(group),
                  'Accepted extraction attempt metadata differs: ' + obs['id'])
            memory, raw = memories.get(record['memory_id']), raw_by_id.get(record['raw_id'])
            check(memory is not None and json.loads(memory['evidence_ids']) == [[record['raw_id'], record['raw_id']]],
                  'Incorrect program-bound evidence: ' + obs['id'])
            check(raw is not None and raw['text'] == obs['text'] and raw['image_path'] == obs['image'],
                  'Raw round differs from input: ' + obs['id'])
            check(memory is not None and memory['image_path'] == (obs['image'] or '') and
                  memory['image_caption'] == extraction['image_caption'] and
                  memory['text'].endswith('\n' + extraction['facts']), 'Memory differs from extraction: ' + obs['id'])
            repair = extraction.get('json_repair')
            check(repair in (None, 'append_final_object_brace') and record.get('json_repair') == repair,
                  'Unexpected extraction JSON repair: ' + obs['id'])
            original_text = call['raw_response'] + ('}' if repair == 'append_final_object_brace' else '')
            model_value = json.loads(original_text)
            # Construction strips surrounding whitespace from both string fields.
            # Reproduce that normalization before comparing saved extractions.
            model_value = {k: v.strip() if isinstance(v, str) else v for k, v in model_value.items()}
            discarded = record.get('discarded_caption_without_image', '')
            check(discarded == extraction.get('discarded_caption_without_image', ''),
                  'Discarded caption log mismatch: ' + obs['id'])
            if not obs['image']:
                check(discarded == model_value['image_caption'].strip(),
                      'Unsupported caption not recorded: ' + obs['id'])
                model_value['image_caption'] = ''
            else:
                check(not discarded, 'Caption discarded despite real image: ' + obs['id'])
            check(model_value == {k: extraction[k] for k in ('facts', 'image_caption')},
                  'Extraction differs from Qwen output: ' + obs['id'])
        searches = rows(out / 'searches.jsonl')
        limits = protocol['retrieval_limits']
        check(all(s['stage'] == 'retrieval' and len(s['results']) <= limits['final'] for s in searches),
              'Search during construction or final Top-K exceeded')
        operations = read(out / 'semantic_log.json')
        check(all(op['op'] in ('add', 'search') for op in operations), 'Memory updated/deleted')
        for op in operations:
            if op['op'] == 'search':
                check(op['limits'] == limits and len(op['text_candidate_ids']) <= limits['text_fused'] and
                      len(op['image_candidate_ids']) <= limits['image'] and len(op['final_ids']) <= limits['final'],
                      'Actual search limits differ from protocol')
    if protocol.get('answer_format') == 'open':
        check(all(abs(summary['metrics'][k] - sum(s[k] for s in scores) / len(scores)) < 1e-12
                  for k in scores[0]), 'Aggregate open-answer metrics differ')
    for obs in observations:
        check(set(obs) == {'id','round','timestamp','session_id','text','image','image_sha256'},
              'Unexpected construction field: ' + obs['id'])
        if obs['image']:
            check(sha(obs['image']) == obs['image_sha256'], 'Changed observation image: ' + obs['id'])
    tool_failures = {}
    for call in calls:
        for message in call['messages']:
            content = message.get('content')
            if (message.get('type') == 'tool' and isinstance(content, str)
                    and content.startswith(('Tool error:', 'Error creating memory:', 'Error parsing', 'Tool call failed:'))):
                tool_failures[message.get('tool_call_id')] = content
    result = {'passed': not errors, 'errors': errors,
              'questions': len(labels), 'correct': correct, 'accuracy': correct / len(labels),
              'original_rounds': len(observations), 'semantic_memories': len(before),
              'invalid_evidence_memory_ids': unknown_evidence,
              'memories_without_original_round_evidence': unsupported,
              'agent_calls': len(calls), 'is_complete_task': protocol['is_complete_task'],
              'unique_tool_failures': tool_failures,
              'artifact_hashes_checked': len(manifest['artifacts'])}
    (out / 'independent_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    result = audit(parser.parse_args().output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
