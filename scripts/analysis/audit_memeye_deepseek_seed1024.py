"""Read-only final MCQ/metric audit; partial sources never become final results.

No inference or HTTP calls. Native refused, malformed, token-limited and empty
outputs remain in the fixed 371-question denominator. Open/Judge auditing is a
separate prerequisite of the companion final workbook exporter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import memeye_deepseek_seed1024_metrics as mapping

OUT, MODEL, METHODS = mapping.OUT, mapping.MODEL, mapping.METHODS
NATIVE = {'gen_agents': {(0., 2048)}, 'memoryos': {(.7, 2000), (.3, 100), (0., 10)},
          'a_mem': {(.7, 1000)}, 'simplemem': {(.1, 16384), (.2, 16384), (.3, 16384)}}
TRACKED = {}


def track(path, value):
    path = Path(path).resolve(); stat = path.stat(); previous = TRACKED.get(str(path))
    require(previous is None or previous['sha256'] == value, 'Audited artifact changed during this audit: ' + str(path))
    TRACKED[str(path)] = dict(sha256=value, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return value


def read(path):
    data = Path(path).read_bytes(); track(path, hashlib.sha256(data).hexdigest())
    return json.loads(data)


def rows(path):
    data = Path(path).read_bytes(); track(path, hashlib.sha256(data).hexdigest())
    return [json.loads(s) for s in data.decode().splitlines() if s.strip()]


def sha(path):
    with Path(path).open('rb') as handle:
        value = hashlib.file_digest(handle, 'sha256').hexdigest()
    return track(path, value)


def digest(value, compact=True):
    kwargs = dict(sort_keys=True, ensure_ascii=False, allow_nan=False)
    if compact:
        kwargs['separators'] = (',', ':')
    return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(values, key='id'):
    result = {r[key]: r for r in values}
    require(len(result) == len(values), 'Duplicate ' + key)
    return result


def hashes(values):
    for path, expected in values.items():
        require(sha(path) == expected, 'Pinned artifact changed: ' + str(path))


def method_paths(method, output=OUT):
    output = Path(output)
    if method in mapping.OURS:
        source = output / 'ours'
        return dict(source=source, mcq=source / 'reader/responses.jsonl', summary=source / 'summary.json',
                    marker=source / 'independent_audit.json', opened=source / 'open' / method,
                    metric=mapping.metric_path(method, output), condition=mapping.OURS[method][1])
    group = 'text_methods' if method in mapping.TEXT else 'visual_methods'
    source = output / group / method
    online = source if method in mapping.TEXT else source / 'online'
    return dict(source=source, mcq=online / 'qa_responses.jsonl', summary=online / 'summary.json',
                marker=online / ('completion_manifest.json' if method != 'clip_image_dialogue' else 'completion.json'),
                opened=output / 'open/text_methods' / method if method in mapping.TEXT else source / 'open',
                metric=mapping.metric_path(method, output))


def read_mcq(method, output=OUT, reference=False):
    if reference:
        source = Path(mapping.reference_mapping()[method]['source_root'])
        if method in mapping.OURS:
            condition = mapping.OURS[method][1]
            values = [r for folder in ('weighted_evaluation', 'complementary_evaluation')
                      for r in rows(source / folder / 'reader/responses.jsonl') if r['condition'] == condition]
        else:
            values = rows(source / ('online/qa_responses.jsonl' if method == 'hippomm' else 'qa_responses.jsonl'))
    else:
        paths = method_paths(method, output)
        values = rows(paths['mcq'])
        if 'condition' in paths:
            values = [r for r in values if r['condition'] == paths['condition']]
    return [dict(r, id=r.get('id', r.get('question_id'))) for r in values]


def prediction(method, row):
    if method == 'hippomm':
        return row['answer']
    raw = row.get('raw_response')
    if method in mapping.TEXT:
        raw = raw.strip() if isinstance(raw, str) else ''
        return raw if raw in ('A', 'B', 'C', 'D') and not row.get('hit_token_limit') and not row.get('refusal') else None
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', raw.strip(), re.I) if isinstance(raw, str) else None
    return match.group(1).upper() if match else None


def audit_text_calls(source, p):
    raw = rows(source / 'api_responses.jsonl')
    by_id = unique([dict(r, id=r['output']['response_id']) for r in raw])
    calls, settings = {}, Counter()
    memory_caps = invalid = 0
    for path in sorted((source / 'calls').glob('*.json')):
        call = read(path); req, out = call['request'], call['output']
        require(digest(req, False) == path.stem and req['seed'] == 1024
                and req['experiment_signature'] == p['experiment_signature'], 'Text logical call identity differs')
        logged = by_id[out['response_id']]; wire = logged['request']
        require(wire['seed'] == 1024 and wire['model'] == MODEL and wire['enable_thinking'] is False
                and wire['messages'] == req['messages'] and wire['max_tokens'] == req['max_new_tokens']
                and wire['temperature'] == req['temperature'], 'Text actual native wire differs')
        require(out['seed'] == 1024 and out['requested_model'] == out['returned_model'] == MODEL
                and out['enable_thinking'] is False and out['request_sha256'] == digest(wire)
                and all(out.get(k) == v for k, v in logged['output'].items())
                and 'inherited_from' not in call, 'Text actual wire/raw output join differs')
        setting = (req['temperature'], req['max_new_tokens'])
        require(setting in (NATIVE[p['mode']] if req['kind'] == 'memory' else {(0., 16)}), 'Text native temperature/cap changed')
        require(out['hit_token_limit'] == (out['finish_reason'] == 'length'), 'Text token-limit flag differs')
        require(out['input_tokens'] + req['max_new_tokens'] <= p['context_budget'], 'Text native context budget exceeded')
        settings[(req['kind'], *setting)] += 1
        memory_caps += req['kind'] == 'memory' and out['hit_token_limit']
        invalid += out.get('structured_output_valid') is False
        calls[path.stem] = call
    history = text_attempt_history(source, p, raw)
    return calls, dict(memory_token_limit_hits=memory_caps, invalid_structured_outputs=invalid,
        native_call_settings=[dict(kind=k[0], temperature=k[1], cap=k[2], calls=n) for k, n in settings.items()],
        actual_text_calls_checked=len(calls), raw_successful_response_records=len(raw), **history)


def text_attempt_history(source, p, raw):
    """Retain aborted-run failures; attempt numbers restart per model invocation."""
    attempts = rows(source / 'api_attempts.jsonl')
    by_response = unique([dict(r, id=r['output']['response_id']) for r in raw])
    for record in raw:
        wire, out = record['request'], record['output']
        require(wire['seed'] == 1024 and wire['model'] == MODEL and wire['enable_thinking'] is False
                and (wire['temperature'], wire['max_tokens']) in NATIVE[p['mode']] | {(0., 16)}, 'Historical successful native wire settings changed')
        require(out['request_sha256'] == digest(wire) and out['seed'] == 1024
                and out['requested_model'] == out['returned_model'] == MODEL, 'Historical successful raw output identity differs')
    successful = set(); errors = []; latest_success = {}
    for number, event in enumerate(attempts, 1):
        require(event['status'] in ('ok', 'error') and type(event['attempt']) is int and 1 <= event['attempt'] <= 3,
                'Native API attempt event schema differs')
        if event['status'] == 'ok':
            record = by_response[event['response_id']]
            require(event['request_sha256'] == record['output']['request_sha256'] == digest(record['request']),
                    'Successful attempt does not join directly saved actual wire/output')
            successful.add(event['response_id']); latest_success[event['request_sha256']] = number
        else:
            require(isinstance(event.get('retryable'), bool), 'Failed native API attempt lost retryability metadata')
            errors.append(dict(line=number, **event))
    require(successful == set(by_response), 'Actual raw successes are missing their original attempt records')
    resolved = [e for e in errors if latest_success.get(e['request_sha256'], 0) > e['line']]
    snapshots = []
    recovery = source.parents[1] / 'recovery'
    for path in sorted(recovery.rglob('recovery_manifest.json')) if recovery.exists() else []:
        manifest = read(path)
        if Path(manifest.get('source_root', '')).resolve() != source.resolve():
            continue
        require(manifest.get('passed') is True, 'Historical failure snapshot was not captured consistently')
        retained = cache_calls = 0
        for item in manifest['files']:
            old = Path(item['source'])
            if old.name in ('api_attempts.jsonl', 'api_responses.jsonl', 'call_events.jsonl') and old.parent == source:
                archived = path.parent / item['archive_relative']
                require(sha(archived) == item['sha256'] and old.read_bytes().startswith(archived.read_bytes()),
                        'Historical failed-run append-only ledger was changed or truncated')
                retained += 1
            elif old.parent == source / 'calls':
                archived = path.parent / item['archive_relative']
                require(sha(archived) == item['sha256'] and old.exists() and sha(old) == item['sha256'],
                        'Previously successful native cache was replaced or removed after a failed run')
                cache_calls += 1
        require(retained == 3, 'Failure archive lacks all three original call/attempt/response ledgers')
        snapshots.append(dict(path=str(path), sha256=sha(path), append_only_ledgers_verified=retained,
                              successful_cached_calls_retained=cache_calls))
    return dict(api_attempt_events=len(attempts), historical_api_error_events=len(errors),
        historical_api_errors_with_later_exact_wire_success=len(resolved),
        historical_api_errors_without_later_exact_wire_success=len(errors) - len(resolved),
        historical_api_error_kinds=dict(Counter(str(e.get('error_type')) + ':' + str(e.get('provider_error_code')) for e in errors)),
        historical_error_records=errors, failure_snapshots_verified=snapshots,
        attempt_counting_policy='Append-only event counts, including aborted runs; attempt numbers restart for each native APIModel invocation. No global retry count is invented.')


def source_provenance(method, paths, values):
    source, summary = paths['source'], read(paths['summary'])
    extra = {}
    if method in mapping.TEXT:
        from scripts.memory import text_memory_baselines as baseline
        p = read(source / 'protocol.json')
        require(digest({k: v for k, v in p.items() if k != 'experiment_signature'}, False) == p['experiment_signature'], 'Text protocol signature differs')
        require(p['seed'] == 1024 and p['model'] == MODEL and p['mode'] == method, 'Text protocol model/seed differs')
        marker = read(paths['marker'])
        hashes({str(source / name): marker[key] for name, key in (
            ('protocol.json', 'protocol_sha256'), ('qa_responses.jsonl', 'responses_sha256'), ('stability_audit.json', 'audit_sha256'))})
        observations = read(source / 'observations.json')
        require(sum(map(len, observations.values())) == 848 and len(p['tasks']) == 8, 'Text history coverage differs')
        for task in p['tasks']:
            hashes({str(source / task / 'responses.jsonl'): read(source / task / 'complete.json')['responses_sha256']})
            require(read(source / task / 'memory_stats.json')['stored_rounds'] == len(observations[task]), 'Incomplete task construction')
        calls, extra = audit_text_calls(source, p)
        for r in values:
            call = calls[r['call_key']]
            require(call['request']['messages'] == baseline.reader_messages(r, r['retrieved_context'], p)
                    and call['output']['raw_response'] == r['raw_response']
                    and digest(call['request']['messages'], False) == r['reader_messages_sha256'], 'Text actual MCQ reader payload differs')
    elif method in mapping.OURS:
        from scripts.evaluation import memeye_deepseek_seed1024_transport as transport
        p = read(source / 'protocol.json')
        require(digest(p['contract']) == p['experiment_signature'] and p['contract']['seed'] == 1024
                and p['contract']['model'] == MODEL, 'Ours source protocol differs')
        hashes(read(paths['marker'])['artifact_sha256'])
        requests = unique(rows(source / 'reader/requests.jsonl'), 'record_id')
        memories = unique(rows(source / 'inputs/memories.jsonl'))
        questions = unique(rows(source / 'inputs/questions.jsonl'))
        canonical = unique(rows(mapping.BENCHMARK))
        encoding = read(source / 'inputs/encoding.json')
        retrieval = unique(rows(source / 'retrieval_results.jsonl'))
        hashes(read(source / 'retrieval_audit.json')['output_sha256'])
        seen = set()
        for r in values:
            q = requests[r['record_id']]
            original = questions[r['id']]
            require(all(original[k] == canonical[r['id']][k] for k in ('task', 'question', 'options')), 'Ours canonical fixed question/options differ')
            selected = q['top_memory_ids']
            ranked = next(c for c in retrieval[r['id']]['conditions'] if c['name'] == paths['condition'])
            require(selected == ranked['top5_memory_ids'] and len(selected) == len(set(selected)) == 5
                    and all(memories[i]['task'] == original['task'] for i in selected), 'Ours actual reader Top5 differs from retrieval')
            context = '\n\n'.join(f'Memory {j}: ' + memories[mid]['dialogue'] + '\nMemory description: ' + memories[mid]['caption']
                                  for j, mid in enumerate(selected, 1))
            text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context +
                    '\n\nQuestion:\n' + original['question'] + '\n\nOptions:\n' +
                    '\n'.join(f'{c}. {original["options"][c]}' for c in 'ABCD') + '\n\nAnswer with one letter only.')
            require(q['system'] == encoding['reader_system_prompt'] and q['user_text'] == text
                    and q['query_images'] == original['query_images'], 'Ours complete canonical reader evidence differs')
            require(r['experiment_signature'] == p['experiment_signature'] and r['run_id'] == p['run_id']
                    and r['historical_answer_reuse'] is False and r['payload_sha256'] == q['payload_sha256'], 'Ours new-run response provenance differs')
            path = source / 'api_calls' / (r['request_sha256'] + '.json')
            call = read(path)
            payload = {k: q[k] for k in ('system', 'user_text', 'query_images')}
            wire = transport.build_request(payload, model=MODEL, max_output_tokens=16, image_detail='high')
            require(call['request'] == wire and digest(wire) == r['request_sha256']
                    and call['response'].get('raw_response') == r['raw_response'], 'Ours actual wire/raw output differs')
            require(wire['seed'] == 1024 and wire['enable_thinking'] is False, 'Ours actual wire seed differs')
            require(r.get('returned_model') == MODEL or (r['status'] == 'provider_input_rejected' and r.get('returned_model') is None), 'Ours model identity differs')
            seen.add(r['request_sha256'])
        extra['actual_unique_answer_wires_checked'] = len(seen)
    elif method == 'clip_image_dialogue':
        from scripts.evaluation.memeye_deepseek_seed1024_audit_open_judge import Audit
        visual = Audit(source.parents[1])
        p = read(source / 'online/protocol.json')
        require(p['seed'] == 1024 and p['answer_model'] == MODEL
                and digest({k: v for k, v in p.items() if k != 'experiment_signature'}) == p['experiment_signature'], 'CLIP protocol model/seed/signature differs')
        hashes({str(paths['mcq']): read(paths['marker'])['responses_sha256']})
        questions = unique(rows(source / 'online/qa_manifest.jsonl'))
        from scripts.memory import clip_image_dialogue
        clip_image_dialogue.verify(Path(p['source_manifest']).parent)
        canonical = unique(rows(mapping.BENCHMARK))
        for r in values:
            q = questions[r['id']]
            require(all(q[k] == canonical[r['id']][k] for k in ('task', 'question', 'options', 'gold_choice', 'gold_ids')), 'CLIP canonical question/options differ')
            require(r['experiment_signature'] == p['experiment_signature'] and r['payload_sha256'] == q['payload_sha256'], 'CLIP response payload join differs')
            messages = copy.deepcopy(q['reader_messages'])
            for message in messages:
                if isinstance(message['content'], list):
                    for part in message['content']:
                        if part['type'] == 'image':
                            part['sha256'] = q['reader_image_hashes'][part['path']]
            wire = visual.wire(messages, cap=16); key = digest(wire)
            folder = source / 'online/calls' / key
            actual = json.loads(gzip.decompress((folder / 'wire.json.gz').read_bytes()))
            require(actual == wire and r['request_sha256'] == key, 'CLIP original-image MCQ wire differs')
            attempts = rows(folder / 'attempts.jsonl')
            require([a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1)) and 1 <= len(attempts) <= 3, 'CLIP bounded native attempt sequence differs')
            if r.get('status') == 'provider_rejection':
                require(r['seed'] == 1024 and r.get('returned_model') is None and not r['raw_response']
                        and attempts[-1]['status'] == 'error' and 'data_inspection_failed' in
                        (str(attempts[-1].get('code')) + str(attempts[-1].get('kind'))), 'CLIP rejection lacks actual permanent provider evidence')
            else:
                out = read(folder / 'response.json'); visual.response(out, key)
                require(out['raw_response'] == r['raw_response'] and attempts[-1]['status'] == 'ok'
                        and attempts[-1]['response'] == out, 'CLIP raw answer attempt differs')
        extra['actual_unique_answer_wires_checked'] = len(values)
        for path, info in visual.files.items():
            track(path, info['sha256'])
    else:
        from scripts.evaluation.memeye_deepseek_seed1024_audit_open_judge import Audit
        visual = Audit(source.parents[1])
        p = read(source / 'online/protocol.json')['contract']
        require(p['seed'] == 1024 and p['model'] == MODEL and summary['model'] == MODEL, 'HippoMM model/seed differs')
        require(read(paths['marker'])['provider_and_image_payload_audit'] == 'passed', 'HippoMM native provider/image audit did not pass')
        canonical = unique(rows(mapping.BENCHMARK)); questions = unique(rows(source / 'online/questions.jsonl'))
        require(set(questions) == set(canonical) and all(all(q[k] == canonical[i][k] for k in ('task', 'question', 'options'))
                for i, q in questions.items()), 'HippoMM canonical question/options differ')
        role_counts = Counter()
        for folder in ('offline/llm_cache', 'online/llm_cache'):
            for path in sorted((source / folder).rglob('call_*.json')):
                call = visual.hippo_call(path)
                role_counts[call['request']['role']] += 1
        require(role_counts['summary'] == read(source / 'offline/memory.json')['events'], 'HippoMM fresh summary-call coverage differs')
        for r in values:
            for name in r['llm_calls']:
                visual.hippo_call(source / 'online' / name)
        extra.update(actual_visual_role_calls_checked=sum(role_counts.values()), native_visual_roles=dict(role_counts))
        for path, info in visual.files.items():
            track(path, info['sha256'])
    return extra


def expected_storage(method, paths, metric):
    source, files = paths['source'], []
    if method in mapping.OURS:
        kind = mapping.OURS[method][2]
        stores = sorted((source / 'memory_stores').glob('*/' + kind))
        require(len(stores) == 8, 'Ours storage domain count differs')
        leaves = 0
        for store in stores:
            manifest = read(store / 'memory.json'); files.append(store / 'memory.json')
            require(set(manifest['files']) == ({'leaves'} if kind == 'natural' else {'leaves', 'graph', 'visual_sums'}), 'Ours storage schema changed')
            for item in manifest['files'].values():
                path = store / item['path']; require(sha(path) == item['sha256'], 'Ours stored payload changed'); files.append(path)
            leaf_rows = read(store / manifest['files']['leaves']['path'])
            require(len(leaf_rows) == manifest['leaf_count'], 'Ours persisted leaf count differs')
            require(all(set(r) == {'image_id', 'memory_id', 'dialogue', 'caption'} for r in leaf_rows), 'Question data entered stored leaves')
            leaves += len(leaf_rows)
        require(leaves == 438, 'Ours full memory leaf coverage differs')
    elif method in mapping.TEXT:
        p = read(source / 'protocol.json')
        for task in p['tasks']:
            work = source / task
            if method in ('gen_agents', 'a_mem'):
                files.append(work / 'final_memory.json')
            elif method == 'memoryos':
                group = sorted((work / 'runtime').rglob('*.json'))
                require(len(group) == 4 and {p.name for p in group} == {'short_term.json', 'mid_term.json', 'long_term_user.json', 'long_term_assistant.json'}, 'MemoryOS persisted layer set differs')
                files.extend(group)
            else:
                group = [p for p in (work / 'runtime/lancedb').rglob('*') if p.is_file()]
                require(bool(group), 'SimpleMem database missing'); files.extend([*group, work / 'runtime/resolved_config.json'])
    elif method == 'hippomm':
        offline = source / 'offline'
        files = [offline / n for n in mapping.HIP_FILES]
        images = rows(offline / 'representative_images.jsonl')
        files += [offline / r['file'] for r in images]
        actual = {p.resolve() for p in (offline / 'images').iterdir() if p.is_file()}
        require(actual == {p.resolve() for p in files[len(mapping.HIP_FILES):]}, 'HippoMM retained image set differs')
        require(sum(p.stat().st_size for p in files) == read(source / 'online/summary.json')['total_memory_bytes'], 'HippoMM actual accounting differs')
    else:
        export = source / 'history_only_memory'
        require({p.name for p in export.iterdir() if p.is_file()} == {'memory.json', 'image_candidates.jsonl', 'image_vectors.npy', 'dialogues.jsonl'}, 'CLIP history-only export schema changed')
        files = [p for p in export.iterdir() if p.is_file()]
        candidates = rows(export / 'image_candidates.jsonl')
        by_image = unique(candidates, 'image_id')
        require(len(by_image) == 433, 'CLIP full history image count differs')
        for r in candidates:
            require(sha(r['path']) == r['image_id'], 'CLIP historical image changed'); files.append(Path(r['path']))
    expected = {str(p.resolve()) for p in files}
    require(len(expected) == len(files), 'Storage double-counts a file')
    actual = metric['storage_components']
    require(len(actual) == len(expected) and {str(Path(c['path']).resolve()) for c in actual} == expected, 'Metric storage policy component set differs')
    for c in actual:
        require(Path(c['path']).stat().st_size == c['bytes'] and sha(c['path']) == c['sha256'], 'Storage component bytes/hash changed')
    total = sum(c['bytes'] for c in actual)
    require(total == metric['storage_bytes'] and abs(total / 1048576 - metric['storage_mib']) < 1e-14, 'Storage sum or MiB conversion differs')
    return dict(storage_files_verified=len(actual), storage_bytes=total, storage_mib=total / 1048576)


def simplemem_seal(output, source, values):
    root = Path(output) / 'metrics/simplemem_source_attribution'
    if not (root / 'annotation_lock.json').exists() or not (root / 'recall_summary.json').exists():
        return None
    p, lock, summary = (read(root / n) for n in ('protocol.json', 'annotation_lock.json', 'recall_summary.json'))
    require(p['seed'] == 42 and p.get('memory_generation_seed') == 1024 and Path(p['source_root']).resolve() == source.resolve(), 'Attribution source/evaluator seed binding differs')
    require(digest({k: v for k, v in p.items() if k != 'signature'}, False) == p['signature'], 'Attribution protocol signature changed')
    hashes(p['input_hashes']); hashes(lock['input_hashes'])
    require(sha(root / 'jobs.jsonl') == p['jobs_sha256'] and sha(root / 'entry_index.json') == p['entry_index_sha256'], 'Attribution jobs/index changed')
    require(lock['status'] == 'sealed_before_scoring' and summary['annotation_lock_sha256'] == sha(root / 'annotation_lock.json'), 'Attribution seal changed')
    require(summary['source_experiment_signature'] == read(source / 'protocol.json')['experiment_signature']
            and summary['responses_sha256'] == sha(source / 'qa_responses.jsonl'), 'Attribution scored another memory run')
    jobs = unique(rows(root / 'jobs.jsonl'))
    require(all(set(j) == {'id', 'task', 'window_index', 'sources', 'targets'} for j in jobs.values()), 'Benchmark fields entered attribution jobs')
    fresh = {e['entry_id']: dict(e, source_task=task) for task in read(source / 'protocol.json')['tasks'] for e in read(source / task / 'final_memory.json')}
    annotations = unique(rows(root / 'frozen_source_annotations.jsonl'), 'entry_id')
    require(set(annotations) <= set(fresh), 'Attribution includes nonfresh memories')
    for identity, item in annotations.items():
        require(item['memory'] == fresh[identity]['lossless_restatement'] and item['task'] == fresh[identity]['source_task'], 'Attribution memory text/task differs from fresh store')
    from scripts.analysis import simplemem_source_attribution as annotation_module
    evaluator = read(mapping.EVALUATOR / 'protocol.json')
    require(all(p[k] == evaluator[k] for k in ('reader_model', 'reader_model_path', 'seed', 'dtype', 'context_budget'))
            and p['max_new_tokens'] == 4096 and p['generation_batch_size'] == 8
            and p['max_padded_batch_tokens'] == 131072, 'Source-attribution evaluator differs from fixed Qwen42 control')
    attribution_calls = caps = 0
    for job in jobs.values():
        for target in job['targets']:
            require(target['memory'] == fresh[target['entry_id']]['lossless_restatement'], 'Annotation job target is not a fresh stored statement')
        for index in (1, 2):
            result = read(root / f'pass{index}' / (job['id'] + '.json'))
            call = read(root / 'calls' / (result['call_key'] + '.json'))
            request = call['request']
            require(digest(request, False) == result['call_key'] and request['signature'] == p['signature']
                    and request['max_new_tokens'] == 4096 and request['do_sample'] is False
                    and request['pass'] == index and request['messages'] == annotation_module.messages(job, index)
                    and call['output'] == result['output'], 'Actual attribution model request/output differs from sealed fresh job')
            attribution_calls += 1; caps += bool(call['output']['hit_token_limit'])
    manual = unique(rows(root / 'manual_review.jsonl'), 'entry_id') if (root / 'manual_review.jsonl').exists() else {}
    queue = unique(rows(root / 'review_queue.jsonl'), 'entry_id')
    require(set(queue) <= set(manual), 'Flagged attribution entries lack explicit review')
    from scripts.analysis.simplemem_attribution_recall import validate_manual
    raw_annotations = unique(rows(root / 'annotations.jsonl'), 'entry_id')
    for identity, item in manual.items():
        record = raw_annotations[identity]; job = jobs[record['job_id']]
        target = next(t for t in job['targets'] if t['entry_id'] == identity)
        validate_manual(item, target, job)
    hits = denominator = empty = 0
    canonical = unique(rows(mapping.BENCHMARK))
    selected = set()
    for r in values:
        ids = r['retrieval_detail']['context_memory_ids']
        require(len(ids) == len(set(ids)) and (len(ids) == 5 or (not ids and r['retrieval_detail'].get('source_empty_preserved'))), 'SimpleMem native final context count differs')
        selected.update(ids); empty += not ids
        edges = set().union(*(set(annotations[i]['source_round_ids']) for i in ids))
        gold = set(canonical[r['id']]['gold_ids']); hits += len(edges & gold); denominator += len(gold)
    require(selected == {i for i, a in annotations.items() if a['selected']}, 'Attribution selected memory set differs')
    require(len(annotations) == lock['entries'] == p['annotation_entries'] and len(manual) == lock['manual_review_entries'], 'Attribution entry/manual counts differ')
    require(hits == summary['gold_round_hits'] and denominator == summary['gold_round_references'] == 1041, 'Sealed attribution source recall differs')
    return dict(recall_numerator=hits, recall_denominator=denominator, recall_at5_micro=hits / denominator,
        annotation_entries=len(annotations), selected_entries=len(selected), manual_review_entries=len(manual),
        native_empty_contexts=empty, attribution_calls_checked=attribution_calls, attribution_output_caps=caps,
        annotation_lock_sha256=sha(root / 'annotation_lock.json'),
        attribution_protocol_sha256=sha(root / 'protocol.json'))


def recall(method, paths, values, canonical, output):
    source = paths['source']; detail = {}
    if method in ('gen_agents', 'memoryos'):
        return dict(recall_at5_micro=None, recall_status='not_reported_native_top10_or_dynamic')
    if method == 'simplemem':
        from scripts.evaluation import memeye_seed1024_text_metrics as context_checker
        context_audit = context_checker.fusion_audit(source, save=False)
        context_audit['source_root'] = str(source.resolve())
        context_audit['source_memory_sha256'] = {str(source / t / 'final_memory.json'): sha(source / t / 'final_memory.json')
            for t in read(source / 'protocol.json')['tasks']}
        context_path = Path(output) / 'audit/simplemem_final_context_audit.json'
        dump(context_path, context_audit); sha(context_checker.__file__); sha(context_path)
        evidence = simplemem_seal(output, source, values)
        return dict(evidence or {}, context_entry_counts=dict(Counter(len(r['retrieval_detail']['context_memory_ids']) for r in values)),
                    final_context_audit_sha256=sha(context_path), attribution_pending=evidence is None)
    if method in mapping.OURS:
        requests = unique(rows(source / 'reader/requests.jsonl'), 'record_id')
        ids = {r['id']: requests[r['record_id']]['top_memory_ids'] for r in values}
    elif method == 'a_mem':
        notes = {t: read(source / t / 'final_memory.json') for t in read(source / 'protocol.json')['tasks']}
        ids = {}
        for r in values:
            mids = r['retrieval_detail']['context_memory_ids']
            require(len(mids) == len(set(mids)) and len(mids) <= 5, 'A-Mem native final count exceeds five')
            selected = [notes[r['task']][i] for i in mids]
            text = ''.join('talk start time:' + n['timestamp'] + 'memory content: ' + n['content'] + 'memory context: ' + n['context']
                           + 'memory keywords: ' + str(n['keywords']) + 'memory tags: ' + str(n['tags']) + '\n' for n in selected)
            require(text == r['retrieved_context'], 'A-Mem selected source notes differ from final reader context')
            ids[r['id']] = [r['task'] + '/' + i for i in mids]
        detail['context_entry_counts'] = dict(Counter(len(v) for v in ids.values()))
    elif method == 'clip_image_dialogue':
        manifest = unique(rows(source / 'online/qa_manifest.jsonl'))
        ids = {r['id']: manifest[r['id']]['top_memory_ids'] for r in values}
    else:
        ids = {r['id']: r['context_round_ids'] for r in values}
    hits = sum(len(set(ids[i]) & set(q['gold_ids'])) for i, q in canonical.items())
    denominator = sum(len(set(q['gold_ids'])) for q in canonical.values())
    require(denominator == 1041, 'Gold source-round denominator differs')
    if method in mapping.OURS or method == 'clip_image_dialogue':
        # Each retrieved image may carry multiple source dialogue IDs for CLIP.
        if method in mapping.OURS:
            require(all(len(v) == len(set(v)) == 5 for v in ids.values()), 'Ours final Top5 differs')
    detail.update(recall_numerator=hits, recall_denominator=denominator)
    if method == 'hippomm':
        detail.update(recall_at5_micro=None, context_source_micro_recall=hits / denominator, recall_is_literal_top5=False)
    else:
        detail['recall_at5_micro'] = hits / denominator
    return detail


def audit_one(method, output=OUT):
    paths = method_paths(method, output)
    if not paths['marker'].exists() or not paths['summary'].exists():
        count = len(read_mcq(method, output)) if paths['mcq'].exists() else 0
        return dict(method=method, status='pending_mcq', observed_records=count)
    summary = read(paths['summary'])
    if method in mapping.OURS:
        ready = summary.get('status') == 'complete'
    elif method == 'hippomm':
        ready = read(paths['marker']).get('status') == 'complete'
    else:
        ready = summary.get('status') == 'complete'
    if not ready:
        return dict(method=method, status='pending_mcq', observed_records=summary.get('questions', 0))
    values, canonical = read_mcq(method, output), unique(rows(mapping.BENCHMARK))
    found = unique(values)
    require(len(found) == 371 and set(found) == set(canonical), 'MCQ coverage is not all 371 unique canonical IDs')
    correct = 0
    for identity, r in found.items():
        q = canonical[identity]
        for key in ('task', 'question', 'options', 'gold_choice', 'gold_ids'):
            if key in r:
                require(r[key] == q[key], 'Canonical MCQ field changed: ' + identity + '/' + key)
        pred = prediction(method, r)
        require(method == 'hippomm' or pred == r['predicted_choice'], 'Saved MCQ grade differs from raw native parse')
        grade = pred == q['gold_choice']; correct += grade
        require('correct' not in r or r['correct'] == grade, 'Saved correct flag differs from raw regrade')
    measured = summary['overall'][paths['condition']] if method in mapping.OURS else summary
    require(correct == measured['correct'] and abs(correct / 371 - measured['accuracy']) < 1e-14, 'MCQ summary does not reproduce')
    result = dict(method=method, status='pending_metrics', mcq_complete=True, questions=371, correct=correct, accuracy=correct / 371,
        invalid_choice_records=sum(prediction(method, r) not in ('A', 'B', 'C', 'D') for r in values),
        format_errors=sum(prediction(method, r) not in ('A', 'B', 'C', 'D') and
                          r.get('status') not in ('provider_input_rejected', 'provider_rejection')
                          and not r.get('hit_token_limit') and not r.get('refusal') for r in values),
        provider_input_rejections=sum(r.get('status') in ('provider_input_rejected', 'provider_rejection') for r in values),
        refused_answers=sum(bool(r.get('refusal')) for r in values),
        answer_token_limit_hits=sum(bool(r.get('hit_token_limit')) for r in values),
        native_outcomes=dict(Counter(r.get('status', 'native_hippomm_answer' if method == 'hippomm' else 'returned_output') for r in values)))
    result.update(source_provenance(method, paths, values))
    old = unique(read_mcq(method, reference=True)); require(set(old) == set(found), 'Seed42 MCQ paired cohort differs')
    require(sum(prediction(method, old[i]) == canonical[i]['gold_choice'] for i in old) ==
            mapping.reference_mapping()[method]['correct'], 'Seed42 reference raw regrade differs from audited count')
    result['paired_outcomes'] = dict(Counter(('correct' if prediction(method, old[i]) == canonical[i]['gold_choice'] else 'wrong') + '42_to_' +
        ('correct' if prediction(method, found[i]) == canonical[i]['gold_choice'] else 'wrong') + '1024' for i in found))
    normalized = lambda value: value if value in ('A', 'B', 'C', 'D') else None
    result['changed_predicted_answers_vs_seed42'] = sum(normalized(prediction(method, old[i])) != normalized(prediction(method, found[i])) for i in found)
    result['changed_native_predictions_vs_seed42'] = sum(prediction(method, old[i]) != prediction(method, found[i]) for i in found)
    result['citations_sha256'] = {str(p): sha(p) for p in (paths['mcq'], paths['summary'])}
    if not paths['metric'].exists():
        return result
    metric = read(paths['metric'])
    if metric.get('storage_bytes') is None or metric.get('status') not in ('complete', 'accuracy_storage_complete_attribution_pending'):
        return result
    require(metric['seed'] == 1024 and metric['model'] == MODEL and metric['correct'] == correct, 'Metric source seed/model/grade differs')
    result.update(expected_storage(method, paths, metric))
    evidence = recall(method, paths, values, canonical, output)
    if evidence is None or evidence.get('attribution_pending'):
        if evidence:
            result.update(evidence)
        result['status'] = 'pending_fresh_simplemem_attribution_seal'
    else:
        for field in ('recall_at5_micro', 'context_source_micro_recall'):
            if evidence.get(field) is not None:
                require(metric.get(field) is not None and abs(evidence[field] - metric[field]) < 1e-14, 'Metric source recall differs')
        result.update(evidence, status='passed', passed=True)
    result['citations_sha256'][str(paths['metric'])] = sha(paths['metric'])
    return result


def audit(output=OUT, methods=METHODS, report_path=None):
    TRACKED.clear()
    output = Path(output); started = datetime.now(timezone.utc).isoformat()
    results, failures = [], []
    plan_path = output / 'plan.json'
    if plan_path.exists():
        hashes(read(plan_path)['source_sha256'])
    index_path = output / 'metrics/metric_summary.json'
    index = read(index_path) if index_path.exists() else None
    if index is not None:
        indexed = [r['method'] for r in index['methods']]
        if len(indexed) != len(set(indexed)) or set(indexed) != set(METHODS):
            failures.append(dict(method='ALL', status='failed', error='Metric index does not contain all10 unique methods'))
    for method in methods:
        try:
            result = audit_one(method, output)
        except Exception as exc:
            result = dict(method=method, status='failed', error_type=type(exc).__name__, error=str(exc))
            failures.append(result)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('citations_sha256', 'native_call_settings')}), flush=True)
    index_ready = index is not None and index.get('status') == 'complete' and index.get('mcq_storage_status') == 'complete'
    passed = not failures and index_ready and len(results) == 10 and all(r['status'] == 'passed' for r in results)
    report = dict(status='passed' if passed else 'failed' if failures else 'partial', passed=passed,
        audit_passed=passed, seed=1024, reference_seed=42, judge_seed=42,
        started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        methods_complete=sum(r.get('mcq_complete', False) for r in results),
        metrics_complete=sum(r['status'] == 'passed' for r in results),
        mcq_records_total=sum(r.get('questions', 0) for r in results), methods=results, failures=failures,
        scope='Independent fixed-cohort raw MCQ regrading, actual text wires, complete stored components, source-round recall and fresh annotation seal. Visual original-image wire audit and Open/Judge audit remain separate.',
        scientific_limits=['Two seeds do not establish robustness or statistical significance.',
            'Fixed seed42 captions exclude caption-generation variation.',
            'Provider alias does not establish fixed weights across dates; UUIDs, timestamps, scheduling and numeric kernels are additional variation sources.',
            'Native construction caps/invalid structured outputs remain in the result; no question filtering or answer imputation.',
            'Storage counts native persisted file serialization and retained versions; GA/A-Mem snapshots omit unpersisted retrieval indices.',
            'SimpleMem recall is conservative model-assisted reconstructed provenance with explicit quote review, not native source IDs or human gold.',
            'HippoMM context source coverage includes fast summaries and is not literal Recall@5.'],
        implementation_sha256=sha(__file__), source_plan_sha256=sha(plan_path) if plan_path.exists() else None)
    report['metric_index_unique_methods'] = len({r['method'] for r in index['methods']}) if index else 0
    report['metric_index_complete'] = index_ready
    report['metric_index_sha256'] = sha(index_path) if index else None
    changed = [path for path, info in TRACKED.items() if Path(path).stat().st_size != info['bytes'] or Path(path).stat().st_mtime_ns != info['mtime_ns']]
    if changed:
        report.update(status='failed', passed=False, audit_passed=False)
        report['failures'].append(dict(method='ALL', error='Audited artifacts changed before audit completion', paths=changed))
    report['artifact_sha256'] = {path: info['sha256'] for path, info in TRACKED.items()}
    destination = report_path or output / 'audit' / ('final_audit_mcq_metrics.json' if len(methods) == 10 else 'mcq_metrics_' + '_'.join(methods) + '.json')
    dump(destination, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report = audit(args.root, METHODS if args.method == 'all' else (args.method,), args.report)
    if report['status'] == 'failed' or (args.require_complete and not report['passed']):
        raise SystemExit(2)
