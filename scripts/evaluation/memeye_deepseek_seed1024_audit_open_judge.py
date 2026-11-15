"""Independent, local-only audit of fresh DeepSeek seed1024 Open/Judge evidence.

This module never calls generation or HTTP functions. It reconstructs each wire
from its native messages, compares durable attempts and raw responses, and
recomputes all 371-question Judge aggregates. Incomplete work remains pending.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation import memeye_deepseek_open_judge as judge
from scripts.memory import hippomm_core, hippomm_original

OUT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106'
CANONICAL = ROOT / 'results/memeye_qwen_semrag_open_20261105/inputs'
MODEL = 'deepseek-v4.1-flash'
GRADES = (0, .25, .5, .75, 1)
FORBIDDEN = {'options', 'gold_choice', 'gold_ids', 'ground_truth', 'answer', 'clue'}


class Pending(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def native_digest(value):
    """Native text/reflexion protocols retain default JSON separators."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Audit:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.files, self.checks = {}, Counter()
        self.native_calls = {}
        self.native_attempts = {}
        self.image_parts = {}

    def require(self, value, description):
        self.checks[description] += 1
        if not value:
            raise ValueError(description)

    def sha(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            with path.open('rb') as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            stat = path.stat()
            self.files[key] = dict(sha256=value, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return self.files[key]['sha256']

    def hashes(self, hashes):
        for path, expected in hashes.items():
            self.require(self.sha(path) == expected, 'pinned_source_hash')

    def unique(self, values, key='id'):
        result = {v[key]: v for v in values}
        self.require(len(result) == len(values), 'unique_' + key)
        return result

    def original_image(self, path, expected=None):
        path = str(Path(path).resolve())
        actual = self.sha(path)
        if expected is not None:
            self.require(actual == expected, 'original_image_sha256')
        if path not in self.image_parts:
            data = Path(path).read_bytes()
            mime = ('image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else
                    'image/jpeg' if data.startswith(b'\xff\xd8\xff') else None)
            self.require(mime is not None, 'original_image_png_jpeg')
            self.image_parts[path] = dict(type='image_url', image_url=dict(detail='high',
                url='data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')))
        return copy.deepcopy(self.image_parts[path])

    def wire(self, native, seed=1024, cap=2048, temperature=0.0):
        messages = copy.deepcopy(native)
        for message in messages:
            self.require(set(message) == {'role', 'content'} and
                         message['role'] in {'system', 'user', 'assistant'}, 'native_message_roles')
            if isinstance(message['content'], str):
                continue
            self.require(message['role'] == 'user' and isinstance(message['content'], list),
                         'native_typed_user_content')
            content = []
            for part in message['content']:
                if part['type'] == 'image':
                    self.require(set(part) == {'type', 'path', 'sha256'}, 'native_image_metadata')
                    content.append(self.original_image(part['path'], part['sha256']))
                else:
                    self.require(set(part) == {'type', 'text'} and part['type'] == 'text' and
                                 isinstance(part['text'], str), 'native_text_part')
                    content.append(part)
            message['content'] = content
        return dict(model=MODEL, messages=messages, max_tokens=cap, temperature=temperature,
                    seed=seed, enable_thinking=False, n=1, stream=False)

    def native_messages(self, row):
        if 'reader_messages' in row:
            return row['reader_messages']
        return [dict(role='system', content=row['system']), dict(role='user', content=[
            *[dict(type='image', path=p, sha256=self.sha(p)) for p in row['query_image_paths']],
            dict(type='text', text=row['user_text'])])]

    def response(self, response, key, seed=1024):
        self.require(response.get('request_sha256') == key and response.get('seed') == seed,
                     'response_actual_wire_seed_hash')
        self.require(response.get('requested_model') == response.get('returned_model') == MODEL and
                     response.get('enable_thinking') is False and response.get('response_id') and
                     isinstance(response.get('raw_response'), str) and
                     response.get('model_identity_kind') == 'provider_alias' and
                     response.get('resolved_model_revision') is None,
                     'response_provider_identity')
        self.require(not response.get('reasoning_content_present') and
                     response.get('reasoning_tokens') in (None, 0), 'disabled_provider_reasoning')
        if 'hit_token_limit' in response:
            self.require(response['hit_token_limit'] == (response['finish_reason'] == 'length'),
                         'honest_output_cap_flag')

    def canonical_question(self, row, canonical):
        self.require(all(row[k] == canonical[row['id']][k] for k in ('task', 'point', 'question')),
                     'canonical_open_question')
        self.require(not FORBIDDEN.intersection(row), 'gold_options_absent_from_reader')

    def native_text_retrieval(self, root, method, protocol, manifest, canonical):
        from scripts.memory import text_memory_baselines as native
        from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM
        source = self.output / 'text_methods' / method
        native_path = str(Path(native.__file__).resolve())
        self.require(protocol['input_sha256'][native_path] == self.sha(native_path),
                     'native_text_query_implementation_hash')
        questions = self.unique(rows(root / 'questions.jsonl'))
        retrieval = self.unique(rows(root / 'retrieval.jsonl'))
        self.require(len(questions) == len(retrieval) == len(manifest) == 371 and
                     set(questions) == set(retrieval) == set(manifest) == set(canonical) and
                     native_digest(list(questions.values())) == protocol['open_evaluation']['questions_sha256'],
                     'native_open_prepared_retrieval_manifest_ids')
        cap_path = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025/captions.jsonl'
        self.require(protocol['caption_sha256'] == self.sha(cap_path), 'native_open_frozen_deepseek_caption_hash')
        caps = self.unique(rows(cap_path))
        task_records, source_entries = [], {}
        for task in protocol['tasks']:
            task_root = root / task
            task_rows = rows(task_root / 'retrieval.jsonl')
            seal = read(task_root / 'complete.json')
            expected_ids = {q['id'] for q in questions.values() if q['task'] == task}
            actual = self.unique(task_rows)
            self.require(seal['status'] == 'complete' and seal['questions'] == len(actual) and
                         set(actual) == expected_ids and
                         seal['retrieval_sha256'] == self.sha(task_root / 'retrieval.jsonl'),
                         'native_open_task_retrieval_completion_hash')
            if method in {'gen_agents', 'a_mem', 'simplemem'}:
                source_entries[task] = read(source / task / 'final_memory.json')
            if method in {'gen_agents', 'a_mem'}:
                actual_memory = read(task_root / 'restored_memory.json')
                source_memory = source_entries[task]
                if method == 'a_mem':
                    strip_access = lambda values: {k: {f: v for f, v in row.items() if f != 'last_accessed'}
                                                   for k, row in values.items()}
                    actual_memory, source_memory = strip_access(actual_memory), strip_access(source_memory)
                self.require(actual_memory == source_memory, 'native_open_restored_new_memory_identity')
                self.sha(task_root / 'restored_memory.json')
            if method in {'memoryos', 'simplemem'}:
                copied = read(task_root / 'runtime_copy_audit.json')
                runtime = source / task / 'runtime'
                expected = {str(p.relative_to(runtime)): self.sha(p) for p in runtime.rglob('*') if p.is_file()}
                self.require(copied['passed'] is True and copied['files'] == expected,
                             'native_open_original_new_runtime_copy')
                self.sha(task_root / 'runtime_copy_audit.json')
            task_records.extend(task_rows); self.sha(task_root / 'complete.json')
        self.require(self.unique(task_records) == retrieval, 'native_task_to_aggregate_retrieval_identity')
        for qid, q in questions.items():
            self.canonical_question(q, canonical)
            self.require(q['query_image_paths'] == canonical[qid]['query_image_paths'],
                         'native_open_canonical_query_image_identity')
            expected_caps = []
            for path in q['query_image_paths']:
                cap = caps[self.sha(path)]
                self.require(cap['seed'] == 42 and cap['model'] == MODEL, 'native_query_caption_original_deepseek42')
                expected_caps.append(cap['caption'])
            self.require(q['query_image_captions'] == expected_caps, 'native_open_query_uses_original_deepseek_captions')
            row, reader = retrieval[qid], manifest[qid]
            query = native.retrieval_query(q)
            # The prepared input retains its old derived retrieval_query for
            # provenance. Recompute the actual query from the true Open text
            # and substituted DeepSeek captions; every other field is exact.
            self.require(all(row.get(k) == v for k, v in q.items() if k != 'retrieval_query') and
                         row['retrieval_query'] == query and row['method'] == method,
                         'native_actual_open_query_deepseek42_and_other_fields_exact')
            self.require(not FORBIDDEN.intersection(row) and row['context'] == row['retrieved_context'] and
                         native_digest(row['context']) == row['context_sha256'],
                         'native_gold_free_context_and_hash')
            self.require(all(reader.get(k) == v for k, v in row.items() if k != 'query_image_paths') and
                         reader['query_image_paths'] == [] and
                         reader['source_query_image_paths'] == row['query_image_paths'],
                         'native_retrieval_to_reader_manifest_exact_join')
            expected_messages = [dict(role='system', content=SYSTEM),
                dict(role='assistant', content='Retrieved memory context:\n' + row['context']),
                dict(role='user', content=query)]
            self.require(reader['system'] == SYSTEM and reader['user_text'] == query and
                         reader['reader_messages'] == expected_messages and
                         reader['reader_messages_sha256'] == reader['payload_sha256'] == native_digest(expected_messages),
                         'native_query_context_and_payload_message_identity')
            detail = row['retrieval_detail']
            if method in {'a_mem', 'simplemem'}:
                ids = detail['context_memory_ids']
                entries = source_entries[q['task']]
                allowed = set(entries) if method == 'a_mem' else {e['entry_id'] for e in entries}
                self.require(len(ids) == len(set(ids)) <= 5 and set(ids) <= allowed,
                             'native_context_ids_from_same_new_memory_task')
                if method == 'simplemem':
                    self.require(len(ids) == 5 or (not ids and detail.get('source_empty_preserved')),
                                 'native_simplemem_five_or_preserved_empty')
            elif method == 'gen_agents':
                ids = detail['context_memory_indices']
                count = len(source_entries[q['task']])
                self.require(detail['context_memory_count'] == len(ids) and
                             all(type(i) is int and 0 <= i < count for i in ids),
                             'native_agents_context_indices_from_same_new_task')
        seal = read(root / 'complete.json')
        self.require(seal['status'] == 'complete' and seal['questions'] == 371 and seal['seed'] == 1024 and
                     seal['retrieval_sha256'] == self.sha(root / 'retrieval.jsonl') and
                     seal['reader_manifest_sha256'] == self.sha(root / 'reader_manifest.jsonl') and
                     seal['protocol_sha256'] == self.sha(root / 'protocol.json'),
                     'native_global_open_retrieval_seal')
        self.sha(root / 'questions.jsonl'); self.sha(root / 'complete.json')

    def generic_open(self, root, method, canonical):
        seal_path = root / 'reader/completion.json'
        if not seal_path.exists() or read(seal_path).get('status') != 'complete':
            raise Pending('371 Open answers have not been sealed')
        protocol_path = root / 'reader/protocol.json'
        protocol = read(protocol_path); contract = protocol['contract']
        source_protocol = read(root / 'protocol.json')
        if method in {'semrag', 'ours_1', 'ours_2', 'ours_1_ours_2'}:
            source = source_protocol['contract']
            self.require(digest(source) == source_protocol['experiment_signature'] and
                         source['seed'] == 1024 and source['method'] == method and
                         source['model'] == MODEL and source['new_memory_built'] is True and
                         source['fresh_candidates'] is True and source['historical_answers_reused'] is False,
                         'ours_open_fresh_memory_and_candidates')
            self.hashes(source['source_sha256'])
        elif method in {'gen_agents', 'memoryos', 'a_mem', 'simplemem'}:
            source = source_protocol
            self.require(native_digest({k: v for k, v in source.items() if k != 'experiment_signature'}) ==
                         source['experiment_signature'] and source['seed'] == 1024 and source['model'] == MODEL and
                         source['open_evaluation']['source_seed'] == 1024 and
                         Path(source['open_evaluation']['source_root']).resolve() == self.output / 'text_methods' / method,
                         'native_open_restores_new_seed1024_memory')
            self.hashes(source['open_evaluation']['source_hashes'])
        else:
            source = source_protocol
            self.require(source['seed'] == 1024 and source['answer_model'] == MODEL and
                         source['fresh_seed1024_memory'] is True, 'clip_open_uses_new_native_index')
            self.hashes(source['source_sha256'])
            self.hashes({str(root / name): expected for name, expected in source['artifact_sha256'].items()})
        self.require(digest(contract) == protocol['experiment_signature'], 'reader_protocol_signature')
        self.require(contract['method_id'] == method and contract['answer_model'] == MODEL and
                     contract['settings'] == dict(model=MODEL, max_tokens=2048, temperature=0,
                         seed=1024, enable_thinking=False, n=1, stream=False),
                     'reader_seed1024_native_2048_settings')
        self.hashes(contract['source_sha256']); self.hashes(contract['image_sha256'])
        self.require(contract['gold_sha256'] == self.sha(CANONICAL / 'gold.jsonl'), 'canonical_gold_hash')
        manifest = self.unique(rows(root / 'reader_manifest.jsonl'))
        if method in {'gen_agents', 'memoryos', 'a_mem', 'simplemem'}:
            self.native_text_retrieval(root, method, source_protocol, manifest, canonical)
        answers = self.unique(rows(root / 'reader/responses.jsonl'))
        self.require(len(answers) == len(manifest) == 371 and set(answers) == set(manifest) == set(canonical),
                     'open_371_unique_id_coverage')
        seal = read(seal_path)
        self.require(seal['responses_sha256'] == self.sha(root / 'reader/responses.jsonl') and
                     seal['protocol_sha256'] == self.sha(protocol_path) and
                     seal['questions_answered'] == 371, 'sealed_open_protocol_and_responses')
        cache = Path(contract['cache_root']).resolve()
        self.require(cache.is_relative_to(self.output), 'answer_cache_belongs_to_new_run')
        cache_protocol = read(cache / 'protocol.json')
        self.require(digest(cache_protocol['contract']) == cache_protocol['experiment_signature'] and
                     cache_protocol['experiment_signature'] == contract['cache_signature'],
                     'shared_answer_cache_signature')
        self.hashes(cache_protocol['contract']['source_sha256'])
        self.require(cache_protocol['contract']['settings'] == contract['settings'], 'shared_cache_seed1024')
        for qid, q in manifest.items():
            self.canonical_question(q, canonical)
            native = self.native_messages(q)
            # The signed Open reader uses integer 0 in its typed-content path;
            # the native text transport emits floating 0.0. Wire hashes retain
            # this original serialization distinction.
            temperature = 0.0 if all(isinstance(m['content'], str) for m in native) else 0
            wire = self.wire(native, temperature=temperature); key = digest(wire)
            directory = cache / 'calls' / key
            wire_path, output_path = directory / 'wire.json.gz', directory / 'response.json'
            actual = json.loads(gzip.decompress(wire_path.read_bytes())); output = read(output_path)
            self.require(actual == wire and digest(actual) == key, 'open_actual_wire_reproduces')
            self.response(output, key)
            attempts = rows(directory / 'attempts.jsonl')
            self.require([a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1)) and
                         1 <= len(attempts) <= 4, 'open_bounded_attempt_sequence')
            successful = [a for a in attempts if a['status'] == 'ok']
            self.require(len(successful) == 1 and successful[0]['response'] == output,
                         'open_raw_response_successful_attempt')
            saved = answers[qid]
            self.require(saved['id'] == qid and saved['method_id'] == method and
                         saved['run_id'] == protocol['run_id'] and
                         saved['experiment_signature'] == protocol['experiment_signature'] and
                         all(saved.get(k) == v for k, v in output.items()), 'open_raw_answer_and_run_identity')
            for path in (wire_path, output_path, directory / 'attempts.jsonl'):
                self.sha(path)
        for path in (root / 'protocol.json', root / 'reader_manifest.jsonl', seal_path, cache / 'protocol.json'):
            self.sha(path)
        return answers, root / 'reader/responses.jsonl', dict(wire_requests_verified=371, cap=2048)

    def hippo_call(self, path):
        path = Path(path).resolve()
        if str(path) in self.native_calls:
            return self.native_calls[str(path)]
        call = read(path); request, response = call['request'], call['response']
        self.require(request['backbone'] == 'deepseek' and request['max_output_tokens'] == 2048,
                     'hippomm_native_2048_cap')
        self.hashes(call['image_sha256'])
        messages = [dict(role='system', content=common.READER_SYSTEM), dict(role='user', content=[
            *[dict(type='image', path=p, sha256=self.sha(p)) for p in request['image_paths']],
            dict(type='text', text=request['prompt'])])]
        wire = self.wire(messages); key = digest(wire)
        wire_path = path.with_suffix('.wire.json.gz')
        actual = json.loads(gzip.decompress(wire_path.read_bytes()))
        self.require(actual == wire and digest(actual) == key == call['request_sha256'] and
                     call['wire_file_sha256'] == self.sha(wire_path), 'hippomm_exact_seed1024_wire')
        self.response(response, key)
        attempts_path = path.parent / 'api_attempts.jsonl'
        if str(attempts_path) not in self.native_attempts:
            grouped = defaultdict(list)
            for attempt in rows(attempts_path):
                grouped[attempt['call_id']].append(attempt)
            self.native_attempts[str(attempts_path)] = grouped
        history = self.native_attempts[str(attempts_path)][call['call_id']]
        self.require([a['attempt'] for a in history] == list(range(1, len(history) + 1)) and
                     1 <= len(history) <= 4 and all(a['request_sha256'] == key for a in history),
                     'hippomm_bounded_native_attempts')
        successes = [a for a in history if a['status'] == 'success']
        self.require(len(successes) == 1 and successes[0]['response'] == response,
                     'hippomm_raw_successful_attempt')
        self.require(response['finish_reason'] == 'stop' and not response.get('refusal') and
                     bool(response['raw_response'].strip()), 'hippomm_normal_native_finish')
        self.sha(path); self.sha(attempts_path)
        self.native_calls[str(path)] = call
        return call

    def hippomm_open(self, root, canonical):
        seal_path = root / 'completion.json'
        if not seal_path.exists() or read(seal_path).get('status') != 'complete':
            raise Pending('371 native HippoMM Open answers have not been sealed')
        protocol = read(root / 'protocol.json')
        self.require(protocol['seed'] == 1024 and protocol['evaluator_seed'] == 42 and
                     protocol['answer_model'] == MODEL and protocol['max_completion_tokens'] == 2048,
                     'hippomm_memory_answer_judge_seeds')
        self.hashes(protocol['source_sha256'])
        self.require(protocol['questions_sha256'] == self.sha(root / 'questions.jsonl'), 'hippomm_question_hash')
        questions = self.unique(rows(root / 'questions.jsonl'))
        answers = self.unique(rows(root / 'responses.jsonl'))
        self.require(len(answers) == len(questions) == 371 and set(answers) == set(questions) == set(canonical),
                     'open_371_unique_id_coverage')
        seal = read(seal_path)
        self.require(seal['answered'] == 371 and seal['responses_sha256'] == self.sha(root / 'responses.jsonl'),
                     'hippomm_sealed_original_responses')
        memory = Path(protocol['memory_dir']).resolve()
        self.require(memory == self.output / 'visual_methods/hippomm/offline', 'hippomm_fresh_memory_root')
        metadata = read(memory / 'memory.json')
        self.require(metadata['seed'] == 1024 and metadata['model'] == MODEL, 'hippomm_fresh_memory_seed1024')
        events = self.unique(rows(memory / 'events.jsonl'), 'event_id')
        dialogues = self.unique(rows(memory / 'dialogues.jsonl'))
        images = self.unique(rows(memory / 'representative_images.jsonl'), 'event_id')
        self.require(len(events) == len(images) == 250, 'hippomm_full_native_memory')
        source_memories = self.unique(rows(ROOT / 'results/natural_observation/memories.jsonl'))
        frozen_captions = self.unique(rows(ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025/captions.jsonl'))
        for event in events.values():
            image = images[event['event_id']]
            path = memory / image['file']; actual = self.sha(path)
            representative_sources = [r for r in event['source_images'] if r['image_ref'] == event['representative_image']]
            self.require(image['image_ref'] == event['representative_image'] and
                         image['task'] == event['task'] and
                         any(actual in source_memories[r['round_id']]['images'] for r in representative_sources),
                         'hippomm_retains_exact_source_original_image')
            cap = frozen_captions[actual]
            self.require(cap['seed'] == 42 and cap['model'] == MODEL and cap['caption'] == event['caption'],
                         'hippomm_shares_exact_original_caption')
            summary_calls = [self.hippo_call(p) for p in (memory / 'llm_cache' / event['event_id']).glob('call_*.json')]
            matched = [c for c in summary_calls if c['request']['role'] == 'summary']
            self.require(len(matched) == 1 and matched[0]['request']['prompt'] ==
                         hippomm_core.summary_prompt(event, dialogues) and
                         matched[0]['response']['raw_response'].strip() == event['summary'],
                         'hippomm_fresh_summary_raw_provenance')
        count_before = len(self.native_calls)
        for qid, q in questions.items():
            self.canonical_question(q, canonical)
            saved = answers[qid]
            self.require(saved['seed'] == 1024 and saved['question'] == q['question'] and
                         saved['task'] == q['task'] and saved['point'] == q['point'], 'hippomm_saved_question_identity')
            calls = []
            for relative in saved['llm_calls']:
                path = (root / relative).resolve()
                self.require(path.is_relative_to(root / 'llm_cache'), 'hippomm_run_local_cache_path')
                calls.append(self.hippo_call(path))
            self.require(bool(calls), 'hippomm_original_calls_exist')
            local = [e for e in events.values() if e['task'] == q['task']]
            first, last = calls[0], calls[-1]
            self.require(first['request']['role'] == 'fast' and first['request']['prompt'] ==
                         hippomm_core.fast_prompt(q, local), 'hippomm_fast_uses_only_same_task_summaries')
            self.require(first['request']['image_paths'] == q['query_image_paths'], 'hippomm_original_query_images')
            self.require(all(events[e]['task'] == q['task'] for e in saved['selected_event_ids']),
                         'hippomm_no_cross_task_event_mix')
            for call in calls[1:-1]:
                request = call['request']; role = request['role']; prompt = request['prompt']
                self.require(not request['image_paths'] and role in
                             {'rewrite', 'rewrite_format_recovery', 'caption_fallback'},
                             'hippomm_native_retrieval_roles_without_images')
                if role == 'rewrite':
                    self.require(prompt.startswith(hippomm_core.query_prompt(q)),
                                 'hippomm_rewrite_uses_true_open_question')
                elif role == 'rewrite_format_recovery':
                    original = prompt.split('Original request:\n', 1)
                    self.require(len(original) == 2 and original[1].startswith(hippomm_core.query_prompt(q)),
                                 'hippomm_explicit_original_query_format_repair')
                else:
                    payload, _ = json.JSONDecoder().raw_decode(prompt.split('\n', 1)[1])
                    event = events[payload['event_id']]
                    self.require(event['task'] == q['task'] and
                                 prompt.startswith(hippomm_core.fallback_prompt(q, event)),
                                 'hippomm_caption_fallback_same_task_true_question')
            if saved['route'] == 'fast':
                parsed = hippomm_core.parse_fast(last['response']['raw_response'], [e['event_id'] for e in local])
                self.require(last['request']['role'] == 'fast' and parsed == saved['fast'] and
                             parsed['answer'].strip() == saved['raw_response'], 'hippomm_fast_raw_answer_identity')
            else:
                self.require(last['request']['role'] in {'answer', 'answer_with_representatives'} and
                             last['response']['raw_response'].strip() == saved['raw_response'],
                             'hippomm_detailed_raw_answer_identity')
                if saved['selected_event_ids']:
                    prompt, paths, order = hippomm_original.reader_payload(q,
                        {'route': 'detailed', 'selected_event_ids': saved['selected_event_ids']},
                        events, images, dialogues, memory)
                    self.require(last['request']['prompt'] == prompt and
                                 last['request']['image_paths'] == [str(p) for p in paths] and
                                 saved['reader_image_order'] == order, 'hippomm_original_representative_order')
            self.require(saved['request_sha256'] == saved['payload_sha256'] == last['request_sha256'] and
                         saved['hit_token_limit'] == (last['response']['finish_reason'] == 'length'),
                         'hippomm_final_wire_response_join')
        self.sha(root / 'protocol.json'); self.sha(seal_path)
        return answers, root / 'responses.jsonl', dict(wire_requests_verified=len(self.native_calls) - count_before,
            fresh_summary_requests_verified=250, cap=2048)

    def judge_audit(self, root, source, answers, response_path, canonical, gold):
        public_path = root / 'llm_judge_summary.json'
        if not public_path.exists() or read(public_path).get('status') != 'complete':
            raise Pending('371 fixed-seed42 Judge grades have not been sealed')
        protocol = read(root / 'protocol.json')
        for field, expected in judge.shared.judge_settings().items():
            self.require(protocol[field] == expected, 'judge_fixed_seed42_settings')
        for field in ('source_sha256', 'implementation_sha256', 'canonical_source_sha256', 'cache_source_sha256'):
            self.hashes(protocol[field])
        self.require((root / 'judge_prompt.txt').read_bytes() == judge.shared.PROMPT.read_bytes(),
                     'judge_official_prompt_bytes')
        self.require(protocol['cache_roots'] == [] and protocol['cached_unique_requests'] == 0 and
                     rows(root / 'reused_scores.jsonl') == [], 'no_historical_judge_grade_reuse')
        questions = self.unique(rows(root / 'question_map.jsonl'))
        inputs = self.unique(rows(source / 'judge_inputs.jsonl'))
        self.require(len(questions) == len(inputs) == 371 and set(questions) == set(inputs) == set(canonical),
                     'judge_371_unique_id_coverage')
        requests = self.unique(rows(root / 'requests.jsonl'), 'request_sha256')
        scores = self.unique(rows(root / 'scores.jsonl'), 'request_sha256')
        self.require(set(requests) == set(scores), 'all_unique_judge_requests_scored')
        histories = defaultdict(list)
        attempts = rows(root / 'attempts.jsonl')
        for attempt in attempts:
            self.require(attempt['request_sha256'] in requests, 'judge_attempt_known_request')
            histories[attempt['request_sha256']].append(attempt)
        template = (root / 'judge_prompt.txt').read_text()
        for key, request in requests.items():
            wire = judge.shared.build_request(request, template)
            self.require(wire == request['request'] and digest(wire) == key and
                         wire['seed'] == 42 and wire['max_tokens'] == 1024, 'judge_wire_seed42_reproduces')
            score = scores[key]
            self.response(score, key, seed=42)
            self.require(score['score_origin'] == 'new_api', 'judge_grade_from_new_api')
            history = histories[key]
            self.require([a['attempt'] for a in history] == list(range(1, len(history) + 1)) and
                         1 <= len(history) <= 5, 'judge_bounded_attempt_sequence')
            self.require(not score.get('recovered_from_preserved_format_error'),
                         'judge_recovery_requires_separate_explicit_audit')
            judge.shared.validate_score(score, key)
            self.require(sum(a['status'] == 'ok' for a in history) == 1 and history[-1]['status'] == 'ok' and
                         all(history[-1].get(k) == score.get(k) for k in
                             ('raw_response', 'response_id', 'request_sha256', 'requested_model',
                              'returned_model', 'seed', 'enable_thinking', 'finish_reason')) and
                         all(score.get(k) == v for k, v in judge.shared.parse_grade(score['raw_response']).items()),
                         'judge_raw_grade_successful_attempt')
        ordered = []
        for qid, q in questions.items():
            self.require(all(q[k] == canonical[qid][k] for k in ('task', 'point', 'question')),
                         'judge_canonical_question_identity')
            self.require(q['ground_truth'] == gold[qid]['answer'] and
                         q['model_output'] == answers[qid]['raw_response'] and
                         q['source_record_id'] == qid and q['source_responses_sha256'] == self.sha(response_path),
                         'judge_preserves_new_open_answer_and_gold')
            self.require(all(q.get(k) == v for k, v in inputs[qid].items()), 'judge_export_exact_source_identity')
            key = q['request_sha256']; wire = judge.shared.build_request(q, template)
            self.require(digest(wire) == key and wire == requests[key]['request'], 'judge_question_wire_mapping')
            ordered.append(scores[key])
        distribution = {str(g): sum(v['score'] == g for v in ordered) for g in GRADES}
        total = sum(v['score'] for v in ordered)
        summary, public = read(root / 'summary.json'), read(public_path)
        for report in (summary, public):
            self.require(report['status'] == 'complete' and report['questions_scored'] == 371 and
                         report['questions_missing'] == 0, 'judge_full_cohort_summary')
            self.require(report['score_sum'] == total and report['llm_judge'] == total / 371 and
                         report['score_distribution'] == distribution, 'judge_score_sum_distribution_mean')
        self.require(public['summary_sha256'] == self.sha(root / 'summary.json'), 'judge_public_summary_hash')
        self.hashes({str(root / name): value for name, value in summary['artifact_sha256'].items()})
        per_question = self.unique(rows(root / 'per_question.jsonl'))
        self.require(set(per_question) == set(questions) and all(per_question[k]['score'] ==
                     scores[q['request_sha256']]['score'] for k, q in questions.items()), 'judge_published_per_question_grades')
        for path in (public_path, root / 'judge_prompt.txt', source / 'judge_inputs.jsonl', root / 'per_question.jsonl'):
            self.sha(path)
        return dict(questions_scored=371, unique_requests=len(requests), score_sum=total,
                    llm_judge=total / 371, score_distribution=distribution,
                    attempts=len(attempts), failed_attempts=sum(a['status'] != 'ok' for a in attempts), seed=42)


def method_roots(output):
    result = [(m, output / 'ours/open' / m) for m in ('semrag', 'ours_1', 'ours_2', 'ours_1_ours_2')]
    result += [(m, output / 'open/text_methods' / m) for m in ('gen_agents', 'memoryos', 'simplemem', 'a_mem')]
    result += [(m, output / 'visual_methods' / m / 'open') for m in ('clip_image_dialogue', 'hippomm')]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--method', choices=tuple(m for m, _ in method_roots(OUT)) + ('all',), default='all')
    args = parser.parse_args()
    audit = Audit(args.output_root)
    plan_path = audit.output / 'plan.json'
    plan = read(plan_path)
    plan_sha256 = audit.sha(plan_path)
    audit.require(plan['seed'] == 1024 and plan['judge_seed'] == 42 and plan['model'] == MODEL,
                  'full_run_seed_model_policy')
    audit.hashes(plan['source_sha256'])
    canonical = audit.unique(rows(CANONICAL / 'questions.jsonl'))
    gold = audit.unique(rows(CANONICAL / 'gold.jsonl'))
    official = read(CANONICAL.parent / 'protocol.json')
    for name in ('questions.jsonl', 'gold.jsonl'):
        audit.require(audit.sha(CANONICAL / name) == official['artifact_sha256']['inputs/' + name],
                      'canonical_official_open_artifact_hash')
    audit.sha(__file__)
    methods, errors, pending = [], [], []
    selected = [(m, p) for m, p in method_roots(audit.output) if args.method in ('all', m)]
    for method, root in selected:
        item = dict(method=method, source_open_root=str(root), open_answers_verified=0, judge_scores_verified=0)
        try:
            answers, response_path, provenance = (audit.hippomm_open(root, canonical) if method == 'hippomm'
                else audit.generic_open(root, method, canonical))
            item.update(open_answers_verified=len(answers), answer_seed=1024, answer_provenance=provenance,
                        empty_answers=sum(not r['raw_response'] for r in answers.values()),
                        token_limit_hits=sum(r['hit_token_limit'] for r in answers.values()))
            measured = audit.judge_audit(root / 'judge', root, answers, response_path, canonical, gold)
            item.update(status='passed', judge=measured, judge_scores_verified=measured['questions_scored'])
        except Pending as error:
            item.update(status='pending', reason=str(error)); pending.append(dict(method=method, reason=str(error)))
        except Exception as error:
            item.update(status='failed', error=str(error))
            errors.append(dict(method=method, error=str(error), traceback=traceback.format_exc()))
        methods.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    for path, stat in audit.files.items():
        current = Path(path).stat()
        if current.st_size != stat['bytes'] or current.st_mtime_ns != stat['mtime_ns']:
            errors.append(dict(path=path, error='Audited source changed during the audit'))
    completed = [m for m in methods if m['status'] == 'passed']
    report = dict(status='failed' if errors else 'pending' if pending else 'passed',
        passed=not errors and not pending, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        expected_methods=len(selected), methods_passed=len(completed),
        expected_open_answers=len(selected) * 371,
        open_answers_verified=sum(m['open_answers_verified'] for m in methods),
        judge_scores_verified=sum(m['judge_scores_verified'] for m in methods),
        unique_judge_requests_verified=sum(m['judge']['unique_requests'] for m in completed),
        score_sum=sum(m['judge']['score_sum'] for m in completed),
        score_distribution={str(g): sum(m['judge']['score_distribution'][str(g)] for m in completed) for g in GRADES},
        answer_seed=1024, judge_seed=42, real_api_calls_by_audit=0,
        plan_sha256_at_read=plan_sha256,
        fabricated_judge_grades=0 if not errors and not pending else None,
        errors=errors, pending=pending, methods=methods, checks=dict(audit.checks),
        artifact_sha256={path: value['sha256'] for path, value in sorted(audit.files.items())})
    destination = args.report or audit.output / 'audit/final_audit_open_judge.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print(json.dumps({k: report[k] for k in ('status', 'methods_passed', 'open_answers_verified', 'judge_scores_verified')}))
    if errors or pending:
        raise SystemExit(1 if errors else 2)


if __name__ == '__main__':
    main()
