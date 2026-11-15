"""Independent offline audit of the exploratory MemEye DeepSeek Flash alias run.

The runner, transport, encoders and graph implementation are not imported. This
script reconstructs outgoing API bodies from original bytes and frozen text,
checks the complete provenance chain, and recomputes retrieval/answer metrics.
It writes only OUTPUT/independent_audit.json. No API, GPU or image copying.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

CONDITIONS = ('baseline_question_only', 'master_answer_query')
MODEL = 'deepseek-v4.1-flash'
MODELS = {MODEL}
BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
CREDENTIAL_VARIABLE = 'DEEPSEEKV4_API_KEY'
PROVIDER = 'DashScope International'
MEMORY_SEPARATOR = '\nMemory description: '
QUERY_SEPARATOR = "\nquestion's image:\nimage_caption: "


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def validate_profile(contract):
    """Independently verify the provider alias and explicitly unresolved weights."""
    require(contract['scope'] == 'all_original_qwen_roles' and contract['model'] == MODEL,
            'Wrong backbone scope/model')
    config = contract['api_config']
    require(config['base_url'] == BASE_URL and config['credential_variable'] == CREDENTIAL_VARIABLE,
            'Wrong API routing/provider')
    require(config['model'] == MODEL, 'API profile model mismatch')
    require(config['api_type'] == 'chat_completions' and contract['provider'] == PROVIDER,
            'Wrong API type/provider')
    require(type(contract['caption_max_output_tokens']) is int and contract['caption_max_output_tokens'] == 1024
            and contract['reader_max_output_tokens'] == 16
            and contract['temperature'] == 0 and contract['image_detail'] == 'high',
            'Generation settings changed')
    require(contract.get('caption_budget_policy') == 'fixed_1024_from_start'
            and 'caption_budget_amendment' not in contract, 'Alias run requires 1024 from its initial contract')
    require(contract.get('model_identity') == {
        'kind': 'provider_alias', 'requested_model': MODEL,
        'resolved_revision': None, 'distinct_weights_vs_1017': None,
    }, 'Alias identity must leave the resolved weights explicitly unconfirmed')
    require(contract['enable_thinking'] is False and contract['web_search'] is False and contract['seed'] is None
            and contract['reasoning_effort'] is None, 'Thinking/seed contract differs')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def index(rows, key='id'):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f'Duplicate {key}')
    return result


def choice(text):
    match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', text.strip(), re.I)
    return match.group(1).upper() if match else None


def aggregate(pairs):
    n = len(pairs)
    base = sum(row['baseline_correct'] for row in pairs)
    new = sum(row['expanded_correct'] for row in pairs)
    wins = sum(not row['baseline_correct'] and row['expanded_correct'] for row in pairs)
    losses = sum(row['baseline_correct'] and not row['expanded_correct'] for row in pairs)
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
            / 2**discordant) if discordant else 1.
    return {'questions': n, 'baseline_correct': base, 'expanded_correct': new,
            'baseline_accuracy': base / n if n else None,
            'expanded_accuracy': new / n if n else None,
            'delta_percentage_points': 100 * (new - base) / n if n else None,
            'wrong_to_correct': wins, 'correct_to_wrong': losses, 'mcnemar_exact_p': p}


def recall(rows, gold):
    total = sum(len(set(gold[row['question_id']]['gold_ids'])) for row in rows)
    result = {'questions': len(rows), 'gold_references': total}
    for condition in CONDITIONS:
        hits, per_question = [], []
        for row in rows:
            targets = set(gold[row['question_id']]['gold_ids'])
            selected = index(row['conditions'], 'name')[condition]['top5_memory_ids']
            require(targets and len(selected) == len(set(selected)) == 5,
                    'Recall requires nonempty gold and five distinct memories')
            n = len(targets.intersection(selected))
            hits.append(n)
            per_question.append(n / len(targets))
        result[condition] = {'gold_hits_top5': sum(hits), 'gold_recall': sum(hits) / total,
                             'macro_recall_at_5': sum(per_question) / len(rows),
                             'questions_with_gold_hit': sum(n > 0 for n in hits),
                             'questions_with_all_gold': sum(r == 1 for r in per_question)}
    return result


def wire_request(payload, model, token_limit, image_detail):
    """Reconstruct exact typed API messages independently of the transport."""
    require(set(payload) <= {'system', 'user_text', 'query_images'} and
            {'user_text', 'query_images'} <= set(payload), 'Unexpected outbound payload fields')
    require(model in MODELS, 'Unapproved model alias')
    content = []
    for image in payload['query_images']:
        require(set(image) == {'path', 'sha256'}, 'Unexpected outbound image metadata')
        data = Path(image['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == image['sha256'], 'Original image bytes changed')
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        else:
            require(data.startswith(b'\xff\xd8\xff'), 'Unsupported original image type')
            mime = 'image/jpeg'
        content.append({'type': 'image_url', 'image_url': {
            'url': f'data:{mime};base64,' + base64.b64encode(data).decode('ascii'), 'detail': image_detail}})
    content.append({'type': 'text', 'text': payload['user_text']})
    messages = [{'role': 'system', 'content': payload['system']}] if 'system' in payload else []
    messages.append({'role': 'user', 'content': content})
    return {'model': model, 'messages': messages, 'temperature': 0, 'n': 1,
            'stream': False, 'max_tokens': token_limit, 'enable_thinking': False}


def validate_response(answer, request, protocol, stage):
    """Reject wrong-run, historical, truncated or malformed saved successes."""
    contract = protocol['contract']
    for key, expected in {'record_id': request['record_id'], 'payload_sha256': request['payload_sha256'],
                          'request_sha256': request['request_sha256'], 'run_id': protocol['run_id'],
                          'experiment_signature': protocol['experiment_signature'],
                          'requested_model': contract['model'], 'returned_model': contract['model'],
                          'status': 'ok', 'finish_reason': 'stop'}.items():
        require(answer.get(key) == expected, f'{stage} response {key} mismatch: {request["record_id"]}')
    require(answer.get('historical_response_reuse') is False, 'Historical API response reused')
    require(isinstance(answer.get('raw_response'), str) and answer['raw_response'].strip(), 'Empty API output')
    require(not answer.get('refusal'), 'API refusal')
    require(isinstance(answer.get('response_id'), str) and answer['response_id'], 'Missing provider response ID')
    require(isinstance(answer.get('execution_id'), str) and answer['execution_id'], 'Missing execution ID')
    require(type(answer['attempts']) is int and 1 <= answer['attempts'] <= contract['max_attempts'],
            'Invalid attempt count')
    require(answer['retry_count'] == answer['attempts'] - 1, 'Retry counter mismatch')
    if stage == 'reader':
        require(choice(answer['raw_response']) is not None, 'Reader output format error')
    usage = answer.get('usage') or {}
    image_tokens = (usage.get('prompt_tokens_details') or {}).get('image_tokens')
    if request.get('query_images'):
        require(type(image_tokens) is int and image_tokens > 0,
                'Image request lacks positive provider image-token usage')
    details = usage.get('completion_tokens_details') or {}
    reasoning = details.get('reasoning_tokens')
    require(answer.get('reasoning_tokens') == reasoning, 'Reasoning usage differs from raw usage')
    require(reasoning is None or type(reasoning) is int and reasoning >= 0, 'Invalid reasoning usage')
    require(reasoning in (None, 0), 'enable_thinking=false returned nonzero reasoning tokens')
    require(not answer.get('reasoning_content'), 'enable_thinking=false returned reasoning content')
    require(answer.get('reasoning_content_present') is False and answer.get('enable_thinking') is False,
            'Missing or non-disabled thinking response provenance')
    return reasoning


class Audit:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.snapshots = {}
        self.checked_hashes = {}
        self.report = {'audit_status': 'running', 'output': str(self.output),
                       'auditor': str(Path(__file__).resolve()), 'auditor_sha256': sha(__file__),
                       'audited_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                       'errors': [], 'completed_sections': []}

    def read(self, path, jsonl=False):
        path = Path(path)
        data = path.read_bytes()
        self.snapshots[str(path)] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines() if line.strip()] if jsonl else json.loads(data)

    def hashes(self, mapping, base=None):
        for name, expected in mapping.items():
            path = Path(name)
            if not path.is_absolute():
                require(base is not None, 'Relative hash path has no base')
                path = Path(base) / path
            key = str(path.resolve())
            actual = self.checked_hashes.setdefault(key, sha(path))
            require(actual == expected, f'SHA256 mismatch: {path}')

    def section(self, name, result):
        self.report[name] = result
        self.report['completed_sections'].append(name)

    def run(self):
        root = self.output
        p = self.read(root / 'protocol.json')
        c = p['contract']
        require(digest(c) == p['experiment_signature'], 'Root experiment signature mismatch')
        validate_profile(c)
        self.identity_and_budget(p)
        require(set(c['replaced_roles']) == {'history_image_caption', 'query_image_caption', 'answer_reader'},
                'Replacement scope misses a Qwen role')
        for key in ('prior_qwen_caption_reuse', 'prior_qwen_retrieval_reuse', 'prior_answer_reuse', 'gold_sent_to_api'):
            require(c[key] is False, f'Incorrect declared policy: {key}')
        self.hashes(c['source_sha256'])
        inputs = self.read(root / 'inputs/protocol.json')
        require(inputs['model'] == c['model'], 'Input preparation model mismatch')
        self.hashes(inputs['source_sha256'])
        self.hashes(inputs['artifact_sha256'], root / 'inputs')
        memories = self.read(root / 'inputs/source_memories.jsonl', True)
        questions = self.read(root / 'inputs/source_questions.jsonl', True)
        gold_rows = self.read(root / 'inputs/evaluation.jsonl', True)
        image_rows = self.read(root / 'inputs/images.jsonl', True)
        caption_jobs = self.read(root / 'inputs/api_requests_caption.jsonl', True)
        mm, qq, gg, images, jobs = [index(rows) for rows in (memories, questions, gold_rows, image_rows, caption_jobs)]
        require(len(mm) == 438 and len(qq) == len(gg) == 371 and len(images) == len(jobs) == 437, 'Input scope incomplete')
        require(set(qq) == set(gg), 'Question/gold ID sets differ')
        require(sum(len(set(row['gold_ids'])) for row in gold_rows) == 1041, 'Gold reference denominator differs')
        require(len(index(memories, 'image_id')) == 438, 'Duplicate image node')
        domains = sorted({r['task'] for r in memories})
        require(len(domains) == 8 and {q['task'] for q in questions} == set(domains), 'Domain coverage differs')
        for memory in memories:
            require(set(memory) == {'id', 'image_id', 'task', 'dialogue', 'timestamp', 'insertion_index',
                                   'image_sha256', 'image_path'}, 'Source memory has old caption/vector/extra fields')
            require(memory['id'].split('/', 1)[0] == memory['task'], 'Memory domain mismatch')
        for q in questions:
            require(set(q) == {'id', 'task', 'question', 'options', 'query_images', 'candidate_statements'},
                    'Source question has old query/gold/extra fields')
            require(set(q['options']) == set(q['candidate_statements']) == set('ABCD'), 'Options/candidates missing')
            require(gg[q['id']]['gold_choice'] in 'ABCD', 'Invalid gold answer')
            require(bool(gg[q['id']]['gold_ids']) and all(mid in mm and mm[mid]['task'] == q['task']
                    for mid in gg[q['id']]['gold_ids']), 'Gold target unreachable in its domain memory pool')
        history_hashes = {m['image_sha256'] for m in memories}
        query_hashes = {im['sha256'] for q in questions for im in q['query_images']}
        image_questions = sum(bool(q['query_images']) for q in questions)
        require((len(history_hashes), len(query_hashes), len(history_hashes & query_hashes), image_questions)
                == (433, 6, 2, 8), 'History/query image scope differs')
        require(set(images) == set(jobs) == history_hashes | query_hashes, 'Image hash scope differs')
        for key, im in images.items():
            require(im['sha256'] == key and im['path'] in im['source_paths'], 'Image manifest identity mismatch')
            self.hashes({path: key for path in im['source_paths']})
            require(im['history_memory_ids'] == [m['id'] for m in memories if m['image_sha256'] == key],
                    'Image history references differ')
            require(im['query_question_ids'] == [q['id'] for q in questions for image in q['query_images']
                                                if image['sha256'] == key], 'Image query references differ')
            require(jobs[key] == {'id': key, 'user_text': inputs['caption_prompt'],
                                 'query_images': [{'path': im['path'], 'sha256': key}]},
                    'Caption input includes non-image benchmark information')
        old_prompt = self.read(Path(inputs['reader_system_prompt_source']))['reader_settings']['reader_system_prompt']
        require(inputs['reader_system_prompt'] == old_prompt, 'Reader prompt differs from Qwen reference')
        self.section('inputs', {'questions': len(qq), 'domains': len(domains), 'memory_leaves': len(mm),
                              'unique_images': len(images), 'history_unique_images': len(history_hashes),
                              'query_unique_images': len(query_hashes), 'image_questions': image_questions,
                              'shared_history_query_images': len(history_hashes & query_hashes),
                              'candidate_statements': sum(len(q['candidate_statements']) for q in questions),
                              'empty_gold': 0, 'unreachable_or_cross_domain_gold': 0,
                              'gold_free_generation_inputs': True})
        cap_payloads = {r['id']: {'user_text': r['user_text'], 'query_images': r['query_images']} for r in caption_jobs}
        cap_requests, cap_fresh = self.stage('caption_generation', cap_payloads, p)
        captions = index(self.read(root / 'captions.jsonl', True))
        require(set(captions) == set(images), 'Generated caption coverage differs')
        for key, caption in captions.items():
            response = cap_fresh[cap_requests[key]['request_sha256']]
            require(caption == {'id': key, 'caption': response['raw_response'].strip(), 'model': c['model'],
                                'source_response_record_id': response['record_id'],
                                'response_id': response['response_id'],
                                'request_sha256': response['request_sha256']}, 'Caption not bound to this fresh API response')
        reader_texts = {m['id']: m['dialogue'] + MEMORY_SEPARATOR + captions[m['image_sha256']]['caption'] for m in memories}
        graph_protocol = self.read(root / 'retrieval_protocol.json')
        require(graph_protocol['source_run_id'] == p['run_id'] and
                graph_protocol['source_experiment_signature'] == p['experiment_signature'], 'Retrieval run binding differs')
        require(digest(graph_protocol['contract']) == graph_protocol['experiment_signature'], 'Retrieval signature mismatch')
        require(graph_protocol['caption_model'] == c['model'], 'Retrieval used another caption backbone')
        self.hashes(graph_protocol['source_sha256'])
        require(graph_protocol['artifact_sha256'] == graph_protocol['output_sha256'], 'Retrieval artifact maps differ')
        self.hashes(graph_protocol['output_sha256'])
        graph_states = self.graphs(domains, memories, captions, inputs, graph_protocol)
        offline = self.read(root / 'offline/memories.jsonl', True)
        expected_offline = [{**{k: v for k, v in m.items() if k not in ('image_path', 'image_sha256')},
                             'caption': captions[m['image_sha256']]['caption'], 'retrieval_text': reader_texts[m['id']]}
                            for m in memories]
        require(offline == expected_offline, 'Offline text store differs from original dialogue and new captions')
        retrieval = self.read(root / 'retrieval_results.jsonl', True)
        rr = index(retrieval, 'question_id')
        require(set(rr) == set(qq), 'Retrieval question IDs differ')
        require(sha(root / 'retrieval_results.jsonl') == graph_protocol['results_sha256'], 'Retrieval results hash differs')
        expected_reader, expected_manifest = {}, {}
        for result in retrieval:
            qid = result['question_id']
            q = qq[qid]
            domain_memory = [m for m in memories if m['task'] == q['task']]
            nodes = graph_states[q['task']]['image_nodes']
            graph_masters = graph_states[q['task']]['masters']
            query_captions = [captions[im['sha256']]['caption'] for im in q['query_images']]
            query_text = q['question'] + (QUERY_SEPARATOR + ' '.join(query_captions) if query_captions else '')
            for key, expected in {'id': qid, 'task': q['task'], 'question': q['question'], 'options': q['options'],
                                  'query_images': q['query_images'], 'query_image_paths': [im['path'] for im in q['query_images']],
                                  'query_image_captions': query_captions, 'baseline_retrieval_query': query_text,
                                  'candidate_statements': q['candidate_statements']}.items():
                require(result[key] == expected, f'Retrieval {key} differs: {qid}')
            master_rows = index(result['masters'], 'master_id')
            require(set(master_rows) == {m['id'] for m in graph_masters}, 'Retrieval master set differs')
            for master in graph_masters:
                row = master_rows[master['id']]
                require(row['member_ids'] == master['member_ids'] and row['n'] == master['n'], 'Retrieval master members differ')
                scores = row['clip_candidate_scores']
                require(set(scores) == set('ABCD') and all(math.isfinite(v) for v in scores.values()), 'Invalid CLIP scores')
                selected = max('ABCD', key=lambda key: scores[key])
                require(row['selected_choice'] == selected and row['selected_statement'] == q['candidate_statements'][selected]
                        and row['augmented_query'] == query_text + '\nCandidate answer: ' + q['candidate_statements'][selected],
                        'Master answer/query selection differs from CLIP argmax')
            leaf_rows = result['leaves']
            require(len(leaf_rows) == len(domain_memory), 'Retrieval does not cover the full domain pool')
            for i, (leaf, memory, node) in enumerate(zip(leaf_rows, domain_memory, nodes)):
                require(leaf['leaf_index'] == i and leaf['memory_id'] == memory['id'] and leaf['image_id'] == memory['image_id']
                        and leaf['master_id'] == node['master_id']
                        and leaf['master_choice'] == master_rows[node['master_id']]['selected_choice'], 'Retrieval leaf identity differs')
            conditions = index(result['conditions'], 'name')
            require(set(conditions) == set(CONDITIONS), 'Missing retrieval condition')
            for name, score_key, rank_key in zip(CONDITIONS, ('baseline_score', 'expanded_score'), ('baseline_rank', 'expanded_rank')):
                condition = conditions[name]
                require(all(math.isfinite(row[score_key]) for row in leaf_rows), 'Nonfinite BGE scores')
                ordered = sorted(range(len(leaf_rows)), key=lambda i: -leaf_rows[i][score_key])
                ids = [domain_memory[i]['id'] for i in ordered]
                require(condition['full_rank_memory_ids'] == ids and condition['top5_memory_ids'] == ids[:5], 'Rank order differs from scores')
                require(condition['top5_leaf_indices'] == ordered[:5] and
                        condition['top5_scores'] == [leaf_rows[i][score_key] for i in ordered[:5]], 'Top5 indices/scores differ')
                for rank, i in enumerate(ordered, 1):
                    require(leaf_rows[i][rank_key] == rank, 'Saved leaf rank differs')
                expected_top = [{**leaf_rows[i], 'rank': rank, 'score': leaf_rows[i][score_key]}
                                for rank, i in enumerate(ordered[:5], 1)]
                require(condition['top5'] == expected_top, 'Expanded Top5 records differ')
                expected_evidence = [{'memory_id': mid, 'filename': Path(mm[mid]['image_path']).name,
                                      'reader_text': reader_texts[mid]} for mid in ids[:5]]
                require(condition['reader_memories'] == expected_evidence, 'Reader evidence contains stale text or hypotheses')
                context = '\n\n'.join(f'Memory {rank}: {reader_texts[mid]}' for rank, mid in enumerate(ids[:5], 1))
                require(condition['reader_context'] == context, 'Reader context differs from ordered evidence')
                text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + context
                        + '\n\nQuestion:\n' + q['question'] + '\n\nOptions:\n'
                        + '\n'.join(f'{key}. {q["options"][key]}' for key in 'ABCD') + '\n\nAnswer with one letter only.')
                record_id = qid + '::' + name
                expected_reader[record_id] = {'system': inputs['reader_system_prompt'], 'user_text': text,
                                              'query_images': q['query_images']}
                expected_manifest[record_id] = {'record_id': record_id, 'question_id': qid, 'task': q['task'],
                                                'condition': name, 'top_memory_ids': ids[:5]}
            require(result['top5_same_order'] == (conditions[CONDITIONS[0]]['top5_memory_ids'] == conditions[CONDITIONS[1]]['top5_memory_ids']),
                    'Same Top5 order flag differs')
        metrics = {'overall': recall(retrieval, gg), 'by_domain': {
            domain: recall([r for r in retrieval if r['task'] == domain], gg) for domain in domains}}
        require(self.read(root / 'retrieval_summary.json') == metrics, 'Saved Recall@5 differs from independent calculation')
        self.section('retrieval', {'questions': len(rr), 'full_domain_pools_checked': len(rr),
                                   'top5_conditions': len(expected_reader), 'metrics': metrics,
                                   'old_query_caption_reuse': 0, 'hypotheses_in_reader_evidence': 0,
                                   'rank_check': 'Stable descending ordering of every saved leaf score; no encoder rerun.'})
        manifest = index(self.read(root / 'reader_manifest.jsonl', True), 'record_id')
        require(manifest == expected_manifest and len(manifest) == 742, 'Reader manifest differs from retrieval')
        requests, fresh = self.stage('reader', expected_reader, p)
        pairs, shared = [], 0
        for q in questions:
            ids = [q['id'] + '::' + name for name in CONDITIONS]
            answers = [fresh[requests[rid]['request_sha256']] for rid in ids]
            choices = [choice(answer['raw_response']) for answer in answers]
            same_payload = requests[ids[0]]['payload_sha256'] == requests[ids[1]]['payload_sha256']
            same_top5 = manifest[ids[0]]['top_memory_ids'] == manifest[ids[1]]['top_memory_ids']
            if same_payload:
                require(answers[0]['response_id'] == answers[1]['response_id'] and choices[0] == choices[1], 'Same payload used different results')
                shared += 1
            pairs.append({'question_id': q['id'], 'domain': q['task'], 'baseline_answer': choices[0],
                          'expanded_answer': choices[1], 'baseline_correct': choices[0] == gg[q['id']]['gold_choice'],
                          'expanded_correct': choices[1] == gg[q['id']]['gold_choice'],
                          'same_reader_payload': same_payload, 'same_top5_order': same_top5})
        require(self.read(root / 'paired_results.jsonl', True) == pairs, 'Paired result choices/flags/correctness differ')
        summary = self.read(root / 'summary.json')
        require(summary['overall'] == aggregate(pairs), 'Overall answer metrics differ')
        require(summary['by_domain'] == {domain: aggregate([r for r in pairs if r['domain'] == domain]) for domain in domains},
                'Per-domain answer metrics differ')
        require(summary['recall_at_5'] == metrics, 'Summary recall differs')
        require(summary['scope'] == c['scope'] and summary['model'] == c['model'] and summary['run_id'] == p['run_id'], 'Summary provenance differs')
        require(summary['historical_qwen_caption_or_answer_reuse'] is False, 'Summary declares historical reuse')
        require(summary['logical_reader_records'] == 742 and summary['same_run_shared_reader_records'] == 742 - len(fresh),
                'Summary reader dedup counts differ')
        for stage in ('caption_generation', 'reader'):
            actual = self.report[stage]
            reported = summary['generation_stages'][stage]
            for key in ('fresh_requests', 'returned_models', 'errors', 'refusals', 'length_stops', 'retry_count', 'usage'):
                require(actual[key] == reported[key], f'Summary {stage}/{key} differs')
        response_ids = [row['response_id'] for row in [*cap_fresh.values(), *fresh.values()]]
        require(len(response_ids) == len(set(response_ids)), 'Provider response reused across distinct generation requests/stages')
        self.section('answers', {'overall': aggregate(pairs), 'by_domain': summary['by_domain'],
                                'logical_records': len(requests), 'unique_fresh_requests': len(fresh),
                                'same_run_shared_records': len(requests) - len(fresh),
                                'same_payload_question_pairs': shared,
                                'image_question_reader_records': sum(bool(v['query_images']) for v in expected_reader.values()),
                                'gold_sent_to_api': False, 'reader_evidence_exact': True})
        # Reject mutation during the audit, including model implementation files.
        for path, expected in {**self.checked_hashes, **self.snapshots}.items():
            require(sha(path) == expected, f'Artifact changed while auditing: {path}')
        self.report.update(audit_status='passed', model=c['model'], run_id=p['run_id'],
                           api_base_url=BASE_URL, credential_variable=CREDENTIAL_VARIABLE, provider=PROVIDER,
                           caption_max_output_tokens=c['caption_max_output_tokens'],
                           reader_max_output_tokens=c['reader_max_output_tokens'],
                           resolved_model_revision=None, distinct_weights_vs_1017=None,
                           checked_source_and_artifact_files=len(set(self.checked_hashes) | set(self.snapshots)),
                           artifact_sha256=self.snapshots,
                           limitations=['No GPU or encoder inference is repeated; saved numerical scores and graph invariants are checked.',
                                        'Local request manifests and transport provenance are verified; provider internals are not observable.',
                                        'Unreported reasoning usage stays unknown; it is never counted as zero.',
                                        'Fresh API provenance establishes non-reuse; coincidentally equal generated text is not evidence of reuse.',
                                        'The Flash alias does not establish a distinct weight revision from the 1017 model.'])
        return self.report

    def identity_and_budget(self, protocol):
        """Report a fresh alias experiment without claiming different weights."""
        c = protocol['contract']
        validate_profile(c)
        self.section('model_identity', {
            **c['model_identity'],
            'interpretation': 'The provider alias was requested and returned. Its resolved weight revision is unconfirmed; differences from the 1017 run cannot establish an ablation across distinct weights.',
        })
        self.section('caption_budget', {
            'caption_max_output_tokens': 1024, 'reader_max_output_tokens': 16,
            'amended': False, 'policy': 'fixed_1024_from_start',
            'output_budget_matches_gpt_comparators': False,
            'output_budget_matches_1017_complete_captions': True,
            'comparison_limitation': 'This alias run uses 1024 tokens from its initial caption request. The completed 1017 run also used 1024; GPT caption runs used 256.',
        })

    def stage(self, name, expected_payloads, protocol):
        root, c = self.output, protocol['contract']
        stage = root / name
        meta = self.read(stage / 'protocol.json')
        require(meta['stage'] == name and meta['run_id'] == protocol['run_id'] and
                meta['experiment_signature'] == protocol['experiment_signature'] and meta['model'] == c['model'],
                f'{name} stage provenance differs')
        self.hashes(meta['source_sha256'])
        rows = self.read(stage / 'requests.jsonl', True)
        require(sha(stage / 'requests.jsonl') == meta['requests_sha256'], f'{name} request file changed')
        requests = index(rows, 'record_id')
        require(set(requests) == set(expected_payloads), f'{name} request coverage differs')
        token_limit = c['caption_max_output_tokens'] if name == 'caption_generation' else c['reader_max_output_tokens']
        require(meta['max_output_tokens'] == token_limit, f'{name} token limit differs')
        representatives = {}
        for row in rows:
            payload = expected_payloads[row['record_id']]
            require(set(row) == set(payload) | {'record_id', 'payload_sha256', 'request_sha256'}, f'{name} unexpected/gold request fields')
            require(all(row[key] == value for key, value in payload.items()), f'{name} outbound content differs')
            require(row['payload_sha256'] == digest(payload), f'{name} payload hash differs')
            body = wire_request(payload, c['model'], token_limit, c['image_detail'])
            require(digest(body) == row['request_sha256'], f'{name} actual wire hash differs')
            representatives.setdefault(row['request_sha256'], row)
        require(meta['logical_records'] == len(rows) and meta['unique_requests'] == len(representatives), f'{name} stage counts differ')
        response_rows = self.read(stage / 'fresh_responses.jsonl', True)
        fresh = index(response_rows, 'request_sha256')
        require(set(fresh) == set(representatives), f'{name} fresh coverage incomplete')
        require(len({r['response_id'] for r in response_rows}) == len(response_rows), f'{name} repeated provider response IDs')
        reasoning = {'reported_zero': 0, 'reported_nonzero': 0, 'unreported': 0, 'reported_token_total': 0}
        for key, answer in fresh.items():
            value = validate_response(answer, representatives[key], protocol, name)
            reasoning['unreported' if value is None else 'reported_zero' if value == 0 else 'reported_nonzero'] += 1
            reasoning['reported_token_total'] += value or 0
        attempts = self.read(stage / 'api_attempts.jsonl', True)
        by_request = defaultdict(list)
        for attempt in attempts:
            key = attempt['request_sha256']
            require(key in representatives and attempt['record_id'] == representatives[key]['record_id']
                    and attempt['stage'] == name and attempt['status'] in {'ok', 'api_error'}, f'{name} attempt linkage differs')
            by_request[key].append(attempt)
        require(set(by_request) == set(fresh), f'{name} API attempt coverage differs')
        for key, answer in fresh.items():
            successful = [a for a in by_request[key] if a['status'] == 'ok']
            require(len(successful) == 1, f'{name} must have one success per unique request')
            success = successful[0]
            require(success['response_id'] == answer['response_id'] and success['execution_id'] == answer['execution_id']
                    and success['attempt'] == answer['attempts'], f'{name} success/source result differs')
            execution = [a for a in by_request[key] if a['execution_id'] == answer['execution_id']]
            require(sorted(a['attempt'] for a in execution) == list(range(1, answer['attempts'] + 1)), f'{name} retry sequence incomplete')
        info = {'logical_records': len(rows), 'fresh_requests': len(fresh),
                'same_run_shared_records': len(rows) - len(fresh), 'actual_wire_hashes_verified': len(rows),
                'returned_models': dict(Counter(r['returned_model'] for r in response_rows)),
                'errors': sum(r['status'] != 'ok' for r in response_rows),
                'refusals': sum(bool(r.get('refusal')) for r in response_rows),
                'length_stops': sum(r['finish_reason'] == 'length' for r in response_rows),
                'empty_outputs': sum(not r['raw_response'].strip() for r in response_rows),
                'format_errors': sum(choice(r['raw_response']) is None for r in response_rows) if name == 'reader' else 0,
                'retry_count': sum(r['retry_count'] for r in response_rows),
                'api_error_attempts': sum(a['status'] == 'api_error' for a in attempts),
                'reasoning_usage': reasoning, 'historical_response_reuse': 0,
                'nonempty_reasoning_outputs': sum(bool(r.get('reasoning_content')) for r in response_rows),
                'max_output_tokens': token_limit,
                'image_requests_with_positive_provider_image_tokens': sum(
                    bool(representatives[key].get('query_images'))
                    and (row.get('usage') or {}).get('prompt_tokens_details', {}).get('image_tokens', 0) > 0
                    for key, row in fresh.items()),
                'usage': {key: sum((r.get('usage') or {}).get(key, 0) for r in response_rows)
                          for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
        self.section(name, info)
        return requests, fresh

    def graphs(self, domains, memories, captions, inputs, protocol):
        states, records = {}, []
        for domain in domains:
            path = self.output / 'domains' / domain
            graph = self.read(path / 'memory/graph.json')
            build = self.read(path / 'build_protocol.json')
            graph_signature = protocol.get('original_strict_stage_signature', protocol['experiment_signature'])
            require(build['source_signature'] == graph_signature, 'Graph build signature differs')
            self.hashes(build['artifact_sha256'])
            domain_memories = [m for m in memories if m['task'] == domain]
            nodes, masters = graph['image_nodes'], graph['masters']
            node_lookup, master_lookup = index(nodes), index(masters)
            require([n['id'] for n in nodes] == [m['image_id'] for m in domain_memories], 'Graph source/order differs')
            require(graph['leaf_visual_dim'] == 2 and graph['text_embedding_cache']['persisted'] is False,
                    'Graph saves unexpected per-image embeddings')
            require(sorted(p.name for p in (path / 'memory').iterdir()) == ['basis.npy', 'graph.json', 'master_sums.npy'],
                    'Unexpected image/vector copies in memory directory')
            require(not (path / 'build_checkpoint.json').exists(), 'Completed graph retains unfinished checkpoint')
            basis = np.load(path / 'memory/basis.npy', allow_pickle=False)
            sums = np.load(path / 'memory/master_sums.npy', allow_pickle=False)
            require(sha(path / 'memory/basis.npy') == inputs['basis_by_domain'][domain]['sha256'], 'Graph uses a different PCA basis')
            require(basis.shape == (graph['visual_dim'], 2) and sums.shape == (len(masters), graph['visual_dim'])
                    and np.isfinite(basis).all() and np.isfinite(sums).all(), 'Invalid saved visual arrays')
            require(np.allclose(basis.T @ basis, np.eye(2), atol=1e-10, rtol=0), 'Visual basis is not orthonormal')
            for node, memory in zip(nodes, domain_memories):
                require(set(node) == {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'},
                        'Leaf stores original image or extra full visual vector')
                require(node['caption'] == captions[memory['image_sha256']]['caption']
                        and node['dialogue'] == memory['dialogue'] and node['timestamp'] == memory['timestamp'],
                        'Graph text differs from new captions and original history')
                coords = np.asarray(node['visual_coords'])
                require(coords.shape == (2,) and np.isfinite(coords).all() and node['master_id'] in master_lookup,
                        'Invalid leaf coordinates/master membership')
            covered, projection_errors = [], []
            for i, master in enumerate(masters):
                require(set(master) <= {'id', 'member_ids', 'n', 'approximate', 'source_overlaps', 'visual_sum_ref', 'visual_mean_ref'},
                        'Master stores unexpected text/per-image content')
                members = master['member_ids']
                require(master['n'] == len(members) > 0 and len(set(members)) == len(members), 'Invalid master membership')
                require(master['visual_sum_ref'] == {'path': 'master_sums.npy', 'row': i} and
                        master['visual_mean_ref'] == {'path': 'master_sums.npy', 'row': i, 'scale': 1 / len(members)},
                        'Master visual index/mean scale differs')
                require(all(mid in node_lookup and node_lookup[mid]['master_id'] == master['id'] for mid in members),
                        'Graph leaf/master links disagree')
                coordinates = np.sum([node_lookup[mid]['visual_coords'] for mid in members], axis=0)
                err = float(np.linalg.norm(basis.T @ sums[i] - coordinates))
                require(np.allclose(basis.T @ sums[i], coordinates, atol=1e-10, rtol=1e-10), 'Master projection differs from sum of leaf coordinates')
                projection_errors.append(err)
                covered.extend(members)
            require(sorted(covered) == sorted(node_lookup), 'Masters are not a complete disjoint leaf partition')
            require(build['images'] == len(nodes) and build['masters'] == len(masters) and
                    build['insertion_order'] == [node['id'] for node in nodes] and
                    build['image_arrival_forwards'] == len(nodes) and build['old_image_reencodes'] == 0 and
                    build['per_image_visual_embeddings_saved'] is False and build['leaf_visual_dimension'] == 2,
                    'Build counters/memory policy differ')
            audits = self.read(path / 'insertion_audit.jsonl', True)
            require(len(audits) == len(nodes), 'Missing insertion audit')
            for i, (audit, node) in enumerate(zip(audits, nodes), 1):
                require(audit['image_id'] == node['id'] and audit['new_image_ids'] == [node['id']]
                        and audit['n_images'] == audit['insertion_index'] == i,
                        'Insertion stream identity differs')
                for key in ('global_conservation_l2_error', 'global_conservation_max_abs_error', 'max_projection_consistency_l2_error'):
                    require(math.isfinite(audit[key]) and audit[key] < 1e-9, 'Visual conservation/projection audit failed')
            records.append({'domain': domain, 'memory_leaves': len(nodes), 'masters': len(masters),
                            'approximate_masters': sum(bool(m['approximate']) for m in masters),
                            'max_projection_error': max(projection_errors), 'leaf_visual_dimension': 2,
                            'image_arrival_forwards': len(nodes), 'old_image_reencodes': 0})
            states[domain] = graph
        require(protocol['old_image_reencodes'] == 0 and protocol['per_image_visual_embeddings_saved'] is False
                and protocol['gold_loaded_for_ranking'] is False, 'Retrieval memory policy differs')
        if 'original_strict_stage_signature' in protocol:
            require(protocol['contract']['original_strict_stage_signature'] == protocol['original_strict_stage_signature']
                    and protocol['completed_graphs_reused_within_same_run'] is True
                    and protocol['prior_run_graphs_reused'] is False and protocol['image_forwards_this_extension'] == 0,
                    'Recovery changed the graph/image-forward policy')
        else:
            require(protocol['image_arrival_forwards'] == 438 and protocol['old_graphs_or_text_embeddings_reused'] is False,
                    'Original retrieval memory policy differs')
        self.encoding_lengths(protocol, inputs, memories, captions)
        self.section('memory_graphs', {'domains': records, 'memory_leaves': sum(r['memory_leaves'] for r in records),
                                      'masters': sum(r['masters'] for r in records), 'old_caption_reuse': 0,
                                      'per_image_full_visual_vectors_saved': 0, 'history_image_copies_in_memory': 0})
        return states

    def encoding_lengths(self, protocol, inputs, memories, captions):
        """Tokenize locally to verify reported BGE window behavior; no inference."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(inputs['text_encoder']['model_path'], local_files_only=True)
        questions = self.read(self.output / 'inputs/source_questions.jsonl', True)
        audits = index(protocol['encoding_audits'], 'task')
        require(set(audits) == {m['task'] for m in memories}, 'Encoding domain coverage differs')
        recovery = 'original_strict_stage_signature' in protocol
        affected, historical_affected = [], []
        if recovery:
            require(protocol['long_text_policy'] == protocol['contract']['long_text_policy'] and
                    protocol['long_text_policy']['name'] == 'original_bge_model_limit_truncation_with_audit',
                    'Unknown or changed long-text recovery policy')
        for domain, audit in audits.items():
            rows = [m for m in memories if m['task'] == domain]
            qs = [q for q in questions if q['task'] == domain]
            texts = {'leaf': [m['dialogue'] + MEMORY_SEPARATOR + captions[m['image_sha256']]['caption'] for m in rows],
                     'query': []}
            ids = {'leaf': [m['id'] for m in rows], 'query': []}
            for q in qs:
                image_captions = [captions[im['sha256']]['caption'] for im in q['query_images']]
                query = q['question'] + (QUERY_SEPARATOR + ' '.join(image_captions) if image_captions else '')
                texts['query'].extend(inputs['query_prefix'] + text for text in [query, *[
                    query + '\nCandidate answer: ' + q['candidate_statements'][key] for key in 'ABCD']])
                ids['query'].extend(q['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D'))
            for kind in ('leaf', 'query'):
                info = audit[kind]
                lengths = [len(tokens) for tokens in tokenizer(texts[kind], truncation=False)['input_ids']]
                require(info['count'] == len(lengths) and info['token_limit'] == 512 and info['max_tokens'] == max(lengths),
                        f'Independent BGE token counts differ: {domain}/{kind}')
                expected = [{'id': identity, 'field': 'dialogue+caption' if kind == 'leaf' else 'query',
                             'original_tokens': length, 'encoded_tokens': 512, 'retained_limit': 512,
                             'dropped_tokens': length - 512}
                            for identity, length in zip(ids[kind], lengths) if length > 512]
                if recovery:
                    require(info['affected_inputs'] == expected and info['truncated_inputs'] == len(expected),
                            f'Reported BGE truncation differs: {domain}/{kind}')
                    require(info['batch_size'] == 16 and info['truncation_side'] == tokenizer.truncation_side,
                            'Recovered encoder batch/truncation settings differ')
                else:
                    require(not expected and info['truncated_inputs'] == 0, 'Unexpected strict-stage BGE truncation')
                affected.extend(expected)
            require(audit['candidate']['truncated_inputs'] == 0 and
                    audit['candidate']['max_tokens'] <= audit['candidate']['token_limit'], 'CLIP candidate truncation')
            reference = self.read(Path(inputs['basis_by_domain'][domain]['protocol_path']))
            old = reference.get('bge_encoding', {})
            if old:
                require(old['token_limit'] == 512 and old['batch_size'] == 16, 'Historical BGE encoder window differs')
                historical_affected.extend(old['truncated_inputs'])
        if recovery:
            require(historical_affected, 'Missing historical proof of original BGE truncation')
        self.section('text_length_policy', {
            'policy': protocol.get('long_text_policy', {'name': 'strict_no_overflow'}),
            'independently_tokenized_inputs': sum(a[k]['count'] for a in audits.values() for k in ('leaf', 'query')),
            'affected_inputs': affected, 'affected_input_count': len(affected),
            'historical_qwen_truncated_inputs': historical_affected,
            'reader_full_text_preserved': True, 'new_chunk_aggregation_applied': False,
            'interpretation': ('The compatibility extension restores the historical BGE model-window truncation; '
                               'it adds an explicit audit and does not introduce chunk-mean embeddings.') if recovery
                              else 'No BGE input exceeds the model window.',
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not (args.output / 'summary.json').exists():
        print(json.dumps({'audit_status': 'waiting_for_complete_outputs', 'output': str(args.output.resolve())}))
        return
    audit = Audit(args.output)
    try:
        report = audit.run()
    except (AuditError, KeyError, OSError, TypeError, ValueError) as error:
        report = audit.report
        report['audit_status'] = 'failed'
        report['errors'].append(f'{type(error).__name__}: {error}')
    target = args.output / 'independent_audit.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('audit_status', 'output', 'completed_sections', 'errors')}, ensure_ascii=False))
    if report['audit_status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
