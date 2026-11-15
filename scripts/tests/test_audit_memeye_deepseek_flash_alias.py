"""Adversarial fixtures for the independent DeepSeek Flash alias auditor."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis.audit_memeye_deepseek_flash_alias import (
    Audit, AuditError, BASE_URL, CREDENTIAL_VARIABLE, CONDITIONS, MODEL, PROVIDER,
    aggregate, digest, recall, sha, validate_profile, validate_response, wire_request,
)


class IndependentDeepSeekFlashAliasAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'original.png'
        self.image.write_bytes(b'\x89PNG\r\n\x1a\noriginal pixels')
        self.payload = {'user_text': 'Describe this image.',
                        'query_images': [{'path': str(self.image), 'sha256': sha(self.image)}]}
        self.protocol = {'run_id': 'this-run', 'experiment_signature': 'this-signature',
                         'contract': {'model': MODEL, 'max_attempts': 3, 'image_detail': 'high',
                                      'scope': 'all_original_qwen_roles', 'provider': PROVIDER,
                                      'api_config': {'model': MODEL, 'api_type': 'chat_completions',
                                                     'base_url': BASE_URL, 'credential_variable': CREDENTIAL_VARIABLE},
                                      'caption_max_output_tokens': 1024, 'reader_max_output_tokens': 16,
                                      'caption_budget_policy': 'fixed_1024_from_start',
                                      'model_identity': {'kind': 'provider_alias', 'requested_model': MODEL,
                                                         'resolved_revision': None, 'distinct_weights_vs_1017': None},
                                      'temperature': 0, 'enable_thinking': False, 'web_search': False,
                                      'seed': None, 'reasoning_effort': None}}

    def fixture(self):
        directory = self.root / 'caption_generation'
        directory.mkdir()
        records = []
        for rid in ('first', 'same_image'):
            records.append({'record_id': rid, **self.payload,
                            'payload_sha256': digest(self.payload),
                            'request_sha256': digest(wire_request(self.payload, self.protocol['contract']['model'], 1024, 'high'))})
        request_file = directory / 'requests.jsonl'
        request_file.write_text(''.join(json.dumps(row) + '\n' for row in records))
        metadata = {'stage': 'caption_generation', 'run_id': self.protocol['run_id'],
                    'experiment_signature': self.protocol['experiment_signature'],
                    'model': self.protocol['contract']['model'], 'max_output_tokens': 1024,
                    'source_sha256': {}, 'requests_sha256': sha(request_file),
                    'logical_records': 2, 'unique_requests': 1}
        (directory / 'protocol.json').write_text(json.dumps(metadata))
        response = {'record_id': 'first', 'request_sha256': records[0]['request_sha256'],
                    'payload_sha256': records[0]['payload_sha256'], 'run_id': self.protocol['run_id'],
                    'experiment_signature': self.protocol['experiment_signature'],
                    'requested_model': self.protocol['contract']['model'],
                    'returned_model': self.protocol['contract']['model'], 'status': 'ok',
                    'finish_reason': 'stop', 'historical_response_reuse': False,
                    'raw_response': 'A public image.', 'refusal': None, 'response_id': 'provider-1',
                    'execution_id': 'execution-1', 'attempts': 1, 'retry_count': 0,
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 4, 'total_tokens': 14,
                              'prompt_tokens_details': {'image_tokens': 8},
                              'completion_tokens_details': {'reasoning_tokens': 0}}, 'reasoning_tokens': 0,
                    'reasoning_content_present': False, 'enable_thinking': False}
        self.write_response(response)
        (directory / 'api_attempts.jsonl').write_text(json.dumps({
            'record_id': 'first', 'request_sha256': records[0]['request_sha256'],
            'stage': 'caption_generation', 'execution_id': 'execution-1', 'attempt': 1,
            'status': 'ok', 'response_id': 'provider-1'}) + '\n')
        return records, response

    def write_response(self, response):
        (self.root / 'caption_generation/fresh_responses.jsonl').write_text(json.dumps(response) + '\n')

    def test_independent_wire_has_user_only_image_first_and_rejects_gold(self):
        body = wire_request(self.payload, self.protocol['contract']['model'], 1024, 'high')
        self.assertEqual([m['role'] for m in body['messages']], ['user'])
        self.assertEqual([p['type'] for p in body['messages'][0]['content']], ['image_url', 'text'])
        self.assertIs(body['enable_thinking'], False)
        self.assertEqual(body['max_tokens'], 1024)
        self.assertNotIn('seed', body)
        self.assertNotIn('store', body)
        self.assertNotIn('reasoning_effort', body)
        self.assertNotIn('path', body['messages'][0]['content'][0]['image_url'])
        with self.assertRaisesRegex(AuditError, 'Unexpected outbound'):
            wire_request({**self.payload, 'gold_choice': 'B'}, self.protocol['contract']['model'], 1024, 'high')
        self.image.write_bytes(b'\x89PNG\r\n\x1a\nchanged pixels')
        with self.assertRaisesRegex(AuditError, 'image bytes changed'):
            wire_request(self.payload, self.protocol['contract']['model'], 1024, 'high')

    def test_canonical_same_run_share_is_audited(self):
        _, response = self.fixture()
        expected = {rid: self.payload for rid in ('first', 'same_image')}
        auditor = Audit(self.root)
        auditor.stage('caption_generation', expected, self.protocol)
        self.assertEqual(auditor.report['caption_generation']['same_run_shared_records'], 1)
        # Same request hashes alone cannot establish canonical fresh provenance.
        response['record_id'] = 'same_image'
        self.write_response(response)
        with self.assertRaisesRegex(AuditError, 'record_id mismatch'):
            Audit(self.root).stage('caption_generation', expected, self.protocol)

    def test_historical_wrong_run_and_nonzero_reasoning_rejected(self):
        records, response = self.fixture()
        for key, value, expected in [('run_id', 'another-run', 'run_id mismatch'),
                                     ('historical_response_reuse', True, 'Historical API')]:
            changed = {**response, key: value}
            with self.assertRaisesRegex(AuditError, expected):
                validate_response(changed, records[0], self.protocol, 'caption_generation')
        changed = copy.deepcopy(response)
        changed['reasoning_tokens'] = 2
        changed['usage']['completion_tokens_details']['reasoning_tokens'] = 2
        with self.assertRaisesRegex(AuditError, 'nonzero reasoning'):
            validate_response(changed, records[0], self.protocol, 'caption_generation')

    def test_provider_profile_and_model_mixing_rejected(self):
        contract = self.protocol['contract']
        validate_profile(contract)
        for key, value in [('base_url', 'https://api.deepseek.com/v1'),
                           ('credential_variable', 'OPENAI_API_KEY'),
                           ('model', 'deepseek-v4.1-flash-1017')]:
            changed = copy.deepcopy(contract)
            changed['api_config'][key] = value
            with self.assertRaises(AuditError):
                validate_profile(changed)
        for key, value in [('provider', 'OpenAI'), ('model', 'gpt-5.4-2026-03-05'),
                           ('enable_thinking', True), ('web_search', True), ('seed', 42),
                           ('reasoning_effort', 'none')]:
            with self.assertRaises(AuditError):
                validate_profile({**contract, key: value})
        records, response = self.fixture()
        for key, value in [('requested_model', 'deepseek-v4.1-flash-1017'),
                           ('returned_model', 'qwen3-vl'), ('returned_model', 'gpt-5.4-2026-03-05')]:
            with self.assertRaisesRegex(AuditError, key + ' mismatch'):
                validate_response({**response, key: value}, records[0], self.protocol, 'caption_generation')
        with self.assertRaisesRegex(AuditError, 'Unapproved model alias'):
            wire_request(self.payload, 'deepseek-v4.1-flash-1017', 1024, 'high')

    def test_thinking_content_or_missing_disabled_provenance_rejected(self):
        records, response = self.fixture()
        for key, value in [('reasoning_content', 'Let me think.'), ('reasoning_content_present', True),
                           ('enable_thinking', True)]:
            with self.assertRaises(AuditError):
                validate_response({**response, key: value}, records[0], self.protocol, 'caption_generation')
        for key in ('reasoning_content_present', 'enable_thinking'):
            changed = dict(response)
            changed.pop(key)
            with self.assertRaisesRegex(AuditError, 'thinking response provenance'):
                validate_response(changed, records[0], self.protocol, 'caption_generation')

    def test_image_usage_requires_actual_positive_provider_image_tokens(self):
        records, response = self.fixture()
        for value in (None, 0, -1, '8'):
            changed = copy.deepcopy(response)
            changed['usage']['prompt_tokens_details']['image_tokens'] = value
            with self.assertRaisesRegex(AuditError, 'image-token usage'):
                validate_response(changed, records[0], self.protocol, 'caption_generation')

    def test_alias_identity_cannot_claim_a_confirmed_or_distinct_revision(self):
        c = self.protocol['contract']
        validate_profile(c)
        audit = Audit(self.root)
        audit.identity_and_budget(self.protocol)
        identity = audit.report['model_identity']
        self.assertIsNone(identity['resolved_revision'])
        self.assertIsNone(identity['distinct_weights_vs_1017'])
        self.assertEqual(audit.report['caption_budget']['policy'], 'fixed_1024_from_start')
        self.assertFalse(audit.report['caption_budget']['amended'])
        for key, value in [('kind', 'pinned_snapshot'), ('requested_model', 'deepseek-v4.1-flash-1017'),
                           ('resolved_revision', '1017'), ('distinct_weights_vs_1017', True),
                           ('distinct_weights_vs_1017', False)]:
            bad = copy.deepcopy(c)
            bad['model_identity'][key] = value
            with self.assertRaisesRegex(AuditError, 'explicitly unconfirmed'):
                validate_profile(bad)
        bad = copy.deepcopy(c)
        bad['model_identity']['unverified_extra'] = 'value'
        with self.assertRaisesRegex(AuditError, 'explicitly unconfirmed'):
            validate_profile(bad)

    def test_alias_budget_is_1024_from_start_without_old_run_amendment(self):
        c = self.protocol['contract']
        for key, value in [('caption_max_output_tokens', 256), ('reader_max_output_tokens', 32),
                           ('caption_budget_policy', 'restarted_1024'), ('caption_budget_amendment', {})]:
            with self.assertRaises(AuditError):
                validate_profile({**c, key: value})
        records, _ = self.fixture()
        self.assertEqual(records[0]['request_sha256'], digest(wire_request(self.payload, MODEL, 1024, 'high')))
        self.assertNotEqual(records[0]['request_sha256'], digest(wire_request(self.payload, MODEL, 256, 'high')))

    def test_missing_reasoning_usage_stays_unknown(self):
        _, response = self.fixture()
        response['reasoning_tokens'] = None
        response['usage'].pop('completion_tokens_details')
        self.write_response(response)
        auditor = Audit(self.root)
        auditor.stage('caption_generation', {rid: self.payload for rid in ('first', 'same_image')}, self.protocol)
        counts = auditor.report['caption_generation']['reasoning_usage']
        self.assertEqual((counts['unreported'], counts['reported_zero']), (1, 0))

    def test_saved_request_gold_field_rejected_even_after_file_hash_rewritten(self):
        records, _ = self.fixture()
        records[0]['gold_ids'] = ['target']
        directory = self.root / 'caption_generation'
        path = directory / 'requests.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        meta = json.loads((directory / 'protocol.json').read_text())
        meta['requests_sha256'] = sha(path)
        (directory / 'protocol.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(AuditError, 'unexpected/gold request fields'):
            Audit(self.root).stage('caption_generation', {rid: self.payload for rid in ('first', 'same_image')}, self.protocol)

    def test_micro_macro_denominators_and_paired_statistics(self):
        gold = {'q1': {'gold_ids': ['a']}, 'q2': {'gold_ids': ['a', 'b', 'c']}}
        rows = [{'question_id': q, 'conditions': [{'name': name, 'top5_memory_ids': ['a', 'd', 'e', 'f', 'g']}
                                                 for name in CONDITIONS]} for q in gold]
        result = recall(rows, gold)
        self.assertEqual(result['gold_references'], 4)
        self.assertEqual(result[CONDITIONS[0]]['gold_recall'], 0.5)
        self.assertAlmostEqual(result[CONDITIONS[0]]['macro_recall_at_5'], 2 / 3)
        pairs = [{'baseline_correct': False, 'expanded_correct': True},
                 {'baseline_correct': True, 'expanded_correct': False},
                 {'baseline_correct': True, 'expanded_correct': True}]
        stats = aggregate(pairs)
        self.assertEqual((stats['baseline_correct'], stats['expanded_correct'], stats['wrong_to_correct'],
                          stats['correct_to_wrong'], stats['mcnemar_exact_p']), (2, 2, 1, 1, 1))

    def test_legacy_truncation_is_independently_counted_and_not_called_chunking(self):
        class Tokenizer:
            truncation_side = 'right'

            def __call__(self, texts, truncation=False):
                self.assert_no_truncation = truncation is False
                return {'input_ids': [[0] * (len(text.split()) + 2) for text in texts]}

        tokenizer = Tokenizer()
        memory = {'id': 'D/M', 'task': 'D', 'dialogue': 'd ' * 570, 'image_sha256': 'image'}
        question = {'id': 'D/Q', 'task': 'D', 'question': 'q', 'query_images': [],
                    'candidate_statements': dict(zip('ABCD', 'abcd'))}
        affected = {'id': 'D/M', 'field': 'dialogue+caption', 'original_tokens': 575,
                    'encoded_tokens': 512, 'retained_limit': 512, 'dropped_tokens': 63}
        policy = {'name': 'original_bge_model_limit_truncation_with_audit'}
        info = {'token_limit': 512, 'batch_size': 16, 'truncation_side': 'right'}
        protocol = {'original_strict_stage_signature': 'old', 'long_text_policy': policy,
                    'contract': {'long_text_policy': policy}, 'encoding_audits': [{
                        'task': 'D', 'leaf': {**info, 'count': 1, 'max_tokens': 575, 'truncated_inputs': 1,
                                             'affected_inputs': [affected]},
                        'query': {**info, 'count': 5, 'max_tokens': 7, 'truncated_inputs': 0, 'affected_inputs': []},
                        'candidate': {'truncated_inputs': 0, 'max_tokens': 3, 'token_limit': 77}}]}
        inputs = {'text_encoder': {'model_path': '/unused'}, 'query_prefix': 'p ',
                  'basis_by_domain': {'D': {'protocol_path': '/reference'}}}
        reference = {'bge_encoding': {'token_limit': 512, 'batch_size': 16,
                                     'truncated_inputs': [{'id': 'D/M#image1', 'original_tokens': 577, 'retained_limit': 512}]}}
        auditor = Audit(self.root)
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer), \
                patch.object(auditor, 'read', side_effect=lambda path, jsonl=False: [question] if jsonl else reference):
            auditor.encoding_lengths(protocol, inputs, [memory], {'image': {'caption': 'caption'}})
            self.assertEqual(auditor.report['text_length_policy']['affected_inputs'], [affected])
            self.assertFalse(auditor.report['text_length_policy']['new_chunk_aggregation_applied'])
            protocol['encoding_audits'][0]['leaf']['affected_inputs'][0]['dropped_tokens'] = 0
            with self.assertRaisesRegex(AuditError, 'Reported BGE truncation differs'):
                auditor.encoding_lengths(protocol, inputs, [memory], {'image': {'caption': 'caption'}})


if __name__ == '__main__':
    unittest.main()
