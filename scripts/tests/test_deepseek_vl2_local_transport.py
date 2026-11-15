"""Local VL2 protocol boundary tests; no network, model, or GPU is used."""
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import deepseek_vl2_local_transport as transport


class DeepSeekVL2LocalTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime_manifest = self.root / 'runtime_manifest.json'
        self.runtime_manifest.write_text(json.dumps({
            'source_sha256': {}, 'fixture': 'offline-unit-test-runtime',
        }))
        manifest_patch = patch.object(transport, 'RUNTIME_MANIFEST', self.runtime_manifest)
        manifest_patch.start()
        self.addCleanup(manifest_patch.stop)
        # A valid one-pixel PNG keeps the test independent of image libraries.
        self.pixels = base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8'
            '/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')
        self.image = self.root / 'query.png'
        self.image.write_bytes(self.pixels)
        self.payload = {
            'record_id': 'internal-record-id',
            'user_text': 'Describe the original image.',
            'query_images': [{'path': str(self.image),
                              'sha256': hashlib.sha256(self.pixels).hexdigest()}],
        }
        self.config = transport.load_config()
        self.request = transport.build_request(self.payload, max_output_tokens=16)
        # Repeated image IDs represent processor-expanded visual positions.
        self.prompt_ids = [1, 100003, 100003, 100003, 100003, 10, 11, 2]

    def tokenization(self, tokens=None):
        tokens = list(self.prompt_ids if tokens is None else tokens)
        return {'count': len(tokens), 'tokens': tokens,
                'max_model_len': transport.CONTEXT_LIMIT}

    def response(self, tokens=None):
        tokens = list(self.prompt_ids if tokens is None else tokens)
        return {
            'id': 'local-vl2-response-id', 'model': transport.MODEL,
            'system_fingerprint': 'local-runtime-fingerprint',
            'prompt_token_ids': tokens,
            'usage': {'prompt_tokens': len(tokens), 'completion_tokens': 2,
                      'total_tokens': len(tokens) + 2},
            'choices': [{'index': 0, 'finish_reason': 'stop', 'stop_reason': 100001,
                         'token_ids': [317, 100001],
                         'message': {'role': 'assistant', 'content': 'A',
                                     'refusal': None}}],
        }

    def test_config_is_pinned_loopback_and_ignores_credentials(self):
        environment = {
            'OPENAI_API_KEY': 'SECRET_OPENAI', 'CODEX_API_KEY': 'SECRET_CODEX',
            'DEEPSEEK_API_KEY': 'SECRET_DEEPSEEK',
            'OPENAI_BASE_URL': 'https://external.invalid/v1',
            'OPENAI_MODEL': 'other-model',
        }
        with patch.dict(os.environ, environment, clear=True):
            config = transport.load_config()
        self.assertEqual(config.model, 'deepseek-vl2-small')
        self.assertEqual(config.base_url, 'http://127.0.0.1:8312/v1')
        public = config.public_dict()
        self.assertEqual(public['credentials'], 'none')
        self.assertEqual(public['checkpoint_revision'], transport.REVISION)
        self.assertEqual(public['context_limit'], 4096)
        self.assertEqual(public['runtime_manifest'], str(self.runtime_manifest))
        self.assertEqual(public['runtime_manifest_sha256'],
                         hashlib.sha256(self.runtime_manifest.read_bytes()).hexdigest())
        self.assertFalse(hasattr(config, 'api_key'))
        self.assertNotIn('SECRET_', repr(config) + json.dumps(public))
        self.assertEqual(transport.load_config(base_url=transport.BASE_URL + '/'), config)
        with self.assertRaises(ValueError):
            transport.load_config(self.root / 'nonexistent-credential-file')
        for model in ('deepseek-chat', 'deepseek-vl2', 'deepseek-vl2-small-latest'):
            with self.subTest(model=model), self.assertRaises(ValueError):
                transport.load_config(model=model)

    def test_runtime_source_hash_is_checked_before_exposing_config(self):
        source = self.root / 'runtime_source.py'
        source.write_bytes(b'original runtime source')
        self.runtime_manifest.write_text(json.dumps({
            'source_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
            'fixture': 'offline-unit-test-runtime',
        }))
        self.assertEqual(self.config.public_dict()['runtime_manifest_sha256'],
                         hashlib.sha256(self.runtime_manifest.read_bytes()).hexdigest())
        source.write_bytes(b'mutated runtime source')
        with self.assertRaisesRegex(ValueError, 'Frozen VL2 runtime source changed'):
            self.config.public_dict()

    def test_changed_endpoint_rejected_before_any_post(self):
        endpoints = (
            'https://external.invalid/v1', 'http://external.invalid/v1',
            'http://localhost:8312/v1', 'http://127.0.0.1:8313/v1',
            transport.BASE_URL + '?destination=external.invalid',
            'http://user:password@127.0.0.1:8312/v1',
        )
        with patch.object(transport, '_post') as post:
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    with self.assertRaises(ValueError):
                        transport.load_config(base_url=endpoint)
                    with self.assertRaises(ValueError):
                        transport.chat_request(transport.LocalConfig(base_url=endpoint), self.request)
            post.assert_not_called()

    def test_request_preserves_original_image_and_prompt(self):
        self.assertEqual(self.request['model'], transport.MODEL)
        self.assertEqual(self.request['seed'], 42)
        self.assertEqual(self.request['temperature'], 0)
        self.assertEqual(self.request['max_tokens'], 16)
        self.assertEqual(self.request['n'], 1)
        self.assertIs(self.request['stream'], False)
        self.assertIs(self.request['return_token_ids'], True)
        self.assertEqual([m['role'] for m in self.request['messages']], ['user'])
        content = self.request['messages'][0]['content']
        self.assertEqual([part['type'] for part in content], ['image_url', 'text'])
        image = content[0]['image_url']
        self.assertEqual(image['detail'], 'high')
        self.assertTrue(image['url'].startswith('data:image/png;base64,'))
        self.assertEqual(base64.b64decode(image['url'].split(',', 1)[1]), self.pixels)
        self.assertEqual(content[-1]['text'], self.payload['user_text'])
        wire = json.dumps(self.request)
        self.assertNotIn(str(self.image), wire)
        self.assertNotIn(self.payload['record_id'], wire)
        for field in ('store', 'api_key', 'tools', 'enable_search', 'gold'):
            self.assertNotIn(field, self.request)

    def test_optional_system_and_text_only_prompts_are_preserved(self):
        request = transport.build_request({**self.payload, 'system': 'Choose one option.'})
        self.assertEqual(request['messages'][0], {'role': 'system', 'content': 'Choose one option.'})
        self.assertEqual(request['messages'][1:], self.request['messages'])
        plain = transport.build_request({'user_text': 'Original question.'})
        self.assertEqual(plain['messages'], [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Original question.'}]}])
        self.assertEqual(plain['max_tokens'], 1024)
        with patch.object(transport, '_post', return_value=self.tokenization([1, 25, 2])):
            self.assertEqual(transport.preflight(self.config, plain)['count'], 3)

    def test_mutated_original_pixels_fail_sha_verification(self):
        self.image.write_bytes(self.pixels + b'mutated')
        with patch.object(transport, '_post') as post:
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                transport.build_request(self.payload)
            post.assert_not_called()

    def test_labels_and_extra_image_metadata_are_rejected(self):
        for field in ('gold', 'gold_choice', 'answer', 'correct_choice', 'target'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                transport.build_request({**self.payload, field: 'PRIVATE_GOLD'})
        nested = dict(self.payload['query_images'][0], gold='PRIVATE_GOLD')
        with self.assertRaises(ValueError):
            transport.build_request({**self.payload, 'query_images': [nested]})
        with self.assertRaises(ValueError):
            transport.build_request({**self.payload, 'query_images': self.payload['query_images'] * 2})

    def test_fixed_generation_settings_are_required_before_post(self):
        changes = ({'seed': 43}, {'seed': None}, {'temperature': 0.1},
                   {'temperature': float('nan')}, {'return_token_ids': False},
                   {'n': 2}, {'stream': True}, {'gold': 'PRIVATE_GOLD'},
                   {'tools': []}, {'model': 'other-model'})
        with patch.object(transport, '_post') as post:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    transport.chat_request(self.config, {**self.request, **change})
            post.assert_not_called()

    def test_invalid_output_budget_is_rejected_before_post(self):
        with patch.object(transport, '_post') as post:
            for budget in (0, -1, True, 1.5, None):
                with self.subTest(budget=budget):
                    with self.assertRaises(ValueError):
                        transport.build_request(self.payload, max_output_tokens=budget)
                    with self.assertRaises(ValueError):
                        transport.chat_request(self.config, {**self.request, 'max_tokens': budget})
            post.assert_not_called()

    def test_preflight_sends_identical_messages_with_generation_prompt(self):
        measured = self.tokenization()
        with patch.object(transport, '_post', return_value=measured) as post:
            result = transport.preflight(self.config, self.request, timeout=23)
        post.assert_called_once_with(
            'http://127.0.0.1:8312/tokenize',
            {'model': transport.MODEL, 'messages': self.request['messages'],
             'add_generation_prompt': True}, 23)
        self.assertEqual(result, measured)

    def test_context_overflow_stops_before_chat_without_truncating(self):
        count = transport.CONTEXT_LIMIT - self.request['max_tokens'] + 1
        measured = self.tokenization([100003, 100003] + [7] * (count - 2))
        original = deepcopy(self.request)
        with patch.object(transport, '_post', return_value=measured) as post:
            with self.assertRaises(transport.APIError) as caught:
                transport.chat_request(self.config, self.request)
        self.assertEqual(caught.exception.kind, 'context_budget')
        self.assertEqual(caught.exception.code, 'context_length_exceeded')
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(self.request, original)
        post.assert_called_once()
        self.assertTrue(post.call_args.args[0].endswith('/tokenize'))

    def test_exact_context_boundary_is_accepted(self):
        count = transport.CONTEXT_LIMIT - self.request['max_tokens']
        ids = [100003, 100003] + [7] * (count - 2)
        with patch.object(transport, '_post', side_effect=[self.tokenization(ids), self.response(ids)]) as post:
            result = transport.chat_request(self.config, self.request)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result['input_tokens'] + self.request['max_tokens'], 4096)
        self.assertIs(result['input_truncated'], False)

    def test_preflight_rejects_missing_multimodal_expansion(self):
        for ids in ([1, 20, 2], [1, 100003, 2]):
            with self.subTest(ids=ids):
                with patch.object(transport, '_post', return_value=self.tokenization(ids)) as post:
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')
                post.assert_called_once()

    def test_preflight_rejects_bad_counts_limits_and_token_types(self):
        bad = (
            {'count': 0}, {'count': True}, {'count': 8.0}, {'count': 9},
            {'max_model_len': 8192}, {'tokens': None},
            {'tokens': [1, 100003, 100003, 100003, 100003, '10', 11, 2]},
            {'tokens': [True, 100003, 100003, 100003, 100003, 10, 11, 2]},
            {'tokens': [-1, 100003, 100003, 100003, 100003, 10, 11, 2]},
        )
        for change in bad:
            with self.subTest(change=change):
                with patch.object(transport, '_post', return_value={**self.tokenization(), **change}) as post:
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')
                post.assert_called_once()

    def test_nonpositive_or_nonfinite_timeout_never_posts(self):
        with patch.object(transport, '_post') as post:
            for timeout in (0, -1, float('nan'), float('inf')):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    transport.chat_request(self.config, self.request, timeout=timeout)
            post.assert_not_called()

    def test_response_keeps_original_token_ids_usage_and_request_provenance(self):
        measured, response = self.tokenization(), self.response()
        original = deepcopy(self.request)
        with patch.object(transport, '_post', side_effect=[measured, response]) as post:
            result = transport.chat_request(self.config, self.request, timeout=23)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[1].args,
                         (transport.BASE_URL + '/chat/completions', original, 23))
        self.assertEqual(self.request, original)
        self.assertEqual(result['raw_response'], 'A')
        self.assertEqual(result['prompt_token_ids'], response['prompt_token_ids'])
        self.assertEqual(result['generated_token_ids'], response['choices'][0]['token_ids'])
        self.assertEqual(result['usage'], response['usage'])
        self.assertEqual(result['input_tokens'], measured['count'])
        self.assertEqual(result['expanded_image_tokens'], 4)
        self.assertEqual(result['response_id'], response['id'])
        self.assertEqual(result['returned_model'], transport.MODEL)
        self.assertEqual(result['checkpoint_revision'], transport.REVISION)
        self.assertEqual(result['stop_reason'], 100001)
        self.assertEqual(result['request_sha256'], transport.request_sha256(original))
        self.assertEqual((result['attempts'], result['retry_count']), (1, 0))
        self.assertIs(result['input_truncated'], False)
        self.assertIs(result['reasoning_content_present'], False)

    def test_response_model_identity_and_prompt_ids_must_match_preflight(self):
        responses = []
        for field, value in (('model', 'deepseek-vl2'), ('id', ''),
                             ('prompt_token_ids', list(reversed(self.prompt_ids))),
                             ('prompt_token_ids', [True] + self.prompt_ids[1:]),
                             ('prompt_token_ids', [1.0] + self.prompt_ids[1:])):
            responses.append({**self.response(), field: value})
        for response in responses:
            with self.subTest(response=response):
                with patch.object(transport, '_post', side_effect=[self.tokenization(), response]):
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')
                self.assertFalse(caught.exception.retryable)

    def test_response_usage_must_match_original_prompt_and_generated_ids(self):
        changes = (
            {'prompt_tokens': 7}, {'prompt_tokens': 8.0},
            {'completion_tokens': 1}, {'completion_tokens': 2.0},
            {'total_tokens': 9}, {'total_tokens': 10.0},
        )
        for change in changes:
            response = self.response()
            response['usage'].update(change)
            with self.subTest(change=change):
                with patch.object(transport, '_post', side_effect=[self.tokenization(), response]):
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')

    def test_response_rejects_invalid_or_overbudget_generated_token_ids(self):
        for ids in ([317, '100001'], [317, True], [317, -1], [317] * 17):
            response = self.response()
            response['choices'][0]['token_ids'] = ids
            response['usage']['completion_tokens'] = len(ids)
            response['usage']['total_tokens'] = len(self.prompt_ids) + len(ids)
            with self.subTest(ids=ids):
                with patch.object(transport, '_post', side_effect=[self.tokenization(), response]):
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')

    def test_response_rejects_unexpected_reasoning_or_finish_reason(self):
        for change in ('reasoning', 'finish_reason', 'missing_ids'):
            response = self.response()
            choice = response['choices'][0]
            if change == 'reasoning':
                choice['message']['reasoning_content'] = 'Unexpected reasoning.'
            elif change == 'finish_reason':
                choice['finish_reason'] = 'tool_calls'
            else:
                del choice['token_ids']
            with self.subTest(change=change):
                with patch.object(transport, '_post', side_effect=[self.tokenization(), response]):
                    with self.assertRaises(transport.APIError) as caught:
                        transport.chat_request(self.config, self.request)
                self.assertEqual(caught.exception.kind, 'protocol')


if __name__ == '__main__':
    unittest.main()
