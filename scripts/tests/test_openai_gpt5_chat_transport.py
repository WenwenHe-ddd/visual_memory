"""Mocked GPT-5.4 schema, route, original-image, and secret-handling checks."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from scripts.evaluation.openai_gpt5_chat_transport import (
    APIConfig, APIError, MODEL, OFFICIAL_BASE_URL, build_request, chat_request, load_config,
)


class GPT54TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / '.env'
        self.env.write_text('OPENAI_API_KEY=OFFICIAL_FAKE_KEY\nOPENAI_MODEL=gpt-4o-mini\n'
                            'OPENAI_BASE_URL=https://api.openai.com/v1\n'
                            'CODEX_API_KEY=THIRD_PARTY_FAKE_KEY\n'
                            'CODEX_BASE_URL=https://third-party.example/v1\n')
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.config = load_config(self.env)
        self.request = build_request('System', 'Question', [])

    def test_alias_is_pinned_with_only_official_credentials(self):
        for model in ('gpt-5.4', MODEL, None):
            config = load_config(self.env, model=model)
            self.assertEqual(config.model, MODEL)
            self.assertEqual(config.base_url, OFFICIAL_BASE_URL)
            self.assertEqual(config.credential_variable, 'OPENAI_API_KEY')
            self.assertEqual(config.api_key, 'OFFICIAL_FAKE_KEY')
            safe = repr(config) + json.dumps(config.public_dict())
            self.assertNotIn('OFFICIAL_FAKE_KEY', safe)
            self.assertNotIn('THIRD_PARTY_FAKE_KEY', safe)
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'PROCESS_FAKE_KEY'}):
            self.assertEqual(load_config(self.env).api_key, 'PROCESS_FAKE_KEY')
        with self.assertRaises(ValueError):
            load_config(self.env, model='gpt-5.4-latest')

    def test_third_party_endpoint_and_codex_only_config_are_rejected(self):
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://third-party.example/v1'}):
            with self.assertRaisesRegex(ValueError, 'official OpenAI endpoint'):
                load_config(self.env)
        self.env.write_text('CODEX_API_KEY=THIRD_PARTY_FAKE_KEY\nCODEX_BASE_URL=https://third-party.example/v1\n')
        with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
            load_config(self.env)
        for config in (APIConfig(MODEL, 'https://third-party.example/v1', 'TEST'),
                       APIConfig(MODEL, OFFICIAL_BASE_URL, 'TEST', 'CODEX_API_KEY')):
            with patch('urllib.request.build_opener') as opener:
                with self.assertRaises(ValueError):
                    chat_request(config, self.request)
                opener.assert_not_called()

    def test_wire_schema_omits_seed_and_legacy_token_limit(self):
        self.assertEqual(self.request['model'], MODEL)
        self.assertEqual(self.request['max_completion_tokens'], 16)
        self.assertEqual(self.request['reasoning_effort'], 'none')
        self.assertEqual(self.request['temperature'], 0)
        self.assertNotIn('seed', self.request)
        self.assertNotIn('max_tokens', self.request)
        self.assertFalse(self.request['store'])
        for kwargs in ({'seed': 42}, {'reasoning_effort': 'low'}, {'temperature': 1},
                       {'max_completion_tokens': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_request('S', 'U', [], **kwargs)
        for key, value in (('seed', 42), ('max_tokens', 16), ('gold_choice', 'A')):
            with self.subTest(key=key), patch('urllib.request.build_opener') as opener:
                with self.assertRaises(ValueError):
                    chat_request(self.config, {**self.request, key: value})
                opener.assert_not_called()

    def test_query_image_bytes_are_exact_and_hash_verified(self):
        data = b'\xff\xd8\xfforiginal JPEG bytes'
        path = self.root / 'query.jpg'
        path.write_bytes(data)
        image = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
        request = build_request('S', 'U', [image], model='gpt-5.4')
        content = request['messages'][1]['content']
        self.assertEqual([p['type'] for p in content], ['image_url', 'text'])
        self.assertEqual(base64.b64decode(content[0]['image_url']['url'].split(',', 1)[1]), data)
        self.assertEqual(content[0]['image_url']['detail'], 'high')
        path.write_bytes(data + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            build_request('S', 'U', [image])

    def test_one_request_normalizes_reasoning_usage_without_seed(self):
        payload = {'id': 'test-response', 'model': MODEL, 'system_fingerprint': None,
                   'usage': {'prompt_tokens': 25, 'completion_tokens': 1, 'total_tokens': 26,
                             'completion_tokens_details': {'reasoning_tokens': 0}},
                   'choices': [{'finish_reason': 'stop', 'message': {'content': 'A', 'refusal': None}}]}
        response, opener = MagicMock(), MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.headers.get.return_value = 'test-request'
        opener.open.return_value.__enter__.return_value = response
        with patch('urllib.request.build_opener', return_value=opener) as factory:
            result = chat_request(self.config, self.request, timeout=21)
        opener.open.assert_called_once()
        sent = opener.open.call_args.args[0]
        self.assertEqual(sent.full_url, OFFICIAL_BASE_URL + '/chat/completions')
        self.assertEqual(json.loads(sent.data), self.request)
        self.assertNotIn('seed', json.loads(sent.data))
        self.assertEqual(result['raw_response'], 'A')
        self.assertEqual(result['reasoning_tokens'], 0)
        self.assertEqual(result['usage'], payload['usage'])
        self.assertIsNone(factory.call_args.args[0].redirect_request(None, None, 302, '', {}, 'https://other.test'))
        self.assertNotIn('OFFICIAL_FAKE_KEY', json.dumps(result))

    def test_errors_are_sanitized_and_never_retried_inside_transport(self):
        for status, code, retryable in ((429, 'insufficient_quota', False),
                                        (429, 'rate_limit_exceeded', True),
                                        (503, 'server_error', True), (401, 'invalid_api_key', False),
                                        (302, None, False)):
            with self.subTest(status=status, code=code):
                payload = json.dumps({'error': {'code': code, 'message': 'OFFICIAL_FAKE_KEY PRIVATE_PROMPT'}}).encode()
                failure = urllib.error.HTTPError(OFFICIAL_BASE_URL, status, 'OFFICIAL_FAKE_KEY', {}, io.BytesIO(payload))
                opener = MagicMock()
                opener.open.side_effect = failure
                with patch('urllib.request.build_opener', return_value=opener):
                    with self.assertRaises(APIError) as raised:
                        chat_request(self.config, self.request)
                opener.open.assert_called_once()
                self.assertEqual(raised.exception.retryable, retryable)
                safe = str(raised.exception) + repr(raised.exception) + json.dumps(vars(raised.exception))
                self.assertNotIn('OFFICIAL_FAKE_KEY', safe)
                self.assertNotIn('PRIVATE_PROMPT', safe)


if __name__ == '__main__':
    unittest.main()
