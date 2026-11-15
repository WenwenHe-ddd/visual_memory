"""Undated API alias isolation, original pixels, and disabled-thinking checks."""
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
from scripts.evaluation.deepseek_flash_seed42_audited_transport import (
    APIConfig, APIError, DEEPSEEK_MODEL, DASHSCOPE_BASE_URL,
    build_request, canonical_model, chat_request, load_config, request_sha256,
)


class DeepSeekFlashSeed42AuditedTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = self.root / '.env'
        self.env.write_text('DEEPSEEKV4_API_KEY=FAKE_DASHSCOPE_SECRET\n'
                            'DEEPSEEKV4_BASE_URL=' + DASHSCOPE_BASE_URL + '\n'
                            'OPENAI_API_KEY=NEVER_USE_THIS\n')
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.data = b'\x89PNG\r\n\x1a\noriginal image bytes'
        self.image = self.root / 'original.png'
        self.image.write_bytes(self.data)
        self.payload = {'record_id': 'internal-image-id', 'user_text': 'Original caption prompt.',
                        'query_images': [{'path': str(self.image),
                                          'sha256': hashlib.sha256(self.data).hexdigest(),
                                          'mime_type': 'image/png'}]}

    def test_credential_and_endpoint_are_bound_without_fallback(self):
        config = load_config(self.env)
        self.assertEqual(config.credential_variable, 'DEEPSEEKV4_API_KEY')
        self.assertEqual(config.base_url, DASHSCOPE_BASE_URL)
        self.assertEqual(config.api_key, 'FAKE_DASHSCOPE_SECRET')
        self.assertNotIn(config.api_key, repr(config) + json.dumps(config.public_dict()))
        with patch.dict(os.environ, {'DEEPSEEKV4_API_KEY': 'PROCESS_SECRET'}):
            self.assertEqual(load_config(self.env).api_key, 'PROCESS_SECRET')
        for endpoint in ('https://api.deepseek.com/v1', 'https://api.openai.com/v1'):
            with self.assertRaises(ValueError):
                load_config(self.env, base_url=endpoint)
            with patch.dict(os.environ, {'DEEPSEEKV4_BASE_URL': endpoint}):
                with self.assertRaises(ValueError):
                    load_config(self.env)
        self.env.write_text('OPENAI_API_KEY=NEVER_USE_THIS\n')
        with self.assertRaisesRegex(ValueError, 'DEEPSEEKV4_API_KEY'):
            load_config(self.env)
        with self.assertRaises(ValueError):
            canonical_model('deepseek-chat')

    def test_caption_wire_exact_pixels_prompt_seed42_no_tools_or_search(self):
        request = build_request(self.payload, model=DEEPSEEK_MODEL, image_detail='high')
        self.assertEqual(set(request), {'model', 'messages', 'temperature', 'max_tokens',
                                        'n', 'stream', 'enable_thinking', 'seed'})
        self.assertEqual((request['temperature'], request['max_tokens'], request['n']), (0, 1024, 1))
        self.assertIs(request['enable_thinking'], False)
        self.assertIs(request['stream'], False)
        self.assertEqual(request['seed'], 42)
        self.assertEqual(len(request['messages']), 1)
        user = request['messages'][0]
        self.assertEqual(user['role'], 'user')
        self.assertEqual([part['type'] for part in user['content']], ['image_url', 'text'])
        visual = user['content'][0]['image_url']
        self.assertEqual(visual['detail'], 'high')
        self.assertEqual(base64.b64decode(visual['url'].split(',', 1)[1]), self.data)
        self.assertEqual(user['content'][-1]['text'], self.payload['user_text'])
        self.assertNotIn(str(self.image), json.dumps(request))
        self.assertNotIn('internal-image-id', json.dumps(request))

    def test_reader_system_and_image_channel_preserved(self):
        request = build_request({**self.payload, 'system': 'Original system.'},
                                model=DEEPSEEK_MODEL, max_output_tokens=16)
        self.assertEqual([m['role'] for m in request['messages']], ['system', 'user'])
        self.assertEqual(request['messages'][0]['content'], 'Original system.')
        self.assertEqual(request['max_tokens'], 16)
        self.assertEqual(request['seed'], 42)
        plain = build_request({'user_text': 'Text only'}, model=DEEPSEEK_MODEL)
        self.assertEqual(plain['messages'][0]['content'], [{'type': 'text', 'text': 'Text only'}])

    def test_gold_mime_and_pixel_mutations_rejected(self):
        with self.assertRaises(ValueError):
            build_request({**self.payload, 'gold_choice': 'A'}, model=DEEPSEEK_MODEL)
        changed = dict(self.payload['query_images'][0], mime_type='image/jpeg')
        with self.assertRaises(ValueError):
            build_request({**self.payload, 'query_images': [changed]}, model=DEEPSEEK_MODEL)
        self.image.write_bytes(self.data + b'mutation')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            build_request(self.payload, model=DEEPSEEK_MODEL)

    @staticmethod
    def mock_opener(model=DEEPSEEK_MODEL, reasoning=0, content=None,
                    response_id='actual-api-response', fingerprint=None):
        payload = {'id': response_id, 'model': model,
                   'choices': [{'finish_reason': 'stop', 'message': {'content': 'A complete caption.',
                                                                   'reasoning_content': content}}],
                   'usage': {'prompt_tokens': 50, 'completion_tokens': 5,
                             'completion_tokens_details': {'reasoning_tokens': reasoning}}}
        if fingerprint is not None:
            payload['system_fingerprint'] = fingerprint
        opener, response = MagicMock(), MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.headers.get.return_value = 'request-id'
        opener.open.return_value.__enter__.return_value = response
        return opener

    def test_success_records_actual_model_response_id_and_wire_hash(self):
        request = build_request(self.payload, model=DEEPSEEK_MODEL)
        opener = self.mock_opener()
        with patch('urllib.request.build_opener', return_value=opener) as factory:
            result = chat_request(load_config(self.env), request)
        sent = opener.open.call_args.args[0]
        self.assertEqual(sent.full_url, DASHSCOPE_BASE_URL + '/chat/completions')
        self.assertEqual(json.loads(sent.data), request)
        self.assertEqual(json.loads(sent.data)['seed'], 42)
        self.assertEqual(result['seed'], 42)
        self.assertEqual(result['response_id'], 'actual-api-response')
        self.assertEqual(result['request_sha256'], request_sha256(request))
        self.assertEqual(result['returned_model'], DEEPSEEK_MODEL)
        self.assertEqual(result['requested_model'], 'deepseek-v4.1-flash')
        self.assertEqual(result['model_identity_kind'], 'provider_alias')
        self.assertIsNone(result['resolved_model_revision'])
        self.assertIsNone(result['system_fingerprint'])
        self.assertIs(result['reasoning_content_present'], False)
        self.assertIs(result['enable_thinking'], False)
        self.assertIsNone(factory.call_args.args[0].redirect_request(None, None, 302, '', {}, 'https://other.test'))
        self.assertNotIn('FAKE_DASHSCOPE_SECRET', json.dumps(result))

    def test_wrong_model_or_reasoning_or_missing_response_identity_fails(self):
        for kwargs, expected in [({'model': 'deepseek-text-only'}, 'model_mismatch'),
                                 ({'model': 'deepseek-v4.1-flash-1017'}, 'model_mismatch'),
                                 ({'reasoning': 1}, 'unexpected_reasoning'),
                                 ({'content': 'Hidden reasoning'}, 'unexpected_reasoning'),
                                 ({'response_id': None}, 'protocol'),
                                 ({'fingerprint': {'unexpected': 'shape'}}, 'protocol')]:
            with self.subTest(kwargs=kwargs):
                opener = self.mock_opener(**kwargs)
                with patch('urllib.request.build_opener', return_value=opener):
                    with self.assertRaises(APIError) as caught:
                        chat_request(load_config(self.env), build_request(self.payload, model=DEEPSEEK_MODEL))
                self.assertEqual(caught.exception.kind, expected)
                self.assertFalse(caught.exception.retryable)
                opener.open.assert_called_once()

    def test_tampered_thinking_tools_or_routing_never_open_network(self):
        request = build_request(self.payload, model=DEEPSEEK_MODEL)
        config = load_config(self.env)
        with patch('urllib.request.build_opener') as opener:
            for variant in [{**request, 'enable_thinking': True}, {**request, 'tools': []},
                            {**request, 'enable_search': True}, {**request, 'stream': True},
                            {**request, 'seed': 7}, {**request, 'n': 2},
                            {**request, 'model': 'deepseek-v4.1-flash-1017'}]:
                with self.assertRaises(ValueError):
                    chat_request(config, variant)
            for conf in [APIConfig(DEEPSEEK_MODEL, 'https://api.deepseek.com/v1', 'secret', 'DEEPSEEKV4_API_KEY'),
                         APIConfig(DEEPSEEK_MODEL, DASHSCOPE_BASE_URL, 'secret', 'OPENAI_API_KEY'),
                         APIConfig('deepseek-v4.1-flash-1017', DASHSCOPE_BASE_URL, 'secret', 'DEEPSEEKV4_API_KEY')]:
                with self.assertRaises(ValueError):
                    chat_request(conf, request)
            opener.assert_not_called()

    def test_provider_error_never_echoes_provider_body_or_key(self):
        opener = MagicMock()
        body = json.dumps({'error': {'message': 'FAKE_DASHSCOPE_SECRET sensitive prompt'}}).encode()
        opener.open.side_effect = urllib.error.HTTPError(DASHSCOPE_BASE_URL, 401, 'secret', {}, io.BytesIO(body))
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as caught:
                chat_request(load_config(self.env), build_request(self.payload, model=DEEPSEEK_MODEL))
        self.assertEqual(caught.exception.kind, 'authentication')
        self.assertNotIn('FAKE_DASHSCOPE_SECRET', str(caught.exception) + json.dumps(vars(caught.exception)))

    def test_only_exact_undated_identifier_is_accepted(self):
        self.assertEqual(canonical_model('deepseek-v4.1-flash'), DEEPSEEK_MODEL)
        for model in ('deepseek-v4.1-flash-1017', 'deepseek-chat', 'deepseek-reasoner',
                      'gpt-4o-mini', 'deepseek-v4.1-flash '):
            with self.subTest(model=model), self.assertRaises(ValueError):
                load_config(self.env, model=model)
            with self.subTest(model=model), self.assertRaises(ValueError):
                build_request(self.payload, model=model)

    def test_optional_fingerprint_is_preserved_without_inventing_a_revision(self):
        for fingerprint in ('provider-fingerprint-123', ''):
            with self.subTest(fingerprint=fingerprint):
                opener = self.mock_opener(fingerprint=fingerprint, reasoning=None)
                with patch('urllib.request.build_opener', return_value=opener):
                    result = chat_request(load_config(self.env), build_request(self.payload, model=DEEPSEEK_MODEL))
                self.assertEqual(result['system_fingerprint'], fingerprint)
                self.assertIsNone(result['resolved_model_revision'])
                self.assertIsNone(result['reasoning_tokens'])
                self.assertEqual(result['model_identity_kind'], 'provider_alias')

    def test_two_original_images_keep_order_and_do_not_emit_audit_metadata(self):
        data = b'\xff\xd8\xffsecond original JPEG bytes'
        second = self.root / 'second.jpg'
        second.write_bytes(data)
        images = [*self.payload['query_images'], {'path': str(second),
                  'sha256': hashlib.sha256(data).hexdigest(), 'mime_type': 'image/jpeg'}]
        request = build_request({**self.payload, 'query_images': images}, model=DEEPSEEK_MODEL)
        content = request['messages'][0]['content']
        self.assertEqual([part['type'] for part in content], ['image_url', 'image_url', 'text'])
        self.assertEqual([base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                          for part in content[:-1]], [self.data, data])
        for forbidden in ('gold_ids', 'gold_choice', 'answer', 'model_identity_kind'):
            with self.subTest(field=forbidden), self.assertRaises(ValueError):
                build_request({**self.payload, forbidden: 'audit only'}, model=DEEPSEEK_MODEL)

    def test_redirect_response_is_not_retried_or_exposed(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            DASHSCOPE_BASE_URL, 302, 'redirect', {'Location': 'https://unexpected.example'},
            io.BytesIO(b'FAKE_DASHSCOPE_SECRET'))
        with patch('urllib.request.build_opener', return_value=opener):
            with self.assertRaises(APIError) as caught:
                chat_request(load_config(self.env), build_request(self.payload, model=DEEPSEEK_MODEL))
        self.assertEqual(caught.exception.kind, 'redirect')
        self.assertFalse(caught.exception.retryable)
        opener.open.assert_called_once()
        self.assertNotIn('FAKE_DASHSCOPE_SECRET', str(caught.exception))


if __name__ == '__main__':
    unittest.main()


class ProviderInspectionCodeTests(unittest.TestCase):
    def test_only_safe_provider_code_is_retained(self):
        from scripts.evaluation import deepseek_flash_seed42_audited_transport as t
        cfg=t.APIConfig(t.DEEPSEEK_MODEL,t.DASHSCOPE_BASE_URL,'FAKE_SECRET','DEEPSEEKV4_API_KEY')
        request=t.build_request({'user_text':'Return A.','query_images':[]},model=t.DEEPSEEK_MODEL,max_output_tokens=16)
        for code,expected in [('data_inspection_failed','data_inspection_failed'),('FAKE_SECRET',None)]:
            opener=MagicMock()
            raw=json.dumps({'error':{'code':code,'message':'FAKE_SECRET prompt echo'}}).encode()
            opener.open.side_effect=urllib.error.HTTPError(t.DASHSCOPE_BASE_URL,400,'bad',{},io.BytesIO(raw))
            with patch('urllib.request.build_opener',return_value=opener):
                with self.assertRaises(t.APIError) as caught:t.chat_request(cfg,request)
            self.assertEqual(caught.exception.provider_error_code,expected)
            self.assertNotIn('FAKE_SECRET',json.dumps(vars(caught.exception)))

    def test_request_is_identical_to_original_seed42_transport(self):
        from scripts.evaluation import deepseek_flash_seed42_audited_transport as new
        from scripts.evaluation import deepseek_flash_seed42_transport as old
        payload={'system':'System.','user_text':'Question.','query_images':[]}
        self.assertEqual(new.build_request(payload,model=new.DEEPSEEK_MODEL,max_output_tokens=16,image_detail='high'),
                         old.build_request(payload,model=old.DEEPSEEK_MODEL,max_output_tokens=16,image_detail='high'))
