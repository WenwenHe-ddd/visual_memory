import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.memory import hippomm_api as method, hippomm_api_transport as transport


class APITests(unittest.TestCase):
    def test_cache_reuses_same_request_without_persisting_transport_hash(self):
        response = {'returned_model': transport.MODELS['gpt5'], 'finish_reason': 'stop',
                    'refusal': None, 'raw_response': 'A', 'request_sha256': 'transport-only'}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(transport.gpt, 'load_config', return_value=None), \
             patch.object(transport.gpt, 'build_request', return_value={}) as build, \
             patch.object(transport.gpt, 'chat_request', return_value=response) as send:
            chat = transport.APIChat(root, 'gpt5', '/unused')
            self.assertEqual(chat('answer', 'test'), 'A')
            self.assertEqual(transport.APIChat(root, 'gpt5', '/unused')('answer', 'test'), 'A')
            self.assertEqual(send.call_count, 1)
            self.assertEqual(build.call_args.kwargs['seed'], 42)
            self.assertNotIn('sha256', (Path(root) / 'call_0001.json').read_text())

    def test_cached_wrong_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, patch.object(transport.deep, 'load_config', return_value=None):
            request = {'role': 'summary', 'prompt': 'test', 'image_paths': [],
                       'backbone': 'deepseek', 'max_output_tokens': 2048}
            (Path(root) / 'call_0001.json').write_text(json.dumps({'request': request,
                'response': {'returned_model': transport.MODELS['gpt5']}}))
            with self.assertRaisesRegex(ValueError, 'caption backbone'):
                transport.APIChat(root, 'deepseek', '/unused')('summary', 'test')

    def test_rewrite_recovery_preserves_model_generated_words(self):
        class Chat:
            last_call_id = 'call_0004'
            def __call__(self, role, prompt):
                self.role = role
                return '{"words":["red","leather","chair"]}'
        chat = Chat()
        calls, recovery = [], []
        result = method.repaired_rewrite(chat, '{"visual_query":"chair"}', 'original', calls, recovery)
        self.assertEqual(json.loads(result), {'visual_query': 'red leather chair'})
        self.assertEqual(calls, ['call_0004'])
        self.assertEqual(chat.role, 'rewrite_format_recovery')

    def test_pipeline_dispatch_is_isolated(self):
        from scripts import visual_memory_pipeline as pipeline
        for phase in ('offline', 'online'):
            with patch.object(method, 'pipeline', return_value='api') as target:
                result = pipeline.main([phase, '--memory', 'hippomm_api', '--dataset', 'memeye',
                                        '--hippomm-backbone', 'deepseek'])
                self.assertEqual(result, 'api')
                self.assertEqual(target.call_args.args[0].hippomm_backbone, 'deepseek')

if __name__ == '__main__':
    unittest.main()
