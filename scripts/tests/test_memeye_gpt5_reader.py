"""GPT-5.4 reader runtime/CLI checks with tiny synthetic, mocked requests."""
from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import visual_memory_pipeline
from scripts.evaluation import memeye_gpt5_reader as reader
from scripts.evaluation.openai_gpt5_chat_transport import APIError, MODEL, load_config
from scripts.tests import test_memeye_api_reader_runtime as legacy_fixture


class GPT54ReaderTests(unittest.TestCase):
    def setUp(self):
        # Reuse the isolated data fixture, without changing the old test module.
        self.fixture = legacy_fixture.MemEyeAPIRuntimeTests('test_parser_is_strict')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.records = self.fixture.root, self.fixture.records
        protocol = reader.read(self.root / 'protocol.json')
        c = protocol['contract']
        c['model'] = MODEL
        c['api_config'] = load_config(c['api_env_file']).public_dict()
        c['seed'] = None
        c['max_completion_tokens'] = c.pop('max_tokens')
        c['reasoning_effort'] = 'none'
        wire = [{'record_id': r['record_id'], 'payload_sha256': r['payload_sha256'],
                 'request_sha256': reader.digest(reader.request_body(protocol, r)),
                 'query_image_count': len(r['query_images'])} for r in self.records]
        c['request_manifest_sha256'] = reader.digest(wire)
        protocol['experiment_signature'] = reader.digest(c)
        reader.write_lines(self.root / 'api_request_manifest.jsonl', wire)
        reader.dump(self.root / 'protocol.json', protocol)

    @staticmethod
    def answer(config, request, timeout):
        text = request['messages'][1]['content'][-1]['text']
        return {'raw_response': 'B' if 'expanded' in text else 'A', 'returned_model': MODEL,
                'response_id': 'resp-' + reader.digest(request)[:8], 'system_fingerprint': None,
                'finish_reason': 'stop', 'usage': {'prompt_tokens': 10, 'completion_tokens': 1,
                                                 'total_tokens': 11, 'completion_tokens_details': {'reasoning_tokens': 0}},
                'reasoning_tokens': 0, 'request_id': 'req-test', 'elapsed_seconds': 0.1, 'refusal': None}

    def run_quietly(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return reader.run(self.root, **kwargs)

    def test_smoke_then_full_resume_uses_frozen_schema_and_shares_once(self):
        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=self.answer) as call:
            smoke = self.run_quietly(smoke=True)
            self.assertTrue(smoke['passed'])
            self.assertEqual({r['query_image_count'] for r in smoke['records']}, {0, 1})
            self.assertEqual(call.call_count, 2)
            result = self.run_quietly()
            self.assertEqual(call.call_count, 3)
            self.assertEqual((result['logical_records'], result['fresh_requests'], result['shared_records']), (4, 3, 1))
            self.assertEqual(result['reasoning_tokens_reported'], 0)
            self.assertEqual(result['reasoning_usage_missing_requests'], 0)
            self.run_quietly()
            self.assertEqual(call.call_count, 3)
            for invocation in call.call_args_list:
                config, request = invocation.args[:2]
                self.assertEqual(config.credential_variable, 'OPENAI_API_KEY')
                self.assertEqual(config.base_url, 'https://api.openai.com/v1')
                self.assertEqual(request['model'], MODEL)
                self.assertEqual(request['reasoning_effort'], 'none')
                self.assertEqual(request['max_completion_tokens'], 16)
                self.assertNotIn('seed', request)
                self.assertNotIn('max_tokens', request)
                self.assertNotIn('GOLD', json.dumps(request))

    def test_other_snapshot_is_rejected_and_not_saved(self):
        def wrong(config, request, timeout):
            return {**self.answer(config, request, timeout), 'returned_model': 'gpt-5.4-2026-04-01'}

        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=wrong) as call:
            with self.assertRaisesRegex(RuntimeError, 'API requests failed'):
                self.run_quietly()
        call.assert_called_once()
        self.assertFalse((self.root / 'fresh_responses.jsonl').exists())
        self.assertEqual(reader.read(self.root / 'failed_requests.json')[0]['last_attempt']['status'], 'model_mismatch')

    def test_nonzero_reasoning_fails_smoke_validation(self):
        def thinking(config, request, timeout):
            return {**self.answer(config, request, timeout), 'reasoning_tokens': 7}

        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=thinking) as call:
            with self.assertRaisesRegex(RuntimeError, 'Smoke outputs failed'):
                self.run_quietly(smoke=True)
        self.assertEqual(call.call_count, 2)
        self.assertFalse(reader.read(self.root / 'smoke_validation.json')['passed'])

    def test_unknown_reasoning_usage_is_reported_as_missing(self):
        def missing(config, request, timeout):
            return {**self.answer(config, request, timeout), 'reasoning_tokens': None, 'usage': None}

        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=missing):
            result = self.run_quietly()
        self.assertEqual(result['reasoning_usage_missing_requests'], 3)

    def test_permanent_quota_error_stops_bounded_batch(self):
        failure = APIError('quota', http_status=429, retryable=False, code='insufficient_quota', message='Quota reached')
        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=failure) as call, \
             patch.object(reader.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'API requests failed'):
                self.run_quietly()
        call.assert_called_once()
        sleep.assert_not_called()

    def test_resume_rejects_changed_returned_model(self):
        with patch('scripts.evaluation.openai_gpt5_chat_transport.chat_request', side_effect=self.answer) as call:
            self.run_quietly()
            completed = reader.lines(self.root / 'fresh_responses.jsonl')
            completed[0]['returned_model'] = 'gpt-4o-mini'
            reader.write_lines(self.root / 'fresh_responses.jsonl', completed)
            with self.assertRaisesRegex(ValueError, 'Saved prediction parser mismatch'):
                self.run_quietly()
            self.assertEqual(call.call_count, 3)

    def test_pipeline_model_guard_blocks_cross_model_resume(self):
        args = SimpleNamespace(output=self.root, stage='run', api_model='gpt-4o-mini', api_smoke=False)
        with patch.object(reader, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'fixed GPT-5.4 snapshot'):
                reader.pipeline(args)
            run.assert_not_called()

    def test_main_pipeline_alias_routes_to_new_profile_and_preserves_mini(self):
        from scripts.evaluation import memeye_api_reader
        with patch.object(reader, 'pipeline') as new, patch.object(memeye_api_reader, 'pipeline') as old:
            visual_memory_pipeline.main(['online', '--memory', 'caption_visual_master',
                                         '--reader-backend', 'api', '--api-model', 'gpt-5.4', '--stage', 'prepare'])
            new.assert_called_once()
            old.assert_not_called()
            args = new.call_args.args[0]
            self.assertEqual(args.api_model, MODEL)
            self.assertEqual(args.output.name, MODEL + '_none')
            visual_memory_pipeline.main(['online', '--memory', 'caption_visual_master',
                                         '--reader-backend', 'api', '--api-model', 'gpt-4o-mini', '--stage', 'prepare'])
            old.assert_called_once()


if __name__ == '__main__':
    unittest.main()
