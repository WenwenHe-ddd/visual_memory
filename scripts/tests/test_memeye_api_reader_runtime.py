"""Small synthetic reader runs; every API call is mocked and labels stay local."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import memeye_api_reader as reader
from scripts.evaluation.openai_chat_transport import APIError, load_config


class MemEyeAPIRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        (self.root / 'inputs').mkdir()
        env = self.root / '.env'
        env.write_text('OPENAI_API_KEY=FAKE_UNIT_TEST_KEY\nOPENAI_MODEL=gpt-4o-mini\n')
        config = load_config(env)
        image = self.root / 'query.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\noriginal synthetic query image')
        image_record = {'path': str(image), 'sha256': reader.sha(image)}
        self.records = []
        for qid, condition, text, images in (
            ('Q1', reader.CONDITIONS[0], 'Q1 identical evidence and question', []),
            ('Q1', reader.CONDITIONS[1], 'Q1 identical evidence and question', []),
            ('Q2', reader.CONDITIONS[0], 'Q2 baseline evidence and question', [image_record]),
            ('Q2', reader.CONDITIONS[1], 'Q2 expanded evidence and question', [image_record]),
        ):
            payload = reader.digest({'system': 'Return one letter.', 'user_text': text, 'images': images})
            self.records.append({'record_id': qid + '::' + condition, 'id': qid, 'task': 'Test',
                                 'condition': condition, 'reader_text': text, 'query_images': images,
                                 'top_memory_ids': ['Test/M1'], 'payload_sha256': payload})
        reader.write_lines(self.root / 'inputs/requests.jsonl', self.records)
        # Intentionally not JSON: successful runtime proves it never parses gold.
        (self.root / 'inputs/gold.jsonl').write_text('MUST NOT LOAD GOLD DURING RUN\n')
        contract = {
            'model': config.model, 'base_url': config.base_url, 'api_config': config.public_dict(),
            'api_env_file': str(env), 'reader_system_prompt': 'Return one letter.',
            'temperature': 0, 'seed': 42, 'max_tokens': 16, 'image_detail': 'high',
            'workers': 1, 'timeout_seconds': 1, 'max_attempts': 3,
            'logical_records': 4, 'unique_payloads': 3,
            'source_sha256': {str(p): reader.sha(p) for p in
                              (self.root / 'inputs/requests.jsonl', self.root / 'inputs/gold.jsonl', image)},
        }
        provisional = {'contract': contract}
        manifest = [{'record_id': r['record_id'], 'payload_sha256': r['payload_sha256'],
                     'request_sha256': reader.digest(reader.request_body(provisional, r)),
                     'query_image_count': len(r['query_images'])} for r in self.records]
        contract['request_manifest_sha256'] = reader.digest(manifest)
        reader.write_lines(self.root / 'api_request_manifest.jsonl', manifest)
        self.protocol = {'run_id': 'unit-test-run', 'experiment_signature': reader.digest(contract),
                         'contract': contract}
        reader.dump(self.root / 'protocol.json', self.protocol)

    @staticmethod
    def answer(config, request, timeout):
        text = request['messages'][1]['content'][-1]['text']
        return {'raw_response': 'B' if 'expanded' in text else 'A',
                'returned_model': 'gpt-4o-mini-2024-07-18', 'response_id': 'resp-' + reader.digest(request)[:8],
                'system_fingerprint': 'fp-test', 'finish_reason': 'stop',
                'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11},
                'request_id': 'req-test', 'elapsed_seconds': 0.1, 'refusal': None}

    def run_quietly(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return reader.run(self.root, **kwargs)

    def test_parser_is_strict(self):
        for text, expected in [('A', 'A'), (' b. ', 'B'), ('Answer: C!', 'C'),
                               ('D because this is correct', None), ('AB', None),
                               ('', None), ('(C)', None)]:
            with self.subTest(text=text):
                self.assertEqual(reader.parse_choice(text), expected)

    def test_fresh_dedup_gold_isolation_and_resume(self):
        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=self.answer) as call:
            result = self.run_quietly()
            self.assertEqual(call.call_count, 3)
            self.assertEqual((result['logical_records'], result['fresh_requests'], result['shared_records']), (4, 3, 1))
            sent = json.dumps([c.args[1] for c in call.call_args_list])
            self.assertNotIn('GOLD', sent)
            self.assertNotIn('gold_choice', sent)
            self.assertNotIn('predicted_choice', sent)
            responses = reader.lines(self.root / 'responses.jsonl')
            first, shared = responses[:2]
            self.assertTrue(first['fresh_inference_executed_for_this_record'])
            self.assertFalse(shared['fresh_inference_executed_for_this_record'])
            self.assertEqual(shared['same_run_shared_inference_record_id'], first['record_id'])
            self.assertEqual(shared['response_id'], first['response_id'])
            original_bytes = (self.root / 'fresh_responses.jsonl').read_bytes()
            resumed = self.run_quietly()
            self.assertEqual(call.call_count, 3)
            self.assertEqual(resumed['fresh_requests'], 3)
            self.assertEqual((self.root / 'fresh_responses.jsonl').read_bytes(), original_bytes)

    def test_request_and_resume_tamper_fail_before_network(self):
        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=self.answer) as call:
            self.run_quietly()
            baseline_calls = call.call_count
            responses = reader.lines(self.root / 'fresh_responses.jsonl')
            responses[0]['request_sha256'] = '0' * 64
            reader.write_lines(self.root / 'fresh_responses.jsonl', responses)
            with self.assertRaisesRegex(ValueError, 'resume provenance mismatch'):
                self.run_quietly()
            self.assertEqual(call.call_count, baseline_calls)
            (self.root / 'inputs/requests.jsonl').write_text('tampered input\n')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                self.run_quietly()
            self.assertEqual(call.call_count, baseline_calls)

    def test_actual_body_must_match_frozen_wire_hash(self):
        real_body = reader.request_body

        def changed_body(protocol, row):
            body = real_body(protocol, row)
            body['messages'][1]['content'][-1]['text'] += 'CHANGED'
            return body

        with patch.object(reader, 'request_body', side_effect=changed_body), \
             patch('scripts.evaluation.openai_chat_transport.chat_request') as call:
            with self.assertRaisesRegex(ValueError, 'differs from frozen request'):
                self.run_quietly()
            call.assert_not_called()

    def test_transient_retry_uses_same_request_and_permanent_error_stops(self):
        attempts = []

        def transient(config, request, timeout):
            attempts.append(reader.digest(request))
            if len(attempts) == 1:
                raise APIError('rate_limit', http_status=429, retryable=True, message='Rate limited')
            return self.answer(config, request, timeout)

        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=transient), \
             patch.object(reader.time, 'sleep') as sleep:
            self.run_quietly()
        self.assertEqual(len(attempts), 4)
        self.assertEqual(attempts[0], attempts[1])
        sleep.assert_called_once_with(2)
        log = reader.lines(self.root / 'api_attempts.jsonl')
        self.assertEqual([r['status'] for r in log[:2]], ['api_error', 'success'])

    def test_smoke_is_fresh_for_both_modalities_then_reused_within_run(self):
        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=self.answer) as call:
            smoke = self.run_quietly(smoke=True)
            self.assertTrue(smoke['passed'])
            self.assertEqual(call.call_count, 2)
            self.assertEqual({r['query_image_count'] for r in smoke['records']}, {0, 1})
            full = self.run_quietly()
            self.assertEqual(call.call_count, 3)
            self.assertEqual(full['logical_records'], 4)

    def test_permanent_auth_failure_stops_after_first_bounded_batch(self):
        failure = APIError('authentication', http_status=401, retryable=False, message='Rejected')
        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=failure) as call, \
             patch.object(reader.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'API requests failed'):
                self.run_quietly()
            call.assert_called_once()
            sleep.assert_not_called()
        self.assertFalse((self.root / 'fresh_responses.jsonl').exists())
        failure_log = reader.read(self.root / 'failed_requests.json')
        self.assertEqual(failure_log[0]['last_attempt']['http_status'], 401)

    def test_format_error_is_retained_without_retry(self):
        def malformed(config, request, timeout):
            return {**self.answer(config, request, timeout), 'raw_response': 'A because of evidence'}

        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=malformed) as call:
            result = self.run_quietly()
        self.assertEqual(call.call_count, 3)
        self.assertEqual(result['format_errors'], 4)
        self.assertTrue(all(r['predicted_choice'] is None for r in reader.lines(self.root / 'responses.jsonl')))

    def test_exact_requested_model_alias_is_accepted(self):
        def alias(config, request, timeout):
            return {**self.answer(config, request, timeout), 'returned_model': config.model}

        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=alias) as call:
            result = self.run_quietly()
        self.assertEqual(call.call_count, 3)
        self.assertEqual(result['returned_models'], ['gpt-4o-mini'])

    def test_dated_snapshot_is_accepted(self):
        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=self.answer):
            result = self.run_quietly()
        self.assertEqual(result['returned_models'], ['gpt-4o-mini-2024-07-18'])

    def test_wrong_model_family_suffix_is_rejected_without_retry(self):
        def wrong_family(config, request, timeout):
            return {**self.answer(config, request, timeout), 'returned_model': 'gpt-4o-mini-tts'}

        with patch('scripts.evaluation.openai_chat_transport.chat_request', side_effect=wrong_family) as call:
            with self.assertRaisesRegex(RuntimeError, 'API requests failed'):
                self.run_quietly()
        call.assert_called_once()
        self.assertFalse((self.root / 'fresh_responses.jsonl').exists())
        failure = reader.read(self.root / 'failed_requests.json')[0]['last_attempt']
        self.assertEqual(failure['status'], 'model_mismatch')
        self.assertEqual(failure['returned_model'], 'gpt-4o-mini-tts')


if __name__ == '__main__':
    unittest.main()
