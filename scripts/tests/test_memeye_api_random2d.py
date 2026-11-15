"""Independent payload/dedup/error checks for the new API basis ablation."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.analysis import evaluate_memeye_api_random2d as target
from scripts.evaluation.openai_chat_transport import APIError


class APIBasisTests(unittest.TestCase):
    def fixture(self):
        memories = {f'D/m{i}': {'task': 'D', 'retrieval_text': f'dialogue {i}\nMemory description: caption {i}'} for i in range(5)}
        q = {'id': 'D/Q1', 'task': 'D', 'question': 'Which color?', 'options': dict(zip('ABCD', ['red', 'blue', 'green', 'white'])), 'query_images': []}
        ids = list(memories)
        context = '\n\n'.join(f'Memory {i+1}: dialogue {i}\nMemory description: caption {i}' for i in range(5))
        condition = {'name': 'pca_2d', 'top5_memory_ids': ids, 'reader_context': context}
        return memories, q, condition

    def test_exact_text_and_generation_parameters(self):
        m, q, c = self.fixture()
        expected = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + c['reader_context']
                    + '\n\nQuestion:\nWhich color?\n\nOptions:\nA. red\nB. blue\nC. green\nD. white\n\nAnswer with one letter only.')
        for backend in ('gpt5', 'deepseek'):
            row = target.build_record(backend, 'system', q, c, m)
            self.assertEqual(row['user_text'], expected)
            payload = {k: row[k] for k in ('system', 'user_text', 'query_images')}
            wire = target.build_wire(backend, payload)
            self.assertEqual(wire['seed'], 42)
            self.assertNotIn('gold_choice', str(wire))
            self.assertNotIn('top_memory_ids', str(wire))
            if backend == 'gpt5':
                self.assertNotIn('temperature', wire)
                self.assertEqual(wire['reasoning_effort'], 'minimal')
                self.assertEqual(wire['max_completion_tokens'], 2048)
            else:
                self.assertIs(wire['enable_thinking'], False)
                self.assertEqual(wire['max_tokens'], 16)
                self.assertEqual(wire['temperature'], 0)

    def test_unexpected_payload_field_and_modified_context_rejected(self):
        m, q, c = self.fixture()
        for backend in ('gpt5', 'deepseek'):
            with self.assertRaises(AssertionError):
                target.build_wire(backend, {'system': '', 'user_text': '', 'query_images': [], 'gold_choice': 'A'})
            with self.assertRaises(AssertionError):
                target.build_record(backend, '', q, {**c, 'reader_context': 'changed'}, m)

    def test_complete_wire_dedup_preserves_order_and_text_changes(self):
        m, q, c = self.fixture()
        rows = [target.build_record('gpt5', '', q, {**c, 'name': name}, m) for name in target.CONDITIONS]
        self.assertEqual(len(target.unique_requests(rows)), 1)
        changed = target.build_record('gpt5', '', {**q, 'question': 'Changed query'}, c, m)
        self.assertEqual(len(target.unique_requests(rows + [changed])), 2)
        key = rows[0]['request_sha256']
        fresh = {key: {**rows[0], 'raw_response': 'B'}}
        out = target.materialize(rows, fresh)
        self.assertEqual([r['condition'] for r in out], list(target.CONDITIONS))
        self.assertEqual(sum(r['same_run_shared'] for r in out), 3)
        self.assertTrue(all(r['raw_response'] == 'B' for r in out))

    def simulate_run(self, second):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / 'reader'
            folder.mkdir()
            wire = lambda backend, payload: {'seed': 42, 'user_text': payload['user_text']}
            records = []
            for qi in range(2):
                for condition in target.CONDITIONS:
                    payload = {'system': '', 'user_text': str(qi), 'query_images': []}
                    records.append({'record_id': f'Q{qi}::{condition}', 'question_id': f'Q{qi}', 'task': 'D',
                                    'condition': condition, 'top_memory_ids': list('abcde'), **payload,
                                    'payload_sha256': target.digest(payload), 'request_sha256': target.digest(wire('', payload))})
            target.write_lines(folder / 'requests.jsonl', records)
            c = {'backend': 'deepseek', 'api_env_file': '', 'api_config': {}, 'workers': 1, 'max_attempts': 3,
                 'logical_records': 8, 'unique_requests': 2, 'timeout_seconds': 1, 'model': 'deepseek-v4.1-flash',
                 'historical_reader_model': 'deepseek-v4.1-flash'}
            p = {'contract': c, 'run_id': 'fresh-test', 'experiment_signature': target.digest(c)}
            good = {'raw_response': 'A', 'finish_reason': 'stop', 'refusal': None,
                    'returned_model': 'deepseek-v4.1-flash', 'requested_model': 'deepseek-v4.1-flash',
                    'response_id': 'test-id', 'request_id': 'test-id'}
            calls = []
            def chat(config, body, timeout):
                calls.append(body)
                if body['user_text'] == '0':
                    return good
                if isinstance(second, Exception):
                    raise second
                return {**good, **second}
            transport = SimpleNamespace(APIError=APIError, load_config=lambda _: SimpleNamespace(public_dict=lambda: {}), chat_request=chat)
            with patch.object(target, 'verify', return_value=p), patch.object(target, 'transport_for', return_value=transport), patch.object(target, 'build_wire', side_effect=wire):
                validation = target.run(output)
                first_count = len(calls)
                target.run(output)
            self.assertEqual(len(calls), first_count, 'Resume must not repeat same-run completed requests')
            self.assertEqual(first_count, 2, 'Only two complete wires should be sent')
            return validation, target.lines(folder / 'responses.jsonl')

    def test_content_rejection_is_terminal_and_reused_without_retry(self):
        error = APIError('http', http_status=400, retryable=True)
        error.provider_error_code = 'data_inspection_failed'
        validation, rows = self.simulate_run(error)
        self.assertEqual(validation['content_rejections'], 4)
        self.assertEqual(validation['api_calls'], 2)
        self.assertEqual(sum(r['status'] == 'content_rejected' for r in rows), 4)

    def test_truncated_letter_is_invalid_and_not_resampled(self):
        validation, rows = self.simulate_run({'raw_response': 'A', 'finish_reason': 'length'})
        self.assertEqual(validation['format_errors'], 4)
        self.assertEqual(validation['token_limit_hits'], 4)
        self.assertTrue(all(r['status'] == 'format_error' for r in rows if r['question_id'] == 'Q1'))
        self.assertTrue(all(r['predicted_choice'] is None and r['parsed_choice'] == 'A' for r in rows if r['question_id'] == 'Q1'))

    def test_refused_letter_is_invalid_and_not_resampled(self):
        validation, rows = self.simulate_run({'raw_response': 'A', 'refusal': 'refused'})
        self.assertEqual(validation['format_errors'], 4)
        self.assertTrue(all(r['status'] == 'format_error' for r in rows if r['question_id'] == 'Q1'))
        self.assertTrue(all(r['predicted_choice'] is None and r['parsed_choice'] == 'A' for r in rows if r['question_id'] == 'Q1'))


if __name__ == '__main__':
    unittest.main()
