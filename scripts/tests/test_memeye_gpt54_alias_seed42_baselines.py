"""Verify alias bytes, credential routing, full scoring and safe resumption."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.analysis import rerun_memeye_gpt54_alias_seed42_baselines as experiment
from scripts.analysis.audit_memeye_gpt54_alias_seed42_baselines import audit
from scripts.evaluation import openai_gpt54_alias_transport as transport
from scripts.evaluation.openai_chat_transport import APIConfig


class AliasTests(unittest.TestCase):
    def test_literal_alias_on_http_wire_and_actual_returned_revision(self):
        config = APIConfig(model='gpt-5.4', base_url=transport.OFFICIAL_BASE_URL, api_key='test')
        request = transport.build_request({'system': 'system', 'user_text': 'question', 'query_images': []}, seed=42)
        response = {'model': 'gpt-5.4-2026-03-05', 'id': 'test', 'choices': [
            {'message': {'content': 'A'}, 'finish_reason': 'stop'}]}
        class Response:
            headers = {'x-request-id': 'request-test'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(response).encode()
        with patch.object(transport.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response()
            result = transport.chat_request(config, request)
            sent = json.loads(opener.return_value.open.call_args.args[0].data)
        self.assertEqual(sent, request)
        self.assertEqual(sent['model'], 'gpt-5.4')
        self.assertEqual(result['returned_model'], 'gpt-5.4-2026-03-05')
        self.assertEqual(result['requested_model'], 'gpt-5.4')
        self.assertFalse(transport.valid_returned_model('gpt-5.4-mini'))
        self.assertFalse(transport.valid_returned_model('gpt-5.5'))

    def test_full_workflow_preserves_controls_and_excludes_old_answers(self):
        calls = []
        lock = threading.Lock()
        def fake(config, body, timeout=60):
            self.assertEqual(body['model'], 'gpt-5.4')
            self.assertEqual(body['seed'], 42)
            self.assertEqual(config.credential_variable, 'OPENAI_API_KEY')
            with lock:
                calls.append(body)
                number = len(calls)
            return dict(raw_response='invalid' if number == 1 else 'A', requested_model='gpt-5.4',
                returned_model='gpt-5.4-2026-03-05', response_id=f'test-{number}',
                request_id=f'request-{number}', system_fingerprint=None, finish_reason='stop',
                usage={'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
                reasoning_tokens=0, elapsed_seconds=0., refusal=None)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            protocol = experiment.prepare(output)
            records = experiment.lines(output/'inputs/requests.jsonl')
            old = experiment.verify(experiment.REFERENCE)
            for row in records:
                before = experiment.previous.request_body(old, row)
                after = experiment.request_body(protocol, row)
                self.assertEqual(before.pop('model'), 'gpt-5.4-2026-03-05')
                self.assertEqual(after.pop('model'), 'gpt-5.4')
                self.assertEqual(before, after)
            with patch.object(transport, 'chat_request', side_effect=fake):
                validation = experiment.run(output)
            summary = experiment.analyze(output)
            verification = audit(output)
            self.assertEqual(len(calls), 720)
            self.assertTrue(verification['passed'])
            self.assertEqual(validation['logical_records'], 742)
            self.assertGreaterEqual(validation['format_errors'], 1)
            self.assertEqual(summary['overall']['questions'], 371)
            self.assertEqual(summary['comparisons']['snapshot_seed42']['baseline_question_only']['reference_correct'], 233)
            self.assertEqual(summary['comparisons']['snapshot_seed42']['master_answer_query']['reference_correct'], 231)
            with patch.object(transport, 'chat_request', side_effect=fake):
                experiment.run(output)
            self.assertEqual(len(calls), 720)
            with self.assertRaises(AssertionError):
                experiment.prepare(output, 1042)


if __name__ == '__main__':
    unittest.main()
