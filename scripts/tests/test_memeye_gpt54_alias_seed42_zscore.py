"""Check z-score controls, scoring, and resume without paid API requests."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.analysis import evaluate_memeye_gpt54_alias_seed42_zscore as experiment
from scripts.analysis.audit_memeye_gpt54_alias_seed42_zscore import audit
from scripts.evaluation import openai_gpt54_alias_transport as transport


class ZScoreAliasTests(unittest.TestCase):
    def test_frozen_controls_fresh_answers_scoring_and_resume(self):
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
                returned_model='gpt-5.4-2026-03-05', response_id=f'zscore-test-{number}',
                request_id=f'zscore-request-{number}', system_fingerprint=None, finish_reason='stop',
                usage={'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2,
                       'completion_tokens_details': {'reasoning_tokens': 0}},
                reasoning_tokens=0, elapsed_seconds=0., refusal=None)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            for name in ('retrieval_results.jsonl', 'retrieval_protocol.json', 'source_validation.json'):
                shutil.copyfile(experiment.BASE/name, output/name)
            protocol = experiment.prepare(output, experiment.SOURCE, experiment.api.DEFAULT_ENV, 3)
            inputs = audit(output, inputs_only=True)
            self.assertTrue(inputs['passed'])
            self.assertEqual(protocol['contract']['prior_unweighted_correct'], {
                'baseline_question_only': 236, 'master_answer_query': 238})
            with patch.object(transport, 'chat_request', side_effect=fake):
                validation = experiment.run(output)
            summary = experiment.analyze(output)
            verification = audit(output)
            self.assertEqual(len(calls), 726)
            self.assertTrue(verification['passed'])
            self.assertTrue(verification['historical_response_ids_excluded'])
            self.assertEqual(validation['logical_records'], 742)
            self.assertEqual(validation['shared_records'], 16)
            self.assertGreaterEqual(validation['format_errors'], 1)
            self.assertEqual(summary['overall']['questions'], 371)
            for name in experiment.NAMES:
                self.assertEqual(sum(v['questions'] for v in summary['historical_comparisons'][name]['request_strata'].values()), 371)
            with patch.object(transport, 'chat_request', side_effect=fake):
                experiment.run(output)
            self.assertEqual(len(calls), 726)


if __name__ == '__main__':
    unittest.main()
