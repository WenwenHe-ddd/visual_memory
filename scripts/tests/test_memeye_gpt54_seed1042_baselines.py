"""Exercise seed 1042 on the wire, frozen controls and full-run scoring."""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.analysis import rerun_memeye_gpt54_seed1042_baselines as experiment
from scripts.evaluation import openai_gpt54_seeded_transport as transport


class Seed1042Tests(unittest.TestCase):
    def test_full_mocked_workflow_and_seed42_control(self):
        calls=[];lock=threading.Lock()
        def fake(config,body,timeout=60):
            self.assertEqual(body['seed'],1042)
            with lock:
                calls.append(body)
                number=len(calls)
            return dict(raw_response='invalid' if number==1 else 'A',returned_model=transport.MODEL,
                response_id=f'test-{number}',request_id=f'request-{number}',system_fingerprint=None,
                finish_reason='stop',usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},
                reasoning_tokens=0,elapsed_seconds=0.,refusal=None)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp)
            protocol=experiment.prepare(output)
            self.assertEqual(protocol['contract']['seed'],1042)
            records=experiment.lines(output/'inputs/requests.jsonl')
            self.assertEqual(records,experiment.lines(experiment.REFERENCE/'inputs/requests.jsonl'))
            old=experiment.verify(experiment.REFERENCE)
            for r in records:
                before=experiment.request_body(old,r)
                after=experiment.request_body(protocol,r)
                self.assertEqual(before.pop('seed'),42)
                self.assertEqual(after.pop('seed'),1042)
                self.assertEqual(before,after)
            with patch.object(transport,'chat_request',side_effect=fake):
                validation=experiment.run(output)
            summary=experiment.analyze(output)
            self.assertEqual(len(calls),720)
            self.assertEqual(validation['logical_records'],742)
            self.assertGreaterEqual(validation['format_errors'],1)
            self.assertEqual(summary['overall']['questions'],371)
            self.assertEqual(summary['comparisons']['seed42']['baseline_question_only']['reference_correct'],233)
            self.assertEqual(summary['comparisons']['seed42']['master_answer_query']['reference_correct'],231)
            gold=experiment.indexed(experiment.lines(output/'inputs/gold.jsonl'),'id')
            responses=experiment.lines(output/'responses.jsonl')
            for condition,key in [('baseline_question_only','baseline_correct'),('master_answer_query','expanded_correct')]:
                expected=sum(r['predicted_choice']==gold[r['id']]['gold_choice'] for r in responses if r['condition']==condition)
                self.assertEqual(summary['overall'][key],expected)
            with patch.object(transport,'chat_request',side_effect=fake):
                experiment.run(output)
            self.assertEqual(len(calls),720)
            with self.assertRaises(AssertionError):
                experiment.prepare(output,42)


if __name__=='__main__':
    unittest.main()
