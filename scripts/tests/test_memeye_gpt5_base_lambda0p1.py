"""Full lambda=.1 mocked scoring, failure-denominator, controls and resumption audit."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from scripts.evaluation import openai_gpt5_base_transport as transport
from scripts.evaluation.openai_chat_transport import APIConfig
from scripts.analysis import evaluate_memeye_gpt5_base_lambda0p1 as experiment
from scripts.analysis.audit_memeye_gpt5_base_lambda0p1 import audit


class GPT5LambdaTests(unittest.TestCase):
    def test_full_reader_workflow_audit_errors_and_resume(self):
        self.assertTrue((experiment.DEFAULT_OUTPUT/'retrieval_protocol.json').is_file(),'Lambda=.1 retrieval required for this integration test')
        calls=[];lock=threading.Lock()
        def fake(config,body,timeout=60):
            self.assertEqual(body['model'],'gpt-5');self.assertEqual(body['seed'],42)
            self.assertEqual(config.credential_variable,'OPENAI_API_KEY')
            with lock:
                calls.append(body);n=len(calls)
            return {'raw_response':'invalid' if n==1 else '' if n in (2,3) else 'A','requested_model':'gpt-5',
                'returned_model':'gpt-5-2025-08-07','response_id':f'gpt5-lambda-test-{n}','request_id':f'gpt5-lambda-test-request-{n}',
                'system_fingerprint':None,'finish_reason':'length' if n==2 else 'stop','usage':{'prompt_tokens':1,'completion_tokens':9,'total_tokens':10,
                'completion_tokens_details':{'reasoning_tokens':8}},'reasoning_tokens':8,'elapsed_seconds':0.,'refusal':'simulated refusal' if n==3 else None}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            source=Path(tmp)/'source'
            shutil.copytree(experiment.DEFAULT_OUTPUT,source,ignore=shutil.ignore_patterns('reader','summary.json','paired_results.json','paired_results.csv','report.html','independent_audit.json','*.log'))
            protocol=experiment.prepare(source)
            self.assertTrue(audit(source,inputs_only=True)['passed'])
            with patch.object(transport,'chat_request',side_effect=fake):
                validation=experiment.run(source/'reader')
            summary=experiment.analyze(source)
            verification=audit(source)
            self.assertTrue(verification['passed'])
            self.assertEqual(validation['logical_records'],742)
            self.assertEqual(len(calls),protocol['contract']['unique_payloads'])
            self.assertGreaterEqual(validation['format_errors'],3)
            self.assertGreaterEqual(validation['empty_responses'],2)
            self.assertGreaterEqual(validation['token_limit_records'],1)
            self.assertEqual(validation['reasoning_tokens_reported'],8*len(calls))
            self.assertEqual(set(summary['overall']),set(experiment.NAMES))
            self.assertTrue(all(s['questions']==371 for s in summary['overall'].values()))
            before=len(calls)
            with patch.object(transport,'chat_request',side_effect=fake):experiment.run(source/'reader')
            self.assertEqual(len(calls),before)


if __name__=='__main__':unittest.main()
