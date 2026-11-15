"""GPT-5 wire isolation and full mocked reader audit/resumption on fresh memory."""
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
from scripts.analysis import evaluate_memeye_gpt5_base as experiment
from scripts.analysis.audit_memeye_gpt5_base import audit


class GPT5BaseTests(unittest.TestCase):
    def test_caption_reader_wire_preserves_alias_and_stage_seed(self):
        config=APIConfig(model='gpt-5',base_url=transport.OFFICIAL_BASE_URL,api_key='test')
        for seed,limit in ((None,2048),(42,2048)):
            body=transport.build_request({'user_text':'Return A.','query_images':[]},seed=seed,max_output_tokens=limit)
            class Response:
                headers={'x-request-id':'test-request'}
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self):return json.dumps({'model':'gpt-5-2025-08-07','id':'test','choices':[{'message':{'content':'A'},'finish_reason':'stop'}]}).encode()
            with patch.object(transport.urllib.request,'build_opener') as op:
                op.return_value.open.return_value=Response()
                result=transport.chat_request(config,body)
                sent=json.loads(op.return_value.open.call_args.args[0].data)
            self.assertEqual(sent,body);self.assertEqual(sent['model'],'gpt-5')
            self.assertNotIn('temperature',sent);self.assertEqual(sent['reasoning_effort'],'minimal')
            self.assertEqual(sent.get('seed'),seed);self.assertEqual(sent['max_completion_tokens'],limit)
            self.assertEqual(result['returned_model'],'gpt-5-2025-08-07')
        self.assertFalse(transport.valid_returned_model('gpt-5.4-2026-03-05'))
        self.assertFalse(transport.valid_returned_model('gpt-5-pro'))

    def test_incompatible_parameters_rejected_before_network(self):
        config=APIConfig(model='gpt-5',base_url=transport.OFFICIAL_BASE_URL,api_key='test')
        body=transport.build_request({'user_text':'Return A.','query_images':[]},seed=42)
        for changes in ({'temperature':0},{'reasoning_effort':'none'},{'model':'gpt-5.4'}):
            with patch.object(transport.urllib.request,'build_opener') as opener:
                with self.assertRaises(ValueError):transport.chat_request(config,{**body,**changes})
                opener.assert_not_called()

    def test_full_reader_workflow_audit_errors_and_resume(self):
        self.assertTrue((experiment.BASE/'zscore/retrieval_protocol.json').is_file(),'Fresh GPT-5 retrieval required for this integration test')
        calls=[];lock=threading.Lock()
        def fake(config,body,timeout=60):
            self.assertEqual(body['model'],'gpt-5');self.assertEqual(body['seed'],42)
            self.assertEqual(config.credential_variable,'OPENAI_API_KEY')
            with lock:
                calls.append(body);n=len(calls)
            return {'raw_response':'invalid' if n==1 else '' if n in (2,3) else 'A','requested_model':'gpt-5',
                'returned_model':'gpt-5-2025-08-07','response_id':f'gpt5_base-test-{n}','request_id':f'gpt5_base-test-request-{n}',
                'system_fingerprint':None,'finish_reason':'length' if n==2 else 'stop','usage':{'prompt_tokens':1,'completion_tokens':9,'total_tokens':10,
                'completion_tokens_details':{'reasoning_tokens':8}},'reasoning_tokens':8,'elapsed_seconds':0.,'refusal':'simulated refusal' if n==3 else None}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            source=Path(tmp)/'source'
            shutil.copytree(experiment.BASE,source,ignore=shutil.ignore_patterns('reader','summary.json','paired_results.json','paired_results.csv','report.html','independent_audit.json','*.log'))
            protocol=experiment.prepare(source)
            self.assertTrue(audit(source,inputs_only=True)['passed'])
            with patch.object(transport,'chat_request',side_effect=fake):
                validation=experiment.run(source/'reader')
            summary=experiment.analyze(source)
            verification=audit(source)
            self.assertTrue(verification['passed'])
            self.assertEqual(validation['logical_records'],1484)
            self.assertEqual(len(calls),protocol['contract']['unique_payloads'])
            self.assertGreaterEqual(validation['format_errors'],3)
            self.assertGreaterEqual(validation['empty_responses'],2)
            self.assertGreaterEqual(validation['token_limit_records'],1)
            self.assertEqual(validation['reasoning_tokens_reported'],8*len(calls))
            self.assertEqual(set(summary['overall']),set(experiment.CONDITIONS))
            self.assertTrue(all(s['questions']==371 for s in summary['overall'].values()))
            before=len(calls)
            with patch.object(transport,'chat_request',side_effect=fake):experiment.run(source/'reader')
            self.assertEqual(len(calls),before)


if __name__=='__main__':unittest.main()
