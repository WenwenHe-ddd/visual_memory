"""GPT-5.5 wire isolation and full mocked reader audit/resumption on fresh memory."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from scripts.evaluation import openai_gpt55_transport as transport
from scripts.evaluation.openai_chat_transport import APIConfig
from scripts.analysis import evaluate_memeye_gpt55 as experiment
from scripts.analysis.audit_memeye_gpt55 import audit


class GPT55Tests(unittest.TestCase):
    def test_caption_reader_wire_preserves_alias_and_stage_seed(self):
        config=APIConfig(model='gpt-5.5',base_url=transport.OFFICIAL_BASE_URL,api_key='test')
        for seed,limit in ((None,256),(42,16)):
            body=transport.build_request({'user_text':'Return A.','query_images':[]},seed=seed,max_output_tokens=limit)
            class Response:
                headers={'x-request-id':'test-request'}
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self):return json.dumps({'model':'gpt-5.5-2026-04-23','id':'test','choices':[{'message':{'content':'A'},'finish_reason':'stop'}]}).encode()
            with patch.object(transport.urllib.request,'build_opener') as op:
                op.return_value.open.return_value=Response()
                result=transport.chat_request(config,body)
                sent=json.loads(op.return_value.open.call_args.args[0].data)
            self.assertEqual(sent,body);self.assertEqual(sent['model'],'gpt-5.5')
            self.assertEqual(sent.get('seed'),seed);self.assertEqual(sent['max_completion_tokens'],limit)
            self.assertEqual(result['returned_model'],'gpt-5.5-2026-04-23')
        self.assertFalse(transport.valid_returned_model('gpt-5.4-2026-03-05'))
        self.assertFalse(transport.valid_returned_model('gpt-5.5-pro'))

    def test_full_reader_workflow_audit_errors_and_resume(self):
        self.assertTrue((experiment.BASE/'zscore/retrieval_protocol.json').is_file(),'Fresh GPT-5.5 retrieval required for this integration test')
        calls=[];lock=threading.Lock()
        def fake(config,body,timeout=60):
            self.assertEqual(body['model'],'gpt-5.5');self.assertEqual(body['seed'],42)
            self.assertEqual(config.credential_variable,'OPENAI_API_KEY')
            with lock:
                calls.append(body);n=len(calls)
            return {'raw_response':'invalid' if n==1 else 'A','requested_model':'gpt-5.5',
                'returned_model':'gpt-5.5-2026-04-23','response_id':f'gpt55-test-{n}','request_id':f'gpt55-test-request-{n}',
                'system_fingerprint':None,'finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2,
                'completion_tokens_details':{'reasoning_tokens':0}},'reasoning_tokens':0,'elapsed_seconds':0.,'refusal':None}
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
            self.assertGreaterEqual(validation['format_errors'],1)
            self.assertEqual(set(summary['overall']),set(experiment.CONDITIONS))
            self.assertTrue(all(s['questions']==371 for s in summary['overall'].values()))
            before=len(calls)
            with patch.object(transport,'chat_request',side_effect=fake):experiment.run(source/'reader')
            self.assertEqual(len(calls),before)


if __name__=='__main__':unittest.main()
