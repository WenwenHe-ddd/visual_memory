"""Seed-on-wire checks and a full mocked frozen-input evaluation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.evaluation import openai_gpt54_seeded_transport as transport
from scripts.evaluation.openai_chat_transport import APIConfig
from scripts.evaluation.openai_backbone_transport import build_request as old_request
from scripts.analysis import rerun_memeye_gpt54_seeded_baselines as experiment


class SeededBaselineTests(unittest.TestCase):
    def test_seed_is_the_only_request_change(self):
        payload={'system':'System','user_text':'Question','query_images':[]}
        expected=old_request(payload,model=transport.MODEL,max_output_tokens=16,image_detail='high')
        actual=transport.build_request(payload,seed=42)
        self.assertEqual(actual.pop('seed'),42)
        self.assertEqual(actual,expected)
        for bad in [True,None,1.5,2**63]:
            with self.assertRaises(ValueError):
                transport.build_request(payload,seed=bad)

    def test_seed_is_transmitted_and_model_checked(self):
        body=transport.build_request({'system':'S','user_text':'Q','query_images':[]})
        config=APIConfig(transport.MODEL,transport.OFFICIAL_BASE_URL,'FAKE_TEST_KEY')
        data={'model':transport.MODEL,'id':'test','choices':[{'message':{'content':'A'},'finish_reason':'stop'}],
              'usage':{'completion_tokens_details':{'reasoning_tokens':0}}}
        response,opener=MagicMock(),MagicMock()
        response.read.return_value=json.dumps(data).encode()
        response.headers.get.return_value='test-request'
        opener.open.return_value.__enter__.return_value=response
        with patch('urllib.request.build_opener',return_value=opener):
            result=transport.chat_request(config,body)
        sent=opener.open.call_args.args[0]
        self.assertEqual(json.loads(sent.data),body)
        self.assertEqual(json.loads(sent.data)['seed'],42)
        self.assertEqual(result['raw_response'],'A')
        data['model']='wrong-model';response.read.return_value=json.dumps(data).encode()
        with patch('urllib.request.build_opener',return_value=opener), self.assertRaises(transport.APIError):
            transport.chat_request(config,body)

    def test_gold_or_missing_seed_is_rejected_before_network(self):
        body=transport.build_request({'system':'S','user_text':'Q','query_images':[]})
        config=APIConfig(transport.MODEL,transport.OFFICIAL_BASE_URL,'FAKE_TEST_KEY')
        for request in [{**body,'gold_choice':'A'},{k:v for k,v in body.items() if k!='seed'}]:
            with patch('urllib.request.build_opener') as opener,self.assertRaises(ValueError):
                transport.chat_request(config,request)
            opener.assert_not_called()

    def test_mocked_full_run_keeps_all_questions_and_no_historical_answers(self):
        calls=[]
        def fake_chat(config,body,timeout=60):
            self.assertEqual(body['seed'],42)
            self.assertEqual(set(body),{'model','messages','temperature','seed','reasoning_effort',
                                      'max_completion_tokens','n','store','stream'})
            calls.append(body)
            return dict(raw_response='invalid' if len(calls)==1 else 'A',returned_model=transport.MODEL,
                response_id=f'test-{len(calls)}',system_fingerprint=None,finish_reason='stop',
                usage={'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},reasoning_tokens=0,
                request_id=f'request-{len(calls)}',elapsed_seconds=0.,refusal=None)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp)
            experiment.prepare(output,42)
            with patch.object(transport,'chat_request',side_effect=fake_chat):
                validation=experiment.run(output)
            result=experiment.analyze(output)
            self.assertEqual(len(calls),720)
            self.assertEqual(validation['logical_records'],742)
            self.assertGreaterEqual(validation['format_errors'],1)
            self.assertEqual(result['overall']['questions'],371)
            self.assertFalse(validation['historical_api_answer_reuse'])
            self.assertFalse(validation['gold_loaded_during_run'])
            before=len(calls)
            with patch.object(transport,'chat_request',side_effect=fake_chat):
                experiment.run(output)
            self.assertEqual(len(calls),before)


if __name__=='__main__':
    unittest.main()
