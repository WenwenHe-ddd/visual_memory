"""Independent invariants beyond the production evaluator's hash checks."""
import unittest

from scripts.analysis.audit_memeye_gpt5_reader import (
    MODEL, check_fresh_flags, expected_request, reasoning_counts,
)


class MemEyeGPT5IndependentAuditTests(unittest.TestCase):
    def test_none_request_does_not_send_legacy_generation_controls(self):
        contract = {'model':MODEL, 'reasoning_effort':'none', 'reader_system_prompt':'System',
                    'seed':42, 'max_tokens':16}
        body = expected_request(contract, [{'type':'text','text':'Question'}])
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertEqual(body['max_completion_tokens'],16)
        self.assertNotIn('seed',body)
        self.assertNotIn('max_tokens',body)
        self.assertEqual(body['messages'][1]['content'], [{'type':'text','text':'Question'}])

    def test_shared_result_cannot_claim_to_be_fresh(self):
        fresh = {'same_payload':{'record_id':'q::baseline'}}
        rows = {
            'q::baseline':{'payload_sha256':'same_payload','fresh_inference_executed_for_this_record':True,
                           'same_run_shared_inference_record_id':None},
            'q::new':{'payload_sha256':'same_payload','fresh_inference_executed_for_this_record':False,
                      'same_run_shared_inference_record_id':'q::baseline'},
        }
        check_fresh_flags(rows,fresh)
        rows['q::new']['fresh_inference_executed_for_this_record']=True
        with self.assertRaisesRegex(AssertionError,'fresh_flag'):
            check_fresh_flags(rows,fresh)

    def test_fresh_source_cannot_have_a_shared_pointer(self):
        fresh = {'payload':{'record_id':'source'}}
        rows = {'source':{'payload_sha256':'payload','fresh_inference_executed_for_this_record':True,
                          'same_run_shared_inference_record_id':'source'}}
        with self.assertRaisesRegex(AssertionError,'shared_source'):
            check_fresh_flags(rows,fresh)

    def test_missing_reasoning_usage_is_unknown_and_nonzero_is_rejected(self):
        report = reasoning_counts([
            {'usage':{'completion_tokens_details':{'reasoning_tokens':0}}},
            {'usage':{'completion_tokens':1}},
            {'usage':None},
        ])
        self.assertEqual(report,{'reported_zero':1,'reported_nonzero':0,'unreported':2,'reported_token_total':0})
        with self.assertRaisesRegex(AssertionError,'nonzero reasoning tokens'):
            reasoning_counts([{'usage':{'completion_tokens_details':{'reasoning_tokens':5}}}])


if __name__ == '__main__':
    unittest.main()
