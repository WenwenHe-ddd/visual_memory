"""A truncated letter must not become a success or disappear from evaluation."""
import unittest
from scripts.analysis.memeye_deepseek_seed42_complementary import valid_answer, metrics, MODEL

class OutcomePolicyTests(unittest.TestCase):
    def test_finish_reason_and_full_letter_format_required(self):
        base={'returned_model':MODEL,'finish_reason':'stop','raw_response':'A','refusal':None}
        self.assertTrue(valid_answer(base))
        self.assertFalse(valid_answer({**base,'finish_reason':'length'}))
        self.assertFalse(valid_answer({**base,'raw_response':'Based on the images'}))
        self.assertFalse(valid_answer({**base,'refusal':'Refused'}))

    def test_failure_stays_in_accuracy_and_recall_denominators(self):
        rows=[{'gold_count':2,'x_correct':True,'x_gold_hits':1,'x_status':'ok'},
              {'gold_count':3,'x_correct':False,'x_gold_hits':2,'x_status':'provider_input_rejected'}]
        result=metrics(rows,'x')
        self.assertEqual(result['questions'],2)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['failed_records'],1)
        self.assertEqual(result['recall_at_5_micro'],.6)
