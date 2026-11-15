"""Selection-policy regression tests: guard against choosing a failing winner."""
import unittest
from scripts.analysis.evaluate_memeye_qwen_zscore_search import choose_shared

class SelectionPolicyTest(unittest.TestCase):
    def row(self,b,m,a=1.,l=.01):
        return {'baseline_correct':b,'master_correct':m,'alpha':a,'lambda':l,
                'passes_all_three':b>184 and m>195 and m>b}
    def test_best_master_is_ineligible_if_baseline_does_not_improve(self):
        valid=self.row(185,202)
        self.assertEqual(choose_shared([self.row(184,220),self.row(200,200),valid]),valid)
    def test_strict_improvement_and_no_forced_winner(self):
        self.assertIsNone(choose_shared([self.row(184,202),self.row(190,195),self.row(201,201)]))
    def test_ties_use_baseline_then_smaller_alpha_then_smaller_lambda(self):
        winner=self.row(189,202,.3,.003)
        cases=[self.row(188,202,.1,.001),self.row(189,201,.1,.001),self.row(189,202,.6,.001),self.row(189,202,.3,.01),winner]
        self.assertEqual(choose_shared(cases),winner)
        self.assertEqual(choose_shared(cases[::-1]),winner)

if __name__=='__main__':unittest.main()
