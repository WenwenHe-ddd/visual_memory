"""Guard completeness and all three strict gates across the five candidate pairs."""
import unittest
from scripts.analysis.evaluate_memeye_gpt5_qwen_five import candidate_table,CONDITIONS,PAIRS,RAW,condition
class FiveCandidateTests(unittest.TestCase):
    def test_all_candidates_retained_and_strict_gates(self):
        self.assertEqual(len(CONDITIONS),12);self.assertEqual(len(set(CONDITIONS)),12)
        overall={RAW[0]:{'correct':211},RAW[1]:{'correct':219}}
        counts=((222,219),(215,221),(211,225),(223,223),(216,218))
        for (l,a),(b,m) in zip(PAIRS,counts):
            overall[condition(RAW[0],l,a)]={'correct':b};overall[condition(RAW[1],l,a)]={'correct':m}
        table=candidate_table(overall)
        self.assertEqual(len(table),5)
        self.assertEqual([r['passes_all_three'] for r in table],[False,True,False,False,False])
        self.assertEqual([r['qwen_rank'] for r in table],[1,2,3,4,5])
    def test_no_forced_success(self):
        overall={n:{'correct':200} for n in CONDITIONS}
        self.assertFalse(any(r['passes_all_three'] for r in candidate_table(overall)))
if __name__=='__main__':unittest.main()
