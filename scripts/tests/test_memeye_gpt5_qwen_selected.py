"""The preregistered validation gate requires all three strict comparisons."""
import unittest
from scripts.analysis.evaluate_memeye_gpt5_qwen_selected import success_checks,CONDITIONS
class ValidationGateTest(unittest.TestCase):
    def check(self,counts):
        return all(success_checks({n:{'correct':v} for n,v in zip(CONDITIONS,counts)}).values())
    def test_three_strict_comparisons(self):
        self.assertTrue(self.check([219,220,223,224]))
        for counts in ([219,219,223,224],[219,220,223,223],[219,224,223,224],[219,218,223,225]):
            self.assertFalse(self.check(counts))
if __name__=='__main__':unittest.main()
