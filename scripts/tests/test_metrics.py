"""Verify ranking, tie denominators, and cue matching used by the experiment."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
import numpy as np
from scripts.analysis.analyze import rank_metrics, match, normalize

class MetricsTests(unittest.TestCase):
    def test_multiple_gold_coverage(self):
        m,r=rank_metrics(np.array([.9,.8,.7,.6,.5,.4]),[0,5])
        self.assertEqual(m['first_gold_rank'],1)
        self.assertEqual(m['mean_gold_rank'],3.5)
        self.assertEqual(m['recall_at_5'],.5)
        self.assertEqual(m['all_annotated_hit_at_5'],0)

    def test_tie_is_not_resolved_by_row_order(self):
        m,r=rank_metrics(np.ones(10),[0,1])
        self.assertEqual(m['mean_gold_rank'],5.5)
        self.assertAlmostEqual(m['first_gold_rank'],11/3)
        self.assertEqual(m['recall_at_5'],.5)
        self.assertAlmostEqual(m['all_annotated_hit_at_5'],2/9)
        self.assertEqual(m['all_annotated_hit_at_1'],0)

    def test_tie_below_cutoff(self):
        m,r=rank_metrics(np.array([.9,.8,.7,.6,.5,.5]),[4])
        self.assertEqual(m['mean_gold_rank'],5.5)
        self.assertEqual(m['recall_at_5'],.5)
        self.assertEqual(m['all_annotated_hit_at_5'],.5)

    def test_lexical_boundaries_and_inflections(self):
        self.assertTrue(match(normalize('Two women carry red bags beside the couches.'),normalize('woman')))
        self.assertTrue(match(normalize('Two women carry red bags beside the couches.'),normalize('bag')))
        self.assertTrue(match(normalize('Two women carry red bags beside the couches.'),normalize('sofa')))
        self.assertFalse(match(normalize('carpet'),normalize('car')))
        self.assertFalse(match(normalize('Player 10'),normalize('Player 1')))

if __name__=='__main__':unittest.main()
