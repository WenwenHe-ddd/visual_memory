"""Independent covariance oracle and pipeline isolation for positive variance."""
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch
import numpy as np

from scripts import visual_memory_pipeline as pipeline
from scripts.memory import online_retrieval as online
from scripts.memory.online_positive_variance import PositiveVarianceWeighting
from scripts.memory.online_zscore import ZScoreWeighting


class PositiveVarianceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(112)
        self.x = rng.normal(size=(7, 11))
        self.x /= np.linalg.norm(self.x, axis=1, keepdims=True)
        self.q = rng.normal(size=11)
        self.q /= np.linalg.norm(self.q)

    def test_positive_covariance_operator_and_existing_zscore_are_distinct(self):
        centered = self.x - self.x.mean(0)
        covariance = centered.T @ centered / (len(self.x)-1)
        for ridge in [.001, .03, 1.]:
            with self.subTest(ridge=ridge):
                result = PositiveVarianceWeighting(ridge, 1).apply(self.q, self.x)
                inverse = np.linalg.solve(covariance + ridge*np.eye(11), ridge*self.q)
                expected = self.x @ (2*self.q - inverse)
                np.testing.assert_allclose(result['positive_variance'], expected, atol=2e-14)
                white = ZScoreWeighting(ridge, 1).apply(self.q, self.x)
                np.testing.assert_allclose(result['positive_variance'] + white['white'], 2*result['raw'], atol=2e-14)
                self.assertNotIn('white', result)
                self.assertNotIn('positive_variance', white)
                self.assertFalse(np.allclose(result['correction'], -white['correction']))

    def test_positive_tanh_and_alpha_zero(self):
        result = PositiveVarianceWeighting(.03, 1).apply(self.q, self.x)
        np.testing.assert_allclose(result['correction'], np.std(result['raw'])*np.tanh(result['z']))
        self.assertLessEqual(abs(result['correction']).max(), result['raw_std'])
        self.assertTrue(np.all(result['correction']*result['z'] >= 0))
        zero = PositiveVarianceWeighting(.03, 0).apply(self.q, self.x)
        np.testing.assert_array_equal(zero['weighted'], self.x@self.q)

    def test_singletons_constant_scores_and_two_leaf_case(self):
        for x in [np.array([[1.,0.]]), np.array([[1.,0.],[1.,0.],[1.,0.]])]:
            result = PositiveVarianceWeighting(.03, 1).apply(np.array([.8,.6]), x)
            np.testing.assert_array_equal(result['raw'], result['weighted'])
        two = PositiveVarianceWeighting(.03, 1).apply(np.array([.8,.6]), np.eye(2))
        np.testing.assert_array_equal(two['z'], [1.,-1.])

    def test_invalid_parameters_and_factory(self):
        for ridge, alpha in [(0,1),(-1,1),(float('nan'),1),(float('inf'),1),(.03,-1),(.03,float('nan'))]:
            with self.assertRaises(ValueError): PositiveVarianceWeighting(ridge,alpha)
        self.assertIsNone(online.make_weighting('none',.03,1))
        self.assertIsInstance(online.make_weighting('positive_variance',.03,1), PositiveVarianceWeighting)
        self.assertIsInstance(online.make_weighting('z_score',.03,1), ZScoreWeighting)
        with self.assertRaises(ValueError): online.make_weighting('typo',.03,1)

    def test_cli_dispatch_for_both_memories_and_legacy_conflict(self):
        for memory, strategy in [('natural','full_scan'),('caption_visual_master','multi_path')]:
            args = ['online','--memory',memory,'--online-weighting','positive_variance',
                    '--whitening-lambda','.03','--reweight-alpha','1']
            with patch.object(online,'online') as run:
                pipeline.main(args)
                actual = run.call_args.args[0]
                self.assertEqual((actual.online_weighting,actual.retrieval,actual.whitening_lambda,actual.reweight_alpha),
                                 ('positive_variance',strategy,.03,1.))
                self.assertEqual(actual.output.name,'positive_variance')
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.main(args+['--retrieval-scoring','z_reweight'])

    def test_same_queries_raw_ranking_and_gold_independence(self):
        leaves = [{'image_id': f'Home/M{i}#image1','memory_id': f'Home/M{i}',
                   'dialogue': f'dialogue {i}','caption': f'caption {i}'} for i in range(len(self.x))]
        memory = online.NaturalCaptionMemory(leaves)
        question = {'id':'Home/Q1','question':'Question','retrieval_query':'Exact query',
                    'gold_ids':['Home/M0'],'gold_choice':'A'}
        def encode(_model,texts,prefix=''):
            return self.x.copy() if texts == memory.texts else self.q[None].copy()
        with patch.object(online,'encode_text',side_effect=encode):
            unweighted = online.retrieve_question(question,memory,None,'prefix ',5)
            result = online.retrieve_question(question,memory,None,'prefix ',5,PositiveVarianceWeighting(.03,1))
            other_gold = online.retrieve_question({**question,'gold_ids':['Home/M6'],'gold_choice':'D'},
                                                  memory,None,'prefix ',5,PositiveVarianceWeighting(.03,1))
        self.assertEqual(unweighted['conditions'][0],result['conditions'][0])
        self.assertEqual(result['groups'][0]['query'],'Exact query')
        self.assertEqual(result['groups'][0]['size'],7)
        self.assertEqual(result['leaf_scores'],other_gold['leaf_scores'])
        self.assertTrue(all('positive_variance_score' in r and 'white_score' not in r for r in result['leaf_scores']))


if __name__ == '__main__':
    unittest.main()
