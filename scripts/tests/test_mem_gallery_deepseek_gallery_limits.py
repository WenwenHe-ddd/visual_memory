import json
import unittest
import numpy as np
from scripts.evaluation import mem_gallery_deepseek_gallery_limits as run
from scripts.analysis.audit_mem_gallery_deepseek_gallery_limits import independent_weights
from scripts.memory.online_zscore import ZScoreWeighting


class MasterComparisonTests(unittest.TestCase):
    def test_hypotheses_see_only_question_and_official_caption(self):
        q=dict(id='scene/Q1',question='Where was it?',query_image_caption='A red object.',
            answer='SECRET_GOLD',clue='SECRET_CLUE',history='SECRET_HISTORY',query_image_paths=['SECRET_PATH'])
        for attempt in (1,2,3):
            payload=run.candidate_payload(q,attempt)
            self.assertNotIn('SECRET_',json.dumps(payload))
            self.assertIn(q['question'],payload['user_text'])
            self.assertIn(q['query_image_caption'],payload['user_text'])
            self.assertEqual(payload['query_images'],[])

    def test_independent_dual_ridge_matches_singletons_pairs_and_large_groups(self):
        rng=np.random.default_rng(42)
        for n in (1,2,7,40):
            x=rng.normal(size=(n,18));x/=np.linalg.norm(x,axis=1,keepdims=True)
            q=rng.normal(size=18);q/=np.linalg.norm(q)
            for alpha in (0.,.3,1.):
                white,weighted=independent_weights(x,q,.03,alpha)
                actual=ZScoreWeighting(.03,alpha).apply(q,x)
                np.testing.assert_allclose(white,actual['white'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(weighted,actual['weighted'],atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
