import unittest
from unittest.mock import patch

import numpy as np

from scripts.memory import clip_image_memory as baseline


class ClipImageTests(unittest.TestCase):
    def test_pool_excludes_query_only_images_and_deduplicates_with_provenance(self):
        memories = [{'id': 'A/R1', 'task': 'A', 'images': ['same']},
                    {'id': 'A/R2', 'task': 'A', 'images': ['same']},
                    {'id': 'B/R1', 'task': 'B', 'images': ['other']}]
        images = [{'id': i, 'path': i + '.png'} for i in ['same', 'other', 'query_only']]
        pool = baseline.image_candidates(memories, images, {'A'})
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]['source_memory_ids'], ['A/R1', 'A/R2'])
        self.assertEqual(pool[0]['image_id'], 'same')

    def test_ranking_cannot_use_other_task_or_gold_or_options(self):
        pool = [{'task': t, 'image_id': str(i)} for i, t in enumerate(['A', 'B', 'A', 'A'])]
        vectors = np.array([[0.8, 0.6], [1., 0.], [0.8, -0.6], [0., 1.]])
        q = {'task': 'A', 'gold_ids': ['B/secret'], 'options': {'A': 'secret'}}
        ranked = baseline.rank_images(q, pool, vectors, np.array([1., 0.]), 2)
        self.assertEqual([r['image_id'] for r in ranked], ['0', '2'])
        with self.assertRaises(ValueError):
            baseline.rank_images(q, pool, vectors, np.array([1., 0.]), 4)

    def test_reader_separates_query_image_and_has_no_caption_dialogue_or_gold(self):
        q = {'question': 'Which object?', 'options': {k: k + ' option' for k in 'ABCD'},
             'query_image_paths': ['query.png'], 'gold_choice': 'secret_gold',
             'caption': 'secret_caption', 'dialogue': 'secret_dialogue'}
        retrieved = [{'path': f'{i}.png', 'source_memory_ids': ['secret_round']} for i in range(5)]
        messages = baseline.reader_messages(q, retrieved, 'Choose a letter')
        content = messages[1]['content']
        self.assertEqual([c['path'] for c in content if c['type'] == 'image'],
                         ['0.png', '1.png', '2.png', '3.png', '4.png', 'query.png'])
        text = '\n'.join(c['text'] for c in content if c['type'] == 'text')
        self.assertIn('Historical image 5:', text)
        self.assertIn('Question image 1', text)
        self.assertIn('D. D option', text)
        self.assertNotIn('secret', text)

    def test_cli_routes_only_selected_clip_image_method(self):
        from scripts import visual_memory_pipeline as pipeline
        with patch.object(baseline, 'pipeline') as run:
            pipeline.main(['online', '--mode', 'clip_image', '--stage', 'all', '--top-k', '5'])
        self.assertEqual(run.call_args.args[0].mode, 'clip_image')
        self.assertEqual(run.call_args.args[0].top_k, 5)


if __name__ == '__main__':
    unittest.main()
