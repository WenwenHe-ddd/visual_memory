"""Representative image order, frozen evidence and dispatch isolation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.memory import hippomm as base, hippomm_original as method


class OriginalTests(unittest.TestCase):
    def fixture(self):
        question = {'id': 'A/Q1', 'task': 'A', 'question': 'What color?',
                    'query_image_paths': ['/query.png'], 'options': {'A': 'red', 'B': 'blue'},
                    'gold': 'SECRET_GOLD'}
        events = {eid: {'event_id': eid, 'task': 'A', 'representative_image': 'I' + eid[1:],
                       'caption': 'A chair.', 'summary': 'User likes the chair.',
                       'source_round_ids': ['A/r1'], 'source_images': [
                           {'image_id': 'D1:IMG_001', 'relation': 'representative'},
                           {'image_id': 'D1:IMG_002', 'relation': 'visually_redundant'}]}
                  for eid in ['E0001', 'E0002']}
        images = {eid: {'file': 'images/' + eid + '.png'} for eid in events}
        dialogues = {'A/r1': {'id': 'A/r1', 'dialogue': 'Original dialogue', 'images': []}}
        trace = {'route': 'detailed', 'selected_event_ids': ['E0002', 'E0001']}
        return question, trace, events, images, dialogues

    def test_image_order_matches_ranked_events_and_preserves_query_first(self):
        q, trace, events, images, dialogues = self.fixture()
        prompt, paths, order = method.reader_payload(q, trace, events, images, dialogues, '/store')
        self.assertEqual(paths, ['/query.png', '/store/images/E0002.png', '/store/images/E0001.png'])
        self.assertEqual([r.get('event_id') for r in order], [None, 'E0002', 'E0001'])
        self.assertEqual(order[1]['source_image_ids'], ['D1:IMG_001'])
        self.assertIn('Original dialogue', prompt)
        self.assertIn('visually_redundant', prompt)
        self.assertNotIn('SECRET_GOLD', prompt)
        self.assertNotIn('Original historical images are unavailable', prompt)

    def test_empty_selection_does_not_invent_history_images(self):
        q, trace, events, images, dialogues = self.fixture()
        trace['selected_event_ids'] = []
        _, paths, order = method.reader_payload(q, trace, events, images, dialogues, '/store')
        self.assertEqual(paths, ['/query.png'])
        self.assertEqual(len(order), 1)

    def test_fast_and_cross_task_payloads_rejected(self):
        q, trace, events, images, dialogues = self.fixture()
        trace['route'] = 'fast'
        with self.assertRaises(ValueError):
            method.reader_payload(q, trace, events, images, dialogues, '/store')
        trace['route'] = 'detailed'
        events['E0002']['task'] = 'B'
        with self.assertRaises(ValueError):
            method.reader_payload(q, trace, events, images, dialogues, '/store')

    def test_retrieval_parity_and_fast_answer_are_required(self):
        original = {'id': 'q', 'route': 'fast', 'selected_event_ids': ['E0001'], 'answer': 'A'}
        method.check_trace(copy.deepcopy(original), original)
        with self.assertRaises(ValueError):
            method.check_trace({**original, 'answer': 'B'}, original)
        with self.assertRaises(ValueError):
            method.check_trace({**original, 'selected_event_ids': []}, original)
        empty = {**original, 'route': 'detailed', 'selected_event_ids': []}
        with self.assertRaises(ValueError):
            method.check_trace({**empty, 'answer': 'B'}, empty)

    def test_offline_copy_keeps_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / 'source.png', root / 'memory/images/E0001.png'
            source.write_bytes(b'original-image-bytes')
            method.copy_exact(source, target)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            source.unlink()
            self.assertEqual(target.read_bytes(), b'original-image-bytes')

    def test_dispatch_is_independent_of_no_original_method(self):
        from scripts import visual_memory_pipeline as cli
        with patch.object(method, 'prepare') as prep, patch.object(base, 'pipeline') as old:
            cli.main(['online', '--memory', 'hippomm_original', '--dataset', 'mem_gallery'])
        args = prep.call_args.args[0]
        self.assertIsNone(args.question_ids)
        self.assertEqual(args.top_k, 5)
        old.assert_not_called()
        with patch.object(method, 'build') as build, patch.object(base, 'pipeline') as old:
            cli.main(['offline', '--memory', 'hippomm_original'])
        build.assert_called_once()
        old.assert_not_called()


if __name__ == '__main__':
    unittest.main()
