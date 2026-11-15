import threading
import unittest
from unittest.mock import patch

from scripts.evaluation import mem_gallery_gpt5_memoryos as run
from scripts.evaluation.mem_gallery_qwen_text_methods_v2 import gallery_helpers


class MemoryOSBackboneTests(unittest.TestCase):
    def test_input_metadata_matches_existing_gallery_memoryos_control(self):
        row = dict(id='D2:7', timestamp='2025-02-03', user='Keep original question.',
                   assistant='Keep original answer.',
                   raw=dict(input_image=['image.jpg'], image_id=['D2:IMG_007'], image_caption=['GPT caption']))
        self.assertEqual(run.user_text(row), gallery_helpers('memoryos')['_round_user_with_captions'](row))
        for value in ('D2:7', '2025-02-03', 'D2:IMG_007', 'GPT caption', row['user']):
            self.assertIn(value, run.user_text(row))
        self.assertNotIn(row['assistant'], run.user_text(row))  # Passed separately to native add_memory.

    def test_memory_generation_failure_sets_fatal_marker(self):
        for response in [dict(finish_reason='length', raw_response='partial', refusal=None),
                         dict(finish_reason='stop', raw_response='', refusal=None),
                         dict(finish_reason='stop', raw_response='', refusal='refused')]:
            model = object.__new__(run.Model)
            model.lock, model.fatal_errors = threading.Lock(), []
            with patch.object(run.transport.Model, 'one', return_value=dict(response, response_id='test')):
                with self.assertRaises(RuntimeError):
                    model.one([], 2048, 0)
            self.assertEqual(len(model.fatal_errors), 1)

    def test_reader_keeps_original_image_and_excludes_labels(self):
        q = next(q for q in run.rows(run.SOURCE / 'inputs/questions.jsonl') if q['query_image_paths'])
        contaminated = dict(q, answer='SECRET_GOLD', gold_ids=['SECRET_ID'], clue='SECRET_CLUE')
        payload = run.ga.reader_payload(contaminated, 'fixed memory')
        wire = run.images.wire_for(payload)
        import json
        self.assertNotIn('SECRET_', json.dumps(wire))
        self.assertEqual(wire['model'], 'gpt-5')
        self.assertEqual(wire['seed'], 42)
        self.assertNotIn('temperature', wire)
        self.assertEqual([i['path'] for i in payload['query_images']], q['query_image_paths'])


if __name__ == '__main__':
    unittest.main()
