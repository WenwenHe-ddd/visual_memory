import base64
import copy
import unittest

from scripts.evaluation import clip_image_dialogue_api as api


class FrozenReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = api.rows(api.DEFAULT_SOURCE / 'qa_manifest.jsonl')

    def test_native_bytes_interleaving_and_model_settings(self):
        for has_query in (False, True):
            q = next(q for q in self.questions if bool(q['query_image_paths']) == has_query)
            original = copy.deepcopy(q)
            for model in api.MODELS:
                wire = api.build_request(q['reader_messages'], q['reader_image_hashes'], model)
                self.assertEqual(wire['messages'][0], q['reader_messages'][0])
                for old, new in zip(q['reader_messages'][1]['content'], wire['messages'][1]['content'], strict=True):
                    if old['type'] == 'text':
                        self.assertEqual(old, new)
                    else:
                        data = base64.b64decode(new['image_url']['url'].split(',', 1)[1])
                        self.assertEqual(api.hashlib.sha256(data).hexdigest(), q['reader_image_hashes'][old['path']])
                self.assertNotIn('gold_choice', wire)
                self.assertEqual(wire['seed'], 42)
            self.assertEqual(q, original)

    def test_changed_image_rejected(self):
        q = self.questions[0]
        hashes = dict(q['reader_image_hashes'])
        hashes[next(iter(hashes))] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            api.build_request(q['reader_messages'], hashes, 'gpt-5')

    def test_answer_parser_does_not_extract_from_explanations(self):
        self.assertEqual(api.parse_choice('Answer: B'), 'B')
        self.assertIsNone(api.parse_choice('A or B'))


if __name__ == '__main__':
    unittest.main()
