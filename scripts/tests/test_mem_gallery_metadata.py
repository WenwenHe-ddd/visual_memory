import unittest
from scripts.evaluation import mem_gallery_qwen_text_methods_v2 as v2


class MetadataTests(unittest.TestCase):
    def test_all_source_image_ids_and_dates_reach_native_input(self):
        original = v2.base.read(v2.base.OUTPUT / 'observations.json')
        visual = 0
        for method in ('gen_agents', 'memoryos'):
            helper = v2.gallery_helpers(method)
            for scenario in original.values():
                for row in scenario:
                    if method == 'gen_agents':
                        text = helper['_build_round_text'](row, row['speaker'], 'assistant')
                        self.assertIn(row['assistant'].strip(), text)
                    else:
                        text = helper['_round_user_with_captions'](row)
                    self.assertIn('Date: ' + row['timestamp'], text)
                    self.assertIn('Round: ' + row['id'], text)
                    self.assertIn(row['user'].strip(), text)
                    for image_id, caption in zip(row['raw']['image_id'], row['raw']['image_caption']):
                        self.assertIn('image_id: ' + image_id, text)
                        self.assertIn('image_caption: ' + caption, text)
                        visual += 1
        self.assertEqual(visual, 2 * 1003)

    def test_native_simplemem_and_amem_include_image_ids(self):
        original = v2.base.read(v2.base.OUTPUT / 'observations.json')
        for method, name in [('simplemem', '_build_round_text'), ('a_mem', 'build_a_mem_note_text')]:
            helper = v2.gallery_helpers(method)
            for scenario in original.values():
                for row in scenario:
                    text = helper[name](row, row['speaker'], 'assistant')
                    for image_id in row['raw']['image_id']:
                        self.assertIn('image_id: ' + image_id, text)


if __name__ == '__main__':
    unittest.main()
