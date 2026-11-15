"""Dataset boundaries that must hold before running a long M2A queue."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import m2a_inputs


class GalleryInputsTests(unittest.TestCase):
    def fixture(self, root):
        base = root / 'data/Mem-Gallery/data'
        (base / 'dialog').mkdir(parents=True)
        (base / 'image').mkdir()
        (base / 'image/one.png').write_bytes(b'original image fixture')
        source = base / 'dialog/Example.json'
        source.write_text(json.dumps({
            'multi_session_dialogues': [{'session_id':'D1', 'date':'2024-01-02', 'dialogues': [
                {'round':'D1:1', 'user':'Look at this.', 'assistant':'A photograph.',
                 'input_image':['../image/one.png'], 'image_id':['D1:IMG_001'],
                 'image_caption':['DO NOT IMPORT HISTORY CAPTION']}]}],
            'human-annotated QAs': [{'question':'When was this?', 'question_image':'../image/one.png',
                'image_caption':'DO NOT IMPORT QUERY CAPTION', 'answer':'GOLD SECRET',
                'clue':['D1:1'], 'point':'TR'}]}))
        return source

    def test_original_pixels_and_image_ids_without_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch.object(m2a_inputs, 'ROOT', root):
                obs, qs, sources = m2a_inputs.load_task('memgallery', 'Example')
            self.assertIn('Image ID: D1:IMG_001', obs[0]['text'])
            self.assertNotIn('CAPTION', json.dumps(obs))
            self.assertNotIn('GOLD SECRET', json.dumps(obs))
            self.assertNotIn('CAPTION', json.dumps(qs))
            self.assertEqual(qs[0]['id'], 'Example/QA_0000')
            self.assertEqual(qs[0]['query_image_paths'], [obs[0]['image']])
            self.assertIn(obs[0]['image'], sources)
            projection = {'question':qs[0]['question'], 'images':qs[0]['query_image_paths']}
            self.assertNotIn('GOLD SECRET', json.dumps(projection))

    def test_invalid_gold_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            data = json.loads(source.read_text())
            data['human-annotated QAs'][0]['clue'] = ['D9:99']
            source.write_text(json.dumps(data))
            with patch.object(m2a_inputs, 'ROOT', root), self.assertRaises(AssertionError):
                m2a_inputs.load_task('memgallery', 'Example')

    def test_multiple_images_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            data = json.loads(source.read_text())
            data['multi_session_dialogues'][0]['dialogues'][0]['input_image'] *= 2
            source.write_text(json.dumps(data))
            with patch.object(m2a_inputs, 'ROOT', root), self.assertRaises(ValueError):
                m2a_inputs.load_task('memgallery', 'Example')


if __name__ == '__main__':
    unittest.main()
