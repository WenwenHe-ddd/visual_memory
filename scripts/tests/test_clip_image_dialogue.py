import copy
import unittest
from unittest.mock import patch

from scripts.memory import clip_image_dialogue as dialogue
from scripts.memory import clip_image_memory as original


class ClipImageDialogueTests(unittest.TestCase):
    def setUp(self):
        self.row = {'question': 'Which object?', 'options': {k: k + ' option' for k in 'ABCD'},
                    'query_image_paths': ['query.png'], 'gold_choice': 'secret_gold',
                    'retrieved_images': [
                        {'path': 'one.png', 'image_id': 'one', 'source_memory_ids': ['T/R1', 'T/R3']},
                        {'path': 'two.png', 'image_id': 'two', 'source_memory_ids': ['T/R2']}],
                    'dialogue_by_memory_id': {'T/R1': 'User: first\nAssistant: first reply',
                                             'T/R3': 'User: third\nAssistant: third reply',
                                             'T/R2': 'User: second\nAssistant: second reply'}}

    def test_dialogue_insertions_preserve_every_original_payload_block(self):
        before = copy.deepcopy(self.row)
        base = original.reader_messages(self.row, self.row['retrieved_images'], 'same system')
        updated = dialogue.reader_messages(self.row, 'same system')
        retained = [x for x in updated[1]['content'] if not x.get('text', '').startswith('Dialogue associated')]
        self.assertEqual(retained, base[1]['content'])
        self.assertEqual(self.row, before)
        self.assertNotIn('secret_gold', str(updated))
        self.assertEqual(updated[1]['content'][2], {'type': 'image', 'path': 'one.png'})
        self.assertEqual(updated[1]['content'][3]['text'],
                         'Dialogue associated with historical image 1:\nUser: first\nAssistant: first reply\n\nUser: third\nAssistant: third reply')
        self.assertEqual(updated[1]['content'][6]['text'],
                         'Dialogue associated with historical image 2:\nUser: second\nAssistant: second reply')

    def test_only_corresponding_image_rounds_are_attached(self):
        records = {mid: {'text': text, 'image_ids': ['two' if mid == 'T/R2' else 'one']}
                   for mid, text in self.row['dialogue_by_memory_id'].items()}
        records['T/unretrieved'] = {'text': 'UNRETRIEVED', 'image_ids': ['three']}
        self.assertEqual(dialogue.attach_dialogues(self.row, records), self.row['dialogue_by_memory_id'])
        records['T/R1']['image_ids'] = ['three']
        with self.assertRaisesRegex(ValueError, 'different image'):
            dialogue.attach_dialogues(self.row, records)

    def test_cli_routes_dialogue_variant(self):
        from scripts import visual_memory_pipeline as pipeline
        with patch.object(dialogue, 'pipeline') as run:
            pipeline.main(['online', '--mode', 'clip_image_dialogue', '--stage', 'all', '--top-k', '5'])
        self.assertEqual(run.call_args.args[0].mode, 'clip_image_dialogue')


if __name__ == '__main__':
    unittest.main()
