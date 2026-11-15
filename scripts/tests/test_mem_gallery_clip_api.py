"""Guard the reader-only boundary and provider image encoding."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.evaluation import mem_gallery_clip_image_dialogue_api as api

class FrozenReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name)/'image.png')
        Image.new('RGB',(10,10),'red').save(self.path)
        self.hashes = {self.path:api.sha(self.path)}
        self.messages = [dict(role='system',content='system'),dict(role='user',content=[])]
        for i in range(5):
            self.messages[1]['content'].extend([dict(type='text',text=f'dialogue {i}'),dict(type='image',path=self.path)])
        self.messages[1]['content'].append(dict(type='text',text='Question only'))
        api.encoded_image.cache_clear()

    def test_interleaving_and_source_immutability(self):
        before=copy.deepcopy(self.messages)
        body=api.build_request(self.messages,self.hashes,'gpt-5')
        self.assertEqual(self.messages,before)
        for original,wire in zip(before[1]['content'],body['messages'][1]['content']):
            if original['type']=='text':self.assertEqual(original,wire)
            else:self.assertEqual(wire['type'],'image_url')
        self.assertNotIn('temperature',body)
        self.assertEqual(body['max_completion_tokens'],2048)

    def test_changed_original_rejected(self):
        Path(self.path).write_bytes(b'changed')
        with self.assertRaises(ValueError):api.build_request(self.messages,self.hashes,'gpt-5')

    def test_no_silent_extra_fields_or_history_loss(self):
        self.messages[1]['content'][0]['answer']='forbidden'
        with self.assertRaises(ValueError):api.build_request(self.messages,self.hashes,'gpt-5')
        del self.messages[1]['content'][0]['answer']
        self.messages[1]['content'].pop(1)
        with self.assertRaises(ValueError):api.build_request(self.messages,self.hashes,'gpt-5')
        self.assertEqual(api.settings('deepseek-v4.1-flash')['max_tokens'],256)

if __name__=='__main__':unittest.main()
