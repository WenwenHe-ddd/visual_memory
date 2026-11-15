"""The new backbone must receive no cached Qwen descriptions or answers."""
from pathlib import Path
import tempfile
import unittest

from scripts.data.memeye_backbone_inputs import (
    commit, image_jobs, sanitize_memory, sanitize_question, sha,
)


class MemEyeBackboneInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root/'original.png'
        self.image.write_bytes(b'fixture original image bytes')
        self.image_hash = sha(self.image)
        self.arrival = {
            'memory_id':'Scene/R1','image_id':'Scene/R1#image1','dialogue':'User: Original dialogue.',
            'timestamp':'2026-01-01','insertion_index':1,'sha256':self.image_hash,'source_path':str(self.image),
            'caption':'OLD_QWEN_CAPTION','retrieval_text':'OLD_QWEN_RETRIEVAL','embedding':[1,2,3],
        }
        self.question = {
            'id':'Scene/Q1','task':'Scene','question':'What is visible?',
            'options':{'A':'red','B':'blue','C':'green','D':'white'},
            'query_image_paths':[str(self.image)],'gold_choice':'SECRET_GOLD','gold_ids':['SECRET_GOLD_ID'],
            'query_image_captions':['OLD_QWEN_QUERY_CAPTION'],'retrieval_query':'OLD_QWEN_EXPANDED_QUERY',
        }
        self.candidate = {key:self.question[key] for key in ('id','question','options')}
        self.candidate['candidate_statements'] = {'A':'It is red.','B':'It is blue.','C':'It is green.','D':'It is white.'}

    def test_source_allowlists_remove_every_old_generative_output(self):
        memory = sanitize_memory(self.arrival)
        question = sanitize_question(self.question,self.candidate)
        self.assertEqual(set(memory),{'id','image_id','task','dialogue','timestamp','insertion_index','image_sha256','image_path'})
        self.assertEqual(set(question),{'id','task','question','options','query_images','candidate_statements'})
        self.assertNotIn('OLD_QWEN',str([memory,question]))
        self.assertNotIn('SECRET_GOLD',str([memory,question]))
        self.assertEqual(question['options'],self.question['options'])
        self.assertEqual(question['candidate_statements'],self.candidate['candidate_statements'])

    def test_history_and_query_share_one_caption_request_by_content(self):
        first = sanitize_memory(self.arrival)
        alias = self.root/'same-image-other-path.png'
        alias.write_bytes(self.image.read_bytes())
        second = {**first,'id':'Scene/R2','image_id':'Scene/R2#image1','image_path':str(alias),'insertion_index':2}
        question = sanitize_question(self.question,self.candidate)
        images,requests = image_jobs([first,second],[question],'Fixed caption instruction')
        self.assertEqual(len(images),1)
        self.assertEqual(len(requests),1)
        self.assertEqual(images[0]['history_memory_ids'],['Scene/R1','Scene/R2'])
        self.assertEqual(images[0]['query_question_ids'],['Scene/Q1'])
        self.assertEqual(len(images[0]['source_paths']),2)
        self.assertEqual(set(requests[0]),{'id','user_text','query_images'})
        self.assertEqual(requests[0]['user_text'],'Fixed caption instruction')
        self.assertNotIn('What is visible?',str(requests))
        self.assertNotIn('Original dialogue',str(requests))
        self.assertNotIn('SECRET_GOLD',str(requests))

    def test_changed_image_and_wrong_candidate_options_are_rejected(self):
        memory = sanitize_memory(self.arrival)
        self.image.write_bytes(b'tampered image')
        with self.assertRaisesRegex(ValueError,'Image content differs'):
            image_jobs([memory],[],'Caption')
        candidate = {**self.candidate,'options':{**self.candidate['options'],'A':'altered option'}}
        with self.assertRaisesRegex(ValueError,'Candidate question/options mismatch'):
            sanitize_question(self.question,candidate)

    def test_frozen_model_inputs_cannot_be_overwritten_or_repaired_silently(self):
        artifacts={'source_memories.jsonl':b'original memories\n','protocol.json':b'model-A\n'}
        commit(self.root,artifacts)
        before={name:(self.root/name).stat().st_mtime_ns for name in artifacts}
        commit(self.root,artifacts)
        self.assertEqual(before,{name:(self.root/name).stat().st_mtime_ns for name in artifacts})
        with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
            commit(self.root,{**artifacts,'protocol.json':b'model-B\n'})
        (self.root/'source_memories.jsonl').unlink()
        with self.assertRaisesRegex(ValueError,'Frozen input artifact is missing'):
            commit(self.root,artifacts)
        self.assertFalse((self.root/'source_memories.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
