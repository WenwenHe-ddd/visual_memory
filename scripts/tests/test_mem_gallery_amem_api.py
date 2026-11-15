import unittest
from types import SimpleNamespace
from unittest.mock import patch
from scripts.evaluation import mem_gallery_amem_api as run

class AMemGalleryTests(unittest.TestCase):
    def test_context_reconstruction_matches_real_bounded_link_order(self):
        notes={str(i):dict(id=str(i),timestamp='2024-01-01',content='source '+str(i),context='topic',keywords=['k'],tags=['t'],links=[]) for i in range(7)}
        notes['0']['links']=['5','6']
        system=SimpleNamespace(memories={k:SimpleNamespace(**v) for k,v in notes.items()},retriever=SimpleNamespace(search=lambda q,k:[0,1,2,3,4]))
        context,detail=run.base.native.bounded_a_mem_context(system,'query',5,5)
        self.assertEqual(detail['context_memory_ids'],['0','5','6','1','2'])
        self.assertEqual(context,run.render_notes(notes,detail['context_memory_ids']))
    def test_gpt_routing_uses_gpt_transport(self):
        original=(run.runner.memory_api,run.runner.reader_api,run.runner.images,run.runner.NativeMemory)
        try:
            run.configure(dict(backbone='gpt5',expected_model='gpt-5-2025-08-07'))
            with patch.object(run.gpt,'Model') as model:
                run.runner.memory_api.Model('root','env',8,2048)
                model.assert_called_once_with('root','env','gpt-5-2025-08-07')
            self.assertIs(run.runner.images,run.gpt_images)
        finally:run.runner.memory_api,run.runner.reader_api,run.runner.images,run.runner.NativeMemory=original
    def test_gallery_reader_does_not_introduce_hypotheses_or_options(self):
        q=dict(id='task/QA_1',task='task',point='FR',question='What happened?',query_image_paths=[],query_image_caption='',retrieval_query='What happened?')
        payload=run.base.reader_payload(q,'verified memory')
        self.assertIn('verified memory',payload['user_text'])
        self.assertIn(q['question'],payload['user_text'])
        self.assertNotIn('answer',payload)
        self.assertEqual(payload['query_images'],[])
if __name__=='__main__':unittest.main()
