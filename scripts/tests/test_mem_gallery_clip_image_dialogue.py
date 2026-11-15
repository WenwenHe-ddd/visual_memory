import unittest
import numpy as np
from scripts.evaluation.mem_gallery_clip_image_dialogue import candidates_from, messages_for, rank_images


class ImageRouteTests(unittest.TestCase):
    def test_deduplicates_within_scenario_and_retains_source_dialogues(self):
        def memory(task,mid,image=True):
            return dict(task=task,id=mid,is_visual=image,image_sha256='samebytes',image_path='/tmp/a.png',
                round=mid,timestamp='2026-01-01',image_id=mid+'_IMG',dialogue=mid+' dialogue',caption='DO_NOT_USE_CAPTION')
        candidates=candidates_from([memory('a','a/1'),memory('a','a/2'),memory('b','b/1'),memory('a','a/text',False)])
        self.assertEqual(len(candidates),2)
        self.assertEqual(candidates[0]['source_memory_ids'],['a/1','a/2'])
        self.assertEqual(len(candidates[0]['source_rounds']),2)

    def test_reader_keeps_ids_dates_query_pixels_and_excludes_caption_gold(self):
        m=dict(task='a',id='a/1',is_visual=True,image_sha256='hash',image_path='/tmp/history.png',
            round='S1:R1',timestamp='2026-01-01',image_id='D1:IMG_003',dialogue='User: hi\nAssistant: hello',
            caption='SECRET_CAPTION')
        q=dict(question='Which image?',point='VS',query_image_paths=['/tmp/query.png'],
               query_image_caption='SECRET_QUERY_CAPTION',answer='SECRET_ANSWER',gold_ids=['SECRET_CLUE'])
        messages=messages_for(q,candidates_from([m]))
        text='\n'.join(c['text'] for c in messages[1]['content'] if c['type']=='text')
        for secret in ['SECRET_CAPTION','SECRET_QUERY_CAPTION','SECRET_ANSWER','SECRET_CLUE']:
            self.assertNotIn(secret,text)
        self.assertIn('D1:IMG_003',text)
        self.assertIn('2026-01-01',text)
        self.assertIn('User: hi',text)
        self.assertEqual([c['path'] for c in messages[1]['content'] if c['type']=='image'],['/tmp/history.png','/tmp/query.png'])

    def test_rank_is_same_scenario_and_stable_on_ties(self):
        candidates=[dict(task='a',image_id=str(i)) for i in range(6)]+[dict(task='b',image_id='outside')]
        vectors=np.array([[1.,0.]]*6+[[2.,0.]])
        result=rank_images(dict(task='a'),candidates,vectors,np.array([1.,0.]),5)
        self.assertEqual([r['image_id'] for r in result],list('01234'))


if __name__=='__main__':unittest.main()
