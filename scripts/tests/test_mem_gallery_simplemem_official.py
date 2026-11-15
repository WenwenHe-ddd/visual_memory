import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.evaluation.mem_gallery_simplemem_official import GalleryMemory
from scripts.memory.simplemem_official import OfficialSimpleMemMemory

class GalleryOfficialTests(unittest.TestCase):
    def test_original_image_ids_captions_and_metadata_reach_official_store(self):
        row=dict(id='D2:4',session_id='D2',timestamp='2025-02-03',speaker='user',user='Remember these.',assistant='OK',raw=dict(image_id=['D2:IMG_01','D2:IMG_02'],image_caption=['Red car.','Blue bike.']))
        original=copy.deepcopy(row)
        memory=object.__new__(GalleryMemory)
        with patch.object(OfficialSimpleMemMemory,'store') as store:
            memory.store(row)
        ingested=store.call_args.args[0]
        self.assertEqual(ingested['raw']['image_caption'],['image_id: D2:IMG_01\nimage_caption: Red car.','image_id: D2:IMG_02\nimage_caption: Blue bike.'])
        self.assertIn('Date: 2025-02-03\nRound: D2:4\nRemember these.',ingested['user'])
        self.assertEqual(row,original)
    def test_finalize_precedes_retrieval(self):
        memory=object.__new__(GalleryMemory)
        events=[]
        with patch.object(OfficialSimpleMemMemory,'finalize',side_effect=lambda:events.append('finalize')),patch.object(OfficialSimpleMemMemory,'recall',side_effect=lambda q:events.append(q)):
            memory.recall('question without gold')
        self.assertEqual(events,['finalize','question without gold'])
if __name__=='__main__':unittest.main()
