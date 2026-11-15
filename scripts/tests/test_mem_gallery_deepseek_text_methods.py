import copy
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

from scripts.evaluation import mem_gallery_deepseek_text_methods as run
from scripts.evaluation import gallery_deepseek_memory_transport as api
from scripts.analysis.audit_mem_gallery_deepseek_text_methods import independent_fusion, summary
from scripts.memory.simplemem_fusion import fuse_candidates


class MatchedGalleryTests(unittest.TestCase):
    def test_empty_caption_keeps_original_image_and_round_metadata(self):
        row=dict(id='D1:2',timestamp='2025-01-01',user='Remember.',assistant='OK',
                 raw=dict(image_id=['D1:IMG1'],image_caption=[''],input_image=['image.png']))
        before=copy.deepcopy(row)
        for text in (run.round_text(row,'user','assistant'),run.user_text(row)):
            self.assertIn('Date: 2025-01-01\nRound: D1:2',text)
            self.assertIn('image_id: D1:IMG1\nimage_caption: ',text)
        self.assertEqual(row,before)

    def test_network_failure_is_sanitized_and_retryable(self):
        request=api.build_request([dict(role='user',content='hello')],2048,0.1)
        with patch.object(api,'_validate_config'),patch.object(api.urllib.request,'build_opener') as opener:
            opener.return_value.open.side_effect=urllib.error.URLError('private network detail')
            with self.assertRaises(api.APIError) as caught:
                api.chat_request(SimpleNamespace(api_key='test-only'),request)
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn('private',str(caught.exception))

    def test_cap_policy_preserves_native_temperature(self):
        model=object.__new__(api.Model);model.fatal_errors=[];model.lock=threading.Lock()
        out=dict(finish_reason='stop',refusal=None,raw_response='valid')
        for floor,expected in [(2048,2048),(0,80)]:
            model.min_output_tokens=floor
            with patch.object(api.APIModel,'one',return_value=out) as generate:
                model.one(['messages'],80,0.4)
            generate.assert_called_once_with(['messages'],expected,0.4)

    def test_rrf_audit_catches_context_tampering_and_duplicate_votes(self):
        batches=[dict(channel='semantic',query='q',top_k=10,result_ids=list('abcdef')),
                 dict(channel='keyword',query='q',top_k=5,result_ids=['new','c']),
                 dict(channel='structured',query={'person':'P'},top_k=5,result_ids=['new','b'])]
        batches.append(copy.deepcopy(batches[0]))
        ids=list(dict.fromkeys(i for b in batches for i in b['result_ids']))
        ranked,fusion=fuse_candidates([SimpleNamespace(entry_id=i) for i in ids],batches)
        detail=dict(original_item_ids=ids,search_batches=batches,fusion=fusion,
                    context_memory_ids=[e.entry_id for e in ranked[:5]])
        independent_fusion(detail)
        self.assertEqual(detail['context_memory_ids'][0],'new')
        detail['context_memory_ids'].reverse()
        with self.assertRaises(AssertionError):independent_fusion(detail)

    def test_no_window_based_simplemem_recall_is_reported(self):
        row=dict(gold_count=3,gold_hits=None,source_ids=None,em=1.,f1=1.,bleu_1=1.,
                 bleu_2=1.,bleu=1.,context_count=5,extra_knowledge_count=0,provider_rejected=False)
        result=summary([row])
        self.assertIsNone(result['source_recall_micro'])
        self.assertIsNone(result['source_recall_macro'])
        self.assertEqual(result['correct'],1)


if __name__=='__main__':unittest.main()
