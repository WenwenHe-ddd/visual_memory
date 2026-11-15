import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.evaluation.deepseek_text_memory_transport import build_request, chat_request
from scripts.evaluation.memeye_deepseek_text_methods import bounded_simplemem, base


@dataclass
class Retrieval:
    items: list
    expansion_candidates: list
    def to_dict(self):
        return dict(items=self.items, expansion_candidates=self.expansion_candidates)


class DeepSeekTextTests(unittest.TestCase):
    def test_memoryos_single_summary_normalization_is_lossless(self):
        from scripts.evaluation.continue_deepseek_memoryos_format import normalize_summaries
        original={'summary':'Exact generated content.','keywords':['King Guz','Alley Oop']}
        before=copy.deepcopy(original)
        normalized,changed=normalize_summaries(original)
        self.assertTrue(changed)
        self.assertEqual(normalized,[{'content':original['summary'],'keywords':original['keywords']}])
        self.assertEqual(original,before)
        valid=[{'theme':'topic','content':'content','keywords':[]}]
        self.assertEqual(normalize_summaries(valid),(valid,False))
        self.assertEqual(normalize_summaries('invalid'),([],True))

    def test_streaming_broker_does_not_wait_for_slow_neighbor(self):
        from scripts.evaluation.continue_deepseek_text_methods_streaming import StreamingBroker
        started=threading.Event();release=threading.Event()
        class Model:
            def one(self,messages,cap,temperature):
                if messages[0]['content']=='slow':
                    started.set();release.wait(5)
                return dict(raw_response='A',input_tokens=1,generated_tokens=1,hit_token_limit=False)
        with tempfile.TemporaryDirectory() as directory:
            broker=StreamingBroker(Model(),Path(directory),dict(seed=42,experiment_signature='test'),4)
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    slow=pool.submit(broker.request,[dict(role='user',content='slow')],16)
                    self.assertTrue(started.wait(2))
                    fast=pool.submit(broker.request,[dict(role='user',content='fast')],16)
                    self.assertEqual(fast.result(timeout=2)['raw_response'],'A')
                    self.assertFalse(slow.done())
                    release.set()
                    self.assertEqual(slow.result(timeout=2)['raw_response'],'A')
            finally:
                release.set();broker.close()

    def test_wire_seed_and_native_temperature(self):
        messages=[dict(role='system',content='schema'),dict(role='assistant',content='memory'),dict(role='user',content='query')]
        request=build_request(messages,1000,0.7)
        self.assertEqual(request['seed'],42)
        self.assertEqual(request['temperature'],0.7)
        self.assertEqual(request['model'],'deepseek-v4.1-flash')
        self.assertFalse(request['enable_thinking'])
        self.assertEqual(request['messages'],messages)

    def test_no_metadata_or_native_images_in_text_messages(self):
        for messages in [[dict(role='user',content='q',gold_choice='A')],
                         [dict(role='user',content=[{'type':'image_url'}])],
                         [dict(role='tool',content='q')],[]]:
            with self.assertRaises(ValueError):build_request(messages,16,0)

    def test_wire_seed_cannot_be_overridden(self):
        request=build_request([dict(role='user',content='q')],16,0)
        request['seed']=0
        with patch('scripts.evaluation.deepseek_text_memory_transport._validate_config'):
            with self.assertRaises(ValueError): chat_request(None,request)

    def test_final_cap_preserves_order_and_complete_units(self):
        items=[{'id':str(i),'summary':'complete'+str(i),'details':{'text':'long body'}} for i in range(8)]
        source=Retrieval([items[0],items[0],*items[1:]],['0','6'])
        bounded,detail=bounded_simplemem(source)
        self.assertEqual(bounded.items,items[:5])
        self.assertEqual(bounded.expansion_candidates,['0'])
        self.assertEqual(detail['context_memory_count'],5)
        self.assertEqual(len(source.items),9)

    def test_sparse_and_empty_context_are_never_filled(self):
        for n in [0,1,3]:
            source=Retrieval([dict(id=str(i),summary='text') for i in range(n)],[])
            bounded,detail=bounded_simplemem(source)
            self.assertEqual(bounded.items,source.items)
            self.assertEqual(detail['context_memory_count'],n)
            self.assertEqual(detail['source_empty_preserved'],n==0)

    def test_gold_and_options_do_not_affect_retrieval(self):
        q=dict(question='Which cup?',options=dict(A='one',B='two',C='three',D='four'),query_image_captions=['Blue cup.'])
        altered=copy.deepcopy(q)
        altered.update(gold_choice='B',gold_ids=['secret'],options={'A':'CHANGED'})
        self.assertEqual(base.retrieval_query(q),base.retrieval_query(altered))
        p=dict(reader_system_prompt='Choose a letter.')
        with_gold=dict(q,gold_choice='C',gold_ids=['hidden'])
        self.assertEqual(base.reader_messages(q,'memory',p),base.reader_messages(with_gold,'memory',p))


if __name__=='__main__':unittest.main()
