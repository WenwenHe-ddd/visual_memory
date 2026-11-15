import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.memory import clip_caption as method


class ClipCaptionTests(unittest.TestCase):
    def record(self, ident='a', task='A'):
        return dict(id=ident, task=task, image_sha256='hash', caption='A blue chair.',
                    image_ids=['D1:IMG_001'], source_memory_ids=['hidden-round'])

    def test_ranking_uses_only_vectors_with_stable_ties_and_task_boundary(self):
        records=[self.record(str(i), t) for i,t in enumerate(['A','B','A','A'])]
        vectors=np.array([[.8,.6],[1.,0.],[.8,-.6],[0.,1.]],dtype=np.float32)
        before=method.rank(records,vectors,'A',np.array([1.,0.]),2)
        self.assertEqual([i for i,_ in before],[0,2])
        records[3]['caption']='Exactly matches question and correct answer'
        self.assertEqual(before,method.rank(records,vectors,'A',np.array([1.,0.]),2))
        with self.assertRaises(ValueError):
            method.rank(records,vectors,'A',np.array([1.,0.]),4)

    def test_reader_whitelists_captions_ids_question_and_options(self):
        record=self.record()
        record.update(dialogue='hidden-dialogue', source_path='hidden-image.png', date='hidden-date')
        question=dict(question='Which chair?',options={k:k+' option' for k in 'ABCD'},
                      gold_choice='hidden-gold',query_image_paths=['hidden-query.png'],
                      query_image_caption='hidden-query-caption',point='VS')
        opened=method.reader_payload(question,[record],'open')
        mcq=method.reader_payload(question,[record],'mcq')
        for payload in [opened,mcq]:
            text=json.dumps(payload)
            self.assertNotIn('hidden',text)
            self.assertIn('A blue chair.',text)
            self.assertIn('D1:IMG_001',text)
            self.assertIn('Which chair?',text)
        self.assertNotIn('A option',opened['user_text'])
        self.assertIn('A option',mcq['user_text'])

    def test_portable_memory_needs_no_source_images_and_rejects_pixels(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            record=self.record();record['source_path']='/unavailable/source.png'
            method.write_rows(root/'records.jsonl',[method.persistent_record(record)])
            np.save(root/'image_embeddings.npy',np.array([[1.,0.]],dtype=np.float32))
            manifest=dict(offline_method='clip_caption',dimensions=2,candidates=1,
                          files={n:method.sha(root/n) for n in ['records.jsonl','image_embeddings.npy']})
            manifest['memory_signature']=method.digest(manifest)
            method.dump(root/'memory.json',manifest)
            _,records,_=method.load_memory(root)
            self.assertNotIn('source_path',records[0])
            (root/'original.png').write_bytes(b'pixels')
            with self.assertRaises(ValueError):method.load_memory(root)

    def test_actual_batch_interface_supplies_no_vision_arguments(self):
        import torch
        from transformers import BatchEncoding
        class Processor:
            tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2)
            def apply_chat_template(self,messages,**kwargs):
                assert all(isinstance(m['content'],str) for m in messages)
                return json.dumps(messages)
            def __call__(self,**kwargs):
                assert set(kwargs)=={'text','padding','truncation','return_tensors'}
                return BatchEncoding(dict(input_ids=torch.tensor([[3,4]]),attention_mask=torch.tensor([[1,1]])))
            def batch_decode(self,*args,**kwargs):return ['A']
        class Model:
            config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=100))
            generation_config=SimpleNamespace(eos_token_id=2)
            def parameters(self):yield torch.zeros(1)
            def generate(self,**kwargs):
                assert not any('image' in k or 'pixel' in k for k in kwargs)
                return torch.tensor([[3,4,5,2]])
        outputs=method.text_batch((Processor(),Model(),{}),[{'payload':{'system':'s','user_text':'q'}}],4)
        self.assertEqual(outputs[0]['reader_image_count'],0)
        self.assertFalse(outputs[0]['hit_token_limit'])
        self.assertEqual(outputs[0]['generated_tokens'],2)

    def test_cli_routes_explicit_method_and_self_describing_memory(self):
        from scripts import visual_memory_pipeline as cli
        with patch.object(method,'pipeline') as run:
            cli.main(['offline','--memory','clip_caption','--output','/tmp/new-memory'])
        self.assertEqual(run.call_args.args[0].memory,'clip_caption')
        with TemporaryDirectory() as tmp:
            Path(tmp,'memory.json').write_text(json.dumps({'offline_method':'clip_caption'}))
            with patch.object(method,'pipeline') as run:
                cli.main(['online','--memory-path',tmp,'--output','/tmp/new-evaluation'])
            self.assertEqual(run.call_args.args[0].mode,'clip_caption')
            with self.assertRaises(SystemExit):
                cli.main(['online','--memory-path',tmp,'--mode','natural'])


if __name__=='__main__':unittest.main()
