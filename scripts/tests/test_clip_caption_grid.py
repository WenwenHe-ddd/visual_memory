import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from scripts.evaluation import clip_caption_grid as grid
from scripts.evaluation import openai_gpt5_base_transport as gpt
from scripts.evaluation import deepseek_flash_seed42_audited_transport as deep

class ClipCaptionGridTests(unittest.TestCase):
    def test_every_api_seed_is_on_wire_and_message_is_text_only(self):
        payload=dict(system='Read captions.',user_text='Caption: a chair. Question: what object?')
        for model in ['gpt5','deepseek']:
            for seed in grid.SEEDS:
                with patch.object(gpt,'load_config',return_value=SimpleNamespace()),patch.object(deep,'load_config',return_value=SimpleNamespace()):
                    gateway=grid.Gateway(dict(backbone=model,seed=seed))
                body=grid.request(payload,model,seed,gateway)
                self.assertEqual(body['seed'],seed)
                self.assertEqual(body['messages'],[dict(role='system',content=payload['system']),dict(role='user',content=payload['user_text'])])
                self.assertEqual(gateway.wire_from_native(body),body)
                self.assertEqual(body.get('max_tokens',body.get('max_completion_tokens')),2048 if model=='gpt5' else 256)
                with self.assertRaises(ValueError):grid.request(dict(payload,query_images=['secret.png']),model,seed,gateway)
                with self.assertRaises(ValueError):grid.request(payload,model,seed+1,gateway)

    def test_replace_only_caption_keeps_visual_memory_and_missing_caption(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/'source/mem_gallery/memory';original.mkdir(parents=True)
            records=[dict(id='A/sha',task='A',caption='old qwen caption',image_sha256='sha',image_ids=['D1:IMG_001'],source_memory_ids=['A/D1:1'])]
            grid.write_rows(original/'records.jsonl',records)
            np.save(original/'image_embeddings.npy',np.array([[1.,0.]],dtype=np.float32))
            manifest=dict(offline_method='clip_caption',dimensions=2,candidates=1,caption_source_model='Qwen',
                files={n:grid.sha(original/n) for n in ['records.jsonl','image_embeddings.npy']})
            manifest['memory_signature']=grid.digest(manifest);grid.dump(original/'memory.json',manifest)
            old_hashes={p.name:grid.sha(p) for p in original.iterdir()}
            audit=dict(model='deepseek-v4.1-flash',seed=42,rejected_image_hashes=['sha'])
            with patch.object(grid,'SOURCE',root/'source'),patch.object(grid,'caption_cache',return_value=({'sha':''},audit)):
                target,_=grid.memory(root/'output','mem_gallery','deepseek')
            _,result,_=grid.method.load_memory(target)
            self.assertEqual(result[0]['caption'],'')
            self.assertEqual({k:v for k,v in result[0].items() if k!='caption'},{k:v for k,v in records[0].items() if k!='caption'})
            self.assertEqual(grid.sha(target/'image_embeddings.npy'),old_hashes['image_embeddings.npy'])
            self.assertEqual({p.name:grid.sha(p) for p in original.iterdir()},old_hashes)

if __name__=='__main__':unittest.main()
