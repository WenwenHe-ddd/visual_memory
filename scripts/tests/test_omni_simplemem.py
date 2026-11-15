"""Behavioral guards for isolation, multimodal payloads and retrieval budgets."""
import base64
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.memory import omni_simplemem as omni
from scripts.evaluation.omni_qwen_transport import OmniBroker


class OmniTests(unittest.TestCase):
    def test_upstream_bytes_are_pinned(self):
        manifest = omni.verify_installation()
        self.assertEqual(manifest['version'], '0.3.0')
        self.assertIn('simplemem/multimodal/processors/image_processor.py', manifest['files'])

    def test_keyword_can_displace_semantic_and_duplicates_do_not_vote_twice(self):
        semantic = [dict(id='a'), dict(id='b'), dict(id='c')]
        keywords = [dict(id='c'), dict(id='c'), dict(id='d')]
        selected, scores = omni.fuse_channels(dict(semantic=semantic, keyword=keywords), limit=2)
        self.assertEqual(selected, ['c', 'a'])
        self.assertAlmostEqual(scores['c'], 1 / 63 + 1 / 61)

    def test_image_wire_is_content_addressed_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            broker = OmniBroker.__new__(OmniBroker)
            broker.root = Path(directory)
            broker.log_lock = threading.Lock()
            data = b'local test pixels'
            source = [dict(role='user', content=[dict(type='text', text='before'),
                dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(data).decode())),
                dict(type='text', text='after')])]
            result = broker.normalize(source)
            part = result[0]['content'][1]
            self.assertEqual(Path(part['path']).read_bytes(), data)
            self.assertEqual(part['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(result[0]['content'][0]['text'], 'before')
            self.assertEqual(result[0]['content'][2]['text'], 'after')
            self.assertEqual(source[0]['content'][1]['type'], 'image_url')

    def test_concurrent_identical_requests_generate_once(self):
        class Model:
            calls = 0
            def generate(self, conversations, cap, sampling_temperature=0.):
                self.calls += len(conversations)
                time.sleep(.1)
                return [dict(raw_response='ok', input_tokens=2, generated_tokens=1, hit_token_limit=False)
                        for _ in conversations]
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            broker = OmniBroker(model, Path(directory), dict(generation_batch_size=8, seed=42,
                                                            experiment_signature='dedup-test'))
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    outputs = list(pool.map(lambda _: broker.request([dict(role='user', content='same')], 5), range(4)))
                self.assertEqual(model.calls, 1)
                self.assertEqual(len({r['call_key'] for r in outputs}), 1)
            finally:
                broker.close()

    def test_invalid_entity_shape_is_regenerated_without_inventing_fields(self):
        import json
        class Model:
            calls = 0
            def generate(self, conversations, cap, sampling_temperature=0.):
                self.calls += 1
                text = ('{"entities":["Architects"],"relations":[]}' if self.calls == 1 else
                        '{"entities":[{"name":"Architects","type":"PERSON"}],"relations":[]}')
                return [dict(raw_response=text, input_tokens=2, generated_tokens=20, hit_token_limit=False)]
        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            broker = OmniBroker(model, Path(directory), dict(generation_batch_size=1, seed=42,
                memory_max_new_tokens=4096, experiment_signature='schema-test'))
            try:
                response = broker.client().create(model='local-qwen', response_format={'type':'json_object'},
                    messages=[dict(role='system', content='You are an expert knowledge extraction system.'),
                              dict(role='user', content='Architects design buildings.')])
                self.assertEqual(model.calls, 2)
                self.assertEqual(json.loads(response.choices[0].message.content)['entities'][0]['name'], 'Architects')
                self.assertEqual(broker.errors, [])
                events = json.loads((Path(directory) / 'transport_recoveries.jsonl').read_text())
                self.assertEqual(events['causes'], ['format'])
                first = json.loads((Path(directory) / 'calls' / (events['attempt_keys'][0] + '.json')).read_text())
                self.assertEqual(json.loads(first['output']['raw_response'])['entities'], ['Architects'])
                self.assertFalse(first['output']['structured_output_valid'])
            finally:
                broker.close()

    def test_native_unified_store_returns_visual_maus_for_text_query_vector(self):
        omni.activate()
        from simplemem.multimodal.storage.vector_store import HybridVectorStore
        with tempfile.TemporaryDirectory() as directory:
            store = HybridVectorStore(directory, text_dim=512, visual_dim=512)
            image = [1.] + [0.] * 511
            text = [0., 1.] + [0.] * 510
            # The official orchestrator stores equal-dimension visual units here.
            store.add_text('visual_mau', image)
            store.add_text('text_mau', text)
            self.assertEqual(store.search(image, top_k=1)[0][0], 'visual_mau')

    def test_graph_constructor_has_no_shared_default_directory(self):
        omni.activate(); omni.install_isolation()
        from simplemem.multimodal import orchestrator
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            token = omni._graph_path.set(a)
            try:
                first = orchestrator.KnowledgeGraph()
            finally:
                omni._graph_path.reset(token)
            token = omni._graph_path.set(b)
            try:
                second = orchestrator.KnowledgeGraph()
            finally:
                omni._graph_path.reset(token)
            self.assertEqual(first.storage_path, a)
            self.assertEqual(second.storage_path, b)
            self.assertNotEqual(first.storage_path, second.storage_path)


if __name__ == '__main__':
    unittest.main()
