import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts.evaluation import mem_gallery_gpt5_gen_agents as run
from scripts.evaluation import gallery_gpt5_transport as transport


class Tests(unittest.TestCase):
    def test_network_error_is_reported_without_shadowing_http_module(self):
        config = type('C', (), dict(base_url='https://api.openai.com/v1', credential_variable='OPENAI_API_KEY',
                                  model='gpt-5', api_key='test'))()
        wire = transport.build_request([dict(role='user', content='test')], 2048, 0)
        with patch('urllib.request.OpenerDirector.open', side_effect=urllib.error.URLError('offline')):
            with self.assertRaises(transport.APIError) as caught:
                transport.chat(config, wire)
        self.assertEqual(caught.exception.kind, 'network')

    def test_metadata_and_source_text_are_preserved(self):
        row = dict(id='D1:2', timestamp='2025-01-02', user='user text', assistant='assistant text',
                   raw=dict(input_image=['img.jpg'], image_id=['D1:IMG_002'], image_caption=['fresh caption']))
        text = run.round_text(row, 'user', 'assistant')
        for field in ['Date: 2025-01-02', 'Round: D1:2', 'user: user text', 'assistant: assistant text',
                      'image_id: D1:IMG_002', 'image_caption: fresh caption']:
            self.assertIn(field, text)

    def test_answer_keeps_query_image_and_excludes_gold_fields(self):
        qs = run.rows(run.SOURCE / 'questions.jsonl')
        q = next(q for q in qs if q['query_image_paths'])
        contaminated = dict(q, answer='SECRET-GOLD', gold_ids=['SECRET-ID'], clue='SECRET-CLUE')
        payload = run.reader_payload(contaminated, 'fixed context')
        wire = run.wire_for(payload)
        serialized = json.dumps(wire)
        self.assertNotIn('SECRET-', serialized)
        self.assertEqual([v['path'] for v in payload['query_images']], q['query_image_paths'])
        self.assertTrue(any(v['type'] == 'image_url' for v in wire['messages'][-1]['content']))
        self.assertEqual(wire['model'], 'gpt-5')
        self.assertEqual(wire['seed'], 42)
        self.assertNotIn('temperature', wire)

    def test_same_run_cache_rejects_changed_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'response.json'
            p = dict(caption_max_tokens=2048, max_new_tokens=2048, experiment_signature='sig', expected_model='gpt-5-2025-08-07')
            payload = dict(user_text='original', query_images=[])
            run.dump(path, dict(payload=payload, experiment_signature='sig', output=dict(
                request_sha256=run.request_sha256(run.wire_for(payload)), returned_model=p['expected_model'])))
            with self.assertRaises(AssertionError):
                run.cached_completion(None, path, dict(payload, user_text='changed'), p, 'caption', 'x')

    def test_native_ga_reflections_and_captured_context(self):
        import torch
        class FakeModel:
            def __init__(self):
                self.torch = torch
                self.fatal_errors = []
            def log(self, *args):
                pass
            def one(self, messages, cap, temperature):
                prompt = messages[-1]['content']
                raw = ('8' if prompt.startswith('Rate the importance') else
                       'Which item is blue?\nWhich item is red?' if 'recent memories' in prompt else
                       'The blue item is remembered.\nThe red item is remembered.')
                request = transport.build_request(messages, cap, temperature)
                return dict(raw_response=raw, hit_token_limit=False, response_id='test-response', input_tokens=10, generated_tokens=5,
                            request_sha256=run.request_sha256(request))
        embedding = run.native.EMBEDDING / 'snapshots' / (run.native.EMBEDDING / 'refs/main').read_text().strip()
        p = dict(mode='gen_agents', method_defaults=run.native.effective_method_defaults('gen_agents'),
                 embedding_model_path=str(embedding), seed=42, memory_max_new_tokens=2048, experiment_signature='test')
        with tempfile.TemporaryDirectory() as tmp:
            broker = run.make_broker('gen_agents', FakeModel(), Path(tmp), p)
            try:
                run.native.install_runtime(broker, p)
                memory = run.Memory(p, Path(tmp))
                for i in range(12):
                    memory.store(dict(id=f'D1:{i}', timestamp='2025-01-01', speaker='user',
                        user=f'Item {i} is blue.', assistant='Noted.',
                        raw=dict(input_image=[], image_id=[], image_caption=[])))
                context, detail = memory.recall('Which items are blue?')
                notes = memory.core.storage.memory_list
                self.assertEqual(len([n for n in notes if n.get('source') is False]), 12)
                self.assertGreater(len(notes), 12)
                ids = detail['selected_memory_ids']
                self.assertEqual(len(ids), 10)
                full = '\n' + '\n\n'.join(f'[{i}] ' + notes[mid]['text'] for i, mid in enumerate(ids)) + '\n'
                self.assertEqual(context, ' '.join(full.split(' ')[:4000]))
            finally:
                broker.close()


if __name__ == '__main__':
    unittest.main()
