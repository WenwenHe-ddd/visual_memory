import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
from scripts.memory import hippomm_gallery_transport as transport, hippomm_gallery_api as method


class GalleryAPITests(unittest.TestCase):
    def test_gallery_dispatch_preserves_memeye_adapter(self):
        from scripts import visual_memory_pipeline as pipeline
        with patch.object(method, 'pipeline', return_value='gallery') as target:
            self.assertEqual(pipeline.main(['online', '--memory', 'hippomm_api', '--dataset', 'mem_gallery']), 'gallery')
            self.assertEqual(target.call_args.args[0].dataset, 'mem_gallery')

    def test_mime_uses_content_not_extension(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'wrong.jpg'
            Image.new('RGB', (8, 6), 'blue').save(p, format='WEBP')
            wire, encodings = transport.build_wire(transport.gpt, 'gpt5', 'hello', [p], 2048)
            image = wire['messages'][-1]['content'][0]['image_url']
            self.assertTrue(image['url'].startswith('data:image/webp;base64,'))
            self.assertEqual(encodings[0]['dimensions'], [8, 6])
            self.assertFalse(encodings[0]['reencoded'])

    def test_oversized_image_keeps_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / 'img.png'
            Image.new('RGB', (20, 16), 'red').save(p)
            with patch.object(transport, 'MAX_IMAGE_BYTES', 1):
                data, meta = transport.encoded_image(p, 'deepseek')
            self.assertTrue(data.startswith(b'\xff\xd8\xff'))
            self.assertTrue(meta['reencoded'])
            self.assertEqual(meta['dimensions'], [20, 16])
            self.assertFalse(meta['resized'])
            self.assertFalse(meta['cropped'])

    def test_provider_rejection_is_cached_and_never_resubmitted(self):
        error = transport.deep.APIError('bad_request', retryable=False, message='rejected')
        error.provider_error_code = 'data_inspection_failed'
        with tempfile.TemporaryDirectory() as root, \
             patch.object(transport.deep, 'load_config', return_value=None), \
             patch.object(transport.deep_chat, 'chat', side_effect=error) as call:
            for _ in range(2):
                with self.assertRaises(transport.ProviderRejected):
                    transport.APIChat(root, 'deepseek', '/unused')('fast', 'question')
            self.assertEqual(call.call_count, 1)
            row = json.loads((Path(root) / 'call_0001.json').read_text())
            self.assertEqual(row['response']['raw_response'], '')
            self.assertEqual(row['response']['status'], 'provider_rejection')
            self.assertNotIn('sha256', json.dumps(row))

    def test_gallery_fast_answer_is_audited_without_mcq_call(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            args = SimpleNamespace(output=root, memory_dir=root/'offline', hippomm_backbone='gpt5',
                                   hippomm_gallery_inputs=root, dataset='mem_gallery')
            q = {'id': 'A/Q1', 'task': 'A', 'point': 'AR', 'query_image_paths': [], 'question': 'What?'}
            fast = {'answer': 'blue', 'confidence': .9, 'evidence_event_ids': []}
            response = {'id': q['id'], 'task': 'A', 'backbone': 'gpt5', 'model': transport.MODELS['gpt5'],
                        'route': 'fast', 'selected_event_ids': [], 'context_round_ids': [], 'answer': 'blue',
                        'llm_calls': ['call.json'], 'fast_context_event_ids': [], 'fast': fast,
                        'reader_image_order': [], 'historical_image_reads': 0}
            call = {'request': {'backbone': 'gpt5', 'role': 'fast', 'image_paths': []},
                    'response': {'returned_model': transport.MODELS['gpt5'], 'finish_reason': 'stop',
                                 'raw_response': json.dumps(fast)}, 'image_encoding': []}
            (root/'call.json').write_text(json.dumps(call))
            (root/'qa_responses.jsonl').write_text(json.dumps(response)+'\n')
            memory = {'model': transport.MODELS['gpt5'], 'events': 0, 'original_image_bytes': 0,
                      'memory_artifact_bytes': 1, 'caption_source': 'fixture'}
            with patch.object(method, 'load_run', return_value=({'evaluation_source': str(root)}, memory, [], {}, {}, {}, [q])), \
                 patch.object(method, 'caption_source', return_value=(root, {})), \
                 patch.object(method.inputs, 'evaluation_gold', return_value={q['id']: {'gold_ids': [], 'answer': 'blue'}}):
                method.analyze(args)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['correct'], 1)
            self.assertEqual(summary['bleu_1'], 1.)

if __name__ == '__main__':
    unittest.main()
