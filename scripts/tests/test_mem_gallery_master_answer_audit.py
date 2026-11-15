"""CPU integration checks for master-conditioned query routing and dispatch.

The encoders are deterministic fixtures; the actual retrieval implementation
and its serialization run unchanged. No checkpoints or source images are read.
"""
import json
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.memory import mem_gallery_master_answer as method
from scripts import visual_memory_pipeline as pipeline


class FakeTokens(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return FakeTokens(input_ids=torch.tensor([[int(text[-1])] for text in texts]))


class FakeClip:
    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids, **kwargs):
        vectors = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        return SimpleNamespace(text_embeds=vectors[input_ids[:, 0]])


class MemGalleryMasterAuditTests(unittest.TestCase):
    def test_master_mapping_and_pure_text_fallback_use_complete_shared_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            memories = [
                {'id': 'Scene/text', 'task': 'Scene', 'is_visual': False,
                 'retrieval_text': 'TEXT'},
                {'id': 'Scene/image_A', 'node_id': 'A', 'task': 'Scene', 'is_visual': True,
                 'caption': 'A caption', 'dialogue': 'A dialogue', 'retrieval_text': 'IMAGE_A'},
                {'id': 'Scene/image_B', 'node_id': 'B', 'task': 'Scene', 'is_visual': True,
                 'caption': 'B caption', 'dialogue': 'B dialogue', 'retrieval_text': 'IMAGE_B'},
            ]
            # Deliberately disagree in master-list order, sidecar row order,
            # image-node order, and memory-source order.
            graph = {'masters': [
                {'id': 'master_B', 'n': 1, 'visual_mean_ref': {'row': 1}},
                {'id': 'master_A', 'n': 1, 'visual_mean_ref': {'row': 0}},
            ], 'image_nodes': [
                {'id': 'B', 'master_id': 'master_B', 'caption': 'B caption', 'dialogue': 'B dialogue'},
                {'id': 'A', 'master_id': 'master_A', 'caption': 'A caption', 'dialogue': 'A dialogue'},
            ]}
            questions = [{'id': f'Scene/QA_{i:04d}', 'task': 'Scene', 'retrieval_query': 'QUERY'}
                         for i in range(1711)]
            candidates = [{'id': q['id'], 'candidate_statements': ['candidate0', 'candidate1',
                                                                   'candidate2', 'candidate3']}
                          for q in questions]
            method.write_lines(output / 'offline/memories.jsonl', memories)
            method.write_lines(output / 'inputs/questions.jsonl', questions)
            method.write_lines(output / 'candidates.jsonl', candidates)
            method.dump(output / 'candidate_generation_protocol.json', {})
            method.dump(output / 'offline/protocol.json', {
                'source_sha256': {}, 'artifact_sha256': {}, 'text_encoder': {},
                'image_encoder': {}, 'query_prefix': '',
            })
            method.dump(output / 'offline/memory/Scene/graph.json', graph)
            np.save(output / 'offline/memory/Scene/master_sums.npy', np.eye(2))
            embeddings = {
                'TEXT': [1., 0., 0.], 'IMAGE_A': [0., 0., 1.], 'IMAGE_B': [0., 1., 0.],
                'QUERY': [.8, .7, .6],
                'QUERY\nCandidate answer: candidate0': [.1, .2, .9],
                'QUERY\nCandidate answer: candidate1': [.2, .95, .1],
                'QUERY\nCandidate answer: candidate2': [1., 0., 0.],
                'QUERY\nCandidate answer: candidate3': [0., 1., 0.],
            }

            def encode(model, texts, prefix=''):
                vectors = np.asarray([embeddings[t] for t in texts])
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

            original_lines = method.lines
            read_paths = []

            def checked_lines(path):
                read_paths.append(str(path))
                self.assertNotIn('gold', str(path))
                return original_lines(path)

            with patch.object(method, 'configure_device'), \
                    patch.object(method, 'load_text_encoder', return_value=object()), \
                    patch.object(method, 'load_clip_text_encoder',
                                 return_value=(FakeClip(), FakeTokenizer(), 77)), \
                    patch.object(method, 'encode_text', side_effect=encode), \
                    patch.object(method, 'lines', side_effect=checked_lines), \
                    patch.object(method, 'analyze_retrieval', return_value='rankings-frozen'):
                self.assertEqual(method.retrieve(output, device='cpu', top_k=5), 'rankings-frozen')

            results = original_lines(output / 'retrieval_results.jsonl')
            self.assertEqual(len(results), 1711)
            for row in results:
                self.assertEqual(row['candidate_count'], 3)
                base, new = row['conditions']
                self.assertEqual(base['top5_memory_ids'], ['Scene/text', 'Scene/image_B', 'Scene/image_A'])
                self.assertEqual(new['top5_memory_ids'], ['Scene/image_B', 'Scene/image_A', 'Scene/text'])
                self.assertEqual(new['top5_scores'][2], base['top5_scores'][0])
                self.assertEqual(set(new['ranking_memory_ids']), set(base['ranking_memory_ids']))
                self.assertEqual([(m['master_id'], m['candidate_index']) for m in row['master_choices']],
                                 [('master_B', 1), ('master_A', 0)])
            self.assertFalse((output / 'inputs/gold.jsonl').exists())

    def test_pipeline_routes_gallery_without_home_defaults(self):
        expected = pipeline.ROOT / 'results/mem_gallery_master_answer_20261021'
        with patch.object(method, 'offline') as offline:
            pipeline.main(['offline', '--memory', 'caption_visual_master', '--dataset', 'mem_gallery'])
            args = offline.call_args.args[0]
            self.assertEqual(args.output, expected)
            self.assertEqual(args.device, 'cuda:0')
            self.assertIsNone(args.image_files)
        with patch.object(method, 'online') as online:
            pipeline.main(['online', '--memory', 'caption_visual_master', '--dataset', 'mem_gallery'])
            args = online.call_args.args[0]
            self.assertEqual(args.output, expected)
            self.assertEqual(args.device, 'cuda:0')
            self.assertEqual(args.top_k, 5)
            self.assertEqual(args.stage, 'prepare')
            self.assertIsNone(args.question_ids)

    def test_pipeline_rejects_nonfive_gallery_retrieval_before_dispatch(self):
        with patch.object(method, 'online') as online, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as raised:
                pipeline.main(['online', '--memory', 'caption_visual_master',
                               '--dataset', 'mem_gallery', '--top-k', '3'])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn('requires --top-k 5', errors.getvalue())
            online.assert_not_called()


if __name__ == '__main__':
    unittest.main()
