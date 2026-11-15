"""Small CPU-only checks for fresh-caption rebuilding and paired ranking."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.memory import memeye_backbone_retrieval as module


class TextEncoder:
    max_seq_length = 12

    def __init__(self):
        self.calls = []

    def tokenizer(self, inputs, **kwargs):
        assert kwargs['truncation'] is False
        return {'input_ids': [[1] * (len(value.split()) + 2) for value in inputs]}

    def encode(self, inputs, **kwargs):
        self.calls.append(list(inputs))
        return np.asarray([[1.0, (sum(map(ord, value)) % 7 + 1) / 10,
                            (len(value) % 5 + 1) / 10] for value in inputs])


class ImageEncoder:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.calls = []

    def encode(self, path):
        index = int(Path(path).stem)
        if index == self.fail_at:
            self.fail_at = None
            raise RuntimeError('simulated interruption')
        self.calls.append(index)
        return np.eye(4)[index]


def fixture_rows(root, count=4):
    rows = []
    for index in range(count):
        image = root / f'{index}.jpg'
        image.write_bytes(f'fake-image-{index}'.encode())
        rows.append({'id': f'T/R{index}', 'image_id': f'T/R{index}#image1', 'task': 'T',
                     'dialogue': f'User: turn {index}', 'timestamp': f'D{index}',
                     'insertion_index': index + 1, 'image_path': str(image),
                     'image_sha256': module.sha(image), 'caption': f'caption {index}',
                     'retrieval_text': f'User: turn {index}\nMemory description: caption {index}'})
    return rows


class FreshCaptionTests(unittest.TestCase):
    def test_digest_matches_root_protocol_and_immutable_outputs_reject_drift(self):
        from scripts.evaluation.memeye_gpt5_reader import digest
        value = {'source': '原文', 'model': 'model-x', 'parameters': {'temperature': 0}}
        self.assertEqual(module.digest(value), digest(value))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'results.jsonl'
            module.write_lines(path, [{'id': 'one'}])
            original = path.stat().st_mtime_ns
            module.write_lines(path, [{'id': 'one'}])
            self.assertEqual(path.stat().st_mtime_ns, original)
            with self.assertRaisesRegex(ValueError, 'overwrite changed artifact'):
                module.write_lines(path, [{'id': 'other'}])

    def test_history_and_query_join_by_image_hash_and_literal_format(self):
        memories = [{'id': 'T/R1', 'image_id': 'T/R1#image1', 'task': 'T',
                     'dialogue': 'User: hello\nAssistant: hi', 'image_sha256': 'h'}]
        questions = [{'id': 'T/Q1', 'task': 'T', 'question': 'Which color?',
                      'options': dict(zip('ABCD', 'rgbx')), 'query_images': [{'path': '/image', 'sha256': 'h'}],
                      'candidate_statements': {key: f'candidate {key}' for key in 'ABCD'}}]
        captions = [{'id': 'h', 'caption': 'Fresh red item.', 'model': 'm',
                     'source_response_record_id': 'r', 'request_sha256': 's'}]
        stored, query, model = module.attach_captions(memories, questions, captions)
        self.assertEqual(model, 'm')
        self.assertEqual(stored[0]['retrieval_text'], 'User: hello\nAssistant: hi\nMemory description: Fresh red item.')
        self.assertEqual(query[0]['retrieval_query'], "Which color?\nquestion's image:\nimage_caption: Fresh red item.")
        self.assertNotIn('caption', memories[0])
        self.assertNotIn('retrieval_query', questions[0])
        altered = copy.deepcopy(memories)
        altered[0]['caption'] = 'old Qwen caption'
        with self.assertRaisesRegex(ValueError, 'old caption'):
            module.attach_captions(altered, questions, captions)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            module.attach_captions(memories, questions, [])
        leaked = copy.deepcopy(questions)
        leaked[0]['gold_choice'] = 'A'
        with self.assertRaisesRegex(ValueError, 'evaluation'):
            module.attach_captions(memories, leaked, captions)

    def test_prefix_counts_toward_bge_limit_before_inference(self):
        encoder = TextEncoder()
        encoder.max_seq_length = 5
        vectors, audit = module.encode_checked(encoder, ['one two'], 'prefix ')
        self.assertEqual(encoder.calls, [['prefix one two']])
        self.assertEqual(audit['max_tokens'], 5)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        with self.assertRaisesRegex(ValueError, 'refusing silent truncation'):
            module.encode_checked(encoder, ['one two'], 'two word prefix ')
        self.assertEqual(len(encoder.calls), 1)


class StreamingBuildTests(unittest.TestCase):
    def test_resume_uses_text_and_master_state_without_old_image_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows(root)
            basis = np.eye(4)[:, :2]
            image = ImageEncoder(fail_at=2)
            text = TextEncoder()
            target = root / 'resumed'
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                module._build_domain(rows, target, basis, image, text, 's')
            self.assertEqual(image.calls, [0, 1])
            checkpoint = module.read(target / 'build_checkpoint.json')
            self.assertEqual(len(checkpoint['graph']['image_nodes']), 2)
            graph, sums, protocol = module._build_domain(rows, target, basis, image, text, 's')
            self.assertEqual(image.calls, [0, 1, 2, 3])
            self.assertFalse((target / 'build_checkpoint.json').exists())
            self.assertTrue(all(set(node) == {'id', 'caption', 'dialogue', 'timestamp', 'visual_coords', 'master_id'}
                                and len(node['visual_coords']) == 2 for node in graph['image_nodes']))
            uninterrupted, expected, _ = module._build_domain(rows, root / 'uninterrupted', basis, ImageEncoder(), TextEncoder(), 's')
            self.assertEqual(graph, uninterrupted)
            np.testing.assert_allclose(sums, expected, atol=1e-12)
            self.assertEqual(protocol['image_arrival_forwards'], 4)
            self.assertEqual(protocol['old_image_reencodes'], 0)
            module._build_domain(rows, target, basis, image, text, 's')
            self.assertEqual(image.calls, [0, 1, 2, 3])
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                module._build_domain(rows, target, basis, image, text, 'changed')
            graph_path = target / 'memory/graph.json'
            graph_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                module._build_domain(rows, target, basis, image, text, 's')

    def test_checkpoint_refuses_changed_caption_or_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = fixture_rows(root, 3)
            basis = np.eye(4)[:, :2]
            with self.assertRaises(RuntimeError):
                module._build_domain(rows, root / 'state', basis, ImageEncoder(fail_at=2), TextEncoder(), 's')
            checkpoint = module.read(root / 'state/build_checkpoint.json')
            changed = copy.deepcopy(rows)
            changed[0]['caption'] = 'other caption'
            with self.assertRaisesRegex(ValueError, 'chronological source'):
                module.restore_checkpoint(checkpoint, basis, changed, TextEncoder(), 's')
            checkpoint['master_sums'][0][0] += 1
            with self.assertRaises(AssertionError):
                module.restore_checkpoint(checkpoint, basis, rows, TextEncoder(), 's')


class RankingTests(unittest.TestCase):
    def test_own_master_candidate_then_global_stable_rank_keeps_reader_text_pure(self):
        memories = [{'id': f'T/R{i}', 'image_id': f'I{i}', 'image_path': f'/I{i}.jpg',
                     'caption': f'caption{i}', 'dialogue': f'dialogue{i}',
                     'retrieval_text': f'dialogue{i}\nMemory description: caption{i}'} for i in range(3)]
        graph = {'image_nodes': [{'id': f'I{i}', 'master_id': 'M0' if i < 2 else 'M1',
                                  'caption': f'caption{i}', 'dialogue': f'dialogue{i}'} for i in range(3)],
                 'masters': [{'id': 'M0', 'n': 2, 'member_ids': ['I0', 'I1'],
                              'visual_mean_ref': {'row': 0, 'scale': .5}},
                             {'id': 'M1', 'n': 1, 'member_ids': ['I2'],
                              'visual_mean_ref': {'row': 1, 'scale': 1.}}]}
        question = {'id': 'T/Q1', 'task': 'T', 'question': 'Which?', 'retrieval_query': 'Which?',
                    'options': dict(zip('ABCD', 'abcd')), 'query_image_paths': [], 'query_images': [],
                    'query_image_captions': [], 'candidate_statements': {key: f'hypothesis_{key}' for key in 'ABCD'}}
        queries = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]])
        leaves = np.asarray([[1, 0], [0, 1], [-1, 0]])
        candidates = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])
        result = module.rank_question(question, memories, graph, np.asarray([[2, 0], [0, 1]]),
                                      queries, leaves, candidates, top_k=2)
        self.assertEqual([row['selected_choice'] for row in result['masters']], ['A', 'B'])
        first, second = result['conditions']
        self.assertEqual(first['top5_memory_ids'], ['T/R0', 'T/R1'])
        self.assertEqual(second['top5_memory_ids'], ['T/R1', 'T/R2'])
        self.assertEqual(second['full_rank_memory_ids'], ['T/R1', 'T/R2', 'T/R0'])
        self.assertEqual(second['top5_scores'], [1, 1])
        self.assertNotIn('hypothesis_', second['reader_context'])
        self.assertNotIn('gold', json.dumps(result))
        self.assertEqual(second['reader_context'], 'Memory 1: dialogue1\nMemory description: caption1\n\nMemory 2: dialogue2\nMemory description: caption2')
        tie = module.rank_question(question, memories, graph, np.asarray([[2, 0], [0, 1]]),
                                  queries, leaves, np.ones((4, 2)), top_k=2)
        self.assertEqual([row['selected_choice'] for row in tie['masters']], ['A', 'A'])


if __name__ == '__main__':
    unittest.main()
