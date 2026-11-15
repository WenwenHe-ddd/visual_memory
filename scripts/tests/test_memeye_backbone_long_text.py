"""No-model tests for the long-input recovery policy."""
import unittest

import numpy as np

from scripts.memory import memeye_backbone_long_text as module


class Tokenizer:
    truncation_side = 'right'
    def __call__(self, texts, add_special_tokens=True, truncation=False):
        assert truncation is False
        def encode(text):
            ids = [ord(word[0]) for word in text.split()]
            return [101, *ids, 102] if add_special_tokens else ids
        return {'input_ids': [encode(text) for text in texts] if isinstance(texts, list) else encode(texts)}

    def num_special_tokens_to_add(self, pair=False):
        return 2


class Encoder:
    max_seq_length = 6
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.asarray([[min(len(text.split()), self.max_seq_length - 2), 1.] for text in texts])


class CompleteEncodingTests(unittest.TestCase):
    def test_short_texts_identical_to_original_path(self):
        texts = ['a b', 'c']
        model = Encoder()
        expected, _ = module.strict.encode_checked(model, texts, 'p ')
        actual, audit = module.encode_complete(model, texts, 'p ', ids=['x', 'y'])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(audit['truncated_inputs'], 0)

    def test_long_input_preserves_full_text_and_uses_original_model_limit(self):
        model = Encoder()
        actual, audit = module.encode_complete(model, ['a b c d e f', 'x'], ids=['M1', 'M2'], field='dialogue+caption')
        self.assertEqual(model.calls[0][0], ['a b c d e f', 'x'])
        self.assertEqual(model.calls[0][1], {'batch_size': 16, 'normalize_embeddings': True,
                                           'convert_to_numpy': True, 'show_progress_bar': False, 'prompt': ''})
        np.testing.assert_allclose(actual[0], module.strict.normalize([4., 1.]))
        row = audit['affected_inputs'][0]
        self.assertEqual(row['id'], 'M1')
        self.assertEqual(row['original_tokens'], 8)
        self.assertEqual(row['encoded_tokens'], 6)
        self.assertEqual(row['dropped_tokens'], 2)
        self.assertEqual(audit['truncated_inputs'], 1)

    def test_query_prefix_counted_once_in_original_limit(self):
        model = Encoder()
        vectors, audit = module.encode_complete(model, ['a b c d e'], 'p ', ids=['Q1/A'], field='query')
        self.assertEqual(model.calls[0][0], ['p a b c d e'])
        np.testing.assert_allclose(vectors[0], module.strict.normalize([4., 1.]))
        self.assertEqual(audit['affected_inputs'][0]['original_tokens'], 8)
        self.assertEqual(audit['affected_inputs'][0]['encoded_tokens'], 6)

    def test_adapter_source_never_calls_image_encoder_or_graph_builder(self):
        import ast
        from pathlib import Path
        tree = ast.parse(Path(module.__file__).read_text())
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertNotIn('ImageEncoder', calls)
        self.assertNotIn('_build_domain', calls)
        self.assertNotIn('rebuild_and_retrieve', calls)


if __name__ == '__main__':
    unittest.main()
