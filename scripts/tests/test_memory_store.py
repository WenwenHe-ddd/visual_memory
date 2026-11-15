"""Self-describing memory path, method compatibility and frozen encoding policy."""
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from scripts import visual_memory_pipeline as pipeline
from scripts.memory import memory_store, online_retrieval as online


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'; self.source.mkdir()
        self.leaf = {'image_id': 'Home/M1#image1', 'memory_id': 'Home/M1', 'dialogue': 'dialogue', 'caption': 'caption'}
        (self.source / 'inputs.json').write_text(json.dumps({'leaves': [self.leaf], 'questions': [{'gold_choice': 'A'}]}))
        (self.source / 'protocol.json').write_text(json.dumps({'text_encoder': {}, 'query_prefix': 'prefix '}))
        self.store = self.root / 'store'

    def test_offline_export_and_online_path_infer_method_without_external_source(self):
        with patch('builtins.print'):
            pipeline.main(['offline', '--memory', 'natural', '--memory-source', str(self.source), '--output', str(self.store)])
        manifest = memory_store.read_manifest(self.store)
        self.assertEqual(manifest['offline_method'], 'natural')
        self.assertNotIn('questions', (self.store / 'leaves.json').read_text())
        # Loading does not depend on the old source, only on the selected memory path.
        (self.source / 'inputs.json').unlink()
        args = SimpleNamespace(mode='natural', memory_dir=self.store)
        memory, _, _, devices = online.load_memory(args, [{'id': 'Home/Q1'}])
        self.assertEqual(memory.leaves, [self.leaf]); self.assertEqual(devices, {})
        for weighting in ('none', 'z_score'):
            with patch.object(online, 'online') as dispatch:
                pipeline.main(['online', '--memory-path', str(self.store), '--online-weighting', weighting])
            args = dispatch.call_args.args[0]
            self.assertEqual((args.mode, args.memory_dir, args.retrieval, args.online_weighting),
                             ('natural', self.store, 'full_scan', weighting))
        with patch.object(online, 'online') as dispatch:
            pipeline.main(['online', '--memory-path', str(self.store)])
        self.assertEqual(dispatch.call_args.args[0].online_weighting, 'none')

    def test_wrong_method_and_unsupported_retrieval_fail_before_model_load(self):
        memory_store.export_store(self.source, self.store, 'natural')
        for flags in (['--memory', 'caption_visual_master'], ['--retrieval', 'multi_path'],
                      ['--memory-dir', '/tmp/another-memory']):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), patch.object(online, 'online') as run:
                pipeline.main(['online', '--memory-path', str(self.store), *flags])
            run.assert_not_called()

    def test_memory_integrity_and_no_overwrite(self):
        memory_store.export_store(self.source, self.store, 'natural')
        with self.assertRaises(FileExistsError):
            memory_store.export_store(self.source, self.store, 'natural')
        (self.store / 'leaves.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'differs from its manifest'):
            memory_store.read_manifest(self.store)

    def test_historical_truncation_is_explicit_and_text_pinned(self):
        memory = online.NaturalCaptionMemory([self.leaf])
        memory.allowed_leaf_truncations = [{'id': self.leaf['image_id'], 'original_tokens': 577, 'retained_limit': 512,
            'text_sha256': hashlib.sha256(memory.texts[0].encode()).hexdigest()}]
        encoder = MagicMock(max_seq_length=512)
        encoder.tokenizer.return_value = {'input_ids': [list(range(577))]}
        encoder.encode.return_value = np.array([[1., 0.]])
        np.testing.assert_array_equal(online.encode_memory_text(memory, encoder), [[1., 0.]])
        memory.texts[0] += ' changed text'
        with self.assertRaisesRegex(ValueError, 'truncation differs'):
            online.encode_memory_text(memory, encoder)


if __name__ == '__main__':
    unittest.main()
