"""Continuation orchestration tests: no models, benchmark data, or API calls."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.evaluation import memeye_deepseek_backbone as completion
from scripts.evaluation.deepseek_backbone_transport import DEEPSEEK_MODEL


class DeepSeekBackboneCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = {name: self.stack.enter_context(patch.object(completion, name)) for name in (
            'prepare', 'verify', 'generate_captions', 'verify_captions', 'verify_retrieval',
            'freeze', 'analyze', 'prepare_reader', 'execute_stage',
        )}
        self.calls['verify'].return_value = {'contract': {'model': DEEPSEEK_MODEL}}
        self.calls['generate_captions'].return_value = True
        self.calls['execute_stage'].return_value = True
        self.calls['analyze'].return_value = {'status': 'synthetic'}
        strict = ModuleType('scripts.memory.memeye_backbone_retrieval')
        extension = ModuleType('scripts.memory.memeye_backbone_long_text')
        self.strict = strict.rebuild_and_retrieve = MagicMock()
        self.extension = extension.complete_retrieval = MagicMock()
        self.stack.enter_context(patch.dict(sys.modules, {
            strict.__name__: strict, extension.__name__: extension,
        }))
        self.args = SimpleNamespace(api_workers=3, api_limit=0, api_model=DEEPSEEK_MODEL,
                                    api_env_file=self.output / 'unused.env', output=self.output,
                                    stage='retrieve', device='cuda:0')
        self.original_source = Path(completion.__file__)
        self.original_digest = hashlib.sha256(self.original_source.read_bytes()).hexdigest()

    def tearDown(self):
        # The continuation must not rewrite the signed original runner.
        self.assertEqual(hashlib.sha256(self.original_source.read_bytes()).hexdigest(), self.original_digest)

    def run_pipeline(self):
        with redirect_stdout(io.StringIO()):
            completion.pipeline(self.args)

    def test_normal_retrieval_keeps_strict_path(self):
        self.run_pipeline()
        self.strict.assert_called_once_with(self.output, device='cuda:0')
        self.extension.assert_not_called()
        self.calls['freeze'].assert_not_called()
        self.calls['verify_captions'].assert_called_once_with(self.output)
        self.calls['analyze'].assert_called_once_with(self.output, require_reader=False)

    def test_only_explicit_bge_overflow_runs_original_policy_continuation(self):
        self.strict.side_effect = ValueError('leaf retrieval text exceeds BGE token limit 512: 577')
        self.run_pipeline()
        self.extension.assert_called_once_with(self.output, device='cuda:0')
        self.calls['freeze'].assert_called_once()
        path, transition = self.calls['freeze'].call_args.args
        self.assertEqual(path, self.output / 'long_text_transition.json')
        self.assertEqual(transition['reason'], str(self.strict.side_effect))
        self.assertIn('original BGE 512-token encoding limit', transition['action'])
        self.assertFalse(transition['caption_generation_changed'])
        self.assertFalse(transition['memory_construction_changed'])
        self.assertFalse(transition['reader_evidence_changed'])
        self.calls['analyze'].assert_called_once_with(self.output, require_reader=False)

    def test_unrelated_value_error_propagates_without_continuation(self):
        self.strict.side_effect = ValueError('Visual source image SHA-256 mismatch')
        with self.assertRaisesRegex(ValueError, 'Visual source image SHA-256 mismatch'):
            self.run_pipeline()
        self.extension.assert_not_called()
        self.calls['freeze'].assert_not_called()
        self.calls['analyze'].assert_not_called()

    def test_completed_retrieval_is_verified_and_skips_gpu_rebuild(self):
        (self.output / 'retrieval_protocol.json').write_text('{}\n')
        self.args.stage = 'all'
        self.run_pipeline()
        self.calls['verify_retrieval'].assert_called_once_with(self.output)
        self.strict.assert_not_called()
        self.extension.assert_not_called()
        self.calls['prepare_reader'].assert_called_once_with(self.output)
        self.calls['execute_stage'].assert_called_once_with(self.output, 'reader', 0)
        self.assertEqual(self.calls['analyze'].call_count, 2)

    def test_caption_request_limit_stops_before_retrieval_and_reader(self):
        self.args.stage = 'all'
        self.args.api_limit = 2
        self.calls['generate_captions'].return_value = False
        self.run_pipeline()
        self.calls['generate_captions'].assert_called_once_with(self.output, 2)
        self.strict.assert_not_called()
        self.extension.assert_not_called()
        self.calls['verify_captions'].assert_not_called()
        self.calls['prepare_reader'].assert_not_called()
        self.calls['execute_stage'].assert_not_called()
        self.calls['analyze'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
