"""API reader CLI routing only: no environment files, models, or network."""
from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import visual_memory_pipeline as pipeline
from scripts.evaluation import memeye_api_reader


class MemEyeAPIPipelineTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.dispatch = stack.enter_context(patch.object(memeye_api_reader, 'pipeline'))
        self.legacy = [stack.enter_context(patch.object(pipeline, name))
                       for name in ('prepare_online', 'run_online', 'analyze_online')]
        self.reader = stack.enter_context(patch.object(pipeline.qa_reader, 'run'))
        self.process = stack.enter_context(patch.object(pipeline.subprocess, 'run'))
        self.base = ['online', '--memory', 'caption_visual_master', '--reader-backend', 'api']

    def assert_no_local_work(self):
        for method in self.legacy + [self.reader, self.process]:
            method.assert_not_called()

    def test_defaults_dispatch_full_frozen_memeye_to_api_prepare(self):
        pipeline.main(self.base)
        self.dispatch.assert_called_once()
        args = self.dispatch.call_args.args[0]
        self.assertEqual(args.dataset, 'memeye')
        self.assertEqual(args.api_model, 'gpt-4o-mini')
        self.assertEqual(args.stage, 'prepare')
        self.assertIsNone(args.question_ids)
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.api_workers, 3)
        self.assertEqual(args.output, pipeline.ROOT / 'results/memeye_reader_ablation_20261021/gpt-4o-mini')
        self.assertEqual(args.api_env_file, Path('/storage/hewenwen/NTU Decentralized Agent/.env'))
        self.assert_no_local_work()

    def test_explicit_api_smoke_arguments_reach_only_api_runner(self):
        pipeline.main(self.base + ['--stage', 'run', '--api-smoke', '--api-model', 'gpt-4o',
                                   '--api-env-file', '/tmp/unit-test.env', '--api-workers', '1',
                                   '--api-source-reader', '/tmp/frozen-reader', '--output', '/tmp/api-results'])
        args = self.dispatch.call_args.args[0]
        self.assertEqual((args.api_model, args.api_workers, args.stage, args.api_smoke), ('gpt-4o', 1, 'run', True))
        self.assertEqual(args.output, Path('/tmp/api-results'))
        self.assertEqual(args.api_source_reader, Path('/tmp/frozen-reader'))
        self.assertEqual(args.api_env_file, Path('/tmp/unit-test.env'))
        self.assert_no_local_work()

    def test_rejects_changed_question_pool_retrieval_or_invalid_flags(self):
        cases = [
            (['--question-ids', 'Q10'], 'freezes all 371'),
            (['--task', 'Brand'], 'freezes all 371'),
            (['--limit-questions', '1'], 'freezes all 371'),
            (['--top-k', '3'], 'frozen Top5'),
            (['--dataset', 'mem_gallery'], 'requires caption_visual_master on MemEye'),
            (['--api-workers', '0'], 'must be positive'),
            (['--api-smoke'], 'requires --stage run or all'),
        ]
        for flags, expected in cases:
            with self.subTest(flags=flags):
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                    pipeline.main(self.base + flags)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected, errors.getvalue())
        self.dispatch.assert_not_called()
        self.assert_no_local_work()

    def test_rejects_other_memory_method_before_any_api_work(self):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            pipeline.main(['online', '--memory', 'natural', '--reader-backend', 'api'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn('requires caption_visual_master on MemEye', errors.getvalue())
        self.dispatch.assert_not_called()
        self.assert_no_local_work()


if __name__ == '__main__':
    unittest.main()
