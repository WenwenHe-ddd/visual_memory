"""CLI routing boundaries for caption clusters and visual master retrieval.

Encoder entrypoints are mocked: these checks must never load models, require a
GPU, alter experiment outputs, or call the QA reader.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import visual_memory_pipeline as pipeline
from scripts.memory import caption_visual_master


class CaptionVisualPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.offline = self.stack.enter_context(patch.object(caption_visual_master, 'offline'))
        self.online = self.stack.enter_context(patch.object(caption_visual_master, 'online'))
        self.legacy = {
            name: self.stack.enter_context(patch.object(pipeline, name))
            for name in ('prepare_online', 'run_online', 'analyze_online')
        }
        self.reader = self.stack.enter_context(patch.object(
            pipeline.qa_reader, 'run', side_effect=AssertionError('unexpected reader call')))
        self.process = self.stack.enter_context(patch.object(
            pipeline.subprocess, 'run', side_effect=AssertionError('unexpected external process')))

    def assert_no_legacy_online(self):
        for method in self.legacy.values():
            method.assert_not_called()
        self.reader.assert_not_called()
        self.process.assert_not_called()

    def assert_cli_error(self, argv, expected_message):
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as exc:
            pipeline.main(argv)
        self.assertEqual(exc.exception.code, 2)
        self.assertIn(expected_message, errors.getvalue())

    def test_offline_build_dispatches_eight_examples_on_gpu(self):
        pipeline.main(['offline', '--memory', 'caption_visual_master'])
        self.offline.assert_called_once()
        args = self.offline.call_args.args[0]
        self.assertEqual(args.device, 'cuda:0')
        self.assertEqual(args.image_files, [
            'D7_IMG_001.jpg', 'D7_IMG_002.jpg', 'D7_IMG_003.jpg',
            'D7_IMG_004.jpg', 'D7_IMG_005.jpg', 'D7_IMG_006.jpg',
            'D13_IMG_001.jpg', 'D13_IMG_002.jpg',
        ])
        self.assertEqual(args.output, pipeline.ROOT / 'results/debug_caption_master_answer/offline')
        self.online.assert_not_called()
        self.assert_no_legacy_online()

    def test_online_defaults_match_offline_output_and_retrieve_three_questions(self):
        pipeline.main(['online', '--memory', 'caption_visual_master'])
        self.online.assert_called_once()
        args = self.online.call_args.args[0]
        self.assertEqual(args.device, 'cuda:0')
        self.assertEqual(args.stage, 'prepare')
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.question_ids, ['Q8', 'Q10', 'Q27'])
        self.assertEqual(args.memory_dir, pipeline.ROOT / 'results/debug_caption_master_answer/offline/memory')
        self.assertEqual(args.output, pipeline.ROOT / 'results/debug_caption_master_answer/online')
        self.offline.assert_not_called()
        self.assert_no_legacy_online()

    def test_legacy_offline_filters_and_resume_flag_are_rejected(self):
        for flags in (['--task', 'Brand'], ['--limit', '2'], ['--from-start']):
            with self.subTest(flags=flags):
                self.assert_cli_error([
                    'offline', '--memory', 'caption_visual_master', *flags,
                ], 'selected by --image-files')
        self.offline.assert_not_called()
        self.online.assert_not_called()
        self.assert_no_legacy_online()

    def test_explicit_device_subset_paths_and_mode_alias_reach_online(self):
        pipeline.main([
            'online', '--mode', 'caption_visual_master', '--device', 'cpu',
            '--memory-dir', '/tmp/custom-memory', '--question-ids', 'Q10',
            '--top-k', '2', '--output', '/tmp/custom-retrieval',
        ])
        args = self.online.call_args.args[0]
        self.assertEqual((args.device, args.question_ids, args.top_k), ('cpu', ['Q10'], 2))
        self.assertEqual(args.memory_dir, Path('/tmp/custom-memory'))
        self.assertEqual(args.output, Path('/tmp/custom-retrieval'))
        self.assert_no_legacy_online()

    def test_reader_and_analysis_stages_rejected_before_work(self):
        for stage in ('run', 'analyze', 'all'):
            with self.subTest(stage=stage):
                self.assert_cli_error([
                    'online', '--memory', 'caption_visual_master', '--stage', stage,
                ], 'retrieval only')
        self.online.assert_not_called()
        self.offline.assert_not_called()
        self.assert_no_legacy_online()

    def test_legacy_question_filters_do_not_silently_change_new_subset(self):
        for flags in (['--task', 'Brand'], ['--limit-questions', '1']):
            with self.subTest(flags=flags):
                self.assert_cli_error([
                    'online', '--memory', 'caption_visual_master', *flags,
                ], '--question-source and --question-ids')
        self.online.assert_not_called()
        self.assert_no_legacy_online()

    def test_new_mode_still_checks_positive_top_k(self):
        self.assert_cli_error([
            'online', '--memory', 'caption_visual_master', '--top-k', '0',
        ], '--top-k must be positive')
        self.online.assert_not_called()
        self.assert_no_legacy_online()

    def test_all_keeps_existing_methods_and_stage_dispatch(self):
        self.assertEqual(pipeline.selected_modes('all'), ('natural', 'fc_t'))
        pipeline.main(['online', '--mode', 'all', '--stage', 'all'])
        for method in self.legacy.values():
            method.assert_called_once()
            self.assertEqual(method.call_args.args[0].mode, 'all')
        self.offline.assert_not_called()
        self.online.assert_not_called()
        self.reader.assert_not_called()

    def test_existing_visual_graph_builder_remains_separate_and_cpu_by_default(self):
        with patch('scripts.graph.build_visual_embedding_graph.build') as build:
            pipeline.main(['offline', '--memory', 'visual_embedding_graph', '--limit', '2'])
        build.assert_called_once()
        args = build.call_args.args[0]
        self.assertEqual(args.visual_device, 'cpu')
        self.assertEqual(args.limit, 2)
        self.assertEqual(args.graph_root, pipeline.ROOT / 'results/visual_embedding_graph')
        self.offline.assert_not_called()
        self.online.assert_not_called()
        self.assert_no_legacy_online()
        self.assert_cli_error(['online', '--memory', 'visual_embedding_graph'], 'invalid choice')

    def test_existing_full_context_rejects_top_k(self):
        self.assert_cli_error(['online', '--memory', 'fc_t', '--top-k', '5'], 'omit --top-k')
        self.assert_no_legacy_online()


if __name__ == '__main__':
    unittest.main()
