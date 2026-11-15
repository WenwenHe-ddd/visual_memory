
# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.memory.full_context_memory import captioned_histories, fit_history, format_history, sha
from scripts.data.generate import signature
from scripts.visual_memory_pipeline import output_dir, selected_modes
from types import SimpleNamespace


class FullContextTests(unittest.TestCase):
    def setUp(self):
        self.question = {'question': 'Where now?', 'options': dict(zip('ABCD', ['a', 'b', 'c', 'd'])),
                         'gold_choice': 'SECRET_GOLD', 'gold_ids': ['SECRET_CLUE']}

    def test_fifo_preserves_latest_round_and_question(self):
        rounds = [{'id': str(i), 'text': 'text ' * 20 + str(i)} for i in range(3)]
        latest_text = format_history(self.question, rounds[-1:])
        text, kept, dropped, count = fit_history(self.question, rounds, len, len(latest_text) + 16, 16)
        self.assertEqual(text, latest_text)
        self.assertEqual(kept, rounds[-1:])
        self.assertEqual(dropped, ['0', '1'])
        self.assertEqual(count, len(text))
        self.assertIn('D. d', text)
        self.assertNotIn('SECRET', text)
        with self.assertRaisesRegex(ValueError, 'exceed'):
            fit_history(self.question, rounds, len, 10, 16)

    def test_full_history_does_not_rank_or_drop_rounds(self):
        rounds = [{'id': 'old', 'text': 'old'}, {'id': 'new', 'text': 'new'}]
        _, kept, dropped, _ = fit_history(self.question, rounds, len, 10000, 16)
        self.assertEqual(kept, rounds)
        self.assertEqual(dropped, [])

    def test_caption_source_whitelist_and_missing_caption_failure(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            def write_jsonl(name, rows):
                (root / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            source = root / 'task_Open.json'
            source.write_text(json.dumps({'multi_session_dialogues': [
                {'session_id': 's', 'date': 'today', 'dialogues': [
                    {'round': 'r1', 'user': 'original user', 'assistant': 'original assistant',
                     'input_image': ['image.png'], 'image_caption': ['FORBIDDEN_SUPPLIED_CAPTION']},
                    {'round': 'r2', 'user': 'text only', 'assistant': 'reply'}]}],
                'human-annotated QAs': [{'answer': 'FORBIDDEN_GOLD'}]}))
            config = dict.fromkeys(['provenance', 'caption_prompt', 'cue_prompt', 'image_max_pixels',
                                   'caption_max_new_tokens', 'cue_max_new_tokens', 'seed',
                                   'do_sample', 'dtype'], 'test')
            config['source_files'] = [{'path': str(source), 'sha256': sha(source)}]
            write_jsonl('captions.jsonl', [{'id': 'img', 'caption': 'EXISTING_QWEN_CAPTION',
                                          'protocol_signature': signature(config)}])
            write_jsonl('images.jsonl', [{'id': 'img', 'source_paths': ['image.png']}])
            write_jsonl('memories.jsonl', [{'id': 'task/r1', 'task': 'task', 'images': ['img']}])
            histories, _ = captioned_histories(root, config, {'task'})
            text = format_history(self.question, histories['task'])
            self.assertEqual([r['id'] for r in histories['task']], ['task/r1', 'task/r2'])
            for expected in ('original user', 'original assistant', 'text only', 'EXISTING_QWEN_CAPTION'):
                self.assertIn(expected, text)
            self.assertNotIn('FORBIDDEN', text)
            self.assertNotIn('SECRET', text)
            write_jsonl('captions.jsonl', [])
            with self.assertRaisesRegex(ValueError, 'missing Qwen caption'):
                captioned_histories(root, config, {'task'})

    def test_fc_only_prepare_skips_retriever_and_graph(self):
        from scripts import visual_memory_pipeline as pipeline
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'protocol.json').write_text('{}')
            (root / 'questions.jsonl').write_text(json.dumps({'id': 'q', 'task': 't'}) + '\n')
            args = SimpleNamespace(benchmark_manifest=root / 'questions.jsonl', task=None,
                                   limit_questions=0, natural_root=root, mode='fc_t')
            with patch('scripts.memory.full_context_memory.prepare_full_context') as prepare:
                pipeline.prepare_online(args)
                prepare.assert_called_once()

    def test_output_not_dependent_on_top_k(self):
        args = SimpleNamespace(output_root=Path('/tmp/output'), top_k=5)
        self.assertIn('fc_t', selected_modes('all'))
        self.assertEqual(output_dir(args, 'fc_t'), Path('/tmp/output/fc_t'))
        self.assertEqual(output_dir(args, 'natural'), Path('/tmp/output/top5/natural'))

    def test_analysis_counts_invalid_answer_and_reports_fifo(self):
        from scripts.visual_memory_pipeline import analyze_online
        with TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'fc_t'
            target.mkdir()
            (target / 'protocol.json').write_text(json.dumps({'experiment_signature': 'sig'}))
            questions, responses = [], []
            for i in range(2):
                questions.append({**self.question, 'id': str(i), 'task': 'task',
                                  'gold_choice': 'A', 'query_image_paths': [],
                                  'payload_sha256': str(i), 'any_gold_topk': i == 0,
                                  'all_gold_topk': i == 0, 'input_tokens': 100 + i,
                                  'dropped_round_ids': [] if i == 0 else ['old']})
                responses.append({'id': str(i), 'experiment_signature': 'sig',
                                  'payload_sha256': str(i),
                                  'predicted_choice': 'A' if i == 0 else None,
                                  'status': 'ok' if i == 0 else 'format_error',
                                  'hit_token_limit': False})
            for name, rows in [('qa_manifest.jsonl', questions), ('qa_responses.jsonl', responses)]:
                (target / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):
                analyze_online(SimpleNamespace(output_root=root, mode='fc_t', top_k=None))
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual(summary['accuracy'], 0.5)
            self.assertEqual(summary['format_errors'], 1)
            self.assertEqual(summary['context_truncated_questions'], 1)
            self.assertIsNone(summary['top_k'])


if __name__ == '__main__':
    unittest.main()
