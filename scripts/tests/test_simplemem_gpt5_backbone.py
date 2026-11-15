import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import simplemem_gpt5_backbone as runner
from scripts import visual_memory_pipeline


class GPT5BackboneTests(unittest.TestCase):
    def test_caption_replacement_preserves_all_other_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'image.png'
            image.write_bytes(b'image identity fixture')
            identity = runner.base.sha(image)
            obs = {'task': [{'id': 'round', 'user': 'keep dialogue', 'assistant': 'keep reply',
                'raw': {'input_image': [str(image)], 'image_caption': ['old'], 'image_id': ['img']}}]}
            qs = [{'id': 'question', 'question': 'keep question', 'options': {'A': 'keep option'},
                   'gold_ids': ['keep gold'], 'query_image_paths': [str(image)], 'query_image_captions': ['old']}]
            before_obs, before_qs = copy.deepcopy(obs), copy.deepcopy(qs)
            changed_obs, changed_qs, uses = runner.replace_captions(obs, qs, {identity: 'GPT-5 fixture'})
            self.assertEqual(obs, before_obs)
            self.assertEqual(qs, before_qs)
            self.assertEqual(len(uses), 2)
            self.assertEqual(changed_obs['task'][0]['raw'].pop('image_caption'), ['GPT-5 fixture'])
            before_obs['task'][0]['raw'].pop('image_caption')
            self.assertEqual(changed_obs, before_obs)
            self.assertEqual(changed_qs[0].pop('query_image_captions'), ['GPT-5 fixture'])
            before_qs[0].pop('query_image_captions')
            self.assertEqual(changed_qs, before_qs)
            with self.assertRaises(KeyError):
                runner.replace_captions(obs, qs, {'wrong-image': 'unrelated caption'})

    def test_pipeline_routes_explicit_gpt5_without_changing_local_default(self):
        with patch.object(runner, 'pipeline') as run:
            visual_memory_pipeline.main(['online', '--mode', 'simplemem', '--simplemem-backbone', 'gpt-5',
                '--api-model', 'gpt-5', '--simplemem-context-limit', '5', '--simplemem-reflection',
                '--output-root', '/tmp/explicit-gpt5-fixture'])
            args = run.call_args.args[0]
            self.assertEqual(args.simplemem_semantic_top_k, 10)
            self.assertEqual(args.simplemem_context_fusion, 'channel_rrf')
            self.assertTrue(args.simplemem_reflection)
        with patch.object(runner.base, 'prepare') as prepare:
            visual_memory_pipeline.main(['online', '--mode', 'simplemem', '--stage', 'prepare'])
            self.assertEqual(prepare.call_args.args[0].simplemem_backbone, 'local-qwen')

    def test_terminal_api_failure_prevents_more_requests(self):
        model = object.__new__(runner.GPT5Model)
        model.fatal_errors = [{'error_type': 'quota'}]
        with patch.object(runner.api.APIModel, 'one') as one:
            with self.assertRaisesRegex(RuntimeError, 'terminal API failure'):
                model.one([{'role': 'user', 'content': 'test'}], 16, 0)
            one.assert_not_called()

    def test_memory_wire_keeps_roles_and_supported_gpt5_settings(self):
        messages = [{'role': 'system', 'content': 'extract'}, {'role': 'user', 'content': 'history'}]
        req = runner.api.build_request(messages, 16384, 0.3)
        self.assertEqual(req['model'], 'gpt-5')
        self.assertEqual(req['messages'], messages)
        self.assertEqual(req['max_completion_tokens'], 16384)
        self.assertEqual(req['reasoning_effort'], 'minimal')
        self.assertEqual(req['seed'], 42)
        self.assertNotIn('temperature', req)
        self.assertFalse(req['store'])


if __name__ == '__main__':
    unittest.main()
