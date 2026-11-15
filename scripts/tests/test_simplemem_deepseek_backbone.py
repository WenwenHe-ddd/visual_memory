import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import simplemem_deepseek_backbone as runner


class DeepSeekBackboneTests(unittest.TestCase):
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
            changed_obs, changed_qs, uses = runner.replace_captions(obs, qs, {identity: 'DeepSeek fixture'})
            self.assertEqual(obs, before_obs)
            self.assertEqual(qs, before_qs)
            self.assertEqual(len(uses), 2)
            self.assertEqual(changed_obs['task'][0]['raw'].pop('image_caption'), ['DeepSeek fixture'])
            before_obs['task'][0]['raw'].pop('image_caption')
            self.assertEqual(changed_obs, before_obs)
            self.assertEqual(changed_qs[0].pop('query_image_captions'), ['DeepSeek fixture'])
            before_qs[0].pop('query_image_captions')
            self.assertEqual(changed_qs, before_qs)
            with self.assertRaises(KeyError):
                runner.replace_captions(obs, qs, {'wrong-image': 'unrelated caption'})

    def test_terminal_api_failure_prevents_more_requests(self):
        model = object.__new__(runner.DeepSeekModel)
        model.fatal_errors = [{'error_type': 'quota'}]
        with patch.object(runner.api.APIModel, 'one') as one:
            with self.assertRaisesRegex(RuntimeError, 'terminal API failure'):
                model.one([{'role': 'user', 'content': 'test'}], 16, 0)
            one.assert_not_called()

    def test_memory_and_answer_wire_keep_exact_model_seed_and_native_budgets(self):
        messages = [{'role': 'system', 'content': 'extract'}, {'role': 'user', 'content': 'history'}]
        req = runner.api.build_request(messages, 16384, 0.3)
        self.assertEqual(req['model'], 'deepseek-v4.1-flash')
        self.assertEqual(req['messages'], messages)
        self.assertEqual(req['max_tokens'], 16384)
        self.assertFalse(req['enable_thinking'])
        self.assertEqual(req['seed'], 42)
        self.assertEqual(req['temperature'], 0.3)
        answer = runner.api.build_request(messages, 16, 0)
        self.assertEqual(answer['max_tokens'], 16)
        self.assertEqual(answer['temperature'], 0)
        with self.assertRaises(ValueError):
            runner.api.build_request([{'role': 'user', 'content': 'history', 'gold': 'B'}], 16, 0)


if __name__ == '__main__':
    unittest.main()
