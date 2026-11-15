
# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.memory.reflexion_memory import (make_memory, upstream_helpers, trial_text, reflection_prompt,
                              apply_reflection, answer_messages, run, hash_object, read_rows)


class ReflexionTests(unittest.TestCase):
    def test_upstream_recall_is_query_independent_and_keeps_newest_words(self):
        memory = make_memory([{'text': ' '.join('w'+str(i) for i in range(7000))}])
        context = memory.recall('first topic')
        self.assertEqual(context, memory.recall('unrelated topic'))
        self.assertEqual(len(context.split(' ')), 6000)
        self.assertNotIn('w0 ', context)
        self.assertIn('w6999', context)
        memory.insight['global_insight'] = 'SECRET_PRIOR_INSIGHT'
        self.assertNotIn('SECRET_PRIOR_INSIGHT', memory.recall('query'))

    def test_real_optimize_updates_and_reuses_insight(self):
        memory = make_memory([{'text': 'visible observation'}])
        context = memory.recall('query')
        trial = trial_text('question', context, 'A', 'B')
        prompt = reflection_prompt(memory, trial)
        self.assertIn('Ground truth:\nB', prompt)
        self.assertIn('Correct: no', prompt)
        apply_reflection(memory, trial, prompt, 'Use the newer evidence.')
        self.assertEqual(memory.insight['global_insight'], 'Use the newer evidence.')
        self.assertIn('Use the newer evidence.', memory.recall('second question'))
        fresh = make_memory([{'text': 'another task'}])
        self.assertEqual(fresh.insight['global_insight'], '')

    def test_current_gold_not_in_answer_and_query_image_is_captioned(self):
        q = {'question': 'Which?', 'options': dict(zip('ABCD', ['one','two','three','four'])),
             'query_image_paths': ['not-loaded.png'], 'query_image_captions': ['Qwen query caption'],
             'gold_choice': 'NEVER_SHOW_CURRENT_GOLD', 'gold_ids': ['NEVER_SHOW_CLUE']}
        memory = make_memory([{'text': 'observation'}])
        question, query, context, messages = answer_messages(q, memory, {'reader_system_prompt': 'system'}, upstream_helpers())
        self.assertIn('Qwen query caption', query)
        self.assertNotIn('NEVER_SHOW', str(messages))
        self.assertNotIn('not-loaded.png', str(messages))
        self.assertEqual(messages[1]['role'], 'assistant')

    def test_official_note_format_uses_only_supplied_qwen_caption(self):
        helper = upstream_helpers()['build_reflexion_note_text']
        text = helper({'user': 'u', 'assistant': 'a', 'raw': {'image_caption': ['QWEN'],
            'answer': 'FORBIDDEN', 'clue': 'FORBIDDEN'}}, 'user (name)', 'assistant')
        self.assertEqual(text, 'user (name): u\nassistant: a\nimage:\nimage_caption: QWEN')

    def test_transport_rejects_wrong_precomputed_prompt(self):
        memory = make_memory([])
        with self.assertRaisesRegex(ValueError, 'prompt/result mismatch'):
            apply_reflection(memory, 'trial', 'not the real prompt', 'reflection')

    def test_checkpoint_resume_preserves_task_state_without_reanswering(self):
        class FakeQwen:
            environment = {'test': True}
            def __init__(self, protocol):
                self.n = 0
            def generate(self, messages, max_new_tokens):
                self.n += 1
                return [{'raw_response': 'A' if max_new_tokens == 16 else f'insight {self.n}',
                         'input_tokens': 50, 'generated_tokens': 2, 'hit_token_limit': False}
                        for _ in messages]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'reflexion'
            target.mkdir()
            questions = [{'id': i, 'task': task, 'question': 'Which?',
                'options': dict(zip('ABCD', ['one', 'two', 'three', 'four'])),
                'gold_choice': 'B', 'gold_ids': [], 'query_image_paths': [],
                'query_image_captions': [], 'x': 'X1', 'y': 'Y1'}
                for i, task in [('a1','a'), ('a2','a'), ('b1','b')]]
            observations = {'a': [{'text': 'history A'}], 'b': [{'text': 'history B'}]}
            protocol = {'input_hashes': {}, 'prepared_questions_sha256': hash_object(questions),
                'observations_sha256': hash_object(observations), 'experiment_signature': 'sig',
                'task_batch_size': 2, 'reader_system_prompt': 'system', 'max_new_tokens': 16,
                'reflection_max_new_tokens': 512}
            (target / 'protocol.json').write_text(json.dumps(protocol))
            (target / 'observations.json').write_text(json.dumps(observations))
            (target / 'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
            with patch('scripts.memory.reflexion_memory.LocalQwen', FakeQwen):
                run(SimpleNamespace(output_root=root))
            states = {s['id']: s for s in (json.loads(p.read_text()) for p in (target/'checkpoints').glob('*.json'))}
            self.assertEqual(states['a1']['insight_before'], '')
            self.assertEqual(states['b1']['insight_before'], '')
            self.assertEqual(states['a2']['insight_before'], states['a1']['insight_after'])
            self.assertEqual(len(read_rows(target/'qa_responses.jsonl')), 3)
            with patch('scripts.memory.reflexion_memory.LocalQwen') as engine:
                run(SimpleNamespace(output_root=root))
                engine.assert_not_called()
            path = target / 'checkpoints' / (hash_object('a2')+'.json')
            bad = json.loads(path.read_text()); bad['insight_before'] = 'wrong prior state'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'checkpoint provenance'):
                run(SimpleNamespace(output_root=root))


if __name__ == '__main__':
    unittest.main()
