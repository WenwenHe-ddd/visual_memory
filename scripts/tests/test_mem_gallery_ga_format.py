import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.evaluation.continue_mem_gallery_ga_format import reflection_lines


class Template:
    def __init__(self, input_variables, template):
        self.template = template

    def format(self, **values):
        return self.template.format(**values)


class FormatRetryTests(unittest.TestCase):
    def invoke(self, responses):
        prompts, events = [], []
        values = iter(responses)

        def ask(prompt):
            prompts.append(prompt)
            return next(values)

        instance = SimpleNamespace(config=SimpleNamespace(question_number=2,
            question_prompt=SimpleNamespace(input_variables=['information'], template='Memory: {information}')),
            llm=SimpleNamespace(fast_run=ask))
        with patch.dict(sys.modules, {'langchain.prompts': SimpleNamespace(PromptTemplate=Template)}):
            output = reflection_lines(instance, {'information': 'original source'}, 'question', 'task', events.append)
        return output, prompts, events

    def test_valid_output_is_unchanged_and_not_retried(self):
        output, prompts, events = self.invoke(['One?\nTwo?'])
        self.assertEqual(output, ['One?', 'Two?'])
        self.assertEqual(prompts, ['Memory: original source'])
        self.assertFalse(events)

    def test_retry_only_adds_format_instruction(self):
        output, prompts, events = self.invoke(['Combined?', 'One?\nTwo?'])
        self.assertEqual(output, ['One?', 'Two?'])
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[1].startswith(prompts[0] + '\n\nOutput format requirement:'))
        self.assertIn('exactly 2 questions', prompts[1])
        self.assertTrue(events[0]['valid'])
        self.assertEqual(events[0]['original_response'], 'Combined?')

    def test_failed_format_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'after three format retries'):
            self.invoke(['One?'] * 4)


if __name__ == '__main__':
    unittest.main()
