"""Regressions for the actual failed response and pinned Qwen template."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation.omni_qwen_transport_v3 import ENTITY_SCHEMA, MergedSystemProcessor, merge_system_messages


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor
        cls.original = AutoProcessor.from_pretrained(str(ROOT / 'models/Qwen3-VL-8B-Instruct'), local_files_only=True)
        cls.merged = MergedSystemProcessor(cls.original)

    def test_original_failure_type_is_valid_for_native_parser(self):
        import jsonschema
        from scripts.memory import omni_simplemem
        omni_simplemem.activate()
        from simplemem.multimodal.knowledge.entity_extractor import ExtractedEntity
        value = dict(entities=[dict(name='Clinical Research Assistant', type='JOB_TITLE')], relations=[])
        jsonschema.validate(value, ENTITY_SCHEMA)
        entity = ExtractedEntity(name=value['entities'][0]['name'], entity_type=value['entities'][0]['type'])
        self.assertEqual(entity.entity_type, 'JOB_TITLE')
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(dict(entities=['Clinical Research Assistant'], relations=[]), ENTITY_SCHEMA)

    def test_real_template_keeps_schema_native_prompt_and_retry(self):
        messages = [dict(role='system', content='Return JSON matching the schema.'),
                    dict(role='system', content='Format retry 1: correct the object shape.'),
                    dict(role='system', content='You are an expert knowledge extraction system.'),
                    dict(role='user', content='Extract entities from a resume.')]
        old = self.original.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertNotIn(messages[1]['content'], old)
        new = self.merged.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for message in messages:
            self.assertIn(message['content'], new)
        self.assertNotEqual(new, old)
        self.assertEqual(len(messages), 4)

    def test_single_system_multimodal_prompt_is_byte_identical(self):
        messages = [dict(role='system', content='Describe the picture.'),
                    dict(role='user', content=[dict(type='image', path='/tmp/example.jpg', sha256='example'),
                                               dict(type='text', text='What is shown?')])]
        self.assertEqual(self.original.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                         self.merged.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        self.assertEqual(messages, merge_system_messages(messages))

    def test_later_system_cannot_silently_disappear(self):
        with self.assertRaises(AssertionError):
            merge_system_messages([dict(role='user', content='Hi'), dict(role='system', content='Hidden')])


if __name__ == '__main__':
    unittest.main()
