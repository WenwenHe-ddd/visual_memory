"""Extraction schema, input isolation, resumability and cache provenance."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import visual_memory_pipeline as pipeline
from scripts.memory import object_relation_memory as extraction


CAPTION = 'A cat sits on a chair.'
OBJECTS = {'objects': ['cat', 'chair', 'cup']}
OUTPUT = {'relations': [{'subject': 'cat', 'predicate': 'sits on', 'object': 'chair'}]}


class FakeWriter:
    last_raw = None

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
        self.image_paths = []
        self.stages = []
        self.max_tokens = []
        self.generation_options = []

    def _ask_plain(self, prompt, stage, max_tokens, *, image_path, **generation_options):
        assert Path(image_path).is_file()
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        self.stages.append(stage)
        self.max_tokens.append(max_tokens)
        self.generation_options.append(generation_options)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        self.last_raw = response
        return self.last_raw


class ObjectRelationTests(unittest.TestCase):
    def test_object_validation_and_normalization(self):
        self.assertEqual(extraction.parse_objects('{"objects": [" cat ", "chair", "cat", "coffee  cup"]}'),
                         ['cat', 'chair', 'coffee cup'])
        self.assertEqual(extraction.parse_objects('{"objects": []}'), [])
        for obj in ([], {}, {'objects': 'cat'}, {'objects': ['']}, {'objects': [None]},
                    {'objects': [{'name': 'cat'}]}, {'objects': ['cat'], 'relations': []}):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                extraction.parse_objects(json.dumps(obj))

    def test_relation_validation(self):
        self.assertEqual(extraction.parse_output(json.dumps(OUTPUT)), OUTPUT['relations'])
        for obj in ([['cat', 'sits on'], [None, 'chair']], {}, {'relations': {}},
                    {'relations': [{'subject': 'cat', 'object': 'chair'}]},
                    {'relations': [{'subject': 'cat', 'predicate': '', 'object': 'chair'}]}):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                extraction.parse_output(json.dumps(obj))

    def test_isolated_objects_preserved_and_relations_deduplicated(self):
        generated = [
            {'subject': ' cat ', 'predicate': 'sits  on', 'object': 'chair'},
            {'subject': 'cat', 'predicate': 'sits on', 'object': 'chair'},
            {'subject': 'chair', 'predicate': 'supports', 'object': 'cat'},
            {'subject': 'cat', 'predicate': 'beside', 'object': 'chair'}]
        result = extraction.process_output(json.dumps({'relations': generated}), OBJECTS['objects'])
        self.assertEqual(result['generated_relations'], [
            'cat sits on chair.', 'cat sits on chair.', 'chair supports cat.', 'cat beside chair.'])
        self.assertEqual(result['initial_objects'], ['cat', 'chair', 'cup'])
        self.assertEqual(result['added_objects'], [])
        self.assertEqual(result['objects'], ['cat', 'chair', 'cup'])
        self.assertEqual(result['relations'], [
            'cat sits on chair.', 'chair supports cat.', 'cat beside chair.'])
        self.assertTrue(all(isinstance(relation, str) for relation in result['relations']))
        self.assertEqual(extraction.process_output('{"relations": []}', OBJECTS['objects']),
                         {'generated_relations': [], 'initial_objects': OBJECTS['objects'],
                          'added_objects': [], 'objects': OBJECTS['objects'], 'relations': []})

    def test_missing_endpoints_complete_objects_in_encounter_order(self):
        initial = ['cat', 'chair', 'cup']
        generated = [
            {'subject': ' child ', 'predicate': 'holds', 'object': ' red  ball '},
            {'subject': 'cat', 'predicate': 'watches', 'object': 'child'},
            {'subject': 'red ball', 'predicate': 'beside', 'object': ' chair '},
            {'subject': 'child', 'predicate': 'holds', 'object': 'red ball'},
            {'subject': 'kitten', 'predicate': 'near', 'object': 'cat'}]
        result = extraction.process_output(json.dumps({'relations': generated}), initial)
        self.assertEqual(initial, ['cat', 'chair', 'cup'])
        self.assertEqual(result['initial_objects'], initial)
        self.assertEqual(result['generated_relations'], [
            'child holds red ball.', 'cat watches child.', 'red ball beside chair.',
            'child holds red ball.', 'kitten near cat.'])
        self.assertEqual(result['added_objects'], ['child', 'red ball', 'kitten'])
        self.assertEqual(result['objects'], ['cat', 'chair', 'cup', 'child', 'red ball', 'kitten'])
        self.assertEqual(result['relations'], [
            'child holds red ball.', 'cat watches child.', 'red ball beside chair.', 'kitten near cat.'])
        for relation in extraction.organize_relations(generated):
            for endpoint in ('subject', 'object'):
                self.assertIn(relation[endpoint], result['objects'])

    def test_sentence_deduplication_preserves_all_triple_endpoints(self):
        generated = [
            {'subject': 'person in blue', 'predicate': 'holds', 'object': 'cup'},
            {'subject': 'cat', 'predicate': 'near', 'object': 'cup'},
            {'subject': 'person', 'predicate': 'in blue holds', 'object': 'cup'}]
        result = extraction.process_output(json.dumps({'relations': generated}), ['cat'])
        self.assertEqual(result['generated_relations'], [
            'person in blue holds cup.', 'cat near cup.', 'person in blue holds cup.'])
        self.assertEqual(result['relations'], ['person in blue holds cup.', 'cat near cup.'])
        self.assertEqual(result['initial_objects'], ['cat'])
        self.assertEqual(result['added_objects'], ['person in blue', 'cup', 'person'])
        self.assertEqual(result['objects'], ['cat', 'person in blue', 'cup', 'person'])

    def test_completion_only_normalizes_whitespace_and_exact_duplicates(self):
        objects = [' cat ', 'cat', 'Cat', 'coffee  cup', 'isolated object']
        relations = [
            {'subject': ' cat ', 'predicate': 'near', 'object': 'coffee cup'},
            {'subject': 'cat', 'predicate': 'near', 'object': 'a cat'},
            {'subject': 'a cat', 'predicate': 'near', 'object': ' Cat '}]
        completed = extraction.complete_objects(objects, relations)
        self.assertEqual(completed, ['cat', 'Cat', 'coffee cup', 'isolated object', 'a cat'])
        self.assertEqual(objects, [' cat ', 'cat', 'Cat', 'coffee  cup', 'isolated object'])

    def fixture(self, root):
        natural = root / 'natural'
        natural.mkdir()
        config = {k: '' for k in ('caption_prompt', 'cue_prompt')}
        config.update(provenance={'caption_model': {'repo': 'test', 'revision': 'pinned'}},
                      caption_model_path='unused', image_max_pixels=1024,
                      caption_max_new_tokens=256, cue_max_new_tokens=256,
                      seed=42, do_sample=False, dtype='bfloat16')
        extraction.write_json(natural / 'protocol.json', config)
        image_rows = []
        for i, color in enumerate(('red', 'green', 'blue')):
            path = natural / f'image{i}.png'
            Image.new('RGB', (8, 8), color).save(path)
            image_rows.append({'id': f'image{i}', 'path': str(path)})
        self.image_ids = [r['id'] for r in image_rows]
        extraction.write_jsonl(natural / 'captions.jsonl', [
            {'id': i, 'caption': CAPTION, 'hit_token_limit': False} for i in self.image_ids])
        extraction.write_jsonl(natural / 'images.jsonl', image_rows)
        extraction.write_jsonl(natural / 'memories.jsonl', [
            {'id': 'task/r1', 'task': 'task', 'images': self.image_ids[:2]},
            {'id': 'task/r2', 'task': 'task', 'images': self.image_ids[:1]}])
        return SimpleNamespace(natural_root=natural, object_relation_root=root / 'output',
                               task=None, from_start=False, limit=0, extraction_stage='run',
                               extraction_max_new_tokens=2048, memory='object_relation')

    def test_prepare_deduplicates_and_excludes_query_images(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.extraction_stage = 'prepare'
            summary = extraction.build(args, lambda: self.fail('prepare loaded model'))
            self.assertEqual(summary['requested_images'], 2)
            self.assertEqual(summary['memory_references'], 3)
            self.assertEqual(summary['completed_images'], 0)
            requests = extraction.read_jsonl(args.object_relation_root / 'requests.jsonl')
            self.assertEqual([r['image_id'] for r in requests], self.image_ids[:2])
            self.assertEqual([r['cache_name'] for r in requests], ['image0', 'image1'])
            self.assertNotIn('image_sha256', requests[0])
            self.assertNotIn('caption_sha256', requests[0])
            self.assertEqual(requests[0]['prompt'], extraction.prompt(CAPTION))
            self.assertNotIn('task/r1', requests[0]['prompt'])

    def test_resume_limit_and_tamper_detection(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)])
            self.assertEqual(extraction.build(args, lambda: writer)['completed_images'], 1)
            self.assertEqual(writer.image_paths, [str(args.natural_root / 'image0.png')] * 2)
            self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction'])
            self.assertTrue((args.object_relation_root / 'generations/image0.json').is_file())
            for sent_prompt in writer.prompts:
                self.assertIn(CAPTION, sent_prompt)
            self.assertEqual(writer.prompts[1], extraction.relation_prompt(CAPTION, OBJECTS['objects']))
            self.assertIn('OBJECTS_JSON', writer.prompts[1])
            self.assertIn(json.dumps(OBJECTS['objects']), writer.prompts[1])
            self.assertEqual(extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)]))['status'], 'complete')
            extraction.build(args, lambda: self.fail('resume loaded model'))
            path = next((args.object_relation_root / 'generations').glob('*.json'))
            row = json.loads(path.read_text())
            row['relations'][0] = 'cat beside chair.'
            extraction.write_json(path, row)
            with self.assertRaisesRegex(ValueError, 'differs from raw'):
                extraction.build(args)

    def test_from_start_regenerates_with_limit(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)] * 2))
            untouched_path = args.object_relation_root / 'generations/image1.json'
            untouched = untouched_path.read_bytes()
            args.from_start = True
            args.limit = 1
            replacement = {'relations': [{'subject': 'cat', 'predicate': 'beside', 'object': 'chair'}]}
            writer = FakeWriter([json.dumps(OBJECTS), json.dumps(replacement)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(len(writer.prompts), 2)
            self.assertEqual(summary['completed_images'], 2)
            saved = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            self.assertEqual(saved['relations'], ['cat beside chair.'])
            self.assertEqual(saved['raw_objects_output'], json.dumps(OBJECTS))
            self.assertEqual(saved['raw_output'], json.dumps(replacement))
            self.assertEqual(untouched_path.read_bytes(), untouched)

    def test_new_endpoints_saved_and_reused_without_third_model_call(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            output = {'relations': [
                {'subject': 'child', 'predicate': 'holds', 'object': 'red ball'},
                {'subject': 'cat', 'predicate': 'watches', 'object': 'child'}]}
            writer = FakeWriter([json.dumps(OBJECTS), json.dumps(output)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction'])
            self.assertEqual(writer.image_paths, [str(args.natural_root / 'image0.png')] * 2)
            self.assertEqual(writer.prompts[1], extraction.relation_prompt(CAPTION, OBJECTS['objects']))
            self.assertEqual(summary['objects'], 5)
            self.assertEqual(summary['relations'], 2)
            path = args.object_relation_root / 'generations/image0.json'
            saved = json.loads(path.read_text())
            self.assertEqual(saved['initial_objects'], OBJECTS['objects'])
            self.assertEqual(saved['added_objects'], ['child', 'red ball'])
            self.assertEqual(saved['objects'], ['cat', 'chair', 'cup', 'child', 'red ball'])
            self.assertEqual(saved['raw_output'], json.dumps(output))
            self.assertEqual(saved['relations'], ['child holds red ball.', 'cat watches child.'])
            self.assertEqual(len(saved['attempts']), 1)
            args.extraction_stage = 'prepare'
            reloaded = extraction.build(args, lambda: self.fail('completed cache loaded model'))
            self.assertEqual(reloaded, summary)
            exported = extraction.read_jsonl(args.object_relation_root / 'extractions.jsonl')[0]
            for key in ('initial_objects', 'added_objects', 'objects', 'relations'):
                self.assertEqual(exported[key], saved[key])

    def test_legacy_triple_caches_upgrade_in_place_without_model(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            output = {'relations': [
                {'subject': ' child ', 'predicate': 'holds', 'object': 'red  ball'},
                {'subject': 'child', 'predicate': 'holds', 'object': 'red ball'},
                {'subject': 'cat', 'predicate': 'watches', 'object': 'child'}]}
            writer = FakeWriter([json.dumps(OBJECTS), json.dumps(output),
                                 json.dumps(OBJECTS), '{"relations": []}', '{"relations": []}'])
            summary = extraction.build(args, lambda: writer)
            paths = sorted((args.object_relation_root / 'generations').glob('*.json'))
            originals = []
            for path in paths:
                legacy = json.loads(path.read_text())
                legacy['relations'] = extraction.organize_relations(
                    extraction.parse_output(legacy['raw_output']))
                extraction.write_json(path, legacy)
                originals.append(legacy)
            exported = extraction.read_jsonl(args.object_relation_root / 'extractions.jsonl')
            for row, legacy in zip(exported, originals):
                row['relations'] = legacy['relations']
            extraction.write_jsonl(args.object_relation_root / 'extractions.jsonl', exported)

            reloaded = extraction.build(args, lambda: self.fail('cache upgrade loaded model'))
            self.assertEqual(reloaded, summary)
            self.assertEqual(sorted((args.object_relation_root / 'generations').glob('*.json')), paths)
            upgraded = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(upgraded[0]['relations'], ['child holds red ball.', 'cat watches child.'])
            self.assertEqual(upgraded[0]['generated_relations'], [
                'child holds red ball.', 'child holds red ball.', 'cat watches child.'])
            self.assertEqual(upgraded[0]['added_objects'], ['child', 'red ball'])
            self.assertEqual(upgraded[1]['relations'], [])
            for original, saved in zip(originals, upgraded):
                self.assertEqual({key: value for key, value in original.items() if key != 'relations'},
                                 {key: value for key, value in saved.items() if key != 'relations'})
            exported = extraction.read_jsonl(args.object_relation_root / 'extractions.jsonl')
            for row, saved in zip(exported, upgraded):
                for key in ('initial_objects', 'added_objects', 'objects', 'generated_relations', 'relations'):
                    self.assertEqual(row[key], saved[key])
            bytes_after_upgrade = [path.read_bytes() for path in paths]
            self.assertEqual(extraction.build(args, lambda: self.fail('upgraded cache loaded model')), summary)
            self.assertEqual([path.read_bytes() for path in paths], bytes_after_upgrade)

    def test_corrupt_legacy_cache_is_rejected_without_overwrite(self):
        for corrupt_field in ('relations', 'generated_relations', 'objects'):
            with self.subTest(corrupt_field=corrupt_field), TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                args.limit = 1
                extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)]))
                path = args.object_relation_root / 'generations/image0.json'
                legacy = json.loads(path.read_text())
                legacy['relations'] = extraction.organize_relations(extraction.parse_output(legacy['raw_output']))
                if corrupt_field == 'relations':
                    legacy['relations'][0]['predicate'] = 'beside'
                elif corrupt_field == 'generated_relations':
                    legacy['generated_relations'][0] = 'cat beside chair.'
                else:
                    legacy['objects'].append('unlisted object')
                extraction.write_json(path, legacy)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'differs from raw'):
                    extraction.build(args, lambda: self.fail('corrupt cache loaded model'))
                self.assertEqual(path.read_bytes(), before)

    def test_format_failure_retained_and_recoverable(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with self.assertRaises(ValueError):
                extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), 'invalid', '{}']))
            summary = json.loads((args.object_relation_root / 'summary.json').read_text())
            self.assertEqual(summary['failed_images'], 1)
            self.assertEqual(summary['completed_images'], 0)
            path = args.object_relation_root / 'generations/image0.json'
            failed = json.loads(path.read_text())
            self.assertEqual(failed['initial_objects'], OBJECTS['objects'])
            self.assertEqual(failed['objects'], OBJECTS['objects'])
            self.assertEqual(failed['raw_objects_output'], json.dumps(OBJECTS))
            self.assertEqual(len(failed['object_attempts']), 1)
            self.assertEqual(len(failed['attempts']), 2)
            args.limit = 1
            extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)]))
            self.assertEqual(len(json.loads(path.read_text())['attempts']), 3)

    def test_continue_on_error_preserves_failed_image_and_processes_next(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.continue_on_error = True
            writer = FakeWriter([json.dumps(OBJECTS), 'invalid', '{}',
                                 json.dumps(OBJECTS), json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['failed_images'], 1)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['pending_images'], 1)
            self.assertEqual(summary['status'], 'incomplete')
            self.assertEqual(summary['objects'], 3)
            self.assertEqual(summary['relations'], 1)
            self.assertEqual(writer.image_paths,
                             [str(args.natural_root / 'image0.png')] * 3
                             + [str(args.natural_root / 'image1.png')] * 2)
            failed = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['initial_objects'], OBJECTS['objects'])
            self.assertEqual([attempt['raw_output'] for attempt in failed['attempts']], ['invalid', '{}'])
            self.assertTrue(all(attempt['error'] for attempt in failed['attempts']))
            completed = json.loads((args.object_relation_root / 'generations/image1.json').read_text())
            self.assertEqual(completed['status'], 'completed')
            exported = extraction.read_jsonl(args.object_relation_root / 'extractions.jsonl')
            self.assertEqual([row['image_id'] for row in exported], ['image1'])

    def test_null_endpoint_retry_preserves_inputs_and_completes_new_object(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            malformed = json.dumps({'relations': [{'subject': 'cat', 'predicate': 'is black', 'object': None}]})
            recovered = {'relations': [{'subject': 'cat', 'predicate': 'near', 'object': 'window'}]}
            writer = FakeWriter([json.dumps(OBJECTS), malformed, json.dumps(recovered),
                                 json.dumps(OBJECTS), json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['completed_images'], 2)
            self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction',
                                             'relation_extraction', 'object_extraction', 'relation_extraction'])
            self.assertEqual(writer.image_paths,
                             [str(args.natural_root / 'image0.png')] * 3
                             + [str(args.natural_root / 'image1.png')] * 2)
            self.assertEqual(writer.max_tokens, [args.extraction_max_new_tokens] * 5)
            self.assertEqual(writer.generation_options, [{}] * 5)
            base_prompt = extraction.relation_prompt(CAPTION, OBJECTS['objects'])
            correction = (' Extract only relationships between two entities.'
                          ' Exclude standalone descriptions or attributes of a single entity.'
                          ' Both endpoints must be nonempty entity names, never null.')
            self.assertEqual(writer.prompts[2], base_prompt
                             + '\nRecheck both the image and caption. Previous issue: '
                             + 'relation fields must be nonempty strings.' + correction)
            self.assertEqual(writer.prompts[1], base_prompt)
            self.assertEqual(writer.prompts[4], base_prompt)
            self.assertNotIn(correction, base_prompt)
            saved = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            self.assertEqual(saved['attempts'][0]['raw_output'], malformed)
            self.assertEqual(saved['attempts'][0]['error'], 'relation fields must be nonempty strings')
            self.assertIsNone(saved['attempts'][1]['error'])
            self.assertEqual(saved['initial_objects'], OBJECTS['objects'])
            self.assertEqual(saved['added_objects'], ['window'])
            self.assertEqual(saved['objects'], ['cat', 'chair', 'cup', 'window'])
            self.assertEqual(saved['relations'], ['cat near window.'])

    def test_continue_on_error_limit_counts_failed_images(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.continue_on_error = True
            args.limit = 1
            writer = FakeWriter([json.dumps(OBJECTS), 'invalid', '{}'])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['failed_images'], 1)
            self.assertEqual(summary['completed_images'], 0)
            self.assertEqual(len(writer.prompts), 3)
            self.assertFalse((args.object_relation_root / 'generations/image1.json').exists())
            self.assertEqual(extraction.read_jsonl(args.object_relation_root / 'extractions.jsonl'), [])

    def test_continue_on_error_still_aborts_for_runtime_failure(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.continue_on_error = True
            writer = FakeWriter([json.dumps(OBJECTS), RuntimeError('model execution failed')])
            with self.assertRaisesRegex(RuntimeError, 'model execution failed'):
                extraction.build(args, lambda: writer)
            self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction'])
            self.assertFalse((args.object_relation_root / 'generations/image1.json').exists())
            summary = json.loads((args.object_relation_root / 'summary.json').read_text())
            self.assertEqual(summary['failed_images'], 1)
            self.assertEqual(summary['completed_images'], 0)
            failed = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(len(failed['attempts']), 1)
            self.assertEqual(failed['attempts'][0]['error'], 'model execution failed')

    def test_empty_first_response_is_rechecked_with_same_image(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter([json.dumps(OBJECTS), '{"relations": []}', json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['empty_images'], 0)
            self.assertEqual(summary['relations'], 1)
            self.assertEqual(writer.image_paths, [str(args.natural_root / 'image0.png')] * 3)
            self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction', 'relation_extraction'])
            self.assertIn('Recheck', writer.prompts[2])
            self.assertIn(CAPTION, writer.prompts[2])
            self.assertIn(json.dumps(OBJECTS['objects']), writer.prompts[2])

    def test_empty_after_recheck_remains_empty(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter([json.dumps(OBJECTS), '{"relations": []}', '{"relations": []}'])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(len(writer.prompts), 3)
            self.assertEqual(summary['empty_images'], 1)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['objects'], 3)
            self.assertEqual(summary['relations'], 0)

    def test_empty_initial_objects_can_be_recovered_from_relations(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter(['{"objects": []}', '{"objects": []}', json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(writer.stages, ['object_extraction', 'object_extraction', 'relation_extraction'])
            self.assertEqual(writer.image_paths, [str(args.natural_root / 'image0.png')] * 3)
            self.assertEqual(writer.prompts[2], extraction.relation_prompt(CAPTION, []))
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['objects'], 2)
            self.assertEqual(summary['relations'], 1)
            path = args.object_relation_root / 'generations/image0.json'
            saved = json.loads(path.read_text())
            self.assertEqual(saved['initial_objects'], [])
            self.assertEqual(saved['added_objects'], ['cat', 'chair'])
            self.assertEqual(saved['objects'], ['cat', 'chair'])
            self.assertEqual(saved['raw_output'], json.dumps(OUTPUT))
            self.assertEqual(saved['raw_objects_output'], '{"objects": []}')
            self.assertEqual(len(saved['attempts']), 1)
            self.assertEqual(saved['generated_relations'], ['cat sits on chair.'])
            self.assertEqual(len(saved['object_attempts']), 2)
            args.extraction_stage = 'prepare'
            self.assertEqual(extraction.build(args, lambda: self.fail('completed cache loaded model')), summary)

    def test_empty_both_stages_rechecked_and_cached(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter(['{"objects": []}', '{"objects": []}',
                                 '{"relations": []}', '{"relations": []}'])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(writer.stages, ['object_extraction', 'object_extraction',
                                             'relation_extraction', 'relation_extraction'])
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['empty_images'], 1)
            self.assertEqual(summary['objects'], 0)
            self.assertEqual(summary['relations'], 0)
            path = args.object_relation_root / 'generations/image0.json'
            saved = json.loads(path.read_text())
            for key in ('initial_objects', 'added_objects', 'objects', 'generated_relations', 'relations'):
                self.assertEqual(saved[key], [])
            self.assertEqual(saved['raw_output'], '{"relations": []}')
            self.assertEqual(len(saved['object_attempts']), 2)
            self.assertEqual(len(saved['attempts']), 2)
            args.extraction_stage = 'prepare'
            self.assertEqual(extraction.build(args, lambda: self.fail('empty cache loaded model')), summary)

    def test_cached_objects_must_match_first_stage_output(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            extraction.build(args, lambda: FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)]))
            path = args.object_relation_root / 'generations/image0.json'
            saved = json.loads(path.read_text())
            saved['objects'].append('unlisted object')
            extraction.write_json(path, saved)
            with self.assertRaisesRegex(ValueError, 'differs from raw'):
                extraction.build(args)

    def test_changed_input_and_bad_caption_rejected(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.extraction_stage = 'prepare'
            extraction.build(args)
            path = args.natural_root / 'captions.jsonl'
            rows = extraction.read_jsonl(path)
            rows[0]['caption'] += ' More text.'
            extraction.write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, 'sources/settings changed'):
                extraction.build(args)
            rows[0]['hit_token_limit'] = True
            extraction.write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, 'truncated caption'):
                extraction.prepare(args)

    def test_pipeline_dispatch(self):
        with patch.object(extraction, 'build') as build:
            args = SimpleNamespace(memory='object_relation')
            pipeline.offline(args)
            build.assert_called_once_with(args)

    def test_image_id_is_not_a_content_hash(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = args.natural_root / 'image0.png'
            Image.new('RGB', (8, 8), 'black').save(path)
            _, _, requests = extraction.prepare(args)
            self.assertEqual(requests[0]['image_id'], 'image0')
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                extraction.prepare(args)

    def test_cache_filename_without_path_traversal(self):
        root = Path('/tmp/output')
        self.assertEqual(extraction.cache_path(root, 'image_123'), root / 'generations/image_123.json')
        for image_id in ('../image', '/image', 'a/b', 'a\\b', '', '.', '..'):
            with self.subTest(image_id=image_id), self.assertRaises(ValueError):
                extraction.cache_path(root, image_id)

    def test_same_filename_in_different_tasks(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = args.natural_root / 'images.jsonl'
            images = extraction.read_jsonl(path)
            for i, task in enumerate(('Home', 'Health')):
                destination = args.natural_root / task / 'D1_IMG_001.jpg'
                destination.parent.mkdir()
                Image.new('RGB', (8, 8), 'white').save(destination)
                images[i]['path'] = str(destination)
            extraction.write_jsonl(path, images)
            _, _, requests = extraction.prepare(args)
            self.assertEqual([r['cache_name'] for r in requests],
                             ['Home__D1_IMG_001', 'Health__D1_IMG_001'])

    def test_retry_also_receives_image(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter(['invalid', json.dumps(OBJECTS), json.dumps(OUTPUT)])
            extraction.build(args, lambda: writer)
            self.assertEqual(writer.image_paths, [str(args.natural_root / 'image0.png')] * 3)
            self.assertEqual(writer.stages, ['object_extraction', 'object_extraction', 'relation_extraction'])
            self.assertEqual(writer.generation_options, [{}, {}, {}])

    def test_token_limit_recovery_uses_local_repetition_override(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            writer = FakeWriter([ValueError('generation reached token limit'),
                                 json.dumps(OBJECTS), json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(writer.stages, ['object_extraction', 'object_extraction', 'relation_extraction'])
            self.assertEqual(writer.generation_options, [{}, {'repetition_penalty': 1.1}, {}])
            saved = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            first, recovery = saved['object_attempts']
            self.assertEqual(first['error'], 'generation reached token limit')
            self.assertNotIn('generation_options', first)
            self.assertIsNone(recovery['error'])
            self.assertEqual(recovery['generation_options'], {'repetition_penalty': 1.1})
            self.assertEqual(recovery['raw_output'], json.dumps(OBJECTS))
            self.assertEqual(len(saved['attempts']), 1)
            self.assertNotIn('generation_options', saved['attempts'][0])
            self.assertEqual(saved['relations'], ['cat sits on chair.'])

    def test_repeated_object_token_limits_resume_with_main_objects(self):
        for token_limit in (2048, 256):
            with self.subTest(token_limit=token_limit), TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                args.limit = 1
                args.extraction_max_new_tokens = token_limit
                failed_writer = FakeWriter([ValueError('generation reached token limit'),
                                            ValueError('generation reached token limit')])
                with self.assertRaises(ValueError):
                    extraction.build(args, lambda: failed_writer)
                self.assertEqual(failed_writer.max_tokens, [token_limit, token_limit])
                self.assertEqual(failed_writer.generation_options, [{}, {'repetition_penalty': 1.1}])
                path = args.object_relation_root / 'generations/image0.json'
                failed = json.loads(path.read_text())
                self.assertEqual(len(failed['object_attempts']), 2)
                self.assertTrue(all('token limit' in row['error'] for row in failed['object_attempts']))
                writer = FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)])
                summary = extraction.build(args, lambda: writer)
                self.assertEqual(summary['completed_images'], 1)
                self.assertEqual(summary['failed_images'], 0)
                self.assertEqual(writer.stages, ['object_extraction', 'relation_extraction'])
                self.assertEqual(writer.max_tokens, [min(token_limit, 512), token_limit])
                self.assertEqual(writer.generation_options, [{'repetition_penalty': 1.0}, {}])
                self.assertTrue(writer.prompts[0].startswith(extraction.prompt(CAPTION)))
                self.assertEqual(writer.prompts[1], extraction.relation_prompt(CAPTION, OBJECTS['objects']))
                saved = json.loads(path.read_text())
                recovery = saved['object_attempts'][-1]
                self.assertEqual(recovery['retry_strategy'], 'main_objects')
                self.assertEqual(recovery['generation_options'],
                                 {'repetition_penalty': 1.0, 'max_new_tokens': min(token_limit, 512)})
                self.assertIsNone(recovery['error'])
                self.assertNotIn('retry_strategy', saved['attempts'][0])
                self.assertNotIn('generation_options', saved['attempts'][0])

    def test_initial_generation_allows_more_than_30_objects_and_relations(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            objects = [f'object {i}' for i in range(31)]
            relations = [{'subject': objects[i], 'predicate': 'near', 'object': objects[(i + 1) % 31]}
                         for i in range(31)]
            writer = FakeWriter([json.dumps({'objects': objects}), json.dumps({'relations': relations})])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['objects'], 31)
            self.assertEqual(summary['relations'], 31)
            self.assertEqual(writer.generation_options, [{}, {}])

    def test_object_token_retry_over_30_is_rejected_then_resumes_with_main_objects(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            too_many = json.dumps({'objects': [f'object {i}' for i in range(31)]})
            writer = FakeWriter([ValueError('generation reached token limit'), too_many])
            with self.assertRaisesRegex(ValueError, extraction.OBJECT_RETRY_LIMIT_ERROR):
                extraction.build(args, lambda: writer)
            self.assertEqual(writer.generation_options, [{}, {'repetition_penalty': 1.1}])
            path = args.object_relation_root / 'generations/image0.json'
            failed = json.loads(path.read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['object_attempts'][-1]['raw_output'], too_many)
            self.assertEqual(failed['object_attempts'][-1]['error'], extraction.OBJECT_RETRY_LIMIT_ERROR)
            self.assertEqual(failed['attempts'], [])
            recovery = FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: recovery)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['failed_images'], 0)
            self.assertEqual(recovery.max_tokens, [512, args.extraction_max_new_tokens])
            self.assertEqual(recovery.generation_options, [{'repetition_penalty': 1.0}, {}])
            saved = json.loads(path.read_text())
            self.assertEqual(saved['object_attempts'][-1]['retry_strategy'], 'main_objects')
            self.assertEqual(saved['objects'], OBJECTS['objects'])

    def test_relation_token_retry_over_30_is_rejected(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            relations = [{'subject': 'cat', 'predicate': 'near', 'object': f'object {i}'}
                         for i in range(31)]
            too_many = json.dumps({'relations': relations})
            writer = FakeWriter([json.dumps(OBJECTS), ValueError('generation reached token limit'), too_many])
            with self.assertRaisesRegex(ValueError, 'relation retry returned more than 30 distinct relations'):
                extraction.build(args, lambda: writer)
            self.assertEqual(writer.generation_options, [{}, {}, {'repetition_penalty': 1.1}])
            path = args.object_relation_root / 'generations/image0.json'
            failed = json.loads(path.read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['initial_objects'], OBJECTS['objects'])
            self.assertEqual(failed['attempts'][-1]['raw_output'], too_many)
            self.assertNotIn('retry_strategy', failed['attempts'][-1])
            summary = json.loads((args.object_relation_root / 'summary.json').read_text())
            self.assertEqual(summary['completed_images'], 0)
            self.assertEqual(summary['failed_images'], 1)

    def test_token_retry_limits_count_distinct_objects_and_relations(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            objects = [f'object {i}' for i in range(30)] + [' object  0 ']
            relations = [{'subject': 'object 0', 'predicate': 'near', 'object': f'object {i}'}
                         for i in range(30)]
            relations.append({'subject': ' object  0 ', 'predicate': ' near ', 'object': 'object 0'})
            raw_objects = json.dumps({'objects': objects})
            raw_relations = json.dumps({'relations': relations})
            writer = FakeWriter([ValueError('generation reached token limit'), raw_objects,
                                 ValueError('generation reached token limit'), raw_relations])
            summary = extraction.build(args, lambda: writer)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(summary['objects'], 30)
            self.assertEqual(summary['relations'], 30)
            self.assertEqual(writer.generation_options,
                             [{}, {'repetition_penalty': 1.1}, {}, {'repetition_penalty': 1.1}])
            saved = json.loads((args.object_relation_root / 'generations/image0.json').read_text())
            self.assertEqual(saved['raw_objects_output'], raw_objects)
            self.assertEqual(saved['raw_output'], raw_relations)
            self.assertEqual(len(saved['initial_objects']), 30)
            self.assertEqual(len(saved['generated_relations']), 31)
            self.assertEqual(len(saved['relations']), 30)

    def test_main_object_limit_rejects_and_stays_active_on_retry_and_resume(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.limit = 1
            with self.assertRaises(ValueError):
                extraction.build(args, lambda: FakeWriter([
                    ValueError('generation reached token limit'),
                    ValueError('generation reached token limit')]))
            too_many = json.dumps({'objects': ['cat', 'chair', 'cup', 'ball', 'child', 'table']})
            writer = FakeWriter([too_many, too_many])
            with self.assertRaises(ValueError):
                extraction.build(args, lambda: writer)
            self.assertEqual(writer.stages, ['object_extraction', 'object_extraction'])
            self.assertEqual(writer.max_tokens, [512, 512])
            self.assertEqual(writer.generation_options, [{'repetition_penalty': 1.0}] * 2)
            path = args.object_relation_root / 'generations/image0.json'
            failed = json.loads(path.read_text())
            self.assertEqual(failed['status'], 'failed')
            for attempt in failed['object_attempts'][-2:]:
                self.assertEqual(attempt['retry_strategy'], 'main_objects')
                self.assertEqual(attempt['generation_options'],
                                 {'repetition_penalty': 1.0, 'max_new_tokens': 512})
                self.assertEqual(attempt['raw_output'], too_many)
                self.assertTrue(attempt['error'])
            recovery_writer = FakeWriter([json.dumps(OBJECTS), json.dumps(OUTPUT)])
            summary = extraction.build(args, lambda: recovery_writer)
            self.assertEqual(summary['completed_images'], 1)
            self.assertEqual(recovery_writer.max_tokens, [512, args.extraction_max_new_tokens])
            self.assertEqual(recovery_writer.generation_options, [{'repetition_penalty': 1.0}, {}])
            saved = json.loads(path.read_text())
            self.assertEqual(saved['object_attempts'][-1]['retry_strategy'], 'main_objects')
            self.assertEqual(saved['initial_objects'], OBJECTS['objects'])

    def test_old_evidence_cache_rejected(self):
        with TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.extraction_stage = 'prepare'
            extraction.build(args)
            path = args.object_relation_root / 'run_config.json'
            old = json.loads(path.read_text())
            old['format'] = 'image_caption_object_relation_v2'
            extraction.write_json(path, old)
            with self.assertRaisesRegex(ValueError, 'sources/settings changed'):
                extraction.build(args)


if __name__ == '__main__':
    unittest.main()
