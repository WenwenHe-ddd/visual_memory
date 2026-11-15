"""Behavioral invariants for the isolated image-only HippoMM adaptation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.memory import hippomm as method
from scripts.memory import hippomm_core as core
from scripts.memory import hippomm_inputs as inputs


def event(eid, task='A', mids=None):
    return {'event_id': eid, 'task': task, 'representative_image': eid,
            'caption': 'A red chair.', 'summary': 'A chair was returned.',
            'source_images': [], 'source_round_ids': mids or [], 'summary': 'A chair was returned.'}


class CoreTests(unittest.TestCase):
    def test_redundancy_keeps_all_dialogue_and_never_merges_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, rounds, vectors = [], [], {}
            for i, (task, v) in enumerate([('A', [1, 0]), ('A', [.99, .01]), ('B', [1, 0])]):
                path = Path(tmp) / f'{i}.png'
                path.write_bytes(str(i).encode())
                h, mid = inputs.sha(path), f'{task}/r{i}'
                rounds.append(dict(id=mid, task=task, round=mid, session_id='s', timestamp=str(i),
                    dialogue='buy' if i == 0 else 'returned', insertion_index=i,
                    images=[dict(image_sha256=h, image_id=f'image{i}')]))
                images.append(dict(task=task, image_sha256=h, path=str(path), caption='caption', source_round_ids=[mid]))
                vectors[path] = v
            events, stored, mapping = method.construct_events(rounds, images, lambda p: vectors[p])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]['source_round_ids'], ['A/r0', 'A/r1'])
        self.assertEqual(events[0]['source_images'][1]['relation'], 'visually_redundant')
        self.assertFalse(mapping[1]['retained'])
        self.assertEqual(set(stored), {e['event_id'] for e in events})
        self.assertNotIn('path', json.dumps(events))
        self.assertNotIn('path', json.dumps(mapping))

    def test_fallback_is_per_image_before_global_ranking(self):
        events = [event('visual'), event('fallback'), event('excluded'), event('foreign', 'B')]
        vectors = {'visual': [.5, np.sqrt(.75)], 'fallback': [.2, np.sqrt(.96)],
                   'excluded': [0, 1], 'foreign': [1, 0]}
        prompts = []

        def llm(role, prompt):
            prompts.append(prompt)
            return json.dumps({'selected_event_ids': ['fallback'] if '"fallback"' in prompt else []})

        ranked, audit = core.detailed_search({'task': 'A', 'question': 'chair', 'id': 'q',
            'options': {'A': 'SECRET'}, 'gold_ids': ['SECRET']}, events, vectors, [1, 0], llm, top_k=2)
        self.assertEqual([r['event_id'] for r in ranked], ['fallback', 'visual'])
        self.assertEqual(ranked[0]['score'], .6)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(len(audit), 3)
        self.assertFalse(any('SECRET' in p for p in prompts))

    def test_fast_reads_all_same_task_summaries_but_no_raw_dialogue(self):
        events = [event('e1', mids=['r1']), event('e2', mids=['r2']), event('secret', 'B')]
        calls = []
        q = {'id': 'q', 'task': 'A', 'question': 'Why?', 'options': {'A': 'OPTIONS_ONLY_FINAL'}}

        def llm(role, prompt):
            calls.append((role, prompt))
            if role == 'fast':
                return json.dumps({'answer': 'returned', 'confidence': .9, 'evidence_event_ids': ['e1']})
            return 'A'

        result = core.answer_question(q, events, {}, {'r1': 'RAW_SECRET'}, llm,
                                      lambda _: self.fail('fast path encoded a query'))
        self.assertEqual(result['route'], 'fast')
        self.assertEqual(result['context_round_ids'], ['r1', 'r2'])
        self.assertEqual([r for r, _ in calls], ['fast', 'format_mcq'])
        self.assertIn('e2', calls[0][1])
        self.assertNotIn('secret', calls[0][1])
        self.assertNotIn('RAW_SECRET', calls[0][1])
        self.assertNotIn('OPTIONS_ONLY_FINAL', calls[0][1])
        self.assertIn('OPTIONS_ONLY_FINAL', calls[1][1])

    def test_invalid_confidence_routes_to_detail_and_keeps_only_retrieved_evidence(self):
        events = [event('e1', mids=['r1']), event('e2', mids=['r2'])]
        replies = {'fast': '{"answer":"guess","confidence":NaN,"evidence_event_ids":[]}',
                   'rewrite': '{"visual_query":"red chair"}',
                   'caption_fallback': '{"selected_event_ids":[]}', 'answer': 'The chair.'}
        calls = []

        def llm(role, prompt):
            calls.append((role, prompt))
            return replies[role]

        result = core.answer_question({'id': 'q', 'task': 'A', 'question': 'Which?'}, events,
            {'e1': [1, 0], 'e2': [0, 1]}, {'r1': {'dialogue': 'buy'}, 'r2': {'dialogue': 'DO_NOT_READ'}},
            llm, lambda _: ([1, 0], {}))
        self.assertEqual(result['route'], 'detailed')
        self.assertIsNotNone(result['fast_parse_error'])
        self.assertEqual(result['selected_event_ids'], ['e1'])
        self.assertNotIn('DO_NOT_READ', calls[-1][1])

    def test_empty_retrieval_is_preserved_and_not_filled(self):
        def llm(role, prompt):
            return {'fast': '{"answer":null,"confidence":0,"evidence_event_ids":[]}',
                    'rewrite': '{"visual_query":"blue couch"}',
                    'caption_fallback': '{"selected_event_ids":[]}',
                    'answer': 'Not mentioned.'}[role]
        result = core.answer_question({'id': 'q', 'task': 'A', 'question': 'Where?'}, [event('e')],
            {'e': [0, 1]}, {}, llm, lambda _: ([1, 0], {}))
        self.assertEqual(result['selected_event_ids'], [])
        self.assertEqual(result['context_round_ids'], [])
        self.assertEqual(result['answer'], 'Not mentioned.')

    def test_strict_threshold_and_no_unknown_citations(self):
        with self.assertRaises(ValueError):
            core.parse_fast('{"answer":"x","confidence":0.9,"evidence_event_ids":["foreign"]}', ['e'])
        with self.assertRaises(ValueError):
            core.parse_selection('{"selected_event_ids":["other"]}', 'e')
        with self.assertRaises(ValueError):
            core.unit_vector([float('nan')])
        with self.assertRaises(ValueError):
            core.parse_query('{"visual_query":"one"}')


class PipelineTests(unittest.TestCase):
    def test_new_branch_does_not_enter_legacy_dispatch_or_home_defaults(self):
        from scripts import visual_memory_pipeline as cli
        with patch.object(method, 'prepare') as prep, patch.object(cli, 'prepare_online') as legacy:
            cli.main(['online', '--memory', 'hippomm', '--dataset', 'mem_gallery'])
        args = prep.call_args.args[0]
        self.assertIsNone(args.question_ids)
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.hippomm_confidence_threshold, .75)
        self.assertEqual(args.output, method.ROOT / 'results/visual_memory_pipeline/hippomm/mem_gallery/online')
        legacy.assert_not_called()

    def test_new_offline_defaults_no_image_filters(self):
        from scripts import visual_memory_pipeline as cli
        with patch.object(method, 'build') as build, patch.object(cli, 'offline') as legacy:
            cli.main(['offline', '--memory', 'hippomm'])
        self.assertIsNone(build.call_args.args[0].image_files)
        legacy.assert_not_called()

    def test_other_method_defaults_unchanged(self):
        from scripts import visual_memory_pipeline as cli
        with patch.object(cli, 'prepare_online') as prep, patch.object(method, 'pipeline') as new:
            cli.main(['online', '--memory', 'natural'])
        args = prep.call_args.args[0]
        self.assertEqual(args.question_ids, ['Q8', 'Q10', 'Q27'])
        self.assertEqual(args.top_k, 5)
        new.assert_not_called()

    def test_conflicting_flags_and_nan_rejected_before_work(self):
        from scripts import visual_memory_pipeline as cli
        for extra in (['--online-weighting', 'none'], ['--hippomm-confidence-threshold', 'nan'],
                      ['--reader-backend', 'api'], ['--output-root', '/tmp/other']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main(['online', '--memory', 'hippomm', *extra])


class StoreTests(unittest.TestCase):
    def test_store_is_queryable_after_historical_image_and_input_sources_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This store is a synthetic integration fixture, not a model result.
            source = root / 'original.png'
            source.write_bytes(b'original')
            build_contract = {'source_hashes': {str(source): inputs.sha(source)}}
            signed = method.contract_at(root / 'build_contract.json', build_contract)
            e = event('e', mids=['A/r'])
            rounds = [{'id': 'A/r', 'task': 'A', 'images': [{'image_id': 'i'}]}]
            method.write_lines(root / 'events.jsonl', [e])
            method.write_lines(root / 'dialogues.jsonl', rounds)
            method.write_lines(root / 'image_mapping.jsonl', [])
            np.savez(root / 'visual_embeddings.npz', event_ids=np.array(['e']), vectors=np.array([[1., 0.]]))
            memory = {'offline_method': 'hippomm', 'schema': 1, 'dataset': 'memeye',
                'build_signature': signed['signature'], 'counts': {'events': 1, 'dialogue_rounds': 1},
                'artifact_sha256': {p: inputs.sha(root / p) for p in
                    ['events.jsonl', 'dialogues.jsonl', 'image_mapping.jsonl', 'visual_embeddings.npz']}}
            memory['signature'] = method.digest(memory)
            method.dump(root / 'memory.json', memory)
            source.unlink()
            _, events, vectors, dialogues = method.verify_memory(root)
            self.assertEqual(events[0]['event_id'], 'e')
            self.assertEqual(list(vectors), ['e'])
            self.assertIn('A/r', dialogues)
            with (root / 'events.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                method.verify_memory(root)

    def test_incomplete_and_corrupt_responses_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = {'id': 'q', 'question': 'where?'}
            response = {'id': 'q', 'experiment_signature': 'signature', 'question_sha256': method.digest(q),
                        'answer': 'answer'}
            response['record_sha256'] = method.digest(response)
            method.write_lines(root / 'qa_responses.jsonl', [response, response])
            with self.assertRaises(ValueError):
                method.checked_responses(root, {'signature': 'signature'}, [q])
            response['answer'] = 'tampered'
            method.write_lines(root / 'qa_responses.jsonl', [response])
            with self.assertRaises(ValueError):
                method.checked_responses(root, {'signature': 'signature'}, [q])


if __name__ == '__main__':
    unittest.main()
