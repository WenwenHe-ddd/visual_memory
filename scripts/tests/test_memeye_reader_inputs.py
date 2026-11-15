"""Guard the fixed-evidence, gold-free boundary of the API reader experiment."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation.memeye_reader_inputs import (
    REQUEST_FIELDS, commit_artifacts, digest, paired_summary,
    request_from_row, sha, validate_responses, verify_hash,
)


class MemEyeReaderInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.system = 'Return one letter.'
        self.original = {
            'id': 'Scene/Q1', 'task': 'Scene', 'question': 'Which color?',
            'options': dict(zip('ABCD', ['blue', 'green', 'red', 'white'])),
            'gold_choice': 'B', 'gold_ids': ['PRIVATE_GOLD_REFERENCE'], 'query_image_paths': [],
        }
        self.memories = {f'Scene/R{i}': f'User: Look at {i}.\nMemory description: Original full caption {i}.'
                         for i in range(5)}
        text = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
                + '\n\n'.join(f'Memory {i}: {value}' for i, value in enumerate(self.memories.values(), 1))
                + '\n\nQuestion:\nWhich color?\n\nOptions:\nA. blue\nB. green\nC. red\nD. white'
                + '\n\nAnswer with one letter only.')
        self.row = {**self.original, 'record_id': 'Scene/Q1::baseline_question_only',
                    'condition': 'baseline_question_only', 'reader_text': text, 'query_images': [],
                    'top_memory_ids': list(self.memories),
                    'candidate_statements': {'A': 'PRIVATE_CANDIDATE'},
                    'private_answer': 'PRIVATE_LABEL'}
        self.row['payload_sha256'] = digest({'system': self.system, 'user_text': text, 'query_images': []})

    def test_request_has_only_allowlisted_fields_and_exact_evidence(self):
        request = request_from_row(self.row, self.original, self.memories, self.system)
        self.assertEqual(set(request), set(REQUEST_FIELDS))
        self.assertEqual(request['reader_text'], self.row['reader_text'])
        self.assertNotIn('gold_choice', request)
        self.assertNotIn('gold_ids', request)
        self.assertNotIn('PRIVATE_', str(request))

    def test_modified_prompt_and_original_image_bytes_are_rejected(self):
        changed = copy.deepcopy(self.row)
        changed['reader_text'] += '\nAdded clue'
        with self.assertRaisesRegex(ValueError, 'evidence differs'):
            request_from_row(changed, self.original, self.memories, self.system)
        path = self.directory / 'image.png'
        path.write_bytes(b'original pixels')
        image = {'path': str(path), 'sha256': sha(path)}
        changed = copy.deepcopy(self.row)
        original = copy.deepcopy(self.original)
        changed['query_images'] = [image]
        changed['query_image_paths'] = original['query_image_paths'] = [str(path)]
        changed['payload_sha256'] = digest({'system': self.system, 'user_text': changed['reader_text'],
                                             'query_images': [image]})
        request_from_row(changed, original, self.memories, self.system)
        path.write_bytes(b'changed pixels')
        with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
            request_from_row(changed, original, self.memories, self.system)

    def test_reentry_preserves_bytes_and_rejects_drift_before_writing(self):
        artifacts = {'requests.jsonl': b'original\n', 'gold.jsonl': b'label\n'}
        commit_artifacts(self.directory, artifacts)
        times = {p.name: p.stat().st_mtime_ns for p in self.directory.iterdir()}
        commit_artifacts(self.directory, artifacts)
        self.assertEqual(times, {p.name: p.stat().st_mtime_ns for p in self.directory.iterdir()})
        (self.directory / 'responses.jsonl').write_text('already started\n')
        with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
            commit_artifacts(self.directory, {**artifacts, 'requests.jsonl': b'new\n'})
        with self.assertRaisesRegex(ValueError, 'after responses exist'):
            commit_artifacts(self.directory, {**artifacts, 'new.json': b'new\n'})
        self.assertFalse((self.directory / 'new.json').exists())
        with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
            verify_hash(self.directory / 'requests.jsonl', '0' * 64)

    def test_reference_scoring_is_recomputed_and_sharing_must_match(self):
        second = {**self.row, 'record_id': 'Scene/Q1::master_answer_query', 'condition': 'master_answer_query'}
        protocol = {'experiment_run_id': 'run', 'experiment_signature': 'signature'}
        first = {
            'record_id': self.row['record_id'], 'question_id': self.row['id'], 'condition': self.row['condition'],
            'run_id': 'run', 'experiment_signature': 'signature', 'payload_sha256': self.row['payload_sha256'],
            'gold_choice': 'B', 'raw_response': 'B', 'predicted_choice': 'B', 'correct': True,
            'status': 'ok', 'hit_token_limit': False, 'historical_response_reuse': False,
            'query_image_count': 0, 'batch_size': 1, 'fresh_inference_executed_for_this_record': True,
            'same_run_shared_inference_record_id': None, 'rendered_prompt_sha256': 'rendered',
            'generated_token_ids': [1, 2], 'execution_id': 'execution',
        }
        shared = {**first, 'record_id': second['record_id'], 'condition': second['condition'],
                  'fresh_inference_executed_for_this_record': False,
                  'same_run_shared_inference_record_id': first['record_id']}
        validate_responses([self.row, second], [first, shared], protocol)
        with self.assertRaisesRegex(ValueError, 'scoring mismatch'):
            validate_responses([self.row, second], [{**first, 'correct': False}, shared], protocol)
        with self.assertRaisesRegex(ValueError, 'shared inference mismatch'):
            validate_responses([self.row, second], [first, {**shared, 'generated_token_ids': [3]}], protocol)

    def test_paired_summary_uses_question_denominator_and_both_directions(self):
        pairs = [
            {'baseline_correct': False, 'expanded_correct': True, 'baseline_answer': 'A', 'expanded_answer': 'B'},
            {'baseline_correct': True, 'expanded_correct': False, 'baseline_answer': 'B', 'expanded_answer': 'A'},
            {'baseline_correct': True, 'expanded_correct': True, 'baseline_answer': 'B', 'expanded_answer': 'B'},
        ]
        for row in pairs:
            row.update(same_reader_payload=False, same_top5_order=False, same_top5_set=False)
        result = paired_summary(pairs)
        self.assertEqual(result['questions'], 3)
        self.assertEqual((result['baseline_correct'], result['expanded_correct']), (2, 2))
        self.assertEqual((result['wrong_to_correct'], result['correct_to_wrong']), (1, 1))
        self.assertEqual(result['answer_changed'], 2)
        self.assertEqual(result['delta_percentage_points'], 0)
        self.assertEqual(result['baseline_accuracy'], 2 / 3)


if __name__ == '__main__':
    unittest.main()
