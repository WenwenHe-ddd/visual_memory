"""Complete but tiny-text synthetic paired results; never calls an API."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from scripts.evaluation import memeye_api_reader as reader


class MemEyeAPIAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'inputs').mkdir()
        image = self.root / 'query.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\nsynthetic original query pixels')
        image_record = {'path': str(image), 'sha256': reader.sha(image)}
        self.records, gold, pairs, qwen = [], [], [], []
        contract = {'model': 'gpt-4o-mini', 'reader_system_prompt': 'One letter.',
                    'max_tokens': 16, 'seed': 42, 'temperature': 0, 'image_detail': 'high',
                    'interpretation': 'Synthetic fixed-order test, not real benchmark results.'}
        wire, originals = [], {}
        # analyze intentionally enforces the real shape. Texts are minimal fixtures.
        for index in range(371):
            qid = 'Q' + str(index)
            domain = 'Domain' + str(index % 8)
            same = index < 23
            has_image = index < 8
            gold.append({'id': qid, 'gold_choice': 'A'})
            pairs.append({'question_id': qid, 'domain': domain, 'query_has_image': has_image,
                          'same_reader_payload': same, 'same_top5_order': same, 'same_top5_set': same})
            for condition in reader.CONDITIONS:
                text = qid if same else qid + condition
                images = [image_record] if has_image else []
                payload = reader.digest({'text': text, 'images': images})
                row = {'record_id': qid + '::' + condition, 'id': qid, 'task': domain,
                       'condition': condition, 'reader_text': text, 'query_images': images,
                       'top_memory_ids': [domain + '/M' + str(k) for k in range(5)],
                       'payload_sha256': payload}
                body_hash = reader.digest(reader.request_body({'contract': contract}, row))
                self.records.append(row)
                wire.append({'record_id': row['record_id'], 'payload_sha256': payload,
                             'request_sha256': body_hash, 'query_image_count': len(images)})
                answer = 'B' if (index == 1 or
                                  (condition == reader.CONDITIONS[0] and index in {23, 25}) or
                                  (condition == reader.CONDITIONS[1] and index == 24)) else 'A'
                # One invalid answer per condition stays in the denominator.
                if index == 30:
                    answer = 'Unparseable answer'
                originals.setdefault(payload, {
                    'record_id': row['record_id'], 'id': qid, 'task': domain, 'condition': condition,
                    'payload_sha256': payload, 'request_sha256': body_hash, 'raw_response': answer,
                    'predicted_choice': reader.parse_choice(answer),
                    'returned_model': 'gpt-4o-mini-2024-07-18', 'response_id': 'response-' + payload,
                    'execution_id': 'synthetic-execution', 'run_id': 'synthetic-run',
                    'usage': None if index == 0 else {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11},
                    'status': 'format_error' if index == 30 else 'ok',
                    'hit_token_limit': False, 'historical_response_reuse': False,
                })
                qwen.append({'record_id': row['record_id'], 'condition': condition, 'payload_sha256': payload,
                             'raw_response': 'B' if index == 0 else 'A' if index == 1 else answer})
        reader.write_lines(self.root / 'inputs/requests.jsonl', self.records)
        reader.write_lines(self.root / 'inputs/gold.jsonl', gold)
        reader.dump(self.root / 'inputs/pairs.json', pairs)
        reader.dump(self.root / 'inputs/qwen_reference.json', {
            'responses': qwen, 'summary': {'overall': {
                'baseline_correct': 367, 'expanded_correct': 368,
                'baseline_accuracy': 367 / 371, 'expanded_accuracy': 368 / 371,
                'delta_percentage_points': 100 / 371,
            }},
        })
        contract['source_sha256'] = {str(path): reader.sha(path) for path in (self.root / 'inputs').iterdir()}
        contract['source_sha256'][str(image)] = reader.sha(image)
        contract['request_manifest_sha256'] = reader.digest(wire)
        signature = reader.digest(contract)
        reader.dump(self.root / 'protocol.json', {
            'run_id': 'synthetic-run', 'experiment_signature': signature, 'contract': contract,
        })
        reader.write_lines(self.root / 'api_request_manifest.jsonl', wire)
        for item in originals.values():
            item['experiment_signature'] = signature
        reader.write_lines(self.root / 'fresh_responses.jsonl', list(originals.values()))
        logical = reader.logical_responses(self.records, originals)
        reader.write_lines(self.root / 'responses.jsonl', logical)
        reader.dump(self.root / 'validation_output.json', {
            'passed': True, 'fresh_requests': 719, 'shared_records': 23,
            'returned_models': ['gpt-4o-mini-2024-07-18'], 'format_errors': 2,
            'empty_responses': 0, 'token_limit_records': 0,
            'responses_sha256': reader.sha(self.root / 'responses.jsonl'),
            'fresh_responses_sha256': reader.sha(self.root / 'fresh_responses.jsonl'),
        })

    def analyze_quietly(self):
        with redirect_stdout(io.StringIO()):
            return reader.analyze(self.root)

    def replace_logical(self, responses):
        reader.write_lines(self.root / 'responses.jsonl', responses)
        validation = reader.read(self.root / 'validation_output.json')
        validation['responses_sha256'] = reader.sha(self.root / 'responses.jsonl')
        reader.dump(self.root / 'validation_output.json', validation)

    def test_denominator_changes_shared_usage_and_missing_usage(self):
        summary = self.analyze_quietly()
        overall = summary['overall']
        self.assertEqual(overall['questions'], 371)
        self.assertEqual((overall['baseline_correct'], overall['expanded_correct']), (367, 368))
        self.assertEqual((overall['wrong_to_correct'], overall['correct_to_wrong']), (2, 1))
        self.assertAlmostEqual(overall['baseline_accuracy'], 367 / 371)
        self.assertAlmostEqual(overall['expanded_accuracy'], 368 / 371)
        self.assertAlmostEqual(overall['delta_percentage_points'], 100 / 371)
        self.assertEqual(overall['mcnemar_exact_p'], 1.)
        self.assertEqual(sum(v['questions'] for v in summary['by_domain'].values()), 371)
        self.assertEqual(summary['by_query_modality']['text_only']['questions'], 363)
        self.assertEqual(summary['by_query_modality']['with_query_image']['questions'], 8)
        self.assertEqual(summary['usage_fresh_requests_only'], {
            'prompt_tokens': 7180, 'completion_tokens': 718, 'total_tokens': 7898,
        })
        self.assertEqual(summary['validation']['shared_records'], 23)
        self.assertEqual(summary['validation']['format_errors'], 2)
        for condition in reader.CONDITIONS:
            self.assertEqual(summary['reader_model_changes'][condition], {
                'qwen_wrong_api_correct': 1, 'qwen_correct_api_wrong': 1,
            })
        self.assertEqual(len(reader.read(self.root / 'paired_results.json')), 371)
        self.assertIn('367/371', (self.root / 'comparison.md').read_text())

    def test_raw_file_tamper_rejected_before_analysis(self):
        path = self.root / 'responses.jsonl'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Response output hash mismatch'):
            self.analyze_quietly()

    def test_changed_shared_source_rejected_even_with_refreshed_file_hash(self):
        responses = reader.lines(self.root / 'responses.jsonl')
        shared = next(r for r in responses if not r['fresh_inference_executed_for_this_record'])
        shared['same_run_shared_inference_record_id'] = 'Q999::baseline_question_only'
        self.replace_logical(responses)
        with self.assertRaisesRegex(ValueError, 'Shared inference source mismatch'):
            self.analyze_quietly()

    def test_changed_shared_answer_rejected_even_with_refreshed_file_hash(self):
        responses = reader.lines(self.root / 'responses.jsonl')
        shared = next(r for r in responses if not r['fresh_inference_executed_for_this_record'])
        shared['raw_response'] = 'C'
        shared['predicted_choice'] = 'C'
        self.replace_logical(responses)
        with self.assertRaisesRegex(ValueError, 'Shared/fresh result differs'):
            self.analyze_quietly()

    def test_missing_condition_is_not_silently_removed_from_denominator(self):
        responses = reader.lines(self.root / 'responses.jsonl')
        self.replace_logical(responses[:-1])
        with self.assertRaisesRegex(ValueError, 'Incomplete paired benchmark coverage'):
            self.analyze_quietly()


if __name__ == '__main__':
    unittest.main()
