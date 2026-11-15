"""Mocked full-backbone stages: no external requests and no historical outputs."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import memeye_deepseek_flash_alias as runner
from scripts.evaluation.deepseek_flash_alias_transport import (
    APIError, DEEPSEEK_MODEL, build_request, load_config,
)


class DeepSeekFlashAliasRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / '.env'
        self.env.write_text('DEEPSEEKV4_API_KEY=FAKE_FULL_BACKBONE_KEY\nDEEPSEEKV4_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1\n')
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.images = []
        for index in range(2):
            path = self.root / f'image{index}.png'
            path.write_bytes(b'\x89PNG\r\n\x1a\noriginal image ' + str(index).encode())
            self.images.append({'path': str(path), 'sha256': runner.sha(path)})
        config = load_config(self.env, DEEPSEEK_MODEL)
        self.protocol = {'run_id': 'full-backbone-test-run', 'experiment_signature': 'synthetic-signature',
                         'contract': {'model': DEEPSEEK_MODEL, 'api_config': config.public_dict(),
                                      'api_env_file': str(self.env), 'caption_max_output_tokens': 1024,
                                      'reader_max_output_tokens': 16, 'image_detail': 'high',
                                      'workers': 1, 'max_attempts': 2, 'timeout_seconds': 1}}
        # The heavyweight public-input verifier has its own tests. This fixture
        # focuses on the additional per-stage sources, wire hashes and results.
        verification = patch.object(runner, 'verify', return_value=self.protocol)
        verification.start()
        self.addCleanup(verification.stop)
        self.source = self.root / 'source.txt'
        self.source.write_text('new-model-only source, no Qwen caption or ranking')
        self.records = [
            {'record_id': 'first', 'user_text': 'Original factual caption prompt.',
             'query_images': [self.images[0]], 'gold_choice': 'GOLD_NEVER_SENT'},
            {'record_id': 'same_pixels', 'user_text': 'Original factual caption prompt.',
             'query_images': [self.images[0]]},
            {'record_id': 'second', 'user_text': 'Original factual caption prompt.',
             'query_images': [self.images[1]]},
        ]

    @staticmethod
    def answer(config, body, timeout):
        return {'raw_response': 'A newly generated factual caption.', 'returned_model': config.model,
                'requested_model': config.model, 'response_id': 'new-response-' + runner.digest(body)[:8],
                'request_id': 'new-request', 'finish_reason': 'stop', 'refusal': None,
                'usage': {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18},
                'reasoning_tokens': None, 'reasoning_content_present': False, 'enable_thinking': False, 'system_fingerprint': 'test', 'elapsed_seconds': 0.1,
                'request_sha256': runner.digest(body), 'attempts': 1, 'retry_count': 0}

    def prepare_stage(self, stage='caption_generation', records=None):
        return runner.stage_requests(self.root, stage, self.records if records is None else records, [self.source])

    def execute(self, stage='caption_generation', limit=0):
        with redirect_stdout(io.StringIO()):
            return runner.execute_stage(self.root, stage, limit)

    def test_caption_stage_has_exact_user_only_prompt_and_no_gold(self):
        meta = self.prepare_stage()
        self.assertEqual((meta['logical_records'], meta['unique_requests'], meta['max_output_tokens']), (3, 2, 1024))
        rows = runner.lines(self.root / 'caption_generation/requests.jsonl')
        for row in rows:
            sanitized = runner.sanitized_payload(row)
            self.assertEqual(set(sanitized), {'user_text', 'query_images'})
            body = build_request(sanitized, model=DEEPSEEK_MODEL, max_output_tokens=1024, image_detail='high')
            self.assertEqual(len(body['messages']), 1)
            self.assertEqual(body['messages'][0]['role'], 'user')
            self.assertEqual(body['messages'][0]['content'][-1]['text'], 'Original factual caption prompt.')
            self.assertEqual(runner.digest(body), row['request_sha256'])
            self.assertNotIn('GOLD_NEVER_SENT', json.dumps(body))

    def test_partial_then_full_stage_resume_and_same_run_dedup(self):
        self.prepare_stage()
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer) as call:
            self.assertFalse(self.execute(limit=1))
            self.assertEqual(call.call_count, 1)
            self.assertTrue(self.execute())
            self.assertEqual(call.call_count, 2)
            self.assertTrue(self.execute())
            self.assertEqual(call.call_count, 2)
        fresh = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')
        self.assertEqual(len(fresh), 2)
        self.assertTrue(all(r['run_id'] == self.protocol['run_id'] and not r['historical_response_reuse'] for r in fresh))

    def test_resume_rejects_response_from_another_run(self):
        self.prepare_stage()
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer) as call:
            self.execute(limit=1)
            path = self.root / 'caption_generation/fresh_responses.jsonl'
            completed = runner.lines(path)
            completed[0]['run_id'] = 'another-run'
            runner.write_lines(path, completed)
            with self.assertRaisesRegex(ValueError, 'Saved response provenance differs'):
                self.execute()
            self.assertEqual(call.call_count, 1)

    def test_changed_stage_source_rejected_before_any_call(self):
        self.prepare_stage()
        self.source.write_text('changed source')
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request') as call:
            with self.assertRaisesRegex(ValueError, 'Stage input changed'):
                self.execute()
            call.assert_not_called()

    def test_truncated_caption_is_saved_for_diagnosis_and_blocks_resume(self):
        self.prepare_stage()

        def truncated(config, body, timeout):
            return {**self.answer(config, body, timeout), 'finish_reason': 'length'}

        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=truncated) as call:
            with self.assertRaisesRegex(RuntimeError, 'Generation failed validation'):
                self.execute()
            call.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'earlier response failed validation'):
                self.execute()
            call.assert_called_once()
        saved = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')
        self.assertEqual(saved[0]['status'], 'invalid_response')
        self.assertEqual(saved[0]['finish_reason'], 'length')

    def test_transient_retry_preserves_wire_and_records_retry_count(self):
        self.prepare_stage(records=self.records[:1])
        seen = []

        def retry_once(config, body, timeout):
            seen.append(runner.digest(body))
            if len(seen) == 1:
                raise APIError('rate_limit', http_status=429, retryable=True, message='Rate limited')
            return self.answer(config, body, timeout)

        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=retry_once), \
             patch.object(runner.time, 'sleep') as sleep:
            self.assertTrue(self.execute())
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        sleep.assert_called_once_with(2)
        saved = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')[0]
        self.assertEqual((saved['attempts'], saved['retry_count']), (2, 1))

    def test_reader_output_must_be_choice_but_caption_accepts_plain_text(self):
        self.prepare_stage('reader', records=[{**self.records[0], 'system': 'Return one letter.'}])
        meta = runner.read(self.root / 'reader/protocol.json')
        self.assertEqual(meta['max_output_tokens'], 16)
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer):
            with self.assertRaisesRegex(RuntimeError, 'Generation failed validation'):
                self.execute('reader')
        self.assertEqual(runner.lines(self.root / 'reader/fresh_responses.jsonl')[0]['status'], 'invalid_response')

    def complete_caption_fixture(self):
        records = [{'record_id': image['sha256'], 'user_text': 'Original factual caption prompt.',
                    'query_images': [image]} for image in self.images]
        self.prepare_stage(records=records)
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer):
            self.assertTrue(self.execute())
        requests = runner.lines(self.root / 'caption_generation/requests.jsonl')
        fresh = runner.indexed(runner.lines(self.root / 'caption_generation/fresh_responses.jsonl'), 'request_sha256')
        captions = [{'id': r['record_id'], 'caption': fresh[r['request_sha256']]['raw_response'].strip(),
                     'model': DEEPSEEK_MODEL, 'source_response_record_id': fresh[r['request_sha256']]['record_id'],
                     'request_sha256': r['request_sha256'],
                     'response_id': fresh[r['request_sha256']]['response_id']} for r in requests]
        runner.write_lines(self.root / 'captions.jsonl', captions)
        return runner.indexed(captions, 'id')

    def prepare_evidence_fixture(self):
        captions = self.complete_caption_fixture()
        inputs = self.root / 'inputs'
        inputs.mkdir()
        memories = [{'id': f'Test/M{i}', 'task': 'Test', 'dialogue': f'User: Original dialogue {i}.\nAssistant: Original reply.',
                     'image_sha256': self.images[i % 2]['sha256']} for i in range(5)]
        question = {'id': 'Test/Q1', 'task': 'Test', 'question': 'Which scene was shown?',
                    'options': dict(zip('ABCD', ['first', 'second', 'third', 'fourth'])),
                    'query_images': [self.images[0]],
                    'candidate_statements': {'A': 'RETRIEVAL_ONLY_HYPOTHESIS'}}
        runner.write_lines(inputs / 'source_memories.jsonl', memories)
        runner.write_lines(inputs / 'source_questions.jsonl', [question])
        protocol_path = self.root / 'data/independent_visual_inputs/protocol.json'
        runner.dump(protocol_path, {'reader_system_prompt': 'Original reader system.'})
        conditions = []
        for name, selected in zip(runner.CONDITIONS, (memories, list(reversed(memories)))):
            evidence = [{'memory_id': m['id'], 'reader_text': m['dialogue'] + '\nMemory description: '
                         + captions[m['image_sha256']]['caption']} for m in selected]
            context = '\n\n'.join(f'Memory {i}: {m["reader_text"]}' for i, m in enumerate(evidence, 1))
            conditions.append({'name': name, 'top5_memory_ids': [m['id'] for m in selected],
                               'reader_memories': evidence, 'reader_context': context})
        rows = [{'question_id': question['id'], 'question': question['question'], 'options': question['options'],
                 'query_image_paths': [self.images[0]['path']], 'conditions': conditions}]
        runner.write_lines(self.root / 'retrieval_results.jsonl', rows)
        retrieval_protocol = {'source_run_id': self.protocol['run_id'],
                              'source_experiment_signature': self.protocol['experiment_signature'],
                              'source_sha256': {str(self.root / 'captions.jsonl'): runner.sha(self.root / 'captions.jsonl')},
                              'output_sha256': {str(self.root / 'retrieval_results.jsonl'): runner.sha(self.root / 'retrieval_results.jsonl')}}
        runner.dump(self.root / 'retrieval_protocol.json', retrieval_protocol)
        return rows

    def test_caption_artifact_is_bound_to_this_run_fresh_response(self):
        self.complete_caption_fixture()
        self.assertEqual(len(runner.verify_captions(self.root)), 2)
        captions = runner.lines(self.root / 'captions.jsonl')
        captions[0]['caption'] = 'STALE_QWEN_CAPTION'
        runner.write_lines(self.root / 'captions.jsonl', captions)
        with self.assertRaisesRegex(ValueError, 'not this run fresh API result'):
            runner.verify_captions(self.root)

    def test_caption_file_records_requested_model_and_actual_response_identity(self):
        source = self.root / 'inputs/api_requests_caption.jsonl'
        source.parent.mkdir()
        runner.write_lines(source, [
            {'id': image['sha256'], 'user_text': 'Original factual caption prompt.',
             'query_images': [image]} for image in self.images])
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer), \
             redirect_stdout(io.StringIO()):
            self.assertTrue(runner.generate_captions(self.root))
        captions = runner.verify_captions(self.root)
        self.assertEqual(len(captions), 2)
        for row in captions.values():
            self.assertEqual(row['model'], DEEPSEEK_MODEL)
            self.assertTrue(row['response_id'].startswith('new-response-'))
            self.assertEqual(row['source_response_record_id'], row['id'])
        changed = list(captions.values())
        changed[0]['response_id'] = 'another-api-response'
        runner.write_lines(self.root / 'captions.jsonl', changed)
        with self.assertRaisesRegex(ValueError, 'not this run fresh API result'):
            runner.verify_captions(self.root)

    def test_resume_rejects_saved_reasoning_before_sending_any_more_requests(self):
        self.prepare_stage()
        with patch('scripts.evaluation.deepseek_flash_alias_transport.chat_request', side_effect=self.answer) as call:
            self.assertFalse(self.execute(limit=1))
            path = self.root / 'caption_generation/fresh_responses.jsonl'
            responses = runner.lines(path)
            responses[0]['reasoning_content_present'] = True
            runner.write_lines(path, responses)
            with self.assertRaisesRegex(ValueError, 'Saved response failed validation'):
                self.execute()
            self.assertEqual(call.call_count, 1)

    def test_read_only_stage_verifier_rejects_cross_run_answers(self):
        self.complete_caption_fixture()
        runner.verify_stage(self.root, 'caption_generation', require_complete=True)
        path = self.root / 'caption_generation/fresh_responses.jsonl'
        completed = runner.lines(path)
        completed[0]['run_id'] = 'a-historical-run'
        runner.write_lines(path, completed)
        with self.assertRaisesRegex(ValueError, 'Saved response provenance differs'):
            runner.verify_stage(self.root, 'caption_generation', require_complete=True)

    def test_reader_uses_new_caption_evidence_and_omits_retrieval_hypotheses(self):
        self.prepare_evidence_fixture()
        with patch.object(runner, 'ROOT', self.root):
            runner.prepare_reader(self.root)
        rows = runner.lines(self.root / 'reader/requests.jsonl')
        self.assertEqual(len(rows), 2)
        for row in rows:
            text = row['user_text']
            self.assertIn('A newly generated factual caption.', text)
            self.assertIn('User: Original dialogue', text)
            self.assertIn('Options:\nA. first\nB. second\nC. third\nD. fourth', text)
            self.assertNotIn('RETRIEVAL_ONLY_HYPOTHESIS', text)
            self.assertNotIn('gold', runner.sanitized_payload(row))
            self.assertEqual(row['query_images'], [self.images[0]])
            self.assertEqual(row['system'], 'Original reader system.')
        self.assertNotEqual(rows[0]['request_sha256'], rows[1]['request_sha256'])

    def test_self_consistent_stale_retrieval_evidence_is_rejected(self):
        rows = self.prepare_evidence_fixture()
        condition = rows[0]['conditions'][0]
        condition['reader_memories'][0]['reader_text'] = 'Original dialogue\nMemory description: STALE_QWEN_CAPTION'
        condition['reader_context'] = '\n\n'.join(f'Memory {i}: {m["reader_text"]}'
                                                      for i, m in enumerate(condition['reader_memories'], 1))
        path = self.root / 'retrieval_results.jsonl'
        runner.write_lines(path, rows)
        protocol_path = self.root / 'retrieval_protocol.json'
        protocol = runner.read(protocol_path)
        protocol['output_sha256'][str(path)] = runner.sha(path)
        runner.dump(protocol_path, protocol)
        with patch.object(runner, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'this backbone fresh captions'):
                runner.prepare_reader(self.root)
        self.assertFalse((self.root / 'reader/requests.jsonl').exists())

    def test_retrieval_from_another_run_is_rejected(self):
        self.prepare_evidence_fixture()
        path = self.root / 'retrieval_protocol.json'
        protocol = runner.read(path)
        protocol['source_run_id'] = 'prior-Qwen-run'
        runner.dump(path, protocol)
        with self.assertRaisesRegex(ValueError, 'Retrieval belongs to another experiment'):
            runner.verify_retrieval(self.root)


class DeepSeekFullBackboneRecallTests(unittest.TestCase):
    def setUp(self):
        self.evaluation = {'Q1': {'gold_ids': ['M1', 'M2']}, 'Q2': {'gold_ids': ['M3']}}
        self.rows = [
            {'question_id': 'Q1', 'conditions': [
                {'name': runner.CONDITIONS[0], 'top5_memory_ids': ['M1', 'X1', 'X2', 'X3', 'X4']},
                {'name': runner.CONDITIONS[1], 'top5_memory_ids': ['M1', 'M2', 'X2', 'X3', 'X4']}]},
            {'question_id': 'Q2', 'conditions': [
                {'name': runner.CONDITIONS[0], 'top5_memory_ids': ['M3', 'X1', 'X2', 'X3', 'X4']},
                {'name': runner.CONDITIONS[1], 'top5_memory_ids': ['X0', 'X1', 'X2', 'X3', 'X4']}]},
        ]

    def test_micro_and_macro_recall_use_distinct_denominators(self):
        summary = runner.retrieval_metrics(self.rows, self.evaluation)
        self.assertEqual(summary['gold_references'], 3)
        b, m = [summary[name] for name in runner.CONDITIONS]
        self.assertAlmostEqual(b['gold_recall'], 2 / 3)
        self.assertAlmostEqual(m['gold_recall'], 2 / 3)
        self.assertEqual(b['macro_recall_at_5'], .75)
        self.assertEqual(m['macro_recall_at_5'], .5)
        self.assertEqual((b['questions_with_gold_hit'], m['questions_with_gold_hit']), (2, 1))

    def test_empty_gold_and_duplicate_top5_are_rejected(self):
        empty = copy.deepcopy(self.evaluation)
        empty['Q1']['gold_ids'] = []
        with self.assertRaisesRegex(ValueError, 'nonempty gold'):
            runner.retrieval_metrics(self.rows, empty)
        duplicate = copy.deepcopy(self.rows)
        duplicate[0]['conditions'][0]['top5_memory_ids'][-1] = 'M1'
        with self.assertRaisesRegex(ValueError, 'five distinct'):
            runner.retrieval_metrics(duplicate, self.evaluation)


class DeepSeekAliasContractTests(unittest.TestCase):
    def test_signed_contract_keeps_identity_unknown_and_budget_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {'model': DEEPSEEK_MODEL, 'model_identity': dict(runner.MODEL_IDENTITY),
                        'caption_budget_policy': 'fixed_1024_from_start',
                        'caption_max_output_tokens': 1024, 'reader_max_output_tokens': 16,
                        'source_sha256': {}}

            def save(value):
                runner.dump(root / 'protocol.json', {'contract': value,
                                                     'experiment_signature': runner.digest(value)})

            save(contract)
            with patch('scripts.data.memeye_backbone_inputs.prepare_inputs') as prepare:
                runner.verify(root)
                prepare.assert_called_once_with(root / 'inputs', DEEPSEEK_MODEL)
                for changes in ({'caption_max_output_tokens': 256},
                                {'caption_budget_amendment': {}},
                                {'model': 'deepseek-v4.1-flash-1017'},
                                {'model_identity': {**runner.MODEL_IDENTITY,
                                                    'distinct_weights_vs_1017': True}}):
                    with self.subTest(changes=changes):
                        save({**contract, **changes})
                        with self.assertRaisesRegex(ValueError, 'Alias identity or fixed'):
                            runner.verify(root)

    def test_cli_defaults_to_gpu_and_unversioned_model(self):
        import argparse
        parser = argparse.ArgumentParser()
        runner.add_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.api_model, 'deepseek-v4.1-flash')
        self.assertEqual(args.device, 'cuda:0')


if __name__ == '__main__':
    unittest.main()
