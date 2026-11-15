"""Local VL2 provenance, context limits and resume behavior; no GPU or service."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.evaluation import memeye_deepseek_vl2 as runner
from scripts.evaluation.openai_chat_transport import APIError


class LocalVL2StagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.profile = {'model': runner.MODEL, 'base_url': runner.BASE_URL,
                        'model_revision': runner.MODEL_REVISION}
        self.config = SimpleNamespace(model=runner.MODEL, base_url=runner.BASE_URL,
                                      public_dict=lambda: copy.deepcopy(self.profile))
        self.protocol = {'run_id': 'local-only-run', 'experiment_signature': 'test-signature',
                         'contract': {'model': runner.MODEL, 'local_config': self.profile,
                                      'caption_max_output_tokens': 1024, 'reader_max_output_tokens': 16,
                                      'context_limit': 4096, 'image_detail': 'high', 'workers': 1,
                                      'max_attempts': 2, 'timeout_seconds': 600}}
        self.stack.enter_context(patch.object(runner, 'verify', return_value=self.protocol))
        transport = ModuleType('scripts.evaluation.deepseek_vl2_local_transport')
        transport.APIError = APIError
        transport.canonical_model = lambda model: model
        transport.load_config = MagicMock(return_value=self.config)
        transport.build_request = self.build_request
        transport.chat_request = MagicMock(side_effect=self.answer)
        self.transport = transport
        self.stack.enter_context(patch.dict(sys.modules, {transport.__name__: transport}))
        self.source = self.root / 'frozen_source.txt'
        self.source.write_text('Fresh local run source')
        self.images = []
        for index in range(2):
            path = self.root / f'original{index}.png'
            path.write_bytes(b'\x89PNG\r\n\x1a\n' + bytes([index]))
            self.images.append({'path': str(path), 'sha256': runner.sha(path)})
        self.records = [
            {'record_id': 'first', 'user_text': 'Fixed caption instruction.',
             'query_images': self.images[:1], 'gold_choice': 'NEVER_PASS_GOLD'},
            {'record_id': 'duplicate', 'user_text': 'Fixed caption instruction.',
             'query_images': self.images[:1]},
            {'record_id': 'second', 'user_text': 'Fixed caption instruction.',
             'query_images': self.images[1:]},
        ]

    @staticmethod
    def build_request(payload, *, model, max_output_tokens, image_detail):
        if set(payload) - {'user_text', 'query_images', 'system'}:
            raise ValueError('Unexpected local payload field')
        return {'model': model, 'payload': payload, 'max_tokens': max_output_tokens,
                'image_detail': image_detail, 'temperature': 0, 'seed': 42}

    @staticmethod
    def answer(config, body, timeout):
        return {'returned_model': config.model, 'requested_model': config.model,
                'raw_response': 'A factual image description.',
                'response_id': 'local-' + runner.digest(body)[:10],
                'finish_reason': 'stop', 'refusal': None, 'elapsed_seconds': 0.1,
                'checkpoint_revision': runner.MODEL_REVISION, 'context_limit': 4096,
                'input_tokens': 120, 'input_truncated': False,
                'prompt_token_ids': [10] * 120, 'generated_token_ids': [20] * 10,
                'usage': {'prompt_tokens': 120, 'completion_tokens': 10, 'total_tokens': 130}}

    def prepare_stage(self, stage='caption_generation', records=None):
        return runner.stage_requests(self.root, stage, self.records if records is None else records,
                                     [self.source])

    def execute(self, stage='caption_generation', limit=0):
        with redirect_stdout(io.StringIO()):
            return runner.execute_stage(self.root, stage, limit)

    def test_allowlist_budget_and_same_run_deduplication(self):
        meta = self.prepare_stage()
        self.assertEqual((meta['logical_records'], meta['unique_requests'], meta['max_output_tokens']),
                         (3, 2, 1024))
        self.assertFalse(self.execute(limit=1))
        self.assertTrue(self.execute())
        self.assertTrue(self.execute())
        self.assertEqual(self.transport.chat_request.call_count, 2)
        self.transport.load_config.assert_called_with(None, model=runner.MODEL)
        for call in self.transport.chat_request.call_args_list:
            body = call.args[1]
            self.assertNotIn('gold_choice', body['payload'])
            self.assertEqual((body['seed'], body['temperature']), (42, 0))
        responses = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(row['historical_response_reuse'] is False for row in responses))
        self.assertTrue(all(row['context_budget_audit']['input_tokens'] == 120 for row in responses))

    def test_other_run_response_is_rejected_before_resume(self):
        self.prepare_stage()
        self.execute(limit=1)
        path = self.root / 'caption_generation/fresh_responses.jsonl'
        rows = runner.lines(path)
        rows[0]['run_id'] = 'older-run'
        runner.write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.execute()
        self.assertEqual(self.transport.chat_request.call_count, 1)

    def test_context_accounting_tampering_blocks_resume(self):
        self.prepare_stage()
        self.execute(limit=1)
        path = self.root / 'caption_generation/fresh_responses.jsonl'
        rows = runner.lines(path)
        rows[0]['context_budget_audit']['input_tokens'] = 1
        runner.write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'context budget audit differs'):
            self.execute()
        self.assertEqual(self.transport.chat_request.call_count, 1)

    def test_stage_source_change_blocks_new_generation(self):
        self.prepare_stage()
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Stage input changed'):
            self.execute()
        self.transport.chat_request.assert_not_called()

    def test_overflow_is_recorded_and_blocks_resume(self):
        self.prepare_stage()

        def overflow(*args, **kwargs):
            response = self.answer(*args, **kwargs)
            response['usage']['prompt_tokens'] = 3073
            response['input_tokens'] = 3073
            response['prompt_token_ids'] = [10] * 3073
            return response

        self.transport.chat_request.side_effect = overflow
        with self.assertRaisesRegex(RuntimeError, 'failed validation'):
            self.execute()
        rows = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')
        self.assertEqual(rows[0]['status'], 'invalid_response')
        self.assertIn('Context budget exceeded', rows[0]['validation_error'])
        with self.assertRaisesRegex(ValueError, 'earlier response failed'):
            self.execute()
        self.assertEqual(self.transport.chat_request.call_count, 1)

    def test_length_stop_remains_diagnostic_failure(self):
        self.prepare_stage()
        self.transport.chat_request.side_effect = lambda *a, **k: {**self.answer(*a, **k), 'finish_reason': 'length'}
        with self.assertRaises(RuntimeError):
            self.execute()
        saved = runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')
        self.assertEqual(saved[0]['finish_reason'], 'length')
        self.assertEqual(saved[0]['status'], 'invalid_response')

    def test_reader_has_16_token_budget_and_requires_choice(self):
        self.prepare_stage('reader', [dict(self.records[0], system='Original reader instruction.')])
        with self.assertRaises(RuntimeError):
            self.execute('reader')
        body = self.transport.chat_request.call_args.args[1]
        self.assertEqual(body['max_tokens'], 16)
        self.assertEqual(body['payload']['system'], 'Original reader instruction.')
        saved = runner.lines(self.root / 'reader/fresh_responses.jsonl')[0]
        self.assertIn('A/B/C/D', saved['validation_error'])

    def test_transient_service_retry_preserves_frozen_input(self):
        self.prepare_stage(records=self.records[:1])
        calls = []

        def retry(config, body, timeout):
            calls.append(runner.digest(body))
            if len(calls) == 1:
                raise APIError('network', retryable=True, message='Transient test failure')
            return self.answer(config, body, timeout)

        self.transport.chat_request.side_effect = retry
        with patch.object(runner.time, 'sleep'):
            self.assertTrue(self.execute())
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(runner.lines(self.root / 'caption_generation/fresh_responses.jsonl')[0]['retry_count'], 1)

    def test_fresh_caption_artifact_retains_response_provenance(self):
        source = self.root / 'inputs/api_requests_caption.jsonl'
        source.parent.mkdir()
        runner.write_lines(source, [{'id': image['sha256'], 'user_text': 'Fixed instruction.',
                                     'query_images': [image]} for image in self.images])
        with redirect_stdout(io.StringIO()):
            self.assertTrue(runner.generate_captions(self.root))
        captions = runner.verify_captions(self.root)
        self.assertEqual(set(captions), {image['sha256'] for image in self.images})
        rows = list(captions.values())
        rows[0]['caption'] = 'historical replacement'
        runner.write_lines(self.root / 'captions.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'not this run fresh'):
            runner.verify_captions(self.root)

    def test_reader_keeps_both_top5_orders_full_text_and_original_query_image(self):
        inputs = self.root / 'inputs'
        inputs.mkdir()
        image = self.images[0]
        captions = {image['sha256']: {'id': image['sha256'], 'caption': 'Fresh complete caption.'}}
        memories = [{'id': f'Task/M{i}', 'task': 'Task', 'dialogue': f'Original dialogue {i}.',
                     'image_sha256': image['sha256']} for i in range(5)]
        question = {'id': 'Task/Q1', 'task': 'Task', 'question': 'Exact original question?',
                    'options': dict(zip('ABCD', ['one', 'two', 'three', 'four'])),
                    'query_images': [image], 'candidate_statements': {'A': 'RETRIEVAL_ONLY_CANDIDATE'}}
        runner.write_lines(inputs / 'source_memories.jsonl', memories)
        runner.write_lines(inputs / 'source_questions.jsonl', [question])
        runner.write_lines(self.root / 'captions.jsonl', list(captions.values()))
        source_protocol = self.root / 'data/independent_visual_inputs/protocol.json'
        runner.dump(source_protocol, {'reader_system_prompt': 'Original system instruction.'})
        conditions = []
        for name, selected in zip(runner.CONDITIONS, [memories, list(reversed(memories))]):
            evidence = [{'memory_id': memory['id'],
                         'reader_text': memory['dialogue'] + '\nMemory description: Fresh complete caption.'}
                        for memory in selected]
            conditions.append({'name': name, 'top5_memory_ids': [m['id'] for m in selected],
                               'reader_memories': evidence,
                               'reader_context': '\n\n'.join(f'Memory {i}: {m["reader_text"]}'
                                                             for i, m in enumerate(evidence, 1))})
        runner.write_lines(self.root / 'retrieval_results.jsonl', [{
            'question_id': question['id'], 'question': question['question'], 'options': question['options'],
            'query_image_paths': [image['path']], 'conditions': conditions}])
        runner.dump(self.root / 'retrieval_protocol.json', {})
        with patch.object(runner, 'ROOT', self.root), \
                patch.object(runner, 'verify_captions', return_value=captions), \
                patch.object(runner, 'verify_retrieval'):
            runner.prepare_reader(self.root)
        rows = runner.lines(self.root / 'reader/requests.jsonl')
        self.assertEqual(len(rows), 2)
        for row, condition in zip(rows, conditions):
            expected = ('Retrieved dialogue and image descriptions (ranked by relevance):\n\n'
                        + condition['reader_context'] + '\n\nQuestion:\n' + question['question']
                        + '\n\nOptions:\nA. one\nB. two\nC. three\nD. four\n\nAnswer with one letter only.')
            self.assertEqual(row['user_text'], expected)
            self.assertEqual(row['system'], 'Original system instruction.')
            self.assertEqual(row['query_images'], [image])
            self.assertNotIn('RETRIEVAL_ONLY_CANDIDATE', row['user_text'])
        manifest = runner.lines(self.root / 'reader_manifest.jsonl')
        self.assertEqual(manifest[0]['top_memory_ids'], [m['id'] for m in memories])
        self.assertEqual(manifest[1]['top_memory_ids'], [m['id'] for m in reversed(memories)])


class LocalVL2BudgetTests(unittest.TestCase):
    def test_exact_context_boundary_and_overflow(self):
        self.assertEqual(runner.validate_context_budget(3072, 1024)['context_limit'], 4096)
        self.assertEqual(runner.validate_context_budget(4080, 16)['input_tokens'], 4080)
        for args in [(3073, 1024), (4081, 16), (4096, 16)]:
            with self.assertRaisesRegex(ValueError, 'Context budget exceeded'):
                runner.validate_context_budget(*args)

    def test_context_count_cannot_be_missing_or_boolean(self):
        for value in [None, True, 0, -1, 1.5, '100']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_context_budget(value, 16)
        with self.assertRaises(ValueError):
            runner.validate_context_budget(100, 16, 100000000)

    def test_response_requires_untruncated_complete_accounting(self):
        response = {'returned_model': runner.MODEL, 'finish_reason': 'stop',
                    'raw_response': 'A', 'response_id': 'local-test',
                    'checkpoint_revision': runner.MODEL_REVISION, 'context_limit': 4096,
                    'input_tokens': 100, 'input_truncated': False,
                    'prompt_token_ids': [10] * 100, 'generated_token_ids': [20],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 1}}
        contract = {'model': runner.MODEL, 'reader_max_output_tokens': 16,
                    'caption_max_output_tokens': 1024, 'context_limit': 4096}
        runner.validate_response(response, 'reader', contract)
        for change in [{'usage': None}, {'input_truncated': True}, {'input_tokens': 99},
                       {'checkpoint_revision': 'different'}, {'context_limit': 8192},
                       {'prompt_token_ids': []}, {'generated_token_ids': [True]},
                       {'usage': {'prompt_tokens': 100, 'completion_tokens': 17}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_response({**response, **change}, 'reader', contract)

    def test_caption_token_audit_never_truncates_and_separates_graph_inputs(self):
        tokenizer = MagicMock()
        tokenizer.encode_batch.side_effect = lambda texts, **kw: [SimpleNamespace(ids=list(range(int(t)))) for t in texts]
        captions = [{'id': 'history', 'caption': '512'}, {'id': 'query-only', 'caption': '513'}]
        audit = runner.caption_token_audit(captions, {'history'}, tokenizer)
        self.assertEqual(audit['status'], 'complete')
        self.assertEqual(audit['max_tokens'], 513)
        self.assertEqual(len(audit['overlong_captions']), 1)
        self.assertEqual(audit['overlong_history_captions'], [])
        tokenizer.no_truncation.assert_called_once()
        tokenizer.no_padding.assert_called_once()
        tokenizer.encode_batch.assert_called_once_with(['512', '513'], add_special_tokens=True)
        captions[0]['caption'] = '513'
        audit = runner.caption_token_audit(captions, {'history'}, tokenizer)
        self.assertEqual(audit['status'], 'blocked')
        self.assertEqual(audit['overlong_history_captions'][0]['image_sha256'], 'history')

    def test_missing_history_caption_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing history'):
            runner.caption_token_audit([{'id': 'other', 'caption': '10'}], {'history'}, MagicMock())


class LocalVL2ConfigurationTests(unittest.TestCase):
    def test_verify_rechecks_the_frozen_runtime_manifest_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            profile = {'model': runner.MODEL, 'base_url': runner.BASE_URL,
                       'runtime_manifest_sha256': 'frozen-runtime'}
            contract = {'model_identity': runner.MODEL_IDENTITY, 'model': runner.MODEL,
                        'caption_budget_policy': 'fixed_1024_from_start',
                        'caption_max_output_tokens': 1024, 'reader_max_output_tokens': 16,
                        'context_limit': 4096, 'input_truncation': False, 'workers': 1,
                        'seed': 42, 'temperature': 0, 'do_sample': False,
                        'endpoint': runner.BASE_URL, 'source_sha256': {},
                        'local_config': copy.deepcopy(profile)}
            protocol = {'run_id': 'configuration-test', 'contract': contract,
                        'experiment_signature': runner.digest(contract)}
            runner.dump(output / 'protocol.json', protocol)
            transport = ModuleType('scripts.evaluation.deepseek_vl2_local_transport')
            transport.load_config = MagicMock(return_value=SimpleNamespace(
                public_dict=lambda: copy.deepcopy(profile)))
            with patch.dict(sys.modules, {transport.__name__: transport}), \
                    patch('scripts.data.memeye_backbone_inputs.prepare_inputs') as inputs:
                self.assertEqual(runner.verify(output), protocol)
                transport.load_config.assert_called_with(None, model=runner.MODEL)
                inputs.assert_called_once_with(output / 'inputs', runner.MODEL)
                inputs.reset_mock()
                profile['runtime_manifest_sha256'] = 'different-runtime'
                with self.assertRaisesRegex(ValueError, 'Frozen local runtime configuration changed'):
                    runner.verify(output)
                inputs.assert_not_called()


class LocalVL2PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = {name: self.stack.enter_context(patch.object(runner, name)) for name in (
            'prepare', 'verify', 'generate_captions', 'audit_caption_lengths', 'verify_retrieval',
            'freeze', 'analyze', 'prepare_reader', 'execute_stage')}
        self.calls['verify'].return_value = {'contract': {'model': runner.MODEL}}
        self.calls['generate_captions'].return_value = True
        self.calls['execute_stage'].return_value = True
        self.calls['analyze'].return_value = {'status': 'synthetic'}
        strict = ModuleType('scripts.memory.memeye_backbone_retrieval')
        extension = ModuleType('scripts.memory.memeye_backbone_long_text')
        transport = ModuleType('scripts.evaluation.deepseek_vl2_local_transport')
        transport.canonical_model = lambda model: model
        self.strict = strict.rebuild_and_retrieve = MagicMock()
        self.extension = extension.complete_retrieval = MagicMock()
        self.stack.enter_context(patch.dict(sys.modules, {
            strict.__name__: strict, extension.__name__: extension, transport.__name__: transport}))
        self.order = MagicMock()
        for name, call in {**self.calls, 'strict': self.strict, 'extension': self.extension}.items():
            self.order.attach_mock(call, name)
        self.args = SimpleNamespace(api_workers=1, api_limit=0, api_model=runner.MODEL,
                                    output=self.output, stage='all', device='cuda:1')

    def run_pipeline(self):
        with redirect_stdout(io.StringIO()):
            runner.pipeline(self.args)

    def test_all_runs_fresh_captions_both_rankings_reader_and_analysis_in_order(self):
        self.run_pipeline()
        self.assertEqual([call[0] for call in self.order.mock_calls], [
            'prepare', 'generate_captions', 'audit_caption_lengths', 'strict', 'analyze',
            'prepare_reader', 'execute_stage', 'analyze'])
        self.calls['prepare'].assert_called_once_with(self.output, runner.MODEL, None, 1)
        self.strict.assert_called_once_with(self.output, device='cuda:1')
        self.extension.assert_not_called()
        self.calls['execute_stage'].assert_called_once_with(self.output, 'reader', 0)
        self.assertEqual(self.calls['analyze'].call_args_list[0].kwargs, {'require_reader': False})
        self.assertEqual(self.calls['analyze'].call_args_list[1].kwargs, {})

    def test_bge_retrieval_overflow_uses_only_the_audited_continuation(self):
        self.strict.side_effect = ValueError('leaf retrieval text exceeds BGE token limit 512; refusing silent truncation')
        self.run_pipeline()
        self.extension.assert_called_once_with(self.output, device='cuda:1')
        path, transition = self.calls['freeze'].call_args.args
        self.assertEqual(path, self.output / 'long_text_transition.json')
        self.assertFalse(transition['caption_generation_changed'])
        self.assertFalse(transition['memory_construction_changed'])
        self.assertFalse(transition['reader_evidence_changed'])
        self.calls['execute_stage'].assert_called_once_with(self.output, 'reader', 0)

    def test_incomplete_caption_stage_stops_before_retrieval_and_reader(self):
        self.args.api_limit = 1
        self.calls['generate_captions'].return_value = False
        self.run_pipeline()
        self.calls['generate_captions'].assert_called_once_with(self.output, 1)
        self.strict.assert_not_called()
        self.calls['audit_caption_lengths'].assert_not_called()
        self.calls['prepare_reader'].assert_not_called()

    def test_overlong_history_caption_stops_before_any_graph_build(self):
        self.calls['audit_caption_lengths'].side_effect = ValueError('Fresh history captions exceed BGE 512 tokens')
        with self.assertRaisesRegex(ValueError, 'history captions exceed'):
            self.run_pipeline()
        self.strict.assert_not_called()
        self.extension.assert_not_called()
        self.calls['prepare_reader'].assert_not_called()

    def test_completed_retrieval_is_verified_without_rebuilding(self):
        (self.output / 'retrieval_protocol.json').write_text('{}\n')
        self.run_pipeline()
        self.calls['verify_retrieval'].assert_called_once_with(self.output)
        self.strict.assert_not_called()
        self.extension.assert_not_called()
        self.calls['execute_stage'].assert_called_once_with(self.output, 'reader', 0)

    def test_unrelated_retrieval_failure_is_not_treated_as_length_overflow(self):
        self.strict.side_effect = ValueError('Frozen source image changed')
        with self.assertRaisesRegex(ValueError, 'source image changed'):
            self.run_pipeline()
        self.extension.assert_not_called()
        self.calls['freeze'].assert_not_called()
        self.calls['prepare_reader'].assert_not_called()

    def test_incomplete_reader_stage_stops_before_final_analysis(self):
        self.calls['execute_stage'].return_value = False
        self.run_pipeline()
        self.calls['analyze'].assert_called_once_with(self.output, require_reader=False)


if __name__ == '__main__':
    unittest.main()
