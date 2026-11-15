"""Amendment freeze/restart and immutable failed-run provenance, without API."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation import memeye_deepseek_complete_captions as restart


class DeepSeekCaptionRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failed = self.root / 'failed'
        self.output = self.root / 'new'
        self.env = self.root / '.env'
        self.env.write_text('unused')
        self.run = restart.run
        self.files = ('source_memories.jsonl', 'source_questions.jsonl', 'evaluation.jsonl',
                      'images.jsonl', 'api_requests_caption.jsonl')
        (self.failed / 'inputs').mkdir(parents=True)
        for name in self.files:
            (self.failed / 'inputs' / name).write_text('{"same_prompt":"80-120 words"}\n')
        contract = {'model': restart.MODEL, 'caption_max_output_tokens': 256,
                    'reader_max_output_tokens': 16, 'source_sha256': {},
                    'workers': 3, 'api_env_file': str(self.env.resolve())}
        self.original = {'run_id': 'failed-run', 'contract': contract,
                         'experiment_signature': self.run.digest(contract)}
        self.run.dump(self.failed / 'protocol.json', self.original)
        stage = self.failed / 'caption_generation'
        stage.mkdir()
        requests = [{'record_id': 'image', 'request_sha256': 'wire256', 'payload_sha256': 'same-prompt'}]
        self.run.write_lines(stage / 'requests.jsonl', requests)
        self.run.dump(stage / 'protocol.json', {
            'run_id': 'failed-run', 'experiment_signature': self.original['experiment_signature'],
            'max_output_tokens': 256, 'requests_sha256': self.run.sha(stage / 'requests.jsonl')})
        self.run.write_lines(stage / 'fresh_responses.jsonl', [{
            **requests[0], 'run_id': 'failed-run', 'experiment_signature': self.original['experiment_signature'],
            'returned_model': restart.MODEL, 'historical_response_reuse': False,
            'status': 'invalid_response', 'finish_reason': 'length',
            'usage': {'completion_tokens': 256}, 'response_id': 'failed-api-response'}])
        self.run.write_lines(stage / 'api_attempts.jsonl', [{'status': 'invalid_response'}])
        self.run.dump(stage / 'progress.json', {'completed_unique': 1})
        self.old_hashes = {str(p): self.run.sha(p) for p in self.failed.rglob('*') if p.is_file()}
        p = patch.object(self.run, 'verify', side_effect=self.verify_saved)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(self.run, 'prepare', side_effect=self.prepare_base)
        self.base_prepare = p.start()
        self.addCleanup(p.stop)
        p = patch('scripts.evaluation.deepseek_backbone_transport.chat_request')
        self.network = p.start()
        self.addCleanup(p.stop)

    def verify_saved(self, output):
        p = self.run.read(Path(output) / 'protocol.json')
        if p['experiment_signature'] != self.run.digest(p['contract']):
            raise ValueError('Experiment signature mismatch')
        for source, expected in p['contract']['source_sha256'].items():
            if self.run.sha(source) != expected:
                raise ValueError('Frozen input changed')
        return p

    def prepare_base(self, output, model, env_file, workers):
        output = Path(output)
        (output / 'inputs').mkdir(parents=True)
        for name in self.files:
            (output / 'inputs' / name).write_bytes((self.failed / 'inputs' / name).read_bytes())
        contract = dict(self.original['contract'])
        contract.update(workers=workers, api_env_file=str(Path(env_file).resolve()))
        p = {'run_id': 'temporary-preparation', 'contract': contract,
             'experiment_signature': self.run.digest(contract)}
        self.run.dump(output / 'protocol.json', p)
        return p

    def prepare(self):
        return restart.prepare(self.output, env_file=self.env, failed_output=self.failed)

    def test_fresh_contract_changes_only_budget_and_binds_failed_evidence(self):
        new = self.prepare()
        c = new['contract']
        self.assertEqual((c['caption_max_output_tokens'], c['reader_max_output_tokens']), (1024, 16))
        self.assertNotIn(new['run_id'], ('failed-run', 'temporary-preparation'))
        amendment = c['caption_budget_amendment']
        self.assertTrue(amendment['all_captions_regenerated'])
        self.assertFalse(amendment['previous_caption_response_reuse'])
        self.assertFalse(amendment['caption_prompt_changed'])
        self.assertEqual(amendment['failed_run_id'], 'failed-run')
        self.assertEqual(amendment['length_stopped_records'][0]['response_id'], 'failed-api-response')
        self.assertEqual(c['source_sha256'][str(Path(restart.__file__).resolve())], self.run.sha(restart.__file__))
        self.assertTrue(all(c['source_sha256'][p] == h for p, h in amendment['failed_artifact_sha256'].items()))
        for name in self.files:
            self.assertEqual((self.output / 'inputs' / name).read_bytes(), (self.failed / 'inputs' / name).read_bytes())
        self.assertFalse((self.output / 'caption_generation').exists())
        self.assertFalse((self.output / 'captions.jsonl').exists())
        self.assertFalse((self.output / 'reader').exists())
        self.network.assert_not_called()

    def test_reentry_verifies_without_new_run_or_contract_rewrite(self):
        first = self.prepare()
        before = (self.output / 'protocol.json').read_bytes()
        second = self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(before, (self.output / 'protocol.json').read_bytes())
        self.base_prepare.assert_called_once()
        self.network.assert_not_called()

    def test_failed_root_and_nonempty_destination_are_never_amended(self):
        with self.assertRaisesRegex(ValueError, 'distinct output'):
            restart.prepare(self.failed, env_file=self.env, failed_output=self.failed)
        self.output.mkdir()
        (self.output / 'old-response.jsonl').write_text('old generated caption')
        with self.assertRaisesRegex(ValueError, 'fresh empty'):
            self.prepare()
        self.base_prepare.assert_not_called()
        self.network.assert_not_called()

    def test_unamended_existing_contract_cannot_be_upgraded_in_place(self):
        self.prepare_base(self.output, restart.MODEL, self.env, 3)
        with self.assertRaisesRegex(ValueError, 'independently signed'):
            self.prepare()
        self.assertEqual(self.run.read(self.output / 'protocol.json')['contract']['caption_max_output_tokens'], 256)
        self.base_prepare.assert_not_called()

    def test_prior_failure_and_original_outputs_remain_unchanged(self):
        self.prepare()
        self.assertEqual(self.old_hashes, {str(p): self.run.sha(p) for p in self.failed.rglob('*') if p.is_file()})

    def test_prior_evidence_mutation_invalidates_resume(self):
        self.prepare()
        (self.failed / 'caption_generation/progress.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            self.prepare()
        self.base_prepare.assert_called_once()

    def test_unrelated_failure_does_not_authorize_budget_amendment(self):
        path = self.failed / 'caption_generation/fresh_responses.jsonl'
        rows = self.run.lines(path)
        rows[0]['finish_reason'] = 'content_filter'
        self.run.write_lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'only observed'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.base_prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
