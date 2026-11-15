"""CPU-only checks for report provenance and paired uncertainty calculations."""
import copy
import unittest

from scripts.analysis.report_mem_gallery_master_answer import (
    CONDITIONS, audit_shared_responses, batch_seconds, paired_bootstrap,
)


class ResultReportTests(unittest.TestCase):
    def fixture(self):
        protocol = {'run_id': 'run', 'experiment_signature': 'signature',
                    'contract': {'unique_payloads': 1}}
        manifest = [{'record_id': f'q::{name}', 'id': 'q', 'condition': name,
                     'payload_sha256': 'payload', 'top5_memory_ids': ['m1', 'm2', 'm3', 'm4', 'm5']}
                    for name in CONDITIONS]
        responses = []
        for i, item in enumerate(manifest):
            responses.append({**item, 'run_id': 'run', 'experiment_signature': 'signature',
                'fresh_inference_executed_for_this_record': i == 0,
                'same_run_shared_inference_record_id': None if i == 0 else manifest[0]['record_id'],
                'raw_response': 'answer', 'status': 'ok', 'input_tokens': 100, 'generated_tokens': 3,
                'hit_token_limit': False, 'execution_id': 'exec', 'batch_size': 1,
                'batch_elapsed_seconds': 2.0})
        return responses, manifest, protocol

    def test_valid_same_run_reuse_and_batch_accounting(self):
        responses, manifest, protocol = self.fixture()
        audit = audit_shared_responses(responses, manifest, protocol)
        self.assertEqual((audit['logical_records'], audit['fresh_inferences'], audit['shared_records']), (2, 1, 1))
        self.assertEqual(batch_seconds(responses, 'reader')['summed_generation_batch_seconds'], 2.0)

    def test_reject_shared_response_tampering(self):
        rows, manifest, protocol = self.fixture()
        for key, bad in [('raw_response', 'different'), ('same_run_shared_inference_record_id', 'old-run'),
                         ('execution_id', 'other-exec'), ('hit_token_limit', True)]:
            with self.subTest(key=key):
                changed = copy.deepcopy(rows)
                changed[1][key] = bad
                with self.assertRaises(ValueError):
                    audit_shared_responses(changed, manifest, protocol)

    def test_reject_extra_or_missing_coverage(self):
        rows, manifest, protocol = self.fixture()
        with self.assertRaises(ValueError):
            audit_shared_responses(rows[:1], manifest, protocol)

    def test_bootstrap_zero_effect_and_clustered_constant_effect(self):
        rows = [{'task': task, CONDITIONS[0]: {'f1': .2}, CONDITIONS[1]: {'f1': .2}}
                for task in ('a', 'a', 'a', 'b')]
        result = paired_bootstrap(rows, repetitions=100)
        self.assertEqual(result['paired_question_percentile_95'], [0., 0.])
        self.assertEqual(result['paired_domain_cluster_percentile_95'], [0., 0.])
        for row in rows:
            row[CONDITIONS[1]]['f1'] = .5
        result = paired_bootstrap(rows, repetitions=100)
        for bounds in ('paired_question_percentile_95', 'paired_domain_cluster_percentile_95'):
            for value in result[bounds]:
                self.assertAlmostEqual(value, .3)


if __name__ == '__main__':
    unittest.main()
