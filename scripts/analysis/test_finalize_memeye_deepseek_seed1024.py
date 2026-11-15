"""Offline fixture tests for finalization safety gates; never use run data."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).with_name('finalize_memeye_deepseek_seed1024.py')
spec = importlib.util.spec_from_file_location('fixture_finalizer', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


class FinalizationGates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='memeye-finalizer-test-')
        self.root = Path(self.temp.name)
        self.implementation = self.root / 'implementation.py'; self.implementation.write_text('fixture = True\n')
        self.pin = hashlib.sha256(self.implementation.read_bytes()).hexdigest()
        self.plan = dict(model='deepseek-v4.1-flash', seed=1024, caption_seed=42, judge_seed=42,
                         source_sha256={str(self.implementation): self.pin}, stages=[])
        self.state = dict(model='deepseek-v4.1-flash', seed=1024, judge_seed=42,
            status='generation_complete_pending_final_audit', current_stage=None, unfinished_stages=[],
            pid=2_000_000_000, stages={})
        self.root.joinpath('suite.lock').touch()
        write(self.root / 'data_export_manifest.json', dict(fixture=True))
        write(self.root / 'export_authorization.json', dict(approved=True, source='explicit_user_reply',
            model='deepseek-v4.1-flash', destination='https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
            manifest_sha256=hashlib.sha256((self.root / 'data_export_manifest.json').read_bytes()).hexdigest()))
        for name in sorted(module.STAGES):
            path = self.root / 'markers' / (name + '.json')
            write(path, dict(status='complete'))
            self.plan['stages'].append(dict(name=name, marker=str(path), marker_key='status',
                                           marker_values=['complete'], depends_on=[]))
            self.state['stages'][name] = dict(status='complete')
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        write(self.root / 'plan.json', self.plan); write(self.root / 'progress.json', self.state)

    def test_completed_generation_and_all47_real_markers_pass(self):
        inspector = module.Inspector(self.root)
        inspector.generation(); inspector.stages()

    def test_active_dispatcher_is_rejected_even_if_status_says_finished(self):
        self.state['pid'] = os.getpid(); self.save()
        with self.assertRaisesRegex(ValueError, 'PID is still active'):
            module.Inspector(self.root).generation()

    def test_running_generation_cannot_be_finalized(self):
        self.state['status'] = 'running'; self.save()
        with self.assertRaisesRegex(ValueError, 'has not reached'):
            module.Inspector(self.root).generation()

    def test_unfinished_current_stage_is_rejected(self):
        self.state['current_stage'] = 'simplemem_open_retrieve'; self.save()
        with self.assertRaisesRegex(ValueError, 'active/unfinished'):
            module.Inspector(self.root).generation()

    def test_live_suite_lock_rejected_even_if_recorded_pid_absent(self):
        with (self.root / 'suite.lock').open('r+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, 'holds suite.lock'):
                module.Inspector(self.root).generation()

    def test_source_pin_mismatch_rejected(self):
        self.implementation.write_text('fixture = False\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            module.Inspector(self.root).generation()

    def test_changed_export_authorization_manifest_rejected(self):
        write(self.root / 'data_export_manifest.json', dict(fixture='changed'))
        with self.assertRaisesRegex(ValueError, 'authorization no longer matches'):
            module.Inspector(self.root).generation()

    def test_fewer_than47_stages_rejected(self):
        self.plan['stages'].pop(); self.save()
        with self.assertRaisesRegex(ValueError, 'exact 47-stage'):
            module.Inspector(self.root).stages()

    def test_skipped_or_failed_stage_not_healed_from_real_marker(self):
        for status in ('failed', 'skipped_dependency', 'running'):
            with self.subTest(status=status):
                self.state['stages']['simplemem_open_retrieve']['status'] = status; self.save()
                with self.assertRaisesRegex(ValueError, 'actually complete/adopt'):
                    module.Inspector(self.root).stages()

    def test_missing_actual_marker_rejected_even_if_all_progress_complete(self):
        Path(self.plan['stages'][0]['marker']).unlink()
        with self.assertRaises(FileNotFoundError):
            module.Inspector(self.root).stages()

    def test_stale_actual_marker_rejected(self):
        write(Path(self.plan['stages'][0]['marker']), dict(status='partial'))
        with self.assertRaisesRegex(ValueError, 'Actual marker not complete'):
            module.Inspector(self.root).stages()

    def test_pending_paired_report_rejected(self):
        write(self.root / 'metrics/paired_mcq_audit.json', dict(status='pending'))
        with self.assertRaisesRegex(ValueError, 'paired MCQ audit is not complete'):
            module.Inspector(self.root).paired_mcq()

    def make_audit_fixture(self):
        def sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        write(self.root / 'metrics/metric_summary.json', dict(fixture=True))
        mcq = dict(status='passed', passed=True, methods_complete=10, metrics_complete=10,
            mcq_records_total=3710, metric_index_complete=True, metric_index_unique_methods=10, failures=[],
            seed=1024, reference_seed=42, judge_seed=42,
            artifact_sha256={str(self.implementation): self.pin},
            source_plan_sha256=sha(self.root / 'plan.json'),
            implementation_sha256=sha(module.ROOT / 'scripts/analysis/audit_memeye_deepseek_seed1024.py'),
            metric_index_sha256=sha(self.root / 'metrics/metric_summary.json'),
            methods=[dict(method=m, status='passed') for m in module.METHODS])
        open_code = module.ROOT / 'scripts/evaluation/memeye_deepseek_seed1024_audit_open_judge.py'
        opened = dict(status='passed', passed=True, methods_passed=10, expected_methods=10,
            open_answers_verified=3710, judge_scores_verified=3710, answer_seed=1024, judge_seed=42,
            real_api_calls_by_audit=0, plan_sha256_at_read=sha(self.root / 'plan.json'), errors=[], pending=[],
            artifact_sha256={str(open_code): sha(open_code)},
            methods=[dict(method=m, status='passed', open_answers_verified=371, judge_scores_verified=371)
                     for m in module.METHODS])
        write(self.root / 'audit/final_audit_mcq_metrics.json', mcq)
        write(self.root / 'audit/final_audit_open_judge.json', opened)
        return opened

    def test_complete_current_audit_schema_passes(self):
        self.make_audit_fixture()
        module.Inspector(self.root).independent_audits()

    def test_open_audit_wrong_seed_or_old_plan_rejected(self):
        for field, value in (('answer_seed', 42), ('plan_sha256_at_read', '0' * 64),
                             ('pending', [{'method': 'simplemem'}])):
            with self.subTest(field=field):
                opened = self.make_audit_fixture(); opened[field] = value
                write(self.root / 'audit/final_audit_open_judge.json', opened)
                with self.assertRaisesRegex(ValueError, 'Open/Judge audit code/plan/seed/completion'):
                    module.Inspector(self.root).independent_audits()

    def test_open_audit_must_explicitly_pin_current_auditor_code(self):
        opened = self.make_audit_fixture(); opened['artifact_sha256'] = {str(self.implementation): self.pin}
        write(self.root / 'audit/final_audit_open_judge.json', opened)
        with self.assertRaisesRegex(ValueError, 'Open/Judge audit code/plan/seed/completion'):
            module.Inspector(self.root).independent_audits()

    def test_unsealed_attribution_rejected(self):
        source = self.root / 'text_methods/simplemem'
        write(source / 'protocol.json', dict(experiment_signature='native-fixture'))
        folder = self.root / 'metrics/simplemem_source_attribution'
        write(folder / 'protocol.json', dict(seed=42, memory_generation_seed=1024, fresh_source_only=True,
            gold_blind=True, source_root=str(source), source_experiment_signature='native-fixture'))
        write(folder / 'annotation_lock.json', dict(status='unsealed'))
        write(folder / 'recall_summary.json', dict(status='complete'))
        with self.assertRaisesRegex(ValueError, 'sealed complete fresh scoring'):
            module.Inspector(self.root).attribution()

    def test_missing_final_workbook_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            module.Inspector(self.root).workbook()

    def test_missing_audits_metrics_and_recovery_rejected_by_drycheck_without_writes(self):
        before = snapshot(self.root)
        report = module.Inspector(self.root).check()
        self.assertFalse(report['passed'])
        self.assertEqual(report['status'], 'not_ready')
        failed = {i['gate'] for i in report['issues']}
        self.assertTrue({'independent_all10_audits', 'all10_metric_index', 'full_paired_mcq',
                         'fresh_simplemem_seal_and_score', 'final_workbook_json_csv',
                         'preserved_recovery_evidence'} <= failed)
        self.assertEqual(before, snapshot(self.root))

    def test_apply_rejects_incomplete_evidence_and_preserves_all_fixture_bytes(self):
        before = snapshot(self.root)
        with self.assertRaisesRegex(ValueError, 'Finalization refused'):
            module.apply(self.root)
        self.assertEqual(before, snapshot(self.root))
        self.assertFalse((self.root / 'completion_manifest.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
