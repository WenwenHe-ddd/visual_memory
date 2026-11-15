import tempfile
import threading
import unittest
from pathlib import Path
from scripts.evaluation import recover_mem_gallery_amem_api as recovery


class Fake(recovery.BudgetRecovery):
    wire = staticmethod(recovery.DeepModel.wire)

    def __init__(self, root, reasons):
        self.root = root
        self.environment = {}
        self.fatal_errors = []
        self.lock = threading.Lock()
        self.reasons = iter(reasons)
        self.calls = []
        self.logs = []
        self.init_recovery()

    def log(self, name, record):
        self.logs.append((name, record))

    def unchecked_one(self, messages, cap, temperature):
        self.calls.append(self.wire(messages, cap, temperature))
        return dict(request_sha256=recovery.run.base.request_sha256(self.calls[-1]),
                    finish_reason=next(self.reasons), raw_response='{}', refusal=None,
                    response_id=str(len(self.calls)))


class RecoveryTests(unittest.TestCase):
    def test_only_budget_changes_and_archived_truncation_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as d:
            model = Fake(Path(d), ['length', 'stop', 'stop'])
            messages = [dict(role='user', content='original memory')]
            out = model.one(messages, 1000, .7)
            self.assertEqual(out['budget_recovery']['successful_cap'], 4096)
            self.assertEqual(model.calls[0] | {'max_tokens': 4096}, model.calls[1])
            model.one(messages, 1000, .7)
            self.assertEqual([r['max_tokens'] for r in model.calls], [2048, 4096, 4096])

    def test_non_length_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            model = Fake(Path(d), ['content_filter'])
            with self.assertRaises(RuntimeError):
                model.one([dict(role='user', content='original')], 1000, .7)
            self.assertEqual(len(model.calls), 1)
            self.assertTrue(model.fatal_errors)

    def test_budget_escalation_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            model = Fake(Path(d), ['length'] * 4)
            with self.assertRaises(RuntimeError):
                model.one([dict(role='user', content='original')], 1000, .7)
            self.assertEqual([r['max_tokens'] for r in model.calls], list(recovery.CAPS))


if __name__ == '__main__':
    unittest.main()
