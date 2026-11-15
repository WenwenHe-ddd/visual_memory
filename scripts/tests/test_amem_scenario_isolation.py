import threading
import unittest
from unittest.mock import patch
from scripts.evaluation import continue_amem_deepseek_isolated as run


class IsolationTests(unittest.TestCase):
    def model(self):
        m = object.__new__(run.IsolatedModel)
        m.rejected = set(); m.fatal_errors = []; m.lock = threading.Lock()
        m.config = None; m.log = lambda *args: None
        return m

    def test_rejection_does_not_poison_other_tasks_and_is_never_resent(self):
        m = self.model()
        exc = run.api.APIError('http', retryable=False, message='rejected')
        exc.provider_error_code = 'data_inspection_failed'
        messages = [dict(role='user', content='test')]
        with patch.object(run.api, 'chat_request', side_effect=exc) as chat:
            for _ in range(2):
                with self.assertRaises(run.ScenarioRejected):
                    m.unchecked_one(messages, 2048, .7)
            self.assertEqual(chat.call_count, 1)
        self.assertEqual(m.fatal_errors, [])
        with patch.object(run.api, 'chat_request', return_value={'response_id':'ok'}) as chat:
            self.assertEqual(m.unchecked_one([dict(role='user',content='other')],2048,.7)['response_id'],'ok')
            chat.assert_called_once()

    def test_terminal_non_content_error_still_stops_run(self):
        m = self.model()
        exc = run.api.APIError('quota', retryable=False, message='quota exhausted')
        with patch.object(run.api, 'chat_request', side_effect=exc):
            with self.assertRaises(run.api.APIError):
                m.unchecked_one([dict(role='user', content='test')],2048,.7)
        self.assertEqual(len(m.fatal_errors),1)

    def test_blocked_scenario_does_not_execute_but_other_scenario_does(self):
        with run.AvailableExecutor(max_workers=1) as pool:
            with self.assertRaises(run.ScenarioRejected):
                pool.submit(lambda task:self.fail('blocked task executed'),next(iter(run.KNOWN_BLOCKED))).result()
            self.assertEqual(pool.submit(lambda task:task,'available').result(),'available')


if __name__ == '__main__':
    unittest.main()
