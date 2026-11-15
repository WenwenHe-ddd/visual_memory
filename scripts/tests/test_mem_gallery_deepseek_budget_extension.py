import threading
import unittest
from unittest.mock import patch,Mock
from scripts.evaluation import mem_gallery_deepseek_budget_extension as ext


class BudgetTests(unittest.TestCase):
    def model(self):
        m=object.__new__(ext.BudgetModel);m.min_output_tokens=8192
        m.fatal_errors=[];m.lock=threading.Lock();m.log=Mock();return m

    def output(self,finish,identity):
        return dict(finish_reason=finish,refusal=None,raw_response='actual output',
                    response_id=identity,request_sha256=identity+'hash')

    def test_full_output_is_preserved_without_extra_request(self):
        m=self.model();out=self.output('stop','a')
        with patch.object(ext.api.APIModel,'one',return_value=out) as call:
            self.assertIs(m.one(['same'],2048,0.7),out)
        call.assert_called_once_with(['same'],8192,0.7);m.log.assert_not_called()

    def test_only_budget_changes_after_truncation_and_only_complete_output_returns(self):
        m=self.model();first=self.output('length','a');second=self.output('stop','b')
        with patch.object(ext.api.APIModel,'one',side_effect=[first,second]) as call:
            self.assertIs(m.one(['same'],2048,0.7),second)
        self.assertEqual([c.args for c in call.call_args_list],[(['same'],8192,0.7),(['same'],32768,0.7)])
        self.assertEqual(m.log.call_args_list[-1].args[1]['discarded_response_id'],'a')
        self.assertEqual(m.log.call_args_list[-1].args[1]['replacement_response_id'],'b')

    def test_truncation_at_ceiling_is_terminal(self):
        m=self.model()
        with patch.object(ext.api.APIModel,'one',side_effect=[self.output('length','a'),self.output('length','b')]):
            with self.assertRaises(RuntimeError):m.one(['same'],2048,0.7)
        self.assertTrue(m.fatal_errors)


if __name__=='__main__':unittest.main()
