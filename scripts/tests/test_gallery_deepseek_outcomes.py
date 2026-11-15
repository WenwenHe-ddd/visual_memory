from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.evaluation import gallery_deepseek_outcomes as api


class RejectionOutcomeTests(unittest.TestCase):
    def test_content_rejection_is_recorded_once_without_stopping_other_requests(self):
        error=api.strict.provider.APIError('http',http_status=400,retryable=False,message='rejected')
        error.provider_error_code='data_inspection_failed'
        with tempfile.TemporaryDirectory() as tmp, patch.object(api.strict.provider,'load_config',return_value=object()):
            model=api.Model(Path(tmp),Path('/unused'))
            with patch.object(api.strict,'chat',side_effect=error) as chat:
                out=model.invoke({'test':'rejected'},'caption','x')
                self.assertEqual(chat.call_count,1)
            self.assertEqual(out['status'],'provider_rejection')
            self.assertEqual(out['raw_response'],'')
            self.assertIsNone(out['returned_model'])
            self.assertIsNone(out['response_id'])
            self.assertEqual(model.fatal_errors,[])
            with patch.object(api.strict,'chat',return_value={'response_id':'ok'}):
                self.assertEqual(model.invoke({'test':'allowed'},'caption','y'),{'response_id':'ok'})


if __name__=='__main__':unittest.main()
