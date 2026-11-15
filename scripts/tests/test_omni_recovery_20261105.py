"""Replay the exact three failed cache entries; verify bounded, auditable repair."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.evaluation.omni_qwen_recovery_20261105 import OUTPUT, FIRST_FAILED, RecoveryBroker


class RecoveryTests(unittest.TestCase):
    def exercise(self, succeeds):
        p = json.loads((OUTPUT / 'protocol.json').read_text())
        first = json.loads((OUTPUT / 'memory/calls' / (FIRST_FAILED + '.json')).read_text())
        keys = [FIRST_FAILED,
            'd428e841d16a52f6a6426604624b722032f410f8954c2605f26996077fea2449',
            'c517147ca14f9578f08495b77e9e5f81d5cb88d63ae35282e29d672a657ff1dd']
        test = self
        class Model:
            generated = 0
            def generate(self, conversations, cap, sampling_temperature=0.):
                self.generated += len(conversations)
                outputs = []
                for messages in conversations:
                    instruction = messages[1]['content']
                    test.assertIn('Validator feedback:', instruction)
                    test.assertIn('/relations/2/object', instruction)
                    test.assertIn('never null', instruction)
                    test.assertEqual(messages[2:], first['request']['messages'][1:])
                    test.assertEqual(cap, first['request']['max_new_tokens'])
                    value = ('{"entities":[{"name":"desk lamp","type":"OBJECT","attributes":{"state":"on"}}],"relations":[]}'
                             if succeeds else first['output']['raw_response'])
                    outputs.append(dict(raw_response=value, input_tokens=600, generated_tokens=100, hit_token_limit=False))
                return outputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'calls').mkdir()
            for key in keys:
                shutil.copyfile(OUTPUT / 'memory/calls' / (key + '.json'), root / 'calls' / (key + '.json'))
            model = Model()
            broker = RecoveryBroker(model, root, p)
            try:
                def call():
                    return broker.client().create(model='local-qwen', messages=first['request']['messages'][1:],
                        response_format={'type':'json_object'}, temperature=0)
                if succeeds:
                    call()
                    self.assertEqual(model.generated, 1)
                    self.assertEqual(broker.errors, [])
                    row = json.loads((root / 'transport_recoveries.jsonl').read_text())
                    self.assertEqual(row['attempt_keys'][:3], keys)
                    self.assertEqual(len(row['attempt_keys']), 4)
                    self.assertEqual(row['recovery_amendment'], '20261105')
                else:
                    with self.assertRaisesRegex(RuntimeError, 'All six'):
                        call()
                    self.assertEqual(model.generated, 3)
                    self.assertEqual(len(broker.errors[0]['attempt_keys']), 6)
                for key in keys:
                    self.assertEqual((root / 'calls' / (key + '.json')).read_bytes(),
                                     (OUTPUT / 'memory/calls' / (key + '.json')).read_bytes())
            finally:
                broker.close()

    def test_three_cached_failures_then_one_regeneration(self):
        self.exercise(True)

    def test_exhaustion_preserves_all_six_attempts(self):
        self.exercise(False)


if __name__ == '__main__':
    unittest.main()
