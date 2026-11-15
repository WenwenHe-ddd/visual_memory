
# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import unittest

from scripts.graph.graph_json_format import readable_dumps


class ReadableGraphJsonTests(unittest.TestCase):
    def test_round_trip_keeps_vectors_inline_and_facts_separate(self):
        graph = {'clusters': [{'id': 'M1', 'shared': ['red sign', 'desk'],
                               'embedding': [0.1, -0.2, 0.3]}],
                 'shared_edges': []}
        rendered = readable_dumps(graph)
        self.assertEqual(json.loads(rendered), graph)
        self.assertIn('"embedding": [0.1, -0.2, 0.3]', rendered)
        self.assertIn('"red sign",\n', rendered)


if __name__ == '__main__':
    unittest.main()
