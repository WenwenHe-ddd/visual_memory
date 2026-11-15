import json
import unittest

from scripts.data.normalize_mem_gallery_candidate_format import parse_four_singleton_lists


class CandidateFormatRepairTests(unittest.TestCase):
    def test_preserves_exact_decoded_strings(self):
        statements = ['A red chair.', 'A shelf costs $80.', 'A quote says "Hello".', 'A café has blue walls.']
        raw = '\n'.join(json.dumps([statement]) for statement in statements)
        self.assertEqual(parse_four_singleton_lists(raw), statements)

    def test_rejects_ambiguous_or_incomplete_formats(self):
        examples = [
            '["A"]\n["B"]\n["C"]',
            '["A"]\n["B"]\n["C"]\n["D", "E"]',
            '["A"]\n["B"]\n["C"]\n["D"',
            '["A"]\n["B"]\n["C"]\n[42]',
            '["A"]\n["B"]\n["C"]\n[" D "]',
            'Here are the candidates:\n["A"]\n["B"]\n["C"]\n["D"]',
            '["A", "B", "C", "D"]',
        ]
        for raw in examples:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_four_singleton_lists(raw)


if __name__ == '__main__':
    unittest.main()
