"""Regression for floating-point ties in independently audited RRF rankings."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analysis.audit_simplemem_fusion_ranks import audit


class RankAuditTests(unittest.TestCase):
    def test_chronological_roundoff_and_tampered_context(self):
        # Same mathematical sum, different chronological IEEE-754 rounding.
        # B is seen first, but A's accumulated score is one ULP higher.
        batches = []
        for query, (identity, rank) in enumerate(
                [('B', 9), ('B', 7), ('B', 4), ('A', 7), ('A', 4), ('A', 9)]):
            ids = [f'filler_{query}_{n}' for n in range(1, rank)] + [identity]
            batches.append({'channel': 'semantic', 'query': str(query),
                            'result_ids': ids, 'top_k': 10})
        candidates = list(dict.fromkeys(i for b in batches for i in b['result_ids']))
        filler_ranks = {i: n for b in batches for n, i in enumerate(b['result_ids'], 1)
                        if i not in ('A', 'B')}
        expected = ['A', 'B'] + sorted(filler_ranks, key=filler_ranks.get)
        row = {'id': 'roundoff', 'retrieval_detail': {
            'original_item_ids': candidates, 'search_batches': batches,
            'context_memory_ids': expected[:5], 'fusion': {
                'channel_rankings': {'semantic': expected, 'keyword': [], 'structured': []},
                'ranked_candidates': [{'id': i, 'score': 1 / (60 + n)}
                                      for n, i in enumerate(expected, 1)]}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'protocol.json').write_text(json.dumps(
                {'questions': 1, 'experiment_signature': 'roundoff-regression'}))
            (root / 'retrieval_checkpoints').mkdir()
            checkpoint = root / 'retrieval_checkpoints' / 'roundoff.json'
            checkpoint.write_text(json.dumps(row))
            result = audit(root)
            self.assertTrue(result['audit_passed'], result['failures'])
            self.assertTrue(result['complete'])
            row['retrieval_detail']['context_memory_ids'].reverse()
            checkpoint.write_text(json.dumps(row))
            self.assertEqual(audit(root)['failures'],
                             [{'id': 'roundoff', 'check': 'selected_five'}])


if __name__ == '__main__':
    unittest.main()
