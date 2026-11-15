"""Regression cases from failed M2A runs and native tool/store integration."""
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / '.vendor_m2a'), str(ROOT / 'scripts/vendor_m2a/runtime')]
from scripts.memory.m2a_evidence import normalize_evidence, install_evidence_policy


class EvidenceTests(unittest.TestCase):
    def test_actual_singleton(self):
        self.assertEqual(normalize_evidence('[[83]]', {83}), [[83, 83]])

    def test_actual_three_ids_do_not_add_intervening_messages(self):
        result = normalize_evidence('[[99, 101, 103]]', {99, 101, 103})
        self.assertEqual(result, [[99, 99], [101, 101], [103, 103]])
        self.assertEqual({i for a,b in result for i in range(a,b+1)}, {99, 101, 103})

    def test_pairs_are_still_inclusive_ranges(self):
        self.assertEqual(normalize_evidence([[99, 103]], set(range(99,104))), [[99,103]])
        with self.assertRaises(ValueError):
            normalize_evidence([[99,103]], {99,101,103})

    def test_mixed_groups_preserve_order_and_input(self):
        original = [[289,291],[293],[11,9,7]]
        saved = copy.deepcopy(original)
        result = normalize_evidence(original)
        self.assertEqual(result, [[289,291],[293,293],[11,11],[9,9],[7,7]])
        self.assertEqual(original, saved)
        self.assertEqual(normalize_evidence(result), result)

    def test_invalid_or_ambiguous_inputs_are_rejected(self):
        for value in ['not json', {}, None, [1,3], [[]], [['83']], [[True]],
                      [[1.0]], [[0]], [[-3]], [[5,3]], [[[1,2]]]]:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                normalize_evidence(value)

    def test_unknown_and_huge_ranges_are_rejected(self):
        for value in [[[8]], [[1,10**12]], [[1,3,8]]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_evidence(value, set(range(1,8)))


class NativeInterfacesTests(unittest.TestCase):
    def setUp(self):
        from agent.stores import RawMessageStore, RawMessage, ImageManager
        from agent.agents.memory_manager import MemoryManagerTools
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.raw = RawMessageStore(str(self.directory / 'raw.db'))
        for i in range(7):
            self.raw.append(RawMessage(0, datetime(2026,1,1), 'conversation_round', f'message {i+1}', None))
        self.store = Mock()
        self.store.add.return_value = 42
        self.tools = MemoryManagerTools(self.raw, self.store, ImageManager())
        self.cleanup = install_evidence_policy(self.directory)

    def tearDown(self):
        self.cleanup()
        self.temp.cleanup()

    def test_native_add_tool_normalizes_before_writing(self):
        result = self.tools.get_add_memory().invoke({'text':'three facts', 'evidence_ids':'[[1,3,7]]'})
        self.assertIn('Created memory', result)
        self.assertEqual(self.store.add.call_args.args[0].evidence_ids, [[1,1],[3,3],[7,7]])
        log = json.loads((self.directory/'evidence_validation.jsonl').read_text())
        self.assertEqual(log['original'], '[[1,3,7]]')
        self.assertEqual(log['canonical'], [[1,1],[3,3],[7,7]])

    def test_invalid_add_returns_correction_without_writing(self):
        for value in ['[[8]]', '[[7,1]]', '[[false]]']:
            result = self.tools.get_add_memory().invoke({'text':'fact', 'evidence_ids':value})
            self.assertTrue(result.startswith('Error creating memory:'))
        self.store.add.assert_not_called()

    def test_raw_fetch_normalizes_exact_ids(self):
        with patch.object(self.raw, 'fetch_by_ids', wraps=self.raw.fetch_by_ids) as fetch:
            result = self.tools.get_fetch_raw_messages().invoke({'id_ranges':'[[1,3,7]]'})
        self.assertIsInstance(result, list)
        self.assertEqual(fetch.call_args.args[0], [[1,1],[3,3],[7,7]])

    def test_legacy_semantic_read_does_not_rewrite_database(self):
        from agent.stores import SemanticStore
        saved = {'id':58, 'text':'fact', 'image_caption':'', 'image_path':'',
                 'evidence_ids':'[[99,101,103]]'}
        store = SemanticStore.__new__(SemanticStore)
        store.db = Mock()
        store.db.get.return_value = [saved]
        memory = store._get_memory_by_id(58)
        self.assertEqual(memory.evidence_ids, [[99,99],[101,101],[103,103]])
        self.assertEqual(saved['evidence_ids'], '[[99,101,103]]')
        store.db.upsert.assert_not_called()

    def test_store_validates_before_first_insert(self):
        from agent.stores import SemanticStore, SemanticMemory
        store = SemanticStore.__new__(SemanticStore)
        store.db = Mock()
        with self.assertRaises(ValueError):
            store.add(SemanticMemory(text='fact', evidence_ids=[[3,1]]))
        store.db.insert.assert_not_called()


if __name__ == '__main__':
    unittest.main()
