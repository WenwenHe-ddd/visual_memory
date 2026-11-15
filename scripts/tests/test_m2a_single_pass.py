import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / '.vendor_m2a'), str(ROOT / '.vendor'),
                str(ROOT / 'scripts/vendor_m2a/runtime')]
from scripts.memory.m2a_single_pass import build_round, parse_extraction, RetrievalLimits, search


class SinglePassTests(unittest.TestCase):
    def test_only_final_object_brace_can_be_completed(self):
        self.assertEqual(parse_extraction('{"facts":"Already complete.","image_caption":""',False),
                         dict(facts='Already complete.',image_caption=''))
        for broken in ['{"facts":"Partial', '{"facts":"x","image_caption":"cut',
                       '{"facts":"x",', '{"facts":"x"',
                       '```json\n{"facts":"x","image_caption":""}\n```']:
            with self.assertRaises(ValueError):
                parse_extraction(broken, True)

    def test_reject_fabricated_ids_and_invalid_caption(self):
        for content, image in [('{"facts":"x","image_caption":"","evidence_ids":[99]}', False),
                               ('{"facts":[],"image_caption":""}', True)]:
            with self.assertRaises(ValueError):
                parse_extraction(content, image)

    def test_no_image_caption_is_discarded_not_used_as_visual_evidence(self):
        self.assertEqual(parse_extraction('{"facts":"Discussed a dog.","image_caption":"A brown dog."}', False),
                         dict(facts='Discussed a dog.',image_caption=''))
        with self.assertRaises(ValueError):
            parse_extraction('{"facts":"","image_caption":"Invented scene"}', False)

    def test_one_call_program_binds_evidence_and_image(self):
        raw, semantic, images, llm = Mock(), Mock(), Mock(), Mock()
        raw.append.return_value, semantic.add.return_value = 42, 88
        llm.invoke.return_value.content = json.dumps(dict(facts='User likes red.', image_caption='Red card.'))
        obs = dict(id='S1:R2', timestamp='2026-11-13T00:00:00', session_id=1,
                   round=2, text='User: red please', image='/original.png')
        with tempfile.TemporaryDirectory() as tmp:
            result = build_round(obs, raw, semantic, images, llm, Path(tmp))
        self.assertEqual(llm.invoke.call_count, 1)
        self.assertEqual(raw.append.call_count, 1)
        self.assertEqual(semantic.add.call_count, 1)
        memory = semantic.add.call_args.args[0]
        self.assertEqual(memory.evidence_ids, [[42, 42]])
        self.assertEqual(memory.image_path, '/original.png')
        self.assertEqual(result['memory_id'], 88)
        self.assertEqual([c[0] for c in semantic.mock_calls], ['add'])

    def test_limits_reach_database_and_override_model_request(self):
        from agent.stores.semantic import SemanticStore
        db = Mock()
        db.hybrid_search.return_value = [[{'id': n} for n in range(1, 6)]]
        db.search.return_value = [[{'id': n} for n in (6, 7, 8)]]
        store = SimpleNamespace(db=db, log=[], _embed_text=lambda _: [0.] * 384,
            _embed_image=lambda _: [0.] * 768, _get_memory_by_id=lambda n: n)
        store._rrf = lambda paths, k: SemanticStore._rrf(store, paths, k)
        result = search(store, RetrievalLimits(), 'red', '/image.png', top_k=100)
        self.assertEqual(len(result), 5)
        self.assertEqual(len(set(result)), 5)
        kw = db.hybrid_search.call_args.kwargs
        self.assertEqual([r.limit for r in kw['reqs']], [5, 5])
        self.assertEqual(kw['limit'], 5)
        self.assertEqual(db.search.call_args.kwargs['limit'], 3)
        self.assertEqual(store.log[0]['requested_top_k'], 100)
        db.reset_mock()
        search(store, RetrievalLimits(), 'red')
        db.search.assert_not_called()

    def test_format_retry_only_commits_one_memory_and_logs_rejected_response(self):
        raw, semantic, images, llm = Mock(), Mock(), Mock(), Mock()
        raw.append.return_value, semantic.add.return_value = 42, 88
        llm.invoke.side_effect = [SimpleNamespace(content='User: ordinary prose'),
            SimpleNamespace(content='{"facts":"User likes red.","image_caption":""}')]
        obs = dict(id='S1:R2', timestamp='2026-11-13T00:00:00', session_id=1,
                   round=2, text='User: red please', image=None)
        with tempfile.TemporaryDirectory() as tmp:
            result = build_round(obs, raw, semantic, images, llm, Path(tmp), max_attempts=3)
            retries = [json.loads(x) for x in (Path(tmp)/'extraction_retries.jsonl').read_text().splitlines()]
        self.assertEqual(result['extraction_attempts'], 2)
        self.assertEqual(raw.append.call_count, 1)
        self.assertEqual(semantic.add.call_count, 1)
        self.assertEqual(len(retries), 1)
        self.assertTrue(retries[0]['will_retry'])
        self.assertIn('FORMAT RETRY 1', llm.invoke.call_args.args[0][1].content[-1]['text'])
        self.assertEqual(semantic.add.call_args.args[0].evidence_ids, [[42, 42]])

    def test_format_retries_stop_at_limit_without_committing_invalid_memory(self):
        raw, semantic, images, llm = Mock(), Mock(), Mock(), Mock()
        raw.append.return_value = 42
        llm.invoke.return_value.content = 'Still plain text'
        obs = dict(id='S1:R2', timestamp='2026-11-13T00:00:00', session_id=1,
                   round=2, text='User: red please', image=None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_round(obs, raw, semantic, images, llm, Path(tmp), max_attempts=3)
            retries = [json.loads(x) for x in (Path(tmp)/'extraction_retries.jsonl').read_text().splitlines()]
        self.assertEqual(llm.invoke.call_count, 3)
        self.assertEqual(raw.append.call_count, 1)
        semantic.add.assert_not_called()
        self.assertEqual([r['will_retry'] for r in retries], [True, True, False])


if __name__ == '__main__':
    unittest.main()
