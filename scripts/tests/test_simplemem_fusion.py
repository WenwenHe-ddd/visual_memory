"""Check that later retrieval channels and reflection can reach the final reader."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.evaluation import mcq_reader
from scripts.memory import simplemem_official as official
from scripts.memory.simplemem_fusion import fuse_candidates
from scripts.memory import text_memory_baselines as baseline
from scripts import visual_memory_pipeline as pipeline


def batch(channel, query, ids, phase='initial'):
    return dict(channel=channel, query=query, result_ids=ids, phase=phase)


class FusionTests(unittest.TestCase):
    def test_later_channels_survive_semantic_first_ten_and_repeated_queries_do_not_add_votes(self):
        ids = [str(i) for i in range(10)] + ['keyword-only', 'structured-only', 'joint']
        entries = [SimpleNamespace(entry_id=i) for i in ids]
        batches = [batch('semantic', 'q', ids[:10]),
                   batch('keyword', 'q', ['keyword-only', 'joint']),
                   batch('structured', 'q', ['structured-only', 'joint'])]
        selected, detail = fuse_candidates(entries, batches)
        self.assertEqual(selected[0].entry_id, 'joint')
        self.assertTrue({'keyword-only', 'structured-only'}.issubset({e.entry_id for e in selected[:5]}))
        repeated, repeated_detail = fuse_candidates(entries, batches + [batches[0]] * 20)
        self.assertEqual([e.entry_id for e in repeated], [e.entry_id for e in selected])
        self.assertEqual(repeated_detail, detail)
        extra_queries = [batch('semantic', 'query ' + str(i), ids[:10]) for i in range(10)]
        many_queries, _ = fuse_candidates(entries, batches + extra_queries)
        self.assertEqual([e.entry_id for e in many_queries], [e.entry_id for e in selected])
        self.assertEqual(len(selected), len({e.entry_id for e in selected}))

    def test_new_reflection_only_evidence_can_replace_initial_semantic_hits(self):
        initial = ['old' + str(i) for i in range(10)]
        extra1 = ['new', *['a' + str(i) for i in range(9)]]
        extra2 = ['new', *['b' + str(i) for i in range(9)]]
        entries = [SimpleNamespace(entry_id=i) for i in dict.fromkeys(initial + extra1 + extra2)]
        batches = [batch('semantic', 'original', initial),
                   batch('semantic', 'missing detail A', extra1, 'reflection'),
                   batch('semantic', 'missing detail B', extra2, 'reflection')]
        result, _ = fuse_candidates(entries, batches)
        self.assertEqual(result[0].entry_id, 'new')

    def test_empty_channels_and_missing_trace_are_explicit(self):
        entries = [SimpleNamespace(entry_id=str(i)) for i in range(10)]
        result, detail = fuse_candidates(entries, [batch('semantic', 'q', [e.entry_id for e in entries])])
        self.assertEqual(result, entries)
        self.assertEqual(detail['channel_rankings']['keyword'], [])
        self.assertEqual(fuse_candidates([], [batch('semantic', 'q', [])])[0], [])
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            fuse_candidates(entries, [])

    def test_real_upstream_searches_reflection_and_whole_entry_cap(self):
        official.bootstrap()
        from simplemem.core.hybrid_retriever import HybridRetriever
        from simplemem.core.answer_generator import AnswerGenerator
        from simplemem.core.models.memory_entry import MemoryEntry
        entries = {i: MemoryEntry(entry_id=i, lossless_restatement='Complete evidence ' + i)
                   for i in [str(n) for n in range(10)] + ['keyword', 'structured', 'reflected']}
        searches = []

        def semantic(query, top_k):
            searches.append((query, top_k))
            ids = ['reflected', *[str(n) for n in range(9)]] if query == 'more evidence' else [str(n) for n in range(10)]
            return [entries[i] for i in ids]

        retriever = object.__new__(HybridRetriever)
        retriever.enable_planning = True
        retriever.enable_reflection = True
        retriever.enable_parallel_retrieval = False
        retriever.semantic_top_k, retriever.keyword_top_k, retriever.structured_top_k = 10, 5, 5
        retriever.max_reflection_rounds = 2
        retriever.vector_store = SimpleNamespace(semantic_search=semantic,
            keyword_search=lambda *a, **k: [entries['keyword'], entries['reflected']],
            structured_search=lambda **k: [entries['structured']])
        retriever._analyze_information_requirements = lambda q: {'required_info': []}
        retriever._generate_targeted_queries = lambda *a: ['question']
        retriever._analyze_query = lambda q: {'keywords': ['evidence'], 'persons': ['Alice']}
        statuses = iter(['incomplete', 'complete'])
        retriever._analyze_information_completeness = lambda *a: next(statuses)
        retriever._generate_missing_info_queries = lambda *a: ['more evidence']
        memory = object.__new__(official.OfficialSimpleMemMemory)
        memory.protocol = {'context_fusion': 'channel_rrf', 'context_memory_limit': 5, 'preserve_empty_context': True}
        memory.finalized = True
        memory.core = SimpleNamespace(hybrid_retriever=retriever, answer_generator=object.__new__(AnswerGenerator))
        memory._trace_retrieval_channels()
        context, detail = memory.recall('question')
        self.assertEqual(searches, [('question', 10), ('more evidence', 10)])
        self.assertEqual(len(detail['context_memory_ids']), 5)
        self.assertEqual(len(set(detail['context_memory_ids'])), 5)
        self.assertIn('reflected', detail['context_memory_ids'])
        self.assertIn('structured', detail['context_memory_ids'])
        self.assertNotEqual(detail['context_memory_ids'], detail['original_item_ids'][:5])
        self.assertEqual(context.count('[Context '), 5)
        for identity in detail['context_memory_ids']:
            self.assertIn(entries[identity].lossless_restatement, context)
        self.assertEqual([b['channel'] for b in detail['search_batches']], ['semantic', 'keyword', 'structured', 'semantic'])
        self.assertEqual(detail['search_batches'][-1]['phase'], 'reflection')
        # A second recall must not contain searches from the first question.
        retriever.enable_reflection = False
        _, detail = memory.recall('question')
        self.assertEqual(len(detail['search_batches']), 3)

    def test_cli_exposes_semantic_k_fusion_and_reflection(self):
        with patch.object(baseline, 'prepare') as prepare:
            pipeline.main(['online', '--mode', 'simplemem', '--stage', 'prepare',
                           '--simplemem-semantic-top-k', '10', '--simplemem-context-limit', '5',
                           '--simplemem-context-fusion', 'channel_rrf', '--simplemem-reflection'])
        args = prepare.call_args.args[0]
        self.assertEqual(args.simplemem_semantic_top_k, 10)
        self.assertEqual(args.simplemem_context_limit, 5)
        self.assertTrue(args.simplemem_reflection)
        with self.assertRaises(SystemExit):
            pipeline.main(['online', '--mode', 'simplemem', '--simplemem-semantic-top-k', '0'])


if __name__ == '__main__':
    unittest.main()
