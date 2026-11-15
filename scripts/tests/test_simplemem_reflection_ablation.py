"""Exercise real upstream reflection and its effect on a final first-five cap."""
from types import SimpleNamespace
import unittest

from scripts.evaluation import simplemem_reflection_ablation as experiment
from scripts.memory import simplemem_official as official


class ReflectionContextTests(unittest.TestCase):
    def test_real_two_round_reflection_appends_without_changing_initial_five(self):
        official.bootstrap()
        from simplemem.core.hybrid_retriever import HybridRetriever
        from simplemem.core.models.memory_entry import MemoryEntry
        initial = [MemoryEntry(entry_id=str(i), lossless_restatement=f'Initial fact {i}') for i in range(7)]
        new = [MemoryEntry(entry_id=str(i), lossless_restatement=f'New fact {i}') for i in range(7, 10)]
        retriever = object.__new__(HybridRetriever)
        retriever.max_reflection_rounds = 2
        retriever.enable_parallel_retrieval = False
        retriever.semantic_top_k = 5
        retriever._analyze_information_completeness = lambda *args: 'incomplete'
        retriever._generate_missing_info_queries = lambda *args: ['additional query']
        retriever.vector_store = SimpleNamespace(semantic_search=lambda query, top_k: [new[0], initial[1], *new[1:]])
        memory = SimpleNamespace(core=SimpleNamespace(hybrid_retriever=retriever))
        trace = dict(question_id='q', phase='initial', searches=[], assessments=[], additional_queries=[])
        control = {'q': {'retrieval_detail': {'original_item_ids': [e.entry_id for e in initial]}}}
        experiment.instrument(memory, trace, control)
        results = retriever._retrieve_with_intelligent_reflection('q', initial, {})
        self.assertEqual([e.entry_id for e in results], [str(i) for i in range(10)])
        self.assertEqual(results[:5], initial[:5])
        self.assertEqual(len(trace['assessments']), 2)
        self.assertEqual(len(trace['searches']), 2)
        self.assertTrue(all(s['top_k'] == 5 and s['phase'] == 'reflection' for s in trace['searches']))


if __name__ == '__main__':
    unittest.main()
