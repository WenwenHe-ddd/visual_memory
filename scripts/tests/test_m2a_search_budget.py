"""Offline tests of the search cap across the actual nested agent graphs."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / '.vendor_m2a'), str(ROOT / '.vendor'),
                str(ROOT / 'scripts/vendor_m2a/runtime')]

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent.agents.chat_agent import ChatAgent, ChatAgentState
from agent.agents.memory_manager import MemoryManager, MemoryManagerTools
from agent.config import ChatAgentConfig, MemoryManagerConfig
from agent.stores import ImageManager
from scripts.memory.m2a_search_budget import BudgetedLLM, SearchBudget


class SearchingLLM:
    stage = 'retrieval'

    def __init__(self, mode):
        self.mode = mode
        self.counter = 0

    def invoke(self, messages):
        return AIMessage(content='Answer from retrieved evidence.')

    def bind_tools(self, tools, **kwargs):
        names = {tool.name for tool in tools}

        def respond(messages):
            self.counter += 1
            if 'search_semantic_memories' in names:
                if self.mode == 'outer' and any(isinstance(m, ToolMessage) for m in messages):
                    return self.invoke(messages)
                name, args = 'search_semantic_memories', {'query_text': 'test'}
            elif 'query_memory' in names:
                name, args = 'query_memory', {'text': 'search again'}
            else:
                return self.invoke(messages)
            return AIMessage(content='', tool_calls=[dict(
                name=name, args=args, id=f'call_{self.counter}', type='tool_call')])

        return SimpleNamespace(invoke=respond)


class SearchBudgetTests(unittest.TestCase):
    def run_graph(self, mode, additional=2):
        budget = SearchBudget(additional)
        raw, semantic = Mock(), Mock()

        def search(**kwargs):
            budget.consume()
            return []

        semantic.hybrid_search.side_effect = search
        images = ImageManager()
        llm = BudgetedLLM(SearchingLLM(mode), budget)
        mm = MemoryManager(MemoryManagerTools(raw, semantic, images), raw, semantic,
                           llm, images, MemoryManagerConfig(context_window=5, max_iteration=15))
        for _ in range(2):
            # Both agents share the budget, which resets only at the next question.
            budget.reset()
            chat = ChatAgent(mm, raw, llm, images,
                             ChatAgentConfig(max_query_iteration=5), update_memory=False)
            state = ChatAgentState(messages=[HumanMessage('Remember the event.'),
                AIMessage(content='', tool_calls=[dict(name='query_memory',
                    args={'text': 'Remember the event.'}, id='initial', type='tool_call')])])
            state = chat._query_memory(state)
            result = chat.graph.invoke(state, config={'recursion_limit': 100})
            self.assertEqual(budget.used, 1 + additional)
            self.assertEqual(result['response'], 'Answer from retrieved evidence.')
            self.assertFalse(any('Tool error:' in str(m.content) for m in result['messages']))
        self.assertEqual(semantic.hybrid_search.call_count, 2 * (1 + additional))

    def test_inner_loop_cannot_exceed_three_searches(self):
        self.run_graph('inner')

    def test_repeated_outer_queries_share_three_searches(self):
        self.run_graph('outer')

    def test_zero_additional_rounds_allows_one_search(self):
        self.run_graph('inner', additional=0)

    def test_one_additional_round_caps_both_loops_at_two_searches(self):
        self.run_graph('inner', additional=1)
        self.run_graph('outer', additional=1)

    def test_exhaustion_preserves_raw_fetch_and_construction_tools(self):
        llm = Mock(stage='retrieval')
        budget = SearchBudget(0)
        budget.consume()
        proxy = BudgetedLLM(llm, budget)
        search = SimpleNamespace(name='search_semantic_memories')
        query = SimpleNamespace(name='query_memory')
        raw = SimpleNamespace(name='fetch_raw_messages')
        proxy.bind_tools([search, raw], parallel_tool_calls=False)
        llm.bind_tools.assert_called_once_with([raw], parallel_tool_calls=False)
        messages = [HumanMessage('Question')]
        proxy.bind_tools([query]).invoke(messages)
        sent = llm.invoke.call_args.args[0]
        self.assertIn('1 searches have been used', sent[0].content)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, 'Question')
        llm.stage = 'construction'
        proxy.bind_tools([search, raw])
        llm.bind_tools.assert_called_with([search, raw])

    def test_execution_guard_blocks_extra_calls_and_invalid_limits(self):
        budget = SearchBudget(2)
        for _ in range(3):
            budget.consume()
        with self.assertRaises(RuntimeError):
            budget.consume()
        self.assertEqual(budget.used, 3)
        with self.assertRaises(ValueError):
            SearchBudget(-1)


if __name__ == '__main__':
    unittest.main()
