"""Regression tests for the reproducibility-critical M2A integration boundaries."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor_m2a'))
sys.path.insert(0, str(ROOT / 'scripts/vendor_m2a/runtime'))
from scripts.memory.m2a_local import parse_calls


class ToolTransportTests(unittest.TestCase):
    schemas = [{'type': 'function', 'function': {'name': 'search_semantic_memories'}}]

    def test_native_qwen_tool_call(self):
        text, calls = parse_calls('Checking memory.\n<tool_call>{"name":"search_semantic_memories",'
            '"arguments":{"query_text":"white cat"}}</tool_call>', self.schemas)
        self.assertEqual(text, 'Checking memory.')
        self.assertEqual(calls[0]['args'], {'query_text': 'white cat'})

    def test_malformed_or_unbound_call_is_not_silently_answered(self):
        for raw in ['<tool_call>{"name":"search_semantic_memories"}',
                    '<tool_call>{"name":"delete_memory","arguments":{}}</tool_call>',
                    '<tool_call>{"name":"search_semantic_memories","arguments":"cat"}</tool_call>']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_calls(raw, self.schemas)

    def test_empty_update_response_is_legal(self):
        self.assertEqual(parse_calls('', self.schemas), ('', []))

    def test_unbound_tool_markup_is_plain_text(self):
        raw = '<tool_call>{"name":"update_memory","arguments":{"text":"cat"}}</tool_call>'
        self.assertEqual(parse_calls(raw, []), (raw, []))


class ProvenanceTests(unittest.TestCase):
    def test_upstream_files_match_git_blobs(self):
        base = ROOT / 'scripts/vendor_m2a'
        meta = json.loads((base / 'UPSTREAM.json').read_text())
        for name, expected in meta['git_blobs'].items():
            data = (base / 'upstream' / name).read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            self.assertEqual(actual, expected, name)

    def test_retrieved_image_token_resolves_to_original_file(self):
        from agent.stores.image_manager import ImageManager
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'cat.png')
            Image.new('RGB', (8, 8), 'white').save(path)
            manager = ImageManager()
            blocks = manager.format_image(path)
            self.assertEqual(manager.image_paths, [path])
            self.assertEqual(manager.image_token_to_image('<image0>'), path)
            self.assertTrue(next(b['url'] for b in blocks if b['type'] == 'image').startswith('data:image'))


class GraphTests(unittest.TestCase):
    def test_mandatory_first_query_reads_memory_without_writing_raw_history(self):
        from agent.agents.chat_agent import ChatAgent, ChatAgentState
        from agent.config import ChatAgentConfig, MemoryManagerConfig
        from agent.stores import ImageManager, RawMessageStore
        from langchain_core.messages import AIMessage, HumanMessage

        class Manager:
            config = MemoryManagerConfig()
            def query(self, **kwargs):
                self.request = kwargs
                return [{'type': 'text', 'text': 'Player 0 held two cards.'}]

        class LLM:
            def bind_tools(self, *args, **kwargs):
                return self
            def invoke(self, messages):
                return AIMessage(content='Two cards.')

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.db'
            raw, manager = RawMessageStore(str(path)), Manager()
            before = path.read_bytes()
            agent = ChatAgent(manager, raw, LLM(), ImageManager(), ChatAgentConfig(), update_memory=False)
            state = ChatAgentState(messages=[HumanMessage('How many cards?'), AIMessage(content='',
                tool_calls=[{'name': 'query_memory', 'args': {'text': 'How many cards?'}, 'id': 'mandatory'}])])
            state = agent._query_memory(state)
            result = agent.graph.invoke(state)
            self.assertEqual(manager.request['query_text'], 'How many cards?')
            self.assertEqual(result['response'], 'Two cards.')
            self.assertEqual(path.read_bytes(), before)

    def test_update_and_query_execute_tools_and_terminate(self):
        from agent.agents.memory_manager import MemoryManager, MemoryManagerTools
        from agent.stores import ImageManager, RawMessageStore
        from agent.config import MemoryManagerConfig
        from langchain_core.messages import AIMessage

        class Store:
            def __init__(self):
                self.memories = []

            def add(self, value):
                self.memories.append(value)
                return len(self.memories)

            def hybrid_search(self, **kwargs):
                return []

        class LLM:
            def __init__(self):
                self.outputs = iter([
                    AIMessage(content='', tool_calls=[{'name': 'add_memory',
                        'args': {'text': 'Mimi is a cat', 'evidence_ids': '[[1,1]]'}, 'id': 'add1'}]),
                    AIMessage(content=''),
                    AIMessage(content='', tool_calls=[{'name': 'search_semantic_memories',
                        'args': {'query_text': 'Mimi'}, 'id': 'search1'}]),
                    AIMessage(content='Mimi is a cat.')])

            def bind_tools(self, *args, **kwargs):
                return self

            def invoke(self, messages):
                return next(self.outputs)

        with tempfile.TemporaryDirectory() as tmp:
            raw, store, images, llm = RawMessageStore(str(Path(tmp) / 'raw.db')), Store(), ImageManager(), LLM()
            manager = MemoryManager(MemoryManagerTools(raw, store, images), raw, store, llm, images,
                                    MemoryManagerConfig(max_iteration=2))
            self.assertEqual(manager.update([], query_text='Remember Mimi'), '')
            self.assertEqual(store.memories[0].evidence_ids, [[1, 1]])
            self.assertEqual(manager.query([], query_text='Who is Mimi?'),
                             [{'type': 'text', 'text': 'Mimi is a cat.'}])


if __name__ == '__main__':
    unittest.main()
