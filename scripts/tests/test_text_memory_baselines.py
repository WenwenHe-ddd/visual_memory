
# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from scripts.evaluation import mcq_reader  # Establish project's isolated dependencies.
from scripts.memory import text_memory_baselines as baseline
from scripts.memory.text_memory_transport import Broker, PerRowTemperatureSampler


class FakeModel:
    def __init__(self):
        self.calls = []
    def generate(self, messages, cap, sampling_temperature=0.0):
        self.calls.append((messages, cap))
        return [{'raw_response': 'A', 'input_tokens': 20, 'generated_tokens': 2,
                 'hit_token_limit': False} for _ in messages]


class TextMemoryProtocolTests(unittest.TestCase):
    def test_simplemem_ablation_run_writes_task_and_combined_answers(self):
        from scripts.evaluation import simplemem_context_ablation as ablation
        class Reader:
            environment = {'device': 'test'}
            def generate(self, messages, cap, sampling_temperature=0):
                return [{'raw_response': 'A', 'input_tokens': 20, 'generated_tokens': 2,
                         'hit_token_limit': False} for _ in messages]
        protocol = {'seed': 42, 'experiment_signature': 'smoke', 'generation_batch_size': 2,
                    'task_workers': 2, 'tasks': ['one', 'two'], 'max_new_tokens': 16,
                    'reader_system_prompt': 'Choose one letter.'}
        rows = [{**self.q, 'id': task, 'task': task, 'retrieved_context': 'full context',
                 'retrieval_detail': {}} for task in protocol['tasks']]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(ablation, 'verify', return_value=(protocol, rows)), \
                    patch.object(ablation, 'SingleGpuReader', return_value=Reader()):
                ablation.run(root)
            self.assertEqual([r['id'] for r in baseline.read_rows(root / 'qa_responses.jsonl')], ['one', 'two'])
            self.assertTrue(all((root / task / 'responses.jsonl').exists() for task in protocol['tasks']))

    def test_simplemem_ablation_reader_accepts_broker_temperature_argument(self):
        from scripts.evaluation.simplemem_context_ablation import SingleGpuReader
        reader = object.__new__(SingleGpuReader)
        messages = [[{'role': 'user', 'content': 'question'}]]
        with patch.object(baseline.LocalQwen, 'generate', return_value=[{'raw_response': 'A'}]) as generate:
            self.assertEqual(reader.generate(messages, 16, sampling_temperature=0), [{'raw_response': 'A'}])
            generate.assert_called_once_with(messages, 16)

    def test_simplemem_context_cap_preserves_whole_items_and_order(self):
        @dataclass
        class Result:
            items: list
            expansion_candidates: list
            def to_dict(self):
                return {'items': self.items, 'expansion_candidates': self.expansion_candidates}
        items = [{'id': str(i), 'summary': f'complete summary {i}',
                  'details': {'text': 'complete original text'}} for i in range(7)]
        result = Result([items[0], items[0], *items[1:]], ['0', '6'])
        bounded, detail = baseline.bounded_simplemem_result(result, 5)
        self.assertEqual(bounded.items, items[:5])
        self.assertEqual(bounded.expansion_candidates, ['0'])
        self.assertEqual(detail['context_memory_ids'], ['0', '1', '2', '3', '4'])
        self.assertEqual(len(result.items), 8)
        with self.assertRaisesRegex(ValueError, 'retrieved only 1'):
            baseline.bounded_simplemem_result(Result([items[0]] * 6, []), 5)
        empty, detail = baseline.bounded_simplemem_result(Result([], []), 5, allow_empty=True)
        self.assertEqual(empty.items, [])
        self.assertTrue(detail['source_empty_preserved'])
        with self.assertRaisesRegex(ValueError, 'retrieved only 0'):
            baseline.bounded_simplemem_result(Result([], []), 5)
        args = SimpleNamespace(mode='simplemem', output_root=Path('/outputs'), simplemem_context_limit=5)
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top5/simplemem_context5'))
        args.simplemem_context_limit = None
        self.assertEqual(baseline.output_dir(args), Path('/outputs/simplemem'))

    def setUp(self):
        self.q = dict(question='Which chair was chosen?', options={k: 'option_' + k for k in 'ABCD'},
                      gold_choice='C', gold_ids=['secret_evidence'], query_image_captions=['A red chair.'])

    def test_labels_and_options_cannot_change_retrieval(self):
        altered = copy.deepcopy(self.q)
        altered.update(gold_choice='A', gold_ids=['different_secret'], options={'A': 'RETRIEVAL_LEAK'})
        self.assertEqual(baseline.retrieval_query(self.q), baseline.retrieval_query(altered))
        self.assertNotIn('option_', baseline.retrieval_query(self.q))

    def test_gold_is_not_used_to_make_answer_prompt(self):
        protocol = {'reader_system_prompt': 'Choose one letter.'}
        altered = dict(self.q, gold_choice='D', gold_ids=['other'])
        before = baseline.reader_messages(self.q, 'same memory', protocol)
        after = baseline.reader_messages(altered, 'same memory', protocol)
        self.assertEqual(before, after)
        self.assertIn('C. option_C', before[-1]['content'])
        self.assertTrue(all(isinstance(m['content'], str) for m in before))

    def test_pinned_source_files(self):
        self.assertEqual(baseline.verify_vendor()['commit'], '0358e70d714980980dfd8c87903384db474f0b16')

    def test_a_mem_k_reaches_agent_and_isolates_outputs(self):
        config = baseline.effective_method_defaults('a_mem')
        self.assertEqual(config['retrieve_k'], 5)
        self.assertEqual(baseline.effective_method_defaults('simplemem')['retrieve_k'], 20)
        protocol = {'method_defaults': config, 'embedding_model_path': 'test-encoder'}
        seen = []
        class Agent:
            def __init__(self, cfg, model):
                self.retrieve_k = cfg['retrieve_k']
            def generate_query(self, query):
                return 'keywords'
            def retrieve_memory(self, query):
                seen.append((query, self.retrieve_k))
                return 'primary hits and linked neighbors'
        with patch.object(baseline, 'helpers', return_value={'AMemAgent': Agent}):
            memory = baseline.Memory('a_mem', protocol, Path('/unused'))
            context, detail = memory.recall('question')
        self.assertEqual(seen, [('keywords', 5)])
        self.assertEqual(detail['primary_top_k'], 5)
        self.assertIn('linked neighbors', context)
        args = SimpleNamespace(mode='a_mem', output_root=Path('/outputs'), top_k=None)
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top5/a_mem'))
        args.a_mem_context_limit = 5
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top5/a_mem_context5'))
        args.a_mem_context_limit = None
        args.top_k = 10
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top10/a_mem'))
        self.assertEqual(baseline.effective_method_defaults('a_mem', 10)['retrieve_k'], 10)
        with self.assertRaises(ValueError):
            baseline.effective_method_defaults('a_mem', 0)

    def test_a_mem_bounded_context_preserves_neighbors_and_complete_unique_notes(self):
        notes = {str(i): SimpleNamespace(id=str(i), timestamp='date',
                 content=f'complete note {i}\ntalk start time: quoted marker', context='context',
                 keywords=['key'], tags=['tag'], links=[]) for i in range(7)}
        notes['0'].links = [1, '1', 'missing', '2', 3]
        searches = []
        def search(query, k):
            searches.append((query, k))
            return [0, 2, 4, 5, 6]
        system = SimpleNamespace(memories=notes, retriever=SimpleNamespace(search=search))
        memory = object.__new__(baseline.Memory)
        memory.method = 'a_mem'
        memory.protocol = {'context_memory_limit': 5}
        memory.core = SimpleNamespace(memory_system=system, retrieve_k=5,
                                      generate_query=lambda query: 'generated keywords')
        context, detail = memory.recall('question')
        self.assertEqual(searches, [('generated keywords', 5)])
        self.assertEqual(detail['primary_memory_ids'], ['0', '2', '4', '5', '6'])
        self.assertEqual(detail['context_memory_ids'], ['0', '1', '2', '3', '4'])
        self.assertEqual(detail['context_memory_count'], 5)
        self.assertEqual(detail['generated_keywords'], 'generated keywords')
        expected = ''.join('talk start time:date' + 'memory content: ' + notes[str(i)].content
                           + "memory context: contextmemory keywords: ['key']memory tags: ['tag']\n"
                           for i in range(5))
        self.assertEqual(context, expected)
        messages = baseline.reader_messages(self.q, context, {'reader_system_prompt': 'Choose.'})
        self.assertEqual(messages[1]['content'], 'Retrieved memory context:\n' + context)

    def test_a_mem_bounded_context_rejects_insufficient_distinct_memories(self):
        note = SimpleNamespace(id='only', links=['only', 0])
        system = SimpleNamespace(memories={'only': note},
                                 retriever=SimpleNamespace(search=lambda q, k: [0]))
        with self.assertRaisesRegex(ValueError, 'requires 5 distinct notes; retrieved only 1'):
            baseline.bounded_a_mem_context(system, 'query', 5, 5)
        system.memories = {}
        with self.assertRaisesRegex(ValueError, 'retrieved only 0'):
            baseline.bounded_a_mem_context(system, 'query', 5, 5)

    def test_graph_paths_are_bound_before_loading_in_each_task(self):
        local = threading.local()
        barrier = threading.Barrier(2)
        class Graph:
            def __init__(self, storage_path=None):
                self.loaded_path = storage_path
        factory = baseline.task_graph_factory(Graph, local)
        def construct(name):
            local.knowledge_graph_path = Path(name) / 'knowledge_graph'
            barrier.wait()
            return factory().loaded_path
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(set(pool.map(construct, ['task_a', 'task_b'])),
                             {'task_a/knowledge_graph', 'task_b/knowledge_graph'})
        with self.assertRaises(AttributeError):
            factory()  # Never fall back to shared storage without a task binding.

    def test_official_note_helpers_use_replaced_caption(self):
        row = {'user': 'hello', 'assistant': 'yes',
               'raw': {'input_image': ['image.jpg'], 'image_caption': ['QWEN_CAPTION'], 'image_id': ['I1']},
               'answer': 'DO_NOT_PASS', 'clue': 'DO_NOT_PASS'}
        for method in baseline.METHODS:
            h = baseline.helpers(method)
            if method == 'memoryos':
                text = h['_round_user_with_captions'](row)
            else:
                key = 'build_a_mem_note_text' if method == 'a_mem' else '_build_round_text'
                text = h[key](row, 'user', 'assistant')
            self.assertIn('QWEN_CAPTION', text)
            self.assertNotIn('DO_NOT_PASS', text)

    def test_transport_cache_binds_messages_cap_and_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel()
            protocol = {'seed': 42, 'experiment_signature': 'first'}
            broker = Broker(model, Path(d), protocol)
            try:
                messages = [{'role': 'user', 'content': 'question'}]
                first = broker.request(messages, 16)
                second = broker.request(messages, 16)
                third = broker.request(messages, 17)
                self.assertEqual(first, second)
                self.assertNotEqual(first['call_key'], third['call_key'])
                self.assertEqual(len(model.calls), 2)
                events = baseline.read_rows(Path(d) / 'call_events.jsonl')
                self.assertEqual([e['reused'] for e in events], [False, True, False])
            finally:
                broker.close()

    def test_native_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            broker = Broker(FakeModel(), Path(d), {'seed': 42, 'experiment_signature': 'test'})
            try:
                with self.assertRaisesRegex(ValueError, 'native images'):
                    broker.request([{'role': 'user', 'content': [{'type': 'image_url'}]}], 16)
            finally:
                broker.close()

    def test_long_batches_split_without_changing_requests(self):
        class Processor:
            class Tokenizer:
                def encode(self, rendered, **kwargs):
                    return list(rendered)
            tokenizer = Tokenizer()
            def apply_chat_template(self, messages, **kwargs):
                return ''.join(m['content'] for m in messages)
        with tempfile.TemporaryDirectory() as d:
            model = FakeModel()
            model.processor = Processor()
            broker = Broker(model, Path(d), {'seed': 42, 'experiment_signature': 'split',
                            'max_padded_batch_tokens': 30}, batch_size=8)
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    pending = [pool.submit(broker.request, [{'role': 'user', 'content': str(i)*20}], 8) for i in range(4)]
                    results = [future.result() for future in pending]
                self.assertEqual(len(results), 4)
                self.assertEqual(len(model.calls), 4)
                for messages, cap in model.calls:
                    self.assertEqual(len(messages), 1)
                    self.assertEqual(len(messages[0][0]['content']), 20)
                    self.assertEqual(cap, 8)
            finally:
                broker.close()

    def test_sampling_rng_is_independent_of_batch_neighbors(self):
        import torch
        solo = PerRowTemperatureSampler(.7, 42)
        batch = PerRowTemperatureSampler(.7, 42)
        scores = torch.tensor([[.1,.2,.3,.4]])
        for _ in range(20):
            one = solo(None, scores)
            many = batch(None, torch.cat([scores, scores * -2], dim=0))
            self.assertTrue(torch.equal(one[0], many[0]))
            self.assertEqual(torch.isfinite(one).sum().item(), 1)


if __name__ == '__main__':
    unittest.main()
