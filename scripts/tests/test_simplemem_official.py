"""Official core integration: real MiniLM/LanceDB, scripted LLM responses.

These tests validate wiring and storage, not Qwen quality or benchmark accuracy.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.evaluation import mcq_reader  # Project dependency bootstrap.
from scripts.memory import text_memory_baselines as baseline
from scripts.memory import simplemem_official as official
from scripts import visual_memory_pipeline as pipeline


class ScriptedModel:
    environment = {'device': 'scripted LLM; real CPU MiniLM and LanceDB'}

    def __init__(self, protocol):
        self.extractions = 0

    def generate(self, messages, cap, sampling_temperature=0.0):
        outputs = []
        for conversation in messages:
            system = conversation[0]['content']
            if 'information extraction assistant' in system:
                self.extractions += 1
                word = 'Coffeebean' if self.extractions == 1 else 'Zebrafinch'
                response = [{'lossless_restatement': f'Alice likes {word} item {i}.',
                             'keywords': [word], 'persons': ['Alice'], 'entities': [word],
                             'timestamp': '2026-01-01T12:00:00', 'topic': 'preferences'} for i in range(6)]
            elif 'information requirement analyst' in system:
                response = {'question_type': 'factual', 'key_entities': ['Alice'],
                            'required_info': [], 'relationships': [], 'minimal_queries_needed': 1}
            elif 'query generation specialist' in system:
                response = {'queries': ['Alice Zebrafinch'], 'reasoning': 'Find preferences.'}
            elif 'query analysis assistant' in system:
                response = {'keywords': ['Zebrafinch'], 'persons': ['Alice'],
                            'entities': [], 'location': None, 'time_expression': None}
            elif system == 'Choose one letter.':
                outputs.append({'raw_response': 'A', 'input_tokens': 10,
                                'generated_tokens': 1, 'hit_token_limit': False})
                continue
            else:
                raise AssertionError('unexpected LLM operation: ' + system)
            outputs.append({'raw_response': json.dumps(response), 'input_tokens': 10,
                            'generated_tokens': 100, 'hit_token_limit': False})
        return outputs


class OfficialSimpleMemTests(unittest.TestCase):
    def test_query_analysis_repairs_nested_array_parse_without_changing_valid_objects(self):
        official.bootstrap()
        from simplemem.core.utils.llm_client import LLMClient
        # Real failing Qwen syntax: upstream falls back to the inner array.
        malformed = '{"keywords": ["pantry"], "time_expression": "early", "later"}'
        parsed = object.__new__(LLMClient).extract_json(malformed)
        self.assertIsInstance(parsed, list)
        valid = {'keywords': ['pantry'], 'persons': [], 'entities': [],
                 'time_expression': 'early and later', 'location': None}
        requests = []

        def chat(messages, temperature):
            requests.append([dict(m) for m in messages])
            return '[]' if len(requests) == 1 else json.dumps(valid)

        with tempfile.TemporaryDirectory() as directory:
            memory = object.__new__(official.OfficialSimpleMemMemory)
            memory.workdir = Path(directory)
            retriever = SimpleNamespace(_analyze_query=lambda query: parsed)
            memory.core = SimpleNamespace(hybrid_retriever=retriever,
                                          llm_client=SimpleNamespace(chat_completion=chat))
            memory._guard_query_analysis()
            self.assertEqual(retriever._analyze_query('pantry query'), valid)
            self.assertEqual(len(requests), 2)
            self.assertNotEqual(requests[0], requests[1])
            repairs = json.loads((memory.workdir / 'query_analysis_repairs.json').read_text())
            self.assertEqual([r['success'] for r in repairs], [False, True])
            retriever._analyze_query = lambda query: valid
            memory._guard_query_analysis()
            self.assertIs(retriever._analyze_query('valid query'), valid)
            self.assertEqual(len(requests), 2)

    def test_cli_defaults_to_official_and_isolates_historical_results(self):
        with patch.object(baseline, 'prepare') as prepare:
            pipeline.main(['online', '--mode', 'simplemem', '--stage', 'prepare'])
        args = prepare.call_args.args[0]
        self.assertEqual(args.simplemem_backend, official.BACKEND)
        self.assertEqual(args.simplemem_semantic_top_k, 10)
        self.assertEqual(args.simplemem_context_fusion, 'channel_rrf')
        args.output_root = Path('/outputs')
        args.simplemem_context_limit = 5
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top5/simplemem_official_text_context5'))
        args.simplemem_backend = 'omni_legacy'
        self.assertEqual(baseline.output_dir(args), Path('/outputs/top5/simplemem_context5'))

    def test_real_core_builds_tail_window_retrieves_and_uses_common_reader(self):
        metadata = official.verify_installation()
        import openai
        import sentence_transformers
        from simplemem.core.database.vector_store import VectorStore
        from simplemem.core.hybrid_retriever import HybridRetriever
        original_semantic_search = VectorStore.semantic_search
        embedding = baseline.EMBEDDING / 'snapshots' / (baseline.EMBEDDING / 'refs/main').read_text().strip()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(mode='simplemem', simplemem_backend='official_text',
                                   simplemem_context_limit=5, output_root=Path(directory))
            root = baseline.output_dir(args)
            root.mkdir(parents=True)
            observations = {'test': [{'id': f'R{i}', 'session_id': 'S1',
                'timestamp': '2026-01-01T12:00:00', 'speaker': 'Alice',
                'user': f'Alice discusses preference {i}.', 'assistant': 'Understood.',
                'raw': {'image_caption': [f'caption marker {i}']}} for i in range(21)]}
            questions = [{'id': 'test/Q1', 'task': 'test', 'x': 'X1', 'y': 'Y1',
                'question': 'What does Alice like?', 'query_image_captions': ['A bird.'],
                'options': dict(A='OPTION_SENTINEL', B='b', C='c', D='d'),
                'gold_choice': 'A', 'gold_ids': ['GOLD_SENTINEL']}]
            protocol = dict(mode='simplemem', simplemem_backend='official_text', upstream=metadata,
                method_defaults=official.method_defaults(), embedding_model_path=str(embedding),
                seed=42, experiment_signature='integration-test-only', generation_batch_size=1,
                task_workers=1, tasks=['test'], memory_max_new_tokens=2048, max_new_tokens=16,
                context_memory_limit=5, preserve_empty_context=True, input_hashes={},
                reader_system_prompt='Choose one letter.', retrieval_count_policy='test: official retrieval then five',
                prepared_questions_sha256=baseline.hash_object(questions),
                observations_sha256=baseline.hash_object(observations))
            baseline.write_json(root / 'protocol.json', protocol)
            baseline.write_json(root / 'observations.json', observations)
            baseline.write_rows(root / 'questions.jsonl', questions)
            # Restore transport / encoder globals changed by production bootstrap.
            with patch.object(openai, 'OpenAI', openai.OpenAI), \
                 patch.object(sentence_transformers, 'SentenceTransformer', sentence_transformers.SentenceTransformer), \
                 patch.object(baseline, 'LocalMemoryQwen', ScriptedModel), \
                 patch.object(baseline, 'Memory', side_effect=AssertionError('legacy core must not be used')), \
                 patch.object(VectorStore, 'semantic_search', autospec=True,
                              side_effect=original_semantic_search) as semantic_search, \
                 patch.object(HybridRetriever, '_retrieve_with_intelligent_reflection', autospec=True,
                              side_effect=AssertionError('reflection must be disabled')) as reflection:
                baseline.run(args)
                reflection.assert_not_called()
                self.assertEqual(semantic_search.call_count, 2)
                self.assertTrue(all(call.kwargs['top_k'] == 10 for call in semantic_search.call_args_list))
            target = root / 'test'
            windows = json.loads((target / 'runtime/construction_windows.json').read_text())
            self.assertEqual([len(w['dialogue_ids']) for w in windows], [40, 4])
            self.assertEqual(windows[1]['dialogue_ids'], [39, 40, 41, 42])
            self.assertEqual(windows[1]['source_round_ids'], ['R19', 'R20'])
            self.assertTrue(all(w['parsed_successfully'] for w in windows))
            stats = json.loads((target / 'memory_stats.json').read_text())
            self.assertEqual(stats, {'stored_rounds': 21, 'stored_dialogues': 42, 'memory_entries': 12})
            import lancedb
            table = lancedb.connect(str(target / 'runtime/lancedb')).open_table('memory_entries')
            self.assertEqual(len(table.search('Zebrafinch').limit(20).to_list()), 6)
            self.assertEqual(len(table.search().where("array_has_any(persons, make_array('Alice'))").limit(20).to_list()), 12)
            row = baseline.read_rows(root / 'qa_responses.jsonl')[0]
            self.assertEqual(row['retrieval_detail']['backend'], 'official_text')
            self.assertEqual(len(set(row['retrieval_detail']['context_memory_ids'])), 5)
            self.assertEqual(row['retrieved_context'].count('[Context '), 5)
            calls = [json.loads(p.read_text())['request'] for p in (root / 'calls').glob('*.json')]
            memory_calls = [c for c in calls if c['kind'] == 'memory']
            payload = json.dumps(memory_calls)
            for forbidden in ('OPTION_SENTINEL', 'GOLD_SENTINEL'):
                self.assertNotIn(forbidden, payload)
            for expected in ('caption marker 20', 'requirement analyst', 'query generation specialist', 'query analysis assistant'):
                self.assertIn(expected, payload)
            for disabled in ('completeness evaluator', 'missing information query generator'):
                self.assertNotIn(disabled, payload)
            self.assertEqual(len(memory_calls), 5)  # Two windows, planning twice, query analysis once.
            resolved = json.loads((target / 'runtime/resolved_config.json').read_text())
            self.assertFalse(resolved['settings']['ENABLE_REFLECTION'])
            self.assertEqual(resolved['settings']['SEMANTIC_TOP_K'], 10)
            answer = next(c for c in calls if c['kind'] == 'answer')
            self.assertEqual(answer['messages'][1]['content'], 'Retrieved memory context:\n' + row['retrieved_context'])
            self.assertTrue((target / 'complete.json').exists())

    def test_failed_parse_is_not_treated_as_valid_empty_gating(self):
        official.bootstrap()
        from simplemem.core.memory_builder import MemoryBuilder
        from simplemem.core.models.memory_entry import Dialogue
        from simplemem.core.utils.llm_client import LLMClient
        for response, valid in [('not JSON', False), ('[]', True)]:
            with self.subTest(response=response), tempfile.TemporaryDirectory() as directory:
                client = object.__new__(LLMClient)
                client.chat_completion = lambda *a, **k: response
                builder = MemoryBuilder(client, SimpleNamespace(), window_size=40)
                memory = object.__new__(official.OfficialSimpleMemMemory)
                memory.workdir, memory.windows = Path(directory), []
                memory.dialogue_sources = {1: {'round_id': 'R1'}}
                memory.core = SimpleNamespace(memory_builder=builder)
                memory._audit_builder()
                dialogue = Dialogue(dialogue_id=1, speaker='Alice', content='Hello.')
                if valid:
                    self.assertEqual(builder._generate_memory_entries([dialogue]), [])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'failed to parse'):
                        builder._generate_memory_entries([dialogue])
                self.assertEqual(memory.windows[0]['parsed_successfully'], valid)

    def test_cap_deduplicates_whole_entries_and_preserves_only_true_empty(self):
        official.bootstrap()
        from simplemem.core.models.memory_entry import MemoryEntry
        from simplemem.core.answer_generator import AnswerGenerator
        entries = [MemoryEntry(entry_id=str(i), lossless_restatement='Complete content ' + str(i),
                               keywords=[]) for i in range(6)]
        memory = object.__new__(official.OfficialSimpleMemMemory)
        memory.finalized = True
        memory.protocol = {'context_memory_limit': 5, 'preserve_empty_context': True}
        memory.core = SimpleNamespace(hybrid_retriever=SimpleNamespace(retrieve=lambda q: [entries[0], *entries]),
                                     answer_generator=AnswerGenerator(None))
        context, detail = memory.recall('query')
        self.assertEqual(detail['context_memory_ids'], ['0', '1', '2', '3', '4'])
        self.assertNotIn('Complete content 5', context)
        memory.core.hybrid_retriever.retrieve = lambda q: entries[:2]
        with self.assertRaisesRegex(ValueError, 'retrieved only 2'):
            memory.recall('query')
        memory.core.hybrid_retriever.retrieve = lambda q: []
        self.assertEqual(memory.recall('query')[0], '')
        self.assertTrue(memory.recall('query')[1]['source_empty_preserved'])
        memory.protocol['preserve_empty_context'] = False
        with self.assertRaisesRegex(ValueError, 'retrieved only 0'):
            memory.recall('query')


if __name__ == '__main__':
    unittest.main()
