"""Pinned official SimpleMem TEXT core, adapted to the controlled Qwen pipeline.

The upstream builder/retriever stay unmodified. Transport, input provenance,
final-context bounds and post-ingest FTS refresh belong to this adapter.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

from scripts.memory.reflexion_memory import ROOT, write_json

RUNTIME = ROOT / '.vendor/simplemem_official_runtime'
SOURCE_LOCK = Path(__file__).with_name('simplemem_official_sources.json')
REQUIREMENTS = ROOT / 'scripts/requirements-simplemem-official.txt'
BACKEND = 'official_text'


def bootstrap():
    if not (RUNTIME / 'simplemem/text/system.py').is_file():
        raise RuntimeError('Install official SimpleMem first: python3 -m pip install --no-deps '
                           '--target .vendor/simplemem_official_runtime '
                           '-r scripts/requirements-simplemem-official.txt')
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))


def verify_installation():
    bootstrap()
    from packaging.requirements import Requirement
    for line in REQUIREMENTS.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        if requirement.specifier and not requirement.specifier.contains(importlib.metadata.version(requirement.name)):
            raise ValueError('incompatible SimpleMem dependency; reinstall ' + str(requirement))
    metadata = json.loads(SOURCE_LOCK.read_text())
    for name, expected in metadata['git_blob_hashes'].items():
        content = (RUNTIME / name).read_bytes()
        blob = b'blob ' + str(len(content)).encode() + b'\0' + content
        if hashlib.sha1(blob).hexdigest() != expected:
            raise ValueError('official SimpleMem source differs from pinned commit: ' + name)
    import simplemem
    if Path(simplemem.__file__).resolve() != RUNTIME / 'simplemem/__init__.py':
        raise ValueError('a different SimpleMem installation is already imported')
    if simplemem.__version__ != metadata['version']:
        raise ValueError('unexpected SimpleMem version')
    metadata['runtime_versions'] = {name: importlib.metadata.version(name) for name in (
        'simplemem', 'lancedb', 'pylance', 'tantivy', 'pyarrow', 'dateparser', 'pydantic',
        'numpy', 'openai', 'sentence-transformers', 'torch')}
    return metadata


def source_files():
    metadata = json.loads(SOURCE_LOCK.read_text())
    return [Path(__file__), Path(__file__).with_name('simplemem_fusion.py'), SOURCE_LOCK, REQUIREMENTS,
            *[RUNTIME / name for name in metadata['git_blob_hashes']]]


def method_defaults(semantic_top_k=10, enable_reflection=False):
    bootstrap()
    from simplemem.core.settings import _DEFAULTS
    # Pin settings explicitly so a working-directory config.py / env cannot
    # silently change this comparison. Use sequential windows with prior context.
    if semantic_top_k < 1:
        raise ValueError('semantic_top_k must be positive')
    return dict(_DEFAULTS, ENABLE_PARALLEL_PROCESSING=False,
                ENABLE_PARALLEL_RETRIEVAL=False, USE_STREAMING=False,
                SEMANTIC_TOP_K=semantic_top_k, ENABLE_REFLECTION=enable_reflection)


def configure_runtime(broker, protocol):
    bootstrap()
    from simplemem.core.settings import settings
    for key, value in protocol['method_defaults'].items():
        setattr(settings, key, value)
    settings.EMBEDDING_MODEL = protocol['embedding_model_path']
    settings.LLM_MODEL = 'local-qwen'
    settings.OPENAI_API_KEY = 'local-only'
    settings.OPENAI_BASE_URL = 'http://127.0.0.1:1/v1'
    from simplemem.core.utils import llm_client
    from scripts.memory.text_memory_transport import client_factory
    llm_client.OpenAI = client_factory(broker)
    # Import before task threads; weights are shared by install_runtime().
    from simplemem.text.system import SimpleMemSystem  # noqa: F401


class OfficialSimpleMemMemory:
    def __init__(self, protocol, workdir):
        from simplemem import create
        self.protocol, self.workdir = protocol, workdir
        self.stats = {'stored_rounds': 0, 'stored_dialogues': 0, 'memory_entries': 0}
        self.dialogue_sources, self.windows = {}, []
        self.finalized = False
        self.core = create(mode='text', api_key='local-only', model='local-qwen',
                           base_url='http://127.0.0.1:1/v1', db_path=str(workdir / 'lancedb'),
                           table_name='memory_entries', clear_db=False,
                           use_streaming=False, enable_thinking=False)
        write_json(workdir / 'resolved_config.json', {
            'backend': BACKEND, 'settings': protocol['method_defaults'],
            'embedding_model_path': protocol['embedding_model_path'],
            'embedding_dimension': self.core.embedding_model.dimension,
            'llm_model': 'local-qwen', 'db_path': str(workdir / 'lancedb')})
        self._audit_builder()
        self._guard_query_analysis()
        if protocol.get('context_fusion', 'ordered') == 'channel_rrf':
            self._trace_retrieval_channels()
        elif protocol.get('context_fusion', 'ordered') != 'ordered':
            raise ValueError('unsupported SimpleMem context fusion')

    def _trace_retrieval_channels(self):
        """Observe real upstream searches without changing planning or reflection."""
        retriever = self.core.hybrid_retriever
        self.search_batches, self.retrieval_phase = [], 'initial'
        semantic = retriever._semantic_search
        keyword = retriever._keyword_search
        structured = retriever._structured_search
        reflection = retriever._retrieve_with_intelligent_reflection

        def record(channel, query, top_k, result):
            self.search_batches.append({'channel': channel, 'phase': self.retrieval_phase,
                'query': query, 'top_k': top_k, 'result_ids': [e.entry_id for e in result]})
            return result

        def semantic_search(query):
            return record('semantic', query, retriever.semantic_top_k, semantic(query))

        def keyword_search(query, analysis):
            return record('keyword', {'query': query, 'analysis': analysis},
                          retriever.keyword_top_k, keyword(query, analysis))

        def structured_search(analysis):
            return record('structured', analysis, retriever.structured_top_k, structured(analysis))

        def reflect(query, entries, plan):
            previous = self.retrieval_phase
            self.retrieval_phase = 'reflection'
            try:
                return reflection(query, entries, plan)
            finally:
                self.retrieval_phase = previous

        retriever._semantic_search = semantic_search
        retriever._keyword_search = keyword_search
        retriever._structured_search = structured_search
        retriever._retrieve_with_intelligent_reflection = reflect

    def _guard_query_analysis(self):
        retriever = self.core.hybrid_retriever
        original_analyze = retriever._analyze_query

        def analyze(query):
            result = original_analyze(query)
            if isinstance(result, dict):
                return result
            # Upstream extract_json can recover an inner keywords array from
            # a malformed object. Request a valid object before dict consumers
            # run; keep successful upstream analyses exactly as returned.
            messages = [
                {'role': 'system', 'content': 'You are a query analysis assistant. You must output valid JSON format.'},
                {'role': 'user', 'content': 'The previous query analysis did not parse as a JSON object. '
                    'Analyze this query and return exactly one valid JSON object, without Markdown. '
                    'Use these fields: keywords, persons, entities (arrays of strings), '
                    'time_expression and location (each a single string or null). '
                    'Do not wrap the object in an array. Do not put multiple values after one key.\n\nQuery: ' + query}]
            audit_path = self.workdir / 'query_analysis_repairs.json'
            repairs = json.loads(audit_path.read_text()) if audit_path.exists() else []
            for attempt in range(1, 4):
                response = self.core.llm_client.chat_completion(messages, temperature=0.1)
                try:
                    text = response.strip()
                    if text.startswith('```') and text.endswith('```'):
                        text = '\n'.join(text.splitlines()[1:-1])
                    repaired = json.loads(text)
                    if not isinstance(repaired, dict):
                        raise ValueError('query analysis must be a JSON object')
                    for key in ('keywords', 'persons', 'entities'):
                        if not isinstance(repaired.get(key), list) or not all(isinstance(v, str) for v in repaired[key]):
                            raise ValueError(f'{key} must be an array of strings')
                    for key in ('time_expression', 'location'):
                        if key not in repaired or (repaired[key] is not None and not isinstance(repaired[key], str)):
                            raise ValueError(f'{key} must be one string or null')
                except (ValueError, TypeError) as exc:
                    repairs.append({'query': query, 'attempt': attempt, 'original_type': type(result).__name__,
                                    'success': False, 'error': str(exc)})
                    write_json(audit_path, repairs)
                    # A changed prompt permits an actual retry rather than
                    # replaying the same invalid deterministic request cache.
                    messages += [{'role': 'assistant', 'content': response},
                                 {'role': 'user', 'content': f'Invalid query-analysis format: {exc}. Return the corrected JSON object only.'}]
                    continue
                repairs.append({'query': query, 'attempt': attempt, 'original_type': type(result).__name__,
                                'success': True, 'analysis': repaired})
                write_json(audit_path, repairs)
                return repaired
            raise RuntimeError('Official SimpleMem query analysis remained invalid after three format repairs')

        retriever._analyze_query = analyze

    def _audit_builder(self):
        builder = self.core.memory_builder
        original_parse = builder._parse_llm_response
        original_generate = builder._generate_memory_entries

        def parse(response, dialogue_ids):
            result = original_parse(response, dialogue_ids)
            self._parsed_window = True
            return result

        def generate(dialogues):
            self._parsed_window = False
            entries = original_generate(dialogues)
            ids = [d.dialogue_id for d in dialogues]
            self.windows.append({'dialogue_ids': ids,
                'source_round_ids': list(dict.fromkeys(self.dialogue_sources[i]['round_id'] for i in ids)),
                'entry_ids': [e.entry_id for e in entries], 'parsed_successfully': self._parsed_window})
            write_json(self.workdir / 'construction_windows.json', self.windows)
            # Upstream returns [] for both legitimate semantic gating and failed
            # parsing. Preserve valid empty windows, but surface failed builds.
            if not self._parsed_window:
                raise RuntimeError('Official SimpleMem failed to parse a construction window; inspect calls/ and construction_windows.json')
            return entries

        builder._parse_llm_response = parse
        builder._generate_memory_entries = generate

    def store(self, row):
        if self.finalized:
            raise RuntimeError('cannot ingest after SimpleMem finalization')
        from simplemem.core.models.memory_entry import Dialogue
        user = str(row.get('user') or '')
        captions = row.get('raw', {}).get('image_caption', [])
        if captions:
            user += '\nimage_caption: ' + '\n'.join(captions)
        dialogues = []
        for speaker, content in ((row['speaker'], user), ('assistant', row.get('assistant', ''))):
            if not content or not content.strip():
                continue
            dialogue_id = self.stats['stored_dialogues'] + 1
            dialogues.append(Dialogue(dialogue_id=dialogue_id, speaker=speaker,
                                      content=content, timestamp=row['timestamp'] or None))
            self.dialogue_sources[dialogue_id] = {'round_id': row['id'],
                'session_id': row['session_id'], 'speaker': speaker, 'timestamp': row['timestamp']}
            self.stats['stored_dialogues'] += 1
        # Explicit IDs avoid add_dialogue()'s processed_count-based IDs drifting
        # with overlapping or legitimately empty windows. This is a public API.
        self.core.add_dialogues(dialogues)
        self.stats['stored_rounds'] += 1

    def finalize(self):
        if self.finalized:
            return
        self.core.finalize()
        backend = self.core.vector_store.backend
        self.stats['memory_entries'] = backend.count()
        if self.stats['memory_entries']:
            # Upstream creates a Tantivy index only on the first insert. Rebuild
            # after all windows so lexical retrieval includes the entire history.
            # Call directly to propagate errors instead of silently losing FTS.
            backend.table.create_fts_index('lossless_restatement', use_tantivy=True,
                                           tokenizer_name='en_stem', replace=True)
            backend._fts_initialized = True
        self.finalized = True
        write_json(self.workdir / 'dialogue_sources.json', self.dialogue_sources)

    def recall(self, query):
        if not self.finalized:
            raise RuntimeError('finalize SimpleMem before retrieval')
        if self.protocol.get('context_fusion', 'ordered') == 'channel_rrf':
            self.search_batches.clear()
            self.retrieval_phase = 'initial'
        entries = self.core.hybrid_retriever.retrieve(query)
        candidates, fusion = entries, None
        if self.protocol.get('context_fusion', 'ordered') == 'channel_rrf':
            from scripts.memory.simplemem_fusion import fuse_candidates
            candidates, fusion = fuse_candidates(entries, self.search_batches,
                self.protocol.get('fusion_rank_constant', 60))
        selected, seen = [], set()
        limit = self.protocol.get('context_memory_limit')
        if limit is not None and limit < 1:
            raise ValueError('SimpleMem context limit must be positive')
        for entry in candidates:
            if entry.entry_id not in seen:
                seen.add(entry.entry_id)
                selected.append(entry)
        if limit is not None:
            selected = selected[:limit]
            if len(selected) != limit and not (self.protocol.get('preserve_empty_context') and not entries):
                raise ValueError(f'SimpleMem context requires {limit} distinct notes; retrieved only {len(selected)}')
        detail = {'backend': BACKEND, 'items': [dict(e.model_dump(), id=e.entry_id) for e in selected],
                  'original_item_ids': [e.entry_id for e in entries],
                  'context_memory_ids': [e.entry_id for e in selected],
                  'context_memory_count': len(selected), 'context_memory_limit': limit,
                  'source_empty_preserved': bool(self.protocol.get('preserve_empty_context') and not entries)}
        if fusion is not None:
            detail.update(search_batches=list(self.search_batches), fusion=fusion)
        return self.core.answer_generator._format_contexts(selected), detail

    def export(self, path):
        write_json(path / 'final_memory.json', [e.model_dump() for e in self.core.get_all_memories()])
        write_json(path / 'memory_stats.json', self.stats)
