"""MemEye Gen.Ag / MemoryOS / A-Mem / SM(T), with a controlled local reader.

Third-party memory cores are pinned, not reimplementations. Changes to transport,
query/answer protocol, and missing upstream configuration are explicit in every
result protocol. Gold labels never enter construction or retrieval.
"""
import ast
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from scripts.memory.reflexion_memory import ROOT, LocalQwen, sha, hash_object, read_rows, write_json, write_rows
from scripts.memory.text_memory_transport import Broker, client_factory, PerRowTemperatureSampler

VENDOR = ROOT / 'scripts/vendor_memeye_text'
METHODS = ('gen_agents', 'memoryos', 'a_mem', 'simplemem')
EMBEDDING = Path('/storage/hewenwen/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2')
_STATE_PATHS = threading.local()


def effective_method_defaults(method, top_k=None):
    """Keep the pinned upstream files intact; override A-Mem's primary retrieval K."""
    config = yaml.safe_load((VENDOR / (method + '.yaml')).read_text())
    if method == 'a_mem':
        config['retrieve_k'] = 5 if top_k is None else top_k
        if config['retrieve_k'] < 1:
            raise ValueError('A-Mem top-k must be positive')
    return config


def output_dir(args):
    if args.mode == 'simplemem' and getattr(args, 'simplemem_backend', None) == 'official_text':
        limit = getattr(args, 'simplemem_context_limit', None)
        return (args.output_root / 'simplemem_official_text' if limit is None else
                args.output_root / f'top{limit}' / f'simplemem_official_text_context{limit}')
    if args.mode == 'simplemem' and getattr(args, 'simplemem_context_limit', None) is not None:
        limit = args.simplemem_context_limit
        return args.output_root / f'top{limit}' / f'simplemem_context{limit}'
    if args.mode == 'a_mem':
        k = effective_method_defaults(args.mode, getattr(args, 'top_k', None))['retrieve_k']
        limit = getattr(args, 'a_mem_context_limit', None)
        name = args.mode if limit is None else f'a_mem_context{limit}'
        return args.output_root / f'top{k}' / name
    return args.output_root / args.mode


def task_graph_factory(graph_type, state_paths):
    """Bind the upstream default graph to the current task before it loads data."""
    def create(*args, **kwargs):
        kwargs['storage_path'] = str(state_paths.knowledge_graph_path)
        return graph_type(*args, **kwargs)
    return create


def verify_vendor():
    metadata = json.loads((VENDOR / 'UPSTREAM.json').read_text())
    for name, digest in metadata['files'].items():
        if sha(VENDOR / name) != digest:
            raise ValueError('pinned upstream source changed: ' + name)
    return metadata


def helpers(method):
    """Load selected original helper definitions without the benchmark router."""
    source = VENDOR / (method + '_adapter.py')
    nodes = ast.parse(source.read_text()).body
    wanted = [n for n in nodes if isinstance(n, ast.FunctionDef)]
    if method == 'gen_agents':
        cls = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'GAMethod')
        wanted += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_build_ga_config']
    if method == 'a_mem':
        wanted += [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'AMemAgent']
    import openai
    ns = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
              Path=Path, os=os, json=json, sys=sys, time=time, re=re,
              OpenAI=openai.OpenAI, MemoryBenchmarkDataset=Any,
              _LOCAL_ENV_PATH=Path('/nonexistent-local-env'))
    for name in ('APIConnectionError', 'APITimeoutError', 'InternalServerError', 'RateLimitError'):
        ns[name] = getattr(openai, name)
    # Official MemoryOS helper delegates caption extraction to dataset.py.
    ns['round_image_captions'] = lambda payload: payload['raw']['image_caption']
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(source), 'exec'), ns)
    return ns


def prepare(args):
    from scripts.memory.full_context_memory import captioned_histories
    official = args.mode == 'simplemem' and getattr(args, 'simplemem_backend', None) == 'official_text'
    if official:
        from scripts.memory import simplemem_official
    metadata = simplemem_official.verify_installation() if official else verify_vendor()
    questions = read_rows(args.benchmark_manifest)
    if args.task:
        questions = [q for q in questions if q['task'] == args.task]
    if args.limit_questions:
        questions = questions[:args.limit_questions]
    if not questions:
        raise ValueError('no selected questions')
    tasks = sorted({q['task'] for q in questions})
    natural = json.loads((args.natural_root / 'protocol.json').read_text())
    baseline = json.loads((args.benchmark_manifest.parent / 'protocol.json').read_text())
    _, sources = captioned_histories(args.natural_root, natural, set(tasks))
    captions = {r['id']: r['caption'] for r in read_rows(args.natural_root / 'captions.jsonl')}
    images = {p: r['id'] for r in read_rows(args.natural_root / 'images.jsonl') for p in r['source_paths']}
    observations = {}
    for task in tasks:
        source = next(ROOT / s['path'] for s in natural['source_files'] if Path(s['path']).name == task + '_Open.json')
        data = json.loads(source.read_text())
        profile = data.get('character_profile') or {}
        name = str(profile.get('name') or (profile.get('primary_user') if args.mode == 'gen_agents' else '') or '').strip()
        speaker = f'user ({name})' if name else 'user'
        observations[task] = []
        for session in data['multi_session_dialogues']:
            for turn in session['dialogues']:
                refs = turn.get('input_image') or []
                if isinstance(refs, str):
                    refs = [refs]
                observations[task].append({'id': turn['round'], 'session_id': session['session_id'],
                    'timestamp': str(session.get('date') or '').strip(), 'speaker': speaker,
                    'user': turn.get('user', ''), 'assistant': turn.get('assistant', ''),
                    'raw': {'input_image': refs, 'image_id': turn.get('image_id', []),
                            'image_caption': [captions[images[p]] for p in refs]}}) # A-Mem核心代码，会把对话内容和image caption作为观察
    sanitized = []
    for q in questions:
        row = {k: q[k] for k in ('id', 'task', 'x', 'y', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')}
        row['query_image_captions'] = [captions[sha(Path(p))] for p in q['query_image_paths']]
        sources += [Path(p) for p in q['query_image_paths']]
        sanitized.append(row)
    embedding_path = EMBEDDING / 'snapshots' / (EMBEDDING / 'refs/main').read_text().strip()
    config = (simplemem_official.method_defaults(
                  semantic_top_k=getattr(args, 'simplemem_semantic_top_k', 10),
                  enable_reflection=getattr(args, 'simplemem_reflection', False)) if official else
              effective_method_defaults(args.mode, getattr(args, 'top_k', None)))
    sources += [args.benchmark_manifest, args.benchmark_manifest.parent / 'protocol.json', Path(__file__),
                Path(__file__).with_name('text_memory_transport.py'), Path(__file__).with_name('reflexion_memory.py'),
                Path(__file__).with_name('full_context_memory.py')]
    sources += simplemem_official.source_files() if official else [VENDOR / 'UPSTREAM.json']
    sources += [p for p in embedding_path.rglob('*') if p.is_file()]
    device_map_path = getattr(args, 'memory_device_map', None)
    if device_map_path is not None:
        sources.append(device_map_path)
    protocol = {k: baseline[k] for k in ('reader_model', 'reader_model_path', 'reader_system_prompt',
                                       'seed', 'do_sample', 'dtype', 'max_new_tokens')}
    protocol.update(mode=args.mode, study='MemEye official memory core / controlled local Qwen MCQ reader',
        upstream=metadata, method_defaults=config, embedding_model_path=str(embedding_path),
        tasks=tasks, questions=len(questions), memory_rounds=sum(map(len, observations.values())),
        caption_model_path=natural['caption_model_path'], context_budget=args.context_budget,
        memory_max_new_tokens=args.memory_max_new_tokens, task_workers=args.memory_workers,
        max_padded_batch_tokens=48000,
        generation_batch_size=args.memory_batch_size, feedback_policy='none; labels used only after prediction for scoring',
        memory_sampling='upstream per-call temperature; full categorical sampling (top_p=1, no top_k); independent seed42 RNG per request',
        query_policy='question plus cached Qwen query-image captions; exclude options and gold from retrieval',
        answer_prompt_policy='same local reader system prompt and 16-token answer budget; assistant-role memory context; fixed presented options',
        query_image_policy='cached Qwen captions only, no native images',
        differences_from_paper=['Qwen for all LLM operations; greedy final answers, native memory-operation temperatures with per-request seed42',
            'cached local Qwen captions, one fixed MCQ option permutation',
            'controlled common MCQ reader replaces each method-specific answer prompt',
            'answer-free retrieval question, no MCQ options in retrieval',
            'JSON schema provided as text instructions, no server constrained decoding',
            'additional pinned SimpleMem config.py restores missing release file',
            'SimpleMem knowledge graph storage isolated per task (upstream default is a shared working-directory path)',
            'method internal max_tokens honored; unspecified memory outputs capped',
            'MiniLM encoder on CPU with shared read-only weights; per-task state isolated'],
        input_hashes={str(p.resolve()): sha(p) for p in sources},
        prepared_questions_sha256=hash_object(sanitized), observations_sha256=hash_object(observations))
    if args.mode == 'a_mem':
        upstream_k = yaml.safe_load((VENDOR / 'a_mem.yaml').read_text())['retrieve_k']
        protocol.update(retrieval_top_k=config['retrieve_k'], upstream_retrieval_top_k=upstream_k,
                        retrieval_count_policy='K primary nearest memories, followed by upstream link-neighbor expansion; final context is not capped to K')
        protocol['differences_from_paper'].append(
            f'A-Mem primary retrieval K={config["retrieve_k"]}; upstream K={upstream_k}; link expansion retained')
        limit = getattr(args, 'a_mem_context_limit', None)
        if limit is not None:
            if limit < 1:
                raise ValueError('A-Mem context limit must be positive')
            protocol.update(context_memory_limit=limit,
                retrieval_count_policy='K primary nearest memories with upstream interleaved link expansion; deduplicate by memory ID in first-occurrence order, then take exactly context_memory_limit complete notes; fail if insufficient')
            protocol['differences_from_paper'].append(
                f'A-Mem expanded answer context deduplicated and limited to exactly {limit} memory notes')
    if args.mode == 'simplemem' and getattr(args, 'simplemem_context_limit', None) is not None:
        limit = args.simplemem_context_limit
        if limit < 1:
            raise ValueError('SimpleMem context limit must be positive')
        protocol.update(context_memory_limit=limit,
            retrieval_count_policy='upstream adaptive retrieval and BM25 union; keep first N distinct memory IDs in returned order; require exactly N answer-context items')
        protocol['preserve_empty_context'] = getattr(args, 'simplemem_preserve_empty', False)
        protocol['differences_from_paper'].append(f'Final SimpleMem context limited to exactly {limit} distinct memory units')
    if official:
        fusion = getattr(args, 'simplemem_context_fusion', 'channel_rrf')
        protocol.update(simplemem_backend='official_text',
            context_fusion=fusion, fusion_rank_constant=60,
            study='Official SimpleMem text core / controlled local Qwen MCQ reader',
            construction_policy='ordered user and assistant Dialogue turns; captions appended to user text; '
                'explicit sequential dialogue IDs; 40-turn windows with 2-turn overlap; finalize before retrieval',
            retrieval_count_policy=f'official planning; semantic top-k={config["SEMANTIC_TOP_K"]} per query, '
                f'keyword top-k={config["KEYWORD_TOP_K"]}, structured top-k={config["STRUCTURED_TOP_K"]}; '
                f'reflection={config["ENABLE_REFLECTION"]}; ' +
                ('deduplicate by entry_id; two-stage equal-channel RRF (rank constant 60); '
                 if fusion == 'channel_rrf' else 'deduplicate by entry_id in returned order; ') + (
                'take exactly N complete entries, optionally preserve originally empty retrievals'
                if 'context_memory_limit' in protocol else 'retain all returned entries'))
        protocol['differences_from_paper'] = [
            'local Qwen for memory operations; MiniLM-L6-v2 CPU embeddings instead of default Qwen3 embeddings',
            'cached Qwen image captions appended to dialogue; no native images',
            'controlled common MCQ reader replaces upstream final answer generation; options and gold excluded from retrieval',
            'non-streaming local transport; upstream call temperatures, per-request seed42, configured memory output token cap',
            'sequential construction and retrieval within each task; independent task databases',
            'explicit settings from pinned defaults, ignoring external config.py and environment overrides',
            f'controlled comparison: semantic top-k={config["SEMANTIC_TOP_K"]} per query (upstream 25); '
                f'reflection={config["ENABLE_REFLECTION"]} (upstream enabled); query planning retained',
            'Tantivy full-text index rebuilt after finalize to include all construction windows',
            'upstream parse retries retained; abort after exhausted construction parse retries instead of treating failure as empty gating']
        if fusion == 'channel_rrf':
            protocol['differences_from_paper'].append(
                'Adapter fuses distinct queries within each channel with RRF, then equally fuses semantic, keyword and '
                'structured rankings before applying the final context limit; reflection contributes semantic candidates')
        if 'context_memory_limit' in protocol:
            protocol['differences_from_paper'].append(
                f'Final answer context limited to exactly {protocol["context_memory_limit"]} distinct whole entries; '
                f'preserve_empty={protocol["preserve_empty_context"]}')
    if device_map_path is not None:
        device_map = json.loads(device_map_path.read_text())
        if not isinstance(device_map, dict) or not device_map or any(
                not isinstance(name, str) or not (device == 'cpu' or
                (type(device) is int and device >= 0)) for name, device in device_map.items()):
            raise ValueError('memory device map must map module names to GPU indices or cpu')
        protocol['reader_device_map'] = device_map
    protocol['experiment_signature'] = hash_object(protocol)
    target = output_dir(args)
    old = target / 'protocol.json'
    if old.exists() and json.loads(old.read_text()) != protocol:
        raise ValueError('existing protocol differs; use a new output root')
    target.mkdir(parents=True, exist_ok=True)
    write_json(old, protocol)
    write_json(target / 'observations.json', observations)
    write_rows(target / 'questions.jsonl', sanitized)
    snapshots = {}
    for source, digest in protocol['input_hashes'].items():
        if Path(source).suffix == '.py':
            snapshot = target / 'source_snapshot' / (digest[:12] + '_' + Path(source).name)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            snapshots[source] = {'snapshot': str(snapshot.resolve()), 'sha256': digest}
    write_json(target / 'source_snapshot/manifest.json', snapshots)
    print(f'{args.mode}: prepared {len(questions)} questions / {protocol["memory_rounds"]} rounds', flush=True)


def install_runtime(broker, protocol):
    """Replace transport and device placement only; algorithms remain upstream."""
    sys.path.insert(0, str(ROOT / '.vendor'))
    os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      LITELLM_LOCAL_MODEL_COST_MAP='True', ANONYMIZED_TELEMETRY='False')
    import openai
    openai.OpenAI = client_factory(broker)
    import sentence_transformers
    real_encoder = sentence_transformers.SentenceTransformer
    encoder = real_encoder(protocol['embedding_model_path'], device='cpu', local_files_only=True)
    lock = threading.RLock()
    original_encode = encoder.encode
    def encode(*args, **kwargs):
        with lock:
            return original_encode(*args, **kwargs)
    encoder.encode = encode
    def shared_encoder(model_name, *args, **kwargs):
        if str(model_name) not in ('all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2', protocol['embedding_model_path']):
            raise ValueError('unexpected embedding model: ' + str(model_name))
        return encoder
    sentence_transformers.SentenceTransformer = shared_encoder
    if protocol.get('simplemem_backend') == 'official_text':
        from scripts.memory import simplemem_official
        simplemem_official.configure_runtime(broker, protocol)
        return encoder
    from langchain_core.prompts import PromptTemplate
    package = types.ModuleType('langchain')
    prompts = types.ModuleType('langchain.prompts')
    prompts.PromptTemplate = PromptTemplate
    package.prompts = prompts
    sys.modules['langchain'] = package
    sys.modules['langchain.prompts'] = prompts
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(VENDOR / 'amem_core'))
    # Pre-import once on the main thread (avoid lazy-import races).
    if protocol['mode'] == 'gen_agents':
        import memengine.function.Retrieval as retrieval
        class MiniLM:
            def __init__(self, config):
                self.device = 'cpu'
            def reset(self):
                pass
            def __call__(self, text, return_type='numpy'):
                result = encoder.encode([str(text)], normalize_embeddings=True)
                return result if return_type == 'numpy' else broker.model.torch.from_numpy(np.asarray(result, dtype=np.float32))
        retrieval.OpenAIEmbeddingEncoder = MiniLM
        # Same numeric result, without evaluating arbitrary generated Python.
        from memengine.function.Judge import LLMJudge
        def safe_scale(self, value):
            number = float(ast.literal_eval(value.strip()))
            if not 1 <= number <= 10:
                raise ValueError('importance outside 1..10')
            return number / self.config.post_scale
        LLMJudge.__post_scale__ = safe_scale
    elif protocol['mode'] == 'a_mem':
        # Python 3.14 removed this unused legacy AST import in upstream A-Mem.
        if not hasattr(ast, 'Str'):
            ast.Str = ast.Constant
        import memory_layer
    elif protocol['mode'] == 'memoryos':
        import memoryos_core
    else:
        import omni_memory
        import omni_memory.orchestrator as orchestrator
        orchestrator.KnowledgeGraph = task_graph_factory(orchestrator.KnowledgeGraph, _STATE_PATHS)
    return encoder


def method_config(protocol):
    cfg = dict(protocol['method_defaults'])
    cfg.update(llm_model='local-qwen', llm_api_key='local-only', llm_base_url='http://127.0.0.1:1/v1',
               base_url='http://127.0.0.1:1/v1', embedding_model=protocol['embedding_model_path'])
    model = dict(model='local-qwen', api_key='local-only', base_url='http://127.0.0.1:1/v1', provider='openai_api')
    return cfg, model


def bounded_a_mem_context(system, query, primary_k, limit):
    """Cap whole notes after upstream-order expansion, never split rendered text."""
    if limit < 1:
        raise ValueError('A-Mem context limit must be positive')
    memories = list(system.memories.values())
    indices = list(system.retriever.search(query, primary_k)) if memories else []

    def expanded():
        for i in indices:
            note = memories[i]
            yield note
            j = 0
            for neighbor in note.links:
                linked = (system.memories.get(neighbor) if isinstance(neighbor, str)
                          else memories[neighbor] if isinstance(neighbor, int) and neighbor < len(memories)
                          else None)
                if linked is None:
                    continue
                yield linked
                # Preserve upstream's per-primary neighbor bound and order.
                if j >= primary_k:
                    break
                j += 1

    selected, seen = [], set()
    for note in expanded():
        if note.id in seen:
            continue
        seen.add(note.id)
        selected.append(note)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f'A-Mem context requires {limit} distinct notes; retrieved only {len(selected)}')
    context = ''.join('talk start time:' + note.timestamp + 'memory content: ' + note.content
                      + 'memory context: ' + note.context + 'memory keywords: ' + str(note.keywords)
                      + 'memory tags: ' + str(note.tags) + '\n' for note in selected)
    return context, {'primary_memory_ids': [memories[i].id for i in indices],
                     'context_memory_ids': [note.id for note in selected],
                     'context_memory_count': len(selected), 'context_memory_limit': limit}


def bounded_simplemem_result(result, limit, allow_empty=False):
    """Keep complete items in original vector/BM25 order, deduplicated by MAU ID."""
    if limit < 1:
        raise ValueError('SimpleMem context limit must be positive')
    selected, seen = [], set()
    for item in result.items:
        if item['id'] in seen:
            continue
        seen.add(item['id'])
        selected.append(item)
        if len(selected) == limit:
            break
    if len(selected) != limit and not (allow_empty and not result.items):
        raise ValueError(f'SimpleMem context requires {limit} distinct notes; retrieved only {len(selected)}')
    bounded = replace(result, items=selected,
                      expansion_candidates=[mid for mid in result.expansion_candidates if mid in seen])
    detail = bounded.to_dict()
    detail.update(context_memory_limit=limit, context_memory_count=len(selected),
                  context_memory_ids=[item['id'] for item in selected],
                  original_item_ids=[item['id'] for item in result.items],
                  source_empty_preserved=bool(allow_empty and not result.items))
    return bounded, detail


class Memory:
    def __init__(self, method, protocol, workdir):
        self.method, self.protocol = method, protocol
        self.helper = helpers(method)
        cfg, model = method_config(protocol)
        self.stats = {'stored_rounds': 0}
        if method == 'gen_agents':
            from memengine.config.Config import ArgConfig
            from memengine.memory.GAMemory import GAMemory
            config = self.helper['_build_ga_config'](types.SimpleNamespace(_empty_memory_text='No relevant memory retrieved.'), cfg, model)
            self.core = GAMemory(types.SimpleNamespace(args=ArgConfig(config)))
            # Upstream assumes all score tensors share the encoder device.
            for retriever in (self.core.recall_op.text_retrieval, self.core.recall_op.time_retrieval,
                              self.core.recall_op.importance_retrieval):
                retriever.device = 'cpu'
        elif method == 'memoryos':
            from memoryos_core import Memoryos
            keys = ('short_term_capacity', 'mid_term_capacity', 'long_term_knowledge_capacity',
                    'retrieval_queue_capacity', 'mid_term_heat_threshold', 'mid_term_similarity_threshold')
            self.core = Memoryos(user_id='memorybench_user', assistant_id='memorybench_assistant',
                openai_api_key='local-only', openai_base_url=cfg['llm_base_url'],
                data_storage_path=str(workdir), llm_model='local-qwen',
                embedding_model_name=cfg['embedding_model'], embedding_model_kwargs={'device': 'cpu'},
                **{k: cfg[k] for k in keys})
        elif method == 'a_mem':
            self.core = self.helper['AMemAgent'](cfg, model)
        else:
            from omni_memory import OmniMemoryConfig, OmniMemoryOrchestrator
            _STATE_PATHS.knowledge_graph_path = workdir / 'knowledge_graph'
            config = OmniMemoryConfig.create_default()
            config.set_unified_model('local-qwen')
            config.llm.api_key = 'local-only'
            config.llm.api_base_url = cfg['base_url']
            config.llm.temperature = cfg['temperature']
            config.llm.max_tokens = cfg['max_tokens']
            config.embedding.model_name = cfg['embedding_model']
            config.embedding.embedding_dim = cfg['embedding_dim']
            config.entropy_trigger.visual_encoder = 'none'
            config.entropy_trigger.enable_visual_trigger = False
            config.retrieval.default_top_k = cfg['retrieve_k']
            config.enable_self_evolution = False
            self.core = OmniMemoryOrchestrator(config=config, data_dir=str(workdir))
            self.core.query_processor._modality_keywords = {}
            write_json(workdir / 'resolved_config.json', config.to_dict())

    def store(self, row):
        method, h = self.method, self.helper
        if method == 'memoryos':
            text = h['_round_user_with_captions'](row)
            self.core.add_memory(text, row['assistant'], timestamp=row['timestamp'])
        else:
            text = h['build_a_mem_note_text' if method == 'a_mem' else '_build_round_text'](row, row['speaker'], 'assistant')
            if method == 'gen_agents':
                self.core.store({'text': text, 'timestamp': row['timestamp'], 'dialogue_id': row['id'], 'source': False})
                reflector = self.core.reflect_op.reflector
                if reflector.get_current_accmulated_importance() >= reflector.get_reflection_threshold():
                    self.core.manage('reflect')
            elif method == 'a_mem':
                self.core.memory_system.add_note(text, time=row['timestamp'], id=row['id'])
            else:
                tags = h['_build_round_tags'](row['id'], row['session_id'], row['timestamp'], row['raw'])
                result = self.core.add_text(text, tags=tags, force=True) # simplemem语义压缩 如果force=false是文本冗余过滤
                if not result.success or result.mau is None:
                    raise RuntimeError('SM(T) rejected input: ' + str(result.error))
        self.stats['stored_rounds'] += 1

    def recall(self, query):
        if self.method == 'gen_agents':
            context = self.core.recall({'text': query, 'timestamp': 'query'})
            self.stats['total_memories_with_reflections'] = self.core.storage.get_element_number()
            return context, {}
        if self.method == 'memoryos':
            result = self.core.retriever.retrieve_context(user_query=query, user_id=self.core.user_id)
            return self.helper['_format_retrieval_context'](result), result
        if self.method == 'a_mem':
            keywords = self.core.generate_query(query)
            limit = self.protocol.get('context_memory_limit')
            if limit is not None:
                context, detail = bounded_a_mem_context(
                    self.core.memory_system, keywords, self.core.retrieve_k, limit)
                return context, dict(detail, generated_keywords=keywords, primary_top_k=self.core.retrieve_k)
            return self.core.retrieve_memory(keywords), {'generated_keywords': keywords,
                                                        'primary_top_k': self.core.retrieve_k}
        # This is answer()'s default retrieval path with unchanged strategy,
        # BM25 union, graph context, and lazy expansion. Final reader is shared.
        result = self.core.query(query, top_k=self.protocol['method_defaults']['retrieve_k'], auto_expand=True)
        if self.protocol.get('context_memory_limit') is not None:
            result, detail = bounded_simplemem_result(result, self.protocol['context_memory_limit'],
                                                      self.protocol.get('preserve_empty_context', False))
            return self.core.retriever.format_for_llm(result, include_instructions=False), detail
        return self.core.retriever.format_for_llm(result, include_instructions=False), result.to_dict()

    def export(self, path):
        if self.method == 'gen_agents':
            write_json(path / 'final_memory.json', [self.core.storage.get_memory_element_by_mid(i)
                for i in range(self.core.storage.get_element_number())])
        elif self.method == 'a_mem':
            write_json(path / 'final_memory.json', {k: vars(v) for k, v in self.core.memory_system.memories.items()})
        elif self.method == 'simplemem':
            self.core.close()
        else:
            self.core.client.shutdown()
        write_json(path / 'memory_stats.json', self.stats)


def retrieval_query(q):
    query = q['question']
    if q['query_image_captions']:
        query += "\nquestion's image:\nimage_caption: " + ' '.join(q['query_image_captions'])
    return query


def reader_messages(q, context, protocol):
    return [{'role': 'system', 'content': protocol['reader_system_prompt']},
            {'role': 'assistant', 'content': 'Retrieved memory context:\n' + context},
            {'role': 'user', 'content': retrieval_query(q) + '\n\nOptions:\n' +
                '\n'.join(k + '. ' + q['options'][k] for k in 'ABCD') + '\n\nAnswer with one letter only.'}]


class LocalMemoryQwen(LocalQwen):
    """Original local reader plus native-temperature memory generation.

An independent RNG per row prevents task batching from changing the random
stream. Sampling is categorical over temperature-scaled model probabilities;
the selected token is supplied to generate() as the only finite logit.
"""
    def generate(self, messages, max_new_tokens, sampling_temperature=0.0):
        if sampling_temperature <= 0:
            return super().generate(messages, max_new_tokens)
        torch = self.torch
        rendered = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        inputs = self.processor(text=rendered, padding=True, return_tensors='pt').to(
            getattr(self, 'input_device', self.model.device))
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        if max(lengths) + max_new_tokens > self.protocol['context_budget']:
            raise ValueError('memory input exceeds context budget')
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                logits_processor=[PerRowTemperatureSampler(sampling_temperature, self.protocol['seed'])],
                pad_token_id=self.processor.tokenizer.pad_token_id)
        tails = output[:, inputs['input_ids'].shape[1]:]
        texts = self.processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        eos = self.processor.tokenizer.eos_token_id
        result = []
        for raw, tokens, length in zip(texts, tails.tolist(), lengths):
            emitted = next((i + 1 for i, token in enumerate(tokens) if token == eos), len(tokens))
            result.append({'raw_response': raw.strip(), 'input_tokens': int(length),
                           'generated_tokens': emitted, 'hit_token_limit': eos not in tokens and emitted >= max_new_tokens})
        return result


def run(args):
    root = output_dir(args)
    protocol = json.loads((root / 'protocol.json').read_text())
    official = protocol.get('simplemem_backend') == 'official_text'
    if official:
        from scripts.memory import simplemem_official
        if simplemem_official.verify_installation() != protocol['upstream']:
            raise ValueError('official SimpleMem installation changed; use a new output root')
    else:
        verify_vendor()
    for p, digest in protocol['input_hashes'].items():
        if sha(Path(p)) != digest:
            raise ValueError('input changed: ' + p)
    questions = read_rows(root / 'questions.jsonl')
    observations = json.loads((root / 'observations.json').read_text())
    if hash_object(questions) != protocol['prepared_questions_sha256'] or hash_object(observations) != protocol['observations_sha256']:
        raise ValueError('prepared data changed')
    pending = [t for t in protocol['tasks'] if not (root / t / 'complete.json').exists()]
    if not pending:
        analyze(args)
        return
    model = LocalMemoryQwen(protocol)
    write_json(root / 'generation_environment.json', model.environment)
    broker = Broker(model, root, protocol, protocol['generation_batch_size'])
    encoder = install_runtime(broker, protocol)
    def task_run(task):
        broker.local.owner = task
        target = root / task
        workdir = target / 'runtime'
        # Rebuild interrupted task from audited request cache; never ingest onto stale state.
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        memory = (simplemem_official.OfficialSimpleMemMemory(protocol, workdir) if official else
                  Memory(args.mode, protocol, workdir))
        for i, observation in enumerate(observations[task], 1):
            memory.store(observation)
            write_json(target / 'progress.json', {'phase': 'ingest', 'done': i, 'total': len(observations[task])})
            if i % 10 == 0:
                print(f'{args.mode} {task}: stored {i}/{len(observations[task])}', flush=True)
        if official:
            write_json(target / 'progress.json', {'phase': 'finalize'})
            memory.finalize()
            memory.export(target)
        rows = []
        for q in [q for q in questions if q['task'] == task]:
            query = retrieval_query(q)
            context, detail = memory.recall(query)
            context = str(context)
            messages = reader_messages(q, context, protocol)
            output = broker.request(messages, protocol['max_new_tokens'], kind='answer', owner=q['id'])
            raw = output['raw_response'].strip()
            row = {**q, **output, 'predicted_choice': raw if raw in 'ABCD' and len(raw) == 1 else None,
                   'retrieval_query': query, 'retrieved_context': context, 'retrieval_detail': detail,
                   'reader_messages_sha256': hash_object(messages), 'experiment_signature': protocol['experiment_signature']}
            rows.append(row)
            write_rows(target / 'responses.jsonl', rows)
            write_json(target / 'progress.json', {'phase': 'answer', 'done': len(rows),
                'total': sum(x['task'] == task for x in questions)})
        memory.export(target)
        write_json(target / 'complete.json', {'experiment_signature': protocol['experiment_signature'],
            'responses_sha256': sha(target / 'responses.jsonl'), 'questions': len(rows)})
        print(f'{args.mode} {task}: complete {sum(r["predicted_choice"] == r["gold_choice"] for r in rows)}/{len(rows)}', flush=True)
    errors = []
    try:
        with ThreadPoolExecutor(max_workers=protocol['task_workers']) as pool:
            futures = {pool.submit(task_run, t): t for t in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    import traceback
                    detail = traceback.format_exc()
                    errors.append({'task': futures[future], 'error': str(exc), 'traceback': detail})
                    write_json(root / 'run_errors.json', errors)
                    print(detail, flush=True)
    finally:
        broker.close()
    analyze(args)
    if errors:
        raise RuntimeError(f'{len(errors)} task(s) failed; inspect run_errors.json')


def analyze(args):
    root = output_dir(args)
    protocol = json.loads((root / 'protocol.json').read_text())
    questions = read_rows(root / 'questions.jsonl')
    rows, complete = [], []
    for task in protocol['tasks']:
        target = root / task
        if (target / 'complete.json').exists():
            checkpoint = json.loads((target / 'complete.json').read_text())
            if checkpoint['experiment_signature'] != protocol['experiment_signature'] or checkpoint['responses_sha256'] != sha(target / 'responses.jsonl'):
                raise ValueError('completed task checkpoint changed')
            rows.extend(read_rows(target / 'responses.jsonl'))
            complete.append(task)
    byid = {r['id']: r for r in rows}
    if len(byid) != len(rows):
        raise ValueError('duplicate question responses')
    rows = [byid[q['id']] for q in questions if q['id'] in byid]
    for row in rows:
        row['correct'] = row['predicted_choice'] == row['gold_choice']
    def metric(subset):
        n = len(subset)
        return {'questions': n, 'correct': sum(r['correct'] for r in subset),
                'accuracy': sum(r['correct'] for r in subset)/n if n else None}
    cells = {f'{y}/{x}': metric([r for r in rows if r['x'] == x and r['y'] == y]) for y in ('Y1','Y2','Y3') for x in ('X1','X2','X3','X4')}
    calls = [json.loads(p.read_text()) for p in (root / 'calls').glob('*.json')]
    summary = {'mode': args.mode, 'status': 'complete' if len(rows) == len(questions) else 'partial',
        **metric(rows), 'expected_questions': len(questions), 'complete_tasks': complete,
        'format_errors': sum(r['predicted_choice'] is None for r in rows),
        'answer_token_limit_hits': sum(r['hit_token_limit'] for r in rows),
        'memory_token_limit_hits': sum(c['output']['hit_token_limit'] for c in calls if c['request']['kind'] == 'memory'),
        'unique_llm_calls': len(calls), 'by_task': {t: metric([r for r in rows if r['task'] == t]) for t in protocol['tasks']},
        'by_cell': cells, 'macro_12_cells': float(np.mean([c['accuracy'] for c in cells.values()])) if all(c['questions'] for c in cells.values()) else None,
        'experiment_signature': protocol['experiment_signature']}
    if protocol.get('simplemem_backend'):
        summary['simplemem_backend'] = protocol['simplemem_backend']
    if args.mode == 'a_mem':
        summary.update(retrieval_top_k=protocol['retrieval_top_k'],
                       retrieval_count_policy=protocol['retrieval_count_policy'])
        if 'context_memory_limit' in protocol:
            limit = protocol['context_memory_limit']
            for row in rows:
                ids = row['retrieval_detail']['context_memory_ids']
                if len(ids) != limit or len(set(ids)) != limit:
                    raise ValueError('A-Mem response violates the distinct context memory limit')
            summary['context_memory_limit'] = limit
    if args.mode == 'simplemem' and 'context_memory_limit' in protocol:
        limit = protocol['context_memory_limit']
        checked = [r for r in rows if not (protocol.get('preserve_empty_context') and
                                           r['retrieval_detail'].get('source_empty_preserved'))]
        if any(len(r['retrieval_detail']['items']) != limit or
               len({i['id'] for i in r['retrieval_detail']['items']}) != limit for r in checked):
            raise ValueError('SimpleMem response violates the distinct context memory limit')
        summary.update(context_memory_limit=limit, preserved_empty_contexts=len(rows)-len(checked),
                       retrieval_count_policy=protocol['retrieval_count_policy'])
    write_rows(root / 'qa_responses.jsonl', rows)
    write_json(root / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('mode','status','questions','correct','accuracy','format_errors','memory_token_limit_hits')}), flush=True)
