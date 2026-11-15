"""Isolated pinned Omni-SimpleMem backend. Never imports the text runtime.

The upstream sources are byte-for-byte pinned. This adapter supplies local
models, source metadata, isolated graph paths, channel RRF and bounded evidence.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / '.vendor/simplemem_omni_030_runtime'
LOCK = ROOT / 'scripts/memory/simplemem_omni_sources.json'
_graph_path = ContextVar('omni_graph_path')


def verify_installation():
    manifest = json.loads(LOCK.read_text())
    for name, expected in manifest['files'].items():
        data = (RUNTIME / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise ValueError('Pinned Omni source changed: ' + name)
    return manifest


def activate():
    verify_installation()
    loaded = sys.modules.get('simplemem')
    if loaded and not Path(loaded.__file__).resolve().is_relative_to(RUNTIME):
        raise RuntimeError('Omni needs its own process; another SimpleMem runtime is loaded')
    sys.path.insert(0, str(RUNTIME))
    import simplemem
    assert simplemem.__version__ == '0.3.0'
    return simplemem


def fuse_channels(channels, limit=5, constant=60):
    """One vote per ID per channel; deterministic first-seen tie breaking."""
    scores, order = {}, {}
    for entries in channels.values():
        seen = set()
        for rank, item in enumerate(entries, 1):
            identity = item['id']
            if identity in seen:
                continue
            seen.add(identity)
            order.setdefault(identity, len(order))
            scores[identity] = scores.get(identity, 0.) + 1. / (constant + rank)
    ranked = sorted(scores, key=lambda key: (-scores[key], order[key]))
    return ranked[:limit], scores


class SharedCLIP:
    """Load once; use upstream CLIP text/image feature extraction unchanged."""
    def __init__(self, path, device='cpu'):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(str(path), local_files_only=True).to(device).eval()
        # Transformers 5 returns a ModelOutput where upstream expects a tensor.
        # pooler_output already contains the learned CLIP projection; do not
        # project twice or use unprojected last_hidden_state.
        for name in ('get_text_features', 'get_image_features'):
            original = getattr(self.model, name)
            def projected(*args, _original=original, **kwargs):
                value = _original(*args, **kwargs)
                return value.pooler_output if hasattr(value, 'pooler_output') else value
            setattr(self.model, name, projected)
        self.processor = CLIPProcessor.from_pretrained(str(path), local_files_only=True)
        self.device, self.lock = device, threading.RLock()
        self.cache = {}
        from simplemem.multimodal.core.config import OmniMemoryConfig
        from simplemem.multimodal.utils.embedding import EmbeddingService
        config = OmniMemoryConfig()
        config.embedding.model_name = 'openai/clip-vit-base-patch32'
        self.service = EmbeddingService(config)
        self.service._clip_model = self.model
        self.service._clip_processor = self.processor
        self.service._clip_backend = 'transformers_clip'
        self.service._clip_device = device
        self.torch = torch

    def embed_text(self, text):
        with self.lock:
            if text not in self.cache:
                result = self.service._embed_clip_text(text)
                if len(result) != 512:
                    raise RuntimeError('CLIP text embedding failed')
                self.cache[text] = result
            return list(self.cache[text])


def install_isolation():
    """Fix the upstream default graph directory through constructor injection."""
    from simplemem.multimodal import orchestrator
    from simplemem.multimodal.knowledge.knowledge_graph import KnowledgeGraph
    class IsolatedGraph(KnowledgeGraph):
        def __init__(self, *args, **kwargs):
            kwargs['storage_path'] = str(_graph_path.get())
            super().__init__(*args, **kwargs)
    orchestrator.KnowledgeGraph = IsolatedGraph


class OmniMemory:
    def __init__(self, workdir, client, clip, dump, task):
        from simplemem import create
        from simplemem.multimodal.core.config import OmniMemoryConfig
        config = OmniMemoryConfig()
        config.set_unified_model('local-qwen')
        config.llm.api_key = 'local-only'
        config.llm.api_base_url = 'http://127.0.0.1:1/v1'
        config.llm.max_tokens = 1000
        config.embedding.model_name = 'openai/clip-vit-base-patch32'
        config.embedding.embedding_dim = config.embedding.visual_embedding_dim = 512
        config.embedding.visual_embedding_model = 'openai/clip-vit-base-patch32'
        config.entropy_trigger.visual_model_name = 'openai/clip-vit-base-patch32'
        config.retrieval.default_top_k = 10
        config.enable_self_evolution = False
        token = _graph_path.set(workdir / 'knowledge_graph')
        try:
            self.core = create(mode='omni', config=config, data_dir=str(workdir))
        finally:
            _graph_path.reset(token)
        self.workdir, self.task, self.dump = workdir, task, dump
        self.sources, self.ingest = {}, []
        self.active_source = None
        self.session = None
        for component in (self.core, self.core.text_processor, self.core.image_processor,
                          self.core.audio_processor, self.core.video_processor,
                          self.core.query_processor, self.core.event_manager,
                          self.core.entity_extractor, self.core.expansion_manager):
            component._llm_client = client
        self.core.retriever._embedding_service = clip
        # TextProcessor otherwise loads a fresh encoder for every input.
        self.core.text_processor.generate_embedding = lambda data: clip.embed_text(str(data)[:8000])
        trigger = self.core.image_processor.trigger
        trigger._encoder, trigger._processor = clip.model, clip.processor
        trigger._clip_backend, trigger._device = 'transformers_clip', clip.device
        extract = trigger._extract_embedding
        def locked_image(data):
            with clip.lock:
                result = extract(data)
                if result is None or len(result) != 512:
                    raise RuntimeError('CLIP visual embedding failed; fallback forbidden')
                return result
        trigger._extract_embedding = locked_image
        for processor in (self.core.text_processor, self.core.image_processor):
            original = processor.create_mau
            def create_mau(*args, _original=original, **kwargs):
                mau = _original(*args, **kwargs)
                src = dict(self.active_source)
                mau.id = 'omni_' + hashlib.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()[:24]
                mau.timestamp = datetime.fromisoformat(src['timestamp']).replace(tzinfo=timezone.utc).timestamp()
                mau.metadata.source = src['round_id']
                mau.metadata.speaker_id = src['speaker']
                self.sources[mau.id] = src
                return mau
            processor.create_mau = create_mau
        self.channels = {}
        preview = self.core.retriever.retrieve_preview
        def traced_preview(*args, **kwargs):
            # Preserve native query filters; fix the candidate budget at ten.
            kwargs['top_k'] = 10
            result = preview(*args, **kwargs)
            self.channels['semantic'] = [dict(x) for x in result.items]
            return result
        self.core.retriever.retrieve_preview = traced_preview
        search = self.core.bm25_store.search
        def traced_bm25(*args, **kwargs):
            kwargs['top_k'] = 5
            result = search(*args, **kwargs)
            self.channels['keyword'] = [dict(id=key, score=float(score)) for key, score in result]
            return result
        self.core.bm25_store.search = traced_bm25
        dump(workdir / 'resolved_config.json', config.to_dict())
        assert self.core.knowledge_graph.storage_path == workdir / 'knowledge_graph'

    def _source(self, row, modality, image_id=None, image_path=None):
        return dict(task=self.task, round_id=self.task + '/' + row['id'],
                    round=row['id'], session=row['session_id'], timestamp=row['timestamp'],
                    speaker=row['speaker'], modality=modality,
                    image_id=image_id, image_path=str(image_path) if image_path else None)

    def store(self, row, images):
        if self.session != row['session_id']:
            if self.session is not None:
                self.core.end_session()
            self.core.start_session(row['session_id'])
            self.core.image_processor.reset_trigger()
            self.core.text_processor.reset()
            self.session = row['session_id']
        tags = ['round_id:' + self.task + '/' + row['id'], 'session_id:' + row['session_id'],
                'date:' + row['timestamp']]
        text = ('Date: ' + row['timestamp'] + '\nRound: ' + row['id'] + '\n' +
                row['speaker'] + ': ' + row['user'] + '\nAssistant: ' + row['assistant'])
        if images:
            text += '\nImage IDs: ' + ', '.join(image['image_id'] for image in images)
        if len(text) > self.core.text_processor.max_length:
            raise ValueError('Refusing upstream text truncation')
        self.active_source = self._source(row, 'text')
        result = self.core.add_text(text, session_id=self.session, tags=tags, force=False)
        records = [self._record(result, self.active_source)]
        ids = [result.mau.id] if result.success else []
        for image in images:
            self.active_source = self._source(row, 'visual', image['image_id'], image['image_path'])
            result = self.core.add_image(image['image_path'], session_id=self.session,
                                         tags=tags + ['image_id:' + image['image_id']], force=False)
            records.append(self._record(result, self.active_source))
            if result.success:
                assert result.mau.raw_pointer and len(result.mau.embedding) == 512
                ids.append(result.mau.id)
        # Same linkage as native add_multimodal(), generalized to multiple images.
        for identity in ids:
            mau = self.core.mau_store.get(identity)
            # Upstream serializes an empty related list as null; restore its
            # list meaning after the disk round trip, before native add_related.
            if mau.links.related_mau_ids is None:
                mau.links.related_mau_ids = []
            for other in ids:
                if other != identity:
                    mau.add_related(other)
            self.core.mau_store.update(mau)
        self.ingest.extend(records)
        self.dump(self.workdir / 'ingest.json', self.ingest)
        self.dump(self.workdir / 'sources.json', self.sources)

    @staticmethod
    def _record(result, source):
        if not result.success and not result.skipped:
            raise RuntimeError('Omni processing failed: ' + str(result.error))
        record = result.to_dict()
        record['source'] = source
        if result.trigger_result:
            record['trigger_reason'] = result.trigger_result.reason
        return record

    def finalize(self):
        self.core.end_session()
        self.core.build_bm25_index()
        self.core.save()
        self.core.knowledge_graph.save()
        self.dump(self.workdir / 'sources.json', self.sources)
        self.dump(self.workdir / 'final_memory.json', [self.core.mau_store.get(k).to_dict() for k in self.sources])

    def recall(self, query):
        self.channels = {}
        result = self.core.query(query, top_k=10, auto_expand=False)
        selected, scores = fuse_channels(self.channels)
        # Fetch the actual MAUs before expansion; BM25-only IDs are also valid.
        expansion = self.core.expand(selected)
        if [x['id'] for x in expansion.items] != selected:
            raise RuntimeError('Missing selected memory during evidence expansion')
        evidence, context_items = [], []
        for item in expansion.items:
            src = self.sources[item['id']]
            item['tags'] = item['metadata'].get('tags', [])
            raw = item.pop('raw_content', None)
            if src['modality'] == 'visual':
                import base64
                if not raw or not raw.get('base64'):
                    raise RuntimeError('Selected visual MAU failed to expand pixels')
                content = base64.b64decode(raw['base64'], validate=True)
                path = self.workdir / 'reader_images' / (item['id'] + '.jpg')
                path.parent.mkdir(exist_ok=True)
                if path.exists():
                    assert path.read_bytes() == content
                else:
                    path.write_bytes(content)
                evidence.append(dict(mau_id=item['id'], image_id=src['image_id'],
                                     path=str(path), sha256=hashlib.sha256(content).hexdigest()))
            context_items.append(item)
        bounded = replace(expansion, items=context_items)
        context = self.core.retriever.format_for_llm(bounded, include_instructions=False)
        graph = getattr(result, 'graph_entities', None)
        detail = dict(channels=self.channels, rrf_scores=scores, context_memory_ids=selected,
                      context_count=len(selected), source_ids=[self.sources[k]['round_id'] for k in selected],
                      context_items=context_items, history_images=evidence,
                      native_candidate_ids=[x['id'] for x in result.items],
                      graph_context=graph.to_dict() if hasattr(graph, 'to_dict') else graph)
        return context, detail
