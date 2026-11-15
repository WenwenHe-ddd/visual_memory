"""Fresh MemGallery text-memory replication of the exact seed-42 table rows.

The native code is loaded from each historical run's immutable snapshots. Only
the seed, fresh caption/input roots and run-local output paths change. Neither
historical responses nor constructed memories are imported into the new run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  LITELLM_LOCAL_MODEL_COST_MAP='True', ANONYMIZED_TELEMETRY='False')
SEED = 1042
RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
REFERENCE = ROOT / 'results/mem_gallery_qwen_seed42_text_methods_20261026_v2'
SM_REFERENCE = ROOT / 'results/simplemem_official_qwen_memgallery_k10_rrf5_reflection_20261029'
METHODS = ('gen_agents', 'memoryos', 'a_mem', 'simplemem')
EXPECTED_CORRECT = dict(gen_agents=472, memoryos=679, a_mem=660, simplemem=570)
LABELS = dict(gen_agents='Generative Agents (Top10)', memoryos='MemoryOS',
              a_mem='A-Mem (final 5)', simplemem='SimpleMem (official 0.3.0, final 5)')
CORE_OUTPUT_KEYS = ('raw_response', 'input_tokens', 'generated_tokens', 'hit_token_limit')
TRUNCATED_INSTRUCTION = (
    'Your earlier extraction exceeded the output length limit. Generate a complete, concise JSON array from ONLY the supplied dialogue window. '
    'Include each supported fact once; do not enumerate imagined observations, times, or repeated variants. '
    'Keep lossless_restatement concise and keywords short. All scalar fields (timestamp, location, topic) must be a single string or null, never an array. '
    'Close the JSON array and stop. Do not add Markdown or explanations.')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def durable_dump(path, value):
    """Seal a complete raw generation before native callers can materialize it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))
    temporary.replace(path)


def frozen_json(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Existing signed artifact differs: ' + str(path))
    else:
        dump(path, value)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


class SnapshotLoader(importlib.abc.Loader):
    def __init__(self, original, entry):
        self.original, self.entry = original, entry

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = Path(self.entry['path'])
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.entry['sha256']:
            raise ValueError('Historical snapshot changed: ' + str(path))
        module.__file__ = self.original  # Preserve native relative resource paths.
        module.__experiment_snapshot__ = dict(self.entry)
        exec(compile(source, str(path), 'exec'), module.__dict__)


class SnapshotFinder(importlib.abc.MetaPathFinder):
    def __init__(self, entries):
        self.entries = entries
        self.modules = {}
        for original, entry in entries.items():
            path = Path(original)
            if not path.suffix == '.py':
                continue
            if path.is_relative_to(ROOT / 'scripts'):
                relative = path.relative_to(ROOT).with_suffix('')
            elif path.is_relative_to(ROOT / '.vendor/simplemem_official_runtime'):
                relative = path.relative_to(ROOT / '.vendor/simplemem_official_runtime').with_suffix('')
            else:
                continue
            parts = list(relative.parts)
            package = parts[-1] == '__init__'
            if package:
                parts.pop()
            self.modules['.'.join(parts)] = (original, entry, package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.modules:
            return None
        original, entry, package = self.modules[fullname]
        spec = importlib.util.spec_from_loader(fullname, SnapshotLoader(original, entry),
                                              origin=original, is_package=package)
        if package:
            spec.submodule_search_locations = [str(Path(original).parent)]
        return spec


def load_native(method):
    """A process executes one method so incompatible historical modules never mix."""
    if method not in METHODS:
        raise ValueError('Unknown method: ' + method)
    reference = SM_REFERENCE if method == 'simplemem' else REFERENCE
    entries = read(reference / 'source_snapshot/manifest.json')
    finder = SnapshotFinder(entries)
    for name in finder.modules:
        if name in sys.modules:
            raise ValueError('Native module imported before its frozen snapshot: ' + name)
    sys.meta_path.insert(0, finder)
    base = importlib.import_module('scripts.evaluation.mem_gallery_qwen_text_methods')
    if method == 'simplemem':
        adapter = importlib.import_module('scripts.evaluation.mem_gallery_simplemem_official')
    else:
        adapter = importlib.import_module('scripts.evaluation.mem_gallery_qwen_text_methods_v2')
    return SimpleNamespace(base=base, adapter=adapter, finder=finder, entries=entries,
                           reference=reference, native=base.native, gallery=base.gallery)


def checked_shared_inputs(run):
    marker = run / 'caption_methods/stages/shared-inputs.json'
    if not marker.exists() or read(marker).get('status') not in ('complete', 'passed'):
        raise ValueError('Fresh seed-1042 shared inputs are not complete: ' + str(marker))
    cap = run / 'captions'
    if read(cap / 'protocol.json')['seed'] != SEED:
        raise ValueError('Wrong caption-generation seed')
    if read(cap / 'independent_audit.json')['status'] != 'passed':
        raise ValueError('Caption audit did not pass')
    if read(cap / 'completion_manifest.json').get('status') != 'complete':
        raise ValueError('Fresh caption stage is not sealed')
    source = run / 'inputs'
    for name, expected in read(source / 'artifact_manifest.json').items():
        if sha(source / name) != expected:
            raise ValueError('Shared input artifact changed: ' + name)
    qs, memories = rows(source / 'questions.jsonl'), rows(source / 'memories.jsonl')
    oldq = rows(REFERENCE / 'inputs/questions.jsonl')
    if qs != oldq or sha(source / 'gold.jsonl') != sha(REFERENCE / 'inputs/gold.jsonl'):
        raise ValueError('Questions, official query captions or gold differ from seed42')
    if (len(qs) != 1711 or len({q['id'] for q in qs}) != 1711
            or len(memories) != 3962 or sum(m['is_visual'] for m in memories) != 1003):
        raise ValueError('MemGallery coverage must be 1711 questions / 3962 rounds / 1003 images')
    if any({'answer', 'ground_truth', 'options', 'clue', 'gold_ids'} & q.keys() for q in qs):
        raise ValueError('Evaluation labels leaked into prepared questions')
    return qs, memories


def signature(protocol):
    return digest({k: v for k, v in protocol.items() if k != 'experiment_signature'})


def batch_request(stage, batch, cap, temperature, protocol):
    if stage not in ('memory', 'reader'):
        raise ValueError('Unknown raw generation stage')
    image_hashes = {}
    if stage == 'reader':
        for item in batch:
            for path in item['query_image_paths']:
                image_hashes[path] = sha(path)
    return {'schema_version': 1, 'stage': stage, 'seed': SEED,
        'cap': int(cap), 'temperature': float(temperature), 'batch_size': len(batch),
        'batch': copy.deepcopy(batch), 'query_image_sha256': image_hashes,
        'experiment_signature': protocol['experiment_signature'],
        'batch_policy': 'Reuse only an identical full ordered native batch, cap, temperature and signed run; never splice outputs across batches.'}


def validate_raw_batch(record, request, path=None):
    if (record.get('status') != 'success' or record.get('request') != request
            or record.get('request_sha256') != digest(request)
            or (path is not None and Path(path).stem != digest(request))):
        raise ValueError('Durable raw generation request/contract differs')
    outputs, evidence = record.get('outputs'), record.get('generation_evidence', {})
    if (not isinstance(outputs, list) or len(outputs) != request['batch_size']
            or record.get('outputs_sha256') != digest(outputs)):
        raise ValueError('Durable raw generation outputs differ')
    expected_sampler = SEED if request['temperature'] > 0 else None
    if (request['seed'] != SEED or evidence.get('actual_torch_seed') != SEED
            or evidence.get('actual_categorical_seed') != expected_sampler
            or not evidence.get('observed_model_generate')
            or any(evidence.get(k) != request[k] for k in ('stage', 'cap', 'temperature'))):
        raise ValueError('Durable raw generation lacks the correct actual seed evidence')
    for output in outputs:
        if (not isinstance(output.get('raw_response'), str)
                or type(output.get('input_tokens')) is not int
                or type(output.get('generated_tokens')) is not int
                or type(output.get('hit_token_limit')) is not bool):
            raise ValueError('Invalid native raw output in durable generation record')
    return record


def durable_batch(path, request, generate):
    """An exact-batch success survives a crash before Broker/reader cache writes."""
    path = Path(path)
    if path.exists():
        record = validate_raw_batch(read(path), request, path)
        return copy.deepcopy(record['outputs']), record['generation_evidence'], 'durable_raw_recovery'
    outputs, evidence = generate()
    record = {'status': 'success', 'request': request, 'request_sha256': digest(request),
        'outputs': outputs, 'outputs_sha256': digest(outputs),
        'generation_evidence': evidence, 'generated_at_utc': now()}
    validate_raw_batch(record, request, path)
    durable_dump(path, record)
    return outputs, evidence, 'fresh_model_generation'


def prepare(ctx, method, run=RUN, device='cuda:1'):
    output, root = run / 'text_methods', run / 'text_methods' / method
    if (root / 'protocol.json').exists():
        return verify(ctx, method, run, device)
    preflight_path = output / 'preflight' / (method + '.json')
    preflight_result = read(preflight_path)
    if preflight_result.get('status') != 'passed' or preflight_result.get('source_sha256') != sha(__file__):
        raise ValueError('Run CPU preflight with this exact driver before preparing the experiment')
    qs, memories = checked_shared_inputs(run)
    # The original observation builder already checks dialogue, dates and images.
    ctx.base.DEFAULT_CAPTIONS = run / 'captions'
    observations = ctx.base.make_observations(memories)
    if sum(map(len, observations.values())) != 3962 or len(observations) != 20:
        raise ValueError('Invalid task-isolated observation coverage')
    inputs = output / 'inputs'
    for source in (run / 'inputs').iterdir():
        if source.is_file():
            target = inputs / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != sha(source):
                raise ValueError('Existing local input copy differs: ' + str(target))
            if not target.exists():
                shutil.copyfile(source, target)
    frozen_json(output / 'observations.json', observations)
    reference = ctx.reference / method
    old = read(reference / 'protocol.json')
    summary = read((ctx.reference / 'final_results.json') if method == 'simplemem' else reference / 'summary.json')
    if summary['overall']['correct'] != EXPECTED_CORRECT[method]:
        raise ValueError('Historical source does not match the screenshot')
    p = copy.deepcopy(old)
    p.pop('experiment_signature', None)
    p.pop('input_sha256', None)
    p.update(seed=SEED, study='MemGallery fresh-caption full-flow seed-1042 stability replication',
        reader_device=str(device), reader_model_path=str(ctx.gallery.MODEL),
        questions_sha256=ctx.base.obj(qs), observations_sha256=ctx.base.obj(observations),
        original_baseline_root=str(run), no_external_api=True,
        memory_sampling='Native per-call temperatures; independent seed1042 categorical RNG per request, top_p=1, no top_k.',
        caption_policy='Fresh seed1042 Qwen history captions, verbatim; original official QA captions fixed',
        query_caption_policy='Original official QA image_caption, verbatim; no regeneration for retrieval',
        historical_model=copy.deepcopy(old['model']))
    p['model'] = ctx.gallery.model_provenance()
    if p['model'] != old['model']:
        raise ValueError('Qwen weights or model settings differ from the screenshot source')
    p['stability_replication'] = {
        'reference_root': str(reference), 'reference_seed': 42, 'seed': SEED,
        'reference_correct': EXPECTED_CORRECT[method], 'reference_questions': 1711,
        'reference_protocol_sha256': sha(reference / 'protocol.json'),
        'reference_responses_sha256': sha(reference / 'reader/responses.jsonl'),
        'construction_policy': 'Fresh native stores; no seed42 memories or model responses imported',
        'retrieval_policy': 'Fresh native retrieval including query keywords, planning and reflection',
        'caption_policy': 'New 1042 history caption outputs; fixed original official question-image captions',
        'reader_policy': 'Original shared Gallery prompts and original query pixels; greedy256; original per-method batch sizes',
        'embedding_device': str(device), 'judge_seed': 42,
        'native_nondeterminism': 'Native uuid4 identities and wall-clock access/maintenance timestamps are preserved; seed does not control these.',
        'generation_seed_evidence': 'Runtime model.generate wrapper verifies Torch seed1042 and every actual categorical sampler seed; durable messages/output hashes connect each cached memory call and final answer to an actual fresh generation.',
        'raw_generation_durability': 'Full ordered batch inputs, original decoded outputs and actual seed evidence are fsynced before return to native Broker/reader. Resume reuses only an identical batch request within this signed run. If native task scheduling changes the full batch, a separately recorded generation is required; outputs are never spliced across batches.',
        'simplemem_conditional_recovery': 'For builder extraction only: first native request; on an output-limit hit, one concise retry at unchanged cap using the historical recovery instruction. On a native parser location-list failure, serialize string locations with semicolon-space; preserve other fields and raw generations. All recoveries recorded; no quality-based retries.',
        'ga_conditional_recovery': 'Original reflection first; invalid line count only, at most three existing count/line-format retries.',
    }
    dependencies = [Path(__file__).resolve(), reference / 'protocol.json', reference / 'reader/responses.jsonl',
        ctx.reference / 'source_snapshot/manifest.json',
        preflight_path, output / 'preflight/durability_review.json',
        output / 'observations.json', run / 'captions/protocol.json',
        run / 'captions/independent_audit.json', run / 'captions/completion_manifest.json',
        run / 'caption_methods/stages/shared-inputs.json']
    dependencies += [v for v in inputs.iterdir() if v.is_file()]
    dependencies += [Path(v['path']) for v in ctx.entries.values()]
    embedding = Path(p['embedding_model_path'])
    dependencies += [v for v in embedding.rglob('*') if v.is_file()]
    if method == 'simplemem':
        dependencies += ctx.adapter.official.source_files()
        dependencies += [SM_REFERENCE / 'recovery_amendment.json']
    else:
        dependencies += [ctx.native.VENDOR / 'UPSTREAM.json']
    p['input_sha256'] = {str(v.resolve()): sha(v) for v in dependencies}
    p['input_sha256'].update({str(ctx.gallery.MODEL / name): item['sha256']
                             for name, item in p['model']['files'].items()})
    p['snapshot_sources'] = copy.deepcopy(ctx.entries)
    p['experiment_signature'] = signature(p)
    frozen_json(root / 'protocol.json', p)
    frozen_json(root / 'source_snapshot_manifest.json', ctx.entries)
    dump(root / 'stage_status.json', {'status': 'prepared', 'seed': SEED, 'method': method,
         'questions': len(qs), 'memory_rounds': 3962, 'updated_at': now()})
    return verify(ctx, method, run, device)


def verify(ctx, method, run=RUN, device='cuda:1'):
    root = run / 'text_methods' / method
    p = read(root / 'protocol.json')
    if (p['seed'] != SEED or p['experiment_signature'] != signature(p)
            or p['reader_device'] != device or p['mode'] != method):
        raise ValueError('Invalid seed, method, device or signed protocol')
    for path, expected in p['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Pinned run input changed: ' + path)
    if p['snapshot_sources'] != ctx.entries or ctx.finder not in sys.meta_path:
        raise ValueError('The signed native source snapshots are not active')
    for name, (original, entry, _) in ctx.finder.modules.items():
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__experiment_snapshot__', None) != entry:
            raise ValueError('Native implementation not loaded from the snapshot: ' + name)
    reference = read(ctx.reference / method / 'protocol.json')
    unchanged = ('method_defaults', 'embedding_model_path', 'max_new_tokens', 'memory_max_new_tokens',
                 'context_budget', 'generation_batch_size', 'task_workers', 'reader_batch_size_text',
                 'reader_batch_size_images', 'query_image_max_pixels', 'reader_system_prompt', 'task_instructions')
    for key in unchanged:
        if p.get(key) != reference.get(key):
            raise ValueError('Unexpected change from seed42: ' + key)
    ctx.native.verify_vendor()
    if method == 'simplemem' and ctx.adapter.official.verify_installation() != p['upstream']:
        raise ValueError('Official SimpleMem source or runtime version changed')
    qs = rows(run / 'text_methods/inputs/questions.jsonl')
    obs = read(run / 'text_methods/observations.json')
    if ctx.base.obj(qs) != p['questions_sha256'] or ctx.base.obj(obs) != p['observations_sha256']:
        raise ValueError('Prepared input contents changed')
    return p


def serialized_location(response):
    text = response.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[len('```json'):-3].strip()
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError('Expected a native extraction array')
    fixed, changes = copy.deepcopy(values), []
    for index, entry in enumerate(fixed):
        if not isinstance(entry, dict):
            raise ValueError('Expected native extraction entry objects')
        value = entry.get('location')
        if isinstance(value, list):
            if not value or not all(isinstance(s, str) for s in value):
                raise ValueError('Only nonempty string location lists can be serialized')
            entry['location'] = '; '.join(value)
            changes.append({'index': index, 'before': value, 'after': entry['location']})
    if not changes:
        raise ValueError('No list-valued location to repair')
    return json.dumps(fixed, ensure_ascii=False), changes


def is_builder_extraction(messages):
    return (len(messages) >= 2 and isinstance(messages[0].get('content'), str)
        and messages[0]['content'].startswith('You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations.')
        and '[Current Window Dialogues]' in messages[-1].get('content', '')
        and TRUNCATED_INSTRUCTION not in messages[-1].get('content', ''))


def install_runtime(ctx, method, run, protocol, device):
    """Use original generation code; observe and check its actual seed boundary."""
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ctx.gallery.SEED = SEED
    if method != 'simplemem':
        ctx.adapter.OUTPUT = run / 'text_methods'
        ctx.native.helpers = ctx.adapter.gallery_helpers
        ctx.native.install_runtime = ctx.adapter.install
    ctx.base.verify = lambda output, selected: verify(ctx, selected, run, device)
    event_path = run / 'text_methods' / method / 'runtime_generation_calls.jsonl'
    raw_batches = run / 'text_methods' / method / 'raw_generation_batches'
    events_lock = threading.Lock()
    original_qwen = ctx.base.LocalQwen
    original_reader = ctx.gallery.generate_batch
    active = threading.local()

    class AuditedQwen(original_qwen):
        def __init__(self, p, selected_device):
            if p['seed'] != SEED or ctx.gallery.SEED != SEED:
                raise ValueError('Actual runtime seed differs')
            super().__init__(p, selected_device)
            self.environment.update(seed=SEED, native_snapshot_generation=True,
                seed1042_driver_sha256=sha(__file__))
            self.actual_generate = self.model.generate

            def checked_generate(*args, **kwargs):
                context = getattr(active, 'context', None)
                if context is None or torch.initial_seed() != SEED:
                    raise ValueError('Model generation lacks the signed seed1042 boundary')
                if kwargs.get('do_sample') is not False or kwargs['max_new_tokens'] != context['cap']:
                    raise ValueError('Actual generation parameters differ from native call')
                processors = kwargs.get('logits_processor', [])
                categorical = [v for v in processors if isinstance(v, ctx.native.PerRowTemperatureSampler)]
                temperature = context['temperature']
                if temperature > 0:
                    if len(categorical) != 1 or categorical[0].seed != SEED or categorical[0].temperature != temperature:
                        raise ValueError('Actual native categorical sampler did not receive seed1042')
                elif processors:
                    raise ValueError('Greedy generation unexpectedly contains a sampling processor')
                context['observed_model_generate'] = True
                context['actual_torch_seed'] = torch.initial_seed()
                context['actual_categorical_seed'] = categorical[0].seed if categorical else None
                return self.actual_generate(*args, **kwargs)

            self.model.generate = checked_generate

        def generate(self, messages, max_new_tokens, sampling_temperature=0.0):
            request = batch_request('memory', messages, max_new_tokens, sampling_temperature, protocol)
            path = raw_batches / (digest(request) + '.json')

            def generate():
                torch.manual_seed(SEED)
                active.context = {'stage': 'memory', 'cap': max_new_tokens,
                    'temperature': sampling_temperature, 'observed_model_generate': False}
                started = time.monotonic()
                output = super(AuditedQwen, self).generate(messages, max_new_tokens, sampling_temperature)
                evidence = copy.deepcopy(active.context)
                if not evidence['observed_model_generate']:
                    raise ValueError('Native model did not execute')
                evidence['original_generation_elapsed_seconds'] = time.monotonic() - started
                del active.context
                return output, evidence

            result, evidence, origin = durable_batch(path, request, generate)
            for conversation, output in zip(messages, result):
                event = {**evidence, 'messages_sha256': digest(conversation),
                    'output_sha256': digest({k: output[k] for k in CORE_OUTPUT_KEYS}),
                    'batch_size': len(messages), 'generation_origin': origin,
                    'raw_batch_path': str(path), 'raw_batch_sha256': sha(path),
                    'raw_batch_request_sha256': digest(request),
                    'experiment_signature': protocol['experiment_signature']}
                with events_lock:
                    append(event_path, event)
            return result

    def checked_reader(runtime, manifest, cap):
        request = batch_request('reader', manifest, cap, 0., protocol)
        path = raw_batches / (digest(request) + '.json')

        def generate():
            active.context = {'stage': 'reader', 'cap': cap, 'temperature': 0.0,
                               'observed_model_generate': False}
            result = original_reader(runtime, manifest, cap)
            evidence = copy.deepcopy(active.context)
            if not evidence['observed_model_generate']:
                raise ValueError('Final reader model did not execute')
            del active.context
            return result, evidence

        result, evidence, origin = durable_batch(path, request, generate)
        for item, output in zip(manifest, result):
            event = {**evidence, 'payload_sha256': item['payload_sha256'],
                'id': item['id'], 'output_sha256': digest({k: output[k] for k in CORE_OUTPUT_KEYS}),
                'batch_size': len(manifest), 'generation_origin': origin,
                'raw_batch_path': str(path), 'raw_batch_sha256': sha(path),
                'raw_batch_request_sha256': digest(request),
                'experiment_signature': protocol['experiment_signature']}
            with events_lock:
                append(event_path, event)
        return result

    ctx.base.LocalQwen = AuditedQwen
    ctx.gallery.generate_batch = checked_reader
    if method == 'simplemem':
        original_broker = ctx.base.Broker

        class ConditionalRecoveryBroker(original_broker):
            def request(self, messages, *args, **kwargs):
                output = super().request(messages, *args, **kwargs)
                if is_builder_extraction(messages) and output['hit_token_limit']:
                    repaired = copy.deepcopy(messages)
                    repaired[-1]['content'] += '\n\n' + TRUNCATED_INSTRUCTION
                    retry = super().request(repaired, *args, **kwargs)
                    with self.log_lock:
                        append(self.root / 'construction_recoveries.jsonl', {
                            'kind': 'truncated_window', 'original_call': output['call_key'],
                            'retry_call': retry['call_key'], 'same_generation_cap': True,
                            'owner': kwargs.get('owner') or getattr(self.local, 'owner', None),
                            'retry_hit_token_limit': retry['hit_token_limit']})
                    return retry
                return output

        class ConditionalRecoveryMemory(ctx.adapter.GalleryMemory):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                original_parse = self.core.memory_builder._parse_llm_response

                def parse(response, identities):
                    try:
                        return original_parse(response, identities)
                    except Exception:
                        # Only the historical location-array schema repair is permitted.
                        try:
                            corrected, changes = serialized_location(response)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            raise
                        result = original_parse(corrected, identities)
                        append(self.workdir / 'location_serialization_repairs.jsonl', {
                            'raw_response_sha256': hashlib.sha256(response.encode()).hexdigest(),
                            'corrected_response': corrected, 'changes': changes,
                            'dialogue_ids': identities, 'raw_generation_preserved': True})
                        return result

                self.core.memory_builder._parse_llm_response = parse

        ctx.base.Broker = ConditionalRecoveryBroker
        ctx.base.Memory = ConditionalRecoveryMemory


def run_method(ctx, method, run=RUN, device='cuda:1'):
    p = verify(ctx, method, run, device)
    root = run / 'text_methods' / method
    dump(root / 'stage_status.json', {'status': 'running', 'phase': 'native_build_retrieve_reader',
         'seed': SEED, 'method': method, 'updated_at': now()})
    install_runtime(ctx, method, run, p, device)
    ctx.base.run(run / 'text_methods', method, device)
    return export(ctx, method, run, device)


def validate_generation_events(events, protocol):
    batches = {}
    for event in events:
        if (event['experiment_signature'] != protocol['experiment_signature']
                or event['actual_torch_seed'] != SEED or not event['observed_model_generate']
                or (event['temperature'] > 0 and event['actual_categorical_seed'] != SEED)
                or (event['temperature'] <= 0 and event['actual_categorical_seed'] is not None)):
            raise ValueError('Actual generation seed audit failed')
        path = event['raw_batch_path']
        if path not in batches:
            record = read(path)
            validate_raw_batch(record, record['request'], path)
            batches[path] = (sha(path), record)
        expected_hash, record = batches[path]
        request = record['request']
        if (event['raw_batch_sha256'] != expected_hash
                or event['raw_batch_request_sha256'] != record['request_sha256']
                or request['experiment_signature'] != protocol['experiment_signature']
                or event['generation_origin'] not in ('fresh_model_generation', 'durable_raw_recovery')
                or any(event[k] != request[k] for k in ('stage', 'cap', 'temperature', 'batch_size'))):
            raise ValueError('Generation event does not match its durable raw batch')
        if event['stage'] == 'memory':
            expected = {(digest(conversation), digest({k: output[k] for k in CORE_OUTPUT_KEYS}))
                        for conversation, output in zip(request['batch'], record['outputs'])}
            identity = (event['messages_sha256'], event['output_sha256'])
        else:
            expected = {(item['id'], item['payload_sha256'], digest({k: output[k] for k in CORE_OUTPUT_KEYS}))
                        for item, output in zip(request['batch'], record['outputs'])}
            identity = (event['id'], event['payload_sha256'], event['output_sha256'])
        if identity not in expected:
            raise ValueError('Generation event output is not present in its durable raw batch')
    return batches


def audit(ctx, method, run=RUN, device='cuda:1'):
    p = verify(ctx, method, run, device)
    root = run / 'text_methods' / method
    questions = rows(run / 'text_methods/inputs/questions.jsonl')
    observations = read(run / 'text_methods/observations.json')
    rlist, answers, manifest = (rows(root / n) for n in (
        'retrieval.jsonl', 'reader/responses.jsonl', 'reader/manifest.jsonl'))
    maps = [{r['id']: r for r in values} for values in (questions, rlist, answers, manifest)]
    qmap, rmap, amap, mmap = maps
    if not all(len(m) == 1711 and set(m) == set(qmap) for m in maps) or any(len(v) != 1711 for v in (rlist, answers, manifest)):
        raise ValueError('Expected one response and retrieval for every one of 1711 questions')
    seal = read(root / 'reader/complete.json')
    if (seal['experiment_signature'] != p['experiment_signature']
            or seal['responses_sha256'] != sha(root / 'reader/responses.jsonl')
            or seal['manifest_sha256'] != sha(root / 'reader/manifest.jsonl')):
        raise ValueError('Final reader completion hashes differ')
    events = rows(root / 'runtime_generation_calls.jsonl')
    raw_batches = validate_generation_events(events, p)
    memory_events = {(e['messages_sha256'], e['cap'], e['temperature'], e['output_sha256']) for e in events if e['stage'] == 'memory'}
    reader_events = {(e['id'], e['payload_sha256'], e['output_sha256']) for e in events if e['stage'] == 'reader'}
    sampling, artifacts, token_limits, invalid_schema = Counter(), {}, 0, 0
    for path in (root / 'calls').glob('*.json'):
        call = read(path)
        req, out = call['request'], call['output']
        if (ctx.base.obj(req) != path.stem or req['seed'] != SEED or req['kind'] != 'memory'
                or req['experiment_signature'] != p['experiment_signature'] or 'inherited_from' in call
                or req['do_sample'] != (req['temperature'] > 0)):
            raise ValueError('Memory cache is not a fresh seed1042 request: ' + str(path))
        identity = (digest(req['messages']), req['max_new_tokens'], req['temperature'],
                    digest({k: out[k] for k in CORE_OUTPUT_KEYS}))
        if identity not in memory_events:
            raise ValueError('Memory output lacks actual seed1042 generation evidence: ' + path.stem)
        artifacts[str(path)] = sha(path)
        sampling[req['temperature']] += 1
        token_limits += bool(out['hit_token_limit'])
        invalid_schema += out.get('structured_output_valid') is False
    notes, construction = {}, []
    helper = ctx.native.helpers(method) if method == 'a_mem' else None
    for task, obs in observations.items():
        target = root / task
        state, stats = read(target / 'complete.json'), read(target / 'memory_stats.json')
        if (state['experiment_signature'] != p['experiment_signature'] or stats['stored_rounds'] != len(obs)
                or state['retrieval_sha256'] != sha(target / 'retrieval.jsonl')
                or state['questions'] != sum(q['task'] == task for q in questions)):
            raise ValueError('Incomplete native construction/retrieval: ' + task)
        construction.append({'task': task, 'stored_rounds': stats['stored_rounds'], 'questions': state['questions']})
        if method in ('gen_agents', 'a_mem', 'simplemem'):
            notes[task] = read(target / 'final_memory.json')
        if method == 'a_mem':
            if set(notes[task]) != {o['id'] for o in obs}:
                raise ValueError('A-Mem lost source rounds')
            for o in obs:
                note = notes[task][o['id']]
                if note['timestamp'] != o['timestamp'] or note['content'] != helper['build_a_mem_note_text'](o, o['speaker'], 'assistant'):
                    raise ValueError('A-Mem source text or timestamp changed')
        if method == 'gen_agents':
            originals = {n['dialogue_id']: n for n in notes[task] if n.get('source') is False}
            if set(originals) != {o['id'] for o in obs}:
                raise ValueError('GA source-round coverage differs')
            ga_helper = ctx.adapter.gallery_helpers(method)
            if any(originals[o['id']]['text'] != ga_helper['_build_round_text'](o, o['speaker'], 'assistant') for o in obs):
                raise ValueError('GA source dialogue, caption or original metadata changed')
        if method == 'simplemem':
            windows = read(target / 'runtime/construction_windows.json')
            sources = read(target / 'runtime/dialogue_sources.json')
            if ({s['round_id'] for s in sources.values()} != {o['id'] for o in obs}
                    or {i for w in windows for i in w['dialogue_ids']} != {int(i) for i in sources}
                    or not all(w['parsed_successfully'] for w in windows)):
                raise ValueError('Official SimpleMem did not parse and ingest every original window')
        for file in target.rglob('*'):
            if file.is_file() and file.suffix in ('.json', '.jsonl'):
                artifacts[str(file)] = sha(file)
    counts = Counter()
    for q in questions:
        identity = q['id']
        retrieval, answer, payload = rmap[identity], amap[identity], mmap[identity]
        if retrieval['retrieval_query'] != q['retrieval_query']:
            raise ValueError('Native retrieval query differs from original official caption policy')
        expected = dict(id=identity, task=q['task'], point=q['point'], system=ctx.gallery.READER_SYSTEM,
            user_text=ctx.base.reader_text(q, retrieval['retrieved_context']), query_image_paths=q['query_image_paths'])
        if {k: v for k, v in payload.items() if k != 'payload_sha256'} != expected:
            raise ValueError('Original Gallery reader text or pixels changed: ' + identity)
        if ctx.base.obj(expected) != payload['payload_sha256'] or answer['payload_sha256'] != payload['payload_sha256']:
            raise ValueError('Reader input hash mismatch')
        if retrieval['experiment_signature'] != p['experiment_signature']:
            raise ValueError('Retrieval/answer belongs to another protocol')
        if answer['experiment_signature'] != p['experiment_signature']:
            raise ValueError('Reader belongs to another protocol')
        if (identity, payload['payload_sha256'], digest({k: answer[k] for k in CORE_OUTPUT_KEYS})) not in reader_events:
            raise ValueError('Final answer lacks actual seed1042 generation evidence')
        detail, context = retrieval['retrieval_detail'], retrieval['retrieved_context']
        if method == 'a_mem':
            ids = detail['context_memory_ids']
            if len(ids) != 5 or len(set(ids)) != 5:
                raise ValueError('A-Mem final context must contain five distinct notes')
            chosen = [notes[q['task']][i] for i in ids]
            rendered = ''.join('talk start time:' + n['timestamp'] + 'memory content: ' + n['content']
                + 'memory context: ' + n['context'] + 'memory keywords: ' + str(n['keywords'])
                + 'memory tags: ' + str(n['tags']) + '\n' for n in chosen)
            if rendered != context or detail['primary_top_k'] != 5:
                raise ValueError('A-Mem context expansion/rendering changed')
            counts[len(ids)] += 1
        elif method == 'simplemem':
            ids = detail['context_memory_ids']
            if len(ids) != len(set(ids)) or len(ids) not in (0, 5) or (not ids and not detail['source_empty_preserved']):
                raise ValueError('Official SimpleMem final-context5 policy changed')
            known = {n['entry_id'] for n in notes[q['task']]}
            if not set(ids) <= known or [n['id'] for n in detail['items']] != ids:
                raise ValueError('SimpleMem retrieved entries not present in the constructed store')
            from simplemem.core.answer_generator import AnswerGenerator
            rendered = AnswerGenerator._format_contexts(None,
                [SimpleNamespace(**{k: v for k, v in item.items() if k != 'id'}) for item in detail['items']])
            if rendered != context:
                raise ValueError('Official SimpleMem context rendering differs')
            counts[len(ids)] += 1
        elif method == 'gen_agents':
            # Native renderer numbers each complete returned memory.
            import re
            count = len(re.findall(r'(?m)^\[\d+\] ', context))
            if not 1 <= count <= 10:
                raise ValueError('GA native Top10 context count differs')
            counts[count] += 1
        elif method == 'memoryos':
            helper = ctx.adapter.gallery_helpers(method)
            if helper['_format_retrieval_context'](detail) != context:
                raise ValueError('Native MemoryOS multi-tier context rendering differs')
            count = len(detail['retrieved_pages'])
            if not 0 <= count <= 10:
                raise ValueError('MemoryOS native page limit changed')
            counts[count] += 1
    for path in (root / 'retrieval.jsonl', root / 'reader/responses.jsonl', root / 'reader/manifest.jsonl',
                 root / 'reader/complete.json', root / 'runtime_generation_calls.jsonl'):
        artifacts[str(path)] = sha(path)
    physical_generated_rows = 0
    for path in (root / 'raw_generation_batches').glob('*.json'):
        record = raw_batches[str(path)][1] if str(path) in raw_batches else read(path)
        validate_raw_batch(record, record['request'], path)
        if record['request']['experiment_signature'] != p['experiment_signature']:
            raise ValueError('Durable raw batch belongs to another experiment')
        physical_generated_rows += len(record['outputs'])
        artifacts[str(path)] = sha(path)
    report = {'status': 'passed', 'seed': SEED, 'method': method, 'questions': 1711,
        'memory_rounds': 3962, 'tasks': construction, 'historical_memory_imports': 0,
        'historical_response_imports': 0, 'source_implementation': 'Verified original per-method snapshot bytes',
        'native_memory_calls': sum(sampling.values()), 'memory_sampling_temperature_counts': dict(sorted(sampling.items())),
        'original_native_memory_token_limit_hits': token_limits, 'original_native_schema_invalid': invalid_schema,
        'final_reader_token_limit_hits': sum(a['hit_token_limit'] for a in answers),
        'empty_answers': sum(not a['raw_response'].strip() for a in answers),
        'context_count_distribution': dict(sorted(counts.items())),
        'actual_seed_generation_events': len(events), 'original_query_captions_fixed': True,
        'durable_raw_batches': len(list((root / 'raw_generation_batches').glob('*.json'))),
        'physical_successful_generation_rows': physical_generated_rows,
        'exact_batch_raw_recovery_rows': sum(e['generation_origin'] == 'durable_raw_recovery' for e in events),
        'artifact_sha256': artifacts, 'protocol_sha256': sha(root / 'protocol.json'),
        'experiment_signature': p['experiment_signature'], 'audited_at': now()}
    dump(root / 'independent_audit.json', report)
    return report


def export(ctx, method, run=RUN, device='cuda:1'):
    report = audit(ctx, method, run, device)
    root = run / 'text_methods' / method
    questions = rows(run / 'text_methods/inputs/questions.jsonl')
    # Gold is parsed only after every generated response and native state passes audit.
    gold = {r['id']: r for r in rows(run / 'text_methods/inputs/gold.jsonl')}
    answers = {r['id']: r for r in rows(root / 'reader/responses.jsonl')}
    records, scored = [], []
    source_hash = sha(root / 'reader/responses.jsonl')
    for q in questions:
        answer = answers[q['id']]
        metric = ctx.gallery.score_answer(answer['raw_response'], gold[q['id']]['answer'])
        scored.append({'id': q['id'], 'task': q['task'], 'point': q['point'], **metric})
        records.append({**{k: q[k] for k in ('id', 'task', 'point', 'question')},
            'ground_truth': gold[q['id']]['answer'], 'model_output': answer['raw_response'],
            'answer_model': 'Qwen/Qwen3-VL-8B-Instruct', 'source_seed': SEED,
            'source_response_file': str(root / 'reader/responses.jsonl'),
            'source_response_sha256': source_hash, 'source_responses_sha256': source_hash,
            'source_protocol_file': str(root / 'protocol.json'), 'source_protocol_sha256': report['protocol_sha256'],
            'source_record_id': q['id'], 'source_hit_token_limit': answer['hit_token_limit'],
            'source_payload_sha256': answer['payload_sha256'], 'source_em': bool(metric['em'])})
    summary = {'status': 'complete', 'seed': SEED, 'method': method, 'label': LABELS[method],
        'questions': 1711, 'correct': sum(int(r['em']) for r in scored),
        **{key: sum(r[key] for r in scored) / 1711 for key in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu')},
        'seed42_correct': EXPECTED_CORRECT[method], 'seed42_em': EXPECTED_CORRECT[method] / 1711,
        'source_sha256': {str(root / 'reader/responses.jsonl'): source_hash,
            str(root / 'protocol.json'): report['protocol_sha256'],
            str(root / 'independent_audit.json'): sha(root / 'independent_audit.json'),
            str(run / 'inputs/questions.jsonl'): sha(run / 'inputs/questions.jsonl'),
            str(run / 'inputs/gold.jsonl'): sha(run / 'inputs/gold.jsonl')},
        'llm_judge': None, 'judge_seed': 42,
        'empty_answers': report['empty_answers'], 'answer_token_limit_hits': report['final_reader_token_limit_hits']}
    summary['em_delta_pp'] = (summary['em'] - summary['seed42_em']) * 100
    evaluation = run / 'evaluations' / method
    write_rows(evaluation / 'records.jsonl', records)
    write_rows(root / 'scored_questions.jsonl', scored)
    dump(root / 'summary.json', summary)
    dump(evaluation / 'metrics.json', summary)
    marker = {'status': 'complete', 'seed': SEED, 'method': method, 'questions': 1711,
        'responses_sha256': source_hash, 'protocol_sha256': report['protocol_sha256'],
        'audit_sha256': sha(root / 'independent_audit.json'),
        'metrics_sha256': sha(evaluation / 'metrics.json'), 'records_sha256': sha(evaluation / 'records.jsonl'),
        'completed_at': now()}
    dump(root / 'completion_manifest.json', marker)
    dump(root / 'stage_status.json', marker)
    print(json.dumps(summary), flush=True)
    return summary


def generation_boundary_preflight(ctx, method, reference_protocol):
    """Execute the signed native generate functions using CPU tensor fixtures."""
    import torch

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)

        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

        def __call__(self, *, text, **kwargs):
            return Inputs(input_ids=torch.ones((len(text), 3), dtype=torch.long),
                          attention_mask=torch.ones((len(text), 3), dtype=torch.long))

        def batch_decode(self, tails, **kwargs):
            return ['CPU seed-boundary fixture' for _ in tails]

    class Model:
        def __init__(self):
            self.device = torch.device('cpu')
            self.actual_generate_calls = 0
            self.parameter = torch.zeros(1)
            self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=262144))
            self.generation_config = SimpleNamespace(eos_token_id=2)

        def parameters(self):
            yield self.parameter

        def generate(self, *, input_ids, attention_mask, **kwargs):
            self.actual_generate_calls += 1
            scores = torch.zeros((input_ids.shape[0], 97))
            for processor in kwargs.get('logits_processor', []):
                scores = processor(input_ids, scores)
            first = scores.argmax(-1, keepdim=True)
            return torch.cat((input_ids, first, torch.full_like(first, 2)), dim=1)

    protocol = copy.deepcopy(reference_protocol)
    protocol.update(seed=SEED, experiment_signature='cpu-generation-fixture')
    original_load = ctx.gallery.load_runtime
    ctx.gallery.load_runtime = lambda device: (Processor(), Model(), {'cpu_fixture': True})
    with tempfile.TemporaryDirectory(prefix='gallery1042_text_preflight_', dir='/tmp') as temporary:
        fixture = Path(temporary)
        try:
            install_runtime(ctx, method, fixture, protocol, 'cuda:1')
            model = ctx.base.LocalQwen(protocol, 'cuda:1')
            conversations = [[{'role': 'user', 'content': 'fixture ' + str(i)}] for i in range(2)]
            original_outputs = []
            for temperature in (0., .5, .7):
                original_outputs.append(model.generate(conversations, 8, sampling_temperature=temperature))
            manifest = [{'id': 'fixture/' + str(i), 'system': 'fixture', 'user_text': 'test',
                         'query_image_paths': [], 'payload_sha256': 'fixture-payload-' + str(i)} for i in range(2)]
            original_outputs.append(ctx.gallery.generate_batch((model.processor, model.model, model.environment), manifest, 256))
            if model.model.actual_generate_calls != 4:
                raise ValueError('CPU generation fixture did not execute four actual batches')
            # Simulate process loss after durable raw success and before native
            # Broker calls/reader responses or generation-ledger rows commit.
            event_path = fixture / 'text_methods' / method / 'runtime_generation_calls.jsonl'
            event_path.unlink()
            if (fixture / 'text_methods' / method / 'calls').exists():
                raise ValueError('Crash fixture unexpectedly materialized native Broker output')
            resumed = ctx.base.LocalQwen(protocol, 'cuda:1')
            recovered_outputs = [resumed.generate(conversations, 8, sampling_temperature=t) for t in (0., .5, .7)]
            recovered_outputs.append(ctx.gallery.generate_batch((resumed.processor, resumed.model, resumed.environment), manifest, 256))
            if original_outputs != recovered_outputs or resumed.model.actual_generate_calls != 0:
                raise ValueError('A durable success was regenerated or changed after the simulated crash')
            sample_path = next((fixture / 'text_methods' / method / 'raw_generation_batches').glob('*.json'))
            sample = read(sample_path)
            for wrong in ({**sample['request'], 'seed': 42},
                          {**sample['request'], 'experiment_signature': 'wrong-signed-run'},
                          {**sample['request'], 'cap': sample['request']['cap'] + 1}):
                try:
                    validate_raw_batch(sample, wrong, sample_path)
                except ValueError:
                    pass
                else:
                    raise ValueError('Wrong-contract raw recovery was accepted')
            original_sampler = ctx.native.PerRowTemperatureSampler

            class WrongSampler(original_sampler):
                def __init__(self, temperature, seed):
                    super().__init__(temperature, 42)

            ctx.native.PerRowTemperatureSampler = WrongSampler
            try:
                resumed.generate([[{'role': 'user', 'content': 'fresh wrong-seed fixture'}]], 8, sampling_temperature=.7)
            except ValueError as exc:
                if 'seed1042' not in str(exc):
                    raise
                wrong_seed_rejected = True
            else:
                raise ValueError('A categorical sampler using seed42 escaped the seed1042 guard')
            finally:
                ctx.native.PerRowTemperatureSampler = original_sampler
            events = rows(fixture / 'text_methods' / method / 'runtime_generation_calls.jsonl')
            validate_generation_events(events, protocol)
            if len(events) != 8:
                raise ValueError('Incorrect CPU generation evidence coverage')
            if any(e['generation_origin'] != 'durable_raw_recovery' for e in events):
                raise ValueError('Simulated crash recovery did not use its durable raw batches')
        finally:
            ctx.gallery.load_runtime = original_load
    return {'actual_native_generation_functions_exercised': True, 'cpu_generation_events': 8,
            'wrong_actual_sampler_seed_rejected': wrong_seed_rejected,
            'durable_raw_before_broker_cache_crash_recovery': True,
            'resumed_actual_model_calls': 0, 'wrong_raw_seed_contract_cap_rejected': True}


def preflight(ctx, method, run=RUN):
    """CPU-only source/configuration and payload regression against all old rows."""
    reference = ctx.reference / method
    p = read(reference / 'protocol.json')
    if p['seed'] != 42 or p['max_new_tokens'] != 256:
        raise ValueError('Wrong historical reader settings')
    if method == 'simplemem':
        if (ctx.adapter.official.verify_installation() != p['upstream']
                or not p['method_defaults']['ENABLE_REFLECTION']
                or p['method_defaults']['SEMANTIC_TOP_K'] != 10
                or p['context_memory_limit'] != 5):
            raise ValueError('Incorrect official SimpleMem variant')
    elif ctx.native.verify_vendor() != p['upstream']:
        raise ValueError('Native vendored source differs')
    qs = rows(ctx.reference / 'inputs/questions.jsonl')
    retrieval = {r['id']: r for r in rows(reference / 'retrieval.jsonl')}
    manifest = rows(reference / 'reader/manifest.jsonl')
    for q, payload in zip(qs, manifest):
        expected = {'id': q['id'], 'task': q['task'], 'point': q['point'],
            'system': ctx.gallery.READER_SYSTEM,
            'user_text': ctx.base.reader_text(q, retrieval[q['id']]['retrieved_context']),
            'query_image_paths': q['query_image_paths']}
        if ({k: v for k, v in payload.items() if k != 'payload_sha256'} != expected
                or ctx.base.obj(expected) != payload['payload_sha256']):
            raise ValueError('Native reader differs from the exact old payload: ' + q['id'])
    # Exercise the actual categorical sampler on CPU: different seeds, same
    # reproducibility per independent row, and zero reliance on global RNG.
    import torch
    sampler = ctx.native.PerRowTemperatureSampler
    logits = torch.zeros((2, 97))
    ids = torch.zeros((2, 1), dtype=torch.long)
    first, second = sampler(.7, SEED), sampler(.7, SEED)
    seq_a = [first(ids, logits).argmax(-1).tolist() for _ in range(16)]
    torch.manual_seed(999)
    seq_b = [second(ids, logits).argmax(-1).tolist() for _ in range(16)]
    old = sampler(.7, 42)
    seq_old = [old(ids, logits).argmax(-1).tolist() for _ in range(16)]
    if seq_a != seq_b or seq_a == seq_old or any(a[0] != a[1] for a in seq_a):
        raise ValueError('Categorical seed1042 regression failed')
    corrected, changes = serialized_location('[{"location":["London","Singapore"],"topic":"x"}]')
    if json.loads(corrected) != [{'location': 'London; Singapore', 'topic': 'x'}] or len(changes) != 1:
        raise ValueError('Historical narrow SimpleMem location repair changed')
    generation = generation_boundary_preflight(ctx, method, p)
    evidence = {'status': 'passed', 'method': method, 'seed': SEED,
        'historical_correct': EXPECTED_CORRECT[method], 'reader_payloads_exact': len(manifest),
        'historical_snapshot_files_verified': len(ctx.entries),
        'categorical_sampler_seed1042_reproducible': True, 'categorical_sampler_distinct_from_seed42': True,
        **generation, 'cpu_only': True, 'source_sha256': sha(__file__), 'updated_at': now()}
    dump(run / 'text_methods/preflight' / (method + '.json'), evidence)
    print(json.dumps(evidence), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'prepare', 'run', 'audit', 'export'))
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / 'results') or run in (REFERENCE, SM_REFERENCE):
        raise ValueError('New output must be an independent results directory')
    ctx = load_native(args.method)
    if args.phase == 'preflight':
        preflight(ctx, args.method, run)
    elif args.phase == 'prepare':
        prepare(ctx, args.method, run, args.device)
    else:
        root = run / 'text_methods' / args.method
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'process.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.phase == 'run':
                run_method(ctx, args.method, run, args.device)
            elif args.phase == 'audit':
                print(json.dumps(audit(ctx, args.method, run, args.device)), flush=True)
            else:
                export(ctx, args.method, run, args.device)


if __name__ == '__main__':
    main()
