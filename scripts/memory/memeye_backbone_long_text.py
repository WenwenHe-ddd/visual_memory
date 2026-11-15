"""Restore the prior experiment's BGE model limit without shortening evidence.

The original frozen pipeline, captions and graphs remain untouched. Retrieval
uses the original SentenceTransformer.encode behavior, including its audited
model-limit truncation. Complete captions still enter the reader and memory.
No original image is encoded and no graph is rebuilt by this extension.
"""
from __future__ import annotations

from pathlib import Path
import json
import time
from types import SimpleNamespace

import numpy as np

from scripts.memory import memeye_backbone_retrieval as strict

POLICY = {
    'name': 'original_bge_model_limit_truncation_with_audit',
    'encoder_call': 'SentenceTransformer.encode(full_inputs, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt="").',
    'long_input': 'The original tokenizer/model max_seq_length rule truncates retrieval-only encoding; all affected IDs and token counts are recorded.',
    'normalization': 'float32 model output followed by float64 L2 normalization, matching original encode_text_features.',
    'query_prefix': 'Original fixed prefix is prepended once before model tokenization, exactly as in the original experiment.',
    'reader_evidence': 'Original complete dialogue and fresh complete caption; no truncation of persisted or reader text.',
    'reference_implementation': 'scripts/analysis/prepare_all_domain_adaptive_inputs.py:encode_text_features',
    'reference_evidence': 'Frozen Card_Playlog_Test protocol records original Qwen PLAY_S2:R5 caption+dialogue at 577 tokens with retained_limit 512.',
    'new_semantic_thresholds': 0,
}


def encode_complete(model, texts, prefix='', *, ids=None, field='text'):
    """Pass full text to the original encoder and audit its retained token limit."""
    if not texts or (ids is not None and len(ids) != len(texts)):
        raise ValueError('Texts and audit identities must be nonempty and aligned')
    identities = list(ids) if ids is not None else [str(i) for i in range(len(texts))]
    inputs = [prefix + strict._text(text, field) for text in texts]
    lengths = [len(row) for row in model.tokenizer(inputs, truncation=False)['input_ids']]
    limit = int(model.max_seq_length)
    long_indices = [i for i, length in enumerate(lengths) if length > limit]
    # Do not slice strings, change prompts, or hand-build token batches. This
    # reproduces the prior experiment's exact encoder call for the full batch.
    vectors = strict.normalize(model.encode(inputs, batch_size=16, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False, prompt=''))
    if vectors.ndim != 2 or len(vectors) != len(texts):
        raise ValueError('Text encoder returned an incompatible batch')
    audit = {'field': field, 'count': len(texts), 'max_tokens': max(lengths), 'token_limit': limit,
             'batch_size': 16, 'truncation_side': model.tokenizer.truncation_side,
             'truncated_inputs': len(long_indices), 'affected_inputs': [
                 {'id': identities[i], 'field': field, 'original_tokens': lengths[i],
                  'encoded_tokens': limit, 'retained_limit': limit, 'dropped_tokens': lengths[i] - limit}
                 for i in long_indices]}
    return vectors, audit


def _verified_completed_graphs(output, device):
    """Recreate the exact strict-stage contract before loading existing graphs."""
    inputs = output / 'inputs'
    preparation = strict.read(inputs / 'protocol.json')
    root = strict.read(output / 'protocol.json')
    if strict.digest(root['contract']) != root['experiment_signature']:
        raise ValueError('Root experiment signature changed')
    strict.verify_hashes(root['contract']['source_sha256'])
    strict.verify_hashes(preparation['source_sha256'])
    strict.verify_hashes({name: value for name, value in preparation['artifact_sha256'].items()
                          if name != 'evaluation.jsonl'}, inputs)
    memories, questions, model_name = strict.attach_captions(
        strict.lines(inputs / 'source_memories.jsonl'), strict.lines(inputs / 'source_questions.jsonl'),
        strict.lines(output / 'captions.jsonl'))
    domains = sorted({row['task'] for row in memories})
    if (model_name != root['contract']['model'] or len(memories) != 438 or len(questions) != 371
            or len(domains) != 8 or {row['task'] for row in questions} != set(domains)
            or sum(bool(row['query_images']) for row in questions) != 8):
        raise ValueError('Fresh backbone inputs differ from full MemEye run')
    # This list intentionally reproduces the immutable strict module's contract;
    # adding this extension here would invalidate the already built graphs.
    paths = [output / 'protocol.json', inputs / 'protocol.json', inputs / 'source_memories.jsonl',
             inputs / 'source_questions.jsonl', output / 'captions.jsonl', Path(strict.__file__),
             strict.ROOT / 'scripts/graph/caption_ward_visual.py',
             strict.ROOT / 'scripts/graph/visual_image_encoder.py',
             strict.ROOT / 'scripts/memory/caption_visual_master.py',
             strict.ROOT / 'scripts/analysis/prepare_all_domain_adaptive_inputs.py',
             strict.ROOT / 'scripts/analysis/probe_caption_visual_online.py']
    for domain in domains:
        basis = preparation['basis_by_domain'][domain]
        strict.verify_hashes({basis['path']: basis['sha256'], basis['protocol_path']: basis['protocol_sha256']})
        reference = strict.read(basis['protocol_path'])
        if (reference['image_encoder'] != preparation['image_encoder']
                or reference['text_encoder']['files_sha256'] != preparation['text_encoder']['files_sha256']
                or reference['query_prefix'] != preparation['query_prefix']):
            raise ValueError('Frozen encoder metadata differs between domains')
        rows = [row for row in memories if row['task'] == domain]
        if [row['insertion_index'] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError('Chronological source order changed')
        paths.extend([Path(basis['path']), Path(basis['protocol_path'])])
    hashes = {**preparation['source_sha256'], **root['contract']['source_sha256'],
              **{str(inputs / name): value for name, value in preparation['artifact_sha256'].items()
                 if name != 'evaluation.jsonl'}, **{str(path): strict.sha(path) for path in paths}}
    original_contract = {'source_sha256': hashes, 'device': device, 'caption_model': model_name, 'top_k': 5,
                         'source_run_id': root['run_id'], 'source_experiment_signature': root['experiment_signature'],
                         'memory_count': 438, 'question_count': 371, 'domains': domains,
                         'image_encoder': preparation['image_encoder'], 'text_encoder': preparation['text_encoder'],
                         'query_prefix': preparation['query_prefix']}
    original_signature = strict.digest(original_contract)
    graphs, artifacts, summaries = {}, [], []
    for domain in domains:
        directory = output / 'domains' / domain
        build_path = directory / 'build_protocol.json'
        if not build_path.is_file():
            raise ValueError('Long-text recovery requires all eight completed memory builds')
        build = strict.read(build_path)
        if build['source_signature'] != original_signature:
            raise ValueError('Completed memory does not belong to the original strict contract')
        strict.verify_hashes(build['artifact_sha256'])
        graph = strict.read(directory / 'memory/graph.json')
        rows = [row for row in memories if row['task'] == domain]
        if ([node['id'] for node in graph['image_nodes']] != [row['image_id'] for row in rows]
                or any(node['caption'] != row['caption'] or node['dialogue'] != row['dialogue']
                       or node['timestamp'] != row['timestamp'] or len(node['visual_coords']) != 2
                       for node, row in zip(graph['image_nodes'], rows))):
            raise ValueError('Completed graph text or chronology differs from fresh inputs')
        graphs[domain] = graph, np.load(directory / 'memory/master_sums.npy', allow_pickle=False)
        artifacts.extend(Path(path) for path in build['artifact_sha256'])
        artifacts.append(build_path)
        summaries.append({'task': domain, 'images': len(rows), 'masters': len(graph['masters']),
                          'clustering_seconds': build['clustering_seconds']})
    return preparation, root, memories, questions, graphs, artifacts, summaries, original_contract


def complete_retrieval(output, device='cuda:0'):
    """Resume only retrieval using validated graphs and fresh complete captions."""
    output = Path(output).resolve()
    preparation, root, memories, questions, graphs, artifacts, summaries, original = _verified_completed_graphs(output, device)
    extension_paths = [Path(__file__).resolve(),
                       strict.ROOT / 'scripts/evaluation/memeye_backbone_completion.py',
                       strict.ROOT / 'scripts/visual_memory_pipeline.py',
                       output / 'long_text_transition.json']
    source_hashes = {**original['source_sha256'], **{str(path): strict.sha(path) for path in artifacts},
                     **{str(path): strict.sha(path) for path in extension_paths}}
    contract = {'original_strict_stage_signature': strict.digest(original), 'source_sha256': source_hashes,
                'source_run_id': root['run_id'], 'source_experiment_signature': root['experiment_signature'],
                'device': device, 'long_text_policy': POLICY}
    signature = strict.digest(contract)
    marker = output / 'retrieval_protocol.json'
    if marker.exists():
        prior = strict.read(marker)
        if prior['contract'] != contract or prior['experiment_signature'] != signature:
            raise ValueError('Existing retrieval differs from this long-text policy or its sources')
        strict.verify_hashes(prior['output_sha256'])
        return prior
    runtime = SimpleNamespace(device=device)
    strict.configure_device(runtime)
    text_encoder = strict.load_text_encoder(preparation['text_encoder'], runtime.device)
    if text_encoder.max_seq_length != preparation['text_encoder']['max_seq_length']:
        raise ValueError('Actual BGE token limit differs from the frozen reference')
    clip, tokenizer, clip_limit = strict.load_clip_text_encoder(preparation['image_encoder'], runtime.device)
    started = time.perf_counter()
    results, encoding = [], []
    for domain in original['domains']:
        rows = [row for row in memories if row['task'] == domain]
        qs = [row for row in questions if row['task'] == domain]
        leaf_vectors, leaf_info = encode_complete(text_encoder, [row['retrieval_text'] for row in rows],
                                                   ids=[row['id'] for row in rows], field='dialogue+caption')
        query_texts, query_ids, statements = [], [], []
        for question in qs:
            candidates = [question['candidate_statements'][key] for key in strict.LETTERS]
            statements.extend(candidates)
            query_texts.extend([question['retrieval_query'], *[
                question['retrieval_query'] + '\nCandidate answer: ' + candidate for candidate in candidates]])
            query_ids.extend([question['id'] + suffix for suffix in ('/baseline', '/A', '/B', '/C', '/D')])
        query_vectors, query_info = encode_complete(text_encoder, query_texts, preparation['query_prefix'],
                                                    ids=query_ids, field='query')
        candidate_vectors, candidate_info = strict._clip_candidates(clip, tokenizer, clip_limit, statements)
        query_vectors = query_vectors.reshape(len(qs), 5, -1)
        candidate_vectors = candidate_vectors.reshape(len(qs), 4, -1)
        graph, sums = graphs[domain]
        for index, question in enumerate(qs):
            results.append(strict.rank_question(question, rows, graph, sums, query_vectors[index],
                                                 leaf_vectors, candidate_vectors[index]))
        encoding.append({'task': domain, 'leaf': leaf_info, 'query': query_info, 'candidate': candidate_info})
        print(json.dumps({'stage': 'backbone_long_text_retrieval', 'task': domain,
                          'questions': len(qs), 'truncated_leaves': leaf_info['truncated_inputs'],
                          'truncated_queries': query_info['truncated_inputs']}), flush=True)
    if len(results) != 371 or {row['id'] for row in results} != {row['id'] for row in questions}:
        raise ValueError('Incomplete retrieval coverage')
    strict.write_lines(output / 'offline/memories.jsonl', [
        {key: value for key, value in row.items() if key not in ('image_path', 'image_sha256')}
        for row in memories])
    strict.write_lines(output / 'retrieval_results.jsonl', results)
    artifacts.extend([output / 'offline/memories.jsonl', output / 'retrieval_results.jsonl'])
    strict.verify_hashes(source_hashes)
    output_hashes = {str(path): strict.sha(path) for path in artifacts}
    protocol = {'status': 'complete', 'contract': contract, 'experiment_signature': signature,
                'source_run_id': root['run_id'], 'source_experiment_signature': root['experiment_signature'],
                'source_sha256': source_hashes, 'output_sha256': output_hashes, 'artifact_sha256': output_hashes,
                'results_sha256': strict.sha(output / 'retrieval_results.jsonl'),
                'caption_model': root['contract']['model'], 'questions': len(results), 'memory_nodes': len(memories),
                'domains': summaries, 'encoding_audits': encoding, 'long_text_policy': POLICY,
                'execution_device': runtime.device, 'elapsed_seconds': time.perf_counter() - started,
                'original_strict_stage_signature': strict.digest(original),
                'completed_graphs_reused_within_same_run': True, 'prior_run_graphs_reused': False,
                'image_forwards_this_extension': 0, 'old_image_reencodes': 0,
                'gold_loaded_for_ranking': False, 'per_image_visual_embeddings_saved': False,
                'leaf_visual_dimension': 2, 'candidate_statements_in_reader_evidence': False,
                'manual_similarity_thresholds': 0, 'score_fusion_weights': 0, 'leaf_distance_used': False,
                'identical_ordered_top5': sum(row['top5_same_order'] for row in results)}
    strict.dump(marker, protocol)
    return protocol
