"""Caption-Ward memory construction and visual master answer query expansion.

Called by visual_memory_pipeline.py with --memory caption_visual_master.
The current input adapters use the frozen Home examples. Online performs
retrieval only; it does not call the QA reader or modify full-domain results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
from scripts.analysis.probe_caption_visual_online import TOKENIZER_HASHES, normalize, read_json, sha, write_json
from scripts.analysis.prepare_all_domain_adaptive_inputs import save_memory
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder

HOME = 'Home_Renovation_Interior_Design'
FROZEN = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain' / HOME
PILOT = ROOT / 'reports/home_caption_ward_visual_20261020'


def load_text_encoder(metadata, device):
    from sentence_transformers import SentenceTransformer
    path = Path(metadata['model_path'])
    for filename, expected in metadata['files_sha256'].items():
        if sha(path / filename) != expected:
            raise ValueError(f'BGE checkpoint differs: {filename}')
    return SentenceTransformer(str(path), device=device, local_files_only=True)


def encode_text(model, texts, prefix=''):
    inputs = [prefix + text for text in texts]
    lengths = [len(tokens) for tokens in model.tokenizer(inputs, truncation=False)['input_ids']]
    if max(lengths) > model.max_seq_length:
        raise ValueError('BGE input exceeds its token limit; refusing silent truncation.')
    return normalize(model.encode(inputs, batch_size=16, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False, prompt=''))


def offline(args, *, basis_override=None, basis_metadata=None):
    # The random entry supplies both; the existing PCA call keeps its original
    # basis loading, output format and numerical path.
    if (basis_override is None) != (basis_metadata is None):
        raise ValueError('An overridden basis requires its construction metadata.')
    output = args.output.resolve()
    if basis_override is None and (output / '.random_2d.lock').exists():
        raise FileExistsError('Random-2D construction is using this --output; use a separate directory.')
    if basis_override is None and (output / 'protocol.json').is_file():
        previous = read_json(output / 'protocol.json')
        if previous.get('basis_kind') == 'random-2d':
            raise FileExistsError('PCA construction cannot overwrite Random-2D memory; use a separate --output.')
    configure_device(args)
    protocol = read_json(args.models_protocol)
    rows = read_json(args.input_manifest)['home']
    requested = set(args.image_files)
    if len(requested) != len(args.image_files):
        raise ValueError('Duplicate --image-files entries.')
    # The source order, rather than the CLI argument order, determines arrivals.
    rows = [row for row in rows if row['filename'] in requested]
    if {row['filename'] for row in rows} != requested:
        raise ValueError('Some selected image files do not exist in the Home manifest.')
    if len(rows) < 2:
        raise ValueError('Select at least two images for this debugging example.')
    canonical = {row['image_id']: row for row in read_json(FROZEN / 'inputs.json')['leaves']}
    for row in rows:
        for key in ('caption', 'dialogue', 'filename'):
            if row[key] != canonical[row['image_id']][key]:
                raise ValueError(f'Input text differs from the frozen experiment: {row["image_id"]}')

    basis = (np.load(args.basis, allow_pickle=False) if basis_override is None
             else np.asarray(basis_override, dtype=np.float64))
    memory = CaptionWardVisualMemory(basis)
    visual_meta = protocol['image_encoder']
    image_encoder = ImageEncoder(visual_meta['model_name'], revision=visual_meta['revision'],
                                 device=args.device, local_files_only=True)
    if {k: v for k, v in image_encoder.metadata.items() if k != 'device'} != {
        k: v for k, v in visual_meta.items() if k != 'device'
    }:
        raise ValueError('CLIP vision encoder differs from the frozen experiment.')
    text_encoder = load_text_encoder(protocol['text_encoder'], args.device)
    audits = []
    for step, row in enumerate(rows, 1):
        if sha(row['source_path']) != row['image_sha256']:
            raise ValueError(f'Incoming image content changed: {row["filename"]}')
        # OFFLINE BREAKPOINT: inspect only the current incoming image vector.
        visual_embedding = normalize(image_encoder.encode(row['source_path']))
        caption_embedding = encode_text(text_encoder, [row['caption']])[0]
        # Step into add() to inspect Ward partitioning and master visual updates.
        audit = memory.add(row['image_id'], row['caption'], caption_embedding,
                           visual_embedding, dialogue=row['dialogue'], timestamp=row['timestamp'])
        del visual_embedding, caption_embedding
        audit.update(step=step, filename=row['filename'])
        audits.append(audit)
        print(json.dumps({'stage': 'offline', 'step': step, 'image': row['filename'],
                          'masters': len(memory.masters), 'splits': audit['split_master_ids']},
                         ensure_ascii=False), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    graph, sums = save_memory(memory, output / 'memory')
    # Keep names for debugging outside the memory; no image paths or image copies.
    write_json(output / 'examples.json', [{k: row[k] for k in ('image_id', 'memory_id', 'filename')}
                                         for row in rows])
    with (output / 'insertion_audit.jsonl').open('w') as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + '\n')
    basis_record = ({'basis_source': str(args.basis.resolve()),
                     'basis_sha256': sha(args.basis), 'basis_recalibrated': False}
                    if basis_metadata is None else
                    {**basis_metadata, 'basis_sha256': sha(output / 'memory/basis.npy')})
    write_json(output / 'protocol.json', {
        'purpose': 'small_subset_debug; not the full-domain benchmark', 'images': len(rows),
        'masters': len(memory.masters), 'insertion_order': [row['image_id'] for row in rows],
        'image_encoder': image_encoder.metadata, 'text_encoder': protocol['text_encoder'],
        'execution_devices': {'image_encoder': str(image_encoder.device),
                              'text_encoder': str(text_encoder.device)},
        'query_prefix': protocol['query_prefix'], **basis_record,
        'leaf_visual_dimension': 2, 'old_image_reencodes': 0, 'image_copies': 0,
        'per_image_visual_embeddings_saved': False,
        'memory_sha256': {name: sha(output / 'memory' / name)
                          for name in ('graph.json', 'basis.npy', 'master_sums.npy')},
        'note': 'A small subset can have different clusters and rankings from the 89-image Home experiment.',
    })
    assert len(graph['image_nodes']) == len(rows) and sums.shape[1] == 512
    print(f'Offline complete: {output / "memory/graph.json"}', flush=True)


def load_clip_text_encoder(metadata, device):
    import torch
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    path = Path(metadata['model_name'])
    for filename, expected in {**metadata['files_sha256'], **TOKENIZER_HASHES}.items():
        if sha(path / filename) != expected:
            raise ValueError(f'CLIP checkpoint differs: {filename}')
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model, loading = CLIPTextModelWithProjection.from_pretrained(
        str(path), config=config.text_config, local_files_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('CLIP text/projection weights are incomplete.')
    model.to(device=device, dtype=torch.float32).eval()
    return model, tokenizer, config.text_config.max_position_embeddings


def choose_master_answers(centers, statements, model, tokenizer, token_limit):
    import torch
    texts = [statements[key] for key in 'ABCD']
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
    if encoded['input_ids'].shape[1] > token_limit:
        raise ValueError('CLIP candidate exceeds its token limit.')
    encoded = encoded.to(next(model.parameters()).device)
    with torch.inference_mode():
        candidate_vectors = normalize(model(**encoded, return_dict=True).text_embeds.double().cpu().numpy())
    # ONLINE BREAKPOINT 1: rows are masters; columns are A/B/C/D hypotheses.
    clip_scores = centers @ candidate_vectors.T
    selected_choices = np.argmax(clip_scores, axis=1)
    return clip_scores, selected_choices


def retrieve_question(question, statements, masters, centers, nodes, leaf_vectors,
                      leaf_master_indices, text_encoder, clip_encoder, tokenizer,
                      clip_limit, query_prefix, top_k):
    clip_scores, selected_choices = choose_master_answers(
        centers, statements, clip_encoder, tokenizer, clip_limit)
    # ONLINE BREAKPOINT 2: four expanded queries are shared by matching masters.
    expanded_queries = [question['retrieval_query'] + '\nCandidate answer: ' + statements[k]
                        for k in 'ABCD']
    query_vectors = encode_text(text_encoder, [question['retrieval_query'], *expanded_queries], query_prefix)
    scores_by_query = query_vectors @ leaf_vectors.T
    # Each leaf uses only the query chosen by its own master; no distance term.
    leaf_scores = scores_by_query[1 + selected_choices[leaf_master_indices], np.arange(len(nodes))]
    # ONLINE BREAKPOINT 3: global ranking across all masters' leaves.
    ranking = np.argsort(-leaf_scores, kind='stable')
    baseline_ranking = np.argsort(-scores_by_query[0], kind='stable')
    master_choices = [{
        'master_id': master['id'], 'n': master['n'],
        'clip_candidate_scores': dict(zip('ABCD', clip_scores[i].tolist())),
        'selected_choice': 'ABCD'[selected_choices[i]],
        'expanded_query': expanded_queries[selected_choices[i]],
    } for i, master in enumerate(masters)]
    # Labels are added only after ranking; they never select a hypothesis.
    gold = set(question['gold_ids'])
    conditions = []
    for name, order, scores in [('baseline_question_only', baseline_ranking, scores_by_query[0]),
                                 ('master_answer_query', ranking, leaf_scores)]:
        top = [{'rank': rank, 'image_id': nodes[i]['id'],
                'memory_id': nodes[i]['id'].rsplit('#image', 1)[0],
                'master_id': nodes[i]['master_id'], 'score': float(scores[i]),
                'master_choice': 'ABCD'[selected_choices[leaf_master_indices[i]]],
                'dialogue': nodes[i]['dialogue'], 'caption': nodes[i]['caption']}
               for rank, i in enumerate(order[:top_k], 1)]
        conditions.append({'name': name, 'top5': top,
                           'gold_hits': sum(leaf['memory_id'] in gold for leaf in top)})
    return {
        'question_id': question['id'], 'question': question['question'], 'options': question['options'],
        'query_image_paths': question['query_image_paths'], 'candidate_statements': statements,
        'master_choices': master_choices, 'conditions': conditions,
        'gold_ids': question['gold_ids'],
        'gold_ids_available_in_subset': sorted(gold & {node['id'].rsplit('#image', 1)[0] for node in nodes}),
        'reader_called': False, 'leaf_distance_used': False,
    }


def online(args):
    configure_device(args)
    state = args.memory_dir.resolve()
    if not (state / 'graph.json').exists():
        raise FileNotFoundError(f'Run offline --memory caption_visual_master first: {state / "graph.json"}')
    protocol = read_json(state.parent / 'protocol.json')
    for name, expected in protocol['memory_sha256'].items():
        if sha(state / name) != expected:
            raise ValueError(f'Offline memory changed since export: {name}')
    graph = read_json(state / 'graph.json')
    masters, nodes = graph['masters'], graph['image_nodes']
    sums = np.load(state / 'master_sums.npy', allow_pickle=False)
    centers = normalize([sums[master['visual_mean_ref']['row']] * master['visual_mean_ref']['scale']
                         for master in masters]) # 检索加载时才计算均值
    master_lookup = {master['id']: index for index, master in enumerate(masters)}
    leaf_master_indices = np.asarray([master_lookup[node['master_id']] for node in nodes])
    questions = read_json(args.question_source)['questions']
    by_id = {q['id']: q for q in questions}
    question_ids = [qid if '/' in qid else f'{HOME}/{qid}' for qid in args.question_ids]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError('Duplicate --question-ids entries.')
    questions = [by_id[qid] for qid in question_ids]
    candidates = {row['id']: row['candidate_statements']
                  for line in args.candidates.read_text().splitlines() if line.strip()
                  for row in [json.loads(line)]}
    text_encoder = load_text_encoder(protocol['text_encoder'], args.device)
    leaf_texts = [node['dialogue'] + '\nMemory description: ' + node['caption'] for node in nodes]
    # Rebuild from this newly constructed graph, never the full-domain cache.
    leaf_vectors = encode_text(text_encoder, leaf_texts)
    clip_encoder, tokenizer, clip_limit = load_clip_text_encoder(protocol['image_encoder'], args.device)
    results = []
    for question in questions:
        result = retrieve_question(question, candidates[question['id']], masters, centers, nodes,
                                   leaf_vectors, leaf_master_indices, text_encoder, clip_encoder,
                                   tokenizer, clip_limit, protocol['query_prefix'], args.top_k)
        results.append(result)
        print(json.dumps({'stage': 'online', 'question': question['id'],
                          'top5': [row['memory_id'] for row in result['conditions'][1]['top5']]},
                         ensure_ascii=False), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'retrieval_results.json', {
        'purpose': 'small_subset_debug; not full-domain evaluation',
        'memory_dir': str(state), 'memory_graph_sha256': sha(state / 'graph.json'),
        'images': len(nodes), 'masters': len(masters), 'questions': results,
        'execution_devices': {'clip_text_encoder': str(next(clip_encoder.parameters()).device),
                              'text_encoder': str(text_encoder.device)},
    })
    print(f'Online complete: {args.output / "retrieval_results.json"}', flush=True)



def configure_device(args):
    """Use the requested encoder device; never silently fall back from CUDA."""
    import torch
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use a GPU-enabled session, or explicitly set --device cpu.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable.')
        device = torch.device('cuda', index)
    args.device = str(device)
    torch.set_num_threads(4)
