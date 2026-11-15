"""Build visual-only masters from pinned images and unchanged Qwen captions.

Each task is a separate memory library, traversed in source dialogue order.
Only the current image is encoded; its vector is never written to disk.
"""

import fcntl
import hashlib
import json
from pathlib import Path

from scripts.data.generate import signature as caption_signature
from scripts.data.dialogues import load_dialogues
from scripts.graph.graph_json_format import readable_dumps
from scripts.graph.visual_embedding_graph import VisualEmbeddingGraph
from scripts.graph.visual_embedding_storage import (
    cleanup_unreferenced_vectors, externalize_snapshot, hydrate_snapshot,
)
from scripts.graph.visual_image_encoder import ImageEncoder, inspect_model_source

ROOT = Path(__file__).resolve().parents[2]
BUILD_FORMAT = 'visual_embedding_graph_build_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, key, label):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f'duplicate {label}: {row[key]}')
        result[row[key]] = row
    return result


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content, encoding='utf-8')
    tmp.replace(path)


def write_json(path, value):
    atomic_text(path, readable_dumps(value) + '\n')


def prepare_inputs(natural_root, task=None):
    """Whitelist memory images/text/time; never read QA labels into the graph."""
    natural_root = Path(natural_root)
    files = [natural_root / name for name in
             ('protocol.json', 'images.jsonl', 'memories.jsonl', 'captions.jsonl')]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'missing {path}; prepare the natural caption cache first')
    config = json.loads(files[0].read_text())
    images = unique(read_jsonl(files[1]), 'id', 'image ID')
    memories = read_jsonl(files[2])
    unique(memories, 'id', 'memory ID')
    captions = unique(read_jsonl(files[3]), 'id', 'caption ID')
    expected_signature = caption_signature(config)
    if set(captions) - set(images):
        raise ValueError('caption cache contains unknown image IDs')
    for row in captions.values():
        if (row.get('protocol_signature') != expected_signature
                or not isinstance(row.get('caption'), str) or not row['caption'].strip()):
            raise ValueError('invalid or incompatible cached Qwen caption: ' + row['id'])
    tasks = sorted({m['task'] for m in memories})
    if task is not None:
        if task not in tasks:
            raise ValueError('unknown task: ' + task)
        tasks = [task]
    for name in tasks:
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid task directory name')
    dialogues, source_paths = load_dialogues(natural_root, set(tasks))
    dates = {}
    for path in source_paths[3:]:
        data = json.loads(path.read_text())
        name = path.stem.removesuffix('_Open')
        for session in data['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = name + '/' + turn['round']
                if mid in dialogues:
                    dates[mid] = session.get('date')
    by_task = {}
    verified_images = set()
    for name in tasks:
        selected = [m for m in memories if m['task'] == name]
        expected_order = [mid for mid in dialogues if mid.startswith(name + '/')]
        if [m['id'] for m in selected] != expected_order:
            raise ValueError('memory order differs from original dialogue order: ' + name)
        records = []
        for memory in selected:
            timestamp = memory.get('date')
            if (timestamp != dates[memory['id']]
                    or timestamp is not None and not isinstance(timestamp, str)):
                raise ValueError('memory date disagrees with original dialogue: ' + memory['id'])
            for index, image_id in enumerate(memory['images'], 1):
                if image_id not in images or image_id not in captions:
                    raise ValueError('missing image or full caption: ' + image_id)
                path = Path(images[image_id]['path'])
                if not path.is_absolute():
                    path = ROOT / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                # Content-addressed input IDs pin the actual pixels, not just paths.
                if image_id not in verified_images:
                    if sha(path) != image_id:
                        raise ValueError('image content hash changed: ' + str(path))
                    verified_images.add(image_id)
                records.append(dict(id=f"{memory['id']}#image{index}",
                                    source_image_id=image_id, image_path=str(path),
                                    dialogue=dialogues[memory['id']],
                                    caption=captions[image_id]['caption'], timestamp=timestamp))
        by_task[name] = records
    return by_task, {str(p.resolve()): sha(p) for p in files}


def protocol(args, source_hashes):
    code_files = [Path(__file__), ROOT / 'scripts/graph/visual_embedding_graph.py',
                  ROOT / 'scripts/graph/visual_image_encoder.py',
                  ROOT / 'scripts/graph/visual_embedding_storage.py',
                  ROOT / 'scripts/visual_memory_pipeline.py']
    return dict(format=BUILD_FORMAT, natural_root=str(args.natural_root.resolve()),
                input_hashes=source_hashes,
                code_hashes={str(p.resolve()): sha(p) for p in code_files},
                encoder=inspect_model_source(args.visual_encoder,
                                             revision=args.visual_encoder_revision),
                device=args.visual_device,
                similarity='raw cosine [-1, 1]; normalized image versus normalized master mean',
                admission='strictly greater than threshold; highest eligible cosine; creation-order ties',
                initial_threshold='0 for empty memory; max cosine to existing masters otherwise',
                updated_threshold='norm of mean of unit image vectors',
                distance='new edge exact after center update; old edges add norm(old_mean)-dot(old_mean,new_center)',
                distance_accuracy='individual old edges approximate; mean distance exact; no clipping',
                memory_scope='one independent library per task, in original dialogue order',
                image_embedding_storage='none: only master aggregate vectors persist',
                master_vector_storage='float64 NPY matrix; relative file path and master row references',
                caption_source='unchanged validated natural Qwen caption cache',
                timestamp_source='original session date, null only when absent',
                online_supported=False)


def verify_resume(graph, logs, records):
    snapshot = graph.to_dict()
    nodes = snapshot['image_nodes']
    if len(nodes) > len(records) or [n['id'] for n in nodes] != [r['id'] for r in records[:len(nodes)]]:
        raise ValueError('checkpoint is not an exact prefix of the source image memories')
    if [r['image_id'] for r in logs] != [n['id'] for n in nodes]:
        raise ValueError('checkpoint decision log coverage/order mismatch')
    for node, record in zip(nodes, records):
        for field in ('image_path', 'dialogue', 'caption', 'timestamp'):
            if node[field] != record[field]:
                raise ValueError('checkpoint changed source ' + field + ': ' + node['id'])


def save_checkpoint(task_dir, graph, logs, protocol_digest, records_digest, encoder_metadata):
    graph.validate()
    snapshot = externalize_snapshot(graph.to_dict(), task_dir)
    state = dict(format=BUILD_FORMAT, protocol_digest=protocol_digest,
                 records_digest=records_digest, encoder_metadata=encoder_metadata,
                 graph=snapshot, write_log=logs)
    # The immutable vector blob is committed first. This atomic checkpoint then
    # pins that exact blob; stale exports can be regenerated after a crash.
    write_json(task_dir / 'checkpoint.json', state)
    write_json(task_dir / 'graph.json', snapshot)
    atomic_text(task_dir / 'write_log.jsonl', ''.join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in logs))
    cleanup_unreferenced_vectors(task_dir)


def build(args, encoder_factory=ImageEncoder):
    """Resume only exact input prefixes under an unchanged encoder/protocol."""
    args.natural_root, args.graph_root = Path(args.natural_root), Path(args.graph_root)
    output = args.graph_root.resolve()
    natural = args.natural_root.resolve()
    if output == natural or output in natural.parents or natural in output.parents:
        raise ValueError('visual graph output must be separate from the natural caption cache')
    records_by_task, source_hashes = prepare_inputs(args.natural_root, args.task)
    run = protocol(args, source_hashes)
    run_digest = digest(run)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another visual graph build is writing this output directory') from exc
        config_path = output / 'run_config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != run:
                raise ValueError('visual graph sources/encoder/code changed; use a new output directory')
        elif any(output.iterdir()):
            if any(p.name != '.build.lock' for p in output.iterdir()):
                raise ValueError('output directory contains another experiment; choose an empty directory')
        write_json(config_path, run)
        # A remote model ID can resolve to a different commit on a later run.
        # Pin its loaded identity across *all* tasks, including completed tasks
        # that do not instantiate an encoder during this invocation.
        encoder_pin_path = output / 'encoder_metadata.json'
        pinned_encoder = (json.loads(encoder_pin_path.read_text())
                          if encoder_pin_path.exists() else None)
        for checkpoint_path in output.glob('*/checkpoint.json'):
            state = json.loads(checkpoint_path.read_text())
            if state.get('format') != BUILD_FORMAT or state.get('protocol_digest') != run_digest:
                raise ValueError('incompatible visual graph checkpoint: ' + str(checkpoint_path))
            metadata = state.get('encoder_metadata')
            if metadata is not None:
                if pinned_encoder is not None and pinned_encoder != metadata:
                    raise ValueError('encoder metadata differs across task checkpoints')
                pinned_encoder = metadata
        if pinned_encoder is not None:
            write_json(encoder_pin_path, pinned_encoder)
        encoder = None
        task_summaries = {}
        for task, records in records_by_task.items():
            task_dir = output / task
            state_path = task_dir / 'checkpoint.json'
            records_digest = digest(records)
            graph, logs, encoder_metadata = VisualEmbeddingGraph(), [], None
            if state_path.exists() and not args.from_start:
                state = json.loads(state_path.read_text())
                if (state.get('format') != BUILD_FORMAT
                        or state.get('protocol_digest') != run_digest
                        or state.get('records_digest') != records_digest):
                    raise ValueError('incompatible visual graph checkpoint: ' + task)
                graph = VisualEmbeddingGraph.from_dict(hydrate_snapshot(state['graph'], task_dir))
                logs, encoder_metadata = state['write_log'], state['encoder_metadata']
                verify_resume(graph, logs, records)
            elif not args.from_start and task_dir.exists() and any(task_dir.iterdir()):
                raise ValueError('task files exist without an authoritative checkpoint: ' + task)
            start = len(logs)
            stop = min(len(records), start + args.limit) if args.limit else len(records)
            for record in records[start:stop]:
                if encoder is None:
                    encoder = encoder_factory(args.visual_encoder,
                                              revision=args.visual_encoder_revision,
                                              device=args.visual_device,
                                              local_files_only=args.visual_local_files_only,
                                              cache_dir=str(ROOT / 'models/huggingface'))
                if pinned_encoder is not None and pinned_encoder != encoder.metadata:
                    raise ValueError('loaded encoder differs from the output directory encoder')
                if encoder_metadata is not None and encoder_metadata != encoder.metadata:
                    raise ValueError('loaded encoder differs from checkpoint encoder')
                encoder_metadata = encoder.metadata
                if pinned_encoder is None:
                    pinned_encoder = encoder_metadata
                    write_json(encoder_pin_path, pinned_encoder)
                vector = encoder.encode(record['image_path'])
                decision = graph.insert(record['id'], vector, dialogue=record['dialogue'],
                                        caption=record['caption'], timestamp=record['timestamp'],
                                        image_path=record['image_path'])
                del vector # 减少存储的关键
                logs.append(decision)
                save_checkpoint(task_dir, graph, logs, run_digest, records_digest, encoder_metadata)
                print(f"{task}: {len(logs)}/{len(records)} images, {len(graph.masters)} masters", flush=True)
            # Regenerate exports even when a complete checkpoint required no encoder call.
            save_checkpoint(task_dir, graph, logs, run_digest, records_digest, encoder_metadata)
            verify_resume(graph, logs, records)
            summary = dict(task=task, expected_images=len(records), processed_images=len(logs),
                           new_images=len(logs) - start, masters=len(graph.masters),
                           complete=len(logs) == len(records),
                           persistent_image_embeddings=0,
                           individual_old_edge_distances='approximate',
                           mean_edge_distance_check='passed',
                           encoder=encoder_metadata)
            write_json(task_dir / 'summary.json', summary)
            task_summaries[task] = summary
        result = dict(format=BUILD_FORMAT, tasks=task_summaries,
                      complete=all(t['complete'] for t in task_summaries.values()))
        write_json(output / 'summary.json', result)
        return result
