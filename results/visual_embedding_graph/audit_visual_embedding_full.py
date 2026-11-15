#!/usr/bin/env python3
"""Independent offline result audit; never loads an encoder/member embedding.

Rebuilds the source-record inventory directly from manifests and dialogue JSON,
then independently replays scalar decisions and center-update formulae. Existing
load_graph is used only as an additional format/reader compatibility check.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

import numpy as np

ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(ROOT))
from scripts.graph.visual_embedding_storage import load_graph


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Auditor:
    def __init__(self):
        self.checked = Counter()
        self.errors = []
        self.warnings = []
        self.max_abs_error = Counter()

    def check(self, truth, category, detail=''):
        self.checked[category] += 1
        if not truth:
            self.errors.append({'check': category, 'detail': str(detail)})

    def equal(self, got, expected, category, detail=''):
        self.check(got == expected, category, detail)

    def close(self, got, expected, category, detail='', tol=1e-9):
        difference = abs(got - expected)
        self.max_abs_error[category] = max(self.max_abs_error[category], difference)
        self.check(math.isfinite(got) and difference <= tol, category,
                   f'{detail}: actual={got!r}, expected={expected!r}, abs_error={difference!r}')


def source_records(natural, auditor):
    config = read(natural / 'protocol.json')
    image_rows, memory_rows, caption_rows = [lines(natural / name) for name in
        ('images.jsonl', 'memories.jsonl', 'captions.jsonl')]
    images = {row['id']: row for row in image_rows}
    memories = {row['id']: row for row in memory_rows}
    captions = {row['id']: row for row in caption_rows}
    for label, rows, keyed in [('images', image_rows, images), ('memories', memory_rows, memories), ('captions', caption_rows, captions)]:
        auditor.equal(len(rows), len(keyed), 'source_unique_ids', label)
    source_image_ids = {rel: row['id'] for row in image_rows for rel in row['source_paths']}
    signature_fields = ('provenance', 'caption_prompt', 'cue_prompt', 'image_max_pixels',
                        'caption_max_new_tokens', 'cue_max_new_tokens', 'seed', 'do_sample', 'dtype')
    signature = hashlib.sha256(json.dumps({k: config[k] for k in signature_fields}, sort_keys=True).encode()).hexdigest()
    for row in caption_rows:
        auditor.equal(row['protocol_signature'], signature, 'caption_protocol_identity', row['id'])
        auditor.check(isinstance(row['caption'], str) and bool(row['caption'].strip()), 'nonempty_caption', row['id'])
    tasks = sorted({row['task'] for row in memory_rows})
    by_task = {}
    checked_pixels = set()
    hashes = {}
    for task in tasks:
        sources = [s for s in config['source_files'] if Path(s['path']).name == task + '_Open.json']
        auditor.equal(len(sources), 1, 'one_pinned_dialogue_source', task)
        source = sources[0]
        source_path = (ROOT / source['path']).resolve()
        actual_hash = sha(source_path)
        auditor.equal(actual_hash, source['sha256'], 'dialogue_source_hash', task)
        hashes[str(source_path)] = actual_hash
        records, expected_memory_ids = [], []
        for session in read(source_path)['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = task + '/' + turn['round']
                refs = turn.get('input_image') or []
                refs = [refs] if isinstance(refs, str) else refs
                if not refs:
                    auditor.check(mid not in memories, 'source_image_round_coverage', mid)
                    continue
                expected_memory_ids.append(mid)
                memory = memories[mid]
                auditor.equal(memory['task'], task, 'memory_task_identity', mid)
                auditor.equal(memory['images'], [source_image_ids[ref] for ref in refs], 'memory_image_order', mid)
                auditor.equal(memory.get('date'), session.get('date'), 'memory_timestamp_source', mid)
                for i, image_id in enumerate(memory['images'], 1):
                    image_path = (ROOT / images[image_id]['path']).resolve()
                    if image_id not in checked_pixels:
                        auditor.equal(sha(image_path), image_id, 'image_pixel_hash', image_id)
                        checked_pixels.add(image_id)
                    records.append(dict(id=f'{mid}#image{i}', source_image_id=image_id,
                        image_path=str(image_path), dialogue={role: turn[role] for role in ('user', 'assistant')},
                        caption=captions[image_id]['caption'], timestamp=session.get('date')))
        auditor.equal([row['id'] for row in memory_rows if row['task'] == task], expected_memory_ids,
                      'memory_source_order', task)
        by_task[task] = records
    return by_task, hashes, len(checked_pixels)


def replay(task, records, logs, auditor):
    """Use only decision scalars to derive expected center norms and edge shifts."""
    masters, edges = {}, {}
    auditor.equal([row['image_id'] for row in logs], [r['id'] for r in records], 'log_complete_order', task)
    for row in logs:
        iid = row['image_id']
        candidates = row['candidates']
        auditor.equal([c['master_id'] for c in candidates], list(masters), 'candidate_complete_order', iid)
        for candidate in candidates:
            mid = candidate['master_id']
            auditor.close(candidate['threshold'], masters[mid]['threshold'], 'candidate_previous_threshold', iid + ':' + mid)
            auditor.equal(candidate['eligible'], candidate['similarity'] > candidate['threshold'], 'strict_threshold_eligibility', iid + ':' + mid)
            auditor.check(-1 <= candidate['similarity'] <= 1, 'candidate_similarity_range', iid + ':' + mid)
        eligible = [c for c in candidates if c['eligible']]
        selected = max(eligible, key=lambda c: c['similarity']) if eligible else None
        if selected is None:
            mid = 'M' + str(len(masters) + 1)
            expected_threshold = max((c['similarity'] for c in candidates), default=0.0)
            auditor.equal(row['new_master'], True, 'new_master_decision', iid)
            auditor.equal(row['master_id'], mid, 'master_creation_order', iid)
            auditor.equal(row['similarity_before'], None, 'new_master_similarity_null', iid)
            auditor.equal(row['threshold_before'], None, 'new_master_threshold_null', iid)
            auditor.close(row['threshold_after'], expected_threshold, 'new_master_initial_threshold', iid)
            auditor.close(row['old_edges_distance_delta'], 0.0, 'new_master_old_delta', iid)
            auditor.close(row['new_edge_distance'], 0.0, 'new_master_edge_zero', iid)
            masters[mid] = dict(count=1, norm=1.0, threshold=expected_threshold)
        else:
            mid = selected['master_id']
            auditor.equal(row['new_master'], False, 'join_existing_decision', iid)
            auditor.equal(row['master_id'], mid, 'highest_eligible_cosine_creation_order_ties', iid)
            state = masters[mid]
            n, norm, sim = state['count'], state['norm'], selected['similarity']
            new_norm = math.sqrt((n * norm)**2 + 1.0 + 2.0 * n * norm * sim) / (n + 1)
            denominator = (n + 1) * new_norm
            old_to_new = (n * norm**2 + norm * sim) / denominator
            expected_delta = norm - old_to_new
            expected_new_distance = 1.0 - (n * norm * sim + 1.0) / denominator
            auditor.close(row['similarity_before'], sim, 'selected_similarity', iid)
            auditor.close(row['threshold_before'], state['threshold'], 'selected_threshold', iid)
            auditor.close(row['threshold_after'], new_norm, 'scalar_replayed_center_norm', iid)
            auditor.close(row['old_edges_distance_delta'], expected_delta, 'scalar_replayed_old_distance_delta', iid)
            auditor.close(row['new_edge_distance'], expected_new_distance, 'scalar_replayed_new_edge_distance', iid)
            for edge in edges.values():
                if edge['master_id'] == mid:
                    edge['distance'] += row['old_edges_distance_delta']
                    edge['distance_semantics'] = 'uniform_mean_shift_approximation'
            state.update(count=n + 1, norm=new_norm, threshold=row['threshold_after'])
        edges[iid] = dict(master_id=mid, image_id=iid, distance=row['new_edge_distance'],
                          distance_semantics='exact_at_insertion')
    return masters, edges


def audit(output):
    auditor = Auditor()
    run = read(output / 'run_config.json')
    encoder = read(output / 'encoder_metadata.json')
    summary = read(output / 'summary.json')
    natural = Path(run['natural_root'])
    records_by_task, source_hashes, unique_images = source_records(natural, auditor)
    expected_inputs = {str((natural / filename).resolve()): sha(natural / filename) for filename in
                       ('protocol.json', 'images.jsonl', 'memories.jsonl', 'captions.jsonl')}
    auditor.equal(run['input_hashes'], expected_inputs, 'input_manifest_hashes')
    for filename, expected in run['code_hashes'].items():
        auditor.equal(sha(filename), expected, 'runtime_code_hashes', filename)
    auditor.equal(run['format'], 'visual_embedding_graph_build_v1', 'run_format')
    auditor.equal(run['online_supported'], False, 'offline_only')
    auditor.equal(run['image_embedding_storage'], 'none: only master aggregate vectors persist', 'no_image_embedding_policy')
    for field, value in run['encoder'].items():
        auditor.equal(encoder.get(field), value, 'encoder_identity', field)
    auditor.equal(encoder['device'], run['device'], 'encoder_device')
    auditor.equal(encoder['model_class'], 'CLIPVisionModelWithProjection', 'clip_visual_model')
    auditor.equal(encoder['processor_config_sha256'], digest(encoder['processor_config']), 'processor_config_hash')
    if encoder['source_type'] == 'local':
        for filename, expected in encoder['files_sha256'].items():
            auditor.equal(sha(Path(encoder['model_name']) / filename), expected, 'encoder_model_file_hash', filename)
    auditor.equal(set(summary['tasks']), set(records_by_task), 'task_coverage')
    auditor.equal(summary['complete'], True, 'run_complete')
    auditor.equal(set(p.parent.name for p in output.glob('*/graph.json')), set(records_by_task), 'graph_task_coverage')
    cluster_rows, task_stats, output_hashes = [], {}, {}
    for task, records in records_by_task.items():
        td = output / task
        graph = read(td / 'graph.json')
        state = read(td / 'checkpoint.json')
        logs = lines(td / 'write_log.jsonl')
        task_summary = read(td / 'summary.json')
        auditor.equal(state['graph'], graph, 'checkpoint_graph_equivalence', task)
        auditor.equal(state['write_log'], logs, 'checkpoint_log_equivalence', task)
        auditor.equal(state['format'], run['format'], 'checkpoint_format', task)
        auditor.equal(state['protocol_digest'], digest(run), 'checkpoint_protocol_digest', task)
        auditor.equal(state['records_digest'], digest(records), 'checkpoint_source_records_digest', task)
        auditor.equal(state['encoder_metadata'], encoder, 'checkpoint_encoder_identity', task)
        auditor.equal(task_summary, summary['tasks'][task], 'summary_export_equivalence', task)
        auditor.equal(task_summary['expected_images'], len(records), 'summary_expected_count', task)
        auditor.equal(task_summary['processed_images'], len(records), 'summary_processed_count', task)
        auditor.equal(task_summary['complete'], True, 'task_complete', task)
        auditor.equal(task_summary['persistent_image_embeddings'], 0, 'summary_no_member_vectors', task)
        auditor.equal(task_summary['encoder'], encoder, 'summary_encoder_identity', task)
        nodes, masters, edges = graph['image_nodes'], graph['masters'], graph['edges']
        auditor.equal([n['id'] for n in nodes], [r['id'] for r in records], 'node_complete_source_order', task)
        auditor.equal([e['image_id'] for e in edges], [r['id'] for r in records], 'edge_complete_source_order', task)
        auditor.equal(len({n['id'] for n in nodes}), len(nodes), 'unique_node_occurrences', task)
        for node, record in zip(nodes, records):
            auditor.equal(set(node), {'id', 'image_id', 'image_path', 'dialogue', 'caption', 'timestamp'}, 'node_schema_no_vectors', node['id'])
            auditor.equal(node['image_id'], record['id'], 'node_occurrence_identity', node['id'])
            for field in ('image_path', 'dialogue', 'caption', 'timestamp'):
                auditor.equal(node[field], record[field], 'verbatim_source_' + field, node['id'])
        storage = graph['vector_storage']
        vector_path = td / storage['path']
        auditor.check(vector_path.resolve().is_relative_to(td.resolve()), 'task_local_vector_path', task)
        auditor.equal(storage['path'], f"vectors/master_visual_means.{storage['sha256']}.npy", 'content_addressed_vector_path', task)
        auditor.equal(sha(vector_path), storage['sha256'], 'master_vector_hash', task)
        matrix = np.load(vector_path, allow_pickle=False)
        auditor.equal(storage['shape'], [len(masters), encoder['embedding_dim']], 'master_vector_metadata_shape', task)
        auditor.equal(list(matrix.shape), storage['shape'], 'master_vector_file_shape', task)
        auditor.equal(str(matrix.dtype), 'float64', 'master_vector_dtype', task)
        auditor.equal(storage['dtype'], str(matrix.dtype), 'master_vector_metadata_dtype', task)
        auditor.check(bool(np.isfinite(matrix).all()), 'master_vector_finite', task)
        auditor.equal(sorted(p.relative_to(td).as_posix() for p in td.rglob('*.npy')), [storage['path']], 'only_one_current_master_matrix', task)
        auditor.equal(graph['embedding_dim'], encoder['embedding_dim'], 'graph_encoder_dimension', task)
        loaded = load_graph(td / 'graph.json')
        auditor.equal(loaded.insertion_order, [n['id'] for n in nodes], 'load_graph_reader_compatibility', task)
        expected_masters, expected_edges = replay(task, records, logs, auditor)
        auditor.equal([m['id'] for m in masters], list(expected_masters), 'replay_master_order', task)
        auditor.equal(task_summary['masters'], len(masters), 'summary_master_count', task)
        distances_by_master = {m['id']: [] for m in masters}
        for edge in edges:
            iid = edge['image_id']
            expected = expected_edges[iid]
            auditor.equal(set(edge), {'master_id', 'image_id', 'distance', 'distance_semantics'}, 'edge_schema_no_vectors', iid)
            auditor.equal(edge['master_id'], expected['master_id'], 'replay_edge_assignment', iid)
            auditor.equal(edge['distance_semantics'], expected['distance_semantics'], 'replay_edge_semantics', iid)
            auditor.close(edge['distance'], expected['distance'], 'replay_final_edge_distance', iid)
            distances_by_master[edge['master_id']].append(edge['distance'])
        for i, master in enumerate(masters):
            mid = master['id']
            expected = expected_masters[mid]
            auditor.equal(set(master), {'id', 'visual_mean_ref', 'member_count', 'join_threshold'}, 'master_schema_aggregate_ref_only', task + ':' + mid)
            auditor.equal(master['visual_mean_ref'], {'row': i}, 'master_vector_row_reference', task + ':' + mid)
            auditor.equal(master['member_count'], expected['count'], 'replay_master_count', task + ':' + mid)
            auditor.equal(master['member_count'], len(distances_by_master[mid]), 'master_edge_member_count', task + ':' + mid)
            auditor.close(master['join_threshold'], expected['threshold'], 'replay_master_threshold', task + ':' + mid)
            norm = math.hypot(*matrix[i].tolist())
            auditor.close(norm, expected['norm'], 'replay_final_vector_norm', task + ':' + mid)
            average_distance = math.fsum(distances_by_master[mid]) / master['member_count']
            auditor.close(average_distance, 1.0 - norm, 'final_mean_distance_identity', task + ':' + mid)
            if master['member_count'] > 1:
                auditor.close(master['join_threshold'], norm, 'final_multi_member_threshold_identity', task + ':' + mid)
            else:
                auditor.close(norm, 1.0, 'final_singleton_unit_norm', task + ':' + mid)
            cluster_rows.append(dict(task=task, master_id=mid, member_count=master['member_count'],
                singleton=master['member_count'] == 1, join_threshold=master['join_threshold'],
                visual_mean_norm=norm, mean_edge_distance=average_distance,
                min_edge_distance=min(distances_by_master[mid]), max_edge_distance=max(distances_by_master[mid]),
                member_ids='|'.join(e['image_id'] for e in edges if e['master_id'] == mid)))
        task_stats[task] = dict(expected_images=len(records), image_nodes=len(nodes), masters=len(masters),
            singleton_masters=sum(m['member_count'] == 1 for m in masters),
            cluster_size_distribution=dict(sorted(Counter(m['member_count'] for m in masters).items())),
            max_cluster_size=max(m['member_count'] for m in masters),
            threshold_range=[min(m['join_threshold'] for m in masters), max(m['join_threshold'] for m in masters)],
            edge_distance_range=[min(e['distance'] for e in edges), max(e['distance'] for e in edges)],
            approximate_edges=sum(e['distance_semantics'] == 'uniform_mean_shift_approximation' for e in edges),
            master_vector_file=storage['path'], master_vector_sha256=storage['sha256'])
        for filename in ('graph.json', 'checkpoint.json', 'write_log.jsonl', 'summary.json'):
            output_hashes[str((td / filename).relative_to(output))] = sha(td / filename)
        output_hashes[str(vector_path.relative_to(output))] = sha(vector_path)
    # The launcher pins supporting source modules beyond run_config's core list.
    evidence = {}
    for filename in ('execution.json',):
        path = output / filename
        auditor.check(path.is_file(), 'execution_evidence_exists', filename)
        if path.is_file():
            evidence[filename] = dict(sha256=sha(path), value=read(path))
    execution = evidence['execution.json']['value']
    auditor.equal(execution['exit_code'], 0, 'execution_success')
    auditor.equal(execution['source_unchanged_during_run'], True, 'execution_source_unchanged')
    command = execution['command']
    auditor.check('--limit' not in command and '--task' not in command, 'execution_unrestricted_scope')
    for flag, expected in [('--memory', 'visual_embedding_graph'), ('--natural-root', str(natural)),
                           ('--graph-root', str(output)), ('--visual-device', run['device']),
                           ('--visual-encoder', encoder['model_name']),
                           ('--visual-encoder-revision', encoder['revision'])]:
        auditor.equal(command[command.index(flag) + 1], expected, 'execution_command_identity', flag)
    auditor.check('--visual-local-files-only' in command, 'execution_local_files_only')
    auditor.equal(execution['environment']['HF_HUB_OFFLINE'], '1', 'execution_offline_environment')
    started = datetime.fromisoformat(execution['started_at'])
    finished = datetime.fromisoformat(execution['finished_at'])
    auditor.check(finished >= started and execution['elapsed_seconds'] > 0, 'execution_valid_timing')
    auditor.close((finished - started).total_seconds(), execution['elapsed_seconds'], 'execution_duration_consistency', tol=0.01)
    snapshot_root = output / 'source_snapshot'
    for source_path, expected_hash in run['code_hashes'].items():
        relative = Path(source_path).relative_to(ROOT).as_posix()
        auditor.equal(execution['source_hashes'].get(relative), expected_hash, 'execution_core_code_hash_identity', relative)
    for relative, expected_hash in execution['source_hashes'].items():
        copied = snapshot_root / relative
        auditor.check(copied.is_file(), 'source_snapshot_file_present', str(relative))
        if copied.is_file():
            auditor.equal(sha(copied), expected_hash, 'source_snapshot_file_hash', str(relative))
        auditor.equal(sha(ROOT / relative), expected_hash, 'execution_source_still_unchanged', relative)
    runtime_log = Path(execution['log_path'])
    auditor.check(runtime_log.is_file(), 'runtime_log_present')
    runtime_text = runtime_log.read_text(encoding='utf-8')
    runtime_progress = {}
    for task in records_by_task:
        pattern = re.compile(r'^' + re.escape(task) + r': (\d+)/(\d+) images, (\d+) masters$', re.MULTILINE)
        progress = [tuple(map(int, match.groups())) for match in pattern.finditer(runtime_text)]
        expected_count = len(records_by_task[task])
        auditor.equal([count for count, total, masters in progress], list(range(1, expected_count + 1)), 'runtime_log_complete_progress', task)
        auditor.check(all(total == expected_count for count, total, masters in progress), 'runtime_log_expected_total', task)
        auditor.equal(progress[-1][2], task_stats[task]['masters'], 'runtime_log_final_master_count', task)
        runtime_progress[task] = dict(progress_rows=len(progress), final_progress=list(progress[-1]))
    evidence['runtime_log'] = dict(path=str(runtime_log), sha256=sha(runtime_log), task_progress=runtime_progress)
    auditor.check('Traceback (most recent call last)' not in runtime_text, 'runtime_no_python_traceback')
    report = dict(status='passed' if not auditor.errors else 'failed',
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        audit_script=str(Path(__file__).resolve()), audit_script_sha256=sha(__file__),
        root=str(output), scope='complete offline build; independent source reconstruction and scalar-only decision replay',
        totals=dict(tasks=len(records_by_task), expected_image_occurrences=sum(map(len, records_by_task.values())),
            unique_image_contents=unique_images, image_nodes=sum(s['image_nodes'] for s in task_stats.values()),
            masters=len(cluster_rows), singleton_masters=sum(s['singleton_masters'] for s in task_stats.values()),
            edges=sum(s['image_nodes'] for s in task_stats.values()), persisted_member_embedding_files=0),
        checks=dict(auditor.checked), max_abs_errors=dict(auditor.max_abs_error),
        errors=auditor.errors, warnings=auditor.warnings, tasks=task_stats,
        source_dialogue_hashes=source_hashes, input_hashes=expected_inputs, code_hashes=run['code_hashes'],
        encoder_identity=encoder, execution_evidence=evidence, output_hashes=output_hashes,
        limitations=['No image encoder was rerun: candidate cosine values are audited for ranges and internal decision/update consistency, not independently recomputed from pixels.',
                     'Old individual edge distances are the specified uniform-shift approximation; the population mean-distance identity is checked exactly within 1e-9.',
                     'Singleton master aggregate vectors equal their sole member representation by construction; no separate image-vector cache is stored.',
                     'This is an offline build audit, not an online retrieval or answer-accuracy evaluation.'])
    with (output / 'cluster_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cluster_rows[0]))
        writer.writeheader()
        writer.writerows(cluster_rows)
    (output / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'totals', 'errors', 'tasks')}, indent=2, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'results/visual_embedding_graph')
    args = parser.parse_args()
    sys.exit(audit(args.root.resolve()))
