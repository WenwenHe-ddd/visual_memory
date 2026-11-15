#!/usr/bin/env python3
"""Isolated full MemEye rerun of the four proposed-method conditions.

The original captions and authored MCQ candidates remain immutable inputs.
Every domain memory is rebuilt from image arrivals, followed by fresh retrieval
and reader inference. The PCA and Ward algorithms have no random initialization.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for _key, _value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                     'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(_key, _value)

import numpy as np

from scripts.analysis import prepare_all_domain_adaptive_inputs as build
from scripts.analysis import run_caption_ward_visual_pilot as home_build
from scripts.analysis.evaluate_adaptive_changed_qa import reader_text
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.memory_store import export_store

OUTPUT = ROOT / 'results/memeye_qwen_seed1024_full_20261105/ours'
REFERENCE = ROOT / 'results/memeye_qwen_zscore_joint_search_20261024'
INPUTS = build.DEFAULT_OUTPUT
QUESTIONS = ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'
TEXTS = ROOT / 'data/independent_visual_inputs/retrieval_memories.jsonl'
CANDIDATES = ROOT / 'reports/all_master_answer_query_20261021/candidates.jsonl'
CONDITIONS = ('baseline_question_only', 'master_answer_query',
              'baseline_question_only__l0p03_a1', 'master_answer_query__l0p03_a1')
SEED = 1024


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_lines(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def verify_hashes(hashes):
    for path, expected in hashes.items():
        assert sha(path) == expected, f'Source changed: {path}'


def verify_signature(protocol):
    signed = dict(protocol)
    expected = signed.pop('experiment_signature')
    assert digest(signed) == expected


def seed_all():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def prepare(output=OUTPUT, device='cuda:0'):
    output = Path(output).resolve()
    if (output / 'protocol.json').exists():
        protocol = read(output / 'protocol.json')
        verify_signature(protocol)
        verify_hashes(protocol['source_sha256'])
        assert protocol['seed'] == SEED and protocol['conditions'] == list(CONDITIONS)
        assert protocol['reader_device'] == device
        return protocol
    old = read(REFERENCE / 'reader/protocol.json')
    verify_signature(old)
    assert old['reader_settings']['seed'] == 42 and not old['reader_settings']['do_sample']
    qa, memories, streams, _, offline, source_hashes = build.collect_sources()
    assert len(qa) == 371 and len(memories) == 438 and len(streams) == 8
    candidates = lines(CANDIDATES)
    assert len(candidates) == len({r['id'] for r in candidates}) == 371
    assert {r['id'] for r in candidates} == {r['id'] for r in qa}
    historical = lines(REFERENCE / 'reader/qa_manifest.jsonl')
    historical = [r for r in historical if r['condition'] in CONDITIONS]
    assert len(historical) == 1484
    # Historical source code is provenance, not a current-code assertion. Verify
    # all frozen data and model configuration bytes, then pin current code anew.
    checked_historical_data = {}
    for path, expected in old['source_sha256'].items():
        if Path(path).suffix in ('.json', '.jsonl', '.npy', '.npz'):
            assert sha(path) == expected, path
            checked_historical_data[path] = expected
    verify_hashes(old['model_source_sha256'])
    for weight in old['weight_files']:
        stat = Path(weight['path']).stat()
        assert stat.st_size == weight['bytes'] and stat.st_mtime_ns == weight['mtime_ns']
    code = [Path(__file__), Path(build.__file__), Path(home_build.__file__),
            ROOT / 'scripts/graph/caption_ward_visual.py', ROOT / 'scripts/graph/visual_image_encoder.py',
            ROOT / 'scripts/memory/caption_visual_master.py', ROOT / 'scripts/memory/online_retrieval.py',
            ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py',
            ROOT / 'scripts/memory/memory_store.py', ROOT / 'scripts/evaluation/paired_single_gpu_reader.py',
            ROOT / 'scripts/analysis/evaluate_adaptive_changed_qa.py',
            ROOT / 'scripts/analysis/evaluate_all_master_answer_reader.py']
    sources = [QUESTIONS, TEXTS, CANDIDATES, REFERENCE / 'reader/protocol.json',
               REFERENCE / 'reader/qa_manifest.jsonl', REFERENCE / 'reader/responses.jsonl',
               REFERENCE / 'summary.json', *code]
    source_hashes.update(checked_historical_data)
    source_hashes.update({str(p): sha(p) for p in sources})
    for domain, stream in streams.items():
        for row in stream:
            assert sha(row['source_path']) == row['sha256']
            source_hashes[row['source_path']] = row['sha256']
        for filename in ('inputs.json', 'protocol.json', 'memory/graph.json',
                         'memory/basis.npy', 'memory/master_sums.npy'):
            path = INPUTS / domain / filename
            source_hashes[str(path)] = sha(path)
    settings = copy.deepcopy(old['reader_settings'])
    settings['seed'] = SEED
    output.mkdir(parents=True, exist_ok=True)
    dump(output / 'inputs.json', {'questions': qa, 'streams': streams,
                                 'original_offline_protocol': offline})
    source_hashes[str(output / 'inputs.json')] = sha(output / 'inputs.json')
    protocol = {'design': 'Full chronological memory rebuild, fresh retrieval and fresh MCQ answers; frozen original captions.',
                'experiment_run_id': str(uuid.uuid4()), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'seed': SEED, 'conditions': list(CONDITIONS), 'questions': 371, 'memory_leaves': 438,
                'domains': sorted(streams), 'lambda': .03, 'alpha': 1., 'reader_device': device,
                'reader_settings': settings, 'offline_device': 'cpu', 'offline_threads': 4,
                'historical_answers_reused': False, 'historical_memories_reused': False,
                'caption_policy': 'Original Qwen captions frozen; no caption-generation calls.',
                'candidate_policy': 'Original authored MCQ candidates frozen; no generation model or seed.',
                'offline_algorithm': 'Original PCA-2D and caption-Ward/silhouette; chronological arrivals. Home caption batch1, other domains batch16.',
                'seed_scope': 'Python, NumPy and Torch at stage boundaries; reader Torch seed reset per request. Greedy decoding.',
                'source_sha256': source_hashes, 'model_source_sha256': old['model_source_sha256'],
                'weight_files': old['weight_files'],
                'weight_validation': 'Pinned historical SHA256 retained; size and mtime checked during prepare; full rehash during run.'}
    protocol['experiment_signature'] = digest(protocol)
    dump(output / 'protocol.json', protocol)
    dump(output / 'validation_input.json', {'passed': True, 'questions': 371, 'leaves': 438,
        'domains': 8, 'conditions': 4, 'historical_selected_records': 1484,
        'historical_selected_unique_payloads': len({r['payload_sha256'] for r in historical}),
        'frozen_data_and_current_code_hashed': True, 'model_configs_verified': True,
        'weight_sizes_and_mtimes_verified': True, 'gpu_loaded': False})
    print(json.dumps({'stage': 'prepared', 'output': str(output), 'questions': 371, 'conditions': 4}), flush=True)
    return protocol


def rebuild_memories(output, protocol):
    index_path = output / 'memory_manifest.json'
    if index_path.exists():
        index = read(index_path)
        assert index['experiment_signature'] == protocol['experiment_signature']
        verify_hashes(index['output_sha256'])
        return index
    from sentence_transformers import SentenceTransformer
    seed_all()
    inputs = read(output / 'inputs.json')
    streams = inputs['streams']
    reference_models = inputs['original_offline_protocol']
    text_meta = reference_models['text_encoder']
    vision_meta = reference_models['image_encoder']
    verify_hashes({str(Path(text_meta['model_path']) / k): v for k, v in text_meta['files_sha256'].items()})
    encoder = ImageEncoder(vision_meta['model_name'], revision=vision_meta['revision'], device='cpu', local_files_only=True)
    assert encoder.metadata['files_sha256'] == vision_meta['files_sha256']
    text_model = SentenceTransformer(text_meta['model_path'], device='cpu', local_files_only=True)
    home_rows, home_calibration, _ = home_build.prepare_inputs()
    assert [r['image_id'] for r in home_rows] == [r['image_id'] for r in streams[build.HOME]]
    # Home's original sample-covariance PCA and single-caption encode layout are
    # kept distinct from the seven later domains' population-PCA/batch16 path.
    home_basis, home_meta = home_build.calibrate_basis(encoder, home_calibration)
    other_domains = [d for d in sorted(streams) if d != build.HOME]
    bases, calibration = build.calibrate_external_bases(streams, other_domains, encoder, output)
    bases[build.HOME] = home_basis
    calibration[build.HOME] = home_meta
    audits, output_hashes = [], {}
    for domain in sorted(streams):
        seed_all()
        target = output / 'offline' / domain
        if (target / 'memory').exists():
            # A partial domain is always reconstructed instead of mixing state.
            import shutil
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        memory = CaptionWardVisualMemory(bases[domain])
        stream = streams[domain]
        if domain != build.HOME:
            caption_vectors = build.normalize(text_model.encode([r['caption'] for r in stream],
                batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt=''))
        insertion = []
        global_sum = np.zeros(512)
        for index, row in enumerate(stream):
            if domain == build.HOME:
                visual = home_build.unit(encoder.encode(row['source_path']))
                text = home_build.unit(text_model.encode(row['caption'], normalize_embeddings=True, show_progress_bar=False))
            else:
                visual = np.asarray(encoder.encode(row['source_path']), dtype=np.float64)
                visual /= np.max(np.abs(visual))
                visual /= np.linalg.norm(visual)
                text = caption_vectors[index]
            global_sum += visual
            audit = memory.add(row['image_id'], row['caption'], text, visual,
                               dialogue=row['dialogue'], timestamp=row['timestamp'])
            del visual, text
            actual_sum = np.sum([m['visual_sum'] for m in memory.masters.values()], axis=0)
            assert np.linalg.norm(actual_sum - global_sum) < 1e-8
            assert audit['max_projection_consistency_l2_error'] < 1e-8
            insertion.append({'step': index + 1, **audit})
            if (index + 1) % 20 == 0 or index + 1 == len(stream):
                print(json.dumps({'stage': 'memory_rebuild', 'domain': domain,
                                  'arrivals': index + 1, 'total': len(stream), 'masters': len(memory.masters)}), flush=True)
        graph, sums = build.save_memory(memory, target / 'memory')
        write_lines(target / 'insertion_audit.jsonl', insertion)
        historical = read(INPUTS / domain / 'protocol.json')
        domain_protocol = {**historical, 'seed': SEED, 'rebuild_run_id': protocol['experiment_run_id'],
                           'rebuild_source_signature': protocol['experiment_signature'],
                           'calibration': calibration[domain], 'old_member_reencoding_within_stream': 0}
        # Old metadata describes immutable encoder/length policy; current source
        # provenance is this run's signed protocol, not historical code hashes.
        domain_protocol.pop('source_sha256', None)
        dump(target / 'protocol.json', domain_protocol)
        by_id = {r['image_id']: r for r in stream}
        leaves = [{**{key: by_id[n['id']][key] for key in ('image_id', 'memory_id', 'filename', 'dialogue', 'caption', 'retrieval_text')},
                   'master_id': n['master_id']} for n in graph['image_nodes']]
        dump(target / 'inputs.json', {'questions': [q for q in inputs['questions'] if q['task'] == domain],
                                     'leaves': leaves, 'masters': [{k: m[k] for k in ('id', 'n', 'member_ids')} for m in graph['masters']]})
        for method in ('natural', 'caption_visual_master'):
            destination = output / 'stores' / domain / method
            if destination.exists():
                import shutil
                shutil.rmtree(destination)
            export_store(target, destination, method)
        old_graph = read(INPUTS / domain / 'memory/graph.json')
        comparisons = {}
        for name in ('graph.json', 'basis.npy', 'master_sums.npy'):
            new, old_path = target / 'memory' / name, INPUTS / domain / 'memory' / name
            result = {'sha256': sha(new), 'seed42_sha256': sha(old_path), 'byte_identical': sha(new) == sha(old_path)}
            if name.endswith('.npy'):
                before, after = np.load(old_path), np.load(new)
                result['same_shape'] = before.shape == after.shape
                result['max_abs_difference'] = float(np.max(np.abs(before - after))) if before.shape == after.shape else None
            comparisons[name] = result
        audits.append({'domain': domain, 'arrivals': len(stream), 'masters': len(graph['masters']),
            'seed42_masters': len(old_graph['masters']), 'files': comparisons,
            'same_memberships': [(m['id'], m['member_ids']) for m in graph['masters']] ==
                                [(m['id'], m['member_ids']) for m in old_graph['masters']],
            'image_arrival_forwards': len(stream), 'old_member_image_reencodes': 0})
        for folder in (target, output / 'stores' / domain):
            output_hashes.update({str(p): sha(p) for p in folder.rglob('*') if p.is_file()})
    del text_model, encoder
    gc.collect()
    index = {'completed': True, 'experiment_signature': protocol['experiment_signature'],
             'seed': SEED, 'domains': audits, 'memory_leaves': sum(a['arrivals'] for a in audits),
             'masters': sum(a['masters'] for a in audits), 'calibration_image_forwards': 344 + 433,
             'image_arrival_forwards': 438, 'historical_memories_reused': False,
             'output_sha256': output_hashes}
    assert index['memory_leaves'] == 438
    dump(index_path, index)
    return index


def retrieve(output, protocol, device):
    from types import SimpleNamespace
    from scripts.memory import online_retrieval as online
    from scripts.memory.online_zscore import ZScoreWeighting
    if (output / 'retrieval_audit.json').exists():
        audit = read(output / 'retrieval_audit.json')
        assert audit['experiment_signature'] == protocol['experiment_signature']
        verify_hashes(audit['output_sha256'])
        return
    seed_all()
    original = lines(QUESTIONS)
    by_id = {q['id']: q for q in original}
    texts = {m['id']: m['reader_text'] for m in lines(TEXTS)}
    old = {r['record_id']: r for r in lines(REFERENCE / 'reader/qa_manifest.jsonl') if r['condition'] in CONDITIONS}
    records, retrieved = [], []
    args = SimpleNamespace(device=device, candidates=CANDIDATES)
    online.configure_device(args)
    for domain in sorted({q['task'] for q in original}):
        questions = [q for q in original if q['task'] == domain]
        encoder = None
        for method, strategy, names in (
            ('natural', online.FullScanRetrieval(), (CONDITIONS[0], CONDITIONS[2])),
            ('caption_visual_master', online.MultiPathRetrieval(), (CONDITIONS[1], CONDITIONS[3]))):
            args.mode, args.retrieval = method, strategy.name
            args.memory_dir = output / 'stores' / domain / method
            memory, metadata, _, _ = online.load_memory(args, questions)
            if encoder is None:
                encoder = online.load_text_encoder(metadata['text_encoder'], args.device)
            for q in questions:
                sanitized = {k: v for k, v in q.items() if k not in ('gold_ids', 'gold_choice')}
                result = online.retrieve_question(sanitized, memory, encoder, metadata['query_prefix'], 5,
                                                 ZScoreWeighting(.03, 1.), strategy)
                result['method'] = method
                result['conditions'][0]['condition'] = names[0]
                result['conditions'][1]['condition'] = names[1]
                retrieved.append(result)
                for condition, rr in zip(names, result['conditions']):
                    ids = rr['top5_memory_ids']
                    assert len(ids) == len(set(ids)) == 5
                    assert all(mid.split('/', 1)[0] == domain for mid in ids)
                    text = reader_text(q, ids, texts)
                    images = [{'path': p, 'sha256': sha(p)} for p in q['query_image_paths']]
                    payload = {'system': protocol['reader_settings']['reader_system_prompt'],
                               'user_text': text, 'query_images': images}
                    records.append({**{k: q[k] for k in ('id', 'task', 'question', 'options', 'gold_choice', 'gold_ids', 'query_image_paths')},
                        'record_id': q['id'] + '::' + condition, 'condition': condition,
                        'top_memory_ids': ids, 'top_query_scores': [r['score'] for r in rr['top5']],
                        'reader_text': text, 'query_images': images, 'payload_sha256': digest(payload)})
            del memory
        del encoder
        gc.collect()
        print(json.dumps({'stage': 'retrieval', 'domain': domain, 'logical_records': len(records)}), flush=True)
    manifest_by_id = {r['record_id']: r for r in records}
    assert len(manifest_by_id) == len(records) == 1484
    manifest = []
    for i, q in enumerate(original):
        batch = [manifest_by_id[q['id'] + '::' + c] for c in CONDITIONS]
        offset = i % 4
        manifest.extend(batch[offset:] + batch[:offset])
    write_lines(output / 'retrieval_results.jsonl', retrieved)
    write_lines(output / 'reader/qa_manifest.jsonl', manifest)
    metrics = {}
    for c in CONDITIONS:
        rows = [r for r in manifest if r['condition'] == c]
        metrics[c] = {'questions': len(rows), 'gold_hits_top5': sum(len(set(r['top_memory_ids']) & set(r['gold_ids'])) for r in rows),
            'gold_references': sum(len(r['gold_ids']) for r in rows),
            'changed_top5_order_vs_seed42': sum(r['top_memory_ids'] != old[r['record_id']]['top_memory_ids'] for r in rows),
            'changed_top5_set_vs_seed42': sum(set(r['top_memory_ids']) != set(old[r['record_id']]['top_memory_ids']) for r in rows),
            'changed_payload_vs_seed42': sum(r['payload_sha256'] != old[r['record_id']]['payload_sha256'] for r in rows)}
        metrics[c]['recall_at_5_micro'] = metrics[c]['gold_hits_top5'] / metrics[c]['gold_references']
    old_protocol = read(REFERENCE / 'reader/protocol.json')
    sources = [output / 'protocol.json', output / 'memory_manifest.json', output / 'retrieval_results.jsonl',
               output / 'reader/qa_manifest.jsonl', Path(__file__), ROOT / 'scripts/evaluation/paired_single_gpu_reader.py']
    reader_protocol = {'experiment_run_id': protocol['experiment_run_id'], 'condition_names': list(CONDITIONS),
        'questions': 371, 'logical_conditions': 1484, 'reader_settings': protocol['reader_settings'],
        'unique_reader_payloads': len({r['payload_sha256'] for r in manifest}),
        'placement': {'gpu_index': int(device.split(':')[1]), 'minimum_free_gib': 24},
        'attention_implementation': 'sdpa', 'historical_response_reuse': False,
        'source_sha256': {str(p): sha(p) for p in sources},
        'model_source_sha256': old_protocol['model_source_sha256'], 'weight_files': old_protocol['weight_files'],
        'query_images': list({i['path']: i for r in manifest for i in r['query_images']}.values()),
        'deduplication': 'Identical complete reader payloads share one fresh inference only within this run.',
        'interpretation': 'Fixed-option MCQ accuracy; not four-rotation official EM.'}
    reader_protocol['experiment_signature'] = digest(reader_protocol)
    dump(output / 'reader/protocol.json', reader_protocol)
    dump(output / 'retrieval_audit.json', {'passed': True, 'experiment_signature': protocol['experiment_signature'],
        'logical_records': 1484, 'unique_reader_payloads': reader_protocol['unique_reader_payloads'],
        'metrics': metrics, 'source_questions_and_texts_unchanged': True,
        'gold_used_only_after_ranking': True,
        'output_sha256': {str(p): sha(p) for p in (output / 'retrieval_results.jsonl',
            output / 'reader/qa_manifest.jsonl', output / 'reader/protocol.json')}})


def analyze(output=OUTPUT):
    output = Path(output).resolve()
    protocol = read(output / 'protocol.json')
    verify_signature(protocol)
    verify_hashes(protocol['source_sha256'])
    reader_protocol = read(output / 'reader/protocol.json')
    verify_signature(reader_protocol)
    verify_hashes(reader_protocol['source_sha256'])
    manifest = {r['record_id']: r for r in lines(output / 'reader/qa_manifest.jsonl')}
    responses = lines(output / 'reader/responses.jsonl')
    by_id = {r['record_id']: r for r in responses}
    assert len(responses) == len(by_id) == len(manifest) == 1484, 'MCQ run incomplete'
    assert set(by_id) == set(manifest)
    historical = {r['record_id']: r for r in lines(REFERENCE / 'reader/responses.jsonl') if r['condition'] in CONDITIONS}
    fresh = 0
    for r in responses:
        source = manifest[r['record_id']]
        match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', r['raw_response'].strip(), re.I)
        choice = match.group(1).upper() if match else None
        assert r['predicted_choice'] == choice and r['correct'] == (choice == source['gold_choice'])
        assert r['experiment_signature'] == reader_protocol['experiment_signature']
        assert r['payload_sha256'] == source['payload_sha256']
        assert r['run_id'] == protocol['experiment_run_id'] and not r['historical_response_reuse']
        if r['fresh_inference_executed_for_this_record']:
            fresh += 1
        else:
            origin = by_id[r['same_run_shared_inference_record_id']]
            assert origin['fresh_inference_executed_for_this_record']
            for key in ('payload_sha256', 'raw_response', 'generated_token_ids', 'execution_id'):
                assert r[key] == origin[key]
    assert fresh == reader_protocol['unique_reader_payloads']
    metrics = copy.deepcopy(read(output / 'retrieval_audit.json')['metrics'])
    for c in CONDITIONS:
        rows = [r for r in responses if r['condition'] == c]
        old_correct = sum(historical[r['record_id']]['correct'] for r in rows)
        correct = sum(r['correct'] for r in rows)
        metrics[c].update({'correct': correct, 'accuracy': correct / 371, 'seed42_correct': old_correct,
            'accuracy_delta': (correct - old_correct) / 371,
            'changed_answers_vs_seed42': sum(r['predicted_choice'] != historical[r['record_id']]['predicted_choice'] for r in rows),
            'format_errors': sum(r['status'] != 'ok' for r in rows),
            'token_limit_hits': sum(r['hit_token_limit'] for r in rows)})
    summary = {'status': 'complete', 'seed': SEED, 'questions': 371, 'overall': metrics,
        'logical_records': len(responses), 'fresh_inferences': fresh, 'shared_records': len(responses) - fresh,
        'memory_rebuild': read(output / 'memory_manifest.json'), 'historical_answers_reused': 0,
        'judge_seed': 42, 'judge_status': 'separate Open-answer workflow',
        'interpretation': 'Seed-only rerun with original captions and greedy decoding. Reproducibility check; two seeds do not establish general robustness.'}
    dump(output / 'summary.json', summary)
    paired = []
    for q in lines(QUESTIONS):
        item = {'question_id': q['id'], 'domain': q['task']}
        for c in CONDITIONS:
            row = by_id[q['id'] + '::' + c]
            item[c] = {k: row[k] for k in ('predicted_choice', 'correct', 'payload_sha256')}
            item[c]['seed42_choice'] = historical[row['record_id']]['predicted_choice']
        paired.append(item)
    dump(output / 'paired_results.json', paired)
    dump(output / 'independent_audit.json', {'passed': True, 'questions_per_condition': 371,
        'logical_records': 1484, 'responses_regraded_from_raw_text': True,
        'new_execution_unique_payloads': fresh, 'historical_answer_reuse': False,
        'all_signed_current_inputs_verified': True,
        'output_sha256': {str(p): sha(p) for p in (output / 'summary.json', output / 'paired_results.json',
              output / 'reader/responses.jsonl', output / 'retrieval_audit.json')}})
    print(json.dumps({'stage': 'analyzed', 'fresh': fresh, 'correct': {c: metrics[c]['correct'] for c in CONDITIONS}}), flush=True)
    return summary


def run(output=OUTPUT, device='cuda:0'):
    from scripts.evaluation.paired_single_gpu_reader import run as run_reader
    output = Path(output).resolve()
    protocol = prepare(output, device)
    for weight in protocol['weight_files']:
        assert sha(weight['path']) == weight['sha256'], weight['path']
    rebuild_memories(output, protocol)
    retrieve(output, protocol, device)
    run_reader(output / 'reader', lambda _: analyze(output), resume=(output / 'reader/responses.jsonl').exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.output, args.device)
    elif args.phase == 'run':
        run(args.output, args.device)
    else:
        analyze(args.output)


if __name__ == '__main__':
    main()
