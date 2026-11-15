#!/usr/bin/env python3
"""Isolated Mem-Gallery seed1042 captions and three caption methods.

The historical implementation is called with new output roots and its actual
runtime seed changed. Historical answers/captions are provenance only. Shared
question-image captions remain the official QA annotations, as in seed42.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
for _name, _value in {'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                      'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}.items():
    os.environ.setdefault(_name, _value)

RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
CAPTIONS = RUN / 'captions'
GROUP = RUN / 'caption_methods'
BASE = GROUP / 'base'
ZSCORE = GROUP / 'zscore'
SEED = 1042
REFERENCE = ROOT / 'results/mem_gallery_master_answer_20261021'
REFERENCE_Z = ROOT / 'results/mem_gallery_qwen_zscore_l0p03_a1_20261027'
AUDIT = ROOT / 'scripts/analysis/audit_memgallery_qwen_seed1042_caption.py'
PHASES = ('prepare', 'caption-worker', 'caption-finalize', 'shared-inputs',
          'base-build', 'base-candidates', 'base-retrieve', 'base-reader',
          'base-analyze', 'zscore-prepare', 'zscore-reader', 'zscore-analyze', 'audit')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_hashes(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def marker_name(phase, worker_index=None):
    return phase + (f'-{worker_index}' if phase == 'caption-worker' else '')


def marker_path(phase, worker_index=None):
    return GROUP / 'stages' / (marker_name(phase, worker_index) + '.json')


def validate_marker(phase, worker_index=None):
    path = marker_path(phase, worker_index)
    if not path.exists():
        return False
    value = read(path)
    assert value['status'] == 'complete' and value['seed'] == SEED
    assert value['plan_signature'] == verify_plan()['experiment_signature']
    verify_hashes(value['outputs_sha256'])
    return True


def require_marker(phase, worker_index=None):
    if not validate_marker(phase, worker_index):
        raise ValueError(f'Required stage is incomplete: {marker_name(phase, worker_index)}')


def complete(phase, paths, worker_index=None, **details):
    value = {'status': 'complete', 'stage': marker_name(phase, worker_index), 'seed': SEED,
             'plan_signature': verify_plan()['experiment_signature'],
             'completed_utc': datetime.now(timezone.utc).isoformat(),
             'outputs_sha256': {str(Path(p).resolve()): sha(p) for p in paths}, **details}
    dump(marker_path(phase, worker_index), value)
    print(json.dumps({'stage': value['stage'], 'status': 'complete', **details}), flush=True)
    return value


def code_paths():
    return [Path(__file__), AUDIT, *[ROOT / p for p in (
        'scripts/data/generate_mem_gallery_qwen_captions.py', 'scripts/data/mem_gallery_inputs.py',
        'scripts/memory/mem_gallery_master_answer.py', 'scripts/evaluation/mem_gallery_master_reader.py',
        'scripts/evaluation/mem_gallery_zscore.py', 'scripts/memory/caption_visual_master.py',
        'scripts/memory/online_zscore.py', 'scripts/memory/ridge_whitening.py',
        'scripts/graph/caption_ward_visual.py', 'scripts/graph/visual_image_encoder.py',
        'scripts/analysis/prepare_all_domain_adaptive_inputs.py',
        'scripts/analysis/probe_caption_visual_online.py',
        'scripts/analysis/audit_mem_gallery_zscore.py', 'scripts/memory/memory_store.py')]]


def verify_plan():
    plan = read(GROUP / 'plan.json')
    unsigned = dict(plan)
    expected = unsigned.pop('experiment_signature')
    assert digest(unsigned) == expected and plan['seed'] == SEED
    verify_hashes(plan['source_sha256'])
    verify_hashes(plan['snapshot_sha256'])
    return plan


def prepare():
    if (GROUP / 'plan.json').exists():
        verify_plan()
        assert read(CAPTIONS / 'protocol.json')['seed'] == SEED
        return complete('prepare', [GROUP / 'plan.json', CAPTIONS / 'protocol.json', CAPTIONS / 'image_jobs.jsonl'])
    from scripts.data import generate_mem_gallery_qwen_captions as native
    GROUP.mkdir(parents=True, exist_ok=True)
    # Preparation produces no model outputs. Replace the unsigned configuration
    # before launching any worker, then sign the actual seed1042 configuration.
    if not (CAPTIONS / 'protocol.json').exists():
        native.prepare(CAPTIONS)
    cfg = read(CAPTIONS / 'protocol.json')
    assert not any(CAPTIONS.glob('captions.worker*.jsonl'))
    cfg.pop('protocol_signature', None)
    cfg.update(seed=SEED, physical_gpus=[1],
               execution_policy='Original two job shards executed serially on physical GPU1; batch4 unchanged.',
               seed_scope='Torch seed1042 in the native caption worker; greedy decoding; no historical caption reuse.',
               driver_sha256=sha(__file__))
    cfg['protocol_signature'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    dump(CAPTIONS / 'protocol.json', cfg)
    reference_files = [REFERENCE / 'reader/summary.json', REFERENCE / 'reader/protocol.json',
        REFERENCE / 'inputs/questions.jsonl', REFERENCE / 'inputs/gold.jsonl',
        REFERENCE / 'offline/protocol.json', REFERENCE / 'candidate_generation_protocol.json',
        REFERENCE_Z / 'reader/summary.json', REFERENCE_Z / 'reader/protocol.json',
        REFERENCE_Z / 'retrieval_protocol.json',
        ROOT / 'data/Mem-Gallery/source_manifest.json', ROOT / 'data/Mem-Gallery/image_manifest.json',
        ROOT / 'data/Mem-Gallery/captions/qwen3-vl-8b-instruct_20261018_v1/protocol.json',
        ROOT / 'reports/home_caption_ward_visual_20261020/memory/basis.npy',
        ROOT / 'reports/home_caption_ward_visual_20261020/input_manifest.json',
        ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/protocol.json']
    snapshots = {}
    for source in code_paths():
        target = GROUP / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        snapshots[str(target)] = sha(target)
    old, old_z = [read(p / 'reader/summary.json')['overall'] for p in (REFERENCE, REFERENCE_Z)]
    references = []
    for method, condition, origin, metrics in (
        ('semantic_rag', 'baseline_question_only', REFERENCE, old),
        ('ours_1', 'master_answer_query', REFERENCE, old),
        ('ours_1_ours_2', 'master_answer_query+zscore', REFERENCE_Z, old_z)):
        score = metrics[condition]
        references.append({'method': method, 'root': str(origin), 'condition': condition,
                           'questions': 1711, 'correct': round(score['em'] * 1711),
                           'em': score['em'], 'f1': score['f1'], 'bleu_1': score['bleu_1']})
    assert [r['correct'] for r in references] == [778, 798, 756]
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
            'dataset': 'Mem-Gallery', 'questions': 1711, 'rounds': 3962, 'visual_rounds': 1003,
            'caption_unique_images': 1393, 'caption_image_files': 1490,
            'methods': references, 'top_k': 5, 'lambda': .03, 'alpha': 1.,
            'caption_policy': 'Fresh Qwen seed1042 image-only captions, same generic prompt, batch4, caps256/512/1024.',
            'question_image_caption_policy': 'Official QA.image_caption verbatim, preserved from seed42; newly generated query captions are unused by retrieval.',
            'graph_policy': 'Fresh chronological Ward/silhouette graph; fixed original external PCA basis; no LLM graph summaries.',
            'candidate_policy': 'Fresh Qwen seed1042, same gold-free prompts/parser, four hypotheses, batch32, cap384, three format attempts.',
            'reader_policy': 'Fresh Qwen seed1042 greedy bf16 SDPA; text batch16/image batch4, cap256, same official prompts, no input truncation.',
            'judge_policy': 'Owned by suite; fixed original DeepSeek seed42 five-grade judge.',
            'historical_model_outputs_reused': False,
            'source_sha256': {str(p): sha(p) for p in [*code_paths(), *reference_files]},
            'snapshot_sha256': snapshots,
            'caption_protocol_sha256': sha(CAPTIONS / 'protocol.json'),
            'phases': list(PHASES), 'markers_directory': str(GROUP / 'stages')}
    plan['experiment_signature'] = digest(plan)
    dump(GROUP / 'plan.json', plan)
    return complete('prepare', [GROUP / 'plan.json', CAPTIONS / 'protocol.json', CAPTIONS / 'image_jobs.jsonl'])


def actual_seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    from scripts.evaluation import mem_gallery_master_reader as reader
    reader.SEED = SEED
    assert reader.SEED == read(CAPTIONS / 'protocol.json')['seed'] == SEED


def caption_worker(index):
    require_marker('prepare')
    assert index in (0, 1)
    cfg = read(CAPTIONS / 'protocol.json')
    assert cfg['seed'] == SEED and cfg['do_sample'] is False and cfg['batch_size'] == 4
    from scripts.data import generate_mem_gallery_qwen_captions as native
    actual_seed()
    native.worker(CAPTIONS, index)
    status = read(CAPTIONS / f'worker{index}_status.json')
    assert status['status'] == 'complete'
    return complete('caption-worker', [CAPTIONS / f'captions.worker{index}.jsonl',
        CAPTIONS / f'attempts.worker{index}.jsonl'], index, captions=status['completed'])


def caption_finalize():
    for index in (0, 1):
        require_marker('caption-worker', index)
    from scripts.data import generate_mem_gallery_qwen_captions as native
    from scripts.analysis.audit_memgallery_qwen_seed1042_caption import audit_captions
    native.finalize(CAPTIONS)
    audited = audit_captions(RUN)
    cfg = read(CAPTIONS / 'protocol.json')
    dump(CAPTIONS / 'completion_manifest.json', {'status': 'complete', 'seed': SEED,
         'protocol_signature': cfg['protocol_signature'], 'fresh_generation': True,
         'questions': 1711, 'unique_images': 1393, 'mapped_image_files': 1490,
         'historical_rounds': 1003, 'question_image_rounds': 487,
         'gold_free_caption_inputs': True, 'independent_audit_passed': audited['status'] == 'passed'})
    manifest = {str(p.relative_to(CAPTIONS)): sha(p) for p in sorted(CAPTIONS.rglob('*'))
                if p.is_file() and p.name != 'artifact_manifest.json' and p.suffix != '.log'
                and '.tmp-' not in p.name}
    dump(CAPTIONS / 'artifact_manifest.json', manifest)
    return complete('caption-finalize', [CAPTIONS / 'artifact_manifest.json',
        CAPTIONS / 'independent_audit.json', CAPTIONS / 'completion_manifest.json'])


def shared_inputs():
    require_marker('caption-finalize')
    from scripts.data.mem_gallery_inputs import prepare as native_prepare
    native_prepare(RUN / 'inputs', caption_dir=CAPTIONS)
    # Official queries/gold are fixed; only historical image descriptions change.
    for name in ('questions.jsonl', 'gold.jsonl', 'sanitized_candidate_inputs.jsonl'):
        assert sha(RUN / 'inputs' / name) == sha(REFERENCE / 'inputs' / name), name
    return complete('shared-inputs', list((RUN / 'inputs').glob('*.json*')),
                    questions=1711, memories=3962, official_question_captions_preserved=True)


def modules():
    from scripts.data import mem_gallery_inputs as inputs
    from scripts.memory import mem_gallery_master_answer as original
    from scripts.evaluation import mem_gallery_master_reader as reader
    from scripts.evaluation import mem_gallery_zscore as zscore
    actual_prepare = inputs.prepare

    def fixed_prepare(output_dir=inputs.DEFAULT_OUTPUT, *, root=ROOT, caption_dir=None):
        if caption_dir is not None and Path(caption_dir).resolve() != CAPTIONS:
            raise ValueError('Only fresh seed1042 captions are accepted')
        return actual_prepare(output_dir, root=root, caption_dir=CAPTIONS)

    inputs.prepare = fixed_prepare
    original.DEFAULT_OUTPUT = BASE
    zscore.SOURCE = BASE
    reader.SEED = SEED
    return original, reader, zscore


def copy_inputs(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for source in (RUN / 'inputs').glob('*.json*'):
        target = destination / source.name
        if target.exists():
            assert sha(target) == sha(source)
        else:
            shutil.copyfile(source, target)


def analyze_combo(reader):
    condition = 'master_answer_query+zscore'
    questions = lines(ZSCORE / 'inputs/questions.jsonl')
    gold = {r['id']: r for r in lines(ZSCORE / 'inputs/gold.jsonl')}
    answers = {r['id']: r for r in lines(ZSCORE / 'reader/responses.jsonl')}
    assert len(answers) == len(questions) == 1711
    rows = []
    for question in questions:
        actual = answers[question['id']]
        target = gold[question['id']]
        assert actual['condition'] == condition
        score = reader.score_answer(actual['raw_response'], target['answer'])
        value = {**score, 'raw_response': actual['raw_response'],
                 'top5_memory_ids': actual['top5_memory_ids'],
                 'gold_hits': len(set(actual['top5_memory_ids']) & set(target['gold_ids'])),
                 'gold_clues': len(set(target['gold_ids'])),
                 'hit_token_limit': actual['hit_token_limit']}
        rows.append({'id': question['id'], 'task': question['task'], 'point': question['point'],
                     condition: value})
    overall = {'questions': 1711, condition: {
        key: sum(r[condition][key] for r in rows) / 1711
        for key in ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')}}
    overall[condition].update(empty_responses=sum(not r[condition]['raw_response'] for r in rows),
        token_limit_records=sum(r[condition]['hit_token_limit'] for r in rows))
    dump(ZSCORE / 'reader/paired_results.json', rows)
    dump(ZSCORE / 'reader/summary.json', {'status': 'complete', 'seed': SEED,
         'metric_policy': 'Official normalized EM, stemmed token F1 and smoothed BLEU; all1711 questions.',
         'gold_loaded_only_after_generation': True, 'overall': overall,
         'condition_policy': 'Only requested master_answer_query+zscore has answer inference; other retrieval diagnostics are retained.'})


def publish_evaluations(report):
    paths = []
    for method in report['methods']:
        method_id = 'semrag' if method['id'] == 'semantic_rag' else method['id']
        root = Path(method['root'])
        sources = {str(path): sha(path) for path in (
            root / 'reader/responses.jsonl', root / 'reader/paired_results.json',
            root / 'reader/summary.json', root / 'reader/protocol.json',
            root / 'reader/qa_manifest.jsonl', root / 'retrieval_results.jsonl',
            RUN / 'inputs/questions.jsonl', RUN / 'inputs/gold.jsonl', GROUP / 'plan.json')}
        rows = lines(method['judge_input_path'])
        output = RUN / 'evaluations' / method_id
        output.mkdir(parents=True, exist_ok=True)
        for row in rows:
            row.update(source_condition=method['condition'], source_response_path=str(root / 'reader/responses.jsonl'),
                source_per_question_path=str(root / 'reader/paired_results.json'),
                source_per_question_sha256=sources[str(root / 'reader/paired_results.json')])
        records_path = output / 'records.jsonl'
        records_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        metrics = {k: method[k] for k in ('seed', 'questions', 'correct', 'em', 'f1', 'bleu_1',
                                          'empty_answers', 'token_limit_records')}
        metrics.update(status='complete', method=method_id, source_condition=method['condition'],
                       records_sha256=sha(records_path), source_sha256=sources)
        dump(output / 'metrics.json', metrics)
        paths.extend([records_path, output / 'metrics.json'])
    return paths


def stage(phase, device):
    require_marker('shared-inputs')
    actual_seed()
    original, reader, zscore = modules()
    if phase.startswith('base-'):
        copy_inputs(BASE / 'inputs')
        reader.CONDITIONS = original.CONDITIONS
    if phase == 'base-build':
        original.build(BASE, device)
        return complete(phase, [BASE / 'offline/protocol.json'])
    if phase == 'base-candidates':
        require_marker('base-build')
        reader.generate_candidates(BASE, device=device, batch_size=32, max_attempts=3)
        assert read(BASE / 'candidate_generation_protocol.json')['contract']['seed'] == SEED
        return complete(phase, [BASE / 'candidates.jsonl', BASE / 'candidate_generation_protocol.json'])
    if phase == 'base-retrieve':
        require_marker('base-candidates')
        original.retrieve(BASE, device, 5)
        return complete(phase, [BASE / 'retrieval_results.jsonl', BASE / 'retrieval_protocol.json'])
    if phase == 'base-reader':
        require_marker('base-retrieve')
        reader.run_reader(BASE, device=device, batch_size=16, image_batch_size=4)
        assert read(BASE / 'reader/protocol.json')['contract']['seed'] == SEED
        return complete(phase, [BASE / 'reader/responses.jsonl', BASE / 'reader/protocol.json',
                                BASE / 'reader/validation_output.json'])
    if phase == 'base-analyze':
        require_marker('base-reader')
        reader.analyze(BASE)
        return complete(phase, [BASE / 'reader/summary.json', BASE / 'reader/paired_results.json'])
    if phase == 'zscore-prepare':
        require_marker('base-analyze')
        zscore.prepare(ZSCORE, device, .03, 1.)
        # This annotation is ours; the native numerical implementation is unchanged.
        protocol = read(ZSCORE / 'retrieval_protocol.json')
        for value in protocol['conditions'].values():
            value['5_answering'] = value['5_answering'].replace('seed42', 'seed1042')
        protocol['seed'] = SEED
        protocol['caption_graph_candidates_reused_unchanged'] = True
        protocol['caption_graph_candidates_source'] = str(BASE)
        dump(ZSCORE / 'retrieval_protocol.json', protocol)
        return complete(phase, [ZSCORE / 'retrieval_protocol.json', ZSCORE / 'retrieval_results.jsonl',
                               ZSCORE / 'weighting_diagnostics.jsonl'])
    if phase == 'zscore-reader':
        require_marker('zscore-prepare')
        reader.CONDITIONS = ('master_answer_query+zscore',)
        native_manifest = reader.build_reader_manifest

        def selected_manifest(questions, memories, retrieval, image_hash=reader.sha):
            selected = copy.deepcopy(retrieval)
            for row in selected:
                row['conditions'] = [c for c in row['conditions'] if c['name'] in reader.CONDITIONS]
            return native_manifest(questions, memories, selected, image_hash=image_hash)

        reader.build_reader_manifest = selected_manifest
        reader.run_reader(ZSCORE, device=device, batch_size=16, image_batch_size=4)
        assert read(ZSCORE / 'reader/protocol.json')['contract']['seed'] == SEED
        return complete(phase, [ZSCORE / 'reader/responses.jsonl', ZSCORE / 'reader/protocol.json',
                                ZSCORE / 'reader/validation_output.json'])
    if phase == 'zscore-analyze':
        require_marker('zscore-reader')
        analyze_combo(reader)
        return complete(phase, [ZSCORE / 'reader/summary.json', ZSCORE / 'reader/paired_results.json'])
    if phase == 'audit':
        require_marker('base-analyze')
        require_marker('zscore-analyze')
        from scripts.analysis.audit_memgallery_qwen_seed1042_caption import audit_methods
        report = audit_methods(RUN)
        selected = {'status': 'complete', 'seed': SEED, 'questions_per_method': 1711,
                    'methods': report['methods'], 'source_reference': verify_plan()['methods'],
                    'independent_audit': str(GROUP / 'independent_audit.json')}
        dump(GROUP / 'method_results.json', selected)
        published = publish_evaluations(report)
        return complete(phase, [GROUP / 'independent_audit.json', GROUP / 'method_results.json', *published])
    raise ValueError(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--worker-index', type=int, choices=(0, 1))
    parser.add_argument('--device', default='cuda:0', help='Logical CUDA device; use CUDA_VISIBLE_DEVICES=1 for physical GPU1.')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
        return
    verify_plan()
    if validate_marker(args.phase, args.worker_index):
        print(json.dumps({'stage': marker_name(args.phase, args.worker_index), 'status': 'already_complete'}))
        return
    if args.phase == 'caption-worker':
        caption_worker(args.worker_index)
    elif args.phase == 'caption-finalize':
        caption_finalize()
    elif args.phase == 'shared-inputs':
        shared_inputs()
    else:
        stage(args.phase, args.device)


if __name__ == '__main__':
    main()
