#!/usr/bin/env python3
"""Fresh seed-1024 Open retrieval/answers using the newly rebuilt ours stores."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]

import numpy as np

from scripts.evaluation import memeye_seed1024_ours as rebuilt
from scripts.evaluation import memeye_seed1024_open as fresh
from scripts.evaluation import mem_gallery_master_reader as common
from scripts.evaluation.memeye_open_semrag_inputs import SYSTEM, payload_hash
from scripts.memory import online_retrieval as online
from scripts.memory.caption_visual_master import choose_master_answers
from scripts.memory.online_zscore import ZScoreWeighting

OUTPUT = rebuilt.OUTPUT / 'open'
METHODS = ('semantic_rag', 'ours_1', 'ours_2', 'ours_1_ours_2')
HISTORICAL = ROOT / 'results/memeye_qwen_other_open_20261105/ours'
sha, digest, read, lines, dump = rebuilt.sha, rebuilt.digest, rebuilt.read, rebuilt.lines, rebuilt.dump


class OpenMultiPath:
    name = 'multi_path'

    def plan(self, question, memory):
        candidate = memory.candidates[question['id']]
        assert candidate['question'] == question['question']
        statements = dict(zip('ABCD', candidate['candidate_statements']))
        assert len(statements) == 4
        scores, choices = choose_master_answers(memory.centers, statements,
            memory.clip_encoder, memory.tokenizer, memory.clip_limit)
        queries = [question['retrieval_query'] + '\nCandidate answer: ' + statements[c] for c in 'ABCD']
        groups = [online.RetrievalGroup(master['id'], indices, int(choice), {
            'candidate_index': int(choice), 'selected_statement': statements['ABCD'[choice]],
            'clip_statement_scores': {c: float(scores[i, j]) for j, c in enumerate('ABCD')},
        }) for i, (master, indices, choice) in enumerate(zip(memory.masters, memory.indices, choices))]
        return queries, groups


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    protocol_path = output / 'preparation_protocol.json'
    if protocol_path.exists():
        protocol = read(protocol_path)
        rebuilt.verify_signature(protocol)
        rebuilt.verify_hashes(protocol['source_sha256'])
        rebuilt.verify_hashes(protocol['artifact_sha256'])
        return protocol
    questions = lines(fresh.CANONICAL / 'questions.jsonl')
    assert len(questions) == len({q['id'] for q in questions}) == 371
    assert all(not set(q) & {'options', 'gold_ids', 'gold_choice', 'clue', 'answer'} for q in questions)
    original_mcq = {q['id']: q for q in lines(rebuilt.QUESTIONS)}
    assert set(original_mcq) == {q['id'] for q in questions}
    sanitized = [{'id': q['id'], 'question': q['question'],
                  'query_image_caption': '\n'.join(q['query_image_captions'])} for q in questions]
    output.mkdir(parents=True, exist_ok=True)
    for name in ('questions.jsonl', 'gold.jsonl'):
        values = lines(fresh.CANONICAL / name)
        fresh.frozen_rows(output / 'inputs' / name, values)
    fresh.frozen_rows(output / 'inputs/sanitized_candidate_inputs.jsonl', sanitized)
    sources = [Path(__file__), Path(fresh.__file__), Path(common.__file__),
               rebuilt.OUTPUT / 'protocol.json', rebuilt.QUESTIONS, rebuilt.TEXTS,
               fresh.CANONICAL / 'questions.jsonl', fresh.CANONICAL / 'gold.jsonl',
               ROOT / 'scripts/memory/online_retrieval.py', ROOT / 'scripts/memory/caption_visual_master.py',
               ROOT / 'scripts/memory/online_zscore.py', ROOT / 'scripts/memory/ridge_whitening.py',
               ROOT / 'scripts/evaluation/memeye_open_semrag_inputs.py']
    historical_paths = [ROOT / 'results/memeye_qwen_semrag_open_20261105/retrieval.jsonl']
    historical_paths += [HISTORICAL / method / 'retrieval.jsonl' for method in METHODS[1:]]
    sources.extend(p for p in historical_paths if p.exists())
    protocol = {'dataset': 'MemEye Open', 'questions': 371, 'methods': list(METHODS),
        'seed': 1024, 'judge_seed': 42, 'top_k': 5, 'lambda': .03, 'alpha': 1.,
        'new_memory_source': str(rebuilt.OUTPUT / 'stores'), 'historical_memories_reused': False,
        'original_captions_frozen': True, 'historical_candidate_reuse': False,
        'candidate_policy': 'Original MemGallery prompt/parser, four fresh hypotheses from Open question and cached question-image caption; batch8, greedy seed1024.',
        'query_policy': 'Official Open question and original cached query-image caption; no MCQ options, gold or history.',
        'changed_open_questions_vs_mcq': sum(q['question'] != original_mcq[q['id']]['question'] for q in questions),
        'reader_policy': 'Original Open system/template and native query images; five original dialogue/full-caption memories.',
        'source_sha256': {str(p): sha(p) for p in sources},
        'artifact_sha256': {str(output / 'inputs' / name): sha(output / 'inputs' / name) for name in
                           ('questions.jsonl', 'gold.jsonl', 'sanitized_candidate_inputs.jsonl')}}
    protocol['experiment_signature'] = digest(protocol)
    dump(protocol_path, protocol)
    print(json.dumps({'stage': 'open_prepared', 'output': str(output), 'questions': 371,
                      'changed_open_questions': protocol['changed_open_questions_vs_mcq']}), flush=True)
    return protocol


def candidates(output, device='cuda:0', model=None):
    prepare(output)
    saved = (common.SEED, common.load_runtime, common.generate_batch, common.ensure_protocol)
    seed_model = model
    existing = lines(Path(output) / 'candidates.jsonl') if (Path(output) / 'candidates.jsonl').exists() else []
    if len(existing) < 371 and seed_model is None:
        # Its constructor calls common.load_runtime, so construct it before the
        # candidate adapter temporarily replaces that function.
        seed_model = fresh.SeededMultimodalQwen(device, seed=1024)

    def load(requested_device):
        assert seed_model is not None and seed_model.seed == 1024
        return seed_model, None, seed_model.environment

    def generate(runtime, rows, cap):
        conversations = [[{'role': 'system', 'content': row['system']},
                          {'role': 'user', 'content': [{'type': 'text', 'text': row['user_text']}]}]
                         for row in rows]
        started = time.perf_counter()
        outputs = runtime[0].generate(conversations, cap, sampling_temperature=0.)
        return [{**row, 'batch_elapsed_seconds': time.perf_counter() - started} for row in outputs]

    def ensure(path, contract):
        contract = copy.deepcopy(contract)
        contract['source_sha256'].update({str(Path(__file__)): sha(__file__), str(Path(fresh.__file__)): sha(fresh.__file__)})
        contract['transport'] = 'Fresh native seed1024 Qwen; original candidate prompt, parser and request layout.'
        return saved[3](path, contract)

    try:
        common.SEED, common.load_runtime, common.generate_batch, common.ensure_protocol = 1024, load, generate, ensure
        report = common.generate_candidates(output, device=device, batch_size=8, max_attempts=3)
    finally:
        common.SEED, common.load_runtime, common.generate_batch, common.ensure_protocol = saved
    return report, seed_model


def retrieve(output=OUTPUT, device='cpu'):
    output = Path(output).resolve()
    preparation = prepare(output)
    if (output / 'retrieval_validation.json').exists():
        audit = read(output / 'retrieval_validation.json')
        rebuilt.verify_hashes(audit['output_sha256'])
        return audit
    memory_index = read(rebuilt.OUTPUT / 'memory_manifest.json')
    assert memory_index['completed'] and memory_index['memory_leaves'] == 438
    rebuilt.verify_hashes(memory_index['output_sha256'])
    validation = read(output / 'candidate_validation.json')
    assert validation['passed'] and validation['questions'] == 371
    assert validation['candidates_sha256'] == sha(output / 'candidates.jsonl')
    generated = common.by_id(lines(output / 'candidates.jsonl'))
    sanitized = common.by_id(lines(output / 'inputs/sanitized_candidate_inputs.jsonl'))
    questions = lines(output / 'inputs/questions.jsonl')
    for q in questions:
        candidate = generated[q['id']]
        assert candidate['source_input_sha256'] == common.digest(sanitized[q['id']])
        assert not candidate['gold_or_history_used']
    # The shared memory loader only needs ID lookup. OpenMultiPath validates the
    # question and consumes newly generated hypotheses without fabricating options.
    loader_candidates = [{**generated[q['id']], 'question': q['question']} for q in questions]
    fresh.frozen_rows(output / 'open_candidates.jsonl', loader_candidates)
    rebuilt.seed_all()
    args = SimpleNamespace(device=device, candidates=output / 'open_candidates.jsonl')
    online.configure_device(args)
    retrieved = {m: [] for m in METHODS}
    manifests = {m: [] for m in METHODS}
    diagnostics = []
    for task in sorted({q['task'] for q in questions}):
        selected = [q for q in questions if q['task'] == task]
        encoder = None
        for mode, strategy, methods in (
            ('natural', online.FullScanRetrieval(), (METHODS[0], METHODS[2])),
            ('caption_visual_master', OpenMultiPath(), (METHODS[1], METHODS[3]))):
            args.mode, args.retrieval = mode, strategy.name
            args.memory_dir = rebuilt.OUTPUT / 'stores' / task / mode
            memory, metadata, _, _ = online.load_memory(args, selected)
            if encoder is None:
                encoder = online.load_text_encoder(metadata['text_encoder'], device)
            for q in selected:
                result = online.retrieve_question(q, memory, encoder, metadata['query_prefix'], 5,
                                                 ZScoreWeighting(.03, 1.), strategy)
                diagnostics.append({'id': q['id'], 'memory_type': mode, 'groups': result['groups'],
                                    'leaf_scores': result['leaf_scores']})
                for method, condition, score_name in zip(methods, result['conditions'], ('raw_score', 'weighted_score')):
                    ids = condition['top5_memory_ids']
                    assert len(ids) == len(set(ids)) == 5
                    top = condition['top5']
                    scores = np.array([leaf[score_name] for leaf in result['leaf_scores']])
                    order = np.argsort(-scores, kind='stable')
                    all_ids = [leaf['memory_id'] for leaf in result['leaf_scores']]
                    context = 'Retrieved dialogue and image descriptions (ranked by relevance):\n\n' + '\n\n'.join(
                        f'Memory {rank}: {memory.texts[all_ids.index(mid)]}' for rank, mid in enumerate(ids, 1))
                    user_text = context + '\n\nQuestion:\n' + q['question'] + '\n\nAnswer concisely.'
                    retrieved[method].append({'id': q['id'], 'task': task, 'retrieval_query': q['retrieval_query'],
                        'top_memory_ids': ids, 'top_scores': [r['score'] for r in top],
                        'full_rank_memory_ids': [all_ids[i] for i in order],
                        'full_rank_scores': [float(scores[i]) for i in order]})
                    manifests[method].append({'id': q['id'], 'task': task, 'point': q['point'],
                        'question': q['question'], 'top_memory_ids': ids, 'context': context,
                        'query_image_paths': q['query_image_paths'], 'system': SYSTEM, 'user_text': user_text,
                        'payload_sha256': payload_hash(SYSTEM, user_text, q['query_image_paths'])})
            del memory
        del encoder
        gc.collect()
        print(json.dumps({'stage': 'open_retrieval', 'domain': task, 'completed': len(retrieved[METHODS[0]])}), flush=True)
    gold = common.by_id(lines(output / 'inputs/gold.jsonl'))
    sources = {**preparation['source_sha256'], **{str(p): sha(p) for p in (
        output / 'preparation_protocol.json', output / 'candidates.jsonl', output / 'candidate_generation_protocol.json',
        output / 'candidate_validation.json', rebuilt.OUTPUT / 'memory_manifest.json')}}
    sources.update(memory_index['output_sha256'])
    summaries, output_hashes = {}, {}
    for method in METHODS:
        root = output / method
        by_id = common.by_id(retrieved[method])
        reader_by_id = common.by_id(manifests[method])
        assert len(by_id) == len(reader_by_id) == 371
        values = [by_id[q['id']] for q in questions]
        reader_values = [reader_by_id[q['id']] for q in questions]
        fresh.frozen_rows(root / 'retrieval.jsonl', values)
        fresh.frozen_rows(root / 'reader_manifest.jsonl', reader_values)
        historical_path = (ROOT / 'results/memeye_qwen_semrag_open_20261105/retrieval.jsonl'
                           if method == 'semantic_rag' else HISTORICAL / method / 'retrieval.jsonl')
        hits = sum(len(set(v['top_memory_ids']) & set(gold[v['id']]['gold_ids'])) for v in values)
        denominator = sum(len(gold[v['id']]['gold_ids']) for v in values)
        summary = {'questions': 371, 'gold_hits_top5': hits, 'gold_references': denominator,
                   'recall_at_5_micro': hits / denominator}
        if historical_path.exists():
            historical = common.by_id(lines(historical_path))
            summary['changed_top5_order_vs_seed42'] = sum(v['top_memory_ids'] != historical[v['id']]['top_memory_ids'] for v in values)
            summary['changed_top5_set_vs_seed42'] = sum(set(v['top_memory_ids']) != set(historical[v['id']]['top_memory_ids']) for v in values)
        summaries[method] = summary
        protocol = {'dataset': 'MemEye Open', 'method': method, 'seed': 1024, 'judge_seed': 42,
            'questions': 371, 'memory_leaves': 438, 'masters': memory_index['masters'],
            'top_k': 5, 'lambda': .03 if method in METHODS[2:] else None,
            'alpha': 1. if method in METHODS[2:] else 0., 'memory_rebuilt': True,
            'historical_candidate_reuse': False, 'retrieval_device': device,
            'source_sha256': sources, 'retrieval_metrics': summary,
            'artifact_sha256': {str(p): sha(p) for p in (root / 'retrieval.jsonl', root / 'reader_manifest.jsonl')}}
        protocol['experiment_signature'] = digest(protocol)
        dump(root / 'protocol.json', protocol)
        for p in (root / 'protocol.json', root / 'retrieval.jsonl', root / 'reader_manifest.jsonl'):
            output_hashes[str(p)] = sha(p)
    fresh.frozen_rows(output / 'master_query_audit.jsonl', diagnostics)
    audit = {'passed': True, 'seed': 1024, 'questions_per_method': 371, 'methods': summaries,
        'fresh_hypothesis_questions': 371, 'fresh_hypotheses': 1484, 'memory_rebuilt': True,
        'all_contexts_five_distinct_memories': True, 'historical_vectors_reused': False,
        'mcq_options_or_gold_used_for_retrieval': False, 'output_sha256': output_hashes}
    dump(output / 'retrieval_validation.json', audit)
    return audit


def run(output=OUTPUT, device='cuda:0', retrieval_device='cpu'):
    output = Path(output).resolve()
    prepare(output)
    model = None
    _, model = candidates(output, device)
    retrieve(output, retrieval_device)
    if model is None and any(not (output / m / 'reader/completion.json').exists() for m in METHODS):
        model = fresh.SeededMultimodalQwen(device, seed=1024)
    for method in METHODS:
        fresh.run_manifest(output / method, method, device, model=model)
    return analyze(output)


def analyze(output=OUTPUT):
    output = Path(output).resolve()
    retrieval = read(output / 'retrieval_validation.json')
    rebuilt.verify_hashes(retrieval['output_sha256'])
    methods = copy.deepcopy(retrieval['methods'])
    for method in METHODS:
        root = output / method
        status = read(root / 'reader/completion.json')
        assert status['status'] == 'complete' and status['answered'] == 371
        assert status['responses_sha256'] == sha(root / 'reader/responses.jsonl')
        methods[method]['open_answers'] = status
        historical_path = (ROOT / 'results/memeye_qwen_semrag_open_20261105/reader/responses.jsonl'
                           if method == 'semantic_rag' else HISTORICAL / method / 'reader/responses.jsonl')
        if historical_path.exists():
            historical = common.by_id(lines(historical_path))
            now = lines(root / 'reader/responses.jsonl')
            methods[method]['changed_open_answers_vs_seed42'] = sum(r['raw_response'] != historical[r['id']]['raw_response'] for r in now)
        fresh.collect(root)
    summary = {'status': 'open_answers_complete', 'seed': 1024, 'judge_seed': 42,
               'questions_per_method': 371, 'methods': methods, 'judge_status': 'pending separate fixed-seed42 judge',
               'historical_answers_reused': False}
    dump(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'candidates', 'retrieve', 'run', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--retrieval-device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'open_run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'prepare':
            prepare(args.output)
        elif args.phase == 'candidates':
            candidates(args.output, args.device)
        elif args.phase == 'retrieve':
            retrieve(args.output, args.retrieval_device)
        elif args.phase == 'run':
            run(args.output, args.device, args.retrieval_device)
        else:
            analyze(args.output)


if __name__ == '__main__':
    main()
