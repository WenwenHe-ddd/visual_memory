#!/usr/bin/env python3
"""Independent image mappings, raw answers and metrics for seed1042 Gallery."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import string
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
DEFAULT_RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
FIELD = 'image_caption_qwen3_vl_8b_instruct'


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def audit_captions(run=DEFAULT_RUN):
    run = Path(run).resolve()
    cache = run / 'captions'
    cfg = read(cache / 'protocol.json')
    unsigned = dict(cfg)
    signature = unsigned.pop('protocol_signature')
    assert hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() == signature
    assert cfg['seed'] == 1042 and cfg['do_sample'] is False and cfg['batch_size'] == 4
    assert cfg['initial_max_new_tokens'] == 256 and cfg['retry_max_new_tokens'] == [512, 1024]
    jobs = lines(cache / 'image_jobs.jsonl')
    records = [r for index in (0, 1) for r in lines(cache / f'captions.worker{index}.jsonl')]
    attempts = [r for index in (0, 1) for r in lines(cache / f'attempts.worker{index}.jsonl')]
    by_id = {r['id']: r for r in records}
    assert len(jobs) == len(records) == len(by_id) == 1393
    assert set(by_id) == {j['id'] for j in jobs}
    for job in jobs:
        row = by_id[job['id']]
        assert row['protocol_signature'] == signature
        assert row['caption'].strip() and row['ended_with_eos'] and not row['hit_token_limit']
        assert all(row[k] == v for k, v in job.items())
        accepted = [a for a in attempts if a['id'] == row['id'] and a['caption'] == row['caption']
                    and a['ended_with_eos'] and not a['hit_token_limit']]
        assert accepted and all(all(row[k] == v for k, v in a.items()) for a in accepted)
    assert all(a['protocol_signature'] == signature for a in attempts)
    dataset = ROOT / 'data/Mem-Gallery'
    manifest = read(dataset / 'image_manifest.json')
    mappings = lines(cache / 'image_path_captions.jsonl')
    assert len(mappings) == len(manifest['files']) == 1490
    mapped = {row['image_path']: row for row in mappings}
    for entry in manifest['files']:
        assert sha(dataset / entry['path']) == entry['sha256']
        actual = mapped[entry['path']]
        assert actual['image_sha256'] == entry['sha256']
        assert actual[FIELD] == by_id[entry['sha256']]['caption']
    source = read(dataset / 'source_manifest.json')
    histories, query_images = 0, 0
    for file in source['files']:
        path = ROOT / file['path']
        assert sha(path) == file['sha256']
        original = read(path)
        derived = read(cache / 'dialog_with_qwen_caption' / path.name)
        restored = copy.deepcopy(derived)
        for session in restored['multi_session_dialogues']:
            for row in session['dialogues']:
                captions = row.pop(FIELD, None)
                if row.get('input_image'):
                    histories += 1
                    expected = [by_id[sha((path.parent / relative).resolve())]['caption']
                                for relative in row['input_image']]
                    assert captions == expected
        for qa in restored['human-annotated QAs']:
            caption = qa.pop(FIELD, None)
            if qa.get('question_image'):
                query_images += 1
                assert caption == by_id[sha((path.parent / qa['question_image']).resolve())]['caption']
        assert restored == original
    assert histories == 1003 and query_images == 487
    result = {'status': 'passed', 'seed': 1042, 'unique_generated_captions': 1393,
              'mapped_image_files': 1490, 'history_round_records': histories,
              'question_image_records': query_images, 'source_dialog_files': len(source['files']),
              'successful_attempt_backing_verified': True, 'original_fields_exactly_preserved': True,
              'all_raw_image_sha256_verified': True, 'historical_model_outputs_reused': False,
              'caption_prompt_inputs': 'Image and generic instruction only; no question/gold/dialogue.',
              'query_caption_policy': 'Generated query captions retained for compatibility; official QA caption remains retrieval input.',
              'protocol_signature': signature, 'audit_script_sha256': sha(__file__)}
    dump(cache / 'independent_audit.json', result)
    return result


def normalize(value):
    value = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', str(value).lower())
    value = value.replace('_', 'UNDERSCOREPLACEHOLDER')
    value = re.sub(r'\b(a|an|the|and)\b', ' ', value)
    value = ''.join(c if c not in string.punctuation else ' ' for c in value)
    return ' '.join(value.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_').split())


def audit_methods(run=DEFAULT_RUN):
    from nltk.stem import PorterStemmer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from scripts.evaluation import mem_gallery_master_reader as native_reader
    from scripts.evaluation import mem_gallery_zscore as native_zscore
    from scripts.analysis import audit_mem_gallery_zscore as native_audit
    run = Path(run).resolve()
    group, base, zroot = run / 'caption_methods', run / 'caption_methods/base', run / 'caption_methods/zscore'
    assert read(run / 'captions/independent_audit.json')['status'] == 'passed'
    questions = {r['id']: r for r in lines(run / 'inputs/questions.jsonl')}
    gold = {r['id']: r for r in lines(run / 'inputs/gold.jsonl')}
    memories = {r['id']: r for r in lines(run / 'inputs/memories.jsonl')}
    assert len(questions) == len(gold) == 1711 and len(memories) == 3962
    assert sum(m['is_visual'] for m in memories.values()) == 1003
    canonical = ROOT / 'results/mem_gallery_master_answer_20261021/inputs'
    for name in ('questions.jsonl', 'gold.jsonl', 'sanitized_candidate_inputs.jsonl'):
        assert sha(run / 'inputs' / name) == sha(canonical / name)
    basis_hash = sha(ROOT / 'reports/home_caption_ward_visual_20261020/memory/basis.npy')
    checked_nodes = 0
    for graph_path in (base / 'offline/memory').glob('*/graph.json'):
        graph = read(graph_path)
        assert sha(graph_path.parent / 'basis.npy') == basis_hash
        for node in graph['image_nodes']:
            leaf = next(m for m in memories.values() if m['node_id'] == node['id'])
            assert (node['caption'], node['dialogue']) == (leaf['caption'], leaf['dialogue'])
            checked_nodes += 1
    assert checked_nodes == 1003
    candidate_protocol = read(base / 'candidate_generation_protocol.json')['contract']
    assert candidate_protocol['seed'] == 1042 and candidate_protocol['do_sample'] is False
    assert candidate_protocol['batch_size'] == 32 and candidate_protocol['max_new_tokens'] == 384
    candidates = lines(base / 'candidates.jsonl')
    assert len(candidates) == 1711 and {c['id'] for c in candidates} == set(questions)
    native_zscore.SOURCE = base
    native_reader.SEED = 1042
    zscore_audit = native_audit.audit_retrieval(zroot)
    methods, total_checked = [], 0
    stemmer, smoothing = PorterStemmer(), SmoothingFunction().method1
    selections = [(base, 'baseline_question_only', 'semantic_rag'),
                  (base, 'master_answer_query', 'ours_1'),
                  (zroot, 'master_answer_query+zscore', 'ours_1_ours_2')]
    for root, condition, method_id in selections:
        protocol = read(root / 'reader/protocol.json')
        contract = protocol['contract']
        for name, expected in {'seed': 1042, 'do_sample': False, 'max_new_tokens': 256,
                               'batch_size_text': 16, 'batch_size_images': 4,
                               'dtype': 'bfloat16', 'query_image_max_pixels': 1638400,
                               'input_truncation': False}.items():
            assert contract[name] == expected, (condition, name)
        all_answers = lines(root / 'reader/responses.jsonl')
        manifest = {m['record_id']: m for m in lines(root / 'reader/qa_manifest.jsonl')}
        answers = {a['record_id']: a for a in all_answers}
        expected_records = 3422 if root == base else 1711
        assert len(all_answers) == len(answers) == len(manifest) == expected_records
        fresh = {a['payload_sha256']: a for a in all_answers if a['fresh_inference_executed_for_this_record']}
        assert len(fresh) == sum(a['fresh_inference_executed_for_this_record'] for a in all_answers)
        ranking = {r['id']: r for r in lines(root / 'retrieval_results.jsonl')}
        pair = {r['id']: r for r in read(root / 'reader/paired_results.json')}
        scored, records = [], []
        response_hash = sha(root / 'reader/responses.jsonl')
        for qid, question in questions.items():
            rid = qid + '::' + condition
            actual, payload = answers[rid], manifest[rid]
            assert actual['experiment_signature'] == protocol['experiment_signature']
            assert actual['run_id'] == protocol['run_id']
            assert actual['raw_response'] == fresh[actual['payload_sha256']]['raw_response']
            expected_rank = next(c for c in ranking[qid]['conditions'] if c['name'] == condition)
            ids = expected_rank['top5_memory_ids']
            assert len(ids) == len(set(ids)) == 5
            assert ids == actual['top5_memory_ids'] == payload['top5_memory_ids']
            assert payload['reader_text'] == native_reader.reader_text(question, ids, memories)
            assert payload['query_image_paths'] == question['query_image_paths']
            assert all(sha(image['path']) == image['sha256'] for image in payload['query_images'])
            p, g = normalize(actual['raw_response']), normalize(gold[qid]['answer'])
            pt, gt = p.split(), g.split()
            common = sum((Counter(stemmer.stem(w) for w in pt) & Counter(stemmer.stem(w) for w in gt)).values())
            score = {'em': float(p == g), 'f1': 2 * common / (len(pt) + len(gt)) if common else 0.,
                     'bleu_1': float(sentence_bleu([gt], pt, weights=(1, 0, 0, 0), smoothing_function=smoothing)) if pt and gt else 0.}
            for metric, value in score.items():
                assert abs(pair[qid][condition][metric] - value) < 1e-12, (qid, metric)
            scored.append(score)
            records.append({'id': qid, 'task': question['task'], 'point': question['point'],
                'question': question['question'], 'ground_truth': gold[qid]['answer'],
                'model_output': actual['raw_response'], 'source_record_id': rid,
                'answer_model': 'Qwen/Qwen3-VL-8B-Instruct', 'answer_seed': 1042,
                'source_hit_token_limit': actual['hit_token_limit'],
                'source_responses_sha256': response_hash})
        summary = {'id': method_id, 'condition': condition, 'root': str(root), 'seed': 1042,
                   'questions': len(scored), 'correct': sum(int(s['em']) for s in scored),
                   **{m: sum(s[m] for s in scored) / len(scored) for m in ('em', 'f1', 'bleu_1')},
                   'empty_answers': sum(not a['model_output'] for a in records),
                   'token_limit_records': sum(a['source_hit_token_limit'] for a in records)}
        native_summary = read(root / 'reader/summary.json')['overall'][condition]
        for metric in ('em', 'f1', 'bleu_1'):
            assert abs(summary[metric] - native_summary[metric]) < 1e-12
        judge_input = group / f'{method_id}_judge_inputs.jsonl'
        judge_input.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        summary.update(judge_input_path=str(judge_input), judge_input_sha256=sha(judge_input),
                       responses_sha256=sha(root / 'reader/responses.jsonl'))
        methods.append(summary)
        total_checked += len(scored)
    report = {'status': 'passed', 'seed': 1042, 'methods': methods,
              'questions_checked': total_checked, 'graph_nodes_checked': checked_nodes,
              'official_question_captions_preserved': True, 'actual_reader_contexts_checked': True,
              'all_generation_outputs_backed_by_same_run_fresh_inference': True,
              'independent_em_f1_bleu1_checked': True, 'zscore_retrieval_audit': zscore_audit,
              'audit_script_sha256': sha(__file__)}
    dump(group / 'independent_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--phase', choices=('captions', 'methods'), required=True)
    args = parser.parse_args()
    print(json.dumps(audit_captions(args.run) if args.phase == 'captions' else audit_methods(args.run), ensure_ascii=False))


if __name__ == '__main__':
    main()
