#!/usr/bin/env python3
"""Freeze verified seed42 answers for the Mem-Gallery seed1042 comparison."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
RUN = ROOT / 'results/memgallery_qwen_seed1042_full_20261106'
SOURCE = ROOT / 'results/mem_gallery_master_answer_20261021'
INPUT_MANIFEST = ROOT / 'results/memgallery_qwen_remaining_judge_inputs_20261104/all_eight_methods_manifest.json'
OLD_JUDGE = ROOT / 'results/memgallery_qwen_all_methods_deepseek_judge_20261104'
METHOD_IDS = ('semrag', 'ours_1', 'ours_1_ours_2', 'gen_agents', 'memoryos',
              'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm')
LABELS = ('Semantic RAG（caption 基线）', 'ours_1', 'ours_1＋ours_2（λ=0.03，α=1）',
          'Generative Agents（Top10）', 'MemoryOS', 'SimpleMem(T)（官方，最终5条）',
          'A-Mem（最终5条）', 'VisRAG（CLIP→原图＋历史对话）', 'HippoMM')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@lru_cache(maxsize=None)
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == raw, f'Refusing to change frozen reference: {path}'
    else:
        path.write_text(raw)


def unique(rows, key='id'):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), f'Duplicate {key}'
    return result


def validate_ready(run=RUN):
    run = Path(run).resolve()
    marker = read(run / 'reference_ready.json')
    assert marker['status'] == 'complete' and marker['methods'] == list(METHOD_IDS)
    for path, expected in marker['output_sha256'].items():
        assert sha(path) == expected, path
    return marker


def canonical():
    q = unique(lines(SOURCE / 'inputs/questions.jsonl'))
    g = unique(lines(SOURCE / 'inputs/gold.jsonl'))
    assert len(q) == len(g) == 1711 and set(q) == set(g)
    return q, g


def prepare(run=RUN):
    from scripts.evaluation.mem_gallery_master_reader import score_answer
    from scripts.analysis import export_memgallery_five_grade_judge as common
    run = Path(run).resolve()
    if (run / 'reference_ready.json').exists():
        return validate_ready(run)
    qmap, gold = canonical()
    baseline = common.load_baseline()
    completed = {m['summary']['method_id']: m for m in common.load_multi(OLD_JUDGE, baseline)}
    source_methods = unique(read(INPUT_MANIFEST)['methods'])
    old_base = read(SOURCE / 'reader/summary.json')['overall']
    specs = {}
    for mid, condition in (('semrag', 'baseline_question_only'), ('ours_1', 'master_answer_query')):
        specs[mid] = {'source_root': str(SOURCE), 'source_condition': condition,
            'expected_em_correct': round(old_base[condition]['em'] * 1711), **old_base[condition],
            'raw_path': str(SOURCE / 'reader/responses.jsonl'),
            'scored_path': str(SOURCE / 'reader/paired_results.json'),
            'source_sha256': {str(p): sha(p) for p in (SOURCE / 'reader/responses.jsonl',
                SOURCE / 'reader/paired_results.json', SOURCE / 'reader/summary.json',
                SOURCE / 'reader/protocol.json', SOURCE / 'reader/qa_manifest.jsonl')}}
    for mid in METHOD_IDS[2:]:
        old_id = 'ours1_ours2' if mid == 'ours_1_ours_2' else mid
        value = dict(source_methods[old_id])
        hashes = value['source_sha256']
        raw = [Path(p) for p in hashes if p.endswith('/responses.jsonl') or p.endswith('/qa_responses.jsonl')]
        assert len(raw) == 1, (mid, raw)
        value['raw_path'] = str(raw[0])
        origin = Path(value['source_root'])
        if mid == 'ours_1_ours_2':
            scored = origin / 'reader/paired_results.json'
        elif mid in ('gen_agents', 'memoryos', 'a_mem'):
            scored = origin / mid / 'scored_questions.jsonl'
        elif mid == 'simplemem':
            scored = origin / 'scored_questions.jsonl'
        else:
            scored = origin / 'per_question.jsonl'
        assert scored.is_file(), scored
        value['scored_path'] = str(scored)
        value['source_sha256'] = {**hashes, str(scored): sha(scored)}
        specs[mid] = value
    references, outputs = [], {}
    for mid, label in zip(METHOD_IDS, LABELS):
        source = specs[mid]
        for path, expected in source['source_sha256'].items():
            assert sha(path) == expected, path
        paired = mid in ('semrag', 'ours_1', 'ours_1_ours_2')
        condition = source.get('source_condition', source.get('code_method', mid))
        raw_rows = lines(source['raw_path'])
        if paired:
            raw_rows = [r for r in raw_rows if r['condition'] == condition]
        actual = unique(raw_rows)
        scored_rows = read(source['scored_path']) if paired else lines(source['scored_path'])
        scored = unique(scored_rows)
        assert set(actual) == set(scored) == set(qmap)
        old_id = 'ours1_ours2' if mid == 'ours_1_ours_2' else mid
        if mid == 'semrag':
            judge = baseline
            judge_path = common.BASELINE / 'summary.json'
        elif mid == 'ours_1':
            judge = None
            judge_path = run / 'judge_seed42_ours_1/methods/ours_1/summary.json'
        else:
            judge = completed[old_id]
            judge_path = OLD_JUDGE / 'methods' / old_id / 'summary.json'
        judge_rows = unique(judge['rows']) if judge else None
        rows = []
        for qid, question in qmap.items():
            raw = actual[qid]
            prediction = raw['answer'] if mid == 'hippomm' else raw['raw_response']
            assert isinstance(prediction, str)
            score = score_answer(prediction, gold[qid]['answer'])
            prior = scored[qid][condition] if paired else scored[qid]
            for metric in ('em', 'f1', 'bleu_1'):
                assert abs(score[metric] - prior[metric]) < 1e-12, (mid, qid, metric)
            if judge_rows:
                original = judge_rows[qid]
                assert all(original[k] == v for k, v in {
                    'question': question['question'], 'ground_truth': gold[qid]['answer'],
                    'model_output': prediction}.items())
                judge_score = original['score']
            else:
                judge_score = None
            rows.append({'id': qid, 'task': question['task'], 'point': question['point'],
                'question': question['question'], 'ground_truth': gold[qid]['answer'],
                'model_output': prediction, **{k: score[k] for k in ('em', 'f1', 'bleu_1')},
                'llm_judge': judge_score, 'answer_seed': 42,
                'source_record_id': raw.get('record_id', qid), 'source_condition': condition,
                'source_response_path': source['raw_path'], 'source_responses_sha256': sha(source['raw_path']),
                'source_per_question_path': source['scored_path'],
                'source_per_question_sha256': sha(source['scored_path']),
                'source_hit_token_limit': raw.get('hit_token_limit', False)})
        values = {key: sum(r[key] for r in rows) / 1711 for key in ('em', 'f1', 'bleu_1')}
        correct = sum(int(r['em']) for r in rows)
        assert correct == source['expected_em_correct']
        for metric in ('em', 'f1', 'bleu_1'):
            assert abs(values[metric] - source[metric]) < 1e-12, (mid, metric)
        output = run / 'reference' / mid / 'records.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_output = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
        if output.exists():
            assert output.read_text() == raw_output
        else:
            output.write_text(raw_output)
        hashes = {**source['source_sha256'], str(INPUT_MANIFEST): sha(INPUT_MANIFEST),
                  str(SOURCE / 'inputs/questions.jsonl'): sha(SOURCE / 'inputs/questions.jsonl'),
                  str(SOURCE / 'inputs/gold.jsonl'): sha(SOURCE / 'inputs/gold.jsonl'),
                  str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
        if judge:
            hashes[str(judge_path)] = sha(judge_path)
            if mid != 'semrag':
                jp = OLD_JUDGE / 'methods' / old_id / 'per_question.jsonl'
                hashes[str(jp)] = sha(jp)
            else:
                for name in ('protocol.json', 'requests.jsonl', 'scores.jsonl'):
                    path = common.BASELINE / name
                    hashes[str(path)] = sha(path)
        references.append({'id': mid, 'label': label, 'seed': 42, 'questions': 1711,
            'correct': correct, **values,
            'llm_judge': judge['summary']['llm_judge'] if judge else None,
            'records_path': str(output), 'records_sha256': sha(output),
            'source_condition': condition, 'source_root': source['source_root'],
            'source_sha256': hashes, 'judge_source_path': str(judge_path),
            'judge_reference_policy': 'Existing actual fixed scores' if judge else 'Pending new judge of frozen seed42 answers; this reference file stays unchanged.'})
        outputs[str(output)] = sha(output)
    reference = {'status': 'complete', 'seed': 42, 'questions_per_method': 1711,
        'method_count': 9, 'methods': references,
        'correction': 'Screenshot combo EM760/1711 belongs to GPT-5. Correct Qwen source has756/1711, F1=.6413725327, BLEU1=.5887262497, Judge=.7296902396.',
        'answer_policy': 'Original frozen raw outputs, including historical token-limit outputs; no model/API generation.',
        'ours_1_judge_policy': 'Null in frozen reference; final report reads the specified later judge file.',
        'all_reference_aggregates_and_raw_outputs_recomputed': True}
    path = run / 'reference_seed42.json'
    dump(path, reference)
    outputs[str(path)] = sha(path)
    ready = {'status': 'complete', 'seed': 42, 'methods': list(METHOD_IDS), 'questions_per_method': 1711,
             'records_checked': 9 * 1711, 'all_original_raw_answers_and_metrics_verified': True,
             'output_sha256': outputs, 'builder_sha256': sha(Path(__file__).resolve())}
    dump(run / 'reference_ready.json', ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--phase', choices=('prepare', 'audit'), default='prepare')
    args = parser.parse_args()
    print(json.dumps(prepare(args.run) if args.phase == 'prepare' else validate_ready(args.run), ensure_ascii=False))


if __name__ == '__main__':
    main()
