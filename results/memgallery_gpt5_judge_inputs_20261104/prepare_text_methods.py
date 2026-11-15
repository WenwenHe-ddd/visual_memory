"""Freeze and audit the four existing GPT-5 MemGallery text-method answers.

This preparation performs no model calls and never changes source artifacts.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import score_answer

OUT = Path(__file__).resolve().parent
CANON = ROOT / 'results/memgallery_qwen_semrag_deepseek_judge_20261104/requests.jsonl'
CONFIGS = [
    ('gen_agents', 'Generative Agents', 'gen_agents',
     'mem_gallery_gpt5_gen_agents_20261028', 'gen_agents/reader', 376),
    ('memoryos', 'MemoryOS', 'memoryos',
     'mem_gallery_gpt5_memoryos_20261030', 'memoryos/reader', 676),
    ('simplemem', 'SimpleMem (official 0.3.0, final 5)', 'simplemem',
     'mem_gallery_gpt5_simplemem030_20261031', 'reader', 575),
    ('a_mem', 'A-Mem (final 5)', 'a_mem',
     'mem_gallery_amem_backbones_20261104/gpt5', 'reader', 679),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows):
    result = {r['id']: r for r in rows}
    assert len(result) == len(rows) == 1711
    return result


def dump(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == content, f'Existing frozen artifact differs: {path}'
    else:
        path.write_text(content)


def main():
    canonical_rows = read(CANON)
    canonical = keyed(canonical_rows)
    methods = []
    for mid, label, code_method, root_name, reader_name, expected_correct in CONFIGS:
        source = ROOT / 'results' / root_name
        reader = source / reader_name
        paths = [source / 'inputs/questions.jsonl', source / 'inputs/gold.jsonl',
                 reader / 'responses.jsonl', reader / 'manifest.jsonl',
                 source / 'scored_questions.jsonl', source / 'protocol.json',
                 source / 'final_results.json']
        questions, gold, responses = (keyed(read(p)) for p in paths[:3])
        scored = keyed(read(source / 'scored_questions.jsonl'))
        for mapping in (questions, gold, responses, scored):
            assert set(mapping) == set(canonical)
        final = json.loads((source / 'final_results.json').read_text())
        protocol = json.loads((source / 'protocol.json').read_text())
        assert final['status'] == 'complete'
        assert final['model'] == 'gpt-5-2025-08-07'
        records = []
        metrics = []
        for cid in canonical:
            c, q, g, response, s = (canonical[cid], questions[cid], gold[cid],
                                  responses[cid], scored[cid])
            assert all(q[k] == c[k] for k in ('task', 'point', 'question'))
            assert g['answer'] == c['ground_truth'] == s['answer']
            assert g['point'] == s['point'] == c['point']
            assert response['raw_response'] == s['raw_response']
            assert response['returned_model'] == final['model']
            recomputed = score_answer(response['raw_response'], g['answer'])
            for metric in ('em', 'f1', 'bleu_1'):
                assert abs(recomputed[metric] - s[metric]) < 1e-12, (mid, cid, metric)
            record = {k: c[k] for k in ('id', 'task', 'point', 'question', 'ground_truth')}
            record.update(model_output=response['raw_response'], em=int(recomputed['em']),
                          source_record_id=response['id'])
            for key in ('hit_token_limit', 'finish_reason', 'returned_model', 'response_id',
                        'request_sha256'):
                if key in response:
                    record['source_' + key] = response[key]
            records.append(record)
            metrics.append(recomputed)
        correct = sum(row['em'] for row in records)
        assert correct == expected_correct == final['overall']['correct']
        recomputed_summary = {metric: sum(r[metric] for r in metrics) / 1711
                              for metric in ('em', 'f1', 'bleu_1')}
        for metric, value in recomputed_summary.items():
            if metric in final['overall']:
                assert abs(final['overall'][metric] - value) < 1e-12
        records_path = OUT / f'{mid}.jsonl'
        serialized = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        if records_path.exists():
            assert records_path.read_text() == serialized
        else:
            records_path.write_text(serialized)
        context_counts = dict(sorted(Counter(str(r['context_count']) for r in scored.values()).items()))
        context_policy = protocol.get('context_policy', protocol.get('method'))
        method = {
            'id': mid, 'label': label, 'display_label': label, 'code_method': code_method,
            'answer_model': final['model'], 'expected_em_correct': correct,
            'records_path': str(records_path), 'records_sha256': sha(records_path),
            'source_sha256': {str(p): sha(p) for p in paths + [CANON]},
            'source_root': str(source), 'source_condition': code_method,
            **recomputed_summary, 'context': context_policy,
            'context_counts': context_counts,
            'source_token_limit_records': sum(r.get('source_hit_token_limit') is True for r in records),
            'source_token_limit_known_records': sum('source_hit_token_limit' in r for r in records),
            'source_empty_answer_records': sum(not r['model_output'].strip() for r in records),
            'validation': {
                'question_and_gold_exact_canonical_match': True,
                'responses_exact_match_scored_questions': True,
                'all_em_f1_bleu1_recomputed_per_question': True,
                'source_answer_model_counts': dict(Counter(r['returned_model'] for r in responses.values())),
                'all_original_answers_preserved': True,
            },
        }
        methods.append(method)
        print(mid, {k: method[k] for k in ['expected_em_correct', 'em', 'f1', 'bleu_1',
                                         'source_token_limit_records', 'source_empty_answer_records']})
    manifest = {
        'dataset': 'MemGallery', 'answer_model': 'gpt-5-2025-08-07',
        'questions_per_method': 1711, 'methods': methods,
        'input_preparation': str(Path(__file__).resolve()),
        'input_preparation_sha256': sha(Path(__file__)),
        'metrics_implementation': str(ROOT / 'scripts/evaluation/mem_gallery_master_reader.py'),
        'metrics_implementation_sha256': sha(ROOT / 'scripts/evaluation/mem_gallery_master_reader.py'),
        'policy': 'Frozen original GPT-5 answers; exact canonical question/gold; no model calls or source edits.',
    }
    dump(OUT / 'text_methods_manifest.json', manifest)


if __name__ == '__main__':
    main()
