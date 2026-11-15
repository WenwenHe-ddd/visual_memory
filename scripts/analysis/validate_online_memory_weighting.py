"""GPU retrieval audit of both online memory objects against frozen 371-question runs.

This is not a new QA accuracy evaluation. Raw columns must reproduce the frozen
natural full scan and the query expansion that previously yielded 195/371.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts import visual_memory_pipeline
from scripts.memory.caption_visual_master import read_json, sha, write_json
from scripts.memory.memory_store import read_manifest

SOURCE = ROOT / 'reports/adaptive_retrieval_comparison_20261020/inputs_by_domain'
REFERENCE = ROOT / 'reports/all_master_answer_query_20261021'
STORES = ROOT / 'results/memory_stores_20261023'


def run(output, device):
    if (output / 'validation.json').exists():
        raise FileExistsError('Use a fresh validation output directory.')
    refs = {r['question_id']: r for line in (REFERENCE / 'retrieval_results.jsonl').read_text().splitlines()
            if line.strip() for r in [json.loads(line)]}
    assert len(refs) == 371
    reference_hashes = {str(p): sha(p) for p in [REFERENCE / 'retrieval_results.jsonl',
                       REFERENCE / 'protocol.json', REFERENCE / 'candidates.jsonl', Path(__file__)]}
    pinned = read_json(REFERENCE / 'protocol.json')['source_sha256']
    stats = {name: {'questions': 0, 'raw_ordered_top5_identical': 0, 'raw_score_max_abs_difference': 0.,
                    'weighted_top5_changed': 0, 'group_count_sum': 0, 'differences': []}
             for name in ['natural', 'caption_visual_master']}
    seen, outputs = {name: set() for name in stats}, {}
    for domain in sorted(SOURCE.iterdir()):
        if not (domain / 'inputs.json').exists():
            continue
        # Only still-relevant frozen sources, not retired historical dependencies.
        for path in [domain / p for p in ('inputs.json', 'protocol.json', 'memory/graph.json', 'memory/master_sums.npy')] + [REFERENCE / 'candidates.jsonl']:
            assert sha(path) == pinned[str(path)], str(path)
        inputs = read_json(domain / 'inputs.json')
        leaves = {leaf['memory_id']: leaf for leaf in inputs['leaves']}
        for name in stats:
            memory_dir = STORES / domain.name / name
            if not (memory_dir / 'memory.json').exists():
                visual_memory_pipeline.main(['offline', '--memory', name, '--memory-source', str(domain),
                                             '--output', str(memory_dir)])
            descriptor = read_manifest(memory_dir)
            assert descriptor['offline_method'] == name
            for path, digest in descriptor['source_sha256'].items():
                assert sha(Path(path)) == digest
            destination = output / domain.name / name
            visual_memory_pipeline.main(['online', '--memory-path', str(memory_dir),
                '--retrieval', 'full_scan' if name == 'natural' else 'multi_path', '--online-weighting', 'z_score',
                '--whitening-lambda', '0.1', '--reweight-alpha', '1', '--device', device,
                '--question-source', str(domain / 'inputs.json'),
                '--output', str(destination), '--top-k', '5'])
            path = destination / 'retrieval_results.json'
            report = read_json(path)
            outputs[str(path)] = sha(path)
            assert report['memory_unchanged'] and not report['vectors_persisted'] and not report['reader_called']
            assert report['execution_devices']['text_encoder'] == device
            assert ('clip_text_encoder' in report['execution_devices']) == (name == 'caption_visual_master')
            stat = stats[name]
            for row in report['questions']:
                qid = row['question_id']; ref = refs[qid]
                assert qid not in seen[name]
                seen[name].add(qid)
                stat['questions'] += 1
                for field in ('question', 'options', 'query_image_paths', 'evaluation'):
                    assert row[field] == ref[field], (qid, field)
                assert row['retrieval_query'] == ref['baseline_retrieval_query']
                raw, weighted = row['conditions']
                reference_condition = ref['conditions'][0 if name == 'natural' else 1]
                same = raw['top5_memory_ids'] == reference_condition['top5_memory_ids']
                stat['raw_ordered_top5_identical'] += same
                stat['weighted_top5_changed'] += raw['top5_memory_ids'] != weighted['top5_memory_ids']
                if not same:
                    stat['differences'].append({'question_id': qid, 'old': reference_condition['top5_memory_ids'],
                                                'new': raw['top5_memory_ids']})
                reference_leaves = {leaf['image_id']: leaf for leaf in ref['leaves']}
                assert set(reference_leaves) == {leaf['image_id'] for leaf in row['leaf_scores']}
                for leaf in row['leaf_scores']:
                    expected = reference_leaves[leaf['image_id']]['baseline_score' if name == 'natural' else 'expanded_score']
                    stat['raw_score_max_abs_difference'] = max(stat['raw_score_max_abs_difference'], abs(leaf['raw_score'] - expected))
                for condition in row['conditions']:
                    for leaf in condition['top5']:
                        assert leaf['reader_text'] == leaves[leaf['memory_id']]['retrieval_text']
                stat['group_count_sum'] += len(row['groups'])
                if name == 'natural':
                    assert len(row['groups']) == 1 and row['groups'][0]['size'] == len(leaves)
                    assert row['groups'][0]['query'] == ref['baseline_retrieval_query']
                else:
                    assert len(row['groups']) == len(ref['masters'])
                    for group, master in zip(row['groups'], ref['masters']):
                        assert (group['group_id'], group['size'], group['choice'], group['selected_statement'], group['query']) == (
                            master['master_id'], master['n'], master['selected_choice'], master['selected_statement'], master['augmented_query'])
            print(json.dumps({'audit_domain': domain.name, 'memory': name, 'cumulative': stat}), flush=True)
    assert all(ids == set(refs) for ids in seen.values())
    for path, expected in reference_hashes.items():
        assert sha(Path(path)) == expected
    passed = all(s['raw_ordered_top5_identical'] == 371 and s['raw_score_max_abs_difference'] < 2e-6 for s in stats.values())
    report = {'passed': passed, 'device': device, 'ridge_lambda': .1, 'alpha': 1,
              'qa_evaluation_run': False, 'stats': stats, 'output_sha256': outputs,
              'reference_sha256': reference_hashes,
              'meaning': 'Fresh online raw retrieval matches frozen baselines; weighted QA accuracy is not measured.'}
    write_json(output / 'validation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('output_sha256', 'reference_sha256')}, indent=2))
    assert passed, 'See validation.json for retrieval differences.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    run(args.output.resolve(), args.device)
