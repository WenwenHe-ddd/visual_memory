"""Independent checks of full rankings, actual reader payloads and scored answers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import string
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
import numpy as np
from scripts.evaluation import mem_gallery_master_reader as reader
from scripts.evaluation import mem_gallery_zscore as experiment


def audit_retrieval(output):
    p = experiment.verify(output)
    rows = experiment.lines(output / 'retrieval_results.jsonl')
    diagnostics = experiment.lines(output / 'weighting_diagnostics.jsonl')
    questions = experiment.lines(output / 'inputs/questions.jsonl')
    memories = experiment.lines(output / 'inputs/memories.jsonl')
    refs = reader.by_id(experiment.lines(experiment.SOURCE / 'retrieval_results.jsonl'))
    rmap, dmap, qmap = map(reader.by_id, (rows, diagnostics, questions))
    assert len(rmap) == len(dmap) == len(qmap) == 1711
    assert set(rmap) == set(dmap) == set(qmap) == set(refs)
    group_count = 0
    for qid, q in qmap.items():
        r, d = rmap[qid], dmap[qid]
        pool = [m for m in memories if m['task'] == q['task']]
        ids = [m['id'] for m in pool]
        assert d['memory_ids'] == ids
        assert r['master_choices'] == refs[qid]['master_choices']
        graph = experiment.read(experiment.SOURCE / 'offline/memory' / q['task'] / 'graph.json')
        nodes = {n['id']: n for n in graph['image_nodes']}
        groups = {m['id']: [] for m in graph['masters']}
        for i, m in enumerate(pool):
            if m['is_visual']:
                groups[nodes[m['node_id']]['master_id']].append(i)
        choices = {c['master_id']: c['candidate_index'] for c in r['master_choices']}
        for ci, condition in enumerate(experiment.CONDITIONS):
            stats = d['conditions'][condition]
            raw = np.array(stats['raw']); weighted = np.array(stats['weighted'])
            assert np.isfinite(raw).all() and np.isfinite(weighted).all()
            expected_groups = [(list(range(len(pool))), 0)] if ci == 0 else [(idx, 1+choices[mid]) for mid, idx in groups.items()]
            assert [(g['indices'], g['query_index']) for g in stats['groups']] == expected_groups
            expected = raw.copy()
            for g in stats['groups']:
                group_count += 1
                idx = g['indices']; a = raw[idx]; white = np.array(g['white'])
                np.testing.assert_array_equal(a, g['raw'])
                tol = 32*np.finfo(float).eps*max(1., np.abs(white).max())
                if len(idx) == 1:
                    z = white.copy()
                elif white.std() <= tol:
                    z = np.zeros(len(idx))
                else:
                    z = (white-white.mean())/white.std()
                np.testing.assert_allclose(z, g['z'], atol=1e-12)
                corrected = a.copy()
                if len(idx) > 1 and g['raw_status'] != 'zero_variance' and g['z_status'] != 'zero_variance':
                    corrected += p['alpha']*a.std()*np.tanh(z)
                np.testing.assert_allclose(corrected, g['weighted'], atol=1e-12)
                assert np.max(np.abs(corrected-a)) <= p['alpha']*a.std()+1e-12
                expected[idx] = corrected
            np.testing.assert_allclose(weighted, expected, atol=1e-12)
            order = np.argsort(-weighted, kind='stable')
            rank = [ids[i] for i in order]
            c = r['conditions'][ci]
            assert c['name'] == condition and c['ranking_memory_ids'] == rank
            assert c['top5_memory_ids'] == rank[:5] and len(set(rank[:5])) == 5
            np.testing.assert_array_equal(c['top5_scores'], weighted[order[:5]])
            ref = refs[qid]['conditions'][ci]
            raw_order = np.argsort(-raw, kind='stable')
            assert [ids[i] for i in raw_order[:5]] == ref['top5_memory_ids']
            np.testing.assert_allclose(raw[raw_order[:5]], ref['top5_scores'], atol=2e-6, rtol=0)
    report = {'passed': True, 'questions': len(qmap), 'conditions': len(experiment.CONDITIONS),
              'groups_checked': group_count, 'raw_top5_matches_original': 3422,
              'full_weighted_rankings_checked': 3422, 'master_choices_preserved': True,
              'text_only_master_fallback_preserved': True, 'gold_loaded': False,
              'retrieval_sha256': reader.sha(output / 'retrieval_results.jsonl')}
    reader.dump(output / 'retrieval_audit.json', report)
    return report


def normalize(text):
    import re
    value = str(text).lower()
    value = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', value)
    value = value.replace('_', 'UNDERSCOREPLACEHOLDER')
    value = re.sub(r'\b(a|an|the|and)\b', ' ', value)
    value = ''.join(c if c not in string.punctuation else ' ' for c in value)
    return ' '.join(value.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_').split())


def audit_final(output):
    from nltk.stem import PorterStemmer
    retrieval = audit_retrieval(output)
    p = experiment.read(output / 'reader/protocol.json')
    contract = p['contract']
    experiment.verify_hashes(contract['source_sha256'])
    old = experiment.read(experiment.SOURCE / 'reader/protocol.json')['contract']
    old_summary = experiment.read(experiment.SOURCE / 'reader/summary.json')
    experiment.verify_hashes(old_summary['artifacts_sha256'])
    old_pairs = reader.by_id(experiment.read(experiment.SOURCE / 'reader/paired_results.json'))
    old_answers = reader.by_id(experiment.lines(experiment.SOURCE / 'reader/responses.jsonl'), 'record_id')
    for k in ('model', 'system_prompt', 'task_instructions', 'seed', 'do_sample', 'max_new_tokens',
              'batch_size_text', 'batch_size_images', 'dtype', 'query_image_max_pixels', 'input_truncation'):
        assert contract[k] == old[k]
    manifest = experiment.lines(output / 'reader/qa_manifest.jsonl')
    responses = experiment.lines(output / 'reader/responses.jsonl')
    assert reader.digest(manifest) == contract['manifest_sha256']
    amap, mmap = reader.by_id(responses, 'record_id'), reader.by_id(manifest, 'record_id')
    assert len(amap) == len(mmap) == 3422 and set(amap) == set(mmap)
    qs = reader.by_id(experiment.lines(output / 'inputs/questions.jsonl'))
    ms = reader.by_id(experiment.lines(output / 'inputs/memories.jsonl'))
    gold = reader.by_id(experiment.lines(output / 'inputs/gold.jsonl'))
    pairs = reader.by_id(experiment.read(output / 'reader/paired_results.json'))
    ranked = reader.by_id(experiment.lines(output / 'retrieval_results.jsonl'))
    image_cache, scored, stemmer = {}, [], PorterStemmer()
    same_payload = {c: {'questions': 0, 'answer_changes': 0, 'em_gained': 0, 'em_lost': 0}
                    for c in experiment.CONDITIONS}
    for record_id, m in mmap.items():
        a = amap[record_id]; q = qs[m['id']]; target = gold[m['id']]
        assert a['run_id'] == p['run_id'] and a['experiment_signature'] == p['experiment_signature']
        c = next(c for c in ranked[m['id']]['conditions'] if c['name'] == m['condition'])
        assert a['top5_memory_ids'] == m['top5_memory_ids'] == c['top5_memory_ids']
        assert m['reader_text'] == reader.reader_text(q, c['top5_memory_ids'], ms)
        assert m['query_image_paths'] == q['query_image_paths']
        for im in m['query_images']:
            if im['path'] not in image_cache:
                image_cache[im['path']] = reader.sha(im['path'])
            assert image_cache[im['path']] == im['sha256']
        payload = {'system': reader.READER_SYSTEM, 'user_text': m['reader_text'], 'query_images': m['query_images']}
        assert reader.digest(payload) == a['payload_sha256'] == m['payload_sha256']
        pa, ga = normalize(a['raw_response']), normalize(target['answer'])
        pt, gt = pa.split(), ga.split()
        common = sum((Counter(stemmer.stem(w) for w in pt) & Counter(stemmer.stem(w) for w in gt)).values())
        f1 = 2*common/(len(pt)+len(gt)) if common else 0.
        s = pairs[m['id']][m['condition']]
        assert s['em'] == float(pa == ga) and abs(s['f1']-f1) < 1e-15
        assert s['gold_hits'] == len(set(target['gold_ids']) & set(m['top5_memory_ids']))
        assert s['gold_clues'] == len(set(target['gold_ids']))
        baseline = experiment.BASE_CONDITIONS[experiment.CONDITIONS.index(m['condition'])]
        old_answer = old_answers[m['id'] + '::' + baseline]
        old_scored = old_pairs[m['id']][baseline]
        assert old_scored['raw_response'] == old_answer['raw_response']
        assert old_scored['em'] == float(normalize(old_answer['raw_response']) == ga)
        assert old_scored['gold_hits'] == len(set(target['gold_ids']) & set(old_answer['top5_memory_ids']))
        if old_answer['payload_sha256'] == a['payload_sha256']:
            bucket = same_payload[m['condition']]
            bucket['questions'] += 1
            bucket['answer_changes'] += old_answer['raw_response'] != a['raw_response']
            bucket['em_gained'] += s['em'] > old_scored['em']
            bucket['em_lost'] += s['em'] < old_scored['em']
        if not a['fresh_inference_executed_for_this_record']:
            src = amap[a['same_run_shared_inference_record_id']]
            assert src['fresh_inference_executed_for_this_record']
            assert src['payload_sha256'] == a['payload_sha256'] and src['raw_response'] == a['raw_response']
        scored.append(s)
    report = {'passed': True, 'retrieval': retrieval, 'reader_records': len(scored),
              'fresh_inferences': sum(a['fresh_inference_executed_for_this_record'] for a in responses),
              'unique_payloads': len({m['payload_sha256'] for m in manifest}),
              'query_images_verified': len(image_cache), 'empty_answers': sum(not a['raw_response'] for a in responses),
              'token_limit_records': sum(a['hit_token_limit'] for a in responses),
              'caption_graph_and_query_inputs_unchanged': True, 'same_reader_configuration_as_original': True,
              'independent_normalized_em_f1_and_recall_verified': True,
              'historical_same_payload_variation': same_payload,
              'audit_source_sha256': {str(Path(__file__).resolve()): reader.sha(__file__)},
              'sha256': {str(output / f): reader.sha(output / f) for f in
                         ['retrieval_results.jsonl', 'retrieval_protocol.json', 'reader/protocol.json',
                          'reader/qa_manifest.jsonl', 'reader/responses.jsonl', 'reader/paired_results.json',
                          'comparison.json', 'metrics.csv', 'report.html']}}
    assert report['fresh_inferences'] == report['unique_payloads']
    reader.dump(output / 'final_audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=experiment.DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=['retrieval', 'final'], default='final')
    args = parser.parse_args()
    result = audit_retrieval(args.output) if args.stage == 'retrieval' else audit_final(args.output)
    print(json.dumps(result, ensure_ascii=False))
