"""Independently reconstruct source inputs, rankings and observation statistics."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.vendor'))
import numpy as np


def main():
    out = Path(sys.argv[1])
    official = Path(sys.argv[2])
    read = lambda name: [json.loads(x) for x in (out / name).read_text().splitlines()]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = json.loads((out/'protocol.json').read_text())
    for path, digest in protocol.get('caption_input_hashes', {}).items():
        assert sha(ROOT/path) == digest, path
    caption_dir = protocol.get('caption_dir')
    caption_field = 'image_caption_qwen3_vl_8b_instruct' if caption_dir else 'image_caption'
    mem, queries, gold, details = [read(n+'.jsonl') for n in ('memories', 'queries', 'gold', 'retrieval_details')]
    vectors = np.load(out / 'embeddings.npz')
    m, q = vectors['memory'], vectors['query']
    mi = {r['id']: i for i, r in enumerate(mem)}
    assert list(vectors['memory_ids']) == [r['id'] for r in mem]
    assert list(vectors['query_ids']) == [r['id'] for r in queries]
    manifest = json.loads((ROOT / 'data/Mem-Gallery/source_manifest.json').read_text())
    raw = {}
    for item in manifest['files']:
        path = ROOT / item['path']
        assert sha(path) == item['sha256'], path
        raw[path.stem] = json.loads(path.read_text())
        if caption_dir:
            raw[path.stem] = json.loads((Path(caption_dir)/'dialog_with_qwen_caption'/path.name).read_text())
        for session in raw[path.stem]['multi_session_dialogues']:
            for row in session['dialogues']:
                if row.get('input_image'):
                    assert mem[mi[path.stem+'/'+row['round']]]['caption'] == ' '.join(row[caption_field])
    cls = next(n for n in ast.parse(official.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'DialogueAgent')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'memory_recall')
    env = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(official), 'exec'), env)
    capture = types.SimpleNamespace(is_multimodal=False, memory=types.SimpleNamespace(recall=lambda x: x))
    expected, primary_ids = {}, set()
    for i, (query, g, detail) in enumerate(zip(queries, gold, details)):
        assert query['id'] == g['id'] == detail['question_id']
        orig = raw[query['task']]['human-annotated QAs'][query['qa_index_0_based']]
        image = {'path': orig['question_image'], 'caption': orig[caption_field]} if orig.get('question_image') else None
        assert query['retrieval_text'] == env['memory_recall'](capture, orig['question'], image), query['id']
        assert not any(k in query for k in ['answer', 'clue', 'clue_ids'])
        gids = {query['task']+'/'+x for x in orig['clue']}
        assert gids == set(g['all_clue_ids'])
        inds = [j for j, r in enumerate(mem) if r['task'] == query['task']]
        score = m[inds] @ q[i]
        order = np.argsort(-np.round(score.astype(float), 6), kind='stable')
        ranked = [mem[inds[j]]['id'] for j in order]
        assert ranked == [r['id'] for r in detail['ranking']], query['id']
        selected = [r for r in ranked if r not in gids][:5]
        assert selected == detail['competitor_ids'] and not gids.intersection(selected)
        primary = bool(orig.get('question_image')) and bool(gids) and gids <= set(ranked)
        assert primary == g['primary_cohort']
        for mid in g['visual_clue_ids']:
            key = query['id']+'|'+mid
            # Accumulate in float64, matching the metric's arithmetic definition.
            sim = float(np.mean([float(m[mi[mid]] @ m[mi[x]]) for x in selected])) if len(selected) == 5 else None
            expected[key] = (ranked.index(mid)+1, sim)
            if primary:
                primary_ids.add(key)
    with (out / 'per_target.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(expected) == len({r['sample_id'] for r in rows})
    for row in rows:
        rank, sim = expected[row['sample_id']]
        assert int(row['target_query_rank']) == rank
        assert (row['retrieved_at_5'] == 'True') == (rank <= 5)
        assert abs(float(row['competitor_similarity'])-sim) < 1e-7, row['sample_id']
    summary = json.loads((out / 'summary.json').read_text())['primary']
    primary = [expected[k] for k in primary_ids]
    assert len(primary) == summary['targets']
    assert sum(rank <= 5 for rank, sim in primary) == summary['retrieved_targets']
    for field, hit in [('retrieved_mean', True), ('missed_mean', False)]:
        assert abs(float(np.mean([s for r, s in primary if (r <= 5) == hit]))-summary[field]) < 1e-7
    artifacts = json.loads((out / 'artifact_manifest.json').read_text())
    assert all(sha(out / n) == digest for n, digest in artifacts.items())
    audit = {'status': 'passed', 'checked_questions': len(queries), 'checked_memory_captions': len(mem),
             'caption_field_verified': caption_field, 'caption_source_hashes_verified': len(protocol.get('caption_input_hashes', {})),
             'checked_target_rows': len(rows), 'primary_targets': len(primary),
             'official_query_builder_exact_matches': len(queries), 'source_sha256_verified': len(manifest['files']),
             'artifact_hashes_verified': len(artifacts), 'full_rankings_recomputed': len(details),
             'competitors_exclude_all_clues': True, 'primary_group_means_independently_recomputed': True,
             'tolerance': 1e-7, 'official_run_bench_sha256': sha(official), 'audit_script_sha256': sha(Path(__file__))}
    (out / 'independent_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    artifacts['independent_audit.json'] = sha(out / 'independent_audit.json')
    (out / 'artifact_manifest.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
