#!/usr/bin/env python3
"""Export rebuilt DeepSeek memories and measure the original table storage policy."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_seed1024_ours as main
from scripts.memory.memory_store import read_manifest

OLD = ROOT / 'reports/memory_storage_deepseek_memeye_20261102/memory_stores'


def measure(root=main.OUTPUT):
    root = Path(root).resolve()
    protocol = main.read(root / 'protocol.json')
    main.verify(protocol)
    memory_proof = main.read(root / 'memory_manifest.json')
    main.native.verify_hashes(memory_proof['output_sha256'])
    retrieval = main.read(root / 'retrieval_audit.json')
    main.native.verify_hashes(retrieval['output_sha256'])
    metadata = main.read(root / 'inputs/encoding.json')
    memories = main.lines(root / 'inputs/memories.jsonl')
    encodings = {a['task']: a['leaf'] for a in retrieval['encoding']}
    components, comparisons = [], []
    totals = {}
    for kind in ('natural', 'caption_visual_master'):
        total = old_total = leaves_count = masters_count = 0
        for task in sorted({m['task'] for m in memories}):
            source = root / 'domains' / task / 'memory'
            graph = main.read(source / 'graph.json')
            original = {m['image_id']: m for m in memories if m['task'] == task}
            leaves = [{'image_id': n['id'], 'memory_id': original[n['id']]['id'],
                       'dialogue': n['dialogue'], 'caption': n['caption']} for n in graph['image_nodes']]
            assert all(n['caption'] == original[n['id']]['caption'] and n['dialogue'] == original[n['id']]['dialogue']
                       for n in graph['image_nodes'])
            dest = root / 'memory_stores' / task / kind
            main.freeze(dest / 'leaves.json', leaves)
            files = {'leaves': 'leaves.json'}
            encoding = {k: copy.deepcopy(metadata[k]) for k in ('text_encoder', 'query_prefix')}
            encoding['allowed_leaf_truncations'] = [dict(id=leaf['image_id'],
                original_tokens=record['original_tokens'], retained_limit=record['retained_limit'],
                text_sha256=hashlib.sha256((leaf['dialogue'] + '\nMemory description: ' + leaf['caption']).encode()).hexdigest())
                for record in encodings[task]['affected_inputs'] for leaf in leaves
                if record['id'] in (leaf['memory_id'], leaf['image_id'])]
            sources = {str(p): main.sha(p) for p in (root / 'inputs/encoding.json',
                root / 'inputs/memories.jsonl', main.SOURCE / 'captions.jsonl')}
            if kind == 'caption_visual_master':
                for key, name in (('graph', 'graph.json'), ('visual_sums', 'master_sums.npy')):
                    target = dest / name
                    if target.exists():
                        assert main.sha(target) == main.sha(source / name)
                    else:
                        shutil.copyfile(source / name, target)
                    files[key] = name
                    sources[str(source / name)] = main.sha(source / name)
                encoding['image_encoder'] = metadata['image_encoder']
                masters_count += len(graph['masters'])
            manifest = {'schema_version': 1, 'offline_method': kind, 'domain': task,
                'leaf_count': len(leaves), 'leaf_text_format': 'dialogue + newline + Memory description: + full caption',
                'files': {key: {'path': name, 'sha256': main.sha(dest / name)} for key, name in files.items()},
                'encoding': encoding, 'source_sha256': sources,
                'export_policy': 'existing DeepSeek Flash offline memory; no new clustering; no QA data included'}
            main.freeze(dest / 'memory.json', manifest)
            read_manifest(dest)
            leaves_count += len(leaves)
            old = OLD / task / kind
            old_manifest = read_manifest(old)
            manifest_without_sources, old_without_sources = copy.deepcopy(manifest), copy.deepcopy(old_manifest)
            manifest_without_sources.pop('source_sha256')
            old_without_sources.pop('source_sha256')
            comparisons.append({'task': task, 'kind': kind,
                'manifest_same_except_source_provenance': manifest_without_sources == old_without_sources})
            for name in ['memory.json', *files.values()]:
                path, old_path = dest / name, old / name
                size, old_size = path.stat().st_size, old_path.stat().st_size
                total += size
                old_total += old_size
                components.append({'kind': kind, 'task': task, 'file': name, 'path': str(path),
                    'bytes': size, 'sha256': main.sha(path), 'seed42_bytes': old_size,
                    'seed42_sha256': main.sha(old_path), 'byte_identical': main.sha(path) == main.sha(old_path)})
        assert leaves_count == 438
        totals[kind] = {'storage_bytes': total, 'storage_mib': total / 1048576,
            'seed42_storage_bytes': old_total, 'storage_delta_bytes': total - old_total,
            'memory_leaves': leaves_count, 'masters': masters_count if kind == 'caption_visual_master' else None}
    measured = main.read(root / 'summary.json')['overall'] if (root / 'summary.json').exists() else {}
    rows = []
    for method, condition, kind in (
        ('semrag', main.CONDITIONS[0], 'natural'), ('ours_1', main.CONDITIONS[1], 'caption_visual_master'),
        ('ours_2', main.CONDITIONS[2], 'natural'), ('ours_1_ours_2', main.CONDITIONS[3], 'caption_visual_master')):
        row = {'method': method, 'condition': condition, **totals[kind],
               **{k: v for k, v in retrieval['metrics'][condition].items() if k != 'questions'}}
        if condition in measured:
            row.update({k: measured[condition][k] for k in ('correct', 'accuracy', 'changed_answers_vs_seed42')})
        rows.append(row)
    result = {'rows': rows, 'policy_source': str(ROOT / 'scripts/analysis/audit_deepseek_memeye_storage.py'),
        'policy': 'Original DeepSeek table schema: memory.json plus its files map, all eight domains. Complete dialogue+caption and duplicate graph/leaves text count. Basis is offline-update state and excluded by the original table policy. Temporary BGE/zscore vectors, QA, model weights, logs and construction audits excluded. MiB=bytes/1048576.',
        'memory_rebuilt': True, 'manifest_comparisons': comparisons,
        'build_core_file_byte_matches': sum(f['byte_identical'] for d in memory_proof['domains'] for f in d['files'].values()),
        'build_core_files_compared': sum(len(d['files']) for d in memory_proof['domains'])}
    main.dump(root / 'storage_summary.json', result)
    for name, values in (('storage_summary.csv', rows), ('storage_files.csv', components)):
        with (root / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    main.dump(root / 'storage_audit.json', {'passed': True, 'store_files': len(components),
        'source_sha256': {str(p): main.sha(p) for p in (Path(__file__), root / 'protocol.json',
            root / 'memory_manifest.json', root / 'retrieval_audit.json')},
        'artifact_sha256': {str(root / n): main.sha(root / n) for n in
                           ('storage_summary.json', 'storage_summary.csv', 'storage_files.csv')}})
    print({r['method']: r['storage_mib'] for r in rows})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=main.OUTPUT)
    measure(parser.parse_args().root)
