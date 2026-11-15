#!/usr/bin/env python3
"""Audit rebuilt-memory parity and storage using the original table's policy."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_seed1024_ours as run
from scripts.memory.memory_store import read_manifest

MIB = 1024 ** 2
METHODS = (
    ('semantic_rag', 'Semantic RAG (baseline)', run.CONDITIONS[0], 'natural'),
    ('ours_1', 'ours_1', run.CONDITIONS[1], 'caption_visual_master'),
    ('ours_2', 'ours_2 (lambda=0.03, alpha=1)', run.CONDITIONS[2], 'natural'),
    ('ours_1_ours_2', 'ours_1 + ours_2 (lambda=0.03, alpha=1)', run.CONDITIONS[3], 'caption_visual_master'),
)


def audit(root=run.OUTPUT):
    root = Path(root).resolve()
    index = run.read(root / 'memory_manifest.json')
    assert index['completed'] and index['memory_leaves'] == 438 and len(index['domains']) == 8
    run.verify_hashes(index['output_sha256'])
    historical_root = ROOT / 'results/memory_stores_20261023'
    components, totals, checks, sources = [], {}, [], {}
    for kind in ('natural', 'caption_visual_master'):
        total = previous_total = payload_total = old_payload_total = 0
        leaves = masters = 0
        domains = sorted((root / 'stores').glob('*/' + kind))
        assert len(domains) == 8
        for domain in domains:
            manifest = read_manifest(domain)
            old = historical_root / domain.parent.name / kind
            old_manifest = read_manifest(old)
            leaves += manifest['leaf_count']
            current_without_sources, old_without_sources = copy.deepcopy(manifest), copy.deepcopy(old_manifest)
            current_without_sources.pop('source_sha256')
            old_without_sources.pop('source_sha256')
            assert current_without_sources == old_without_sources
            payload_equal = True
            for category, filename in [('manifest', 'memory.json')] + [(key, value['path']) for key, value in manifest['files'].items()]:
                path, old_path = domain / filename, old / filename
                size, old_size = path.stat().st_size, old_path.stat().st_size
                current_sha, old_sha = run.sha(path), run.sha(old_path)
                sources[str(path)], sources[str(old_path)] = current_sha, old_sha
                total += size
                previous_total += old_size
                if category != 'manifest':
                    payload_total += size
                    old_payload_total += old_size
                    payload_equal = payload_equal and current_sha == old_sha
                components.append({'store_kind': kind, 'domain': domain.parent.name,
                    'category': category, 'path': str(path), 'bytes': size, 'sha256': current_sha,
                    'seed42_path': str(old_path), 'seed42_bytes': old_size, 'seed42_sha256': old_sha,
                    'byte_identical': current_sha == old_sha})
            if kind == 'caption_visual_master':
                graph = run.read(domain / manifest['files']['graph']['path'])
                masters += len(graph['masters'])
            assert payload_equal
            checks.append({'domain': domain.parent.name, 'store_kind': kind,
                'all_memory_payload_files_byte_identical': payload_equal,
                'manifest_identical_after_removing_source_provenance': True})
        assert leaves == 438 and payload_total == old_payload_total
        totals[kind] = {'storage_bytes': total, 'storage_mib': total / MIB,
            'seed42_storage_bytes': previous_total, 'seed42_storage_mib': previous_total / MIB,
            'storage_delta_bytes': total - previous_total, 'memory_payload_bytes': payload_total,
            'seed42_memory_payload_bytes': old_payload_total, 'memory_leaves': leaves,
            'masters': masters if kind == 'caption_visual_master' else None,
            'files': sum(c['store_kind'] == kind for c in components)}
    summary_path = root / 'summary.json'
    measured = run.read(summary_path)['overall'] if summary_path.exists() else {}
    rows = []
    for method, label, condition, kind in METHODS:
        row = {'method': method, 'label': label, 'condition': condition, **totals[kind]}
        if condition in measured:
            row.update({k: measured[condition][k] for k in ('correct', 'questions', 'accuracy',
                'gold_hits_top5', 'gold_references', 'recall_at_5_micro', 'changed_answers_vs_seed42')})
        rows.append(row)
    # Core graphs/bases/sums are checked against the historical build as well as
    # exported reader payloads. Do not infer byte parity from equal file sizes.
    graph_files = [f for domain in index['domains'] for f in domain['files'].values()]
    graph_byte_matches = sum(f['sha256'] == f['seed42_sha256'] for f in graph_files)
    result = {'rows': rows, 'policy_source': str(ROOT / 'scripts/analysis/audit_memory_storage.py'),
        'policy': 'Sum memory.json and each unique file named in its files map, across all eight domains. Uncompressed logical bytes; MiB=bytes/1048576. No model weights, result logs, queries, answers, build audits or temporary BGE/zscore arrays.',
        'serialization': 'Graph and leaves duplicate text is counted as stored. Source-path provenance is retained in manifest bytes.',
        'size_difference_explanation': 'All core payloads match seed42 byte-for-byte. New source-directory strings change only manifest provenance length: natural -128 bytes, master -192 bytes.',
        'new_vs_seed42_core_build_byte_identical': graph_byte_matches,
        'core_build_files_compared': len(graph_files), 'byte_comparisons': checks,
        'bge_leaf_numeric_payload_not_persisted_or_counted_bytes': 438 * 384 * 8}
    run.dump(root / 'storage_summary.json', result)
    with (root / 'storage_files.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(components[0]))
        writer.writeheader()
        writer.writerows(components)
    with (root / 'storage_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources[str(Path(__file__))] = run.sha(__file__)
    sources[str(root / 'memory_manifest.json')] = run.sha(root / 'memory_manifest.json')
    sources[str(ROOT / 'scripts/analysis/audit_memory_storage.py')] = run.sha(ROOT / 'scripts/analysis/audit_memory_storage.py')
    run.dump(root / 'storage_audit.json', {'passed': True, 'native_store_files': len(components),
        'domain_stores': 16, 'rebuilt_graph_array_files_compared': len(graph_files),
        'rebuilt_graph_array_byte_matches': graph_byte_matches,
        'all_exported_payloads_byte_identical': True, 'source_sha256': sources,
        'artifact_sha256': {str(p): run.sha(p) for p in (root / 'storage_summary.json',
            root / 'storage_files.csv', root / 'storage_summary.csv')}})
    print({r['method']: {'bytes': r['storage_bytes'], 'MiB': r['storage_mib']} for r in rows})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=run.OUTPUT)
    audit(parser.parse_args().root)
