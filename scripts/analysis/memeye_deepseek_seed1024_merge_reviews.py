"""Validate reviewer-authored packet results and merge the original manual schema.

No decisions are inferred. Reads only this run's review packets, their three
allowlisted attribution inputs, and reviewer-written result envelopes. Does not
follow protocol source paths, freeze annotations, score, load a model or use API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis import memeye_deepseek_seed1024_review_packets as packets
from scripts.analysis.simplemem_attribution_recall import validate_manual, duplicate_evidence_sources

FIELDS = {'entry_id', 'decision', 'evidence', 'ambiguous_sources', 'note', 'reviewer'}
ENVELOPE = {'packet_id', 'packet_sha256', 'input_manifest_sha256', 'reviews'}


def require(value, message):
    if not value:
        raise ValueError(message)


def safe_child(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path != root, 'Review paths must stay under this attribution directory')
    return path


def prepared_packets(root, packet_root, synthetic_fixture=False):
    manifest_path = packet_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    sizes = manifest['requested_packet_size']
    _, generated, index, basis = packets.build(root, sizes['minimum'], sizes['preferred'],
                                               sizes['maximum'], synthetic_fixture)
    packet_bytes = {item['file']: packets.encode(packet) for item, packet in zip(index, generated)}
    for item in index:
        item['sha256'] = hashlib.sha256(packet_bytes[item['file']].encode()).hexdigest()
    expected = dict(basis, status='prepared_for_independent_review', packet_index=index,
                    tool_sha256=packets.file_sha(packets.__file__))
    require(manifest == expected, 'Packet manifest no longer matches its original inputs/tool')
    for name, expected_text in packet_bytes.items():
        path = safe_child(root, packet_root / name)
        require(path.read_text() == expected_text, 'Original review packet changed: ' + name)
    return manifest, packets.file_sha(manifest_path), generated


def validate_record(item, record, target, job):
    require(isinstance(item, dict) and set(item) == FIELDS, 'Manual row must contain exactly the original six fields')
    require(item['entry_id'] == record['entry_id'], 'Manual entry identity differs')
    require(isinstance(item['note'], str) and isinstance(item['reviewer'], str) and
            item['note'].strip() and item['reviewer'].strip(), 'Every decision needs a reviewer and substantive reason')
    require(isinstance(item['evidence'], list) and isinstance(item['ambiguous_sources'], list), 'Evidence and ambiguity must be arrays')
    require(len(item['ambiguous_sources']) == len(set(item['ambiguous_sources'])), 'Duplicate ambiguity aliases')
    sources = {s['alias']: s for s in job['sources']}
    edges = set()
    for evidence in item['evidence']:
        require(isinstance(evidence, dict) and set(evidence) == {'source', 'field', 'quote'}, 'Every evidence edge needs source/field/quote only')
        alias, field, quote = evidence['source'], evidence['field'], evidence['quote']
        require(alias in sources and isinstance(quote, str), 'Unknown source alias or non-text quote')
        source = sources[alias]
        allowed = {'memory'} if source['kind'] == 'prior_entry' else {'user', 'assistant', 'caption'}
        require(field in allowed and isinstance(source.get(field), str), 'Quote field differs from its source kind')
        require(quote in source[field], 'Quote must be copied verbatim from the named original field')
        edge = (alias, field, quote)
        require(edge not in edges, 'Duplicate evidence edge')
        edges.add(edge)
    require(not (set(e['source'] for e in item['evidence']) & set(item['ambiguous_sources'])),
            'The same source cannot be simultaneously definite and ambiguous')
    if item['decision'] == 'partial':
        require(bool(item['evidence']), 'Partial means some content has definite source evidence; otherwise use unresolved/unmatched')
    # The unchanged evaluator enforces normalized 4-word/20-character quotes,
    # known aliases/statuses, no evidence on unresolved/unmatched and the
    # evidence/no-ambiguity requirements for accept. This call performs no I/O.
    accepted = validate_manual(item, target, job)
    duplicates = duplicate_evidence_sources(accepted, job)
    prior = sorted({e['source'] for e in item['evidence'] if sources[e['source']]['kind'] == 'prior_entry'})
    warnings = []
    if duplicates:
        warnings.append(dict(entry_id=item['entry_id'], kind='literal_quote_has_alternative_sources',
                             alternatives=duplicates, note='The reviewer must justify semantic uniqueness; matching text alone does not establish provenance.'))
    if prior:
        warnings.append(dict(entry_id=item['entry_id'], kind='previous_memory_requires_conservative_propagation', aliases=prior))
    return warnings


def validate(root, packet_root, review_root, synthetic_fixture=False):
    # The packet loader validates the fresh-run boundary before opening inputs.
    root, _, queue, jobs, input_hashes = packets.load_inputs(root, synthetic_fixture)
    packet_root = safe_child(root, packet_root or root / 'review_packets')
    review_root = safe_child(root, review_root or root / 'review_results')
    manifest, manifest_sha, generated = prepared_packets(root, packet_root, synthetic_fixture)
    packet_index = {item['packet_id']: item for item in manifest['packet_index']}
    expected_by_packet = {p['packet_id']: {e['entry_id'] for j in p['jobs'] for e in j['entries']} for p in generated}
    input_files, manual, warnings, complete_packets = {}, {}, [], set()
    files = sorted(review_root.glob('packet_*.json')) if review_root.exists() else []
    for candidate in files:
        path = safe_child(root, candidate)
        require(path.parent == review_root and path.stem in packet_index, 'Unexpected reviewer packet result path')
        raw = path.read_bytes(); envelope = json.loads(raw)
        packets.assert_safe(envelope)
        require(isinstance(envelope, dict) and set(envelope) == ENVELOPE, 'Reviewer output must contain the four envelope fields')
        identity = envelope['packet_id']
        require(identity == path.stem and identity not in complete_packets, 'Duplicate/misnamed packet result')
        require(envelope['input_manifest_sha256'] == manifest_sha and
                envelope['packet_sha256'] == packet_index[identity]['sha256'], 'Reviewer result is bound to different packet bytes')
        require(isinstance(envelope['reviews'], list), 'Reviewer reviews must be an array')
        actual = packets.unique(envelope['reviews'], 'entry_id')
        require(set(actual) == expected_by_packet[identity], 'A packet result must cover exactly its assigned entries')
        for entry_id, item in actual.items():
            require(entry_id not in manual, 'Entry reviewed more than once across packets')
            record = queue[entry_id]; job = jobs[record['job_id']]
            target = next(t for t in job['targets'] if t['entry_id'] == entry_id)
            warnings.extend(validate_record(item, record, target, job))
            manual[entry_id] = item
        complete_packets.add(identity)
        input_files[str(path)] = hashlib.sha256(raw).hexdigest()
    require(set(manual) <= set(queue), 'Review includes an unassigned entry')
    missing_packets = sorted(set(packet_index) - complete_packets)
    missing_entries = set(queue) - set(manual)
    report = dict(status='ready_to_merge' if not missing_entries else 'awaiting_reviewer_results',
        flagged_entries=len(queue), decisions_validated=len(manual), missing_entries=len(missing_entries),
        complete_packets=len(complete_packets), expected_packets=len(packet_index), missing_packets=missing_packets,
        decision_counts=dict(Counter(item['decision'] for item in manual.values())),
        warnings=warnings, warning_counts=dict(Counter(item['kind'] for item in warnings)),
        input_sha256=input_hashes, packet_manifest_sha256=manifest_sha, reviewer_file_sha256=input_files,
        validation_tool_sha256=packets.file_sha(__file__), generated_decisions=0,
        synthetic_fixture=synthetic_fixture)
    packets.assert_safe(report)
    return root, [manual[identity] for identity in queue if identity in manual], report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('validate', 'merge'), required=True)
    parser.add_argument('--root', type=Path, default=packets.DEFAULT)
    parser.add_argument('--packets', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    root, manual, report = validate(args.root, args.packets, args.reviews, args.synthetic_fixture)
    if args.phase == 'merge':
        require(not (root / 'annotation_lock.json').exists(), 'Annotations are sealed; preserve all review decisions')
        require(report['status'] == 'ready_to_merge', 'Every flagged entry must have a validated reviewer result before merge')
        path = safe_child(root, args.output or root / 'manual_review.jsonl')
        require(path == root / 'manual_review.jsonl', 'Merged output must be the original evaluator manual_review.jsonl under this attribution root')
        text = ''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n' for item in manual)
        if path.exists():
            require(path.read_text() == text, 'Existing manual review differs; refuse to overwrite prior decisions')
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        report.update(status='merged_validated_reviewer_results', merged_sha256=packets.file_sha(path),
                      merged_entries=len(manual))
    report_path = safe_child(root, args.report or root / 'review_validation.json')
    require(re.fullmatch(r'review_validation(?:_[A-Za-z0-9_-]+)?\.json', report_path.name),
            'Validation reports must use a dedicated review_validation*.json filename')
    report_path.write_text(packets.encode(report))
    print(json.dumps({k: v for k, v in report.items() if k not in
        {'input_sha256', 'reviewer_file_sha256', 'warnings'}}, ensure_ascii=False))
    if report['status'] == 'awaiting_reviewer_results':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
