"""Pack this run's flagged SimpleMem source reviews without evaluation data.

Only review_queue.jsonl, jobs.jsonl and protocol.json are opened from the fresh
attribution directory. No source hashes are recursively followed. Packet fields
are explicitly projected, sources are shared through job-local alias bindings,
and no review decisions or manual_review.jsonl are created.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'results/memeye_deepseek_seed1024_full_20261106/metrics/simplemem_source_attribution'
INPUT_NAMES = ('review_queue.jsonl', 'jobs.jsonl', 'protocol.json')
SOURCE_FIELDS = ('kind', 'round_id', 'timestamp', 'speaker', 'user', 'caption',
                 'assistant', 'entry_id', 'memory')
ANNOTATION_FIELDS = ('id', 'status', 'evidence', 'ambiguous_sources', 'note')
EVIDENCE_FIELDS = ('source', 'field', 'quote')
FORBIDDEN_KEYS = {'selected', 'qid', 'qids', 'question', 'question_id', 'question_ids',
                  'gold', 'gold_ids', 'gold_choice', 'ground_truth', 'options', 'correct',
                  'accuracy', 'prediction', 'predicted_choice', 'model_output'}


def digest(value):
    value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def annotation_protocol_digest(value):
    # Preserve reflexion_memory.hash_object's original default JSON separators;
    # packet/source identifiers above deliberately use compact serialization.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def assert_safe(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError('Unexpected evaluation metadata in projected packet')
        for item in value.values():
            assert_safe(item)
    elif isinstance(value, list):
        for item in value:
            assert_safe(item)


def annotation(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Expected one original pass annotation or null')
    out = {k: value[k] for k in ANNOTATION_FIELDS if k in value}
    if 'evidence' in out:
        out['evidence'] = [{k: edge[k] for k in EVIDENCE_FIELDS if k in edge}
                           for edge in out['evidence']]
    return out


def unique(rows, key):
    result = {row[key]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError('Duplicate ' + key)
    return result


def partition(groups, minimum=15, preferred=20, maximum=25):
    """Keep each job intact; minimize undersized packets before size deviation."""
    if not 1 <= minimum <= preferred <= maximum:
        raise ValueError('Expected 1 <= minimum <= preferred <= maximum')
    sizes = [len(group['entries']) for group in groups]
    if any(n > maximum or n < 1 for n in sizes):
        raise ValueError('A job exceeds the packet maximum; use a larger --max-entries')
    best = [(0, 0, 0)] + [None] * len(groups)
    previous = [None] * (len(groups) + 1)
    for end in range(1, len(groups) + 1):
        size = 0
        for start in range(end - 1, -1, -1):
            size += sizes[start]
            if size > maximum:
                break
            if best[start] is None:
                continue
            cost = (best[start][0] + int(size < minimum),
                    best[start][1] + max(0, minimum - size),
                    best[start][2] + (size - preferred) ** 2)
            if best[end] is None or cost < best[end]:
                best[end], previous[end] = cost, start
    result, end = [], len(groups)
    while end:
        start = previous[end]
        if start is None:
            raise ValueError('Could not partition complete jobs')
        result.append(groups[start:end]); end = start
    return result[::-1]


def load_inputs(root, synthetic_fixture=False):
    root = Path(root).resolve()
    if synthetic_fixture:
        if not root.is_relative_to(Path('/tmp').resolve()):
            raise ValueError('Synthetic fixture inputs must be under /tmp')
    elif root != DEFAULT.resolve():
        raise ValueError('Only this fresh seed1024 attribution directory is allowed')
    paths = {name: root / name for name in INPUT_NAMES}
    for path in paths.values():
        if path.resolve().parent != root:
            raise ValueError('Attribution input symlinks may not escape this directory')
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Attribution review inputs are not ready: ' + ', '.join(missing))
    # Read and hash the same in-memory bytes; never follow protocol input paths.
    raw = {name: path.read_bytes() for name, path in paths.items()}
    protocol = json.loads(raw['protocol.json'])
    if (protocol.get('seed') != 42 or protocol.get('memory_generation_seed') != 1024 or
            protocol.get('signature') != annotation_protocol_digest({k: v for k, v in protocol.items() if k != 'signature'})):
        raise ValueError('Expected a valid fresh-memory1024/evaluator42 protocol')
    if synthetic_fixture and protocol.get('synthetic_fixture') is not True:
        raise ValueError('Synthetic fixtures must be explicitly marked in their protocol')
    if not synthetic_fixture:
        expected = DEFAULT.parents[1] / 'text_methods/simplemem'
        if Path(protocol['source_root']).resolve() != expected.resolve():
            raise ValueError('Attribution source belongs to a different memory run')
    queue = [json.loads(line) for line in raw['review_queue.jsonl'].splitlines() if line.strip()]
    jobs = [json.loads(line) for line in raw['jobs.jsonl'].splitlines() if line.strip()]
    bindings = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    return root, protocol, unique(queue, 'entry_id'), unique(jobs, 'id'), bindings


def build(root=DEFAULT, minimum=15, preferred=20, maximum=25, synthetic_fixture=False):
    root, protocol, queue, jobs, bindings = load_inputs(root, synthetic_fixture)
    grouped = defaultdict(list)
    for record in queue.values():
        job = jobs[record['job_id']]
        target = next((t for t in job['targets'] if t['entry_id'] == record['entry_id']), None)
        if target is None or any(target[k] != record[k] for k in ('alias', 'memory')):
            raise ValueError('Review entry differs from its original job target')
        if record['task'] != job['task'] or record.get('sources') != job['sources']:
            raise ValueError('Review entry source/task differs from the original job')
        item = {k: record[k] for k in ('entry_id', 'alias', 'memory', 'quote_issues',
            'review_reasons', 'duplicate_evidence_sources')}
        item.update(automatic_status=record['status'], pass1=annotation(record['pass1']),
                    pass2=annotation(record['pass2']))
        assert_safe(item)
        grouped[job['id']].append(item)
    groups = []
    for identity in sorted(grouped, key=lambda k: (jobs[k]['task'], jobs[k]['window_index'], k)):
        job = jobs[identity]
        entries = unique(grouped[identity], 'entry_id')
        ordered = [entries[t['entry_id']] for t in job['targets'] if t['entry_id'] in entries]
        groups.append(dict(job_id=identity, task=job['task'], window_index=job['window_index'],
                           sources=job['sources'], entries=ordered))
    packets, index, seen = [], [], []
    for number, chunk in enumerate(partition(groups, minimum, preferred, maximum), 1):
        source_map, packet_jobs = {}, []
        for group in chunk:
            aliases, source_bindings = set(), []
            for source in group['sources']:
                alias = source['alias']
                if not re.fullmatch(r'[RP][1-9][0-9]*', alias) or alias in aliases:
                    raise ValueError('Source aliases must remain unique within each original job')
                aliases.add(alias)
                projected = {k: source[k] for k in SOURCE_FIELDS if k in source}
                source_id = 'S' + digest(projected)
                if source_id in source_map and source_map[source_id] != projected:
                    raise ValueError('Source hash collision')
                source_map[source_id] = projected
                source_bindings.append(dict(alias=alias, source_id=source_id))
            packet_jobs.append({k: group[k] for k in ('job_id', 'task', 'window_index', 'entries')} |
                               dict(source_bindings=source_bindings))
        count = sum(len(job['entries']) for job in packet_jobs)
        packet_id = f'packet_{number:04d}'
        packet = dict(schema='simplemem_source_review_packet_v1', packet_id=packet_id,
            instructions='Review every listed memory against only its job-bound original sources. '
                'Preserve source aliases in evidence. Compare complete semantics, time and referents; '
                'literal substring matching alone is insufficient. Do not infer provenance from topic similarity. '
                'Only the reviewer writes the final decision, evidence, ambiguity, note and identity.',
            original_annotation_rules=protocol['rules'],
            review_output_fields=['entry_id', 'decision', 'evidence', 'ambiguous_sources', 'note', 'reviewer'],
            allowed_decision_labels=['accept', 'partial', 'unresolved', 'unmatched'],
            jobs=packet_jobs, sources=source_map, entry_count=count, input_sha256=bindings)
        assert_safe(packet)
        ids = [e['entry_id'] for job in packet_jobs for e in job['entries']]
        seen.extend(ids); packets.append(packet)
        index.append(dict(packet_id=packet_id, file=packet_id + '.json', entries=count,
            jobs=len(packet_jobs), unique_sources=len(source_map), entry_ids=ids,
            source_references=sum(len(j['source_bindings']) for j in packet_jobs),
            within_requested_range=minimum <= count <= maximum,
            size_exception_reason=None if minimum <= count <= maximum else
                'Keep complete original jobs together; the remaining group sizes cannot all meet the minimum.'))
    if len(seen) != len(set(seen)) or set(seen) != set(queue):
        raise ValueError('Every flagged entry must occur in exactly one packet')
    return root, packets, index, dict(input_sha256=bindings, flagged_entries=len(queue),
        jobs_with_flagged_entries=len(groups), packets=len(packets),
        requested_packet_size=dict(minimum=minimum, preferred=preferred, maximum=maximum),
        every_flagged_entry_exactly_once=True, review_decisions_written=0,
        source_files_read=list(INPUT_NAMES), evaluator_seed=42, memory_generation_seed=1024,
        synthetic_fixture=synthetic_fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'verify'), required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--min-entries', type=int, default=15)
    parser.add_argument('--target-entries', type=int, default=20)
    parser.add_argument('--max-entries', type=int, default=25)
    parser.add_argument('--synthetic-fixture', action='store_true')
    args = parser.parse_args()
    try:
        root, packets, index, manifest = build(args.root, args.min_entries,
            args.target_entries, args.max_entries, args.synthetic_fixture)
    except FileNotFoundError as error:
        print(json.dumps(dict(status='awaiting_attribution_review_inputs', reason=str(error))))
        raise SystemExit(2)
    output = (args.output or root / 'review_packets').resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError('Packet output must be a dedicated directory under the input root')
    content = {item['file']: encode(packet) for item, packet in zip(index, packets)}
    for item in index:
        item['sha256'] = hashlib.sha256(content[item['file']].encode()).hexdigest()
    manifest.update(status='prepared_for_independent_review', packet_index=index,
                    tool_sha256=file_sha(__file__))
    assert_safe(manifest)
    content['manifest.json'] = encode(manifest)
    if args.phase == 'verify':
        for name, expected in content.items():
            if (output / name).read_text() != expected:
                raise ValueError('Packet or manifest changed: ' + name)
    else:
        if output.exists():
            existing = {p.name for p in output.iterdir()}
            if existing != set(content):
                raise ValueError('Refuse to alter a nonempty or differently prepared packet directory')
            for name, expected in content.items():
                if (output / name).read_text() != expected:
                    raise ValueError('Existing review packet differs; preserve previous packets')
        else:
            output.mkdir(parents=True)
            for name, value in content.items():
                (output / name).write_text(value)
    print(json.dumps(dict(status='verified' if args.phase == 'verify' else manifest['status'],
        flagged_entries=manifest['flagged_entries'], packets=manifest['packets'],
        counts=[item['entries'] for item in index], manifest_sha256=file_sha(output / 'manifest.json'),
        review_decisions_written=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
