"""Read-only replay of evidence normalization over frozen pre-QA snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory.m2a_evidence import normalize_evidence, POLICY


def audit(source, output):
    if output.exists():
        raise ValueError('Use a fresh report file')
    reports = []
    hashes = {}
    for path in sorted(source.glob('*/*/attempt_*/semantic_before_qa.json')):
        raw_path = path.with_name('raw_before_qa.json')
        if not raw_path.exists():
            continue
        data, raw_data = path.read_bytes(), raw_path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        hashes[str(raw_path)] = hashlib.sha256(raw_data).hexdigest()
        valid = {r['msg_id'] for r in json.loads(raw_data)}
        memories = json.loads(data)
        changes, rejected = [], []
        for memory in memories:
            original = memory['evidence_ids']
            try:
                normalized = normalize_evidence(original, valid)
                expected = json.loads(original) if isinstance(original, str) else original
                if normalized != expected:
                    changes.append(dict(memory_id=memory['id'], original=original, canonical=normalized))
                # Exercise the exact unpacking operation that previously crashed.
                expanded = {rid for start,end in normalized for rid in range(start,end+1)}
                assert expanded <= valid
            except (TypeError, ValueError) as exc:
                rejected.append(dict(memory_id=memory['id'], original=original, error=str(exc)))
        reports.append(dict(snapshot=str(path), memories=len(memories), changes=changes, rejected=rejected))
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during replay: ' + path)
    result = dict(policy=POLICY, snapshots=len(reports), memories=sum(r['memories'] for r in reports),
        normalized_memories=sum(len(r['changes']) for r in reports),
        rejected_memories=sum(len(r['rejected']) for r in reports),
        source_files_unchanged=True, source_sha256=hashes, details=reports,
        scope='Read-only format/ID-existence replay; not QA rerun or semantic factuality verification',
        implementation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [Path(__file__), ROOT/'scripts/memory/m2a_evidence.py']})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.source.resolve(), args.output.resolve())
    print(json.dumps({k:v for k,v in report.items() if k not in ('details','source_sha256','implementation_sha256','policy')}, indent=2))
