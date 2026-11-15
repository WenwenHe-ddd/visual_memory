"""Independently recompute two-stage RRF from recorded search lists.

This audit deliberately does not import or call the production fusion function.
It can audit completed retrieval checkpoints before reader generation finishes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    failures, comparisons = [], []
    files = sorted((root / 'retrieval_checkpoints').glob('*.json'))
    for path in files:
        row = json.loads(path.read_text())
        detail = row['retrieval_detail']
        candidates = list(dict.fromkeys(detail['original_item_ids']))
        distinct = {}
        for batch in detail['search_batches']:
            key = batch['channel'], json.dumps(batch['query'], sort_keys=True, ensure_ascii=False)
            distinct.setdefault(key, list(dict.fromkeys(batch['result_ids'])))
        channel_lists = {}
        for channel in ('semantic', 'keyword', 'structured'):
            lists = [ids for (ch, _), ids in distinct.items() if ch == channel]
            first = list(dict.fromkeys(identity for ids in lists for identity in ids))
            scores = {}
            for identity in first:
                # Production accumulates each search chronologically with +=.
                # Python 3.12+ sum() uses improved float summation, which can
                # change one-ULP ties and therefore the recorded ranking.
                total = 0.0
                for ids in lists:
                    if identity in ids:
                        total += 1 / (60 + ids.index(identity) + 1)
                scores[identity] = total
            channel_lists[channel] = sorted(first, key=lambda identity: -scores[identity])
        scores = {identity: sum(1 / (60 + ids.index(identity) + 1) for ids in channel_lists.values() if identity in ids)
                  for identity in candidates}
        ranked = sorted(candidates, key=lambda identity: -scores[identity])
        checks = {
            'channel_rankings': channel_lists == detail['fusion']['channel_rankings'],
            'candidate_ranking': ranked == [r['id'] for r in detail['fusion']['ranked_candidates']],
            'candidate_scores': all(scores[r['id']] == r['score'] for r in detail['fusion']['ranked_candidates']),
            'selected_five': ranked[:5] == detail['context_memory_ids'],
            'search_limits': all(b['top_k'] == (10 if b['channel'] == 'semantic' else 5)
                                 for b in detail['search_batches']),
        }
        for name, passed in checks.items():
            if not passed:
                failures.append({'id': row['id'], 'check': name})
        comparisons.append({'id': row['id'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    protocol = json.loads((root / 'protocol.json').read_text())
    return {'audit_passed': not failures, 'questions': len(files),
            'complete': len(files) == protocol['questions'], 'failures': failures,
            'experiment_signature': protocol['experiment_signature'],
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_hashes': comparisons}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    (args.root / 'independent_rank_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoint_hashes'}))
    if not result['audit_passed']:
        raise SystemExit(1)
