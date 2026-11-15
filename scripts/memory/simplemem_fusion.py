"""Auditable rank fusion for the official SimpleMem adapter (not upstream code).

Fuse query lists inside each channel first, then fuse the three ranked channels
with equal weight. Repeating a query cannot give a channel an extra vote.
"""
from collections import defaultdict
import json

CHANNELS = ('semantic', 'keyword', 'structured')
RANK_CONSTANT = 60


def fuse_candidates(entries, batches, rank_constant=RANK_CONSTANT):
    if rank_constant <= 0:
        raise ValueError('rank_constant must be positive')
    by_id = {e.entry_id: e for e in entries}
    seen_queries = set()
    scores = {channel: defaultdict(float) for channel in CHANNELS}
    first_seen = {channel: {} for channel in CHANNELS}
    observed = set()
    for batch in batches:
        channel = batch['channel']
        if channel not in CHANNELS:
            raise ValueError('unknown retrieval channel: ' + channel)
        # The reflection phase may repeat an earlier query. Count it once.
        key = (channel, json.dumps(batch['query'], sort_keys=True, ensure_ascii=False))
        ids = list(dict.fromkeys(batch['result_ids']))
        if any(identity not in by_id for identity in ids):
            raise ValueError('search trace contains a candidate absent from the final union')
        observed.update(ids)
        if key in seen_queries:
            continue
        seen_queries.add(key)
        for rank, identity in enumerate(ids, 1):
            first_seen[channel].setdefault(identity, len(first_seen[channel]))
            scores[channel][identity] += 1.0 / (rank_constant + rank)
    if observed != set(by_id):
        raise ValueError('search trace does not cover the final candidate union')
    channel_rankings = {}
    contributions = {identity: {} for identity in by_id}
    for channel in CHANNELS:
        ordered = sorted(scores[channel], key=lambda i: (-scores[channel][i], first_seen[channel][i]))
        channel_rankings[channel] = ordered
        for rank, identity in enumerate(ordered, 1):
            contributions[identity][channel] = {'rank': rank, 'rrf_score': 1.0 / (rank_constant + rank)}
    totals = {identity: sum(v['rrf_score'] for v in channels.values())
              for identity, channels in contributions.items()}
    order = {}
    for i, entry in enumerate(entries):
        order.setdefault(entry.entry_id, i)
    ranked_ids = sorted(by_id, key=lambda i: (-totals[i], order[i]))
    audit = {'method': 'channel_rrf', 'rank_constant': rank_constant,
             'policy': 'RRF within each channel over distinct queries, then equal-weight RRF across channel rankings; '
                       'ties preserve upstream candidate order; reflection searches join the semantic channel',
             'channel_rankings': channel_rankings,
             'ranked_candidates': [{'id': i, 'score': totals[i], 'channels': contributions[i]} for i in ranked_ids]}
    return [by_id[i] for i in ranked_ids], audit
