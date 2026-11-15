"""Empirical adaptive search; predicted scores are NOT certified upper bounds.

The running largest observed member-minus-centroid score is transferred to
unvisited clusters. A second estimate scales by cluster radius. Their maximum
is the conservative choice between these two empirical predictions, but is
still not a mathematical bound. Fit uses memory only and no QA-derived tuning.
"""
from __future__ import annotations

import numpy as np

from scripts.graph.text_similarity_bounds import _partition, _unit_query, _unit_rows


class ObservedSpreadSearch:
    def __init__(self, vectors, groups):
        self.vectors = _unit_rows(vectors)
        self.groups = _partition(groups, len(self.vectors))
        self.centers = np.array([self.vectors[g].mean(axis=0) for g in self.groups])
        self.radii = np.array([np.max(np.linalg.norm(self.vectors[g] - center, axis=1))
                               for g, center in zip(self.groups, self.centers)])

    @classmethod
    def fit(cls, vectors, groups):
        return cls(vectors, groups)

    def search(self, query, initial_master_order, limit, record_trace=True,
               prediction="max_both", reuse_centroid=False):
        if prediction not in {"offset", "radius_ratio", "max_both"}:
            raise ValueError("Unknown empirical prediction")
        if not 1 <= limit <= len(self.vectors):
            raise ValueError("Invalid evidence count")
        order = [int(i) for i in initial_master_order]
        if sorted(order) != list(range(len(self.groups))):
            raise ValueError("Invalid master permutation")
        q = _unit_query(query, self.vectors.shape[1])
        means = self.centers @ q
        scored, visited, skipped, tail, trace = {}, [], [], [], []
        offset, ratio = 0., 0.
        selected = []
        exact_leaf_dots = 0
        inferred_scores = {}
        refinements_this_visit = []
        rounding_margin = 64 * np.finfo(np.float64).eps * (self.vectors.shape[1] + 1)

        def resolve_near_ties():
            # Inferred scores have an accumulation error envelope. Recompute
            # ambiguous candidates instead of changing stable tie semantics.
            nonlocal exact_leaf_dots
            while True:
                ranked = sorted(scored, key=lambda i: (-scored[i], i))
                ambiguous = set()
                for left, right in zip(ranked[:limit], ranked[1:limit + 1]):
                    allowance = inferred_scores.get(left, 0.) + inferred_scores.get(right, 0.)
                    if allowance and scored[left] - scored[right] <= allowance:
                        ambiguous.update(i for i in (left, right) if i in inferred_scores)
                if len(ranked) >= limit:
                    last = ranked[limit - 1]
                    lower_cutoff = scored[last] - inferred_scores.get(last, 0.)
                    selected_set = set(ranked[:limit])
                    ambiguous.update(i for i, error in inferred_scores.items()
                                     if i not in selected_set and scored[i] + error >= lower_cutoff)
                if not ambiguous:
                    break
                for i in ambiguous:
                    scored[i] = float(np.einsum("ij,j->i", self.vectors[i:i + 1], q,
                                               optimize=False)[0])
                    exact_leaf_dots += 1
                    inferred_scores.pop(i)
                    refinements_this_visit.append({
                        "leaf_index": i, "score": scored[i],
                        "coordinate_products": self.vectors.shape[1],
                    })
            return sorted(scored, key=lambda i: (-scored[i], i))[:limit]

        for pos, mi in enumerate(order):
            cutoff = scored[selected[-1]] if len(selected) == limit else None
            raw = means + offset
            scaled = means + ratio * self.radii
            predicted = raw if prediction == "offset" else scaled
            if prediction == "max_both":
                predicted = np.maximum(raw, scaled)
            remaining = order[pos:]
            max_remaining = float(np.max(predicted[remaining]))
            entry = {
                "position": pos, "master_index": mi, "cutoff_before": cutoff,
                "observed_offset": offset, "observed_radius_ratio": ratio,
                "predicted_max": float(predicted[mi]),
                "max_remaining_predicted": max_remaining,
                "is_certified_bound": False,
            }
            if cutoff is not None and max_remaining < cutoff:
                tail = remaining
                if record_trace:
                    trace.append({**entry, "action": "empirical_stop"})
                break
            if cutoff is not None and predicted[mi] < cutoff:
                skipped.append(mi)
                if record_trace:
                    trace.append({**entry, "action": "empirical_skip"})
                continue
            members = self.groups[mi]
            refinements_this_visit = []
            inferred_index = None
            dots_this_visit = len(members)
            if reuse_centroid:
                values = np.empty(len(members))
                values[:-1] = np.einsum("ij,j->i", self.vectors[members[:-1]], q,
                                        optimize=False)
                values[-1] = len(members) * means[mi] - values[:-1].sum()
                inferred_index = members[-1]
                inferred_scores[inferred_index] = len(members) * rounding_margin
                dots_this_visit -= 1
            else:
                values = self.vectors[members] @ q
            exact_leaf_dots += dots_this_visit
            scored.update((i, float(value)) for i, value in zip(members, values))
            visited.append(mi)
            selected = resolve_near_ties() if reuse_centroid else sorted(scored, key=lambda i: (-scored[i], i))[:limit]
            # Refinement can change a prior member's score by an ulp. Its
            # cluster's observed spread must use that same refined score.
            update_groups = visited if refinements_this_visit else [mi]
            for observed_mi in update_groups:
                deviation = max(scored[i] for i in self.groups[observed_mi]) - means[observed_mi]
                offset = max(offset, float(deviation))
                if self.radii[observed_mi] > 0:
                    ratio = max(ratio, float(deviation / self.radii[observed_mi]))
            if record_trace:
                trace.append({**entry, "action": "visit", "leaf_indices": members,
                              "scores": values.tolist(), "observed_max": float(values.max()),
                              "direct_leaf_dots": dots_this_visit, "centroid_inferred_leaf": inferred_index,
                              "tie_refinements": refinements_this_visit,
                              "cutoff_after": scored[selected[-1]] if len(selected) == limit else None})
        assert len(selected) == limit
        assert sorted(visited + skipped + tail) == list(range(len(self.groups)))
        return {
            "selected_indices": selected, "selected_scores": [scored[i] for i in selected],
            "scored_leaves": len(scored), "scored_leaf_indices": list(scored),
            "visited_masters": len(visited), "visited_master_indices": visited,
            "skipped_master_indices": skipped, "tail_master_indices": tail,
            "bound_dot_count": len(self.groups), "exact_leaf_dot_count": exact_leaf_dots,
            "centroid_inferred_score_count": len(inferred_scores),
            "scalar_coordinate_products": (len(self.groups) + exact_leaf_dots) * self.vectors.shape[1],
            "prediction": prediction, "exact_guarantee": False, "trace": trace,
            "reuse_centroid": reuse_centroid,
        }
