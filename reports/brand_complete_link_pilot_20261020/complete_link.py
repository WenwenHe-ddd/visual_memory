"""Prefix-only complete linkage from rank-two local memory.

No encoder, image path, old-image callback, original-vector cache or evaluation
data is available to this class. Pair distances and the merge tree are transient.
This full-prefix reference costs quadratic working memory, not a scalable index.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STRICT = HERE.parent / "brand_no_original_pilot_20261019"
sys.path.insert(0, str(STRICT))
from run_no_original import NoOriginalGraph, NoReplaySketch, write_json
from run_streaming_pilot import otsu_cut


def represented_distances(sketches, member_ids):
    """Pair cosines factored through master bases; never build an n x d array."""
    position = {member: row for row, member in enumerate(member_ids)}
    n = len(member_ids)
    output = np.empty((n, n))
    norms = []
    for s in sketches:
        norm2 = float(s.mean @ s.mean) + 2 * s.coordinates @ (s.basis.T @ s.mean)
        norm2 += np.sum(s.coordinates**2, axis=1)
        norms.append(np.sqrt(np.maximum(norm2, 0)))
    for ai, a in enumerate(sketches):
        arows = [position[i] for i in a.member_ids]
        for bi in range(ai, len(sketches)):
            b = sketches[bi]
            brows = [position[i] for i in b.member_ids]
            dot = float(a.mean @ b.mean)
            dot = dot + (a.coordinates @ (a.basis.T @ b.mean))[:, None]
            dot = dot + (b.coordinates @ (b.basis.T @ a.mean))[None, :]
            dot = dot + a.coordinates @ (a.basis.T @ b.basis) @ b.coordinates.T
            denominator = norms[ai][:, None] * norms[bi][None, :]
            cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
            # Numerical zero only, identical to the previous strict pilot.
            cosine[np.abs(1 - cosine) <= np.finfo(float).eps * a.dimension] = 1
            block = 1 - np.clip(cosine, -1, 1)
            output[np.ix_(arows, brows)] = block
            output[np.ix_(brows, arows)] = block.T
    output = (output + output.T) / 2
    np.fill_diagonal(output, 0)
    return output


def partition(distances, member_ids):
    """Otsu on merge heights, not on all pair distances; no chosen K or cutoff."""
    n = len(member_ids)
    if n == 1:
        return [list(member_ids)], {"cut": None, "cut_reason": "first_image"}
    tree = linkage(squareform(distances, checks=True), method="complete")
    heights = tree[:, 2]
    cut = otsu_cut(heights)
    if n == 2 and heights[0] != 0:
        # One nonzero height provides no empirical separation; defer the merge.
        return [[i] for i in member_ids], {"cut": None, "cut_reason": "insufficient_merge_heights"}
    if cut is None:
        # Constant heights cannot identify two classes; same convention as the
        # previous Otsu prototype. This is an assumption, not learned evidence.
        applied = float(heights[-1])
        reason = "constant_merge_heights_keep_all"
    else:
        applied = cut
        reason = "otsu_between_class_variance"
    labels = fcluster(tree, t=applied, criterion="distance")
    groups = defaultdict(list)
    for member, label in zip(member_ids, labels):
        groups[int(label)].append(member)
    return sorted(groups.values(), key=min), {"cut": cut, "applied_cut": applied, "cut_reason": reason}


def max_diameter(distances, groups, member_ids):
    positions = {member: row for row, member in enumerate(member_ids)}
    return max(float(distances[np.ix_([positions[i] for i in group],
                                     [positions[i] for i in group])].max()) for group in groups)


class CompleteLinkGraph(NoOriginalGraph):
    def __init__(self):
        super().__init__()
        self.method = "no_original_complete_link_otsu_prefix"

    def add(self, image_id, z):
        if image_id in self.seen:
            raise ValueError("each image may arrive only once")
        z = np.asarray(z, dtype=float)
        if z.ndim != 1 or not np.isfinite(z).all() or not np.isclose(z @ z, 1):
            raise ValueError("expected normalized embedding")
        started = time.perf_counter()
        singleton = NoReplaySketch(max_rank=2)
        singleton.append(image_id, z)
        seen = self.seen + [image_id]
        distances = represented_distances(self.sketches + [singleton], seen)
        groups, decision = partition(distances, seen)
        before_diameter = max_diameter(distances, groups, seen)
        if "applied_cut" in decision:
            assert before_diameter <= decision["applied_cut"] + 1e-12
        destination = {i: g for g, group in enumerate(groups) for i in group}
        pieces = defaultdict(list)
        splits = 0
        for sketch in self.sketches:
            parts = defaultdict(list)
            for i in sketch.member_ids:
                parts[destination[i]].append(i)
            if len(parts) == 1:
                pieces[next(iter(parts))].append(sketch)
            else:
                splits += 1
                keys = sorted(parts)
                for k, child in zip(keys, sketch.split([parts[k] for k in keys])):
                    pieces[k].append(child)
        pieces[destination[image_id]].append(singleton)
        updated = []
        merges = 0
        for g in range(len(groups)):
            combined = pieces[g][0]
            for piece in pieces[g][1:]:
                combined = combined.merged(piece)
                merges += 1
            updated.append(combined)
        self.seen = seen
        self.sketches = updated
        elapsed = time.perf_counter() - started
        # Diagnostics use only compressed memory, and cannot alter admission.
        audit_started = time.perf_counter()
        after = represented_distances(updated, seen)
        after_diameter = max_diameter(after, groups, seen)
        applied = decision.get("applied_cut")
        violations = sum(
            float(after[np.ix_(group, group)].max()) > applied + 1e-12
            for group in [[seen.index(i) for i in g] for g in groups]
        ) if applied is not None else 0
        self.log.append({
            "prefix_count": len(seen), "arriving_id": image_id,
            "cluster_count": len(groups), "memberships": groups,
            "cluster_sizes": sorted(map(len, groups), reverse=True),
            "singleton_count": sum(len(g) == 1 for g in groups),
            **decision, "splits": splits, "merges": merges,
            "admission_seconds": elapsed,
            "compressed_geometry_diagnostic_seconds": time.perf_counter() - audit_started,
            "maximum_cluster_diameter_at_decision": before_diameter,
            "maximum_cluster_diameter_after_compression": after_diameter,
            "clusters_exceeding_decision_cut_after_compression": violations,
            "approximate_mean_master_count": sum(not s.mean_is_exact for s in updated),
            "sketch_array_bytes": sum(s.summary()["array_bytes"] for s in updated),
            "transient_pair_distance_matrix_bytes": int(distances.nbytes),
            "edges": [],
        })
        return self.log[-1]

    def save(self, directory, nodes):
        result = super().save(directory, nodes)
        result["distance_semantics"] = (
            "Each arrival recomputes complete linkage from current rank-two "
            "reconstructions plus the arriving exact vector. Otsu cuts merge "
            "heights. Pair distances and the merge tree are discarded. "
            "The diameter criterion holds before the subsequent rank-two update; "
            "post-update drift is separately audited."
        )
        result["partition_policy"] = {
            "linkage": "complete", "cut": "exact_otsu_on_prefix_merge_heights",
            "target_cluster_count": None, "hand_set_similarity_cutoff": None,
            "two_nonidentical_inputs": "defer_merge",
            "constant_merge_heights": "keep_all",
            "update_scope": "entire_seen_prefix",
        }
        write_json(Path(directory) / "graph.json", result)
        return result
