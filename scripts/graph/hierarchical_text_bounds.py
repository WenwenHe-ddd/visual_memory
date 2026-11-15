"""Exact cosine search through a rebuildable tree of text-vector spherical bands.

Each node uses an actual member as its angular reference. Computing that pivot's
dot product therefore also gives an exact leaf score, which enters Top-L at
once.  The pivot is removed before its children are built.  No member dot
product is repeated, and no extra full-dimensional centroid dots are hidden.
Bounds concern only the still-unscored descendants: their minimum AND maximum
cosines to the pivot define a spherical band, excluding the already known pivot
when the geometry permits. No empirical stopping or similarity threshold is used.

Construction uses only text vectors and the supplied master membership:
choose the member nearest the mean as pivot, then bisect the remaining members
at the median projection along a farthest-pair direction.  There is no manually
chosen similarity threshold.  Text vectors are retained in this external index;
no per-image visual vectors or original images are involved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
from numbers import Integral
from typing import Any, Sequence

import numpy as np

from scripts.graph.text_similarity_bounds import _margin, _partition, _unit_query, _unit_rows


@dataclass(frozen=True, slots=True)
class TextPivotNode:
    node_id: int
    master_index: int
    pivot_index: int
    members: tuple[int, ...]
    children: tuple[int, ...]
    min_cosine: float
    max_cosine: float
    safe_angular_radius: float
    safe_min_angular_distance: float


@dataclass(frozen=True, slots=True)
class HierarchicalTextBoundIndex:
    """Text-only pivot forest with one node and one stored vector per member."""

    vectors: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    nodes: tuple[TextPivotNode, ...]
    roots: tuple[int, ...]

    @classmethod
    def fit(cls, vectors: Any, groups: Sequence[Sequence[int]]) -> "HierarchicalTextBoundIndex":
        text = _unit_rows(vectors)
        partition = _partition(groups, len(text))
        margin = _margin(text.shape[1])
        nodes: list[TextPivotNode | None] = []

        def build(indices: list[int], master: int) -> int:
            # Original leaf index resolves all construction ties.
            indices = sorted(indices)
            block = text[indices]
            mean_similarities = block @ block.mean(axis=0)
            pivot = indices[int(np.argmax(mean_similarities))]
            remaining = [i for i in indices if i != pivot]
            descendant_cosines = text[remaining] @ text[pivot] if remaining else np.array([1.])
            minimum = float(np.clip(np.min(descendant_cosines), -1.0, 1.0))
            maximum = float(np.clip(np.max(descendant_cosines), -1.0, 1.0))
            radius = min(math.pi, float(np.nextafter(
                math.acos(float(np.clip(minimum - margin, -1.0, 1.0))), np.inf)) + margin)
            inner_radius = max(0.0, float(np.nextafter(
                math.acos(float(np.clip(maximum + margin, -1.0, 1.0))), -np.inf)) - margin)
            node_id = len(nodes)
            nodes.append(None)
            children = []
            if len(remaining) == 1:
                children.append(build(remaining, master))
            elif remaining:
                data = text[remaining]
                pair_cosines = data @ data.T
                # Upper triangle permits a deterministic off-diagonal pair,
                # including duplicate or antipodal member vectors.
                first, second = min(
                    ((i, j) for i in range(len(remaining)) for j in range(i + 1, len(remaining))),
                    key=lambda pair: (pair_cosines[pair], remaining[pair[0]], remaining[pair[1]]),
                )
                axis = data[first] - data[second]
                projection = data @ axis
                projected_order = sorted(range(len(remaining)),
                                         key=lambda i: (projection[i], remaining[i]))
                ordered = [remaining[i] for i in projected_order]
                midpoint = len(ordered) // 2
                children.extend((build(ordered[:midpoint], master),
                                 build(ordered[midpoint:], master)))
            nodes[node_id] = TextPivotNode(
                node_id=node_id, master_index=master, pivot_index=pivot,
                members=tuple(indices), children=tuple(children),
                min_cosine=minimum, max_cosine=maximum,
                safe_angular_radius=radius, safe_min_angular_distance=inner_radius,
            )
            return node_id

        roots = tuple(build(group, master) for master, group in enumerate(partition))
        assert len(nodes) == len(text)
        assert sorted(node.pivot_index for node in nodes) == list(range(len(text)))
        text.setflags(write=False)
        return cls(text, tuple(tuple(group) for group in partition), tuple(nodes), roots)

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]

    @property
    def n_members(self) -> int:
        return len(self.vectors)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def numerical_margin(self) -> float:
        return _margin(self.dimension)

    def _cap_upper(self, node: TextPivotNode, pivot_score: float) -> float:
        margin = self.numerical_margin
        b_upper = float(np.clip(pivot_score + margin, -1.0, 1.0))
        b_lower = float(np.clip(pivot_score - margin, -1.0, 1.0))
        theta_lower = max(0.0, float(np.nextafter(math.acos(b_upper), -np.inf)) - margin)
        theta_upper = min(math.pi, float(np.nextafter(math.acos(b_lower), np.inf)) + margin)
        gap_lower = max(0.0, theta_lower - node.safe_angular_radius,
                        node.safe_min_angular_distance - theta_upper)
        # Keep the tiny positive roundoff envelope above one: an IEEE dot
        # product of normalized identical vectors may itself be >1 by an ulp.
        return min(1.0 + margin, float(np.nextafter(math.cos(gap_lower), np.inf)) + margin)

    def search(
        self, query: Any, initial_master_order: Sequence[int], limit: int,
        policy: str = "clip", record_trace: bool = True,
    ) -> dict[str, Any]:
        """Return exact stable Top-L, with mutually exclusive dot-cost counters.

        ``clip`` considers master roots in the supplied order and searches each
        subtree best-bound-first. ``best_bound`` first scores all root pivots,
        then explores a global bound queue; CLIP order resolves bound ties.
        Query normalization is common to a full scan and is excluded from the
        reported retrieval coordinate products. Heap and scalar cap operations
        remain real runtime work and are reported separately.
        """
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, Integral) or not 1 <= limit <= self.n_members:
            raise ValueError("limit must be an integer between one and the member count")
        if policy not in {"clip", "best_bound"}:
            raise ValueError("policy must be 'clip' or 'best_bound'")
        try:
            master_order = list(initial_master_order)
        except TypeError as exc:
            raise ValueError("initial_master_order must be a complete permutation") from exc
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in master_order):
            raise ValueError("master-order indices must be integers")
        master_order = [int(i) for i in master_order]
        if sorted(master_order) != list(range(self.n_groups)):
            raise ValueError("initial_master_order must contain every master exactly once")
        q = _unit_query(query, self.dimension)
        priority = {master: i for i, master in enumerate(master_order)}
        scores: dict[int, float] = {}
        top: list[tuple[float, int]] = []
        trace: list[dict[str, Any]] = []
        visited, expanded_masters = [], set()
        bound_dots = leaf_dots = bound_evaluations = heap_pushes = heap_pops = 0
        pruned_nodes: list[int] = []

        def cutoff() -> float | None:
            return top[0][0] if len(top) == limit else None

        def score_and_bound(node_id: int, frontier: list[tuple[float, int, int]]) -> None:
            nonlocal bound_dots, leaf_dots, bound_evaluations, heap_pushes
            node = self.nodes[node_id]
            leaf = node.pivot_index
            # Pivots are removed before splitting, so this invariant is also
            # the accounting proof that no exact member dot is repeated.
            if leaf in scores:
                raise RuntimeError("tree attempted to score a pivot twice")
            before = cutoff()
            value = float(self.vectors[leaf] @ q)
            scores[leaf] = value
            if node.master_index not in visited:
                visited.append(node.master_index)
            quality = (value, -leaf)
            if len(top) < limit:
                heapq.heappush(top, quality)
            elif quality > top[0]:
                heapq.heapreplace(top, quality)
            if node.children:
                bound_dots += 1
                purpose = "bound_pivot"
            else:
                leaf_dots += 1
                purpose = "terminal_leaf"
            if record_trace:
                trace.append({
                    "action": "score", "node_id": node_id, "master_index": node.master_index,
                    "leaf_index": leaf, "purpose": purpose, "score": value,
                    "threshold_before": before, "threshold_after": cutoff(),
                    "coordinate_products": self.dimension,
                })
            if not node.children:
                return
            bound_evaluations += 1
            upper = self._cap_upper(node, value)
            threshold = cutoff()
            if record_trace:
                trace.append({
                    "action": "bound", "node_id": node_id, "master_index": node.master_index,
                    "pivot_index": leaf, "upper_bound": upper,
                    "target": "unscored_descendants_excluding_pivot",
                    "threshold": threshold, "additional_coordinate_products": 0,
                })
            if threshold is not None and upper < threshold:
                pruned_nodes.append(node_id)
                if record_trace:
                    trace.append({
                        "action": "prune", "node_id": node_id, "master_index": node.master_index,
                        "upper_bound": upper, "threshold": threshold,
                        "unscored_members": [j for j in node.members if j not in scores],
                    })
                return
            heapq.heappush(frontier, (-upper, priority[node.master_index], node_id))
            heap_pushes += 1

        def drain(frontier: list[tuple[float, int, int]], scope: str) -> None:
            nonlocal heap_pops
            while frontier:
                negative_upper, _, node_id = heapq.heappop(frontier)
                heap_pops += 1
                upper, threshold = -negative_upper, cutoff()
                if threshold is not None and upper < threshold:
                    pending = [node_id, *(item[2] for item in frontier)]
                    pruned_nodes.extend(pending)
                    if record_trace:
                        trace.append({
                            "action": "stop_frontier", "scope": scope,
                            "max_remaining_upper_bound": upper, "threshold": threshold,
                            "pending_node_ids": pending,
                            "unscored_members": sorted({
                                j for pending_id in pending for j in self.nodes[pending_id].members
                                if j not in scores
                            }),
                        })
                    frontier.clear()
                    return
                node = self.nodes[node_id]
                expanded_masters.add(node.master_index)
                if record_trace:
                    trace.append({
                        "action": "expand", "node_id": node_id, "master_index": node.master_index,
                        "upper_bound": upper, "threshold": threshold,
                        "children": list(node.children),
                    })
                for child in node.children:
                    score_and_bound(child, frontier)

        if policy == "clip":
            for master in master_order:
                frontier = []
                score_and_bound(self.roots[master], frontier)
                drain(frontier, f"master_{master}")
        else:
            frontier = []
            for master in master_order:
                score_and_bound(self.roots[master], frontier)
            drain(frontier, "all_masters")
        selected = [(-negative_leaf, value) for value, negative_leaf in top]
        selected.sort(key=lambda pair: (-pair[1], pair[0]))
        assert len(selected) == limit
        assert len(scores) == bound_dots + leaf_dots <= self.n_members
        return {
            "policy": policy, "L": int(limit),
            "selected_indices": [leaf for leaf, _ in selected],
            "selected_scores": [value for _, value in selected],
            "scored_leaves": len(scores), "scored_leaf_count": len(scores),
            "scored_leaf_indices": list(scores),
            "visited_masters": visited, "visited_master_count": len(visited),
            "expanded_masters": sorted(expanded_masters),
            "bound_dot_count": bound_dots, "leaf_dot_count": leaf_dots,
            "exact_dot_count": len(scores), "additional_nonleaf_dot_count": 0,
            "scalar_coordinate_products": self.dimension * (bound_dots + leaf_dots),
            "bound_evaluations": bound_evaluations,
            "scalar_bound_acos_calls": 2 * bound_evaluations,
            "scalar_bound_cos_calls": bound_evaluations,
            "heap_pushes": heap_pushes, "heap_pops": heap_pops,
            "pruned_node_ids": pruned_nodes,
            "pruned_leaf_count": self.n_members - len(scores),
            "query_normalization_excluded_from_products": True,
            "trace": trace,
        }

    def export_state(self) -> dict[str, Any]:
        """Rebuildable text-cache state, including text vectors but no visuals."""
        return {
            "schema_version": 1, "kind": "text_member_pivot_band_forest",
            "dimension": self.dimension, "normalization": "robust_float64_l2",
            "numerical_margin": self.numerical_margin,
            "pivot_rule": "member_with_max_mean_cosine",
            "split_rule": "median_projection_on_farthest_member_pair",
            "text_vectors": self.vectors.copy(),
            "groups": [list(group) for group in self.groups],
            "root_node_ids": list(self.roots),
            "nodes": [asdict(node) for node in self.nodes],
        }
