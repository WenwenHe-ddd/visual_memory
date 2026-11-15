"""Exact text Top-L with progressively refined shared PCA coordinates.

Only reconstructible text vectors enter this index. The seed visits whole
clusters in the supplied (for example CLIP) order until L leaves are scored.
Subsequent projections and exact scores are selected without labels or a
precomputed query-by-leaf score matrix.

One PC is an indivisible refinement step, not a fitted block-size parameter.
Refinement stops when its next D + active_count coordinate multiplications
cost at least scoring all remaining active leaves (D * active_count).
All earlier costs remain included in the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np


def _unit_rows(value: Any) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("text vectors must be real")
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("text vectors must be a finite nonempty matrix")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("text vectors must have finite nonzero norms")
    return x / norms[:, None]


def _unit_query(value: Any, dimension: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("query must be real")
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (dimension,) or not np.isfinite(q).all():
        raise ValueError("query dimension or values are invalid")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("query norm must be finite and nonzero")
    return q / norm


@dataclass
class ProgressiveTextBounds:
    """Shared, mean-centered PCA index over normalized leaf text vectors."""

    vectors: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    mean: np.ndarray
    basis: np.ndarray
    coordinates: np.ndarray
    residual_norms: np.ndarray
    coordinate_norms: np.ndarray
    orthogonality_error: float
    numerical_margin: float
    fit_singular_values: np.ndarray

    @classmethod
    def fit(cls, leaf_vectors: Any, groups: Sequence[Sequence[int]]) -> "ProgressiveTextBounds":
        x = _unit_rows(leaf_vectors)
        n, dimension = x.shape
        prepared = []
        for group in groups:
            members = tuple(group)
            if not members or any(not isinstance(i, Integral) or isinstance(i, bool) for i in members):
                raise ValueError("groups must contain nonempty integer index lists")
            prepared.append(tuple(int(i) for i in members))
        flat = [i for group in prepared for i in group]
        if sorted(flat) != list(range(n)):
            raise ValueError("groups must partition all text vectors exactly once")
        mean = x.mean(axis=0)
        centered = x - mean
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        # This rank tolerance only resolves machine-precision zero singular
        # values. Discarded components remain covered by actual residual norms.
        tolerance = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        basis = right[:rank].copy()
        for row in basis:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        coordinates = centered @ basis.T
        margin = float(64 * np.finfo(np.float64).eps * (dimension + rank + 1))
        orth_error = (
            float(np.max(np.sum(np.abs(basis @ basis.T - np.eye(rank)), axis=1))) + margin
            if rank else 0.0
        )
        residual_norms = np.empty((n, rank + 1))
        residual = centered.copy()
        residual_norms[:, 0] = np.linalg.norm(residual, axis=1) + margin
        for k in range(rank):
            residual -= coordinates[:, k, None] * basis[k]
            residual_norms[:, k+1] = np.linalg.norm(residual, axis=1) + margin
        coordinate_norms = np.concatenate(
            [np.zeros((n, 1)), np.sqrt(np.cumsum(coordinates**2, axis=1))], axis=1)
        result = cls(
            vectors=x.copy(), groups=tuple(prepared), mean=mean, basis=basis,
            coordinates=coordinates, residual_norms=residual_norms,
            coordinate_norms=coordinate_norms, orthogonality_error=orth_error,
            numerical_margin=margin, fit_singular_values=singular,
        )
        for array in (
            result.vectors, result.mean, result.basis, result.coordinates,
            result.residual_norms, result.coordinate_norms, result.fit_singular_values,
        ):
            array.setflags(write=False)
        return result

    @property
    def dimension(self) -> int:
        return self.vectors.shape[1]

    @property
    def rank(self) -> int:
        return self.basis.shape[0]

    @property
    def n_leaves(self) -> int:
        return len(self.vectors)

    @property
    def numeric_bytes(self) -> int:
        return sum(array.nbytes for array in (
            self.vectors, self.mean, self.basis, self.coordinates,
            self.residual_norms, self.coordinate_norms, self.fit_singular_values,
        ))

    def _bounds(
        self, indices: np.ndarray, components: int, partial: np.ndarray,
        mean_score: float, query_projected_squared_norm: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative Cauchy bounds including nonorthogonality allowances.

        If B contains the current directions, q_r=q-B.T(Bq). Its squared
        norm is bounded by 1-||Bq||²+||BB.T-I||*||Bq||². The tiny B-residual
        cross term is bounded separately rather than assuming exact float64
        orthogonality. Actual leaf residual norms include discarded SVD modes.
        """
        s = query_projected_squared_norm
        margin = self.numerical_margin
        q_residual_upper = np.sqrt(max(0.0, 1-s+self.orthogonality_error*s+margin))
        cross = np.sqrt(max(0.0, s)) * (
            self.orthogonality_error*self.coordinate_norms[indices, components]+margin)
        width = q_residual_upper*self.residual_norms[indices, components]+cross+margin
        center = mean_score+partial[indices]
        lower = np.maximum(-1-margin, center-width)
        upper = np.minimum(1+margin, center+width)
        return lower, upper

    def search(
        self, query: Any, initial_master_order: Sequence[int], limit: int, *,
        record_trace: bool = True,
    ) -> dict:
        """Return canonical exact Top-L; every full leaf score is counted.

        Full scores use a fixed per-row reduction (einsum optimize=False), so
        tie ordering does not depend on the size of an exact-score batch.
        Equal scores are ordered by the original leaf index. A bound equal to
        the current cutoff is never pruned.
        """
        q = _unit_query(query, self.dimension)
        if not isinstance(limit, Integral) or isinstance(limit, bool) or not 1 <= limit <= self.n_leaves:
            raise ValueError("limit must be between one and the number of leaves")
        limit = int(limit)
        order = list(initial_master_order)
        if any(not isinstance(i, Integral) or isinstance(i, bool) for i in order):
            raise ValueError("master order must contain integer indices")
        order = [int(i) for i in order]
        if sorted(order) != list(range(len(self.groups))):
            raise ValueError("master order must be a complete permutation")
        n, dimension = self.vectors.shape
        scored: dict[int, float] = {}
        active = np.ones(n, dtype=bool)
        selected: list[int] = []
        partial = np.zeros(n)
        trace = []
        exact_order = []
        cost = {
            "mean_dot_scalar_products": 0,
            "query_projection_scalar_products": 0,
            "leaf_partial_scalar_products": 0,
            "exact_leaf_scalar_products": 0,
            "bound_evaluations": 0,
            "query_normalization_scalar_products_common_to_full_scan": dimension,
        }

        def cutoff():
            return scored[selected[-1]] if len(selected) == limit else None

        def score(indices):
            nonlocal selected
            indices = [int(i) for i in indices]
            if any(i in scored for i in indices):
                raise AssertionError("An exact leaf score must never be computed twice")
            values = np.einsum("ij,j->i", self.vectors[indices], q, optimize=False)
            for i, value in zip(indices, values):
                scored[i] = float(value)
                active[i] = False
                exact_order.append(i)
            cost["exact_leaf_scalar_products"] += len(indices)*dimension
            selected = sorted(scored, key=lambda i: (-scored[i], i))[:limit]

        seed_masters = []
        for master in order:
            before = cutoff()
            score(self.groups[master])
            seed_masters.append(master)
            if record_trace:
                trace.append({
                    "action": "seed_master", "master_index": master,
                    "scored_indices": list(self.groups[master]),
                    "threshold_before": before, "threshold_after": cutoff(),
                    "remaining_active": int(active.sum()),
                })
            if len(scored) >= limit:
                break
        components = 0
        query_projected_sq = 0.0
        mean_score = None
        reason = "seed_scored_all_leaves"
        if active.any():
            mean_score = float(self.mean @ q)
            cost["mean_dot_scalar_products"] = dimension
        while active.any():
            ids = np.flatnonzero(active)
            next_cost = dimension+len(ids)
            exact_remaining_cost = dimension*len(ids)
            if components == self.rank or next_cost >= exact_remaining_cost:
                reason = "all_components_used" if components == self.rank else "exact_remaining_is_no_costlier"
                before = cutoff()
                score(ids)
                if record_trace:
                    trace.append({
                        "action": "exact_remaining", "reason": reason,
                        "components": components, "scored_indices": ids.tolist(),
                        "threshold_before": before, "threshold_after": cutoff(),
                        "next_projection_scalar_products": next_cost,
                        "exact_remaining_scalar_products": exact_remaining_cost,
                    })
                break
            before = cutoff()
            component = float(self.basis[components] @ q)
            cost["query_projection_scalar_products"] += dimension
            partial[ids] += self.coordinates[ids, components]*component
            cost["leaf_partial_scalar_products"] += len(ids)
            query_projected_sq += component**2
            components += 1
            lower, upper = self._bounds(ids, components, partial, mean_score, query_projected_sq)
            cost["bound_evaluations"] += len(ids)
            # A certified lower bound above the current exact cutoff identifies
            # a candidate whose exact score can immediately improve that cutoff.
            refined = []
            for position in sorted(range(len(ids)), key=lambda j: (-lower[j], int(ids[j]))):
                i = int(ids[position])
                if lower[position] > cutoff():
                    score([i])
                    refined.append(i)
                else:
                    break
            pruned = [int(i) for i, ub in zip(ids, upper) if active[i] and ub < cutoff()]
            active[pruned] = False
            if record_trace:
                trace.append({
                    "action": "project_component", "components": components,
                    "active_before": len(ids), "active_after": int(active.sum()),
                    "threshold_before": before, "threshold_after": cutoff(),
                    "pruned_indices": pruned, "refined_exact_indices": refined,
                    "upper_max": float(upper.max()), "lower_max": float(lower.max()),
                    "query_projected_squared_norm": query_projected_sq,
                    "next_projection_scalar_products": next_cost,
                    "exact_remaining_scalar_products": exact_remaining_cost,
                })
            reason = "all_remaining_candidates_pruned_or_scored"
        assert len(selected) == limit
        leaf_owners = {leaf: mi for mi, members in enumerate(self.groups) for leaf in members}
        touched_masters = sorted({leaf_owners[i] for i in scored})
        cost["total_scalar_products"] = sum(cost[key] for key in (
            "mean_dot_scalar_products", "query_projection_scalar_products",
            "leaf_partial_scalar_products", "exact_leaf_scalar_products"))
        cost["full_scan_leaf_scalar_products"] = n*dimension
        return {
            "L": limit, "selected_indices": selected,
            "selected_scores": [scored[i] for i in selected],
            "scored_indices": exact_order, "exact_score_count": len(scored),
            "scored_leaves": len(scored),
            "visited_masters": len(touched_masters),
            "visited_master_indices": touched_masters,
            "scalar_coordinate_products": cost["total_scalar_products"],
            "bound_dot_count": None,
            "pruned_leaf_count": n-len(scored), "projected_components": components,
            "seed_master_indices": seed_masters, "touched_master_indices": touched_masters,
            "stop_reason": reason, "trace": trace, "cost": cost,
            "candidate_refinement_rule": "certified_lower_bound_above_current_exact_cutoff",
            "cost_scope": "Scalar multiplications within dot/projection operations; "
                          "bound sqrt/arithmetic, sorting, indexing and Python overhead are not included. "
                          "Query normalization is listed separately and is common to full scan.",
        }


# Explicit longer name for callers that prefer the index suffix.
ProgressiveTextBoundIndex = ProgressiveTextBounds
