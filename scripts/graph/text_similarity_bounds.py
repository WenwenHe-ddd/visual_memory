"""Rebuildable text-only cluster bounds for cosine retrieval.

Fit consumes text vectors and a partition; no query or target labels are used.
Only one aggregate row per cluster is retained.  The index is independent of
visual master state and can be rebuilt from saved dialogue/full-caption text.

For unit members x, let c=mean(x), u=c/||c|| and a=min(u.x).  A unit query q
has b=q.u.  The maximum q.x over the containing spherical cap u.x>=a is
1 if b>=a, otherwise a*b + sqrt((1-a*a)*(1-b*b)).

The implementation evaluates the equivalent angular expression and expands
its feasible cap using a dimension-scaled float64 roundoff allowance.  This
allowance is numerical conservatism, not a tunable retrieval threshold.
Bounds may be loose; they do not guarantee a reduction in visited clusters.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
_ROUNDING_FACTOR = 64.0


def _margin(dimension: int) -> float:
    return float(_ROUNDING_FACTOR * np.finfo(np.float64).eps * (dimension + 1))


def _numeric_array(value: Any, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must contain real numbers")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result.copy()


def _unit_rows(value: Any) -> np.ndarray:
    rows = _numeric_array(value, "vectors")
    if rows.ndim != 2 or min(rows.shape) == 0:
        raise ValueError("vectors must have nonempty shape (n_members, dimension)")
    scales = np.max(np.abs(rows), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("vectors must have nonzero row norms")
    # Scaling avoids overflow/underflow for otherwise valid finite inputs.
    rows /= scales
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows


def _unit_query(value: Any, dimension: int) -> np.ndarray:
    query = _numeric_array(value, "query")
    if query.ndim != 1 or query.shape != (dimension,):
        raise ValueError(f"query must have shape ({dimension},)")
    return _unit_rows(query[None, :])[0]


def _partition(groups: Sequence[Sequence[int]], count: int) -> list[list[int]]:
    try:
        raw = list(groups)
    except TypeError as exc:
        raise ValueError("groups must be a nonempty partition of member indices") from exc
    if not raw:
        raise ValueError("groups must be nonempty")
    result, seen = [], set()
    for group in raw:
        try:
            members = list(group)
        except TypeError as exc:
            raise ValueError("each group must be a nonempty index sequence") from exc
        if not members:
            raise ValueError("empty groups are not allowed")
        normalized = []
        for member in members:
            if isinstance(member, (bool, np.bool_)) or not isinstance(member, Integral):
                raise ValueError("member indices must be integers")
            index = int(member)
            if index < 0 or index >= count or index in seen:
                raise ValueError("member indices must be in range and occur exactly once")
            normalized.append(index)
            seen.add(index)
        result.append(normalized)
    if seen != set(range(count)):
        raise ValueError("groups must cover every input member exactly once")
    return result


@dataclass(frozen=True, slots=True)
class TextBoundIndex:
    """Per-cluster aggregates; no member vectors or membership lists retained.

    Inputs are normalized internally, so all methods operate on cosine
    similarity even if the caller provides positively scaled text vectors.
    Arrays in the object are owned, float64 (except counts), and read-only.
    """

    centers: np.ndarray
    directions: np.ndarray
    min_cosine: np.ndarray
    radii: np.ndarray
    counts: np.ndarray

    def __post_init__(self) -> None:
        centers = _numeric_array(self.centers, "centers")
        directions = _numeric_array(self.directions, "directions")
        minima = _numeric_array(self.min_cosine, "min_cosine")
        radii = _numeric_array(self.radii, "radii")
        raw_counts = np.asarray(self.counts)
        if centers.ndim != 2 or min(centers.shape) == 0:
            raise ValueError("centers must have nonempty shape (n_groups, dimension)")
        k, dimension = centers.shape
        if directions.shape != centers.shape or minima.shape != (k,) or radii.shape != (k,):
            raise ValueError("aggregate shapes are inconsistent")
        if raw_counts.shape != (k,) or raw_counts.dtype.kind not in "iu" or np.any(raw_counts <= 0):
            raise ValueError("counts must be positive integers, one per group")
        counts = np.array(raw_counts, dtype=np.int64, copy=True)
        if np.any(counts <= 0):
            raise ValueError("counts exceed the supported integer range")
        margin = _margin(dimension)
        norm_c = np.linalg.norm(centers, axis=1)
        norm_u = np.linalg.norm(directions, axis=1)
        fallback = norm_u == 0
        if np.any(norm_c > 1 + margin) or np.any(np.abs(norm_u[~fallback] - 1) > margin):
            raise ValueError("centers must be unit-ball means and nonzero directions must be unit length")
        if np.any((minima < -1) | (minima > 1)) or np.any((radii < 0) | (radii > 2 + margin)):
            raise ValueError("min_cosine or radius is outside its valid range")
        if np.any(norm_c[fallback] > margin) or np.any(minima[fallback] != -1):
            raise ValueError("zero-direction fallback requires a negligible center and min_cosine=-1")
        active = ~fallback
        if np.any(active & (norm_c == 0)):
            raise ValueError("a zero center must use the fallback direction")
        if active.any() and np.max(np.abs(directions[active] - centers[active] / norm_c[active, None])) > 4 * margin:
            raise ValueError("directions must align with centers")
        singleton = counts == 1
        if np.any(np.abs(norm_c[singleton] - 1) > margin) or np.any(radii[singleton] > margin):
            raise ValueError("singleton aggregates must represent a unit member")
        for name, value in (
            ("centers", centers), ("directions", directions),
            ("min_cosine", minima), ("radii", radii), ("counts", counts),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def fit(cls, vectors: Any, groups: Sequence[Sequence[int]]) -> "TextBoundIndex":
        """Fit from normalized text geometry and an exact member partition."""
        members = _unit_rows(vectors)
        partition = _partition(groups, len(members))
        dimension = members.shape[1]
        centers, directions, minima, radii, counts = [], [], [], [], []
        for indices in partition:
            group = members[indices]
            center = group.mean(axis=0)
            center_norm = float(np.linalg.norm(center))
            if center_norm <= _margin(dimension):
                direction, minimum = np.zeros(dimension), -1.0
            else:
                direction = center / center_norm
                minimum = float(np.clip(np.min(group @ direction), -1.0, 1.0))
            centers.append(center)
            directions.append(direction)
            minima.append(minimum)
            radii.append(float(np.max(np.linalg.norm(group - center, axis=1))))
            counts.append(len(indices))
        return cls(np.asarray(centers), np.asarray(directions), np.asarray(minima),
                   np.asarray(radii), np.asarray(counts, dtype=np.int64))

    @property
    def dimension(self) -> int:
        return self.centers.shape[1]

    @property
    def n_groups(self) -> int:
        return len(self.counts)

    @property
    def numerical_margin(self) -> float:
        """Fixed float64 roundoff allowance; not a data-dependent threshold."""
        return _margin(self.dimension)

    def mean_scores(self, query: Any) -> np.ndarray:
        """Exact member cosine arithmetic means, up to floating-point error."""
        return self.centers @ _unit_query(query, self.dimension)

    def upper_bounds(self, query: Any) -> np.ndarray:
        """Conservative upper bounds on every cluster's maximum leaf cosine.

        For safe Top-L stopping use max(unvisited_bounds) < current_Lth_score;
        equality must continue or be handled with an explicit stable tie rule.
        Bounds guarantee ranking consistency, not semantic answer sufficiency.
        """
        q = _unit_query(query, self.dimension)
        margin = self.numerical_margin
        b = self.directions @ q
        # Lower a expands the containing cap. Higher b brings the query closer
        # to that cap. Both perturbations make the resulting score an upper bound.
        a_lower = np.clip(self.min_cosine - margin, -1.0, 1.0)
        b_upper = np.clip(b + margin, -1.0, 1.0)
        rho_upper = np.minimum(np.pi, np.nextafter(np.arccos(a_lower), np.inf) + margin)
        theta_lower = np.maximum(0.0, np.nextafter(np.arccos(b_upper), -np.inf) - margin)
        gap_lower = np.maximum(0.0, theta_lower - rho_upper)
        upper = np.minimum(1.0, np.nextafter(np.cos(gap_lower), np.inf) + margin)
        # A singleton is a known point, so it does not need an angular cap
        # expansion around acos(1), whose condition number is poor.
        singleton = self.counts == 1
        upper[singleton] = np.minimum(1.0, np.nextafter(b[singleton], np.inf) + margin)
        upper[np.linalg.norm(self.directions, axis=1) == 0] = 1.0
        return np.clip(upper, -1.0, 1.0)

    def radius_upper_bounds(self, query: Any) -> np.ndarray:
        """Simpler diagnostic bound q.c + r, intersected with the cosine range."""
        q = _unit_query(query, self.dimension)
        return np.clip(np.nextafter(self.centers @ q + self.radii, np.inf)
                       + 2 * self.numerical_margin, -1.0, 1.0)

    def export_state(self) -> dict[str, Any]:
        """Return independent aggregate arrays suitable for a text-index sidecar.

        Callers should additionally record member/text/model provenance outside
        this numerical object and rebuild affected bounds when those inputs change.
        No query, label, member vector, or visual state is exported.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "dimension": self.dimension,
            "normalization": "robust_float64_l2",
            "numerical_margin": self.numerical_margin,
            "centers": self.centers.copy(),
            "directions": self.directions.copy(),
            "min_cosine": self.min_cosine.copy(),
            "radii": self.radii.copy(),
            "counts": self.counts.copy(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TextBoundIndex":
        """Load an aggregate-only state; provenance remains the caller's duty."""
        if not isinstance(state, Mapping) or state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported text-bound schema")
        if state.get("normalization") != "robust_float64_l2":
            raise ValueError("unsupported text-vector normalization")
        try:
            result = cls(*(state[name] for name in ("centers", "directions", "min_cosine", "radii", "counts")))
        except KeyError as exc:
            raise ValueError("incomplete text-bound state") from exc
        if state.get("dimension") != result.dimension or state.get("numerical_margin") != result.numerical_margin:
            raise ValueError("text-bound numerical metadata is inconsistent")
        return result
