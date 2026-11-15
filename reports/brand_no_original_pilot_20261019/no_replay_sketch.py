"""Rank-two streaming memory with no historical-image replay capability.

Every arrival supplies one transient encoder vector. Persistent visual state is
one full-dimensional mean, up to two directions, and small member coordinates.
Splits only recenter those coordinates; they cannot recover discarded content.

Means remain exact through append/whole-cluster merge until a lossy cluster is
split. Such child means are explicitly estimates. Their weighted sum remains
the parent mean, and the global first moment therefore remains exact. Error
bounds below are conservative mathematical bounds, subject to floating-point
roundoff; they are not measured errors or admission thresholds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class NoReplaySketch:
    """Low-rank coordinates, honest centroid provenance, and error bounds."""

    def __init__(self, max_rank: int = 2) -> None:
        if not isinstance(max_rank, int) or max_rank < 1:
            raise ValueError("max_rank must be a positive integer")
        self.max_rank = max_rank
        self.member_ids: list[Any] = []
        self.mean = np.empty(0, dtype=np.float64)
        self.basis = np.empty((0, 0), dtype=np.float64)
        self.coordinates = np.empty((0, 0), dtype=np.float64)
        self.singular_values = np.empty(0, dtype=np.float64)
        self.mean_is_exact = True
        self.representation_is_exact = True
        self.mean_error_l2_upper_bound = 0.0
        self.reconstruction_frobenius_error_upper_bound = 0.0
        self.accumulated_truncation_energy = 0.0
        self.svd_calls = 0

    @property
    def n(self) -> int:
        return len(self.member_ids)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def dimension(self) -> int:
        return self.mean.size

    @property
    def coords(self) -> np.ndarray:
        return self.coordinates

    @staticmethod
    def _svd(rows: np.ndarray, max_rank: int):
        """The tolerance removes numerical zero only; it is not semantic."""
        if not min(rows.shape):
            return np.empty((rows.shape[1], 0)), np.empty(0), 0.0, False
        _, values, vh = np.linalg.svd(rows, full_matrices=False)
        tolerance = np.finfo(np.float64).eps * max(rows.shape) * values[0]
        nonzero_rank = int(np.count_nonzero(values > tolerance))
        rank = min(max_rank, nonzero_rank)
        basis = vh[:rank].T.copy()
        for column in range(rank):
            pivot = np.argmax(np.abs(basis[:, column]))
            if basis[pivot, column] < 0:
                basis[:, column] *= -1
        discarded = float(np.sum(values[rank:] ** 2))
        return basis, values[:rank].copy(), discarded, nonzero_rank > rank

    def append(self, member_id: Any, embedding: np.ndarray) -> dict[str, Any]:
        if member_id in self.member_ids:
            raise ValueError(f"duplicate member: {member_id}")
        z = np.asarray(embedding, dtype=np.float64)
        if z.ndim != 1 or not z.size or not np.all(np.isfinite(z)):
            raise ValueError("embedding must be a finite nonempty vector")
        if self.n and z.size != self.dimension:
            raise ValueError("embedding dimension changed")
        if not self.n:
            self.mean = z.copy()
            self.basis = np.empty((z.size, 0))
            self.coordinates = np.empty((1, 0))
            self.member_ids.append(member_id)
            return {"discarded_energy": 0.0, "svd_shape": None}

        count = self.n
        old_basis = self.basis
        delta = z - self.mean
        shift = delta / (count + 1)
        mean = self.mean + shift
        rows = np.vstack((self.singular_values[:, None] * old_basis.T,
                          np.sqrt(count / (count + 1)) * delta))
        basis, values, discarded, lossy = self._svd(rows, self.max_rank)
        old_coords = self.coordinates @ (old_basis.T @ basis) - shift @ basis
        self.coordinates = np.vstack((old_coords, (z - mean) @ basis))
        self.mean = mean
        self.basis = basis
        self.singular_values = values
        self.member_ids.append(member_id)
        # Old mean bias is diluted by the one exact arrival. The old errors and
        # new truncation need not be orthogonal after an approximate split.
        self.mean_error_l2_upper_bound *= count / (count + 1)
        self.reconstruction_frobenius_error_upper_bound += np.sqrt(discarded)
        self.accumulated_truncation_energy += discarded
        self.representation_is_exact = self.representation_is_exact and not lossy
        self.svd_calls += 1
        return {"discarded_energy": discarded, "svd_shape": list(rows.shape)}

    def clone(self) -> NoReplaySketch:
        other = type(self)(self.max_rank)
        other.member_ids = self.member_ids.copy()
        for name in ("mean", "basis", "coordinates", "singular_values"):
            setattr(other, name, getattr(self, name).copy())
        for name in ("mean_is_exact", "representation_is_exact",
                     "mean_error_l2_upper_bound",
                     "reconstruction_frobenius_error_upper_bound",
                     "accumulated_truncation_energy", "svd_calls"):
            setattr(other, name, getattr(self, name))
        return other

    def merged(self, other: NoReplaySketch) -> NoReplaySketch:
        if self.max_rank != other.max_rank:
            raise ValueError("inconsistent maximum ranks")
        if not self.n:
            return other.clone()
        if not other.n:
            return self.clone()
        if self.dimension != other.dimension:
            raise ValueError("embedding dimensions differ")
        if set(self.member_ids).intersection(other.member_ids):
            raise ValueError("overlapping members")

        result = type(self)(self.max_rank)
        total = self.n + other.n
        result.mean = self.mean + other.n / total * (other.mean - self.mean)
        rows = np.vstack((self.singular_values[:, None] * self.basis.T,
                          other.singular_values[:, None] * other.basis.T,
                          np.sqrt(self.n * other.n / total) * (self.mean - other.mean)))
        result.basis, result.singular_values, discarded, lossy = self._svd(rows, self.max_rank)
        result.coordinates = np.vstack((
            self.coordinates @ (self.basis.T @ result.basis)
            + (self.mean - result.mean) @ result.basis,
            other.coordinates @ (other.basis.T @ result.basis)
            + (other.mean - result.mean) @ result.basis,
        ))
        result.member_ids = self.member_ids + other.member_ids
        result.mean_is_exact = self.mean_is_exact and other.mean_is_exact
        result.representation_is_exact = (
            self.representation_is_exact and other.representation_is_exact and not lossy
        )
        result.mean_error_l2_upper_bound = (
            self.n * self.mean_error_l2_upper_bound
            + other.n * other.mean_error_l2_upper_bound
        ) / total
        result.reconstruction_frobenius_error_upper_bound = float(np.hypot(
            self.reconstruction_frobenius_error_upper_bound,
            other.reconstruction_frobenius_error_upper_bound,
        ) + np.sqrt(discarded))
        result.accumulated_truncation_energy = (
            self.accumulated_truncation_energy + other.accumulated_truncation_energy
            + discarded
        )
        result.svd_calls = self.svd_calls + other.svd_calls + 1
        return result

    def split(self, groups: Sequence[Sequence[Any]]) -> list[NoReplaySketch]:
        """Partition represented vectors using only m x r coordinate SVDs.

        Each represented vector is unchanged (up to roundoff). An individual
        child can inherit all the parent's unknown error, so bounds are copied
        conservatively and are not additive error measurements.
        """
        groups = [list(group) for group in groups]
        flattened = [member for group in groups for member in group]
        if not groups or any(not group for group in groups):
            raise ValueError("partition must contain nonempty groups")
        if len(flattened) != len(set(flattened)) or set(flattened) != set(self.member_ids):
            raise ValueError("groups must partition the members exactly once")
        if len(groups) == 1 and groups[0] == self.member_ids:
            return [self.clone()]

        positions = {member: row for row, member in enumerate(self.member_ids)}
        children = []
        for members in groups:
            child = type(self)(self.max_rank)
            child.member_ids = members
            coordinates = self.coordinates[[positions[member] for member in members]]
            coordinate_mean = coordinates.mean(axis=0)
            child.mean = self.mean + self.basis @ coordinate_mean
            centered = coordinates - coordinate_mean
            rotation, values, discarded, lossy = self._svd(centered, self.max_rank)
            # r <= max_rank, so this SVD can only remove numerical zero.
            if lossy:
                raise AssertionError("coordinate-only split unexpectedly lost a dimension")
            child.basis = self.basis @ rotation
            child.coordinates = centered @ rotation
            child.singular_values = values
            child.mean_is_exact = self.representation_is_exact
            child.representation_is_exact = self.representation_is_exact
            parent_error = self.reconstruction_frobenius_error_upper_bound
            child.reconstruction_frobenius_error_upper_bound = float(parent_error + np.sqrt(discarded))
            if child.mean_is_exact:
                child.mean_error_l2_upper_bound = 0.0
            elif self.mean_is_exact:
                # Sum of parent row errors is zero. Subset and complement
                # Cauchy-Schwarz bounds combine to this tighter expression.
                child.mean_error_l2_upper_bound = float(
                    parent_error * np.sqrt((self.n - child.n) / (child.n * self.n))
                )
            else:
                child.mean_error_l2_upper_bound = float(parent_error / np.sqrt(child.n))
            # This is operation history, never a true reconstruction SSE. A
            # split cannot attribute parent's discarded energy to each child.
            child.accumulated_truncation_energy = 0.0
            child.svd_calls = 1 if self.rank else 0
            children.append(child)
        return children

    def array_state(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in
                ("mean", "basis", "coordinates", "singular_values")}

    def summary(self) -> dict[str, Any]:
        arrays = self.array_state()
        return {
            "n": self.n, "dimension": self.dimension, "rank": self.rank,
            "max_rank": self.max_rank,
            "mean_is_exact": self.mean_is_exact,
            "mean_semantics": "exact_average_up_to_roundoff" if self.mean_is_exact
                              else "estimated_average_after_lossy_partition",
            "representation_is_exact_up_to_roundoff": self.representation_is_exact,
            "mean_error_l2_upper_bound": float(self.mean_error_l2_upper_bound),
            "reconstruction_frobenius_error_upper_bound": float(self.reconstruction_frobenius_error_upper_bound),
            "represented_scatter_energy": float(np.sum(self.singular_values ** 2)),
            "accumulated_truncation_energy_since_last_split": float(self.accumulated_truncation_energy),
            "truncation_energy_is_true_reconstruction_error": False,
            "svd_calls": self.svd_calls,
            "array_float_count": sum(array.size for array in arrays.values()),
            "array_bytes": sum(array.nbytes for array in arrays.values()),
        }


def run_math_audit() -> dict[str, Any]:
    """Synthetic oracle tests; real image vectors are never loaded here."""
    rng = np.random.default_rng(90210)
    truth = rng.normal(size=(64, 32))
    truth /= np.linalg.norm(truth, axis=1, keepdims=True)
    checks = []
    records = []

    def record(name, condition, **details):
        checks.append({"name": name, "passed": bool(condition), **details})

    def inspect(sketch, name):
        exact = truth[sketch.member_ids]
        # Only synthetic tests materialize full historical rows.
        represented = sketch.mean + sketch.coordinates @ sketch.basis.T
        mean_error = float(np.linalg.norm(sketch.mean - exact.mean(axis=0)))
        full_error = float(np.linalg.norm(represented - exact))
        bound_mean = sketch.mean_error_l2_upper_bound
        bound_full = sketch.reconstruction_frobenius_error_upper_bound
        record(name + "_mean_bound", mean_error <= bound_mean + 1e-11)
        record(name + "_reconstruction_bound", full_error <= bound_full + 1e-11)
        record(name + "_exact_flag", not sketch.mean_is_exact or mean_error < 1e-11)
        record(name + "_coordinate_center", np.linalg.norm(sketch.coordinates.sum(axis=0)) < 1e-10)
        record(name + "_coordinate_spectrum", np.allclose(
            sketch.coordinates.T @ sketch.coordinates,
            np.diag(sketch.singular_values ** 2), atol=1e-10))
        records.append({"name": name, "n": sketch.n, "rank": sketch.rank,
                        "mean_is_exact": sketch.mean_is_exact,
                        "actual_mean_l2_error": mean_error,
                        "mean_error_l2_upper_bound": float(bound_mean),
                        "actual_reconstruction_frobenius_error": full_error,
                        "reconstruction_frobenius_error_upper_bound": float(bound_full)})

    sketch = NoReplaySketch()
    for i in range(12):
        sketch.append(i, truth[i])
        inspect(sketch, f"append_{i + 1}")
    record("append_centroid_stays_exact", sketch.mean_is_exact)
    record("rank_two_is_lossy_for_generic_12_points", not sketch.representation_is_exact)
    before = sketch.mean + sketch.coordinates @ sketch.basis.T
    children = sketch.split([list(range(3)), list(range(3, 8)), list(range(8, 12))])
    after = np.zeros_like(before)
    for k, child in enumerate(children):
        after[child.member_ids] = child.mean + child.coordinates @ child.basis.T
        inspect(child, f"split_child_{k}")
    record("split_preserves_represented_vectors", np.allclose(before, after, atol=1e-12),
           max_change=float(np.max(np.abs(before - after))))
    record("lossy_split_marks_estimated_means", all(not child.mean_is_exact for child in children))
    global_mean = sum(child.n * child.mean for child in children) / 12
    record("split_weighted_centroid_is_exact", np.allclose(global_mean, truth[:12].mean(axis=0), atol=1e-12),
           l2_error=float(np.linalg.norm(global_mean - truth[:12].mean(axis=0))))
    previous_bound = children[0].mean_error_l2_upper_bound
    children[0].append(12, truth[12])
    inspect(children[0], "append_after_approximate_split")
    record("append_dilutes_mean_error_bound", np.isclose(children[0].mean_error_l2_upper_bound, previous_bound * 3 / 4))
    record("append_does_not_restore_exact_mean_flag", not children[0].mean_is_exact)
    joined = children[0].merged(children[1]).merged(children[2])
    inspect(joined, "merge_after_approximate_split")
    record("merging_all_siblings_restores_actual_global_mean", np.allclose(joined.mean, truth[:13].mean(axis=0), atol=1e-12))
    record("exact_flag_stays_conservative_without_lineage_proof", not joined.mean_is_exact)

    # Repeated lossy repartition, append, and merge exercises inherited bounds.
    active = [joined]
    for new_id in range(13, 40):
        largest = max(active, key=lambda item: item.n)
        if largest.n > 3:
            active.remove(largest)
            active.extend(largest.split([largest.member_ids[::2], largest.member_ids[1::2]]))
        target = new_id % len(active)
        active[target].append(new_id, truth[new_id])
        if len(active) > 3:
            active = [active[0].merged(active[1])] + active[2:]
        weighted = sum(item.n * item.mean for item in active) / (new_id + 1)
        record(f"dynamic_prefix_{new_id + 1}_global_first_moment", np.allclose(weighted, truth[:new_id + 1].mean(axis=0), atol=1e-11))
        for k, item in enumerate(active):
            inspect(item, f"dynamic_prefix_{new_id + 1}_cluster_{k}")

    tiny = NoReplaySketch()
    for i in range(3):
        tiny.append(i, truth[i])
    tiny_children = tiny.split([[0], [1, 2]])
    record("lossless_small_split_keeps_exact_flags", all(item.mean_is_exact for item in tiny_children))
    for k, item in enumerate(tiny_children):
        inspect(item, f"lossless_small_child_{k}")
    empty = NoReplaySketch()
    record("empty_merge_clones", np.array_equal(empty.merged(tiny).mean, tiny.mean))
    singleton = NoReplaySketch()
    singleton.append(40, truth[40])
    record("singleton_split_rank_zero", singleton.split([[40]])[0].rank == 0)
    duplicate = NoReplaySketch()
    for member in range(3):
        duplicate.append(member, truth[0])
    duplicate_parts = duplicate.split([[0], [1, 2]])
    duplicate_merged = duplicate_parts[0].merged(duplicate_parts[1])
    record("identical_vectors_have_no_residual_directions", duplicate_merged.rank == 0)
    record("identical_vectors_keep_exact_mean", duplicate_merged.mean_is_exact
           and np.allclose(duplicate_merged.mean, truth[0], atol=1e-12))
    record("split_clone_does_not_alias_source", not np.shares_memory(
        singleton.split([[40]])[0].mean, singleton.mean))
    for invalid in ([[0], [0, 1, 2]], [[0], [1]], [[], [0, 1, 2]]):
        caught = False
        try:
            tiny.split(invalid)
        except ValueError:
            caught = True
        record("invalid_partition_rejected_" + str(invalid), caught)
    import inspect as inspect_module
    api = " ".join(str(inspect_module.signature(method)) for method in
                   (NoReplaySketch.append, NoReplaySketch.merged, NoReplaySketch.split))
    record("no_encoder_or_replay_callback_in_api", "encode" not in api and "callback" not in api)
    record("no_persistent_historical_full_vector_array", all(
        array.ndim != 2 or array.shape != (item.n, item.dimension)
        for item in active for array in item.array_state().values()))
    return {"passed": all(check["passed"] for check in checks), "check_count": len(checks),
            "checks": checks, "synthetic_error_measurements": records,
            "scope": "Synthetic normalized vectors only; no images, encoder, file paths, or Brand oracle input.",
            "bound_semantics": "Conservative mathematical bounds plus numerical test tolerance; no claim that accumulated truncation energy equals true reconstruction error after splitting."}


if __name__ == "__main__":
    import json
    from pathlib import Path
    result = run_math_audit()
    destination = Path(__file__).with_name("math_audit.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "check_count": result["check_count"], "path": str(destination)}))
    raise SystemExit(0 if result["passed"] else 1)
