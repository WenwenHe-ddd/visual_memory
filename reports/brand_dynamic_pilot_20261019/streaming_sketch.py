"""Small streaming residual sketch for the Brand pilot, not production code.

The persistent state contains one exact mean, at most ``max_rank`` directions,
and at most ``max_rank`` coordinates per member. Historical image embeddings
are never cached. Each append performs an SVD on at most (max_rank + 1) x d.

Important: after truncation, this is *streaming approximate PCA*, not batch PCA
of all original images. When the basis changes, old coordinates are projected
from their previous reconstruction; discarded directions cannot be recovered.
The exact mean is unaffected by this approximation. Splitting/reassigning old
members with exact means requires a fresh embedding callback.

An additional exact invariant is available: the centered reconstruction Y and
its accumulated error E satisfy Y.T @ E == 0 (up to roundoff). Center shifts
preserve it because the rows of E sum to zero; each truncated SVD preserves it
because discarded and retained singular components are orthogonal. Therefore
the sum of discarded squared singular values equals ||E||_F**2, even though the
missing directions and per-image errors cannot be reconstructed from that sum.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np


class ClusterSketch:
    """An exact mean and a rank-limited streaming residual representation.

    ``append(member_id, embedding)`` consumes one transient input embedding.
    ``reconstruct()`` creates temporary approximate full-dimensional vectors;
    callers must not mistake them for exact historical embeddings.
    """

    def __init__(self, max_rank: int = 2) -> None:
        if not isinstance(max_rank, int) or max_rank < 1:
            raise ValueError("max_rank must be a positive integer")
        self.max_rank = max_rank
        self.member_ids: list[str] = []
        self.mean = np.empty(0, dtype=np.float64)
        self.basis = np.empty((0, 0), dtype=np.float64)
        self.singular_values = np.empty(0, dtype=np.float64)
        self.coordinates = np.empty((0, 0), dtype=np.float64)
        self.exact_scatter_trace = 0.0
        self.cumulative_discarded_scatter_energy = 0.0
        self.reconstruction_error_frobenius_upper_bound = 0.0
        self.svd_calls = 0

    @property
    def n(self) -> int:
        return len(self.member_ids)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def coords(self) -> np.ndarray:
        """Short alias used by the pilot's candidate scorer."""
        return self.coordinates

    @property
    def scatter_trace(self) -> float:
        return self.exact_scatter_trace

    def append(self, member_id: str, embedding: np.ndarray) -> dict[str, Any]:
        """Add one real embedding and update the old projected coordinates.

        The mean update is exact up to floating point. The rank truncation is a
        modeling approximation. A machine-precision rank test below only drops
        numerical zero singular values; it is not a cluster-admission threshold.
        """
        if member_id in self.member_ids:
            raise ValueError(f"duplicate member_id: {member_id}")
        z = np.asarray(embedding, dtype=np.float64)
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
            raise ValueError("embedding must be a nonempty finite vector")
        if self.n and z.size != self.dimension:
            raise ValueError("embedding dimension changed")
        if self.n == 0:
            self.mean = z.copy()
            self.basis = np.empty((z.size, 0), dtype=np.float64)
            self.coordinates = np.empty((1, 0), dtype=np.float64)
            self.member_ids.append(member_id)
            return {"rank": 0, "discarded_scatter_energy": 0.0, "svd_shape": None}

        old_n = self.n
        old_basis = self.basis
        delta = z - self.mean
        shift = delta / (old_n + 1)
        new_mean = self.mean + shift
        # Welford's scatter update: S' = S + n/(n+1) delta delta^T.
        new_scatter_row = np.sqrt(old_n / (old_n + 1)) * delta
        small = np.vstack(
            (self.singular_values[:, None] * old_basis.T, new_scatter_row[None, :])
        )
        _, all_singular_values, vh = np.linalg.svd(small, full_matrices=False)
        self.svd_calls += 1
        numerical_tolerance = (
            np.finfo(np.float64).eps * max(small.shape) * all_singular_values[0]
        )
        nonzero_rank = int(np.count_nonzero(all_singular_values > numerical_tolerance))
        new_rank = min(self.max_rank, nonzero_rank)
        new_basis = vh[:new_rank].T.copy()
        new_singular_values = all_singular_values[:new_rank].copy()
        # Choose a deterministic sign without changing the represented subspace.
        for j in range(new_rank):
            pivot = int(np.argmax(np.abs(new_basis[:, j])))
            if new_basis[pivot, j] < 0:
                new_basis[:, j] *= -1

        # No n x d historical reconstruction is needed here. With a fixed basis
        # the center translation is exact. A changed basis reprojects the *old
        # approximation*, not unavailable original embeddings.
        old_coordinates = self.coordinates @ (old_basis.T @ new_basis)
        old_coordinates -= shift @ new_basis
        new_coordinate = (z - new_mean) @ new_basis
        self.coordinates = np.vstack((old_coordinates, new_coordinate))
        self.mean = new_mean
        self.basis = new_basis
        self.singular_values = new_singular_values
        self.member_ids.append(member_id)

        discarded = float(np.sum(all_singular_values[new_rank:] ** 2))
        self.exact_scatter_trace += float(new_scatter_row @ new_scatter_row)
        self.cumulative_discarded_scatter_energy += discarded
        # Also retain a conservative triangle-inequality bound. The sharper
        # squared-error identity is explained in the module documentation.
        self.reconstruction_error_frobenius_upper_bound += float(np.sqrt(discarded))
        return {
            "rank": self.rank,
            "discarded_scatter_energy": discarded,
            "svd_shape": list(small.shape),
        }

    def reconstruct(self, ids: Sequence[str] | None = None) -> np.ndarray:
        """Return approximate vectors in original dimensions, only on demand."""
        if ids is None:
            coordinates = self.coordinates
        else:
            positions = {member_id: i for i, member_id in enumerate(self.member_ids)}
            coordinates = self.coordinates[[positions[member_id] for member_id in ids]]
        return self.mean[None, :] + coordinates @ self.basis.T

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project a transient real vector into the current residual basis."""
        if not self.n:
            raise ValueError("cannot project into an empty sketch")
        z = np.asarray(embedding, dtype=np.float64)
        if z.shape != self.mean.shape or not np.all(np.isfinite(z)):
            raise ValueError("embedding shape or values invalid")
        return (z - self.mean) @ self.basis

    def clone(self) -> ClusterSketch:
        """Copy only compressed state for a reversible candidate evaluation."""
        other = type(self)(max_rank=self.max_rank)
        other.member_ids = self.member_ids.copy()
        for name in ("mean", "basis", "singular_values", "coordinates"):
            setattr(other, name, getattr(self, name).copy())
        for name in (
            "exact_scatter_trace",
            "cumulative_discarded_scatter_energy",
            "reconstruction_error_frobenius_upper_bound",
            "svd_calls",
        ):
            setattr(other, name, getattr(self, name))
        return other

    def merged(self, other: ClusterSketch) -> ClusterSketch:
        """Combine two masters, keeping the exact mean; leave inputs untouched.

        The scatter sketch SVD has at most (2 * max_rank + 1) rows. Historical
        coordinates are reprojected directly without materializing n x d vectors.
        """
        if self.max_rank != other.max_rank:
            raise ValueError("merged sketches must use the same maximum rank")
        if not self.n:
            return other.clone()
        if not other.n:
            return self.clone()
        if self.dimension != other.dimension:
            raise ValueError("merged embedding dimensions differ")
        if set(self.member_ids).intersection(other.member_ids):
            raise ValueError("merged sketches contain overlapping member IDs")
        combined = type(self)(max_rank=self.max_rank)
        n_total = self.n + other.n
        combined.mean = self.mean + other.n / n_total * (other.mean - self.mean)
        center_row = np.sqrt(self.n * other.n / n_total) * (self.mean - other.mean)
        small = np.vstack(
            (
                self.singular_values[:, None] * self.basis.T,
                other.singular_values[:, None] * other.basis.T,
                center_row[None, :],
            )
        )
        _, singular_values, vh = np.linalg.svd(small, full_matrices=False)
        numerical_tolerance = np.finfo(np.float64).eps * max(small.shape) * singular_values[0]
        new_rank = min(
            self.max_rank, int(np.count_nonzero(singular_values > numerical_tolerance))
        )
        combined.basis = vh[:new_rank].T.copy()
        combined.singular_values = singular_values[:new_rank].copy()
        for j in range(new_rank):
            pivot = int(np.argmax(np.abs(combined.basis[:, j])))
            if combined.basis[pivot, j] < 0:
                combined.basis[:, j] *= -1
        combined.coordinates = np.vstack(
            (
                self.coordinates @ (self.basis.T @ combined.basis)
                + (self.mean - combined.mean) @ combined.basis,
                other.coordinates @ (other.basis.T @ combined.basis)
                + (other.mean - combined.mean) @ combined.basis,
            )
        )
        combined.member_ids = self.member_ids + other.member_ids
        discarded = float(np.sum(singular_values[new_rank:] ** 2))
        combined.exact_scatter_trace = (
            self.exact_scatter_trace
            + other.exact_scatter_trace
            + float(center_row @ center_row)
        )
        combined.cumulative_discarded_scatter_energy = (
            self.cumulative_discarded_scatter_energy
            + other.cumulative_discarded_scatter_energy
            + discarded
        )
        combined.reconstruction_error_frobenius_upper_bound = float(
            np.hypot(
                self.reconstruction_error_frobenius_upper_bound,
                other.reconstruction_error_frobenius_upper_bound,
            )
            + np.sqrt(discarded)
        )
        combined.svd_calls = self.svd_calls + other.svd_calls + 1
        return combined

    def merge(self, other: ClusterSketch) -> ClusterSketch:
        """Alias for ``merged``; neither operand is mutated."""
        return self.merged(other)

    @classmethod
    def from_encoder(
        cls,
        member_ids: Iterable[str],
        encode: Callable[[str], np.ndarray],
        max_rank: int = 2,
    ) -> ClusterSketch:
        """Build/rebuild with exact means from one transient real vector at a time."""
        sketch = cls(max_rank=max_rank)
        for member_id in member_ids:
            sketch.append(member_id, encode(member_id))
        return sketch

    def partition_reencoded(
        self,
        groups: Sequence[Sequence[str]],
        encode: Callable[[str], np.ndarray],
    ) -> list[ClusterSketch]:
        """Split with a required real-image callback; never silently fake means.

        Passing ``reconstruct`` as this callback would invalidate the exact-mean
        contract. The caller must re-encode originals or provide verified exact
        source embeddings held only transiently for this operation.
        """
        flattened = [member_id for group in groups for member_id in group]
        if any(not group for group in groups):
            raise ValueError("empty partition group")
        if len(flattened) != len(set(flattened)) or set(flattened) != set(self.member_ids):
            raise ValueError("groups must partition all members exactly once")
        return [self.from_encoder(group, encode, self.max_rank) for group in groups]

    def summary(self) -> dict[str, Any]:
        retained = float(np.sum(self.singular_values**2))
        arrays = self.array_state()
        float_count = sum(int(array.size) for array in arrays.values())
        return {
            "n": self.n,
            "dimension": self.dimension,
            "rank": self.rank,
            "max_rank": self.max_rank,
            "mean_is_exact_streaming_average": True,
            "historical_vectors_are_approximated": bool(
                self.cumulative_discarded_scatter_energy > 0.0
            ),
            "exact_scatter_trace": self.exact_scatter_trace,
            "retained_sketch_scatter_energy": retained,
            "cumulative_discarded_scatter_energy": self.cumulative_discarded_scatter_energy,
            "scatter_trace_balance_error": abs(
                self.exact_scatter_trace
                - retained
                - self.cumulative_discarded_scatter_energy
            ),
            "retained_scatter_fraction": (
                retained / self.exact_scatter_trace if self.exact_scatter_trace > 0 else 1.0
            ),
            "reconstruction_error_frobenius_upper_bound": (
                self.reconstruction_error_frobenius_upper_bound
            ),
            "discarded_energy_is_actual_squared_frobenius_error": True,
            "rms_reconstruction_error_per_image": (
                float(np.sqrt(self.cumulative_discarded_scatter_energy / self.n))
                if self.n else 0.0
            ),
            "svd_calls": self.svd_calls,
            "array_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "array_float64_count": float_count,
            "array_bytes": sum(int(array.nbytes) for array in arrays.values()),
            "full_embedding_float64_count_for_comparison": self.n * self.dimension,
            "storage_note": "Array bytes exclude IDs, Python objects and scalar metadata.",
        }

    def array_state(self) -> dict[str, np.ndarray]:
        """Copies suitable for np.savez; no full per-image embedding array."""
        return {
            "mean": self.mean.copy(),
            "basis": self.basis.copy(),
            "singular_values": self.singular_values.copy(),
            "coordinates": self.coordinates.copy(),
        }


def _self_test() -> dict[str, Any]:
    """Small mathematical checks, with raw synthetic vectors only in the test."""
    rng = np.random.default_rng(20261019)
    d = 16
    x = rng.normal(size=(30, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sketch = ClusterSketch()
    errors = []
    max_mean_error = 0.0
    max_trace_balance_error = 0.0
    for i, z in enumerate(x):
        event = sketch.append(str(i), z)
        true_mean = x[: i + 1].mean(axis=0)
        np.testing.assert_allclose(sketch.mean, true_mean, atol=2e-15)
        max_mean_error = max(max_mean_error, float(np.max(np.abs(sketch.mean - true_mean))))
        error = float(np.linalg.norm(x[: i + 1] - sketch.reconstruct()))
        errors.append(error)
        np.testing.assert_allclose(
            error**2, sketch.cumulative_discarded_scatter_energy, atol=1e-12
        )
        np.testing.assert_allclose(
            (sketch.reconstruct() - sketch.mean).T @ (x[: i + 1] - sketch.reconstruct()),
            0.0,
            atol=1e-12,
        )
        if i < 3:
            assert error < 1e-12, (i, error)
        assert error <= sketch.reconstruction_error_frobenius_upper_bound + 1e-12
        if event["svd_shape"] is not None:
            assert event["svd_shape"][0] <= 3
        np.testing.assert_allclose(sketch.coordinates.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(
            sketch.coordinates.T @ sketch.coordinates,
            np.diag(sketch.singular_values**2),
            atol=1e-12,
        )
        exact_scatter_trace = float(np.sum((x[: i + 1] - true_mean) ** 2))
        np.testing.assert_allclose(sketch.exact_scatter_trace, exact_scatter_trace, atol=1e-12)
        max_trace_balance_error = max(
            max_trace_balance_error, sketch.summary()["scatter_trace_balance_error"]
        )

    # The fixed-basis center-translation identity is exact for retained coords.
    basis, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    center_a, center_b = rng.normal(size=(2, d))
    old_coordinates = (x - center_a) @ basis
    translated = old_coordinates - (center_b - center_a) @ basis
    np.testing.assert_allclose(translated, (x - center_b) @ basis, atol=1e-14)

    # Data intrinsically contained in an affine 2-plane never requires a third
    # direction; evolving basis updates therefore stay exact even after n > 3.
    planar = rng.normal(size=(40, 2)) @ basis.T + rng.normal(size=d)
    planar_sketch = ClusterSketch()
    for i, z in enumerate(planar):
        planar_sketch.append(str(i), z)
    planar_error = float(np.linalg.norm(planar_sketch.reconstruct() - planar))
    assert planar_error < 1e-11

    # Exact means after a split come from the callback, not low-rank reconstructions.
    callback_calls = []

    def encode(member_id: str) -> np.ndarray:
        callback_calls.append(member_id)
        return x[int(member_id)]

    groups = [[str(i) for i in range(15)], [str(i) for i in range(15, 30)]]
    split = sketch.partition_reencoded(groups, encode)
    assert callback_calls == [str(i) for i in range(30)]
    for child, group in zip(split, groups):
        np.testing.assert_allclose(child.mean, x[list(map(int, group))].mean(axis=0), atol=1e-15)

    merged = split[0].merged(split[1])
    np.testing.assert_allclose(merged.mean, x.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(merged.scatter_trace, sketch.scatter_trace, atol=1e-12)
    np.testing.assert_allclose(
        merged.coords.T @ merged.coords, np.diag(merged.singular_values**2), atol=1e-12
    )
    assert merged.summary()["scatter_trace_balance_error"] < 1e-12
    merge_error = float(np.linalg.norm(merged.reconstruct() - x))
    assert merge_error <= merged.reconstruction_error_frobenius_upper_bound + 1e-12
    np.testing.assert_allclose(
        merge_error**2, merged.cumulative_discarded_scatter_energy, atol=1e-12
    )
    assert split[0].n == 15 and split[1].n == 15
    singleton_merges = ClusterSketch()
    for i, z in enumerate(x[:3]):
        singleton = ClusterSketch()
        singleton.append(str(i), z)
        singleton_merges = singleton_merges.merged(singleton)
    np.testing.assert_allclose(singleton_merges.reconstruct(), x[:3], atol=1e-12)

    identical = ClusterSketch()
    for i in range(4):
        identical.append(str(i), x[0])
    assert identical.rank == 0
    np.testing.assert_allclose(identical.reconstruct(), np.repeat(x[:1], 4, axis=0))

    copied = sketch.clone()
    copied.coordinates[0, 0] += 1
    assert copied.coordinates[0, 0] != sketch.coordinates[0, 0]
    np.testing.assert_allclose(sketch.reconstruct(["2", "0"]), sketch.reconstruct()[[2, 0]])
    assert sketch.coordinates.shape == (30, 2)
    assert sketch.basis.shape == (d, 2)
    assert all(array.shape != (30, d) for array in sketch.array_state().values())
    assert errors[-1] > 0.0
    return {
        "status": "passed",
        "checks": [
            "exact streaming mean",
            "rank <= 2 / SVD rows <= 3",
            "n <= 3 reconstruction exact",
            "fixed-basis center translation exact",
            "affine-rank-2 data reconstruction exact",
            "coordinates centered and match sketch singular values",
            "exact scatter trace and cumulative discarded trace balance",
            "discarded energy equals actual accumulated squared reconstruction error",
            "retained reconstruction and discarded error have zero cross scatter",
            "actual error below accumulated Frobenius upper bound",
            "split means exact via one callback per affected member",
            "merge exact mean and scatter trace, projected coordinates, no input mutation",
            "three-singleton merge reconstruction exact",
            "duplicate vectors need no residual direction",
            "clone and selected-ID reconstruction",
            "state has no n x d historical image vector array",
        ],
        "max_exact_mean_error": max_mean_error,
        "max_scatter_trace_balance_error": max_trace_balance_error,
        "n1_to_n3_reconstruction_frobenius_errors": errors[:3],
        "n30_reconstruction_frobenius_error": errors[-1],
        "n30_actual_squared_reconstruction_error": errors[-1] ** 2,
        "n30_discarded_scatter_energy_equals_actual_squared_error": (
            sketch.cumulative_discarded_scatter_energy
        ),
        "planar_n40_reconstruction_frobenius_error": planar_error,
        "merged_two_halves_n30_reconstruction_frobenius_error": merge_error,
        "final_summary": sketch.summary(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(_self_test(), indent=2))
