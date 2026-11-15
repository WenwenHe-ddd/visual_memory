"""Caption-only Ward clustering with compact, independently updated vision.

Text vectors determine the partition.  Each leaf retains exactly two visual
coordinates in a fixed orthonormal basis; masters retain unnormalised visual
sums.  Old visual residuals cannot be recovered after a split: they are shared
among new clusters in proportion to the overlap with each old cluster.

The clustering selector considers 2..N-1 clusters, not the otherwise undefined
silhouette endpoints.  N<=2 keeps singletons.  For N>=3, exactly identical
normalised text vectors form one cluster as an explicit degeneracy rule.
"""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score


METHOD = "caption_ward_silhouette_visual"
SCHEMA_VERSION = 1
SCORE_TIE_TOLERANCE = 1e-12


def _vector(value: Any, label: str, dimension: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric vector") from exc
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty finite vector")
    if dimension is not None and result.size != dimension:
        raise ValueError(f"{label} must have dimension {dimension}")
    return result.copy()


def _unit(value: Any, label: str, dimension: int | None = None) -> np.ndarray:
    vector = _vector(value, label, dimension)
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError(f"{label} must have nonzero norm")
    vector /= scale
    vector /= np.linalg.norm(vector)
    return vector


def _basis(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or result.shape[0] < 2
        or result.shape[1] != 2
        or not np.isfinite(result).all()
    ):
        raise ValueError("basis must be a finite D-by-2 array with D>=2")
    if not np.allclose(result.T @ result, np.eye(2), rtol=0, atol=1e-10):
        raise ValueError("basis columns must be orthonormal")
    return result.copy()


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


def choose_caption_partition(text_embeddings: Any) -> dict[str, Any]:
    """Return exact-K Ward cuts and the best Euclidean silhouette partition.

    Rows are normalised here, so callers may pass either unit or raw vectors.
    Cluster order follows the first input member; ties within 1e-12 prefer fewer
    clusters.  Linkage rows are replayed instead of distance-threshold cuts,
    which can skip K when several merges have equal heights.
    """
    vectors = np.asarray(text_embeddings, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("text_embeddings must be a nonempty-width 2D array")
    if not np.isfinite(vectors).all():
        raise ValueError("text_embeddings must be finite")
    n = len(vectors)
    if n == 0:
        return {
            "partition": [], "candidate_scores": [], "selected_k": 0,
            "selection_reason": "empty", "linkage": None,
        }
    vectors = np.stack([_unit(row, "text embedding") for row in vectors])
    if n <= 2: # 只有一个节点，直接返回
        return {
            "partition": [[i] for i in range(n)],
            "candidate_scores": [], "selected_k": n,
            "selection_reason": "n_le_2_singletons", "linkage": None,
        }
    if np.array_equal(vectors, np.broadcast_to(vectors[0], vectors.shape)):
        return {
            "partition": [list(range(n))],
            "candidate_scores": [], "selected_k": 1,
            "selection_reason": "all_text_vectors_identical", "linkage": None,
        }

    tree = linkage(vectors, method="ward", metric="euclidean")
    distances = squareform(pdist(vectors, metric="euclidean"))
    active = {i: [i] for i in range(n)}
    candidates: list[dict[str, Any]] = []
    best_partition: list[list[int]] | None = None
    best_score = -np.inf
    best_k = n
    for merge_index, row in enumerate(tree):
        left, right = int(row[0]), int(row[1])
        members = sorted(active.pop(left) + active.pop(right))
        active[n + merge_index] = members
        k = len(active)
        if not 2 <= k <= n - 1:
            continue
        partition = sorted(active.values(), key=lambda group: group[0])
        labels = np.empty(n, dtype=np.int64)
        for label, group in enumerate(partition):
            labels[group] = label
        score = float(silhouette_score(distances, labels, metric="precomputed"))
        candidates.append({"k": k, "score": score})
        if (
            score > best_score + SCORE_TIE_TOLERANCE
            or (
                abs(score - best_score) <= SCORE_TIE_TOLERANCE
                and k < best_k
            )
        ): # 分数更高就替换当前最佳结果；分数基本相同时，选组数更少的。
            best_score, best_k = score, k
            best_partition = [list(group) for group in partition]
    if best_partition is None:
        raise RuntimeError("no admissible Ward cut found")
    return {
        "partition": best_partition,
        "candidate_scores": sorted(candidates, key=lambda row: row["k"]),
        "selected_k": best_k,
        "selected_score": best_score,
        "selection_reason": "maximum_mean_euclidean_silhouette",
        "linkage": tree,
    }


def update_visual_partition(
    old_masters: Mapping[str, Mapping[str, Any]],
    old_coords: Mapping[str, Any],
    partition: Sequence[Sequence[str]],
    new_vectors: Mapping[str, Any],
    basis: Any,
    next_master_index: int = 1,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], int]:
    """Redistribute one immutable old snapshot into a new membership partition.

    ``new_vectors`` contains only the current insertion's transient image
    vectors; this pure helper does not normalise them.  If C is an old master,
    its unobserved residual is R_C=S_C-U sum(z_i).  A new group receives its
    members' exact U-coordinates plus |C intersect group|/|C| of R_C.

    Master IDs are reused only for identical member sets.  Full-cluster merges
    and additions preserve old aggregates, but previous approximation flags
    persist.  Every partial old-cluster redistribution is marked approximate.
    """
    u = _basis(basis)
    dim = u.shape[0]
    coords = {key: _vector(z, f"coordinates of {key}", 2)
              for key, z in old_coords.items()}
    new = {key: _vector(v, f"new vector {key}", dim)
           for key, v in new_vectors.items()}
    if set(coords) & set(new):
        raise ValueError("new_vectors and old_coords must have disjoint IDs")
    if not isinstance(next_master_index, int) or next_master_index < 1:
        raise ValueError("next_master_index must be a positive integer")

    owner: dict[str, str] = {}
    prepared: dict[str, dict[str, Any]] = {}
    identical_sets: dict[frozenset[str], str] = {}
    for master_id, master in old_masters.items():
        members = list(master["member_ids"])
        if not members or len(set(members)) != len(members):
            raise ValueError("old master members must be nonempty and unique")
        if master.get("n", len(members)) != len(members):
            raise ValueError("old master count does not match its members")
        for member in members:
            if member in owner or member not in coords:
                raise ValueError("old masters must partition old coordinate IDs")
            owner[member] = master_id
        total = _vector(master["visual_sum"], f"sum of {master_id}", dim)
        coord_sum = np.sum([coords[member] for member in members], axis=0)
        prepared[master_id] = {
            "members": members, "sum": total, "n": len(members),
            "residual": total - u @ coord_sum,
        }
        identical_sets[frozenset(members)] = master_id
    if set(owner) != set(coords):
        raise ValueError("old masters must cover every old coordinate ID")
    groups = [list(group) for group in partition]
    flattened = [member for group in groups for member in group]
    if (
        any(not group for group in groups)
        or len(set(flattened)) != len(flattened)
        or set(flattened) != set(coords) | set(new)
    ):
        raise ValueError("partition must cover every old and new ID exactly once")

    updated: dict[str, dict[str, Any]] = {}
    overlaps: list[dict[str, Any]] = []
    unchanged_ids: list[str] = []
    changed_ids: list[str] = []
    partial_ids: list[str] = []
    merged_ids: list[str] = []
    old_destinations: dict[str, list[str]] = {key: [] for key in old_masters}
    for members in groups:
        member_set = frozenset(members)
        old_id = identical_sets.get(member_set)
        sources: dict[str, list[str]] = {}
        for member in members:
            if member in owner:
                sources.setdefault(owner[member], []).append(member)
        if old_id is not None:
            master_id = old_id
            # Copying preserves bits and prevents mutation of the old snapshot.
            master = copy.deepcopy(dict(old_masters[old_id]))
            master["visual_sum"] = prepared[old_id]["sum"].copy()
            master.setdefault("approximate", False)
            master.setdefault("source_overlaps", [])
            unchanged_ids.append(master_id)
        else:
            while True:
                master_id = f"master_{next_master_index:06d}"
                next_master_index += 1
                if master_id not in old_masters and master_id not in updated:
                    break
            total = np.zeros(dim, dtype=np.float64)
            approximate = False
            partial = False
            for source, selected in sources.items():
                old = prepared[source]
                approximate |= bool(old_masters[source].get("approximate", False))
                if len(selected) == old["n"]: # 取了旧组全部成员
                    total += old["sum"]
                else: # 只取旧组部分成员
                    total += u @ np.sum([coords[m] for m in selected], axis=0)
                    total += (len(selected) / old["n"]) * old["residual"]
                    partial = approximate = True
            for member in members:
                if member in new:
                    total += new[member]
            source_records = [
                {"old_master_id": source, "count": len(selected),
                 "old_n": prepared[source]["n"],
                 "fraction": len(selected) / prepared[source]["n"]}
                for source, selected in sources.items()
            ]
            master = {
                "id": master_id, "member_ids": list(members), "n": len(members),
                "visual_sum": total, "approximate": approximate,
                "source_overlaps": source_records,
            }
            changed_ids.append(master_id)
            if partial:
                partial_ids.append(master_id)
        updated[master_id] = master
        if len(sources) >= 2:
            merged_ids.append(master_id)
        for source, selected in sources.items():
            old_destinations[source].append(master_id)
            overlaps.append({
                "old_master_id": source, "new_master_id": master_id,
                "count": len(selected), "old_n": prepared[source]["n"],
                "fraction": len(selected) / prepared[source]["n"],
                "member_ids": list(selected),
            })

    before_sum = np.zeros(dim, dtype=np.float64)
    for old in prepared.values():
        before_sum += old["sum"]
    for vector in new.values():
        before_sum += vector
    after_sum = np.zeros(dim, dtype=np.float64)
    projection_errors = []
    for master in updated.values():
        after_sum += master["visual_sum"]
        expected = np.zeros(2, dtype=np.float64)
        for member in master["member_ids"]:
            expected += coords[member] if member in coords else u.T @ new[member]
        projection_errors.append(float(np.linalg.norm(
            u.T @ master["visual_sum"] - expected
        )))
    conservation_delta = after_sum - before_sum
    audit = {
        "clusters_partition": [list(master["member_ids"])
                               for master in updated.values()],
        "clusters": [
            {"master_id": master_id, "member_ids": list(master["member_ids"]),
             "n": master["n"], "approximate": master["approximate"]}
            for master_id, master in updated.items()
        ],
        "unchanged_master_ids": unchanged_ids,
        "changed_master_ids": changed_ids,
        "retired_master_ids": [key for key in old_masters if key not in updated],
        "source_overlaps": overlaps,
        "split_master_ids": [key for key, targets in old_destinations.items()
                             if len(targets) > 1],
        "merged_master_ids": merged_ids,
        "partial_redistribution_master_ids": partial_ids,
        "old_members_in_changed_clusters": sum(
            sum(member in coords for member in updated[key]["member_ids"])
            for key in changed_ids
        ),
        "new_image_ids": list(new),
        "global_conservation_l2_error": float(np.linalg.norm(conservation_delta)),
        "global_conservation_max_abs_error": float(np.max(
            np.abs(conservation_delta), initial=0.0
        )),
        "max_projection_consistency_l2_error": max(projection_errors, default=0.0),
    }
    return updated, audit, next_master_index


class CaptionWardVisualMemory:
    """Incremental public interface; persistent leaf vision is exactly 2-D."""

    def __init__(self, basis: Any) -> None:
        self.basis = _basis(basis)
        self.basis.setflags(write=False)
        self.visual_dim = self.basis.shape[0]
        self.text_dim: int | None = None
        self.masters: dict[str, dict[str, Any]] = {}
        self.image_nodes: dict[str, dict[str, Any]] = {}
        # Reconstructible text cache, independent from the graph's leaf fields.
        self.text_embedding_cache: dict[str, np.ndarray] = {}
        self._next_master_index = 1

    def add(
        self,
        image_id: str,
        caption: str,
        text_embedding: Any,
        visual_embedding: Any,
        dialogue: Any = None,
        timestamp: Any = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a nonempty string")
        if image_id in self.image_nodes:
            raise ValueError(f"duplicate image_id: {image_id}")
        if not isinstance(caption, str):
            raise ValueError("caption must be a string")
        text = _unit(text_embedding, "text embedding", self.text_dim)
        visual = _unit(visual_embedding, "visual embedding", self.visual_dim)
        node = {
            "id": image_id,
            "caption": caption,
            "dialogue": _json_copy(dialogue, "dialogue"),
            "timestamp": _json_copy(timestamp, "timestamp"),
            "visual_coords": (self.basis.T @ visual).tolist(),
        }
        image_ids = [*self.image_nodes, image_id]
        matrix = np.stack([*self.text_embedding_cache.values(), text])
        before_clustering = time.perf_counter()
        choice = choose_caption_partition(matrix)
        after_clustering = time.perf_counter()
        partition = [[image_ids[index] for index in group]
                     for group in choice["partition"]]
        old_coords = {key: old["visual_coords"]
                      for key, old in self.image_nodes.items()}
        updated, audit, next_index = update_visual_partition(
            self.masters, old_coords, partition, {image_id: visual}, self.basis,
            self._next_master_index,
        )
        after_visual = time.perf_counter()
        self.image_nodes[image_id] = node
        self.text_embedding_cache[image_id] = text
        self.text_dim = len(text)
        self.masters = updated
        self._next_master_index = next_index
        for master_id, master in self.masters.items():
            for member in master["member_ids"]:
                self.image_nodes[member]["master_id"] = master_id
        audit.update({
            "image_id": image_id,
            "n_images": len(self.image_nodes),
            "n_clusters": len(self.masters),
            "selected_k": choice["selected_k"],
            "selected_score": choice.get("selected_score"),
            "candidate_scores": choice["candidate_scores"],
            "selection_reason": choice["selection_reason"],
            "timings_seconds": {
                "prepare": before_clustering - started,
                "clustering": after_clustering - before_clustering,
                "visual_update": after_visual - after_clustering,
                "total": time.perf_counter() - started,
            },
        })
        return audit

    def snapshot(self) -> dict[str, Any]:
        """JSON graph with sidecar references, never per-image full vectors."""
        masters = []
        for master_id, master in self.masters.items():
            row = {key: copy.deepcopy(value) for key, value in master.items()
                   if key != "visual_sum"}
            row["visual_sum_ref"] = f"masters/{master_id}/visual_sum"
            row["visual_mean_ref"] = f"masters/{master_id}/visual_mean"
            masters.append(row)
        return {
            "method": METHOD,
            "schema_version": SCHEMA_VERSION,
            "visual_dim": self.visual_dim,
            "leaf_visual_dim": 2,
            "text_dim": self.text_dim,
            "basis_ref": "shared_basis",
            "masters": masters,
            "image_nodes": copy.deepcopy(list(self.image_nodes.values())),
            "text_embedding_cache": {
                "storage": "external_reconstructible_cache",
                "count": len(self.text_embedding_cache),
            },
        }

    def export_visual_arrays(self) -> dict[str, np.ndarray]:
        """Copy sidecar arrays; leaf vectors and residuals are never exported."""
        arrays = {"shared_basis": self.basis.copy()}
        for master_id, master in self.masters.items():
            arrays[f"masters/{master_id}/visual_sum"] = master["visual_sum"].copy()
            arrays[f"masters/{master_id}/visual_mean"] = (
                master["visual_sum"] / master["n"]
            )
        return arrays
