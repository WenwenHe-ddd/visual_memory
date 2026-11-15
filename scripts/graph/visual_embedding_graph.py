"""Offline visual clustering without retaining individual image embeddings.

The only persisted vectors are master means of unit image vectors.  A master's
mean norm is exactly the mean cosine similarity to its normalized center.  Old
image-edge distances receive a shared *approximate* correction after a center
move; their mean, together with the exact new edge, stays exact.  Individual
old distances cannot be recovered from a mean vector.
"""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Iterable


METHOD = "visual_embedding_dynamic"
SCHEMA_VERSION = 1
EXACT_DISTANCE = "exact_at_insertion"
APPROXIMATE_DISTANCE = "uniform_mean_shift_approximation"
_TOLERANCE = 1e-9


def _number(value: Any, label: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _vector(value: Iterable[float], label: str) -> list[float]:
    if isinstance(value, (str, bytes, dict)):
        raise ValueError(f"{label} must be a nonempty numeric vector")
    try:
        result = [_number(part, label) for part in value]
    except TypeError as exc:
        raise ValueError(f"{label} must be a nonempty numeric vector") from exc
    if not result:
        raise ValueError(f"{label} must be a nonempty numeric vector")
    return result


def _unit(value: Iterable[float]) -> list[float]:
    vector = _vector(value, "embedding")
    # Scaling avoids overflow/underflow for otherwise valid finite inputs.
    scale = max(abs(part) for part in vector)
    if scale == 0:
        raise ValueError("embedding must have nonzero norm")
    scaled = [part / scale for part in vector]
    length = math.hypot(*scaled)
    return [part / length for part in scaled]


def _dot(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _json_value(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


def _keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must have exactly these fields: {sorted(expected)}")


class VisualEmbeddingGraph:
    """Insertion-ordered graph with one aggregate vector per master.

    ``image_id`` identifies a memory occurrence, so repeated uses of one image
    should have different IDs.  Image paths are provenance, not vector caches.
    Ties are resolved by master creation order.  Admission is strictly greater
    than the existing threshold, with no margin ranking or tolerance.
    """

    def __init__(self) -> None:
        self.embedding_dim: int | None = None
        self.masters: list[dict[str, Any]] = []
        self.image_nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    @property
    def insertion_order(self) -> list[str]:
        return list(self.image_nodes)

    def insert(
        self,
        image_id: str,
        embedding: Iterable[float],
        *,
        dialogue: Any,
        caption: str,
        timestamp: Any,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        """Insert one memory and return a scalar-only clustering audit.

        Inputs are checked before mutation.  The temporary normalized image
        vector is used for admission and aggregate updates, then discarded.
        """
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("image_id must be a nonempty string")
        if image_id in self.image_nodes:
            raise ValueError(f"duplicate image_id: {image_id}")
        if not isinstance(caption, str):
            raise ValueError("caption must be a string")
        if image_path is not None and not isinstance(image_path, str):
            raise ValueError("image_path must be a string or null")
        node = {
            "id": image_id,
            "image_id": image_id,
            "image_path": image_path,
            "dialogue": _json_value(dialogue, "dialogue"),
            "caption": caption,
            "timestamp": _json_value(timestamp, "timestamp"),
        }
        vector = _unit(embedding)
        if self.embedding_dim is not None and len(vector) != self.embedding_dim:
            raise ValueError(
                f"embedding dimension {len(vector)} does not match {self.embedding_dim}"
            )

        candidates = []
        chosen = None
        chosen_similarity = -math.inf
        for master in self.masters:
            # Only numerical cosine roundoff is bounded; stored edge distances
            # are never clipped, which would break the mean-distance invariant.
            similarity = max(-1.0, min(1.0, _dot(vector, _unit(master["visual_mean"]))))
            threshold = master["join_threshold"]
            eligible = similarity > threshold
            candidates.append({
                "master_id": master["id"],
                "similarity": similarity,
                "threshold": threshold,
                "eligible": eligible,
            })
            if eligible and similarity > chosen_similarity:
                chosen, chosen_similarity = master, similarity

        if chosen is None:
            threshold = max((row["similarity"] for row in candidates), default=0.0)
            chosen = {
                "id": f"M{len(self.masters) + 1}",
                "visual_mean": vector,
                "member_count": 1,
                "join_threshold": threshold,
            }
            self.masters.append(chosen)
            similarity_before = None
            threshold_before = None
            distance_delta = 0.0
            new_distance = 0.0
            is_new = True
        else:
            old_mean = chosen["visual_mean"]
            count = chosen["member_count"]
            updated_mean = [
                (count * old + new) / (count + 1)
                for old, new in zip(old_mean, vector)
            ]
            updated_center = _unit(updated_mean)
            updated_threshold = math.hypot(*updated_mean)
            distance_delta = math.hypot(*old_mean) - _dot(old_mean, updated_center)
            new_distance = 1.0 - _dot(vector, updated_center)
            similarity_before = chosen_similarity
            threshold_before = chosen["join_threshold"]
            # Delta describes the old population only, not the change in the
            # combined threshold (which also includes the newly added image).
            for edge in self.edges.values():
                if edge["master_id"] == chosen["id"]:
                    edge["distance"] += distance_delta
                    edge["distance_semantics"] = APPROXIMATE_DISTANCE
            chosen.update(
                visual_mean=updated_mean,
                member_count=count + 1,
                join_threshold=updated_threshold,
            )
            is_new = False

        self.embedding_dim = len(vector)
        self.image_nodes[image_id] = node
        self.edges[image_id] = {
            "master_id": chosen["id"],
            "image_id": image_id,
            "distance": new_distance,
            "distance_semantics": EXACT_DISTANCE,
        }
        return {
            "image_id": image_id,
            "master_id": chosen["id"],
            "new_master": is_new,
            "similarity_before": similarity_before,
            "threshold_before": threshold_before,
            "threshold_after": chosen["join_threshold"],
            "old_edges_distance_delta": distance_delta,
            "new_edge_distance": new_distance,
            "candidates": candidates,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON-ready snapshot; node/edge order is insertion order."""
        self.validate()
        return copy.deepcopy({
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "embedding_dim": self.embedding_dim,
            "masters": self.masters,
            "image_nodes": list(self.image_nodes.values()),
            "edges": list(self.edges.values()),
        })

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any]) -> "VisualEmbeddingGraph":
        """Restore strictly: unexpected node fields (including embeddings) fail."""
        _keys(snapshot, {"schema_version", "method", "embedding_dim", "masters",
                         "image_nodes", "edges"}, "snapshot")
        if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported visual graph schema_version")
        if snapshot["method"] != METHOD:
            raise ValueError("unsupported visual graph method")
        for key in ("masters", "image_nodes", "edges"):
            if not isinstance(snapshot[key], list):
                raise ValueError(f"{key} must be a list")
        graph = cls()
        graph.embedding_dim = snapshot["embedding_dim"]
        graph.masters = copy.deepcopy(snapshot["masters"])
        for node in snapshot["image_nodes"]:
            _keys(node, {"id", "image_id", "image_path", "dialogue", "caption", "timestamp"},
                  "image node")
            node_id = node["id"]
            if not isinstance(node_id, str) or not node_id or node_id in graph.image_nodes:
                raise ValueError("invalid or duplicate image node id")
            graph.image_nodes[node_id] = copy.deepcopy(node)
        for edge in snapshot["edges"]:
            _keys(edge, {"master_id", "image_id", "distance", "distance_semantics"}, "edge")
            image_id = edge["image_id"]
            if not isinstance(image_id, str) or image_id in graph.edges:
                raise ValueError("invalid or duplicate edge image_id")
            graph.edges[image_id] = copy.deepcopy(edge)
        graph.validate()
        return graph

    def validate(self) -> None:
        """Check structure and exact aggregate identities without member vectors."""
        if not isinstance(self.masters, list) or not isinstance(self.image_nodes, dict) or not isinstance(self.edges, dict):
            raise ValueError("invalid graph containers")
        if not self.masters:
            if self.embedding_dim is not None or self.image_nodes or self.edges:
                raise ValueError("an empty graph cannot contain nodes, edges, or a dimension")
            return
        if type(self.embedding_dim) is not int or self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        if list(self.image_nodes) != list(self.edges):
            raise ValueError("image nodes and edges must have matching insertion order")
        master_by_id = {}
        distances: dict[str, list[float]] = {}
        for index, master in enumerate(self.masters, 1):
            _keys(master, {"id", "visual_mean", "member_count", "join_threshold"}, "master")
            if master["id"] != f"M{index}":
                raise ValueError("master ids must follow creation order M1, M2, ...")
            mean = _vector(master["visual_mean"], "visual_mean")
            if len(mean) != self.embedding_dim:
                raise ValueError("master visual_mean dimension mismatch")
            norm = math.hypot(*mean)
            if not 0 < norm <= 1.0 + _TOLERANCE:
                raise ValueError("master visual_mean must have norm in (0, 1]")
            count = master["member_count"]
            if type(count) is not int or count < 1:
                raise ValueError("member_count must be a positive integer")
            threshold = _number(master["join_threshold"], "join_threshold")
            if not -1.0 - _TOLERANCE <= threshold <= 1.0 + _TOLERANCE:
                raise ValueError("join_threshold must lie in [-1, 1]")
            if count > 1 and not math.isclose(threshold, norm, abs_tol=_TOLERANCE, rel_tol=0):
                raise ValueError("multi-member threshold must equal visual_mean norm")
            if count == 1:
                if not math.isclose(norm, 1.0, abs_tol=_TOLERANCE, rel_tol=0):
                    raise ValueError("single-member visual_mean must be a unit vector")
                if index == 1 and threshold != 0.0:
                    raise ValueError("first singleton master must have initial threshold 0")
            master_by_id[master["id"]] = master
            distances[master["id"]] = []
        for image_id, node in self.image_nodes.items():
            _keys(node, {"id", "image_id", "image_path", "dialogue", "caption", "timestamp"},
                  "image node")
            if not isinstance(image_id, str) or not image_id or node["id"] != image_id or node["image_id"] != image_id:
                raise ValueError("invalid image node id")
            if not isinstance(node["caption"], str):
                raise ValueError("caption must be a string")
            if node["image_path"] is not None and not isinstance(node["image_path"], str):
                raise ValueError("image_path must be a string or null")
            _json_value(node["dialogue"], "dialogue")
            _json_value(node["timestamp"], "timestamp")
            edge = self.edges[image_id]
            _keys(edge, {"master_id", "image_id", "distance", "distance_semantics"}, "edge")
            if edge["image_id"] != image_id or not isinstance(edge["master_id"], str) or edge["master_id"] not in master_by_id:
                raise ValueError("edge must connect its image to an existing master")
            if edge["distance_semantics"] not in (EXACT_DISTANCE, APPROXIMATE_DISTANCE):
                raise ValueError("unsupported distance semantics")
            distance = _number(edge["distance"], "distance")
            if distance < -_TOLERANCE:
                raise ValueError("edge distance must not be negative")
            # An approximated individual edge can exceed 2. Never clip it.
            distances[edge["master_id"]].append(distance)
        for master_id, master in master_by_id.items():
            member_distances = distances[master_id]
            if len(member_distances) != master["member_count"]:
                raise ValueError("master member_count does not match its edges")
            mean_distance = math.fsum(member_distances) / master["member_count"]
            expected = 1.0 - math.hypot(*master["visual_mean"])
            if not math.isclose(mean_distance, expected, abs_tol=_TOLERANCE, rel_tol=0):
                raise ValueError("mean edge distance must equal 1 - norm(visual_mean)")
