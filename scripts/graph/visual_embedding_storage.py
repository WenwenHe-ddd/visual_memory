"""Portable, immutable storage for master aggregate vectors only.

The graph JSON retains row references; float64 master means live together in a
content-addressed NPY file beside it. Legacy graphs with inline means remain
readable. Keep previous vector files until both checkpoint and graph exports
have committed, then call ``cleanup_unreferenced_vectors`` under the build lock.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from scripts.graph.visual_embedding_graph import VisualEmbeddingGraph


_VECTOR_NAME = re.compile(r"master_visual_means\.[0-9a-f]{64}\.npy\Z")
_STORAGE_KEYS = {"format", "path", "sha256", "shape", "dtype"}


def _task_path(task_dir: str | Path, relative: str) -> Path:
    """Resolve a task-local reference, rejecting traversal and outside symlinks."""
    root = Path(task_dir).resolve()
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or "\\" in relative
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("vector path must be a relative path within the task directory")
    result = root / relative
    if not result.resolve().is_relative_to(root):
        raise ValueError("vector path escapes the task directory")
    return result


def _metadata(snapshot: dict[str, Any], task_dir: str | Path) -> tuple[dict, Path]:
    storage = snapshot.get("vector_storage")
    if not isinstance(storage, dict) or set(storage) != _STORAGE_KEYS:
        raise ValueError("invalid vector_storage metadata fields")
    checksum = storage["sha256"]
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("invalid vector_storage sha256")
    if storage["format"] != "npy" or storage["dtype"] != "float64":
        raise ValueError("vector_storage requires npy format and float64 dtype")
    path = _task_path(task_dir, storage["path"])
    if storage["path"] != f"vectors/master_visual_means.{checksum}.npy":
        raise ValueError("vector_storage path must match its content-addressed filename")
    shape = storage["shape"]
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(size) is not int or size < 0 for size in shape)):
        raise ValueError("vector_storage shape must contain two nonnegative integers")
    masters = snapshot.get("masters")
    dimension = snapshot.get("embedding_dim")
    if not isinstance(masters, list):
        raise ValueError("masters must be a list")
    if masters and (type(dimension) is not int or dimension < 1):
        raise ValueError("embedding_dim must be a positive integer")
    if shape != [len(masters), dimension if masters else 0]:
        raise ValueError("vector_storage shape disagrees with graph masters/dimension")
    return storage, path


def externalize_snapshot(snapshot: dict[str, Any], task_dir: str | Path) -> dict[str, Any]:
    """Write master means atomically and return JSON containing only references.

    The supplied snapshot must use the core inline schema and is never mutated.
    Existing content-addressed files are verified and reused, never overwritten.
    This function does not commit any graph/checkpoint JSON or delete old files.
    """
    result = VisualEmbeddingGraph.from_dict(snapshot).to_dict()
    masters = result["masters"]
    matrix = (np.asarray([master["visual_mean"] for master in masters], dtype=np.float64)
              if masters else np.empty((0, 0), dtype=np.float64))
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    relative = f"vectors/master_visual_means.{checksum}.npy"
    path = _task_path(task_dir, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A hard link publishes the complete temporary file atomically and refuses
    # to overwrite any existing generation, even under concurrent writers.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".master_visual_means.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("existing content-addressed master vector file is corrupt")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result["vector_storage"] = dict(format="npy", path=relative, sha256=checksum,
                                    shape=list(matrix.shape), dtype="float64")
    for row, master in enumerate(masters):
        del master["visual_mean"]
        master["visual_mean_ref"] = {"row": row}
    return result


def hydrate_snapshot(snapshot: dict[str, Any], task_dir: str | Path) -> dict[str, Any]:
    """Restore and validate an external or legacy-inline graph without mutation."""
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be an object")
    if "vector_storage" not in snapshot:
        return VisualEmbeddingGraph.from_dict(snapshot).to_dict()
    storage, path = _metadata(snapshot, task_dir)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != storage["sha256"]:
        raise ValueError("master vector file sha256 mismatch")
    try:
        matrix = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError("invalid master vector NPY file") from exc
    if not isinstance(matrix, np.ndarray):
        if hasattr(matrix, "close"):
            matrix.close()
        raise ValueError("master vector file must contain a single NPY array")
    if matrix.dtype != np.dtype("float64"):
        raise ValueError("master vector file dtype must be float64")
    if list(matrix.shape) != storage["shape"]:
        raise ValueError("master vector file shape mismatch")
    if not np.isfinite(matrix).all():
        raise ValueError("master vector file contains non-finite values")
    result = copy.deepcopy(snapshot)
    del result["vector_storage"]
    for row, master in enumerate(result["masters"]):
        if not isinstance(master, dict) or "visual_mean" in master:
            raise ValueError("external master must have a row reference and no inline visual_mean")
        reference = master.pop("visual_mean_ref", None)
        if (not isinstance(reference, dict) or set(reference) != {"row"}
                or type(reference["row"]) is not int or reference["row"] != row):
            raise ValueError("master visual_mean_ref rows must follow master order without gaps")
        master["visual_mean"] = matrix[row].tolist()
    return VisualEmbeddingGraph.from_dict(result).to_dict()


def load_graph(path: str | Path) -> VisualEmbeddingGraph:
    """Load a graph.json file, resolving its vector path relative to that file."""
    path = Path(path)
    return VisualEmbeddingGraph.from_dict(hydrate_snapshot(
        json.loads(path.read_text(encoding="utf-8")), path.parent))


def cleanup_unreferenced_vectors(task_dir: str | Path) -> list[Path]:
    """Remove only this scheme's files unused by current graph/checkpoint JSON.

    Call after both JSON files have committed, while holding the writer lock.
    Both documents are inspected before deleting anything; malformed references
    abort cleanup. Missing documents are tolerated, but at least one is needed.
    """
    root = Path(task_dir)
    retained: set[Path] = set()
    found_document = False
    for name in ("graph.json", "checkpoint.json"):
        path = root / name
        if not path.exists():
            continue
        found_document = True
        document = json.loads(path.read_text(encoding="utf-8"))
        snapshot = document if name == "graph.json" else document["graph"]
        hydrate_snapshot(snapshot, root)
        if "vector_storage" in snapshot:
            retained.add(_metadata(snapshot, root)[1])
    if not found_document:
        return []
    vectors = _task_path(root, "vectors")
    removed = []
    if vectors.is_dir():
        for path in vectors.iterdir():
            if _VECTOR_NAME.fullmatch(path.name) and path.is_file() and path not in retained:
                path.unlink()
                removed.append(path)
    return removed
