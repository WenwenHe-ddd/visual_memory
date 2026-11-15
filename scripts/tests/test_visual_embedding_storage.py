"""Persistence, migration, portability, and integrity of external master means."""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from scripts.graph.visual_embedding_graph import VisualEmbeddingGraph
from scripts.graph.visual_embedding_storage import (
    cleanup_unreferenced_vectors,
    externalize_snapshot,
    hydrate_snapshot,
    load_graph,
)


def insert(graph, image_id, vector):
    return graph.insert(image_id, vector, dialogue=[{"user": "完整对话"}],
                        caption="complete caption", timestamp="2026-10-19")


def example_graph():
    graph = VisualEmbeddingGraph()
    insert(graph, "a", [1, 0, 0])
    insert(graph, "b", [0, 1, 0])
    insert(graph, "c", [1, 0.2, 0])
    return graph


class VisualEmbeddingStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / "task"
        self.task.mkdir()
        self.graph = example_graph()
        self.inline = self.graph.to_dict()

    def write_graph(self, snapshot, name="graph.json"):
        (self.task / name).write_text(json.dumps(snapshot), encoding="utf-8")

    def write_matrix(self, external, matrix):
        snapshot = copy.deepcopy(external)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        payload = buffer.getvalue()
        checksum = hashlib.sha256(payload).hexdigest()
        relative = f"vectors/master_visual_means.{checksum}.npy"
        (self.task / relative).write_bytes(payload)
        snapshot["vector_storage"].update(path=relative, sha256=checksum)
        return snapshot

    def test_roundtrip_is_exact_and_keeps_source_unchanged(self):
        original = copy.deepcopy(self.inline)
        external = externalize_snapshot(self.inline, self.task)
        self.assertEqual(self.inline, original)
        self.assertNotIn('"visual_mean":', json.dumps(external))
        self.assertEqual(external["vector_storage"]["shape"], [2, 3])
        self.assertEqual(external["masters"][1]["visual_mean_ref"], {"row": 1})
        self.assertEqual(external["image_nodes"], self.inline["image_nodes"])
        self.assertEqual(external["edges"], self.inline["edges"])
        restored = hydrate_snapshot(external, self.task)
        self.assertEqual(restored, self.inline)
        self.assertNotIn("visual_mean", external["masters"][0])
        self.write_graph(external)
        self.assertEqual(load_graph(self.task / "graph.json").to_dict(), self.inline)
        # float64 serialization preserves future clustering decisions as well.
        resumed = VisualEmbeddingGraph.from_dict(restored)
        self.assertEqual(insert(resumed, "next", [0.1, 1, 0]),
                         insert(self.graph, "next", [0.1, 1, 0]))
        self.assertEqual(resumed.to_dict(), self.graph.to_dict())

    def test_legacy_inline_graph_loads_and_can_be_migrated(self):
        self.write_graph(self.inline)
        legacy = load_graph(self.task / "graph.json")
        self.assertEqual(legacy.to_dict(), self.inline)
        external = externalize_snapshot(legacy.to_dict(), self.task)
        self.write_graph(external)
        self.assertEqual(load_graph(self.task / "graph.json").to_dict(), self.inline)

    def test_task_directory_is_portable(self):
        external = externalize_snapshot(self.inline, self.task)
        self.write_graph(external)
        moved = self.root / "moved"
        shutil.move(self.task, moved)
        self.assertEqual(load_graph(moved / "graph.json").to_dict(), self.inline)

    def test_empty_graph_roundtrip(self):
        inline = VisualEmbeddingGraph().to_dict()
        external = externalize_snapshot(inline, self.task)
        self.assertEqual(external["vector_storage"]["shape"], [0, 0])
        self.assertEqual(hydrate_snapshot(external, self.task), inline)

    def test_repeated_save_reuses_immutable_file(self):
        first = externalize_snapshot(self.inline, self.task)
        path = self.task / first["vector_storage"]["path"]
        inode = path.stat().st_ino
        second = externalize_snapshot(self.inline, self.task)
        self.assertEqual(first, second)
        self.assertEqual(path.stat().st_ino, inode)
        self.assertEqual(len(list(path.parent.iterdir())), 1)
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            externalize_snapshot(self.inline, self.task)
        self.assertEqual(path.read_bytes(), b"corrupt")
        self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_missing_and_corrupt_vector_files_are_rejected(self):
        external = externalize_snapshot(self.inline, self.task)
        path = self.task / external["vector_storage"]["path"]
        payload = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            hydrate_snapshot(external, self.task)
        path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            hydrate_snapshot(external, self.task)

    def test_wrong_dtype_shape_nonfinite_or_file_format_is_rejected(self):
        external = externalize_snapshot(self.inline, self.task)
        for matrix, error in [(np.zeros((2, 3), dtype=np.float32), "dtype"),
                              (np.zeros((2, 4)), "shape"),
                              (np.full((2, 3), np.nan), "non-finite")]:
            with self.subTest(error=error):
                changed = self.write_matrix(external, matrix)
                with self.assertRaisesRegex(ValueError, error):
                    hydrate_snapshot(changed, self.task)
        changed = copy.deepcopy(external)
        payload = b"this is not an NPY file"
        checksum = hashlib.sha256(payload).hexdigest()
        relative = f"vectors/master_visual_means.{checksum}.npy"
        (self.task / relative).write_bytes(payload)
        changed["vector_storage"].update(path=relative, sha256=checksum)
        with self.assertRaisesRegex(ValueError, "NPY"):
            hydrate_snapshot(changed, self.task)

    def test_wrong_or_duplicate_row_references_are_rejected(self):
        external = externalize_snapshot(self.inline, self.task)
        for reference in ({"row": -1}, {"row": 2}, {"row": True}, {"row": 0}, {},
                          {"row": 1, "extra": 0}):
            changed = copy.deepcopy(external)
            changed["masters"][1]["visual_mean_ref"] = reference
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "rows"):
                hydrate_snapshot(changed, self.task)
        changed = copy.deepcopy(external)
        changed["masters"][0]["visual_mean"] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, "no inline"):
            hydrate_snapshot(changed, self.task)

    def test_invalid_metadata_and_unsafe_paths_are_rejected(self):
        external = externalize_snapshot(self.inline, self.task)
        for key, value in [("path", "../outside.npy"), ("path", "/tmp/outside.npy"),
                           ("path", "vectors/../outside.npy"), ("path", "vectors/x.npy"),
                           ("dtype", "float32"), ("shape", [2, True]),
                           ("shape", [3, 3]), ("sha256", "bad"), ("format", "pickle")]:
            changed = copy.deepcopy(external)
            changed["vector_storage"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                hydrate_snapshot(changed, self.task)

    def test_symlink_outside_task_is_rejected_for_read_and_write(self):
        external = externalize_snapshot(self.inline, self.task)
        outside = self.root / "outside"
        shutil.move(self.task / "vectors", outside)
        (self.task / "vectors").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            hydrate_snapshot(external, self.task)
        with self.assertRaisesRegex(ValueError, "escapes"):
            externalize_snapshot(self.inline, self.task)

    def test_generations_survive_partial_commits_then_cleanup_only_unreferenced(self):
        first = externalize_snapshot(self.inline, self.task)
        insert(self.graph, "next", [1, 0.1, 0])
        second = externalize_snapshot(self.graph.to_dict(), self.task)
        first_path = self.task / first["vector_storage"]["path"]
        second_path = self.task / second["vector_storage"]["path"]
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(hydrate_snapshot(first, self.task), self.inline)
        # A crash between checkpoint and graph writes preserves both generations.
        self.write_graph(first)
        self.write_graph({"graph": second}, "checkpoint.json")
        unrelated = self.task / "vectors" / "unrelated.npy"
        unrelated.write_bytes(b"keep")
        self.assertEqual(cleanup_unreferenced_vectors(self.task), [])
        self.assertTrue(first_path.exists() and second_path.exists())
        self.write_graph(second)
        self.assertEqual(cleanup_unreferenced_vectors(self.task), [first_path])
        self.assertTrue(second_path.exists() and unrelated.exists())
        self.assertEqual(load_graph(self.task / "graph.json").to_dict(), self.graph.to_dict())

    def test_cleanup_needs_valid_json_references_and_never_deletes_first(self):
        first = externalize_snapshot(self.inline, self.task)
        self.assertEqual(cleanup_unreferenced_vectors(self.task), [])
        self.write_graph(first)
        (self.task / "checkpoint.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            cleanup_unreferenced_vectors(self.task)
        self.assertTrue((self.task / first["vector_storage"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
