"""Offline input/provenance/resume tests; vectors are synthetic, never CLIP quality results."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.graph import build_visual_embedding_graph as builder
from scripts.graph import visual_embedding_storage as storage


class BuildVisualEmbeddingGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.natural = self.root / "natural"
        self.natural.mkdir()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type": "clip"}')
        self.task = "Fixture"
        self.source = self.root / (self.task + "_Open.json")
        self.images, self.captions, self.paths, self.vectors = [], [], [], {}
        for index, angle in enumerate((0, 20, 180, 175)):
            path = self.root / f"image{index}.ppm"
            # Real, tiny PPM images keep this test independent of ML/Pillow packages.
            path.write_text(f"P3\n1 1\n255\n{index * 40} 50 80\n")
            image_id = builder.sha(path)
            self.paths.append(str(path))
            self.images.append({"id": image_id, "path": str(path),
                                "source_paths": [f"fixture/image{index}.ppm"]})
            self.captions.append({"id": image_id,
                                  "caption": f"完整 caption {index}.\nA full detail remains. "})
            radians = math.radians(angle)
            self.vectors[str(path)] = [math.cos(radians), math.sin(radians)]
        # Deliberately nonlexical rounds ensure original dialogue ordering is used.
        self.turns = [
            {"round": "S1:R2", "user": "第一轮 user", "assistant": "原始 answer",
             "input_image": ["fixture/image0.ppm", "fixture/image1.ppm"],
             "gold_answer": "must not enter the graph"},
            {"round": "S1:R9", "user": "text only", "assistant": "not a child"},
            {"round": "S1:R1", "user": "second user", "assistant": "second answer",
             "input_image": "fixture/image2.ppm"},
            {"round": "S1:R3", "user": "third user", "assistant": "third answer",
             "input_image": ["fixture/image3.ppm"]},
        ]
        self.write_json(self.source, {"multi_session_dialogues": [
            {"date": "2026-10-01 10:20:30", "dialogues": self.turns}
        ]})
        self.memories = [
            {"id": f"{self.task}/{round_id}", "task": self.task,
             "date": "2026-10-01 10:20:30", "images": [self.images[i]["id"] for i in indices],
             "text": "derived text must not replace the original dialogue"}
            for round_id, indices in (("S1:R2", [0, 1]), ("S1:R1", [2]), ("S1:R3", [3]))
        ]
        self.config = {
            "source_files": [{"path": str(self.source), "sha256": builder.sha(self.source)}],
            "provenance": {"caption_model": "fixture-only"},
            "caption_prompt": "unchanged full caption prompt", "cue_prompt": "cue",
            "image_max_pixels": 16, "caption_max_new_tokens": 100, "cue_max_new_tokens": 10,
            "seed": 42, "do_sample": False, "dtype": "float32",
        }
        signature = builder.caption_signature(self.config)
        for caption in self.captions:
            caption["protocol_signature"] = signature
        self.write_json(self.natural / "protocol.json", self.config)
        self.write_jsonl(self.natural / "images.jsonl", self.images)
        self.write_jsonl(self.natural / "memories.jsonl", self.memories)
        self.write_jsonl(self.natural / "captions.jsonl", self.captions)
        self.encoded, self.factory_calls = [], []
        self.metadata = {"encoder": "synthetic-test-vectors", "dimension": 2}

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")

    @staticmethod
    def read_json(path):
        return json.loads(path.read_text())

    def args(self, output="graph", **overrides):
        args = dict(natural_root=self.natural, graph_root=self.root / output, task=None,
                    from_start=False, limit=0, visual_encoder=str(self.model),
                    visual_encoder_revision=None, visual_device="cpu", visual_local_files_only=True)
        args.update(overrides)
        return SimpleNamespace(**args)

    def factory(self, model, **kwargs):
        self.factory_calls.append((model, kwargs))

        def encode(path):
            self.encoded.append(path)
            return list(self.vectors[path])

        return SimpleNamespace(metadata=deepcopy(self.metadata), encode=encode)

    def build(self, args, factory=None):
        with redirect_stdout(io.StringIO()):
            return builder.build(args, encoder_factory=factory or self.factory)

    def add_task(self, task="Other"):
        source = self.root / (task + "_Open.json")
        source.write_bytes(self.source.read_bytes())
        self.config["source_files"].append({"path": str(source), "sha256": builder.sha(source)})
        self.write_json(self.natural / "protocol.json", self.config)
        extra = [{**row, "task": task, "id": row["id"].replace(self.task + "/", task + "/")}
                 for row in self.memories]
        self.write_jsonl(self.natural / "memories.jsonl", self.memories + extra)

    def assert_external_snapshot(self, snapshot, task_dir):
        metadata = snapshot["vector_storage"]
        self.assertEqual(metadata["format"], "npy")
        self.assertEqual(metadata["dtype"], "float64")
        self.assertEqual(metadata["shape"], [len(snapshot["masters"]), 2])
        vector_path = task_dir / metadata["path"]
        self.assertEqual(vector_path.parent, task_dir / "vectors")
        self.assertEqual(builder.sha(vector_path), metadata["sha256"])
        for row, master in enumerate(snapshot["masters"]):
            self.assertNotIn("visual_mean", master)
            self.assertEqual(master["visual_mean_ref"], {"row": row})
        self.assertNotIn('"visual_mean":', json.dumps(snapshot))
        return vector_path

    def expected_graph(self):
        graph = builder.VisualEmbeddingGraph()
        records, _ = builder.prepare_inputs(self.natural)
        for record in records[self.task]:
            graph.insert(record["id"], self.vectors[record["image_path"]],
                         dialogue=record["dialogue"], caption=record["caption"],
                         timestamp=record["timestamp"], image_path=record["image_path"])
        return graph.to_dict()

    def test_graph_and_checkpoint_reference_one_lossless_master_vector_file(self):
        self.build(self.args())
        task_dir = self.root / "graph" / self.task
        snapshot = self.read_json(task_dir / "graph.json")
        state = self.read_json(task_dir / "checkpoint.json")
        vector_path = self.assert_external_snapshot(snapshot, task_dir)
        self.assertEqual(self.assert_external_snapshot(state["graph"], task_dir), vector_path)
        self.assertEqual(snapshot, state["graph"])
        self.assertEqual(storage.hydrate_snapshot(snapshot, task_dir), self.expected_graph())
        self.assertEqual(storage.load_graph(task_dir / "graph.json").to_dict(),
                         self.expected_graph())
        self.assertEqual(list((task_dir / "vectors").iterdir()), [vector_path])

    def test_missing_or_corrupt_current_vectors_fail_before_encoder_load(self):
        for corruption in ("missing", "modified"):
            with self.subTest(corruption=corruption):
                args = self.args(output=corruption, limit=1)
                self.build(args)
                task_dir = args.graph_root / self.task
                checkpoint = task_dir / "checkpoint.json"
                committed = checkpoint.read_bytes()
                state = self.read_json(checkpoint)
                vector_path = task_dir / state["graph"]["vector_storage"]["path"]
                if corruption == "missing":
                    vector_path.unlink()
                else:
                    payload = bytearray(vector_path.read_bytes())
                    payload[-1] ^= 1
                    vector_path.write_bytes(payload)
                calls, encoded = len(self.factory_calls), list(self.encoded)
                args.limit = 0
                with self.assertRaises((FileNotFoundError, ValueError, OSError)):
                    self.build(args)
                self.assertEqual(len(self.factory_calls), calls)
                self.assertEqual(self.encoded, encoded)
                self.assertEqual(checkpoint.read_bytes(), committed)

    def test_failures_before_checkpoint_commit_preserve_resumable_previous_vectors(self):
        for failure in ("vector_write", "checkpoint_write"):
            with self.subTest(failure=failure):
                args = self.args(output=failure, limit=1)
                self.build(args)
                task_dir = args.graph_root / self.task
                checkpoint = task_dir / "checkpoint.json"
                committed = checkpoint.read_bytes()
                snapshot = self.read_json(task_dir / "graph.json")
                vector_path = task_dir / snapshot["vector_storage"]["path"]
                real_write = builder.write_json

                def fail_checkpoint(path, value):
                    if path.name == "checkpoint.json":
                        raise OSError("simulated checkpoint write interruption")
                    return real_write(path, value)

                if failure == "vector_write":
                    interruption = patch.object(builder, "externalize_snapshot",
                                                side_effect=OSError("simulated vector write interruption"))
                else:
                    interruption = patch.object(builder, "write_json", side_effect=fail_checkpoint)
                with interruption, self.assertRaisesRegex(OSError, "simulated"):
                    self.build(args)
                self.assertEqual(checkpoint.read_bytes(), committed)
                self.assertTrue(vector_path.is_file())
                self.assertEqual(len(storage.load_graph(task_dir / "graph.json").image_nodes), 1)
                # The failed second image was not committed and must be encoded again.
                encoded_before = len(self.encoded)
                args.limit = 0
                self.assertTrue(self.build(args)["complete"])
                self.assertEqual(self.encoded[encoded_before:], self.paths[1:])
                self.assertEqual(storage.load_graph(task_dir / "graph.json").to_dict(),
                                 self.expected_graph())

    def test_export_crash_retains_both_referenced_vector_files_until_resume(self):
        self.build(self.args(limit=1))
        task_dir = self.root / "graph" / self.task
        old_snapshot = self.read_json(task_dir / "graph.json")
        old_vector = task_dir / old_snapshot["vector_storage"]["path"]
        real_write = builder.write_json

        def fail_graph_export(path, value):
            if path.name == "graph.json":
                raise OSError("simulated graph export interruption")
            return real_write(path, value)

        with patch.object(builder, "write_json", side_effect=fail_graph_export):
            with self.assertRaisesRegex(OSError, "simulated graph export"):
                self.build(self.args(limit=1))
        state = self.read_json(task_dir / "checkpoint.json")
        current_vector = task_dir / state["graph"]["vector_storage"]["path"]
        self.assertNotEqual(old_vector, current_vector)
        storage.cleanup_unreferenced_vectors(task_dir)
        self.assertTrue(old_vector.is_file())
        self.assertTrue(current_vector.is_file())
        self.assertEqual(storage.hydrate_snapshot(old_snapshot, task_dir)["image_nodes"],
                         state["graph"]["image_nodes"][:1])
        self.assertEqual(self.encoded, self.paths[:2])
        self.assertTrue(self.build(self.args())["complete"])
        self.assertEqual(self.encoded, self.paths)
        final_snapshot = self.read_json(task_dir / "graph.json")
        final_vector = task_dir / final_snapshot["vector_storage"]["path"]
        self.assertEqual(set((task_dir / "vectors").iterdir()), {final_vector})

    def test_cleanup_only_removes_unreferenced_master_vector_blobs(self):
        self.build(self.args())
        task_dir = self.root / "graph" / self.task
        snapshot = self.read_json(task_dir / "graph.json")
        current_vector = task_dir / snapshot["vector_storage"]["path"]
        orphan = task_dir / "vectors" / ("master_visual_means." + "a" * 64 + ".npy")
        orphan.write_bytes(b"uncommitted orphan")
        unrelated = task_dir / "vectors" / "other_experiment.npy"
        unrelated.write_bytes(b"unrelated data must remain")
        storage.cleanup_unreferenced_vectors(task_dir)
        self.assertFalse(orphan.exists())
        self.assertTrue(current_vector.exists())
        self.assertEqual(unrelated.read_bytes(), b"unrelated data must remain")

    def test_inline_checkpoint_with_current_protocol_resumes_and_externalizes(self):
        self.build(self.args(limit=2))
        task_dir = self.root / "graph" / self.task
        checkpoint = task_dir / "checkpoint.json"
        state = self.read_json(checkpoint)
        state["graph"] = storage.hydrate_snapshot(state["graph"], task_dir)
        # Exercise the inline loader while retaining the current, pinned protocol;
        # this does not authorize arbitrary old-code runs to resume.
        self.write_json(checkpoint, state)
        self.write_json(task_dir / "graph.json", state["graph"])
        for path in (task_dir / "vectors").iterdir():
            path.unlink()
        self.assertTrue(self.build(self.args())["complete"])
        self.assertEqual(self.encoded, self.paths)
        snapshot = self.read_json(task_dir / "graph.json")
        self.assert_external_snapshot(snapshot, task_dir)
        self.assertEqual(storage.load_graph(task_dir / "graph.json").to_dict(),
                         self.expected_graph())

    def test_prepare_preserves_source_dialogue_full_caption_date_and_content_hash(self):
        by_task, hashes = builder.prepare_inputs(self.natural)
        records = by_task[self.task]
        self.assertEqual([record["id"] for record in records], [
            "Fixture/S1:R2#image1", "Fixture/S1:R2#image2",
            "Fixture/S1:R1#image1", "Fixture/S1:R3#image1",
        ])
        self.assertEqual(len(hashes), 4)
        for index, record in enumerate(records):
            source_turn = self.turns[0 if index < 2 else index]
            self.assertEqual(record["dialogue"],
                             {role: source_turn[role] for role in ("user", "assistant")})
            self.assertEqual(record["caption"], self.captions[index]["caption"])
            self.assertEqual(record["timestamp"], "2026-10-01 10:20:30")
            self.assertEqual(record["source_image_id"], builder.sha(record["image_path"]))
            self.assertNotIn("gold_answer", record)
        self.assertNotIn("must not enter", json.dumps(by_task))

    def test_partial_resume_matches_one_shot_without_reencoding_completed_images(self):
        partial = self.build(self.args(limit=2))
        self.assertFalse(partial["complete"])
        self.assertEqual(self.encoded, self.paths[:2])
        complete = self.build(self.args())
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["tasks"][self.task]["new_images"], 2)
        self.assertEqual(self.encoded, self.paths)
        self.assertEqual(len(self.factory_calls), 2)
        self.build(self.args(output="one-shot"))
        for name in ("graph.json", "checkpoint.json", "write_log.jsonl"):
            self.assertEqual((self.root / "graph" / self.task / name).read_bytes(),
                             (self.root / "one-shot" / self.task / name).read_bytes())
        graph = self.read_json(self.root / "graph" / self.task / "graph.json")
        self.assertEqual(len(graph["image_nodes"]), 4)
        self.assertEqual(len(graph["masters"]), 2)
        self.assertTrue(all(set(node) == {
            "id", "image_id", "image_path", "dialogue", "caption", "timestamp"
        } for node in graph["image_nodes"]))

    def test_complete_resume_regenerates_missing_or_stale_exports_without_loading_encoder(self):
        self.build(self.args())
        task_dir = self.root / "graph" / self.task
        expected_graph = (task_dir / "graph.json").read_bytes()
        expected_log = (task_dir / "write_log.jsonl").read_bytes()
        (task_dir / "graph.json").write_text("truncated")
        (task_dir / "write_log.jsonl").unlink()

        def forbidden_factory(*args, **kwargs):
            self.fail("a complete checkpoint must not load the image encoder")

        result = self.build(self.args(), forbidden_factory)
        self.assertEqual(result["tasks"][self.task]["new_images"], 0)
        self.assertEqual((task_dir / "graph.json").read_bytes(), expected_graph)
        self.assertEqual((task_dir / "write_log.jsonl").read_bytes(), expected_log)

    def test_crash_after_checkpoint_commit_resumes_without_reencoding_committed_image(self):
        real_write = builder.write_json

        def fail_first_export(path, value):
            if path.name == "graph.json":
                raise OSError("simulated interruption after checkpoint commit")
            return real_write(path, value)

        with patch.object(builder, "write_json", side_effect=fail_first_export):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                self.build(self.args())
        state = self.read_json(self.root / "graph" / self.task / "checkpoint.json")
        self.assertEqual(len(state["graph"]["image_nodes"]), 1)
        self.assertEqual(self.encoded, self.paths[:1])
        self.assertTrue(self.build(self.args())["complete"])
        self.assertEqual(self.encoded, self.paths)

    def test_encoder_metadata_drift_rejected_before_next_image_is_encoded(self):
        self.build(self.args(limit=1))
        self.metadata["preprocessing_version"] = "changed"
        with self.assertRaisesRegex(ValueError, "loaded encoder differs"):
            self.build(self.args())
        self.assertEqual(self.encoded, self.paths[:1])

    def test_remote_encoder_drift_between_completed_and_new_tasks_is_rejected(self):
        self.add_task()
        self.metadata["resolved_commit"] = "old-revision"
        self.build(self.args(task=self.task, visual_encoder="fixture/remote-clip"))
        original = (self.root / "graph" / self.task / "checkpoint.json").read_bytes()
        self.metadata["resolved_commit"] = "new-revision"
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.build(self.args(task="Other", visual_encoder="fixture/remote-clip"))
        self.assertEqual(self.encoded, self.paths)
        self.assertEqual((self.root / "graph" / self.task / "checkpoint.json").read_bytes(), original)
        self.assertFalse((self.root / "graph" / "Other" / "checkpoint.json").exists())

    def test_completed_checkpoint_restores_missing_global_encoder_pin_without_model_load(self):
        self.add_task()
        self.metadata["resolved_commit"] = "original-revision"
        self.build(self.args(task=self.task, visual_encoder="fixture/remote-clip"))
        pin = self.root / "graph" / "encoder_metadata.json"
        pin.unlink(missing_ok=True)

        def forbidden_factory(*args, **kwargs):
            self.fail("recovering metadata from complete checkpoints must not load the encoder")

        self.build(self.args(task=self.task, visual_encoder="fixture/remote-clip"), forbidden_factory)
        self.assertTrue(pin.is_file())
        # The recovered identity must govern a task that has never been started.
        self.metadata["resolved_commit"] = "different-revision"
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.build(self.args(task="Other", visual_encoder="fixture/remote-clip"))
        self.assertEqual(self.encoded, self.paths)

    def test_completed_unselected_checkpoint_must_match_global_encoder_pin(self):
        self.add_task()
        self.metadata["resolved_commit"] = "original-revision"
        self.build(self.args(visual_encoder="fixture/remote-clip"))
        state_path = self.root / "graph" / "Other" / "checkpoint.json"
        state = self.read_json(state_path)
        state["encoder_metadata"]["resolved_commit"] = "different-revision"
        self.write_json(state_path, state)
        # Selecting only the clean, complete task must still expose mixed provenance.
        factory_calls = len(self.factory_calls)
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.build(self.args(task=self.task, visual_encoder="fixture/remote-clip"))
        self.assertEqual(len(self.factory_calls), factory_calls)

    def test_missing_global_pin_does_not_accept_conflicting_completed_task_encoders(self):
        self.add_task()
        self.metadata["resolved_commit"] = "original-revision"
        self.build(self.args(visual_encoder="fixture/remote-clip"))
        (self.root / "graph" / "encoder_metadata.json").unlink(missing_ok=True)
        state_path = self.root / "graph" / "Other" / "checkpoint.json"
        state = self.read_json(state_path)
        state["encoder_metadata"]["resolved_commit"] = "different-revision"
        self.write_json(state_path, state)
        factory_calls = len(self.factory_calls)
        with self.assertRaisesRegex(ValueError, "encoder"):
            self.build(self.args(task=self.task, visual_encoder="fixture/remote-clip"))
        self.assertEqual(len(self.factory_calls), factory_calls)

    def test_local_model_changes_rejected_before_loading_encoder(self):
        self.build(self.args(limit=1))
        (self.model / "config.json").write_text('{"model_type": "clip", "changed": true}')
        with self.assertRaisesRegex(ValueError, "sources/encoder/code changed"):
            self.build(self.args())
        self.assertEqual(len(self.factory_calls), 1)

    def test_code_identity_change_rejected_without_mutating_checkpoint(self):
        self.build(self.args(limit=1))
        checkpoint = self.root / "graph" / self.task / "checkpoint.json"
        before = checkpoint.read_bytes()
        original_sha = builder.sha

        def changed_code_sha(path):
            return "changed-code-fingerprint" if Path(path) == Path(builder.__file__) else original_sha(path)

        with patch.object(builder, "sha", side_effect=changed_code_sha):
            with self.assertRaisesRegex(ValueError, "sources/encoder/code changed"):
                self.build(self.args())
        self.assertEqual(checkpoint.read_bytes(), before)

    def test_changed_caption_cache_refuses_resume_even_when_signature_is_valid(self):
        self.build(self.args(limit=1))
        self.captions[0]["caption"] += "new content"
        self.write_jsonl(self.natural / "captions.jsonl", self.captions)
        with self.assertRaisesRegex(ValueError, "sources/encoder/code changed"):
            self.build(self.args())

    def test_pinned_dialogue_source_change_and_image_content_change_are_rejected(self):
        original_source = self.source.read_bytes()
        self.source.write_bytes(original_source + b"\n")
        with self.assertRaisesRegex(ValueError, "dialogue source changed"):
            builder.prepare_inputs(self.natural)
        self.source.write_bytes(original_source)
        Path(self.paths[0]).write_text("P3\n1 1\n255\n255 255 255\n")
        with self.assertRaisesRegex(ValueError, "image content hash changed"):
            builder.prepare_inputs(self.natural)

    def test_duplicate_manifest_ids_and_incompatible_caption_signatures_are_rejected(self):
        for filename, rows, label in (
            ("images.jsonl", self.images, "image ID"),
            ("memories.jsonl", self.memories, "memory ID"),
            ("captions.jsonl", self.captions, "caption ID"),
        ):
            with self.subTest(filename=filename):
                self.write_jsonl(self.natural / filename, [*rows, rows[0]])
                with self.assertRaisesRegex(ValueError, "duplicate " + label):
                    builder.prepare_inputs(self.natural)
                self.write_jsonl(self.natural / filename, rows)
        self.captions[0]["protocol_signature"] = "different-protocol"
        self.write_jsonl(self.natural / "captions.jsonl", self.captions)
        with self.assertRaisesRegex(ValueError, "incompatible cached Qwen caption"):
            builder.prepare_inputs(self.natural)

    def test_original_date_and_order_are_required(self):
        self.memories[0]["date"] = "2026-10-02"
        self.write_jsonl(self.natural / "memories.jsonl", self.memories)
        with self.assertRaisesRegex(ValueError, "date disagrees"):
            builder.prepare_inputs(self.natural)
        self.memories[0]["date"] = "2026-10-01 10:20:30"
        self.write_jsonl(self.natural / "memories.jsonl", list(reversed(self.memories)))
        with self.assertRaisesRegex(ValueError, "order differs"):
            builder.prepare_inputs(self.natural)

    def test_visual_graph_is_offline_only_and_cli_dispatches_its_encoder_options(self):
        from scripts import visual_memory_pipeline as pipeline

        argv = ["pipeline", "offline", "--memory", "visual_embedding_graph", "--task", self.task,
                "--visual-encoder", str(self.model), "--visual-encoder-revision", "pinned",
                "--visual-device", "cpu", "--visual-local-files-only", "--limit", "2"]
        with patch("sys.argv", argv), patch.object(pipeline, "offline") as offline:
            pipeline.main()
        args = offline.call_args.args[0]
        self.assertEqual(args.visual_encoder, str(self.model))
        self.assertEqual(args.visual_encoder_revision, "pinned")
        self.assertTrue(args.visual_local_files_only)
        self.assertEqual(args.graph_root, pipeline.ROOT / "results/visual_embedding_graph")
        with patch("sys.argv", ["pipeline", "online", "--mode", "visual_embedding_graph"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                pipeline.main()
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
