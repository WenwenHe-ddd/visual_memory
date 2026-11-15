"""Random-2D entry boundaries and compressed-memory invariants, without models.

The encoder stand-ins below return small deterministic fixtures. They never
decode images, load weights, use CUDA, or alter existing experiment artifacts.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import visual_memory_pipeline as pipeline
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.memory import caption_visual_master as base
from scripts.memory import caption_visual_master_random_2d as random_2d


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member_sets(memory):
    return {frozenset(master["member_ids"]) for master in memory.masters.values()}


class RandomBasisCliTests(unittest.TestCase):
    def test_default_and_explicit_pca_keep_existing_dispatch_and_output(self):
        for flags in ([], ["--visual-basis", "pca-2d"]):
            with self.subTest(flags=flags), patch.object(base, "offline") as pca, \
                    patch.object(random_2d, "offline") as random:
                pipeline.main(["offline", "--memory", "caption_visual_master", *flags])
                pca.assert_called_once()
                random.assert_not_called()
                args = pca.call_args.args[0]
                self.assertEqual(args.visual_basis, "pca-2d")
                self.assertIsNone(args.basis_seed)
                self.assertEqual(args.device, "cuda:0")
                self.assertEqual(args.output,
                                 pipeline.ROOT / "results/debug_caption_master_answer/offline")
                self.assertEqual(args.basis, base.PILOT / "memory/basis.npy")

    def test_random_dispatch_has_separate_seed_specific_output(self):
        for seed, flags in ((0, []), (7, ["--basis-seed", "7"])):
            with self.subTest(seed=seed), patch.object(base, "offline") as pca, \
                    patch.object(random_2d, "offline") as random:
                pipeline.main(["offline", "--memory", "caption_visual_master",
                               "--visual-basis", "random-2d", *flags])
                pca.assert_not_called()
                random.assert_called_once()
                args = random.call_args.args[0]
                self.assertEqual(args.basis_seed, seed)
                self.assertEqual(args.device, "cuda:0")
                self.assertEqual(args.output, pipeline.ROOT /
                                 f"results/debug_caption_master_answer_random_2d/seed_{seed}/offline")

    def test_explicit_random_paths_and_inputs_are_preserved(self):
        with patch.object(random_2d, "offline") as run:
            pipeline.main([
                "offline", "--memory", "caption_visual_master", "--visual-basis", "random-2d",
                "--basis-seed", "19", "--device", "cpu", "--output", "/tmp/random-new-run",
                "--image-files", "D7_IMG_005.jpg", "D7_IMG_006.jpg",
            ])
        args = run.call_args.args[0]
        self.assertEqual(args.output, Path("/tmp/random-new-run"))
        self.assertEqual(args.image_files, ["D7_IMG_005.jpg", "D7_IMG_006.jpg"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.basis_seed, 19)

    def test_incompatible_flags_fail_before_any_work(self):
        cases = [
            ["--visual-basis", "pca-2d", "--basis-seed", "0"],
            ["--visual-basis", "random-2d", "--basis-seed", "-1"],
            ["--visual-basis", "random-2d", "--basis", "/tmp/custom.npy"],
            ["--visual-basis", "random-2d", "--dataset", "mem_gallery"],
            ["--visual-basis", "random-2d", "--memory-source", "/tmp/source",
             "--output", "/tmp/export"],
        ]
        for flags in cases:
            with self.subTest(flags=flags), patch.object(pipeline, "offline") as run, \
                    patch("scripts.memory.memory_store.export_store") as export:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    pipeline.main(["offline", "--memory", "caption_visual_master", *flags])
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()
                export.assert_not_called()
        for method in ("natural", "visual_embedding_graph", "hippomm"):
            with self.subTest(method=method), patch.object(pipeline, "offline") as run, \
                    patch("scripts.memory.hippomm.pipeline") as other:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    pipeline.main(["offline", "--memory", method,
                                   "--visual-basis", "random-2d"])
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()
                other.assert_not_called()

    def test_online_consumes_explicit_random_memory_through_unchanged_retriever(self):
        with patch("scripts.memory.online_retrieval.online") as run:
            pipeline.main([
                "online", "--memory", "caption_visual_master", "--retrieval", "multi_path",
                "--online-weighting", "none", "--memory-dir", "/tmp/random-run/offline/memory",
                "--output", "/tmp/random-run/online", "--question-ids", "Q10",
            ])
        run.assert_called_once()
        args = run.call_args.args[0]
        self.assertEqual(args.memory_dir, Path("/tmp/random-run/offline/memory"))
        self.assertEqual(args.output, Path("/tmp/random-run/online"))
        self.assertEqual(args.question_ids, ["Q10"])
        self.assertEqual(args.online_weighting, "none")


class RandomBasisMathTests(unittest.TestCase):
    def test_invalid_dimension_or_seed_is_rejected(self):
        for dimension, seed in ((1, 0), (0, 0), (512, -1), (512, 1.5), (True, 0), (512, True)):
            with self.subTest(dimension=dimension, seed=seed), self.assertRaises(ValueError):
                random_2d.make_random_basis(dimension, seed)

    def test_orthonormal_seeded_basis_is_reproducible_without_global_rng_changes(self):
        state = np.random.get_state()
        try:
            first = random_2d.make_random_basis(512, 0)
            after = np.random.get_state()
            self.assertEqual(state[0], after[0])
            np.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
            np.testing.assert_array_equal(first, random_2d.make_random_basis(512, 0))
            self.assertFalse(np.array_equal(first, random_2d.make_random_basis(512, 1)))
            self.assertEqual(first.shape, (512, 2))
            self.assertEqual(first.dtype, np.float64)
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_allclose(first.T @ first, np.eye(2), atol=1e-13)
        finally:
            np.random.set_state(state)

    def test_changing_basis_preserves_every_caption_partition_and_global_sum(self):
        pca = CaptionWardVisualMemory(np.eye(6)[:, :2])
        random = CaptionWardVisualMemory(random_2d.make_random_basis(6, 23))
        texts = [[1., 0., 0.], [0., 1., 0.], [1., .02, 0.], [.02, 1., 0.],
                 [1., -.02, 0.], [0., 0., 1.], [0., .02, 1.], [-1., 0., 0.]]
        visuals = np.random.default_rng(921).normal(size=(len(texts), 6))
        visuals /= np.linalg.norm(visuals, axis=1, keepdims=True)
        for index, (text, visual) in enumerate(zip(texts, visuals)):
            for memory in (pca, random):
                memory.add(str(index), f"caption {index}", text, visual,
                           dialogue=f"dialogue {index}", timestamp=index)
                np.testing.assert_allclose(
                    sum(master["visual_sum"] for master in memory.masters.values()),
                    visuals[:index + 1].sum(axis=0), atol=1e-12,
                )
                for master in memory.masters.values():
                    coords = np.array([memory.image_nodes[key]["visual_coords"]
                                       for key in master["member_ids"]])
                    np.testing.assert_allclose(memory.basis.T @ master["visual_sum"],
                                               coords.sum(axis=0), atol=1e-12)
            self.assertEqual(member_sets(pca), member_sets(random))
        for node in random.snapshot()["image_nodes"]:
            self.assertEqual(len(node["visual_coords"]), 2)
            self.assertEqual({key for key in node if "visual" in key}, {"visual_coords"})


class RandomBasisOfflineTests(unittest.TestCase):
    def test_active_random_lock_rejects_reentry_and_pca_then_cleans_up_on_success_or_failure(self):
        original_offline = base.offline
        for fail in (False, True):
            with self.subTest(builder_fails=fail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol_path = root / "models.json"
                protocol_path.write_text(json.dumps({"image_encoder": {"embedding_dim": 512}}))
                output = root / "memory-run"
                args = SimpleNamespace(output=output, models_protocol=protocol_path, basis_seed=0)
                lock = output / ".random_2d.lock"
                artifact = output / "builder-artifact.txt"

                def builder(*unused_args, **unused_kwargs):
                    self.assertTrue(lock.is_file())
                    lock_contents = lock.read_bytes()
                    with self.assertRaises(FileExistsError):
                        random_2d.offline(args)
                    self.assertEqual(lock.read_bytes(), lock_contents)
                    with self.assertRaises(FileExistsError):
                        original_offline(args)
                    self.assertEqual(lock.read_bytes(), lock_contents)
                    artifact.write_text("preserved builder output")
                    if fail:
                        raise RuntimeError("synthetic builder failure")
                    return "completed"

                with patch.object(base, "offline", side_effect=builder) as shared, \
                        patch.object(base, "configure_device") as configure, \
                        patch.object(base, "ImageEncoder") as image:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "synthetic builder failure"):
                            random_2d.offline(args)
                    else:
                        self.assertEqual(random_2d.offline(args), "completed")
                    shared.assert_called_once()
                    configure.assert_not_called()
                    image.assert_not_called()
                self.assertFalse(lock.exists())
                self.assertEqual(artifact.read_text(), "preserved builder output")

    def test_pca_refuses_to_overwrite_existing_random_memory_before_device_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protocol = output / "protocol.json"
            protocol.write_text(json.dumps({"basis_kind": "random-2d", "basis_seed": 0}))
            original = protocol.read_bytes()
            with patch.object(base, "configure_device") as configure:
                with self.assertRaises(FileExistsError):
                    base.offline(SimpleNamespace(output=output))
                configure.assert_not_called()
            self.assertEqual(protocol.read_bytes(), original)

    def test_nonempty_output_refused_before_encoders_or_shared_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = output / "existing-result.json"
            existing.write_bytes(b"preserve this result")
            args = SimpleNamespace(output=output, basis_seed=0)
            with patch.object(base, "offline") as shared, patch.object(base, "ImageEncoder") as image:
                with self.assertRaises((ValueError, FileExistsError)):
                    random_2d.offline(args)
                shared.assert_not_called()
                image.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"preserve this result")
            self.assertEqual(list(output.iterdir()), [existing])

    def test_mocked_offline_build_never_reads_pca_basis_and_keeps_two_coordinate_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "random"
            visual_vectors = np.random.default_rng(92).normal(size=(4, 512))
            visual_vectors /= np.linalg.norm(visual_vectors, axis=1, keepdims=True)
            texts = np.array([[1., 0., 0.], [1., .01, 0.], [0., 1., 0.], [.01, 1., 0.]])
            rows = []
            for index in range(4):
                image_path = source / f"image_{index}.fixture"
                image_path.write_bytes(f"synthetic source {index}".encode())
                rows.append({"image_id": f"HOME_S1:R{index}#image1", "memory_id": f"HOME_S1:R{index}",
                             "filename": f"image_{index}.jpg", "source_path": str(image_path),
                             "image_sha256": sha_file(image_path), "caption": f"caption {index}",
                             "dialogue": f"dialogue {index}", "timestamp": index})
            (source / "inputs.json").write_text(json.dumps({"leaves": rows}))
            manifest = source / "input_manifest.json"
            manifest.write_text(json.dumps({"home": rows}))
            visual_meta = {"model_name": "mock-clip", "revision": "fixture", "embedding_dim": 512,
                           "device": "cpu"}
            protocol_path = source / "protocol.json"
            protocol_path.write_text(json.dumps({"image_encoder": visual_meta,
                                                 "text_encoder": {"model_path": "mock-bge"},
                                                 "query_prefix": "fixture prefix: "}))
            args = SimpleNamespace(output=output, models_protocol=protocol_path, input_manifest=manifest,
                                   image_files=[row["filename"] for row in rows], device="cpu",
                                   basis_seed=3, visual_basis="random-2d",
                                   basis=root / "does-not-exist-pca.npy")
            fake_image = SimpleNamespace(metadata=visual_meta, device="cpu")
            visual_by_path = {row["source_path"]: vector for row, vector in zip(rows, visual_vectors)}
            fake_image.encode = lambda path: visual_by_path[str(path)].copy()
            fake_text = SimpleNamespace(device="cpu")
            text_by_caption = {row["caption"]: vector for row, vector in zip(rows, texts)}
            with patch.object(base, "FROZEN", source), \
                    patch.object(base, "configure_device"), \
                    patch.object(base, "ImageEncoder", return_value=fake_image), \
                    patch.object(base, "load_text_encoder", return_value=fake_text), \
                    patch.object(base, "encode_text", side_effect=lambda model, captions:
                                 np.array([text_by_caption[caption] for caption in captions])), \
                    redirect_stdout(io.StringIO()):
                random_2d.offline(args)
            basis_path = output / "memory/basis.npy"
            basis = np.load(basis_path, allow_pickle=False)
            np.testing.assert_array_equal(basis, random_2d.make_random_basis(512, 3))
            graph = json.loads((output / "memory/graph.json").read_text())
            sums = np.load(output / "memory/master_sums.npy", allow_pickle=False)
            protocol = json.loads((output / "protocol.json").read_text())
            self.assertEqual(protocol["basis_kind"], "random-2d")
            self.assertEqual(protocol["basis_source"], "generated:gaussian_qr")
            self.assertEqual(protocol["basis_seed"], 3)
            self.assertEqual(protocol["external_calibration_images"], 0)
            self.assertTrue(protocol["basis_generation"]["fixed_throughout_arrivals"])
            self.assertEqual(protocol["basis_sha256"], sha_file(basis_path))
            self.assertFalse(protocol["per_image_visual_embeddings_saved"])
            self.assertEqual(protocol["old_image_reencodes"], 0)
            self.assertEqual(protocol["image_copies"], 0)
            self.assertEqual(protocol["insertion_order"], [row["image_id"] for row in rows])
            for name, digest in protocol["memory_sha256"].items():
                self.assertEqual(digest, sha_file(output / "memory" / name))
            self.assertEqual(graph["leaf_visual_dim"], 2)
            for index, node in enumerate(graph["image_nodes"]):
                self.assertEqual(set(node), {"id", "caption", "dialogue", "timestamp",
                                             "visual_coords", "master_id"})
                np.testing.assert_allclose(node["visual_coords"], visual_vectors[index] @ basis,
                                           atol=1e-12)
            np.testing.assert_allclose(sums.sum(axis=0), visual_vectors.sum(axis=0), atol=1e-12)
            self.assertEqual({path.name for path in (output / "memory").iterdir()},
                             {"graph.json", "basis.npy", "master_sums.npy"})
            self.assertFalse(args.basis.exists())


if __name__ == "__main__":
    unittest.main()
