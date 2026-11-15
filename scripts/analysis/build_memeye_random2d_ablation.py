#!/usr/bin/env python3
"""Replay MemEye once into matched PCA-2D and ten Random-2D memories.

Only the arriving image is encoded. Its transient vector is shared across
conditions, and no individual full visual embeddings or original images are
persisted. Caption partitions are computed once per prefix and passed to the
unchanged CaptionWardVisualMemory.add implementation in this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from scripts.analysis.prepare_all_domain_adaptive_inputs import save_memory
from scripts.analysis.probe_caption_visual_online import read_json, sha, write_json
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory, _unit, choose_caption_partition
from scripts.graph.visual_image_encoder import ImageEncoder
from scripts.memory.caption_visual_master_random_2d import make_random_basis

SOURCE = ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
CAPTION = ROOT / "reports/memeye_caption_embedding_pca_20261104"
OUTPUT = ROOT / "reports/memeye_qwen_random2d_ablation_20261105/offline"
HOME = "Home_Renovation_Interior_Design"
SEEDS = list(range(10))
CONDITIONS = ["pca_2d", *[f"random_2d_seed_{seed}" for seed in SEEDS]]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def member_sets(groups):
    return {frozenset(group) for group in groups}


def preflight(source, caption):
    manifest = read_json(source / "manifest.json")
    assert manifest["completed"] and manifest["n_leaves"] == 438 and manifest["n_domains"] == 8
    caption_protocol = read_json(caption / "protocol.json")
    assert caption_protocol["encoding_input"] == "graph.image_nodes[].caption only"
    assert caption_protocol["caption_version"] == "original Qwen3-VL-8B-Instruct"
    assert caption_protocol["dialogue_encoded"] is False
    for path, expected in caption_protocol["source_sha256"].items():
        assert sha(Path(path)) == expected, f"Caption cache source changed: {path}"
    with np.load(caption / "caption_embeddings.npz", allow_pickle=False) as archive:
        ids = archive["image_ids"].tolist()
        matrix = archive["caption_embeddings"].copy()
    assert len(set(ids)) == 438 and matrix.shape == (438, 384) and np.isfinite(matrix).all()
    assert np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1)) < 1e-12
    text_by_id = dict(zip(ids, matrix))
    plotted = read_json(caption / "plot_data.json")
    caption_by_id = {p["image_id"]: p["caption"] for d in plotted["domains"] for p in d["points"]}
    tracked = [source / "manifest.json", caption / "protocol.json",
               caption / "caption_embeddings.npz", caption / "plot_data.json",
               ROOT / "results/natural_observation/protocol.json",
               ROOT / "scripts/graph/caption_ward_visual.py",
               ROOT / "scripts/graph/visual_image_encoder.py",
               ROOT / "scripts/memory/caption_visual_master_random_2d.py",
               ROOT / "scripts/analysis/prepare_all_domain_adaptive_inputs.py",
               Path(__file__).resolve()]
    natural = read_json(ROOT / "results/natural_observation/protocol.json")
    assert "Qwen3-VL-8B-Instruct" in natural["caption_model_path"]
    domains = []
    all_ids = []
    for record in manifest["domains"]:
        name = record["domain"]
        directory = source / name
        for filename, expected in record["sha256"].items():
            path = directory / filename
            assert sha(path) == expected, f"Frozen domain source changed: {path}"
            tracked.append(path)
        graph = read_json(directory / "memory/graph.json")
        rows = read_json(directory / "input_manifest.json")["arrivals"]
        protocol = read_json(directory / "protocol.json")
        assert protocol["text_encoder"] == caption_protocol["text_encoder"]
        assert [row["image_id"] for row in rows] == [node["id"] for node in graph["image_nodes"]]
        assert len(rows) == record["n_leaves"]
        for row, node in zip(rows, graph["image_nodes"]):
            assert row["caption"] == node["caption"] == caption_by_id[row["image_id"]]
            assert row["dialogue"] == node["dialogue"] and row["timestamp"] == node["timestamp"]
        audit_path = (ROOT / "reports/home_caption_ward_visual_20261020/insertion_audit.jsonl"
                      if name == HOME else directory / "insertion_audit.jsonl")
        historical_audits = read_lines(audit_path)
        assert len(historical_audits) == len(rows)
        tracked.append(audit_path)
        texts = np.stack([_unit(text_by_id[row["image_id"]], "cached caption", 384) for row in rows])
        choices = []
        largest_score_delta = 0.0
        for n, historical in enumerate(historical_audits, 1):
            choice = choose_caption_partition(texts[:n])
            groups = [[rows[index]["image_id"] for index in group] for group in choice["partition"]]
            assert groups == historical["clusters_partition"], (name, n, "prefix partition differs")
            assert choice["selected_k"] == historical["selected_k"]
            if choice.get("selected_score") is not None:
                largest_score_delta = max(largest_score_delta,
                    abs(choice["selected_score"] - historical["selected_score"]))
            choices.append(choice)
        final_groups = [[rows[index]["image_id"] for index in group] for group in choices[-1]["partition"]]
        assert member_sets(final_groups) == member_sets(m["member_ids"] for m in graph["masters"])
        domains.append({"name": name, "record": record, "directory": directory,
                        "rows": rows, "graph": graph, "protocol": protocol,
                        "texts": texts, "choices": choices,
                        "historical_audits": historical_audits,
                        "caption_prefix_score_max_abs_delta": largest_score_delta})
        all_ids.extend(row["image_id"] for row in rows)
        print(json.dumps({"stage": "preflight", "domain": name, "prefix_partitions_identical": len(rows),
                          "max_caption_silhouette_delta": largest_score_delta}), flush=True)
    assert set(all_ids) == set(ids) and len(all_ids) == 438
    source_hashes = {str(path.resolve()): sha(path) for path in tracked}
    return domains, source_hashes, caption_protocol


def verify_completed(output, name, frozen_digest):
    completion = output / "domain_completion" / f"{name}.json"
    if not completion.exists():
        return None
    value = read_json(completion)
    assert value["completed"] and value["frozen_input_digest"] == frozen_digest
    for condition in CONDITIONS:
        directory = output / condition / name
        for filename, expected in value["conditions"][condition]["sha256"].items():
            assert sha(directory / filename) == expected, (condition, name, filename)
    return value


def build_domain(domain, output, encoder, source_hashes, frozen_digest, random_bases):
    started = time.perf_counter()
    name, rows = domain["name"], domain["rows"]
    staging = output / "_working" / name
    if staging.exists():
        # This path is exclusively an unpublished scratch directory of this run.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for condition in CONDITIONS:
        if (output / condition / name).exists():
            raise FileExistsError(f"Incomplete published domain requires inspection: {output / condition / name}")
    pca_basis = np.load(domain["directory"] / "memory/basis.npy", allow_pickle=False)
    bases = {"pca_2d": pca_basis, **random_bases}
    memories = {condition: CaptionWardVisualMemory(basis) for condition, basis in bases.items()}
    handles = {}
    state_audits = {c: {"max_global_sum_l2_error": 0.0, "max_projection_l2_error": 0.0,
                         "split_steps": 0, "partial_redistribution_steps": 0} for c in CONDITIONS}
    total_sum = np.zeros(512, dtype=np.float64)
    for condition in CONDITIONS:
        (staging / condition).mkdir()
        handles[condition] = (staging / condition / "insertion_audit.jsonl").open("w")
    image_calls = 0
    try:
        for index, (row, text, choice) in enumerate(zip(rows, domain["texts"], domain["choices"]), 1):
            assert sha(row["source_path"]) == row["sha256"], f"Incoming image changed: {row['image_id']}"
            current = np.asarray(encoder.encode(row["source_path"]), dtype=np.float64)
            image_calls += 1
            # Match the historical stream's pre-normalization, then add()'s normalization.
            current = _unit(current, "arriving visual", 512)
            total_sum += _unit(current, "core visual", 512)
            with patch("scripts.graph.caption_ward_visual.choose_caption_partition", return_value=choice):
                for condition, memory in memories.items():
                    audit = memory.add(row["image_id"], row["caption"], text, current,
                                       dialogue=row["dialogue"], timestamp=row["timestamp"])
                    historical = domain["historical_audits"][index - 1]
                    assert audit["clusters_partition"] == historical["clusters_partition"]
                    assert audit["clusters"] == historical["clusters"], (condition, name, index)
                    err = float(np.linalg.norm(np.sum([m["visual_sum"] for m in memory.masters.values()], axis=0) - total_sum))
                    projection = audit["max_projection_consistency_l2_error"]
                    assert err < 1e-10 and projection < 1e-10, (condition, name, index, err, projection)
                    stats = state_audits[condition]
                    stats["max_global_sum_l2_error"] = max(stats["max_global_sum_l2_error"], err)
                    stats["max_projection_l2_error"] = max(stats["max_projection_l2_error"], projection)
                    stats["split_steps"] += bool(audit["split_master_ids"])
                    stats["partial_redistribution_steps"] += bool(audit["partial_redistribution_master_ids"])
                    audit.update(step=index, filename=row["filename"], caption_partition_shared=True,
                                 caption_partition_matches_historical=True, image_forward_shared_across_conditions=True,
                                 cumulative_global_sum_l2_error=err)
                    handles[condition].write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
            del current
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({"stage": "offline", "domain": name, "arrivals": index,
                                  "total": len(rows), "masters": len(memories["pca_2d"].masters),
                                  "conditions": len(memories)}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    assert image_calls == len(rows)
    frozen_graph = domain["graph"]
    frozen_sums = np.load(domain["directory"] / "memory/master_sums.npy", allow_pickle=False)
    frozen_coords = np.asarray([n["visual_coords"] for n in frozen_graph["image_nodes"]])
    pca = memories["pca_2d"]
    pca_sums = np.stack([m["visual_sum"] for m in pca.masters.values()])
    pca_coords = np.asarray([n["visual_coords"] for n in pca.image_nodes.values()])
    assert list(pca.masters) == [m["id"] for m in frozen_graph["masters"]]
    max_sum_delta = float(np.max(np.abs(pca_sums - frozen_sums)))
    max_coord_delta = float(np.max(np.abs(pca_coords - frozen_coords)))
    # FP32 CUDA versus historical FP32 CPU encoder differences are recorded.
    # A larger discrepancy must stop, rather than silently accepting drift.
    assert max_sum_delta < 1e-4 and max_coord_delta < 1e-5, (name, max_sum_delta, max_coord_delta)
    summary = {"completed": True, "domain": name, "n_leaves": len(rows),
               "n_masters": len(pca.masters), "image_encoder_calls": image_calls,
               "frozen_input_digest": frozen_digest, "conditions": {},
               "all_caption_prefix_partitions_identical": True,
               "caption_prefix_score_max_abs_delta": domain["caption_prefix_score_max_abs_delta"],
               "pca_parity": {"basis_identical": np.array_equal(pca_basis, pca.basis),
                              "all_prefix_memberships_and_master_ids_identical": True,
                              "master_sums_max_abs_delta": max_sum_delta,
                              "leaf_coords_max_abs_delta": max_coord_delta,
                              "historical_encoder_device": "cpu", "replay_encoder_device": str(encoder.device)}}
    for condition, memory in memories.items():
        directory = staging / condition
        graph, sums = save_memory(memory, directory / "memory")
        assert len(graph["image_nodes"]) == len(rows) and sums.shape == frozen_sums.shape
        assert member_sets(m["member_ids"] for m in graph["masters"]) == member_sets(m["member_ids"] for m in frozen_graph["masters"])
        seed = None if condition == "pca_2d" else int(condition.rsplit("_", 1)[1])
        memory_hashes = {f"memory/{name}": sha(directory / "memory" / name)
                         for name in ("graph.json", "basis.npy", "master_sums.npy")}
        protocol = {
            "completed": True, "dataset": "MemEye", "domain": name,
            "purpose": "full_domain_random_2d_basis_ablation", "condition": condition,
            "basis_kind": "pca-2d" if seed is None else "random-2d", "basis_seed": seed,
            "basis_source": str(domain["directory"] / "memory/basis.npy") if seed is None else "generated:gaussian_qr",
            "basis_sha256": memory_hashes["memory/basis.npy"], "basis_fixed_throughout_arrivals": True,
            "basis_recalibrated": False, "external_calibration_images_used_in_this_replay": 0,
            "random_basis_shared_across_domains_for_same_seed": seed is not None,
            "random_basis_generation": None if seed is None else {
                "generator": "numpy.random.Generator(PCG64)", "distribution": "independent standard normal",
                "orthogonalization": "reduced QR; columns signed for nonnegative diagonal of R",
                "shape": [512, 2], "numpy_version": np.__version__},
            "images": len(rows), "masters": len(graph["masters"]),
            "insertion_order": [r["image_id"] for r in rows],
            "caption_model": "Qwen3-VL-8B-Instruct", "caption_input": "unchanged full caption only",
            "caption_cache": str(CAPTION / "caption_embeddings.npz"),
            "caption_partition_policy": "one original Ward-silhouette choice per prefix, shared by unchanged add() calls",
            "all_caption_prefix_partitions_match_historical": True,
            "image_encoder": encoder.metadata, "text_encoder": domain["protocol"]["text_encoder"],
            "query_prefix": domain["protocol"]["query_prefix"],
            "image_forward_policy": "one forward per arrival shared across all 11 conditions; then released",
            "image_encoder_calls_shared": image_calls, "old_image_reencodes": 0, "image_copies": 0,
            "per_image_visual_embeddings_saved": False, "leaf_visual_dimension": 2,
            "qa_used_for_memory_construction": False, "frozen_input_digest": frozen_digest,
            "source_sha256": {k: v for k, v in source_hashes.items() if name in k or "/scripts/" in k},
            "memory_sha256": {Path(k).name: v for k, v in memory_hashes.items()},
            "validation": state_audits[condition], "pca_parity": summary["pca_parity"] if seed is None else None,
        }
        write_json(directory / "protocol.json", protocol)
        summary["conditions"][condition] = {
            "directory": str(output / condition / name), **state_audits[condition],
            "sha256": {**memory_hashes, "protocol.json": sha(directory / "protocol.json"),
                       "insertion_audit.jsonl": sha(directory / "insertion_audit.jsonl")}}
    # Publish only complete artifacts; domain_completion is the resume authority.
    for condition in CONDITIONS:
        (output / condition).mkdir(exist_ok=True)
        (staging / condition).rename(output / condition / name)
    staging.rmdir()
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output / "domain_completion").mkdir(exist_ok=True)
    write_json(output / "domain_completion" / f"{name}.json", summary)
    print(json.dumps({"stage": "domain_complete", "domain": name,
                      "seconds": summary["elapsed_seconds"], "pca_parity": summary["pca_parity"]}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--caption-cache", type=Path, default=CAPTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    domains, source_hashes, caption_protocol = preflight(args.source, args.caption_cache)
    frozen_digest = digest(source_hashes)
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "n_images": 438, "n_domains": 8,
                          "frozen_input_digest": frozen_digest}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    run_protocol = {"dataset": "MemEye", "caption_model": "Qwen3-VL-8B-Instruct", "conditions": CONDITIONS,
                    "random_seeds": SEEDS, "device": args.device, "source_sha256": source_hashes,
                    "frozen_input_digest": frozen_digest, "caption_cache_protocol": caption_protocol,
                    "memory_policy": "streaming, exactly two per-leaf visual coordinates; no full visual cache",
                    "started_utc": datetime.now(timezone.utc).isoformat()}
    frozen_path = args.output / "frozen_inputs.json"
    if frozen_path.exists():
        saved = read_json(frozen_path)
        assert saved["frozen_input_digest"] == frozen_digest and saved["device"] == args.device
    else:
        assert not any(args.output.iterdir()), "Output must be fresh or contain this run's frozen_inputs.json"
        write_json(frozen_path, run_protocol)
    complete = {d["name"]: verify_completed(args.output, d["name"], frozen_digest) for d in domains}
    pending = [d for d in domains if complete[d["name"]] is None]
    if pending:
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable; no silent CPU fallback")
        visual_meta = domains[0]["protocol"]["image_encoder"]
        encoder = ImageEncoder(visual_meta["model_name"], revision=visual_meta["revision"],
                               device=args.device, local_files_only=True)
        strip_device = lambda record: {k: v for k, v in record.items() if k != "device"}
        assert strip_device(encoder.metadata) == strip_device(visual_meta), "Frozen CLIP metadata differs"
        assert all(strip_device(d["protocol"]["image_encoder"]) == strip_device(visual_meta) for d in domains)
        random_bases = {f"random_2d_seed_{seed}": make_random_basis(512, seed) for seed in SEEDS}
        for domain in pending:
            complete[domain["name"]] = build_domain(domain, args.output, encoder, source_hashes,
                                                    frozen_digest, random_bases)
        del encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    for path, expected in source_hashes.items():
        assert sha(path) == expected, f"Frozen input changed during run: {path}"
    summaries = [complete[d["name"]] for d in domains]
    result = {"completed": True, "dataset": "MemEye", "caption_model": "Qwen3-VL-8B-Instruct",
              "n_domains": 8, "n_leaves": sum(s["n_leaves"] for s in summaries),
              "n_masters_per_condition": sum(s["n_masters"] for s in summaries),
              "conditions": CONDITIONS, "random_seeds": SEEDS, "device": args.device,
              "image_forwards_all_conditions_shared": sum(s["image_encoder_calls"] for s in summaries),
              "old_image_reencodes": 0, "image_copies": 0, "per_image_visual_embeddings_saved": False,
              "all_caption_prefix_partitions_match_historical": True, "source_files_unchanged": True,
              "frozen_input_digest": frozen_digest, "domains": summaries,
              "elapsed_seconds_this_invocation": time.perf_counter() - started,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    assert result["n_leaves"] == 438 and result["n_masters_per_condition"] == 88
    assert result["image_forwards_all_conditions_shared"] == 438
    write_json(args.output / "manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "domains"}), flush=True)


if __name__ == "__main__":
    main()
