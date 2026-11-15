#!/usr/bin/env python3
"""Prepare frozen, auditable inputs for all-domain adaptive retrieval comparisons.

No reader is called and no retrieval method is evaluated here.  Home reuses its
frozen visual memory.  Other domains replay the original insertion order with
caption-only Ward clustering and two visual coordinates per leaf.  Calibration
keeps running moments, never a collection of individual visual embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from scripts.analysis.probe_caption_visual_online import (
    OFFLINE, QA_ROOT, TOKENIZER_HASHES, normalize, read_json, sha, write_json,
)
from scripts.graph.caption_ward_visual import CaptionWardVisualMemory
from scripts.graph.visual_image_encoder import ImageEncoder

HOME = "Home_Renovation_Interior_Design"
FROZEN_ONLINE = ROOT / "reports/home_online_routing_dialogue_caption_20261020"
DEFAULT_OUTPUT = ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain"
SEPARATOR = "\nMemory description: "


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def assert_unique(rows, key="id"):
    mapping = {row[key]: row for row in rows}
    assert len(mapping) == len(rows), key
    return mapping


def collect_sources():
    """Recover exact reader text and chronology without using QA to construct it."""
    qa = read_jsonl(QA_ROOT / "qa_manifest.jsonl")
    reader = assert_unique(read_jsonl(QA_ROOT / "retrieval_memories.jsonl"))
    memories = assert_unique(read_jsonl(ROOT / "results/natural_observation/memories.jsonl"))
    images = assert_unique(read_jsonl(ROOT / "results/natural_observation/images.jsonl"))
    source_protocol = read_json(QA_ROOT / "protocol.json")
    offline = read_json(OFFLINE / "protocol.json")
    assert len(qa) == 371 and len(reader) == len(memories) == 438
    assert_unique(qa)
    domains = sorted({row["task"] for row in qa})
    tracked = [
        QA_ROOT / name for name in
        ("qa_manifest.jsonl", "retrieval_memories.jsonl", "protocol.json", "arrivals.jsonl")
    ]
    tracked += [ROOT / "results/natural_observation" / f"{name}.jsonl"
                for name in ("memories", "images")]
    streams = {}
    for domain in domains:
        dialog_path = ROOT / "data/MemEye/data/dialog" / f"{domain}_Open.json"
        original_protocol = read_json(ROOT / "results/natural_observation/protocol.json")
        expected = next(row["sha256"] for row in original_protocol["source_files"]
                        if Path(row["path"]).name == dialog_path.name)
        assert sha(dialog_path) == expected
        tracked.append(dialog_path)
        full_by_id = {row["image_id"]: row for row in read_jsonl(QA_ROOT / "arrivals.jsonl")
                      if row["memory_id"].startswith(domain + "/")}
        stream = []
        for session in read_json(dialog_path)["multi_session_dialogues"]:
            for turn in session["dialogues"]:
                for number, relative in enumerate(turn.get("input_image", []), 1):
                    memory_id = f"{domain}/{turn['round']}"
                    image_id = f"{memory_id}#image{number}"
                    path = ROOT / "data/MemEye/data/image" / relative
                    digest = sha(path)
                    assert len(memories[memory_id]["images"]) == 1
                    assert digest == memories[memory_id]["images"][number - 1]
                    assert digest in images
                    dialogue, caption = reader[memory_id]["reader_text"].split(SEPARATOR, 1)
                    assert caption == full_by_id[image_id]["caption"], image_id
                    expected_dialogue = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                    assert dialogue == expected_dialogue, image_id
                    stream.append({
                        "image_id": image_id, "memory_id": memory_id,
                        "filename": path.name, "source_path": str(path), "sha256": digest,
                        "caption": caption, "dialogue": dialogue,
                        "retrieval_text": reader[memory_id]["reader_text"],
                        "timestamp": session["date"], "session_id": session["session_id"],
                        "round": turn["round"], "insertion_index": len(stream) + 1,
                    })
        assert len(stream) == len(full_by_id)
        assert {r["memory_id"] for r in stream} == {
            key for key, value in memories.items() if value["task"] == domain
        }
        assert_unique(stream, "image_id")
        streams[domain] = stream
    assert sum(map(len, streams.values())) == 438
    assert sum(bool(q["query_image_paths"]) for q in qa) == 8
    source_hashes = {str(path): sha(path) for path in tracked}
    return qa, reader, streams, source_protocol, offline, source_hashes


def verify_qa_reuse_source(qa, reader, protocol, output):
    """Verify the comparison cache already retained with the independent experiment."""
    source = DEFAULT_OUTPUT / "qa_reuse_source.json"
    value = read_json(source)
    records = {row["id"]: row for row in value["records"]}
    assert len(records) == len(qa) == 371
    settings = value["source_reader_protocol"]
    for key, expected in settings.items():
        if key != "experiment_signature":
            assert protocol[key] == expected, key
    for question in qa:
        saved = records[question["id"]]
        user_text = (
            "Retrieved dialogue and image descriptions (ranked by relevance):\n\n"
            + "\n\n".join(f"Memory {i}: {reader[mid]['reader_text']}"
                            for i, mid in enumerate(saved["ordered_memory_ids"], 1))
            + "\n\nQuestion:\n" + question["question"] + "\n\nOptions:\n"
            + "\n".join(f"{label}. {question['options'][label]}" for label in "ABCD")
            + "\n\nAnswer with one letter only."
        )
        images = [{"path": p, "sha256": sha(p)} for p in question["query_image_paths"]]
        payload = {"system": settings["reader_system_prompt"], "user_text": user_text,
                   "query_images": images}
        assert images == saved["query_images"]
        assert hashlib.sha256(user_text.encode()).hexdigest() == saved["reader_text_sha256"]
        assert json_digest(payload) == saved["payload_sha256"] == saved["response"]["payload_sha256"]
        assert saved["response"]["experiment_signature"] == settings["experiment_signature"]
        assert saved["historical_correct"] == (saved["response"]["predicted_choice"] == question["gold_choice"])
    target = output / "qa_reuse_source.json"
    if target.resolve() != source.resolve():
        target.write_bytes(source.read_bytes())
    return value


def make_questions(qa, domain):
    return [{key: row[key] for key in (
        "id", "question", "gold_ids", "retrieval_query", "query_image_paths",
        "query_image_captions", "options", "gold_choice",
    )} | {"query_has_image": bool(row["query_image_paths"])}
            for row in qa if row["task"] == domain]


def encode_text_features(model, questions, leaves, prefix):
    """Match the frozen Home query+leaf batch layout exactly."""
    texts = [prefix + q["retrieval_query"] for q in questions]
    texts += [leaf["retrieval_text"] for leaf in leaves]
    lengths = [len(row) for row in model.tokenizer(texts, truncation=False)["input_ids"]]
    vectors = normalize(model.encode(
        texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, prompt="",
    ))
    nq = len(questions)
    ids = [q["id"] for q in questions] + [leaf["image_id"] for leaf in leaves]
    metadata = {
        "batch_size": 16, "token_limit": model.max_seq_length,
        "query_prefix": prefix, "query_input": "manifest.retrieval_query",
        "leaf_input": "original dialogue + '\\nMemory description: ' + unchanged full caption",
        "query_token_counts": lengths[:nq], "leaf_token_counts": lengths[nq:],
        "truncated_inputs": [
            {"id": identity, "original_tokens": length, "retained_limit": model.max_seq_length}
            for identity, length in zip(ids, lengths) if length > model.max_seq_length
        ],
        "normalization": "float32 model output followed by float64 L2 normalization",
    }
    return vectors[:nq], vectors[nq:], metadata


def save_domain(output, domain, inputs, bge_query, bge_leaf, clip_scores, protocol):
    target = output / domain
    target.mkdir(parents=True, exist_ok=True)
    assert bge_query.shape == (len(inputs["questions"]), 384)
    assert bge_leaf.shape == (len(inputs["leaves"]), 384)
    assert clip_scores.shape == (len(inputs["questions"]), len(inputs["masters"]))
    clip_order = np.argsort(-clip_scores, axis=1, kind="stable")
    np.savez(target / "features.npz", bge_query=bge_query, bge_leaf=bge_leaf,
             clip_master=clip_scores, clip_order=clip_order)
    write_json(target / "inputs.json", inputs)
    write_json(target / "protocol.json", protocol)
    return target


def prepare_home(output, qa, streams, model, protocol):
    source = read_json(FROZEN_ONLINE / "inputs.json")
    questions = make_questions(qa, HOME)
    assert [q["id"] for q in questions] == [q["id"] for q in source["questions"]]
    expected = {row["image_id"]: row for row in streams[HOME]}
    for leaf in source["leaves"]:
        for key in ("memory_id", "filename", "dialogue", "caption", "retrieval_text"):
            assert leaf[key] == expected[leaf["image_id"]][key], (leaf["image_id"], key)
    inputs = {"questions": questions, "leaves": source["leaves"], "masters": source["masters"]}
    query_vectors, leaf_vectors, text_meta = encode_text_features(
        model, questions, inputs["leaves"], protocol["query_prefix"]
    )
    with np.load(FROZEN_ONLINE / "score_matrices.npz") as source_scores:
        reference_scores = source_scores["bge_leaf"].copy()
        clip_scores = source_scores["clip_master"].copy()
    delta = float(np.max(np.abs(query_vectors @ leaf_vectors.T - reference_scores)))
    assert delta == 0.0, ("Home scores changed", delta)
    target = save_domain(output, HOME, inputs, query_vectors, leaf_vectors, clip_scores, {
        **protocol, "domain": HOME, "memory_policy": "byte-for-byte frozen Home memory",
        "bge_encoding": text_meta, "clip_policy": "reuse frozen question-text CLIP master scores",
        "clip_source": str(FROZEN_ONLINE / "score_matrices.npz"),
        "home_bge_score_max_abs_difference": delta,
    })
    memory_target = target / "memory"
    memory_target.mkdir(exist_ok=True)
    copied = {}
    for name in ("graph.json", "basis.npy", "master_sums.npy"):
        source_path = OFFLINE / "memory" / name
        shutil.copyfile(source_path, memory_target / name)
        assert sha(source_path) == sha(memory_target / name)
        copied[name] = sha(source_path)
    write_json(target / "input_manifest.json", {"domain": HOME, "arrivals": streams[HOME],
                                               "memory_frozen_sha256": copied})
    write_json(target / "validation.json", {
        "passed": True, "n_questions": len(questions), "n_leaves": 89, "n_masters": 25,
        "home_bge_score_max_abs_difference": delta,
        "frozen_visual_memory_identical": True, "frozen_clip_scores_identical": True,
        "leaf_visual_coordinates": 2, "full_individual_visual_vectors_saved": False,
    })
    print(json.dumps({"ready_domain": HOME, "n_questions": len(questions),
                      "n_leaves": 89, "n_masters": 25, "home_bge_max_abs_delta": delta}), flush=True)


def calibrate_external_bases(streams, domains, encoder, output):
    """One image at a time updates each target's strictly external moments."""
    by_hash = {}
    hashes_by_domain = {}
    for domain, stream in streams.items():
        hashes_by_domain[domain] = {row["sha256"] for row in stream}
        for row in stream:
            by_hash.setdefault(row["sha256"], row["source_path"])
    assert len(by_hash) == 433
    dimension = encoder.embedding_dim
    accumulators = {domain: {
        "sum": np.zeros(dimension), "second": np.zeros((dimension, dimension)), "count": 0,
    } for domain in domains}
    for index, digest in enumerate(sorted(by_hash), 1):
        current = np.asarray(encoder.encode(by_hash[digest]), dtype=np.float64)
        second = np.outer(current, current)
        for domain in domains:
            if digest not in hashes_by_domain[domain]:
                accumulator = accumulators[domain]
                accumulator["sum"] += current
                accumulator["second"] += second
                accumulator["count"] += 1
        del current, second
        if index % 100 == 0 or index == len(by_hash):
            print(json.dumps({"calibration_processed": index, "unique_images": len(by_hash)}), flush=True)
    bases, metadata = {}, {}
    for domain in domains:
        accumulator = accumulators.pop(domain)
        n = accumulator["count"]
        mean = accumulator["sum"] / n
        covariance = accumulator["second"] / n - np.outer(mean, mean)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
        order = np.argsort(-eigenvalues, kind="stable")
        eigenvalues = eigenvalues[order]
        basis = eigenvectors[:, order[:2]].copy()
        for col in range(2):
            if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
                basis[:, col] *= -1
        assert np.allclose(basis.T @ basis, np.eye(2), rtol=0, atol=1e-12)
        selected = [digest for digest in sorted(by_hash) if digest not in hashes_by_domain[domain]]
        assert n == len(selected) and not set(selected) & hashes_by_domain[domain]
        bases[domain] = basis
        metadata[domain] = {
            "method": "centered PCA of unique other-domain image vectors via streaming moments",
            "unique_images": n, "excluded_target_image_shas": sorted(hashes_by_domain[domain]),
            "calibration_sha256": selected, "target_hash_overlap": 0,
            "fixed_before_target_stream": True, "qa_used": False,
            "eigenvalues": eigenvalues.tolist(),
            "first_two_explained_ratio": float(eigenvalues[:2].sum() / eigenvalues.sum()),
            "per_image_visual_vectors_retained": False,
        }
    write_json(output / "calibration_manifest.json", {
        "unique_source_images": 433, "single_pass_image_forwards": 433,
        "scope": "external-domain offline calibration; target-domain images excluded by SHA",
        "individual_vectors_saved": False, "domains": metadata,
    })
    return bases, metadata


def save_memory(memory, target):
    target.mkdir(parents=True, exist_ok=True)
    graph = memory.snapshot()
    sums = np.stack([m["visual_sum"] for m in memory.masters.values()])
    np.save(target / "basis.npy", memory.basis)
    np.save(target / "master_sums.npy", sums)
    graph["basis_ref"] = {"path": "basis.npy", "shape": list(memory.basis.shape)}
    for index, master in enumerate(graph["masters"]):
        master["visual_sum_ref"] = {"path": "master_sums.npy", "row": index}
        master["visual_mean_ref"] = {
            "path": "master_sums.npy", "row": index, "scale": 1 / master["n"],
        }
    graph["text_embedding_cache"].update(
        persisted=False, rebuild_from="image_nodes[].caption"
    )
    assert all(len(node["visual_coords"]) == 2 for node in graph["image_nodes"])
    assert all(set(node) == {"id", "caption", "dialogue", "timestamp", "visual_coords", "master_id"}
               for node in graph["image_nodes"])
    write_json(target / "graph.json", graph)
    return graph, sums


def encode_clip_questions(questions, model, tokenizer, limit):
    import torch
    vectors, metadata = [], []
    for question in questions:
        raw_ids = tokenizer(question["question"], truncation=False)["input_ids"]
        inputs = tokenizer(question["question"], return_tensors="pt", truncation=True,
                           max_length=limit)
        with torch.inference_mode():
            vector = model(**inputs, return_dict=True).text_embeds[0].double().numpy()
        vectors.append(normalize(vector))
        metadata.append({
            "id": question["id"], "original_tokens": len(raw_ids),
            "encoded_tokens": inputs["input_ids"].shape[1],
            "truncated": len(raw_ids) > limit, "input_ids": inputs["input_ids"][0].tolist(),
        })
    return np.stack(vectors), metadata


def prepare_new_domain(domain, output, qa, stream, basis, calibration, image_encoder,
                       text_model, clip_model, tokenizer, clip_limit, protocol):
    target = output / domain
    target.mkdir(parents=True, exist_ok=True)
    memory = CaptionWardVisualMemory(basis)
    # This reconstructible text cache is independent of vision; only the current
    # caption vector is submitted at each add, so future text cannot affect a cut.
    captions = [row["caption"] for row in stream]
    caption_lengths = [len(ids) for ids in text_model.tokenizer(
        captions, truncation=False)["input_ids"]]
    caption_vectors = normalize(text_model.encode(
        captions, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, prompt="",
    ))
    total_sum = np.zeros(image_encoder.embedding_dim)
    max_sum_error = max_projection_error = 0.0
    split_steps = 0
    with (target / "insertion_audit.jsonl").open("w") as handle:
        for index, row in enumerate(stream):
            current = np.asarray(image_encoder.encode(row["source_path"]), dtype=np.float64)
            # Match the core's stable unit normalization for aggregate validation.
            current /= np.max(np.abs(current))
            current /= np.linalg.norm(current)
            total_sum += current
            audit = memory.add(
                row["image_id"], row["caption"], caption_vectors[index], current,
                dialogue=row["dialogue"], timestamp=row["timestamp"],
            )
            del current
            assert set(memory.image_nodes) == {r["image_id"] for r in stream[:index + 1]}
            error = float(np.linalg.norm(
                np.sum([m["visual_sum"] for m in memory.masters.values()], axis=0) - total_sum
            ))
            max_sum_error = max(max_sum_error, error)
            max_projection_error = max(
                max_projection_error, audit["max_projection_consistency_l2_error"]
            )
            assert error < 1e-10 and max_projection_error < 1e-10
            split_steps += bool(audit["split_master_ids"])
            handle.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
    graph, sums = save_memory(memory, target / "memory")
    by_id = {row["image_id"]: row for row in stream}
    leaves = []
    for node in graph["image_nodes"]:
        row = by_id[node["id"]]
        leaves.append({key: row[key] for key in (
            "image_id", "memory_id", "filename", "dialogue", "caption", "retrieval_text",
        )} | {"master_id": node["master_id"]})
    masters = [{key: master[key] for key in ("id", "n", "member_ids")}
               for master in graph["masters"]]
    questions = make_questions(qa, domain)
    bge_query, bge_leaf, text_metadata = encode_text_features(
        text_model, questions, leaves, protocol["query_prefix"]
    )
    clip_query, clip_metadata = encode_clip_questions(
        questions, clip_model, tokenizer, clip_limit
    )
    means = sums / np.array([master["n"] for master in masters])[:, None]
    clip_scores = clip_query @ normalize(means).T
    inputs = {"questions": questions, "leaves": leaves, "masters": masters}
    save_domain(output, domain, inputs, bge_query, bge_leaf, clip_scores, {
        **protocol, "domain": domain,
        "memory_policy": "fresh chronological caption-Ward stream, fixed external PCA, two leaf coordinates",
        "calibration": calibration, "bge_encoding": text_metadata,
        "caption_clustering_encoding": {
            "input": "full caption only", "batch_size": 16, "prefix": "",
            "max_tokens": max(caption_lengths),
            "truncated_count": sum(n > text_model.max_seq_length for n in caption_lengths),
            "future_prefix_information_passed_to_core": False,
        },
        "clip_policy": "question text only, CLIP model token limit with explicit truncation audit",
        "clip_token_limit": clip_limit, "clip_query_encoding": clip_metadata,
        "image_arrival_forwards": len(stream), "old_member_reencoding_within_stream": 0,
    })
    write_json(target / "input_manifest.json", {"domain": domain, "arrivals": stream})
    write_json(target / "validation.json", {
        "passed": True, "n_questions": len(questions), "n_leaves": len(leaves),
        "n_masters": len(masters), "prefix_causality_checked_steps": len(stream),
        "steps_with_old_cluster_split": split_steps,
        "max_global_sum_l2_error": max_sum_error,
        "max_projection_consistency_l2_error": max_projection_error,
        "leaf_visual_coordinates": 2, "full_individual_visual_vectors_saved": False,
        "source_images_copied": False, "calibration_target_sha_overlap": 0,
        "semantic_or_qa_quality_evaluated": False,
    })
    print(json.dumps({"ready_domain": domain, "n_questions": len(questions),
                      "n_leaves": len(leaves), "n_masters": len(masters)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    qa, reader, streams, source_protocol, offline, source_hashes = collect_sources()
    verify_qa_reuse_source(qa, reader, source_protocol, output)
    import torch
    import transformers
    import sentence_transformers
    from transformers import AutoConfig, AutoTokenizer, CLIPTextModelWithProjection
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    text_info, image_info = offline["text_encoder"], offline["image_encoder"]
    for path, hashes in (
        (Path(text_info["model_path"]), text_info["files_sha256"]),
        (Path(image_info["model_name"]), {**image_info["files_sha256"], **TOKENIZER_HASHES}),
    ):
        for name, expected in hashes.items():
            assert sha(path / name) == expected, str(path / name)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_encoder": image_info, "text_encoder": text_info,
        "query_prefix": source_protocol["query_prefix"],
        "bge_query_policy": "exact source manifest retrieval_query including cached query-image caption",
        "clip_query_policy": "question text only; no native image query fusion",
        "query_options_or_gold_encoded": False, "qa_used_for_clustering_or_calibration": False,
        "visual_state": "full master sums; fixed D-by-2 basis; exactly two leaf coordinates",
        "visual_split": "partial old-cluster residual allocated proportional to overlap; approximate",
        "source_originals_policy": "read-only experiment arrivals; no original images copied into memory",
        "clustering": offline["method"], "candidate_k": offline["candidate_k"],
        "source_experiment_signature": source_protocol["experiment_signature"],
        "device": "cpu", "torch_threads": 4,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__},
        "reader_calls": 0,
    }
    text_model = SentenceTransformer(text_info["model_path"], device="cpu", local_files_only=True)
    prepare_home(output, qa, streams, text_model, protocol)
    domains = [domain for domain in sorted(streams) if domain != HOME]
    image_encoder = ImageEncoder(image_info["model_name"], revision=image_info["revision"],
                                 device="cpu", local_files_only=True)
    assert image_encoder.metadata["files_sha256"] == image_info["files_sha256"]
    bases, calibration = calibrate_external_bases(streams, domains, image_encoder, output)
    config = AutoConfig.from_pretrained(image_info["model_name"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(image_info["model_name"], local_files_only=True)
    clip_model, loading = CLIPTextModelWithProjection.from_pretrained(
        image_info["model_name"], config=config.text_config, local_files_only=True,
        output_loading_info=True,
    )
    assert not any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
    assert all(k.startswith(("vision_model.", "visual_projection.")) or k == "logit_scale"
               for k in loading.get("unexpected_keys", []))
    clip_model.to(device="cpu", dtype=torch.float32).eval()
    for domain in domains:
        prepare_new_domain(
            domain, output, qa, streams[domain], bases[domain], calibration[domain],
            image_encoder, text_model, clip_model, tokenizer,
            config.text_config.max_position_embeddings, protocol,
        )
    for path, expected in source_hashes.items():
        assert sha(path) == expected, path
    entries = []
    for domain in sorted(streams):
        target = output / domain
        validation = read_json(target / "validation.json")
        entries.append({
            "domain": domain, "directory": str(target),
            "n_questions": validation["n_questions"], "n_leaves": validation["n_leaves"],
            "n_masters": validation["n_masters"],
            "n_image_queries": sum(bool(q["query_image_paths"])
                                   for q in qa if q["task"] == domain),
            "sha256": {name: sha(target / name) for name in (
                "inputs.json", "features.npz", "protocol.json", "input_manifest.json",
                "memory/graph.json", "memory/basis.npy", "memory/master_sums.npy",
            )},
        })
    assert sum(row["n_questions"] for row in entries) == 371
    assert sum(row["n_leaves"] for row in entries) == 438
    manifest = {
        "completed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_domains": 8, "n_questions": 371, "n_leaves": 438,
        "n_image_queries": 8, "n_text_only_queries": 363,
        "home_frozen": True, "leaf_visual_coordinates": 2,
        "full_individual_visual_vectors_saved": False,
        "single_pass_calibration_image_forwards": 433,
        "new_memory_stream_image_forwards": 349,
        "feature_keys": {"bge_query": "[question,384]", "bge_leaf": "[leaf,384]",
                         "clip_master": "[question,master]", "clip_order": "[question,master]"},
        "domains": entries, "source_sha256": source_hashes,
        "script_sha256": sha(Path(__file__)),
        "source_files_unchanged": True, "reader_calls": 0,
        "new_retrieval_evaluation_performed": False,
        "qa_reuse_source": "qa_reuse_source.json",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"completed": True, "output": str(output),
                      "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
