"""Isolated two-image/caption residual diagnostic; never edits memory pipelines.

The user explicitly requested these embeddings for analysis. They are saved as
experiment artifacts, not added to the strict two-coordinate memory prototype.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".vendor"))
from graph.visual_image_encoder import ImageEncoder

SOURCE_GRAPH = ROOT / "results/visual_embedding_graph/Home_Renovation_Interior_Design/graph.json"
SOURCE_CACHE = ROOT / "results/natural_observation/captions.jsonl"
SOURCE_DIALOG = ROOT / "data/MemEye/data/dialog/Home_Renovation_Interior_Design.json"
SOURCE_PROTOCOL = ROOT / "results/natural_observation/protocol.json"
CLIP_MODEL = ROOT / "models/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
IMAGE_NAMES = ["D7_IMG_005.jpg", "D7_IMG_006.jpg"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    assert np.isfinite(vectors).all() and (norms > 0).all()
    return vectors / norms


def gather_inputs():
    graph = json.loads(SOURCE_GRAPH.read_text())
    cache = [json.loads(line) for line in SOURCE_CACHE.read_text().splitlines() if line.strip()]
    dialog = json.loads(SOURCE_DIALOG.read_text())
    protocol = json.loads(SOURCE_PROTOCOL.read_text())
    records = []

    def find_rounds(obj, relative_image):
        found = []
        if isinstance(obj, dict):
            if relative_image in obj.get("input_image", []):
                found.append(obj)
            for value in obj.values():
                found.extend(find_rounds(value, relative_image))
        elif isinstance(obj, list):
            for value in obj:
                found.extend(find_rounds(value, relative_image))
        return found

    for name in IMAGE_NAMES:
        path = ROOT / "data/MemEye/data/image/Home_Renovation_Interior_Design" / name
        matching = [n for n in graph["image_nodes"] if Path(n.get("image_path", "")).resolve() == path]
        assert len(matching) == 1
        node = matching[0]
        digest = sha(path)
        matching_cache = [c for c in cache if c.get("id") == digest]
        assert len(matching_cache) == 1
        caption_row = matching_cache[0]
        assert caption_row["caption"] == node["caption"] and not caption_row["hit_token_limit"]
        rounds = find_rounds(dialog, f"Home_Renovation_Interior_Design/{name}")
        assert len(rounds) == 1
        assert node["id"].split("/")[-1].split("#")[0] == rounds[0]["round"]
        records.append({
            "filename": name, "image_path": str(path), "image_sha256": digest,
            "node_id": node["id"], "timestamp": node["timestamp"],
            "caption": node["caption"],
            "caption_sha256": hashlib.sha256(node["caption"].encode()).hexdigest(),
            "caption_source": str(SOURCE_GRAPH),
            "caption_cache_record": caption_row,
            "dataset_short_caption_reference_only": rounds[0]["image_caption"],
            "caption_is_full_pipeline_caption_not_dataset_short_caption": True,
            "dialogue_reference_only": node["dialogue"],
        })
        (HERE / f"{Path(name).stem}.caption.txt").write_text(node["caption"] + "\n")
    return records, protocol


def decompose(vectors, label, arrays):
    assert vectors.shape[0] == 2 and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    residual = vectors[0] - vectors[1]
    u, s, vh = np.linalg.svd(residual[None, :], full_matrices=False)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    cu, cs, cvh = np.linalg.svd(centered, full_matrices=False)
    arrays.update({
        f"{label}_embeddings": vectors, f"{label}_residual_005_minus_006": residual,
        f"{label}_residual_U": u, f"{label}_residual_S": s, f"{label}_residual_Vh": vh,
        f"{label}_centered_pair": centered, f"{label}_centered_U": cu,
        f"{label}_centered_S": cs, f"{label}_centered_Vh": cvh,
    })
    cosine = float(vectors[0] @ vectors[1])
    reconstruction_error = float(np.max(np.abs((u * s) @ vh - residual[None, :])))
    assert reconstruction_error < 1e-12
    assert np.isclose(s[0], np.sqrt(2 - 2 * cosine), atol=1e-12)
    assert np.isclose(cs[0], s[0] / np.sqrt(2), atol=1e-12)
    rank = int(np.linalg.matrix_rank(residual[None, :]))
    centered_rank = int(np.linalg.matrix_rank(centered))
    assert rank == centered_rank == 1
    return {
        "embedding_shape": list(vectors.shape), "residual_matrix_shape": [1, vectors.shape[1]],
        "cosine_similarity": cosine, "residual_l2_norm": float(np.linalg.norm(residual)),
        "residual_singular_values": s.tolist(), "residual_rank": rank,
        "residual_svd_max_reconstruction_error": reconstruction_error,
        "centered_pair_shape": list(centered.shape), "centered_pair_singular_values": cs.tolist(),
        "centered_pair_rank": centered_rank,
        "residual_formula": "embedding_005 - embedding_006 after L2 normalization",
        "interpretation": "Single-vector SVD is rank one; singular value is pair separation, not detail count or fidelity.",
    }


def caption_chunks(text, tokenizer, limit):
    """Greedy whole-sentence packing, no truncation, exact token coverage checked."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        assert len(tokenizer.encode(sentence, add_special_tokens=True, truncation=False)) <= limit
        candidate = current + " " + sentence if current else sentence
        if len(tokenizer.encode(candidate, add_special_tokens=True, truncation=False)) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    assert " ".join(chunks) == text
    original_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunk_ids = [tokenizer.encode(c, add_special_tokens=False, truncation=False) for c in chunks]
    assert sum(chunk_ids, []) == original_ids
    return chunks, [len(ids) for ids in chunk_ids]


def score_probes(vectors, probe_vectors, probes):
    scores = vectors @ probe_vectors.T
    delta = vectors[0] - vectors[1]
    results = []
    for i, probe in enumerate(probes):
        local = scores[:, 2*i:2*i+2]
        a = float(local[0, 0] - local[0, 1])
        b = float(local[1, 1] - local[1, 0])
        q = probe_vectors[2*i] - probe_vectors[2*i+1]
        difference = float(delta @ q)
        assert np.isclose(difference, a + b, atol=1e-12)
        results.append({
            **probe, "cosine_scores_rows_005_006_cols_prompt005_prompt006": local.tolist(),
            "correct_choice_margin_005": a, "correct_choice_margin_006": b,
            "correct_choice_005": a > 0, "correct_choice_006": b > 0,
            "residual_dot_prompt_difference": difference,
            "pair_relative_direction_correct": difference > 0,
            "normalized_residual_prompt_difference_cosine": float(unit(delta) @ unit(q)),
        })
    primary = [r for r in results if r["primary"]]
    return {
        "cases": results, "primary_prompt_pair_count": len(primary),
        "primary_relative_directions_correct": sum(r["pair_relative_direction_correct"] for r in primary),
        "primary_individual_choices_correct": sum(r["correct_choice_005"] + r["correct_choice_006"] for r in primary),
        "primary_individual_choice_count": 2 * len(primary),
        "warning": "Hand-authored AI visual diagnostics, not independent human labels. Margins are not calibrated across encoders.",
    }


def main():
    if (HERE / "results.json").exists():
        raise FileExistsError("Completed experiment already exists; choose a new output directory.")
    import torch
    import transformers
    import sentence_transformers
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection, AutoConfig
    torch.set_num_threads(4)
    started = time.perf_counter()
    inputs, natural_protocol = gather_inputs()
    reference_path = HERE / "detail_reference.json"
    reference = json.loads(reference_path.read_text())
    reference_sha = sha(reference_path)
    probes = reference["probes"]
    assert len(probes) == 10 and sum(p["primary"] for p in probes) == 8
    probe_texts = [p[k] for p in probes for k in ("prompt_005", "prompt_006")]
    captions = [row["caption"] for row in inputs]
    write_json(HERE / "inputs.json", inputs)
    write_json(HERE / "protocol.json", {
        "created_before_encoding_utc": datetime.now(timezone.utc).isoformat(),
        "images_in_order": IMAGE_NAMES, "residual_order": "005 minus 006",
        "primary_image_model": str(CLIP_MODEL), "primary_text_model": natural_protocol["retriever_model_path"],
        "all_embeddings_l2_normalized": True, "caption_input": "full existing Qwen caption; no dialogue/query prefix",
        "detail_reference_sha256": reference_sha, "prompt_tuning_after_results": False,
        "clip_full_caption_control": "greedy whole-sentence chunks, normalized chunk vectors, token-count-weighted mean then L2 normalization",
        "no_silent_text_truncation": True, "no_arbitrary_vector_reshape": True,
        "no_pipeline_or_graph_changes": True,
        "analytical_scope": "one image pair; scalar SVD cannot identify a modality winner",
    })

    # Primary representations, consistent with existing pipeline encoders.
    image_encoder = ImageEncoder(str(CLIP_MODEL), revision=CLIP_REVISION, local_files_only=True)
    image_vectors = unit([image_encoder.encode(row["image_path"]) for row in inputs])
    image_metadata = image_encoder.metadata
    del image_encoder
    text_model_path = Path(natural_protocol["retriever_model_path"])
    text_encoder = sentence_transformers.SentenceTransformer(
        str(text_model_path), device="cpu", local_files_only=True)
    texts = captions + probe_texts
    token_counts = [len(text_encoder.tokenizer.encode(t, truncation=False)) for t in texts]
    assert max(token_counts) <= text_encoder.max_seq_length
    text_vectors = unit(text_encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    text_metadata = {
        "model_path": str(text_model_path), "revision": natural_protocol["retriever_revision"],
        "embedding_dim": int(text_vectors.shape[1]), "max_seq_length": text_encoder.max_seq_length,
        "pooling_config": json.loads((text_model_path / "1_Pooling/config.json").read_text()),
        "normalization": "SentenceTransformer normalized output, float64 renormalization",
        "query_prefix": None, "caption_token_counts": token_counts[:2],
        "probe_token_counts": token_counts[2:], "truncated_inputs": 0,
        "files_sha256": {str(p.relative_to(text_model_path)): sha(p) for p in sorted(text_model_path.rglob("*"))
                         if p.is_file() and (p.suffix in {".json", ".safetensors", ".bin", ".txt"})
                         and "onnx" not in p.parts},
    }
    del text_encoder
    print("Primary CLIP image and full BGE caption embeddings complete.", flush=True)

    # Same CLIP checkpoint, short semantic probes and explicit full-coverage
    # chunk aggregation control. Never silently use the first 77 caption tokens.
    tokenizer = CLIPTokenizer.from_pretrained(str(HERE / "tokenizer"), local_files_only=True)
    config = AutoConfig.from_pretrained(str(CLIP_MODEL), local_files_only=True)
    clip_limit = config.text_config.max_position_embeddings
    clip_text_encoder, loading_info = CLIPTextModelWithProjection.from_pretrained(
        str(CLIP_MODEL), config=config.text_config, local_files_only=True, output_loading_info=True)
    assert not any(loading_info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs"))
    clip_text_encoder.to(dtype=torch.float32, device="cpu").eval()

    def encode_clip_text(values):
        counts = [len(tokenizer.encode(t, truncation=False)) for t in values]
        assert max(counts) <= clip_limit
        batch = tokenizer(values, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            output = clip_text_encoder(**batch, return_dict=True).text_embeds
        return unit(output.detach().cpu().to(torch.float64).numpy())

    clip_probes = encode_clip_text(probe_texts)
    chunk_control, chunk_records = [], []
    for caption in captions:
        chunks, counts = caption_chunks(caption, tokenizer, clip_limit)
        vectors = encode_clip_text(chunks)
        aggregate = unit(np.average(vectors, weights=counts, axis=0))
        chunk_control.append(aggregate)
        chunk_records.append({
            "full_caption_tokens_with_special_tokens": len(tokenizer.encode(caption, truncation=False)),
            "chunks": chunks, "content_tokens_per_chunk": counts,
            "chunk_tokens_with_special_tokens": [len(tokenizer.encode(c, truncation=False)) for c in chunks],
            "all_original_content_tokens_preserved": True, "chunk_vectors": vectors.tolist(),
        })
    clip_control = np.stack(chunk_control)
    del clip_text_encoder
    arrays = {
        "bge_probe_embeddings": text_vectors[2:], "clip_probe_embeddings": clip_probes,
    }
    representations = {
        "clip_image": image_vectors, "bge_full_caption": text_vectors[:2],
        "clip_caption_chunk_mean_control": clip_control,
    }
    decompositions = {name: decompose(vectors, name, arrays) for name, vectors in representations.items()}
    for label, vectors in representations.items():
        for name, vector in zip(IMAGE_NAMES, vectors):
            np.save(HERE / f"{Path(name).stem}.{label}.npy", vector)
        np.save(HERE / f"{label}.residual_005_minus_006.npy", vectors[0] - vectors[1])
    details = {
        "clip_image": score_probes(image_vectors, clip_probes, probes),
        "bge_full_caption": score_probes(text_vectors[:2], text_vectors[2:], probes),
        "clip_caption_chunk_mean_control": score_probes(clip_control, clip_probes, probes),
    }
    # Common-space alignment is not available for the primary CLIP/BGE pair.
    aligned_cosine = float(unit(image_vectors[0] - image_vectors[1]) @ unit(clip_control[0] - clip_control[1]))
    np.savez(HERE / "embeddings_residuals_svd.npz", **arrays)
    write_json(HERE / "detail_probe_results.json", details)
    assert reference_sha == sha(reference_path)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "image_encoder": image_metadata, "text_encoder": text_metadata,
        "clip_caption_control": {
            "same_checkpoint_as_image": True, "context_limit": clip_limit,
            "protocol": "whole-sentence chunking; token-count-weighted mean of normalized projected text features, then normalize",
            "standard_single_pass_full_caption_embedding": False,
            "truncated_inputs": 0, "captions": chunk_records,
            "residual_alignment_with_image_cosine": aligned_cosine,
            "tokenizer_provenance": json.loads((HERE / "tokenizer/provenance.json").read_text()),
        },
        "decompositions": decompositions,
        "detail_probe_summary": {name: {k: v for k, v in r.items() if k != "cases"} for name, r in details.items()},
        "provenance": {
            "sources_sha256": {str(p): sha(p) for p in (SOURCE_GRAPH, SOURCE_CACHE, SOURCE_DIALOG, SOURCE_PROTOCOL)},
            "detail_reference_sha256": reference_sha,
            "script_sha256": sha(Path(__file__)),
            "image_encoder_script_sha256": sha(ROOT / "scripts/graph/visual_image_encoder.py"),
            "array_artifact_sha256": sha(HERE / "embeddings_residuals_svd.npz"),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "sentence_transformers_version": sentence_transformers.__version__, "numpy_version": np.__version__,
            "runtime_seconds_including_model_loads": time.perf_counter() - started,
        },
        "limitations": [
            "One residual vector has rank at most one; SVD cannot count semantic details.",
            "Residual magnitude measures pair separation and is not calibrated as fidelity across CLIP/BGE.",
            "Exactly two centered embeddings have rank at most one; this does not test two-dimensional memory truncation.",
            "Probe cases and paraphrases are AI-authored diagnostics, not independent human labels or a held-out benchmark.",
            "BGE probes inspect caption text already filtered by caption generation.",
            "CLIP caption control uses chunk pooling, an additional protocol with no guarantee of exact full-text representation.",
            "CLIP global features and resizing can suppress small visual details; probe failures do not prove information is absent.",
            "Different viewpoints, object configurations and captions vary together; this is not a causal modality ablation.",
        ],
    }
    write_json(HERE / "results.json", result)
    print(json.dumps({"decompositions": decompositions, "probes": result["detail_probe_summary"],
                      "clip_caption_tokens": [c["full_caption_tokens_with_special_tokens"] for c in chunk_records]},
                     indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
