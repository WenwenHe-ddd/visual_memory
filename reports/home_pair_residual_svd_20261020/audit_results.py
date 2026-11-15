"""Independent artifact audit; no model inference or downloads.

Only reads saved experiment/source artifacts and writes independent_audit.json.
Tokenizers are reloaded locally to verify caption/probe token coverage.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKS = []


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(name, passed, evidence=None):
    row = {"check": name, "passed": bool(passed)}
    if evidence is not None:
        row["evidence"] = evidence
    CHECKS.append(row)


def close(name, actual, expected, atol=1e-12):
    a, e = np.asarray(actual), np.asarray(expected)
    valid = a.shape == e.shape and np.allclose(a, e, rtol=0, atol=atol)
    error = float(np.max(np.abs(a - e))) if a.shape == e.shape and a.size else None
    check(name, valid, {"maximum_absolute_error": error})


def load(name):
    return json.loads((HERE / name).read_text())


def normalized(vector):
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array, axis=-1, keepdims=True)


def main():
    result, protocol, inputs, reference, details = [
        load(name) for name in (
            "results.json", "protocol.json", "inputs.json", "detail_reference.json",
            "detail_probe_results.json")
    ]
    arrays = np.load(HERE / "embeddings_residuals_svd.npz", allow_pickle=False)
    provenance = result["provenance"]
    source = (HERE / "run_analysis.py").read_text()
    parsed = ast.parse(source)
    check("analysis_source_hash", sha(HERE / "run_analysis.py") == provenance["script_sha256"])
    check("image_encoder_source_hash",
          sha(ROOT / "scripts/graph/visual_image_encoder.py") ==
          provenance["image_encoder_script_sha256"])
    check("array_artifact_hash",
          sha(HERE / "embeddings_residuals_svd.npz") == provenance["array_artifact_sha256"])
    ref_hash = sha(HERE / "detail_reference.json")
    check("frozen_probe_hash_in_protocol_and_result",
          ref_hash == protocol["detail_reference_sha256"] ==
          provenance["detail_reference_sha256"])
    check("protocol_timestamp_precedes_result",
          datetime.fromisoformat(protocol["created_before_encoding_utc"]) <
          datetime.fromisoformat(result["created_at_utc"]))
    check("inputs_copied_without_changes", inputs == result["inputs"])
    check("two_requested_images_in_correct_order",
          [r["filename"] for r in inputs] == protocol["images_in_order"] ==
          ["D7_IMG_005.jpg", "D7_IMG_006.jpg"])
    check("residual_order_documented", protocol["residual_order"] == "005 minus 006")
    check("no_caption_query_prefix", result["text_encoder"]["query_prefix"] is None)
    check("standalone_analysis_scope", protocol["no_pipeline_or_graph_changes"])
    check("no_arbitrary_embedding_reshape_in_analysis_source",
          not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "reshape" for node in ast.walk(parsed)))
    check("source_freezes_probe_hash_before_encoding",
          source.index("reference_sha = sha(reference_path)") <
          source.index("image_encoder = ImageEncoder"))
    check("source_rechecks_probe_hash_after_scoring",
          source.index("assert reference_sha == sha(reference_path)") >
          source.index('"clip_image": score_probes'))
    check("source_encodes_original_captions_without_probe_modification",
          'captions = [row["caption"] for row in inputs]' in source and
          'texts = captions + probe_texts' in source and
          'text_vectors[:2]' in source)
    for path, digest in provenance["sources_sha256"].items():
        check(f"source_hash:{Path(path).name}", sha(path) == digest)
    graph_path = Path(inputs[0]["caption_source"])
    graph = json.loads(graph_path.read_text())
    cache = [json.loads(line) for line in
             (ROOT / "results/natural_observation/captions.jsonl").read_text().splitlines()
             if line.strip()]
    for index, record in enumerate(inputs):
        prefix = record["filename"]
        check(f"{prefix}:image_content_hash", sha(record["image_path"]) == record["image_sha256"])
        check(f"{prefix}:full_caption_hash",
              hashlib.sha256(record["caption"].encode()).hexdigest() == record["caption_sha256"])
        matches = [row for row in graph["image_nodes"] if row["id"] == record["node_id"]]
        cached = [row for row in cache if row.get("id") == record["image_sha256"]]
        check(f"{prefix}:unique_graph_node", len(matches) == 1)
        check(f"{prefix}:unique_caption_cache_record", len(cached) == 1)
        check(f"{prefix}:caption_exactly_matches_graph_and_cache",
              matches[0]["caption"] == cached[0]["caption"] == record["caption"])
        check(f"{prefix}:cache_metadata_exact", cached[0] == record["caption_cache_record"])
        check(f"{prefix}:caption_generation_did_not_hit_limit", not cached[0]["hit_token_limit"])
        check(f"{prefix}:correct_graph_image_path",
              Path(matches[0]["image_path"]).resolve() == Path(record["image_path"]).resolve())
        check(f"{prefix}:saved_caption_text_complete",
              (HERE / f"{Path(prefix).stem}.caption.txt").read_text() == record["caption"] + "\n")
        check(f"{prefix}:caption_distinct_from_dataset_short_reference",
              record["caption"] not in record["dataset_short_caption_reference_only"])
        check(f"{prefix}:dialogue_not_used_as_caption",
              matches[0]["dialogue"] == record["dialogue_reference_only"])
    for key in ("image_encoder", "text_encoder"):
        metadata = result[key]
        model_path = Path(metadata.get("model_name", metadata.get("model_path")))
        for filename, digest in metadata["files_sha256"].items():
            check(f"{key}:model_file:{filename}", sha(model_path / filename) == digest)
    for item in result["clip_caption_control"]["tokenizer_provenance"]:
        local = HERE / "tokenizer" / item["filename"]
        check(f"clip_tokenizer_hash:{item['filename']}", sha(local) == item["sha256"])
        check(f"clip_tokenizer_size:{item['filename']}", local.stat().st_size == item["bytes"])
        check(f"clip_tokenizer_pinned_revision:{item['filename']}",
              "/openai/clip-vit-base-patch32/resolve/" +
              result["image_encoder"]["revision"] + "/" in item["source_url"])

    feature_summary = {}
    for label, dimension in (("clip_image", 512), ("bge_full_caption", 384),
                             ("clip_caption_chunk_mean_control", 512)):
        embeddings = arrays[f"{label}_embeddings"]
        residual = arrays[f"{label}_residual_005_minus_006"]
        saved = result["decompositions"][label]
        check(f"{label}:shape_and_finite",
              embeddings.shape == (2, dimension) and np.isfinite(embeddings).all())
        close(f"{label}:unit_norms", np.linalg.norm(embeddings, axis=1), np.ones(2))
        close(f"{label}:residual_sign_and_values", residual, embeddings[0] - embeddings[1])
        for i, stem in enumerate(("D7_IMG_005", "D7_IMG_006")):
            close(f"{label}:{stem}:npy_matches_npz",
                  np.load(HERE / f"{stem}.{label}.npy", allow_pickle=False), embeddings[i])
        close(f"{label}:residual_npy_matches_npz",
              np.load(HERE / f"{label}.residual_005_minus_006.npy", allow_pickle=False), residual)
        u, s, vh = [arrays[f"{label}_residual_{name}"] for name in ("U", "S", "Vh")]
        check(f"{label}:SVD_is_single_row_not_reshaped",
              u.shape == (1, 1) and s.shape == (1,) and vh.shape == (1, dimension))
        close(f"{label}:SVD_reconstruction", (u * s) @ vh, residual[np.newaxis, :])
        close(f"{label}:singular_value_equals_residual_norm", s[0], np.linalg.norm(residual))
        cosine = embeddings[0] @ embeddings[1]
        close(f"{label}:singular_value_equals_cosine_identity", s[0], np.sqrt(2 - 2 * cosine))
        check(f"{label}:rank_one", np.linalg.matrix_rank(residual[np.newaxis, :]) ==
              saved["residual_rank"] == 1)
        close(f"{label}:reported_cosine", cosine, saved["cosine_similarity"])
        close(f"{label}:reported_singular_values", s, saved["residual_singular_values"])
        centered = embeddings - embeddings.mean(axis=0)
        close(f"{label}:centered_pair", arrays[f"{label}_centered_pair"], centered)
        cu, cs, cvh = [arrays[f"{label}_centered_{name}"] for name in ("U", "S", "Vh")]
        close(f"{label}:centered_SVD_reconstruction", (cu * cs) @ cvh, centered)
        close(f"{label}:centered_first_singular_value", cs[0], s[0] / np.sqrt(2))
        check(f"{label}:centered_rank_one",
              np.linalg.matrix_rank(centered) == saved["centered_pair_rank"] == 1)
        close(f"{label}:one_affine_component_reconstructs_both",
              embeddings.mean(axis=0) + (cu[:, :1] * cs[:1]) @ cvh[:1], embeddings)
        probe_vectors = arrays["bge_probe_embeddings" if label == "bge_full_caption"
                               else "clip_probe_embeddings"]
        check(f"{label}:probe_dimensions", probe_vectors.shape == (20, dimension))
        close(f"{label}:probe_unit_norms", np.linalg.norm(probe_vectors, axis=1), np.ones(20))
        scores = embeddings @ probe_vectors.T
        cases = details[label]["cases"]
        check(f"{label}:probe_count", len(cases) == len(reference["probes"]) == 10)
        features = {}
        for i, (case, frozen) in enumerate(zip(cases, reference["probes"])):
            name = f"{label}:{frozen['id']}"
            check(f"{name}:frozen_probe_text_and_metadata",
                  all(case[key] == value for key, value in frozen.items()))
            local = scores[:, 2*i:2*i+2]
            a, b = float(local[0, 0] - local[0, 1]), float(local[1, 1] - local[1, 0])
            direction = probe_vectors[2*i] - probe_vectors[2*i+1]
            margin = float(residual @ direction)
            close(f"{name}:score_matrix",
                  local, case["cosine_scores_rows_005_006_cols_prompt005_prompt006"])
            close(f"{name}:individual_margins",
                  [a, b], [case["correct_choice_margin_005"], case["correct_choice_margin_006"]])
            close(f"{name}:difference_margin",
                  margin, case["residual_dot_prompt_difference"])
            close(f"{name}:difference_equals_sum_of_individual_margins", margin, a + b)
            close(f"{name}:direction_cosine", normalized(residual) @ normalized(direction),
                  case["normalized_residual_prompt_difference_cosine"])
            check(f"{name}:flags_use_correct_sign",
                  case["correct_choice_005"] == (a > 0) and
                  case["correct_choice_006"] == (b > 0) and
                  case["pair_relative_direction_correct"] == (margin > 0))
            features.setdefault(frozen["feature_id"], []).append({
                "variant": frozen["variant"], "relative_margin": margin,
                "relative_direction_correct": margin > 0,
                "individual_choices_correct": int(a > 0) + int(b > 0),
                "primary": frozen["primary"],
            })
        primary = [case for case in cases if case["primary"]]
        # Explicit int conversion avoids numpy boolean addition behaving as OR.
        expected_counts = {
            "primary_prompt_pair_count": len(primary),
            "primary_relative_directions_correct":
                sum(int(case["pair_relative_direction_correct"]) for case in primary),
            "primary_individual_choices_correct":
                sum(int(case["correct_choice_005"]) + int(case["correct_choice_006"])
                    for case in primary),
            "primary_individual_choice_count": 2 * len(primary),
        }
        for key, value in expected_counts.items():
            check(f"{label}:count:{key}",
                  details[label][key] == result["detail_probe_summary"][label][key] == value)
        feature_summary[label] = features

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer, CLIPTokenizer
    bge = AutoTokenizer.from_pretrained(result["text_encoder"]["model_path"], local_files_only=True)
    clip = CLIPTokenizer.from_pretrained(str(HERE / "tokenizer"), local_files_only=True)
    probe_texts = [p[key] for p in reference["probes"] for key in ("prompt_005", "prompt_006")]
    bge_counts = [len(bge.encode(r["caption"], truncation=False)) for r in inputs]
    probe_counts = [len(bge.encode(t, truncation=False)) for t in probe_texts]
    check("BGE_caption_tokens_verified_locally", bge_counts == result["text_encoder"]["caption_token_counts"])
    check("BGE_probe_tokens_verified_locally", probe_counts == result["text_encoder"]["probe_token_counts"])
    check("BGE_no_input_exceeds_limit", max(bge_counts + probe_counts) <= result["text_encoder"]["max_seq_length"])
    clip_limit = result["clip_caption_control"]["context_limit"]
    clip_probe_counts = [len(clip.encode(t, truncation=False)) for t in probe_texts]
    check("CLIP_all_probes_within_limit", max(clip_probe_counts) <= clip_limit)
    for i, (record, chunked) in enumerate(zip(inputs, result["clip_caption_control"]["captions"])):
        chunks = chunked["chunks"]
        check(f"caption{i}:chunk_text_covers_original_exactly", " ".join(chunks) == record["caption"])
        full_ids = clip.encode(record["caption"], add_special_tokens=False, truncation=False)
        chunk_ids = [clip.encode(c, add_special_tokens=False, truncation=False) for c in chunks]
        check(f"caption{i}:chunk_tokens_cover_original_exactly", sum(chunk_ids, []) == full_ids)
        check(f"caption{i}:content_counts_verified",
              [len(ids) for ids in chunk_ids] == chunked["content_tokens_per_chunk"])
        with_special = [len(clip.encode(c, truncation=False)) for c in chunks]
        check(f"caption{i}:chunk_limits_verified",
              with_special == chunked["chunk_tokens_with_special_tokens"] and max(with_special) <= clip_limit)
        check(f"caption{i}:full_token_count_verified",
              len(clip.encode(record["caption"], truncation=False)) ==
              chunked["full_caption_tokens_with_special_tokens"])
        chunk_vectors = np.asarray(chunked["chunk_vectors"])
        close(f"caption{i}:chunk_vector_unit_norms", np.linalg.norm(chunk_vectors, axis=1), np.ones(len(chunks)))
        aggregate = normalized(np.average(chunk_vectors, weights=[len(x) for x in chunk_ids], axis=0))
        close(f"caption{i}:weighted_chunk_aggregate_recomputed",
              aggregate, arrays["clip_caption_chunk_mean_control_embeddings"][i])
    close("common_CLIP_residual_alignment_recomputed",
          normalized(arrays["clip_image_residual_005_minus_006"]) @
          normalized(arrays["clip_caption_chunk_mean_control_residual_005_minus_006"]),
          result["clip_caption_control"]["residual_alignment_with_image_cosine"])
    passed = sum(int(row["passed"]) for row in CHECKS)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha(Path(__file__)),
        "scope": "Independent read-only numerical/provenance/tokenizer audit of saved experiment artifacts; no encoder model inference, no downloads.",
        "checks_passed": passed, "checks_total": len(CHECKS),
        "all_checks_pass": passed == len(CHECKS),
        "checks": CHECKS, "feature_variant_summary": feature_summary,
        "interpretation_caveats": [
            "Only two examples: rank-one residual SVD measures pair separation, not semantic-detail count or model fidelity.",
            "The centered two-point cloud is exactly representable by one affine principal component, so this does not measure prior 2D memory truncation loss.",
            "CLIP/BGE residual magnitudes and probe margins are not cross-model calibrated.",
            "Two paraphrases of four primary features are repeated diagnostics, not eight independent examples.",
            "Positive pair-relative margin need not mean both individual examples select their own correct prompt.",
            "BGE sees generated captions; caption omissions and errors cannot be attributed solely to the text encoder.",
            "CLIP full-caption control is sentence-chunk pooling with full token coverage, not standard single-pass CLIP text encoding.",
            "Frozen probe hashes and source order support this run's freeze claim; static artifacts alone cannot prove no earlier exploratory runs or subsequent external tampering ever occurred.",
            "Source hashes verify consistency with recorded artifacts, not an independently witnessed before/after history; source code writes experiment paths only.",
            "No encoder inference was repeated in this audit; encoder outputs are checked for internal consistency and checkpoint/source provenance.",
        ],
    }
    (HERE / "independent_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks_passed": passed, "checks_total": len(CHECKS),
                      "all_checks_pass": report["all_checks_pass"],
                      "failed": [c for c in CHECKS if not c["passed"]]}, ensure_ascii=False))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
