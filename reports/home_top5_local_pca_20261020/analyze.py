"""Compare within-Top5 concentration: independent PCA per QA, equal-QA means."""
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vendor"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualmemory_home_local_pca")
import numpy as np

DOMAIN = "Home_Renovation_Interior_Design"
from scripts.analysis import frozen_comparison_inputs as frozen

RUN = ROOT / "data/independent_visual_inputs"
NAT = ROOT / "results/natural_observation"
PRIOR = ROOT / "reports/home_q10_top5_embeddings_20261020"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_save(name, values):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def unit(values):
    x = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    assert np.isfinite(x).all() and (norms > 0).all()
    return x / norms


def spectrum(x):
    centered = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular ** 2
    if energy.sum() <= np.finfo(np.float64).eps ** 2:
        raise ValueError("Zero within-set variance; do not invent an explained-variance curve.")
    tol = max(centered.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int((singular > tol).sum())
    assert 1 <= rank <= 4
    ratio = energy / energy.sum()
    probabilities = ratio[:rank]
    return {
        "rank": rank,
        "singular_values": singular.tolist(),
        "explained_variance_ratio": ratio[:4].tolist(),
        "cumulative": np.cumsum(ratio[:4]).tolist(),
        "total_centered_energy": float(energy.sum()),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
    }


def summarize(spectra):
    cumulative = np.array([r["cumulative"] for r in spectra])
    effective = np.array([r["effective_rank"] for r in spectra])
    return {
        "n_questions": len(spectra),
        "mean_cumulative": cumulative.mean(axis=0).tolist(),
        "median_cumulative": np.median(cumulative, axis=0).tolist(),
        "p25_cumulative": np.quantile(cumulative, .25, axis=0).tolist(),
        "p75_cumulative": np.quantile(cumulative, .75, axis=0).tolist(),
        "mean_effective_rank": float(effective.mean()),
        "median_effective_rank": float(np.median(effective)),
        "ranks": [r["rank"] for r in spectra],
    }


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Completed analysis already exists.")
    import torch
    from sentence_transformers import SentenceTransformer
    from scripts.graph.visual_image_encoder import ImageEncoder

    torch.set_num_threads(4)
    run_protocol = json.loads((RUN / "protocol.json").read_text())
    natural_protocol = json.loads((NAT / "protocol.json").read_text())
    prior_protocol = json.loads((PRIOR / "protocol.json").read_text())
    questions = [q for q in frozen.questions() if q["task"] == DOMAIN]
    assert len(questions) == 52 and len({q["id"] for q in questions}) == 52
    assert all(len(q["top_memory_ids"]) == len(set(q["top_memory_ids"])) == 5 for q in questions)
    ids = list(dict.fromkeys(mid for q in questions for mid in q["top_memory_ids"]))
    assert len(ids) == 71
    memories = {r["id"]: r for r in rows(NAT / "memories.jsonl")}
    images = {r["id"]: r for r in rows(NAT / "images.jsonl")}
    texts = {r["id"]: r for r in rows(RUN / "retrieval_memories.jsonl")}
    nodes = frozen.memories()
    inputs = []
    for mid in ids:
        image_hash, = memories[mid]["images"]
        path = Path(images[image_hash]["path"])
        assert path.parent.name == DOMAIN and sha(path) == image_hash
        caption = texts[mid]["reader_text"].split("\nMemory description: ", 1)[1]
        assert caption == nodes[mid]["caption"]
        inputs.append({
            "memory_id": mid, "image_path": str(path), "filename": path.name,
            "image_sha256": image_hash, "full_caption": caption,
            "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
        })
    assert len({r["image_sha256"] for r in inputs}) == 71
    question_manifest = []
    for q in questions:
        for mid in q["top_memory_ids"]:
            assert texts[mid]["reader_text"] in q["reader_text"]
        question_manifest.append({k: q[k] for k in (
            "id", "task", "question", "top_memory_ids", "payload_sha256")})
    save("inputs.json", inputs)
    save("question_manifest.json", question_manifest)

    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision=prior_protocol["image_encoder"]["revision"],
                           device="cpu", local_files_only=True)
    assert encoder.metadata["files_sha256"] == prior_protocol["image_encoder"]["files_sha256"]
    visual_metadata = encoder.metadata
    image_vectors = []
    for i, record in enumerate(inputs, 1):
        image_vectors.append(encoder.encode(record["image_path"]))
        if i % 20 == 0 or i == len(inputs):
            print(f"Images encoded: {i}/{len(inputs)}", flush=True)
    image_vectors = unit(image_vectors)
    del encoder

    text_path = Path(natural_protocol["retriever_model_path"])
    assert str(text_path) == prior_protocol["text_encoder"]["model_path"]
    text_hashes = prior_protocol["text_encoder"]["files_sha256"]
    assert all(sha(text_path / name) == digest for name, digest in text_hashes.items())
    encoder = SentenceTransformer(str(text_path), device="cpu", local_files_only=True)
    captions = [r["full_caption"] for r in inputs]
    token_lengths = [len(encoder.tokenizer.encode(c, truncation=False)) for c in captions]
    assert max(token_lengths) <= encoder.max_seq_length
    text_vectors = unit(encoder.encode(captions, batch_size=16, normalize_embeddings=True,
                                      show_progress_bar=False))
    text_max_seq_length = encoder.max_seq_length
    del encoder
    assert image_vectors.shape == (71, 512) and text_vectors.shape == (71, 384)
    print("Full captions encoded without truncation.", flush=True)
    for record, length in zip(inputs, token_lengths):
        record["caption_tokens"] = length
    save("inputs.json", inputs)

    index = {mid: i for i, mid in enumerate(ids)}
    spectra = {"image": [], "caption": []}
    grams = {"image": [], "caption": []}
    validation = {"status": "passed", "qa_count": 52, "distinct_memory_count": 71,
                  "caption_truncations": 0, "per_image_embeddings_persisted": False,
                  "original_images_copied": False}
    question_rows = []
    for q in questions:
        indices = [index[mid] for mid in q["top_memory_ids"]]
        result = {"id": q["id"], "top_memory_ids": q["top_memory_ids"]}
        for modality, array in (("image", image_vectors), ("caption", text_vectors)):
            x = array[indices]
            spec = spectrum(x)
            spectra[modality].append(spec)
            grams[modality].append(x @ x.T)
            result[modality] = spec
        result["caption_minus_image_first_two"] = result["caption"]["cumulative"][1] - result["image"]["cumulative"][1]
        question_rows.append(result)

    # Check the previously measured five-image case against freshly encoded data.
    q10_index = next(i for i, q in enumerate(questions) if q["id"].endswith("/Q10"))
    q10_ids = questions[q10_index]["top_memory_ids"]
    with np.load(PRIOR / "embeddings.npz", allow_pickle=False) as old:
        assert old["memory_ids"].tolist() == q10_ids
        errors = {}
        for modality, array, key in (
                ("image", image_vectors, "image_embeddings"),
                ("caption", text_vectors, "text_full_caption")):
            x = array[[index[mid] for mid in q10_ids]]
            errors[modality] = float(np.max(np.abs(x - old[key])))
            assert errors[modality] < 2e-6
    old_pca = json.loads((ROOT / "reports/home_q10_residual_pca_20261020/results.json").read_text())
    curve_errors = {}
    for modality in spectra:
        curve_errors[modality] = float(np.max(np.abs(
            np.array(spectra[modality][q10_index]["cumulative"]) -
            np.array(old_pca["primary_balanced_residual_pca"][modality]["cumulative_explained_variance"][:4]))))
        assert curve_errors[modality] < 2e-6
    validation["q10_max_embedding_abs_errors_vs_prior"] = errors
    validation["q10_max_cumulative_abs_errors_vs_prior"] = curve_errors

    # Gram matrices retain pairwise similarities for an independent spectral audit.
    # No individual full-dimensional embedding is written.
    np.savez_compressed(OUT / "grams.npz", image=np.array(grams["image"]),
                        caption=np.array(grams["caption"]),
                        question_ids=np.array([q["id"] for q in questions]),
                        top_memory_ids=np.array([q["top_memory_ids"] for q in questions]))
    unique_indices = []
    seen = set()
    duplicates = {}
    for i, q in enumerate(questions):
        key = tuple(sorted(q["top_memory_ids"]))
        duplicates.setdefault(key, []).append(q["id"])
        if key not in seen:
            seen.add(key)
            unique_indices.append(i)
    assert len(unique_indices) == 50
    aggregate = {m: summarize(values) for m, values in spectra.items()}
    deduplicated = {m: summarize([values[i] for i in unique_indices]) for m, values in spectra.items()}
    delta = np.array([q["caption_minus_image_first_two"] for q in question_rows])
    # A numerical tolerance is only used to label effectively equal results.
    results = {
        "domain": DOMAIN, "n_questions": len(questions), "n_unique_top5_sets": len(unique_indices),
        "n_unique_memories": len(ids), "aggregate": aggregate,
        "deduplicated_set_sensitivity": deduplicated,
        "duplicate_sets": [v for v in duplicates.values() if len(v) > 1],
        "paired_first_two_caption_minus_image": {
            "mean": float(delta.mean()), "median": float(np.median(delta)),
            "p25": float(np.quantile(delta, .25)), "p75": float(np.quantile(delta, .75)),
            "image_more_concentrated_count": int((delta < -1e-8).sum()),
            "caption_more_concentrated_count": int((delta > 1e-8).sum()),
            "numerical_tie_count": int((np.abs(delta) <= 1e-8).sum()),
            "count_tolerance": 1e-8,
        },
        "per_question": question_rows,
        "interpretation": "Average of independently fitted local PCA spectra, one equal vote per QA. "
                          "PC indices are local ranks, not shared axes. IQR describes variation "
                          "among the saved questions, not statistical confidence or independence. "
                          "Curve flatness describes dispersion of representation differences, "
                          "not semantic detail quality. CLIP and BGE have different training objectives.",
    }
    save("results.json", results)
    flat_rows = []
    for q in question_rows:
        flat = {"question_id": q["id"], "top_memory_ids": " | ".join(q["top_memory_ids"])}
        for modality in spectra:
            flat.update({f"{modality}_cumulative_pc{k + 1}": value
                         for k, value in enumerate(q[modality]["cumulative"])})
            flat[f"{modality}_effective_rank"] = q[modality]["effective_rank"]
            flat[f"{modality}_rank"] = q[modality]["rank"]
        flat["caption_minus_image_first_two"] = q["caption_minus_image_first_two"]
        flat_rows.append(flat)
    csv_save("per_question.csv", flat_rows)
    csv_save("mean_curves.csv", [{
        "principal_components": k + 1,
        **{f"{m}_{stat}": aggregate[m][stat][k] for m in spectra
           for stat in ("mean_cumulative", "median_cumulative", "p25_cumulative", "p75_cumulative")}
    } for k in range(4)])

    source_paths = [RUN / name for name in ("protocol.json", "qa_manifest.jsonl", "retrieval_memories.jsonl")]
    source_paths += [NAT / name for name in ("protocol.json", "memories.jsonl", "images.jsonl")]
    source_paths += [frozen.COMPARISON, PRIOR / "protocol.json", PRIOR / "embeddings.npz",
                     ROOT / "scripts/graph/visual_image_encoder.py"]
    save("protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "domain": DOMAIN,
        "source_run": str(RUN), "source_experiment_signature": run_protocol["experiment_signature"],
        "source_sha256": {str(p): sha(p) for p in source_paths}, "script_sha256": sha(Path(__file__)),
        "image_encoder": visual_metadata, "text_encoder": prior_protocol["text_encoder"],
        "caption_max_tokens_observed": max(token_lengths), "caption_max_seq_length": text_max_seq_length,
        "caption_truncations": 0, "text_input": "unchanged 190-run full caption, no dialogue/unique caption/prefix",
        "normalization": "L2 before local centering; no residual normalization, coordinate scaling or whitening",
        "pca": "Independent float64 SVD of each five-row centered matrix, separately for image/caption",
        "denominator": "Sum of all squared singular values before taking first four",
        "aggregation": "Arithmetic mean of 52 within-QA cumulative ratios; not pooled covariance",
        "sensitivity": "Repeat summary with one equal weight per distinct Top5 set (50)",
        "memory_policy": "Full image/text vectors exist in RAM only. Persist local 5x5 Gram matrices, "
                         "input provenance and derived spectra; no copied original images.",
    })
    plot(aggregate)
    validation["artifact_sha256"] = {name: sha(OUT / name) for name in (
        "inputs.json", "question_manifest.json", "grams.npz", "results.json", "protocol.json",
        "per_question.csv", "mean_curves.csv", "cumulative_explained_variance.png")}
    save("validation.json", validation)
    print(json.dumps({"aggregate": aggregate,
                      "paired_first_two": results["paired_first_two_caption_minus_image"],
                      "q10_validation": curve_errors}, ensure_ascii=False, indent=2), flush=True)


def plot(aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=.10, right=.96, top=.79, bottom=.21)
    x = np.arange(1, 5)
    for modality, label, color in (
            ("image", "Image · CLIP ViT-B/32", "#2563EB"),
            ("caption", "Full caption · BGE-small", "#E86F26")):
        stat = aggregate[modality]
        ax.fill_between(x, stat["p25_cumulative"], stat["p75_cumulative"], color=color, alpha=.12, linewidth=0)
        ax.plot(x, stat["mean_cumulative"], color=color, marker="o", linewidth=2.5,
                markersize=7, label=label)
    ax.set_xticks(x)
    ax.set_xlim(.9, 4.1)
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Number of local principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(axis="y", color="#E6E9EF")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Home: concentration of differences within retrieved Top5", fontsize=15, y=.95)
    fig.text(.10, .85, "Mean of 52 independently fitted QA spectra · Same Top5 for both modalities",
             fontsize=10.5, color="#475569")
    fig.text(.10, .08, "Shading: 25th–75th percentiles across questions (not confidence intervals).\n"
             "Each five-image set has at most 4 nonzero components; axes are fitted separately per QA.",
             fontsize=9.5, color="#64748B")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"cumulative_explained_variance.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
