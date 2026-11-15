"""PCA of all pairwise residuals for the five prepared Q10 image/caption pairs."""
import csv
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "reports/home_q10_top5_embeddings_20261020"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualmemory_q10_residual_pca_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pca(x):
    """Conventional feature-centered PCA via SVD; no feature standardization."""
    centered = x - x.mean(axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2 / (len(x) - 1)
    ratio = variance / variance.sum()
    tol = max(centered.shape) * np.finfo(centered.dtype).eps * singular[0]
    rank = int(np.sum(singular > tol))
    return {
        "mean": x.mean(axis=0), "scores": u * singular, "components": vt,
        "singular_values": singular, "explained_variance": variance,
        "explained_variance_ratio": ratio, "cumulative": np.cumsum(ratio), "rank": rank,
    }


def summarize(result):
    p = result["explained_variance_ratio"][:result["rank"]]
    return {
        "rank": result["rank"],
        "explained_variance_ratio": result["explained_variance_ratio"].tolist(),
        "cumulative_explained_variance": result["cumulative"].tolist(),
        "singular_values": result["singular_values"].tolist(),
        "explained_variance": result["explained_variance"].tolist(),
        "effective_rank_entropy": float(np.exp(-np.sum(p * np.log(p)))),
    }


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Completed analysis exists; use a new output directory.")
    source_validation = json.loads((SOURCE / "validation.json").read_text())
    assert sha(SOURCE / "embeddings.npz") == source_validation["artifact_sha256"]["embeddings.npz"]
    inputs = json.loads((SOURCE / "inputs.json").read_text())
    with np.load(SOURCE / "embeddings.npz", allow_pickle=False) as z:
        names = z["filenames"].tolist()
        ids = z["memory_ids"].tolist()
        originals = {"image": z["image_embeddings"].copy(), "caption": z["text_full_caption"].copy()}
    assert names == [r["filename"] for r in inputs]
    assert ids == [r["memory_id"] for r in inputs]
    assert originals["image"].shape == (5, 512)
    assert originals["caption"].shape == (5, 384)
    assert all(np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1.0,
                                                  rtol=0, atol=1e-12)
               for x in originals.values())
    pairs = list(itertools.combinations(range(5), 2))
    arrays = {"pair_row_indices": np.asarray(pairs), "filenames": np.asarray(names),
              "memory_ids": np.asarray(ids)}
    primary, oriented, source_pca, checks = {}, {}, {}, {}
    residuals = {}
    for modality, x in originals.items():
        # Ten unique pairs, each initially oriented by the saved retrieval order.
        residual = np.stack([x[i] - x[j] for i, j in pairs])
        # Both orientations make covariance independent of arbitrary pair signs.
        # The twenty rows remain dependent and do not increase the sample size.
        balanced = np.concatenate([residual, -residual], axis=0)
        residuals[modality] = residual
        main_pca, one_sign, original_pca = pca(balanced), pca(residual), pca(x)
        assert main_pca["rank"] == original_pca["rank"] == 4
        center = x - x.mean(axis=0)
        gram_error = float(np.max(np.abs(residual.T @ residual - len(x) * center.T @ center)))
        spectrum_error = float(np.max(np.abs(main_pca["explained_variance_ratio"][:4] -
                                             original_pca["explained_variance_ratio"][:4])))
        assert gram_error < 1e-12 and spectrum_error < 1e-12
        perm = np.array([2, 4, 0, 3, 1])
        xp = x[perm]
        rp = np.stack([xp[i] - xp[j] for i, j in pairs])
        perm_pca = pca(np.concatenate([rp, -rp], axis=0))
        permutation_error = float(np.max(np.abs(perm_pca["explained_variance_ratio"][:4] -
                                                main_pca["explained_variance_ratio"][:4])))
        assert permutation_error < 1e-12
        for key, value in [
            ("residuals", residual), ("balanced_residuals", balanced),
            ("pca_mean", main_pca["mean"]), ("pca_scores", main_pca["scores"]),
            ("pca_components", main_pca["components"]),
            ("pca_singular_values", main_pca["singular_values"]),
            ("pca_explained_variance_ratio", main_pca["explained_variance_ratio"]),
        ]:
            arrays[f"{modality}_{key}"] = value
        primary[modality] = summarize(main_pca)
        oriented[modality] = summarize(one_sign)
        source_pca[modality] = summarize(original_pca)
        checks[modality] = {
            "all_pairs_gram_identity_abs_error": gram_error,
            "explained_variance_ratio_vs_original_samples_abs_error": spectrum_error,
            "permutation_spectrum_abs_error": permutation_error,
            "max_balanced_mean_abs": float(np.max(np.abs(main_pca["mean"]))),
            "rank": main_pca["rank"],
        }

    np.savez_compressed(OUT / "residuals_and_pca.npz", **arrays)
    with np.load(OUT / "residuals_and_pca.npz", allow_pickle=False) as z:
        assert all(np.array_equal(z[k], v) for k, v in arrays.items())
    pair_rows = []
    for k, (i, j) in enumerate(pairs):
        pair_rows.append({
            "residual_row_index": k, "rank_a": i + 1, "rank_b": j + 1,
            "image_a": names[i], "image_b": names[j],
            "memory_a": ids[i], "memory_b": ids[j], "residual_direction": "a_minus_b",
            "image_residual_l2": float(np.linalg.norm(residuals["image"][k])),
            "caption_residual_l2": float(np.linalg.norm(residuals["caption"][k])),
            "image_cosine": float(originals["image"][i] @ originals["image"][j]),
            "caption_cosine": float(originals["caption"][i] @ originals["caption"][j]),
        })
    write_csv("residual_pairs.csv", pair_rows)
    curve_rows = [{
        "principal_components": k + 1,
        "image_explained_variance_ratio": primary["image"]["explained_variance_ratio"][k],
        "image_cumulative": primary["image"]["cumulative_explained_variance"][k],
        "caption_explained_variance_ratio": primary["caption"]["explained_variance_ratio"][k],
        "caption_cumulative": primary["caption"]["cumulative_explained_variance"][k],
    } for k in range(4)]
    write_csv("cumulative_variance.csv", curve_rows)
    save("results.json", {
        "n_original_samples": 5, "n_unique_pairs": 10, "n_signed_residual_rows": 20,
        "maximum_possible_rank": 4, "filenames_in_order": names,
        "primary_balanced_residual_pca": primary,
        "sensitivity_centered_single_orientation_residual_pca": oriented,
        "original_five_samples_centered_pca": source_pca,
        "interpretation": [
            "Within these five samples, image variance is more concentrated and caption variance more dispersed.",
            "Both modalities have four nonzero components; the observed difference is variance allocation.",
            "Curve shape alone cannot establish semantic detail, encoder quality, or generalization.",
            "All-pair residuals add no independent evidence: balanced residual PCA has the same explained variance ratios as PCA of the five centered original vectors.",
        ],
    })

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#344054", "text.color": "#243247",
        "xtick.color": "#344054", "ytick.color": "#344054", "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    fig.subplots_adjust(left=.11, right=.95, top=.79, bottom=.16)
    fig.text(.11, .94, "PCA of pairwise residuals", fontsize=20, weight="bold")
    fig.text(.11, .875, "5 matched images / full captions · 10 pairs · both residual directions",
             fontsize=11, color="#667085")
    colors = {"image": "#2563B9", "caption": "#D66C27"}
    labels = {"image": "Image · CLIP ViT-B/32", "caption": "Full caption · BGE"}
    pc = np.arange(1, 5)
    for modality, marker in [("image", "o"), ("caption", "s")]:
        values = np.asarray(primary[modality]["cumulative_explained_variance"][:4])
        ax.plot(pc, values, color=colors[modality], marker=marker, markersize=7,
                linewidth=2.5, label=labels[modality])
        for k in range(3):
            offset = 13 if modality == "image" else -19
            ax.annotate(f"{100 * values[k]:.2f}%", (pc[k], values[k]),
                        textcoords="offset points", xytext=(0, offset),
                        ha="center", color=colors[modality], fontsize=11, weight="medium")
    ax.annotate("100% (both)", (4, 1), xytext=(-3, 13), textcoords="offset points",
                ha="right", color="#344054", fontsize=11)
    ax.set(xlim=(.8, 4.2), ylim=(0, 1.10), xticks=pc,
           xlabel="Number of retained principal components",
           ylabel="Cumulative explained variance")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", color="#E8ECF2", linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.text(.11, .035, "Residuals are not re-normalized. Five source samples impose a maximum rank of 4.",
             fontsize=10, color="#667085")
    assert len(ax.lines) == 2
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"cumulative_explained_variance.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    save("protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(SOURCE / name): sha(SOURCE / name) for name in [
            "embeddings.npz", "inputs.json", "protocol.json", "validation_independent.json",
        ]},
        "script_sha256": sha(Path(__file__)),
        "selected_arrays": {"image": "image_embeddings", "caption": "text_full_caption"},
        "caption_source": "190/371 run's full caption only, excluding dialogue and unique caption.",
        "image_model": "CLIP ViT-B/32, 512-D", "caption_model": "BGE-small-en-v1.5, 384-D",
        "input_normalization": "Already L2-normalized, retained unchanged.",
        "pair_order": "itertools.combinations(range(5), 2); initial residual is x[i]-x[j].",
        "primary_pca": "Feature-centered PCA on [R; -R], R has all 10 unordered pair residuals.",
        "reason_for_both_signs": "Avoid an arbitrary pair orientation changing the residual mean and centered covariance.",
        "residual_normalization": None, "feature_standardization": None,
        "pca_implementation": "NumPy SVD of column-centered data, variance=s^2/(n_rows-1).",
        "explained_variance_denominator": "Sum over all singular values, without truncating before normalization.",
        "plotted_components": [1, 2, 3, 4], "plotted_curve_count": 2,
        "mathematical_identity": "R.T@R = n*X_centered.T@X_centered; balanced residual EVR equals original centered sample EVR.",
        "sensitivity": "Ordinary centered PCA of the 10 singly oriented residuals is saved separately, not plotted.",
    })
    save("validation.json", {
        "status": "passed", "source_bundle_hash_matches": True,
        "normalization_and_dimensions_checked": True, "residual_pairs": len(pairs),
        "npz_roundtrip_exact": True, "checks": checks, "plotted_curves": 2,
        "output_sha256": {name: sha(OUT / name) for name in [
            "residuals_and_pca.npz", "residual_pairs.csv", "cumulative_variance.csv",
            "results.json", "protocol.json", "cumulative_explained_variance.png",
            "cumulative_explained_variance.svg", "cumulative_explained_variance.pdf",
        ]},
    })
    print(json.dumps({"curves": curve_rows, "validation": checks, "output": str(OUT)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
