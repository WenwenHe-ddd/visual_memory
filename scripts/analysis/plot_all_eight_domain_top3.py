"""Plot six Top-3 interference curves across all eight MemEye domains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
OUT_RAW = ROOT / "reports/all_eight_domain_top3_interference_20261101"
OUT_ZSCORE = ROOT / "reports/all_eight_domain_top3_interference_zscore_20261103"
DOMAINS = (
    ("Brand", "Brand_Memory_Test", 25, "brand_top3_interference_15_20261101"),
    ("Card", "Card_Playlog_Test", 48, "card_top3_interference_15_20261101"),
    ("Cartoon", "Cartoon_Entertainment_Companion", 76, "cartoon_top3_interference_15_20261101"),
    ("Home", "Home_Renovation_Interior_Design", 52, "home_top3_interference_15_20261031"),
    ("Cross-Scene", "Multi-Scene_Visual_Case_Archive_Assistant", 50, "multiscene_top3_interference_15_20261101"),
    ("Outdoor", "Outdoor_Navigation_Route_Memory_Assistant", 28, "outdoor_top3_interference_15_20261101"),
    ("Health", "Personal_Health_Dashboard_Assistant", 51, "health_top3_interference_15_20261101"),
    ("Social", "Social_Chat_Memory_Test", 36, "social_top3_interference_15_20261101"),
)
SERIES = (
    ("similar", "text", "Text-Similar", "#B7A625", "o", "-"),
    ("similar", "image", "Visual-Similar", "#185C40", "s", "-"),
    ("unrelated", "text", "Text-Dissimilar", "#B7A625", "^", "--"),
    ("unrelated", "image", "Visual-Dissimilar", "#185C40", "D", "--"),
)
CVM_SERIES = (
    ("similar", "caption_visual_master", "CVM-Similar", "#B13C3C", "P", "-"),
    ("unrelated", "caption_visual_master", "CVM-Dissimilar", "#B13C3C", "X", "--"),
)
CVM_ZSCORE_SERIES = (
    ("similar", "caption_visual_master_zscore_l0p03_a1", "CVM+z-score Similar", "#B13C3C", "P", "-"),
    ("unrelated", "caption_visual_master_zscore_l0p03_a1", "CVM+z-score Dissimilar", "#B13C3C", "X", "--"),
)
CVM_RESULTS = {
    "Brand": "brand_top3_interference_15_cvm_20261103",
    "Card": "card_top3_interference_15_cvm_20261103",
    "Cartoon": "cartoon_top3_interference_cvm_zscore_20261101",
    "Home": "home_top3_interference_15_cvm_20261031",
    "Cross-Scene": "multiscene_top3_interference_15_cvm_20261103",
    "Outdoor": "outdoor_top3_interference_15_cvm_20261103",
    "Health": "health_top3_interference_15_cvm_20261103",
    "Social": "social_top3_interference_15_cvm_20261103",
}
CVM_ZSCORE_RESULTS = {**CVM_RESULTS, "Home": "home_top3_interference_15_cvm_zscore_l0p03_a1_20261031"}


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(variant: str = "raw") -> None:
    if variant not in {"raw", "zscore"}:
        raise ValueError(f"Unknown CVM variant: {variant}")
    out = OUT_RAW if variant == "raw" else OUT_ZSCORE
    cvm_series = CVM_SERIES if variant == "raw" else CVM_ZSCORE_SERIES
    cvm_results = CVM_RESULTS if variant == "raw" else CVM_ZSCORE_RESULTS
    curves = []
    overview = []
    for label, task, count, result_dir in DOMAINS:
        path = ROOT / "results" / result_dir / "summary.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            summary = list(csv.DictReader(stream))
        by_key = {
            (row["condition"], row["insertion_method"], int(row["distractors_added"])): row
            for row in summary
        }
        record = {"domain": label, "task": task, "questions": count}
        for condition, method, name, _, _, _ in SERIES:
            for n in range(16):
                row = by_key[condition, method, n]
                if int(row["questions"]) != count:
                    raise ValueError(f"Question count changed: {task}, N={n}")
                curves.append({
                    "domain": label, "task": task, "questions": count,
                    "condition": condition, "insertion_method": method,
                    "series": name, "distractors": n,
                    "mean_weighted_rank_score_x100": float(row["mean_rank_score"]) * 100,
                    "gold_in_top3_rate_x100": float(row["gold_in_top3_rate"]) * 100,
                })
            for n in (1, 5, 10, 15):
                key = {
                    ("similar", "text"): "similar_text",
                    ("similar", "image"): "similar_image",
                    ("unrelated", "text"): "unrelated_text",
                    ("unrelated", "image"): "unrelated_image",
                }[condition, method]
                record[f"n{n}_{key}"] = float(
                    by_key[condition, method, n]["mean_rank_score"]
                ) * 100
        cvm_path = ROOT / "results" / cvm_results[label] / "summary.csv"
        with cvm_path.open(encoding="utf-8-sig", newline="") as stream:
            cvm_summary = list(csv.DictReader(stream))
        cvm_by_key = {
            (row["condition"], row["insertion_method"], int(row["distractors_added"])): row
            for row in cvm_summary
        }
        for condition, method, name, _, _, _ in cvm_series:
            for n in range(16):
                row = cvm_by_key[condition, method, n]
                if int(row["questions"]) != count:
                    raise ValueError(f"{label} CVM question count changed: N={n}")
                curves.append({
                    "domain": label, "task": task, "questions": count,
                    "condition": condition, "insertion_method": method,
                    "series": name, "distractors": n,
                    "mean_weighted_rank_score_x100": float(row["mean_rank_score"]) * 100,
                    "gold_in_top3_rate_x100": float(row["gold_in_top3_rate"]) * 100,
                })
            key = "cvm_similar" if condition == "similar" else "cvm_dissimilar"
            for n in (1, 5, 10, 15):
                record[f"n{n}_{key}"] = float(
                    cvm_by_key[condition, method, n]["mean_rank_score"]
                ) * 100
        overview.append(record)
    if len(curves) != 8 * 6 * 16 or sum(row["questions"] for row in overview) != 366:
        raise ValueError("Eight-domain cohort changed")
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "all_eight_curves.csv", curves)
    save_csv(out / "eight_domain_summary.csv", overview)
    (out / "plot_protocol.json").write_text(json.dumps({
        "cvm_variant": variant,
        "cvm_parameters": {"lambda": 0.03, "alpha": 1.0} if variant == "zscore" else None,
        "cvm_role": "candidate insertion order only",
        "retrieval": "question-to-original-caption frozen BGE cosine, Top-3 for every series",
        "qa_reader_run": False,
        "cvm_result_directories": cvm_results,
    }, ensure_ascii=False, indent=2) + "\n")

    font_dir = os.environ.get("MEMEYE_TIMES_FONT_DIR")
    if not font_dir:
        candidate = Path.home() / ".local/share/fonts/TimesNewRoman"
        font_dir = str(candidate) if candidate.is_dir() else None
    if font_dir:
        for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
            path = Path(font_dir) / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing Times New Roman style: {path}")
            font_manager.fontManager.addfont(str(path))
    try:
        font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
        font_name = "Times New Roman"
    except ValueError:
        font_name = "STIXGeneral"
        font_path = font_manager.findfont(font_name, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": font_name, "font.size": 12,
        "axes.linewidth": 1.75, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    (out / "font_provenance.json").write_text(json.dumps({
        "requested_font": "Times New Roman",
        "rendered_font": font_name,
        "rendered_font_path": font_path,
        "rendered_font_sha256": hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
        "font_distribution": "Microsoft Core Fonts for the Web, times32.exe",
        "source_archive_sha256": "db56595ec6ef5d3de5c24994f001f03b2a13e37cee27bc25c58f6f43e8f807ab",
        "note": "Exact Times New Roman is unavailable on this host" if font_name != "Times New Roman" else "",
    }, ensure_ascii=False, indent=2) + "\n")
    fig, axs = plt.subplots(2, 4, figsize=(18.6, 8.2), sharex=True, sharey=False, dpi=200)
    for ax, (label, _, _, _) in zip(axs.flat, DOMAINS):
        domain_values = [
            row["mean_weighted_rank_score_x100"] for row in curves
            if row["domain"] == label and row["distractors"] > 0
        ]
        lower = math.floor(min(domain_values) / 10) * 10
        upper = math.ceil(max(domain_values) / 10) * 10
        if lower == upper:
            lower = max(0, lower - 10)
        series = SERIES + cvm_series
        for condition, method, name, color, marker, linestyle in series:
            selected = [
                row for row in curves
                if row["domain"] == label and row["condition"] == condition
                and row["insertion_method"] == method and row["distractors"] > 0
            ]
            ax.plot(
                [row["distractors"] for row in selected],
                [row["mean_weighted_rank_score_x100"] for row in selected],
                color=color, marker=marker, markersize=4.7,
                linewidth=2.3, linestyle=linestyle, label=name,
            )
        ax.set_title(label, fontsize=14.5, pad=8)
        ax.set_xlim(.7, 15.3)
        ax.set_ylim(lower - 0.5, upper + 0.5)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_yticks(range(lower, upper + 1, 10))
        ax.grid(axis="y", color="#D7D7D7", linestyle=(0, (4, 3)),
                linewidth=1.3, alpha=.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#888888")
        ax.tick_params(axis="both", width=1.0, length=5)
    for ax in axs[1]:
        ax.set_xlabel("Number of distractors")
    for ax in axs[:, 0]:
        ax.set_ylabel("Target-Rank Score")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .006),
               ncol=3, frameon=False, handlelength=2.2, columnspacing=2.2,
               prop={"size": 11, "weight": "bold"})
    fig.subplots_adjust(left=.065, right=.99, top=.93, bottom=.195, wspace=.18, hspace=.26)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            out / f"eight_domain_top3_interference.{extension}",
            dpi=300, facecolor="white",
        )
    plt.close(fig)
    print(f"{out / 'eight_domain_top3_interference.png'} · {len(curves)} points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cvm-variant", choices=("raw", "zscore"), default="raw")
    main(parser.parse_args().cvm_variant)
