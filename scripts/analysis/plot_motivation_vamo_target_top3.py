"""Plot Top-3 curves after VAMO target-to-candidate selection and BGE ordering."""

from __future__ import annotations

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
BASE = ROOT / "reports/all_eight_domain_top3_interference_20261101/all_eight_curves.csv"
VAMO = ROOT / "results/vamo_target_selection_top3_20261114"
OUT = ROOT / "reports/motivation_vamo_target_selection_20261114"
DOMAINS = (
    ("Brand", "brand", 25), ("Card", "card", 48),
    ("Cartoon", "cartoon", 76), ("Home", "home", 52),
    ("Cross-Scene", "multiscene", 50), ("Outdoor", "outdoor", 28),
    ("Health", "health", 51), ("Social", "social", 36),
)
SERIES = (
    ("Text-Similar", "Text-Similar Disruption Selection", "#B7A625", "o", "-"),
    ("Visual-Similar", "Visul-Similar Disruption Selection", "#185C40", "s", "-"),
    ("Text-Dissimilar", "Text-Dissimilar", "#A6A6A6", "^", "--"),
    ("VAMO", "VAMO Disruption Selection", "#B13C3C", "*", "-"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    baseline = {(row["domain"], row["series"], int(row["distractors"])): row
                for row in read_csv(BASE)}
    records = []
    for domain, slug, count in DOMAINS:
        for source, label, _, _, _ in SERIES[:3]:
            for n in range(16):
                row = baseline[domain, source, n]
                if int(row["questions"]) != count:
                    raise ValueError(f"Baseline question count changed: {domain}")
                records.append({**row, "series": label})
        summary = {(row["condition"], row["insertion_method"],
                    int(row["distractors_added"])): row
                   for row in read_csv(VAMO / slug / "evaluation/summary.csv")}
        for n in range(16):
            row = summary["similar", "vamo_target_text", n]
            if int(row["questions"]) != count:
                raise ValueError(f"VAMO question count changed: {domain} N={n}")
            records.append({"domain": domain, "task": baseline[domain, "Text-Similar", n]["task"],
                            "questions": count, "condition": "similar",
                            "insertion_method": "vamo_target_text",
                            "series": "VAMO Disruption Selection", "distractors": n,
                            "mean_weighted_rank_score_x100": float(row["mean_rank_score"]) * 100,
                            "gold_in_top3_rate_x100": float(row["gold_in_top3_rate"]) * 100})
        endpoints = [float(record["mean_weighted_rank_score_x100"])
                     for record in records if record["domain"] == domain
                     and record["distractors"] == 0]
        if max(endpoints) - min(endpoints) > 1e-8:
            raise ValueError(f"Zero-distractor scores differ: {domain}")
    if len(records) != 8 * 4 * 16 or sum(n for _, _, n in DOMAINS) != 366:
        raise ValueError("Eight-domain chart scope changed")
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT / "motivation_curves.csv", records)

    font_dir = os.environ.get("MEMEYE_TIMES_FONT_DIR") or str(
        Path.home() / ".local/share/fonts/TimesNewRoman")
    for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"):
        font = Path(font_dir) / name
        if not font.is_file():
            raise FileNotFoundError(f"Times New Roman font missing: {font}")
        font_manager.fontManager.addfont(str(font))
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.linewidth": 1.75, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    (OUT / "protocol.json").write_text(json.dumps({
        "cohort_questions": 366, "domains": [d for d, _, _ in DOMAINS],
        "candidate_group": "VAMO-reconstructed image-to-image cosine selects the top 15 from the full task image pool",
        "vamo_representation": "saved 4096D reconstructed VAE latents, L2 normalized",
        "vamo_ordering": "question-to-original-caption BGE cosine among VAMO-selected candidates, descending",
        "retrieval": "unchanged original-caption BGE Top-3 within target-plus-N pool",
        "score": "best target rank 1/2/3/4+ -> 100/66.67/33.33/0; mean over questions",
        "qa_reader_run": False, "human_review": "new VAMO-selected candidate sets not individually reviewed",
        "baseline_data": str(BASE), "vamo_result_root": str(VAMO),
        "font": str(font_path), "font_sha256": hashlib.sha256(Path(font_path).read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=2) + "\n")

    fig, axs = plt.subplots(2, 4, figsize=(18.6, 8.2), sharex=True, dpi=200)
    for ax, (domain, _, _) in zip(axs.flat, DOMAINS):
        vals = [float(r["mean_weighted_rank_score_x100"]) for r in records
                if r["domain"] == domain and int(r["distractors"]) > 0]
        lower = math.floor(min(vals) / 10) * 10
        upper = math.ceil(max(vals) / 10) * 10
        if lower == upper:
            lower = max(0, lower - 10)
        for _, name, color, marker, style in SERIES:
            points = sorted((r for r in records if r["domain"] == domain
                             and r["series"] == name and int(r["distractors"]) > 0),
                            key=lambda r: int(r["distractors"]))
            kwargs = {"color": color, "marker": marker,
                      "linestyle": style, "linewidth": 2.3,
                      "markersize": 9.3 if marker == "*" else 5.1,
                      "label": name}
            if marker == "^":
                kwargs.update(markerfacecolor="#7F807F", markeredgecolor="#7F807F")
            ax.plot([int(r["distractors"]) for r in points],
                    [float(r["mean_weighted_rank_score_x100"]) for r in points],
                    **kwargs)
        ax.set_title(domain, fontsize=14.5, pad=8)
        ax.set_xlim(.7, 15.3)
        ax.set_ylim(lower - .5, upper + .5)
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
               ncol=2, frameon=False, handlelength=2.2, columnspacing=2.2,
               prop={"size": 11, "weight": "bold"})
    fig.subplots_adjust(left=.065, right=.99, top=.93, bottom=.20,
                        wspace=.18, hspace=.26)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"motivation_vamo_target_top3.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"{OUT / 'motivation_vamo_target_top3.png'} · {len(records)} points")


if __name__ == "__main__":
    main()
