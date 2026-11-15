"""Audit and summarize fixed-caption Top-3 interference in six MemEye domains."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/other_domain_top3_interference_20261101"
DOMAINS = (
    ("brand", "Brand"), ("card", "Card"), ("health", "Health"),
    ("multiscene", "Multi-Scene"), ("social", "Social"), ("outdoor", "Outdoor"),
)
SERIES = (
    ("similar", "text", "SemRAG (Disruption)", "#185C40", "o"),
    ("similar", "image", "Modality-Preserving (Disruption)", "#B7A626", "s"),
    ("unrelated", "text", "SemRAG (Irrelevant)", "#63869A", "^"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curves = []
    domain_rows = []
    for slug, label in DOMAINS:
        source = ROOT / "results" / f"{slug}_all_questions_distractor_review_20261101"
        result = ROOT / "results" / f"{slug}_top3_interference_15_20261101"
        prep = json.loads((source / "protocol.json").read_text())
        ordering = json.loads((source / "query_similarity_protocol.json").read_text())
        evaluation = json.loads((result / "protocol.json").read_text())
        plans = [json.loads(line) for line in (source / "candidate_order.jsonl").read_text().splitlines()
                 if line.strip()]
        candidate_rows = read_csv(source / "review_candidates.csv")
        question_summary = read_csv(source / "question_summary.csv")
        if not (prep["device"] == ordering["device"] == "cuda:0"):
            raise ValueError(f"GPU 0 provenance missing: {slug}")
        nq = len(plans)
        if (nq != prep["scope"]["questions"] or nq != evaluation["question_count"]
                or len(candidate_rows) != nq * 30 or evaluation["rows"] != nq * 2 * 2 * 16):
            raise ValueError(f"Question/candidate/result counts differ: {slug}")
        if evaluation["full_domain_pool_top15_score_equivalent_to_original_caption_baseline"] != nq:
            raise ValueError(f"Original caption baseline changed: {slug}")
        overlap_by_qid = {}
        for plan in plans:
            if len(plan["similar_ids"]) != 15 or len(plan["unrelated_ids"]) != 15:
                raise ValueError(f"Candidate length differs: {slug}")
            if (len(set(plan["similar_ids"])) != 15 or len(set(plan["unrelated_ids"])) != 15
                    or set(plan["gold_memory_ids"]) &
                    (set(plan["similar_ids"]) | set(plan["unrelated_ids"]))):
                raise ValueError(f"Candidate duplication or gold overlap: {slug}")
            overlap_by_qid[plan["question_id"]] = len(set(plan["similar_ids"]) &
                                                    set(plan["unrelated_ids"]))
        if {row["question_id"] for row in question_summary} != set(overlap_by_qid):
            raise ValueError(f"Question summary differs: {slug}")
        for row in question_summary:
            row["similar_unrelated_overlap_count"] = overlap_by_qid[row["question_id"]]
        write_csv(source / "question_summary.csv", question_summary)
        overlap_questions = sum(value > 0 for value in overlap_by_qid.values())
        prep["scope"]["questions_with_similar_unrelated_overlap"] = overlap_questions
        prep["scope"]["max_similar_unrelated_overlap"] = max(overlap_by_qid.values())
        (source / "protocol.json").write_text(json.dumps(prep, ensure_ascii=False, indent=2) + "\n")
        summary = read_csv(result / "summary.csv")
        indexed = {(row["condition"], row["insertion_method"], int(row["distractors_added"])):
                   row for row in summary}
        for condition, method, name, _, _ in SERIES:
            for n in range(16):
                row = indexed[condition, method, n]
                if int(row["questions"]) != nq:
                    raise ValueError(f"Curve cohort differs: {slug}, {name}, {n}")
                curves.append({"domain": label, "task": prep["study"].split(" original-image")[0],
                               "condition": condition, "insertion_method": method,
                               "series": name, "distractors": n, "questions": nq,
                               "mean_weighted_rank_score_x100": float(row["mean_rank_score"]) * 100,
                               "gold_in_top3_rate_x100": float(row["gold_in_top3_rate"]) * 100})
        item = {"domain": label, "task": prep["study"].split(" original-image")[0],
                "total_questions": prep["scope"]["total_domain_questions"],
                "eligible_questions": nq,
                "excluded_question_ids": " | ".join(prep["scope"]["excluded_question_ids_with_fewer_than_N_non_targets"]),
                "unique_images": prep["scope"]["memory_images"],
                "questions_with_candidate_group_overlap": overlap_questions,
                "max_candidate_group_overlap": max(overlap_by_qid.values()),
                "baseline_exact_id_matches": evaluation["full_domain_pool_top15_exact_id_matches_original_caption_baseline"],
                "baseline_score_equivalent_matches": evaluation["full_domain_pool_top15_score_equivalent_to_original_caption_baseline"]}
        for n in (1, 5, 10, 15):
            for condition, method, name, _, _ in SERIES:
                key = {("similar", "text"): "similar_text",
                       ("similar", "image"): "similar_image",
                       ("unrelated", "text"): "unrelated_text"}[condition, method]
                item[f"n{n}_{key}"] = float(indexed[condition, method, n]["mean_rank_score"]) * 100
        domain_rows.append(item)
    write_csv(OUT / "domain_summary.csv", domain_rows)
    write_csv(OUT / "all_curves.csv", curves)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.linewidth": 1.25, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 3, figsize=(15.2, 8.8), sharex=True, sharey=True, dpi=200)
    for ax, (_, label) in zip(axs.flat, DOMAINS):
        for condition, method, name, color, marker in SERIES:
            selected = [row for row in curves if row["domain"] == label
                        and row["condition"] == condition and row["insertion_method"] == method
                        and row["distractors"] > 0]
            ax.plot([row["distractors"] for row in selected],
                    [row["mean_weighted_rank_score_x100"] for row in selected],
                    color=color, marker=marker, markersize=3.6, linewidth=1.9, label=name)
        meta = next(row for row in domain_rows if row["domain"] == label)
        ax.set_title(f"{label} (n={meta['eligible_questions']})", fontsize=11.5)
        ax.set_xlim(.7, 15.3)
        ax.set_ylim(0, 101)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#BABDBB", linestyle=(0, (4, 3)), alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("Number of distractor images")
    for ax in axs[:, 0]:
        ax.set_ylabel("Mean weighted target-rank score")
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .01),
               frameon=False, fontsize=10)
    fig.subplots_adjust(left=.075, right=.985, top=.94, bottom=.12, wspace=.16, hspace=.25)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"six_domain_top3_interference.{ext}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    print(json.dumps({"domains": len(domain_rows), "curve_rows": len(curves),
                      "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
