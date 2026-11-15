#!/usr/bin/env python3
"""Aggregate completed adaptive-retrieval experiments without rerunning search.

All ratios use sums over the selected questions, so domains of different sizes
are compared to their actual full-scan work. Retrieval wall times are L=5 only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HOME = "Home_Renovation_Interior_Design"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/adaptive_retrieval_comparison_20261020"
METHOD_NAMES = {
    "full_scan": "全量 BGE",
    "spherical_cap_clip_order": "严格球冠界 · CLIP 顺序",
    "spherical_cap_best_bound": "严格球冠界 · 优先最高界",
    "pivot_tree_clip_order": "严格 pivot 树 · CLIP 顺序",
    "pivot_tree_best_bound": "严格 pivot 树 · 优先最高界",
    "progressive_text_pca": "严格渐进 PCA",
    "empirical_offset": "经验最大偏移",
    "empirical_radius_ratio": "经验半径比例",
    "empirical_max_both": "经验两者取大",
    "empirical_max_both_reuse": "经验两者取大 · 中心复用",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                          encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    left = int(position)
    right = min(left+1, len(values)-1)
    return values[left]+(values[right]-values[left])*(position-left)


def load_batch(output, relative, source_paths):
    directory = output / relative
    summary = read(directory / "summary.json")
    runs = read(directory / "runs.json")
    timings = read(directory / "latency_samples.json")
    domain = summary["domain"]
    input_path = output / "inputs_by_domain" / domain / "inputs.json"
    inputs = read(input_path)
    source_paths.update([directory/"summary.json", directory/"runs.json",
                         directory/"latency_samples.json", input_path])
    for source_name, expected_hash in summary.get("source_hashes", {}).items():
        source_path = Path(source_name)
        assert sha(source_path) == expected_hash, source_path
        source_paths.add(source_path)
    assert len(inputs["questions"]) == summary["n_questions"]
    assert len(inputs["leaves"]) == summary["n_leaves"]
    assert len(inputs["masters"]) == summary["n_masters"]
    timed = {}
    for method, samples in timings.items():
        by_query = defaultdict(list)
        for sample in samples:
            by_query[sample["question_index"]].append(sample["nanoseconds"]/1e6)
        assert set(by_query) == set(range(len(inputs["questions"])))
        for qi, values in by_query.items():
            timed[(method, qi)] = {
                "samples_ms": values, "median_ms": statistics.median(values),
                "repetitions": len(values),
            }
    seen = set()
    records = []
    for run in runs:
        qi, method, limit = run["question_index"], run["method"], run["L"]
        question = inputs["questions"][qi]
        assert question["id"] == run["id"]
        key = (method, limit, question["id"])
        assert key not in seen
        seen.add(key)
        selected = [inputs["leaves"][i] for i in run["selected_indices"]]
        reference = [inputs["leaves"][i] for i in run["reference_indices"]]
        selected_ids = {leaf["memory_id"] for leaf in selected}
        reference_ids = {leaf["memory_id"] for leaf in reference}
        gold = set(question["gold_ids"])
        assert len(selected) == len(selected_ids) == limit
        assert run["gold_hits"] == len(gold & selected_ids)
        assert run["gold_total"] == len(gold)
        assert run["exact_match"] == (run["selected_indices"] == run["reference_indices"])
        assert abs(run["topL_overlap"]-len(selected_ids & reference_ids)/limit) < 1e-12
        image_assisted = bool(question.get("query_image_paths"))
        if "query_has_image" in question:
            assert bool(question["query_has_image"]) == image_assisted
        # Every currently implemented non-full method consumes CLIP order,
        # including tie-breaking in the "best bound" policies.
        clip_products = 0 if method == "full_scan" else summary["n_masters"]*512
        records.append({
            "batch": relative, "domain": domain, "method": method, "L": limit,
            "id": run["id"], "question": question["question"], "question_index": qi,
            "image_assisted": image_assisted, "run": run,
            "n_leaves": summary["n_leaves"], "n_masters": summary["n_masters"],
            "full_products": summary["n_leaves"]*384,
            "clip_products": clip_products,
            "selected_leaves": selected, "reference_leaves": reference,
            "gold_ids": gold, "selected_ids": selected_ids, "reference_ids": reference_ids,
            "lost_gold_ids": gold & reference_ids-selected_ids,
            "gained_gold_ids": gold & selected_ids-reference_ids,
            "missing_gold_ids": gold-selected_ids,
            "baseline_missing_gold_ids": gold-reference_ids,
            "timing": timed[(method, qi)],
            "baseline_timing": timed[("full_scan", qi)],
        })
    expected = len(inputs["questions"])*len(summary["methods"])*4
    assert len(records) == expected, (relative, len(records), expected)
    for reported in summary["scopes"]["all"]:
        subset = [r for r in records if r["method"] == reported["method"] and r["L"] == reported["L"]]
        assert len(subset) == reported["n_questions"]
        assert sum(r["run"]["exact_match"] for r in subset) == reported["exact_matches"]
        assert sum(r["run"]["gold_hits"] for r in subset) == reported["gold_hits"]
        assert sum(len(r["gold_ids"]) for r in subset) == reported["gold_total"]
        assert abs(statistics.mean(r["run"]["scalar_coordinate_products"] for r in subset)
                   - reported["mean_coordinate_products"]) < 1e-8
        assert abs(statistics.mean(r["run"]["scored_leaves"] for r in subset)
                   - reported["mean_scored_leaves"]) < 1e-10
    return {"summary": summary, "records": records, "inputs": inputs}


def aggregate(records, scope_id, scope_label):
    by_method = defaultdict(list)
    for record in records:
        by_method[(record["method"], record["L"])].append(record)
    rows = []
    for (method, limit), selected in sorted(
        by_method.items(), key=lambda item: (item[0][0] != "full_scan", item[0][0], item[0][1])
    ):
        count = len(selected)
        full_products = sum(r["full_products"] for r in selected)
        bge_products = sum(r["run"]["scalar_coordinate_products"] for r in selected)
        clip_products = sum(r["clip_products"] for r in selected)
        full_leaves = sum(r["n_leaves"] for r in selected)
        scored = sum(r["run"]["scored_leaves"] for r in selected)
        reference_hits = sum(len(r["gold_ids"] & r["reference_ids"]) for r in selected)
        gold_hits = sum(r["run"]["gold_hits"] for r in selected)
        latency = None
        if limit == 5:
            samples = [value for r in selected for value in r["timing"]["samples_ms"]]
            medians = [r["timing"]["median_ms"] for r in selected]
            latency = {
                "L": 5, "n_measurements": len(samples),
                "pooled_median_ms": statistics.median(samples),
                "pooled_p95_ms": percentile(samples, .95),
                "mean_per_query_median_ms": statistics.mean(medians),
                "median_per_query_median_ms": statistics.median(medians),
                "min_repetitions_per_query": min(r["timing"]["repetitions"] for r in selected),
                "max_repetitions_per_query": max(r["timing"]["repetitions"] for r in selected),
                "mean_paired_full_scan_per_query_median_ms": statistics.mean(
                    r["baseline_timing"]["median_ms"] for r in selected),
            }
            latency["mean_per_query_median_ratio_to_full"] = (
                latency["mean_per_query_median_ms"] /
                latency["mean_paired_full_scan_per_query_median_ms"])
        bound_counts = [r["run"].get("bound_dot_count") for r in selected]
        actual_leaf_dots = [r["run"].get("exact_leaf_dot_count",
                            r["run"].get("exact_score_count", r["run"]["scored_leaves"]))
                            for r in selected]
        row = {
            "scope": scope_id, "scope_label": scope_label, "method": method,
            "method_label": METHOD_NAMES.get(method, method), "L": limit,
            "n_questions": count, "n_domains": len({r["domain"] for r in selected}),
            "n_image_assisted_questions": sum(r["image_assisted"] for r in selected),
            "ordered_topL_exact": sum(r["run"]["exact_match"] for r in selected),
            "ordered_topL_exact_fraction": sum(r["run"]["exact_match"] for r in selected)/count,
            "mean_topL_overlap": statistics.mean(r["run"]["topL_overlap"] for r in selected),
            "lost_reference_leaf_occurrences": sum(len(r["reference_ids"]-r["selected_ids"]) for r in selected),
            "mean_scored_leaves": scored/count, "mean_full_scan_leaves": full_leaves/count,
            "mean_resolved_leaf_scores": scored/count,
            "mean_actual_leaf_dot_count": statistics.mean(actual_leaf_dots),
            "weighted_actual_leaf_dot_reduction": 1-sum(actual_leaf_dots)/full_leaves,
            "mean_centroid_or_bound_dot_count": (
                statistics.mean(bound_counts) if all(v is not None for v in bound_counts) else None),
            "weighted_leaf_score_reduction": 1-scored/full_leaves,
            "mean_visited_masters": statistics.mean(r["run"]["visited_masters"] for r in selected),
            "mean_total_masters": statistics.mean(r["n_masters"] for r in selected),
            "mean_bge_coordinate_products": bge_products/count,
            "mean_clip_master_coordinate_products": clip_products/count,
            "mean_total_coordinate_products_including_clip": (bge_products+clip_products)/count,
            "mean_plain_full_bge_coordinate_products": full_products/count,
            "bge_coordinate_ratio_to_plain_full": bge_products/full_products,
            "total_coordinate_ratio_to_plain_full_including_clip": (bge_products+clip_products)/full_products,
            "bge_coordinate_reduction": 1-bge_products/full_products,
            "total_coordinate_reduction_including_clip": 1-(bge_products+clip_products)/full_products,
            "gold_hits": gold_hits, "full_scan_gold_hits": reference_hits,
            "gold_total": sum(len(r["gold_ids"]) for r in selected),
            "any_gold_queries": sum(bool(r["gold_ids"] & r["selected_ids"]) for r in selected),
            "full_scan_any_gold_queries": sum(bool(r["gold_ids"] & r["reference_ids"]) for r in selected),
            "all_gold_queries": sum(r["gold_ids"] <= r["selected_ids"] for r in selected),
            "full_scan_all_gold_queries": sum(r["gold_ids"] <= r["reference_ids"] for r in selected),
            "gold_occurrences_lost_vs_full": sum(len(r["lost_gold_ids"]) for r in selected),
            "gold_occurrences_gained_vs_full": sum(len(r["gained_gold_ids"]) for r in selected),
            "queries_with_lost_gold": sum(bool(r["lost_gold_ids"]) for r in selected),
            "queries_with_gained_gold": sum(bool(r["gained_gold_ids"]) for r in selected),
            "queries_with_missing_gold": sum(bool(r["missing_gold_ids"]) for r in selected),
            "full_scan_queries_with_missing_gold": sum(bool(r["baseline_missing_gold_ids"]) for r in selected),
            "gold_occurrences_missing_in_both": sum(
                len(r["missing_gold_ids"] & r["baseline_missing_gold_ids"]) for r in selected),
            "latency": latency,
        }
        assert gold_hits-reference_hits == row["gold_occurrences_gained_vs_full"]-row["gold_occurrences_lost_vs_full"]
        rows.append(row)
    return rows


def discrepancy(record):
    run = record["run"]
    return {
        "domain": record["domain"], "id": record["id"], "question": record["question"],
        "query_type": "image_assisted" if record["image_assisted"] else "text_only",
        "method": record["method"], "L": record["L"], "exact_match": run["exact_match"],
        "topL_overlap": run["topL_overlap"],
        "selected_memory_ids": [leaf["memory_id"] for leaf in record["selected_leaves"]],
        "reference_memory_ids": [leaf["memory_id"] for leaf in record["reference_leaves"]],
        "selected_filenames": [leaf["filename"] for leaf in record["selected_leaves"]],
        "reference_filenames": [leaf["filename"] for leaf in record["reference_leaves"]],
        "missed_reference_memory_ids": sorted(record["reference_ids"]-record["selected_ids"]),
        "added_memory_ids": sorted(record["selected_ids"]-record["reference_ids"]),
        "gold_ids": sorted(record["gold_ids"]),
        "lost_gold_ids": sorted(record["lost_gold_ids"]),
        "gained_gold_ids": sorted(record["gained_gold_ids"]),
        "missing_gold_ids": sorted(record["missing_gold_ids"]),
        "baseline_missing_gold_ids": sorted(record["baseline_missing_gold_ids"]),
        "gold_hits": run["gold_hits"],
        "full_scan_gold_hits": len(record["gold_ids"] & record["reference_ids"]),
        "gold_total": len(record["gold_ids"]),
        "scored_leaves": run["scored_leaves"], "full_scan_leaves": record["n_leaves"],
        "actual_leaf_dot_count": run.get("exact_leaf_dot_count", run.get("exact_score_count", run["scored_leaves"])),
        "centroid_or_bound_dot_count": run.get("bound_dot_count"),
        "visited_masters": run["visited_masters"], "n_masters": record["n_masters"],
        "bge_coordinate_products": run["scalar_coordinate_products"],
        "clip_coordinate_products": record["clip_products"],
        "full_scan_bge_coordinate_products": record["full_products"],
        "source_runs": record["batch"]+"/runs.json",
    }


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def build_summary(output):
    sources = {
        Path(__file__), ROOT/"scripts/analysis/compare_adaptive_retrieval.py",
        ROOT/"scripts/graph/observed_spread_search.py",
        ROOT/"scripts/graph/progressive_text_bounds.py",
        ROOT/"scripts/graph/hierarchical_text_bounds.py",
        ROOT/"scripts/graph/text_similarity_bounds.py",
    }
    strict = load_batch(output, "home/"+HOME, sources)
    development = load_batch(output, "home_empirical/"+HOME, sources)
    transfer_batches = [load_batch(output, str(p.relative_to(output)), sources)
                        for p in sorted((output/"all_empirical").iterdir())
                        if (p/"summary.json").exists()]
    assert len(transfer_batches) == 8
    all_records = [r for batch in transfer_batches for r in batch["records"]]
    reuse_batches = [load_batch(output, str(p.relative_to(output)), sources)
                     for p in sorted((output/"all_centroid_reuse").iterdir())
                     if (p/"summary.json").exists()]
    assert len(reuse_batches) == 8
    reuse_records = [r for batch in reuse_batches for r in batch["records"]
                     if r["method"] == "empirical_max_both_reuse"]
    original_both = {(r["id"], r["L"]): r for r in all_records if r["method"] == "empirical_max_both"}
    for record in reuse_records:
        original = original_both[(record["id"], record["L"])]
        assert record["run"]["selected_indices"] == original["run"]["selected_indices"]
        assert record["run"]["scored_leaves"] == original["run"]["scored_leaves"]
        assert record["run"]["visited_masters"] == original["run"]["visited_masters"]
    assert len(reuse_records) == 371*4
    all_records += reuse_records
    development_records = development["records"] + [r for r in reuse_records if r["domain"] == HOME]
    all_questions = {r["id"]: r for r in all_records}
    assert len(all_questions) == 371
    assert sum(r["image_assisted"] for r in all_questions.values()) == 8
    scopes = [
        ("home_strict_52", "Home 开发 · 严格方法 · 52 题", strict["records"]),
        ("home_strict_new_51", "Home 次要对照 · 严格方法 · 新增 51 题",
         [r for r in strict["records"] if not r["id"].endswith("/Q10")]),
        ("home_empirical_52", "Home 开发 · 经验方法 · 52 题", development_records),
        ("home_empirical_new_51", "Home 次要对照 · 经验方法 · 新增 51 题",
         [r for r in development_records if not r["id"].endswith("/Q10")]),
        ("transfer_7_319", "迁移评估 · 另外 7 域 · 319 题",
         [r for r in all_records if r["domain"] != HOME]),
        ("all_8_371", "全部 8 域 · 371 题（包含开发域）", all_records),
        ("all_text_363", "全部域 · 纯文本查询 · 363 题",
         [r for r in all_records if not r["image_assisted"]]),
        ("all_image_8", "全部域 · 带图查询 · 8 题",
         [r for r in all_records if r["image_assisted"]]),
        ("transfer_text_311", "迁移域 · 纯文本查询 · 311 题",
         [r for r in all_records if r["domain"] != HOME and not r["image_assisted"]]),
        ("transfer_image_8", "迁移域 · 带图查询 · 8 题",
         [r for r in all_records if r["domain"] != HOME and r["image_assisted"]]),
    ]
    for batch in transfer_batches:
        domain = batch["summary"]["domain"]
        scopes.append(("domain_"+domain, "单域 · "+domain,
                       batch["records"]+[r for r in reuse_records if r["domain"] == domain]))
    methods = [row for scope_id, label, records in scopes for row in aggregate(records, scope_id, label)]
    disagreements = [discrepancy(r) for r in all_records
                     if r["method"] != "full_scan" and not r["run"]["exact_match"]]
    primary_records = [r for r in all_records if r["method"] == "empirical_max_both" and r["L"] == 5]
    primary_audits = [discrepancy(r) for r in primary_records]
    primary_gold_changes = [d for d in primary_audits if d["lost_gold_ids"] or d["gained_gold_ids"]]
    for path in (output/"experiment_plan.json", output/"experiment_plan_addendum.json"):
        sources.add(path)
    qa_path = output/"reader_comparison.json"
    if not qa_path.exists():
        qa_path = output/"changed_qa/summary.json"
    qa = read(qa_path) if qa_path.exists() else None
    qa_protocol = None
    if qa_path.exists():
        sources.add(qa_path)
        protocol_path = qa_path.parent/"protocol.json"
        if protocol_path.exists():
            qa_protocol = read(protocol_path)
            sources.add(protocol_path)
        if "paired_results" in qa:
            expected_changed = {r["id"] for r in primary_records if not r["run"]["exact_match"]}
            assert {r["question_id"] for r in qa["paired_results"]} == expected_changed
            assert qa["n_questions"] == len(expected_changed)
            assert not qa["full_371_accuracy_measured"]
            qa_by_id = {r["question_id"]: r for r in qa["paired_results"]}
            for row in disagreements:
                pair = (qa_by_id.get(row["id"]) if row["L"] == 5 and
                        row["method"] in {"empirical_max_both", "empirical_max_both_reuse"} else None)
                row["paired_reader_full_answer"] = pair["full_scan_answer"] if pair else None
                row["paired_reader_empirical_answer"] = pair["empirical_answer"] if pair else None
                row["paired_reader_gold_choice"] = pair["gold_choice"] if pair else None
    fit_rows = []
    for stage, batches in (("home_strict", [strict]), ("home_development_empirical", [development]),
                           ("all_domains_empirical", transfer_batches),
                           ("all_domains_centroid_reuse", reuse_batches)):
        for batch in batches:
            s = batch["summary"]
            for method, fit in s["fit"].items():
                fit_rows.append({"stage": stage, "domain": s["domain"], "method": method,
                                 "n_leaves": s["n_leaves"], "n_masters": s["n_masters"], **fit})
    independent = []
    for audit_path in sorted(output.glob("validation*independent.json")):
        audit = read(audit_path)
        sources.add(audit_path)
        independent.append({"path": audit_path.name, "passed": audit.get("passed"),
                            "checks": audit.get("checks")})
    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_domain": HOME, "development_questions": 52,
        "home_new_questions_secondary": 51, "transfer_domains": 7,
        "transfer_questions": 319, "all_questions": 371,
        "text_only_questions": 363, "image_assisted_questions": 8,
        "primary_L": 5, "secondary_L": [1, 3, 10],
        "methods": methods,
        "scope_labels": {scope_id: label for scope_id, label, _ in scopes},
        "method_labels": METHOD_NAMES,
        "strict_expansion_gate_passed_methods": strict["summary"]["expansion_gate_passed_methods"],
        "primary_all_domain_gold_changes": primary_gold_changes,
        "primary_all_domain_question_audit": primary_audits,
        "fit_and_storage": fit_rows,
        "independent_audits": independent,
        "disagreement_rows_all_empirical_all_L": len(disagreements),
        "reader_comparison": qa,
        "reader_protocol": qa_protocol,
        "reader_comparison_path": str(qa_path.relative_to(output)) if qa else None,
        "centroid_reuse_equivalence": {
            "checked_question_budget_pairs": len(reuse_records),
            "same_ordered_leaves_as_original_max_both": True,
            "same_resolved_leaf_and_visited_master_counts": True,
            "change": "Reuse query dot cluster mean to reconstruct the last leaf score; "
                      "near-tie cases fall back to exact scoring. Retrieval policy is unchanged.",
        },
        "aggregation_validation": {
            "source_batches_checked": 2+len(transfer_batches)+len(reuse_batches),
            "source_summary_counts_and_costs_recomputed_from_runs": True,
            "all_gold_sets_recomputed_from_input_memory_ids": True,
            "source_questions_unique_in_primary_scope": len(all_questions),
            "reuse_ordered_output_equivalence_pairs_checked": len(reuse_records),
            "paired_reader_ids_match_changed_top5_ids": (
                True if qa and "paired_results" in qa else None),
        },
        "experiment_plan": read(output/"experiment_plan.json"),
        "experiment_plan_addendum": read(output/"experiment_plan_addendum.json"),
        "cost_boundaries": {
            "bge_products": "Includes exact leaf and BGE center/bound/projection coordinate products.",
            "clip_products": "Adds 512 times the number of masters for every method using CLIP order.",
            "plain_full_bge": "No CLIP query encoder or CLIP master scoring is required by this baseline.",
            "encoder_time": "Not measured; cannot infer end-to-end speedup.",
            "wall_time": "Measured at L=5: encoded-query BGE search, dot products and topL; "
                         "excludes CLIP ordering, encoders, file IO and reader.",
            "aggregation": "Reductions are ratios of sums across queries, not means of domain ratios.",
        },
        "limitations": [
            "Home is development, not held-out transfer.",
            "Empirical spread predictions are not mathematical upper bounds and may miss TopL or gold.",
            "All three empirical policies are reported; target-domain gold did not tune their rules.",
            "TopL fidelity and gold retrieval are separate from downstream QA accuracy.",
            "Lower coordinate counts do not establish lower runtime; all measured search latency is reported.",
        ],
        "source_sha256": {str(p.resolve()): sha(p) for p in sorted(sources)},
    }
    primary = next(r for r in methods if r["scope"] == "all_8_371"
                   and r["method"] == "empirical_max_both" and r["L"] == 5)
    assert primary["ordered_topL_exact"] == 350
    assert primary["gold_hits"] == 491 and primary["full_scan_gold_hits"] == 492
    assert primary["gold_total"] == 1041
    assert primary["gold_occurrences_lost_vs_full"] == 3
    assert primary["gold_occurrences_gained_vs_full"] == 2
    assert len(primary_gold_changes) == 4
    assert not snapshot["strict_expansion_gate_passed_methods"]
    return snapshot, disagreements


_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>动态记忆检索：效果与完整成本对照</title>
<style>
:root{--ink:#21344a;--muted:#65758b;--line:#dce4ed;--blue:#2468a4;--orange:#be7436;--green:#217357;--red:#a74444}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.65 system-ui,sans-serif}
main{max-width:1510px;margin:auto;padding:32px 26px 60px}h1{font-size:29px;margin:0 0 14px;line-height:1.35}
h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:16px 0 8px}p{margin:8px 0 14px}
.muted,small{color:var(--muted)}.panel,.card{background:white;border:1px solid var(--line);border-radius:10px}
.panel{padding:20px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}
.card{padding:16px}.card .value{font-size:25px;font-weight:700}.card .label{font-size:13px;color:var(--muted)}
.note{background:#edf4fa;border-left:4px solid var(--blue);padding:12px 15px;margin:14px 0}
.warning{background:#fff5e8;border-color:var(--orange)}.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0}
select,input,button{font:inherit;color:var(--ink);padding:8px 10px;border:1px solid #bac8d7;border-radius:6px;background:white}
select{max-width:100%;min-width:170px}input{min-width:220px}button{cursor:pointer}button:hover{background:#ecf2f8}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f0f4f8;white-space:nowrap}tr:last-child td{border-bottom:0}tbody tr:hover{background:#f8faff}
.num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.good{color:var(--green)}.bad{color:var(--red)}
.mono{font:12px/1.5 ui-monospace,SFMono-Regular,monospace}.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.tag{display:inline-block;padding:2px 6px;border-radius:4px;background:#e9eff5;font-size:12px}
.gold{background:#fff4d9}.bar-line{display:grid;grid-template-columns:180px 1fr 100px;gap:12px;align-items:center;margin:12px 0}
.track{height:24px;background:#eef2f6;border-radius:4px;overflow:hidden;display:flex}.bar-bge{background:var(--blue)}.bar-clip{background:var(--orange)}
details{border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin:10px 0}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:12px;border-radius:5px;font:12px/1.5 ui-monospace,monospace}
a{color:var(--blue)}.empty{padding:20px;color:var(--muted)}.hidden{display:none}
@media(max-width:900px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.split{grid-template-columns:1fr}.bar-line{grid-template-columns:130px 1fr 75px}}
@media print{body{background:white}.toolbar{display:none}.table-wrap{overflow:visible}.panel{break-inside:avoid}}
</style></head><body><main>
<h1>动态记忆检索：效果与完整成本对照</h1>
<p>Home 先用于方法开发，再把冻结的三条经验规则迁移到另外 7 个领域。
这些规则根据已经访问的簇估计后续簇的最高分，不设固定相似度阈值，但估计值并非数学上界。</p>
<div class="note warning"><b>目前只能说减少了部分评分工作，不能宣称整体提速。</b>
<span id="complete-cost-message"></span>尚未计入 CLIP 编码，已测的 BGE 检索墙时也更慢。</div>
<div id="headline-cards" class="cards"></div>
<section class="panel"><h2>实验口径</h2>
<p>开发域：Home 52 题；剔除已有 Q10 的 51 题作为次要口径。迁移评估：另外 7 域 319 题。
全部 371 题中有 363 道纯文本查询、8 道带图查询；带图题单独报告。L 是最终叶子记忆数量，不是访问簇数。</p>
<p>主对照沿用 L=5，另报告 L=1、3、10。Gold 命中按问题内的记忆 ID 计数；
Top-L 一致性与问答正确率分开记录。全部汇总都按问题实际工作量加总，不对领域百分比简单平均。</p>
<details><summary>六项严格方法的负结果与经验扩展依据</summary><div id="strict-results"></div></details>
<details><summary>三条经验规则如何区别</summary>
<p>最大偏移：使用已访问簇中“真实最高分 − 文本均值分数”的最大值预测未访问簇。
半径比例：用已观察到的偏移与簇半径之比预测。
两者取大：使用这两个预测中较高的一个。它更保守，但仍不能保证覆盖未访问簇的真实最高分。</p>
<p>规则只根据当前问题已访问的簇更新，不读取未访问叶子的真实分数或 gold。
三个版本全部报告，没有用迁移领域的结果挑阈值。Home 的观察属于开发证据。</p>
<p>“中心复用”保留“两者取大”的检索规则：已知查询与簇均值的点积，完整访问该簇时计算 n−1 个叶子，
用 n×中心分数减去前面分数之和恢复最后一个；接近排序边界时回退精算。
它没有改变本次 371×4 个问题/预算的返回列表，但得到分数的叶子数量与真正做完整点积的次数不同。</p>
<p><a href="experiment_plan.json">事前实验计划</a> · <a href="experiment_plan_addendum.json">迁移评估前的扩展记录</a></p>
</details></section>
<section class="panel"><h2>按范围与返回数量比较</h2>
<div class="toolbar"><label>范围 <select id="scope"></select></label><label>返回叶子 L <select id="budget"><option>1</option><option>3</option><option selected>5</option><option>10</option></select></label></div>
<div id="selected-summary"></div><div id="methods-table" class="table-wrap"></div>
<p class="muted">“得到分数的叶子”包含用中心恒等式恢复的分数；实际叶子点积另外列出。
成本减少的负值表示成本增加。渐进 PCA 的查询投影在总成本中计入。</p>
<h3>成本需要计入 CLIP</h3><div id="cost-bars"></div>
<p class="muted">蓝色为 BGE 坐标乘积（含中心、界与投影）；橙色为 CLIP master 排序的 512×簇数。
plain BGE 全量检索不需要 CLIP。这些不是 FLOPs 总量，编码和排序等开销并未全部体现在条形图中。</p>
<h3>实测检索墙时</h3><div id="latency-table" class="table-wrap"></div>
<p class="muted">只在 L=5 重复计时。输入已经编码，计时包含 BGE 索引遍历、点积与 Top-L；
不含 CLIP 排序、编码、文件读取、reader。各方法的相对时间使用同批次全量基线；
中心复用与原版在不同批次测量。其他 L 不复用 L=5 的时间作为测量值。</p>
</section>
<section class="panel"><h2>Gold 损失、增益与未命中</h2>
<p>下面默认检查“经验两者取大”的 L=5；完整的三方法与四个 L 差异保存在 CSV。
有些 gold 在全量 BGE Top5 中本来就缺失，因此“与全量一致”不能替代 gold 命中率。</p>
<div id="gold-summary"></div><div id="gold-changes" class="table-wrap"></div>
<details><summary>全部 371 题的 gold 未命中记录</summary><div id="missing-gold" class="table-wrap"></div></details>
</section>
<section class="panel"><h2>逐题检索差异</h2>
<div class="toolbar"><label>方法 <select id="diff-method"></select></label>
<label>L <select id="diff-budget"><option>1</option><option>3</option><option selected>5</option><option>10</option></select></label>
<input id="diff-search" placeholder="搜索领域、题号或问题"><span id="diff-count" class="muted"></span></div>
<div id="diff-table" class="table-wrap"></div><div id="diff-detail"></div>
<p><a href="disagreements.csv">下载全部不一致记录</a> · <a href="methods.csv">下载全部方法和口径</a></p>
</section>
<section class="panel"><h2>下游问答验证</h2><div id="qa"></div></section>
<section class="panel"><h2>索引与补充负结果</h2>
<details><summary>各领域索引数值存储与构建时间</summary><div id="storage" class="table-wrap"></div></details>
<p><a href="diagnostics/README.md">PCA 与记忆自身邻居保真诊断</a> ·
<a href="diagnostics/progressive_memory_rank_probe_results.json">PCA 逐题结果</a> ·
<a href="diagnostics/neighbor_graph_pilot.json">近邻图探索结果</a> ·
<a href="pilot_centered_pivot_result.json">中心化 pivot 探索结果</a></p>
<p class="muted">严格方法没有通过“Top5 全部一致且坐标成本下降”的扩展门槛。
自动 PCA 维数和近邻图探索也保留了失败结果，没有仅展示有效候选。</p>
<p><a href="summary.json">聚合结果与来源 SHA-256</a> · <a href="methods.csv">方法表</a> ·
<a href="disagreements.csv">差异表</a></p>
<div id="independent-audits"></div>
</section>
</main><script type="application/json" id="data">__DATA__</script><script>
"use strict";
const D=JSON.parse(document.getElementById("data").textContent);
const S=D.summary,rows=S.methods,esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const n=(v,d=2)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(d):"—";
const pct=v=>n(v*100)+"%";
const name=m=>S.method_labels[m]??m;
const short=x=>String(x??"").split("/").pop();
const table=(head,body)=>`<table><thead><tr>${head.map(x=>`<th>${esc(x)}</th>`).join("")}</tr></thead><tbody>${body.map(c=>`<tr>${c.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
const ids=xs=>xs.length?xs.map(x=>`<div class="mono">${esc(short(x))}</div>`).join(""):"—";
const pick=(scope,method,L=5)=>rows.find(r=>r.scope===scope&&r.method===method&&r.L===L);
const primary=pick("all_8_371","empirical_max_both");
const reused=pick("all_8_371","empirical_max_both_reuse");
document.getElementById("complete-cost-message").textContent=
 `计入 CLIP master 排序后，原“两者取大”减少 ${pct(primary.total_coordinate_reduction_including_clip)} 坐标乘积；
  复用簇中心分数的同结果版本减少 ${pct(reused.total_coordinate_reduction_including_clip)}。`;
document.getElementById("headline-cards").innerHTML=[
 ["全部 371 题 Top5 完全一致",`${primary.ordered_topL_exact}/371`,`集合重合率 ${pct(primary.mean_topL_overlap)}`],
 ["中心复用后 BGE 坐标乘积减少",pct(reused.bge_coordinate_reduction),"已包含 BGE 文本中心评分"],
 ["中心复用并计入 CLIP 后减少",pct(reused.total_coordinate_reduction_including_clip),"编码时间仍未计入"],
 ["Gold 命中",`${primary.gold_hits}/${primary.gold_total}`,`全量 ${primary.full_scan_gold_hits}/${primary.gold_total}；并非问答分数`]
].map(([l,v,t])=>`<div class="card"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><small>${esc(t)}</small></div>`).join("");
const strict=rows.filter(r=>r.scope==="home_strict_52"&&r.L===5);
document.getElementById("strict-results").innerHTML=`<p>包含全量基线的六项严格对照均完整运行；五项检索替代方案没有同时获得精确性和成本收益。</p>`+
 table(["方法","ordered Top5","BGE 成本/全量","平均完整评分叶子","平均每题中位墙时 ms"],strict.map(r=>[
 esc(r.method_label),`${r.ordered_topL_exact}/${r.n_questions}`,n(r.bge_coordinate_ratio_to_plain_full,3),n(r.mean_scored_leaves),n(r.latency.mean_per_query_median_ms,4)]));
const scope=document.getElementById("scope"),budget=document.getElementById("budget");
scope.innerHTML=Object.entries(S.scope_labels).map(([id,label])=>`<option value="${esc(id)}">${esc(label)}</option>`).join("");
scope.value="transfer_7_319";
function renderMethods(){
 const current=rows.filter(r=>r.scope===scope.value&&r.L===Number(budget.value));
 const chosen=current.find(r=>r.method==="empirical_max_both_reuse")??current.find(r=>r.method==="empirical_max_both")??current.find(r=>r.method==="progressive_text_pca")??current[0];
 document.getElementById("selected-summary").innerHTML=`<p><b>${esc(S.scope_labels[scope.value])}</b>：${chosen.n_questions} 题，
 ${chosen.n_image_assisted_questions} 道带图查询。该范围平均全量评分 ${n(chosen.mean_full_scan_leaves)} 条叶子。</p>`;
 document.getElementById("methods-table").innerHTML=table(
 ["方法","完全一致","Top-L 重合","得到分数的叶子","实际叶子点积","中心/界点积","BGE 成本减少","计入 CLIP 后减少","Gold / 总数","Any / All gold"],
 current.map(r=>[esc(r.method_label),`${r.ordered_topL_exact}/${r.n_questions}`,pct(r.mean_topL_overlap),n(r.mean_resolved_leaf_scores),
 n(r.mean_actual_leaf_dot_count),n(r.mean_centroid_or_bound_dot_count),pct(r.bge_coordinate_reduction),`<span class="${r.total_coordinate_reduction_including_clip>=0?"good":"bad"}">${pct(r.total_coordinate_reduction_including_clip)}</span>`,
 `${r.gold_hits}/${r.gold_total}`,`${r.any_gold_queries} / ${r.all_gold_queries}`]));
 const largest=Math.max(...current.map(r=>r.mean_total_coordinate_products_including_clip),chosen.mean_plain_full_bge_coordinate_products);
 document.getElementById("cost-bars").innerHTML=current.map(r=>`<div class="bar-line"><span>${esc(r.method_label)}</span>
 <div class="track"><div class="bar-bge" style="width:${100*r.mean_bge_coordinate_products/largest}%"></div>
 <div class="bar-clip" style="width:${100*r.mean_clip_master_coordinate_products/largest}%"></div></div>
 <span class="num">${n(r.mean_total_coordinate_products_including_clip,0)}</span></div>`).join("");
 document.getElementById("latency-table").innerHTML=Number(budget.value)!==5
 ?'<div class="empty">此 L 没有重复墙时测量。切换 L=5 查看实测。</div>'
 :table(["方法","每题中位数的平均 ms","测量中位 ms","测量 P95 ms","同批全量均值 ms","相对同批全量墙时","每题重复数"],current.map(r=>[
 esc(r.method_label),n(r.latency.mean_per_query_median_ms,5),n(r.latency.pooled_median_ms,5),n(r.latency.pooled_p95_ms,5),
 n(r.latency.mean_paired_full_scan_per_query_median_ms,5),`${n(r.latency.mean_per_query_median_ratio_to_full,2)}×`,
 `${r.latency.min_repetitions_per_query}–${r.latency.max_repetitions_per_query}`]));
}
scope.addEventListener("change",renderMethods);budget.addEventListener("change",renderMethods);renderMethods();
document.getElementById("gold-summary").innerHTML=`<p>全部域 L=5：经验两者取大命中 ${primary.gold_hits}/${primary.gold_total}，
 全量命中 ${primary.full_scan_gold_hits}/${primary.gold_total}。相对全量丢失 ${primary.gold_occurrences_lost_vs_full} 个 gold 出现、增加
 ${primary.gold_occurrences_gained_vs_full} 个；${S.primary_all_domain_gold_changes.length} 道问题涉及 gold 变化。
 Any-gold：${primary.any_gold_queries} vs ${primary.full_scan_any_gold_queries}；
 All-gold：${primary.all_gold_queries} vs ${primary.full_scan_all_gold_queries}。</p>`;
document.getElementById("gold-changes").innerHTML=table(["题目","丢失的 gold","新增的 gold","本方案仍缺少","全量原本缺少"],
 S.primary_all_domain_gold_changes.map(r=>[esc(r.id),ids(r.lost_gold_ids),ids(r.gained_gold_ids),ids(r.missing_gold_ids),ids(r.baseline_missing_gold_ids)]));
document.getElementById("missing-gold").innerHTML=table(["题目","经验命中/总数","全量命中/总数","经验漏掉的 gold","全量漏掉的 gold"],
 S.primary_all_domain_question_audit.filter(r=>r.missing_gold_ids.length||r.baseline_missing_gold_ids.length).map(r=>[
 esc(r.id),`${r.gold_hits}/${r.gold_total}`,`${r.full_scan_gold_hits}/${r.gold_total}`,ids(r.missing_gold_ids),ids(r.baseline_missing_gold_ids)]));
const diffMethod=document.getElementById("diff-method"),diffBudget=document.getElementById("diff-budget"),search=document.getElementById("diff-search");
diffMethod.innerHTML=["empirical_max_both_reuse","empirical_max_both","empirical_offset","empirical_radius_ratio"].map(m=>`<option value="${m}">${esc(name(m))}</option>`).join("");
function renderDifferences(){
 const term=search.value.toLowerCase();
 const selected=D.disagreements.filter(r=>r.method===diffMethod.value&&r.L===Number(diffBudget.value)&&(`${r.id} ${r.question}`).toLowerCase().includes(term));
 document.getElementById("diff-count").textContent=`${selected.length} 条不一致记录`;
 document.getElementById("diff-table").innerHTML=table(["题目","查询","Top-L 重合","漏掉全量结果","新增结果","Gold 丢失 / 新增","操作"],
 selected.map((r,i)=>[esc(r.id),esc(r.query_type==="image_assisted"?"带图":"纯文本"),pct(r.topL_overlap),ids(r.missed_reference_memory_ids),
 ids(r.added_memory_ids),`${r.lost_gold_ids.length} / ${r.gained_gold_ids.length}`,`<button data-diff="${i}">详情</button>`]));
 document.getElementById("diff-detail").innerHTML="";
 document.querySelectorAll("[data-diff]").forEach(button=>button.addEventListener("click",()=>{
  const r=selected[Number(button.dataset.diff)];
  document.getElementById("diff-detail").innerHTML=`<details open><summary>${esc(r.id)}</summary><p>${esc(r.question)}</p>
  <div class="split"><div><h3>经验方案</h3>${r.selected_filenames.map((f,i)=>`<div>${i+1}. ${esc(f)}</div>`).join("")}</div>
  <div><h3>全量 BGE</h3>${r.reference_filenames.map((f,i)=>`<div>${i+1}. ${esc(f)}</div>`).join("")}</div></div>
  <p>得到 ${r.scored_leaves}/${r.full_scan_leaves} 条叶子的分数，实际做 ${r.actual_leaf_dot_count} 次叶子点积，
  访问 ${r.visited_masters}/${r.n_masters} 个 master。
  BGE ${r.bge_coordinate_products} + CLIP ${r.clip_coordinate_products} 次坐标乘积；
  plain BGE 全量 ${r.full_scan_bge_coordinate_products}。</p><p><a href="${esc(r.source_runs)}">对应搜索轨迹</a></p>
  <pre>${esc(JSON.stringify(r,null,2))}</pre></details>`;
 }));
}
diffMethod.addEventListener("change",renderDifferences);diffBudget.addEventListener("change",renderDifferences);search.addEventListener("input",renderDifferences);renderDifferences();
document.getElementById("storage").innerHTML=table(["阶段","领域","方法","数值索引字节","构建秒"],S.fit_and_storage.map(r=>[
 esc(r.stage),esc(r.domain),esc(name(r.method)),r.numeric_index_bytes.toLocaleString(),n(r.seconds,5)]));
document.getElementById("independent-audits").innerHTML=`<h3>独立复核</h3>`+
 S.independent_audits.map(a=>`<p><a href="${esc(a.path)}">${esc(a.path)}</a>：
 ${a.passed?"通过":"请查看结果"}，${esc(a.checks??"—")} 项检查。</p>`).join("");
const qa=S.reader_comparison;
if(qa&&qa.paired_results){
 const changed=qa.paired_results.filter(r=>r.answer_changed);
 const outcomeRows=qa.paired_results.map(r=>[
 esc(r.question_id),esc(r.gold_choice),esc(r.full_scan_answer)+(r.full_scan_correct?" ✓":" ✗"),
 esc(r.empirical_answer)+(r.empirical_correct?" ✓":" ✗"),r.answer_changed?"答案变化":"相同"]);
 document.getElementById("qa").innerHTML=`<div class="note warning"><b>只重新测试 ordered Top5 改变的 ${qa.n_questions} 题。</b>
 全量 BGE 答对 ${qa.full_scan_correct}/${qa.n_questions}，经验“两者取大”答对 ${qa.empirical_max_both_correct}/${qa.n_questions}；
 ${qa.n_questions-qa.answer_changed} 题答案相同，${qa.correct_to_wrong} 题由对变错，${qa.wrong_to_correct} 题由错变对。
 这不是重新测量的全 371 题准确率。</div>
 <p>两条件使用同一固定版本 Qwen3-VL-8B-Instruct、batch size 1 和相同 GPU/CPU 混合放置，
 共 ${qa.n_responses} 次新推理；格式错误 ${qa.format_errors}，输出达到 token 上限 ${qa.token_limit_hits}。
 另外 350 题的检索上下文完全相同。中心复用版本返回同样的叶子，适用同一组上下文对照。</p>
 ${changed.map(r=>`<p><b>${esc(r.question_id)}</b>：${esc(r.full_scan_answer)}（${r.full_scan_correct?"对":"错"}）→
 ${esc(r.empirical_answer)}（${r.empirical_correct?"对":"错"}）。</p>`).join("")}
 <p>Outdoor/Q31 的 gold 检索命中反而从 0 增至 1，不能把这次答案变化解释成丢失 gold 所致。
 单题变化也不足以建立具体干扰机制的因果结论。</p>
 <details><summary>全部 ${qa.n_questions} 题的配对结果</summary><div class="table-wrap">${table(["题目","标准答案","全量 BGE","经验方案","变化"],outcomeRows)}</div></details>
 <p><a href="${esc(S.reader_comparison_path)}">问答验证产物</a> · <a href="changed_qa/protocol.json">相同 reader 设置与输入协议</a></p>`;
}else if(qa){
 document.getElementById("qa").innerHTML=`<p><a href="${esc(S.reader_comparison_path)}">独立问答验证产物</a></p><pre>${esc(JSON.stringify(qa,null,2))}</pre>`;
}else{
 document.getElementById("qa").innerHTML='<p>下游 reader 对照仍待验证。本报告的 Top-L 重合、gold 命中与成本不等同于问答正确率。</p>';
}
</script></body></html>
"""


def render(output: Path):
    output = Path(output).resolve()
    summary, disagreements = build_summary(output)
    write(output/"summary.json", summary)
    latency_fields = [
        "L", "n_measurements", "pooled_median_ms", "pooled_p95_ms",
        "mean_per_query_median_ms", "median_per_query_median_ms",
        "min_repetitions_per_query", "max_repetitions_per_query",
        "mean_paired_full_scan_per_query_median_ms", "mean_per_query_median_ratio_to_full",
    ]
    method_rows = []
    for row in summary["methods"]:
        flattened = {k: v for k, v in row.items() if k != "latency"}
        flattened.update({"latency_"+key: (row["latency"] or {}).get(key) for key in latency_fields})
        method_rows.append(flattened)
    csv_write(output/"methods.csv", method_rows)
    csv_write(output/"disagreements.csv", disagreements)
    encoded = json.dumps({"summary": summary, "disagreements": disagreements},
                         ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output/"report.html").write_text(_HTML.replace("__DATA__", encoded), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = render(args.output)
    row = next(r for r in summary["methods"] if r["scope"] == "all_8_371"
               and r["method"] == "empirical_max_both" and r["L"] == 5)
    print(json.dumps({"complete": True, "output": str(args.output.resolve()),
                      "method_rows": len(summary["methods"]),
                      "disagreement_rows": summary["disagreement_rows_all_empirical_all_L"],
                      "all_L5_primary": row}, ensure_ascii=False))


if __name__ == "__main__":
    main()
