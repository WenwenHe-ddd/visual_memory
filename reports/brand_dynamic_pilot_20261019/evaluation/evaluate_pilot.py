"""Post-hoc Brand evaluation, never imported by the admission algorithm.

API: evaluate({full_image_id: master_id}, name="pilot") -> JSON-serializable dict.
CLI: python3 evaluate_pilot.py --graph GRAPH --output-dir DIRECTORY [--name NAME]
The input may contain graph edges, an assignments mapping, a direct ID mapping,
or a list of assignment records (image_id/id plus master_id/cluster_id).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST_PATH = HERE / "evaluation_manifest.json"
FINDINGS_PATH = HERE / "evaluation_findings.json"
PAIRWISE_PATH = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[dict, dict, np.ndarray]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    findings = json.loads(FINDINGS_PATH.read_text())
    if _sha(PAIRWISE_PATH) != findings["evaluation_scalar_distance_source_sha256"]:
        raise ValueError("Pairwise cosine source changed since visual cases were frozen")
    index = {r["id"]: i for i, r in enumerate(manifest["images"])}
    matrix = np.zeros((len(index), len(index)), dtype=float)
    seen = set()
    with PAIRWISE_PATH.open() as handle:
        for row in csv.DictReader(handle):
            if row["task"] != manifest["task"]:
                continue
            i, j = index[row["earlier_id"]], index[row["later_id"]]
            key = tuple(sorted((i, j)))
            if i == j or key in seen:
                raise ValueError("Duplicate/self pair in evaluation cosine source")
            seen.add(key)
            cosine = float(row["cosine"])
            if not np.isfinite(cosine) or not -1.000000000001 <= cosine <= 1.000000000001:
                raise ValueError("Invalid evaluation cosine")
            matrix[i, j] = matrix[j, i] = 1.0 - np.clip(cosine, -1.0, 1.0)
    if len(seen) != len(index) * (len(index) - 1) // 2:
        raise ValueError("Evaluation cosine source does not cover every Brand pair")
    return manifest, findings, matrix


def evaluate(assignments: Mapping[str, Any], *, name: str = "pilot") -> dict:
    """Evaluate final memberships only. No clustering or assignment decisions occur."""
    report: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "scope": "post_hoc_evaluation_only",
        "errors": [],
        "caveats": [
            "Brand labels are a secondary semantic proxy, not exhaustive visual ground truth.",
            "Cosine silhouette measures geometry in the same CLIP encoder, not independent semantic correctness.",
            "Visual sanity cases were frozen before new pilot memberships, but informed by baseline failures; they are diagnostic, not held-out ground truth.",
            "Visual cases were inspected by an AI assistant, not independently annotated by human raters.",
            "Pairwise scalar distances are read only after admission finishes; they must not be used as dynamic admission input.",
            "This evaluator does not assess streaming look-ahead, vector retention, runtime, PCA approximation or payload preservation; audit those separately.",
        ],
    }
    try:
        manifest, findings, matrix = _inputs()
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(str(exc))
        return report
    rows = manifest["images"]
    expected = {r["id"] for r in rows}
    if not isinstance(assignments, Mapping):
        report["errors"].append("assignments must be an image-ID to master-ID mapping")
        return report
    received = set(assignments)
    missing, extra = sorted(expected - received), sorted(received - expected)
    if missing:
        report["errors"].append({"missing_image_ids": missing})
    if extra:
        report["errors"].append({"unexpected_image_ids": extra})
    invalid = [key for key, value in assignments.items() if value is None or isinstance(value, (dict, list, tuple, bool)) or not str(value).strip()]
    if invalid:
        report["errors"].append({"invalid_master_ids_for_images": invalid})
    if report["errors"]:
        return report
    normalized = {key: str(value) for key, value in assignments.items()}
    true = [r["filename_brand_evaluation_label"] for r in rows]
    pred = [normalized[r["id"]] for r in rows]
    cluster_members = defaultdict(list)
    for row in rows:
        cluster_members[normalized[row["id"]]].append(row)
    k = len(cluster_members)
    tp = fp = fn = tn = 0
    intra = []
    inter = []
    for i, j in combinations(range(len(rows)), 2):
        same_true, same_pred = true[i] == true[j], pred[i] == pred[j]
        tp += same_true and same_pred
        fp += not same_true and same_pred
        fn += same_true and not same_pred
        tn += not same_true and not same_pred
        (intra if same_pred else inter).append(float(1.0 - matrix[i, j]))
    silhouette = float(silhouette_score(matrix, pred, metric="precomputed")) if 1 < k < len(rows) else None
    if silhouette is None:
        report["caveats"].append("Silhouette is undefined with one cluster or one cluster per image; reported as null.")
    cases = []
    for original in findings["visual_cases"]:
        case = dict(original)
        same = normalized[case["left_id"]] == normalized[case["right_id"]]
        relation = case["relation_evaluation_only"]
        expected_same = True if relation == "close" else False if relation in {"different", "different_same_brand"} else None
        case.update({"pilot_left_master": normalized[case["left_id"]], "pilot_right_master": normalized[case["right_id"]], "pilot_same_cluster": same, "expected_same_cluster_for_sanity_check": expected_same, "sanity_check_passed": None if expected_same is None else same == expected_same})
        cases.append(case)
    positive = [c for c in cases if c["expected_same_cluster_for_sanity_check"] is True]
    negative = [c for c in cases if c["expected_same_cluster_for_sanity_check"] is False]
    def brief(values: list[float]) -> dict:
        return {"count": len(values), "mean": float(np.mean(values)) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}
    report.update({
        "data": {"image_count": len(rows), "task": manifest["task"], "evaluation_manifest_sha256": _sha(MANIFEST_PATH), "frozen_cases_sha256": _sha(FINDINGS_PATH), "pairwise_cosines_sha256": _sha(PAIRWISE_PATH)},
        "clusters": {"count": k, "sizes": sorted((len(v) for v in cluster_members.values()), reverse=True), "size_histogram": {str(size): count for size, count in sorted(Counter(map(len, cluster_members.values())).items())}, "memberships": {key: [r["id"] for r in group] for key, group in cluster_members.items()}},
        "geometry": {"cosine_silhouette": silhouette, "within_cluster_cosine": brief(intra), "between_cluster_cosine": brief(inter)},
        "secondary_brand_proxy": {"adjusted_rand": float(adjusted_rand_score(true, pred)), "normalized_mutual_information": float(normalized_mutual_info_score(true, pred)), "pair_tp_fp_fn_tn": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}, "pair_precision": float(tp / (tp + fp)) if tp + fp else None, "pair_recall": float(tp / (tp + fn)) if tp + fn else None, "pair_f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None, "precision_zero_denominator_policy": "null when no predicted positive pairs"},
        "visual_sanity": {"close_pairs_kept": sum(c["sanity_check_passed"] for c in positive), "close_pair_count": len(positive), "different_pairs_separated": sum(c["sanity_check_passed"] for c in negative), "different_pair_count": len(negative), "unscored_ambiguous_or_cross_brand_count": len(cases) - len(positive) - len(negative), "cases": cases},
        "baseline_reference": findings["baseline"],
    })
    return report


def read_assignments(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "assignments" in obj:
        obj = obj["assignments"]
    elif isinstance(obj, dict) and "edges" in obj:
        obj = obj["edges"]
    if isinstance(obj, dict):
        return obj
    if not isinstance(obj, list):
        raise ValueError("Input must be a graph, assignments mapping or assignments list")
    output = {}
    for row in obj:
        image_id = row.get("image_id", row.get("id"))
        master_id = row.get("master_id", row.get("cluster_id"))
        if image_id is None or master_id is None:
            raise ValueError("Assignment row needs image_id/id and master_id/cluster_id")
        if image_id in output:
            raise ValueError(f"Duplicate assignment for {image_id}")
        output[image_id] = master_id
    return output


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    """Save JSON and an original-image gallery. Images are never copied or edited."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    html_path = output_dir / "gallery.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_id = {row["id"]: row for row in manifest["images"]}
    esc = html.escape
    pieces = ["<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#172033;background:#f5f6fa}h1{font-size:24px}h2{font-size:20px}.notice{padding:14px;background:#fff2ce;border-radius:8px}.cluster,.case{margin:24px 0;padding:18px;background:white;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}figure{margin:0}img{width:100%;height:230px;object-fit:contain;background:#eee}figcaption{font-size:13px;overflow-wrap:anywhere}details{margin-top:8px}.pass{color:#14713c}.fail{color:#b52828}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:7px 14px;border:1px solid #ccd}</style>",
        f"<title>{esc(report.get('name', 'Brand pilot'))} — Brand evaluation</title><body><h1>{esc(report.get('name', 'Brand pilot'))} — Brand 视觉聚类核查</h1>",
        "<p class='notice'>仅用于构图完成后的评估。品牌名和图像描述没有作为视觉聚类输入。品牌一致不等于视觉一致；下列案例为模型目视诊断，并非独立人工真值。图片直接引用原文件，没有复制或改图。</p>"]
    if report["errors"]:
        pieces.append(f"<h2>评估错误</h2><pre>{esc(json.dumps(report['errors'], ensure_ascii=False, indent=2))}</pre>")
    else:
        g, s, b = report["geometry"], report["visual_sanity"], report["secondary_brand_proxy"]
        def value(number: Any) -> str:
            return "不适用" if number is None else f"{number:.4f}" if isinstance(number, float) else str(number)
        summary = [("图片 / master", f"30 / {report['clusters']['count']}"), ("CLIP cosine silhouette", value(g["cosine_silhouette"])), ("相近案例同簇", f"{s['close_pairs_kept']} / {s['close_pair_count']}"), ("不同案例分离", f"{s['different_pairs_separated']} / {s['different_pair_count']}"), ("品牌辅助 pair F1 / ARI", f"{value(b['pair_f1'])} / {value(b['adjusted_rand'])}")]
        pieces.append("<table>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in summary) + "</table>")
        def image_card(image_id: str) -> str:
            row = by_id[image_id]
            url = quote(os.path.relpath(row["image_path"], output_dir).replace(os.sep, "/"), safe="/")
            return f"<figure><a href='{esc(url, quote=True)}' target='_blank'><img loading='lazy' src='{esc(url, quote=True)}' alt='{esc(row['filename'], quote=True)}'></a><figcaption><strong>{esc(row['filename'])}</strong><br>{esc(image_id)}<br>{esc(row['timestamp'])}<details><summary>视觉描述（仅评估）</summary>{esc(row['visual_description_evaluation_only'])}</details></figcaption></figure>"
        pieces.append("<h2>最终各簇</h2>")
        for master_id, ids in report["clusters"]["memberships"].items():
            pieces.append(f"<section class='cluster'><h3>{esc(master_id)} · {len(ids)} 张</h3><div class='grid'>" + "".join(image_card(i) for i in ids) + "</div></section>")
        pieces.append("<h2>预先固定的视觉案例</h2>")
        for case in s["cases"]:
            passed = case["sanity_check_passed"]
            cls, status = ("pass", "符合诊断预期") if passed is True else ("fail", "未符合诊断预期") if passed is False else ("", "关系有歧义，不计对错")
            relation = "同簇" if case["pilot_same_cluster"] else "分簇"
            pieces.append(f"<section class='case'><h3 class='{cls}'>{esc(case['left_image'])} ↔ {esc(case['right_image'])}：{relation} · {status}</h3><p>{esc(case['reason'])}</p><p>关系类型：{esc(case['relation_evaluation_only'])}；CLIP cosine={case['cosine_from_prior_diagnostic']:.4f}；master：{esc(case['pilot_left_master'])} / {esc(case['pilot_right_master'])}</p><div class='grid'>" + image_card(case["left_id"]) + image_card(case["right_id"]) + "</div></section>")
    pieces.append("<h2>解释范围</h2><ul>" + "".join(f"<li>{esc(c)}</li>" for c in report["caveats"]) + "</ul></body></html>")
    html_path.write_text("\n".join(pieces) + "\n")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="pilot")
    args = parser.parse_args()
    report = evaluate(read_assignments(args.graph), name=args.name)
    report["input_graph_path"] = str(args.graph.resolve())
    report["input_graph_sha256"] = _sha(args.graph)
    paths = write_report(report, args.output_dir)
    print(json.dumps({"errors": report["errors"], "outputs": list(map(str, paths)), "geometry": report.get("geometry"), "visual_sanity_counts": {k: v for k, v in report.get("visual_sanity", {}).items() if k != "cases"}}, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
