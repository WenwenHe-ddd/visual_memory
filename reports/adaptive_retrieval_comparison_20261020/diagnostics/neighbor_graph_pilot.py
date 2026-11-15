#!/usr/bin/env python3
"""Fixed Home L=5 geometry-only graph pilots; no QA-dependent graph tuning."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scripts.analysis.compare_adaptive_retrieval import load_dataset

SOURCE = ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design"


def fit_graphs(vectors, groups):
    n, dimension = vectors.shape
    distances = squareform(pdist(vectors))
    # Numerical equality allowance only; no empirical similarity cutoff.
    tolerance = 64 * np.finfo(float).eps * (dimension + 1) * max(1.0, distances.max())
    medoids = [min(g, key=lambda i: (float(distances[i, g].sum()), i)) for g in groups]
    rng = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if not np.any(np.maximum(distances[i], distances[j]) < distances[i, j] - tolerance):
                rng[i].add(j)
                rng[j].add(i)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            previous = parent[i]
            parent[i] = parent[parent[i]]
            i = previous
        return i

    mst = []
    for _, i, j in sorted((distances[i, j], i, j) for i in range(n) for j in range(i)):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            mst.append((i, j))
    assert len(mst) == n - 1
    assert all(j in rng[i] and i in rng[j] for i, j in mst)
    mrng = [set() for _ in range(n)]
    for i in range(n):
        for j in sorted((x for x in range(n) if x != i), key=lambda x: (distances[i, x], x)):
            if not any(distances[j, k] < distances[i, j] - tolerance for k in mrng[i]):
                mrng[i].add(j)
    for i, j in mst:
        mrng[i].add(j)
        mrng[j].add(i)
    graphs = {name: [sorted(neighbors) for neighbors in adj]
              for name, adj in (("rng", rng), ("mrng_mst", mrng))}
    return graphs, medoids, mst, float(tolerance)


def search(query, vectors, order, adjacency, medoids, mode, limit=5):
    """Only discovered nodes receive scores. No oracle scores are arguments."""
    scores, frontier, expanded, trace = {}, [], [], []

    def discover(i):
        if i in scores:
            return False
        scores[i] = float(vectors[i] @ query)
        heapq.heappush(frontier, (-scores[i], i))
        return True

    seeds = [medoids[order[0]]] if mode == "single_master_seed" else [medoids[m] for m in order]
    for seed in seeds:
        discover(seed)
    while frontier:
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        if len(ranked) >= limit and frontier[0] > (-scores[ranked[limit - 1]], ranked[limit - 1]):
            break
        _, current = heapq.heappop(frontier)
        expanded.append(current)
        discovered = [neighbor for neighbor in adjacency[current] if discover(neighbor)]
        trace.append({"expanded": current, "newly_scored": discovered})
    selected = sorted(scores, key=lambda i: (-scores[i], i))[:limit]
    assert len(selected) == limit and len(expanded) == len(set(expanded))
    return {
        "selected_indices": selected, "selected_scores": [scores[i] for i in selected],
        "scored_indices": list(scores), "scored_leaves": len(scores),
        "scalar_coordinate_products": len(scores) * vectors.shape[1],
        "expanded_indices": expanded, "seeds": seeds, "trace": trace,
        "approximate": True, "global_topL_guaranteed": False,
    }


def run():
    inputs, queries, vectors, groups, orders = load_dataset(SOURCE)
    graphs, medoids, mst, tolerance = fit_graphs(vectors, groups)
    rows = []
    for method, adjacency in graphs.items():
        for mode in ("single_master_seed", "all_master_seeds"):
            for qi, query in enumerate(queries):
                result = search(query, vectors, orders[qi], adjacency, medoids, mode)
                result.update(method=method, mode=mode, question_index=qi,
                              question_id=inputs["questions"][qi]["id"], L=5)
                rows.append(result)
    # Full BGE scores are computed only after all approximate control traces end.
    oracle = queries @ vectors.T
    for row in rows:
        qi = row["question_index"]
        reference = np.argsort(-oracle[qi], kind="stable")[:5].tolist()
        selected = row["selected_indices"]
        assert np.max(np.abs(np.array(row["selected_scores"]) - oracle[qi, selected])) < 1e-12
        gold = set(inputs["questions"][qi]["gold_ids"])
        chosen = {inputs["leaves"][i]["memory_id"] for i in selected}
        full = {inputs["leaves"][i]["memory_id"] for i in reference}
        row.update(reference_indices=reference, exact_match=selected == reference,
                   top5_overlap=len(set(selected) & set(reference)) / 5,
                   gold_hits=len(chosen & gold), full_gold_hits=len(full & gold),
                   gold_total=len(gold), all_gold=gold <= chosen, any_gold=bool(gold & chosen),
                   full_all_gold=gold <= full, full_any_gold=bool(gold & full))
    summary = []
    for method in graphs:
        for mode in ("single_master_seed", "all_master_seeds"):
            for scope in ("all_52", "new_51_excluding_Q10"):
                selected = [r for r in rows if r["method"] == method and r["mode"] == mode
                            and (scope == "all_52" or not r["question_id"].endswith("/Q10"))]
                summary.append({
                    "method": method, "mode": mode, "scope": scope, "n_questions": len(selected),
                    "exact_matches": sum(r["exact_match"] for r in selected),
                    "mean_top5_overlap": float(np.mean([r["top5_overlap"] for r in selected])),
                    "mean_scored_leaves": float(np.mean([r["scored_leaves"] for r in selected])),
                    "min_scored_leaves": min(r["scored_leaves"] for r in selected),
                    "max_scored_leaves": max(r["scored_leaves"] for r in selected),
                    "gold_hits": sum(r["gold_hits"] for r in selected),
                    "full_gold_hits": sum(r["full_gold_hits"] for r in selected),
                    "gold_total": sum(r["gold_total"] for r in selected),
                    "all_gold_queries": sum(r["all_gold"] for r in selected),
                    "any_gold_queries": sum(r["any_gold"] for r in selected),
                    "full_all_gold_queries": sum(r["full_all_gold"] for r in selected),
                    "full_any_gold_queries": sum(r["full_any_gold"] for r in selected),
                })
    result = {
        "protocol": {
            "source": str(SOURCE), "L": 5, "additional_ef_or_beam": None,
            "graph_fit_inputs": "memory BGE text vectors and master memberships only",
            "rng": "strict relative-neighborhood lune; equality retains edge",
            "mrng_mst": "distance-ordered directed greedy pruning plus bidirectional MST",
            "seed": "within-master Euclidean medoid; first CLIP master or all master medoids",
            "all_seed_clip_role": "ordering only; no master is screened out",
            "stop": "best frontier stable score/index rank strictly worse than current TopL worst",
            "numerical_equality_tolerance": tolerance, "similarity_threshold": None,
            "gold_used_only_after_search": True, "oracle_computed_only_after_search": True,
            "strict_top5_guarantee": False, "new_reader_calls": 0,
            "conclusion": "None of the four fixed graph variants achieves near-full Top5 preservation; do not expand or tune using these QA.",
            "source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                              for name in ("inputs.json", "features.npz")},
        },
        "graph": {"medoids": medoids, "mst_edges": mst, "adjacency": graphs,
                  "directed_edge_counts": {name: sum(map(len, adj)) for name, adj in graphs.items()}},
        "summary": summary, "questions": rows,
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"saved": str(path), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    run()
