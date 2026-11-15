#!/usr/bin/env python3
"""Compare answer-free adaptive retrieval against exhaustive BGE scoring.

The final evidence count controls output size, never a cluster visit budget.
Full scores and gold are evaluated only after all search calls for a domain.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from scripts.analysis.evaluate_text_similarity_bounds import adaptive_search
from scripts.graph.text_similarity_bounds import TextBoundIndex


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def array_bytes(obj, seen=None):
    """Count owned/reference array payloads once, excluding Python metadata."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(array_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(array_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return array_bytes(vars(obj), seen)
    if hasattr(type(obj), "__slots__"):
        return sum(array_bytes(getattr(obj, key), seen) for key in type(obj).__slots__
                   if hasattr(obj, key))
    return 0


class FullScan:
    def __init__(self, vectors, groups):
        self.vectors = vectors
        self.groups = groups

    def search(self, query, order, limit, record_trace=True):
        scores = self.vectors @ query
        selected = np.argsort(-scores, kind="stable")[:limit]
        return {
            "selected_indices": selected.tolist(),
            "selected_scores": scores[selected].tolist(),
            "scored_leaves": len(scores), "visited_masters": len(self.groups),
            "scalar_coordinate_products": self.vectors.size,
            "bound_dot_count": 0, "exact_leaf_dot_count": len(scores),
            "trace": [],
        }


class CapSearch:
    def __init__(self, vectors, groups, policy):
        self.vectors = vectors
        self.groups = groups
        self.index = TextBoundIndex.fit(vectors, groups)
        self.policy = policy

    def search(self, query, order, limit, record_trace=True):
        upper = self.index.upper_bounds(query)
        if self.policy == "best_bound":
            # Use CLIP rank only as a deterministic tie-breaker.
            ranks = {int(mi): k for k, mi in enumerate(order)}
            order = sorted(order, key=lambda mi: (-upper[mi], ranks[mi]))
        else:
            order = list(order)

        def score(mi):
            members = self.groups[mi]
            values = self.vectors[members] @ query
            return [(i, float(s)) for i, s in zip(members, values)]

        result = adaptive_search(order, upper, self.groups, score, limit, record_trace=record_trace)
        result.update({
            "scored_leaves": result["scored_leaf_count"],
            "visited_masters": result["visited_master_count"],
            "bound_dot_count": len(self.groups),
            "exact_leaf_dot_count": result["scored_leaf_count"],
            "scalar_coordinate_products":
                (len(self.groups) + result["scored_leaf_count"]) * self.vectors.shape[1],
        })
        if not record_trace:
            result["trace"] = []
        return result


def load_dataset(path):
    inputs = read(path / "inputs.json")
    features = np.load(path / "features.npz", allow_pickle=False)
    queries, vectors = features["bge_query"], features["bge_leaf"]
    orders = features["clip_order"].astype(int).tolist()
    leaves, masters = inputs["leaves"], inputs["masters"]
    indices = {leaf["image_id"]: i for i, leaf in enumerate(leaves)}
    groups = [[indices[key] for key in m["member_ids"]] for m in masters]
    assert sorted(i for g in groups for i in g) == list(range(len(leaves)))
    assert len(set(leaf["memory_id"] for leaf in leaves)) == len(leaves), \
        "This pilot compares image leaves to unique memory rounds."
    assert queries.shape[0] == len(inputs["questions"])
    assert vectors.shape[0] == len(leaves)
    assert queries.shape[1] == vectors.shape[1]
    assert np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)) < 1e-12
    assert np.max(np.abs(np.linalg.norm(queries, axis=1) - 1)) < 1e-12
    assert all(sorted(o) == list(range(len(masters))) for o in orders)
    return inputs, queries, vectors, groups, orders


def normalize_result(result):
    """All implementations expose these shared measurable quantities."""
    result = native(result)
    for key in ("selected_indices", "selected_scores", "scored_leaves",
                "visited_masters", "scalar_coordinate_products"):
        if key not in result:
            raise KeyError(f"Missing standardized search result field: {key}; {list(result)}")
    if isinstance(result["scored_leaves"], list):
        result["scored_leaf_indices"] = result["scored_leaves"]
        result["scored_leaves"] = len(result["scored_leaves"])
    if isinstance(result["visited_masters"], list):
        result["visited_master_indices"] = result["visited_masters"]
        result["visited_masters"] = len(result["visited_masters"])
    return result


def factories(vectors, groups):
    from scripts.graph.hierarchical_text_bounds import HierarchicalTextBoundIndex
    from scripts.graph.progressive_text_bounds import ProgressiveTextBounds
    from scripts.graph.observed_spread_search import ObservedSpreadSearch

    return {
        "full_scan": lambda: FullScan(vectors, groups),
        "spherical_cap_clip_order": lambda: CapSearch(vectors, groups, "clip"),
        "spherical_cap_best_bound": lambda: CapSearch(vectors, groups, "best_bound"),
        "pivot_tree_clip_order": lambda: HierarchicalTextBoundIndex.fit(vectors, groups),
        "pivot_tree_best_bound": lambda: HierarchicalTextBoundIndex.fit(vectors, groups),
        "progressive_text_pca": lambda: ProgressiveTextBounds.fit(vectors, groups),
        "empirical_offset": lambda: ObservedSpreadSearch.fit(vectors, groups),
        "empirical_radius_ratio": lambda: ObservedSpreadSearch.fit(vectors, groups),
        "empirical_max_both": lambda: ObservedSpreadSearch.fit(vectors, groups),
        "empirical_max_both_reuse": lambda: ObservedSpreadSearch.fit(vectors, groups),
    }


def search(index, method, query, order, limit, record_trace):
    kwargs = {"record_trace": record_trace}
    if method.startswith("pivot_tree"):
        kwargs["policy"] = "clip" if method.endswith("clip_order") else "best_bound"
    if method.startswith("empirical_"):
        kwargs["prediction"] = method.removeprefix("empirical_").removesuffix("_reuse")
        kwargs["reuse_centroid"] = method.endswith("_reuse")
    return index.search(query, order, limit, **kwargs)


def summarize(runs, question_indices, inputs, vectors):
    selected = [r for r in runs if r["question_index"] in question_indices]
    aggregate = []
    for method, limit in sorted({(r["method"], r["L"]) for r in selected}):
        rows = [r for r in selected if r["method"] == method and r["L"] == limit]
        if not rows:
            continue
        aggregate.append({
            "method": method, "L": limit, "n_questions": len(rows),
            "exact_matches": sum(r["exact_match"] for r in rows),
            "exact_match_rate": float(np.mean([r["exact_match"] for r in rows])),
            "mean_topL_overlap": float(np.mean([r["topL_overlap"] for r in rows])),
            "mean_scored_leaves": float(np.mean([r["scored_leaves"] for r in rows])),
            "mean_visited_masters": float(np.mean([r["visited_masters"] for r in rows])),
            "mean_coordinate_products": float(np.mean([r["scalar_coordinate_products"] for r in rows])),
            "coordinate_product_ratio_to_full": float(np.mean(
                [r["scalar_coordinate_products"] / vectors.size for r in rows])),
            "gold_hits": sum(r["gold_hits"] for r in rows),
            "gold_total": sum(r["gold_total"] for r in rows),
            "any_gold_queries": sum(r["gold_hits"] > 0 for r in rows),
            "all_gold_queries": sum(r["gold_hits"] == r["gold_total"] for r in rows),
            "max_score_error": max(r["score_error"] for r in rows),
        })
    return aggregate


def evaluate_domain(source, output, methods, repetitions):
    inputs, queries, vectors, groups, orders = load_dataset(source)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Refusing to replace completed evaluation: {output}")
    all_factories = factories(vectors, groups)
    indexes, fits = {}, {}
    for method in methods:
        t0 = time.perf_counter()
        indexes[method] = all_factories[method]()
        fits[method] = {"seconds": time.perf_counter() - t0,
                        "numeric_index_bytes": array_bytes(indexes[method])}
    runs = []
    for qi, query in enumerate(queries):
        for limit in (1, 3, 5, 10):
            if limit > len(vectors):
                continue
            for method in methods:
                result = normalize_result(search(indexes[method], method, query, orders[qi], limit, limit == 5))
                result.update({"method": method, "L": limit, "question_index": qi,
                               "id": inputs["questions"][qi]["id"]})
                runs.append(result)
    # This full score matrix is evaluation-only, never a controller argument.
    scores = queries @ vectors.T
    for result in runs:
        qi, limit = result["question_index"], result["L"]
        reference = np.argsort(-scores[qi], kind="stable")[:limit].tolist()
        selected = result["selected_indices"]
        assert len(selected) == len(set(selected)) == limit
        assert all(0 <= i < len(vectors) for i in selected)
        result["reference_indices"] = reference
        result["exact_match"] = selected == reference
        result["topL_overlap"] = len(set(selected) & set(reference)) / limit
        result["score_error"] = float(np.max(np.abs(np.asarray(result["selected_scores"]) - scores[qi, selected])))
        result["selected_memory_ids"] = [inputs["leaves"][i]["memory_id"] for i in selected]
        gold = set(inputs["questions"][qi]["gold_ids"])
        result["gold_hits"] = len(gold & set(result["selected_memory_ids"]))
        result["gold_total"] = len(gold)
    # Every timing uses an already encoded, normalized query. Repeated calls
    # recompute all query-dependent index information; no result cache exists.
    # Random interleaving avoids always giving one implementation the warm CPU.
    rng = np.random.default_rng(20261020)
    timings = {method: [] for method in methods}
    for qi, query in enumerate(queries):
        for method in methods:
            search(indexes[method], method, query, orders[qi], 5, False)
        for repeat in range(repetitions):
            for method in rng.permutation(methods):
                t0 = time.perf_counter_ns()
                result = search(indexes[method], method, query, orders[qi], 5, False)
                elapsed = time.perf_counter_ns() - t0
                # Integrity checked outside the timed interval.
                expected = next(r["selected_indices"] for r in runs
                                if r["method"] == method and r["L"] == 5 and r["question_index"] == qi)
                assert native(result["selected_indices"]) == expected
                timings[method].append({"question_index": qi, "repeat": repeat,
                                        "nanoseconds": elapsed})
    image_indices = [i for i, q in enumerate(inputs["questions"]) if q.get("query_image_paths")]
    text_indices = [i for i in range(len(queries)) if i not in image_indices]
    scope = {
        "all": summarize(runs, set(range(len(queries))), inputs, vectors),
        "text_only": summarize(runs, set(text_indices), inputs, vectors),
        "image_assisted": summarize(runs, set(image_indices), inputs, vectors),
    }
    elapsed_stats = {}
    for method, values in timings.items():
        times = np.array([row["nanoseconds"] for row in values]) / 1e6
        per_query = [np.median([r["nanoseconds"] for r in values if r["question_index"] == qi]) / 1e6
                     for qi in range(len(queries))]
        elapsed_stats[method] = {
            "repetitions_per_query": repetitions, "measurements": len(times),
            "median_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95)),
            "mean_of_per_query_median_ms": float(np.mean(per_query)),
            "per_query_median_ms": [float(v) for v in per_query],
        }
    summary = {
        "domain": source.name, "n_questions": len(queries), "n_leaves": len(vectors),
        "n_masters": len(groups), "n_image_assisted_questions": len(image_indices),
        "methods": methods, "scopes": scope, "fit": fits, "latency": elapsed_stats,
        "all_exact": all(r["exact_match"] for r in runs),
        "max_score_error": max(r["score_error"] for r in runs),
        "shared_clip_master_coordinate_products_per_query": len(groups) * 512,
        "latency_boundary": "Already encoded queries; index traversal + dot products + topL. No encoding, file IO, or reader.",
        "numeric_cost_boundary": "Vector coordinate products; scalar bounds/heap/sort/normalization overhead is reflected in measured latency, not this count.",
        "source_hashes": {str(p.resolve()): sha(p) for p in [source / "inputs.json", source / "features.npz"]},
    }
    if source.name == "Home_Renovation_Interior_Design":
        summary["expansion_gate_passed_methods"] = [
            r["method"] for r in scope["all"] if r["L"] == 5 and r["method"] != "full_scan"
            and r["exact_matches"] == len(queries) and r["coordinate_product_ratio_to_full"] < 1
        ]
    write(output / "summary.json", summary)
    write(output / "runs.json", runs)
    write(output / "latency_samples.json", timings)
    np.savez(output / "evaluation_scores.npz", bge=scores)
    columns = ["method", "L", "id", "exact_match", "topL_overlap", "scored_leaves",
               "visited_masters", "scalar_coordinate_products", "gold_hits", "gold_total", "score_error"]
    with (output / "per_query.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: r[key] for key in columns} for r in runs)
    print(json.dumps({"domain": source.name, "all_exact": summary["all_exact"],
                      "L5": [r for r in scope["all"] if r["L"] == 5],
                      "gate": summary.get("expansion_gate_passed_methods")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "reports/adaptive_retrieval_comparison_20261020/inputs_by_domain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=[
        "full_scan", "spherical_cap_clip_order", "spherical_cap_best_bound",
        "pivot_tree_clip_order", "pivot_tree_best_bound", "progressive_text_pca"])
    parser.add_argument("--repetitions", type=int, default=11)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    for domain in args.domains:
        evaluate_domain(args.input_root / domain, args.output / domain, args.methods, args.repetitions)


if __name__ == "__main__":
    main()
