#!/usr/bin/env python3
"""Validate per-cluster BGE upper bounds and adaptive CLIP-order traversal.

Only reconstructible text features build the index. Original image pixels and
individual visual embeddings are never read. Gold labels and full score matrices
are used only after the adaptive searches, for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from scripts.analysis.probe_caption_visual_online import read_json, write_json, sha, normalize
from scripts.graph.text_similarity_bounds import TextBoundIndex

SOURCE = ROOT / "reports/home_online_routing_dialogue_caption_20261020"
OFFLINE = ROOT / "reports/home_caption_ward_visual_20261020"
BUDGETS = list(range(1, 11))


def adaptive_search(order, upper_bounds, member_indices, score_cluster, limit, record_trace=True):
    """The callback is the only way this search can obtain any leaf score.

    Strict upper-bound comparisons preserve the full-scan stable Top-L,
    including equal-score ties. Neither gold nor unvisited leaf scores enter
    the traversal. Upper bounds are independent of previously observed scores.
    """
    if not 1 <= limit <= sum(len(m) for m in member_indices):
        raise ValueError("Invalid final evidence count")
    if sorted(order) != list(range(len(member_indices))):
        raise ValueError("Traversal order must contain all clusters once")
    bounds = np.asarray(upper_bounds, dtype=np.float64)
    if bounds.shape != (len(order),) or not np.isfinite(bounds).all():
        raise ValueError("Invalid bounds")
    scored = {}
    selected = []
    visited, skipped, tail = [], [], []
    trace = []
    reason = "all_clusters_considered"
    for pos, master_index in enumerate(order):
        threshold = scored[selected[-1]] if len(selected) == limit else None
        remaining = order[pos:]
        remaining_max = float(bounds[remaining].max())
        base = {
            "position": pos + 1, "master_index": int(master_index),
            "upper_bound": float(bounds[master_index]),
            "threshold_before": threshold, "remaining_max_before": remaining_max,
            "remaining_count": len(remaining),
        }
        if threshold is not None and remaining_max < threshold:
            tail = [int(i) for i in remaining]
            if record_trace:
                trace.append({**base, "action": "stop", "threshold_after": threshold, "n_scores": 0})
            reason = "all_remaining_bounds_below_current_cutoff"
            break
        if threshold is not None and bounds[master_index] < threshold:
            skipped.append(int(master_index))
            if record_trace:
                trace.append({**base, "action": "skip", "threshold_after": threshold, "n_scores": 0})
            continue
        values = score_cluster(int(master_index))
        expected = member_indices[master_index]
        if len(values) != len(expected) or [i for i, _ in values] != expected:
            raise ValueError("Callback must return all current members in original order")
        for i, score in values:
            if i in scored or not np.isfinite(score):
                raise ValueError("Duplicate or invalid leaf score")
            scored[i] = float(score)
        visited.append(int(master_index))
        selected = sorted(scored, key=lambda i: (-scored[i], i))[:limit]
        after = scored[selected[-1]] if len(selected) == limit else None
        if record_trace:
            trace.append({**base, "action": "visit", "threshold_after": after, "n_scores": len(values)})
    assert len(selected) == limit
    assert sorted(visited + skipped + tail) == list(range(len(order)))
    return {
        "L": limit, "selected_indices": selected, "selected_scores": [scored[i] for i in selected],
        "visited_indices": visited, "skipped_indices": skipped, "tail_indices": tail,
        "visited_master_count": len(visited), "scored_leaf_count": len(scored),
        "clip_depth": max((order.index(i) + 1 for i in visited), default=0),
        "stop_reason": reason, "trace": trace,
    }


def encode_texts(inputs, source_protocol):
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    checkpoint = source_protocol["encoders"]["bge"]["checkpoint"]
    model_path = Path(checkpoint["model_path"])
    for name, expected in checkpoint["files_sha256"].items():
        assert sha(model_path / name) == expected, name
    prefix = source_protocol["encoders"]["bge"]["query_prefix"]
    torch.set_num_threads(4)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    texts = [prefix + q["question"] for q in inputs["questions"]]
    texts += [leaf["retrieval_text"] for leaf in inputs["leaves"]]
    lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
    assert max(lengths) <= model.max_seq_length
    vectors = normalize(model.encode(
        texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, prompt="",
    ))
    nq = len(inputs["questions"])
    return vectors[:nq], vectors[nq:], {
        "checkpoint": checkpoint, "query_prefix": prefix,
        "query_inputs": texts[:nq], "leaf_input": "original dialogue + '\\nMemory description: ' + full caption",
        "query_token_counts": lengths[:nq], "leaf_token_counts": lengths[nq:],
        "max_query_tokens": max(lengths[:nq]), "max_leaf_tokens": max(lengths[nq:]),
        "truncated_inputs": 0, "torch_threads": 4, "device": "cpu",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__},
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(question_indices, questions, runs, upper, actual):
    scope_runs = [r for r in runs if r["question_index"] in question_indices]
    slack = upper[question_indices] - actual[question_indices]
    budgets = []
    for limit in BUDGETS:
        selected = [r for r in scope_runs if r["L"] == limit]
        visits = [r["visited_master_count"] for r in selected]
        counts = [r["scored_leaf_count"] for r in selected]
        budgets.append({
            "L": limit, "exact_matches": sum(r["exact_match"] for r in selected),
            "mean_visited_masters": float(np.mean(visits)),
            "min_visited_masters": min(visits), "max_visited_masters": max(visits),
            "mean_scored_leaves": float(np.mean(counts)),
            "min_scored_leaves": min(counts), "max_scored_leaves": max(counts),
            "mean_total_vector_dots": float(upper.shape[1] + np.mean(counts)),
            "mean_clip_depth": float(np.mean([r["clip_depth"] for r in selected])),
            "stopped_early_count": sum(bool(r["tail_indices"]) for r in selected),
            "mean_skipped_masters": float(np.mean([len(r["skipped_indices"]) for r in selected])),
            "mean_masters_unvisited_at_stop": float(np.mean([len(r["tail_indices"]) for r in selected])),
            "mean_search_seconds": float(np.mean([r["search_seconds"] for r in selected])),
            "gold_hits": sum(r["gold_hits"] for r in selected),
            "any_gold_count": sum(r["gold_hits"] > 0 for r in selected),
            "all_gold_count": sum(r["gold_hits"] == len(questions[r["question_index"]]["gold_ids"])
                                  for r in selected),
        })
    return {
        "n_questions": len(question_indices), "n_query_master_pairs": int(slack.size),
        "n_gold_occurrences": sum(len(questions[i]["gold_ids"]) for i in question_indices),
        "bound_violations": int(np.count_nonzero(slack < 0)),
        "slack_mean": float(slack.mean()), "slack_median": float(np.median(slack)),
        "slack_p90": float(np.quantile(slack, .9)), "slack_min": float(slack.min()),
        "slack_max": float(slack.max()), "budgets": budgets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/home_text_similarity_bounds_20261020")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Use a fresh output directory.")
    start = time.perf_counter()
    inputs = read_json(SOURCE / "inputs.json")
    source_protocol = read_json(SOURCE / "protocol.json")
    previous = read_json(SOURCE / "results.json")["questions"]
    questions, leaves, masters = (inputs[key] for key in ("questions", "leaves", "masters"))
    assert (len(questions), len(leaves), len(masters)) == (52, 89, 25)
    graph = read_json(OFFLINE / "memory/graph.json")
    assert [q["id"] for q in questions] == [q["id"] for q in previous]
    assert [node["id"] for node in graph["image_nodes"]] == [leaf["image_id"] for leaf in leaves]
    for node, leaf in zip(graph["image_nodes"], leaves):
        assert leaf["retrieval_text"] == node["dialogue"] + "\nMemory description: " + node["caption"]
    index_by_id = {leaf["image_id"]: i for i, leaf in enumerate(leaves)}
    groups = [[index_by_id[key] for key in m["member_ids"]] for m in masters]
    master_by_id = {m["id"]: i for i, m in enumerate(masters)}
    orders = [[master_by_id[m["master_id"]] for m in q["masters"]] for q in previous]
    sources = [
        SOURCE / "inputs.json", SOURCE / "protocol.json", SOURCE / "results.json",
        SOURCE / "score_matrices.npz", OFFLINE / "memory/graph.json",
        OFFLINE / "memory/master_sums.npy", OFFLINE / "memory/basis.npy", Path(__file__),
        ROOT / "scripts/graph/text_similarity_bounds.py",
    ]
    source_hashes = {str(path.resolve()): sha(path) for path in sources}
    t0 = time.perf_counter()
    queries, leaf_vectors, encoder = encode_texts(inputs, source_protocol)
    encode_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    # The index fitter receives no questions, query vectors, labels or scores.
    index = TextBoundIndex.fit(leaf_vectors, groups)
    state = index.export_state()
    index_seconds = time.perf_counter() - t0
    upper_rows, ball_rows, mean_rows, runs = [], [], [], []
    bounds_seconds = []
    for qi, query in enumerate(queries):
        t0 = time.perf_counter()
        upper = index.upper_bounds(query)
        bounds_seconds.append(time.perf_counter() - t0)
        upper_rows.append(upper)
        # Only comparisons; these are NOT passed to the search controller.
        ball_rows.append(index.radius_upper_bounds(query))
        mean_rows.append(index.mean_scores(query))
        for limit in BUDGETS:
            calls = []

            def score_cluster(mi):
                calls.append(mi)
                members = groups[mi]
                values = leaf_vectors[members] @ query
                return [(j, float(value)) for j, value in zip(members, values)]

            t0 = time.perf_counter()
            result = adaptive_search(orders[qi], upper, groups, score_cluster, limit)
            elapsed = time.perf_counter() - t0
            assert calls == result["visited_indices"]
            result.update({"question_index": qi, "id": questions[qi]["id"],
                           "search_seconds": elapsed + bounds_seconds[-1]})
            runs.append(result)

    # Full-scan reference and gold annotations are evaluated AFTER traversal.
    oracle_start = time.perf_counter()
    scores = queries @ leaf_vectors.T
    oracle_seconds = time.perf_counter() - oracle_start
    upper = np.asarray(upper_rows)
    ball = np.asarray(ball_rows)
    means = np.asarray(mean_rows)
    actual = np.column_stack([scores[:, members].max(axis=1) for members in groups])
    exact_means = np.column_stack([scores[:, members].mean(axis=1) for members in groups])
    old_scores = np.load(SOURCE / "score_matrices.npz", allow_pickle=False)["bge_leaf"]
    source_score_error = float(np.max(np.abs(scores - old_scores)))
    assert source_score_error < 1e-6
    assert float(np.max(np.abs(means - exact_means))) < 1e-12
    assert np.all(upper >= actual), float((upper - actual).min())
    assert np.all(ball >= actual), float((ball - actual).min())
    for run in runs:
        qi, limit = run["question_index"], run["L"]
        reference = np.argsort(-scores[qi], kind="stable")[:limit].tolist()
        run["exact_match"] = run["selected_indices"] == reference
        assert run["exact_match"], (run["id"], limit)
        gold = set(questions[qi]["gold_ids"])
        run["gold_hits"] = sum(leaves[j]["memory_id"] in gold for j in run["selected_indices"])
        run["visited_master_ids"] = [masters[i]["id"] for i in run["visited_indices"]]
        run["skipped_master_ids"] = [masters[i]["id"] for i in run["skipped_indices"]]
        run["unvisited_master_ids"] = [masters[i]["id"] for i in run["tail_indices"]]
        run["selected_leaves"] = [
            {"image_id": leaves[j]["image_id"], "filename": leaves[j]["filename"],
             "score": float(scores[qi, j]), "is_gold": leaves[j]["memory_id"] in gold}
            for j in run["selected_indices"]
        ]
        for event in run["trace"]:
            event["master_id"] = (None if event["action"] == "stop"
                                  else masters[event["master_index"]]["id"])
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    indices = list(range(len(questions)))
    new_indices = [i for i in indices if not questions[i]["id"].endswith("/Q10")]
    numeric_bytes = sum(value.nbytes for value in state.values() if isinstance(value, np.ndarray))
    summary = {
        "scopes": {
            "new_questions": summarize(new_indices, questions, runs, upper, actual),
            "all_questions": summarize(indices, questions, runs, upper, actual),
        },
        "index_stats": {"n_clusters": len(masters), "dimension": leaf_vectors.shape[1],
                        "numeric_bytes": int(numeric_bytes)},
        "comparison": {"ball_slack_mean": float((ball - actual).mean()),
                       "spherical_cap_slack_mean": float((upper - actual).mean()),
                       "cap_tighter_than_ball_count": int(np.count_nonzero(upper < ball)),
                       "source_score_max_error": source_score_error},
    }
    audits = []
    for qi, question in enumerate(questions):
        run = next(r for r in runs if r["question_index"] == qi and r["L"] == 5)
        statuses = {i: "visit" for i in run["visited_indices"]}
        statuses.update({i: "skip" for i in run["skipped_indices"]})
        statuses.update({i: "stop" for i in run["tail_indices"]})
        gold = set(question["gold_ids"])
        audits.append({
            "id": question["id"], "question": question["question"],
            "gold_targets": previous[qi]["gold_targets"], "diagnostic_L5": run,
            "masters": [{
                "master_rank": rank, "master_id": masters[mi]["id"], "n": len(groups[mi]),
                "upper_bound": float(upper[qi, mi]), "ball_upper_bound": float(ball[qi, mi]),
                "actual_max": float(actual[qi, mi]), "mean_score": float(means[qi, mi]),
                "slack": float(upper[qi, mi] - actual[qi, mi]),
                "gold_count": sum(leaves[j]["memory_id"] in gold for j in groups[mi]),
                "action_L5": statuses[mi],
            } for rank, mi in enumerate(orders[qi], 1)],
        })
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "text_index.npz", **state)
    np.save(output / "query_vectors.npy", queries, allow_pickle=False)
    np.savez_compressed(output / "score_audit_matrices.npz", bge_leaf=scores,
                        upper_bound=upper, ball_upper_bound=ball, actual_max=actual, mean_score=means)
    write_json(output / "index_metadata.json", {
        "source_inputs": str(SOURCE / "inputs.json"),
        "source_inputs_sha256": sha(SOURCE / "inputs.json"),
        "master_ids": [m["id"] for m in masters], "member_indices": groups,
        "leaf_ids": [leaf["image_id"] for leaf in leaves],
        "text_model": encoder["checkpoint"], "leaf_input_format": encoder["leaf_input"],
        "fit_has_query_or_gold_input": False,
        "storage": "external reconstructible text index, separate from visual masters",
        "invalidate_on": ["membership change", "dialogue or caption change", "BGE encoder change"],
        "state_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                         for key, value in state.items() if isinstance(value, np.ndarray)},
    })
    write_json(output / "summary.json", summary)
    write_json(output / "adaptive_results.json", {"runs": runs})
    with (output / "query_audit.jsonl").open("w") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(output / "upper_bounds.csv", [
        {"question_id": q["id"], **master} for q in audits for master in q["masters"]])
    write_csv(output / "search_metrics.csv", [{
        "question_id": r["id"], "L": r["L"], "visited_masters": r["visited_master_count"],
        "scored_leaves": r["scored_leaf_count"], "total_vector_dots": 25 + r["scored_leaf_count"],
        "clip_depth": r["clip_depth"], "skipped_masters": len(r["skipped_indices"]),
        "tail_masters": len(r["tail_indices"]), "exact_match": r["exact_match"],
        "gold_hits": r["gold_hits"], "search_seconds": r["search_seconds"],
    } for r in runs])
    write_json(output / "protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "text_encoder": encoder,
        "fit": "all 89 current dialogue+caption BGE unit vectors; no query or gold",
        "upper_bound": "spherical cap around normalized mean; minimum member cosine; conservative float64 expansion",
        "comparison_bound": "query dot unnormalized mean + max Euclidean member radius",
        "ordering": "unchanged saved CLIP master ranking for each question",
        "search": "visit next CLIP-ordered cluster if its bound can reach current Top-L cutoff; stop only if ALL remaining bounds are strictly below cutoff",
        "ties": "strict pruning; leaf original-index order for equal scores",
        "evidence_counts_evaluated": BUDGETS, "default_evidence_count": None,
        "L5_role": "diagnostic continuation of prior Top5 comparison, not a fixed master count or new production default",
        "search_inputs": "CLIP order, per-cluster upper bounds, member index lists, and current-cluster score callback only",
        "oracle": "full leaf score matrix and gold used only AFTER adaptive searches; no original image encoding",
        "retained_image_vectors": 0, "retained_leaf_text_vectors": 0,
        "retained_query_vectors": "evaluation artifact only; queries are text",
        "memory_modified": False, "qa_accuracy_evaluated": False,
        "references": ["https://scikit-learn.org/stable/modules/neighbors.html#ball-tree"],
        "timing_seconds": {"text_load_hash_encode": encode_seconds, "index_fit": index_seconds,
                           "all_query_bounds": float(sum(bounds_seconds)), "full_score_matrix_for_audit": oracle_seconds,
                           "total_before_writing": time.perf_counter() - start},
        "cost_note": "25 bound direction dot products plus visited-leaf dot products; bounds include scalar geometry and Python traversal overhead, so fewer leaves is not measured end-to-end speedup",
        "limitations": [
            "guarantee is exact BGE Top-L, not gold recall or answer sufficiency",
            "bound may be loose; high-dimensional geometry can require most clusters",
            "text cache must be rebuilt or invalidated when text/model/membership changes",
            "finite-precision bounds use conservative rounding, not certified arbitrary-precision interval arithmetic",
        ],
    })
    write_json(output / "validation.json", {
        "passed": True, "query_cluster_pairs": int(upper.size),
        "spherical_cap_violations": int(np.count_nonzero(upper < actual)),
        "ball_violations": int(np.count_nonzero(ball < actual)),
        "adaptive_runs": len(runs), "exact_full_scan_topL_matches": sum(r["exact_match"] for r in runs),
        "source_score_max_error": source_score_error,
        "mean_score_identity_max_error": float(np.max(np.abs(means - exact_means))),
        "all_callbacks_match_visited_clusters": True,
        "all_source_hashes_unchanged": True, "truncated_inputs": 0,
        "original_images_read": 0, "image_encoder_calls": 0, "memory_modified": False,
    })
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
