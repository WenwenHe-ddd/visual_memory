#!/usr/bin/env python3
"""Independent saved-bound and traversal audit without text/image encoding."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path("/storage/hewenwen/VisualMemory")
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/home_online_routing_dialogue_caption_20261020"
sys.path.insert(0, str(ROOT / ".vendor"))
import numpy as np

checks = []
maxima = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=2e-11):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.))
    family = name.split("/")[0]
    maxima[family] = max(maxima.get(family, 0.), error)
    check(name, error <= tolerance)


def main():
    started = time.perf_counter()
    protocol = read(OUT / "protocol.json")
    summary = read(OUT / "summary.json")
    metadata = read(OUT / "index_metadata.json")
    inputs = read(SOURCE / "inputs.json")
    previous = read(SOURCE / "results.json")["questions"]
    source_protocol = read(SOURCE / "protocol.json")
    questions, leaves, masters = (inputs[key] for key in ("questions", "leaves", "masters"))
    for path, expected in protocol["source_sha256"].items():
        check(f"source_sha/{path}", sha(path) == expected)
    check("source_inputs_hash", sha(SOURCE / "inputs.json") == metadata["source_inputs_sha256"])
    check("preserved_query_prefix", protocol["text_encoder"]["query_prefix"] ==
          source_protocol["encoders"]["bge"]["query_prefix"])
    check("preserved_query_texts", protocol["text_encoder"]["query_inputs"] ==
          source_protocol["encoders"]["bge"]["query_inputs"])
    check("preserved_text_checkpoint", protocol["text_encoder"]["checkpoint"] ==
          source_protocol["encoders"]["bge"]["checkpoint"])
    check("no_default_L", protocol["default_evidence_count"] is None)
    check("all_diagnostic_budgets", protocol["evidence_counts_evaluated"] == list(range(1, 11)))
    check("no_memory_changes", protocol["memory_modified"] is False)
    check("no_qa_accuracy_claim", protocol["qa_accuracy_evaluated"] is False)
    check("no_retained_leaf_text_or_visual_vectors",
          protocol["retained_leaf_text_vectors"] == protocol["retained_image_vectors"] == 0)
    check("no_query_or_gold_fit_flag", metadata["fit_has_query_or_gold_input"] is False)
    image_index = {leaf["image_id"]: i for i, leaf in enumerate(leaves)}
    groups = [[image_index[member] for member in master["member_ids"]] for master in masters]
    master_index = {master["id"]: i for i, master in enumerate(masters)}
    orders = [[master_index[master["master_id"]] for master in question["masters"]]
              for question in previous]
    check("index_master_order", metadata["master_ids"] == [master["id"] for master in masters])
    check("index_leaf_order", metadata["leaf_ids"] == [leaf["image_id"] for leaf in leaves])
    check("index_groups", metadata["member_indices"] == groups)
    check("partition_covers89once", sorted(i for group in groups for i in group) == list(range(89)))

    state = dict(np.load(OUT / "text_index.npz", allow_pickle=False))
    vectors = np.load(OUT / "query_vectors.npy", allow_pickle=False)
    matrices = dict(np.load(OUT / "score_audit_matrices.npz", allow_pickle=False))
    scores = matrices["bge_leaf"]
    bounds = matrices["upper_bound"]
    ball_bounds = matrices["ball_upper_bound"]
    check("score_shapes", scores.shape == (52, 89) and bounds.shape == ball_bounds.shape == (52, 25))
    check("query_shape", vectors.shape == (52, 384))
    check("aggregate_only_index_fields", set(state) ==
          {"schema_version", "dimension", "normalization", "numerical_margin", "centers",
           "directions", "min_cosine", "radii", "counts"})
    check("aggregate_shapes", state["centers"].shape == state["directions"].shape == (25, 384) and
          all(state[key].shape == (25,) for key in ("min_cosine", "radii", "counts")))
    check("aggregate_counts", state["counts"].tolist() == list(map(len, groups)))
    close("query_unit_norms", np.linalg.norm(vectors, axis=1), np.ones(52), 1e-12)
    q = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    center_norm = np.linalg.norm(state["centers"], axis=1)
    close("index_direction_norms", np.linalg.norm(state["directions"], axis=1), np.ones(25))
    close("index_direction_matches_center", state["directions"],
          state["centers"] / center_norm[:, None])
    close("radius_cap_identity", state["radii"]**2,
          1 + center_norm**2 - 2 * center_norm * state["min_cosine"])
    old_scores = np.load(SOURCE / "score_matrices.npz", allow_pickle=False)["bge_leaf"]
    close("source_score_matrix_reproduction", scores, old_scores, 1e-12)
    actual_max = np.column_stack([scores[:, group].max(axis=1) for group in groups])
    actual_mean = np.column_stack([scores[:, group].mean(axis=1) for group in groups])
    close("actual_max_recompute", matrices["actual_max"], actual_max)
    close("actual_mean_recompute", matrices["mean_score"], actual_mean)
    close("index_mean_identity", q @ state["centers"].T, actual_mean)
    margin = 64 * np.finfo(np.float64).eps * (384 + 1)
    close("fixed_rounding_margin", state["numerical_margin"], margin, 0.)
    # Independent geometric expression: cosine of the angle separating the
    # query from the containing cap, using the saved (non-query-fitted) cap.
    query_angles = np.arccos(np.clip(q @ state["directions"].T, -1, 1))
    cap_angles = np.arccos(np.clip(state["min_cosine"], -1, 1))
    geometric_cap = np.cos(np.maximum(0., query_angles - cap_angles[None, :]))
    singletons = state["counts"] == 1
    geometric_cap[:, singletons] = (q @ state["directions"].T)[:, singletons]
    close("cap_bound_geometric_recomputation", bounds, geometric_cap, 5e-9)
    theoretical_ball = np.minimum(1., q @ state["centers"].T + state["radii"])
    close("ball_bound_geometric_recomputation", ball_bounds, theoretical_ball, 5e-10)
    # Allow numerical noise only when comparing two independently calculated
    # formulas. Saved upper bounds must contain actual scores with no tolerance.
    check("all1300_spherical_bounds_contain_actual", bool(np.all(bounds >= actual_max)))
    check("all1300_ball_bounds_contain_actual", bool(np.all(ball_bounds >= actual_max)))
    check("padded_cap_not_below_geometric_cap", np.min(bounds - geometric_cap) >= -1e-14)
    check("scores_within_cosine_range", np.max(np.abs(scores)) <= 1 + 1e-12)

    # Extract only the controller into an isolated namespace. It cannot access
    # source files, query/gold arrays, the oracle, or the index fitter.
    main_script = ROOT / "scripts/analysis/evaluate_text_similarity_bounds.py"
    module_ast = ast.parse(main_script.read_text())
    function = next(item for item in module_ast.body
                    if isinstance(item, ast.FunctionDef) and item.name == "adaptive_search")
    check("controller_argument_interface",
          [arg.arg for arg in function.args.args] ==
          ["order", "upper_bounds", "member_indices", "score_cluster", "limit"])
    controller_globals = {"np": np, "__builtins__": __builtins__}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(main_script), "exec"),
         controller_globals)
    controller = controller_globals["adaptive_search"]
    fit_calls = [call for call in ast.walk(module_ast) if isinstance(call, ast.Call) and
                 isinstance(call.func, ast.Attribute) and call.func.attr == "fit" and
                 isinstance(call.func.value, ast.Name) and call.func.value.id == "TextBoundIndex"]
    check("fit_arguments_no_query_or_gold", len(fit_calls) == 1 and
          [arg.id for arg in fit_calls[0].args if isinstance(arg, ast.Name)] == ["leaf_vectors", "groups"] and
          not fit_calls[0].keywords)
    main_function = next(item for item in module_ast.body
                         if isinstance(item, ast.FunctionDef) and item.name == "main")
    search_loop = next(item for item in main_function.body if isinstance(item, ast.For) and
                       any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and
                           child.func.id == "adaptive_search" for child in ast.walk(item)))
    score_assignment = next(item for item in main_function.body if isinstance(item, ast.Assign) and
                            any(isinstance(target, ast.Name) and target.id == "scores" for target in item.targets))
    check("oracle_computed_after_all_searches", score_assignment.lineno > search_loop.end_lineno)

    runs = read(OUT / "adaptive_results.json")["runs"]
    run_map = {(run["question_index"], run["L"]): run for run in runs}
    check("all520_unique_runs", len(runs) == len(run_map) == 520 and
          set(run_map) == {(qi, limit) for qi in range(52) for limit in range(1, 11)})
    callback_total = 0
    examined_events = 0
    for qi in range(52):
        order = orders[qi]
        check(f"clip_order_complete/{qi}", sorted(order) == list(range(25)))
        for limit in range(1, 11):
            saved = run_map[(qi, limit)]
            calls = []
            def guarded_score_cluster(mi):
                check(f"callback_only_once/{qi}/{limit}/{mi}", mi not in calls)
                calls.append(mi)
                return [(i, float(scores[qi, i])) for i in groups[mi]]
            replayed = controller(order, bounds[qi], groups, guarded_score_cluster, limit)
            callback_total += len(calls)
            for key in ("selected_indices", "visited_indices", "skipped_indices", "tail_indices",
                        "visited_master_count", "scored_leaf_count", "clip_depth", "stop_reason"):
                check(f"controller_replay_{key}/{qi}/{limit}", replayed[key] == saved[key])
            check(f"callbacks_equal_visited/{qi}/{limit}", calls == saved["visited_indices"])
            close(f"selected_scores/{qi}/{limit}", replayed["selected_scores"], saved["selected_scores"])
            reference = sorted(range(89), key=lambda i: (-scores[qi, i], i))[:limit]
            check(f"exact_stable_topL/{qi}/{limit}",
                  saved["selected_indices"] == reference and saved["exact_match"] is True)
            check(f"trace_event_count/{qi}/{limit}", len(saved["trace"]) == len(replayed["trace"]))
            # Independently check each action's decision using only previously
            # visited scores, without calling the production controller.
            observed = {}
            independent_visited = []
            for pos, event in enumerate(saved["trace"]):
                mi = order[pos]
                remaining = order[pos:]
                ranked = sorted(observed, key=lambda i: (-observed[i], i))[:limit]
                cutoff = observed[ranked[-1]] if len(ranked) == limit else None
                remaining_upper = max(bounds[qi, rest] for rest in remaining)
                check(f"trace_position/{qi}/{limit}/{pos}", event["position"] == pos + 1 and
                      event["master_index"] == mi and event["remaining_count"] == len(remaining))
                close(f"trace_bound/{qi}/{limit}/{pos}", event["upper_bound"], bounds[qi, mi])
                close(f"remaining_bound/{qi}/{limit}/{pos}", event["remaining_max_before"], remaining_upper)
                if cutoff is None:
                    check(f"empty_cutoff/{qi}/{limit}/{pos}", event["threshold_before"] is None)
                else:
                    close(f"current_cutoff/{qi}/{limit}/{pos}", event["threshold_before"], cutoff)
                should_stop = cutoff is not None and remaining_upper < cutoff
                should_skip = cutoff is not None and bounds[qi, mi] < cutoff
                expected_action = "stop" if should_stop else ("skip" if should_skip else "visit")
                check(f"independent_action/{qi}/{limit}/{pos}", event["action"] == expected_action)
                if expected_action == "visit":
                    independent_visited.append(mi)
                    observed.update({i: scores[qi, i] for i in groups[mi]})
                    check(f"visit_score_count/{qi}/{limit}/{pos}", event["n_scores"] == len(groups[mi]))
                else:
                    check(f"no_unvisited_leaf_reads/{qi}/{limit}/{pos}", event["n_scores"] == 0)
                ranked_after = sorted(observed, key=lambda i: (-observed[i], i))[:limit]
                after = observed[ranked_after[-1]] if len(ranked_after) == limit else None
                if after is None:
                    check(f"after_empty_cutoff/{qi}/{limit}/{pos}", event["threshold_after"] is None)
                else:
                    close(f"after_cutoff/{qi}/{limit}/{pos}", event["threshold_after"], after)
                examined_events += 1
                if expected_action == "stop":
                    check(f"stop_is_final_event/{qi}/{limit}/{pos}", pos == len(saved["trace"]) - 1)
                    check(f"tail_is_exact_remaining_order/{qi}/{limit}/{pos}", saved["tail_indices"] == remaining)
                    break
            check(f"independent_visited_order/{qi}/{limit}", independent_visited == saved["visited_indices"])
            check(f"scored_leaf_count/{qi}/{limit}", len(observed) == saved["scored_leaf_count"])
            check(f"unique_cluster_accounting/{qi}/{limit}", sorted(saved["visited_indices"] +
                  saved["skipped_indices"] + saved["tail_indices"]) == list(range(25)))
            gold = set(questions[qi]["gold_ids"])
            expected_gold = sum(leaves[i]["memory_id"] in gold for i in reference)
            check(f"gold_after_retrieval/{qi}/{limit}", expected_gold == saved["gold_hits"])
            check(f"selected_leaf_ids/{qi}/{limit}", [leaf["image_id"] for leaf in saved["selected_leaves"]] ==
                  [leaves[i]["image_id"] for i in reference])
            check(f"runtime_positive/{qi}/{limit}", saved["search_seconds"] > 0)

    scope_indices = {
        "all_questions": list(range(52)),
        "new_questions": [qi for qi, question in enumerate(questions) if not question["id"].endswith("/Q10")],
    }
    check("scopes52_and51", len(scope_indices["all_questions"]) == 52 and len(scope_indices["new_questions"]) == 51)
    for scope, indices in scope_indices.items():
        recorded = summary["scopes"][scope]
        check(f"scope_count/{scope}", recorded["n_questions"] == len(indices))
        check(f"scope_gold_denominator/{scope}", recorded["n_gold_occurrences"] ==
              sum(len(questions[qi]["gold_ids"]) for qi in indices))
        slack = bounds[indices] - actual_max[indices]
        check(f"scope_pair_count/{scope}", recorded["n_query_master_pairs"] == slack.size)
        check(f"scope_no_bound_violations/{scope}", recorded["bound_violations"] == np.count_nonzero(slack < 0))
        for key, value in {"slack_mean": slack.mean(), "slack_median": np.median(slack),
                           "slack_p90": np.quantile(slack, .9),
                           "slack_min": slack.min(), "slack_max": slack.max()}.items():
            close(f"{key}/{scope}", recorded[key], value)
        for budget in recorded["budgets"]:
            limit = budget["L"]
            selected = [run_map[(qi, limit)] for qi in indices]
            visits = [run["visited_master_count"] for run in selected]
            leaf_counts = [run["scored_leaf_count"] for run in selected]
            expected = {
                "exact_matches": sum(run["exact_match"] for run in selected),
                "mean_visited_masters": np.mean(visits), "min_visited_masters": min(visits),
                "max_visited_masters": max(visits), "mean_scored_leaves": np.mean(leaf_counts),
                "min_scored_leaves": min(leaf_counts), "max_scored_leaves": max(leaf_counts),
                "mean_total_vector_dots": 25 + np.mean(leaf_counts),
                "mean_clip_depth": np.mean([run["clip_depth"] for run in selected]),
                "stopped_early_count": sum(bool(run["tail_indices"]) for run in selected),
                "mean_skipped_masters": np.mean([len(run["skipped_indices"]) for run in selected]),
                "mean_masters_unvisited_at_stop": np.mean([len(run["tail_indices"]) for run in selected]),
                "mean_search_seconds": np.mean([run["search_seconds"] for run in selected]),
                "gold_hits": sum(run["gold_hits"] for run in selected),
                "any_gold_count": sum(run["gold_hits"] > 0 for run in selected),
                "all_gold_count": sum(run["gold_hits"] == len(questions[run["question_index"]]["gold_ids"]) for run in selected),
            }
            for key, value in expected.items():
                close(f"budget_summary_{key}/{scope}/{limit}", budget[key], value)
    check("index_payload_bytes", summary["index_stats"]["numeric_bytes"] ==
          sum(state[key].nbytes for key in ("centers", "directions", "min_cosine", "radii", "counts")))
    query_audits = [json.loads(line) for line in (OUT / "query_audit.jsonl").read_text().splitlines()]
    check("L5_audits_cover52", len(query_audits) == 52)
    for qi, query_audit in enumerate(query_audits):
        check(f"L5_audit_matches_full_run/{qi}", query_audit["diagnostic_L5"] == run_map[(qi, 5)])
        check(f"L5_audit_CLIP_order/{qi}", [m["master_id"] for m in query_audit["masters"]] ==
              [masters[mi]["id"] for mi in orders[qi]])
    files = [OUT / filename for filename in ("protocol.json", "summary.json", "adaptive_results.json",
              "query_audit.jsonl", "text_index.npz", "query_vectors.npy",
              "score_audit_matrices.npz", "index_metadata.json")]
    output = {
        "passed": True, "check_count": len(checks), "checks": checks,
        "query_master_pairs": 1300, "adaptive_runs": 520,
        "trace_events_examined": examined_events, "guarded_callback_invocations": callback_total,
        "all_exact_full_scan_topL_matches": True,
        "max_absolute_discrepancies": maxima,
        "independent_method": "Geometry recomputed from saved query/index arrays; isolated controller with guarded callbacks; separate action-by-action simulation; exact stable full-score oracle",
        "auditor_encoder_calls": 0,
        "limitations": [
            "Containment is measured for all 1300 evaluated pairs; floating-point expansion is not a formal arbitrary-precision certificate",
            "No leaf vectors are saved; fitting provenance and mean/radius identities are checked, but this audit does not independently re-encode the corpus",
            "Saved timings and candidate counts are aggregated independently; no end-to-end speedup is inferred",
            "The guarantee is equality to full BGE Top-L, not answer accuracy or semantic sufficiency",
        ],
        "source_memory_unchanged": all(sha(path) == expected for path, expected in protocol["source_sha256"].items()
                                       if "/memory/" in path),
        "script_sha256": sha(__file__),
        "audited_files_sha256": {str(path.relative_to(OUT)): sha(path) for path in files},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "validation_independent.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks), "trace_events": examined_events,
                      "callback_invocations": callback_total, "seconds": output["elapsed_seconds"],
                      "maxima": maxima}))


if __name__ == "__main__":
    main()
