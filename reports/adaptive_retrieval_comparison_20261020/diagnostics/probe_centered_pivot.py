"""Negative, memory-only common-direction prototype; never used by the core.

Run from the repository root with OPENBLAS_NUM_THREADS=1 python3 <this-file>.
The query-by-leaf matrix is computed solely for post-search verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.graph.hierarchical_text_bounds import HierarchicalTextBoundIndex


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report = ROOT / "reports/adaptive_retrieval_comparison_20261020"
    feature_path = report / "home_features.npz"
    source = ROOT / "reports/home_online_routing_dialogue_caption_20261020"
    data = json.loads((source / "inputs.json").read_text())
    features = np.load(feature_path)
    leaf_index = {row["image_id"]: i for i, row in enumerate(data["leaves"])}
    groups = [[leaf_index[i] for i in master["member_ids"]] for master in data["masters"]]
    orders = np.argsort(-np.load(source / "score_matrices.npz")["clip_master"], axis=1, kind="stable")
    base = HierarchicalTextBoundIndex.fit(features["bge_leaf"], groups)
    x = base.vectors
    g = x.mean(axis=0)
    g /= np.linalg.norm(g)
    alpha = x @ g
    residual = x - alpha[:, None] * g
    beta = np.linalg.norm(residual, axis=1)
    unit_residual = np.divide(residual, beta[:, None], out=np.zeros_like(residual),
                              where=beta[:, None] != 0)
    margin = base.numerical_margin
    metadata = {}
    for node in base.nodes:
        rem = [i for i in node.members if i != node.pivot_index]
        if rem:
            cos = unit_residual[rem] @ unit_residual[node.pivot_index]
            metadata[node.node_id] = (
                float(alpha[rem].min()), float(alpha[rem].max()),
                float(cos.min()), float(cos.max()), float(beta[rem].min()))
    context = {}

    class CenteredPrototype(HierarchicalTextBoundIndex):
        def _cap_upper(self, node, pivot_score):
            old = super()._cap_upper(node, pivot_score)
            if not node.children:
                return old
            b, gamma = context["b"], context["gamma"]
            alo, ahi, tlo, thi, beta_min = metadata[node.node_id]
            bp = beta[node.pivot_index]
            if min(gamma, bp, beta_min) < math.sqrt(margin):
                return old
            residual_cosine = (pivot_score - b * alpha[node.pivot_index]) / (gamma * bp)
            error = 8 * margin / (gamma * bp)
            theta_lo = max(0., math.acos(float(np.clip(residual_cosine + error, -1, 1))) - error)
            theta_hi = min(math.pi, math.acos(float(np.clip(residual_cosine - error, -1, 1))) + error)
            phi_lo = max(0., math.acos(float(np.clip(thi + 8*margin, -1, 1))) - 8*margin)
            phi_hi = min(math.pi, math.acos(float(np.clip(tlo - 8*margin, -1, 1))) + 8*margin)
            residual_upper = min(1., math.cos(max(0., phi_lo-theta_hi, theta_lo-phi_hi)) + 8*margin)
            c = gamma * residual_upper
            lo, hi = max(-1., alo-8*margin), min(1., ahi+8*margin)
            candidates = [lo, hi]
            if c > 0:
                candidates.append(float(np.clip(b / math.hypot(b, c), lo, hi)))
            value = max(b*t+c*math.sqrt(max(0., 1-t*t)) for t in candidates) + 64*margin
            return min(old, value)

    index = CenteredPrototype(base.vectors, base.groups, base.nodes, base.roots)
    rows = []
    bound_checks = violations = 0
    minimum_slack = math.inf
    for policy in ("clip", "best_bound"):
        for limit in (1, 3, 5, 10):
            results = []
            for qi, raw_q in enumerate(features["bge_query"]):
                q = raw_q / np.linalg.norm(raw_q)
                context["b"] = float(q @ g)
                context["gamma"] = math.sqrt(max(0., 1-context["b"]**2))
                result = index.search(q, orders[qi], limit, policy=policy, record_trace=False)
                # Exhaustive scores below cannot affect the search.
                oracle = np.array([v @ q for v in x])
                expected = np.argsort(-oracle, kind="stable")[:limit].tolist()
                products = result["scalar_coordinate_products"] + x.shape[1]
                results.append({
                    "question_index": qi, "question_id": data["questions"][qi]["id"],
                    "exact_match": result["selected_indices"] == expected,
                    "selected_indices": result["selected_indices"],
                    "scored_leaves": result["scored_leaf_count"],
                    "coordinate_products": products,
                    "shared_direction_dot_products": x.shape[1],
                    "pivot_or_leaf_dot_products": result["scalar_coordinate_products"],
                    "bound_evaluations": result["bound_evaluations"],
                })
                if policy == "clip" and limit == 1:
                    for node in base.nodes:
                        rem = [i for i in node.members if i != node.pivot_index]
                        if not rem:
                            continue
                        bound = index._cap_upper(node, float(oracle[node.pivot_index]))
                        slack = float(bound - oracle[rem].max())
                        bound_checks += len(rem)
                        violations += int(np.count_nonzero(oracle[rem] > bound))
                        minimum_slack = min(minimum_slack, slack)
            rows.append({
                "policy": policy, "L": limit, "questions": len(results),
                "exact_matches": sum(row["exact_match"] for row in results),
                "mean_scored_leaves": float(np.mean([r["scored_leaves"] for r in results])),
                "mean_coordinate_products": float(np.mean([r["coordinate_products"] for r in results])),
                "coordinate_product_ratio_to_full": float(np.mean([r["coordinate_products"] for r in results]) / x.size),
                "per_question": results,
            })
    output = {
        "variant": "common_memory_direction_removed_pivot_band_prototype",
        "domain": "Home_Renovation_Interior_Design",
        "sourcecode": str(Path(__file__).relative_to(ROOT)),
        "sourcecode_sha256": sha(__file__),
        "frozen_base_core_sha256": sha(ROOT / "scripts/graph/hierarchical_text_bounds.py"),
        "feature_path": str(feature_path.relative_to(ROOT)),
        "feature_sha256": sha(feature_path),
        "input_sha256": sha(source / "inputs.json"),
        "dimensions": int(x.shape[1]), "leaves": len(x), "masters": len(groups),
        "fit_inputs": "Only normalized memory text embeddings and existing master membership.",
        "formula": {
            "memory": "g=unit(mean x); alpha_i=g·x_i; beta_i=||x_i-alpha_i*g||; w_i=(x_i-alpha_i*g)/beta_i",
            "query": "b=q·g; gamma=sqrt(1-b²); q_res·w_p=(q·x_p-b*alpha_p)/(gamma*beta_p)",
            "residual_bound": "Angular band of w_i around w_p bounds query residual alignment by U_res.",
            "full_bound": "max_{t in [alpha_min,alpha_max]} b*t + gamma*U_res*sqrt(1-t²), intersected with original pivot-band bound.",
            "maximum": "Check both interval endpoints; if gamma*U_res>0 also check clipped stationary t=b/hypot(b,gamma*U_res).",
            "numeric_allowance": "Fixed float64 error guards from 64*eps*(D+1), enlarged for residual division; near-zero denominator falls back to original bound. No learned or QA-tuned cutoff.",
        },
        "cost": "One additional D-coordinate shared-direction dot per query, plus every unique pivot/leaf dot. Scalar arithmetic and Python overhead are additional unmeasured work; no runtime improvement is claimed.",
        "oracle_validation": {
            "bound_leaf_checks": bound_checks, "violations": violations,
            "minimum_upper_minus_actual_max": minimum_slack,
            "alpha_range": [float(alpha.min()), float(alpha.max())],
        },
        "results": rows,
        "conclusion": "Negative result: all 89 leaves are scored at every tested L and policy. Exact results match, but vector-coordinate work rises by 1/89 due to the shared-direction query dot. Not integrated into the six formal methods.",
        "limitations": "Finite-sample bound checks are not a formal floating-point proof of this exploratory prototype. The frozen production pivot bound remains unchanged.",
    }
    destination = report / "pilot_centered_pivot_result.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("oracle_validation", "conclusion")}))


if __name__ == "__main__":
    main()
