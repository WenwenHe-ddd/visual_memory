"""Independent arithmetic and trace audit; imports no production search code."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DATA = BASE / "inputs_by_domain/Home_Renovation_Interior_Design"
OUT = BASE / "home/Home_Renovation_Interior_Design"
checks = 0
statistics = {
    "pca_bound_leaf_checks": 0, "pca_bound_violations": 0,
    "pca_pruned_leaf_decisions": 0, "tree_bound_leaf_checks": 0,
    "tree_bound_violations": 0, "cap_bound_leaf_checks": 0,
    "cap_bound_violations": 0,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, description):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(description)


def same(actual, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            check(key in actual, f"{label}.{key} exists")
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, (list, tuple, np.ndarray)):
        check(len(actual) == len(expected), f"{label} length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            same(a, e, f"{label}[{i}]")
    elif isinstance(expected, (float, np.floating)):
        check(abs(actual-expected) <= 2e-11, f"{label}: {actual} != {expected}")
    else:
        check(actual == expected, f"{label}: {actual} != {expected}")


def robust_unit(x):
    x = np.array(x, dtype=float, copy=True)
    x /= np.max(np.abs(x), axis=-1, keepdims=True)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def stable_top(scores, limit, available=None):
    ids = range(len(scores)) if available is None else available
    return sorted(ids, key=lambda i: (-scores[i], i))[:limit]


def fit_tree(x, groups, margin):
    nodes = []
    def build(ids, master):
        ids = sorted(ids)
        block = x[ids]
        pivot = ids[np.argmax(block @ block.mean(axis=0))]
        rest = [i for i in ids if i != pivot]
        values = x[rest] @ x[pivot] if rest else np.array([1.])
        angular_lo = max(0., np.nextafter(math.acos(float(np.clip(values.max()+margin, -1, 1))), -np.inf)-margin)
        angular_hi = min(math.pi, np.nextafter(math.acos(float(np.clip(values.min()-margin, -1, 1))), np.inf)+margin)
        ni = len(nodes)
        nodes.append(None)
        children = []
        if len(rest) == 1:
            children = [build(rest, master)]
        elif rest:
            cos = x[rest] @ x[rest].T
            i, j = min(((i, j) for i in range(len(rest)) for j in range(i+1, len(rest))),
                       key=lambda ij: (cos[ij], rest[ij[0]], rest[ij[1]]))
            proj = x[rest] @ (x[rest[i]]-x[rest[j]])
            ordered = [rest[k] for k in sorted(range(len(rest)), key=lambda k: (proj[k], rest[k]))]
            mid = len(ordered)//2
            children = [build(ordered[:mid], master), build(ordered[mid:], master)]
        nodes[ni] = dict(pivot=pivot, members=ids, rest=rest, master=master, children=children,
                         angular_lo=angular_lo, angular_hi=angular_hi)
        return ni
    roots = [build(group, mi) for mi, group in enumerate(groups)]
    check(len(nodes) == len(x), "one tree node per leaf")
    check(sorted(n["pivot"] for n in nodes) == list(range(len(x))), "unique tree pivots")
    for node in nodes:
        check(sorted(i for child in node["children"] for i in nodes[child]["members"]) == sorted(node["rest"]),
              "child members exactly partition non-pivot descendants")
    return nodes, roots


def audit_pca(row, q, x, groups, order, pca):
    mean, basis, coords, residual_norm, coordinate_norm, orth, margin = pca
    n, dim = x.shape
    limit = row["L"]
    # Independent per-leaf scoring is used only by this audit.
    scores = np.einsum("ij,j->i", x, q, optimize=False)
    active = set(range(n))
    exact = []
    trace = []
    seeds = []
    cutoff = lambda: None if len(exact) < limit else float(scores[stable_top(scores, limit, exact)[-1]])
    for master in order:
        before = cutoff()
        members = groups[master]
        check(not (set(exact) & set(members)), "PCA seed no repeated scores")
        exact.extend(members)
        active.difference_update(members)
        seeds.append(master)
        trace.append(dict(action="seed_master", master_index=master, scored_indices=members,
                          threshold_before=before, threshold_after=cutoff(), remaining_active=len(active)))
        if len(exact) >= limit:
            break
    mean_cost = dim if active else 0
    mean_score = float(mean @ q) if active else 0
    partial = np.zeros(n)
    sq, components, partial_cost = 0., 0, 0
    reason = "seed_scored_all_leaves"
    while active:
        ids = np.array(sorted(active))
        next_cost, exact_cost = dim+len(ids), dim*len(ids)
        before = cutoff()
        if components == len(basis) or next_cost >= exact_cost:
            reason = "all_components_used" if components == len(basis) else "exact_remaining_is_no_costlier"
            exact.extend(ids.tolist())
            active.clear()
            trace.append(dict(action="exact_remaining", reason=reason, components=components,
                              scored_indices=ids.tolist(), threshold_before=before, threshold_after=cutoff(),
                              next_projection_scalar_products=next_cost, exact_remaining_scalar_products=exact_cost))
            break
        v = float(basis[components] @ q)
        partial[ids] += coords[ids, components]*v
        sq += v*v
        components += 1
        partial_cost += len(ids)
        qradius = math.sqrt(max(0., 1-sq+orth*sq+margin))
        cross = math.sqrt(max(0., sq))*(orth*coordinate_norm[ids, components]+margin)
        width = qradius*residual_norm[ids, components]+cross+margin
        lower = np.maximum(-1-margin, mean_score+partial[ids]-width)
        upper = np.minimum(1+margin, mean_score+partial[ids]+width)
        violations = int(np.count_nonzero((scores[ids] < lower) | (scores[ids] > upper)))
        statistics["pca_bound_leaf_checks"] += len(ids)
        statistics["pca_bound_violations"] += violations
        check(violations == 0, "PCA lower <= exact <= upper at every active leaf")
        refined = []
        for j in sorted(range(len(ids)), key=lambda j: (-lower[j], ids[j])):
            i = int(ids[j])
            if lower[j] <= cutoff():
                break
            check(i not in exact, "PCA exact refinement unique")
            exact.append(i)
            active.remove(i)
            refined.append(i)
        pruned = [int(i) for i, ub in zip(ids, upper) if i in active and ub < cutoff()]
        check(all(scores[i] < cutoff() for i in pruned), "every PCA prune is safe at current cutoff")
        statistics["pca_pruned_leaf_decisions"] += len(pruned)
        active.difference_update(pruned)
        trace.append(dict(action="project_component", components=components, active_before=len(ids),
                          active_after=len(active), threshold_before=before, threshold_after=cutoff(),
                          pruned_indices=pruned, refined_exact_indices=refined,
                          upper_max=float(upper.max()), lower_max=float(lower.max()),
                          query_projected_squared_norm=sq,
                          next_projection_scalar_products=next_cost, exact_remaining_scalar_products=exact_cost))
        reason = "all_remaining_candidates_pruned_or_scored"
    costs = dict(mean_dot_scalar_products=mean_cost, query_projection_scalar_products=components*dim,
                 leaf_partial_scalar_products=partial_cost, exact_leaf_scalar_products=len(exact)*dim,
                 bound_evaluations=partial_cost, query_normalization_scalar_products_common_to_full_scan=dim,
                 total_scalar_products=mean_cost+components*dim+partial_cost+len(exact)*dim,
                 full_scan_leaf_scalar_products=n*dim)
    same(row, dict(selected_indices=stable_top(scores, limit, exact), scored_indices=exact,
                   exact_score_count=len(exact), scored_leaves=len(exact), projected_components=components,
                   scalar_coordinate_products=costs["total_scalar_products"], cost=costs,
                   pruned_leaf_count=n-len(exact), seed_master_indices=seeds, stop_reason=reason), "PCA replay")
    if limit == 5:
        same(row["trace"], trace, "PCA saved L5 trace")


def main():
    inputs, summary, rows = read(DATA/"inputs.json"), read(OUT/"summary.json"), read(OUT/"runs.json")
    feature = np.load(DATA/"features.npz")
    x, queries, orders = feature["bge_leaf"], feature["bge_query"], feature["clip_order"].tolist()
    n, dim = x.shape
    old_dir = ROOT/"reports/home_online_routing_dialogue_caption_20261020"
    old = read(old_dir/"inputs.json")
    old_scores = np.load(old_dir/"score_matrices.npz")
    files = [DATA/"inputs.json", DATA/"features.npz", DATA/"protocol.json", DATA/"input_manifest.json",
             OUT/"runs.json", OUT/"summary.json", OUT/"latency_samples.json", OUT/"evaluation_scores.npz",
             old_dir/"inputs.json", old_dir/"score_matrices.npz"]
    core = [ROOT/p for p in ["scripts/analysis/compare_adaptive_retrieval.py",
                              "scripts/analysis/evaluate_text_similarity_bounds.py",
                              "scripts/graph/text_similarity_bounds.py",
                              "scripts/graph/hierarchical_text_bounds.py",
                              "scripts/graph/progressive_text_bounds.py"]]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files+core}
    for path, expected in summary["source_hashes"].items():
        check(sha(path) == expected, "runtime input hash")
    check((len(queries), n, len(inputs["masters"]), dim) == (52, 89, 25, 384), "fixed Home counts")
    check(sum(len(q["gold_ids"]) for q in inputs["questions"]) == 68, "68 gold occurrences")
    check(not any(q["query_image_paths"] for q in inputs["questions"]), "Home text-only questions")
    check([q["id"] for q in inputs["questions"]] == [q["id"] for q in old["questions"]], "old question ordering")
    for a, b in zip(inputs["questions"], old["questions"]):
        for key in ("id", "question", "gold_ids"):
            same(a[key], b[key], "old question "+key)
        check(a["retrieval_query"] == a["question"], "exact question query")
    same(inputs["masters"], old["masters"], "frozen masters")
    for a, b in zip(inputs["leaves"], old["leaves"]):
        same(a, {k: b[k] for k in a}, "frozen leaves")
        check(a["retrieval_text"] == a["dialogue"]+"\nMemory description: "+a["caption"], "dialogue/full caption input")
    leaf_ids = {leaf["image_id"]: i for i, leaf in enumerate(inputs["leaves"])}
    groups = [[leaf_ids[i] for i in m["member_ids"]] for m in inputs["masters"]]
    check(sorted(i for group in groups for i in group) == list(range(n)), "membership partitions89")
    check(np.array_equal(feature["clip_master"], old_scores["clip_master"]), "CLIP scores exactly reused")
    same(orders, np.argsort(-feature["clip_master"], axis=1, kind="stable"), "all CLIP orders")
    oracle = np.einsum("qd,nd->qn", queries, x, optimize=False)
    check(np.max(np.abs(oracle-old_scores["bge_leaf"])) < 2e-14, "old BGE score consistency")
    check(np.max(np.abs(oracle-np.load(OUT/"evaluation_scores.npz")["bge"])) < 2e-14, "saved full oracle")
    manifest = read(DATA/"input_manifest.json")
    for name, expected in manifest["memory_frozen_sha256"].items():
        check(sha(ROOT/"reports/home_caption_ward_visual_20261020/memory"/name) == expected, "frozen visual memory hash")
    methods = sorted(summary["methods"])
    check(len(rows) == 52*4*6, "all1248runs")
    check(len({(r["question_index"],r["L"],r["method"]) for r in rows}) == len(rows), "no duplicate runs")
    tree_x = robust_unit(x)
    eps = 64*np.finfo(float).eps*(dim+1)
    nodes, roots = fit_tree(tree_x, groups, eps)
    centers = np.array([tree_x[g].mean(axis=0) for g in groups])
    directions = centers/np.linalg.norm(centers,axis=1,keepdims=True)
    minima = np.array([np.min(tree_x[g]@u) for g,u in zip(groups,directions)])
    # Independently refit memory-only PCA; no production class is imported.
    px = x/np.linalg.norm(x,axis=1,keepdims=True)
    mean = px.mean(axis=0)
    centered = px-mean
    _, singular, right = np.linalg.svd(centered,full_matrices=False)
    rank = int(np.count_nonzero(singular > max(centered.shape)*np.finfo(float).eps*singular[0]))
    basis = right[:rank].copy()
    for direction in basis:
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
    coords = centered@basis.T
    margin = 64*np.finfo(float).eps*(dim+rank+1)
    orth = np.max(np.abs(basis@basis.T-np.eye(rank)).sum(axis=1))+margin
    residual = centered.copy()
    residual_norm = [np.linalg.norm(residual,axis=1)+margin]
    for k in range(rank):
        residual -= coords[:,k,None]*basis[k]
        residual_norm.append(np.linalg.norm(residual,axis=1)+margin)
    residual_norm = np.array(residual_norm).T
    coordinate_norm = np.column_stack([np.zeros(n),np.sqrt(np.cumsum(coords**2,axis=1))])
    pca = mean,basis,coords,residual_norm,coordinate_norm,orth,margin
    cap_cache = {}
    for qi,q in enumerate(queries):
        tq = robust_unit(q)
        b = directions@tq
        rho = np.minimum(np.pi,np.nextafter(np.arccos(np.clip(minima-eps,-1,1)),np.inf)+eps)
        theta = np.maximum(0.,np.nextafter(np.arccos(np.clip(b+eps,-1,1)),-np.inf)-eps)
        upper = np.minimum(1.,np.nextafter(np.cos(np.maximum(0.,theta-rho)),np.inf)+eps)
        for mi,g in enumerate(groups):
            if len(g)==1: upper[mi]=min(1.,np.nextafter(b[mi],np.inf)+eps)
            statistics["cap_bound_leaf_checks"] += len(g)
            statistics["cap_bound_violations"] += int(np.count_nonzero(oracle[qi,g]>upper[mi]))
        cap_cache[qi]=upper
    check(statistics["cap_bound_violations"]==0,"every cap upper bounds all cluster members")
    for row in rows:
        qi,limit,method=row["question_index"],row["L"],row["method"]
        expected=stable_top(oracle[qi],limit)
        same(row["selected_indices"],expected,"full oracle ordered TopL")
        same(row["selected_scores"],oracle[qi,expected],"selected scores")
        check(row["exact_match"] and row["topL_overlap"]==1.,"exact flags")
        same(row["reference_indices"],expected,"saved oracle indices")
        selected_ids=[inputs["leaves"][i]["memory_id"] for i in expected]
        same(row["selected_memory_ids"],selected_ids,"memory mapping")
        gold=set(inputs["questions"][qi]["gold_ids"])
        same(row["gold_hits"],len(set(selected_ids)&gold),"gold hits")
        same(row["gold_total"],len(gold),"gold denominator")
        if method=="full_scan":
            same(row,dict(scored_leaves=n,visited_masters=len(groups),scalar_coordinate_products=n*dim),"full cost")
        elif method.startswith("pivot_tree"):
            scored=row["scored_leaf_indices"]
            check(len(scored)==len(set(scored))==row["scored_leaves"],"tree no duplicate scores")
            check(sorted(scored)==list(range(n)),"Home tree scanned every leaf")
            same(row["bound_dot_count"]+row["leaf_dot_count"],len(scored),"mutually exclusive tree dots")
            same(row["scalar_coordinate_products"],len(scored)*dim,"tree products")
            check(row["additional_nonleaf_dot_count"]==0,"pivot reused no extra full dot")
            if limit==5:
                trace_scored=[]
                bcount=lcount=0
                for event in row["trace"]:
                    if event["action"]=="score":
                        node=nodes[event["node_id"]]
                        i=event["leaf_index"]
                        check(i==node["pivot"] and i not in trace_scored,"tree trace unique pivot")
                        same(event["score"],oracle[qi,i],"tree score oracle")
                        trace_scored.append(i)
                        bcount+=int(bool(node["children"]));lcount+=int(not node["children"])
                    elif event["action"]=="bound":
                        node=nodes[event["node_id"]]
                        check(node["pivot"] in trace_scored,"bound reuses scored pivot")
                        rem=node["rest"]
                        statistics["tree_bound_leaf_checks"]+=len(rem)
                        statistics["tree_bound_violations"]+=int(np.count_nonzero(oracle[qi,rem]>event["upper_bound"]))
                        check(np.all(oracle[qi,rem]<=event["upper_bound"]),"tree bound valid for descendants")
                        same(event["additional_coordinate_products"],0,"tree bound extra Dcost0")
                    elif event["action"] in ("prune","stop_frontier"):
                        check(all(oracle[qi,i]<event["threshold"] for i in event["unscored_members"]),"tree prune safe")
                same(trace_scored,scored,"tree trace scored order")
                same([bcount,lcount],[row["bound_dot_count"],row["leaf_dot_count"]],"tree trace dot counters")
        elif method.startswith("spherical_cap"):
            upper=cap_cache[qi]
            order=orders[qi]
            if method.endswith("best_bound"):
                positions={mi:k for k,mi in enumerate(order)}
                order=sorted(order,key=lambda mi:(-upper[mi],positions[mi]))
            visited,skipped,tail,scored=[],[],[],[]
            trace=[]
            for pos,mi in enumerate(order):
                threshold=None if len(scored)<limit else float(oracle[qi,stable_top(oracle[qi],limit,scored)[-1]])
                maximum=float(upper[order[pos:]].max())
                event=dict(position=pos+1,master_index=mi,upper_bound=float(upper[mi]),
                           threshold_before=threshold,remaining_max_before=maximum,remaining_count=len(order)-pos)
                if threshold is not None and maximum<threshold:
                    tail=order[pos:];event.update(action="stop",threshold_after=threshold,n_scores=0);trace.append(event);break
                if threshold is not None and upper[mi]<threshold:
                    skipped.append(mi);event.update(action="skip",threshold_after=threshold,n_scores=0);trace.append(event);continue
                visited.append(mi);scored.extend(groups[mi])
                after=None if len(scored)<limit else float(oracle[qi,stable_top(oracle[qi],limit,scored)[-1]])
                event.update(action="visit",threshold_after=after,n_scores=len(groups[mi]));trace.append(event)
            same(row,dict(visited_indices=visited,skipped_indices=skipped,tail_indices=tail,scored_leaves=len(scored),
                          visited_masters=len(visited),scalar_coordinate_products=(len(groups)+len(scored))*dim),"cap control/cost")
            if limit==5:same(row["trace"],trace,"saved cap trace")
        elif method=="progressive_text_pca":
            q=queries[qi]/np.linalg.norm(queries[qi])
            audit_pca(row,q,px,groups,orders[qi],pca)
        else:
            raise AssertionError(method)
    # Recompute summary numerators, denominators and all cost means.
    for entry in summary["scopes"]["all"]:
        selected=[r for r in rows if r["method"]==entry["method"] and r["L"]==entry["L"]]
        expected=dict(n_questions=len(selected),exact_matches=len(selected),exact_match_rate=1.,mean_topL_overlap=1.,
                      mean_scored_leaves=float(np.mean([r["scored_leaves"] for r in selected])),
                      mean_visited_masters=float(np.mean([r["visited_masters"] for r in selected])),
                      mean_coordinate_products=float(np.mean([r["scalar_coordinate_products"] for r in selected])),
                      coordinate_product_ratio_to_full=float(np.mean([r["scalar_coordinate_products"]/(n*dim) for r in selected])),
                      gold_hits=sum(r["gold_hits"] for r in selected),gold_total=68,
                      any_gold_queries=sum(r["gold_hits"]>0 for r in selected),
                      all_gold_queries=sum(r["gold_hits"]==r["gold_total"] for r in selected))
        same(entry,expected,"summary")
    same(summary["scopes"]["all"],summary["scopes"]["text_only"],"all Home text-only")
    check(summary["scopes"]["image_assisted"]==[],"empty image-assisted scope")
    check(summary["expansion_gate_passed_methods"]==[],"no strict method passes cost gate")
    timings=read(OUT/"latency_samples.json")
    for method, samples in timings.items():
        check(len(samples)==52*11,"latency sample count")
        check(len({(r["question_index"],r["repeat"]) for r in samples})==52*11,"latency uniqueness")
        times=np.array([r["nanoseconds"] for r in samples])/1e6
        per_query=[float(np.median([r["nanoseconds"] for r in samples if r["question_index"]==qi])/1e6) for qi in range(52)]
        same(summary["latency"][method],dict(repetitions_per_query=11,measurements=572,
             median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
             mean_of_per_query_median_ms=float(np.mean(per_query)),per_query_median_ms=per_query),"latency statistics")
    csvrows=list(csv.DictReader((OUT/"per_query.csv").open()))
    check(len(csvrows)==len(rows),"CSV all runs")
    for a,b in zip(csvrows,rows):
        for key in a:check(a[key]==str(b[key]),"CSV field exact")
    for path,expected in hashes.items():check(sha(ROOT/path)==expected,"audited sources unchanged")
    result=dict(passed=True,checks=checks,scope="52 Home queries x 4 L x 6 exact methods; 1,248 runs",
                independent_implementation="This audit imports no production index or search implementation.",
                dimensions=dict(questions=52,leaves=89,masters=25,text_dimension=384,gold_occurrences=68),
                audit_counts=statistics,pca_rank=rank,
                source_hashes=hashes,
                code_hash_boundary="Core hashes are recorded at audit time; formal summary originally stored only input hashes. Current code was reviewed and independently replayed against saved output.",
                L5_summary=[e for e in summary["scopes"]["all"] if e["L"]==5],
                limitations=[
                    "No encoder, image decoding or reader was rerun; encoded inputs are linked to the previously independently audited Home source.",
                    "Upper/lower containment is exhaustively checked on these saved queries and memories, not claimed as a general floating-point proof.",
                    "Coordinate products exclude scalar bounds, heaps, sort and Python overhead; shared CLIP25x512 routing is outside the BGE denominator.",
                    "Latency values are validated from saved samples and code boundaries, not independently re-timed.",
                    "Numeric index bytes exclude Python metadata; frozen six-method exact results do not establish speedup or unseen-domain effectiveness.",
                ])
    destination=BASE/"validation_home_exact_independent.json"
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(passed=True,checks=checks,audit_counts=statistics,pca_rank=rank)))


if __name__=="__main__":
    main()
