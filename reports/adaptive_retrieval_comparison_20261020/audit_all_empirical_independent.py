"""Independently replay every empirical search, with post hoc miss attribution.

No production search/fit function is imported, no model is loaded, and no image
is read. Full-oracle scores are kept outside the replay controller.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import audit_home_exact_independent as H

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def replay(x, centers, radii, groups, q, order, limit, prediction):
    """Controller reads exact values exclusively from each visited group."""
    q = H.robust_unit(q)
    means = centers @ q
    scores = {}
    visited, skipped, tail, trace = [], [], [], []
    offset, ratio = 0., 0.
    selected = []
    exclusion = {}
    for pos, master in enumerate(order):
        cutoff = scores[selected[-1]] if len(selected) == limit else None
        raw, scaled = means + offset, means + ratio*radii
        estimates = raw if prediction == "offset" else scaled
        if prediction == "max_both":
            estimates = np.maximum(raw, scaled)
        remaining = order[pos:]
        largest = float(np.max(estimates[remaining]))
        event = dict(position=pos,master_index=master,cutoff_before=cutoff,
                     observed_offset=offset,observed_radius_ratio=ratio,
                     predicted_max=float(estimates[master]),max_remaining_predicted=largest,
                     is_certified_bound=False)
        if cutoff is not None and largest < cutoff:
            tail = remaining
            trace.append({**event,"action":"empirical_stop"})
            for mi in remaining:
                exclusion[mi] = {**event,"action":"empirical_stop","excluded_master":mi,
                                  "excluded_master_prediction":float(estimates[mi])}
            break
        if cutoff is not None and estimates[master] < cutoff:
            skipped.append(master)
            trace.append({**event,"action":"empirical_skip"})
            exclusion[master] = {**event,"action":"empirical_skip","excluded_master":master,
                                  "excluded_master_prediction":float(estimates[master])}
            continue
        members = groups[master]
        H.check(not any(i in scores for i in members),"empirical no repeated leaf score")
        values = x[members] @ q
        scores.update((i,float(v)) for i,v in zip(members,values))
        visited.append(master)
        deviation = float(values.max()-means[master])
        offset = max(offset,deviation)
        if radii[master] > 0:
            ratio = max(ratio,deviation/radii[master])
        selected = sorted(scores,key=lambda i:(-scores[i],i))[:limit]
        trace.append({**event,"action":"visit","leaf_indices":members,"scores":values.tolist(),
                      "observed_max":float(values.max()),
                      "cutoff_after":scores[selected[-1]] if len(selected)==limit else None})
    H.check(len(selected)==limit,"empirical output count")
    H.check(sorted(visited+skipped+tail)==list(range(len(groups))),"every master has exactly one disposition")
    result = dict(selected_indices=selected,selected_scores=[scores[i] for i in selected],
                  scored_leaves=len(scores),scored_leaf_indices=list(scores),
                  visited_masters=len(visited),visited_master_indices=visited,
                  skipped_master_indices=skipped,tail_master_indices=tail,
                  bound_dot_count=len(groups),exact_leaf_dot_count=len(scores),
                  scalar_coordinate_products=(len(groups)+len(scores))*x.shape[1],
                  prediction=prediction,exact_guarantee=False)
    return result,trace,exclusion


def summary(rows, inputs, n, dim, question_indices):
    results=[]
    for method,limit in sorted({(r["method"],r["L"]) for r in rows}):
        chosen=[r for r in rows if r["question_index"] in question_indices
                and r["method"]==method and r["L"]==limit]
        if not chosen:
            continue
        results.append(dict(method=method,L=limit,n_questions=len(chosen),
            exact_matches=sum(r["exact_match"] for r in chosen),
            exact_match_rate=float(np.mean([r["exact_match"] for r in chosen])),
            mean_topL_overlap=float(np.mean([r["topL_overlap"] for r in chosen])),
            mean_scored_leaves=float(np.mean([r["scored_leaves"] for r in chosen])),
            mean_visited_masters=float(np.mean([r["visited_masters"] for r in chosen])),
            mean_coordinate_products=float(np.mean([r["scalar_coordinate_products"] for r in chosen])),
            coordinate_product_ratio_to_full=float(np.mean([r["scalar_coordinate_products"]/(n*dim) for r in chosen])),
            gold_hits=sum(r["gold_hits"] for r in chosen),gold_total=sum(r["gold_total"] for r in chosen),
            any_gold_queries=sum(r["gold_hits"]>0 for r in chosen),
            all_gold_queries=sum(r["gold_hits"]==r["gold_total"] for r in chosen)))
    return results


def main():
    source = BASE/"all_empirical"
    domain_paths = sorted(p for p in source.iterdir() if p.is_dir())
    H.check(len(domain_paths)==8,"eight domains")
    all_runs,changed,hashes,domain_stats=[],[],{},{}
    trace_counts = dict(visit_events=0,empirical_skip_events=0,empirical_stop_events=0,
                        prediction_underestimation_at_exclusion=0)
    all_ids,all_gold,all_images=[],0,0
    truncated_inputs=[]
    for folder in domain_paths:
        name=folder.name
        data=BASE/"inputs_by_domain"/name
        inputs=H.read(data/"inputs.json")
        features=np.load(data/"features.npz")
        saved=H.read(folder/"summary.json")
        rows=H.read(folder/"runs.json")
        for p in [data/"inputs.json",data/"features.npz",data/"protocol.json",
                  folder/"summary.json",folder/"runs.json",folder/"evaluation_scores.npz",folder/"latency_samples.json"]:
            hashes[str(p.relative_to(ROOT))]=H.sha(p)
        for path,expected in saved["source_hashes"].items():
            H.check(H.sha(path)==expected,"empirical source hash")
        x0,queries,orders=features["bge_leaf"],features["bge_query"],features["clip_order"].tolist()
        nq,n,dim=len(queries),len(x0),x0.shape[1]
        H.check(dim==384,"BGE dimension384")
        H.check(np.max(np.abs(np.linalg.norm(x0,axis=1)-1))<1e-12,"normalized leaves")
        H.check(np.max(np.abs(np.linalg.norm(queries,axis=1)-1))<1e-12,"normalized queries")
        leaves,masters=inputs["leaves"],inputs["masters"]
        index={row["image_id"]:i for i,row in enumerate(leaves)}
        groups=[[index[i] for i in master["member_ids"]] for master in masters]
        owners={i:mi for mi,g in enumerate(groups) for i in g}
        H.check(sorted(owners)==list(range(n)) and sum(map(len,groups))==n,"exact leaf membership")
        H.check(len({leaf["memory_id"] for leaf in leaves})==n,"one image leaf per memory in this experiment")
        H.same(orders,np.argsort(-features["clip_master"],axis=1,kind="stable"),"CLIP initial order")
        for leaf in leaves:
            H.check(leaf["retrieval_text"]==leaf["dialogue"]+"\nMemory description: "+leaf["caption"],"full original dialogue+caption")
        protocol=H.read(data/"protocol.json")
        H.check(protocol["query_options_or_gold_encoded"] is False,"no QA answer/options encoded")
        H.check(protocol["qa_used_for_clustering_or_calibration"] is False,"memory-only fit")
        truncated_inputs.extend({"domain":name,**entry}
                                for entry in protocol["bge_encoding"]["truncated_inputs"])
        all_ids.extend(q["id"] for q in inputs["questions"])
        all_gold+=sum(len(set(q["gold_ids"])) for q in inputs["questions"])
        all_images+=sum(bool(q["query_image_paths"]) for q in inputs["questions"])
        # Derive aggregates only from memories.
        x=H.robust_unit(x0)
        centers=np.array([x[g].mean(axis=0) for g in groups])
        radii=np.array([np.linalg.norm(x[g]-c,axis=1).max() for g,c in zip(groups,centers)])
        oracle=np.einsum("qd,nd->qn",queries,x0,optimize=False)
        H.check(np.max(np.abs(oracle-np.load(folder/"evaluation_scores.npz")["bge"]))<2e-14,"independent full oracle")
        H.check(len(rows)==nq*4*4,"all question/L/method runs")
        H.check(len({(r["question_index"],r["L"],r["method"]) for r in rows})==len(rows),"unique runs")
        for row in rows:
            qi,limit,method=row["question_index"],row["L"],row["method"]
            H.check(row["id"]==inputs["questions"][qi]["id"],"question mapping")
            reference=H.stable_top(oracle[qi],limit)
            selected=row["selected_indices"]
            H.check(len(selected)==len(set(selected))==limit,"unique output count")
            H.same(row["reference_indices"],reference,"oracle TopL")
            H.same(row["selected_scores"],oracle[qi,selected],"all returned exact scores")
            H.same(row["exact_match"],selected==reference,"ordered exact flag")
            H.same(row["topL_overlap"],len(set(selected)&set(reference))/limit,"overlap")
            selected_ids=[leaves[i]["memory_id"] for i in selected]
            H.same(row["selected_memory_ids"],selected_ids,"leaf evidence mapping")
            gold=set(inputs["questions"][qi]["gold_ids"])
            H.same(row["gold_hits"],len(gold&set(selected_ids)),"gold occurrence hits")
            H.same(row["gold_total"],len(gold),"gold occurrence denominator")
            if method=="full_scan":
                H.same(selected,reference,"full baseline")
                H.same(row,dict(scored_leaves=n,visited_masters=len(groups),
                                scalar_coordinate_products=n*dim),"full baseline cost")
                trace,exclusion=[],{}
            else:
                prediction=method.removeprefix("empirical_")
                replayed,trace,exclusion=replay(x,centers,radii,groups,queries[qi],orders[qi],limit,prediction)
                H.same(row,replayed,"independent empirical control and cost")
                H.check(set(selected).issubset(row["scored_leaf_indices"]),"outputs only from visited leaves")
                H.check(set(row["scored_leaf_indices"])=={i for mi in row["visited_master_indices"] for i in groups[mi]},
                        "exact scores exactly equal visited member union")
                if limit==5:
                    H.same(row["trace"],trace,"every saved L5 trace event")
                    for event in trace:
                        trace_counts[event["action"]+"_events"]+=1
                    for mi,event in exclusion.items():
                        if oracle[qi,groups[mi]].max()>event["excluded_master_prediction"]:
                            trace_counts["prediction_underestimation_at_exclusion"]+=1
                else:
                    H.check(row["trace"]==[],"traces only L5")
            row["audit_n_leaves"],row["audit_n_masters"],row["audit_dimension"]=n,len(groups),dim
            row["audit_domain"]=name
            all_runs.append(row)
            if method=="empirical_max_both" and limit==5 and selected!=reference:
                missed=sorted(set(reference)-set(selected),key=reference.index)
                gained=sorted(set(selected)-set(reference),key=selected.index)
                def description(i):
                    return dict(leaf_index=i,image_id=leaves[i]["image_id"],memory_id=leaves[i]["memory_id"],
                                filename=leaves[i]["filename"],master_index=owners[i],master_id=masters[owners[i]]["id"],
                                score=float(oracle[qi,i]),full_rank=H.stable_top(oracle[qi],n).index(i)+1,
                                is_gold=leaves[i]["memory_id"] in gold)
                missed_rows=[]
                for i in missed:
                    H.check(i not in row["scored_leaf_indices"],"every missed full Top5 leaf unvisited")
                    event=exclusion[owners[i]]
                    H.check(event["excluded_master_prediction"]<event["cutoff_before"],"first exclusion predicted below cutoff")
                    item=description(i)
                    item["first_exclusion"]=event
                    item["actual_cluster_max"]=float(oracle[qi,groups[owners[i]]].max())
                    item["actual_leaf_minus_prediction"]=float(oracle[qi,i]-event["excluded_master_prediction"])
                    missed_rows.append(item)
                changed.append(dict(id=row["id"],domain=name,question=inputs["questions"][qi]["question"],
                                    full_top5=[description(i) for i in reference],
                                    adaptive_top5=[description(i) for i in selected],
                                    missed=missed_rows,gained=[description(i) for i in gained],
                                    gold_ids=sorted(gold),
                                    full_gold_hits=len(gold&{leaves[i]["memory_id"] for i in reference}),
                                    adaptive_gold_hits=row["gold_hits"]))
        scopes=dict(all=set(range(nq)),
                    text_only={i for i,q in enumerate(inputs["questions"]) if not q["query_image_paths"]},
                    image_assisted={i for i,q in enumerate(inputs["questions"]) if q["query_image_paths"]})
        for scope,ids in scopes.items():
            H.same(saved["scopes"][scope],summary(rows,inputs,n,dim,ids),"domain summary "+scope)
        timing=H.read(folder/"latency_samples.json")
        for method,samples in timing.items():
            reps=saved["latency"][method]["repetitions_per_query"]
            H.check(len(samples)==nq*reps,"timing count")
            H.check(len({(r["question_index"],r["repeat"]) for r in samples})==nq*reps,"unique timings")
            values=np.array([r["nanoseconds"] for r in samples])/1e6
            H.same(saved["latency"][method]["median_ms"],float(np.median(values)),"latency median")
            H.same(saved["latency"][method]["p95_ms"],float(np.quantile(values,.95)),"latency p95")
        csvrows=list(csv.DictReader((folder/"per_query.csv").open()))
        H.check(len(csvrows)==len(rows),"all CSV rows")
        for a,b in zip(csvrows,rows):
            for key in a:H.check(a[key]==str(b[key]),"CSV values")
        domain_stats[name]=dict(questions=nq,leaves=n,masters=len(groups),image_assisted=len(scopes["image_assisted"]),
                               gold_occurrences=sum(len(set(q["gold_ids"])) for q in inputs["questions"]))
    H.check(len(all_ids)==len(set(all_ids))==371,"371 unique QA")
    H.check(all_gold==1041,"1041 gold occurrences")
    H.check(all_images==8,"8 image-assisted QA")
    H.check(len(all_runs)==371*4*4,"5936 runs")
    aggregate=[]
    for method,limit in sorted({(r["method"],r["L"]) for r in all_runs}):
        chosen=[r for r in all_runs if r["method"]==method and r["L"]==limit]
        full_products=sum(r["audit_n_leaves"]*r["audit_dimension"] for r in chosen)
        adaptive_products=sum(r["scalar_coordinate_products"] for r in chosen)
        full_leaf=sum(r["audit_n_leaves"] for r in chosen)
        adaptive_leaf=sum(r["scored_leaves"] for r in chosen)
        aggregate.append(dict(method=method,L=limit,questions=len(chosen),
            ordered_exact=sum(r["exact_match"] for r in chosen),
            topL_overlap=float(np.mean([r["topL_overlap"] for r in chosen])),
            gold_hits=sum(r["gold_hits"] for r in chosen),gold_total=sum(r["gold_total"] for r in chosen),
            any_gold_queries=sum(r["gold_hits"]>0 for r in chosen),
            all_gold_queries=sum(r["gold_hits"]==r["gold_total"] for r in chosen),
            exact_leaf_scores=adaptive_leaf,full_leaf_scores=full_leaf,
            leaf_reduction=1-adaptive_leaf/full_leaf,
            coordinate_products=adaptive_products,full_coordinate_products=full_products,
            coordinate_product_reduction=1-adaptive_products/full_products,
            shared_clip_master_coordinate_products=sum(r["audit_n_masters"]*512 for r in chosen)))
    max5=next(r for r in aggregate if r["method"]=="empirical_max_both" and r["L"]==5)
    H.check(len(changed)==21 and max5["ordered_exact"]==350,"21 changed maxboth questions")
    missed_count=sum(len(row["missed"]) for row in changed)
    H.check(missed_count==25,"25 missed full Top5 entries")
    H.check(max5["gold_hits"]==491,"maxboth491 gold")
    H.check(next(r for r in aggregate if r["method"]=="full_scan" and r["L"]==5)["gold_hits"]==492,"full492 gold")
    miss_result=dict(method="empirical_max_both",L=5,changed_queries=len(changed),
                     missed_full_top5_entries=missed_count,
                     missed_gold_entries=sum(x["is_gold"] for r in changed for x in r["missed"]),
                     gained_gold_entries=sum(x["is_gold"] for r in changed for x in r["gained"]),
                     attribution="Each missing full Top5 leaf was never exactly scored. Its first skip/stop event is reproduced from visited scores only; reported actual values are post hoc audit oracles.",
                     questions=changed)
    (BASE/"miss_audit.json").write_text(json.dumps(miss_result,ensure_ascii=False,indent=2)+"\n")
    for p in [ROOT/"scripts/graph/observed_spread_search.py",ROOT/"scripts/analysis/compare_adaptive_retrieval.py"]:
        hashes[str(p.relative_to(ROOT))]=H.sha(p)
    for path,expected in hashes.items():H.check(H.sha(ROOT/path)==expected,"audited sources unchanged")
    result=dict(passed=True,checks=H.checks,domains=domain_stats,questions=371,gold_occurrences=1041,
                runs=5936,independence="No production search/fit functions imported. All 4,452 empirical runs independently replayed, and all 1,113 saved L5 traces compared field by field.",
                trace_counts=trace_counts,aggregate=aggregate,source_hashes=hashes,
                encoded_input_truncation=truncated_inputs,
                max_both_miss_summary={k:v for k,v in miss_result.items() if k!="questions"},
                scientific_boundary=[
                    "Predictions are empirical estimates, never certified bounds; exact_guarantee=False and is_certified_bound=False were verified.",
                    "Current-cluster scores alone update observed offset/radius ratio; unvisited exact scores, gold and full oracle never enter the independent replay controller.",
                    "Every centroid dot is charged; coordinate ratios are ratios of summed work, not unweighted domain averages.",
                    "Original dialogue plus complete caption is the reader evidence; no reader was run by this experiment.",
                    "Home is the development domain; other seven domains constitute transfer evaluation.",
                    "Eight Social queries use their frozen manifest retrieval_query with cached query-image caption for BGE; CLIP still uses question text.",
                    "One Card leaf has 577 original BGE tokens truncated at512; both adaptive and exhaustive comparisons use the same fixed encoded vector. Saved reader text remains complete.",
                    "Timing is recomputed from saved samples only; no claim that fewer coordinate products necessarily means lower latency.",
                    "Model encoding and image processing are outside this audit; inputs/hashes and score matrices are checked, not re-encoded.",
                ])
    (BASE/"validation_all_empirical_independent.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(passed=True,checks=H.checks,max_both_L5=max5,miss_summary=result["max_both_miss_summary"]),ensure_ascii=False))


if __name__=="__main__":
    main()
