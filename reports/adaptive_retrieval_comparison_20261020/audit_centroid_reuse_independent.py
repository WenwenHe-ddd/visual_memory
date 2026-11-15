"""Independent replay of centroid identity reuse and tie-cost accounting."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import audit_home_exact_independent as H
import audit_all_empirical_independent as E

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def replay(x,groups,raw_query,order,limit):
    q=H.robust_unit(raw_query)
    centers=np.array([x[g].mean(axis=0) for g in groups])
    means=centers@q
    radii=np.array([np.linalg.norm(x[g]-c,axis=1).max() for g,c in zip(groups,centers)])
    margin=64*np.finfo(float).eps*(x.shape[1]+1)
    scores,inferred={},{}
    direct=set()
    selected=[];visited=[];skipped=[];tail=[];trace=[];fallback=[]
    offset=ratio=0.
    max_inference_error=0.
    for pos,mi in enumerate(order):
        cutoff=scores[selected[-1]] if len(selected)==limit else None
        predicted=np.maximum(means+offset,means+ratio*radii)
        largest=float(predicted[order[pos:]].max())
        entry=dict(position=pos,master_index=mi,cutoff_before=cutoff,observed_offset=offset,
                   observed_radius_ratio=ratio,predicted_max=float(predicted[mi]),
                   max_remaining_predicted=largest,is_certified_bound=False)
        if cutoff is not None and largest<cutoff:
            tail=order[pos:];trace.append({**entry,"action":"empirical_stop"});break
        if cutoff is not None and predicted[mi]<cutoff:
            skipped.append(mi);trace.append({**entry,"action":"empirical_skip"});continue
        members=groups[mi]
        H.check(not (set(members)&set(scores)),"each visited leaf resolved once")
        values=np.empty(len(members))
        values[:-1]=np.einsum("ij,j->i",x[members[:-1]],q,optimize=False)
        values[-1]=len(members)*means[mi]-values[:-1].sum()
        direct.update(members[:-1])
        inferred[members[-1]]=len(members)*margin
        scores.update((i,float(s)) for i,s in zip(members,values))
        # Oracle is audit-only; never passed to the controller or used to rank.
        error=abs(float(values[-1]-x[members[-1]]@q))
        max_inference_error=max(max_inference_error,error)
        H.check(error<=inferred[members[-1]],"identity inference within declared FP envelope")
        visited.append(mi)
        visit_refinements=[]
        while True:
            ranked=sorted(scores,key=lambda i:(-scores[i],i))
            ambiguous=set()
            for a,b in zip(ranked[:limit],ranked[1:limit+1]):
                allowance=inferred.get(a,0.)+inferred.get(b,0.)
                if allowance and scores[a]-scores[b]<=allowance:
                    ambiguous.update(i for i in (a,b) if i in inferred)
            if len(ranked)>=limit:
                kth=ranked[limit-1]
                lower=scores[kth]-inferred.get(kth,0.)
                chosen=set(ranked[:limit])
                ambiguous.update(i for i,e in inferred.items() if i not in chosen and scores[i]+e>=lower)
            if not ambiguous:break
            for i in sorted(ambiguous):
                H.check(i not in direct,"tie fallback never repeats a direct dot")
                before=scores[i]
                scores[i]=float(np.einsum("ij,j->i",x[i:i+1],q,optimize=False)[0])
                direct.add(i);inferred.pop(i)
                fallback.append(dict(master_visit=mi,leaf_index=i,score_before=before,score_after=scores[i]))
                visit_refinements.append(dict(leaf_index=i,score=scores[i],coordinate_products=x.shape[1]))
        selected=sorted(scores,key=lambda i:(-scores[i],i))[:limit]
        for observed_mi in (visited if visit_refinements else [mi]):
            deviation=max(scores[i] for i in groups[observed_mi])-means[observed_mi]
            offset=max(offset,float(deviation))
            if radii[observed_mi]>0:ratio=max(ratio,float(deviation/radii[observed_mi]))
        H.check(not (direct&set(inferred)),"final inferred/direct states mutually exclusive")
        H.check(direct|set(inferred)==set(scores),"all resolved scores have exactly one state")
        trace.append({**entry,"action":"visit","leaf_indices":members,"scores":values.tolist(),
                      "observed_max":float(values.max()),"direct_leaf_dots":len(members)-1,
                      "centroid_inferred_leaf":members[-1],
                      "tie_refinements":visit_refinements,
                      "cutoff_after":scores[selected[-1]] if len(selected)==limit else None})
    result=dict(selected_indices=selected,selected_scores=[scores[i] for i in selected],
                scored_leaves=len(scores),scored_leaf_indices=list(scores),visited_masters=len(visited),
                visited_master_indices=visited,skipped_master_indices=skipped,tail_master_indices=tail,
                bound_dot_count=len(groups),exact_leaf_dot_count=len(direct),
                centroid_inferred_score_count=len(inferred),
                scalar_coordinate_products=(len(groups)+len(direct))*x.shape[1],
                prediction="max_both",exact_guarantee=False,reuse_centroid=True)
    return result,trace,fallback,max_inference_error


def main():
    hashes={}
    paths=sorted(p for p in (BASE/"all_centroid_reuse").iterdir() if p.is_dir())
    H.check(len(paths)==8,"all8 completed domains")
    all_rows=[];fallbacks=[];maxerr=0.;maxscorechange=0.
    for folder in paths:
        name=folder.name;data=BASE/"inputs_by_domain"/name
        inputs=H.read(data/"inputs.json")
        features=np.load(data/"features.npz")
        old=H.read(BASE/"all_empirical"/name/"runs.json")
        oldmap={(r["question_index"],r["L"],r["method"]):r for r in old}
        rows=H.read(folder/"runs.json");summ=H.read(folder/"summary.json")
        x=H.robust_unit(features["bge_leaf"]);queries=features["bge_query"]
        orders=features["clip_order"].tolist()
        leafindex={l["image_id"]:i for i,l in enumerate(inputs["leaves"])}
        groups=[[leafindex[i] for i in m["member_ids"]] for m in inputs["masters"]]
        for path,expected in summ["source_hashes"].items():H.check(H.sha(path)==expected,"input hashes")
        for p in [data/"inputs.json",data/"features.npz",folder/"runs.json",folder/"summary.json",
                  folder/"evaluation_scores.npz",folder/"latency_samples.json",BASE/"all_empirical"/name/"runs.json"]:
            hashes[str(p.relative_to(ROOT))]=H.sha(p)
        H.check(len(rows)==len(queries)*4*2,"all2methods4L")
        H.check(len({(r["question_index"],r["L"],r["method"]) for r in rows})==len(rows),"unique result keys")
        for row in rows:
            qi,L=row["question_index"],row["L"]
            method="full_scan" if row["method"]=="full_scan" else "empirical_max_both"
            reference=oldmap[qi,L,method]
            H.same(row["selected_indices"],reference["selected_indices"],"new/old ordered TopL exactly")
            diff=float(np.max(np.abs(np.array(row["selected_scores"])-reference["selected_scores"])))
            maxscorechange=max(maxscorechange,diff)
            H.check(diff<1e-12,"new/old exact score consistency")
            for key in ("gold_hits","gold_total","exact_match","topL_overlap","selected_memory_ids","reference_indices"):
                H.same(row[key],reference[key],"same downstream "+key)
            if method=="empirical_max_both":
                result,trace,fallback,error=replay(x,groups,queries[qi],orders[qi],L)
                maxerr=max(maxerr,error)
                H.same(row,result,"centroid replay")
                if L==5:H.same(row["trace"],trace,"saved reuse trace")
                else:H.check(row["trace"]==[],"nonL5 trace omitted")
                H.same(row["scored_leaf_indices"],reference["scored_leaf_indices"],"same resolved leaf order")
                H.same(row["visited_master_indices"],reference["visited_master_indices"],"same visited masters")
                H.check(row["exact_leaf_dot_count"]+row["centroid_inferred_score_count"]==row["scored_leaves"],
                        "direct/inferred accounting")
                H.check(row["scalar_coordinate_products"]==(len(groups)+row["exact_leaf_dot_count"])*x.shape[1],
                        "all centroid and direct/fallback dots charged")
                for event in fallback:fallbacks.append(dict(domain=name,question_index=qi,L=L,**event))
            else:
                H.check(row["scalar_coordinate_products"]==x.size,"fullbaseline cost")
            all_rows.append({**row,"domain":name,"n_leaves":len(x),"n_masters":len(groups),
                             "original_products":reference["scalar_coordinate_products"]})
        for scope,indices in {
            "all":set(range(len(queries))),
            "text_only":{i for i,q in enumerate(inputs["questions"]) if not q["query_image_paths"]},
            "image_assisted":{i for i,q in enumerate(inputs["questions"]) if q["query_image_paths"]},
        }.items():
            H.same(summ["scopes"][scope],E.summary(rows,inputs,len(x),x.shape[1],indices),"new summary "+scope)
        timings=H.read(folder/"latency_samples.json")
        for method,samples in timings.items():
            times=np.array([s["nanoseconds"] for s in samples])/1e6
            H.same(summ["latency"][method]["median_ms"],float(np.median(times)),"reuse timing median")
            H.same(summ["latency"][method]["p95_ms"],float(np.quantile(times,.95)),"reuse timing p95")
    # Deliberate exact-tie fixtures exercise fallback paths absent in this run.
    sys.path.insert(0,str(ROOT))
    from scripts.graph.observed_spread_search import ObservedSpreadSearch
    fixtures=[]
    for size,groups,limit in [
        (8,[[0,1],[2,3],[4,5],[6,7]],5),
        (8,[[i] for i in range(8)],5),
        (12,[list(range(0,3)),list(range(3,6)),list(range(6,9)),list(range(9,12))],10),
    ]:
        vectors=np.repeat([[.8,.6]],size,axis=0)
        order=list(reversed(range(len(groups))))
        independent,trace,fallback,error=replay(H.robust_unit(vectors),groups,np.array([.8,.6]),order,limit)
        core=ObservedSpreadSearch.fit(vectors,groups)
        production=core.search(np.array([.8,.6]),order,limit,reuse_centroid=True)
        H.same(production,independent,"synthetic fallback core vs independent")
        # Order inside an ambiguous set is irrelevant; every refinement is logged.
        def canonical_trace(events):
            return [{**event,"tie_refinements":sorted(event["tie_refinements"],key=lambda r:r["leaf_index"])}
                    if "tie_refinements" in event else event for event in events]
        H.same(canonical_trace(production["trace"]),canonical_trace(trace),"synthetic fallback trace semantics")
        H.same(production["selected_indices"],list(range(limit)),"canonical stable identical-vector tie")
        H.check(len(fallback)>0,"tie fixture actually recomputes inferred values")
        H.check(production["exact_leaf_dot_count"]==size-len(groups)+len(fallback),"fallback cost counted")
        plain=core.search(np.array([.8,.6]),order,limit,reuse_centroid=True,record_trace=False)
        H.same(plain["selected_indices"],production["selected_indices"],"trace off output")
        H.same(plain["scalar_coordinate_products"],production["scalar_coordinate_products"],"trace off cost")
        fixtures.append(dict(size=size,groups=groups,L=limit,fall_back_dots=len(fallback),
                             direct_leaf_dots=production["exact_leaf_dot_count"],fallback_log=fallback))
    aggregates=[]
    for L in (1,3,5,10):
        chosen=[r for r in all_rows if r["L"]==L and r["method"]!="full_scan"]
        fullproducts=sum(r["n_leaves"]*384 for r in chosen)
        products=sum(r["scalar_coordinate_products"] for r in chosen)
        original=sum(r["original_products"] for r in chosen)
        aggregates.append(dict(L=L,questions=len(chosen),identical_to_original=len(chosen),
            resolved_leaf_scores=sum(r["scored_leaves"] for r in chosen),
            actual_direct_leaf_dots=sum(r["exact_leaf_dot_count"] for r in chosen),
            inferred_remaining_scores=sum(r["centroid_inferred_score_count"] for r in chosen),
            centroid_dots=sum(r["bound_dot_count"] for r in chosen),
            coordinate_products=products,original_empirical_coordinate_products=original,
            full_scan_coordinate_products=fullproducts,
            reduction_vs_original=1-products/original,reduction_vs_full_scan=1-products/fullproducts))
    for p in [ROOT/"scripts/graph/observed_spread_search.py",ROOT/"scripts/analysis/compare_adaptive_retrieval.py"]:
        hashes[str(p.relative_to(ROOT))]=H.sha(p)
    for p,value in hashes.items():H.check(H.sha(ROOT/p)==value,"sources stable during audit")
    result=dict(passed=True,checks=H.checks,runs=2968,reuse_runs=1484,
        method="empirical_max_both_reuse",independence="Independent formula/controller replay for every saved run; production implementation imported only for synthetic tie fixtures.",
        formula="q·x_last = n*(q·mean(x)) - sum_{other visited members}(q·x_i)",
        max_inferred_minus_direct_abs_error=maxerr,max_selected_score_change_vs_original=maxscorechange,
        formal_fallback_dot_count=len(fallbacks),formal_fallback_events=fallbacks,
        synthetic_tie_fixtures=fixtures,aggregates=aggregates,source_hashes=hashes,
        boundaries=[
            "scored_leaves counts resolved scores; exact_leaf_dot_count counts actual full-dimensional leaf dot products.",
            "Initial direct dots and still-inferred scores are disjoint; each ambiguous inferred score is directly recomputed at most once and then removed from the inferred set.",
            "Formal data triggered no tie recomputation; each visited cluster saves one full-dimensional dot, including singleton reuse.",
            "Visit traces contain initial n-1 direct dots and pre-fallback scores; each additional fallback dot is explicitly logged in tie_refinements and included in final counts.",
            "All saved output indices, membership visits and scores within1e-12 match the original empirical variant; this finite-data equivalence does not establish universal FP-equivalent routing near an empirical skip/stop boundary.",
            "The retrieval policy remains approximate relative to exhaustive BGE; centroid reuse does not turn empirical predictions into certified bounds.",
            "Unchanged ordered evidence means the original max_both reader payloads represent this reuse variant exactly.",
            "This audit uses the final rerun after two independent synthetic counterexamples were fixed: canonical per-row einsum for direct/fallback scoring and spread updates after refinement. Prior pilots remain under diagnostics and are excluded. Code hashes therefore differ from the earlier trace-only pilot.",
        ])
    (BASE/"validation_centroid_reuse_independent.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("passed","checks","max_inferred_minus_direct_abs_error",
                                          "max_selected_score_change_vs_original","formal_fallback_dot_count","aggregates")}))


if __name__=="__main__":
    main()
