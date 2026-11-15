#!/usr/bin/env python3
"""Independent saved-state reconstruction audit; never opens original pixels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "reports/home_caption_ward_visual_20261020"
checks = []
max_errors = {}


def read_json(path):
    return json.loads(Path(path).read_text())


def check(name, condition):
    checks.append({"check": name, "passed": bool(condition)})
    if not condition:
        raise AssertionError(name)


def close(name, actual, expected, tolerance=1e-11):
    delta = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))
    family = name.split("/")[0]
    max_errors[family] = max(max_errors.get(family, 0.0), delta)
    check(name, delta <= tolerance)


def main():
    graph = read_json(SOURCE / "memory/graph.json")
    manifest = read_json(SOURCE / "input_manifest.json")["home"]
    audits = [json.loads(line) for line in (SOURCE / "insertion_audit.jsonl").read_text().splitlines()]
    result = read_json(OUT / "results.json")
    protocol = read_json(OUT / "protocol.json")
    with (OUT / "per_image.csv").open(encoding="utf-8-sig") as handle:
        image_rows = list(csv.DictReader(handle))
    with (OUT / "per_cluster.csv").open(encoding="utf-8-sig") as handle:
        cluster_rows = list(csv.DictReader(handle))
    for name, expected in protocol["source_sha256"].items():
        check(f"source_hash/{name}", hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected)
    nodes = graph["image_nodes"]
    ids = [node["id"] for node in nodes]
    index = {image_id: i for i, image_id in enumerate(ids)}
    count = len(ids)
    check("89_unique_images", len(ids) == len(index) == len(image_rows) == 89)
    check("manifest_order", ids == [row["image_id"] for row in manifest])
    check("output_order", ids == [row["image_id"] for row in image_rows])
    check("all_89_arrivals", ids == [step["image_id"] for step in audits])
    basis = np.load(SOURCE / "memory/basis.npy", allow_pickle=False)
    sums = np.load(SOURCE / "memory/master_sums.npy", allow_pickle=False)
    coords = np.asarray([node["visual_coords"] for node in nodes])
    check("strict_2D", basis.shape == (512, 2) and coords.shape == (89, 2))
    close("orthonormal_basis", basis.T @ basis, np.eye(2))
    gram = np.eye(count)
    seen = set()
    with (ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["earlier_id"] in index and row["later_id"] in index:
                i, j = index[row["earlier_id"]], index[row["later_id"]]
                key = tuple(sorted((i, j)))
                check(f"unique_gram_pair/{i}/{j}", i != j and key not in seen)
                seen.add(key)
                gram[i, j] = gram[j, i] = float(row["cosine"])
    check("complete_gram", len(seen) == count * (count - 1) // 2)
    residual_gram = gram - coords @ coords.T
    check("gram_PSD", np.linalg.eigvalsh(gram).min() >= -1e-11)
    check("residual_gram_PSD", np.linalg.eigvalsh(residual_gram).min() >= -1e-11)

    # Use audited overlap transport matrices, rather than the evaluator's
    # cluster-loop implementation, to reconstruct the residual history.
    old_ids, old_groups = [], {}
    coefficients = np.empty((0, count))
    for position, step in enumerate(audits):
        current_ids = [cluster["master_id"] for cluster in step["clusters"]]
        groups = {cluster["master_id"]: set(cluster["member_ids"]) for cluster in step["clusters"]}
        old_index = {key: i for i, key in enumerate(old_ids)}
        new_index = {key: i for i, key in enumerate(current_ids)}
        transport = np.zeros((len(current_ids), len(old_ids)))
        overlap_keys = set()
        for overlap in step["source_overlaps"]:
            old, new = overlap["old_master_id"], overlap["new_master_id"]
            actual = old_groups[old] & groups[new]
            key = (old, new)
            check(f"overlap_unique/{position}/{old}/{new}", key not in overlap_keys)
            overlap_keys.add(key)
            check(f"overlap_members/{position}/{old}/{new}",
                  actual == set(overlap["member_ids"]) and overlap["count"] == len(actual))
            check(f"overlap_denominator/{position}/{old}/{new}", overlap["old_n"] == len(old_groups[old]))
            close(f"overlap_fraction/{position}/{old}/{new}", overlap["fraction"], len(actual) / len(old_groups[old]))
            transport[new_index[new], old_index[old]] = len(actual) / len(old_groups[old])
        check(f"overlap_complete/{position}", overlap_keys == {
            (old, new) for old in old_ids for new in current_ids if old_groups[old] & groups[new]})
        close(f"transport_column_conservation/{position}", transport.sum(axis=0), np.ones(len(old_ids)))
        coefficients = transport @ coefficients
        destinations = [i for i, key in enumerate(current_ids) if step["image_id"] in groups[key]]
        check(f"new_image_single_destination/{position}", len(destinations) == 1)
        coefficients[destinations[0], position] += 1
        close(f"coefficient_row_counts/{position}", coefficients.sum(axis=1),
              [len(groups[key]) for key in current_ids])
        close(f"coefficient_global_conservation/{position}",
              coefficients.sum(axis=0), np.r_[np.ones(position + 1), np.zeros(count - position - 1)])
        check(f"coefficient_nonnegative/{position}", coefficients.min() >= 0)
        old_ids, old_groups = current_ids, groups
    final_coefficient = {key: coefficients[i] for i, key in enumerate(old_ids)}
    masters = graph["masters"]
    master_index = {master["id"]: i for i, master in enumerate(masters)}
    check("25_final_masters", len(masters) == len(cluster_rows) == 25)
    means, mean_coords, weights = [], [], []
    for master in masters:
        key = master["id"]
        check(f"final_members/{key}", old_groups[key] == set(master["member_ids"]))
        check(f"final_n/{key}", master["n"] == len(master["member_ids"]))
        mean = sums[master["visual_sum_ref"]["row"]] / master["n"]
        zbar = coords[[index[member] for member in master["member_ids"]]].mean(axis=0)
        means.append(mean)
        mean_coords.append(zbar)
        weights.append(final_coefficient[key] / master["n"])
        close(f"master_projection/{key}", mean @ basis, zbar)
    means, mean_coords, weights = map(np.asarray, (means, mean_coords, weights))
    residuals = means - mean_coords @ basis.T
    close("master_residual_Gram", residuals @ residuals.T, weights @ residual_gram @ weights.T)
    close("master_residual_orthogonality", residuals @ basis, np.zeros((len(masters), 2)))

    calculated = {}
    for i, (node, row) in enumerate(zip(nodes, image_rows)):
        m = master_index[node["master_id"]]
        check(f"image_master/{i}", row["master_id"] == node["master_id"])
        check(f"image_filename/{i}", row["filename"] == manifest[i]["filename"])
        check(f"image_cluster_size/{i}", int(row["cluster_size"]) == masters[m]["n"])
        weight = weights[m]
        # Algebraically equivalent reconstruction expressed via its shared
        # orthogonal residual, independently of evaluator's mean+delta function.
        rebuilt = basis @ coords[i] + residuals[m]
        norm = np.linalg.norm(rebuilt)
        dot = coords[i] @ coords[i] + residual_gram[i] @ weight
        close(f"reconstruction_norm/{i}", norm ** 2, coords[i] @ coords[i] + weight @ residual_gram @ weight)
        close(f"reconstruction_projection/{i}", rebuilt @ basis, coords[i])
        cosine = dot / norm
        check(f"unclipped_cosine_range/{i}", -1 - 1e-12 <= cosine <= 1 + 1e-12)
        delta = weight.copy()
        delta[i] -= 1
        squared_error = delta @ residual_gram @ delta
        close(f"reconstruction_error_identity/{i}", squared_error, 1 + norm ** 2 - 2 * dot)
        baseline = (mean_coords[m] @ coords[i] + residual_gram[i] @ weight) / np.linalg.norm(means[m])
        scores = {
            "cosine_original_vs_reconstructed": cosine,
            "cosine_original_vs_master_only": baseline,
            "cosine_gain_from_two_coords": cosine - baseline,
            "reconstruction_raw_l2_error": np.sqrt(max(0.0, squared_error)),
            "reconstruction_norm": norm,
            "normalized_reconstruction_l2_error": np.sqrt(max(0.0, 2 - 2 * cosine)),
        }
        calculated[node["id"]] = scores
        for name, value in scores.items():
            close(f"per_image_{name}/{i}", float(row[name]), value)
        del rebuilt
    for field in ("cosine_original_vs_reconstructed", "cosine_original_vs_master_only",
                  "cosine_gain_from_two_coords", "reconstruction_raw_l2_error", "reconstruction_norm"):
        values = np.array([calculated[key][field] for key in ids])
        expected = {
            "n": count, "mean": values.mean(), "median": np.median(values),
            "min": values.min(), "max": values.max(), "p10": np.percentile(values, 10),
            "p90": np.percentile(values, 90), "std_population": values.std(),
        }
        for name, value in expected.items():
            close(f"summary/{field}/{name}", result[field][name], value)
    for row, master in zip(cluster_rows, masters):
        check(f"cluster_output_order/{master['id']}", row["master_id"] == master["id"])
        values = np.array([calculated[key]["cosine_original_vs_reconstructed"] for key in master["member_ids"]])
        close(f"cluster_mean/{master['id']}", float(row["mean_cosine"]), values.mean())
        close(f"cluster_min/{master['id']}", float(row["min_cosine"]), values.min())
        close(f"cluster_max/{master['id']}", float(row["max_cosine"]), values.max())
        close(f"cluster_L2/{master['id']}", float(row["mean_l2"]),
              np.mean([calculated[key]["reconstruction_raw_l2_error"] for key in master["member_ids"]]))
    names = {row["filename"]: i for i, row in enumerate(manifest)}
    a, b = names["D7_IMG_005.jpg"], names["D7_IMG_006.jpg"]
    ma, mb = master_index[nodes[a]["master_id"]], master_index[nodes[b]["master_id"]]
    ha, hb = basis @ coords[a] + residuals[ma], basis @ coords[b] + residuals[mb]
    pair = result["q10_D7_005_vs_D7_006"]
    check("q10_pair_same_master", ma == mb and pair["same_master"])
    close("q10_original_pair_cosine", pair["original_pair_cosine"], gram[a, b])
    close("q10_reconstructed_pair_cosine", pair["reconstructed_pair_cosine"], ha @ hb / np.linalg.norm(ha) / np.linalg.norm(hb))
    close("q10_pair_distance", pair["reconstructed_squared_distance"], (ha - hb) @ (ha - hb))
    close("q10_pair_distance_from_2D", pair["reconstructed_squared_distance"], (coords[a] - coords[b]) @ (coords[a] - coords[b]))
    close("q10_squared_distance_retention", pair["squared_distance_retention"],
          np.sum((coords[a] - coords[b]) ** 2) / (2 - 2 * gram[a, b]))
    del ha, hb
    with np.load(ROOT / "reports/home_q10_top5_embeddings_20261020/embeddings.npz", allow_pickle=False) as cache:
        for memory_id, original in zip(cache["memory_ids"], cache["image_embeddings"]):
            key = f"{memory_id}#image1"
            i = index[key]
            m = master_index[nodes[i]["master_id"]]
            rebuilt = basis @ coords[i] + residuals[m]
            close(f"direct_original_norm/{key}", np.linalg.norm(original), 1.0)
            close(f"direct_projection/{key}", basis.T @ original, coords[i])
            close(f"direct_cosine/{key}", original @ rebuilt / np.linalg.norm(original) / np.linalg.norm(rebuilt),
                  calculated[key]["cosine_original_vs_reconstructed"])
            close(f"direct_L2/{key}", np.linalg.norm(original - rebuilt), calculated[key]["reconstruction_raw_l2_error"])
            del rebuilt
    check("no_vector_output_artifacts", not [p for p in OUT.iterdir() if p.suffix in (".npz", ".npy", ".pt", ".pkl")])
    validation = {
        "passed": True,
        "checks_passed": len(checks),
        "n_images": count,
        "n_final_masters": len(masters),
        "n_arrivals_replayed": len(audits),
        "independent_method": "overlap transport matrices plus orthogonal residual reconstruction; no evaluator imports",
        "original_pixels_read": 0,
        "image_or_text_encoder_calls": 0,
        "original_or_reconstructed_vectors_written": 0,
        "max_absolute_errors": max_errors,
        "mean_reconstruction_cosine": result["cosine_original_vs_reconstructed"]["mean"],
        "mean_master_only_cosine": result["cosine_original_vs_master_only"]["mean"],
        "checks": checks,
    }
    (OUT / "validation_independent.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in validation.items() if key not in ("checks", "max_absolute_errors")}, indent=2))


if __name__ == "__main__":
    main()
