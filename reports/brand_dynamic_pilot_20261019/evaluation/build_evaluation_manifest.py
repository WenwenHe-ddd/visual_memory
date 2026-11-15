"""Evaluation-only manifest. No labels or descriptions may be clustering inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GRAPH = ROOT / "results/visual_embedding_graph/Brand_Memory_Test/graph.json"
DIALOG = ROOT / "data/MemEye/data/dialog/Brand_Memory_Test.json"
PAIRWISE = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"

# Written after viewing all 30 actual images, before receiving pilot memberships.
DESCRIPTIONS = {
    "CocaCola_1": "Blue football sponsorship advertisement with a player, dense offer text and several partner logos.",
    "CocaCola_2": "White background; two hands clink personalized cola bottles; text on the left.",
    "CocaCola_3": "Red background; smiling man and woman hold cola bottles; white text on the right.",
    "CocaCola_4": "Red background; smiling woman presents a cola bottle; white and yellow text.",
    "CocaCola_5": "Red background; four chilled glass cola bottles with different labels; white text on the right.",
    "BurgerKing_1": "Pale background; woman's legs in red heels and a cardboard brand crown; small salad at lower right.",
    "BurgerKing_2": "Cream background; breakfast bowl and muffin sandwich with large brown headline.",
    "BurgerKing_3": "Dark red background; large central stacked burger with white headline.",
    "BurgerKing_4": "Black background; black-bun burger on wood and sauce bottles, with white headline.",
    "BurgerKing_5": "Pale background; vertically stacked cartoon cows and small burger-brand emblem.",
    "Dunkin_1": "Brown and orange dense campaign infographic with diagrams, photographs and coffee imagery.",
    "Dunkin_2": "White background; tomato and yellow ingredient wedge, small paragraph and bottom brand lockup.",
    "Dunkin_3": "Pale green background; fried snack resembling a spider, small smoothie cup and white text.",
    "Dunkin_4": "White background; single bacon strip, small paragraph and bottom brand lockup.",
    "Dunkin_5": "Yellow background; coin/donut composite, bold pink and orange text.",
    "McDonalds_1": "Red graphic background; runner and large salad bowl with decorative silhouettes.",
    "McDonalds_2": "White background; line-up of burgers, fries and other menu products under a large menu-history headline.",
    "McDonalds_3": "Light blue background; goldfish inside a transparent burger-shaped vessel.",
    "McDonalds_4": "Red background; fries arranged into three fireworks, small bottom text and logo.",
    "McDonalds_5": "Red background; fries arranged into a Wi-Fi symbol, small bottom-right logo.",
    "Pepsi_1": "Blue background; three blue soda cans surrounded by water splashes.",
    "Pepsi_2": "Blue background; large chilled blue soda can and bold white slogan.",
    "Pepsi_3": "Dark blue background; small central blue soda can surrounded by water droplets.",
    "Pepsi_4": "Light blue background; woman drinks from a silver soda can; text and an extra can on the right.",
    "Pepsi_5": "Blue/red/white background; large soda paper cup with ice and splash graphics.",
    "Starbucks_1": "White background; tilted water bottle pours a heart-shaped blue stream, with cause-marketing text.",
    "Starbucks_2": "Warm brown/yellow blurred background; large white takeaway coffee cup on the left, slogan on the right.",
    "Starbucks_3": "Dark brown background; glass coffee cup emits gold sparkles, white slogan on the right.",
    "Starbucks_4": "White campaign board; coffee beans form Braille, explanatory text and demonstration photos.",
    "Starbucks_5": "Four-panel composition of whipped-cream drinks in glass mugs, green bands and large central logo.",
}

# These are limited visual sanity checks, not exhaustive human ground truth.
CASES = [
    ("Pepsi_1", "Pepsi_2", "close", "Blue canned-soda product shots with a large circular logo and chilled/dripping surface."),
    ("Pepsi_1", "Pepsi_3", "close", "Blue canned-soda product shots with water effects on blue backgrounds."),
    ("Pepsi_2", "Pepsi_3", "close", "Blue canned-soda product shots on blue backgrounds; scale and layout differ."),
    ("Dunkin_2", "Dunkin_4", "close", "Shared sparse white product-ad layout, small side paragraph and same bottom lockup."),
    ("McDonalds_4", "McDonalds_5", "close", "Red field with yellow fries arranged into a symbol and small bottom branding."),
    ("CocaCola_3", "CocaCola_4", "close", "Red cola campaigns with smiling people holding bottles and white headline text."),
    ("CocaCola_1", "BurgerKing_1", "different", "Football/offer board versus isolated legs/crown; distinct composition and subject."),
    ("Dunkin_1", "McDonalds_1", "different", "Dense brown infographic versus red salad/runner advertisement."),
    ("CocaCola_1", "CocaCola_5", "different_same_brand", "Blue football/partner board versus red bottle product row."),
    ("BurgerKing_1", "BurgerKing_4", "different_same_brand", "Pale legs/crown concept versus black burger product shot."),
    ("Starbucks_1", "Starbucks_3", "different_same_brand", "White water-bottle/heart scene versus dark glowing coffee cup."),
    ("Pepsi_2", "CocaCola_5", "cross_brand_related", "Both emphasize chilled packaged soda; brand, color and can/bottle geometry differ."),
    ("McDonalds_2", "BurgerKing_3", "cross_brand_related", "Both show burger products; single red hero shot versus white menu line-up."),
    ("CocaCola_4", "Pepsi_4", "cross_brand_related", "Both place a woman with a soda package against a brand-colored background."),
    ("Starbucks_1", "Dunkin_2", "ambiguous_layout", "Both use sparse white backgrounds but depict different objects and product categories."),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    graph = json.loads(GRAPH.read_text())
    dialog = json.loads(DIALOG.read_text())
    rounds = {
        r["round"]: (session["date"], r)
        for session in dialog["multi_session_dialogues"]
        for r in session["dialogues"]
    }
    masters = {e["image_id"]: e["master_id"] for e in graph["edges"]}
    members = defaultdict(list)
    rows = []
    for index, node in enumerate(graph["image_nodes"]):
        path = Path(node["image_path"])
        round_id = node["id"].split("/")[-1].split("#")[0]
        date, original = rounds[round_id]
        assert node["dialogue"] == {k: original[k] for k in ("user", "assistant")}
        assert node["timestamp"] == date
        assert "Brand_Memory_Test/" + path.name in original["input_image"]
        assert node["caption"].strip()
        with Image.open(path) as im:
            image_size = list(im.size)
            im.verify()
        row = {
            "stream_index": index,
            "id": node["id"],
            "image_path": str(path),
            "image_sha256": sha(path),
            "image_size": image_size,
            "filename": path.name,
            "filename_brand_evaluation_label": path.stem.rsplit("_", 1)[0],
            "labels_are_evaluation_only": True,
            "visual_description_evaluation_only": DESCRIPTIONS[path.stem],
            "baseline_master_id": masters[node["id"]],
            "timestamp": node["timestamp"],
            "dialogue_matches_source": True,
            "caption_present": True,
            "caption_characters": len(node["caption"]),
            "caption_sha256": hashlib.sha256(node["caption"].encode()).hexdigest(),
        }
        rows.append(row)
        members[masters[node["id"]]].append(path.name)
    assert len(rows) == 30
    assert len({r["image_sha256"] for r in rows}) == 30
    pair_rows = [r for r in csv.DictReader(PAIRWISE.open()) if r["task"] == "Brand_Memory_Test"]
    assert len(pair_rows) == 435
    id_to_index = {r["id"]: i for i, r in enumerate(rows)}
    by_name = {Path(r["filename"]).stem: r for r in rows}
    distances = np.zeros((30, 30))
    similarities = {}
    for pair in pair_rows:
        i, j = id_to_index[pair["earlier_id"]], id_to_index[pair["later_id"]]
        distances[i, j] = distances[j, i] = max(0.0, 1.0 - float(pair["cosine"]))
        similarities[frozenset((Path(pair["earlier_image"]).stem, Path(pair["later_image"]).stem))] = float(pair["cosine"])
    true = [r["filename_brand_evaluation_label"] for r in rows]
    pred = [r["baseline_master_id"] for r in rows]
    tp = fp = fn = 0
    for i, j in combinations(range(30), 2):
        same_truth, same_pred = true[i] == true[j], pred[i] == pred[j]
        tp += same_truth and same_pred
        fp += not same_truth and same_pred
        fn += same_truth and not same_pred
    probes = []
    for left, right, relation, reason in CASES:
        probes.append({
            "left_id": by_name[left]["id"], "right_id": by_name[right]["id"],
            "left_image": left + ".png", "right_image": right + ".png",
            "relation_evaluation_only": relation, "reason": reason,
            "cosine_from_prior_diagnostic": similarities[frozenset((left, right))],
            "baseline_same_cluster": by_name[left]["baseline_master_id"] == by_name[right]["baseline_master_id"],
        })
    manifest = {
        "schema_version": 1, "task": "Brand_Memory_Test", "purpose": "evaluation_only",
        "prohibition": "Do not feed brand labels, image filenames, descriptions, cases or source dialogue into visual clustering.",
        "source_graph": str(GRAPH), "source_graph_sha256": sha(GRAPH),
        "source_dialog": str(DIALOG), "source_dialog_sha256": sha(DIALOG),
        "count": len(rows), "brand_counts": dict(Counter(true)), "images": rows,
    }
    findings = {
        "evaluation_scope": "All 30 original images visually inspected by the assistant; this is not independent human annotation.",
        "case_selection": "Fixed before inspecting the new pilot's assignments, informed by original images and previous baseline diagnostics; therefore diagnostic and not a held-out benchmark.",
        "brand_label_limit": "Brand identity is only a secondary semantic proxy. Same-brand images can be visually different and cross-brand images can share composition or object type.",
        "payload_audit": {"images_verified": 30, "unique_image_hashes": 30, "dialogues_match_source": 30, "timestamps_match_source": 30, "full_graph_captions_present": 30, "caption_characters_min": min(r["caption_characters"] for r in rows), "caption_characters_max": max(r["caption_characters"] for r in rows)},
        "baseline": {
            "master_count": len(members), "cluster_size_histogram": dict(Counter(map(len, members.values()))),
            "memberships": dict(members), "cosine_silhouette": float(silhouette_score(distances, pred, metric="precomputed")),
            "secondary_brand_adjusted_rand": float(adjusted_rand_score(true, pred)),
            "secondary_brand_normalized_mutual_information": float(normalized_mutual_info_score(true, pred)),
            "secondary_brand_pair_tp_fp_fn": {"tp": int(tp), "fp": int(fp), "fn": int(fn)},
            "secondary_brand_pair_precision": tp / (tp + fp), "secondary_brand_pair_recall": tp / (tp + fn),
            "secondary_brand_pair_f1": 2 * tp / (2 * tp + fp + fn),
        },
        "visual_cases": probes,
        "evaluation_scalar_distance_source": str(PAIRWISE), "evaluation_scalar_distance_source_sha256": sha(PAIRWISE),
        "notes": [
            "No per-image embedding was generated or stored by this evaluation script.",
            "Cached pairwise scalar cosines are evaluation evidence only, never dynamic insertion input.",
            "Captions are present and hashed from the existing graph; caption factual accuracy has not been exhaustively checked.",
            "Use the strong same-layout cases plus geometry and storage/time audits, not only brand ARI, to judge the pilot.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (OUT / "evaluation_findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(rows), "baseline": findings["baseline"], "output": str(OUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
