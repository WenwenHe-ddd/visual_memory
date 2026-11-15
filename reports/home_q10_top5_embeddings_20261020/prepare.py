"""Prepare the five saved Q10 candidates as an isolated analysis artifact."""
import csv
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vendor"))

from scripts.analysis import frozen_comparison_inputs as frozen

RUN = ROOT / "data/independent_visual_inputs"
NAT = ROOT / "results/natural_observation"
PAIR_CSV = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"
PAIR_RESULT = ROOT / "reports/home_q10_top2_similarity_20261020/results.json"
QUESTION_ID = "Home_Renovation_Interior_Design/Q10"
EXPECTED_NAMES = ["D13_IMG_001.jpg", "D7_IMG_005.jpg", "D7_IMG_002.jpg",
                  "D7_IMG_006.jpg", "D13_IMG_002.jpg"]
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
TEXT_VARIANTS = {
    "text_full_caption": "full_caption",
    "text_dialogue_full_caption": "dialogue_full_caption",
    "text_dialogue_unique_caption": "retrieval_text",
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    assert (norms > 0).all()
    return x / norms


def main():
    if (OUT / "embeddings.npz").exists():
        raise FileExistsError("A prepared vector bundle already exists; choose a new directory.")
    import torch
    from sentence_transformers import SentenceTransformer
    from scripts.graph.visual_image_encoder import ImageEncoder

    torch.set_num_threads(4)
    run_protocol = json.loads((RUN / "protocol.json").read_text())
    natural_protocol = json.loads((NAT / "protocol.json").read_text())
    q = next(r for r in frozen.questions() if r["id"] == QUESTION_ID)
    ids = q["top_memory_ids"]
    assert len(ids) == len(set(ids)) == 5
    saved_texts = frozen.q10_memories()
    saved_inputs = {r["memory_id"]: r for r in json.loads((frozen.Q10 / "inputs.json").read_text())}
    memories = {r["id"]: r for r in rows(NAT / "memories.jsonl")}
    images = {r["id"]: r for r in rows(NAT / "images.jsonl")}
    inputs = []
    for row_index, mid in enumerate(ids):
        image_hash, = memories[mid]["images"]
        path = Path(images[image_hash]["path"])
        assert path.name == EXPECTED_NAMES[row_index]
        assert path.parent.name == "Home_Renovation_Interior_Design"
        assert sha(path) == image_hash
        record = saved_texts[mid]
        assert record["reader_text"] in q["reader_text"]
        dialogue, caption = record["reader_text"].split("\nMemory description: ", 1)
        unique = saved_inputs[mid]["unique_caption_or_shared_fallback"]
        assert caption == saved_inputs[mid]["full_caption"]
        assert record["retrieval_text"] == dialogue + "\nMemory description: " + unique
        inputs.append({
            "row_index": row_index, "retrieval_rank": row_index + 1,
            "memory_id": mid, "image_node_id": mid + "#image1", "filename": path.name,
            "image_path": str(path), "image_sha256": image_hash,
            "timestamp": memories[mid]["date"], "is_gold": mid in q["gold_ids"],
            "dialogue": dialogue, "full_caption": caption,
            "unique_caption_or_shared_fallback": unique,
            "dialogue_full_caption": record["reader_text"],
            "retrieval_text": record["retrieval_text"],
            "text_sha256": {
                "full_caption": text_sha(caption),
                "dialogue_full_caption": text_sha(record["reader_text"]),
                "retrieval_text": text_sha(record["retrieval_text"]),
            },
        })
    save("inputs.json", inputs)

    visual_encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                                 revision=CLIP_REVISION, device="cpu", local_files_only=True)
    image_vectors = unit([visual_encoder.encode(r["image_path"]) for r in inputs])
    visual_metadata = visual_encoder.metadata
    assert image_vectors.shape == (5, 512)
    del visual_encoder
    print("Five image representations encoded.", flush=True)

    text_model_path = Path(natural_protocol["retriever_model_path"])
    text_encoder = SentenceTransformer(str(text_model_path), device="cpu", local_files_only=True)
    texts = [r[field] for field in TEXT_VARIANTS.values() for r in inputs]
    lengths = [len(text_encoder.tokenizer.encode(t, truncation=False)) for t in texts]
    assert max(lengths) <= text_encoder.max_seq_length
    text_vectors = unit(text_encoder.encode(texts, batch_size=16, normalize_embeddings=True,
                                           show_progress_bar=False))
    assert text_vectors.shape == (15, 384)
    arrays = {"image_embeddings": image_vectors}
    token_counts = {}
    for i, key in enumerate(TEXT_VARIANTS):
        arrays[key] = text_vectors[i * 5:(i + 1) * 5]
        token_counts[key] = lengths[i * 5:(i + 1) * 5]
    assert all(np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12, rtol=0)
               for a in arrays.values())
    print("Three text representations per image encoded without truncation.", flush=True)

    historical_pairs = {
        tuple(sorted((p["earlier_id"].removesuffix("#image1"),
                      p["later_id"].removesuffix("#image1")))): float(p["cosine"])
        for p in csv.DictReader(PAIR_CSV.open())
    }
    image_errors = [
        abs(float(image_vectors[i] @ image_vectors[j]) -
            historical_pairs[tuple(sorted((ids[i], ids[j])))])
        for i, j in itertools.combinations(range(5), 2)
    ]
    assert max(image_errors) < 1e-7
    cached_retrieval = unit([saved_texts[mid]["embedding"] for mid in ids])
    retrieval_error = float(np.max(np.abs(cached_retrieval - arrays["text_dialogue_unique_caption"])))
    assert retrieval_error < 1e-5
    prior = json.loads(PAIR_RESULT.read_text())
    prior_keys = {
        "image_embeddings": prior["image_clip"]["cosine"],
        "text_full_caption": prior["text_bge"]["full_caption_190"]["cosine"],
        "text_dialogue_full_caption": prior["text_bge"]["dialogue_full_caption_190"]["cosine"],
        "text_dialogue_unique_caption": prior["text_bge"]["actual_retrieval_text_190"]["cosine"],
    }
    first_pair_errors = {k: abs(float(arrays[k][0] @ arrays[k][1]) - v)
                         for k, v in prior_keys.items()}
    assert max(first_pair_errors.values()) < 1e-6

    np.savez_compressed(
        OUT / "embeddings.npz", **arrays, memory_ids=np.asarray(ids),
        filenames=np.asarray(EXPECTED_NAMES), retrieval_ranks=np.arange(1, 6),
    )
    with np.load(OUT / "embeddings.npz", allow_pickle=False) as saved:
        assert saved["memory_ids"].tolist() == ids
        assert saved["filenames"].tolist() == EXPECTED_NAMES
        assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
    index = []
    for i, r in enumerate(inputs):
        index.append({k: r[k] for k in [
            "row_index", "retrieval_rank", "memory_id", "filename",
            "image_path", "image_sha256", "timestamp", "is_gold",
        ]} | {key + "_tokens": token_counts[key][i] for key in TEXT_VARIANTS})
    with (OUT / "index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    source_paths = [
        RUN / "protocol.json", RUN / "qa_manifest.jsonl", frozen.COMPARISON,
        RUN / "retrieval_memories.jsonl", NAT / "protocol.json", NAT / "memories.jsonl",
        NAT / "images.jsonl", frozen.Q10 / "inputs.json", frozen.Q10 / "embeddings.npz", PAIR_CSV, PAIR_RESULT,
        ROOT / "scripts/graph/visual_image_encoder.py",
    ]
    save("protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(RUN), "source_experiment_signature": run_protocol["experiment_signature"],
        "question_id": QUESTION_ID, "row_order": "Saved 190/371 run's Q10 Top1 through Top5",
        "script_sha256": sha(Path(__file__)),
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "text_variant_fields": TEXT_VARIANTS, "text_token_counts": token_counts,
        "image_encoder": visual_metadata,
        "text_encoder": {
            "model_path": str(text_model_path), "revision": natural_protocol["retriever_revision"],
            "model": "BAAI/bge-small-en-v1.5", "dimension": 384,
            "max_seq_length": text_encoder.max_seq_length, "truncated_inputs": 0,
            "pooling": json.loads((text_model_path / "1_Pooling/config.json").read_text()),
            "query_prefix": None,
            "files_sha256": {
                str(p.relative_to(text_model_path)): sha(p)
                for p in sorted(text_model_path.rglob("*"))
                if p.is_file() and p.suffix in {".json", ".bin", ".safetensors", ".txt"}
                and "onnx" not in p.parts
            },
        },
        "normalization": "float32 inference followed by float64 L2 normalization",
        "dimensionality_reduction": None,
        "scope": "Five-image experimental artifact; production memory pipeline unchanged.",
        "original_images_copied": False,
    })
    save("validation.json", {
        "status": "passed", "rows": 5, "matched_saved_top5_order": True,
        "image_hashes_match_registry": True, "texts_match_saved_reader_and_graph": True,
        "all_values_finite": True, "all_vectors_unit_norm": True,
        "no_text_truncation": True, "npz_roundtrip_exact": True,
        "historical_image_pairs_checked": 10,
        "max_historical_image_cosine_abs_error": max(image_errors),
        "max_retrieval_cached_component_abs_error": retrieval_error,
        "first_two_match_previous_experiment_abs_errors": first_pair_errors,
        "artifact_sha256": {name: sha(OUT / name) for name in [
            "embeddings.npz", "inputs.json", "index.csv", "protocol.json", "prepare.py",
        ]},
    })
    print(json.dumps({
        "output": str(OUT / "embeddings.npz"),
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        "filenames": EXPECTED_NAMES, "validation": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
