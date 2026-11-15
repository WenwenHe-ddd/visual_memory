"""Isolated pair comparison using saved run inputs and freshly encoded features."""
import csv
import hashlib
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
IDS = ["Home_Renovation_Interior_Design/HOME_S13:R2",
       "Home_Renovation_Interior_Design/HOME_S7:R6"]
NAMES = ["D13_IMG_001.jpg", "D7_IMG_005.jpg"]
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    assert np.isfinite(v).all()
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    assert (norm > 0).all()
    return v / norm


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Choose a new output directory for another run.")
    import torch
    from sentence_transformers import SentenceTransformer
    from scripts.graph.visual_image_encoder import ImageEncoder

    torch.set_num_threads(4)
    natural_protocol = json.loads((NAT / "protocol.json").read_text())
    run_protocol = json.loads((RUN / "protocol.json").read_text())
    q10 = next(r for r in frozen.questions()
               if r["id"] == "Home_Renovation_Interior_Design/Q10")
    assert q10["top_memory_ids"][:2] == IDS
    texts_by_id = frozen.q10_memories()
    memories = {r["id"]: r for r in read_rows(NAT / "memories.jsonl")}
    images = {r["id"]: r for r in read_rows(NAT / "images.jsonl")}
    natural_captions = {r["id"]: r["caption"] for r in read_rows(NAT / "captions.jsonl")}
    graph_nodes = frozen.memories()
    inputs = []
    for mid, name in zip(IDS, NAMES):
        image_hash, = memories[mid]["images"]
        path = Path(images[image_hash]["path"])
        assert path.name == name and path.parent.name == "Home_Renovation_Interior_Design"
        assert sha(path) == image_hash
        row = texts_by_id[mid]
        assert row["reader_text"] in q10["reader_text"]
        dialogue, caption = row["reader_text"].split("\nMemory description: ", 1)
        assert caption == graph_nodes[mid]["caption"]
        inputs.append({
            "id": mid, "filename": name, "image_path": str(path), "image_sha256": image_hash,
            "full_caption_190": caption, "dialogue": dialogue,
            "dialogue_full_caption_190": row["reader_text"],
            "actual_retrieval_text_190": row["retrieval_text"],
            "natural_full_caption_reference": natural_captions[image_hash],
        })
        (OUT / f"{Path(name).stem}.caption_190.txt").write_text(caption + "\n")
    save("inputs.json", inputs)

    encoder = ImageEncoder(str(ROOT / "models/clip-vit-base-patch32"),
                           revision=CLIP_REV, device="cpu", local_files_only=True)
    image_vectors = unit([encoder.encode(r["image_path"]) for r in inputs])
    image_similarity = float(image_vectors[0] @ image_vectors[1])
    image_metadata = encoder.metadata
    del encoder
    matching = [p for p in csv.DictReader(PAIR_CSV.open())
                if {p["earlier_id"], p["later_id"]} == {mid + "#image1" for mid in IDS}]
    assert len(matching) == 1
    prior_image_similarity = float(matching[0]["cosine"])
    assert abs(image_similarity - prior_image_similarity) < 1e-7

    model_path = Path(natural_protocol["retriever_model_path"])
    encoder = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    variants = ["full_caption_190", "dialogue_full_caption_190",
                "actual_retrieval_text_190", "natural_full_caption_reference"]
    texts = [r[variant] for variant in variants for r in inputs]
    lengths = [len(encoder.tokenizer.encode(t, truncation=False)) for t in texts]
    assert max(lengths) <= encoder.max_seq_length
    vectors = unit(encoder.encode(texts, batch_size=8, normalize_embeddings=True,
                                  show_progress_bar=False))
    results = {
        "images_in_order": NAMES, "source_run": str(RUN),
        "image_clip": {"cosine": image_similarity, "dimension": int(image_vectors.shape[1]),
                       "prior_audit_cosine": prior_image_similarity,
                       "prior_audit_abs_error": abs(image_similarity - prior_image_similarity)},
        "text_bge": {
            variant: {"cosine": float(vectors[2*i] @ vectors[2*i+1]),
                      "dimension": int(vectors.shape[1]), "token_counts": lengths[2*i:2*i+2]}
            for i, variant in enumerate(variants)
        },
    }
    cached_retrieval = unit([texts_by_id[mid]["embedding"] for mid in IDS])
    fresh_retrieval = vectors[4:6]
    cache_max_error = float(np.max(np.abs(cached_retrieval - fresh_retrieval)))
    assert cache_max_error < 1e-5
    results["retrieval_cache_check"] = {
        "cached_pair_cosine": float(cached_retrieval[0] @ cached_retrieval[1]),
        "max_component_abs_error": cache_max_error,
    }
    with np.load(NAT / "embeddings.npz", allow_pickle=False) as z:
        natural_index = {str(mid): i for i, mid in enumerate(z["memory_ids"])}
        cached_natural = unit([z["memory"][natural_index[mid]] for mid in IDS])
    natural_max_error = float(np.max(np.abs(cached_natural - vectors[6:8])))
    assert natural_max_error < 1e-5
    results["natural_caption_cache_check"] = {
        "cached_pair_cosine": float(cached_natural[0] @ cached_natural[1]),
        "max_component_abs_error": natural_max_error,
    }
    source_paths = [
        RUN / "qa_manifest.jsonl", RUN / "retrieval_memories.jsonl", RUN / "protocol.json",
        NAT / "memories.jsonl", NAT / "images.jsonl", NAT / "captions.jsonl",
        NAT / "embeddings.npz", NAT / "protocol.json", PAIR_CSV, frozen.COMPARISON, frozen.Q10 / "inputs.json", frozen.Q10 / "embeddings.npz",
    ]
    save("protocol.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)),
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "image_encoder": image_metadata,
        "text_encoder": {
            "model_path": str(model_path), "revision": natural_protocol["retriever_revision"],
            "dimension": int(vectors.shape[1]), "max_seq_length": encoder.max_seq_length,
            "pooling": json.loads((model_path / "1_Pooling/config.json").read_text()),
            "query_prefix": None, "truncated_inputs": 0,
            "files_sha256": {str(p.relative_to(model_path)): sha(p)
                             for p in sorted(model_path.rglob("*"))
                             if p.is_file() and p.suffix in {".json", ".bin", ".safetensors", ".txt"}
                             and "onnx" not in p.parts},
        },
        "calculation": "Cosine: L2-normalize both vectors and take dot product.",
        "scope": "Similarity between the two memories, not question-to-memory retrieval scores.",
        "per_image_vectors_saved": False, "original_images_copied": False,
        "interpretation": "CLIP and BGE use different spaces; score magnitude alone does not establish which retains more detail.",
    })
    save("results.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
