#!/usr/bin/env python3
"""Fresh paired Q10 reader test for master-selected-answer query expansion.

Only original dialogue and full captions enter the reader. Expansion statements
are retrieval-only. Identical reader payloads share one fresh inference, with
explicit same-run deduplication provenance for both logical conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from scripts.analysis.evaluate_adaptive_changed_qa import (
    digest, dump, lines, read, reader_text, sha,
)

BASE = ROOT / "reports/home_q10_master_answer_query_20261021"
SOURCE = ROOT / "data/independent_visual_inputs"
PREVIOUS = ROOT / "reports/adaptive_retrieval_comparison_20261020/changed_qa/protocol.json"
CONDITIONS = ("baseline_question_only", "master_answer_query")


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "responses.jsonl").exists():
        raise FileExistsError("Refusing to replace inputs after reader responses exist")
    retrieval = read(source)
    qid = retrieval["question_id"]
    assert qid == "Home_Renovation_Interior_Design/Q10"
    original = next(row for row in lines(SOURCE / "qa_manifest.jsonl") if row["id"] == qid)
    for key in ("question", "options", "query_image_paths"):
        assert retrieval[key] == original[key], key
    assert not original["query_image_paths"]
    for key in ("gold_choice", "gold_ids"):
        assert retrieval["evaluation"][key] == original[key], key
    source_memory = {row["id"]: row["reader_text"]
                     for row in lines(SOURCE / "retrieval_memories.jsonl")}
    previous = read(PREVIOUS)
    settings = previous["reader_settings"]
    assert settings["seed"] == 42 and settings["do_sample"] is False
    assert settings["batch_size_text"] == settings["batch_size_query_images"] == 1
    rows = {row["name"]: row for row in retrieval["conditions"]}
    assert set(rows) == set(CONDITIONS) and len(retrieval["conditions"]) == 2
    manifest = []
    for condition in CONDITIONS:
        row = rows[condition]
        ids = row["top5_memory_ids"]
        assert len(ids) == len(set(ids)) == len(row["top5_scores"]) == 5
        assert [m["memory_id"] for m in row["reader_memories"]] == ids
        for memory in row["reader_memories"]:
            assert memory["reader_text"] == source_memory[memory["memory_id"]]
        context = "\n\n".join(f"Memory {i}: {source_memory[mid]}"
                              for i, mid in enumerate(ids, 1))
        assert row["reader_context"] == context
        text = reader_text(original, ids, source_memory)
        payload = {"system": settings["reader_system_prompt"],
                   "user_text": text, "query_images": []}
        manifest.append({
            **{key: original[key] for key in (
                "id", "task", "question", "options", "gold_choice", "gold_ids", "query_image_paths",
            )},
            "record_id": f"{qid}::{condition}", "condition": condition,
            "top_memory_ids": ids, "top_query_scores": row["top5_scores"],
            "reader_text": text, "payload_sha256": digest(payload),
        })
    with (output / "qa_manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    model_hashes = previous["model_source_sha256"]
    for path, expected in model_hashes.items():
        assert sha(path) == expected, path
    weights = previous["weight_files"]
    for weight in weights:
        path = Path(weight["path"])
        stat = path.stat()
        assert stat.st_size == weight["bytes"] and stat.st_mtime_ns == weight["mtime_ns"]
        assert sha(path) == weight["sha256"], path
    source_paths = [source, PREVIOUS, SOURCE / "qa_manifest.jsonl",
                    SOURCE / "retrieval_memories.jsonl", Path(__file__),
                    ROOT / "scripts/analysis/evaluate_adaptive_changed_qa.py",
                    output / "qa_manifest.jsonl"]
    protocol = {
        "design": "Fresh paired Q10 reader inference; only ordered original memory context may differ.",
        "condition_names": list(CONDITIONS), "question_id": qid,
        "logical_conditions": 2,
        "unique_reader_payloads": len({row["payload_sha256"] for row in manifest}),
        "reader_settings": settings, "attention_implementation": "sdpa",
        "placement": {"device_map": "auto", "max_memory": {"0": "4GiB", "1": "4GiB", "cpu": "128GiB"}},
        "retrieval_expansion_sent_to_reader": False,
        "leaf_distance_used": False, "historical_response_reuse": False,
        "deduplication": "Identical payloads share one fresh inference in this run; no historical answer reuse.",
        "evidence": "Original dialogue plus unchanged full caption only; original question and options.",
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "model_source_sha256": model_hashes, "weight_files": weights,
        "interpretation": "Single-question result; not domain accuracy or causal evidence of visual understanding.",
    }
    protocol["experiment_signature"] = digest(protocol)
    dump(output / "protocol.json", protocol)
    dump(output / "validation_input.json", {
        "passed": True, "logical_conditions": 2,
        "unique_reader_payloads": protocol["unique_reader_payloads"],
        "reader_memories_exact_source_text": True, "retrieval_expansion_not_in_reader": True,
        "original_question_options_and_labels_match_source": True,
        "weights_sha256_recomputed_and_match_pinned_revision": True,
    })
    print(json.dumps({"stage": "prepared", "output": str(output),
                      "unique_payloads": protocol["unique_reader_payloads"]}), flush=True)


def verify(protocol: dict) -> None:
    for path, expected in {**protocol["source_sha256"], **protocol["model_source_sha256"]}.items():
        assert sha(path) == expected, path
    for weight in protocol["weight_files"]:
        stat = Path(weight["path"]).stat()
        assert stat.st_size == weight["bytes"] and stat.st_mtime_ns == weight["mtime_ns"]


def run(output: Path) -> None:
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    protocol = read(output / "protocol.json")
    verify(protocol)
    settings = protocol["reader_settings"]
    manifest = lines(output / "qa_manifest.jsonl")
    responses_path = output / "responses.jsonl"
    if responses_path.exists():
        raise FileExistsError("Use analyze for completed runs; do not accidentally repeat or append inference")
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    free = [torch.cuda.mem_get_info(i)[0] for i in range(2)]
    assert min(free) >= 6 * 1024**3, "Insufficient safe GPU headroom for mixed placement"
    processor = AutoProcessor.from_pretrained(settings["reader_model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": 65536,
                                      "longest_edge": settings["query_image_max_pixels"]}
    print("Loading pinned BF16 reader with 4 GiB per GPU and CPU offload.", flush=True)
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["reader_model_path"], dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", device_map="auto",
        max_memory={0: "4GiB", 1: "4GiB", "cpu": "128GiB"},
    )
    model.eval()
    device = model.get_input_embeddings().weight.device
    assert device.type == "cuda"
    run_id = str(uuid.uuid4())
    dump(output / "generation_environment.json", {
        "run_id": run_id, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
        "free_gpu_bytes_before_loading": free, "threads": torch.get_num_threads(),
        "device_map": model.hf_device_map, "input_device": str(device), "dtype": str(model.dtype),
        "context_limit": model.config.text_config.max_position_embeddings,
        "processor_size": processor.image_processor.size,
        "generation_config": model.generation_config.to_dict(),
        "pid": os.getpid(), "load_seconds": time.perf_counter() - started,
    })
    cache = {}
    for row in manifest:
        previous = cache.get(row["payload_sha256"])
        if previous is not None:
            response = {**previous, "record_id": row["record_id"], "condition": row["condition"],
                        "fresh_inference_executed_for_this_record": False,
                        "same_run_shared_inference_record_id": previous["record_id"],
                        "elapsed_seconds": 0.0}
        else:
            rendered = processor.apply_chat_template([
                {"role": "system", "content": settings["reader_system_prompt"]},
                {"role": "user", "content": [{"type": "text", "text": row["reader_text"]}]},
            ], tokenize=False, add_generation_prompt=True)
            encoded = processor(text=[rendered], images=None, padding=True,
                                return_tensors="pt").to(device)
            input_tokens = int(encoded["attention_mask"].sum())
            assert input_tokens + settings["max_new_tokens"] <= model.config.text_config.max_position_embeddings
            torch.manual_seed(settings["seed"])
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False,
                    max_new_tokens=settings["max_new_tokens"], pad_token_id=processor.tokenizer.pad_token_id)
            tails = generated[:, encoded["input_ids"].shape[1]:]
            raw = processor.batch_decode(tails, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0].strip()
            tokens = tails[0].tolist()
            eos = processor.tokenizer.eos_token_id
            emitted = next((i + 1 for i, token in enumerate(tokens) if token == eos), len(tokens))
            match = re.fullmatch(r"(?:Answer\s*:\s*)?([ABCD])[.!]?", raw, re.IGNORECASE)
            choice = match.group(1).upper() if match else None
            response = {
                "record_id": row["record_id"], "question_id": row["id"], "condition": row["condition"],
                "raw_response": raw, "predicted_choice": choice, "gold_choice": row["gold_choice"],
                "correct": choice == row["gold_choice"], "status": "ok" if match else "format_error",
                "input_tokens": input_tokens, "generated_tokens": emitted, "generated_token_ids": tokens,
                "hit_token_limit": emitted >= settings["max_new_tokens"] and eos not in tokens,
                "payload_sha256": row["payload_sha256"],
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "experiment_signature": protocol["experiment_signature"], "run_id": run_id,
                "elapsed_seconds": time.perf_counter() - started,
                "batch_size": 1, "query_image_count": 0,
                "fresh_inference_executed_for_this_record": True,
                "same_run_shared_inference_record_id": None, "historical_response_reuse": False,
            }
            cache[row["payload_sha256"]] = response
            del encoded, generated, tails
        with responses_path.open("a") as handle:
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
        print(json.dumps({"condition": row["condition"], "answer": response["predicted_choice"],
                          "fresh_execution": response["fresh_inference_executed_for_this_record"],
                          "seconds": response["elapsed_seconds"]}), flush=True)
    analyze(output)


def analyze(output: Path) -> None:
    protocol = read(output / "protocol.json")
    verify(protocol)
    manifest = {row["record_id"]: row for row in lines(output / "qa_manifest.jsonl")}
    responses = lines(output / "responses.jsonl")
    assert len(responses) == len(manifest) == 2
    indexed = {row["record_id"]: row for row in responses}
    assert set(indexed) == set(manifest) and len(indexed) == 2
    assert len({row["run_id"] for row in responses}) == 1
    for row in responses:
        assert row["payload_sha256"] == manifest[row["record_id"]]["payload_sha256"]
        assert row["experiment_signature"] == protocol["experiment_signature"]
        assert row["batch_size"] == 1 and row["query_image_count"] == 0
        assert row["historical_response_reuse"] is False
        if not row["fresh_inference_executed_for_this_record"]:
            source = indexed[row["same_run_shared_inference_record_id"]]
            assert source["fresh_inference_executed_for_this_record"] is True
            for key in ("payload_sha256", "rendered_prompt_sha256", "raw_response", "generated_token_ids"):
                assert row[key] == source[key]
    assert sum(row["fresh_inference_executed_for_this_record"] for row in responses) == protocol["unique_reader_payloads"]
    summary = {
        "question_id": protocol["question_id"], "gold_choice": responses[0]["gold_choice"],
        "logical_conditions": 2, "fresh_model_inferences": protocol["unique_reader_payloads"],
        "same_reader_payload": protocol["unique_reader_payloads"] == 1,
        "historical_response_reuse": False,
        "conditions": [{key: row[key] for key in (
            "condition", "predicted_choice", "correct", "status", "input_tokens", "generated_tokens",
            "hit_token_limit", "payload_sha256", "fresh_inference_executed_for_this_record",
            "same_run_shared_inference_record_id", "elapsed_seconds",
        )} for row in responses],
        "format_errors": sum(row["status"] != "ok" for row in responses),
        "token_limit_hits": sum(row["hit_token_limit"] for row in responses),
        "interpretation": protocol["interpretation"],
        "experiment_signature": protocol["experiment_signature"],
    }
    dump(output / "summary.json", summary)
    dump(output / "validation_output.json", {
        "passed": True, "unique_response_ids": 2, "response_payloads_match_manifest": True,
        "same_run_deduplication_validated": True, "historical_response_reuse": False,
        "source_hashes_unchanged": True, "format_errors": summary["format_errors"],
        "token_limit_hits": summary["token_limit_hits"],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--source", type=Path, default=BASE / "retrieval_results.json")
    parser.add_argument("--output", type=Path, default=BASE / "reader")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    else:
        globals()[args.stage](args.output.resolve())
