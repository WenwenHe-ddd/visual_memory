#!/usr/bin/env python3
"""Fresh paired full-371 reader test of master-answer query expansion.

Retrieval hypotheses never enter reader evidence. Identical ordered reader
payloads share one fresh inference within this experiment, with explicit hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

BASE = ROOT / "reports/all_master_answer_query_20261021"
SOURCE = ROOT / "data/independent_visual_inputs"
PREVIOUS = ROOT / "reports/home_q10_master_answer_query_20261021/reader/protocol.json"
CONDITIONS = ("baseline_question_only", "master_answer_query")


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "responses.jsonl").exists():
        raise FileExistsError("Refusing to replace inputs after reader responses exist")
    retrieved = lines(source)
    by_qid = {r["question_id"]: r for r in retrieved}
    originals = lines(SOURCE / "qa_manifest.jsonl")
    assert len(by_qid) == len(retrieved) == len(originals) == 371
    assert set(by_qid) == {q["id"] for q in originals}
    assert len({q["task"] for q in originals}) == 8
    source_memory = {r["id"]: r["reader_text"]
                     for r in lines(SOURCE / "retrieval_memories.jsonl")}
    previous = read(PREVIOUS)
    settings = previous["reader_settings"]
    assert settings["seed"] == 42 and settings["do_sample"] is False
    assert settings["max_new_tokens"] == 16
    assert settings["batch_size_text"] == settings["batch_size_query_images"] == 1
    manifest, pairs, image_records_by_path = [], [], {}
    for index, original in enumerate(originals):
        qid = original["id"]
        retrieval = by_qid[qid]
        for key in ("question", "options", "query_image_paths"):
            assert retrieval[key] == original[key], (qid, key)
        for key in ("gold_choice", "gold_ids"):
            assert retrieval["evaluation"][key] == original[key], (qid, key)
        for path in original["query_image_paths"]:
            image_records_by_path.setdefault(path, {"path": path, "sha256": sha(path)})
        image_records = [image_records_by_path[p] for p in original["query_image_paths"]]
        rows = {r["name"]: r for r in retrieval["conditions"]}
        assert set(rows) == set(CONDITIONS) and len(retrieval["conditions"]) == 2
        pair = []
        for condition in CONDITIONS:
            row = rows[condition]
            ids = row["top5_memory_ids"]
            assert len(ids) == len(set(ids)) == len(row["top5_scores"]) == 5
            assert all(mid.split("/", 1)[0] == original["task"] for mid in ids)
            assert [m["memory_id"] for m in row["reader_memories"]] == ids
            for memory in row["reader_memories"]:
                assert memory["reader_text"] == source_memory[memory["memory_id"]]
            context = "\n\n".join(f"Memory {i}: {source_memory[mid]}"
                                  for i, mid in enumerate(ids, 1))
            assert row["reader_context"] == context
            text = reader_text(original, ids, source_memory)
            payload = {"system": settings["reader_system_prompt"],
                       "user_text": text, "query_images": image_records}
            pair.append({
                **{key: original[key] for key in (
                    "id", "task", "question", "options", "gold_choice", "gold_ids", "query_image_paths",
                )},
                "record_id": f"{qid}::{condition}", "condition": condition,
                "top_memory_ids": ids, "top_query_scores": row["top5_scores"],
                "reader_text": text, "payload_sha256": digest(payload),
                "query_images": image_records,
            })
        first, second = pair
        for key in ("id", "task", "question", "options", "gold_choice", "gold_ids",
                    "query_image_paths", "query_images"):
            assert first[key] == second[key], (qid, key)
        assert first["reader_text"].split("\n\nQuestion:\n", 1)[1] == second["reader_text"].split("\n\nQuestion:\n", 1)[1]
        pairs.append({
            "question_id": qid, "domain": original["task"],
            "query_has_image": bool(original["query_image_paths"]),
            "same_reader_payload": first["payload_sha256"] == second["payload_sha256"],
            "same_top5_order": first["top_memory_ids"] == second["top_memory_ids"],
            "same_top5_set": set(first["top_memory_ids"]) == set(second["top_memory_ids"]),
            "conditions": pair,
        })
        manifest.extend(pair if index % 2 == 0 else list(reversed(pair)))
    assert len(manifest) == 742
    assert sum(bool(q["query_image_paths"]) for q in originals) == 8
    dump(output / "pairs.json", pairs)
    with (output / "qa_manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for path, expected in previous["model_source_sha256"].items():
        assert sha(path) == expected, path
    for weight in previous["weight_files"]:
        path = Path(weight["path"])
        stat = path.stat()
        assert stat.st_size == weight["bytes"] and stat.st_mtime_ns == weight["mtime_ns"]
        assert sha(path) == weight["sha256"], path
    source_paths = [source, PREVIOUS, SOURCE / "qa_manifest.jsonl", SOURCE / "protocol.json",
                    SOURCE / "retrieval_memories.jsonl", Path(__file__),
                    ROOT / "scripts/analysis/evaluate_adaptive_changed_qa.py",
                    output / "qa_manifest.jsonl", output / "pairs.json"]
    protocol = {
        "design": "Fresh paired full-371 reader inference; only ordered original memory context differs.",
        "experiment_run_id": str(uuid.uuid4()), "condition_names": list(CONDITIONS),
        "questions": len(originals), "domains": 8, "logical_conditions": len(manifest),
        "unique_reader_payloads": len({r["payload_sha256"] for r in manifest}),
        "n_same_payload_pairs": sum(p["same_reader_payload"] for p in pairs),
        "n_image_questions": 8, "reader_settings": settings,
        "attention_implementation": "sdpa",
        "placement": {"device_map": "auto", "max_memory": {"0": "4GiB", "1": "4GiB", "cpu": "128GiB"}},
        "condition_order": "Original manifest question order; alternating A/B then B/A.",
        "retrieval_expansion_sent_to_reader": False, "leaf_distance_used": False,
        "historical_response_reuse": False,
        "deduplication": "Identical payloads share one fresh inference within this experiment, no historical answers.",
        "evidence": "Original dialogue plus unchanged full caption; original question and options.",
        "query_image_policy": "Native original query images; retrieval-only query-image captions are not reader text.",
        "query_images": list(image_records_by_path.values()),
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "model_source_sha256": previous["model_source_sha256"], "weight_files": previous["weight_files"],
        "interpretation": "Fixed option-order local-Qwen paired accuracy, not four-rotation paper EM.",
    }
    protocol["experiment_signature"] = digest(protocol)
    dump(output / "protocol.json", protocol)
    dump(output / "validation_input.json", {
        "passed": True, "questions": 371, "logical_conditions": 742,
        "unique_reader_payloads": protocol["unique_reader_payloads"], "image_questions": 8,
        "reader_memories_exact_source_text": True, "retrieval_expansion_not_in_reader": True,
        "original_question_options_labels_images_match_source": True,
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
    for image in protocol["query_images"]:
        assert sha(image["path"]) == image["sha256"]


def run(output: Path, resume: bool = False, analyze_fn=None) -> None:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    analyze_results = analyze if analyze_fn is None else analyze_fn
    protocol = read(output / "protocol.json")
    verify(protocol)
    settings = protocol["reader_settings"]
    manifest = lines(output / "qa_manifest.jsonl")
    by_id = {r["record_id"]: r for r in manifest}
    responses_path = output / "responses.jsonl"
    if responses_path.exists() and not resume:
        raise FileExistsError("Use --resume for interrupted same-experiment inference or --stage analyze")
    completed = lines(responses_path) if responses_path.exists() else []
    assert len({r["record_id"] for r in completed}) == len(completed)
    cache = {}
    for row in completed:
        assert row["experiment_signature"] == protocol["experiment_signature"]
        assert row["run_id"] == protocol["experiment_run_id"]
        assert row["payload_sha256"] == by_id[row["record_id"]]["payload_sha256"]
        if row["fresh_inference_executed_for_this_record"]:
            assert row["payload_sha256"] not in cache
            cache[row["payload_sha256"]] = row
    done = {r["record_id"] for r in completed}
    pending = [r for r in manifest if r["record_id"] not in done]
    if not pending:
        analyze_results(output)
        return
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
    execution_id = str(uuid.uuid4())
    environment = {
        "run_id": protocol["experiment_run_id"], "execution_id": execution_id,
        "resumed": resume, "already_completed_records": len(completed),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
        "free_gpu_bytes_before_loading": free, "threads": torch.get_num_threads(),
        "device_map": model.hf_device_map, "input_device": str(device), "dtype": str(model.dtype),
        "context_limit": model.config.text_config.max_position_embeddings,
        "processor_size": processor.image_processor.size,
        "generation_config": model.generation_config.to_dict(),
        "pid": os.getpid(), "load_seconds": time.perf_counter() - started,
    }
    dump(output / f"generation_environment_{execution_id}.json", environment)
    if not completed:
        dump(output / "generation_environment.json", environment)
    for count, row in enumerate(pending, len(completed) + 1):
        previous = cache.get(row["payload_sha256"])
        if previous is not None:
            response = {**previous, "record_id": row["record_id"], "condition": row["condition"],
                        "question_id": row["id"], "gold_choice": row["gold_choice"],
                        "correct": previous["predicted_choice"] == row["gold_choice"],
                        "fresh_inference_executed_for_this_record": False,
                        "same_run_shared_inference_record_id": previous["record_id"],
                        "elapsed_seconds": 0.0}
        else:
            content = [{"type": "image"} for _ in row["query_image_paths"]]
            content.append({"type": "text", "text": row["reader_text"]})
            rendered = processor.apply_chat_template([
                {"role": "system", "content": settings["reader_system_prompt"]},
                {"role": "user", "content": content},
            ], tokenize=False, add_generation_prompt=True)
            images = []
            for path in row["query_image_paths"]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(text=[rendered], images=images or None, padding=True,
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
                "experiment_signature": protocol["experiment_signature"],
                "run_id": protocol["experiment_run_id"], "execution_id": execution_id,
                "elapsed_seconds": time.perf_counter() - started,
                "batch_size": 1, "query_image_count": len(images),
                "fresh_inference_executed_for_this_record": True,
                "same_run_shared_inference_record_id": None, "historical_response_reuse": False,
            }
            cache[row["payload_sha256"]] = response
            del encoded, generated, tails
            for image in images:
                image.close()
        with responses_path.open("a") as handle:
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
        print(json.dumps({"completed": count, "total": len(manifest), "question": row["id"],
                          "condition": row["condition"], "answer": response["predicted_choice"],
                          "fresh_execution": response["fresh_inference_executed_for_this_record"],
                          "seconds": response["elapsed_seconds"]}), flush=True)
    analyze_results(output)


def aggregate(pairs: list[dict]) -> dict:
    n = len(pairs)
    baseline = sum(r["baseline_correct"] for r in pairs)
    expanded = sum(r["expanded_correct"] for r in pairs)
    wins = sum(not r["baseline_correct"] and r["expanded_correct"] for r in pairs)
    losses = sum(r["baseline_correct"] and not r["expanded_correct"] for r in pairs)
    discordant = wins + losses
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {"questions": n, "baseline_correct": baseline, "expanded_correct": expanded,
            "baseline_accuracy": baseline / n if n else None,
            "expanded_accuracy": expanded / n if n else None,
            "delta_percentage_points": 100 * (expanded - baseline) / n if n else None,
            "wrong_to_correct": wins, "correct_to_wrong": losses,
            "answer_changed": sum(r["baseline_answer"] != r["expanded_answer"] for r in pairs),
            "same_reader_payload": sum(r["same_reader_payload"] for r in pairs),
            "same_top5_order": sum(r["same_top5_order"] for r in pairs),
            "same_top5_set": sum(r["same_top5_set"] for r in pairs),
            "mcnemar_exact_two_sided_p": pvalue}


def analyze(output: Path) -> None:
    protocol = read(output / "protocol.json")
    verify(protocol)
    manifest = {r["record_id"]: r for r in lines(output / "qa_manifest.jsonl")}
    responses = lines(output / "responses.jsonl")
    assert len(responses) == len(manifest) == 742
    indexed = {r["record_id"]: r for r in responses}
    assert set(indexed) == set(manifest) and len(indexed) == 742
    assert {r["run_id"] for r in responses} == {protocol["experiment_run_id"]}
    for row in responses:
        original = manifest[row["record_id"]]
        assert row["payload_sha256"] == original["payload_sha256"]
        assert row["experiment_signature"] == protocol["experiment_signature"]
        assert row["gold_choice"] == original["gold_choice"]
        assert row["correct"] == (row["predicted_choice"] == original["gold_choice"])
        assert row["batch_size"] == 1 and row["query_image_count"] == len(original["query_image_paths"])
        assert row["historical_response_reuse"] is False
        if not row["fresh_inference_executed_for_this_record"]:
            source = indexed[row["same_run_shared_inference_record_id"]]
            assert source["fresh_inference_executed_for_this_record"] is True
            for key in ("payload_sha256", "rendered_prompt_sha256", "raw_response", "generated_token_ids", "execution_id"):
                assert row[key] == source[key]
    assert sum(r["fresh_inference_executed_for_this_record"] for r in responses) == protocol["unique_reader_payloads"]
    paired = []
    for pair in read(output / "pairs.json"):
        qid = pair["question_id"]
        first, second = [indexed[f"{qid}::{name}"] for name in CONDITIONS]
        if pair["same_reader_payload"]:
            assert first["raw_response"] == second["raw_response"]
        paired.append({
            **{key: pair[key] for key in ("question_id", "domain", "query_has_image", "same_reader_payload", "same_top5_order", "same_top5_set")},
            "gold_choice": first["gold_choice"], "baseline_answer": first["predicted_choice"],
            "expanded_answer": second["predicted_choice"], "baseline_correct": first["correct"],
            "expanded_correct": second["correct"],
            "baseline_status": first["status"], "expanded_status": second["status"],
        })
    domains = sorted({r["domain"] for r in paired})
    summary = {
        "overall": aggregate(paired),
        "by_domain": {domain: aggregate([r for r in paired if r["domain"] == domain]) for domain in domains},
        "by_query_modality": {name: aggregate([r for r in paired if r["query_has_image"] == value])
                              for name, value in (("text_only", False), ("with_query_image", True))},
        "conditions": {name: {"questions": 371, "correct": sum(r["correct"] for r in responses if r["condition"] == name),
            "format_errors": sum(r["status"] != "ok" for r in responses if r["condition"] == name),
            "token_limit_hits": sum(r["hit_token_limit"] for r in responses if r["condition"] == name)} for name in CONDITIONS},
        "logical_conditions": 742, "fresh_model_inferences": protocol["unique_reader_payloads"],
        "same_run_shared_records": 742 - protocol["unique_reader_payloads"],
        "generation_executions": len({r["execution_id"] for r in responses}),
        "historical_response_reuse": False, "format_errors": sum(r["status"] != "ok" for r in responses),
        "token_limit_hits": sum(r["hit_token_limit"] for r in responses),
        "fresh_inference_seconds": sum(r["elapsed_seconds"] for r in responses),
        "interpretation": protocol["interpretation"],
    }
    dump(output / "paired_results.json", paired)
    with (output / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    dump(output / "summary.json", summary)
    dump(output / "validation_output.json", {
        "passed": True, "expected_records": 742, "actual_records": len(responses),
        "all_ids_hashes_signatures_and_same_run_reuse_match": True,
        "query_images_use_original_pixels": True, "reader_evidence_matches_original_source": True,
        "only_retrieval_context_differs_between_conditions": True,
        "source_files_unchanged": True, "historical_response_reuse": False,
        "format_errors": summary["format_errors"], "token_limit_hits": summary["token_limit_hits"],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "retrieval_results.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "reader")
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.source, args.output)
    if args.stage in ("run", "all"):
        run(args.output, args.resume)
    if args.stage == "analyze":
        analyze(args.output)


if __name__ == "__main__":
    main()
