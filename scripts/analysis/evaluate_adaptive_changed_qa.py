#!/usr/bin/env python3
"""Paired reader test only where empirical retrieval changes ordered Top5.

Reuses the pinned local reader and original dialogue/full-caption evidence.
Both conditions run freshly with batch size one and identical mixed placement.
The results describe the selected changed-question subset, not full-domain ACC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")
BASE = ROOT / "reports/adaptive_retrieval_comparison_20261020"
SOURCE = ROOT / "data/independent_visual_inputs"
PREVIOUS = ROOT / "reports/home_q10_interference_20261020"
DEFAULT_OUTPUT = BASE / "changed_qa"
CONDITIONS = ("full_scan", "empirical_max_both")


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def reader_text(question, ordered_ids, memory_texts):
    return (
        "Retrieved dialogue and image descriptions (ranked by relevance):\n\n"
        + "\n\n".join(f"Memory {rank}: {memory_texts[mid]}"
                      for rank, mid in enumerate(ordered_ids, 1))
        + "\n\nQuestion:\n" + question["question"] + "\n\nOptions:\n"
        + "\n".join(f"{letter}. {question['options'][letter]}" for letter in "ABCD")
        + "\n\nAnswer with one letter only."
    )


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "responses.jsonl").exists():
        raise FileExistsError("Do not regenerate inputs after reader responses exist")
    source_protocol = read(SOURCE / "protocol.json")
    historical = lines(SOURCE / "qa_manifest.jsonl")
    source_reader = {r["id"]: r["reader_text"] for r in lines(SOURCE / "retrieval_memories.jsonl")}
    original = {q["id"]: q for q in historical}
    results = {}
    source_paths = [SOURCE / name for name in (
        "protocol.json", "qa_manifest.jsonl", "retrieval_memories.jsonl",
    )] + [Path(__file__), PREVIOUS / "protocol.json"]
    source_texts = {}
    for directory in sorted((BASE / "all_empirical").iterdir()):
        if not directory.is_dir():
            continue
        runs_path = directory / "runs.json"
        inputs_path = BASE / "inputs_by_domain" / directory.name / "inputs.json"
        source_paths.extend((runs_path, inputs_path))
        for leaf in read(inputs_path)["leaves"]:
            assert leaf["retrieval_text"] == source_reader[leaf["memory_id"]]
            source_texts[leaf["memory_id"]] = leaf["retrieval_text"]
        for row in read(runs_path):
            if row["L"] == 5 and row["method"] in CONDITIONS:
                key = (row["id"], row["method"])
                assert key not in results
                results[key] = row
    assert len(results) == 742 and len(source_texts) == 438
    changed = [q["id"] for q in historical
               if results[q["id"], CONDITIONS[0]]["selected_memory_ids"]
               != results[q["id"], CONDITIONS[1]]["selected_memory_ids"]]
    assert len(changed) == 21
    pairs, manifest = [], []
    for index, qid in enumerate(changed):
        question = original[qid]
        pair_rows = []
        image_records = [{"path": p, "sha256": sha(p)} for p in question["query_image_paths"]]
        for condition in CONDITIONS:
            retrieval = results[qid, condition]
            ordered_ids = retrieval["selected_memory_ids"]
            assert len(ordered_ids) == len(set(ordered_ids)) == 5
            text = reader_text(question, ordered_ids, source_texts)
            payload = {
                "system": source_protocol["reader_system_prompt"],
                "user_text": text, "query_images": image_records,
            }
            row = {key: question[key] for key in (
                "id", "task", "question", "options", "gold_choice", "gold_ids", "query_image_paths",
            )}
            row.update(condition=condition, record_id=f"{qid}::{condition}",
                       top_memory_ids=ordered_ids, top_query_scores=retrieval["selected_scores"],
                       reader_text=text, payload_sha256=digest(payload),
                       query_images=image_records)
            pair_rows.append(row)
        first, second = pair_rows
        for key in ("id", "task", "question", "options", "gold_choice", "gold_ids",
                    "query_image_paths", "query_images"):
            assert first[key] == second[key], (qid, key)
        assert first["payload_sha256"] != second["payload_sha256"]
        suffix = "\n\nQuestion:\n"
        assert first["reader_text"].split(suffix, 1)[1] == second["reader_text"].split(suffix, 1)[1]
        pairs.append({"question_id": qid, "query_has_image": bool(question["query_image_paths"]),
                      "conditions": pair_rows,
                      "changed_memory_ids": sorted(set(first["top_memory_ids"]) ^
                                                   set(second["top_memory_ids"]))})
        # Counterbalance condition order without using outcomes or QA labels.
        manifest.extend(pair_rows if index % 2 == 0 else list(reversed(pair_rows)))
    assert len(manifest) == 42 and sum(bool(r["query_image_paths"]) for r in manifest) == 2
    dump(output / "pairs.json", pairs)
    with (output / "qa_manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    previous = read(PREVIOUS / "protocol.json")
    settings = {key: source_protocol[key] for key in (
        "reader_model", "reader_model_path", "reader_system_prompt", "seed", "dtype",
        "do_sample", "max_new_tokens", "query_image_max_pixels",
    )}
    settings.update(batch_size_text=1, batch_size_query_images=1)
    model_path = Path(settings["reader_model_path"])
    source_paths += [output / "pairs.json", output / "qa_manifest.jsonl"]
    model_hashes = dict(previous["model_source_sha256"])
    for name in ("preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json"):
        model_hashes[str(model_path / name)] = sha(model_path / name)
    for path, expected in model_hashes.items():
        assert sha(path) == expected, path
    weights = []
    for old in previous["weight_files"]:
        path = Path(old["path"])
        stat = path.stat()
        assert stat.st_size == old["bytes"] and stat.st_mtime_ns == old["mtime_ns"]
        metadata = (model_path / ".cache/huggingface/download" / (path.name + ".metadata")).read_text().splitlines()
        assert metadata[0] == settings["reader_model"]["revision"]
        actual = sha(path)
        assert actual == metadata[1]
        weights.append({**old, "sha256": actual, "snapshot_revision": metadata[0]})
    protocol = {
        "design": "Fresh paired reader inference only for questions whose ordered Top5 changes.",
        "n_source_questions": 371, "n_unchanged_context_questions": 350,
        "n_paired_questions": 21, "n_payloads": 42, "n_image_questions": 1,
        "reader_settings": settings, "attention_implementation": "sdpa",
        "placement": {"device_map": "auto",
                      "max_memory": {"0": "4GiB", "1": "4GiB", "cpu": "128GiB"}},
        "condition_order": "Original manifest question order; alternating A/B then B/A.",
        "condition_names": list(CONDITIONS),
        "evidence": "Original dialogue + unchanged full caption; supplied ordered Top5 preserved.",
        "query_image_policy": "Native original query image; retrieval-only caption not appended to reader.",
        "historical_response_reuse": False,
        "changed_from_190_protocol": ["Both reader batches fixed to one", "Explicit mixed GPU/CPU placement"],
        "unchanged_between_conditions": [
            "question", "options", "query-image content", "model/checkpoint", "system prompt",
            "batch size", "dtype", "attention implementation", "placement", "seed", "generation settings",
        ],
        "interpretation": "Changed-subset paired accuracy only; not a newly measured full-371 accuracy.",
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "model_source_sha256": model_hashes, "weight_files": weights,
    }
    protocol["experiment_signature"] = digest(protocol)
    dump(output / "protocol.json", protocol)
    dump(output / "validation_input.json", {
        "passed": True, "questions": 21, "payloads": 42,
        "per_condition": {name: 21 for name in CONDITIONS}, "native_image_payloads": 2,
        "only_retrieval_context_differs_within_each_pair": True,
        "ordered_top5_matches_retrieval_outputs": True, "full_reader_text_matches_source": True,
        "checkpoint_files_match_pinned_revision_and_weight_hashes": True,
    })
    print(json.dumps({"prepared": 42, "output": str(output)}), flush=True)


def verify(protocol):
    for path, expected in {**protocol["source_sha256"], **protocol["model_source_sha256"]}.items():
        assert sha(path) == expected, path
    for weight in protocol["weight_files"]:
        stat = Path(weight["path"]).stat()
        assert stat.st_size == weight["bytes"] and stat.st_mtime_ns == weight["mtime_ns"]


def run(output):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    protocol = read(output / "protocol.json")
    verify(protocol)
    settings = protocol["reader_settings"]
    manifest = lines(output / "qa_manifest.jsonl")
    by_id = {r["record_id"]: r for r in manifest}
    responses_path = output / "responses.jsonl"
    completed = lines(responses_path) if responses_path.exists() else []
    assert len({r["record_id"] for r in completed}) == len(completed)
    for row in completed:
        assert row["experiment_signature"] == protocol["experiment_signature"]
        assert row["payload_sha256"] == by_id[row["record_id"]]["payload_sha256"]
    done = {r["record_id"] for r in completed}
    pending = [row for row in manifest if row["record_id"] not in done]
    if not pending:
        return analyze(output)
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(2)]
    assert min(free_memory) >= 6 * 1024**3, "Insufficient safe GPU headroom for mixed placement"
    processor = AutoProcessor.from_pretrained(settings["reader_model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {
        "shortest_edge": 65536, "longest_edge": settings["query_image_max_pixels"],
    }
    print("Loading pinned BF16 reader with 4 GiB per GPU and CPU offload.", flush=True)
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["reader_model_path"], dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", device_map="auto",
        max_memory={0: "4GiB", 1: "4GiB", "cpu": "128GiB"},
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    context_limit = model.config.text_config.max_position_embeddings
    assert input_device.type == "cuda"
    environment = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
        "free_gpu_bytes_before_loading": free_memory, "threads": torch.get_num_threads(),
        "device_map": model.hf_device_map, "input_device": str(input_device),
        "dtype": str(model.dtype), "context_limit": context_limit,
        "processor_size": processor.image_processor.size,
        "generation_config": model.generation_config.to_dict(),
        "pid": os.getpid(), "load_seconds": time.perf_counter() - started,
    }
    dump(output / "generation_environment.json", environment)
    for count, row in enumerate(pending, len(completed) + 1):
        content = [{"type": "image"} for _ in row["query_image_paths"]]
        content.append({"type": "text", "text": row["reader_text"]})
        rendered = processor.apply_chat_template(
            [{"role": "system", "content": settings["reader_system_prompt"]},
             {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        images = []
        for path in row["query_image_paths"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        encoded = processor(text=[rendered], images=images or None, padding=True,
                            return_tensors="pt").to(input_device)
        input_tokens = int(encoded["attention_mask"].sum())
        assert input_tokens + settings["max_new_tokens"] <= context_limit
        torch.manual_seed(settings["seed"])
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=settings["max_new_tokens"],
                pad_token_id=processor.tokenizer.pad_token_id,
            )
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
            "elapsed_seconds": time.perf_counter() - started,
            "batch_size": 1, "query_image_count": len(images),
        }
        with responses_path.open("a") as handle:
            handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            handle.flush()
        print(json.dumps({"completed": count, "total": len(manifest),
                          "question": row["id"], "condition": row["condition"],
                          "answer": choice, "seconds": response["elapsed_seconds"]}), flush=True)
        del encoded, generated, tails
        for image in images:
            image.close()
    analyze(output)


def analyze(output):
    protocol = read(output / "protocol.json")
    verify(protocol)
    manifest = {r["record_id"]: r for r in lines(output / "qa_manifest.jsonl")}
    responses = lines(output / "responses.jsonl")
    assert len(responses) == len(manifest) == 42
    indexed = {r["record_id"]: r for r in responses}
    assert len(indexed) == 42 and set(indexed) == set(manifest)
    for row in responses:
        assert row["payload_sha256"] == manifest[row["record_id"]]["payload_sha256"]
        assert row["experiment_signature"] == protocol["experiment_signature"]
        assert row["batch_size"] == 1
        assert row["query_image_count"] == len(manifest[row["record_id"]]["query_image_paths"])
    paired = []
    for pair in read(output / "pairs.json"):
        qid = pair["question_id"]
        first, second = [indexed[f"{qid}::{condition}"] for condition in CONDITIONS]
        paired.append({
            "question_id": qid, "query_has_image": pair["query_has_image"],
            "gold_choice": first["gold_choice"],
            "full_scan_answer": first["predicted_choice"], "empirical_answer": second["predicted_choice"],
            "full_scan_correct": first["correct"], "empirical_correct": second["correct"],
            "answer_changed": first["predicted_choice"] != second["predicted_choice"],
            "correct_to_wrong": first["correct"] and not second["correct"],
            "wrong_to_correct": not first["correct"] and second["correct"],
        })
    summary = {
        "scope": "21 questions selected solely because ordered Top5 differs",
        "n_questions": 21, "n_responses": 42,
        "full_scan_correct": sum(r["full_scan_correct"] for r in paired),
        "empirical_max_both_correct": sum(r["empirical_correct"] for r in paired),
        "correct_to_wrong": sum(r["correct_to_wrong"] for r in paired),
        "wrong_to_correct": sum(r["wrong_to_correct"] for r in paired),
        "answer_changed": sum(r["answer_changed"] for r in paired),
        "format_errors": sum(r["status"] != "ok" for r in responses),
        "token_limit_hits": sum(r["hit_token_limit"] for r in responses),
        "responses_per_condition": {c: sum(r["condition"] == c for r in responses) for c in CONDITIONS},
        "full_371_accuracy_measured": False,
        "paired_results": paired, "experiment_signature": protocol["experiment_signature"],
    }
    dump(output / "summary.json", summary)
    with (output / "paired_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    dump(output / "validation_output.json", {
        "passed": True, "unique_response_ids": 42, "response_payloads_match_manifest": True,
        "source_hashes_unchanged": True, "conditions_have_identical_model_and_batch_settings": True,
        "responses_per_condition": summary["responses_per_condition"],
        "format_errors": summary["format_errors"], "token_limit_hits": summary["token_limit_hits"],
    })
    print(json.dumps({k: v for k, v in summary.items() if k != "paired_results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments.output.resolve())
