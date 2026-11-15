"""Controlled reader-only Q10 ablation; no retrieval or caption regeneration."""
import argparse
import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/independent_visual_inputs"
QID = "Home_Renovation_Interior_Design/Q10"
FILENAMES = ["D13_IMG_001.jpg", "D7_IMG_005.jpg", "D7_IMG_002.jpg",
             "D7_IMG_006.jpg", "D13_IMG_002.jpg"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def payload_hash(protocol, text):
    payload = {"system": protocol["reader_system_prompt"], "user_text": text, "query_images": []}
    return digest(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def prepare():
    base = json.loads((SOURCE / "protocol.json").read_text())
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.analysis import frozen_comparison_inputs as frozen
    manifest = frozen.questions()
    memories = {r["id"]: r for r in read_jsonl(SOURCE / "retrieval_memories.jsonl")}
    comparison = json.loads(frozen.COMPARISON.read_text())
    responses = {r["id"]: r["response"] for r in comparison["records"]}
    text_questions = [q for q in manifest if not q["query_image_paths"]]
    position = next(i for i, q in enumerate(text_questions) if q["id"] == QID)
    start = position // base["batch_size_text"] * base["batch_size_text"]
    batch = text_questions[start:start + base["batch_size_text"]]
    q = next(q for q in batch if q["id"] == QID)
    prefix, body = q["reader_text"].split("Memory 1: ", 1)
    memory_body, suffix = body.split("\n\nQuestion:\n", 1)
    blocks = re.split(r"\n\nMemory \d+: ", memory_body)
    assert len(blocks) == 5
    assert blocks == [memories[mid]["reader_text"] for mid in q["top_memory_ids"]]
    assert payload_hash(base, q["reader_text"]) == q["payload_sha256"]
    assert q["top_memory_ids"][0] == "Home_Renovation_Interior_Design/HOME_S13:R2"
    assert q["gold_choice"] == "C" and responses[QID]["predicted_choice"] == "A"
    suffix = "\n\nQuestion:\n" + suffix
    texts = {
        "baseline_top5": q["reader_text"],
        "drop_top1_original_numbering": prefix + "\n\n".join(
            f"Memory {i}: {block}" for i, block in enumerate(blocks[1:], 2)) + suffix,
        "drop_top1_renumbered": prefix + "\n\n".join(
            f"Memory {i}: {block}" for i, block in enumerate(blocks[1:], 1)) + suffix,
    }
    assert texts["drop_top1_original_numbering"] == q["reader_text"].replace(
        "Memory 1: " + blocks[0] + "\n\n", "", 1)
    conditions = []
    for name, text in texts.items():
        current = copy.deepcopy(batch)
        target = next(row for row in current if row["id"] == QID)
        target["reader_text"] = text
        target["payload_sha256"] = payload_hash(base, text)
        if name != "baseline_top5":
            target["top_memory_ids"] = target["top_memory_ids"][1:]
            if "top_query_scores" in target:
                target["top_query_scores"] = target["top_query_scores"][1:]
        for row in current:
            assert payload_hash(base, row["reader_text"]) == row["payload_sha256"]
        (OUT / f"{name}.txt").write_text(text)
        (OUT / f"{name}.diff").write_text("".join(difflib.unified_diff(
            q["reader_text"].splitlines(True), text.splitlines(True),
            fromfile="original_top5", tofile=name)))
        conditions.append({"condition": name, "questions": current})
    dump("conditions.json", conditions)
    dump("historical_batch_responses.json", [responses[row["id"]] for row in batch])
    dump("removed_memory.json", {
        "id": q["top_memory_ids"][0], "filename": FILENAMES[0], "reader_text": blocks[0]})
    model_path = Path(base["reader_model_path"])
    sources = [SOURCE / name for name in (
        "protocol.json", "qa_manifest.jsonl", "retrieval_memories.jsonl")]
    sources.append(frozen.COMPARISON)
    sources += [Path(__file__), OUT / "conditions.json"]
    sources += [model_path / name for name in (
        "config.json", "generation_config.json", "chat_template.json",
        "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json")]
    protocol = {
        "question_id": QID,
        "baseline_signature": base["experiment_signature"],
        "baseline_source": str(SOURCE),
        "reader_settings": {k: base[k] for k in (
            "reader_model", "reader_model_path", "reader_system_prompt", "seed",
            "dtype", "do_sample", "max_new_tokens", "batch_size_text", "query_image_max_pixels")},
        "attention_implementation": "sdpa",
        "placement": {"device_map": "auto", "max_memory": {"0": "4GiB", "1": "4GiB", "cpu": "128GiB"}},
        "placement_note": "CPU offload is needed because both host GPUs have other active workloads. "
                          "All conditions share this placement, exact weights and bf16 precision.",
        "batch_reconstruction": {
            "rule": "Filter original manifest to text-only questions; preserve order; take original batch of 8.",
            "original_text_index": position, "batch_start": start,
            "question_ids": [row["id"] for row in batch],
        },
        "original_top5": [{"id": mid, "filename": filename}
                          for mid, filename in zip(q["top_memory_ids"], FILENAMES)],
        "intervention": "Remove whole Top1 dialogue plus full caption, retain remaining order, "
                        "do not backfill from Top6. Check original and consecutive memory numbering.",
        "condition_names": list(texts),
        "gold_choice": q["gold_choice"], "options": q["options"],
        "source_sha256": {str(p): digest(p.read_bytes()) for p in sources},
        "weight_files": [{"path": str(p), "bytes": p.stat().st_size,
                          "mtime_ns": p.stat().st_mtime_ns}
                         for p in sorted(model_path.glob("*.safetensors"))],
    }
    protocol["experiment_signature"] = digest(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode())
    dump("protocol.json", protocol)
    print(json.dumps({"prepared": list(texts), "batch_ids": protocol["batch_reconstruction"]["question_ids"]}), flush=True)


def run():
    sys.path.insert(0, str(ROOT / ".vendor"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in protocol["source_sha256"].items():
        assert digest(Path(path).read_bytes()) == expected, f"Source changed: {path}"
    assert not (OUT / "responses.jsonl").exists(), "Do not overwrite completed inference."
    settings = protocol["reader_settings"]
    conditions = json.loads((OUT / "conditions.json").read_text())
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    processor = AutoProcessor.from_pretrained(settings["reader_model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": 65536,
                                      "longest_edge": settings["query_image_max_pixels"]}
    print("Loading original bf16 Qwen model with CPU/GPU offload.", flush=True)
    loaded_at = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["reader_model_path"], dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map="auto", max_memory={0: "4GiB", 1: "4GiB", "cpu": "128GiB"})
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    assert input_device.type == "cuda"
    context_limit = model.config.text_config.max_position_embeddings
    environment = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "context_limit": context_limit, "processor_size": processor.image_processor.size,
        "device_map": model.hf_device_map, "input_device": str(input_device),
        "dtype": str(model.dtype), "pid": os.getpid(), "threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "generation_config": model.generation_config.to_dict(),
        "load_seconds": time.monotonic() - loaded_at,
    }
    dump("generation_environment.json", environment)
    print(f"Loaded in {environment['load_seconds']:.1f}s; input_device={input_device}", flush=True)
    outputs = []
    for condition in conditions:
        name, batch = condition["condition"], condition["questions"]
        rendered = []
        for row in batch:
            assert not row["query_image_paths"]
            messages = [{"role": "system", "content": settings["reader_system_prompt"]},
                        {"role": "user", "content": [{"type": "text", "text": row["reader_text"]}]}]
            rendered.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        target_index = next(i for i, q in enumerate(batch) if q["id"] == QID)
        (OUT / f"{name}_rendered.txt").write_text(rendered[target_index])
        inputs = processor(text=rendered, images=None, padding=True, return_tensors="pt").to(input_device)
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        assert max(lengths) + settings["max_new_tokens"] <= context_limit
        if name == "baseline_top5":
            assert lengths[target_index] == 792
        print(json.dumps({"starting": name, "input_tokens": lengths, "padded_length": inputs["input_ids"].shape[1]}), flush=True)
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"],
                pad_token_id=processor.tokenizer.pad_token_id)
        tails = generated[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elapsed = time.monotonic() - started
        for row, raw, tokens, length in zip(batch, texts, tails.tolist(), lengths):
            raw = raw.strip()
            match = re.fullmatch(r"(?:Answer\s*:\s*)?([ABCD])[.!]?", raw, re.IGNORECASE)
            eos = processor.tokenizer.eos_token_id
            emitted = next((i + 1 for i, token in enumerate(tokens) if token == eos), len(tokens))
            output = {
                "condition": name, "id": row["id"], "raw_response": raw,
                "predicted_choice": match.group(1).upper() if match else None,
                "gold_choice": row["gold_choice"], "correct": bool(match and match.group(1).upper() == row["gold_choice"]),
                "status": "ok" if match else "format_error", "input_tokens": int(length),
                "generated_tokens": emitted, "generated_token_ids": tokens,
                "hit_token_limit": emitted >= settings["max_new_tokens"] and eos not in tokens,
                "payload_sha256": row["payload_sha256"],
                "rendered_prompt_sha256": digest(rendered[batch.index(row)].encode()),
                "experiment_signature": protocol["experiment_signature"],
                "batch_seconds": elapsed, "padded_input_length": int(inputs["input_ids"].shape[1]),
            }
            outputs.append(output)
            with (OUT / "responses.jsonl").open("a") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            if row["id"] == QID:
                print(json.dumps(output, ensure_ascii=False), flush=True)
        del inputs, generated, tails
    old = {r["id"]: r for r in json.loads((OUT / "historical_batch_responses.json").read_text())}
    q10 = [row for row in outputs if row["id"] == QID]
    baseline = [row for row in outputs if row["condition"] == "baseline_top5"]
    summary = {
        "question_id": QID, "question": conditions[0]["questions"][target_index]["question"],
        "gold_choice": protocol["gold_choice"], "options": protocol["options"],
        "historical_q10": old[QID], "q10_results": q10,
        "baseline_reproduced_q10": q10[0]["predicted_choice"] == old[QID]["predicted_choice"],
        "baseline_reproduced_batch": sum(row["raw_response"] == old[row["id"]]["raw_response"] for row in baseline),
        "baseline_batch_size": len(baseline),
        "response_count": len(outputs),
        "format_errors": sum(row["status"] != "ok" for row in outputs),
        "token_limit_hits": sum(row["hit_token_limit"] for row in outputs),
        "remaining_images": FILENAMES[1:],
        "experiment_signature": protocol["experiment_signature"],
    }
    dump("summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage]()
