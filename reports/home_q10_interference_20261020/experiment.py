"""Exhaustive gold-preserving context ablations for the saved Home/Q10 reader."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "reports/home_q10_drop_top1_20261020"
QID = "Home_Renovation_Interior_Design/Q10"
DISTRACTORS = (1, 3, 4, 5)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def payload_sha(system, reader):
    return sha(json.dumps({"system": system, "user_text": reader, "query_images": []},
                          sort_keys=True, ensure_ascii=False).encode())


def prepare():
    previous = json.loads((PREVIOUS / "protocol.json").read_text())
    source = Path(previous["baseline_source"])
    baseline_protocol = json.loads((source / "protocol.json").read_text())
    batch = json.loads((PREVIOUS / "conditions.json").read_text())[0]["questions"]
    assert len(batch) == 8 and batch[0]["id"] == QID
    baseline = batch[0]
    assert not any(row["query_image_paths"] for row in batch)
    header, body = baseline["reader_text"].split("Memory 1: ", 1)
    memory_text, suffix = body.split("\n\nQuestion:\n", 1)
    suffix = "\n\nQuestion:\n" + suffix
    blocks = re.split(r"\n\nMemory \d+: ", memory_text)
    assert len(blocks) == 5 and baseline["gold_choice"] == "C"
    memories = [dict(r, rank=i, reader_text=blocks[i - 1])
                for i, r in enumerate(previous["original_top5"], 1)]
    unique, logical, by_payload = [], [], {}
    prompt_dir = OUT / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    for count in range(4, -1, -1):
        for distractors in itertools.combinations(DISTRACTORS, count):
            kept = sorted((2, *distractors))
            for numbering in ("original", "consecutive"):
                name = "keep_" + "".join(map(str, kept)) + "_" + numbering
                reader = header + "\n\n".join(
                    f"Memory {rank if numbering == 'original' else i}: {blocks[rank - 1]}"
                    for i, rank in enumerate(kept, 1)) + suffix
                digest = payload_sha(baseline_protocol["reader_system_prompt"], reader)
                if digest not in by_payload:
                    rows = copy.deepcopy(batch)
                    rows[0]["reader_text"] = reader
                    rows[0]["payload_sha256"] = digest
                    rows[0]["top_memory_ids"] = [baseline["top_memory_ids"][r - 1] for r in kept]
                    rows[0]["top_query_scores"] = [baseline["top_query_scores"][r - 1] for r in kept]
                    unique.append({"condition": name, "kept_ranks": kept, "questions": rows})
                    by_payload[digest] = name
                    (prompt_dir / f"{name}.txt").write_text(reader)
                logical.append({"condition": name, "kept_ranks": kept, "numbering": numbering,
                                "payload_sha256": digest, "inference_condition": by_payload[digest]})
    assert len(logical) == 32 and len(unique) == 28
    assert unique[0]["questions"] == batch
    dump(OUT / "conditions.json", unique)
    dump(OUT / "logical_conditions.json", logical)
    dump(OUT / "memories.json", memories)
    old = json.loads((PREVIOUS / "historical_batch_responses.json").read_text())
    dump(OUT / "historical_batch_responses.json", old)
    paths = [PREVIOUS / name for name in
             ("protocol.json", "conditions.json", "historical_batch_responses.json",
              "summary.json", "validation_independent.json")]
    paths += [source / name for name in ("protocol.json", "qa_manifest.jsonl", "qa_responses.jsonl")]
    paths += [Path(__file__), OUT / "conditions.json", OUT / "logical_conditions.json", OUT / "memories.json"]
    protocol = {
        "question_id": QID, "baseline_source": str(source),
        "baseline_signature": baseline_protocol["experiment_signature"],
        "reader_settings": previous["reader_settings"],
        "attention_implementation": "sdpa", "placement": previous["placement"],
        "batch_reconstruction": previous["batch_reconstruction"],
        "gold_rank": 2, "gold_choice": baseline["gold_choice"], "options": baseline["options"],
        "design": "Keep original Top2 gold evidence. Exhaust all 16 subsets of the four other "
                  "memories in original order; compare original and consecutive numbering. "
                  "Deduplicate identical full payloads, preserving mappings. Change Q10 only "
                  "within its original batch8; do not retrieve or backfill.",
        "logical_conditions": len(logical), "unique_inference_conditions": len(unique),
        "interpretation": "Use gold-only as an evidence-sufficiency check. Removal from full "
                          "context and addition to gold-only identify different contextual effects. "
                          "Do not assume monotonicity, a single cause, or generalization beyond Q10.",
        "source_sha256": {str(p): sha(p.read_bytes()) for p in paths},
        "model_source_sha256": {p: h for p, h in previous["source_sha256"].items()
                               if "/models/" in p},
        "weight_files": previous["weight_files"],
    }
    protocol["experiment_signature"] = sha(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode())
    dump(OUT / "protocol.json", protocol)
    print(json.dumps({"logical_conditions": len(logical), "unique_conditions": len(unique)}), flush=True)


def run():
    sys.path.insert(0, str(ROOT / ".vendor"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    protocol = json.loads((OUT / "protocol.json").read_text())
    for path, expected in {**protocol["source_sha256"], **protocol["model_source_sha256"]}.items():
        assert sha(Path(path).read_bytes()) == expected, path
    for record in protocol["weight_files"]:
        stat = Path(record["path"]).stat()
        assert stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"]
    assert not (OUT / "responses.jsonl").exists(), "Refuse to overwrite completed responses."
    settings = protocol["reader_settings"]
    conditions = json.loads((OUT / "conditions.json").read_text())
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(8)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    processor = AutoProcessor.from_pretrained(settings["reader_model_path"], local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": 65536,
                                      "longest_edge": settings["query_image_max_pixels"]}
    print("Loading the original bf16 reader with the previously verified placement.", flush=True)
    started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["reader_model_path"], dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="sdpa", device_map="auto",
        max_memory={0: "4GiB", 1: "4GiB", "cpu": "128GiB"})
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    assert input_device.type == "cuda"
    context_limit = model.config.text_config.max_position_embeddings
    dump(OUT / "generation_environment.json", {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": [torch.cuda.get_device_name(i) for i in range(2)],
        "context_limit": context_limit, "processor_size": processor.image_processor.size,
        "device_map": model.hf_device_map, "input_device": str(input_device),
        "dtype": str(model.dtype), "pid": os.getpid(), "threads": torch.get_num_threads(),
        "generation_config": model.generation_config.to_dict(),
        "load_seconds": time.monotonic() - started,
    })
    for index, condition in enumerate(conditions):
        name, batch = condition["condition"], condition["questions"]
        rendered = [processor.apply_chat_template(
            [{"role": "system", "content": settings["reader_system_prompt"]},
             {"role": "user", "content": [{"type": "text", "text": row["reader_text"]}]}],
            tokenize=False, add_generation_prompt=True) for row in batch]
        (OUT / "prompts" / f"{name}_rendered.txt").write_text(rendered[0])
        inputs = processor(text=rendered, images=None, padding=True, return_tensors="pt").to(input_device)
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        assert max(lengths) + settings["max_new_tokens"] <= context_limit
        assert inputs["input_ids"].shape[1] == 827
        started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=settings["max_new_tokens"],
                                       pad_token_id=processor.tokenizer.pad_token_id)
        tails = generated[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(tails, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elapsed = time.monotonic() - started
        for row, raw, tokens, length, prompt in zip(batch, texts, tails.tolist(), lengths, rendered):
            raw = raw.strip()
            match = re.fullmatch(r"(?:Answer\s*:\s*)?([ABCD])[.!]?", raw, re.IGNORECASE)
            eos = processor.tokenizer.eos_token_id
            emitted = next((i + 1 for i, token in enumerate(tokens) if token == eos), len(tokens))
            output = {
                "condition": name, "id": row["id"], "raw_response": raw,
                "predicted_choice": match.group(1).upper() if match else None,
                "gold_choice": row["gold_choice"],
                "correct": bool(match and match.group(1).upper() == row["gold_choice"]),
                "status": "ok" if match else "format_error", "input_tokens": int(length),
                "generated_tokens": emitted, "generated_token_ids": tokens,
                "hit_token_limit": emitted >= settings["max_new_tokens"] and eos not in tokens,
                "payload_sha256": row["payload_sha256"],
                "rendered_prompt_sha256": sha(prompt.encode()),
                "experiment_signature": protocol["experiment_signature"],
                "batch_seconds": elapsed, "padded_input_length": int(inputs["input_ids"].shape[1]),
            }
            with (OUT / "responses.jsonl").open("a") as handle:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            if row["id"] == QID:
                print(json.dumps({"completed": index + 1, "total": len(conditions),
                                  "condition": name, "answer": output["predicted_choice"],
                                  "correct": output["correct"], "seconds": elapsed}), flush=True)
        del inputs, generated, tails
    analyze()


def analyze():
    protocol = json.loads((OUT / "protocol.json").read_text())
    conditions = json.loads((OUT / "conditions.json").read_text())
    logical = json.loads((OUT / "logical_conditions.json").read_text())
    responses = [json.loads(line) for line in (OUT / "responses.jsonl").read_text().splitlines()]
    assert len(responses) == len(conditions) * 8
    assert len({(r["condition"], r["id"]) for r in responses}) == len(responses)
    lookup = {r["condition"]: r for r in responses if r["id"] == QID}
    rows = [dict(c, predicted_choice=lookup[c["inference_condition"]]["predicted_choice"],
                 correct=lookup[c["inference_condition"]]["correct"],
                 input_tokens=lookup[c["inference_condition"]]["input_tokens"]) for c in logical]
    compare = {(tuple(r["kept_ranks"]), r["numbering"]): r for r in rows}
    marginal_effects = []
    for numbering in ("original", "consecutive"):
        for rank in DISTRACTORS:
            changes = []
            others = [r for r in DISTRACTORS if r != rank]
            for n in range(4):
                for combination in itertools.combinations(others, n):
                    before = tuple(sorted((2, *combination)))
                    after = tuple(sorted((*before, rank)))
                    a, b = compare[before, numbering], compare[after, numbering]
                    changes.append({"before": list(before), "after": list(after),
                                    "before_answer": a["predicted_choice"], "after_answer": b["predicted_choice"],
                                    "correct_to_wrong": a["correct"] and not b["correct"],
                                    "wrong_to_correct": not a["correct"] and b["correct"]})
            marginal_effects.append({"rank": rank, "numbering": numbering,
                                     "correct_to_wrong": sum(c["correct_to_wrong"] for c in changes),
                                     "wrong_to_correct": sum(c["wrong_to_correct"] for c in changes),
                                     "changes": changes})
    historical = {r["id"]: r for r in json.loads((OUT / "historical_batch_responses.json").read_text())}
    baseline = [r for r in responses if r["condition"] == conditions[0]["condition"]]
    mismatches = [{"condition": r["condition"], "id": r["id"], "answer": r["raw_response"],
                   "historical": historical[r["id"]]["raw_response"]}
                  for r in responses if r["id"] != QID and r["raw_response"] != historical[r["id"]]["raw_response"]]
    summary = {
        "question_id": QID, "gold_choice": protocol["gold_choice"], "options": protocol["options"],
        "logical_conditions": len(rows), "unique_inference_conditions": len(conditions),
        "responses": len(responses), "results": rows, "marginal_effects": marginal_effects,
        "baseline_reproduced_batch": sum(r["raw_response"] == historical[r["id"]]["raw_response"] for r in baseline),
        "unchanged_question_mismatches": mismatches,
        "format_errors": sum(r["status"] != "ok" for r in responses),
        "token_limit_hits": sum(r["hit_token_limit"] for r in responses),
        "numbering_disagreements": [
            {"kept_ranks": r["kept_ranks"], "original": r["predicted_choice"],
             "consecutive": compare[tuple(r["kept_ranks"]), "consecutive"]["predicted_choice"]}
            for r in rows if r["numbering"] == "original" and
            r["predicted_choice"] != compare[tuple(r["kept_ranks"]), "consecutive"]["predicted_choice"]],
        "experiment_signature": protocol["experiment_signature"],
    }
    dump(OUT / "summary.json", summary)
    with (OUT / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "condition", "kept_ranks", "numbering", "inference_condition", "predicted_choice", "correct", "input_tokens"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "payload_sha256"} for r in rows)
    print(json.dumps({"completed": len(conditions), "format_errors": summary["format_errors"],
                      "numbering_disagreements": summary["numbering_disagreements"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.stage]()
