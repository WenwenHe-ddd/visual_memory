#!/usr/bin/env python3
"""Fresh paired Qwen evaluation of PCA-2D versus ten Random-2D seeds.

Only original question/options, native query images, and ordered Top5 original
dialogue plus full captions reach the reader. Identical complete payloads share
one fresh inference inside this signed experiment; historical answers are never
read or reused. Prepare and analyze require all 371 questions in all 11 arms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT)]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "8")

from scripts.analysis.evaluate_adaptive_changed_qa import digest, dump, lines, read, reader_text, sha
from scripts.analysis.evaluate_all_master_answer_reader import verify
from scripts.evaluation.paired_single_gpu_reader import run as run_reader

BASE = ROOT / "reports/memeye_qwen_random2d_ablation_20261105"
ORIGINAL = ROOT / "data/independent_visual_inputs"
REFERENCE = ROOT / "reports/all_master_answer_query_20261021/reader"
CONDITIONS = ("pca_2d",) + tuple(f"random_2d_seed_{seed}" for seed in range(10))
EXPECTED_QUESTIONS = 371
EXPECTED_RECORDS = EXPECTED_QUESTIONS * len(CONDITIONS)


def write_lines(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_signed(protocol: dict) -> None:
    unsigned = dict(protocol)
    signature = unsigned.pop("experiment_signature")
    assert digest(unsigned) == signature, "Protocol signature changed"
    assert protocol["condition_names"] == list(CONDITIONS)
    assert protocol["logical_conditions"] == EXPECTED_RECORDS
    verify(protocol)


def prepare(source: Path, output: Path, device: str) -> dict:
    source, output = source.resolve(), output.resolve()
    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError("Reader requires an explicit single GPU, e.g. cuda:0")
    if (output / "protocol.json").exists():
        protocol = read(output / "protocol.json")
        verify_signed(protocol)
        assert protocol["retrieval_source"] == str(source)
        assert protocol["placement"]["gpu_index"] == int(match.group(1))
        return protocol
    if (output / "responses.jsonl").exists():
        raise FileExistsError("Responses exist without a protocol; refusing to replace inputs")

    previous = read(REFERENCE / "protocol.json")
    unsigned = dict(previous)
    assert digest({key: value for key, value in unsigned.items() if key != "experiment_signature"}) == previous["experiment_signature"]
    settings = previous["reader_settings"]
    assert settings["reader_model"]["repo"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings["seed"] == 42 and settings["do_sample"] is False
    assert settings["dtype"] == "bfloat16" and settings["max_new_tokens"] == 16
    assert settings["batch_size_text"] == settings["batch_size_query_images"] == 1
    assert previous["attention_implementation"] == "sdpa"

    questions = lines(ORIGINAL / "qa_manifest.jsonl")
    question_by_id = {row["id"]: row for row in questions}
    assert len(questions) == len(question_by_id) == EXPECTED_QUESTIONS
    assert len({q["task"] for q in questions}) == 8
    assert sum(bool(q["query_image_paths"]) for q in questions) == 8
    memories = {m["id"]: m for m in lines(ORIGINAL / "retrieval_memories.jsonl")}
    memory_texts = {mid: row["reader_text"] for mid, row in memories.items()}
    retrieved = lines(source)
    indexed = {(r["question_id"], r["condition"]): r for r in retrieved}
    expected = {(qid, name) for qid in question_by_id for name in CONDITIONS}
    assert len(indexed) == len(retrieved) == EXPECTED_RECORDS, "Incomplete or duplicate retrieval records"
    assert set(indexed) == expected, "Conditions/questions do not match the fixed protocol"

    historical_inputs = {r["record_id"]: r for r in lines(REFERENCE / "qa_manifest.jsonl")}
    manifest, query_images = [], {}
    baseline_order_matches = 0
    baseline_payload_matches = 0
    for qi, q in enumerate(questions):
        for image_path in q["query_image_paths"]:
            query_images.setdefault(image_path, {"path": image_path, "sha256": sha(image_path)})
        images = [query_images[path] for path in q["query_image_paths"]]
        batch = []
        for name in CONDITIONS:
            row = indexed[q["id"], name]
            assert row["domain"] == q["task"]
            for key in ("question", "options", "query_image_paths", "gold_choice", "gold_ids"):
                assert row[key] == q[key], (q["id"], name, key)
            assert len(row["top5"]) == 5
            ids = [m["memory_id"] for m in row["top5"]]
            assert len(set(ids)) == 5
            assert all(mid.split("/", 1)[0] == q["task"] for mid in ids)
            for item in row["top5"]:
                original = memories[item["memory_id"]]
                for key in ("image_id", "dialogue", "caption"):
                    assert item[key] == original[key], (q["id"], name, item["memory_id"], key)
                assert math.isfinite(item["score"])
            text = reader_text(q, ids, memory_texts)
            payload = {"system": settings["reader_system_prompt"], "user_text": text, "query_images": images}
            payload_hash = digest(payload)
            if name == "pca_2d":
                old = historical_inputs[q["id"] + "::master_answer_query"]
                same_order = old["top_memory_ids"] == ids
                baseline_order_matches += same_order
                baseline_payload_matches += old["payload_sha256"] == payload_hash
                if same_order:
                    assert old["reader_text"] == text and old["payload_sha256"] == payload_hash
            batch.append({
                **{key: q[key] for key in ("id", "task", "question", "options", "gold_choice", "gold_ids", "query_image_paths")},
                "record_id": q["id"] + "::" + name,
                "condition": name,
                "top_memory_ids": ids,
                "top_image_ids": [m["image_id"] for m in row["top5"]],
                "top_query_scores": [m["score"] for m in row["top5"]],
                "reader_text": text,
                "query_images": images,
                "payload_sha256": payload_hash,
            })
        # Fixed cyclic counterbalancing, based only on source question index.
        offset = qi % len(CONDITIONS)
        manifest.extend(batch[offset:] + batch[:offset])
    assert len(manifest) == EXPECTED_RECORDS

    print(json.dumps({"stage": "verifying_pinned_model_files"}), flush=True)
    for path, expected_hash in previous["model_source_sha256"].items():
        assert sha(path) == expected_hash, path
    for weight in previous["weight_files"]:
        stat = Path(weight["path"]).stat()
        assert stat.st_size == weight["bytes"] and stat.st_mtime_ns == weight["mtime_ns"]
        assert sha(weight["path"]) == weight["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    write_lines(output / "qa_manifest.jsonl", manifest)
    source_paths = [
        source, Path(__file__), ORIGINAL / "qa_manifest.jsonl", ORIGINAL / "retrieval_memories.jsonl",
        ORIGINAL / "protocol.json", REFERENCE / "protocol.json", REFERENCE / "qa_manifest.jsonl",
        ROOT / "scripts/analysis/evaluate_adaptive_changed_qa.py",
        ROOT / "scripts/analysis/evaluate_all_master_answer_reader.py",
        ROOT / "scripts/evaluation/paired_single_gpu_reader.py", output / "qa_manifest.jsonl",
    ]
    protocol = {
        "design": "Fresh full-MemEye Qwen reader comparison of PCA-2D and Random-2D seeds 0 through 9.",
        "experiment_run_id": str(uuid.uuid4()),
        "condition_names": list(CONDITIONS),
        "questions": EXPECTED_QUESTIONS, "domains": 8, "logical_conditions": EXPECTED_RECORDS,
        "unique_reader_payloads": len({r["payload_sha256"] for r in manifest}),
        "reader_settings": settings, "attention_implementation": "sdpa",
        "placement": {"gpu_index": int(match.group(1)), "minimum_free_gib": 24,
                      "policy": "Entire BF16 model on one GPU; no CPU offload"},
        "condition_order": "Original question order; condition order cyclically rotated by question index.",
        "retrieval_source": str(source),
        "retrieval_expansion_sent_to_reader": False,
        "historical_response_reuse": False,
        "deduplication": "Only identical complete system/text/native-image payloads share one fresh inference within this experiment.",
        "evidence": "Original dialogue plus unchanged full caption in supplied Top5 order; original question and options.",
        "query_image_policy": "Original native query images, with byte hashes in each payload signature.",
        "query_images": list(query_images.values()), "n_image_questions": 8,
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "model_source_sha256": previous["model_source_sha256"], "weight_files": previous["weight_files"],
        "historical_input_check": {"pca_same_top5_order": baseline_order_matches,
                                   "pca_same_complete_payload": baseline_payload_matches,
                                   "historical_answers_read": False},
        "interpretation": "Fixed option-order full-371 paired accuracy; ten seeds summarized with sample standard deviation; not four-rotation paper EM.",
    }
    protocol["experiment_signature"] = digest(protocol)
    dump(output / "protocol.json", protocol)
    verify_signed(protocol)
    validation = {
        "passed": True, "questions": EXPECTED_QUESTIONS, "logical_records": EXPECTED_RECORDS,
        "conditions": len(CONDITIONS), "per_condition": {name: EXPECTED_QUESTIONS for name in CONDITIONS},
        "unique_payloads": protocol["unique_reader_payloads"], "original_image_questions": 8,
        "reader_prompt_identical_to_original_template": True,
        "dialogue_and_caption_exact_original_source": True,
        "question_options_labels_query_images_exact_original_source": True,
        "all_checkpoint_weight_hashes_verified": True,
        "historical_input_check": protocol["historical_input_check"],
        "candidate_statements_sent_to_reader": False,
    }
    dump(output / "validation_input.json", validation)
    print(json.dumps({"stage": "prepared", "records": EXPECTED_RECORDS,
                      "unique_payloads": protocol["unique_reader_payloads"], "output": str(output)}), flush=True)
    return protocol


def paired_statistics(rows: list[dict]) -> dict:
    n = len(rows)
    wins = sum(not r["pca_correct"] and r["random_correct"] for r in rows)
    losses = sum(r["pca_correct"] and not r["random_correct"] for r in rows)
    discordant = wins + losses
    pvalue = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.
    return {"questions": n, "wrong_to_correct": wins, "correct_to_wrong": losses,
            "net_correct": wins - losses, "delta_accuracy": (wins - losses) / n,
            "delta_percentage_points": 100 * (wins - losses) / n,
            "answer_changed": sum(r["pca_answer"] != r["random_answer"] for r in rows),
            "same_reader_payload": sum(r["same_reader_payload"] for r in rows),
            "mcnemar_exact_two_sided_p": pvalue}


def analyze(output: Path) -> dict:
    output = output.resolve()
    protocol = read(output / "protocol.json")
    verify_signed(protocol)
    manifest_rows = lines(output / "qa_manifest.jsonl")
    manifest = {r["record_id"]: r for r in manifest_rows}
    responses = lines(output / "responses.jsonl")
    by_id = {r["record_id"]: r for r in responses}
    assert len(by_id) == len(responses) == len(manifest) == EXPECTED_RECORDS, "Full-condition summary requires all 4,081 signed records"
    assert set(by_id) == set(manifest)
    fresh, per_question = {}, []
    for response in responses:
        original = manifest[response["record_id"]]
        match = re.fullmatch(r"(?:Answer\s*:\s*)?([ABCD])[.!]?", response["raw_response"].strip(), re.I)
        choice = match.group(1).upper() if match else None
        assert response["predicted_choice"] == choice
        assert response["correct"] == (choice == original["gold_choice"])
        assert response["gold_choice"] == original["gold_choice"]
        assert response["question_id"] == original["id"] and response["condition"] == original["condition"]
        assert response["payload_sha256"] == original["payload_sha256"]
        assert response["experiment_signature"] == protocol["experiment_signature"]
        assert response["run_id"] == protocol["experiment_run_id"]
        assert response["historical_response_reuse"] is False and response["batch_size"] == 1
        assert response["query_image_count"] == len(original["query_image_paths"])
        assert response["status"] == ("ok" if match else "format_error")
        if response["fresh_inference_executed_for_this_record"]:
            assert response["payload_sha256"] not in fresh
            fresh[response["payload_sha256"]] = response
        else:
            origin = by_id[response["same_run_shared_inference_record_id"]]
            assert origin["fresh_inference_executed_for_this_record"] is True
            for key in ("payload_sha256", "rendered_prompt_sha256", "raw_response", "generated_token_ids", "execution_id"):
                assert response[key] == origin[key]
        per_question.append({
            "condition": original["condition"], "question_id": original["id"], "domain": original["task"],
            "gold_choice": original["gold_choice"], "predicted_choice": choice, "correct": response["correct"],
            "status": response["status"], "query_has_image": bool(original["query_image_paths"]),
            "payload_sha256": response["payload_sha256"],
            "top5_memory_ids": "|".join(original["top_memory_ids"]),
            "fresh_inference": response["fresh_inference_executed_for_this_record"],
            "shared_from_record_id": response["same_run_shared_inference_record_id"],
        })
    assert len(fresh) == protocol["unique_reader_payloads"]
    assert set(fresh) == {r["payload_sha256"] for r in manifest_rows}

    questions = lines(ORIGINAL / "qa_manifest.jsonl")
    domains = sorted({q["task"] for q in questions})
    overall, by_domain, metric_rows, pairs, comparisons = {}, {}, [], [], {}
    for name in CONDITIONS:
        overall[name] = {}
        by_domain[name] = {}
        for domain in ["ALL", *domains]:
            qs = questions if domain == "ALL" else [q for q in questions if q["task"] == domain]
            selected = [by_id[q["id"] + "::" + name] for q in qs]
            result = {"questions": len(qs), "correct": sum(r["correct"] for r in selected),
                      "accuracy": statistics.mean(r["correct"] for r in selected),
                      "format_errors": sum(r["status"] != "ok" for r in selected),
                      "token_limit_hits": sum(r["hit_token_limit"] for r in selected)}
            metric_rows.append({"condition": name, "domain": domain, **result})
            if domain == "ALL":
                overall[name] = result
            else:
                by_domain[name][domain] = result
        overall[name]["macro_domain_accuracy"] = statistics.mean(r["accuracy"] for r in by_domain[name].values())
        if name == "pca_2d":
            continue
        these_pairs = []
        for q in questions:
            first = by_id[q["id"] + "::pca_2d"]
            second = by_id[q["id"] + "::" + name]
            pair = {"condition": name, "question_id": q["id"], "domain": q["task"],
                    "gold_choice": q["gold_choice"], "pca_answer": first["predicted_choice"],
                    "random_answer": second["predicted_choice"], "pca_correct": first["correct"],
                    "random_correct": second["correct"],
                    "same_reader_payload": first["payload_sha256"] == second["payload_sha256"]}
            if pair["same_reader_payload"]:
                assert first["raw_response"] == second["raw_response"]
            these_pairs.append(pair)
        pairs.extend(these_pairs)
        comparisons[name] = {"overall": paired_statistics(these_pairs),
                             "by_domain": {domain: paired_statistics([r for r in these_pairs if r["domain"] == domain])
                                           for domain in domains}}
    random_accuracy = [overall[name]["accuracy"] for name in CONDITIONS[1:]]
    random_correct = [overall[name]["correct"] for name in CONDITIONS[1:]]
    random_summary = {
        "seeds": list(range(10)), "n_seeds": 10, "questions_per_seed": EXPECTED_QUESTIONS,
        "correct_mean": statistics.mean(random_correct), "correct_std_sample": statistics.stdev(random_correct),
        "accuracy_mean": statistics.mean(random_accuracy), "accuracy_std_sample": statistics.stdev(random_accuracy),
        "accuracy_min": min(random_accuracy), "accuracy_max": max(random_accuracy),
        "delta_accuracy_mean_vs_pca": statistics.mean(random_accuracy) - overall["pca_2d"]["accuracy"],
        "by_domain": {domain: {
            "accuracy_mean": statistics.mean(by_domain[name][domain]["accuracy"] for name in CONDITIONS[1:]),
            "accuracy_std_sample": statistics.stdev(by_domain[name][domain]["accuracy"] for name in CONDITIONS[1:]),
        } for domain in domains},
    }
    summary = {
        "status": "complete", "questions": EXPECTED_QUESTIONS, "conditions": len(CONDITIONS),
        "logical_records": EXPECTED_RECORDS, "fresh_model_inferences": len(fresh),
        "same_run_shared_records": EXPECTED_RECORDS - len(fresh),
        "historical_answers_reused": 0, "overall": overall, "by_domain": by_domain,
        "random_2d_across_seeds": random_summary, "paired_vs_pca": comparisons,
        "format_errors": sum(r["status"] != "ok" for r in responses),
        "token_limit_hits": sum(r["hit_token_limit"] for r in responses),
        "fresh_inference_seconds": sum(r["elapsed_seconds"] for r in responses),
        "generation_executions": len({r["execution_id"] for r in responses}),
        "interpretation": protocol["interpretation"],
        "experiment_signature": protocol["experiment_signature"],
    }
    write_lines(output / "per_question.jsonl", per_question)
    write_csv(output / "per_question.csv", per_question)
    write_csv(output / "paired_vs_pca.csv", pairs)
    write_csv(output / "metrics.csv", metric_rows)
    dump(output / "summary.json", summary)
    dump(output / "validation_output.json", {
        "passed": True, "expected_records": EXPECTED_RECORDS, "actual_records": len(responses),
        "all_11_conditions_cover_exact_original_371_questions": True,
        "all_signatures_payloads_and_dedup_sources_match": True,
        "unique_fresh_inferences": len(fresh), "historical_answers_reused": 0,
        "source_files_unchanged": True, "query_images_use_original_pixels": True,
        "only_retrieved_original_memory_context_differs": True,
        "format_errors": summary["format_errors"], "token_limit_hits": summary["token_limit_hits"],
    })
    print(json.dumps({"stage": "analyzed", "pca": overall["pca_2d"],
                      "random": random_summary, "fresh_model_inferences": len(fresh)}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "retrieval_results.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "reader")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare(args.source, args.output, args.device)
    if args.stage in ("run", "all"):
        verify_signed(read(args.output / "protocol.json"))
        run_reader(args.output, analyze, args.resume)
    elif args.stage == "analyze":
        analyze(args.output)


if __name__ == "__main__":
    main()
