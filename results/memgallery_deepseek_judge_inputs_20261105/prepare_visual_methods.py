"""Freeze the two existing DeepSeek visual-method answers for LLM-Judge.

This script is read-only with respect to benchmark sources and makes no API calls.
Original empty answers and provider rejections are retained in all 1,711 rows.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.mem_gallery_master_reader import score_answer

MODEL = "deepseek-v4.1-flash"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(path):
    data = rows(path)
    result = {row["id"]: row for row in data}
    assert len(result) == len(data) == 1711, path
    return result


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    questions_path = ROOT / "results/mem_gallery_master_answer_20261021/inputs/questions.jsonl"
    gold_path = ROOT / "results/mem_gallery_master_answer_20261021/inputs/gold.jsonl"
    canonical_path = ROOT / "results/memgallery_qwen_semrag_deepseek_judge_20261104/requests.jsonl"
    questions, gold, canonical = (keyed(path) for path in [questions_path, gold_path, canonical_path])
    assert set(questions) == set(gold) == set(canonical)
    old = json.loads((ROOT / "results/memgallery_gpt5_judge_inputs_20261104/visual_methods_manifest.json").read_text())
    templates = {item["id"]: item for item in old["methods"]}
    specs = [
        ("clip_image_dialogue", "results/mem_gallery_clip_image_dialogue_api_20261102/deepseek-v4.1-flash", 656, 13),
        ("hippomm", "results/hippomm_api_memgallery_20261104/deepseek/online", 427, 14),
    ]
    methods, reports = [], []
    for mid, relative_root, expected_correct, expected_empty in specs:
        source = ROOT / relative_root
        response_path, scored_path = source / "qa_responses.jsonl", source / "per_question.jsonl"
        source_question_path = source / ("qa_manifest.jsonl" if mid == "clip_image_dialogue" else "questions.jsonl")
        response_map, scored_map, source_questions = (keyed(path) for path in [response_path, scored_path, source_question_path])
        assert set(response_map) == set(scored_map) == set(source_questions) == set(canonical)
        summary = json.loads((source / "summary.json").read_text())
        protocol = json.loads((source / "protocol.json").read_text())
        assert summary["model"] == MODEL
        if mid == "clip_image_dialogue":
            overall = summary["overall"]
            assert summary["status"] == "complete"
            assert protocol["model"] == protocol["api"]["model"] == MODEL
            assert protocol["top_k"] == 5
            request_path = source / "request_manifest.jsonl"
            request_map = keyed(request_path)
            assert set(request_map) == set(canonical)
        else:
            overall = summary
            assert protocol["contract"]["model"] == MODEL
            assert protocol["contract"]["top_k"] == 5
        records, metrics, empty_ids = [], [], []
        models, statuses = Counter(), Counter()
        known_token_limits, token_limit_hits = 0, 0
        for qid in canonical:
            q, g, canon, original_q = questions[qid], gold[qid], canonical[qid], source_questions[qid]
            response, scored = response_map[qid], scored_map[qid]
            for field in ["task", "point", "question"]:
                assert q[field] == canon[field] == original_q[field], (mid, qid, field)
            assert g["answer"] == canon["ground_truth"]
            answer = response["raw_response"] if mid == "clip_image_dialogue" else response["answer"]
            assert isinstance(answer, str)
            status = response.get("status", "ok")
            assert status == scored["status"]
            if mid == "clip_image_dialogue":
                assert scored["prediction"] == answer and scored["answer"] == g["answer"]
                assert response["request_sha256"] == request_map[qid]["request_sha256"]
                assert response["source_payload_sha256"] == original_q["payload_sha256"] == request_map[qid]["source_payload_sha256"]
                returned_model = response.get("returned_model")
                if status == "ok":
                    assert returned_model == MODEL
                else:
                    assert status == "provider_rejection" and returned_model is None and answer == ""
                models[str(returned_model)] += 1
            else:
                assert response["model"] == MODEL and response["backbone"] == "deepseek"
                models[response["model"]] += 1
            statuses[status] += 1
            if not answer.strip():
                empty_ids.append(qid)
            recomputed = score_answer(answer, g["answer"])
            for metric in ["em", "f1", "bleu_1"]:
                assert abs(recomputed[metric] - scored[metric]) < 1e-12, (mid, qid, metric)
            metrics.append(recomputed)
            record = {"id": qid, "task": q["task"], "point": q["point"],
                      "question": q["question"], "ground_truth": g["answer"],
                      "model_output": answer, "em": int(recomputed["em"]),
                      "source_record_id": qid, "answer_model": MODEL,
                      "source_status": status}
            for source_key, output_key in [
                ("returned_model", "source_returned_model"),
                ("finish_reason", "source_finish_reason"),
                ("request_sha256", "source_request_sha256"),
                ("response_id", "source_response_id"),
            ]:
                if source_key in response:
                    record[output_key] = response[source_key]
            if "hit_token_limit" in response or "hit_token_limit" in scored:
                value = response.get("hit_token_limit", scored.get("hit_token_limit"))
                assert isinstance(value, bool)
                record["source_hit_token_limit"] = value
                known_token_limits += 1
                token_limit_hits += value
            records.append(record)
        means = {name: sum(row[name] for row in metrics) / 1711 for name in ["em", "f1", "bleu_1"]}
        correct = sum(row["em"] for row in records)
        assert correct == expected_correct
        assert len(empty_ids) == expected_empty == statuses["provider_rejection"]
        assert overall["questions"] == 1711
        assert overall.get("correct", overall.get("exact_match_correct")) == correct
        for metric, value in means.items():
            assert abs(value - overall[metric]) < 1e-12, (mid, metric)
        output = HERE / f"{mid}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        source_files = [canonical_path, questions_path, gold_path, response_path, scored_path,
                        source / "summary.json", source / "protocol.json", source_question_path,
                        Path(__file__)]
        if mid == "clip_image_dialogue":
            source_files.append(request_path)
        method = {key: templates[mid][key] for key in ["id", "label", "display_label", "code_method", "source_condition", "context"]}
        if "label_note" in templates[mid]:
            method["label_note"] = templates[mid]["label_note"]
        method.update({"expected_em_correct": correct, "records_path": str(output),
                       "records_sha256": sha(output), "source_sha256": {str(path): sha(path) for path in source_files},
                       "source_root": str(source), "questions": 1711, **means,
                       "answer_model": MODEL, "source_empty_responses": len(empty_ids),
                       "source_provider_rejections": statuses["provider_rejection"],
                       "source_token_limit_count": token_limit_hits if known_token_limits == 1711 else None,
                       "audit": {"canonical_question_gold_task_point_exact_match": 1711,
                                 "unique_source_ids": 1711,
                                 "all_per_question_em_f1_bleu1_recomputed": True,
                                 "raw_frozen_predictions_unchanged": True,
                                 "empty_and_provider_rejected_answers_preserved": True}})
        methods.append(method)
        reports.append({"id": mid, "questions": 1711, "correct": correct, **means,
                        "returned_model_or_declared_model_counts": dict(models),
                        "status_counts": dict(statuses), "source_token_limit_known_count": known_token_limits,
                        "original_empty_answer_ids": empty_ids,
                        "records_sha256": sha(output)})
    manifest = {"dataset": "MemGallery", "answer_model": MODEL,
                "questions_per_method": 1711, "methods": methods}
    dump(HERE / "visual_methods_manifest.json", manifest)
    dump(HERE / "visual_inputs_validation.json", {"status": "passed", "api_calls": 0,
          "methods": reports, "manifest_sha256": sha(HERE / "visual_methods_manifest.json")})
    print(json.dumps({"status": "passed", "methods": [{"id": report["id"], "correct": report["correct"],
                       "questions": report["questions"], "status_counts": report["status_counts"]}
                      for report in reports]}, indent=2))


if __name__ == "__main__":
    main()
