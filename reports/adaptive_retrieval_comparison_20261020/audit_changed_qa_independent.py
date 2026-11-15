"""Audit paired payloads, native query image channels and saved reader outputs."""
from __future__ import annotations
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
import audit_home_exact_independent as H

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
QA=BASE/"changed_qa"
SOURCE=ROOT/"data/independent_visual_inputs"


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def compose(q,ordered_ids,texts):
    chunks=["Retrieved dialogue and image descriptions (ranked by relevance):"]
    chunks.extend(f"Memory {rank}: {texts[mid]}" for rank,mid in enumerate(ordered_ids,1))
    chunks.extend(["Question:\n"+q["question"],
                   "Options:\n"+"\n".join(f"{key}. {q['options'][key]}" for key in "ABCD"),
                   "Answer with one letter only."])
    return "\n\n".join(chunks)


def main():
    protocol=H.read(QA/"protocol.json")
    settings=protocol["reader_settings"]
    pairs=H.read(QA/"pairs.json")
    manifest=lines(QA/"qa_manifest.jsonl")
    outputs=lines(QA/"responses.jsonl")
    result=H.read(QA/"summary.json")
    original_questions=lines(SOURCE/"qa_manifest.jsonl")
    original={q["id"]:q for q in original_questions}
    original_texts={r["id"]:r["reader_text"] for r in lines(SOURCE/"retrieval_memories.jsonl")}
    source_protocol=H.read(SOURCE/"protocol.json")
    sources={**protocol["source_sha256"],**protocol["model_source_sha256"]}
    for path,expected in sources.items():H.check(H.sha(path)==expected,"recorded source/model-config hash")
    signature=protocol["experiment_signature"]
    H.check(digest({k:v for k,v in protocol.items() if k!="experiment_signature"})==signature,"protocol signature")
    H.check(protocol["historical_response_reuse"] is False,"fresh responses")
    for key in ("reader_model","reader_model_path","reader_system_prompt","seed","dtype",
                "do_sample","max_new_tokens","query_image_max_pixels"):
        H.same(settings[key],source_protocol[key],"pinned common reader setting "+key)
    H.check(settings["batch_size_text"]==settings["batch_size_query_images"]==1,"matched batch1")
    H.check(settings["do_sample"] is False and settings["seed"]==42,"greedy common seed42")
    # Validate pinned weight metadata without rereading 17.5 GB of model shards.
    weight_status=[]
    for w in protocol["weight_files"]:
        p=Path(w["path"]);stat=p.stat()
        meta=(p.parent/".cache/huggingface/download"/(p.name+".metadata")).read_text().splitlines()
        H.check(stat.st_size==w["bytes"] and stat.st_mtime_ns==w["mtime_ns"],"unchanged checkpoint file stat")
        H.check(meta[0]==w["snapshot_revision"]==settings["reader_model"]["revision"],"weight pinned revision")
        H.check(meta[1]==w["sha256"],"recorded weight hash matches pinned download metadata")
        weight_status.append(dict(path=str(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                  sha256=w["sha256"],verification="stat and pinned metadata; no duplicate full-shard hash"))
    retrieval={}
    texts={}
    for folder in sorted((BASE/"all_empirical").iterdir()):
        if not folder.is_dir():continue
        inputs=H.read(BASE/"inputs_by_domain"/folder.name/"inputs.json")
        for leaf in inputs["leaves"]:
            H.check(leaf["retrieval_text"]==original_texts[leaf["memory_id"]],"438 complete source reader texts")
            H.check(leaf["retrieval_text"]==leaf["dialogue"]+"\nMemory description: "+leaf["caption"],"no evidence rewrite")
            texts[leaf["memory_id"]]=leaf["retrieval_text"]
        for row in H.read(folder/"runs.json"):
            if row["L"]==5 and row["method"] in ("full_scan","empirical_max_both"):
                retrieval[row["id"],row["method"]]=row
    H.check(len(texts)==438 and len(retrieval)==742,"complete source memory and retrieval coverage")
    changed=[];unchanged=[]
    for q in original_questions:
        a=retrieval[q["id"],"full_scan"]["selected_memory_ids"]
        b=retrieval[q["id"],"empirical_max_both"]["selected_memory_ids"]
        ta,tb=compose(q,a,texts),compose(q,b,texts)
        if a==b:
            H.check(ta==tb,"identical reader context for every unchanged question")
            unchanged.append(dict(id=q["id"],reader_text_sha256=hashlib.sha256(ta.encode()).hexdigest()))
        else:
            H.check(ta!=tb,"changed ID order changes actual reader context")
            changed.append(q["id"])
    H.check(len(changed)==21 and len(unchanged)==350,"changed/unchanged denominator")
    H.same([p["question_id"] for p in pairs],changed,"changed set chosen by retrieval alone")
    H.check(len(manifest)==len(outputs)==42,"42 fresh payloads and outputs")
    indexed={r["record_id"]:r for r in manifest}
    answered={r["record_id"]:r for r in outputs}
    H.check(len(indexed)==len(answered)==42 and set(indexed)==set(answered),"unique complete matching record IDs")
    counterbalanced=[]
    for pi,pair in enumerate(pairs):
        a,b=pair["conditions"]
        q=original[pair["question_id"]]
        for key in ("id","task","question","options","gold_choice","gold_ids","query_image_paths","query_images"):
            H.same(a[key],b[key],"paired invariant "+key)
        for row in (a,b):
            H.same(row,indexed[row["record_id"]],"pair/manifest equality")
            for key in ("id","task","question","options","gold_choice","gold_ids","query_image_paths"):
                H.same(row[key],q[key],"source question/option equality")
            selected=retrieval[q["id"],row["condition"]]
            H.same(row["top_memory_ids"],selected["selected_memory_ids"],"exact ranked Top5")
            H.same(row["top_query_scores"],selected["selected_scores"],"retrieval scores for audit only")
            expected=compose(q,selected["selected_memory_ids"],texts)
            H.same(row["reader_text"],expected,"exact evidence/question/options payload; excludes gold/score metadata")
            H.check(len(row["query_images"])==len(q["query_image_paths"]),"native image channel")
            for image in row["query_images"]:
                H.check(image["path"] in q["query_image_paths"],"original query image path")
                H.check(H.sha(image["path"])==image["sha256"],"native query image byte hash")
            payload={"system":settings["reader_system_prompt"],"user_text":expected,"query_images":row["query_images"]}
            H.check(digest(payload)==row["payload_sha256"],"recomputed payload hash")
        H.check(a["payload_sha256"]!=b["payload_sha256"],"context-only paired payload difference")
        counterbalanced.extend([a["record_id"],b["record_id"]] if pi%2==0 else [b["record_id"],a["record_id"]])
    H.same([r["record_id"] for r in manifest],counterbalanced,"alternating condition order")
    H.same([r["record_id"] for r in outputs],counterbalanced,"actual output order")
    # Independently reconstruct prompt rendering and token/image processing.
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    sys.path[:0]=[str(ROOT/".vendor"),str(ROOT)]
    from transformers import AutoProcessor
    from PIL import Image
    processor=AutoProcessor.from_pretrained(settings["reader_model_path"],local_files_only=True)
    processor.tokenizer.padding_side="left"
    processor.image_processor.size={"shortest_edge":65536,"longest_edge":settings["query_image_max_pixels"]}
    environment=H.read(QA/"generation_environment.json")
    H.check(environment["dtype"]=="torch.bfloat16","BF16 actual model")
    H.same(environment["processor_size"],processor.image_processor.size,"native image resize policy")
    for row in manifest:
        answer=answered[row["record_id"]]
        content=[{"type":"image"} for _ in row["query_image_paths"]]
        content.append({"type":"text","text":row["reader_text"]})
        prompt=processor.apply_chat_template([
            {"role":"system","content":settings["reader_system_prompt"]},
            {"role":"user","content":content}],tokenize=False,add_generation_prompt=True)
        H.check(hashlib.sha256(prompt.encode()).hexdigest()==answer["rendered_prompt_sha256"],"actual rendered prompt hash")
        images=[]
        for path in row["query_image_paths"]:
            with Image.open(path) as image:images.append(image.convert("RGB"))
        encoded=processor(text=[prompt],images=images or None,padding=True,return_tensors="np")
        tokens=int(encoded["attention_mask"].sum())
        H.same(answer["input_tokens"],tokens,"exact native-image/text input token count")
        H.check(tokens+settings["max_new_tokens"]<=environment["context_limit"],"no input truncation")
        decoded=processor.tokenizer.decode(answer["generated_token_ids"],skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False).strip()
        H.same(answer["raw_response"],decoded,"token IDs decode to recorded output")
        H.check(decoded in "ABCD" and len(decoded)==1,"strict single-letter output")
        H.same(answer["predicted_choice"],decoded,"prediction parse")
        H.same(answer["gold_choice"],row["gold_choice"],"evaluation gold only after inference")
        H.same(answer["correct"],decoded==row["gold_choice"],"answer correctness")
        H.check(answer["status"]=="ok" and not answer["hit_token_limit"],"format and length status")
        H.check(answer["payload_sha256"]==row["payload_sha256"] and answer["experiment_signature"]==signature,
                "response provenance")
        H.check(answer["batch_size"]==1 and answer["query_image_count"]==len(images),"matched generation channels")
        H.check(0<answer["generated_tokens"]<=settings["max_new_tokens"],"bounded output count")
        for image in images:image.close()
    # Effective generate kwargs override the model's do_sample=True default.
    tree=ast.parse((ROOT/"scripts/analysis/evaluate_adaptive_changed_qa.py").read_text())
    generation=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute) and node.func.attr=="generate"]
    H.check(len(generation)==1,"one controlled generation call")
    keywords={k.arg:k.value for k in generation[0].keywords}
    H.check(isinstance(keywords["do_sample"],ast.Constant) and keywords["do_sample"].value is False,
            "effective do_sample=False overrides checkpoint default")
    H.check(isinstance(keywords["max_new_tokens"],ast.Subscript),"generation token budget from common settings")
    paired=[]
    for qid in changed:
        a,b=[answered[f"{qid}::{condition}"] for condition in ("full_scan","empirical_max_both")]
        paired.append(dict(question_id=qid,query_has_image=bool(original[qid]["query_image_paths"]),
            gold_choice=a["gold_choice"],full_scan_answer=a["predicted_choice"],empirical_answer=b["predicted_choice"],
            full_scan_correct=a["correct"],empirical_correct=b["correct"],
            answer_changed=a["predicted_choice"]!=b["predicted_choice"],
            correct_to_wrong=a["correct"] and not b["correct"],wrong_to_correct=not a["correct"] and b["correct"]))
    H.same(result["paired_results"],paired,"paired outcome table")
    totals=dict(n_questions=21,n_responses=42,full_scan_correct=sum(p["full_scan_correct"] for p in paired),
                empirical_max_both_correct=sum(p["empirical_correct"] for p in paired),
                correct_to_wrong=sum(p["correct_to_wrong"] for p in paired),
                wrong_to_correct=sum(p["wrong_to_correct"] for p in paired),
                answer_changed=sum(p["answer_changed"] for p in paired),format_errors=0,token_limit_hits=0,
                responses_per_condition={"full_scan":21,"empirical_max_both":21},full_371_accuracy_measured=False)
    H.same(result,totals,"summary numerators")
    changed_answers=[p for p in paired if p["answer_changed"]]
    H.check(len(changed_answers)==1 and changed_answers[0]["question_id"]=="Outdoor_Navigation_Route_Memory_Assistant/Q31",
            "only Q31 changes answer")
    report=dict(passed=True,checks=H.checks,summary=totals,answer_changes=changed_answers,
                unchanged_context_questions=unchanged,complete_source_memory_texts=438,
                payloads_verified=42,rendered_prompts_verified=42,native_image_payloads_verified=2,
                output_token_decodes_verified=42,weight_checks=weight_status,
                effective_generation=dict(do_sample=False,seed=42,max_new_tokens=settings["max_new_tokens"],
                                          model_default_do_sample=environment["generation_config"]["do_sample"]),
                source_hashes={str(p.relative_to(ROOT)):H.sha(p) for p in QA.iterdir() if p.is_file()},
                limitations=[
                    "This is fresh paired inference on21 selected changed-context questions; it is not a newly measured371-question accuracy.",
                    "350 other ordered reader contexts are verified identical, but their reader outputs were not regenerated.",
                    "Only retrieval context differs within a pair; original questions/options, native query images, model and generation settings match.",
                    "One native query image is processed in both conditions; cached retrieval-only query caption is excluded from the reader payload.",
                    "Q31 gains a gold-memory retrieval hit but changes from correct D to wrong A; no isolated-memory causal mechanism is established.",
                    "Encoded payloads and output token IDs are audited; model generation itself was not rerun.",
                    "Model shard metadata/stat and recorded pinned hashes were verified without repeating the full17.5GB hashing pass.",
                ])
    (BASE/"validation_changed_qa_independent.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(passed=True,checks=H.checks,summary=totals,answer_changes=changed_answers)))


if __name__=="__main__":
    main()
