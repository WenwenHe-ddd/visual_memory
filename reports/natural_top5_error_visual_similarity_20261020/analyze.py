"""Audit saved Natural caption Top5 answers; no new model calls or embeddings."""
import csv
import hashlib
import html
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/storage/hewenwen/VisualMemory")
OUT = Path(__file__).resolve().parent
RUN = ROOT / "results/mcq_top5_10_ablation/top5/caption"
NAT = ROOT / "results/natural_observation"
PAIR = ROOT / "reports/visual_embedding_diagnosis_20261019/pairwise_cosines.csv"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


manifest = read_jsonl(RUN / "qa_manifest.jsonl")
responses = {x["id"]: x for x in read_jsonl(RUN / "qa_responses.jsonl")}
memories = {x["id"]: x for x in read_jsonl(NAT / "memories.jsonl")}
images = {x["id"]: x for x in read_jsonl(NAT / "images.jsonl")}
captions = {x["id"]: x["caption"] for x in read_jsonl(NAT / "captions.jsonl")}
protocol = json.loads((RUN / "protocol.json").read_text())
assert len(manifest) == len(responses) == len({x["id"] for x in manifest}) == 371
assert {x["id"] for x in manifest} == set(responses)
assert all(len(m["images"]) == 1 for m in memories.values())
for name in ["memories.jsonl", "captions.jsonl"]:
    assert sha(NAT / name) == protocol["input_hashes"][str((NAT / name).relative_to(ROOT))]

pair_scores = {}
domain_pairs = defaultdict(list)
domain_distinct_pairs = defaultdict(list)
with PAIR.open() as f:
    for row in csv.DictReader(f):
        a, b = (row[k].removesuffix("#image1") for k in ["earlier_id", "later_id"])
        key = tuple(sorted((a, b)))
        assert key not in pair_scores
        assert memories[a]["task"] == memories[b]["task"] == row["task"]
        assert images[memories[a]["images"][0]]["path"] == row["earlier_image"]
        assert images[memories[b]["images"][0]]["path"] == row["later_image"]
        score = float(row["cosine"])
        pair_scores[key] = score
        domain_pairs[row["task"]].append(score)
        if memories[a]["images"] != memories[b]["images"]:
            domain_distinct_pairs[row["task"]].append(score)
for task, n in Counter(m["task"] for m in memories.values()).items():
    assert len(domain_pairs[task]) == n * (n - 1) // 2
domain_pairs = {k: np.sort(v) for k, v in domain_pairs.items()}
domain_distinct_pairs = {k: np.sort(v) for k, v in domain_distinct_pairs.items()}

# Reuse the actual caption encoder cache, in its explicit saved memory-ID order.
# Embeddings are used transiently here; only pairwise scalar scores are written.
with np.load(NAT / "embeddings.npz", allow_pickle=False) as z:
    caption_vectors = {str(k): v.astype(np.float64) for k, v in zip(z["memory_ids"], z["memory"])}
caption_vectors = {k: v / np.linalg.norm(v) for k, v in caption_vectors.items()}
assert set(caption_vectors) == set(memories)


def percentile(array, value):
    # Midrank empirical percentile, including tied pairs.
    return float((np.searchsorted(array, value, "left") +
                  np.searchsorted(array, value, "right")) / (2 * len(array)))


questions, flat_pairs, flat_questions = [], [], []
for m in manifest:
    response = responses[m["id"]]
    assert m["payload_sha256"] == response["payload_sha256"]
    assert response["experiment_signature"] == protocol["experiment_signature"]
    ids = m["top_memory_ids"]
    assert len(ids) == len(set(ids)) == 5
    assert all(memories[x]["task"] == m["task"] for x in ids)
    nodes = []
    for rank, mid in enumerate(ids, 1):
        image_hash = memories[mid]["images"][0]
        node = {
            "rank": rank, "memory_id": mid, "image_sha256": image_hash,
            "image_path": images[image_hash]["path"],
            "image_url": os.path.relpath(images[image_hash]["path"], OUT),
            "caption": captions[image_hash], "date": memories[mid]["date"],
            "is_gold": mid in m["gold_ids"],
        }
        # Captions shown in gallery are exactly the text included in saved reader input.
        assert node["caption"] in m["reader_text"]
        assert Path(node["image_path"]).exists()
        nodes.append(node)
    ps = []
    correct = response["predicted_choice"] == m["gold_choice"]
    for a, b in itertools.combinations(nodes, 2):
        score = pair_scores[tuple(sorted((a["memory_id"], b["memory_id"])))]
        p = {
            "rank_a": a["rank"], "rank_b": b["rank"],
            "memory_a": a["memory_id"], "memory_b": b["memory_id"],
            "image_a": a["image_path"], "image_b": b["image_path"],
            "visual_cosine": score,
            "caption_cosine": float(caption_vectors[a["memory_id"]] @ caption_vectors[b["memory_id"]]),
            "same_image_sha256": a["image_sha256"] == b["image_sha256"],
            "same_domain_pair_percentile": percentile(domain_pairs[m["task"]], score),
            "gold_a": a["is_gold"], "gold_b": b["is_gold"],
        }
        ps.append(p)
        flat_pairs.append({"question_id": m["id"], "task": m["task"], "correct": correct, **p})
    ps.sort(key=lambda p: -p["visual_cosine"])
    visual = [p["visual_cosine"] for p in ps]
    distinct = [p for p in ps if not p["same_image_sha256"]]
    gold_pairs = [p for p in distinct if p["gold_a"] != p["gold_b"]]
    q = {
        "id": m["id"], "task": m["task"], "question": m["question"], "options": m["options"],
        "gold_choice": m["gold_choice"], "predicted_choice": response["predicted_choice"],
        "correct": correct, "response_status": response["status"],
        "hit_token_limit": response["hit_token_limit"], "gold_ids": m["gold_ids"],
        "query_image_paths": m["query_image_paths"],
        "any_gold_top5": any(x in ids for x in m["gold_ids"]),
        "all_gold_top5": all(x in ids for x in m["gold_ids"]),
        "max_visual_cosine": max(visual), "mean_visual_cosine": float(np.mean(visual)),
        "min_visual_cosine": min(visual),
        "max_distinct_visual_cosine": max(p["visual_cosine"] for p in distinct),
        "max_gold_distractor_cosine": max((p["visual_cosine"] for p in gold_pairs), default=None),
        "mean_caption_cosine": float(np.mean([p["caption_cosine"] for p in ps])),
        "has_exact_duplicate": any(p["same_image_sha256"] for p in ps),
        "unique_image_count": len({x["image_sha256"] for x in nodes}),
        "top5": nodes, "pairs": ps,
    }
    assert q["any_gold_top5"] == m["any_gold_topk"]
    assert q["all_gold_top5"] == m["all_gold_topk"]
    questions.append(q)
    flat_questions.append({
        k: v for k, v in q.items()
        if k not in ["options", "gold_ids", "query_image_paths", "top5", "pairs"]
    } | {
        "gold_ids": ";".join(q["gold_ids"]),
        "top5_ids": ";".join(ids),
        "highest_pair": f"{ps[0]['memory_a']} | {ps[0]['memory_b']}",
        "highest_pair_ranks": f"{ps[0]['rank_a']},{ps[0]['rank_b']}",
    })


def summarize(rows):
    n = len(rows)
    result = {
        "questions": n, "exact_duplicate_questions": sum(x["has_exact_duplicate"] for x in rows),
        "any_gold_top5": sum(x["any_gold_top5"] for x in rows),
        "all_gold_top5": sum(x["all_gold_top5"] for x in rows),
        "no_gold_top5": sum(not x["any_gold_top5"] for x in rows),
    }
    for metric in ["max_visual_cosine", "mean_visual_cosine", "max_distinct_visual_cosine",
                   "mean_caption_cosine"]:
        values = [x[metric] for x in rows]
        result[metric] = {
            "median": float(np.median(values)), "mean": float(np.mean(values)),
            "p25": float(np.quantile(values, .25)), "p75": float(np.quantile(values, .75)),
        }
    # Explicit descriptive cutoffs, NOT semantic labels or clustering parameters.
    result["reference_cutoffs"] = {
        str(c): {
            "max_pair_ge": sum(x["max_visual_cosine"] >= c for x in rows),
            "distinct_max_pair_ge": sum(x["max_distinct_visual_cosine"] >= c for x in rows),
            "mean_pair_ge": sum(x["mean_visual_cosine"] >= c for x in rows),
            "all_ten_pairs_ge": sum(x["min_visual_cosine"] >= c for x in rows),
        } for c in [.85, .90, .95, .98]
    }
    return result


wrong = [q for q in questions if not q["correct"]]
right = [q for q in questions if q["correct"]]
domains = {
    task: {
        "all_domain_pairs": len(domain_pairs[task]),
        "all_domain_pair_median": float(np.median(domain_pairs[task])),
        "all_domain_pair_p95": float(np.quantile(domain_pairs[task], .95)),
        "incorrect": summarize([q for q in wrong if q["task"] == task]),
        "correct": summarize([q for q in right if q["task"] == task]),
    } for task in sorted(domain_pairs)
}
summary = {
    "experiment": "Original Natural caption Top5 (not query-image-caption aligned)",
    "source_run": str(RUN), "questions": len(questions), "correct": len(right),
    "incorrect": len(wrong),
    "valid_wrong_letters": sum(q["predicted_choice"] in "ABCD" for q in wrong if q["predicted_choice"]),
    "format_errors": [q["id"] for q in wrong if q["response_status"] != "ok"],
    "wrong": summarize(wrong), "right": summarize(right), "domains": domains,
    "unit": "Each question has five memory occurrences, each with one image; ten unordered image pairs.",
    "interpretation": [
        "Maximum similarity measures at least one similar pair, not all five images.",
        "Reference cutoffs are descriptive and do not define visual ground truth.",
        "Domain composition confounds pooled wrong/correct similarity comparison.",
        "Reader saw retrieved captions, not retrieved image pixels; similarity is posthoc.",
        "Original fixed-permutation local Qwen result; not paper four-rotation EM.",
        "Observational audit cannot establish whether similar images caused an error.",
    ],
}
save("summary.json", summary)
save("questions.json", questions)
write_csv("per_question.csv", flat_questions)
write_csv("incorrect_questions.csv", [r for r in flat_questions if not r["correct"]])
write_csv("top5_image_pairs.csv", flat_pairs)
save("protocol.json", {
    "source_files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [
        RUN / "qa_manifest.jsonl", RUN / "qa_responses.jsonl", RUN / "protocol.json",
        NAT / "memories.jsonl", NAT / "images.jsonl", NAT / "captions.jsonl",
        NAT / "embeddings.npz", NAT / "protocol.json", PAIR,
        ROOT / "results/visual_embedding_graph/encoder_metadata.json",
    ]},
    "script_sha256": sha(Path(__file__)),
    "visual_encoder": "openai/clip-vit-base-patch32",
    "visual_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "visual_score": "Previously audited normalized 512-D CLIP image feature dot product.",
    "caption_score": "Normalized BGE-small-en-v1.5 cached memory vectors, explicit memory_ids mapping.",
    "pair_percentile": "Midrank percentile among all within-domain occurrence pairs, including this pair.",
    "pair_source_count": len(pair_scores), "pairs_in_report": len(flat_pairs),
    "new_encoder_calls": 0, "new_reader_calls": 0, "new_per_image_embeddings_saved": False,
    "original_images_copied": False,
    "image_hash_verification": "Registry hashes used for equality; selected case files separately rehashed.",
    "reader_protocol": protocol,
})

short = {
    "Brand_Memory_Test": "品牌", "Card_Playlog_Test": "卡牌",
    "Cartoon_Entertainment_Companion": "动漫", "Home_Renovation_Interior_Design": "家居",
    "Multi-Scene_Visual_Case_Archive_Assistant": "多场景",
    "Outdoor_Navigation_Route_Memory_Assistant": "户外",
    "Personal_Health_Dashboard_Assistant": "健康仪表盘", "Social_Chat_Memory_Test": "社交",
}
table = ""
for task, d in domains.items():
    a, b = d["incorrect"], d["correct"]
    table += f"""<tr><td>{short[task]}</td><td>{a['questions']} / {b['questions']}</td>
    <td>{a['max_visual_cosine']['median']:.4f} / {b['max_visual_cosine']['median']:.4f}</td>
    <td>{a['mean_visual_cosine']['median']:.4f} / {b['mean_visual_cosine']['median']:.4f}</td>
    <td>{a['exact_duplicate_questions']} / {b['exact_duplicate_questions']}</td></tr>"""
data = json.dumps(questions, ensure_ascii=False).replace("<", "\\u003c")
page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Natural Top5 错题图片相似度检查</title>
<style>
body{font:16px/1.65 system-ui,sans-serif;background:#f5f7fa;color:#202938;margin:0}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:21px}
.panel,details.q{background:white;border:1px solid #dce1e9;border-radius:12px;padding:18px;margin:16px 0}
.note{color:#536078}.stats{display:flex;gap:18px;flex-wrap:wrap}.stat{background:#eff4fa;padding:14px 20px;border-radius:8px}
.stat b{display:block;font-size:27px}table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:8px;border-bottom:1px solid #e5e8ed;text-align:left}input,select{padding:10px;margin:6px;border:1px solid #bac5d5;border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}.card{border:1px solid #dce1e9;border-radius:8px;overflow:hidden}
.card.gold{border:2px solid #21845a}.card img{width:100%;height:220px;object-fit:contain;background:#f1f3f6}
.card div{padding:8px;overflow-wrap:anywhere;font-size:14px}.caption{font-size:13px;line-height:1.55}
.tag{background:#edf1f6;padding:2px 7px;border-radius:4px;font-size:13px}.good{color:#087046}.bad{color:#b23131}
summary{cursor:pointer;font-weight:650}a{color:#225d9f}.scroll{overflow:auto}
@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){.grid{grid-template-columns:1fr}}
</style><main>
<h1>Natural caption Top5：错题里有哪些相似图片？</h1>
<p>原始实验，174 / 371 答对；检索与回答均复用已保存结果。每题检查实际 Top5 的 10 个图片对。</p>
<div class="stats"><div class="stat"><b>197</b>未答对题（含 1 个格式错误）</div>
<div class="stat"><b>141 / 197</b>至少一对 CLIP cosine ≥ 0.90</div>
<div class="stat"><b>98 / 197</b>至少一对 CLIP cosine ≥ 0.95</div>
<div class="stat"><b>17 / 197</b>包含相同图片内容的不同记忆</div></div>
<div class="panel"><h2>如何理解这些数值</h2>
<p>0.90、0.95 只是便于查看的统计刻度，不是通用的“很相似”判定。最高值表示至少一对接近；10 对平均值用于观察整个 Top5。
卡牌、健康仪表盘的共同界面使相似度普遍偏高，因此应当在同一领域内比较。</p>
<p>答对题也有 107 / 174 题最高值 ≥ 0.90，42 / 174 题 ≥ 0.95，5 / 174 题包含相同图片。
总体错题最高值中位数 0.9488，答对题 0.9168；分领域后并没有普遍的“错题更相似”。</p>
<p><b>该 reader 实际只读检索图片的 caption，没有读取这些图片的像素，也没有收到记忆的日期、轮次或对话。</b>
页面为核查补充了原图、日期、ID 和标注，不能把这些补充信息当成当时的模型输入。
是否因相似记忆而答错，需要单独做受控干预实验。</p></div>
<div class="panel"><h2>各领域：错题 / 答对题</h2><div class="scroll"><table>
<tr><th>领域</th><th>题数</th><th>最高图片相似度的中位数</th><th>10 对均值的中位数</th><th>含相同图片的题数</th></tr>
__TABLE__</table></div></div>
<div class="panel"><h2>与刚才两张家居图片相关的具体例子</h2>
<p><a href="#case-Home_Renovation_Interior_Design-Q10" onclick="showCase('Home_Renovation_Interior_Design/Q10')">家居 Q10</a>
问“工作区布局测试图里，哪张桌子已经有显示器？”正确答案为靠窗桌，模型答成两张桌子都有。</p>
<p>Top3 的 D7_IMG_005 是标注目标，caption 说明靠窗桌有显示器，前景桌放的是合上的笔记本电脑。
Top5 的 D7_IMG_006 则描述两张桌子都有显示器；Top1 的 D13_IMG_002 同样描述两张桌子各有显示器。
005 与 006 的图片相似度为 0.9047。这里同时出现了相近场景和回答关键细节的冲突，符合混淆的可能机制。</p>
<p>作为对照，同一对 005 / 006 也同时出现在答对的家居 Q19、Q20 中。因此不能仅凭这对图片出现就断定其导致错误。</p></div>
<div class="panel"><h2>相似图片之外，也要检查 caption 缺失了什么</h2>
<p><a href="#case-Home_Renovation_Interior_Design-Q12" onclick="showCase('Home_Renovation_Interior_Design/Q12')">家居 Q12</a>：
Top1 / Top2 是两张木柜厨房图（D8_IMG_001 / 007，相似度 0.9577），实际布局明显不同。
题目问小碗与砧板哪个更靠近冰箱；正确图片的 caption 提到了两者，但没有保留谁更近。
正确答案是小碗，模型选了砧板。这里存在相似场景，但 caption 信息缺失也是直接可见的问题。</p>
<p><a href="#case-Multi-Scene_Visual_Case_Archive_Assistant-Q10" onclick="showCase('Multi-Scene_Visual_Case_Archive_Assistant/Q10')">多场景 Q10</a>：
Top1 / Top3 的 S4-IMG1 / IMG3 是高度相似的客厅场景（0.9847），钱包和椅背衣物有增减。
题目问什么最靠近桌面右前角；正确答案是塑料包装物，模型选了钱包。
caption 列举了物品却没有保留其相对位置，而且两张图里塑料包装物的位置并没有反转，因此不能把这道题解释为选错相似图片导致。</p>
<p class="note">这三例已逐图目视核对；选择规则为延续前一轮的 005 / 006 讨论，以及家居和多场景领域里无相同图片重复的错题中最高相似度案例。
尚未对全部 197 道错题逐一归因。</p></div>
<div class="panel"><h2>逐题查看原图与实际 caption</h2>
<label>结果 <select id="status"><option value="wrong">未答对</option><option value="right">答对</option><option value="all">全部</option></select></label>
<label>领域 <select id="domain"><option value="">全部领域</option>__OPTIONS__</select></label>
<label>排列 <select id="sort"><option value="id">原顺序</option><option value="max">最高相似度从高到低</option><option value="mean">平均相似度从高到低</option></select></label>
<input id="search" placeholder="搜索题号、问题或图片名"><label><input type="checkbox" id="duplicates">仅含相同图片</label>
<p class="note">绿色边框表示数据集标注的 gold 记忆。相似度矩阵以检索名次表示图片对；图像与文本相似度来自不同 encoder，不能直接比较数值大小。</p>
<p id="count"></p><div id="questions"></div></div>
<p><a href="summary.json">完整统计</a> · <a href="incorrect_questions.csv">197 道未答对题 CSV</a> ·
<a href="top5_image_pairs.csv">全部 3710 个检索图片对</a> · <a href="protocol.json">来源与计算口径</a> ·
<a href="audit.json">独立核查</a></p>
</main><script>
const data=__DATA__;
const el=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const anchor=q=>'case-'+q.id.replaceAll('/','-');
function body(q){
 const cards=q.top5.map(n=>`<article class="card ${n.is_gold?'gold':''}"><a href="${esc(n.image_url)}" target="_blank"><img loading="lazy" src="${esc(n.image_url)}" alt="${esc(n.memory_id)}"></a><div><b>Top${n.rank} ${n.is_gold?'· gold':''}</b><br>${esc(n.memory_id)}<br>${esc(n.image_path.split('/').pop())}<br><span class="note">${esc(n.date)}（核查补充）</span><details><summary>实际 caption</summary><p class="caption">${esc(n.caption)}</p></details></div></article>`).join('');
 const ps=q.pairs.map(p=>`<tr><td>Top${p.rank_a} ↔ Top${p.rank_b}</td><td>${p.visual_cosine.toFixed(4)}</td><td>${p.caption_cosine.toFixed(4)}</td><td>${(p.same_domain_pair_percentile*100).toFixed(1)}%</td><td>${p.same_image_sha256?'相同图片内容':''}</td></tr>`).join('');
 return `<p>${esc(q.question)}</p><p>${Object.entries(q.options).map(([k,v])=>`${k}: ${esc(v)}`).join('　')}</p><p><b class="${q.correct?'good':'bad'}">模型：${esc(q.predicted_choice??'无有效字母')} · 正确：${q.gold_choice}</b> · ${q.response_status} · 标注目标${q.any_gold_top5?'已出现':'未出现'}在 Top5${q.query_image_paths.length?' · 该题另有 query image':''}</p><p class="note">全部 gold：${esc(q.gold_ids.join(', '))}</p><div class="grid">${cards}</div><div class="scroll"><table><tr><th>图片对</th><th>图片 CLIP cosine</th><th>caption BGE cosine</th><th>同领域全部图片对百分位</th><th>重复</th></tr>${ps}</table></div>`;
}
function render(){
 const s=el('status').value,d=el('domain').value,search=el('search').value.toLowerCase();
 let rows=data.filter(q=>(s==='all'||q.correct===(s==='right'))&&(!d||q.task===d)&&(!el('duplicates').checked||q.has_exact_duplicate)&&(!search||[q.id,q.question,...q.top5.map(n=>n.image_path)].join(' ').toLowerCase().includes(search)));
 if(el('sort').value==='max')rows.sort((a,b)=>b.max_visual_cosine-a.max_visual_cosine);
 if(el('sort').value==='mean')rows.sort((a,b)=>b.mean_visual_cosine-a.mean_visual_cosine);
 el('count').textContent=`显示 ${rows.length} 题`;
 el('questions').innerHTML=rows.map(q=>`<details class="q" id="${anchor(q)}" data-id="${esc(q.id)}"><summary>${esc(q.id)} · ${q.correct?'答对':'未答对'} · max ${q.max_visual_cosine.toFixed(4)} / mean ${q.mean_visual_cosine.toFixed(4)} ${q.has_exact_duplicate?'· 含相同图片':''}</summary><div class="content"></div></details>`).join('');
 document.querySelectorAll('details.q').forEach(node=>node.addEventListener('toggle',()=>{if(node.open&&!node.dataset.loaded){node.querySelector('.content').innerHTML=body(data.find(q=>q.id===node.dataset.id));node.dataset.loaded='1';}}));
}
function showCase(id){el('status').value='all';el('domain').value='';el('search').value=id;el('duplicates').checked=false;render();const node=document.querySelector('details.q');if(node)node.open=true;}
['status','domain','sort','duplicates'].forEach(id=>el(id).addEventListener('change',render));el('search').addEventListener('input',render);render();
if(location.hash){const q=data.find(q=>'#'+anchor(q)===location.hash);if(q)showCase(q.id);}
</script></html>"""
page = page.replace("__TABLE__", table).replace("__DATA__", data).replace("__OPTIONS__", "".join(
    f'<option value="{html.escape(t)}">{short[t]}</option>' for t in domains))
(OUT / "report.html").write_text(page)
print(json.dumps({
    "questions": len(questions), "incorrect": len(wrong), "correct": len(right),
    "pair_rows": len(flat_pairs), "output": str(OUT),
    "wrong": summary["wrong"], "right": summary["right"],
}, ensure_ascii=False, indent=2))
