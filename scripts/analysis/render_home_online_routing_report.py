#!/usr/bin/env python3
"""Render the saved Home routing experiment as a self-contained local HTML page.

This module renders existing inputs and results only. It performs no encoding,
retrieval, statistical estimation, or answer generation.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/home_online_routing_dialogue_caption_20261020"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Home · 图文与文文检索对照</title>
<style>
:root{color-scheme:light;--ink:#1e293b;--muted:#64748b;--blue:#235ea8;--line:#dbe3ed;--paper:#fff;--bg:#f3f6fa;--green:#166534;--gold:#fef9e7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1380px;padding:32px 24px 64px;margin:auto}h1{font-size:28px;margin:0 0 8px}h2{font-size:21px;margin:0 0 14px}h3{font-size:17px;margin:14px 0 8px}
p{margin:8px 0}.muted,small{color:var(--muted)}.lead{max-width:1000px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:22px;margin:20px 0}
.cards{display:grid;grid-template-columns:repeat(3,minmax(170px,1fr));gap:12px;margin:16px 0}.card{background:#f7f9fc;border:1px solid var(--line);border-radius:9px;padding:14px}.card strong{font-size:25px;display:block;color:var(--blue)}.card span{display:block}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}table{border-collapse:collapse;width:100%;background:white;font-size:14px}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#eef3fa;white-space:nowrap;font-weight:650}tbody tr:last-child td{border-bottom:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.rank-table>thead>tr>th{position:sticky;top:0;z-index:1}
.target-row>td{background:var(--gold)}.badge{display:inline-block;border-radius:5px;padding:1px 7px;background:#e8eef7;font-size:12px;white-space:nowrap}.badge.gold{color:#7c5300;background:#fff0ba}.badge.green{color:var(--green);background:#e8f5eb}.badge.explore{background:#f3eafa;color:#6b3990}
details{margin:10px 0}summary{cursor:pointer;color:var(--blue);font-weight:550}summary:focus-visible,button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #8bb8ef;outline-offset:2px}
.toolbar{display:flex;align-items:end;gap:14px;flex-wrap:wrap}.field{display:flex;flex-direction:column;gap:5px}.field.grow{flex:1;min-width:240px}
label{font-size:13px;font-weight:600;color:#475569}select,input,button{font:inherit;border:1px solid #c4cfde;border-radius:7px;padding:8px 10px;background:white;color:var(--ink)}
input{width:100%;min-width:180px}select{max-width:100%}button{cursor:pointer}button:hover{background:#edf3fb}button:disabled{opacity:.45;cursor:default}.buttons{display:flex;gap:7px}
.question-title{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.question-text{font-size:18px;white-space:pre-wrap;margin:12px 0 20px}.target-list{display:flex;flex-wrap:wrap;gap:10px}
.target-card{padding:11px 13px;border:1px solid #edd28c;border-radius:8px;background:var(--gold)}.target-card strong{display:block}.target-card small{display:block;overflow-wrap:anywhere}
.legend{border-left:3px solid #abc4e5;padding-left:12px;font-size:13px;color:#52637c}.empty{padding:30px;text-align:center;color:var(--muted)}
.cluster-details{margin:0}.cluster-details>summary{white-space:nowrap}.members{margin-top:12px;min-width:480px;max-width:880px}.member{border:1px solid var(--line);border-radius:8px;margin:12px 0;padding:14px;background:#fafcff}
.member.gold-member{border-color:#e9c768;background:#fffdf3}.member-title{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-weight:600}
.dialogue{white-space:pre-wrap;margin:6px 0 12px;padding:10px;background:#eef3fa;border-radius:6px;font-size:14px}.caption{white-space:pre-wrap;margin:6px 0}
.evidence-label{font-size:12px;font-weight:650;color:#62728a;margin-top:12px}.small-list{padding-left:23px;margin:8px 0}.small-list li{padding:3px 0}
.status{font-size:13px;color:var(--muted);margin-left:auto}.subheading{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.jump{font-size:12px;background:#fffdf4;border-color:#e9c768;margin-top:7px}
.scope-details{border-top:1px solid var(--line);padding-top:13px;margin-top:18px}.score{font-variant-numeric:tabular-nums}.no-wrap{white-space:nowrap}
.highlighted{outline:3px solid #e6b73b;outline-offset:-3px}.footer-note{font-size:13px;color:var(--muted)}
@media(max-width:780px){main{padding:20px 12px}.panel{padding:15px}.cards{grid-template-columns:1fr}.toolbar .field{width:100%}.members{min-width:350px}.question-text{font-size:16px}th,td{padding:8px}}
@media print{body{background:#fff}main{max-width:none;padding:0}.toolbar,.buttons,.jump{display:none}.panel{break-inside:avoid}details:not([open]){display:none}}
</style>
</head>
<body>
<main>
<h1>Home · 图文与文文检索对照</h1>
<p class="lead">逐题查看：CLIP 如何排列记忆簇，目标图片在哪个簇，以及对话与完整 caption 的 BGE 相似度会怎样改变排序。</p>
<p class="muted">这是一份检索结果报告，没有运行回答模型。不同簇包含的图片数不同，按簇数截取候选不等于固定图片预算。</p>
<section id="summary-panel" class="panel" aria-labelledby="summary-title"><h2 id="summary-title">整体结果</h2><div id="summary-content"></div></section>
__SCIENTIFIC_PLOT__
<section class="panel" aria-label="选择问题">
<div class="toolbar">
 <div class="field"><label for="scope-filter">问题范围</label><select id="scope-filter"><option value="all">全部问题（含 Q10）</option><option value="new">除 Q10 外的问题</option><option value="q10">仅 Q10 探索案例</option></select></div>
 <div class="field grow"><label for="query-filter">搜索题目、题号或目标图片</label><input id="query-filter" type="search" placeholder="例如 Q10、D7_IMG_005、monitor" autocomplete="off"></div>
 <div class="field grow"><label for="question-select">选择题目</label><select id="question-select"></select></div>
 <div class="buttons"><button type="button" id="previous">上一题</button><button type="button" id="next">下一题</button></div>
 <span id="question-count" class="status" aria-live="polite"></span>
</div>
</section>
<section id="question-panel" class="panel" aria-live="polite"></section>
<p class="footer-note">此页数据已内嵌，可直接离线打开；没有原图副本或外部资源请求。目标标记来自实验中的参考记忆。</p>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("report-data").textContent);
const leaves=new Map(DATA.inputs.leaves.map(leaf=>[leaf.image_id,leaf]));
const questions=DATA.results.questions.slice().sort((a,b)=>a.id.localeCompare(b.id,undefined,{numeric:true}));
const allById=new Map(questions.map(question=>[question.id,question]));
const select=document.getElementById("question-select");
const scopeFilter=document.getElementById("scope-filter");
const search=document.getElementById("query-filter");
const panel=document.getElementById("question-panel");
let filtered=[];

function node(tag,text,cls){
 const item=document.createElement(tag);
 if(text!==undefined&&text!==null)item.textContent=String(text);
 if(cls)item.className=cls;
 return item;
}
function add(parent,...children){for(const child of children){if(child)parent.appendChild(child);}return parent;}
function shortId(id){return id.split("/").pop();}
function isQ10(id){return shortId(id).toUpperCase()==="Q10";}
function score(value){return Number.isFinite(value)?value.toFixed(4):"—";}
function number(value){return Number.isFinite(value)?String(value):"—";}
function fraction(count,total){return total?`${count}/${total} · ${(100*count/total).toFixed(1)}%`:"—";}
function goldBadge(count=1){return node("span",count>1?`${count} 条目标`:"目标","badge gold");}
function scoreCell(value){
 const td=node("td",score(value),"num score");
 if(Number.isFinite(value))td.title=String(value);
 return td;
}
function table(headers){
 const container=node("div",null,"table-wrap");
 const element=node("table");
 const head=node("thead");const headRow=node("tr");
 for(const header of headers)add(headRow,node("th",header.text??header,header.numeric?"num":null));
 add(head,headRow);const body=node("tbody");add(element,head,body);add(container,element);
 return {container,element,body};
}
function dialogueText(value){
 if(typeof value==="string")return value;
 if(Array.isArray(value))return value.map(turn=>{
  if(typeof turn==="string")return turn;
  return `${turn.role??turn.speaker??""}: ${turn.content??turn.text??""}`;
 }).join("\n");
 if(value&&typeof value==="object")return Object.entries(value).map(([key,text])=>`${key}: ${text}`).join("\n");
 return "";
}
function drawScope(scope,parent,heading,primary){
 add(parent,node("h3",heading));
 const count=scope.n_questions;
 add(parent,node("p",`${count} 道问题，${scope.n_gold_memories} 条目标记忆。`,"muted"));
 const cards=node("div",null,"cards");
 for(const [label,value] of [
  ["CLIP 首簇：至少命中一条目标",scope.clip_top1_any_count],
  ["BGE 均分首簇：至少命中一条目标",scope.text_mean_top1_any_count],
  ["全库 BGE Top5：至少命中一条目标",scope.flat_bge_top5.any_count]
 ]){
  const card=node("div",null,"card");
  add(card,node("span",label),node("strong",fraction(value,count)));add(cards,card);
 }
 add(parent,cards);
 const metricTable=table(["取前 K 个簇","CLIP 至少一条目标","CLIP 全部目标","BGE 均分至少一条目标","CLIP 候选图片数","级联 Top5 至少一条目标"]);
 for(const row of scope.coverage.filter(item=>[1,3,5].includes(item.k))){
  const tr=node("tr");
  add(tr,node("td",row.k),node("td",fraction(row.clip_any_count,count)),
      node("td",fraction(row.clip_all_count,count)),node("td",fraction(row.text_mean_any_count,count)),
      node("td",`${row.candidate_leaves_mean.toFixed(1)} 平均 · ${row.candidate_leaves_min}–${row.candidate_leaves_max}`),
      node("td",fraction(row.cascade_any_count,count)));
  add(metricTable.body,tr);
 }
 add(parent,metricTable.container);
 add(parent,node("p","这里展示 K=1、3、5 的诊断结果，没有把某个 K 设为正式策略。级联 Top5 指先取 CLIP 前 K 个簇，再在候选图片中按 BGE 相似度选前 5 条。","legend"));
}
function drawSummary(){
 const parent=document.getElementById("summary-content");
 const fresh=DATA.summary.scopes.new_questions,all=DATA.summary.scopes.all_questions;
 drawScope(fresh,parent,`主汇总：除 Q10 外的 ${fresh.n_questions} 题`,true);
 const expanded=node("details",null,"scope-details");
 add(expanded,node("summary",`查看全部 ${all.n_questions} 题（包含 Q10 探索案例）`));
 const body=node("div");drawScope(all,body,`全部 ${all.n_questions} 题`,false);add(expanded,body);add(parent,expanded);
 add(parent,node("p","“至少一条目标”与“全部目标”按问题计数；有多条目标的题，两种指标可能不同。BGE 的叶子文本使用原对话与完整 caption。","muted"));
 scopeFilter.options[0].textContent=`全部 ${all.n_questions} 题（含 Q10）`;
 scopeFilter.options[1].textContent=`除 Q10 外的 ${fresh.n_questions} 题`;
}
function leafList(ids,question,leafScores){
 const list=node("ol",null,"small-list");
 const targetIds=new Set(question.gold_targets.map(target=>target.image_id));
 for(const id of ids){
  const leaf=leaves.get(id);const item=node("li");
  add(item,node("span",leaf.filename));
  if(targetIds.has(id))add(item,document.createTextNode(" "),goldBadge());
  if(leafScores.has(id))add(item,node("small",` · BGE ${score(leafScores.get(id).score)} · ${leaf.master_id}`));
  add(list,item);
 }
 return list;
}
function drawMember(member){
 const leaf=leaves.get(member.image_id);
 const article=node("article",null,member.is_gold?"member gold-member":"member");
 const heading=node("div",null,"member-title");
 add(heading,node("span",leaf.filename),node("span",`簇内第 ${member.within_master_rank} · BGE ${score(member.score)}`,"badge"));
 if(member.is_gold)add(heading,goldBadge());
 add(article,heading,node("small",leaf.memory_id),
     node("div","原对话","evidence-label"),node("div",dialogueText(leaf.dialogue),"dialogue"),
     node("div","完整 caption","evidence-label"),node("p",leaf.caption,"caption"));
 return article;
}
function drawQuestion(question){
 panel.replaceChildren();
 const heading=node("div",null,"question-title");
 add(heading,node("h2",shortId(question.id)));
 if(isQ10(question.id))add(heading,node("span","探索案例","badge explore"));
 add(panel,heading,node("p",question.question,"question-text"),node("h3","目标图片及所在簇"));
 const targets=node("div",null,"target-list");
 for(const target of question.gold_targets){
  const card=node("div",null,"target-card");
  add(card,node("strong",target.filename),node("small",target.master_id),
      node("div",`CLIP 簇排名 ${target.clip_master_rank} · BGE 均分簇排名 ${target.text_mean_master_rank}`),
      node("small",`BGE 簇内第 ${target.within_master_text_rank} · 全库图片第 ${target.global_leaf_text_rank}`));
  const jump=node("button","查看所在簇","jump");jump.type="button";
  jump.addEventListener("click",()=>{
   const targetRow=document.getElementById("master-"+target.master_id);
   const details=targetRow?.querySelector("details");
   if(details)details.open=true;
   if(targetRow){targetRow.scrollIntoView({behavior:"smooth",block:"center"});targetRow.classList.add("highlighted");
    window.setTimeout(()=>targetRow.classList.remove("highlighted"),1600);}
  });
  add(card,jump);add(targets,card);
 }
 add(panel,targets,node("h3",`全部 ${question.masters.length} 个簇 · 按 CLIP 图文相似度排序`));
 add(panel,node("p","CLIP 列比较问题文本与 master 的视觉表征；BGE 均分列是问题文本与该簇所有“对话＋完整 caption”的相似度平均值。两种分数分别排序，数值大小不能直接跨模型比较。","legend"));
 const ranked=table([{text:"CLIP 排名",numeric:true},"记忆簇与成员",{text:"图片数",numeric:true},{text:"CLIP 分数",numeric:true},{text:"BGE 均分",numeric:true},{text:"BGE 均分排名",numeric:true},"目标"]);
 ranked.element.classList.add("rank-table");
 const leafScores=new Map();
 for(const master of question.masters.slice().sort((a,b)=>a.master_rank-b.master_rank)){
  const row=node("tr",null,master.gold_count?"target-row":null);row.id="master-"+master.master_id;
  const details=node("details",null,"cluster-details");
  add(details,node("summary",`${master.master_id} · 展开成员`));
  const members=node("div",null,"members");
  add(members,node("small","成员按本题的 BGE 相似度排列。"));
  for(const leaf of master.leaves.slice().sort((a,b)=>a.within_master_rank-b.within_master_rank)){
   leafScores.set(leaf.image_id,leaf);add(members,drawMember(leaf));
  }
  add(details,members);
  const memberCell=node("td");add(memberCell,details);
  const targetCell=node("td");add(targetCell,master.gold_count?goldBadge(master.gold_count):node("span","—","muted"));
  add(row,node("td",master.master_rank,"num"),memberCell,node("td",master.n,"num"),
      scoreCell(master.clip_similarity),scoreCell(master.text_mean_similarity),node("td",master.text_mean_rank,"num"),targetCell);
  add(ranked.body,row);
 }
 add(panel,ranked.container);
 const flat=node("details");add(flat,node("summary","对照：全库直接按 BGE 检索的 Top5"),leafList(question.flat_bge_top5,question,leafScores));add(panel,flat);
 const diagnostics=node("details");
 add(diagnostics,node("summary","展开本题不同 K 的候选覆盖与级联 Top5"));
 add(diagnostics,node("p","下面是不同截取规模的诊断，不代表已选择某个 K。目标覆盖以本题参考目标条数为分母。","muted"));
 const comparisons=table(["K","CLIP 候选图片数","CLIP 目标覆盖","BGE 均分目标覆盖","级联 BGE Top5","候选内按均分重排的首簇"]);
 for(const row of question.diagnostics.slice().sort((a,b)=>a.k-b.k)){
  const tr=node("tr");const cascade=node("td");const cascadeDetails=node("details");
  add(cascadeDetails,node("summary",`${row.cascade_gold_hits}/${question.gold_targets.length} 条目标 · 查看 Top5`),
      leafList(row.cascade_top5,question,leafScores));add(cascade,cascadeDetails);
  const rerank=node("td");add(rerank,node("span",row.mean_rerank_top1_master));
  if(row.mean_rerank_top1_any)add(rerank,document.createTextNode(" "),goldBadge());
  add(tr,node("td",row.k),node("td",row.clip_candidate_leaf_count),
      node("td",`${row.clip_gold_hits}/${question.gold_targets.length}`),
      node("td",`${row.text_mean_gold_hits}/${question.gold_targets.length}`),cascade,rerank);add(comparisons.body,tr);
 }
 add(diagnostics,comparisons.container);add(panel,diagnostics);
}
function currentIndex(){return filtered.findIndex(question=>question.id===select.value);}
function refreshQuestion(){
 const question=allById.get(select.value);
 if(question)drawQuestion(question);else panel.replaceChildren(node("div","没有匹配的题目。请调整搜索内容或问题范围。","empty"));
 const index=currentIndex();
 document.getElementById("previous").disabled=index<=0;
 document.getElementById("next").disabled=index<0||index>=filtered.length-1;
 document.getElementById("question-count").textContent=filtered.length?`${index+1} / ${filtered.length} 题`:"0 题";
}
function applyFilters(){
 const oldSelection=select.value;
 const query=search.value.trim().toLowerCase();
 filtered=questions.filter(question=>{
  const inScope=scopeFilter.value==="all"||(scopeFilter.value==="q10"?isQ10(question.id):!isQ10(question.id));
  const searchable=[question.id,question.question,...question.gold_targets.map(target=>target.filename)].join(" ").toLowerCase();
  return inScope&&(!query||searchable.includes(query));
 });
 select.replaceChildren();
 for(const question of filtered){
  const option=node("option",`${shortId(question.id)}${isQ10(question.id)?" · 探索案例":""} — ${question.question}`);
  option.value=question.id;add(select,option);
 }
 if(filtered.some(question=>question.id===oldSelection))select.value=oldSelection;
 select.disabled=!filtered.length;refreshQuestion();
}
select.addEventListener("change",refreshQuestion);
scopeFilter.addEventListener("change",applyFilters);
search.addEventListener("input",applyFilters);
document.getElementById("previous").addEventListener("click",()=>{const index=currentIndex();if(index>0){select.value=filtered[index-1].id;refreshQuestion();}});
document.getElementById("next").addEventListener("click",()=>{const index=currentIndex();if(index>=0&&index<filtered.length-1){select.value=filtered[index+1].id;refreshQuestion();}});
drawSummary();applyFilters();
</script>
</body></html>
"""


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _validate(data: dict) -> None:
    """Check joins that would otherwise silently hide evidence in the viewer."""
    inputs = data["inputs"]
    results = data["results"]
    leaf_ids = [leaf["image_id"] for leaf in inputs["leaves"]]
    if len(leaf_ids) != len(set(leaf_ids)):
        raise ValueError("Duplicate image IDs in inputs.leaves")
    leaves = dict(zip(leaf_ids, inputs["leaves"]))
    input_questions = {question["id"]: question for question in inputs["questions"]}
    result_ids = [question["id"] for question in results["questions"]]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(input_questions):
        raise ValueError("Result questions do not match inputs.questions")
    master_ids = {master["id"] for master in inputs["masters"]}
    for question in results["questions"]:
        if question["question"] != input_questions[question["id"]]["question"]:
            raise ValueError(f"Question text mismatch: {question['id']}")
        masters = question["masters"]
        if len(masters) != len(master_ids) or {master["master_id"] for master in masters} != master_ids:
            raise ValueError(f"Master coverage mismatch: {question['id']}")
        if sorted(master["master_rank"] for master in masters) != list(range(1, len(masters) + 1)):
            raise ValueError(f"Invalid CLIP ranks: {question['id']}")
        scored_ids = []
        for master in masters:
            if len(master["leaves"]) != master["n"]:
                raise ValueError(f"Cluster size mismatch: {question['id']}/{master['master_id']}")
            for leaf in master["leaves"]:
                scored_ids.append(leaf["image_id"])
                source = leaves[leaf["image_id"]]
                if source["master_id"] != master["master_id"] or source["filename"] != leaf["filename"]:
                    raise ValueError(f"Leaf membership mismatch: {leaf['image_id']}")
        if sorted(scored_ids) != sorted(leaf_ids):
            raise ValueError(f"Leaf scores do not cover each image once: {question['id']}")
        for target in question["gold_targets"]:
            source = leaves[target["image_id"]]
            if source["master_id"] != target["master_id"]:
                raise ValueError(f"Target master mismatch: {question['id']}")
        for image_id in question["flat_bge_top5"]:
            if image_id not in leaves:
                raise ValueError(f"Unknown flat Top5 image: {image_id}")
        for diagnostic in question["diagnostics"]:
            for image_id in diagnostic["cascade_top5"]:
                if image_id not in leaves:
                    raise ValueError(f"Unknown cascade Top5 image: {image_id}")
    scopes = data["summary"]["scopes"]
    if scopes["all_questions"]["n_questions"] != len(result_ids):
        raise ValueError("All-question summary count differs from question results")
    q10_count = sum(question_id.rsplit("/", 1)[-1].upper() == "Q10" for question_id in result_ids)
    if scopes["new_questions"]["n_questions"] != len(result_ids) - q10_count:
        raise ValueError("Non-Q10 summary count differs from question results")


def render(output: Path) -> Path:
    """Read inputs/results/summary JSON and atomically write ``report.html``."""
    output = Path(output).resolve()
    data = {name: _load(output / f"{name}.json") for name in ("inputs", "results", "summary")}
    _validate(data)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Keep all source content inert inside the JSON script element. The browser
    # inserts visible source text using textContent, never innerHTML.
    for character, escape in (("&", r"\u0026"), ("<", r"\u003c"), (">", r"\u003e"),
                              ("\u2028", r"\u2028"), ("\u2029", r"\u2029")):
        payload = payload.replace(character, escape)
    destination = output / "report.html"
    temporary = output / ".report.html.tmp"
    plot_path = output / "routing_diagnostics.png"
    plot = ""
    if plot_path.is_file():
        encoded = base64.b64encode(plot_path.read_bytes()).decode("ascii")
        plot = (
            '<details class="panel"><summary>查看主汇总的检索对照曲线（不含 Q10）</summary>'
            '<p class="muted">候选目标覆盖、级联 Top5 召回与候选图片数量。</p>'
            f'<img style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}" '
            'alt="除 Q10 外的问题：候选目标覆盖、两阶段 Top5 召回与候选图片数对照图"></details>'
        )
    temporary.write_text(HTML.replace("__REPORT_DATA__", payload).replace("__SCIENTIFIC_PLOT__", plot),
                         encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory containing inputs.json, results.json and summary.json")
    args = parser.parse_args()
    print(render(args.output))


if __name__ == "__main__":
    main()
