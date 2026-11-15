#!/usr/bin/env python3
"""Render a self-contained diagnostic report for bounded text search.

The report reads completed experiment artifacts; it never encodes images or
text, computes retrieval decisions, or changes the saved experiment state.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pick(row, *names, default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def _normalize_query(row):
    """Translate the producer schema to the report-only display schema."""
    diagnostic = row["diagnostic_L5"]
    clusters = []
    for master in row["masters"]:
        clusters.append({
            **master, "clip_rank": master["master_rank"],
            "true_max_bge": master["actual_max"], "gap": master["slack"],
            "has_gold": master["gold_count"] > 0,
            "action": "not_reached" if master["action_L5"] == "stop" else master["action_L5"],
        })
    trace = []
    for index, event in enumerate(diagnostic["trace"], 1):
        trace.append({**event, "step": index,
                      "cutoff_before": event.get("threshold_before"),
                      "cutoff_after": event.get("threshold_after")})
    last = diagnostic["trace"][-1] if diagnostic["trace"] else {}
    unvisited = set(diagnostic["skipped_master_ids"]) | set(diagnostic["unvisited_master_ids"])
    max_unvisited = max((m["upper_bound"] for m in row["masters"] if m["master_id"] in unvisited),
                        default=None)
    return {
        **row, "query_id": row["id"], "clusters": clusters,
        "gold_ids": [gold["filename"] for gold in row["gold_targets"]],
        "search": {
            **diagnostic, "visited_clusters": diagnostic["visited_master_count"],
            "scored_leaves": diagnostic["scored_leaf_count"],
            "top_l_exact": diagnostic["exact_match"],
            "final_cutoff": last.get("threshold_after"),
            "max_remaining_upper_bound": max_unvisited,
        },
        "trace": trace, "returned_leaves": diagnostic["selected_leaves"],
    }


def _number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _summary(results, protocol):
    scopes = results["scopes"]
    all_questions = scopes["all_questions"]
    l5 = next(b for b in all_questions["budgets"] if b["L"] == 5)
    new_scope = scopes["new_questions"]
    new_l5 = next(b for b in new_scope["budgets"] if b["L"] == 5)
    index = results["index_stats"]
    names = {"new_questions": "新增问题", "all_questions": "全部问题（含已有对照题）"}
    lines = [
        '<div class="table-wrap"><table><thead><tr><th>范围</th><th>问题数</th>'
        '<th>问题 × master 对数</th><th>上界违例</th><th>平均间隙</th>'
        '<th>间隙中位数</th><th>间隙 P90</th></tr></thead><tbody>'
    ]
    for key, scope in scopes.items():
        lines.append("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in [
            names.get(key, key), scope["n_questions"], scope["n_query_master_pairs"],
            scope["bound_violations"], _number(scope["slack_mean"]),
            _number(scope["slack_median"]), _number(scope["slack_p90"]),
        ])+"</tr>")
    lines.append("</tbody></table></div>")
    lines.append(
        f'<p>固定记忆库含 {index["n_clusters"]} 个 master；文本索引维度为 '
        f'{index["dimension"]}。候选返回量逐一检查 L = 1…10；本页面逐题轨迹展示 L = 5。</p>'
    )
    lines.append(
        '<div class="note"><b>界有效，但本版较松，尚无效率收益。</b>'
        f'在 {new_scope["n_questions"]} 道新增问题、L = 5 的诊断中，平均仍需访问 '
        f'{new_l5["mean_visited_masters"]:.2f} 个簇、评分 '
        f'{new_l5["mean_scored_leaves"]:.2f} 条叶子。加上 {index["n_clusters"]} 个文本中心点积，'
        f'平均为 {new_l5["mean_total_vector_dots"]:.2f} 次向量点积，'
        '多于直接评分全部 89 条叶子；当前不能据此宣称提速。</div>'
    )
    costs = [
        f'<p>文本索引的数值存储量：<b>{index["numeric_bytes"]:,} 字节</b>。'
        '该统计不等同于全部记忆库文件体积。</p>',
        '<p>“向量点积总数”包含全部 master 的文本上界计算，以及真正访问的叶子评分。'
        '这是计算次数对照，不代表实测运行时间或已确认的加速。</p>',
        '<div class="table-wrap"><table><thead><tr><th>范围</th><th>L</th><th>Top-L 完全一致</th>'
        '<th>平均访问簇 [最少, 最多]</th><th>平均评分叶子 [最少, 最多]</th>'
        '<th>平均向量点积数</th><th>平均 CLIP 扫描深度</th><th>提前停止题数</th></tr></thead><tbody>',
    ]
    for key, scope in scopes.items():
        for budget in scope["budgets"]:
            values = [
                names.get(key, key), budget["L"], f'{budget["exact_matches"]}/{scope["n_questions"]}',
                f'{_number(budget["mean_visited_masters"],2)} '
                f'[{budget["min_visited_masters"]}, {budget["max_visited_masters"]}]',
                f'{_number(budget["mean_scored_leaves"],2)} '
                f'[{budget["min_scored_leaves"]}, {budget["max_scored_leaves"]}]',
                _number(budget["mean_total_vector_dots"],2), _number(budget["mean_clip_depth"],2),
                budget["stopped_early_count"],
            ]
            costs.append("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in values)+"</tr>")
    costs.append("</tbody></table></div>")
    method = (
        "<p>每个簇保留文本索引统计量；查询到达后，从这些统计量计算该簇可能达到的最高文本相似度。"
        "访问该簇之前不读取其真实最高分。已收集至少 L 条叶子后，当前第 L 名分数构成可用的下界；"
        "如果未访问簇的上界无法超过它，即可跳过或停止。相等分数遵循确定性排序规则。</p>"
        "<p>当前使用覆盖全部簇成员的球冠区域。对单位文本向量 x，令 c = mean(x)，u = c / ||c||，"
        "a = min(u·x)，单位查询 q 对中心方向的分数为 b = q·u。若 b ≥ a，上界为 1；"
        "否则上界为 a·b + √((1−a²)(1−b²))。实际实现会对 float64 舍入误差做保守扩张。</p>"
        "<p>上界允许在整个覆盖区域内寻找最相似的位置，真实最高分只能来自实际存储的叶子。"
        "区域中可能没有叶子的地方也被包含，因此真实分数可以明显低于界；这正是界较松的原因。</p>"
        "<p>保证的是返回的叶子与同一 BGE 分数、同一排序规则下的全量 Top-L 一致。"
        "这不保证 gold 一定进入 Top-L，也不等同于问答模型答对。</p>"
        "<details><summary>实验协议与具体上界公式</summary><pre>"
        + html.escape(json.dumps(protocol, ensure_ascii=False, indent=2)) + "</pre></details>"
    )
    return {
        "leaf_limit": 5,
        "bound_checks_display": f'{all_questions["n_questions"]} × {index["n_clusters"]}',
        "bound_violations": all_questions["bound_violations"],
        "exact_top_l_display": f'{sum(b["exact_matches"] for b in all_questions["budgets"])}/'
                              f'{all_questions["n_questions"]*len(all_questions["budgets"])}',
        "mean_visits_display": f'{new_l5["mean_visited_masters"]:.2f} / {new_l5["mean_scored_leaves"]:.2f}',
        "scope_html": "".join(lines), "cost_html": "".join(costs), "method_html": method,
    }


_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home：未访问簇的文本相似度上界</title>
<style>
:root{--ink:#24344c;--muted:#64748b;--line:#dce4ed;--bg:#f3f6fa;--blue:#2365a2;--green:#167251;--orange:#ac611b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}
main{max-width:1450px;margin:0 auto;padding:30px 24px 60px}h1{font-size:28px;line-height:1.35;margin:0 0 12px}
h2{font-size:21px;margin:0 0 10px}h3{font-size:17px;margin:12px 0 8px}p{margin:8px 0 14px}
.muted,small{color:var(--muted)}.lead{max-width:1120px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}
.card,.panel{background:white;border:1px solid var(--line);border-radius:10px}.card{padding:15px}
.card .value{font-size:25px;font-weight:700}.card .label{color:var(--muted);font-size:13px}
.panel{padding:20px;margin-top:18px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}
input,select,button{font:inherit;border:1px solid #bac8d7;border-radius:6px;padding:8px 11px;background:white;color:var(--ink)}
input{min-width:220px}select{flex:1;min-width:220px}button{cursor:pointer}button:hover{background:#ecf2f8}button:disabled{opacity:.4;cursor:default}
.note{border-left:4px solid var(--blue);background:#eef5fb;padding:10px 14px;margin:14px 0}
.bad{color:#b13535}.good{color:var(--green)}.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:#f0f4f8;white-space:nowrap;position:sticky;top:0}
tr:last-child td{border-bottom:0}tbody tr:hover{background:#f7faff}.number{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.gold-row{background:#fff9ec}.badge{display:inline-block;padding:2px 7px;border-radius:4px;background:#eaf0f6;font-size:12px;white-space:nowrap}
.badge.visit{background:#e7f2ff;color:#246397}.badge.skip{background:#e6f4ee;color:#167251}.badge.gold{background:#fff0c9;color:#8b5c0d}
.query-text{font-size:17px;white-space:pre-wrap}.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.score{font-variant-numeric:tabular-nums}
details{margin:10px 0;border:1px solid var(--line);border-radius:6px;padding:10px 12px}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#f6f8fb;padding:12px;border-radius:6px}
.result-leaf{border-bottom:1px solid var(--line);padding:8px 0}.result-leaf:last-child{border:0}.hidden{display:none}
.empty{padding:20px;color:var(--muted)}a{color:var(--blue)}.trace-event{border-left:3px solid #c9d8e7;padding:4px 14px;margin:15px 0}
.trace-event.skip{border-color:#57a486}.trace-event.stop{border-color:#e1a447}
@media(max-width:850px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.split{grid-template-columns:1fr}h1{font-size:24px}}
@media print{body{background:white}main{max-width:none}.toolbar{display:none}.panel{break-inside:avoid}.table-wrap{overflow:visible}}
</style></head>
<body><main>
<h1>Home：未访问簇的文本相似度上界</h1>
<p class="lead">对每个问题，先用 CLIP 对视觉 master 排序，再用簇的文本索引计算 BGE 相似度上界。
本报告把上界与簇内真实最高分放在一起，检查是否安全跳过尚未访问的簇。</p>
<div class="note"><b>诊断目标是最终取得 L = <span id="leaf-limit">5</span> 条叶子记忆。</b>
L 表示返回的叶子数量，并非需要访问的簇数，也未被设为正式检索默认值。
真实最高 BGE 分数和 gold 标记仅用于事后验证，不参与跳过判断。</div>
<div id="cards" class="cards"></div>
<section class="panel">
<h2>实验范围与索引成本</h2><div id="scope"></div>
<details><summary>索引统计和计算成本</summary><div id="cost"></div></details>
<details><summary>上界公式与实验限制</summary><div id="method"></div></details>
</section>
<section class="panel">
<h2>逐题检查</h2>
<div class="toolbar">
<input id="query-search" type="search" aria-label="搜索问题" placeholder="搜索题号或问题文本">
<button id="previous" type="button" aria-label="上一题">上一题</button>
<select id="query-select" aria-label="选择问题"></select>
<button id="next" type="button" aria-label="下一题">下一题</button>
</div>
<div id="query-header"></div>
<div class="split">
<div><h3>访问情况</h3><div id="query-stats"></div></div>
<div><h3>停止判断</h3><div id="stop-reason"></div></div>
</div>
<h3>所有 master：按 CLIP 排名展示</h3>
<p class="muted">“间隙 = 上界 − 真实最高 BGE”。间隙非负表示该簇真实分数没有超过界；间隙越小，界越紧。
点击一行可查看该簇的分数和验证细节。表中保留六位小数，原始记录保留完整精度。</p>
<div class="table-wrap"><table><thead><tr>
<th>CLIP 名次</th><th>Master</th><th class="number">叶子数</th>
<th class="number">BGE 上界</th><th class="number">真实最高 BGE</th><th class="number">间隙</th><th>含 gold</th><th>访问决策</th>
</tr></thead><tbody id="cluster-body"></tbody></table></div>
<div id="cluster-detail" class="hidden"></div>
<h3>动态访问、跳过和停止轨迹</h3>
<p class="muted">访问后更新“当前第 L 名叶子的 BGE 分数”。只有已经拿到至少 L 条候选后，这个分数才可用于剪枝；
具体相等分数的处理遵循实验中的确定性排序规则。</p>
<div id="trace"></div>
<div class="split">
<div><h3>动态过程返回的叶子</h3><div id="returned-leaves"></div></div>
<div><h3>Gold 参考（仅用于评估）</h3><div id="exact-leaves"></div></div>
</div>
<details><summary>该题原始审计记录</summary><pre id="raw-query"></pre></details>
</section>
<section class="panel">
<h2>可追溯数据</h2>
<p><a href="summary.json">结果概要</a> · <a href="query_audit.jsonl">逐题记录</a> ·
<a href="protocol.json">实验协议</a> · <a href="validation.json">验证记录</a></p>
<p class="muted">本页面为自包含报告，无外部脚本、模型调用和原图副本。展示的分数与决策来自已完成的实验产物。</p>
</section>
</main>
<script id="report-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("report-data").textContent);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num=(v,d=6)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(d):"—";
const pick=(o,...keys)=>{for(const k of keys){if(o&&o[k]!==undefined)return o[k];}return undefined;};
const yes=v=>v===true?"是":v===false?"否":"—";
const field=(label,value)=>`<p><span class="muted">${esc(label)}：</span>${value}</p>`;
const raw=v=>`<pre>${esc(JSON.stringify(v,null,2))}</pre>`;
const shortId=v=>String(v??"").split("/").pop();
const actionName=v=>({visit:"访问",visited:"已访问",skip:"跳过",skipped:"已跳过",stop:"停止",unvisited:"未访问",bound_pruned:"上界剪枝",not_reached:"停止后未访问"}[v]??v??"—");
const reasonName=v=>({all_remaining_bounds_below_current_cutoff:"所有尚未扫描簇的上界都低于当前第 L 名，安全停止",
all_clusters_considered:"全部簇已扫描；部分簇通过上界判断跳过"}[v]??v??"—");
function decisionExplanation(step){
 if(!step)return "";
 const cutoff=step.threshold_before??step.cutoff_before;
 if(step.action==="stop")return `剩余簇最高界 ${num(step.remaining_max_before)} < 当前第 L 名 ${num(cutoff)}，可以停止。`;
 if(step.action==="skip")return `该簇上界 ${num(step.upper_bound)} < 当前第 L 名 ${num(cutoff)}，不必对簇内叶子评分。`;
 if(step.action==="visit"&&cutoff===null)return "尚未获得 L 条候选叶子，先访问该簇。";
 if(step.action==="visit")return `该簇上界 ${num(step.upper_bound)} 仍可能达到或超过当前第 L 名 ${num(cutoff)}，需要评分确认。`;
 return "";
}
const items=v=>Array.isArray(v)?v:[];
const queries=DATA.queries;
const summary=DATA.summary;
document.getElementById("leaf-limit").textContent=summary.leaf_limit??5;
const cardData=[
 ["问题 × master 验证",summary.bound_checks_display??"—","上界是否覆盖簇内最高真实分数"],
 ["上界违例",summary.bound_violations??"—","仅事后核对，不参与检索"],
 ["Top-L 完全一致",summary.exact_top_l_display??"—","全部问题，L = 1…10"],
 ["平均访问簇 / 叶",summary.mean_visits_display??"—","51 道新增问题，L = 5"]
];
document.getElementById("cards").innerHTML=cardData.map(([l,v,n])=>`<div class="card"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><small>${esc(n)}</small></div>`).join("");
document.getElementById("scope").innerHTML=summary.scope_html??raw(DATA.results);
document.getElementById("cost").innerHTML=summary.cost_html??raw(DATA.results.storage??DATA.results.cost??{});
document.getElementById("method").innerHTML=summary.method_html??raw(DATA.protocol);
const select=document.getElementById("query-select");
let visible=queries.map((q,i)=>i);
const label=q=>`${q.query_id??q.id??"?"} · ${String(q.question??q.query??"").slice(0,100)}`;
function fillSelect(preferred){
 select.innerHTML=visible.map(i=>`<option value="${i}">${esc(label(queries[i]))}</option>`).join("");
 if(visible.includes(preferred))select.value=String(preferred);
 showQuery();
}
function displayLeaves(leaves){
 if(!items(leaves).length)return '<div class="empty">记录未提供叶子列表。</div>';
 return items(leaves).map((x,i)=>{
  const id=typeof x==="string"?x:pick(x,"image_id","leaf_id","id");
  const score=typeof x==="object"?pick(x,"score","bge_score","similarity"):undefined;
  return `<div class="result-leaf"><b>${i+1}. ${esc(x.filename??shortId(id))}</b> <span class="score">${num(score)}</span>${x.gold||x.is_gold?'<span class="badge gold">gold</span>':""}</div>`;
 }).join("");
}
function showCluster(q,c){
 const box=document.getElementById("cluster-detail");
 box.classList.remove("hidden");
 const bound=pick(c,"upper_bound","bound","bge_upper_bound"), actual=pick(c,"true_max_bge","true_max","exact_max_score");
 const event=items(q.trace).find(t=>t.master_id===c.master_id)??items(q.trace).find(t=>t.action==="stop");
 box.innerHTML=`<details open><summary>${esc(c.master_id??c.id)}：验证明细</summary>
 <p>真实最高分 ${num(actual)} ${actual<=bound+1e-10?"≤":"&gt;"} 上界 ${num(bound)}；间隙 ${num(pick(c,"gap","bound_gap")??(bound-actual))}。</p>
 <p>查询与簇平均文本向量的点积：${num(c.mean_score)}；球界（对照）：${num(c.ball_upper_bound)}。</p>
 <p>${esc(decisionExplanation(event))}</p>
 <p class="muted">覆盖整个簇的区域可以包含并无实际叶子的位置，因此界会高于真实最高分。
 这里的真实分数来自事后全量验证；访问前只能使用上界。</p>${raw(c)}</details>`;
}
function displayTrace(trace){
 if(!items(trace).length)return '<div class="empty">记录未提供逐步轨迹。</div>';
 return items(trace).map((step,i)=>{
  const action=pick(step,"action","decision","event");
  const mid=pick(step,"master_id","cluster_id");
  const cutoff=pick(step,"cutoff_before","threshold_before","kth_before","current_kth_score");
  const after=pick(step,"cutoff_after","threshold_after","kth_after");
  const bound=pick(step,"upper_bound","bound","max_unvisited_upper_bound");
  const reason=pick(step,"reason","explanation");
  return `<div class="trace-event ${esc(action)}"><b>步骤 ${step.step??i+1} · ${esc(actionName(action))}${mid?" · "+esc(mid):""}</b>
  <div>当前簇上界 ${num(bound)}；处理前第 L 名 ${num(cutoff)}；处理后第 L 名 ${num(after)}</div>
  <div>剩余簇最高界 ${num(step.remaining_max_before)}；剩余簇数 ${esc(step.remaining_count??"—")}；本步叶子评分 ${esc(step.n_scores??"—")}</div>
  <div>${esc(reason??decisionExplanation(step))}</div><details><summary>该步审计值</summary>${raw(step)}</details></div>`;
 }).join("");
}
function showQuery(){
 const i=Number(select.value),q=queries[i];
 if(!q||!visible.includes(i)){
  document.getElementById("query-header").innerHTML='<div class="empty">没有匹配的问题。</div>';
  ["query-stats","stop-reason","cluster-body","trace","returned-leaves","exact-leaves","raw-query"].forEach(id=>document.getElementById(id).textContent="");
  document.getElementById("cluster-detail").classList.add("hidden");
  document.getElementById("previous").disabled=true;document.getElementById("next").disabled=true;return;
 }
 const pos=visible.indexOf(i);
 document.getElementById("previous").disabled=pos<=0;
 document.getElementById("next").disabled=pos>=visible.length-1;
 document.getElementById("cluster-detail").classList.add("hidden");
 document.getElementById("query-header").innerHTML=`<h3>${esc(q.query_id??q.id)}</h3><p class="query-text">${esc(q.question??q.query??"")}</p>
 <p class="muted">Gold：${esc(items(q.gold_ids??q.gold_image_ids).map(shortId).join("；")||"记录未提供")}</p>`;
 const stats=q.search??q.retrieval??q;
 document.getElementById("query-stats").innerHTML=
  field("访问簇",esc(pick(stats,"visited_clusters","n_visited_clusters","clusters_visited")??"—"))+
  field("实际评分叶子",esc(pick(stats,"scored_leaves","n_scored_leaves","leaves_scored")??"—"))+
  field("与全量 Top-L 一致",yes(pick(stats,"top_l_exact","exact_top_l","matches_exact")));
 document.getElementById("stop-reason").innerHTML=
  field("停止原因",esc(reasonName(pick(stats,"stop_reason","reason")??"见下方轨迹")))+
  field("最终第 L 名分数",num(pick(stats,"final_cutoff","kth_score","cutoff")))+
  field("未访问簇最大上界",num(pick(stats,"max_remaining_upper_bound","max_unvisited_bound")));
 const clusters=items(q.clusters??q.masters).slice().sort((a,b)=>(a.clip_rank??a.rank??0)-(b.clip_rank??b.rank??0));
 document.getElementById("cluster-body").innerHTML=clusters.map((c,idx)=>{
  const bound=pick(c,"upper_bound","bound","bge_upper_bound"), actual=pick(c,"true_max_bge","true_max","exact_max_score");
  const gap=pick(c,"gap","bound_gap")??(bound-actual), gold=pick(c,"has_gold","contains_gold");
  const action=pick(c,"action","decision","status");
  return `<tr tabindex="0" data-cluster="${idx}" class="${gold?"gold-row":""}">
   <td>${esc(c.clip_rank??c.rank??idx+1)}</td><td class="mono">${esc(c.master_id??c.id)}</td>
   <td class="number">${esc(c.n??c.n_leaves??c.size??"—")}</td>
   <td class="number">${num(bound)}</td><td class="number">${num(actual)}</td>
   <td class="number ${gap< -1e-10?"bad":"good"}">${num(gap)}</td><td>${gold?'<span class="badge gold">是</span>':"否"}</td>
   <td><span class="badge ${esc(action??"")}">${esc(actionName(action))}</span></td></tr>`;
 }).join("");
 document.querySelectorAll("[data-cluster]").forEach(el=>{
  const activate=()=>showCluster(q,clusters[Number(el.dataset.cluster)]);
  el.addEventListener("click",activate);el.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" ")activate();});
 });
 document.getElementById("trace").innerHTML=displayTrace(q.trace??stats.trace??stats.events);
 document.getElementById("returned-leaves").innerHTML=displayLeaves(q.returned_leaves??stats.returned_leaves??stats.top_l);
 document.getElementById("exact-leaves").innerHTML=items(q.gold_targets).map(g=>
  `<div class="result-leaf"><b>${esc(g.filename??shortId(g.image_id))}</b>
  <div>CLIP master 名次 ${esc(g.clip_master_rank??"—")}；全量叶子 BGE 名次 ${esc(g.global_leaf_text_rank??"—")}</div>
  <div class="mono">${esc(g.master_id??"")}</div></div>`).join("")||'<div class="empty">没有 gold 参考。</div>';
 document.getElementById("raw-query").textContent=JSON.stringify(q,null,2);
}
select.addEventListener("change",showQuery);
document.getElementById("query-search").addEventListener("input",e=>{
 const term=e.target.value.toLowerCase(),preferred=Number(select.value);
 visible=queries.map((q,i)=>i).filter(i=>label(queries[i]).toLowerCase().includes(term));
 fillSelect(preferred);
});
document.getElementById("previous").addEventListener("click",()=>{
 const pos=visible.indexOf(Number(select.value));if(pos>0){select.value=String(visible[pos-1]);showQuery();}
});
document.getElementById("next").addEventListener("click",()=>{
 const pos=visible.indexOf(Number(select.value));if(pos+1<visible.length){select.value=String(visible[pos+1]);showQuery();}
});
fillSelect(0);
</script></body></html>
"""


def render(output: Path) -> Path:
    output = Path(output).resolve()
    results = _read(output / "summary.json")
    protocol = _read(output / "protocol.json")
    validation = _read(output / "validation.json")
    queries = [_normalize_query(row) for row in _jsonl(output / "query_audit.jsonl")]
    summary = _summary(results, protocol)
    payload = {"results": results, "protocol": protocol, "validation": validation,
               "summary": summary, "queries": queries}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    target = output / "report.html"
    target.write_text(_PAGE.replace("__DATA__", encoded), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(render(args.output))


if __name__ == "__main__":
    main()
