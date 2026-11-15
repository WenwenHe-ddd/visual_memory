#!/usr/bin/env python3
"""Build a self-contained PCA explorer from a report's plot_data.json."""

import argparse
import json
import math
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>重构视觉向量 PCA 查看器</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#172633;background:#f3f6f9}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;padding:24px}main{max-width:1560px;margin:auto}
h1{font-size:24px;margin:0 0 10px}p{line-height:1.7;margin:8px 0}small{color:#536777}
.notice{background:#fff7df;border-left:4px solid #bc8a21;padding:12px 16px;margin:14px 0}
.controls,.metrics{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:16px 0}
label{display:flex;align-items:center;gap:8px}select,button{font:inherit;border:1px solid #cbd4dc;border-radius:7px;background:white;padding:7px 10px;color:inherit}
button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #3c87bc;outline-offset:2px}
.metrics>div{background:white;border:1px solid #e0e6ec;border-radius:8px;padding:10px 14px}
.metrics strong{display:block;margin-top:4px;font-variant-numeric:tabular-nums}
.layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,360px);gap:18px}
.card{background:white;border:1px solid #dce4eb;border-radius:12px;padding:16px;min-width:0}
svg{display:block;width:100%;height:auto;aspect-ratio:960/650;overflow:visible}
svg text{font-size:12px;fill:#526777}svg .marker{cursor:pointer}svg .marker:focus{outline:none;filter:drop-shadow(0 0 3px #111)}
.legend{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px;max-height:220px;overflow:auto;padding:4px}
.legend button{font-size:12px;display:flex;align-items:center;gap:7px}
.legend button[aria-pressed="true"]{border-color:#195e8b;box-shadow:0 0 0 1px #195e8b;background:#edf6fc}
.swatch{width:11px;height:11px;border-radius:50%;display:inline-block;flex-shrink:0}
h2{font-size:17px;margin:0 0 12px}dl{margin:0}dt{font-weight:650;color:#526777;margin-top:16px;font-size:13px}
dd{margin:5px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65;font-size:14px}
.hint{color:#536777;font-size:13px}#selection{margin-bottom:12px}#error{color:#a31f1f}
#memberChooser{display:block;font-size:13px;margin-bottom:16px}#member{display:block;width:100%;margin-top:7px}
@media(max-width:850px){body{padding:12px}.layout{grid-template-columns:1fr}.card{padding:10px}h1{font-size:21px}}
</style>
<main>
<h1>重构视觉向量 · PCA 查看器</h1>
<p class="hint">圆点：图像叶子向量；×：master。横纵轴使用相同单位比例。点击圆点或 × 查看详情，点击图例突出一个簇，其余簇变淡。</p>
<div class="notice">颜色来自已有的 caption 簇，PCA 仅用于可视化。重构过程中的共享残差会天然使同簇向量靠近；图中的分离程度与轮廓系数不能视为原图聚类真值，也不能证明原始视觉空间具有相同结构。</div>
<div class="controls">
<label>数据域 <select id="domain"></select></label>
<label>向量模式 <select id="mode"><option value="raw">原始重构向量（raw）</option><option value="normalized">归一化向量（normalized）</option></select></label>
</div>
<div id="metrics" class="metrics"></div>
<p id="error" role="alert"></p>
<div class="layout">
<section class="card" aria-label="PCA 散点图">
<svg id="plot" viewBox="0 0 960 650" preserveAspectRatio="xMidYMid meet" role="group" aria-label="重构视觉向量 PCA 散点图"></svg>
<p id="selection" class="hint" aria-live="polite"></p>
<div id="legend" class="legend" aria-label="簇筛选"></div>
</section>
<aside class="card"><h2>所选点详情</h2><label id="memberChooser" hidden>所选簇图像（可选择被遮挡的点）<select id="member"></select></label><p id="detailHint" class="hint">点击图中的点查看原始记录。</p><dl id="details"></dl></aside>
</div>
<p class="hint">此页离线运行，全部坐标和指标来自同目录 plot_data.json 的内嵌快照；不加载图片、原始编码向量或外部服务。图像数量和簇规模均按输入报告显示。</p>
</main>
<script id="plot-data" type="application/json">__PLOT_DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("plot-data").textContent);
const el = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
let selectedCluster = null;
let selectedPoint = null;
const textValue = value => value == null ? "未提供" : (typeof value === "object" ? JSON.stringify(value, null, 2) : String(value));
const number = value => typeof value === "number" && Number.isFinite(value) ? value.toLocaleString("zh-CN", {maximumSignificantDigits:5}) : "未提供";
const percent = value => typeof value === "number" && Number.isFinite(value) ? number(value * 100) + "%" : "未提供";
const color = value => typeof value === "string" && /^#[0-9a-f]{3,8}$/i.test(value) ? value : "#667788";
const sameCluster = item => selectedCluster === null || String(item.cluster_label) === selectedCluster;
function svg(tag, attrs = {}, content) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (content !== undefined) node.textContent = content;
  return node;
}
function currentDomain() { return DATA.domains[Number(el("domain").value)]; }
function field(label, value) {
  const dt = document.createElement("dt"), dd = document.createElement("dd");
  dt.textContent = label; dd.textContent = textValue(value);
  el("details").append(dt, dd);
}
function showDetails(item, isMaster) {
  selectedPoint = {item, isMaster};
  el("detailHint").textContent = isMaster ? "已选择 master（×）。" : "已选择图像叶子（圆点）。";
  el("details").replaceChildren();
  field("类型", isMaster ? "master" : "图像叶子");
  if (!isMaster) { field("图像 ID", item.image_id); field("文件名", item.filename); }
  field("Master ID", item.master_id);
  field("簇标签", item.cluster_label);
  if (isMaster) {
    field("簇内图像数", item.n);
    field("近似 master", typeof item.approximate === "boolean" ? (item.approximate ? "是" : "否") : item.approximate);
  } else { field("Caption", item.caption); field("Dialogue", item.dialogue); }
  field("当前 PCA 坐标（PC1, PC2）", item[el("mode").value]);
  el("member").value = !isMaster && sameCluster(item) ? String(currentDomain().points.indexOf(item)) : "";
}
function resetDetails() {
  selectedPoint = null; el("details").replaceChildren();
  el("detailHint").textContent = "点击图中的点查看原始记录。";
}
function metric(label, value) {
  const box = document.createElement("div"), name = document.createElement("small"), val = document.createElement("strong");
  name.textContent = label; val.textContent = value; box.append(name, val); el("metrics").append(box);
}
function drawLegend(domain) {
  el("legend").replaceChildren();
  function button(label, cluster, shade) {
    const b = document.createElement("button");
    b.type = "button"; b.setAttribute("aria-pressed", String(selectedCluster === cluster));
    if (shade) { const dot = document.createElement("span"); dot.className = "swatch"; dot.style.backgroundColor = color(shade); b.append(dot); }
    const name = document.createElement("span"); name.textContent = label; b.append(name);
    b.addEventListener("click", () => { selectedCluster = selectedCluster === cluster ? null : cluster; render(); });
    el("legend").append(b);
  }
  button("全部簇", null);
  for (const m of domain.masters) button("簇 " + textValue(m.cluster_label) + " · " + number(m.n) + " 张", String(m.cluster_label), m.color);
  el("selection").textContent = selectedCluster === null ? "当前突出：全部簇" : "当前突出：簇 " + selectedCluster + "（其余簇变淡）";
  el("memberChooser").hidden = selectedCluster === null;
  el("member").replaceChildren();
  const placeholder=document.createElement("option"); placeholder.value=""; placeholder.textContent="选择图像查看详情";
  el("member").append(placeholder);
  if(selectedCluster !== null) for(const [index,point] of domain.points.entries()) {
    if(!sameCluster(point)) continue;
    const option=document.createElement("option"); option.value=String(index);
    option.textContent=textValue(point.image_id)+" · "+textValue(point.filename);
    el("member").append(option);
  }
}
function drawPlot(domain, mode) {
  const root = el("plot"); root.replaceChildren();
  const records = [...domain.points, ...domain.masters];
  if (!records.length) { root.append(svg("text", {x:480,y:325,"text-anchor":"middle"}, "该数据域没有坐标")); return; }
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const p of records) { const [x,y]=p[mode]; xmin=Math.min(xmin,x); xmax=Math.max(xmax,x); ymin=Math.min(ymin,y); ymax=Math.max(ymax,y); }
  // Both axes share one scale; padding only affects the viewport, never coordinates.
  const span = Math.max(xmax-xmin, ymax-ymin) || 1;
  const xpad = Math.max(xmax-xmin, span * 0.05) * 0.09;
  const ypad = Math.max(ymax-ymin, span * 0.05) * 0.09;
  xmin-=xpad; xmax+=xpad; ymin-=ypad; ymax+=ypad;
  const scale = Math.min(845/(xmax-xmin), 545/(ymax-ymin));
  const width=(xmax-xmin)*scale, height=(ymax-ymin)*scale;
  const left=80+(845-width)/2, top=35+(545-height)/2;
  const px=x=>left+(x-xmin)*scale, py=y=>top+(ymax-y)*scale;
  for (let i=0;i<=5;i++) {
    const x=xmin+(xmax-xmin)*i/5, y=ymin+(ymax-ymin)*i/5;
    root.append(svg("line",{x1:px(x),x2:px(x),y1:top,y2:top+height,stroke:"#e8edf2"}));
    root.append(svg("line",{x1:left,x2:left+width,y1:py(y),y2:py(y),stroke:"#e8edf2"}));
    root.append(svg("text",{x:px(x),y:top+height+22,"text-anchor":"middle"},number(x)));
    root.append(svg("text",{x:left-12,y:py(y)+4,"text-anchor":"end"},number(y)));
  }
  root.append(svg("rect",{x:left,y:top,width,height,fill:"none",stroke:"#aebfcb"}));
  const variance=domain.metrics[mode].explained_ratio || [];
  root.append(svg("text",{x:left+width/2,y:635,"text-anchor":"middle"},"PC1 · 解释方差 " + percent(variance[0])));
  root.append(svg("text",{transform:"translate(18 " + (top+height/2) + ") rotate(-90)","text-anchor":"middle"},"PC2 · 解释方差 " + percent(variance[1])));
  function marker(item, master) {
    const [x,y]=item[mode], cx=px(x), cy=py(y), c=color(item.color);
    const node=svg("g",{class:"marker",tabindex:0,role:"button",opacity:sameCluster(item)?(master?1:0.74):0.08,"aria-label":(master?"Master "+item.master_id:"图像 "+item.image_id)+"，簇 "+item.cluster_label});
    node.append(svg("title",{},(master?"Master: "+item.master_id:"图像: "+item.image_id)+"\n簇: "+item.cluster_label+"\nPC1: "+x+"\nPC2: "+y));
    if (master) {
      node.append(svg("circle",{cx,cy,r:11,fill:"transparent"}));
      node.append(svg("path",{d:`M ${cx-6} ${cy-6} L ${cx+6} ${cy+6} M ${cx-6} ${cy+6} L ${cx+6} ${cy-6}`,stroke:"white","stroke-width":5,fill:"none"}));
      node.append(svg("path",{d:`M ${cx-6} ${cy-6} L ${cx+6} ${cy+6} M ${cx-6} ${cy+6} L ${cx+6} ${cy-6}`,stroke:c,"stroke-width":2.5,fill:"none"}));
    } else node.append(svg("circle",{cx,cy,r:3.5,fill:c,stroke:"white","stroke-width":0.45}));
    node.addEventListener("click",()=>showDetails(item,master));
    node.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();showDetails(item,master);}});
    return node;
  }
  // Draw dimmed points first so selected clusters remain easy to inspect.
  for (const selected of [false,true]) {
    for (const point of domain.points) if (sameCluster(point)===selected) root.append(marker(point,false));
    for (const master of domain.masters) if (sameCluster(master)===selected) root.append(marker(master,true));
  }
}
function render() {
  const domain=currentDomain(), mode=el("mode").value, m=domain.metrics[mode];
  el("metrics").replaceChildren();
  metric("图像数",number(domain.n_images)); metric("Master 数",number(domain.n_masters));
  metric("PCA 2D 累计解释方差",percent(m.pca2_explained_ratio));
  metric("512D 轮廓系数 · 欧氏距离",number(m.silhouette_512d_euclidean));
  metric("PCA 2D 轮廓系数 · 欧氏距离",number(m.silhouette_pca2_euclidean));
  drawPlot(domain,mode); drawLegend(domain);
  if(selectedPoint) showDetails(selectedPoint.item,selectedPoint.isMaster);
}
for (const [index, domain] of DATA.domains.entries()) {
  const option=document.createElement("option"); option.value=String(index);
  option.textContent=domain.short_domain ? domain.short_domain+" · "+domain.domain : domain.domain;
  el("domain").append(option);
}
el("domain").addEventListener("change",()=>{selectedCluster=null;resetDetails();render();});
el("mode").addEventListener("change",render);
el("member").addEventListener("change",()=>{if(el("member").value!=="") showDetails(currentDomain().points[Number(el("member").value)],false);});
if(DATA.domains.length) render(); else el("error").textContent="输入报告没有数据域。";
</script>
</html>
"""


def build_html(data: dict) -> str:
    """Validate drawable coordinates and safely embed the supplied report."""
    if not isinstance(data, dict) or not isinstance(data.get("domains"), list):
        raise ValueError("plot_data.json must contain a domains list")
    for domain in data["domains"]:
        for kind in ("points", "masters"):
            for item in domain[kind]:
                for mode in ("raw", "normalized"):
                    xy = item[mode]
                    if not isinstance(xy, list) or len(xy) != 2 or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in xy
                    ):
                        raise ValueError(f"Invalid {mode} coordinate in {domain['domain']} / {kind}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Escaping '<' prevents captions/dialogue from closing the JSON script element.
    for source, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(source, replacement)
    return HTML.replace("__PLOT_DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path,
                        help="Existing report directory containing plot_data.json")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve(strict=True)
    if not report_dir.is_dir():
        parser.error("--report-dir must be a directory")
    input_path, output_path = report_dir / "plot_data.json", report_dir / "explorer.html"
    # Do not follow an output link beyond the report directory.
    if output_path.is_symlink():
        parser.error("explorer.html must not be a symbolic link")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    html = build_html(data)
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
