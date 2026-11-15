"""Posthoc Brand comparison. This module is never imported by admission."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVALUATION = HERE.parent / "brand_dynamic_pilot_20261019/evaluation"
sys.path.insert(0, str(EVALUATION))
from evaluate_pilot import _inputs, evaluate, read_assignments


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    manifest, _, raw_distances = _inputs()
    index = {row["id"]: i for i, row in enumerate(manifest["images"])}
    filenames = {row["id"]: row["filename"] for row in manifest["images"]}
    orders = ["original", "reverse", "shuffle_20261019", "shuffle_20261020"]
    rows, reports, logs, all_assignments = [], {}, {}, {}
    for order in orders:
        for method in ("complete_link", "previous_nn"):
            directory = HERE / "runs" / order / method
            assignments = read_assignments(directory / "graph.json")
            report = evaluate(assignments, name=f"{method}/{order}")
            assert not report["errors"]
            write_json(directory / "evaluation.json", report)
            reports[(order, method)] = report
            all_assignments[(order, method)] = assignments
            run = json.loads((directory / "run_summary.json").read_text())
            log = [json.loads(line) for line in (directory / "admission_log.jsonl").read_text().splitlines()]
            logs[(order, method)] = log
            raw_diameters = [
                float(raw_distances[np.ix_([index[i] for i in members], [index[i] for i in members])].max())
                for members in report["clusters"]["memberships"].values()
            ]
            original = all_assignments[("original", method)]
            ids = list(assignments)
            disagreement = sum(
                (assignments[a] == assignments[b]) != (original[a] == original[b])
                for n, a in enumerate(ids) for b in ids[n+1:])
            sizes = report["clusters"]["sizes"]
            rows.append({
                "order": order, "method": method, "clusters": len(sizes),
                "sizes": sizes, "singleton_images": sizes.count(1),
                "singleton_image_fraction": sizes.count(1) / len(ids),
                "largest_cluster": max(sizes),
                "close_pairs_kept": report["visual_sanity"]["close_pairs_kept"],
                "close_pair_count": report["visual_sanity"]["close_pair_count"],
                "different_pairs_separated": report["visual_sanity"]["different_pairs_separated"],
                "different_pair_count": report["visual_sanity"]["different_pair_count"],
                "cosine_silhouette": report["geometry"]["cosine_silhouette"],
                "maximum_raw_clip_cluster_diameter": max(raw_diameters),
                "pair_assignments_different_from_original_out_of_435": disagreement,
                "runtime": run["runtime"], "storage": run["storage"], "audit": run["audit"],
            })
    original_log = logs[("original", "complete_link")]
    jumps = []
    for before, after in zip(original_log, original_log[1:]):
        if after["cluster_count"] > before["cluster_count"] + 1:
            jumps.append({
                "arrival": after["prefix_count"],
                "clusters_before": before["cluster_count"], "clusters_after": after["cluster_count"],
                "cut_before": before["cut"], "cut_after": after["cut"],
                "split_old_masters": after["splits"],
            })
    # Reproducing the previous strict results is a control, not a quality claim.
    baseline_match = {}
    old = HERE.parent / "brand_no_original_pilot_20261019"
    for order in orders:
        historical = old / ("original_order" if order == "original" else f"validation/{order}")
        baseline_match[order] = read_assignments(historical / "graph.json") == all_assignments[(order, "previous_nn")]
    assert all(baseline_match.values())
    audit_path = HERE / "independent_audit.json"
    independent = json.loads(audit_path.read_text()) if audit_path.exists() else {"status": "pending"}
    summary = {
        "status": "tested_on_brand_only; anti_chaining_improved_but_fragmentation_remains",
        "conclusion_zh": "完整链接加全局 Otsu 在 Brand 上偏碎；不建议直接替换当前方案。",
        "image_count": 30, "domain": "Brand_Memory_Test",
        "protocol": {
            "sequential_prefix_only": True, "max_visual_coordinates_per_image": 2,
            "retain_original_images_or_paths": False, "old_image_reencoding": False,
            "per_image_full_vector_cache": False, "extra_per_image_visual_scalars": False,
            "masters": "full mean, at most two basis directions, singular values and error metadata",
            "linkage": "complete", "cut": "exact Otsu on prefix merge heights",
            "hand_set_similarity_cutoff": None, "target_number_of_clusters": None,
            "prefix_two_distinct_points": "defer merge", "constant_heights_with_at_least_three_points": "keep all",
            "fresh_streams": 5, "encoder_calls_total": 132,
            "evaluation": "Frozen cases and existing scalar CLIP cosines, posthoc only; no original-image replay.",
        },
        "results": rows, "original_order_split_jumps": jumps,
        "baseline_exactly_reproduced": baseline_match,
        "prefix_audit": json.loads((HERE / "validation/prefix_audit.json").read_text()),
        "independent_audit": independent,
        "limitations": [
            "Frozen 6 close and 5 different pairs are AI diagnostic examples, not held-out human ground truth.",
            "Silhouette evaluates the same CLIP geometry, not independent human visual similarity.",
            "Complete-link controls diameter only in the approximate geometry at decision time.",
            "Low-rank recompression can change geometry afterwards; reverse stream has an observed transient violation.",
            "Full-prefix tree rebuilding is a small-domain quality reference, not an efficient large-scale index.",
            "Data-derived Otsu cut can jump; no hand-set cutoff does not imply no assumptions or stability.",
            "More small masters can remove the storage benefit: include master means and bases in comparisons.",
            "Local means after a lossy split remain estimates; no Brand true-centroid error was measured.",
        ],
    }
    write_json(HERE / "summary.json", summary)
    plot(rows, logs)
    esc = html.escape
    chunks = ["""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brand 完整链接：碎片化验证</title>
<style>body{max-width:1180px;margin:32px auto;padding:0 20px;color:#172033;background:#f6f7fb;font:16px/1.65 system-ui,sans-serif}
h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}
th,td{padding:9px;border:1px solid #dce1eb;text-align:left}.notice{padding:18px;background:#fff1d6;border-left:4px solid #cd8626}
details{background:white;padding:12px 18px;margin:12px 0;border-radius:8px}code{overflow-wrap:anywhere}img{width:100%;background:white}
.muted{color:#526078}a{color:#165bbd}</style><body>
<h1>Brand：完整链接是否会拆得太碎？</h1>
<p class="notice"><strong>会。这一版减少了大簇串联，但原顺序 30 图产生 23 簇，18 张图单独成簇。</strong>
四种顺序均出现较多单图簇；完整链接加全局 Otsu 还不适合作为正式替代方案。</p>
<p>所有实验只处理 Brand；逐张到达，输入临时图片在编码后即删除，每张图只保留最多两个坐标。
不同方法共享当次编码，之后各自更新独立记忆。每个流内没有旧图重编码。原始 benchmark 文件作为外部测试输入保留，记忆内没有原图或路径。</p>
<h2>原始顺序：与当前方案直接比较</h2><table><tr><th>指标</th><th>当前邻接图方案</th><th>完整链接 + Otsu</th></tr>"""]
    base = next(r for r in rows if r["order"] == "original" and r["method"] == "previous_nn")
    new = next(r for r in rows if r["order"] == "original" and r["method"] == "complete_link")
    for label, key in [("簇数量", "clusters"), ("单图簇", "singleton_images"),
                       ("最大簇图片数", "largest_cluster"), ("相似案例同簇（共 6 对）", "close_pairs_kept"),
                       ("不同案例分开（共 5 对）", "different_pairs_separated"),
                       ("CLIP silhouette", "cosine_silhouette")]:
        vals = [f"{r[key]:.4f}" if isinstance(r[key], float) else str(r[key]) for r in (base, new)]
        chunks.append(f"<tr><td>{label}</td><td>{vals[0]}</td><td>{vals[1]}</td></tr>")
    chunks.append("</table><p class='muted'>诊断案例来自此前冻结的模型目视检查，并非独立人工真值；silhouette 也不是人工相似度准确率。</p>")
    chunks.append("<img src='fragmentation.png' alt='Cluster sizes, prefix changes and singleton counts across orders'>")
    chunks.append("<h2>四种加入顺序</h2><table><tr><th>顺序</th><th>方法</th><th>簇</th><th>单图簇</th><th>最大簇</th><th>相似同簇</th><th>不同分开</th><th>相对原顺序关系变化 /435</th></tr>")
    for r in rows:
        chunks.append(f"<tr><td>{r['order']}</td><td>{'完整链接' if r['method']=='complete_link' else '当前邻接图'}</td>"
                      f"<td>{r['clusters']}</td><td>{r['singleton_images']}</td><td>{r['largest_cluster']}</td>"
                      f"<td>{r['close_pairs_kept']}/6</td><td>{r['different_pairs_separated']}/5</td>"
                      f"<td>{r['pair_assignments_different_from_original_out_of_435']}</td></tr>")
    chunks.append("</table><h2>具体相似案例</h2><table><tr><th>图片对</th><th>原方案</th><th>完整链接</th></tr>")
    base_cases = reports[("original", "previous_nn")]["visual_sanity"]["cases"]
    new_cases = reports[("original", "complete_link")]["visual_sanity"]["cases"]
    for a, b in zip(base_cases, new_cases):
        if b["expected_same_cluster_for_sanity_check"] is not True:
            continue
        chunks.append(f"<tr><td>{esc(filenames[b['left_id']])} / {esc(filenames[b['right_id']])}</td>"
                      f"<td>{'同簇' if a['pilot_same_cluster'] else '分开'}</td>"
                      f"<td>{'同簇' if b['pilot_same_cluster'] else '分开'}</td></tr>")
    chunks.append("</table><h2>动态过程暴露的问题</h2>"
                  "<p>没有手设阈值，但自动切点仍会跳变。例如第 9→10 张，距离切点约 0.496→0.191，"
                  "簇数从 4→9；第 20→21 张，约 0.389→0.144，簇数从 9→17。"
                  "一张新图就会触发多个旧簇拆分。完整链接检查最远成员比较保守，"
                  "全局二分切点和二维信息损失也共同影响结果；本次实验未拆分量化这三个因素的贡献。</p>")
    chunks.append("<h2>时间、存储和约束核对</h2>")
    chunks.append(f"<p>原顺序 30 图的聚类更新耗时：当前方案 {base['runtime']['admission_seconds']:.3f} 秒，"
                  f"完整链接 {new['runtime']['admission_seconds']:.3f} 秒。共享编码约 "
                  f"{new['runtime']['encoder_seconds_shared_between_methods']:.2f} 秒；完整链接另有 "
                  f"{new['runtime']['diagnostic_seconds']:.3f} 秒的压缩后几何审计。模型加载与后验评估不在这些时间内。</p>")
    chunks.append(f"<p>视觉数组：当前方案 {base['storage']['sketch_array_bytes']:,} 字节，完整链接 "
                  f"{new['storage']['sketch_array_bytes']:,} 字节。30×512 的完整 float64 向量为 122,880 字节，"
                  "float32 为 61,440 字节。大量小簇的中心和基底抵消了二维坐标的节省；这里没有净存储优势。"
                  "这些数值不含 JSON、日志、模型和临时工作空间。</p>")
    chunks.append("<p>前 12 张独立截断输入与完整流前 12 步一致；四种顺序的旧方案结果均精确复现此前结果。"
                  "每个全量流编码 30 次，截断流编码 12 次，共 132 次；跨实验从头运行不属于单个流内的旧图重编码。</p>")
    chunks.append("<p>完整链接的直径条件适用于划分时的近似表征。反向流在一次后续压缩后出现超出当步切点的簇；"
                  "没有宣称最终分组是固定点，也没有宣称能精确恢复原 embedding。完整前缀建树需要二次临时空间，"
                  "本次耗时只说明 30 图规模能运行。</p>")
    chunks.append("<details><summary>独立审计结果</summary><pre>" + esc(json.dumps(independent, ensure_ascii=False, indent=2)) + "</pre></details>")
    chunks.append("<h2>原顺序的最终成员</h2><p class='muted'>图片名只用于后验阅读，未作为聚类输入。这里不保存或嵌入原图。</p>")
    for mid, members in reports[("original", "complete_link")]["clusters"]["memberships"].items():
        chunks.append(f"<details><summary>{mid} · {len(members)} 张 · " +
                      esc(", ".join(filenames[i] for i in members)) + "</summary><ul>" +
                      "".join(f"<li><code>{esc(i)}</code></li>" for i in members) + "</ul></details>")
    chunks.append("<p><a href='summary.json'>机器可读汇总</a> · <a href='runs/original/complete_link/graph.json'>原顺序 graph</a>"
                  " · <a href='runs/original/complete_link/admission_log.jsonl'>动态加入日志</a> · "
                  "<a href='https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html'>完整链接定义与复杂度：SciPy 官方文档</a></p></body></html>")
    (HERE / "report.html").write_text("\n".join(chunks))
    snapshot = HERE / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in ("complete_link.py", "run_pilot.py", "build_report.py", "audit_complete_link.py"):
        if (HERE / filename).exists():
            shutil.copy2(HERE / filename, snapshot / filename)
    print(json.dumps({"summary": str(HERE / "summary.json"), "results": [
        {k: r[k] for k in ("order", "method", "clusters", "singleton_images", "close_pairs_kept", "different_pairs_separated")}
        for r in rows]}, indent=2))


def plot(rows, logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    colors = {"previous_nn": "#78879b", "complete_link": "#266ac4"}
    labels = {"previous_nn": "Previous NN graph", "complete_link": "Complete link + Otsu"}
    for method in ("previous_nn", "complete_link"):
        row = next(r for r in rows if r["order"] == "original" and r["method"] == method)
        offset = -.19 if method == "previous_nn" else .19
        axes[0].bar(np.arange(len(row["sizes"])) + 1 + offset, row["sizes"],
                    width=.38, label=labels[method], color=colors[method])
        log = logs[("original", method)]
        axes[1].plot([r["prefix_count"] for r in log], [r["cluster_count"] for r in log],
                     color=colors[method], label=labels[method])
        values = [r["singleton_images"] for r in rows if r["method"] == method]
        axes[2].bar(np.arange(4) + offset, values, width=.38, color=colors[method])
    axes[0].set(title="Final cluster sizes (original order)", xlabel="Cluster, sorted by size", ylabel="Images")
    axes[1].set(title="Clusters during ingestion", xlabel="Images received", ylabel="Clusters")
    axes[2].set(title="Singleton images / 30", xticks=np.arange(4),
                xticklabels=["Original", "Reverse", "Shuffle 1", "Shuffle 2"], ylabel="Images")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(HERE / "fragmentation.png", dpi=180)
    fig.savefig(HERE / "fragmentation.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
