"""Readable report and scientific plots from the completed pair experiment."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = {
    "clip_image": "图片 · CLIP",
    "bge_full_caption": "完整 caption · BGE",
    "clip_caption_chunk_mean_control": "完整 caption · CLIP 分段聚合（补充）",
}
ENGLISH = {
    "clip_image": "CLIP image",
    "bge_full_caption": "BGE full caption",
    "clip_caption_chunk_mean_control": "CLIP chunked caption",
}


def main():
    result = json.loads((HERE / "results.json").read_text())
    probes = json.loads((HERE / "detail_probe_results.json").read_text())
    reference = json.loads((HERE / "detail_reference.json").read_text())
    audit_path = HERE / "independent_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"status": "pending"}
    methods = list(LABELS)
    rows = []
    for feature in reference["features"]:
        row = {"feature_id": feature["id"], "label_zh": feature["label_zh"], "primary": feature["primary"]}
        for method in methods:
            cases = [c for c in probes[method]["cases"] if c["feature_id"] == feature["id"]]
            row[method] = {
                "relative_directions_correct": sum(c["pair_relative_direction_correct"] for c in cases),
                "relative_direction_count": len(cases),
                "individual_choices_correct": sum(int(c["correct_choice_005"]) + int(c["correct_choice_006"]) for c in cases),
                "individual_choice_count": 2 * len(cases),
                "signed_margins": [c["residual_dot_prompt_difference"] for c in cases],
            }
        rows.append(row)
    summary = {
        "conclusion_zh": "这对图片的有限细节对照中，CLIP 图像残差的相对方向更稳定；单残差 SVD 不能判断哪一模态保留更多细节，逐图属性判断仍有明显错误。",
        "svd": result["decompositions"],
        "detail_probe_summary": result["detail_probe_summary"], "per_feature": rows,
        "independent_audit": audit,
        "important_denominators": {
            "image_pairs": 1, "primary_visual_features": 4, "wording_variants_per_feature": 2,
            "primary_relative_comparisons": 8, "primary_individual_choices": 16,
            "independent_human_ground_truth": False,
        },
        "caption_sources": "Full pipeline captions exactly matched to original image SHA256 in the Qwen caption cache.",
        "same_clip_control_is_chunk_aggregated": True,
        "limitations": result["limitations"],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "probe_scores.csv").open("w") as f:
        fields = ["method", "feature", "variant", "primary", "prompt_005", "prompt_006",
                  "correct_choice_margin_005", "correct_choice_margin_006",
                  "residual_dot_prompt_difference", "pair_relative_direction_correct",
                  "correct_choice_005", "correct_choice_006"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in methods:
            for case in probes[method]["cases"]:
                writer.writerow({"method": method, "feature": case["feature_id"],
                                 **{k: case[k] for k in fields if k not in ("method", "feature")}})
    # CSVs expose actual embedding coordinates without an arbitrary 2-D reshape.
    with np.load(HERE / "embeddings_residuals_svd.npz", allow_pickle=False) as arrays:
        for method in methods:
            vectors = arrays[method + "_embeddings"]
            residual = arrays[method + "_residual_005_minus_006"]
            with (HERE / f"{method}.coordinates.csv").open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["embedding_dimension_index", "image005_or_caption005", "image006_or_caption006", "residual005_minus006"])
                writer.writerows((i, vectors[0, i], vectors[1, i], residual[i]) for i in range(vectors.shape[1]))
    plot(result, probes)
    esc = html.escape
    chunks = ["""<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOME D7 005/006：图像与 caption 残差 SVD</title>
<style>body{max-width:1200px;margin:30px auto;padding:0 20px;color:#1c2940;background:#f6f7fb;font:16px/1.65 system-ui,sans-serif}
h1{font-size:27px}h2{font-size:21px;margin-top:32px}.notice{padding:18px;background:#fff2d8;border-left:4px solid #bf851c}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{background:white;padding:16px;border-radius:8px}
.card img{width:100%;object-fit:contain}table{width:100%;border-collapse:collapse;background:white;font-size:14px}
td,th{border:1px solid #d9dfe9;padding:9px;text-align:left}.muted{color:#546176}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}
details{background:white;padding:12px;margin:12px 0}a{color:#155eb9}.plot{width:100%}
@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><body>
<h1>两张 Home 图片：图像 / caption embedding 差分与 SVD</h1>
<p class="notice"><strong>有限细节探针中，图像残差的相对方向更稳定；但单个残差的 SVD 不能判断哪个模态保留更多细节。</strong>
这里只有一对图片、4 种主要细节，每种用 2 种措辞重复检查。它不是总体准确率或独立人工标注的基准。</p>
<h2>原图与完整 caption</h2><div class="grid">"""]
    for record in result["inputs"]:
        relative = Path(record["image_path"]).relative_to(ROOT)
        chunks.append(f"<div class='card'><h3>{esc(record['filename'])}</h3>"
                      f"<img src='../../{esc(str(relative))}' alt='{esc(record['filename'])}'>"
                      f"<p><code>{esc(record['node_id'])}</code> · {esc(record['timestamp'])}</p>"
                      f"<p>{esc(record['caption'])}</p>"
                      f"<p><a href='{Path(record['filename']).stem}.caption.txt'>完整 caption 文本</a></p>"
                      "<details><summary>原始数据集短 caption（未用于主实验）</summary><p>"
                      + esc(" ".join(record["dataset_short_caption_reference_only"]))
                      + "</p></details></div>")
    chunks.append("</div><p>使用当前 pipeline 的完整 Qwen caption，并与图片 SHA256 对应的缓存逐字核对。"
                  "没有重新生成 caption，也没有把对话混入文本 embedding。原图直接引用已有文件，没有复制、改图或改变 memory pipeline。</p>")
    chunks.append("<h2>请求的差分与 SVD</h2>"
                  "<p>所有 embedding 先做 L2 归一化，再计算 <code>r = embedding005 − embedding006</code>。"
                  "SVD 输入分别为 <code>1×512</code> 和 <code>1×384</code>，没有任意 reshape 成图片矩阵。</p>"
                  "<table><tr><th>表征</th><th>维数</th><th>两样本余弦</th><th>残差范数 / 唯一非零奇异值</th><th>秩</th></tr>")
    for method in methods:
        d = result["decompositions"][method]
        chunks.append(f"<tr><td>{LABELS[method]}</td><td>{d['embedding_shape'][1]}</td>"
                      f"<td>{d['cosine_similarity']:.6f}</td><td>{d['residual_l2_norm']:.6f}</td><td>{d['residual_rank']}</td></tr>")
    chunks.append("</table><p><strong>一个差向量最多只有一个非零奇异值，且 σ = ‖r‖ = √(2−2cos)。</strong>"
                  "这个数表示两样本在该编码器内的分离程度，不表示保留的细节数量。BGE 数值更大不能据此判为文本更好；"
                  "CLIP 与 BGE 是不同训练目标、不同空间，二者残差也不能直接相减或求夹角。</p>"
                  "<p>另算两样本减去共同均值后的 <code>2×d</code> 矩阵，同样秩为 1；首个奇异值为上面的 σ/√2，"
                  "第二个仅是浮点误差。仅这两个点用一条仿射主方向就能重建，不能用本实验验证先前多图动态二维压缩损失。</p>")
    chunks.append("<h2>残差对实际细节的响应</h2>"
                  "<p>在取得得分前，冻结每个可见差异的两条短句。例如“书柜位于窗右边 / 窗左边”。"
                  "计算 <code>(e005−e006) · (p005−p006)</code>；大于 0，表示两样本的相对变化方向与该描述一致。"
                  "CLIP 图像使用同检查点的 CLIP 文本探针，BGE caption 使用 BGE 文本探针。</p>"
                  "<table><tr><th>细节（每项 2 种措辞）</th><th>CLIP 图片方向正确</th><th>BGE caption 方向正确</th><th>CLIP caption 补充方向正确</th></tr>")
    for row in rows:
        suffix = "" if row["primary"] else "（次要复合对照）"
        chunks.append(f"<tr><td>{esc(row['label_zh'])}{suffix}</td>" +
                      "".join(f"<td>{row[m]['relative_directions_correct']}/{row[m]['relative_direction_count']}</td>" for m in methods) + "</tr>")
    chunks.append("<tr><th>主要 4 项合计</th>" +
                  "".join(f"<th>{probes[m]['primary_relative_directions_correct']}/8</th>" for m in methods) + "</tr></table>")
    chunks.append("<p><strong>方向正确不等于每张图都能独立答对。</strong>逐张在两条描述中择一，CLIP 图片为 "
                  f"{probes['clip_image']['primary_individual_choices_correct']}/16，BGE caption 为 "
                  f"{probes['bge_full_caption']['primary_individual_choices_correct']}/16，CLIP caption 分段对照为 "
                  f"{probes['clip_caption_chunk_mean_control']['primary_individual_choices_correct']}/16。"
                  "部分图像方向余量很小，换措辞仍可能改变判断；不能把 8/8 解释为图像准确率 100%。</p>")
    chunks.append("<img class='plot' src='residual_diagnostics.png' alt='Residual SVD and direction diagnostics'>")
    chunks.append("<h2>先区分 caption 遗漏与编码器的响应</h2><table><tr><th>细节</th><th>005 caption</th><th>006 caption</th></tr>")
    for feature in reference["features"]:
        chunks.append(f"<tr><td>{esc(feature['label_zh'])}</td><td>{esc(feature['caption_005_coverage'])}</td>"
                      f"<td>{esc(feature['caption_006_coverage'])}</td></tr>")
    chunks.append("</table><p>005 的固定椅腿可见，但 caption 没有写出；006 写了脚轮。"
                  "因此这一项不能单独归因于文本 encoder 丢失信息。探针失败也不能证明 embedding 内完全不存在该信息，"
                  "它只说明当前相似度读出没有稳定响应。高长黑色物体是否为 speaker、薄型设备究竟是 laptop 还是 tablet，"
                  "目视不足以强确认，所以未用作主要真值。</p>")
    chunks.append("<h2>完整文本与同模型补充对照</h2>")
    bge_counts = result["text_encoder"]["caption_token_counts"]
    clip_counts = [r["full_caption_tokens_with_special_tokens"] for r in result["clip_caption_control"]["captions"]]
    chunks.append(f"<p>BGE 两段 caption 分别 {bge_counts[0]}、{bge_counts[1]} tokens，均低于 512 上限，完整输入。"
                  f"CLIP 分别 {clip_counts[0]}、{clip_counts[1]} tokens，超过其 77 上限。"
                  "因此补充对照采用按完整句子分段、每段分别编码、按内容 token 数加权平均后归一化。"
                  "<strong>这是额外聚合协议，不能称为 CLIP 单次编码的完整 caption 向量。</strong>"
                  "已核对所有原始内容 tokens 被完整覆盖，没有静默截断。</p>")
    for record, chunks_row in zip(result["inputs"], result["clip_caption_control"]["captions"]):
        chunks.append(f"<details><summary>{esc(record['filename'])} 的 CLIP 分段</summary><ol>" +
                      "".join(f"<li>{esc(text)} <em>（{count} tokens，含特殊符号）</em></li>"
                              for text, count in zip(chunks_row["chunks"], chunks_row["chunk_tokens_with_special_tokens"]))
                      + "</ol></details>")
    chunks.append(f"<p>同一 CLIP 空间中的图像差向量与分段 caption 差向量余弦为 "
                  f"{result['clip_caption_control']['residual_alignment_with_image_cosine']:.4f}。"
                  "它说明两种差分方向不一致，不能单凭这个夹角确定谁更忠实。</p>")
    chunks.append("<h2>可复查产物</h2><ul>"
                  "<li><a href='embeddings_residuals_svd.npz'>所有 embedding、差向量及 SVD 的 U / S / Vh</a></li>"
                  "<li><a href='results.json'>完整数值、模型、token 和来源记录</a></li>"
                  "<li><a href='probe_scores.csv'>逐项探针得分 CSV</a> · <a href='detail_reference.json'>得分前冻结的细节与短句</a></li>"
                  "<li><a href='clip_image.coordinates.csv'>图像向量 CSV</a> · <a href='bge_full_caption.coordinates.csv'>文本向量 CSV</a></li>"
                  "<li><a href='independent_audit.json'>独立验证</a> · <a href='run_analysis.py'>复现实验脚本</a></li></ul>")
    chunks.append("<p class='muted'>图像：CLIP ViT-B/32，投影后的全局 CLS，输入缩放至模型所需分辨率；"
                  "文本主实验：BGE-small-en-v1.5，CLS pooling，无 query prefix。完整检查点与文件哈希记录于 results.json。"
                  "实验向量保存在此独立分析目录，没有写入只保留两个坐标的记忆方案。</p>")
    chunks.append("<details><summary>独立审计</summary><pre>" + esc(json.dumps(audit, ensure_ascii=False, indent=2)) + "</pre></details>")
    chunks.append("<p>方法参考：<a href='https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html'>NumPy SVD</a> · "
                  "<a href='https://github.com/openai/CLIP'>OpenAI CLIP</a> · "
                  "<a href='https://huggingface.co/BAAI/bge-small-en-v1.5'>BGE 官方模型说明</a></p></body></html>")
    (HERE / "report.html").write_text("\n".join(chunks))
    snapshot = HERE / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("run_analysis.py", "build_report.py", "audit_results.py", "detail_reference.json"):
        if (HERE / name).exists():
            shutil.copy2(HERE / name, snapshot / name)
    print(json.dumps({"report": str(HERE / "report.html"), "per_feature": rows}, ensure_ascii=False, indent=2))


def plot(result, probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    methods = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.7), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.75]})
    values = [result["decompositions"][m]["residual_l2_norm"] for m in methods]
    colors = ["#2868b2", "#d68036", "#648d7c"]
    bars = axes[0].bar(np.arange(3), values, color=colors, width=.6)
    axes[0].set_xticks(np.arange(3), [ENGLISH[m].replace(" ", "\n", 1) for m in methods])
    axes[0].set_ylim(0, .58)
    axes[0].set_ylabel("Only nonzero singular value = residual norm")
    axes[0].set_title("Requested residual SVD: rank = 1 for all")
    axes[0].bar_label(bars, fmt="%.4f", padding=3)
    axes[0].text(.5, .93, "Magnitude is NOT a detail-retention score",
                 transform=axes[0].transAxes, ha="center", fontsize=9, color="#805728")
    axes[0].spines[["top", "right"]].set_visible(False)
    primary = [c for c in probes[methods[0]]["cases"] if c["primary"]]
    correctness = np.array([[int(c["pair_relative_direction_correct"]) for c in probes[m]["cases"] if c["primary"]]
                            for m in methods])
    axes[1].imshow(correctness, cmap=ListedColormap(["#e5a37b", "#9bc7df"]), vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(8):
            axes[1].text(j, i, "+" if correctness[i, j] else "−", ha="center", va="center", fontsize=18)
    labels = ["Monitors\nv1", "Monitors\nv2", "Chair base\nv1", "Chair base\nv2",
              "Shelf side\nv1", "Shelf side\nv2", "Desk layout\nv1", "Desk layout\nv2"]
    axes[1].set_xticks(np.arange(8), labels, fontsize=8)
    axes[1].set_yticks(np.arange(3), [ENGLISH[m] for m in methods])
    axes[1].set_title("Relative detail direction: + agrees, − disagrees")
    axes[1].set_xlabel("4 features × 2 phrasings; repeated diagnostics, not independent samples")
    fig.savefig(HERE / "residual_diagnostics.png", dpi=180)
    fig.savefig(HERE / "residual_diagnostics.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
