"""Summarize the strict two-coordinate/no-original experiment without image IO."""
from pathlib import Path
import html
import json

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    run = read("original_order/run_summary.json")
    evaluation = read("original_order/evaluation.json")
    orders = read("validation/order_audit.json")
    audit = read("independent_audit.json")
    assert audit["all_passed"]
    primary = {"name": "original", "masters": run["master_count"],
               "silhouette": evaluation["geometry"]["cosine_silhouette"],
               "close_pairs": evaluation["visual_sanity"]["close_pairs_kept"],
               "different_pairs": evaluation["visual_sanity"]["different_pairs_separated"],
               "pairwise_assignment_disagreements_with_original": 0,
               "runtime": run["runtime"]}
    all_orders = [primary] + orders["orders"]
    summary = {
        "status": "constraints_implemented_and_audited_but_clustering_quality_not_accepted",
        "scope": "Brand only; isolated offline prototype, not integrated into the production pipeline",
        "hard_constraints": {"retain_original_images_in_memory": False,
                             "retain_original_image_paths_in_memory": False,
                             "reencode_old_images": False,
                             "retain_per_image_full_embeddings": False,
                             "max_visual_coordinates_per_image": 2,
                             "retain_extra_per_image_hashes_or_visual_sketches": False,
                             "retain_per_image_edge_distances": False},
        "retained_master_state": "full-dimensional estimated mean, up to two PCA directions, singular values, count and conservative error metadata",
        "retained_child_state": "up to two coordinates, graph IDs, dialogue, full caption and timestamp",
        "empirical_order_results": all_orders,
        "original_order_cluster_sizes": evaluation["clusters"]["sizes"],
        "original_order_runtime": run["runtime"],
        "original_order_storage": run["storage"],
        "original_order_audit": run["audit"],
        "independent_audit_all_passed": audit["all_passed"],
        "prefix_12_matches_full_stream": orders["prefix_12_matches_full_stream"],
        "conclusion": "The data-flow and storage constraints pass. The candidate-neighbor/PCA2 clustering is order-sensitive and can chain unrelated images into a large component. Do not reuse results of the old image-reencoding method or claim good clustering from six positive examples alone.",
        "preserved_external_test_fixtures": "The pre-existing benchmark files are not deleted. The experiment feeds one disposable image input at a time; the memory receives no retained file/path. Existing scalar cosine data is used only after admission for evaluation. No new posthoc image encodings occur.",
        "superseded_experiments": "The sibling brand_dynamic_pilot_20261019 verified method reencodes originals and is ineligible. The intermediate no-original method in superseded_with_stored_distances retained extra edge scalars and is also ineligible under the final two-coordinate constraint."}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    table = "".join(f"<tr><td>{html.escape(row['name'])}</td><td>{row['masters']}</td>"
                    f"<td>{row['silhouette']:.4f}</td><td>{row['close_pairs']}/6</td><td>{row['different_pairs']}/5</td>"
                    f"<td>{row['pairwise_assignment_disagreements_with_original']}/435</td>"
                    f"<td>{row['runtime']['stream_seconds_excluding_model_load_and_posthoc_evaluation']:.2f} s</td></tr>"
                    for row in all_orders)
    document = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brand 严格二维、无原图回放试验</title><style>
body{{font:16px/1.75 system-ui,sans-serif;max-width:1020px;margin:32px auto;padding:0 20px;color:#243142;background:#fafbfd}}
h1{{font-size:29px}}h2{{font-size:22px;margin-top:30px}}table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
td,th{{border:1px solid #ccd4df;padding:9px;text-align:left}}a{{color:#175ea5}}.notice{{padding:18px;background:#fff0d0;border-radius:8px}}
pre{{white-space:pre-wrap;background:#edf1f6;padding:16px}}li{{margin:8px 0}}</style><body>
<h1>严格只留两个坐标：Brand 动态试验</h1><p>2026-10-19 · 30 张图 · CLIP ViT-B/32 · 仅离线实验</p>
<p class="notice"><strong>结论：存储和加入流程符合新约束，聚类质量尚不达标。</strong>没有保留原图、原图路径、每图完整 embedding、额外二进制摘要或历史距离数值；每张图的视觉数值只有最多两个坐标。但 2D 候选图仍会过度合并，并且明显依赖输入顺序。不能把此前重编码核验版的结果当作这版结果。</p>
<p><a href="summary.json">结构化结论</a> · <a href="original_order/graph.json">最终记忆图</a> · <a href="run_no_original.py">实验入口</a> · <a href="no_replay_sketch.py">二维更新实现</a></p>
<h2>保留什么，何时释放</h2>
<ul><li><strong>加入时：</strong>本试验沿用已有完整 caption 缓存和对话。临时图片只交给 CLIP 一次；编码后立刻删除临时输入，再进行聚类；加入完成后释放输入 embedding。</li>
<li><strong>Master：</strong>保留完整 512 维中心、最多两条 PCA 方向、奇异值、计数与误差状态。这里的“完整”表示维数完整，不保证每次拆分后的中心等于真实成员均值。</li>
<li><strong>子节点：</strong>最多两个局部坐标，另保留对话、完整 caption、时间戳和图索引。没有 image_path，也没有额外图片视觉摘要。</li>
<li><strong>图连接：</strong>仅保留邻居 ID 和连接状态。每次新图加入时，按当前中心与两个坐标临时重算候选边距离；用完释放，不把距离继续作为每图额外视觉信息保存。</li></ul>
<h2>中心变化与拆分</h2>
<p>图片的当前近似表示为 μ + Ba。中心和 PCA 方向更新时，旧坐标只从已有近似投影到新基底。拆分一个成员子集 G 时，用坐标重心得到新中心：</p>
<pre>μ_G = μ + B · mean(a_i, i∈G)
a_i,G = a_i − mean(a_j, j∈G)</pre>
<p>必要时只在二维坐标内做 SVD，去除冗余方向。这个拆分保持已有近似向量不变，不读取旧图，也不重建全量历史 n×512 矩阵。曾丢弃方向信息的簇拆分后，子中心被标为近似；后续追加、合并不会无依据地把这个标记改回精确。</p>
<p>所有 master 的加权和 Σnμ 仍保持为已到达真实向量的总和，逐前缀最大绝对误差为 {run['audit']['max_global_embedding_sum_abs_error']:.2e}。这是全局守恒，不代表每个 master 中心都精确。最终 {run['audit']['approximate_master_mean_count']} 个 master 的中心明确标为近似，{run['audit']['exact_master_mean_count']} 个标为精确。误差上界是保守理论界，不是通过回读旧图测出的真实局部误差。</p>
<h2>加入判定</h2>
<p>沿用局部尺度的自动切边规则，没有固定余弦加入阈值或预设簇数。新图与当前压缩表示比较；旧节点只比较已有候选邻居和新图。候选边的相对距离分布通过 Otsu 方差准则决定保留与切断，再据连通分量更新 master。</p>
<p>准确称谓是<strong>动态候选邻接图</strong>。当基底改变后，另一张未保留为候选的旧图可能变得更近；本方案不会自动发现它。一次加入只处理一轮，也不保证分组已收敛为当前重建空间的固定点。这些选择节省计算，但仍会带来候选遗漏和路径依赖。</p>
<h2>实测结果与顺序敏感性</h2>
<table><tr><th>加入顺序</th><th>Master 数</th><th>真实 CLIP silhouette</th><th>相近对同簇</th><th>不同对分开</th><th>与原顺序不同的同簇关系</th><th>流式耗时</th></tr>{table}</table>
<p>原顺序得到的簇大小为 <strong>{evaluation['clusters']['sizes']}</strong>。其中一个簇包含 18/30 张图，6/6 相近对同簇不能证明质量好：大簇也会轻易覆盖这些正例。5 对不同图片中仍有一对误合。最终候选邻居在真实 CLIP 中是最近邻的比例为 15/30；旧图距离没有保存，因此不存在持久距离误差指标。</p>
<p>这些诊断对在此前试验前固定，但不是独立人工真值；品牌标签、caption 和图像文件名不进入聚类。Silhouette 使用已有的、仅供事后评估的真实余弦标量矩阵，没有为了评估再编码旧图。改变顺序后指标和分组明显波动，所以当前不能称为稳定有效的聚类方案。</p>
<h2>成本与审计</h2>
<ul><li>原顺序 <strong>30 次编码、0 次旧图重编码</strong>，CPU 流式处理 {run['runtime']['stream_seconds_excluding_model_load_and_posthoc_evaluation']:.2f} 秒；其中编码 {run['runtime']['encoder_seconds']:.2f} 秒、聚类约 {run['runtime']['admission_seconds']:.3f} 秒。模型加载另 {run['runtime']['model_load_seconds']:.2f} 秒。</li>
<li>最终中心、PCA 方向、坐标和奇异值共 {run['storage']['sketch_array_bytes']:,} 字节；没有历史距离数组。相同 float64 精度的 30 张完整 embedding 为 122,880 字节。NPZ 文件为 {run['storage']['npz_file_bytes']:,} 字节；这些数字不含图 ID、JSON、误差元数据、诊断日志和模型权重。压缩节省也与当前大簇合并过多有关，不能单独视为收益。</li>
<li>542 项合成数学检查与 58 项独立流程、存储、产物检查通过。前 12 张单独截断运行与完整运行的前 12 步一致。</li>
<li>没有增加超平面 hash、随机投影摘要或其他每图视觉特征。实验输入 SHA256 只用于外部溯源，不进入记忆图、候选判定或子节点视觉状态。</li>
<li>已有 benchmark 文件作为外部实验输入保持原样。本实验没有删除用户的数据集；进入记忆系统的临时副本均在编码后删除。审计覆盖代码、对象生命周期和产物，不冒充操作系统级完整访问证明。</li></ul>
<h2>采用范围</h2>
<p>当前保留为隔离实验，不替换正式 pipeline。新的存储与加入约束已经落实；此前依赖旧图核验的方案不再是可选实现。后续算法调整必须继续遵守这两条硬约束：<strong>不保留/回读旧图，每图仅两个视觉坐标</strong>。</p>
<p>已有方向丢失是信息约束；当前额外的链式合并、候选遗漏和顺序依赖是具体算法仍需改进的部分。本试验说明这版算法不够好，并没有证明任何满足这些约束的算法都不可能有效。</p>
<h2>复现与证据</h2><pre>HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
python3 reports/brand_no_original_pilot_20261019/run_no_original.py \\
  --output reports/brand_no_original_pilot_20261019/reproduced</pre>
<p>在 /storage/hewenwen/VisualMemory 运行。本试验尚无断点恢复支持。</p>
<ul><li><a href="original_order/run_summary.json">运行与存储记录</a> · <a href="original_order/encoder_calls.json">一次性编码记录</a></li>
<li><a href="original_order/evaluation.json">原顺序事后评估</a> · <a href="validation/order_audit.json">顺序与前缀验证</a></li>
<li><a href="independent_audit.json">58 项独立审计</a> · <a href="math_audit.json">542 项数学检查</a></li></ul>
</body></html>"""
    (HERE / "report.html").write_text(document)
    print(HERE / "report.html")


if __name__ == "__main__":
    main()
