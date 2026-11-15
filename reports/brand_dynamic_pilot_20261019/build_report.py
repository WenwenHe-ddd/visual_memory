"""Render measured pilot results; this file never makes admission decisions."""
from pathlib import Path
from datetime import datetime, timezone
import html
import json

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads((HERE / path).read_text())


def main():
    selected = read("verified_local_scale_nn/run_summary.json")
    ordered = read("validation/stream_order_audit.json")
    rows = []
    specs = [
        ("原均值加入阈值", "evaluation/original_visual_mean", False),
        ("二维近似＋不切边", "uncut_nn", True),
        ("二维近似＋自动切边", "local_scale_nn", True),
        ("候选核验＋自动切边", "verified_local_scale_nn", True),
    ]
    for name, path, actual in specs:
        e = read(f"{path}/evaluation.json")
        r = read(f"{path}/run_summary.json") if actual else None
        rows.append({"name": name, "path": path, "masters": e["clusters"]["count"],
                     "silhouette": e["geometry"]["cosine_silhouette"],
                     "close_pairs": e["visual_sanity"]["close_pairs_kept"],
                     "different_pairs": e["visual_sanity"]["different_pairs_separated"],
                     "brand_pair_f1_secondary_only": e["secondary_brand_proxy"]["pair_f1"],
                     "stream_seconds": r["runtime"]["stream_seconds_excluding_model_load_and_posthoc_audit"] if r else None,
                     "stream_encoder_calls": sum(r["runtime"]["encode_calls"].values()) if r else None})
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Brand 30 images only; offline pilot; no production pipeline/retrieval change",
        "decision": "Promising improvement over the current mean-threshold baseline, not a claim that visual clustering quality is solved.",
        "selected_pilot": "verified_local_scale_nn",
        "comparison": rows,
        "order_validation": ordered,
        "runtime": selected["runtime"], "storage": selected["storage"],
        "audit": selected["audit"],
        "no_tuned_cutoff": "No cosine cutoff, similarity margin, target K, histogram bins or variance-retention fraction. A per-prefix Otsu boundary is estimated from normalized edge weights. Rank 2 and the 1-NN graph are explicit design choices, not parameter-free guarantees.",
        "no_all_embedding_cache": "One exact mean and <=2 residual directions per master; <=2 coordinates per child; one nearest-neighbor ID and scalar per image. Original vectors only exist transiently for arrival, candidate checking or split rebuilding.",
        "limitations": [
            "Exact retained edges do not guarantee exact candidate recall after rank-2 truncation; the Brand run's exact NN agreement is measured, not a theorem.",
            "The local Otsu rule divides every nonconstant finite edge-weight distribution; it does not test for evidence of two modes.",
            "Nearest-neighbor connected components can join visually different endpoints through a sequence of intermediate images.",
            "The primary stream required 76 extra old-image encodings; no guarantee of constant insertion cost.",
            "Master bases consume high-dimensional storage. Float64 pilot state is larger than all 30 float32 image vectors, although smaller at equal precision.",
            "Tiny diagnostic cases and same-encoder silhouette are not an independent human visual accuracy benchmark.",
            "All explored objectives used this same Brand domain; neither candidate selection nor diagnostic evaluation is held out.",
            "Diagnostic per-prefix logs grow quadratically here. Core state is compact, but large-library scaling and checkpoint resume were not implemented or tested."]}
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    esc = html.escape
    table = "".join(
        f"<tr><td><a href='{row['path']}/gallery.html'>{esc(row['name'])}</a></td>"
        f"<td>{row['masters']}</td><td>{row['silhouette']:.4f}</td><td>{row['close_pairs']}/6</td>"
        f"<td>{row['different_pairs']}/5</td><td>{row['brand_pair_f1_secondary_only']:.4f}</td>"
        f"<td>{row['stream_seconds']:.2f} s / {row['stream_encoder_calls']} 次</td></tr>" if row["stream_seconds"] is not None else
        f"<tr><td><a href='{row['path']}/gallery.html'>{esc(row['name'])}</a></td><td>{row['masters']}</td>"
        f"<td>{row['silhouette']:.4f}</td><td>{row['close_pairs']}/6</td><td>{row['different_pairs']}/5</td>"
        f"<td>{row['brand_pair_f1_secondary_only']:.4f}</td><td>未同轮计时</td></tr>"
        for row in rows
    )
    order_table = "".join(f"<tr><td>{row['name']}</td><td>{row['masters']}</td><td>{row['silhouette']:.4f}</td>"
                          f"<td>{row['close_pairs']}/6，{row['different_pairs']}/5</td><td>{row['encoder_calls']}</td>"
                          f"<td>{row['stream_seconds']:.2f} s</td></tr>" for row in ordered["orders"])
    r = selected["runtime"]
    body = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Brand 动态视觉记忆聚类试验</title><style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#1c2839;background:#fafbfd}}
h1{{font-size:30px}}h2{{font-size:23px;margin-top:36px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}
td,th{{border:1px solid #ccd6df;padding:9px;text-align:left}}code,pre{{background:#edf1f6;border-radius:4px}}pre{{padding:18px;white-space:pre-wrap;overflow-wrap:anywhere}}
a{{color:#175a9f}}.notice{{padding:18px;background:#fff3d5;border-radius:8px}}.good{{padding:18px;background:#eaf4ee;border-radius:8px}}li{{margin:9px 0}}
</style><body><h1>Brand 动态视觉记忆聚类试验</h1>
<p>2026-10-19 · 30 张原图 · CLIP ViT-B/32 · 每张按顺序加入 · 仅 offline 试验</p>
<p class="good">当前更值得继续的方案：<strong>局部二维残差提出候选 → 临时核验候选边 → 按相对局部尺度自动切边 → 动态更新 master</strong>。没有设置 0.85、0.03 或指定簇数；没有保存每张图片的原始 embedding。Brand 原顺序、反序和两个固定随机顺序得到相同最终分组。</p>
<p class="notice">有明确改善，但不能声称已完全达到“效果好且高效”。视觉链式粗合并仍存在；为了控制二维误差，这次额外重编码了旧图 76 次。此次是同一 Brand 数据上的方案探索与诊断，不是独立测试集结论。</p>
<p><a href="verified_local_scale_nn/gallery.html">打开最终分组原图画廊</a> · <a href="verified_local_scale_nn/manual_review.json">逐簇视觉复核</a> · <a href="summary.json">结构化汇总</a> · <a href="run_streaming_pilot.py">运行代码</a></p>
<h2>这次实际做了什么</h2>
<ol>
<li><strong>新图到来。</strong>CLIP 只编码当前图片，得到单位向量 z。第一张成为独立 master；只有一个非零候选边时无法估计强弱分布，暂不合并，精确重复图除外。取消固定初始加入阈值。</li>
<li><strong>压缩候选检索。</strong>每个 master 保存完整均值 μ 和最多两条局部残差主方向 B；每个子节点保存最多两个坐标 a。旧图近似为 μ + Ba。用这些状态比较新图与旧成员；不构建全量历史 embedding 矩阵。</li>
<li><strong>准备改连接时才核验。</strong>某候选可能改善新图或旧图的最近邻时，若其表示已有丢失方向，则临时重新编码该旧图，计算真实余弦距离，使用后立即释放。只保存邻居 ID 和距离标量。所有保存边的距离经过核验；候选筛选仍有近似，不能普遍保证找到了真正最近邻。</li>
<li><strong>让分界由当前数据决定。</strong>令 rᵢ 为 i 当前保存的最近邻距离，对边(i,j)计算 wᵢⱼ = log[dᵢⱼ / min(rᵢ,rⱼ)]。在当前有限边权的不同取值之间枚举分界，最大化两组之间的方差，保留较小一组。实现采用 <a href="https://ieeexplore.ieee.org/document/4310076">Otsu 的类间方差准则</a>，不设直方图 bin。常量边权不强行二分；零距离按代码中的退化分支处理。最终分界比值 1.558988 是运行结果，不能拿它当固定配置。</li>
<li><strong>连通分量生成 master。</strong>新边可以改变已有归属、合并或拆分。整簇合并按成员数更新精确均值；无损小簇拆分可恢复成员，已有压缩损失的大簇拆分则逐张重新编码受影响成员，避免伪造精确中心。</li>
<li><strong>保存。</strong>master 只携带视觉表示与必要的计数、投影统计；子节点保留原 dialogue、完整 caption、timestamp 和图片来源。均值、方向和坐标在 visual_state.npz，graph.json 只放路径/key/row 索引。正式 pipeline 与 online 检索尚未变动。</li>
</ol>
<h2>动态中心与方向如何更新</h2>
<p>均值按 μ′ = μ + (z − μ)/(n + 1) 精确更新。新残差方向由最多 3×512 的矩阵做 SVD；合并两个完整 master 时最多 5×512。旧坐标随中心与方向变换：</p>
<pre>aᵢ′ = B′ᵀ B aᵢ + B′ᵀ(μ − μ′)</pre>
<p>这是旧近似向新基底的投影。被丢弃的方向不能恢复；因此并不宣称二维坐标保留了完整图像几何，也不把它称为全量 batch PCA。保留的完整 master 均值不受这项截断影响。</p>
<h2>真实动态结果</h2>
<table><tr><th>方法</th><th>Master 数</th><th>真实 CLIP silhouette</th><th>相近对同簇</th><th>不同对分开</th><th>品牌 pair F1（辅助）</th><th>CPU 流式处理 / 编码次数</th></tr>{table}</table>
<p>相近 6 对、不同 5 对是在新试验分组之前固定的 AI 目视诊断案例，但参考过原方法的问题，且不是独立人工真值。另有 4 对跨品牌/歧义案例只展示、不计分。品牌标签和描述仅事后评估使用，没有输入聚类。Silhouette 来自同一 CLIP 的真实距离，衡量该编码器内的几何，不等于视觉准确率。</p>
<p>二维近似直接决定连接时，只恢复了 2/6 相近对；加入核验恢复到 4/6，说明本次最大的可修正问题之一是压缩后的边距离失真。只用未切边连通分量虽有 5/6，相似概念被过度串联，只有 4 个大簇，不能据此判为最好。</p>
<h2>顺序、正确性与存储核查</h2>
<table><tr><th>额外加入顺序</th><th>Master 数</th><th>Silhouette</th><th>相近 / 不同案例</th><th>编码总次数</th><th>流式时间</th></tr>{order_table}</table>
<p>前 12 张单独截断重跑，与完整流前 12 个时刻的归属及边一致。未来图片访问被硬性拒绝。24 条合成流覆盖拆分、合并、均值、压缩距离与保存索引；完全重复向量测试也通过。所有 payload 完整保留；master 均值最大绝对误差 {selected['audit']['max_master_mean_abs_error']:.2e}。</p>
<p>该 Brand 主试验每个前缀的分组都与使用精确全部距离的参考算法一致；最终 30/30 个最近邻也一致。这是事后核查结果，参考距离文件只供独立评估读取，没有喂给真实动态算法。四种顺序一致不意味着任意未来数据都对顺序不敏感。</p>
<ul>
<li>主试验流式处理 <strong>{r['stream_seconds_excluding_model_load_and_posthoc_audit']:.2f} 秒</strong>，模型加载另 {r['model_load_seconds']:.2f} 秒，事后审计额外编码 30 次不计入处理时间。</li>
<li>30 次新图编码：{r['arrival_encode_seconds']:.2f} 秒；47 次候选边核验：{r['edge_verification_seconds']:.2f} 秒；29 次拆分重建：{r['split_reencode_seconds']:.2f} 秒；纯聚类计算：{r['admission_seconds_excluding_reencode']:.3f} 秒。计时为单次 CPU 观察，不是稳定性能基准。</li>
<li>最终 10 个完整均值、10 条总残差方向、48 个坐标值、10 个奇异值，共 <strong>82,384 字节</strong> float64 数组；再加 30 个最近邻距离是 82,624 字节。相同 float64 精度的 30 张完整 embedding 为 122,880 字节，少约 32.8%。但完整 float32 embedding 仅 61,440 字节；不能跨精度声称本试验更省。</li>
<li>原方案 21 个 master 均值本身为 86,016 字节；本方案增加了方向与坐标，因此对原 master-only 方案的节省很小。NPZ 文件 92,786 字节，含容器开销；图 JSON、ID、日志和模型权重另计。</li>
<li>核心状态为 O(Kdr + Nr + N)，r≤2；单次比较约 O(Kdr + Nr)，边排序 O(N log N)，再加受影响簇的 SVD 与必要重编码。不是次线性检索，也不保证常数加入时间。本试验逐前缀完整诊断日志累计 O(N²)，正式大规模实现应外写/精简日志。</li>
</ul>
<h2>仍然没解决的地方</h2>
<p><strong>M003 与 M009 仍然过粗。</strong>前者通过咖啡相关图片串起信息图、纸杯、盲文板和硬币甜甜圈；后者通过菜单图串起汉堡、鱼缸汉堡和薯条 Wi-Fi。连通不意味着簇内任意两张都相似。固定的 11 对计分案例没有覆盖这些新问题。</p>
<p>McDonalds_4/5 的 CLIP 相似度约 0.602677，双方都有更近的其他图片，因此单最近邻图不产生它们之间的边。CocaCola_3/4 也各有更近对象；Coke4 对 Pepsi4 与 Coke5 的差异仅约 0.000053，显示单最近邻排序可能敏感。这两处不能单靠改一个加入阈值解释或解决。</p>
<p>Otsu 没有证明当前边权确实有两类，非恒定分布会被切开。自动分界避免了手填数值，仍包含“局部尺度＋两组分离＋图连通”的建模假设。只保存二维残差会影响候选召回；旧图临时重编码控制了本次误差，但也增加计算。</p>
<p><strong>结论：</strong>保留这版作为可复现候选，不立即替换主 pipeline。它已验证动态流程、完整中心、压缩子节点、无固定余弦门槛和相对原版的改善；“稳定视觉紧凑”和大库效率还没有验证成立。下一步应针对图的链式合并与 CLIP 对视觉模板的排序偏差设计对照，不再围绕固定余弦阈值调数值。</p>
<h2>复现与证据</h2>
<pre>HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
python3 reports/brand_dynamic_pilot_20261019/run_streaming_pilot.py \\
  --policy verified_local_scale_nn \\
  --output reports/brand_dynamic_pilot_20261019/reproduced_verified</pre>
<p>从 /storage/hewenwen/VisualMemory 运行。当前是独立试验入口，可保存和核查，未实现断点恢复。</p>
<ul><li><a href="verified_local_scale_nn/run_summary.json">运行时间、模型版本、图像 SHA 与几何审计</a></li>
<li><a href="verified_local_scale_nn/admission_log.jsonl">逐次加入与切边记录</a> · <a href="verified_local_scale_nn/encoder_calls.json">每次编码及其用途</a></li>
<li><a href="validation/artifact_audit.json">独立产物审计</a> · <a href="validation/stream_order_audit.json">顺序与前缀重放审计</a></li>
<li><a href="independent_streaming_review.json">合成流独立复核</a> · <a href="streaming_sketch_math_audit.json">增量 SVD 数学核查</a></li>
<li><a href="source_snapshot/run_streaming_pilot.py">执行版本快照</a> · <a href="design_exploration/local_scale_upperbound.json">精确距离参考结果（非真实压缩流）</a></li></ul>
</body></html>"""
    (HERE / "report.html").write_text(body)
    print(HERE / "report.html")


if __name__ == "__main__":
    main()
