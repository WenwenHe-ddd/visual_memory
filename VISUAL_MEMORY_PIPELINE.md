# VisualMemory 入口

统一入口：`scripts/visual_memory_pipeline.py`。当前保留 Natural、FC(T)、Reflexion、Gen.Ag、MemoryOS、A-Mem、SimpleMem，以及独立的视觉表征聚类和 Object Relation 实验。

当前 caption 分簇与视觉 master 多路径方案的流程、公式和框架图见 [Method 1 README](docs/method_1/README.md)。

组内 ridge 白化与 z-score 在线分数修正的公式、运行方式和简洁框架图见 [Method 2 README](docs/method_2/README.md)。

## Natural：图片描述检索与回答

```bash
python3 scripts/visual_memory_pipeline.py offline --memory natural
python3 scripts/visual_memory_pipeline.py online --mode natural --top-k 5 --query-image-captions --stage prepare
python3 scripts/visual_memory_pipeline.py online --mode natural --top-k 5 --query-image-captions --stage run
python3 scripts/visual_memory_pipeline.py online --mode natural --top-k 5 --query-image-captions --stage analyze
```

Natural 从 `results/natural_observation` 读取缓存的完整图片描述，在同一任务内检索 Top-k。`--query-image-captions` 仅为检索问题追加已有的问题图片描述，不引入选项或标准答案；reader 仍接收原始问题图片。`--mode all` 运行 Natural 与 FC(T)。

在线输出默认位于 `results/visual_memory_pipeline/`。修改实现后，新运行应指定独立的 `--output-root`；现有基线结果保留原协议和历史来源哈希，不改写为新代码的实验结果。

## Offline：其他独立方法

### Visual embedding graph：图片表征动态聚类（offline）

`offline --memory visual_embedding_graph` 使用 CLIP 图片编码器，仅由图片像素决定归属。每个原始 task 是独立记忆库，按源对话顺序逐图插入；完整 caption 逐字复用已验证签名的 natural Qwen 缓存，对话和时间戳来自原始记录。没有接入 online 检索或问答。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 scripts/visual_memory_pipeline.py offline \
  --memory visual_embedding_graph \
  --visual-encoder /storage/hewenwen/VisualMemory/models/clip-vit-base-patch32 \
  --visual-encoder-revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --visual-device cpu --visual-local-files-only \
  --task Brand_Memory_Test --limit 10 \
  --graph-root /storage/hewenwen/VisualMemory/results/visual_embedding_graph_debug \
  --from-start
```

首版使用 `openai/clip-vit-base-patch32` 的 vision tower 和 trained visual projection，512 维输出逐图归一化。`--visual-encoder` 支持本地模型目录或 Hugging Face ID，`--visual-encoder-revision` 可固定远程版本；本地模型内容另存 SHA-256。默认使用上述本地模型目录和 CPU，GPU 可改 `--visual-device cuda:0`。每张图片的向量只临时存在内存，不写图片向量缓存、不保存到子节点或日志；仅 master 保存聚合向量。

规则（均使用原始余弦相似度，范围 `[-1,1]`）：

1. master 的 `visual_mean=u` 是成员单位图片向量的未归一化均值。比较时取中心 `m=u/||u||`。
2. 首个 master 的 `join_threshold=0`。无候选合格而新建后续 master 时，阈值取新图片与所有现有 master 中心的最大余弦相似度，允许为负。
3. 只有 `similarity > join_threshold` 才可加入；多个候选达标时选择原始相似度最高者，同分按 master 创建顺序。此方法固定该规则。
4. 加入已有 master 后，`u'=(n*u+z)/(n+1)`，成员数加一，阈值更新为 `||u'||`，精确等于全体成员到新中心的平均相似度。旧成员不重新分配。
5. 新建 master 的首条边距离为 0。加入已有 master 的新边距离为 `1-dot(z,m')`，在更新中心后精确初始化。旧边统一加 `delta=||u||-dot(u,m')`，逐边是近似值，但全部边的平均距离精确等于 `1-||u'||`。不裁剪边距离，不把初始化阈值变化误作中心移动量。

聚合视觉向量单独存入 task 目录下的 `vectors/master_visual_means.<sha256>.npy`，使用 float64 矩阵，无损保存原来的数值。`graph.json` 顶部的 `vector_storage` 给出相对路径、SHA-256、矩阵形状和数据类型；每个 master 的 `visual_mean_ref.row` 是从 0 开始的行索引，不再内嵌长向量。例如：

```json
{
  "vector_storage": {
    "format": "npy",
    "path": "vectors/master_visual_means.<sha256>.npy",
    "sha256": "<sha256>",
    "shape": [7, 512],
    "dtype": "float64"
  },
  "masters": [
    {"id": "M1", "visual_mean_ref": {"row": 0}, "member_count": 2, "join_threshold": 0.9}
  ]
}
```

以上只是字段示例。`image_nodes` 仍保存 ID/原图路径、完整 `dialogue`、完整 `caption`、原始 `timestamp`。`edges` 明确标注 `exact_at_insertion` 或 `uniform_mean_shift_approximation`。后一类不代表旧图片到当前中心的真实几何距离，统一更新也不会改变旧边之间的排序。

通过 `scripts.graph.visual_embedding_storage.load_graph(graph_path)` 加载图时，会校验向量文件的哈希、形状、类型和索引，再还原内存中的 `visual_mean`。路径相对于 `graph.json` 所在 task 目录，因此整个 task 文件夹可一起移动；不能只复制 JSON 而遗漏 `vectors/`。加载器兼容旧版内嵌向量格式。

输出按 task 保存 `graph.json`、`vectors/`、`write_log.jsonl`、`checkpoint.json` 和 `summary.json`；根目录记录模型、源文件和实现的哈希。checkpoint 也只引用同一份外置向量。每次更新先写入按内容哈希命名的向量文件，再原子写入 checkpoint 和 graph；两份 JSON 均保存成功后清理不再引用的旧向量文件。中断时旧 checkpoint 仍能找到原来的向量，导出文件可重新生成。`--limit N` 表示本次每个 task 最多新增 N 张；去掉 `--from-start` 续跑，去掉 `--limit` 处理剩余图片，去掉 `--task` 处理全部任务。若源文件、模型或实现改变，应使用新的输出目录；旧图不会被静默混用。summary 的 `complete` 只针对当前选择的任务。

VS Code 调试项为 **VisualMemory: Debug Visual Embedding Graph (CLIP)**，默认重建 Brand 任务前 10 张图片。建议断点放在 `visual_embedding_graph.py` 的 `insert()`：候选判断、均值更新、`distance_delta` 和边更新处。

### caption_visual_master：caption 分簇与视觉 master 检索

当前新方案通过主入口显式选择，与旧 `visual_embedding_graph` 分开：

```bash
# Home 的 8 张图，按原始顺序逐张加入
/usr/bin/python3 scripts/visual_memory_pipeline.py offline --memory caption_visual_master
# 读取上一步的 memory，检索 Q8/Q10/Q27
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master
```

两阶段的 CLIP/BGE 默认使用 `--device cuda:0`，CUDA 不可用时明确报错。默认数据范围是已固定输入的 Home 小样例，仍依赖对应的 manifest、模型 protocol、二维基底和候选答案文件。`--image-files` 选择离线图片，`--question-ids` 选择在线问题，`--output` 分别指定各阶段输出；在线使用 `--memory-dir` 读取离线 memory。另有下述 `--dataset mem_gallery` 全量数据适配器。

离线按 caption embedding 做 Ward 分簇，以 silhouette 选择簇数，按成员变化更新 master 完整视觉表征。叶子保存原始对话、完整 caption、时间戳和两个视觉坐标；不保存原图副本或每张图片的完整视觉 embedding。默认重建小样例；不更改既有全领域实验结果。

在线先用每个 master 的 CLIP 视觉中心在预先准备的 A/B/C/D 候选答案陈述中选择最相似的一句，把它追加到问题 query；再用 BGE 与本簇叶子的“对话＋完整 caption”计算相似度，汇总全局 Top5。同一份输出还保留原 query 全量检索基线。候选答案仅用于检索。Home 小样例只支持 `--stage prepare`（默认），不运行 QA reader，其他 stage 会明确报错。

默认输出为 `results/debug_caption_master_answer/offline/` 和 `online/`。离线的 `memory/` 下包含 `graph.json`、`basis.npy`、`master_sums.npy`；在线输出 `retrieval_results.json`。实际编码设备会记录在两阶段输出中。

VS Code 调试项 **VisualMemory: Offline 构建（8 张图）**、**VisualMemory: Online 检索（3 道题）** 直接执行 `scripts/visual_memory_pipeline.py`。断点建议：

- 统一调度：pipeline 的 `offline()` 或 `main()` 中的 `caption_visual_master` 分支。
- 离线编码和逐张加入：`scripts/memory/caption_visual_master.py` 的 `offline()`；进入 `memory.add()` 可检查 `scripts/graph/caption_ward_visual.py` 中的分簇与视觉更新。
- 在线候选匹配、query 扩展和全局排序：`scripts/memory/caption_visual_master.py` 的 `choose_master_answers()`、`retrieve_question()`。

`scripts/debug_master_answer_memory.py` 只保留旧命令兼容转发，不再维护独立实现。

#### 184/371 全量检索基线：五项标注

**方案标识：`baseline_question_only` / `full_scan`；最终问答准确率 `184/371 = 0.495956873`。** 结果来自 [历史 reader summary](reports/all_master_answer_query_20261021/reader/summary.json)。当前在线对应 `natural` memory、`--retrieval full_scan --online-weighting none --top-k 5`。代码中文标注统一使用 `【184/371 全量基线 · 序号. 名称】`，位于 [online_retrieval.py](scripts/memory/online_retrieval.py)。

| 标注 | 该方案的确切定义 | 对应位置 |
| --- | --- | --- |
| **1. 候选记忆结构** | 当前 domain 的全部图片记忆，每条保存 `image_id`、`memory_id`、该轮 User/Assistant 对话和完整 Qwen caption；检索文本为 `dialogue + "\nMemory description: " + caption`。8 个 domain 合计 438 条，分别检索；Home 全量为 89 条。 | 当前：`NaturalCaptionMemory.__init__()`；历史：[`collect_sources()`](scripts/analysis/prepare_all_domain_adaptive_inputs.py#L58)。 |
| **2. 查询内容** | 原问题文本；题目带图时，附上该题图片已有的 Qwen caption。不拼选项、候选答案陈述或历史对话。**“带对话检索”的对话位于记忆侧。** | 当前：`FullScanRetrieval.plan()`；历史：[`make_questions()`](scripts/analysis/prepare_all_domain_adaptive_inputs.py#L154) 保留 `retrieval_query`。 |
| **3. 检索方式** | 同一个 query 与当前 domain 全部候选计算余弦相似度，按分数降序取 Top-5，完全同分按源顺序。基线不使用 master 路由、CLIP 候选匹配、白化或 z-score。 | 当前：`retrieve_question()` 中 `weighting is None` 和 `orders`；历史：[`FullScan.search()`](scripts/analysis/compare_adaptive_retrieval.py#L75)。 |
| **4. 编码方式** | `BAAI/bge-small-en-v1.5`，384 维，查询和记忆都做 L2 归一化。仅查询加前缀 `Represent this sentence for searching relevant passages: `；记忆编码完整对话＋caption。批大小 16，长度上限 512；历史 Card 有一条 577-token 输入截到 512。 | 当前：`retrieve_question()` 的 `query_vectors` / `leaf_vectors` 和 `encode_memory_text()`；历史：[`encode_text_features()`](scripts/analysis/prepare_all_domain_adaptive_inputs.py#L162)。 |
| **5. 回答方式** | Top-5 的原始对话＋完整 caption 按排名拼接，再附原问题和 A/B/C/D 选项；带图题另传原图。使用 Qwen3-VL-8B-Instruct，BF16、seed=42、`do_sample=False`、`max_new_tokens=16`，只输出选项字母。固定选项排列下计算 371 题的最终正确率。 | 当前：`retrieve_question()` 的 `reader_text` 输出边界；历史实际回答：[`prepare()`](scripts/analysis/evaluate_all_master_answer_reader.py#L37) 和 [`run()`](scripts/analysis/evaluate_all_master_answer_reader.py#L165)，参数见 [reader protocol](reports/all_master_answer_query_20261021/reader/protocol.json)。 |

历史实验复用冻结的 BGE 查询和叶子向量；当前 online 入口现场编码并丢弃向量，已核对全部 371 题的原始有序 Top-5 一致。当前 `--stage prepare` 只输出检索结果和可供 reader 使用的记忆文本；上表第 5 项记录产生 **184/371** 的历史回答方式，不表示当前检索入口已执行 QA。历史源代码与原始逐题记录保持原样，表中链接定位其实际实现。

结果文件同样带有 `method_annotations` 五项说明：[历史问答汇总](reports/all_master_answer_query_20261021/reader/summary.json) 和 [历史检索汇总](reports/all_master_answer_query_20261021/retrieval_summary.json) 按 `baseline_question_only` 标注；[可视化报告](reports/all_master_answer_query_20261021/report.html#baseline-method-annotations) 展示对应表格。当前 online 结果已补齐，后续由 `result_method_annotations.py` 自动写入。仅检索的输出明确保存 `reader_called=false`、`measured_qa_accuracy=null`，不会把历史 184/371 当成本次准确率。

标注只新增说明字段；原有指标、分数、排名和回答逐项一致，原始 JSONL 与协议未改写。文件字节哈希会随说明增加而变化，旧核验哈希仍表示标注前的版本；[标注审计](results/result_method_annotations_20261023/audit.json) 保存前后哈希、原文件备份及内容一致性核验。

#### baseline_question_only 加 z-score：λ=0.1、α=1

2026-10-23 已用 GPU 1 完成全部 371 题的配对问答。仅第 **3. 检索方式** 增加在线赋权：每个 domain 的全量候选组成一个簇，白化和标准化后用 `final = raw + std_pop(raw) * tanh(z_white)` 排序取 Top-5。第 1、2、4、5 项的候选文本、查询内容、编码模型和回答配置保持原基线口径；两组本轮重新回答，相同完整输入仅在本轮共享一次新推理。

| 方案 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 |
| --- | --- | --- | --- | --- |
| `baseline_question_only` | 184 / 371 | 49.5957% | 492 / 1041 | 297 / 371 |
| `baseline_question_only_zscore` | 183 / 371 | 49.3261% | 490 / 1041 | 297 / 371 |

3 题错→对、4 题对→错，净减少 1 题（−0.2695 个百分点）；McNemar 双侧 p=1.0，本设置未观察到提升。57 题有序 Top-5 改变，其中 24 题的记忆集合改变。新运行的基线输入和全部 371 个答案与历史基线一致。742 条回答记录来自 428 次新推理和 314 条同轮相同输入共享；零格式错误、零输出截断。此结果采用固定选项排列的本地 Qwen 评测。

可查看 [完整报告及五项标注](reports/baseline_question_only_zscore_20261023/report.html)、[问答统计](reports/baseline_question_only_zscore_20261023/reader/summary.json)、[逐题结果](reports/baseline_question_only_zscore_20261023/reader/paired_results.csv) 和 [独立核验](reports/baseline_question_only_zscore_20261023/reader/independent_audit.json)。核验覆盖全部候选分数、完整输入、原图哈希、生成 token 解码和同轮共享来源，并用冻结 BGE 向量的协方差矩阵求逆独立复算白化公式，分差小于 `5e-7`。

完整执行入口为 [evaluate_baseline_question_only_zscore.py](scripts/analysis/evaluate_baseline_question_only_zscore.py)，固定 λ=0.1、α=1；重新运行必须指定新的输出目录。检索阶段调用统一 online pipeline，回答阶段沿用固定 Qwen reader：

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python3 scripts/analysis/evaluate_baseline_question_only_zscore.py \
  --stage all --device cuda:1 --output reports/baseline_question_only_zscore_rerun
```

#### baseline_question_only 的 λ 消融：0.01 / 0.1 / 1.0

2026-10-23 进一步固定 α=1，在相同 memory、原始查询、对话＋完整 caption 和回答配置下，完成 λ=0.01 与 1.0 的 GPU 1 全量评测。每个 domain 全量候选为一个簇；仅调整第 3 步中白化的 λ。原始基线和 λ=0.1 两列来自上面已核验的实验，两个新设置各自重新评测全部 371 题。

| 设置 | 答对 / 总题数 | 准确率 | 相对原基线：错→对 / 对→错 | Top-5 目标记忆召回 | 至少命中一条目标的题数 |
| --- | --- | --- | --- | --- | --- |
| 未赋权原基线（已有） | 184 / 371 | 49.5957% | — | 492 / 1041 | 297 / 371 |
| λ=0.01（本轮） | 188 / 371 | 50.6739% | 14 / 10 | 488 / 1041 | 300 / 371 |
| λ=0.1（已有） | 183 / 371 | 49.3261% | 3 / 4 | 490 / 1041 | 297 / 371 |
| λ=1.0（本轮） | 183 / 371 | 49.3261% | 0 / 1 | 491 / 1041 | 297 / 371 |

λ=0.01 本次答对数最高，相对原基线净增加 4 题（+1.0782 个百分点，McNemar 双侧 p=0.5413），相对 λ=0.1 净增加 5 题（12 题错→对、7 题对→错）。这是同一测试集上的参数消融，不能作为独立验证的最优参数结论。λ=1.0 与 0.1 的总正确率相同，但逐题答案不同。增大 λ 只减弱白化，α=1 的辅助赋权仍保留。

相对原基线，λ=0.01、0.1、1.0 分别改变 203、57、12 题的 Top-5 顺序，以及 119、24、4 题的 Top-5 集合。两个新条件的未赋权分数与上一轮完全相同；742 条新评测记录由 571 次新推理和 171 条同轮相同输入共享产生，不复用历史答案。独立矩阵求逆、分数标准化、排序、完整输入、原图哈希、生成 token 解码和回答计数全部通过；与参考组输入完全相同的 1,043 次比较均无答案差异，零格式错误及输出截断。

结果及五项标注见 [消融报告](reports/baseline_question_only_lambda_ablation_20261023/report.html)、[完整统计](reports/baseline_question_only_lambda_ablation_20261023/reader/summary.json)、[逐题对照](reports/baseline_question_only_lambda_ablation_20261023/reader/paired_results.csv)、[独立核验](reports/baseline_question_only_lambda_ablation_20261023/reader/independent_audit.json)。执行入口为 [evaluate_baseline_question_only_lambda.py](scripts/analysis/evaluate_baseline_question_only_lambda.py)，默认 `--stage all --device cuda:1`；重跑时指定新的 `--output` 目录。

#### baseline_question_only 的 α 消融：固定 λ=0.01

2026-10-23 在 GPU 1 完成 α=0.01、0.1、10.0 三组各 371 题的新评测，固定 λ=0.01。原始查询、同 domain 全量候选、对话＋完整 caption、BGE 编码及 Qwen 回答配置与此前实验一致。辅助赋权为 `final = raw + alpha * std_pop(raw in domain) * tanh(z_white)`；仅缩放修正量，全部原始分数、白化分数和 z-score 与 α=1 实验逐项完全一致。

| 设置 | 答对 / 总题数 | 准确率 | 相对 α=1：错→对 / 对→错 | Top-5 目标记忆召回 | 至少命中一条目标的题数 |
| --- | --- | --- | --- | --- | --- |
| 未赋权原基线（已有） | 184 / 371 | 49.5957% | 10 / 14 | 492 / 1041 | 297 / 371 |
| α=0.01（本轮） | 183 / 371 | 49.3261% | 9 / 14 | 491 / 1041 | 297 / 371 |
| α=0.1（本轮） | 184 / 371 | 49.5957% | 9 / 13 | 492 / 1041 | 298 / 371 |
| α=1（已有） | 188 / 371 | 50.6739% | — | 488 / 1041 | 300 / 371 |
| α=10.0（本轮） | 192 / 371 | 51.7520% | 20 / 16 | 466 / 1041 | 300 / 371 |

α=10.0 在本次四个 α 中答对数最高：相对 α=1 净增加 4 题（+1.0782 个百分点，McNemar 双侧 p=0.6177）；相对未赋权基线为 27 题错→对、19 题对→错，净增加 8 题（+2.1563 个百分点，p=0.3020）。目标记忆召回总数下降，因此问答提升不能解释为召回更多目标。该结果为同一测试集上的描述性消融，尚不足以确认参数的泛化优势。

相对未赋权基线，α=0.01、0.1、1、10 分别改变 10、40、203、330 题的 Top-5 顺序，以及 4、19、119、258 题的记忆集合。三组新设置共 1,113 条回答记录，来自 728 次新推理和 385 条同轮相同输入共享；不复用历史答案。独立核验重算了协方差求逆、标准化、α 修正、排序、完整输入、原图哈希和生成 token 解码；与参照组输入完全相同的 1,162 次比较均无答案差异，零格式错误、零输出截断。

启动时检测到统一 pipeline 入口版本不同；其余固定输入、白化实现和回答代码均未变。实验协议记录该入口的新旧哈希，仅对这一明确版本变动放行，并在回答前要求所有原始／白化／z 分数完整复现，实测最大差异均为 0。已有实验的协议和结果保持原样。

五项方法标注、参照来源和分 domain 统计见 [α 消融报告](reports/baseline_question_only_alpha_ablation_20261023/report.html)、[完整统计](reports/baseline_question_only_alpha_ablation_20261023/reader/summary.json)、[逐题对照](reports/baseline_question_only_alpha_ablation_20261023/reader/paired_results.csv)、[独立核验](reports/baseline_question_only_alpha_ablation_20261023/reader/independent_audit.json) 和 [实验协议](reports/baseline_question_only_alpha_ablation_20261023/protocol.json)。[执行入口](scripts/analysis/evaluate_baseline_question_only_alpha.py) 固定 λ=0.01，默认 `--stage all --device cuda:1`；重跑需指定新的 `--output` 目录。

#### baseline_question_only 的 α 消融：固定 λ=0.1

2026-10-24 补齐 α=0.01、0.1、10.0 三组，每组评测全部 371 题；α=1 使用此前已核验的 183/371 结果。每个 domain 的全量 natural memory 为一个簇，原始查询、对话＋完整 caption、BGE 编码、Top-5 和 Qwen 回答配置保持一致，仅缩放在线 z-score 修正量。

| 设置 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 | 相对未赋权答对变化 |
| --- | --- | --- | --- | --- | --- |
| 未赋权原基线（已有） | 184 / 371 | 49.5957% | 492 / 1041 | 297 / 371 | — |
| α=0.01（本轮） | 184 / 371 | 49.5957% | 491 / 1041 | 297 / 371 | 0 |
| α=0.1（本轮） | 183 / 371 | 49.3261% | 491 / 1041 | 297 / 371 | −1 |
| α=1.0（已有） | 183 / 371 | 49.3261% | 490 / 1041 | 297 / 371 | −1 |
| **α=10.0（本轮）** | **192 / 371** | **51.7520%** | **492 / 1041** | **300 / 371** | **+8** |

α=10 在本轮四个设置中答对数最高。相对未赋权有 14 题错→对、6 题对→错，净增加 8 题（+2.1563 个百分点，McNemar 双侧 p=0.1153）；相对 α=1 有 12 题错→对、3 题对→错，净增加 9 题（p=0.03516）。p 值未作多重比较校正；这仍是同一测试集上的描述性参数比较。α=10 与未赋权的目标记忆命中总数相同，不能将问答变化直接解释为召回更多目标。

相对未赋权，α=0.01、0.1、1、10 分别改变 1、12、57、156 题的 Top-5 顺序，以及 1、4、24、78 题的记忆集合。三个新设置使用空闲 GPU 0；原基线和 α=1 参照来自同型号 GPU 1。新增 1,113 条逻辑回答记录来自 534 次新推理和 579 条本轮相同完整输入共享，没有复用历史回答。1,823 次与参照完全相同输入的答案比较全部一致；零格式错误、零输出截断。

原始／白化／z 分数与 α=1 参照逐项完全一致，原始有序 Top-5 全部复现。独立协方差求逆核验共检查 89,092 个候选分数，以及全部输入、原图哈希、排序和生成 token 解码。工作区在首次检索期间增加了注释，来源校验因此中断该次尝试，尚未生成回答；正式结果改用固定代码副本。两条新增检索注释移除后严格恢复历史文件哈希，入口版本差异也单独记录；原工作区文件未回退。

五项方法标注、参照来源、分 domain 和逐题变化见 [λ=0.1 的 α 消融报告](reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/report.html)、[完整统计](reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/reader/summary.json)、[逐题 CSV](reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/reader/paired_results.csv)、[独立核验](reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/reader/independent_audit.json) 和 [实验协议](reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/protocol.json)。[执行入口](scripts/analysis/evaluate_baseline_question_only_alpha_lambda0p1.py) 固定 λ=0.1，默认 GPU 0；重跑指定新的 `--output`，仍从协议对应的 [代码副本清单](results/experiment_source_snapshots/baseline_question_only_lambda0p1_alpha_20261024/snapshot_manifest.json) 加载检索实现。

#### 195/371 caption_visual_master：五项标注

**方案标识：`master_answer_query` / `caption_visual_master`＋`multi_path`；最终问答准确率 `195/371 = 0.525606469`。** 当前在线对应 master memory 路径、`--retrieval multi_path --online-weighting none --top-k 5`。代码标注统一使用 `【195/371 多路径方案 · 序号. 名称】`。

| 标注 | 该方案的确切定义 | 对应位置 |
| --- | --- | --- |
| **1. 候选记忆结构** | 同 domain 的图片叶子按 caption embedding 经 Ward/silhouette 离线分簇；master 保存成员及聚合视觉向量，叶子保存该轮 User/Assistant 对话＋完整 Qwen caption。8 个 domain 共 438 条叶子、88 个 master；Home 为 89 条叶子、25 个 master。 | 当前：`CaptionVisualMasterMemory`；历史结构：[各 domain 的 memory](reports/adaptive_retrieval_comparison_20261020/inputs_by_domain)。 |
| **2. 查询内容** | 仅由问题和选项构造、预先冻结四条完整候选陈述。每个 master 选一句，形成 `q_m = retrieval_query + "\nCandidate answer: " + selected_statement`；保留既有题图 caption。拼入的是完整陈述，不是原选项短语。 | 当前：`MultiPathRetrieval.plan()`；历史候选：[candidates.jsonl](reports/all_master_answer_query_20261021/candidates.jsonl)。 |
| **3. 检索方式** | 每个 master 用视觉均值与四条陈述的 CLIP 余弦选择一句，然后用对应 BGE 查询匹配本簇所有叶子。当前 domain 的全部 master 都参与，将所有叶子的原始 BGE 分数合并取全局 Top-5；同分按源顺序。CLIP 分数只选陈述，不融合到叶子分数；原 195/371 不含白化或 z-score。 | 当前：`MultiPathRetrieval.plan()` 和 `retrieve_question()`；历史：[`probe_all_master_answer_query.py`](scripts/analysis/probe_all_master_answer_query.py#L198)。 |
| **4. 编码方式** | CLIP ViT-B/32 以 512 维归一化文本向量匹配已保存、归一化的 master 视觉均值，候选最长 77 token；不重新编码历史图片，不使用叶子二维坐标打分。BGE-small-en-v1.5 以 384 维归一化向量匹配扩展查询与原始对话＋完整 caption，查询前缀、512-token 上限及历史截断策略与全量基线相同。 | 当前：`choose_master_answers()`、`retrieve_question()` 和 `encode_memory_text()`；历史模型与编码参数：[protocol.json](reports/all_master_answer_query_20261021/protocol.json)。 |
| **5. 回答方式** | 多路径 Top-5 的原始对话＋完整 caption 按排名拼接，再加原问题、选项及题目原图（如有）。沿用 Qwen3-VL-8B-Instruct：BF16、seed=42、`do_sample=False`、`max_new_tokens=16`，只输出选项字母。候选陈述不进入 reader。固定选项排列下答对 195/371。 | 当前：`reader_text` 输出边界；历史实际回答：[`evaluate_all_master_answer_reader.py`](scripts/analysis/evaluate_all_master_answer_reader.py#L37)，结果：[reader summary](reports/all_master_answer_query_20261021/reader/summary.json)。 |

历史 [问答汇总](reports/all_master_answer_query_20261021/reader/summary.json) 与 [检索汇总](reports/all_master_answer_query_20261021/retrieval_summary.json) 的 `method_annotations.master_answer_query` 保存这五项，和 `baseline_question_only` 分开标注；[报告中的五项说明](reports/all_master_answer_query_20261021/report.html#master-method-annotations) 同步展示。当前 master 在线结果及后续自动输出也记录 master 结构、完整陈述和 CLIP/BGE 的不同职责。仅检索结果保留 `reader_called=false`、`measured_qa_accuracy=null`；195/371 只归属于历史未赋权 QA，不归属于新 z-score 结果。

本次只补充代码注释、说明及结果元数据，评分逻辑和原有数值不变；[本次标注审计](results/caption_visual_master_annotations_20261023/audit.json) 保存标注前后哈希和内容核验。

#### master_answer_query 加 z-score：λ=0.01、α=1

2026-10-23 已在 GPU 1 完成原 195/371 方案的全部 371 题配对实验。使用 `caption_visual_master` memory 和 `multi_path`，保留已冻结的完整候选陈述、各 master 的视觉匹配选择、扩展查询及原始对话＋完整 caption。每个 master 的全部叶子组成一个簇，在线计算白化和 z-score，按 `final = raw + std_pop(raw in master) * tanh(z_white)` 修正分数，再合并所有叶子取 Top-5。固定 λ=0.01、α=1；单叶子及零方差组不加修正。

| 方案 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 |
| --- | --- | --- | --- | --- |
| `master_answer_query` 未赋权 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 |
| `master_answer_query_zscore` | 201 / 371 | 54.1779% | 473 / 1041 | 301 / 371 |

24 题错→对、18 题对→错，净增加 6 题（+1.6173 个百分点，McNemar 双侧 p=0.4408）。265 题的 Top-5 顺序改变，其中 187 题的记忆集合改变。目标记忆召回总数下降，不能把问答准确率上升解释为召回更多目标；当前结果为同一测试集上的描述性比较，尚未达到统计显著。

两组回答均在本轮重新执行，742 条记录来自 636 次新推理和 106 条同轮相同输入共享，不复用历史回答。未赋权组的全部 371 个 Top-5、回答输入和答案与历史结果一致；3,862 次 master 陈述选择、扩展查询和成员分组全部复现。独立核验通过：逐簇协方差求逆与在线 SVD 分数相符（最大误差小于 `4e-7`），并重算标准化、辅助修正、全局排序、完整输入、原图哈希和 token 解码；零格式错误、零输出截断。

五项方法标注和分 domain 统计见 [完整报告](reports/master_answer_query_zscore_20261023/report.html)、[问答汇总](reports/master_answer_query_zscore_20261023/reader/summary.json)、[逐题对照](reports/master_answer_query_zscore_20261023/reader/paired_results.csv)、[独立核验](reports/master_answer_query_zscore_20261023/reader/independent_audit.json)。执行入口为 [evaluate_master_answer_query_zscore.py](scripts/analysis/evaluate_master_answer_query_zscore.py)，固定 λ=0.01、α=1，默认 `--stage all --device cuda:1`；重跑时指定新的 `--output` 目录。[audit_master_answer_query_zscore.py](scripts/analysis/audit_master_answer_query_zscore.py) 可独立核验完成的输出。

#### master_answer_query 的 α=10：固定 λ=0.01

2026-10-23 已在 GPU 1 完成全部 371 题。沿用上面的 master 分组、完整候选陈述、查询、叶子文本和回答配置，仅将辅助赋权的 α 从 1 调整为 10；`final = raw + 10 * std_pop(raw in master) * tanh(z_white)`。未赋权对照同时重新运行，α=1 使用已核验的上一轮结果。

| 方案 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 |
| --- | --- | --- | --- | --- |
| `master_answer_query` 未赋权，本轮复现 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 |
| λ=0.01、α=1，上一轮参照 | 201 / 371 | 54.1779% | 473 / 1041 | 301 / 371 |
| **λ=0.01、α=10，本轮评测** | **180 / 371** | **48.5175%** | **361 / 1041** | **260 / 371** |

相对 α=1，21 题错→对、42 题对→错，净减少 21 题（−5.6604 个百分点）；331 题的 Top-5 顺序改变，其中 281 题的集合改变。相对未赋权，27 题错→对、42 题对→错，净减少 15 题。McNemar 双侧 p 分别为 0.01114 和 0.09119，均未对同一测试集上的多次参数比较作校正。这次比较中，增大 α 同时降低了问答准确率和目标记忆召回。

控制核验：3,862 次 master 陈述选择、扩展查询和成员分组全部一致；原始分数、白化分数和 z 分数与 α=1 逐项完全相同，修正项为原来的 10 倍（最大浮点误差 `1.12e-16`）。本轮 742 条回答记录来自 719 次新推理和 23 条同轮相同输入共享，不复用历史回答；未赋权组全部 371 个答案再次复现历史结果。独立核验覆盖 22,273 个候选分数、全部回答输入和 token 解码，协方差求逆与在线分数最大误差小于 `4e-7`；零格式错误、零输出截断。

五项方法标注、逐 domain 结果和对照见 [完整报告](reports/master_answer_query_alpha10_20261023/report.html)、[问答汇总](reports/master_answer_query_alpha10_20261023/reader/summary.json)、[逐题对照](reports/master_answer_query_alpha10_20261023/reader/paired_results.csv)、[独立核验](reports/master_answer_query_alpha10_20261023/reader/independent_audit.json)。执行入口为 [evaluate_master_answer_query_alpha10.py](scripts/analysis/evaluate_master_answer_query_alpha10.py)，固定 λ=0.01、α=10；独立核验入口为 [audit_master_answer_query_alpha10.py](scripts/analysis/audit_master_answer_query_alpha10.py)。

#### master_answer_query 的 α 消融：固定 λ=0.01

2026-10-23 补充完成 α=0.1、0.01 的 GPU 1 全量评测。两组各自执行全部 371 题和未赋权对照，沿用相同 memory、完整候选陈述、查询、对话＋完整 caption、Top-5 和回答配置。

| 设置 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 | 相对未赋权净变化 |
| --- | --- | --- | --- | --- | --- |
| 未赋权，本轮两次复现 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 | 0 |
| **α=0.01，本轮评测** | **195 / 371** | **52.5606%** | **500 / 1041** | **300 / 371** | **0** |
| **α=0.1，本轮评测** | **194 / 371** | **52.2911%** | **495 / 1041** | **300 / 371** | **−1** |
| α=1，已完成参照 | 201 / 371 | 54.1779% | 473 / 1041 | 301 / 371 | +6 |
| α=10，已完成参照 | 180 / 371 | 48.5175% | 361 / 1041 | 260 / 371 | −15 |

α=0.01 相对未赋权没有正误翻转，8 题 Top-5 顺序改变，其中 3 题集合改变。α=0.1 有 3 题错→对、4 题对→错，71 题排序改变，其中 30 题集合改变。相对 α=1，两组分别少答对 6 题和 7 题；配对 McNemar 双侧 p 分别为 0.4408、0.3368，未做多重比较校正。当前同一测试集的四个 α 设置中，α=1 的问答准确率最高。

两组均逐项复现 α=1 的原始／白化／z 分数，master 选择和分组一致，只有修正项乘以相应 α。两次运行合计 1,484 条逻辑回答记录，来自 821 次新推理和 663 条各自运行内相同完整输入共享；无历史回答复用。未赋权对照的全部答案均复现历史结果，两组独立核验通过，零格式错误、零输出截断。

汇总包含五项方法标注、来源哈希、分 domain 统计及所有设置的逐题对照：[α 消融报告](reports/master_answer_query_alpha_ablation_20261023/report.html)、[汇总 JSON](reports/master_answer_query_alpha_ablation_20261023/summary.json)、[逐题 CSV](reports/master_answer_query_alpha_ablation_20261023/paired_results.csv)。执行入口为 [evaluate_master_answer_query_alpha.py](scripts/analysis/evaluate_master_answer_query_alpha.py)，通过 `--alpha 0.1` 或 `--alpha 0.01` 选择本轮设置；[audit_master_answer_query_alpha.py](scripts/analysis/audit_master_answer_query_alpha.py) 独立核验单组输出，[summarize_master_answer_query_alpha.py](scripts/analysis/summarize_master_answer_query_alpha.py) 核对后合并全部参照。

#### master_answer_query 的 λ 消融：固定 α=1

2026-10-23 已补齐 λ=0.1、1.0 的全部 371 题问答评测。两组分别在 GPU 1 重新运行检索和回答，并各自复跑未赋权对照。保持原 195/371 方案的完整候选陈述、各 master 查询与成员、对话＋完整 caption、Top-5 和 Qwen 回答配置，只改变白化正则 λ；α 固定为 1。

| 设置 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 | 相对未赋权净变化 |
| --- | --- | --- | --- | --- | --- |
| 未赋权，本轮两次复现 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 | 0 |
| λ=0.01，已完成参照 | 201 / 371 | 54.1779% | 473 / 1041 | 301 / 371 | +6 |
| **λ=0.1，本轮评测** | **202 / 371** | **54.4474%** | **474 / 1041** | **299 / 371** | **+7** |
| **λ=1.0，本轮评测** | **199 / 371** | **53.6388%** | **474 / 1041** | **298 / 371** | **+4** |

相对 λ=0.01，λ=0.1 有 7 题错→对、6 题对→错，净增加 1 题（+0.2695 个百分点，McNemar 双侧 p=1.0）；λ=1.0 有 8 题错→对、10 题对→错，净减少 2 题（−0.5391 个百分点，p=0.8145）。相对未赋权，两组分别为 20 题错→对／13 题对→错，以及 16 题错→对／12 题对→错。λ=0.1 当前答对数最高，但与 0.01 只有一题差异；这是同一测试集上的参数比较，所有 p 值均未作多重比较校正。增大 λ 只减弱白化，α=1 的辅助赋权仍然存在。

两组原始匹配分数与 λ=0.01 逐项完全一致，3,862 次 master 选择、扩展查询及分组全部复现，白化和 z 分数按新 λ 重新计算。两次运行共 1,484 条逻辑回答记录，来自 1,151 次新推理及 333 条各自运行内完全相同输入共享，无历史回答复用；未赋权答案均逐题复现 195/371。独立协方差求逆核验覆盖每组 22,273 个候选分数，并检查全部回答输入、原图与 token 解码；两组均通过，零格式错误、零输出截断。

五项方法标注、来源哈希、各 domain 及逐题比较见 [λ 消融报告](reports/master_answer_query_lambda_ablation_20261023/report.html)、[汇总 JSON](reports/master_answer_query_lambda_ablation_20261023/summary.json)、[逐题 CSV](reports/master_answer_query_lambda_ablation_20261023/paired_results.csv)。执行入口 [evaluate_master_answer_query_lambda.py](scripts/analysis/evaluate_master_answer_query_lambda.py) 使用 `--whitening-lambda 0.1` 或 `--whitening-lambda 1.0`，默认 GPU 1、α=1；[独立核验入口](scripts/analysis/audit_master_answer_query_lambda.py) 和 [汇总入口](scripts/analysis/summarize_master_answer_query_lambda.py) 分别核验单组与合并参照。

#### master_answer_query 固定 λ=0.1 的 α 消融

2026-10-24 已完成 α=0.01、0.1、10.0 的全部 371 题，α=1.0 使用前次已完成且通过严格来源校验的 202/371 参照。保持同一 `caption_visual_master` memory、完整候选陈述、master 查询、对话＋完整 caption、Top-5 和回答配置，固定 λ=0.1，仅改变 α。

| 设置 | 答对 / 总题数 | 准确率 | Top-5 目标记忆召回 | 至少命中一条目标的题数 | 相对未赋权净变化 |
| --- | --- | --- | --- | --- | --- |
| 未赋权，本轮三次复现 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 | 0 |
| α=0.01，本轮评测 | 195 / 371 | 52.5606% | 500 / 1041 | 300 / 371 | 0 |
| α=0.1，本轮评测 | 195 / 371 | 52.5606% | 496 / 1041 | 300 / 371 | 0 |
| **α=1.0，已完成参照** | **202 / 371** | **54.4474%** | **474 / 1041** | **299 / 371** | **+7** |
| α=10.0，本轮评测 | 185 / 371 | 49.8652% | 368 / 1041 | 262 / 371 | −10 |

相对未赋权，α=0.01 没有正误翻转，7 题 Top-5 排序改变，其中 2 题集合改变；α=0.1 有 2 题错→对、2 题对→错，53 题排序改变，其中 22 题集合改变；α=10 有 27 题错→对、37 题对→错，281 题排序改变，其中 259 题集合改变。相对 α=1，三组分别净减少 7、7、17 题。α=10 的配对变化为 21 题错→对、38 题对→错，McNemar 双侧 p=0.03634，未作多重比较校正。当前同一测试集、固定 λ=0.1 的比较中，α=1 答对数最高。

GPU 1 当时有其他任务，新增三组使用同型号的 GPU 0；α=1 参照来自 GPU 1，设备差异记录在协议中。三组原始／白化／z 分数均与参照逐项完全相同，3,862 次 master 选择、查询及成员分组全部复现，仅修正量按 α 缩放。所有未赋权逐题答案复现 195/371，各设置间完全相同输入的答案核对一致。新增 2,226 条逻辑回答记录来自 1,454 次新推理和 772 条各自运行内相同输入共享，不复用历史回答；三组独立协方差求逆、输入、原图及 token 解码核验全部通过，零格式错误、零输出截断。

结果包含五项方法标注、来源、分 domain 统计和四个 α 的逐题对照：[固定 λ=0.1 的 α 消融报告](reports/master_answer_query_lambda0p1_alpha_ablation_20261024/report.html)、[汇总 JSON](reports/master_answer_query_lambda0p1_alpha_ablation_20261024/summary.json)、[逐题 CSV](reports/master_answer_query_lambda0p1_alpha_ablation_20261024/paired_results.csv)。执行入口 [evaluate_master_answer_query_alpha_lambda0p1.py](scripts/analysis/evaluate_master_answer_query_alpha_lambda0p1.py) 固定 λ=0.1，以 `--alpha` 选择设置；默认 GPU 0。重跑需指定新的 `--output`。另有 [独立核验入口](scripts/analysis/audit_master_answer_query_alpha_lambda0p1.py) 和 [汇总入口](scripts/analysis/summarize_master_answer_query_alpha_lambda0p1.py)。

#### Memory 路径、检索策略与在线 z-score

Online 的记忆入口是 `--memory-path`。目录内的 `memory.json` 保存 `offline_method`、domain、叶子数量、文本格式、编码配置和文件哈希；online 自动加载对应 memory 对象。更换路径即更换检索记忆库，不需要重复声明构建方法。`--memory` 保留为旧命令兼容参数；若与路径同时指定，必须与库内方法一致。

三层独立配置：

| 层 | 参数 | 作用 |
| --- | --- | --- |
| 记忆库 | `--memory-path` | 从 `memory.json` 读取 offline 方法、叶子文本及已有 master 结构 |
| 查询／检索 | `--retrieval full_scan` / `multi_path` | 原始查询直接匹配／各 master 视觉匹配完整候选陈述后扩展查询 |
| 可选赋权 | `--online-weighting none` / `z_score` | 原始余弦分数／原始余弦分数加在线修正 |

`natural` 的默认检索是 `full_scan`，完整同 domain 候选记忆作为一个簇；`caption_visual_master` 的默认检索是 `multi_path`，每个已有 master 作为一个簇。master 库也支持 `full_scan`，此时各 master 使用同一原始查询，z-score 仍遵循该库的 master 分组。natural 没有视觉 master，因此不支持 `multi_path`。不传赋权参数时默认 `none`。

两种检索保持原始“对话 + 换行 + Memory description: + 完整 caption”文本。full_scan 对应历史 **184/371 = 49.5957%** 方案；multi_path 使用 **195/371 = 52.5606%** 方案冻结的完整候选陈述，绝不改为原始选项短语。查询中的既有图片 caption 保留。候选陈述是查询输入，由 `--candidates` 提供，默认使用 `reports/all_master_answer_query_20261021/candidates.jsonl`；它不属于离线 memory，也不传给回答模型。新问题需要提供与问题和选项对应的陈述文件。

赋权公式为 `final = raw + alpha * std_pop(raw in group) * tanh(z_white)`。`z_white` 来源于组内全部叶子向量的中心化、PCA、ridge 白化打分和标准化。默认／当前调试固定 **λ=0.1、α=1**；`none` 跳过 PCA，α=0 恢复同一查询、同一文本的原始分数。单叶子及零方差组的修正为零。最终始终合并修正后的匹配分数取 Top-k，不能直接混合各组 z-score 排名。

已有离线结果可一次性导出为自描述 memory；只复制原文本和既有 master 数据，不重新聚类，也不纳入问题、答案或标签。新目录必须为空：

```bash
python3 scripts/visual_memory_pipeline.py offline --memory caption_visual_master \
  --memory-source reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design \
  --output results/memory_stores_20261023/Home_Renovation_Interior_Design/caption_visual_master
```

将上面的 `--memory` 与输出末级目录改为 `natural`，即可导出相同文本的自然记忆库。master 库包含 `memory.json`、`leaves.json`、`graph.json`、`master_sums.npy`；natural 只需前两个文件。导出后 online 不依赖原始实验目录里的 memory 或 protocol 文件；问题文件和候选陈述仍由 online 单独传入。

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory-path results/memory_stores_20261023/Home_Renovation_Interior_Design/caption_visual_master \
  --retrieval multi_path --online-weighting z_score \
  --whitening-lambda 0.1 --reweight-alpha 1 --device cuda:1 \
  --question-source reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/inputs.json \
  --question-ids Q10 --top-k 5 \
  --output results/debug_online_retrieval/caption_visual_master/multi_path/z_score
```

全量检索改为 natural 路径及 `--retrieval full_scan`。关闭赋权使用 `--online-weighting none` 并移除 λ/α 参数；不要改动 memory 路径或检索方式。不传 `--question-ids` 则处理 question-source 的全部问题。旧参数 `--retrieval-scoring z_reweight` 仅作为 `--online-weighting z_score` 的兼容别名。

VS Code 中的 **Online z-score（caption_visual_master · Home Q10）** 和 **Online z-score（natural · Home Q10）** 分别指向完整 Home 的两种库：89 条叶子，master 库有 25 个簇，natural 为一个簇。固定 λ=0.1、α=1；BGE 及多路径 CLIP 使用 GPU 1（`cuda:1`），NumPy 完成 PCA。断点放在 `scripts/memory/online_retrieval.py` 的 `FullScanRetrieval.plan()`、`MultiPathRetrieval.plan()`、`retrieve_question()`，以及 `scripts/memory/online_zscore.py` 的 `ZScoreWeighting.apply()`。

该入口目前支持 MemEye 的 `--stage prepare` 检索，输出未赋权／赋权 Top-k、可直接用于 reader 的原始上下文、逐叶标量分数和数据哈希，尚不调用回答模型。查询、叶子和 PCA 向量每次在线生成、用完即弃，不写回 memory。历史 Card 记忆有一条 577-token 输入按原基线截到 512；此策略随 memory 保存，并校验 ID、完整文本哈希和长度，其他超长输入报错。

2026-10-24 已清理错误的 177/371 原始选项＋caption-only 多路径实现及其 Home 试验、全量赋权、旧 λ 消融、调试输出和专属脚本，同时移除无人调用的旧兼容模块。保留 `master_answer_query` 的 195/371 基线、完整候选陈述、全部有效 z-score 消融、共享 memory 和编码／白化组件；`baseline_question_only_zscore` 的 183/371 等有效全量基线结果也完整保留。删除前后逐文件核对保留项哈希，清单和检查结果见 [清理记录](results/cleanup_option_caption_multipath_20261024/validation.json) 与 [删除清单](results/cleanup_option_caption_multipath_20261024/removal_manifest.json)。

2026-10-23 已在 GPU 1 完成新的 memory-path 入口检索核验：两种记忆各 371 题的原始有序 Top-5 与上述历史方案全部一致，全部 master 的候选陈述选择一致；最大原始分差小于 `5e-7`。λ=0.1、α=1 的赋权分别改变 natural 的 57/371、multi_path 的 213/371 道题的有序 Top-5。natural 的完整 QA 已完成，结果见上文；multi_path 固定 α=1 的 λ=0.01、0.1、1.0 完整 QA 分别为 201、202、199 / 371，详见上面的 λ 消融。早期 [全量核验](results/online_memory_path_validation_20261023/validation.json) 和 [两项 GPU 调试配置及关闭赋权核验](results/online_memory_path_validation_20261023/debug_validation.json) 仅记录当时检索阶段的验证，不包含后续问答结果。46 项相关测试通过。

#### MemEye 回答模型 API 消融

在既有全领域实验上只替换最后的回答模型：保留 8 个领域的 371 道题、固定选项顺序、两种检索各自的有序 Top5、完整对话和 caption，以及题目原图。两种条件是原 query 全量检索与 master 候选答案扩展 query 检索；候选答案不进入回答上下文。检索池沿用 438 条图片叶子，不扩大到纯文本轮次。

```bash
# 核验并冻结全部输入，不调用 API
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-4o-mini --stage prepare
# 纯文本题和带图题各一个；成功结果可供本次全量运行续用
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-4o-mini --stage run --api-smoke
# 核验、续跑全部问题、计算配对指标
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-4o-mini --stage all
```

默认从 `/storage/hewenwen/NTU Decentralized Agent/.env` 读取官方 OpenAI 配置，可用 `--api-env-file` 指定；进程环境变量优先。密钥仅用于鉴权，不写入实验结果。默认输出 `results/memeye_reader_ablation_20261021/gpt-4o-mini/`；模型或配置变化应使用新的 `--output`，不覆盖已固定实验。默认并发数 3，可在 prepare 时用 `--api-workers` 指定；已固定实验续跑时参数需保持一致。这条 API 路径不加载本地 Qwen、CLIP 或 BGE，也不需要 GPU。

`evaluation/memeye_reader_inputs.py` 重建并核对历史 reader 的完整输入，单独存放 `inputs/gold.jsonl`。`evaluation/memeye_api_reader.py` 只将固定 system prompt、user prompt 和原始问题图片交给 `evaluation/openai_chat_transport.py`。请求采用 Chat Completions、temperature=0、seed=42、16-token 输出上限；原图转为 data URL，`detail=high`。API 与 Qwen 的 tokenizer 和图像预处理各自不同，不能称为像素预处理完全相同。API 的 seed 也不保证服务端完全确定性。

742 条逻辑回答对应 719 个不同请求，相同输入仅在本次运行内共享一次新推理。每次调用记录输入哈希、返回的模型版本、请求/响应 ID、system fingerprint、token 用量和结束原因；请求与源码变更会阻止续跑。只对临时网络、限流和服务端错误重试同一请求，认证或配额错误停止后续批次；不因答案错误或格式错误重新生成。错误格式仍计入固定分母。

`comparison.md`、`summary.json`、`paired_results.csv` 由完整运行后的 analyze 阶段生成，比较两种 reader × 两种检索的正确率及逐题变化。2026-10-21 已完成：Qwen 参考为基线 184/371、新方法 195/371；API 实际返回 `gpt-4o-mini-2024-07-18`，两种检索均为 196/371（52.83%），新方法 31 题错转对、31 题对转错。719 次 API 调用、23 条本轮共享均通过独立审计，零重试、拒答、格式错误或截断。详见 [实验报告](results/memeye_reader_ablation_20261021/gpt-4o-mini/comparison.md)。

指标是固定选项排列的配对 MCQ accuracy，不是论文四次旋转 EM。历史来源目录迁移与辅助源码变化记入 `inputs/input_audit.json`，历史协议不改写；当前对照由逐字段一致的固定 reader 输入和原图哈希保证。`public_data_provenance.json` 进一步记录了两条保留对话相对于公开源的轻微改写，以及部分 caption 历史生成链不能完整恢复的限制。两种 reader 均使用相同冻结版本。

##### GPT-5.4 回答模型对照

GPT-5.4 使用单独的 `evaluation/memeye_gpt5_reader.py` 与 `evaluation/openai_gpt5_chat_transport.py`，不修改已完成 GPT-4o-mini 实验冻结的代码和哈希。统一入口会将 `gpt-5.4` 固定为 `gpt-5.4-2026-03-05`；当前 GPT-5 路径只实现此快照的 `reasoning_effort=none` 对照。

```bash
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-5.4-2026-03-05 --stage prepare
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-5.4-2026-03-05 --stage run --api-smoke
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-5.4-2026-03-05 --stage all
```

默认输出为 `results/memeye_reader_ablation_20261021/gpt-5.4-2026-03-05_none/`。使用官方 `https://api.openai.com/v1` 与 `OPENAI_API_KEY`；不会因 NTU 的裸模型名路由改用 CODEX 凭据或第三方端点。生成设置为 temperature=0、max_completion_tokens=16、reasoning_effort=none，不发送 seed。输入、system/user prompt、Top5 和问题原图与 Qwen/GPT-4o-mini 逐条一致；seed 与 API 参数格式的差异在协议中明确记录。

接口返回的 reasoning token 用量逐条保存，缺失表示未知，不当作已核验的零推理。`comparison.md` 同时报告 Qwen、GPT-4o-mini 与本次 GPT-5.4 的两组结果；历史 caption 和两条保留对话改写仍沿用相同冻结输入。完成后可运行 `python3 scripts/analysis/audit_memeye_gpt5_reader.py`，独立重构全部请求、检查回答来源并重算三模型分数，输出 `independent_audit.json`。

2026-10-21 已完成 GPT-5.4 **仅替换回答模型、保留 Qwen caption** 的全量对照：基线 227/371（61.19%），新方法 235/371（63.34%），净增 8 题、+2.16 个百分点；36 题错转对、28 题对转错，McNemar 双侧精确 p=0.382。719 次实际调用均返回指定快照，明确报告 reasoning tokens=0，无用量缺失、格式错误、空答或截断；23 条同次共享已独立核验。详见 [完整报告](results/memeye_reader_ablation_20261021/gpt-5.4-2026-03-05_none/comparison.md) 和 [独立审计](results/memeye_reader_ablation_20261021/gpt-5.4-2026-03-05_none/independent_audit.json)。

另有 **GPT-5.4 生成历史／题图 caption、重建记忆与检索、再由 GPT-5.4 回答** 的完整流程实验：未赋权 baseline 为 **236/371**，master_answer_query 为 **241/371**，见 [完整流程报告](results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05/comparison.md)。它与上述 227/371、235/371 使用不同 caption 和 Top-5，不能混称为同一套 GPT-5.4 baseline。

##### GPT-5.4 caption：未加权基线的 seed=42 回答复测

2026-10-24，确认历史 236/371、241/371 的请求和返回模型均为 `gpt-5.4-2026-03-05` 后，冻结同一 GPT caption 和记忆，仅在回答阶段新增 `seed=42`。未重生成 caption、未重建 memory，也未重新检索。逐条独立重建 742 条 API 请求，证实除 seed 外，system/user、原图字节、Top-5 顺序、模型快照、temperature=0、reasoning_effort=none、16-token 输出上限均一致。

| 未加权方案 | 历史未传 seed | 本次 seed=42 | 准确率 | 错→对 / 对→错 |
|---|---:|---:|---:|---:|
| `baseline_question_only` | 236/371 | **233/371** | **62.8032%** | 14 / 17 |
| `master_answer_query` | 241/371 | **231/371** | **62.2642%** | 7 / 17 |

720 次本轮新调用支撑 742 条逻辑记录，22 条相同完整请求共享本轮新回答。四次先行 seed 探测不参与评分；无历史答案复用、失败、格式错误、空答、截断或非零 reasoning tokens。四项测试（含全量模拟评测与错误保留分母）和独立结果核验通过。

seed 探测已获 HTTP 200，但 Health/Q38 的相同完整请求、相同 seed=42 重复调用分别回答 A、C；返回的 `system_fingerprint` 均为空。该观察说明固定 seed 不保证复现，不证明 seed 被忽略，也不能定位服务端非确定性的具体来源。此轮仅是一次固定 seed 复测，分差不能归因于 seed；此前未传 seed 的 z-score 结果与本轮不构成仅改变 z-score 的对照。五项方法标注、seed 与限制均写入结果文件。

独立入口：

```bash
python3 scripts/analysis/rerun_memeye_gpt54_seeded_baselines.py --stage all --seed 42
python3 scripts/analysis/audit_memeye_gpt54_seeded_baselines.py
```

该输出目录会校验协议并续跑，已完成时不重复发送请求。历史无 seed 适配器、签名、结果均保留；新适配器为 `scripts/evaluation/openai_gpt54_seeded_transport.py`。见 [结果报告](results/memeye_gpt54_seed42_baselines_20261024/report.html)、[逐题对照](results/memeye_gpt54_seed42_baselines_20261024/paired_results.csv)、[seed 探测](results/memeye_gpt54_seed42_baselines_20261024/seed_support_probe.json)、[独立核验](results/memeye_gpt54_seed42_baselines_20261024/independent_audit.json)。

##### GPT-5.4 caption：未加权基线的 seed=1042 复测

2026-10-24，沿用上一节的全部冻结输入和回答设置，仅将 API seed 从 42 改为 1042。独立重建 742 条请求，确认除了 seed 之外均完全一致；没有重生成 caption、重建 memory 或重新检索，两套方案均不叠加 z-score。

| 方案 | 历史未传 seed | seed=42 | seed=1042 | 相对 seed=42：错→对 / 对→错 |
|---|---:|---:|---:|---:|
| `baseline_question_only` | 236/371 | 233/371 | **235/371（63.3423%）** | 7 / 5 |
| `master_answer_query` | 241/371 | 231/371 | **235/371（63.3423%）** | 6 / 2 |

相对 seed=42，两套方案分别有 17、10 题的答案字母改变，净增 2、4 题；配对 McNemar 双侧精确 p 分别为 0.7744、0.2891。每个 seed 只有一轮，差异同时包含跨调用波动，不能据此宣称 seed=1042 稳定更优。

完成 720 次本轮新调用、742 条逻辑记录，其中 22 条相同请求共享本轮回答。文本／带图先行检查使用同一协议并计入本轮；无历史答案复用、请求失败、格式错误、空答、截断或非零 reasoning tokens。全量模拟流程测试与独立请求／评分核验通过；五项方法标注已写入报告和结果文件。历史 seed=42 的代码与产物均未改写。

```bash
python3 scripts/analysis/rerun_memeye_gpt54_seed1042_baselines.py --stage all --seed 1042
python3 scripts/analysis/audit_memeye_gpt54_seed1042_baselines.py
```

见 [对比报告](results/memeye_gpt54_seed1042_baselines_20261024/report.html)、[逐题 seed 对照](results/memeye_gpt54_seed1042_baselines_20261024/seed_comparison.csv)、[完整统计](results/memeye_gpt54_seed1042_baselines_20261024/summary.json)、[独立核验](results/memeye_gpt54_seed1042_baselines_20261024/independent_audit.json)。

##### GPT-5.4 别名：未加权基线的 seed=42 复测

2026-10-24，直接发送 `model="gpt-5.4"`、`seed=42`，完成两套未加权基线的全部 371 题。沿用历史 GPT-5.4 caption、memory、完整排名、有序 Top-5、问题原图及提示词；相对日期快照 seed=42 对照，742 条完整请求仅 `model` 字段不同。回答仍使用 temperature=0、reasoning_effort=none、max_completion_tokens=16。caption 未重新生成，检索未重算，未叠加 z-score。

| 方案 | 快照未传 seed | 快照 seed=42 | 别名 seed=42 | 相对快照 seed=42：错→对 / 对→错 |
|---|---:|---:|---:|---:|
| `baseline_question_only` | 236/371 | 233/371 | **236/371（63.6119%）** | 7 / 4 |
| `master_answer_query` | 241/371 | 231/371 | **238/371（64.1509%）** | 9 / 2 |

实际 720 次新调用全部返回 `gpt-5.4-2026-03-05`，请求别名与返回版本分别记录。742 条逻辑记录中，22 条相同请求共享本轮新回答；文本／图片先行检查计入本轮，没有复用历史或独立探测答案。请求失败、格式错误、空答、截断和非零 reasoning tokens 均为零。两项测试（含完整模拟评测、错误保留分母和续跑）及独立请求／评分审计通过，437 张独立 caption 图片的来源与冻结输入均已核验。

相对快照 seed=42，两套方案分别有 14、11 题的答案字母改变，净增 3、7 题；相对历史未传 seed 为 0、−3 题。两种请求名本轮指向同一返回版本，每种设置仅一轮，分差不能证明别名更准确或更可复现。五项方法标注及逐题对照均写入结果文件。

独立入口保留历史快照代码与结果：

```bash
python3 scripts/analysis/rerun_memeye_gpt54_alias_seed42_baselines.py --stage all --model gpt-5.4 --seed 42
python3 scripts/analysis/audit_memeye_gpt54_alias_seed42_baselines.py
```

见 [别名对比报告](results/memeye_gpt54_alias_seed42_baselines_20261024/report.html)、[逐题 CSV](results/memeye_gpt54_alias_seed42_baselines_20261024/alias_comparison.csv)、[完整统计](results/memeye_gpt54_alias_seed42_baselines_20261024/summary.json)、[独立核验](results/memeye_gpt54_alias_seed42_baselines_20261024/independent_audit.json)。适配器为 `scripts/evaluation/openai_gpt54_alias_transport.py`，直接发送别名；该独立实验入口与历史强制快照入口分开。

##### GPT-5.4 别名 seed=42：两种检索叠加 z-score（λ=0.01、α=1）

2026-10-24，在上一轮 `gpt-5.4` 别名、seed=42 的未加权 236/371、238/371 基础上，只改变在线检索赋权，完成两套方案全部 371 题。沿用同一批 GPT-5.4 caption、原始对话、master、完整候选陈述、题图与固定选项排列；caption 不重新生成。GPU 0 重算 BGE／CLIP 检索，742 组未加权完整排名均与原基线一致，原始分数最大差为 0。4,052 个白化分组通过独立 ridge 逆矩阵公式核验。

赋权沿用 `raw + α × std_pop(raw in group) × tanh(white_z)`，λ=0.01、α=1。`baseline_question_only` 在同 domain 全部记忆上计算；`master_answer_query` 在各 master 内计算，再按最终分数合并取 Top-5。所有回答请求均为字面别名 `gpt-5.4`、seed=42、temperature=0、reasoning_effort=none、max_completion_tokens=16。独立重建 742 条请求，核验相对未加权对照只改变检索上下文，其他请求参数不变。

| 方案 | 同别名／seed 未加权 | 加权后 | 变化 | 错→对 / 对→错 |
|---|---:|---:|---:|---:|
| `baseline_question_only + z-score` | 236/371（63.6119%） | **241/371（64.9596%）** | +5（+1.3477 pp） | 18 / 13 |
| `master_answer_query + z-score` | 238/371（64.1509%） | **235/371（63.3423%）** | −3（−0.8086 pp） | 16 / 19 |

全量方案有 182 题完整请求与未加权对照相同，其中 9 题答案字母改变（4 错→对、4 对→错，净 0）；master 方案有 111 题完整请求相同，其中 3 题答案改变（0 错→对、1 对→错，净 −1）。输入改变的子集分别净 +5、−2。固定 seed 仍存在跨调用波动；全部题目的配对 McNemar 双侧精确 p 分别为 0.4731、0.7359，一轮结果不能证明稳定收益。

目标记忆 Top-5 命中数分别从 513/1041 降到 495/1041、从 514/1041 降到 488/1041；至少命中一条目标的题数分别从 303 降到 300、从 306 降到 301。最终问答准确率与检索召回分别统计。

本轮 726 次独立新调用覆盖 742 条逻辑记录，16 条同轮相同输入共享回答；未复用历史答案。实际返回模型全部为 `gpt-5.4-2026-03-05`，与未加权对照一致。无请求失败、格式错误、空答、截断、拒答或非零 reasoning tokens；全部 fingerprint 为空。模拟完整流程（含格式错误保留分母、独立评分、续跑不重复调用）及结果独立核验通过。五项方法标注保存在 protocol、summary 和 HTML 报告中。

```bash
python3 scripts/analysis/evaluate_memeye_gpt54_alias_seed42_zscore.py --stage all --model gpt-5.4 --seed 42 --whitening-lambda 0.01 --reweight-alpha 1 --device cuda:0
python3 scripts/analysis/audit_memeye_gpt54_alias_seed42_zscore.py
```

默认目录保留此次签名与回答，重复执行只续跑缺失项；要做独立复测须指定新的 `--output`。见 [报告](results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/report.html)、[逐题 CSV](results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/paired_results.csv)、[完整统计](results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/summary.json)、[检索变化](results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/retrieval_comparison.json)、[独立核验](results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/independent_audit.json)。历史未传 seed 的 λ=0.01 加权结果（242/237）保持原记录，不作为本轮同 seed 对照。

##### GPT-5.5：重新生成 caption、重建记忆与检索、重新回答

2026-10-24，按用户要求完成 GPT-5.5 全流程实验。437 张独立图片的 caption 全部新生成，包含历史图与题图；438 条记忆的 caption 及 master 叶子文本逐条核对新 API 输出，未复用 GPT-5.4 或 Qwen caption。caption 与回答都请求字面别名 `gpt-5.5`，实际返回版本全部为 **`gpt-5.5-2026-04-23`**。caption 沿用未传 seed、temperature=0、reasoning_effort=none、256-token 上限；回答沿用 seed=42、temperature=0、reasoning_effort=none、16-token 上限。原图 `detail=high`。

对话、原图字节、题目／选项排列、完整候选答案陈述、BGE、CLIP、外部二维基底和聚类规则保持不变。GPU 0 根据新 caption 重建 **81 个 master**，重新计算两种未加权排名，再叠加 λ=0.01、α=1 的在线 z-score。742 组赋权前完整排名与本轮新基线完全相同，原始分数最大差为 0；3,829 个白化分组通过独立 ridge 逆矩阵公式核验。全量按同 domain 全部记忆分组，master 方案按每个新 master 分组。

| 方案 | GPT-5.4 同设置历史参考 | GPT-5.5 本轮 | 准确率 | 相对 GPT-5.4 正确数变化 |
|---|---:|---:|---:|---:|
| `baseline_question_only` | 236/371 | **250/371** | **67.3854%** | +14 |
| `baseline_question_only + z-score` | 241/371 | **237/371** | **63.8814%** | −4 |
| `master_answer_query` | 238/371 | **247/371** | **66.5768%** | +9 |
| `master_answer_query + z-score` | 235/371 | **251/371** | **67.6550%** | +16 |

同一 GPT-5.5 记忆上的赋权消融：全量方案从 250 降到 237，7 题错→对、20 题对→错，净 −13（−3.5040 pp），配对 McNemar 双侧精确 p=0.0192；master 从 247 升到 251，16 题错→对、12 题对→错，净 +4（+1.0782 pp），p=0.5716。两套方案分别有 169、99 题的完整输入未变，四组中相同完整请求共享本轮新回答，因此这些题的答案保持一致。上述 p 值为单轮、未作多重比较校正的配对统计；本轮 master 的 +4 不构成稳定收益证据。

目标记忆 Top-5 命中数分别为：全量 515/1041、全量加权 510/1041、master 512/1041、master 加权 481/1041；至少命中一条目标的题数分别为 296、295、303、299。问答准确率和检索命中分别统计。BGE 沿用原始 512-token 编码上限，本轮仅 Card 的 `PLAY_S2:R5` 对话＋caption 从 616 tokens 截取前 512 tokens 用于检索编码；记忆持久化和回答上下文仍保留全文，caption 生成本身没有截断。

完成 437 次新 caption 调用及 1,174 次新回答调用，覆盖四组各 371 题（1,484 条逻辑记录），310 条同轮记录共享相同请求的回答。另有一次合成可用性探测，不进入 benchmark。无请求失败、拒答、格式错误、空答、输出截断或非零 reasoning tokens。两个测试（请求字段及完整模拟评分／续跑）与独立 API 字节、caption 来源、检索、评分核验均通过。五项方法标注保存在回答 protocol、summary 与 HTML 报告中。

GPT-5.4 对照来自上一轮别名 seed=42 的 236/238、241/235。跨模型对照同时更换 caption 生成与回答，并因此重建了 master 与排名；它不是只替换 reader 的消融。GPT-5.5 四组之间则共享本轮同一批新 caption 和记忆。

可复现入口（默认目录为本次结果；独立复测时四条命令都传入同一新的 `--output`）：

```bash
python3 scripts/analysis/probe_memeye_gpt55.py
python3 scripts/evaluation/memeye_gpt55_captions.py --stage all
python3 scripts/analysis/retrieve_memeye_gpt55.py --device cuda:0
python3 scripts/analysis/evaluate_memeye_gpt55.py --stage all
python3 scripts/analysis/audit_memeye_gpt55.py
```

见 [完整报告](results/memeye_gpt55_seed42_full_20261024/report.html)、[逐题 CSV](results/memeye_gpt55_seed42_full_20261024/paired_results.csv)、[完整统计](results/memeye_gpt55_seed42_full_20261024/summary.json)、[caption 来源](results/memeye_gpt55_seed42_full_20261024/caption_model_provenance.json)、[独立核验](results/memeye_gpt55_seed42_full_20261024/independent_audit.json)。历史实验脚本与结果保持原样，新入口按阶段保存来源哈希、请求哈希和实际返回版本。

##### GPT-5：重新生成 caption、重建记忆与检索、重新回答

2026-10-24 完成 GPT-5 全流程实验：437 张独立图片的 caption 全部重新生成，覆盖历史图和题图；438 条记忆与新 master 叶子的 caption 均逐条追溯到本轮新 API 输出。caption 和回答都请求字面别名 `gpt-5`，实际返回版本全部为 **`gpt-5-2025-08-07`**，没有复用 GPT-5.4、GPT-5.5 或 Qwen 的 caption／回答。

回答固定 seed=42，caption 沿用未传 seed。GPT-5 的参数适配单独记录：`reasoning_effort=minimal`，不发送 temperature（使用服务端默认值），caption 与回答的 `max_completion_tokens` 均为 2048，包含推理与可见输出预算。此前 GPT-5.4／5.5 使用 none、temperature=0，以及 caption 256／回答 16 的预算，因此跨模型表是全流程历史对照，不能解释为仅替换 reader 或生成参数完全相同的消融。本轮实际报告的 reasoning tokens 均为 0，但请求档位仍为 minimal。

原始对话、图片字节、题目与选项排列、caption／回答提示词、完整候选答案陈述、CLIP、BGE、外部二维基底和聚类规则不变。GPU 0 用新 caption 重建 **72 个 master**；全量检索以同 domain 全部记忆作为赋权组，多路径按每个 master 的叶子分组。固定 λ=0.01、α=1，最终分数仍为 `raw + alpha * std_pop(raw in group) * tanh(white_z)`。742 组原始完整排名及分数复算完全一致；3,597 个白化分组、44,546 个叶子分数经过独立 ridge 逆矩阵公式核验，最大误差 1.23e-15 以内。

| 方案 | GPT-5.4 历史参考 | GPT-5.5 历史参考 | GPT-5 本轮 | 准确率 |
|---|---:|---:|---:|---:|
| `baseline_question_only` | 236/371 | 250/371 | **219/371** | **59.0296%** |
| `baseline_question_only + z-score` | 241/371 | 237/371 | **221/371** | **59.5687%** |
| `master_answer_query` | 238/371 | 247/371 | **223/371** | **60.1078%** |
| `master_answer_query + z-score` | 235/371 | 251/371 | **221/371** | **59.5687%** |

同一 GPT-5 记忆上的 z-score 消融：全量 18 题错→对、16 题对→错，净 **+2**（+0.5391 pp，配对 McNemar 双侧精确 p=0.8642）；master 18 题错→对、20 题对→错，净 **−2**（−0.5391 pp，p=0.8714）。两组分别有 179、102 题的完整输入未变；四个条件中相同完整请求共享本轮新回答。单轮结果不支持稳定收益结论，p 值未作多重比较校正。

Top-5 目标记忆命中数依次为 507、501、507、480 / 1041；至少命中一条目标的题数为 306、303、306、303。仅 Card 的 `PLAY_S2:R5` 对话＋caption 超出 BGE 上限，按原始规则从 636 tokens 编码前 512 tokens；记忆与回答上下文保留全文，caption 生成没有截断。

完成 437 次新 caption 调用和 1,164 次新回答调用，覆盖四组各 371 题、共 1,484 条逻辑记录，320 条同轮记录共享回答。另有一次合成接口探测，不进入 benchmark。没有 API 失败／重试、拒答、格式错误、空答或输出截断。三项预检通过，覆盖请求字段、参数拒绝、模拟完整评分、非零推理用量、异常答案保留分母和续跑；独立来源／API 字节／检索／评分核验全部通过。结果文件保留五项方法标注及与 GPT-5.4、GPT-5.5 的逐题对照。

运行入口（默认目录为本轮结果；独立复测时每条命令传入同一新的 `--output`）：

```bash
python3 scripts/analysis/probe_memeye_gpt5_base.py
python3 scripts/evaluation/memeye_gpt5_base_captions.py --stage all
python3 scripts/analysis/retrieve_memeye_gpt5_base.py --device cuda:0
python3 scripts/analysis/evaluate_memeye_gpt5_base.py --stage all
python3 scripts/analysis/audit_memeye_gpt5_base.py
```

见 [完整报告](results/memeye_gpt5_base_seed42_full_20261024/report.html)、[逐题 CSV](results/memeye_gpt5_base_seed42_full_20261024/paired_results.csv)、[完整统计](results/memeye_gpt5_base_seed42_full_20261024/summary.json)、[caption 来源](results/memeye_gpt5_base_seed42_full_20261024/caption_model_provenance.json)、[独立核验](results/memeye_gpt5_base_seed42_full_20261024/independent_audit.json)。新代码与结果独立保存，既有 GPT-5.4／5.5 签名代码未改写。

##### GPT-5：固定 α=1，λ=0.001 消融

2026-10-24 在上一轮 GPT-5 的同一批 437 条 caption、438 条记忆及 72 个 master 上完成 λ=0.001、α=1 消融。caption 与记忆未重建；GPU 0 重新编码检索文本，742 组未赋权完整排名、44,546 个原始叶子分数及 3,597 个查询／分组与 λ=0.01 完全一致。λ=0.001 白化分数经独立 ridge 逆矩阵公式核验，最大误差 1.12e-15 以内。

两种方案各 371 题全部重新回答；请求 `gpt-5`，实际统一返回 **`gpt-5-2025-08-07`**。回答参数与 λ=0.01 相同：seed=42、reasoning_effort=minimal、不发送 temperature、max_completion_tokens=2048、原图 detail=high。候选记忆、查询、检索、编码和回答五项标注写入 protocol、summary 与 HTML 报告。

| 方案 | 历史未加权 | λ=0.01 历史参考 | λ=0.001 本轮 | 本轮准确率 | 相对 0.01 |
|---|---:|---:|---:|---:|---:|
| `baseline_question_only + z-score` | 219/371 | 221/371 | **214/371** | **57.6819%** | -7 |
| `master_answer_query + z-score` | 223/371 | 221/371 | **217/371** | **58.4906%** | -4 |

全量 24 题错→对、31 题对→错，净 −7（−1.8868 pp）；master 13 题错→对、17 题对→错，净 −4（−1.0782 pp）。配对 McNemar 双侧精确 p 分别为 0.4188、0.5847，未作多重比较校正。

相对 λ=0.01，全量 288 题的有序 Top-5 改变，83 题完整请求相同；这 83 题中有 5 题答案字母改变、净正确数 −1，其余输入变化的题目净 −6。master 171 题的有序 Top-5 改变，200 题完整请求相同；后者有 11 题答案改变、净 −2，其余输入变化的题目净 −2。即使 seed 和请求一致也观察到跨轮 API 波动，不能把全部分差归因于 λ。

Top-5 目标命中为全量 491/1041、master 478/1041，至少命中一条目标的题数为 301、303；λ=0.01 对应为 501/480 和 303/303。BGE 编码截断策略及受影响的 Card 长文本与上一轮完全相同，回答仍保留完整文本。

735 次新 API 调用覆盖 742 条记录，7 条同轮相同请求共享回答，无历史回答复用。无失败、重试、拒答、格式错误、空答、输出截断；实际报告 reasoning tokens=0。完整模拟评分／异常分母／独立核验／断点续跑测试通过，实际输入与回答独立核验通过，原实验文件哈希不变。

运行入口（独立复测为两条命令指定同一新的 `--output`；`--memory` 默认指向上一轮 GPT-5 记忆）：

```bash
python3 scripts/analysis/retrieve_memeye_gpt5_base_lambda0p001.py --device cuda:0
python3 scripts/analysis/evaluate_memeye_gpt5_base_lambda0p001.py --stage all
python3 scripts/analysis/audit_memeye_gpt5_base_lambda0p001.py
```

见 [完整报告](results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/report.html)、[逐题 CSV](results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/paired_results.csv)、[完整统计](results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/summary.json)、[独立核验](results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/independent_audit.json)。

##### GPT-5：固定 α=1，λ=0.1 消融与三个 λ 对比

2026-10-24，在同一批 GPT-5 caption、438 条记忆及 72 个 master 上完成 λ=0.1、α=1 的两种检索各 371 题。caption 与记忆未重建；GPU 0 重算检索，742 组原始完整排名、44,546 个原始分数和 3,597 个查询／分组与 λ=0.01 完全一致，并与 λ=0.001 逐条核对。白化分数经过独立 ridge 逆矩阵公式核验，最大误差 1.12e-15 以内。

回答请求 `gpt-5`，实际全部为 **`gpt-5-2025-08-07`**；三个 λ 均沿用 seed=42、minimal 推理、不传 temperature、2048 token 总预算、原图 detail=high、同一系统与用户提示词格式。每个 λ 分别重新回答，无跨实验回答缓存复用。

| 方案 | λ=0.001 | λ=0.01 | λ=0.1 本轮 | 相对 0.01 |
|---|---:|---:|---:|---:|
| `baseline_question_only + z-score` | 214/371（57.6819%） | 221/371（59.5687%） | **216/371（58.2210%）** | -5 |
| `master_answer_query + z-score` | 217/371（58.4906%） | 221/371（59.5687%） | **214/371（57.6819%）** | -7 |

相对 λ=0.01，全量 17 题错→对、22 题对→错，净 −5（−1.3477 pp，配对 McNemar 双侧精确 p=0.5224）；master 9 题错→对、16 题对→错，净 −7（−1.8868 pp，p=0.2295）。相对 λ=0.001，全量净 +2、master 净 −3。这些均为单轮结果，p 值未作多重比较校正。

全量有 168 题的有序 Top-5 相对 λ=0.01 改变，203 题完整请求相同；相同请求中有 10 题答案字母改变、净正确数 −4，输入变化的题目净 −1。master 有 160 题 Top-5 改变、211 题请求相同；后者 10 题答案改变、净 −2，输入变化的题目净 −5。三个 λ 报告也单列与 λ=0.001 的同请求波动，不能把全部分差归因于 λ。

Top-5 目标命中为全量 508/1041、master 477/1041，至少包含一条目标的题数为 306、302；λ=0.01 对应 501/480 与 303/303，λ=0.001 为 491/478 与 301/303。检索命中与问答准确率分别统计。BGE 上限、受影响的长文本及完整回答上下文策略均不变。

725 次新调用覆盖 742 条评测记录，17 条同轮相同请求共享回答。无失败、重试、拒答、格式错误、空答或输出截断，实际报告 reasoning tokens=0。完整模拟评分／异常分母／独立核验／续跑预检通过，本轮逐题核验与三个 λ 的独立重计及控制变量核验均通过。五项方法标注保存在结果文件中，旧代码与结果哈希保持不变。

运行入口（独立复测给检索和回答传入同一新的 `--output`；`--memory` 默认指向原 GPT-5 记忆）：

```bash
python3 scripts/analysis/retrieve_memeye_gpt5_base_lambda0p1.py --device cuda:0
python3 scripts/analysis/evaluate_memeye_gpt5_base_lambda0p1.py --stage all
python3 scripts/analysis/audit_memeye_gpt5_base_lambda0p1.py
python3 scripts/analysis/compare_memeye_gpt5_base_lambdas.py
```

见 [三个 λ 报告](results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison.html)、[三个 λ 逐题 CSV](results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison.csv)、[对比核验](results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison_audit.json)、[本轮详情](results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/report.html)、[本轮核验](results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/independent_audit.json)。

##### GPT-5：固定 λ=0.01 的 α 消融

2026-10-24，完成 α=0.01、0.1、10.0 的两方案全量消融，α=1 使用已核验的历史 221/371、221/371 参照。沿用 `memeye_gpt5_base_seed42_full_20261024` 的同一批 GPT-5 caption、438 条记忆、72 个 master、查询、完整候选陈述及回答参数（请求 `gpt-5`，实际返回全部为 `gpt-5-2025-08-07`，seed=42、minimal、不传 temperature、2048 token 总预算）。未重生成 caption 或重建 graph。

GPU 1 重算检索；742 组原始排名复现，原始／白化／z 分数和分组与 α=1 完全一致，独立矩阵求逆核验 3,597 个白化组，三个 α 共核验 133,638 个叶子分数。获用户明确授权后执行 1,474 次新 API 调用，覆盖 2,226 条逻辑回答记录；752 条记录共享本轮相同完整请求的回答，没有复用历史答案。完整模拟流程、独立输入及最终回答核验均通过，覆盖异常分母、同轮共享与断点续跑。没有请求失败、拒答、空答或截断；baseline α=10 的 Brand/Q3 有 1 条长解释格式错误，按严格单字母规则计错并保留在分母中，没有追加重试或事后提取字母。实际报告 reasoning tokens 均为 0。

| 设置 | baseline 答对 / 准确率 | master 答对 / 准确率 | baseline 目标命中 | master 目标命中 |
| --- | --- | --- | --- | --- |
| 未赋权（历史） | 219 / 371，59.0296% | 223 / 371，60.1078% | 507 / 1041 | 507 / 1041 |
| α=0.01（本轮） | 212 / 371，57.1429% | 220 / 371，59.2992% | 507 / 1041 | 505 / 1041 |
| α=0.1（本轮） | 211 / 371，56.8733% | 220 / 371，59.2992% | 507 / 1041 | 503 / 1041 |
| α=1（历史） | 221 / 371，59.5687% | 221 / 371，59.5687% | 501 / 1041 | 480 / 1041 |
| α=10（本轮） | 213 / 371，57.4124% | 204 / 371，54.9865% | 484 / 1041 | 389 / 1041 |

α=1 的历史成绩在表中最高，但跨轮回答波动必须单列：baseline α=0.01 相比未赋权只改变 3 题的有序 Top-5，另 368 题完整请求相同，其中 21 题回答改变、净减少 7 题；发生检索变化的 3 题净变化为 0。因此 219→212 的净下降全部位于相同请求子集，不能归因于 α 改动。master α=0.01、0.1 相比历史 α=1 都只少答对 1 题，其中相同请求子集净 −2、改变请求子集净 +1。完整逐题比较记录在 `alpha1_comparisons`、`unweighted_comparisons` 中。

本轮三个新 α 之间，相同完整请求共享回答并核验零答案差异；baseline 三组只相差 1–2 题，master 的 0.01 与 0.1 总分相同但逐题不同。master α=10 相比两组小 α 均少答对 16 题（32 错→对、48 对→错，配对双侧 p=0.0929），同时目标记忆命中明显下降。这是同一测试集上的单轮描述性参数比较，未作多重比较校正，不能宣称独立验证的最优 α。

见 [完整问答报告](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/report.html)、[α 对照 CSV](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/alpha_comparison.csv)、[逐题 CSV](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/paired_results.csv)、[独立核验](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/independent_audit.json)、[格式异常记录](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/response_diagnostics.json)、[冻结协议](results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/reader/protocol.json)。执行入口为 `scripts/analysis/retrieve_memeye_gpt5_base_alpha.py --device cuda:1`、`scripts/analysis/evaluate_memeye_gpt5_base_alpha.py --stage all` 和 `scripts/analysis/audit_memeye_gpt5_base_alpha.py`；独立复测时为检索与回答指定同一新的 `--output`。五项方法标注与实测准确率均保存在结果文件，旧实验代码、协议与结果保持原样。

##### Qwen 共享参数搜索与 GPT-5 迁移验证

2026-10-24，按用户确认，两种检索共用一组 λ、α。预先固定 25 个组合：λ∈{0.001, 0.003, 0.01, 0.03, 0.1}，α∈{0.1, 0.3, 0.6, 1, 2}。两组未赋权对照与 50 组加权条件全部用本地 Qwen 重新回答；沿用 Qwen caption、438 条记忆、88 个 master、同样的 371 道题、选项顺序和 Top-5。共 19,292 条逻辑记录，3,155 次新推理、16,137 条本轮相同完整请求共享；无历史答案复用、格式错误或输出截断。两组未赋权对照仍是 184/371、195/371，逐题答案也与原记录一致。

选择规则在回答前固定：先要求 baseline 加权 > baseline 未赋权、master 加权 > master 未赋权、master 加权 > baseline 加权；符合条件后按 master 正确数、baseline 正确数从高到低，再按 α、λ 从小到大选择。共有 5 组符合：

| λ | α | baseline 正确数 | master 正确数 |
|---:|---:|---:|---:|
| 0.003 | 0.3 | 186/371 | 196/371 |
| 0.01 | 0.6 | 186/371 | 198/371 |
| **0.01** | **1** | **188/371** | **201/371** |
| 0.03 | 1 | 185/371 | 199/371 |
| 0.1 | 0.6 | 186/371 | 199/371 |

因此固定 **λ=0.01、α=1**。λ=0.1、α=1 虽有 master 202/371，但 baseline 为 183/371，未通过第一项条件。选中组合相对未赋权分别净 +4／+6 题，配对 McNemar 双侧精确 p=0.5413／0.4408；这是同一 benchmark 上的探索性选参，不能当作独立测试集泛化或稳定显著提升。

见 [Qwen 完整搜索报告](results/memeye_qwen_zscore_joint_search_20261024/report.html)、[25 组 CSV](results/memeye_qwen_zscore_joint_search_20261024/search_results.csv)、[搜索热图](results/memeye_qwen_zscore_joint_search_20261024/search_heatmap.png)、[冻结选择](results/memeye_qwen_zscore_joint_search_20261024/selected_parameters.json)、[独立核验](results/memeye_qwen_zscore_joint_search_20261024/independent_audit.json)。

GPT-5 验证沿用已核验的 GPT-5 caption、438 条记忆及 72 个 master；不复用 Qwen graph。将 Qwen 选定参数在任何新 GPT-5 调用前冻结。四组条件（全量未赋权／加权、master 未赋权／加权）本轮重新回答，完整请求相同则共享一次新回答。未赋权的 742 组完整排名与原 GPT-5 实验完全一致；λ=0.01、α=1 下需要 1,164 次新回答调用，覆盖 1,484 条记录。验证要求三个严格比较全部通过，不依据 GPT-5 分数再换参。

GPT-5 四组验证已完成，**未通过预定条件**：

| 方案 | 本轮正确数 | 本轮准确率 | 加权净变化 |
|---|---:|---:|---:|
| baseline_question_only | 211/371 | 56.8733% | — |
| baseline_question_only + zscore | 222/371 | 59.8383% | +11 |
| master_answer_query | 219/371 | 59.0296% | — |
| master_answer_query + zscore | 219/371 | 59.0296% | 0 |

baseline 加权优于未赋权（23 错→对、12 对→错，p=0.0895）；master 为 17 错→对、17 对→错，准确率持平，且加权 master 比加权 baseline 少 3 题（37 错→对、40 对→错，p=0.8199）。三项严格比较只有第一项通过；不能声称 Qwen 的收益成功迁移到 GPT-5。完成 1,164 次新调用、320 条本轮共享，实际返回均为 `gpt-5-2025-08-07`，无 API 失败、拒答、格式错误、空答或输出截断，独立核验通过。

四组完整请求均与此前同参数 GPT-5 实验完全相同，seed、返回模型和 caption/graph 也一致；重新回答后，四组分别有 19／24／19／19 题答案改变。历史正确数 219／221／223／221 对应本轮 211／222／219／219。跨轮波动单独记录，当前赋权结论使用本轮四组对照；本轮相同请求共享一个回答，不存在相同请求在处理组与对照组之间答案不一致的情况。

见 [GPT-5 验证报告](results/memeye_gpt5_qwen_selected_validation_20261024/report.html)、[完整结果](results/memeye_gpt5_qwen_selected_validation_20261024/summary.json)、[逐题 CSV](results/memeye_gpt5_qwen_selected_validation_20261024/paired_results.csv)、[独立核验](results/memeye_gpt5_qwen_selected_validation_20261024/independent_audit.json)、[跨轮回答变化](results/memeye_gpt5_qwen_selected_validation_20261024/repeatability.json)。

```bash
python3 scripts/analysis/retrieve_memeye_qwen_zscore_search.py --device cuda:1
python3 scripts/analysis/evaluate_memeye_qwen_zscore_search.py --stage all --device cuda:1
python3 scripts/analysis/retrieve_memeye_gpt5_qwen_selected.py --device cuda:1
python3 scripts/analysis/evaluate_memeye_gpt5_qwen_selected.py --stage all
```

独立复测需给每阶段指定新的 `--output`；GPT-5 检索的 `--selection` 指向该轮 Qwen 的 `selected_parameters.json`。Qwen 同一签名运行中断后用 `--stage run --resume`，GPT-5 同一签名运行会保留并续用本轮已完成请求。结果保留五项方法标注、逐题回答、完整请求哈希和独立核验；历史签名代码及结果保持原样。

##### Qwen 五组候选的 GPT-5 联合复验

在首个候选 λ=0.01、α=1 的 GPT-5 验证失败后，用户明确要求把 Qwen 中满足条件的全部 5 组再放到 GPT-5 验证。此轮单独保存，保留之前的失败记录和原有选择协议。参数按 Qwen 原排名冻结为 `(0.01,1)`、`(0.1,0.6)`、`(0.03,1)`、`(0.01,0.6)`、`(0.003,0.3)`；每组内两种检索共用参数。

沿用同一批 GPT-5 caption、438 条记忆和 72 个 master，seed=42、minimal、不传 temperature、2048 总 token 预算。5 组各两种加权方案加两个公共未赋权对照，共 12×371=4,452 条记录；完整请求去重后需要 1,557 次新 API 调用，2,895 条本轮共享。所有回答在本轮重新生成，历史答案不用于填充。742 组未赋权完整排序及原始分数与 GPT-5 原实验一致；17,985 组白化计算经独立 ridge 公式核验，输入、原图与标准答案隔离检查通过。一次候选清单元数据错误在 API 前被预检拦截，零调用，失败准备记录单独归档；修复后重新生成并通过预检。

每组分别判定：baseline 加权 > baseline 对照、master 加权 > master 对照、master 加权 > baseline 加权；三项同时成立才算该组通过。首轮失败后扩展的五组比较属于探索性复验，不能将从中挑选的结果标为独立留出验证。

五组联合复验已完成并通过独立核验。**本轮未赋权对照为 baseline 217/371、master 221/371；没有一组同时通过三项条件。** 所有结果均来自本轮新回答。

| Qwen 排名 | λ | α | baseline + zscore | master + zscore | baseline 增益 | master 增益 | master − baseline | 全部通过 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.01 | 1 | 222/371 | 221/371 | +5 | +0 | -1 | 否 |
| 2 | 0.1 | 0.6 | 215/371 | 212/371 | -2 | -9 | -3 | 否 |
| 3 | 0.03 | 1 | 217/371 | 222/371 | +0 | +1 | +5 | 否 |
| 4 | 0.01 | 0.6 | 213/371 | 222/371 | -4 | +1 | +9 | 否 |
| 5 | 0.003 | 0.3 | 215/371 | 220/371 | -2 | -1 | +5 | 否 |

λ=0.03、α=1 和 λ=0.01、α=0.6 均使 master 从 221 提高到 222/371，且加权 master 分别比加权 baseline 多 5／9 题；但前者 baseline 持平，后者 baseline 下降 4 题。λ=0.003、α=0.3 也满足加权 master > 加权 baseline，但两种方案都低于各自对照。因此不能把三组的跨方法排序优势写成全部条件通过；master 的 +1 题也只是单轮小差异。

完成 1,557 次新调用、2,895 条本轮共享记录；实际返回均为 `gpt-5-2025-08-07`，无 API 失败、拒答、格式错误、空答或截断。λ=0.01、α=1 及两个原方法对照与上一次单候选实验的完整请求相同，跨轮回答变化单独保存，旧轮失败结果没有改写。

见 [完整报告](results/memeye_gpt5_qwen_five_validation_20261024/report.html)、[五组比较 CSV](results/memeye_gpt5_qwen_five_validation_20261024/candidate_results.csv)、[逐题 CSV](results/memeye_gpt5_qwen_five_validation_20261024/paired_results.csv)、[完整统计和五项标注](results/memeye_gpt5_qwen_five_validation_20261024/summary.json)、[独立核验](results/memeye_gpt5_qwen_five_validation_20261024/independent_audit.json)、[跨轮相同请求变化](results/memeye_gpt5_qwen_five_validation_20261024/previous_first_repeatability.json)。

```bash
python3 scripts/analysis/retrieve_memeye_gpt5_qwen_five.py --device cuda:1
python3 scripts/analysis/evaluate_memeye_gpt5_qwen_five.py --stage all
```

独立重跑为两条命令指定同一个新的 `--output`；GPU 检索支持 `--memory` 指定 GPT-5 记忆路径、`--qwen-search` 指定原 Qwen 搜索目录。中断后 `--stage all` 继续本轮已核验的回答；`--stage analyze` 仅汇总和核验。不得把失败准备归档目录当作完整实验。结果文件保留五项方法标注、全部候选分数及通过状态、15 个配对比较、跨轮相同请求变化、使用量和来源哈希。

##### Qwen caption＋GPT-5.4 回答：两种检索叠加 z-score（λ=0.01、α=1）

2026-10-24 已完成 `baseline_question_only + z-score` 和 `master_answer_query + z-score` 的全部 MemEye 371 题。冻结已核验的两组 Top-5，只将最终回答模型替换为 `gpt-5.4-2026-03-05`；对应 Qwen 参照分别为 188/371 和 201/371。全量方案以同 domain 全量记忆为一簇，多路径方案沿用正确的 195/371 完整候选陈述＋对话／caption 检索，再叠加在线赋权。

| 方案（λ=0.01、α=1） | GPT-5.4 正确数 | 准确率 | Qwen caption＋GPT-5.4 回答的未赋权历史参照 |
| --- | ---: | ---: | ---: |
| `baseline_question_only + z-score` | 236/371 | 63.6119% | 227/371 |
| `master_answer_query + z-score` | 230/371 | 61.9946% | 235/371 |

本轮使用 **Qwen caption**，没有在上述 GPT-5.4 caption 的完整流程实验上叠加 z-score。因此本轮加权 baseline 的 236/371 与完整流程未赋权 baseline 的 236/371 只是正确题数相同，不能据此判断赋权前后不变。[实验口径澄清](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/comparison_scope_clarification.json) 单独保存三组来源，冻结协议与原始结果保持原样。用户要求的 **GPT-5.4 caption＋GPT-5.4 回答**版本已另行完成，见下一节。

本轮加权多路径相对加权全量：29 题错→对、35 题对→错，净减少 6 题（−1.6173 个百分点；McNemar 双侧精确 p=0.5323，未作多重比较校正）。两者相对历史未赋权结果分别多答对 9 题、少答对 5 题；但全量的 168 道相同输入中有 15 道答案改变，多路径的 106 道相同输入中有 11 道答案改变，因此历史对比不能视作隔离 API 波动的 z-score 因果效应。

沿用此前 GPT-5.4 回答配置：官方 API、`reasoning_effort=none`、temperature=0、max_completion_tokens=16、不发送 seed，带图题发送原始题图字节。共 742 条逻辑回答输入、728 个唯一请求，标签单独保存且不进入 API 请求。此前未赋权的 GPT-5.4 227/371、235/371 仅作历史参照；API 不保证确定性，跨轮差异不能完全归因于 z-score。

执行入口为 [evaluate_memeye_gpt54_zscore.py](scripts/analysis/evaluate_memeye_gpt54_zscore.py)，使用独立输出目录及现有 GPT-5.4 transport；[audit_memeye_gpt54_zscore.py](scripts/analysis/audit_memeye_gpt54_zscore.py) 独立重构全部请求、核验原图／标签隔离及最终回答来源。18 项接口测试与完整独立核验均通过。获得本次数据外发授权后，完成 728 次新 API 调用及 14 条本轮相同输入共享，无历史答案复用、失败重试、拒答、格式错误、空答或截断；全部返回指定模型快照并明确报告 reasoning tokens=0。五项方法标注与实测准确率已写入结果文件。

详见 [完整报告](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/report.html)、[逐题 CSV](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/paired_results.csv)、[独立核验](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/independent_audit.json) 和 [运行状态](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/execution_status.json)。输入核验见 [independent_input_audit.json](results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/inputs/independent_input_audit.json)。

```bash
python3 scripts/analysis/evaluate_memeye_gpt54_zscore.py --stage prepare
python3 scripts/analysis/evaluate_memeye_gpt54_zscore.py --stage all
```

`all` 先检查一题纯文本和一题带图，再继续全量并独立核验；同一签名实验可断点续跑，仅在本轮完全相同输入之间共享回答，不复用历史答案。`prepare` 不调用 API；重建输入或改变设置需新的输出目录。

##### GPT-5.4 caption＋GPT-5.4 回答：以 236/371 为基准的 z-score 实验

2026-10-24 已完成用户要求的全部 371 题、λ=0.01、α=1 双方案评测。记忆路径为 `results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05/`：沿用此前 GPT-5.4 生成的历史／题图完整 caption、原始对话、既有 master 和完整候选陈述；GPU 0 现场重算 BGE／CLIP 文本编码并叠加在线赋权，仍由 `gpt-5.4-2026-03-05` 回答。没有混入 Qwen caption，也没有重新生成 caption 或改写旧记忆。

| 方案 | 此前同 caption 未赋权 | 本轮 λ=0.01、α=1 | 本轮准确率 | 正确数变化 |
| --- | ---: | ---: | ---: | ---: |
| `baseline_question_only + z-score` | 236/371 | **242/371** | **65.2291%** | +6 |
| `master_answer_query + z-score` | 241/371 | **237/371** | **63.8814%** | −4 |

全量方案以同 domain 全部叶子为一簇；多路径方案以既有 master 的全部叶子为一簇。两者均保留原始 BGE 匹配分数，最终分数为 `raw + α × std_pop(raw in group) × tanh(white_z)`，合并所有候选取 Top-5。现场向量用完即弃，不写回记忆。原有 BGE 512-token 编码上限保持一致，Card 一条超长叶子的 reader 证据仍保留全文。

核验包括：437 个 GPT caption 的 API 来源；742 个未加权完整排序及全部原始叶子分数与旧实验完全一致（最大分差 0）；4,052 个组的 SVD 白化与独立 ridge 求解一致；44,546 条组内叶子分数；742 个原始回答请求由当前 transport 重构后哈希一致。22 项测试及不联网的全流程模拟通过。实际完成 726 次新 API 调用、16 条本轮相同输入共享，无历史答案复用、失败重试、拒答、格式错误、空答或截断；返回模型与 reasoning tokens=0 均逐条核验。

全量方案相对历史对照 22 题错→对、16 题对→错；多路径 21 题错→对、25 题对→错。历史对照未在本轮重跑：182 道全量相同输入中有 18 道答案改变，111 道多路径相同输入中有 16 道答案改变，所以 +6／−4 不应全部解释为赋权效应。本轮加权多路径相对加权全量为 28 题错→对、33 题对→错，净少 5 题（McNemar 双侧精确 p=0.6089）。

运行入口 [evaluate_memeye_gpt54_caption_zscore.py](scripts/analysis/evaluate_memeye_gpt54_caption_zscore.py) 接受 `--memory` 指定原 GPT caption 实验路径；独立核验入口为 [audit_memeye_gpt54_caption_zscore.py](scripts/analysis/audit_memeye_gpt54_caption_zscore.py)。五项方法标注、原始请求与回答、逐题变化均保存在独立结果目录。

```bash
python3 scripts/analysis/evaluate_memeye_gpt54_caption_zscore.py --stage all --device cuda:0 --memory /storage/hewenwen/VisualMemory/results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05
```

详见 [完整报告](results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/report.html)、[逐题 CSV](results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/paired_results.csv)、[独立核验](results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/independent_audit.json)、[检索复现核验](results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/retrieval_protocol.json) 与 [回答参数／请求一致性](results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/historical_reader_transport_parity.json)。同签名可续跑；修改参数或输入必须换新的输出目录。

##### GPT-5.4 caption＋GPT-5.4 回答：固定 α=1 的 λ 消融

2026-10-24 已完成 **λ=0.1、α=1** 的两套全量评测，继续使用此前 236/371、241/371 完整流程的同一 GPT caption 和记忆；全部答案重新由 `gpt-5.4-2026-03-05` 生成。

| 方案 | λ=0.01（此前） | λ=0.1（本次） | λ=0.1 准确率 | 相对 λ=0.01 |
| --- | ---: | ---: | ---: | ---: |
| `baseline_question_only + z-score` | 242/371 | **236/371** | **63.6119%** | −6 题 |
| `master_answer_query + z-score` | 237/371 | **237/371** | **63.8814%** | 0 题 |

两轮的 caption、原始对话、记忆簇、查询、CLIP 选中陈述与回答参数均相同；4,052 个分组和 44,546 条原始匹配分数逐条完全一致，只改变白化参数 λ。全量有 166 题、多路径有 130 题的 Top-5 内容或顺序变化。GPU 0 完成检索，全部 742 个未赋权完整排序继续精确复现原实验。

λ=0.1 相对 λ=0.01：全量 10 题错→对、16 题对→错（McNemar 双侧精确 p=0.3269）；多路径 9 题错→对、9 题对→错（p=1.0），正确题数持平并不代表逐题结果相同。相同完整输入中，全量 205 题有 8 题跨轮答案改变，多路径 241 题有 6 题改变；因此不能把跨轮分数差异全部归因于 λ，也不能据此认定最优参数。

22 项测试、完整模拟流程、实际回答与独立 λ 对比核验均通过。727 次新 API 调用、15 条本轮相同输入共享；无历史答案复用、失败重试、格式错误、空答或截断。旧实验源码、协议与结果保持原样，两个 λ 的代码分别冻结。

```bash
python3 scripts/analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p1.py --stage all --device cuda:0 --memory /storage/hewenwen/VisualMemory/results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05
```

见 [λ 消融对比报告](results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison.html)、[λ 逐题对比 CSV](results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison.csv)、[本轮报告与五项标注](results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/report.html)、[实际回答核验](results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/independent_audit.json) 及 [独立对比核验](results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison_audit.json)。[评测入口](scripts/analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p1.py)、[本轮核验入口](scripts/analysis/audit_memeye_gpt54_caption_zscore_lambda0p1.py)、[变量控制与对比入口](scripts/analysis/compare_memeye_gpt54_caption_lambdas.py) 分别保留。

##### GPT-5.4 caption＋GPT-5.4 回答：补充 λ=0.001、α=1

2026-10-24 已完成 **λ=0.001、α=1** 的全部 371 题。继续沿用同一 GPT caption、原始对话、记忆簇、候选陈述、BGE／CLIP 编码和 GPT-5.4 回答设置；仅改变 λ，所有答案重新生成。三个 λ 的结果如下：

| 方案 | λ=0.001（本次） | λ=0.01 | λ=0.1 |
| --- | ---: | ---: | ---: |
| `baseline_question_only + z-score` | **243/371（65.4987%）** | 242/371（65.2291%） | 236/371（63.6119%） |
| `master_answer_query + z-score` | **240/371（64.6900%）** | 237/371（63.8814%） | 237/371（63.8814%） |

相对 λ=0.01，全量 18 题错→对、17 题对→错，净 +1 题（McNemar 双侧精确 p=1.0）；多路径 15 题错→对、12 题对→错，净 +3 题（p=0.7011）。相对 λ=0.1，全量为 23／16、净 +7 题（p=0.3368），多路径为 14／11、净 +3 题（p=0.6900）。本轮数值差异尚不支持稳定优势或最优参数结论；这些配对检验未作多重比较校正。

与两组历史实验分别核对，4,052 个分组与查询、44,546 条原始匹配分数均完全相同。相对 λ=0.01，本轮全量 294 题、多路径 175 题的 Top-5 内容或顺序改变；相同完整输入中仍分别有 4/77、6/196 题答案跨轮改变。与 λ=0.1 的相同输入比较为 1/50、5/150，因此不把全部分差归因于 λ。GPU 0 完成检索，未加权完整排序仍与 236/371、241/371 原实验逐条一致。

22 项测试、完整模拟流程、独立回答核验及三组 λ 的对比核验通过。730 次新 API 调用、12 条本轮相同输入共享，无历史答案复用、失败重试、格式错误、空答或截断；全部返回 `gpt-5.4-2026-03-05`。此前两组源码、协议与结果保持原样。

```bash
python3 scripts/analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p001.py --stage all --device cuda:0 --memory /storage/hewenwen/VisualMemory/results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05
```

见 [三个 λ 的对比报告](results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison.html)、[逐题对比 CSV](results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison.csv)、[本轮报告与五项标注](results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/report.html)、[回答核验](results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/independent_audit.json)、[对比核验](results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison_audit.json)。对应 [评测入口](scripts/analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p001.py)、[回答核验入口](scripts/analysis/audit_memeye_gpt54_caption_zscore_lambda0p001.py)、[对比入口](scripts/analysis/compare_memeye_gpt54_caption_lambda0p001.py) 独立保存。

#### Mem-Gallery 全量配对实验

```bash
/usr/bin/python3 scripts/visual_memory_pipeline.py offline --memory caption_visual_master --dataset mem_gallery --device cuda:0
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset mem_gallery --stage prepare --device cuda:0
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset mem_gallery --stage run --device cuda:0
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master --dataset mem_gallery --stage analyze --device cuda:0
```

也可用在线 `--stage all` 依次运行候选生成、检索、答题和分析。此适配器固定处理 20 个领域的全部 3,962 轮记忆和 1,711 个开放式问题，不接受 Home 的图片、问题或领域筛选参数。默认输出为 `results/mem_gallery_master_answer_20261021/`；新实验可在离线和在线都指定同一个 `--output`。已有完整结果校验哈希后复用；输入、模型、代码或协议改变时要求使用新目录，不静默混入旧实验。

历史记忆沿用已生成的完整 Qwen caption，并加入原始 user/assistant 对话。检索问题附图使用数据集已有 caption；回答阶段使用问题原图。两个条件都在同一领域的全部记忆轮次中全量排序并取 Top5，不按问题的 session_id 截断历史。原 query 的 BGE 检索是基线；新方法对 1,003 条图片记忆使用各自 master 选择的候选扩展 query，对 2,959 条纯文本记忆保留基线分数，最后一起排序。

Mem-Gallery 没有选择题选项，因此新增一次候选生成：本地 Qwen3-VL-8B 只看问题及其已有附图 caption，生成四个短假设。它不读取历史、答案或 clue。每个 master 用 CLIP 选择最相近的假设，再追加到 query；所有候选均为检索用途，不加入 reader 证据。此候选生成步骤是对开放式问答的适配，不等同于 MemEye 已给定选项的实验。

离线严格按图片到达顺序更新，与 Home 共用 caption Ward/silhouette 分簇和视觉更新核心。二维基底复用外部 MemEye 校准，不在 Mem-Gallery 全集上重拟合；校准图与当前历史、问题图片不重叠。旧簇拆分或迁移时，两个坐标之外的残差按原簇成员比例近似分配，因此更新后的 master 不应称为原始图片 embedding 的精确均值。持久化视觉记忆仅包含叶子的两个坐标、共用基底和 master 聚合向量；不复制原图，不保存单图完整 embedding，不重编码旧图。

两种检索结果使用同一 Qwen reader、提示词、生成参数和问题原图。上下文严格是各自 Top5 的原始对话、完整 caption、日期及图片 ID。相同有序上下文只在本次实验内共享一次新推理，并记录来源；不导入历史答案。`reader/summary.json` 使用官方确定性 normalized EM、stemmed token F1 和 BLEU，属于开放式答案指标，不是选择题 ACC，也不是 LLM judge。184 道无 clue 题保留在全部答题指标中，仅从有监督检索召回的分母中排除。

代码入口：`scripts/data/mem_gallery_inputs.py` 验证输入；`scripts/memory/mem_gallery_master_answer.py` 构建和检索；`scripts/evaluation/mem_gallery_master_reader.py` 生成候选、配对回答及评分。逐题结果见 `retrieval_results.jsonl` 和 `reader/paired_results.csv`；输入、模型、候选、检索和 reader 协议分别保存以供核验。

完整评分结束后，`python3 scripts/analysis/report_mem_gallery_master_answer.py --output <同一输出目录>` 会重新核验输入与推理来源，生成 `report/comparison.md`、`comparison.json` 和分领域 CSV。独立的 `diagnostics/offline_audit.json`、`retrieval_audit.json` 记录视觉存储、排名和文本长度检查。

2026-10-21 这次运行有 3 道题经过三次生成后仍把四句话写成四个独立的单元素 JSON 数组。`scripts/data/normalize_mem_gallery_candidate_format.py` 只对这种明确的容器格式做严格转换，不改内容、不重新生成；`candidate_format_repair.json` 保留原始输出、失败记录和转换哈希。原生成代码及协议保持冻结，随后原 prepare 阶段校验全部 1,711 条候选通过。其他不完整或含糊的格式仍会报错，不能用此工具静默替代。

### Object Relation：先生成对象，再生成关系

每张图片分两次调用同一个 Qwen。两次都输入原图与已有完整 caption：

1. 第一次只输出 `{"objects": ["person", "cup"]}`。
2. 第二次额外输入该对象列表，输出 `{"relations": [{"subject": "person", "predicate": "holds", "object": "cup"}]}`。对象列表作为起点，允许出现新的主客体；相同实体尽量复用已有名称。
3. 程序将关系中尚未出现的主客体补入对象列表，统一空白并去重，保留方向、不同谓词和没有关系的初始对象。此阶段不调用模型，也不做同义词合并。不生成对象 ID、矩阵或 evidence。

前两阶段各允许一次格式或空结果重试；超长重试要求最多 30 个不同条目，解析后检查该数量限制，并使用 `repetition_penalty=1.1` 抑制重复，该参数保存在对应 attempt 的 `generation_options` 中。初始对象为空时仍生成关系，第三阶段可从关系补出对象。关系复查后仍为空则保留初始对象与空关系列表。

关系字段为空或 `null` 时，重试提示只提取两个实体之间的关系，排除缺少另一端的单实体属性；模型原始输出和重试提示都保留。

对象阶段连续两次超长，或超长重试仍超过 30 个不同对象后，续跑改为提取最多 5 个主要可见对象（最多 512 tokens，`repetition_penalty=1.0`），记录 `retry_strategy=main_objects`；关系阶段仍可引入其他对象，由第三阶段补齐。此恢复不会把截断输出当作成功。

固定输出目录为 `results/object_relation_memory/`，文件仍为 `generations/<图片名>.json`，跨任务重名时才加来源目录前缀。`raw_objects_output` / `object_attempts` 保存第一阶段原始输出和尝试；`raw_output` 保存第二阶段模型的原始文本，`attempts` 保存该阶段的尝试；`initial_objects` 是初始对象，`added_objects` 是第三阶段新增的对象，`objects` 是最终对象列表。`generated_relations` 和 `relations` 都保存句子字符串列表：前者保留生成顺序与重复项，后者整理并去重，例如 `"relations": ["person holds cup."]`。第三阶段在内部解析原始输出时补齐对象。已有三元组格式的 `relations` 在核对原始输出后原地转换，保留原始生成记录。旧实验统一保留在同目录的 `previous_generations.json`。不计算额外哈希，不新增版本目录。

```bash
# 仅准备输入
python3 scripts/visual_memory_pipeline.py offline --memory object_relation --extraction-stage prepare
# 同一目录内重新生成第一张图片
python3 scripts/visual_memory_pipeline.py offline --memory object_relation --limit 1 --from-start
# 正常续跑
python3 scripts/visual_memory_pipeline.py offline --memory object_relation
# 全量时记录单张输出失败并继续，结束后检查 summary 中的 failed_images 再续跑失败项
python3 scripts/visual_memory_pipeline.py offline --memory object_relation --continue-on-error
```

原 VS Code 调试项 **VisualMemory: Debug Object Relation Extraction** 可直接运行三阶段流程，每次重新处理 1 张图片。断点放在 `generate_stage()` 的 `writer._ask_plain(...)`：第一次 `stage=object_extraction`，第二次 `stage=relation_extraction`；查看第二次的 `request_text` 可确认对象列表已加入。`process_output(raw, objects)` 整理关系，`complete_objects()` 是第三阶段补充对象的断点入口。

2026-10-19 已完成全部 8 个任务：438 条 memory 对应 433 张去重图片，初始对象 3,823 条，第三阶段补充 1,769 条，共 5,592 条对象；整理后关系 4,839 条。`generated_relations` 保留 4,869 个原始关系句子，其中 30 条精确重复已在 `relations` 中去除。失败、空对象和空关系均为 0。9 张图片使用主要对象恢复，1 张使用空端点关系恢复，原始尝试全部保留。完整结果见 [extractions.jsonl](results/object_relation_memory/extractions.jsonl)；[全量校验](results/object_relation_memory/validation.json)已通过覆盖范围、原始输出、补齐端点和缓存/导出一致性检查，未逐条人工核验语义。


## Online：完整历史与文本记忆基线

### FC(T)：完整文本历史

```bash
/usr/bin/python3 scripts/visual_memory_pipeline.py online --stage prepare --memory fc_t
/usr/bin/python3 scripts/visual_memory_pipeline.py online --stage run --memory fc_t
/usr/bin/python3 scripts/visual_memory_pipeline.py online --stage analyze --memory fc_t
```

单独选择 `fc_t` 时不传 `--top-k`。历史来自 natural protocol 中已固定哈希的 `*_Open.json`，包含所有 session 的 user/assistant 对话轮次（含无图轮次）、session/date/round 信息；历史图片逐一替换为 natural 缓存中经生成签名校验的 Qwen caption。不会读取数据集自带的 `image_caption`、QA 答案或 clue 来构造历史。任务范围与其他模式一致；不会把 QA 的 session_id 当作查询时间截断历史。

这与当前 natural/graph 的 caption-only 输入有明确区别：FC(T) 还包含原始对话和时间信息，因此比较不能只解释成 Top-K 数量的影响。题目自身的附图仍沿用当前 reader 输入，不属于历史图片。

默认 `--context-budget 128000`，并取它与 reader 模型上下文上限的较小值。prepare 使用实际 Qwen processor 计算含系统提示、题目、选项和题目附图的 token 数，预留现有 `max_new_tokens`；超限按 FIFO 删除完整的最早轮次。每题记录保留/删除的轮次 ID、输入 token 数及截断后的 gold 覆盖；run 再校验上下文上限。FC(T) 使用 batch size 1 降低长上下文显存占用，其他答题参数不变。

FC(T) 输出在 `results/visual_memory_pipeline/fc_t/`，独立于 Top-K。`--mode all --top-k K` 会同时包含 FC(T) 和四个检索模式，比较文件仍写在 `topK/comparison.json`。FC(T) 的 `retrieval_top_k` 为 null；为兼容现有分析保留的 `any_gold_topk`、`all_gold_topk` 表示 FIFO 后的证据覆盖，不是检索召回。

### Reflexion：官方记忆核心，本地 Qwen 适配

```bash
/usr/bin/python3 scripts/visual_memory_pipeline.py offline --memory reflexion
/usr/bin/python3 scripts/visual_memory_pipeline.py online --memory reflexion --stage all --reflexion-batch-size 2
```

Reflexion 必须显式选择，不包含在 `--mode all` 中，因为它沿用官方的**答题后标准答案反馈**。不传 `--top-k`。prepare 复用已校验的 natural Qwen caption；run 加载本地 Qwen 答题和生成反思；analyze 汇总逐题准确率。输出位于 `results/visual_memory_pipeline/reflexion/`。

来源固定为 [MinghoKwok/MemEye 的提交 0358e70](https://github.com/MinghoKwok/MemEye/tree/0358e70d714980980dfd8c87903384db474f0b16)。`scripts/vendor_memeye_reflexion/` 保留原始 RFMemory 核心、adapter 参考文件、许可证和逐文件哈希；`scripts/memory/reflexion_memory.py` 是本地适配器。官方纯函数直接从已校验的 adapter 源码加载，避免引入其他方法的 router 依赖；store、recall、截断、optimize 和反思提示词保持原样，LLM 传输替换为本地 Qwen。

已确认的官方行为：

- 每个 task 先存入全历史对话与图片 caption。召回不做语义检索，query 不参与排序。
- 将全局反思放在全部记忆前面，随后保留最后 6,000 个以空格切分的片段。长历史可能完全截掉前面的反思；逐题记录 `prior_insight_fully_visible`。
- 每题先回答，再用该题的标准答案和对错标记更新全局反思；反思跨同一 task 的后续题目保留，不跨 task。当前题标准答案不会进入该题答题消息，但可经先前反思影响后续题目。
- 题目附图也用现有 Qwen caption 代替，不向回答模型传原图。这是官方 text-only 策略，与当前 FC/Top-K 的题目附图策略不同。

本地实验使用现有 371 题固定选项排列，每题只回答一次，反馈使用该排列对应的 `gold_choice`。不运行官方四次选项轮换：当前上游 runner 的轮换分支仍将完整原始 QA 交给方法，而 Reflexion 读取外层 `qa.answer`；不能把该反馈直接当作每次轮换的正确标签。

回答沿用现有 Qwen/system prompt/seed42/16-token 上限；反思使用官方提示词、同一 Qwen 和确定性解码，新增明确的 `--reflection-max-new-tokens 512` 上限（上游反思 API 未指定生成长度）。`--reflexion-batch-size` 只合并不同 task 的调用，同 task 严格按原题顺序执行。权重按每卡最多 10 GiB 且不超过空闲显存的 60% 自动分配，记录实际放置；无需停止其他 GPU 进程。

每题保存原始消息、输入哈希、反思前后状态和生成 token 数。`pending_answers/` 先保存已完成的回答，`checkpoints/` 在对应反思成功后提交整个状态；续跑验证任务状态链，避免重新生成已保存回答。所有格式错误均计入总分母。此结果是**带反馈、依赖题目顺序的本地适配实验**，不能作为与无反馈 FC/Top-K 完全受控的排名或论文精确复现。

检索模式默认输出为 `results/visual_memory_pipeline/top{k}/{mode}/`，包含 `protocol.json`、`qa_manifest.jsonl`、`qa_responses.jsonl`、`summary.json`、`per_question.jsonl`。`--mode all` 的分析另写 `comparison.json`。已有答案按实验签名和输入哈希校验后续跑；配置不同会报错。

可用 `--task` 和 `--limit-questions` 做小样本调试，建议搭配独立的 `--output-root`。`--benchmark-manifest` 可指定同字段格式的选择题清单；默认使用当前 MemEye 371 题清单。查询默认复用已保存 BGE 向量，`--encode-queries` 可按题目文本重新编码。当前 offline 的底层数据准备脚本仍针对 MemEye，切换新数据集时需要提供对应 manifest 并适配写入器的数据读取。
# Gen.Ag / MemoryOS / A-Mem / SM(T) local comparison

The pipeline accepts `gen_agents`, `memoryos`, `a_mem`, and `simplemem` for both
`offline --memory` and `online --memory`. The offline step reuses the same Qwen
caption store as natural / FC(T). Online preparation validates caption
signatures, source hashes, the question manifest, and pinned upstream files.

```bash
python3 scripts/visual_memory_pipeline.py online --memory gen_agents --stage all
python3 scripts/visual_memory_pipeline.py online --memory memoryos --stage all
python3 scripts/visual_memory_pipeline.py online --memory a_mem --stage all
python3 scripts/visual_memory_pipeline.py online --memory simplemem --stage all
python3 scripts/evaluation/audit_text_memory_results.py
```

These are signed result directories: after a source/configuration change, use a
fresh `--output-root` for a new run. Existing results can be read with
`--stage analyze`; preparation deliberately refuses to overwrite a different
protocol. Each completed run retains its original source snapshot.

Omit `--top-k` except for A-Mem: these methods use their pinned method defaults. Gen.Ag retrieves 10 entries; A-Mem defaults to 5 primary hits plus linked neighbors; MemoryOS has a retrieval queue capacity of 10.
SimpleMem now defaults to the official **text** backend: semantic K=5 per query,
keyword K=5 and structured K=5, with planning enabled and reflection disabled
for the controlled comparison (upstream: semantic K=25 and reflection enabled). Their merged
context is not a fixed Top-K. A-Mem also expands linked neighbors, so its
context can exceed 10 notes. `--memory-workers 8` runs independent task states;
`--memory-batch-size 8` batches their local Qwen calls. Both defaults are 8;
batches split when padded input plus output would exceed 48,000 tokens (a
single larger request is retained unchanged). The completed Gen.Ag run used
4 workers / batch size 4; its original code snapshot is retained.
Questions within one task
remain in source order. `--task` and `--limit-questions` support smaller runs;
all selected tasks still ingest their complete histories.

Gen.Ag, MemoryOS, A-Mem and historical SimpleMem Omni import their memory cores
from `scripts/vendor_memeye_text/`. `UPSTREAM.json` records every source hash:

- MemEye: `0358e70d714980980dfd8c87903384db474f0b16`.
- Gen.Ag's unbundled MemEngine: `67e779ee97599304815a0a820ca7e2e7c8ac18ea`.
- SM(T)'s missing `omni_memory/core/config.py`: restored from SimpleMem
  `db80b6a7c591e0ea730a058e9f5fc4eb06572299`. This is the upstream configuration,
  not proof of the exact uncommitted configuration used for the paper.

The default SimpleMem text backend is separately installed in
`.vendor/simplemem_official_runtime/`, version 0.3.0 at the same pinned SimpleMem
commit above. `scripts/requirements-simplemem-official.txt` installs its missing
text dependencies while reusing the local torch/Qwen/MiniLM environment. Its
adapter is `scripts/memory/simplemem_official.py`; `simplemem_official_sources.json`
checks the official text source against the upstream Git blob hashes.

The adapter calls `create(mode="text")`, `add_dialogues()`, `finalize()`, and
`hybrid_retriever.retrieve()`. The official MemoryBuilder performs stages 1–2
(semantic structured compression and online synthesis) jointly over 40-turn
windows with 2-turn overlap. Stage 3 retains planning and all three retrieval paths;
reflection and its additional searches are disabled. Planning can still generate
multiple queries, so five semantic hits per query is not a cap on the merged
candidate count. Each dialogue round becomes separate user and
assistant turns; cached image captions are appended to the user turn. Construction
and retrieval are sequential within a task, with independent per-task databases.
Finalization processes the tail window and refreshes the Tantivy index after all
inserts. Failed construction parsing aborts after upstream retries; valid empty
outputs remain valid. Window provenance is saved separately and does not imply
entry-level evidence alignment.

`--simplemem-context-limit 5 --simplemem-preserve-empty` keeps the first five
distinct complete entries in upstream returned order, or preserves an originally
empty result. Nonempty results shorter than five fail. New results use
`top5/simplemem_official_text_context5/` (or `simplemem_official_text/` without a
cap). Historical Omni results remain under their old names; select
`--simplemem-backend omni_legacy` to read them through this CLI. Frozen old-context
ablations continue to use `scripts/evaluation/simplemem_context_ablation.py`.
Their published metrics are not results of the new text backend. See
`scripts/README.md` for the install and one-question run command; 8192 output
tokens are suggested for the larger official extraction windows.

Preserved mechanisms: Gen.Ag combines relevance, recency and importance and
reflects after accumulated importance reaches 8; MemoryOS updates short-, mid-
and long-term memory with its original capacities and thresholds; A-Mem creates
structured notes and evolves links and neighbor metadata; historical SM(T) Omni summarizes
captioned rounds, builds its stores and graph, and uses the original query and
expansion path. Runtime compatibility fixes keep Gen.Ag score tensors on the
encoder's CPU, supply the legacy `langchain.prompts` namespace and Python 3.14's
removed `ast.Str` alias, and parse numeric importance without Python `eval`.
These compatibility changes are outside the pinned core files.

This is a **controlled local Qwen comparison**, not the paper's original
end-to-end result. All four methods use Qwen3-VL-8B-Instruct for memory operations
and final answering, the existing Qwen image captions, and MiniLM-L6-v2 for their
native retrieval. They ingest dialogue **and** captions; natural use
caption-only image memories. Retrieval receives question text plus the cached
question-image caption, excluding MCQ options and labels. Final answering uses
the common local system prompt, fixed options, greedy decoding, seed 42 and the
16-token answer cap. The original per-method final answer prompts are replaced.
All four use text descriptions for query images; older natural / FC(T)
runs pass native query images. None of these four methods gets gold feedback;
the existing Reflexion run does and must be marked separately.

The local transport provides JSON schemas as prompt instructions rather than
server-enforced constrained decoding. Memory operations retain their upstream
per-call temperatures (MemoryOS and A-Mem commonly use 0.7); each sampled
request has its own seed-42 RNG stream and uses categorical sampling with no
top-k/top-p restriction. Final MCQ answering remains greedy. A partial
all-greedy MemoryOS diagnostic run was archived as `memoryos_greedy_partial`
when restoring the native temperature protocol. MemoryOS explicitly uses
temperature 0.3 / 100 tokens for continuous meta-summaries, and 0.7 / 2000 tokens
for profile generation; these caps must be counted separately. The partial run is not included as a
completed benchmark result. Invalid structured responses and upstream
fallback warnings are retained for audit, without silent retries. Explicit
upstream memory generation caps are honored; unspecified ones use
`--memory-max-new-tokens 2048`. No external model endpoint is used.

Each output folder contains `protocol.json`, sanitized `observations.json`,
`questions.jsonl`, content-addressed `calls/`, `call_events.jsonl`, per-task
progress and response files, `qa_responses.jsonl`, `summary.json`, and `audit.json`.
Completed tasks are hash-verified before reuse. An interrupted task rebuilds
its state in an isolated runtime directory and replays identical cached local
requests. Upstream wall-clock fields and internal IDs can cause some new
requests during rebuilding; this is recorded as cache misses, not concealed.
The comparison exports micro accuracy over all 371 questions, the unweighted
mean of the 12 X/Y cells, and paired wins/losses against FC(T). Invalid MCQ
outputs count as incorrect in the full denominator.

The A-Mem audit distinguishes schema-valid JSON from merely parseable JSON,
and counts resolvable and unresolved links using the upstream lookup rules.
In particular, a numeric string is treated as a memory key, not automatically
converted to an integer position. These outputs are not silently repaired.


## 独立 CLIP/Ward 输入与历史结果

独立的 CLIP/Ward、视觉近似、master 路由与答题实验继续保留。
`data/independent_visual_inputs/` 从这些实验已经保存的输入清单中提取 438 条完整 caption/对话和 371 道问题；保留原图片哈希、时间顺序、问题文本及选项。该目录不保存旧图分组、shared/unique 或历史检索排名。`protocol.json` 记录来源和导出哈希，`validation.json` 记录本次校验。

当前 `scripts/analysis/` 的 Ward 构建、跨域输入、路由和 reader 入口读取该目录。少数固定候选的视觉诊断通过 `analysis/frozen_comparison_inputs.py` 读取独立实验已经保存的比较缓存；仅复核历史候选，不执行已删除的检索算法。

历史结果、回答和 protocol 保留原内容及哈希。旧协议中指向已删除文件的路径属于历史来源记录，不能证明这些文件仍可复核。原有严格来源审计仍会报告缺失文件或代码变化；不会把旧哈希改成新哈希来掩盖缺口。当前 Ward 输入和数学重放的独立检查位于 `reports/home_caption_ward_visual_20261020/validation_after_cleanup.json`，与原 `validation_independent.json` 分开保存。

新增运行使用独立输出目录；不要覆盖已有完成结果。

## MemEye：完整替换生成 backbone

`backbone` 入口替换 `caption_visual_master` 与其全量检索基线中的全部 Qwen 工作：历史图片 caption、带图问题的 caption、最终回答。每个模型先生成 437 张去重图片的 caption，再按原始到达顺序重建 438 个叶子和全部 8 个领域的 master，重新计算两种方法对 371 道题的 Top5，最后回答。两个方法共享该模型生成的同一份 caption。

```bash
python3 scripts/visual_memory_pipeline.py backbone \
  --memory caption_visual_master --dataset memeye \
  --api-model gpt-5.4-2026-03-05 --device cuda:0 --stage all
```

另一个已配置模型为 `--api-model gpt-4o-mini-2024-07-18`。默认结果分别写到 `results/memeye_full_backbone_20261021/<模型快照>/`，不会复用旧 Qwen caption、分簇、Top5 或答案。`--stage prepare` 只准备输入；`caption` 调 API 生成 caption；`retrieve` 在 GPU 上重建和检索；`reader` 调 API 回答；`analyze` 汇总 Recall@5 与回答准确率。`--api-limit 2 --stage caption` 可先检查两次 caption 请求，随后不带 limit 继续同一运行。

CLIP、BGE、外部二维基底、Ward 分簇规则、检索算法、原始对话、问题选项顺序与 Top5 保持一致。MemEye 的四条候选语句原本是 assistant 按问题和选项编写的固定模板，并非 Qwen 输出；本次仍使用同一份模板。叶子只持有二维视觉坐标，master 保存完整视觉聚合；原图只在输入阶段读取，不复制进记忆库。历史与问题中同一图像内容共享该模型的一次新 caption。

`inputs/` 中的 API 请求与评估标签分开保存；caption 请求只包含图片与原始 caption 指令，回答请求只包含问题、选项、检索到的原始对话＋新完整 caption 和问题原图。请求哈希、实际返回模型、运行 ID、重试、token 用量及各阶段文件哈希用于核查来源。官方 API 的图像输入与固定模型接口依据 [OpenAI vision 文档](https://developers.openai.com/api/docs/guides/images-vision)和 [GPT-5.4 文档](https://developers.openai.com/api/docs/models/gpt-5.4)。

此前 `online --reader-backend api` 和 `results/memeye_reader_ablation_20261021/` **仅替换最终回答模型**，冻结了上游 Qwen caption 与检索结果。它们保留为 reader-only 对照，不能称为完整 backbone 替换结果。

检索编码沿用旧实验的 BGE 512-token 上限，超长输入的具体 ID 与长度会记录在协议中；完整文本继续用于记忆存储与回答。旧 Qwen 实验的 `Card_Playlog_Test/PLAY_S2:R5` 原本已有 577-token 输入，并在旧 `bge_encoding.truncated_inputs` 中记录了 512-token 截断。新流程通过独立的 `memeye_backbone_long_text.py` 恢复这个既有编码规则，两个检索条件共享同一叶子向量。它不改变 caption 生成、分簇规则或已构建 master，也不重新读取已经加入记忆的旧图。本次不采用分块聚合，以免额外改变检索方法。
