# Method 1：Caption 分簇与视觉 Master 引导的多路径检索

本文记录当前 `caption_visual_master` 的实际实现，核对日期为 **2026-10-26**。核心配置是 `caption_visual_master + multi_path + online_weighting=none`；z-score 是可选的在线赋权扩展。

**2026-11-05：** 新增 MemEye 的 Random-2D 基底消融入口，见 §6.5。默认仍为原有 PCA-2D；已有结果表均为原有方案，不代表随机基底的实验结果。

**核心思路：用 caption 决定记忆如何分组，用每组的视觉表征选择检索假设，再用对话与完整 caption 区分具体记忆。** 图片只在加入时编码；记忆库保留每个 master 的完整视觉聚合向量、每条叶子的两个视觉坐标及文本。在线阶段，每个 master 选择一条候选答案陈述，将它追加到问题 query，分别检索所属叶子，最后合并成全局 Top-5。

## 1. 框架

![Method 1 framework](framework.png)

[矢量 SVG](framework.svg) · [PDF](framework.pdf) · [可编辑绘图源代码](draw_framework.py)

主图只保留“离线建库 → 两层记忆 → 在线检索”三个部分。同色表示同一个簇及其检索路径：master 从候选陈述中选择一条来扩展问题，再用文本匹配所属叶子，最后合并全局 Top-5。图中两个簇与 B/C 选择仅为示意，实际所有 master 都参与；候选陈述只用于检索。更新公式、具体编码器和可选 z-score 见下文。

`master → leaf` 是当前保存的两层结构；Ward 聚类过程中生成的层次树不持久化为多层记忆树。

## 2. 数据与模型

一条到达记录包含图片 $I_i$、完整 caption $c_i$、对应轮次对话 $d_i$ 和时间戳 $\tau_i$。同一个 domain 独立构建和检索记忆。

| 组件 | 当前基础实验配置 | 用途 |
| --- | --- | --- |
| CLIP 图像编码器 | ViT-B/32，512 维 | 图片到达时提取视觉表征 |
| CLIP 文本编码器 | 与上述图像编码器配套，512 维 | 将候选答案陈述映射到视觉共享空间 |
| BGE 文本编码器 | `BAAI/bge-small-en-v1.5`，384 维 | caption 分簇，以及在线 query–叶子匹配 |
| 生成式 backbone | 原始配置为 `Qwen3-VL-8B-Instruct`；实验可替换 | 历史图片 caption、带图问题的 caption、最终回答 |

CLIP 和 BGE 属于两个独立空间，不直接比较它们的向量。实际模型路径、版本和文件哈希由实验的 `protocol.json` / `memory.json` 固定。替换生成式 backbone 的完整消融需重做上述三个生成环节，再重建记忆和检索；只更换 reader 属于回答模型消融。

| 记忆层 | 保存内容 | 保存位置 |
| --- | --- | --- |
| 共享视觉基底 | 固定正交基 $U\in\mathbb R^{512\times2}$ | `basis.npy` |
| Master | 视觉和 $S_m\in\mathbb R^{512}$、成员数、成员 ID、近似更新标记与来源信息 | 数组在 `master_sums.npy`；索引和元数据在 `graph.json` |
| 叶子 | 完整 caption、对话、时间戳、master ID、两个视觉坐标 $z_i\in\mathbb R^2$ | `graph.json` |
| Caption embedding | 可从 caption 重建的运行内文本缓存 | 不写入上述持久化图 |

Master 不生成文本摘要；其语义表征是视觉向量。记忆库不保存历史图片副本、原图路径或每张图片的完整视觉 embedding。实验数据目录仍保留源数据用于到达模拟；构建代码不会删除源数据。

## 3. Offline：逐条加入与重新分簇

### 3.1 共享二维坐标系

在目标 domain 的加入流程开始前，用其他 domain、且图片哈希与目标 domain 不重叠的校准图像拟合 PCA。实现逐张累计视觉向量的一阶和二阶矩，构造中心化协方差，取前两个主方向得到 $U$，满足 $U^\top U=I_2$。校准后固定 $U$。

对新图提取归一化向量：

$$
v_i=\operatorname{norm}(E_{\mathrm{CLIP,img}}(I_i)),\qquad
t_i=\operatorname{norm}(E_{\mathrm{BGE}}(c_i)),\qquad
z_i=U^\top v_i.
$$

**分簇编码只使用 caption；在线叶子编码才使用对话＋caption。** PCA 拟合时中心化，但保存的 $z_i$ 是绝对投影，不扣除校准均值，也不以某个 master 为坐标原点。因此旧叶子不需要随 master 中心移动而修改坐标。

### 3.2 用 caption 选择划分

每次加入一条记录，对所有已到达记录的归一化 caption 向量重新运行欧氏 Ward 聚类，并比较树上的不同切分：

$$
K^*=\arg\max_{K\in\{2,\ldots,N-1\}}
\operatorname{Silhouette}_{\rm Euclidean}(t_{1:N},\mathcal P_K).
$$

其中 silhouette 比较每条样本的簇内平均距离与最近其他簇的平均距离。当前规则不需要人为设置“相似度超过多少才能加入”。边界规则如下：

- 空库保持为空；$N\le2$ 时各图片单独成簇。
- $N\ge3$ 且所有归一化 caption 向量完全相同时，合为一簇。
- 其他情况比较 exact-$K$ 划分，$K=2,\ldots,N-1$；不比较单簇与全部单例端点。
- silhouette 差不超过 $10^{-12}$ 视为数值并列，选择更少的簇。

新划分允许旧成员迁移、旧簇拆分或合并。完全相同的成员集合复用 master ID；成员变化时生成新的 ID。因此这是一套**逐条到达、每次全量重分簇**的动态构建实现。

### 3.3 只有两个坐标，如何更新完整视觉 master？

旧簇 $C$ 保存未归一化视觉和 $S_C$，均值为 $\mu_C=S_C/|C|$。定义二维子空间之外的聚合残差：

$$
R_C=S_C-U\sum_{i\in C}z_i.
$$

当新划分得到簇 $G$ 时：

$$
S'_G=
\sum_{C\in\mathcal P_{\mathrm{old}}}
\left[
U\sum_{i\in C\cap G}z_i+
\frac{|C\cap G|}{|C|}R_C
\right]
+\sum_{i\in G\cap\mathrm{new}}v_i.
$$

这条公式的含义是：保留下来的两个方向按实际成员坐标累加；没有保留的方向只能把旧簇残差按迁移人数比例分配。整个旧簇转入时，代码直接加 $S_C$；新到达图片仍使用本次临时的完整 $v_i$。

部分迁移等价于用旧 master 和二维坐标近似重建旧叶子：

$$
\hat v_i=\mu_C+U(z_i-U^\top\mu_C)=Uz_i+R_C/|C|.
$$

用于聚合时不先把每个 $\hat v_i$ 单独归一化。新均值为 $\mu_G=S'_G/|G|$，在线 CLIP 匹配时才归一化其方向。

更新保持全局视觉和与保留方向的投影守恒：

$$
\sum_G S'_G=\sum_C S_C+\sum_{\mathrm{new}}v_i,\qquad
U^\top S'_G=\sum_{i\in G}z_i.
$$

**拆分后的单个 master 仍是近似均值。** 旧图在未保留方向上的个体差异无法恢复；发生部分重分配会设置 `approximate=True`，后续更新继承此标记。完成加入后，当前图片的完整视觉向量即被释放，不再读取旧图。

### 3.4 保存格式

```text
offline/
├── protocol.json
├── insertion_audit.jsonl
└── memory/
    ├── graph.json
    ├── basis.npy          # [512, 2]
    └── master_sums.npy    # [M, 512]
```

`graph.json` 通过引用取得视觉均值，例如一个有 5 个成员的 master：

```json
{"visual_mean_ref": {"path": "master_sums.npy", "row": 0, "scale": 0.2}}
```

这避免将长向量直接写进图的 JSON。只计算视觉数值状态，当前 float64 实现约需 $8(2D+MD+2N)$ 字节，$D=512$；该式不含文本、JSON 元数据和运行内文本缓存，也不表示整个记忆库的大小。

用于在线读取的自描述导出库另含 `memory.json`、`leaves.json`。当前 `export_store()` 导出 graph、视觉和、文本及编码元数据，**不复制 `basis.npy`**；它是在线只读包，不是可直接继续加入图片的完整构建状态。

## 4. Online：视觉选择假设，文本区分叶子

### 4.1 问题与候选答案陈述

当前 MemEye 实现面向四选项问题。为每题的 A/B/C/D 各准备一条完整、自包含的答案假设 $a_1,\ldots,a_4$，例如将“蓝色”扩写为“该场景中的方向箭头是蓝色”。

当前冻结候选由 assistant 根据问题和选项编写，**不是在线调用 Qwen/GPT 自动生成**。候选编写不读取标准答案、记忆文本、图片或检索得分。来源见[候选编写协议](../../reports/all_master_answer_query_20261021/candidate_generation_protocol.json)；新问题必须提供与其问题、选项一致的候选 JSONL。

### 4.2 每个 master 独立选择候选

$$
h_j=\operatorname{norm}(E_{\mathrm{CLIP,text}}(a_j)),\qquad
k_m=\arg\max_j\operatorname{norm}(\mu_m)^\top h_j.
$$

所有 master 都参与，每个 master 选择一个假设；并列时按 A/B/C/D 顺序。CLIP 分数只用于选择陈述，不与后续 BGE 分数相加。

### 4.3 扩展 query 与簇内文本匹配

为 master $m$ 构造：

```text
q_m = retrieval_query + "\nCandidate answer: " + selected_statement
leaf_text_i = dialogue_i + "\nMemory description: " + full_caption_i
```

答案陈述追加到 **query**，叶子文本不改写。同一个 master 的叶子共享该 query；多个 master 选中同一假设时共用相同 query 向量，所以每题只需编码四条不同的扩展 query。

BGE 的查询侧加前缀 `Represent this sentence for searching relevant passages: `，记忆侧不加前缀。归一化后：

$$
q_m^B=\operatorname{norm}(E_{\mathrm{BGE}}(\mathrm{prefix}\oplus q_m)),\quad
x_i=\operatorname{norm}(E_{\mathrm{BGE}}(\mathrm{leaf\_text}_i)),\quad
s_i=x_i^\top q_{m(i)}^B.
$$

遍历全部 master 及其叶子，将所有 $s_i$ 按降序合并取全局 Top-5；完全同分保持源记忆顺序。二维坐标及 master–leaf 距离在本版本中只服务于离线视觉更新，不参与在线打分。这里也没有提前停止或只检索前几个 master 的步骤。

### 4.4 回答与带图问题

回答模型接收：原问题、原选项、按排名排列的 Top-5“对话＋完整 caption”；带图问题额外接收该问题的原图。检索阶段使用的候选陈述不作为额外证据拼入 reader 上下文，master 向量和分数也不传给 reader。

若问题包含图片，预先生成的**题图 caption** 会拼入 `retrieval_query`；当前 CLIP 假设选择没有融合题图 embedding。历史图片仍不需要重新读取。`gold_ids` / `gold_choice` 仅用于排序后的评估。

因此视觉信息通过“选择假设 → 改变检索结果 → 改变回答上下文”间接帮助回答，当前 reader 不直接消费 master embedding。

## 5. 可选扩展：组内白化与 z-score 赋权

`--online-weighting none` 是核心方法；显式设置 `z_score` 才启用以下修正。对于每个 master 的全部叶子文本向量，用中心化矩阵的 SVD 得到主方向 $p_j$ 与样本协方差特征值 $\nu_j$：

$$
r_i=x_i^\top q_m^B,\qquad
w_i=r_i-\sum_j\frac{\nu_j}{\nu_j+\lambda}
(x_i^\top p_j)(p_j^\top q_m^B).
$$

中心化只用于拟合方向，打分仍使用原始归一化向量。然后在该 master 内标准化：

$$
\zeta_i=\frac{w_i-\operatorname{mean}(w)}{\operatorname{std}_{pop}(w)},\qquad
s_i=r_i+\alpha\operatorname{std}_{pop}(r)\tanh(\zeta_i).
$$

修正幅度不超过 $\alpha\operatorname{std}_{pop}(r)$；单叶子或零方差组修正为零。最终合并的是修正后的相似度，不是直接跨簇比较 z-score。该模块不改变 CLIP 选择、caption 或持久化 memory。

显式启用后 CLI 的默认值是 $\lambda=0.1,\alpha=1$。它引入两个可配置参数，不能因此把整个扩展方案称为“无超参数”。文中二维视觉坐标 $z_i$ 与这里的标量 $\zeta_i$ 是不同变量。

## 6. 运行与调试

以下命令在项目根目录 `/storage/hewenwen/VisualMemory` 执行，默认使用 GPU。它们依赖当前 checkout 已有的 Home 输入、冻结 caption、候选陈述、二维基底及协议指定的本地 CLIP/BGE 权重。

### 6.1 从 8 张图逐条构建

```bash
python3 scripts/visual_memory_pipeline.py offline \
  --memory caption_visual_master --device cuda:0 \
  --image-files D7_IMG_001.jpg D7_IMG_002.jpg D7_IMG_003.jpg \
    D7_IMG_004.jpg D7_IMG_005.jpg D7_IMG_006.jpg \
    D13_IMG_001.jpg D13_IMG_002.jpg \
  --output results/method1_debug/offline
```

图片按源记录顺序到达，不按命令中的排列顺序到达。本调试入口读取已有 caption，执行图片编码、文本编码、聚类和视觉更新，不现场调用 caption 生成模型。

### 6.2 检索三个问题

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory caption_visual_master \
  --memory-dir results/method1_debug/offline/memory \
  --retrieval multi_path --online-weighting none \
  --question-ids Q8 Q10 Q27 --top-k 5 \
  --device cuda:0 --stage prepare \
  --output results/method1_debug/online
```

结果在 `results/method1_debug/online/retrieval_results.json`，包含各 master 的候选选择、扩展 query、逐叶分数和 Top-5 上下文。这里 `raw` 与 `weighted` 两个条件相同；`raw` 指多路径的未赋权分数，不是原问题的 full-scan 基线。

### 6.3 同一候选池的全量检索基线

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory caption_visual_master \
  --memory-dir results/method1_debug/offline/memory \
  --retrieval full_scan --online-weighting none \
  --question-ids Q8 Q10 Q27 --top-k 5 \
  --device cuda:0 --stage prepare \
  --output results/method1_debug/full_scan
```

关闭赋权时，这条命令对相同叶子池使用同一个原始 query，等价于全量 BGE 余弦检索。若要试 z-score，将多路径命令中的 `none` 改为 `z_score`，追加 `--whitening-lambda 0.1 --reweight-alpha 1`，并使用独立输出目录。

以上 online 入口**只检索，不调用 QA reader**，8 图小库也不代表完整 Home 域的效果。完整域自描述库可通过 `--memory-path` 加载；`--memory-source` 只是导出已有离线结果，不会重建。完整 backbone/reader 实验入口见[主 pipeline 文档](../../VISUAL_MEMORY_PIPELINE.md)。不要将示例目录改为旧的默认 `results/debug_caption_master_answer/offline/memory`：组件模式对该默认路径有自动切换到完整域库的兼容逻辑。

### 6.4 代码入口与断点

| 阶段 | 文件与函数 |
| --- | --- |
| CLI 参数和分发 | [visual_memory_pipeline.py](../../scripts/visual_memory_pipeline.py)：`main()` / `offline()` |
| 新图编码和逐条加入 | [caption_visual_master.py](../../scripts/memory/caption_visual_master.py)：`offline()` |
| caption 分簇、旧成员重分配 | [caption_ward_visual.py](../../scripts/graph/caption_ward_visual.py)：`choose_caption_partition()`、`update_visual_partition()`、`CaptionWardVisualMemory.add()` |
| 外域 PCA 基底、图与数组落盘 | [prepare_all_domain_adaptive_inputs.py](../../scripts/analysis/prepare_all_domain_adaptive_inputs.py)：`calibrate_external_bases()`、`save_memory()` |
| CLIP 候选答案匹配 | [caption_visual_master.py](../../scripts/memory/caption_visual_master.py)：`choose_master_answers()` |
| 当前在线 query、BGE 分数、Top-5 | [online_retrieval.py](../../scripts/memory/online_retrieval.py)：`MultiPathRetrieval.plan()` / `retrieve_question()` |
| 可选赋权 | [online_zscore.py](../../scripts/memory/online_zscore.py)：`ZScoreWeighting.apply()`；[ridge_whitening.py](../../scripts/memory/ridge_whitening.py) |
| 原始配对 QA reader | [evaluate_all_master_answer_reader.py](../../scripts/analysis/evaluate_all_master_answer_reader.py)：`prepare()` / `run()` |

重新生成框架图：`python3 docs/method_1/draw_framework.py`。该脚本使用 Matplotlib 和当前 checkout 中的 SimSun 中文字体，输出 SVG、PDF、PNG；也可通过 `METHOD1_CJK_FONT=/path/to/font.ttf` 指定其他中文字体。

### 6.5 Random-2D：独立的基底消融入口

`--visual-basis pca-2d` 是 offline 默认值：仍加载原来的 `--basis` 文件，使用原来的默认输出目录。随机入口使用 `--visual-basis random-2d --basis-seed 0`，从局部 PCG64 随机数生成器采样一个 `512 × 2` 标准高斯矩阵，做 reduced QR，并固定 R 对角线非负的符号约定。得到的正交基在整个加入过程中保持不变；不读取 PCA 基底或外部校准图片。

caption 分簇、视觉 master 更新公式、每个叶子的两个坐标、CLIP/BGE 编码器和 online 检索均复用现有实现。新图仍只在到达时编码，之后不保存原图或完整图片 embedding。

```bash
python3 scripts/visual_memory_pipeline.py offline \
  --memory caption_visual_master --visual-basis random-2d \
  --basis-seed 0 --device cuda:0
```

这条命令仍构建现有的 8 张 Home 调试图片，按源顺序加入；可以用原有 `--image-files` 选择更多 Home 图片。默认输出为 `results/debug_caption_master_answer_random_2d/seed_0/offline`，其他种子使用对应的 `seed_N` 子目录。可通过 `--output` 指定新的目录；Random-2D 拒绝覆盖非空目录，PCA 入口也拒绝覆盖标记为 Random-2D 的输出。

在线阶段显式指定这个 memory，避免读取原来的 PCA 默认库：

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory caption_visual_master \
  --memory-dir results/debug_caption_master_answer_random_2d/seed_0/offline/memory \
  --retrieval multi_path --online-weighting none \
  --question-ids Q8 Q10 Q27 --top-k 5 \
  --device cuda:0 --stage prepare \
  --output results/debug_caption_master_answer_random_2d/seed_0/online
```

入口代码是 [caption_visual_master_random_2d.py](../../scripts/memory/caption_visual_master_random_2d.py)：`make_random_basis()` 生成基底，`offline()` 检查独立输出并调用共享 builder。`protocol.json` 记录 `basis_kind=random-2d`、随机种子、生成器和 QR 约定、NumPy 版本，以及实际保存的 `memory/basis.npy` 的 SHA-256。`graph.json` 和数组的格式与原有方案兼容。

这一入口当前用于 MemEye Home 构建适配器。Random-2D 与 `--dataset mem_gallery`、自定义 `--basis` 或 `--memory-source` 导出不能混用；`--basis-seed` 仅用于 Random-2D。比较两组时需分别从相同到达序列建库，不能直接替换已构建记忆的 `basis.npy`。完整多领域消融需另行运行，此入口本身不产生全量评估结论。

## 7. 基线、已有结果与适用边界

| 项目 | `full_scan + none` | Method 1：`multi_path + none` |
| --- | --- | --- |
| 叶子内容 | 对话＋完整 caption | 相同 |
| Query | 原问题，带图时附题图 caption | 每个 master 追加其选中的完整候选陈述 |
| 视觉信息 | 不参与排名 | CLIP 视觉 master 选择陈述 |
| 候选池 | 当前 domain 全部叶子 | 相同，全部 master 都参与 |
| 最终排序 | BGE 余弦，全局 Top-5 | BGE 余弦，全局 Top-5 |
| Reader | 原问题、选项、Top-5 文本、可选题图 | 相同格式与模型配置 |

因此现有比较同时包含“候选答案扩展 query”与“视觉选择候选”的作用；仅凭这组基线不能把全部收益归因于视觉 master。分离贡献需要保持候选陈述一致，再对比视觉选择、其他选择方式与不分簇的扩展检索。

以下是 **2026-10-21、原始 Qwen caption/reader、无 z-score** 的已保存配对结果，作为该核心版本的可追溯参照，不代表其他 backbone 或新增数据：

| 范围 | 方法 | Recall@5（命中标注记忆数 / 标注总数） | QA 正确率 |
| --- | --- | --- | --- |
| MemEye 全部 371 题 | 全量基线 | 492/1041 = 47.26% | 184/371 = 49.60% |
| MemEye 全部 371 题 | Method 1 | 500/1041 = 48.03% | 195/371 = 52.56% |
| Home 52 题 | 全量基线 | 54/68 = 79.41% | 29/52 = 55.77% |
| Home 52 题 | Method 1 | 60/68 = 88.24% | 33/52 = 63.46% |

来源：[检索汇总](../../reports/all_master_answer_query_20261021/retrieval_summary.json)、[回答汇总](../../reports/all_master_answer_query_20261021/reader/summary.json)。Recall@5 为上述标注计数的 micro 比例；QA 是固定选项排列下的准确率，不是四次轮换的官方 EM。结果在 domain 之间有升有降；全量配对 McNemar 双侧 $p\approx0.200$，不能据此声称显著、稳定的提升。

当前实现的主要边界：

- **离线成本：** 每次加入重跑 Ward 并评估多个切分，构造两两距离；还不是局部增量聚类算法。
- **在线成本：** 仍打分全部叶子，不减少候选数；组件入口目前每题重新编码叶子文本，不能声称已经实现编码缓存带来的加速。
- **视觉压缩：** 两个坐标只保留两个方向的个体差异；拆分时的 master 更新可能累积误差。master 数量接近叶子数时，视觉存储压缩收益也会缩小。
- **划分偏好：** 无固定相似度阈值不等于无设计选择；Ward、silhouette、允许的 $K$ 范围、二维基底都会影响结果。
- **任务范围：** 当前核心候选流程依赖四选项题；开放问答的候选生成与选择协议尚需另行定义。
- **长度策略：** 完整文本保留给 reader；BGE 上限为 512 token，CLIP 陈述上限为 77 token。普通入口拒绝无记录的截断；已有全量实验只沿用并核验显式记录的超长文本策略。
