# Method 2：组内 Ridge 白化与 z-score 分数修正

本文对应当前 `online_weighting=z_score` 的实际实现，核对日期为 **2026-10-26**。它是独立的在线评分模块，可叠加在 [Method 1 的视觉 master 多路径检索](../method_1/README.md)上，也可用于 Natural 全量文本检索。

**核心思路：先用组内协方差调整文本匹配的方向权重，再将调整后的分数标准化，用有界修正项更新原始 BGE 相似度，最后统一排序。** 它保留原有 query、记忆文本和分组，仅改变叶子的检索分数。

## 1. 简洁框架

![Method 2 framework](framework.png)

[矢量 SVG](framework.svg) · [PDF](framework.pdf) · [可编辑绘图代码](draw_framework.py)

图按“**已有查询与记忆 → 组内调整分数 → 全局重新排序**”展开。A–D 表示四条记忆，蓝色与橙色区分两个既有分组；灰色条表示原始匹配分数，彩色条表示加上修正后的分数。相同字母和颜色贯穿全图，直观展示记忆如何进入最终排名。图中分数与 Top-3 排名仅为流程示意，不是实验结果；实际默认取 Top-5。公式与边界情况见下文。

## 2. 模块放在哪里？

当前 pipeline 将**记忆结构、query 策略、在线赋权**分开配置：

| 配置 | 每组有哪些叶子？ | 该组使用什么 query？ |
| --- | --- | --- |
| `natural + full_scan + z_score` | 当前 domain 的全部叶子，组成一个组 | 原 `retrieval_query` |
| `caption_visual_master + multi_path + z_score` | 每个已保存 master 的全部叶子 | 原 query ＋该 master 选出的完整候选陈述 |
| `caption_visual_master + full_scan + z_score` | 仍按已有 master 分组 | 所有组共用原 `retrieval_query` |

**分组由 memory 决定，不由 `full_scan` / `multi_path` 决定。** 因而最后一行不等价于 Natural 的单组赋权，尽管它们使用相同的原问题 query。

在 Method 1 上使用时，CLIP 已经根据 master 视觉均值选择了候选陈述，之后才进入本模块。固定输入时，开启 z-score 不会改变这次 CLIP 选择。叶子二维视觉坐标也不参与本模块；白化发生在 **BGE 文本空间**。

## 3. 输入与符号

对一个包含 $n_g$ 条叶子的组 $g$，构造：

```text
leaf_text_i = dialogue_i + "\nMemory description: " + full_caption_i

full_scan:   query_g = retrieval_query
multi_path:  query_g = retrieval_query + "\nCandidate answer: " + selected_statement_g
```

带图问题的 `retrieval_query` 可包含事先生成的题图 caption。多路径候选陈述来自冻结候选文件，标准答案不参与选择或评分。

用同一个 BGE 编码器得到 L2 归一化向量：

$$
x_i=\operatorname{norm}(E_B(\mathrm{leaf\_text}_i)),\qquad
q_g=\operatorname{norm}(E_B(\mathrm{prefix}\oplus\mathrm{query}_g)).
$$

当前基础配置为 `BAAI/bge-small-en-v1.5`，维度 $d=384$；query 前缀为 `Represent this sentence for searching relevant passages: `，记忆侧不加前缀。实际编码器和长度策略由 memory 协议固定。

令 $X_g\in\mathbb R^{n_g\times d}$ 的第 $i$ 行为 $x_i^\top$，原始余弦分数为：

$$
r_i=x_i^\top q_g.
$$

下文用 $\zeta_i$ 表示标量 z-score，避免与 Method 1 的二维视觉坐标 $z_i$ 混淆。

## 4. 三步评分

### 4.1 用组内文本向量拟合协方差

对 $n_g>1$：

$$
\bar x_g=\frac1{n_g}\sum_i x_i,\qquad
\widetilde X_g=X_g-\mathbf1\bar x_g^\top,\qquad
C_g=\frac{\widetilde X_g^\top\widetilde X_g}{n_g-1}.
$$

实现对 $\widetilde X_g$ 做 SVD，取得主方向 $p_j$ 及样本协方差特征值 $\nu_j=\sigma_j^2/(n_g-1)$。最多保留 $\min(n_g-1,d)$ 个方向，数值零方差方向的特征值设为零；这里没有另设“只取前两个主成分”的超参数。

拟合使用**组内全部叶子**，发生在 Top-K 选择之前；不使用 gold 标签，也不跨 domain 混合样本。

### 4.2 Ridge 白化匹配分数

给定 $\lambda>0$，计算：

$$
\boxed{
w_i=r_i-\sum_j\frac{\nu_j}{\nu_j+\lambda}
(x_i^\top p_j)(p_j^\top q_g)
}
$$

等价矩阵形式为：

$$
A_g=\lambda(C_g+\lambda I)^{-1},\qquad
w_i=x_i^\top A_gq_g.
$$

沿主方向 $p_j$ 的匹配贡献保留比例为 $\lambda/(\nu_j+\lambda)$。高方差方向受到更强抑制，协方差零空间保留原始贡献。其动机是减弱组内高方差方向对匹配的支配，但**高方差不自动意味着噪声**，有效的区分信号也可能被抑制。

这里的“白化”准确地说是**带 ridge 正则的逆协方差匹配**。若令 $T_g=A_g^{1/2}$，则 $w_i=(T_gx_i)^\top(T_gq_g)$；代码直接计算分数，不显式生成这两个变换后的向量。中心化仅用于拟合 $C_g$，实际打分仍用原始 $x_i,q_g$，变换后也不再做 L2 归一化。因此它不是“中心化白化 embedding 后再计算余弦”的流程。

### 4.3 z-score 与有界残差修正

在同一组内对白化分数使用总体标准差（`ddof=0`）：

$$
\mu_{w,g}=\operatorname{mean}_{i\in g}(w_i),\qquad
\sigma_{w,g}=\operatorname{std}_{pop,i\in g}(w_i),\qquad
\zeta_i=\frac{w_i-\mu_{w,g}}{\sigma_{w,g}}.
$$

再以原始余弦分数的组内标准差确定修正尺度：

$$
\sigma_{r,g}=\operatorname{std}_{pop,i\in g}(r_i),\qquad
\Delta_i=\alpha\sigma_{r,g}\tanh(\zeta_i),\qquad
\boxed{s_i=r_i+\Delta_i},\qquad\alpha\ge0.
$$

三个量各有作用：

- $\zeta_i$：表示叶子的白化分数在本组中的相对位置。
- $\tanh$：限制极端 z-score 的影响，满足 $|\Delta_i|\le\alpha\sigma_{r,g}$。
- $r_i$ 与 $\sigma_{r,g}$：保留原始匹配分数作为基底，并让修正随组内原始分差缩放。

最终跨组合并的是 $s_i$，不是 $\zeta_i$。这保留了原始相似度的基底，但不构成不同 query 或不同组之间的概率校准保证。

## 5. 排序、边界情况与输出

对当前 domain 的全部叶子按 $s_i$ 降序取全局 Top-K，默认 $K=5$；完全同分保持源记忆顺序。所有组都参与，没有先选少数 master，也没有提前停止。

| 情况 | 当前实现行为 |
| --- | --- |
| 单叶子组 | 无协方差方向，修正为 0；诊断字段 `white_z` 保留原分数，状态为 `singleton_raw`，并非有效标准分数 |
| 原始分数或白化分数数值零方差 | 修正为 0，保留原始分数 |
| 两叶子且白化分数不相等 | z-score 精确设为 $+1,-1$；相等时为零 |
| $\alpha=0$ | 最终分数严格退化为原始分数，但仍执行白化计算 |
| $\lambda\to\infty$ | $w_i\to r_i$，但 z-score 修正一般仍存在，**不等于关闭赋权** |

零方差判定使用浮点容差，而非仅检查 `std == 0`。两叶子组在非退化情况下没有随 $\lambda$ 连续变化的 z-score 幅值，这会限制该参数在小组上的影响。

reader 使用排序后的原始“对话＋完整 caption”，配合原问题、选项及原始题图（若有）。白化向量、分数及检索用候选陈述不作为额外证据传给 reader。当前组件入口只生成检索结果，QA 需另行运行。

输出 `retrieval_results.json` 同时保留：

| 字段 | 含义 |
| --- | --- |
| `groups` | 分组大小、实际 query、候选选择及标准差/退化状态 |
| `leaf_scores` | 每个叶子的 `raw_score`、`white_score`、`white_z`、`correction`、`weighted_score` |
| `conditions` | `raw` 与 `weighted` 两套 Top-K，字段名沿用 `top5` |
| `source_sha256`、`memory_unchanged` | 输入与代码哈希，以及运行前后输入一致性检查 |
| `reader_called`、`vectors_persisted` | 本入口均为 `false` |

同一次多路径运行中的 `raw` 是**多路径未赋权结果**，不能把它称为原问题全量检索基线。在线 BGE 向量、PCA 方向和白化状态均为临时计算，不写回 memory。

## 6. 运行示例

在项目根目录 `/storage/hewenwen/VisualMemory` 执行。下面使用 checkout 已有的完整 Home 域输入、记忆和冻结候选；本次文档编写未重跑模型实验。

### 6.1 Method 1 ＋ Method 2

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory caption_visual_master \
  --memory-dir reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/memory \
  --question-source reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/inputs.json \
  --candidates reports/all_master_answer_query_20261021/candidates.jsonl \
  --retrieval multi_path --online-weighting z_score \
  --whitening-lambda 0.01 --reweight-alpha 1 \
  --question-ids Q8 Q10 Q27 --top-k 5 \
  --device cuda:0 --stage prepare \
  --output results/method2_debug/multi_path_zscore
```

这里显式使用既有部分实验的 $\lambda=0.01,\alpha=1$。**CLI 默认值实际为 $\lambda=0.1,\alpha=1$**，不是 0.01；复现时应显式指定。两者都是可配置超参数，不能将本扩展称为无超参数方案。

### 6.2 Natural 全量检索 ＋ Method 2

```bash
python3 scripts/visual_memory_pipeline.py online \
  --memory natural \
  --memory-dir reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design \
  --question-source reports/adaptive_retrieval_comparison_20261020/inputs_by_domain/Home_Renovation_Interior_Design/inputs.json \
  --retrieval full_scan --online-weighting z_score \
  --whitening-lambda 0.01 --reweight-alpha 1 \
  --question-ids Q8 Q10 Q27 --top-k 5 \
  --device cuda:0 --stage prepare \
  --output results/method2_debug/full_scan_zscore
```

关闭赋权的配对对照：将对应命令中的 `z_score` 改为 `none`，移除 `--whitening-lambda`、`--reweight-alpha`，并使用新的输出目录。显式保留 `--online-weighting none`，确保仍走相同的组件路径；省略相关路由参数可能进入历史 caption-only 入口。

已有 `memory.json` 的自描述库可以用 `--memory-path` 加载，由文件确定 memory 类型。不同 domain 需替换对应记忆与问题源；不要把多个 domain 直接合成一个白化组。

### 6.3 数值与路由验证

以下已有测试无需加载模型，覆盖逆协方差等价式、有界修正、退化组、显式关闭赋权及 query/文本/分组不变性：

```bash
python3 -m unittest \
  scripts.tests.test_ridge_whitening \
  scripts.tests.test_caption_z_reweight_pipeline -v
```

本次文档核对时上述 **12 项测试全部通过**；这验证数值和路由行为，不代表重新完成了 QA 实验。

重新生成图：

```bash
python3 docs/method_2/draw_framework.py
```

中文绘图优先使用本机已有的 Noto Sans CJK 字体；其他环境可追加 `--font /path/to/chinese-font.otf`。SVG/PDF 已嵌入字形，查看成图无需安装字体，文字和布局可通过绘图代码修改。

## 7. 代码对应与实验边界

| 阶段 | 实现入口 |
| --- | --- |
| 参数、默认值与路由 | [visual_memory_pipeline.py](../../scripts/visual_memory_pipeline.py)：`main()` |
| query 与组定义 | [online_retrieval.py](../../scripts/memory/online_retrieval.py)：`FullScanRetrieval.plan()`、`MultiPathRetrieval.plan()`、memory 的 `group_definitions()` |
| SVD、ridge 分数、标准化 | [ridge_whitening.py](../../scripts/memory/ridge_whitening.py)：`fit()`、`score()`、`standardize()` |
| 有界修正 | [online_zscore.py](../../scripts/memory/online_zscore.py)：`ZScoreWeighting.apply()` |
| 全局排序和审计输出 | [online_retrieval.py](../../scripts/memory/online_retrieval.py)：`retrieve_question()`、`online()` |

该模块不会减少候选数量。当前组件入口每题重新编码叶子并拟合组内 SVD，尚不能宣称具有跨问题缓存带来的加速。小组的协方差估计受样本数限制，singleton 不获得任何修正；不同分组和 query 策略下，效果需要分别测量。

评估应冻结记忆、caption、query、候选陈述、文本长度策略、Top-K 和 reader 参数，分别报告检索 Recall@K 与最终 QA accuracy。检索分数变化、召回变化都不能直接代替 QA 收益；现有不同 backbone/配置的结果也不支持预设稳定提升。已有实验索引见 [scripts README](../../scripts/README.md)，具体复现以各实验协议及独立核验文件为准。
