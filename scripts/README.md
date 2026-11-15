# Scripts

**Qwen 五组候选的 GPT-5 联合复验已完成，0/5 组通过全部条件。** 用户在单候选失败后要求测试所有 5 组：`(λ,α)=(0.01,1)/(0.1,0.6)/(0.03,1)/(0.01,0.6)/(0.003,0.3)`。两方法每组共用参数，12 个条件包含两个新未赋权对照，4,452 条记录仅需 1,557 次新调用；沿用 GPT-5 caption 和 72 个 master，相同完整请求只共享本轮新回答。每组必须同时通过三项严格比较，旧失败结果保持原样；这是扩展后的探索性复验。入口为 `analysis/retrieve_memeye_gpt5_qwen_five.py --device cuda:1`、`analysis/evaluate_memeye_gpt5_qwen_five.py --stage all`；本轮未赋权对照为 217/371、221/371；五组加权的 baseline/master 依次为 222/221、215/212、217/222、213/222、215/220。后 3 组满足加权 master > 加权 baseline，但均不满足两种方案分别优于未赋权对照。1,557 次新调用及 2,895 条共享均通过独立核验，零失败／拒答／格式错误／空答／截断。见 [完整报告](../results/memeye_gpt5_qwen_five_validation_20261024/report.html)、[五组比较 CSV](../results/memeye_gpt5_qwen_five_validation_20261024/candidate_results.csv)、[独立核验](../results/memeye_gpt5_qwen_five_validation_20261024/independent_audit.json)。

**Qwen 共享 λ/α 搜索与 GPT-5 验证已完成；GPT-5 未通过预定条件。** 25 组共享参数、两个新对照、共 19,292 条记录，3,155 次本地新推理；5 组满足三项严格比较，预定规则选中 λ=0.01、α=1。Qwen 全量 184→188/371、master 195→201/371；无格式错误或截断，独立核验通过。GPT-5 沿用其原有 caption 与 72 个 master，四组重新回答、相同完整请求共享：全量 211→222/371、master 219→219/371；加权 master 低于加权全量，三项严格比较仅第一项通过。1,164 次新调用、320 条共享均通过独立核验，无失败或格式问题。四组完整请求与旧轮相同，跨轮答案波动单独记录。见 [GPT-5 验证报告](../results/memeye_gpt5_qwen_selected_validation_20261024/report.html)、[核验](../results/memeye_gpt5_qwen_selected_validation_20261024/independent_audit.json)、[跨轮统计](../results/memeye_gpt5_qwen_selected_validation_20261024/repeatability.json)。见 [Qwen 搜索报告](../results/memeye_qwen_zscore_joint_search_20261024/report.html)、[25 组 CSV](../results/memeye_qwen_zscore_joint_search_20261024/search_results.csv)、[固定参数](../results/memeye_qwen_zscore_joint_search_20261024/selected_parameters.json)。入口为 `analysis/retrieve_memeye_qwen_zscore_search.py`、`analysis/evaluate_memeye_qwen_zscore_search.py`、`analysis/retrieve_memeye_gpt5_qwen_selected.py`、`analysis/evaluate_memeye_gpt5_qwen_selected.py`；独立复测使用新 `--output`，Qwen 续跑用 `--resume`。


统一入口为 `visual_memory_pipeline.py`，完整命令见 [pipeline 说明](../VISUAL_MEMORY_PIPELINE.md)。

MemEye 的 **GPT-5.4 caption＋GPT-5.4 回答＋z-score（λ=0.01、α=1）** 已按此前 **236/371 baseline** 的完整流程完成：`baseline_question_only + z-score` 为 **242/371（65.2291%）**，`master_answer_query + z-score` 为 **237/371（63.8814%）**。沿用原 GPT caption 和记忆，GPU 0 重算检索；未加权的 742 个完整排序、原始分数和回答参数均复现旧实验。726 次新调用、16 条本轮共享通过独立核验，无请求失败、格式错误、空答或截断。相对历史未赋权 236/371、241/371 为 +6／−4 题；API 跨轮波动单独记录。见 [正确口径报告](../results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/report.html)、[逐题 CSV](../results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/paired_results.csv)、[核验结果](../results/memeye_gpt54_caption_zscore_lambda0p01_alpha1_20261024/independent_audit.json)；[评测入口](analysis/evaluate_memeye_gpt54_caption_zscore.py) 接受 `--memory` 指定 GPT caption 实验路径，[核验入口](analysis/audit_memeye_gpt54_caption_zscore.py)。

此前 **Qwen caption＋GPT-5.4 回答＋z-score** 的单独实验为 **236/371、230/371**，其未赋权历史参照为 **227/371、235/371**。这组结果保留作不同 caption 来源的实验记录，不作为上述 GPT caption 方案的消融对照。见 [旧口径报告](../results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/report.html)、[核验结果](../results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/independent_audit.json)；[旧评测入口](analysis/evaluate_memeye_gpt54_zscore.py)、[旧核验入口](analysis/audit_memeye_gpt54_zscore.py)。

同一 **GPT-5.4 caption＋GPT-5.4 回答**方案的 **λ=0.1、α=1** 消融已完成：全量 **236/371（63.6119%）**、多路径 **237/371（63.8814%）**；相对 λ=0.01 的 242/371、237/371 为 −6／0 题。两轮原始分数、分组、查询和回答参数逐条相同；全量 10 题错→对、16 题对→错，多路径各 9 题。727 次新调用及 15 条本轮共享通过独立核验，无失败或格式问题；API 跨轮波动单独记录。见 [λ 对比报告](../results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison.html)、[逐题对比](../results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison.csv)、[对比核验](../results/memeye_gpt54_caption_zscore_lambda0p1_alpha1_20261024/lambda_comparison_audit.json)；[评测入口](analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p1.py)、[本轮核验](analysis/audit_memeye_gpt54_caption_zscore_lambda0p1.py)、[对比入口](analysis/compare_memeye_gpt54_caption_lambdas.py)。旧 λ=0.01 源码及结果未改写。

同一 GPT caption 流程的 **λ=0.001、α=1** 已完成：全量 **243/371（65.4987%）**、多路径 **240/371（64.6900%）**。相对 λ=0.01 为 +1／+3 题，相对 λ=0.1 为 +7／+3 题；两组历史输入与原始分数均逐条一致，730 次新调用、12 条本轮共享通过独立核验，无失败、格式错误、空答或截断。API 波动与逐题变化单独记录，当前小差异不构成稳定优势结论。见 [三个 λ 对比报告](../results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison.html)、[逐题对比](../results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison.csv)、[对比核验](../results/memeye_gpt54_caption_zscore_lambda0p001_alpha1_20261024/lambda_comparison_audit.json)；[评测入口](analysis/evaluate_memeye_gpt54_caption_zscore_lambda0p001.py)、[回答核验](analysis/audit_memeye_gpt54_caption_zscore_lambda0p001.py)、[对比入口](analysis/compare_memeye_gpt54_caption_lambda0p001.py)。此前 λ=0.01／0.1 的源码和结果未改写。

2026-10-24 按用户要求，另行完成两套 **GPT caption 未加权基线的 seed=42 回答复测**：`baseline_question_only` **233/371（62.8032%）**，`master_answer_query` **231/371（62.2642%）**。历史未传 seed 的结果仍为 236/371、241/371；742 条完整请求逐条核验仅新增 seed，caption、memory、排序、Top-5、原图和其他生成参数均冻结。720 次新调用、22 条本轮共享，零失败／格式错误／空答／截断。独立 seed 探测中，Health/Q38 相同请求重复返回 A、C，说明参数被接受不代表确定性；本次分差不能归因于 seed，也不能直接作为此前未传 seed 的 z-score 消融对照。见 [复测报告](../results/memeye_gpt54_seed42_baselines_20261024/report.html)、[探测记录](../results/memeye_gpt54_seed42_baselines_20261024/seed_support_probe.json)、[独立核验](../results/memeye_gpt54_seed42_baselines_20261024/independent_audit.json)。独立入口：[rerun_memeye_gpt54_seeded_baselines.py](analysis/rerun_memeye_gpt54_seeded_baselines.py)，默认 `--seed 42`；新适配器 [openai_gpt54_seeded_transport.py](evaluation/openai_gpt54_seeded_transport.py)，历史适配器和结果未改写。

同一冻结输入的 **seed=1042** 复测也已完成：两套未加权方案均为 **235/371（63.3423%）**。相对 seed=42，`baseline_question_only` 为 7 题错→对、5 题对→错，净 +2；`master_answer_query` 为 6 题错→对、2 题对→错，净 +4。全部 742 条完整请求逐条核验仅 seed 不同，720 次新调用及 22 条本轮共享通过独立核验，零失败／格式错误／空答／截断。各 seed 只有一轮，分差不能完全归因于 seed。见 [seed 对比报告](../results/memeye_gpt54_seed1042_baselines_20261024/report.html)、[逐题 CSV](../results/memeye_gpt54_seed1042_baselines_20261024/seed_comparison.csv)、[独立核验](../results/memeye_gpt54_seed1042_baselines_20261024/independent_audit.json)；入口为 [rerun_memeye_gpt54_seed1042_baselines.py](analysis/rerun_memeye_gpt54_seed1042_baselines.py)，默认 `--seed 1042`，复用冻结的 seed-enabled transport 与运行循环，保留 seed=42 源码及结果。

直接使用 **`gpt-5.4` 别名、seed=42** 的同一冻结输入复测已完成：两套未加权方案分别为 **236/371（63.6119%）**、**238/371（64.1509%）**。相对日期快照 seed=42 为 +3／+7 题，错→对／对→错分别为 7／4、9／2；答案字母改变 14／11 题。742 条请求仅 model 名称不同，720 次新调用实际全部返回 `gpt-5.4-2026-03-05`，22 条同轮共享，零失败／格式错误／空答／截断，独立核验通过。一次复测不能证明别名更准确或更可复现。见 [别名报告](../results/memeye_gpt54_alias_seed42_baselines_20261024/report.html)、[逐题对照](../results/memeye_gpt54_alias_seed42_baselines_20261024/alias_comparison.csv)、[独立核验](../results/memeye_gpt54_alias_seed42_baselines_20261024/independent_audit.json)；运行 [rerun_memeye_gpt54_alias_seed42_baselines.py](analysis/rerun_memeye_gpt54_alias_seed42_baselines.py) 的 `--stage all --model gpt-5.4 --seed 42`。新 [alias transport](evaluation/openai_gpt54_alias_transport.py) 保留字面别名并分别记录请求名与返回版本，历史快照代码和结果不变。

在上述 **`gpt-5.4` 别名、seed=42** 基础上叠加 **λ=0.01、α=1** 的 z-score 已完成：`baseline_question_only` 为 **241/371（64.9596%）**，相对未加权 236 净 +5（18 错→对、13 对→错）；`master_answer_query` 为 **235/371（63.3423%）**，相对未加权 238 净 −3（16 错→对、19 对→错）。GPU 0 重算检索，742 组未加权完整排名及原始分数完全一致；仅加权后的上下文改变，其余回答参数冻结。726 次新调用、742 条逻辑记录，实际全部返回 `gpt-5.4-2026-03-05`，零失败／格式错误／空答／截断，独立核验通过。完整请求未变的 182／111 题中仍有 9／3 题答案改变，故单轮分差不能全部归因于 z-score。见 [报告](../results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/report.html)、[逐题 CSV](../results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/paired_results.csv)、[独立核验](../results/memeye_gpt54_alias_seed42_zscore_lambda0p01_alpha1_20261024/independent_audit.json)；运行 [evaluate_memeye_gpt54_alias_seed42_zscore.py](analysis/evaluate_memeye_gpt54_alias_seed42_zscore.py) 的 `--stage all --model gpt-5.4 --seed 42 --whitening-lambda 0.01 --reweight-alpha 1 --device cuda:0`，独立复测须指定新的 `--output`。历史未传 seed 的加权 242/237 保持原记录。

**GPT-5 新 caption＋新记忆＋新回答** 已完成。437 张独立图片重新生成 caption，GPU 0 重建 438 条记忆及 72 个 master；caption 和回答实际均为 `gpt-5-2025-08-07`。回答 seed=42，caption 未传 seed；使用 minimal 推理、默认温度（不传 temperature）、2048 总 token 预算，与此前 GPT-5.4／5.5 的生成参数存在适配差异。固定 λ=0.01、α=1，四组成绩为：全量 **219/371（59.0296%）**、全量加 z-score **221/371（59.5687%）**、master **223/371（60.1078%）**、master 加 z-score **221/371（59.5687%）**。赋权分别净 +2／−2，不支持稳定收益结论；437 次新 caption 调用及 1,164 次新回答调用覆盖 1,484 条评测记录，零失败／拒答／格式错误／空答／输出截断，三项预检与独立核验通过。见 [报告](../results/memeye_gpt5_base_seed42_full_20261024/report.html)、[逐题 CSV](../results/memeye_gpt5_base_seed42_full_20261024/paired_results.csv)、[独立核验](../results/memeye_gpt5_base_seed42_full_20261024/independent_audit.json)。按顺序运行 [接口探测](analysis/probe_memeye_gpt5_base.py)、[caption 生成](evaluation/memeye_gpt5_base_captions.py)、[GPU 记忆／检索重建](analysis/retrieve_memeye_gpt5_base.py)、[四组问答](analysis/evaluate_memeye_gpt5_base.py)；独立复测指定同一新 `--output`。测试入口：`python3 -m unittest scripts.tests.test_memeye_gpt5_base -v`。

同一 **GPT-5 caption＋GPT-5 回答** 的 **λ=0.001、α=1** 消融已完成：全量 **214/371（57.6819%）**、master **217/371（58.4906%）**，相对 λ=0.01 的 221/221 为 −7／−4 题。沿用同一 caption、438 条记忆与 72 个 master，原始分数／查询／分组及回答参数均一致；735 次新回答覆盖 742 条记录，零失败／拒答／格式错误／空答／输出截断，独立核验通过。两组分别有 83／200 题完整请求相同，其中仍有 5／11 题答案改变，故全部分差不能归因于 λ。见 [报告](../results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/report.html)、[逐题 CSV](../results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/paired_results.csv)、[核验](../results/memeye_gpt5_base_seed42_zscore_lambda0p001_alpha1_20261024/independent_audit.json)。运行 [GPU 检索](analysis/retrieve_memeye_gpt5_base_lambda0p001.py) 后运行 [回答评测](analysis/evaluate_memeye_gpt5_base_lambda0p001.py)；支持 `--memory` 和新的 `--output`，历史代码与结果不变。

同一 GPT-5 流程的 **λ=0.1、α=1** 已完成：全量 **216/371（58.2210%）**、master **214/371（57.6819%）**；相对 λ=0.01 的 221/221 为 −5／−7，相对 λ=0.001 的 214/217 为 +2／−3。caption、记忆、原始分数／查询／分组及回答参数均一致；725 次新调用覆盖 742 条记录，零失败／拒答／格式错误／空答／输出截断。与 λ=0.01 相同的 203／211 个完整请求中，两组各有 10 题答案改变，故不能将全部分差归因于 λ。预检、本轮独立核验和三个 λ 的独立重计均通过。见 [三个 λ 报告](../results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison.html)、[逐题 CSV](../results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison.csv)、[对比核验](../results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/lambda_comparison_audit.json)、[本轮核验](../results/memeye_gpt5_base_seed42_zscore_lambda0p1_alpha1_20261024/independent_audit.json)。入口为 [GPU 检索](analysis/retrieve_memeye_gpt5_base_lambda0p1.py)、[回答评测](analysis/evaluate_memeye_gpt5_base_lambda0p1.py)、[三个 λ 汇总](analysis/compare_memeye_gpt5_base_lambdas.py)。

**GPT-5 固定 λ=0.01 的 α 消融**已完成：按 α=0.01／0.1／1／10 排列，baseline 为 **212／211／221／213 / 371**，master 为 **220／220／221／204 / 371**；α=1 为历史参照，其余三组为本轮新回答。GPU 1 重算检索，沿用同一批 GPT-5 caption、438 条记忆及 72 个 master，原始／白化／z 分数与历史完全一致。1,474 次新 API 调用覆盖 2,226 条记录，752 条同轮相同请求共享回答；完整模拟及独立核验通过，无请求失败，1 条格式错误按既定规则计错。跨轮波动不能归因于 α：baseline α=0.01 仅改变 3 题的 Top-5，但相对未赋权的净 −7 全部发生在 368 个相同完整请求的子集中。见 [完整报告](../results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/report.html)、[α 对照表](../results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/alpha_comparison.csv)、[逐题 CSV](../results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/paired_results.csv)、[独立核验](../results/memeye_gpt5_base_seed42_zscore_lambda0p01_alpha_ablation_20261024/independent_audit.json)。入口为 [GPU 检索](analysis/retrieve_memeye_gpt5_base_alpha.py)、[回答评测](analysis/evaluate_memeye_gpt5_base_alpha.py)、[独立核验](analysis/audit_memeye_gpt5_base_alpha.py)。

**GPT-5.5 新 caption＋新记忆＋新回答** 已完成。437 张独立图片全部重新生成 caption，GPU 0 重建 438 条记忆及 81 个 master；caption 和回答实际均返回 `gpt-5.5-2026-04-23`。回答 seed=42，caption 未传 seed。四组成绩为：全量 **250/371（67.3854%）**、全量加 z-score **237/371（63.8814%）**、master **247/371（66.5768%）**、master 加 z-score **251/371（67.6550%）**。固定 λ=0.01、α=1，赋权在全量净 −13（7 错→对、20 对→错），在 master 净 +4（16 错→对、12 对→错）；同一完整请求在四组间共享本轮新回答。437 次 caption 调用和 1,174 次回答调用覆盖 1,484 条评测记录，零失败／拒答／格式错误／空答／输出截断，独立核验通过。GPT-5.4 历史参考同时改变 caption 与回答，不是只替换 reader 的对照。见 [报告](../results/memeye_gpt55_seed42_full_20261024/report.html)、[逐题 CSV](../results/memeye_gpt55_seed42_full_20261024/paired_results.csv)、[caption 来源](../results/memeye_gpt55_seed42_full_20261024/caption_model_provenance.json)、[独立核验](../results/memeye_gpt55_seed42_full_20261024/independent_audit.json)。按顺序运行 [接口探测](analysis/probe_memeye_gpt55.py)、[caption 生成](evaluation/memeye_gpt55_captions.py)、[GPU 记忆／检索重建](analysis/retrieve_memeye_gpt55.py)、[四组问答](analysis/evaluate_memeye_gpt55.py)；独立复测时指定同一新 `--output`。测试入口为 `python3 -m unittest scripts.tests.test_memeye_gpt55 -v`。

此前 **GPT-5.4 生成 caption 并重建记忆／检索** 的未赋权结果见 [完整流程报告](../results/memeye_full_backbone_20261021/gpt-5.4-2026-03-05/comparison.md)。旧口径加权 baseline 与完整流程未加权 baseline 恰好都为 236/371，输入并不相同。详见 [口径澄清](../results/memeye_gpt54_zscore_lambda0p01_alpha1_20261024/comparison_scope_clarification.json)。

**184/371（0.495956873）全量基线**的候选记忆结构、查询内容、检索方式、编码方式和回答方式，见 [五项标注](../VISUAL_MEMORY_PIPELINE.md#184371-全量检索基线五项标注)。`memory/online_retrieval.py` 的对应代码位置带有 `【184/371 全量基线 · ...】` 中文注释；历史回答入口也在注释和说明中明确标出。

2026-10-23 已在该全量基线的检索步骤叠加 **λ=0.1、α=1 的在线 z-score**，完成 GPU 1 上的全部 371 题配对问答：原始 184/371，赋权后 183/371（3 题错→对、4 题对→错）。每个 domain 全部记忆为一个簇，查询和对话＋完整 caption 不变。见 [报告](../reports/baseline_question_only_zscore_20261023/report.html)、[执行入口](analysis/evaluate_baseline_question_only_zscore.py) 与 [独立核验入口](analysis/audit_baseline_question_only_zscore.py)。

同一全量基线的 **λ 消融**已完成：固定 α=1，λ=0.01 为 **188/371**，λ=0.1 为 **183/371**，λ=1.0 为 **183/371**；0.01 相对原基线净增加 4 题，配对检验 p=0.5413。两个新设置完成全部 371 题的新评测，结果见 [消融报告](../reports/baseline_question_only_lambda_ablation_20261023/report.html)；[执行入口](analysis/evaluate_baseline_question_only_lambda.py) 调用统一 online pipeline，并在分析阶段核验公式、输入和答案。

固定 **λ=0.01 的 α 消融**也已完成：α=0.01、0.1、1、10 分别为 **183、184、188、192 / 371**。α=10 相对 α=1 净增加 4 题（20 题错→对、16 题对→错，p=0.6177），相对未赋权原基线净增加 8 题。三组新设置在 GPU 1 完成全部评测，原始／白化／z 分数与参照完全一致；见 [α 消融报告](../reports/baseline_question_only_alpha_ablation_20261023/report.html) 和 [执行入口](analysis/evaluate_baseline_question_only_alpha.py)。

2026-10-24，全量基线固定 **λ=0.1 的 α 消融**也已完成：α=0.01、0.1、1、10 分别为 **184、183、183、192 / 371**。α=1 为核验后的既有结果，其余三组在 GPU 0 新评测；α=10 比未赋权的 184/371 多答对 8 题（14 题错→对、6 题对→错，p=0.1153）。三组原始／白化／z 分数与参照完全一致，全部回答通过独立核验。见 [完整报告](../reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/report.html)、[逐题 CSV](../reports/baseline_question_only_lambda0p1_alpha_ablation_20261024/reader/paired_results.csv) 和 [执行入口](analysis/evaluate_baseline_question_only_alpha_lambda0p1.py)。运行从固定代码副本加载检索实现，避免工作区同时编辑注释影响已冻结实验；重跑需新的输出目录。

**195/371（0.525606469）caption_visual_master 多路径方案**同样标注这五项，见 [五项标注](../VISUAL_MEMORY_PIPELINE.md#195371-caption_visual_master五项标注)。代码使用 `【195/371 多路径方案 · ...】`，历史结果使用 `method_annotations.master_answer_query`；完整候选陈述、master 内匹配与全局 Top-5、CLIP/BGE 各自职责均明确记录。195/371 属于未赋权方案。

2026-10-24 已删除错误的 177/371 原始选项＋caption-only 方案专属脚本、Home／全量／旧 λ 消融报告及调试输出。`master_answer_query`、当前在线赋权、共享白化组件及全部有效基线／消融保留，清理前后核对文件哈希。详见 [清理核验](../results/cleanup_option_caption_multipath_20261024/validation.json) 和 [删除清单](../results/cleanup_option_caption_multipath_20261024/removal_manifest.json)。

该 `master_answer_query` 方案叠加 **λ=0.01、α=1** 的簇内 z-score 后，GPU 1 全部 371 题得到 **201/371（54.1779%）**，未赋权对照本轮复现 195/371。24 题错→对、18 题对→错，净增加 6 题，配对检验 p=0.4408；目标记忆召回为 473/1041，低于未赋权的 500/1041。见 [完整报告](../reports/master_answer_query_zscore_20261023/report.html)、[执行入口](analysis/evaluate_master_answer_query_zscore.py) 和 [独立核验入口](analysis/audit_master_answer_query_zscore.py)。

同一 `master_answer_query` 方案固定 **λ=0.01、α=10** 已在 GPU 1 完成全部 371 题，得到 **180/371（48.5175%）**。相对 α=1 的 201/371，21 题错→对、42 题对→错，净减少 21 题；Top-5 目标记忆召回从 473/1041 降至 361/1041。本轮未赋权对照仍为 195/371，原始／白化／z 分数与 α=1 完全一致，仅修正项扩大 10 倍。见 [完整报告](../reports/master_answer_query_alpha10_20261023/report.html)、[执行入口](analysis/evaluate_master_answer_query_alpha10.py) 和 [独立核验入口](analysis/audit_master_answer_query_alpha10.py)。

`master_answer_query` 固定 **λ=0.01** 的小 α 消融也已完成：**α=0.01 为 195/371（52.5606%），α=0.1 为 194/371（52.2911%）**。两组各自重新执行 371 题和未赋权对照，后者均复现 195/371，且独立核验通过；原始／白化／z 分数完全一致。结合此前 α=1 的 201/371、α=10 的 180/371，当前测试中 α=1 准确率最高。见 [完整 α 消融报告](../reports/master_answer_query_alpha_ablation_20261023/report.html)、[执行入口](analysis/evaluate_master_answer_query_alpha.py)、[独立核验入口](analysis/audit_master_answer_query_alpha.py) 和 [汇总入口](analysis/summarize_master_answer_query_alpha.py)。

`master_answer_query` 固定 **α=1 的 λ 消融**已补齐：λ=0.01、0.1、1.0 分别为 **201、202、199 / 371**（54.1779%、54.4474%、53.6388%）。两组新设置在 GPU 1 完成全部问答及独立核验，未赋权对照均复现 195/371。λ=0.1 相对 0.01 净增加 1 题（7 题错→对、6 题对→错，p=1.0），λ=1.0 净减少 2 题；当前数值差异不支持独立验证的最优参数结论。见 [λ 消融报告](../reports/master_answer_query_lambda_ablation_20261023/report.html)、[执行入口](analysis/evaluate_master_answer_query_lambda.py)、[独立核验入口](analysis/audit_master_answer_query_lambda.py) 和 [汇总入口](analysis/summarize_master_answer_query_lambda.py)。

2026-10-24，`master_answer_query` 固定 **λ=0.1 的 α 消融**完成：α=0.01、0.1、1.0、10.0 分别为 **195、195、202、185 / 371**。三个新设置在空闲 GPU 0 完整执行并通过独立核验，α=1 使用已验证的 GPU 1 既有结果；原始／白化／z 分数完全一致，未赋权逐题答案均复现 195/371，相同完整输入的答案也一致。当前 α=1 最高，α=10 比其少答对 17 题。见 [完整报告](../reports/master_answer_query_lambda0p1_alpha_ablation_20261024/report.html)、[执行入口](analysis/evaluate_master_answer_query_alpha_lambda0p1.py)、[独立核验入口](analysis/audit_master_answer_query_alpha_lambda0p1.py) 和 [汇总入口](analysis/summarize_master_answer_query_alpha_lambda0p1.py)。

| 目录或文件 | 用途 |
| --- | --- |
| `data/` | 数据准备、缓存 caption 生成、原始对话加载 |
| `memory/` | FC(T)、Reflexion、文本记忆适配、Object Relation |
| `evaluation/mcq_reader.py` | 共用的 Qwen 选择题回答器 |
| `evaluation/mcq_top15_caption_qa.py` | Natural caption Top15 基线 |
| `graph/` | 独立的 CLIP 聚类、Ward 聚类、向量存储与检索界限算法 |
| `analysis/` | 保留方法的观测、干扰、检索与误差分析 |
| `tests/` | 本地回归测试 |
| `vendor_memeye_text/`、`vendor_memeye_reflexion/` | 固定版本第三方实现 |

```bash
python3 scripts/visual_memory_pipeline.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -t . -p 'test_*.py' -q
```

Natural 与 FC(T) 直接从 `results/natural_observation` 读取缓存。Object Relation 使用独立的 `memory/qwen_generation.py`，视觉聚类通过 `data/dialogues.py` 加载原始对话。

## Caption 分簇与视觉 master 检索

统一入口新增 `--memory caption_visual_master`，当前默认运行 Home 的 8 张图和 Q8/Q10/Q27：

```bash
python3 scripts/visual_memory_pipeline.py offline --memory caption_visual_master
python3 scripts/visual_memory_pipeline.py online --memory caption_visual_master
```

默认 `--device cuda:0`。实现位于 `memory/caption_visual_master.py`，caption 聚类和视觉更新核心位于 `graph/caption_ward_visual.py`。旧 `debug_master_answer_memory.py` 仅保留为转发到 pipeline 的兼容入口。VS Code 的 Offline/Online 两个调试项直接运行 pipeline。

在线检索按三层独立配置：`--memory-path` 从库内 `memory.json` 识别 offline 方法，`--retrieval full_scan|multi_path` 选择查询构造，`--online-weighting none|z_score` 选择是否赋权。`memory/online_retrieval.py` 加载 memory 和检索策略，`memory/online_zscore.py` 仅修正匹配分数；两种检索保持历史 184/371、195/371 的对话＋完整 caption 口径。`offline --memory-source ... --memory natural|caption_visual_master --output ...` 可导出既有离线结果为自描述记忆库。VS Code 两个 Home Q10 配置使用 GPU 1，固定 λ=0.1、α=1。详见 [memory 路径与在线赋权](../VISUAL_MEMORY_PIPELINE.md#memory-路径检索策略与在线-z-score)。

Home 小样例 Online 仅支持 `--stage prepare`（默认），执行检索并输出 baseline 与新方法的 Top5，不调用回答模型。加上 `--dataset mem_gallery` 可运行 Mem-Gallery 全量离线构建及在线 `prepare/run/analyze/all`；包含全部 20 个领域、1,711 道开放式问题的两条件配对回答。实现为 `memory/mem_gallery_master_answer.py` 和 `evaluation/mem_gallery_master_reader.py`，默认 GPU0。详细输入、断点和输出见 [pipeline 说明](../VISUAL_MEMORY_PIPELINE.md#caption_visual_mastercaption-分簇与视觉-master-检索)。

MemEye 的回答模型消融已接入同一入口：`online --memory caption_visual_master --dataset memeye --reader-backend api --api-model gpt-4o-mini --stage all`。它复用已固定的 371 道题及两种检索 Top5，仅替换最终 reader，不重新构建或检索。先用 `--stage prepare` 核验输入，或用 `--stage run --api-smoke` 检查一道文本题和一道带图题。接口实现位于 `evaluation/memeye_api_reader.py` 与 `evaluation/openai_chat_transport.py`；默认从 NTU 项目 `.env` 读取官方 API 配置，不保存密钥，不需要 GPU。详情见 [API 消融说明](../VISUAL_MEMORY_PIPELINE.md#memeye-回答模型-api-消融)。

GPT-5.4 对照使用相同命令，将 `--api-model` 改为 `gpt-5.4-2026-03-05`。新适配器为 `evaluation/memeye_gpt5_reader.py` 和 `evaluation/openai_gpt5_chat_transport.py`，固定官方 API、推理 none、temperature=0、max_completion_tokens=16、不发送 seed；默认独立目录后缀为 `_none`。既有 GPT-4o-mini 实现及协议保持冻结。完整运行后，`analysis/audit_memeye_gpt5_reader.py` 独立重建请求和评分。

## A-Mem 主检索 Top5

当前 A-Mem 默认主检索 K=5，也可显式设置 `online --memory a_mem --top-k 5`。适配层覆盖固定上游配置中的 K=10，不修改第三方源码或 YAML；协议同时记录实际 K 和上游 K。记忆写入时的近邻选择逻辑不变。

A-Mem 结果存入 `<output-root>/top5/a_mem/`（指定其他 K 时对应 `top<K>`），与历史 `<output-root>/a_mem/` 区分。旧结果保留其原协议和原分数。

这里的 K 限制主检索命中数；随后仍按原算法展开 `links` 邻居，最终回答上下文可能超过 K 条，不能直接称为最终上下文严格 Top5。SimpleMem 仍沿用原配置。

新增严格 5 条上下文版本：

```bash
python3 scripts/visual_memory_pipeline.py online --memory a_mem --top-k 5 --a-mem-context-limit 5 --stage all
```

主检索和邻居展开保持原顺序（主节点、它的邻居、下一个主节点……）；按记忆 ID 去重后，仅取前 5 条完整记忆进入回答上下文。因此邻居可以占用最终的 5 个名额，并非保留全部 5 个主节点后继续追加邻居。条数不足时报错，不重复填充。每题记录主节点 ID、最终上下文 ID 和条数；结果独立存入 `<output-root>/top5/a_mem_context5/`。省略 `--a-mem-context-limit` 仍运行原来的不封顶版本。

只改变 A-Mem 的 K 时，可在新目录完成 `prepare` 后，用 `scripts/memory/replay_a_mem_cache.py <原实验目录> <新实验目录>` 迁移兼容的记忆构建及检索关键词调用缓存。它核对模型、题目、观察数据和上游配置，记录来源签名，不复制答题调用。迁移后使用 `--stage run`，不要重新 `prepare`；节点一致性及答案是否新生成由结果审计检查。


## 历史基线结果

保留实验的 protocol、输入、输出和来源快照继续代表当时的运行。历史哈希不改写；修改后的入口用于新输出目录。模型、数据及第三方实现保持原样。

## 独立视觉实验输入

保留的 CLIP/Ward 实验改用 `data/independent_visual_inputs/`，包含已核对的完整 caption、对话和问题。当前运行不需要被删除的旧 caption 图。`analysis/frozen_comparison_inputs.py` 只读取独立视觉诊断的固定历史候选缓存，不进行旧方法检索。历史协议中的来源哈希保持原样；当前可复核范围见该输入目录的 `validation.json` 和 pipeline 说明。

## SimpleMem 最终上下文 5 条

当前 `online --mode simplemem` 默认使用 **aiming-lab/SimpleMem 官方文本接口**，固定提交 `db80b6a7c591e0ea730a058e9f5fc4eb06572299`，包版本 0.3.0。安装位置 `.vendor/simplemem_official_runtime/`；第三方源码由 `scripts/memory/simplemem_official_sources.json` 校验。

在已有 Qwen / MiniLM 环境中安装缺失的文本接口依赖：

```bash
python3 -m pip install --no-deps --target .vendor/simplemem_official_runtime -r scripts/requirements-simplemem-official.txt
```

官方流程由 `scripts/memory/simplemem_official.py` 接入：`create(mode="text")` → `add_dialogues()` → `finalize()` → `hybrid_retriever.retrieve()`。语义结构化压缩和在线整合由官方 MemoryBuilder 在写入窗口中完成（40 个说话轮次，重叠 2 个）；第三阶段执行检索规划及语义/关键词/结构化检索。为对齐本地比较，当前默认 `ENABLE_REFLECTION=False`，关闭反思补充检索，`SEMANTIC_TOP_K=5`，每个语义查询最多返回 5 条（上游默认 25 条）。关键词和结构化检索各最多 5 条。规划仍可生成多个查询，因此合并候选数可能超过 5；最终回答上下文另由下面的限制控制。用户与助手分别作为 Dialogue，图片 caption 追加到用户文本。离线步骤仍只准备图片 caption；实际建库发生在 online 的 run 阶段。

保留本地 Qwen、CPU MiniLM、统一选择题回答器。任务内部顺序建库/检索，任务间可并行；结束写入后重建全文索引，使全部窗口参与关键词检索。`runtime/construction_windows.json` 记录窗口来源和产出的 entry ID；这些是窗口级来源，不能视为每条记忆精确对应的证据轮次。模型、依赖版本及上述适配均写入 protocol。

从头构建并运行：

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/visual_memory_pipeline.py online --mode simplemem --stage all \
  --task Brand_Memory_Test --limit-questions 1 --memory-workers 1 --memory-batch-size 1 \
  --memory-max-new-tokens 8192 --simplemem-context-limit 5 --simplemem-preserve-empty \
  --output-root results/simplemem_official_k5_no_reflection_20261026
```

`--simplemem-context-limit 5` 在官方检索之后，按 entry_id 去重，保留返回顺序中的前 5 个完整条目进入统一回答器；原结果空时可由 `--simplemem-preserve-empty` 保留为空，其他不足 5 条的情况报错。官方默认输出为 `simplemem_official_text/`；加 5 条限制后为 `top5/simplemem_official_text_context5/`。更改源码或参数后使用新 `--output-root`。

全量 MemEye 测试去掉 `--task` 和 `--limit-questions`，覆盖 8 个任务、848 轮历史和 371 道题。显存受限时，`--memory-device-map devices.json` 可指定模块到 GPU 编号或 `"cpu"` 的映射；映射内容及文件哈希写入 protocol，实际分配写入 `generation_environment.json`，CPU 模块常驻 CPU。该选项只调整设备位置，不改变权重、精度和生成参数；未指定时沿用自动分配。结果可用 `python3 -m scripts.analysis.audit_simplemem_official_run <含 protocol.json 的结果目录>` 核对调用缓存、输入哈希、窗口覆盖、实际回答上下文及关闭反思的设置。

查询分析有格式保护：若上游解析器从格式错误的 JSON 中提取出内部数组，适配层最多进行三次仅修复 JSON 格式的 LLM 请求，成功的对象原样保留。重试记录在 `runtime/query_analysis_repairs.json`，不会启动反思补充检索。已完成建库的中断实验可通过 `python3 -m scripts.evaluation.resume_simplemem_official --source <旧结果目录> --target <新结果目录>` 续跑缺失问题；新协议保留父实验签名、调用来源和已生成答案，审计核对原有记忆、答案与上下文均未改动。

历史 MemEye SM(T) 使用 OmniMemoryOrchestrator，可用 `--simplemem-backend omni_legacy` 明确选择。历史 `simplemem.yaml` 的 `retrieve_k=20` 不是固定 Top20：其 QueryProcessor 会使用 5／10／20；旧流程没有建立 BM25 索引，不能据源码存在 BM25 分支就宣称旧结果用了它。历史 55.80% 及下方冻结检索消融均属于旧实现，不代表新官方文本接口结果。

历史上下文对照直接复用已核验的旧检索记录。原记录逐字还原后才截取，答案全部重新生成：

```bash
python3 scripts/evaluation/simplemem_context_ablation.py --context-limit 5 --preserve-empty --stage all
```

已有准备文件时用 `--stage run`，完成后用 `--stage analyze`。默认结果目录 `results/visual_memory_pipeline/top5/simplemem_context5/`；独立目录可用 `--output` 指定。该目录属于冻结检索消融，使用上述脚本续跑，不用普通 pipeline 的重新构建流程覆盖。

历史 371 题实际上下文为：365 题 10 条、4 题 5 条、1 题 20 条、1 题 0 条。按用户确认，Social_Chat_Memory_Test/Q38 保持空上下文，其余 370 题限制为 5 条。报告必须保留这一例外，不能声称全部 371 题都有 5 条。
