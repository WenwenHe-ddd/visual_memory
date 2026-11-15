# Visual Memory

MemEye / Mem-Gallery 的视觉记忆研究代码，包括 caption 构建、记忆组织、检索、回答、评测与分析工具。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/visual_memory_pipeline.py` | 统一 offline / online 入口 |
| `scripts/data/` | 数据准备与 caption 生成 |
| `scripts/memory/`、`scripts/graph/` | 记忆方法与检索实现 |
| `scripts/evaluation/`、`scripts/analysis/` | 评测、核验及分析源码 |
| `scripts/tests/` | 现有回归测试 |
| `scripts/vendor_*/` | 第三方源码、适配器、上游信息与许可证 |
| `data/` | 数据集文本、caption 缓存及独立实验输入 |
| `docs/` | 方法说明与图示生成源码 |
| `results/`、`reports/` | **仅显式保留的 caption / 输入文件与实验脚本**，不包含实验结果 |
| `release_manifest.json` | 本次发布文件来源、SHA-256 与 caption 记录数 |

## 发布范围

本仓库从本机研究工作目录整理，保留代码、文档、数据集文本和文本 caption。历史 caption 保留原相对路径，以兼容已有读取代码；`.gitignore` 对 `results/` 使用逐文件白名单。

未上传模型权重、原始图片、embedding、检索排序、模型回答、评分表、实验报告、运行日志、API 响应缓存、安装依赖目录或任何凭据。原工作目录未清理或覆盖。文档和源码中的历史结果引用仅为研究记录，对应产物不在此次发布中。

不同模型 / 协议的 caption 分别保留，未合并或替换。caption 文件和输入数据保持原字节内容；其中的历史绝对路径与 provenance 也保持原样，跨机器运行时需要配置本机路径。

## 环境与运行

建议使用 Python 3.11 或兼容的独立环境。基础依赖见 `requirements.txt`；第三方文本记忆组件和 SimpleMem 的附加依赖分别见 `scripts/requirements-text-memory.txt` 与 `scripts/requirements-simplemem-official.txt`；M2A 依赖见 `scripts/vendor_m2a/requirements-lock.txt`。这些环境按具体方法选择安装。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/visual_memory_pipeline.py --help
```

运行实验前，还需准备相应模型和原始图片，并检查方法参数、模型路径和数据路径。部分历史脚本依赖未发布的固定 memory、embedding、completion manifest 或实验产物，需要先按对应流程重新生成。此仓库是源码与 caption 发布包，不声称每个历史实验都能从当前文件直接复现。

完整参数和方法入口见 [VISUAL_MEMORY_PIPELINE.md](VISUAL_MEMORY_PIPELINE.md) 与 [scripts/README.md](scripts/README.md)。`run_experiment.sh` 是原自然记忆实验的历史批处理入口，会运行生成与评测，不建议把它当作安装检查。

如所选方法需要 API，请通过该方法支持的环境变量或本地配置提供凭据；部分历史脚本默认引用原机器的其他项目配置路径，使用前应通过参数修改。不要提交 `.env`、token 或 SSH 私钥。

## 第三方代码与数据

第三方代码保留各自的 `LICENSE*`、`THIRD_PARTY_LICENSES.md` 与上游记录；请遵守对应许可。数据来源和约束见数据目录中的说明。此发布不额外授予第三方代码或数据的新许可。
