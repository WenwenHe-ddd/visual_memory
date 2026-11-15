# 第二观察：指定图片的 caption 相似度

独立入口 `second_observation.py`，不接入 `visual_memory_pipeline.py`。
读取已有 caption，用本地 BGE 模型对文本编码，计算全部图片对的余弦相似度。
图片只用于定位 caption 和记录内容哈希；不重新生成 caption，不计算视觉特征相似度。
两端都是 caption，不添加 query prefix。默认模型读取自 natural observation 的 `protocol.json`。

## 使用已有 caption 缓存

从项目根目录运行（也可以使用脚本和输入文件的绝对路径在任意目录执行）：

```bash
python3 scripts/tests/second_observation.py \
  --images data/MemEye/data/image/Brand_Memory_Test/CocaCola_1.png \
           data/MemEye/data/image/Brand_Memory_Test/BurgerKing_1.png \
           data/MemEye/data/image/Brand_Memory_Test/Starbucks_1.png \
  --captions results/natural_observation/captions.jsonl \
  --output-dir results/second_observation/example
```

`--images` 至少传入两张图片。`--captions` 接受一个共享文件，或与图片一一对应的多个文件。
共享 JSONL 每行一个对象；JSON 支持对象列表或单个对象。记录使用 `caption` 字段，
通过 `id`（图片文件 SHA-256）、`image_sha256` 或 `image_path`/`path` 匹配。
记录中的相对图片路径相对于 caption 文件所在目录；不按文件名猜测匹配。

若 caption 分别保存为纯文本，按图片顺序传入：

```bash
python3 scripts/tests/second_observation.py \
  --images /path/to/image_a.png /path/to/image_b.png \
  --captions /path/to/caption_a.txt /path/to/caption_b.txt \
  --output-dir results/second_observation/my_comparison
```

逐图传入时也支持仅含 `{"caption": "..."}` 的 JSON 文件；带有匹配字段的记录必须匹配，
不会在匹配失败时退回按位置分配。多个图片可以来自不同 caption 文件。
可选参数：`--model-path` 指定本地 SentenceTransformer 模型目录、`--device`（默认 `cpu`）、
`--batch-size`（默认 32）。运行需要项目已有的 torch、sentence-transformers 和 numpy 环境，
脚本会加载项目 `.vendor`，不会下载模型。

## 输出与检查

- `similarity.json`：输入图片顺序、图片/来源文件哈希、实际 caption、模型路径、token 数、完整矩阵及配对分数。
- `similarity_matrix.csv`：按 `--images` 顺序排列的 N×N 矩阵。
- `pairs.csv`：N(N−1)/2 个不重复图片对及相似度；终端也打印这些分数。

余弦相似度范围为 −1 到 1，越高表示该模型下文本越相似；不是概率或图片事实一致率。
缺失、歧义、空 caption、已标记生成截断或 embedding 输入超长均报错，避免静默使用不完整文本。
默认输出目录是项目下 `results/second_observation`；已有同名结果时拒绝覆盖，请另选目录。

针对性测试：`python3 -m unittest scripts.tests.test_second_observation -v`。
