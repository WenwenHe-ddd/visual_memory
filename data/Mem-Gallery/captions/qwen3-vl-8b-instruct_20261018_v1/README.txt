Mem-Gallery Qwen captions — qwen3-vl-8b-instruct_20261018_v1

原始数据和官方 image_caption 保留不变。
新字段：image_caption_qwen3_vl_8b_instruct

history_captions.jsonl：历史图片 caption，按对话轮组织；新字段为列表。
query_captions.jsonl：问题图片 caption，按任务和 QA 下标组织；新字段为字符串。
captions_by_image.jsonl：按原图文件 SHA256 去重后的生成结果，caption 为 Qwen 输出。
image_path_captions.jsonl：每个原图相对路径与 Qwen caption 的映射。
dialog_with_qwen_caption/：完整对话 JSON 副本；原字段不变，图片轮和图片问题增加新字段。
image -> ../../data/image：复用原图，使副本中的 ../image/... 路径仍可访问；未复制或改写原图。

使用新 caption 做实验时，须显式读取 image_caption_qwen3_vl_8b_instruct。
原来读取 image_caption 的程序仍会读取官方 caption。
image_path_captions.jsonl 的 image_path 相对于 data/Mem-Gallery/。
完全相同的图片文件共用一个 Qwen caption；模型仅接收图片与通用描述提示词，不接收问题、答案、对话或官方 caption。
GIF/MPO 仅读取首帧，与此前 caption 流程一致。

protocol.json：模型、提示词、参数、数据版本及输入校验信息。
attempts.worker*.jsonl：包括重试在内的全部模型输出。
audit.json：完成后的数量和完整性验证；status.json 为当前状态。
artifact_manifest.json：输出文件 SHA256；model_file_sha256.json：模型文件 SHA256。
independent_audit.json：生成完成后另行核对映射、源文件、原字段与图片链接的结果。

本次只生成 caption。已有线索过载实验仍使用官方 caption，尚未用新 caption 重跑。
