---
language:
- en
license: apache-2.0
size_categories:
- n<1K
task_categories:
- image-text-to-text
pretty_name: MemEye
tags:
- multimodal
- memory
- benchmark
- long-context
---

# MemEye

[**Paper**](https://huggingface.co/papers/2605.15128) | [**Project Page**](https://minghokwok.github.io/MemEye/) | [**Official Code**](https://github.com/MinghoKwok/MemEye)

MemEye is a visual-centric multimodal memory benchmark for evaluating agents that need to remember and reason over long-running image-grounded dialogues. It evaluates memory capabilities across two axes: visual evidence granularity (from scene-level to pixel-level) and memory reasoning depth (from atomic retrieval to evolutionary synthesis).

The dataset includes 371 mirrored MCQ + open-ended questions across 8 life-scenario tasks, featuring human-annotated clues and validation gates for answerability and visual necessity.

## Core Scenarios

| Scenario | Sessions | Dialogue rounds | QAs | Images |
| --- | ---: | ---: | ---: | ---: |
| Brand Memory Test | 42 | 72 | 29 | 30 |
| Card Playlog Test | 4 | 30 | 48 | 30 |
| Cartoon Entertainment Companion | 86 | 299 | 76 | 119 |
| Home Renovation Interior Design | 13 | 120 | 52 | 90 |
| Multi-Scene Visual Case Archive Assistant | 15 | 117 | 50 | 57 |
| Outdoor Navigation Route Memory Assistant | 10 | 60 | 28 | 40 |
| Personal Health Dashboard Assistant | 12 | 97 | 51 | 62 |
| Social Chat Memory Test | 39 | 53 | 37 | 67 |

## Sample Usage

### Loading with Python

You can download the dataset directly with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

data_dir = snapshot_download(
    repo_id="MemEyeBench/MemEye",
    repo_type="dataset",
)
print(data_dir)
```

### Running Evaluation

After setting up the [official code](https://github.com/MinghoKwok/MemEye), you can run a benchmark evaluation:

```bash
python run_benchmark.py \
  --task-config config/tasks_external/brand_memory_test.yaml \
  --model-config config/models/gpt_4_1_nano.yaml \
  --method-config config/methods/full_context_multimodal.yaml
```

## JSON Format

Each task JSON follows the benchmark schema:

```json
{
  "character_profile": { "...": "..." },
  "multi_session_dialogues": [
    {
      "session_id": "D1",
      "date": "2026-03-01",
      "dialogues": [
        {
          "round": "D1:1",
          "user": "...",
          "assistant": "...",
          "input_image": ["../image/<Task_Name>/<file>.png"]
        }
      ]
    }
  ],
  "human-annotated QAs": [
    {
      "question": "...",
      "answer": "...",
      "point": [["X2"], ["Y1"]],
      "session_id": ["D1"],
      "clue": ["D1:1"]
    }
  ]
}
```

## Intended Use

MemEye is intended for research on multimodal memory, long-context retrieval, image-grounded dialogue agents, and benchmark method comparisons. It is not intended for training or deploying systems that identify real people or infer sensitive personal attributes.

## Citation

```bibtex
@inproceedings{guo2026memeye,
  title={MemEye: A Visual-Centric Evaluation Framework for Multimodal Agent Memory},
  author={Guo, Minghao and Jiao, Qingyue and Shi, Zeru and Quan, Yihao and Zhang, Boxuan and Li, Danrui and Che, Liwei and Xu, Wujiang and Liu, Shilong and Liu, Zirui and Kapadia, Mubbasir and Pavlovic, Vladimir and Liu, Jiang and Wang, Mengdi and Shi, Yiyu and Metaxas, Dimitris N. and Tang, Ruixiang},
  booktitle={arXiv preprint arXiv:2605.15128},
  year={2026}
}
```

## License

The code and dataset documentation are released under the [Apache License 2.0](https://github.com/MinghoKwok/MemEye/blob/main/LICENSE).