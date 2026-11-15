# HippoMM image adaptation in the pipeline

This is an isolated `--memory hippomm` method. It adapts the memory and retrieval
ideas from [HippoMM](https://arxiv.org/abs/2504.10739), with the public implementation
pinned for reference at
[`d7000b4`](https://github.com/linyueqian/HippoMM/tree/d7000b40d327017f0a47886b83033683910177f5).
It is **not** a reproduction of the complete audiovisual system or its published scores.

The existing methods, their defaults, and the `all` selector are unchanged.
HippoMM dispatches before the legacy Home example defaults and other-method validation.
It supports the existing MemEye MCQ manifest and Mem-Gallery open-answer inputs.

## Agreed adaptation

1. Ingest historical images directly; no video segmentation, round segmentation,
   temporal window search, audio model, or ImageBind dependency.
2. Use the local `models/clip-vit-base-patch32` image and text towers from the same
   checkpoint. L2-normalize projected features. Images are scoped to their task.
3. Process distinct images in source order. Retain the first image; associate a
   subsequent image with the most similar retained representative if cosine is
   at least `0.9`. Otherwise retain it as a new representative. This is greedy
   novelty filtering, not clustering or feature averaging.
4. One **retained image** is one ThetaEvent. Store its CLIP vector, existing verified
   Qwen caption, all associated original dialogues, source IDs/times, and a freshly
   generated Qwen summary. Redundant images lose their visual feature, not their
   source dialogue. Keep explicit `visually_redundant` mappings; a representative's
   appearance must not be asserted to be identical to its aliases.
5. Query first attempts an answer from **all same-task event summaries only**.
   Use the LLM's explicit confidence with a strict `> 0.75` acceptance condition.
   Invalid confidence/JSON fails closed to detailed retrieval; it is not a cosine score.
6. Detailed retrieval rewrites the question into a concrete 2–5 word visual cue,
   encodes it with CLIP text, and compares against retained image vectors. For each
   single-image event with cosine `< 0.4`, ask the LLM to select or reject its caption.
   Selected caption candidates receive the fixed score `0.6`; rejected candidates
   are omitted. Merge all candidates, sort stably by score, then take global Top-K
   (default 5, fewer if fewer candidates exist). Empty retrieval remains empty.
7. Answer from the selected captions and associated original dialogues/times.
   There is no history-image reread, vector decoding, additional text-vector search,
   cross-modal sequential search, or answer-reflection step.

The duplicate threshold `0.9` follows the public visual key-frame filter; it is
distinct from the paper's segment-consolidation threshold `0.85`. The `0.75`
confidence threshold follows the paper; the public code has different per-type
gates. The caption fallback thresholds follow the public code. These are explicit
adaptation choices, not claims that paper and code are identical.

## No-original-image boundary

The new memory store saves **no historical image files or reloadable image paths**.
It contains numeric features and text. The shared source dataset is read-only and
is never deleted. This preserves other methods while enforcing no historical pixel
access at query time. Shared source-cache validation stays in the read-only input adapter.
Online memory validation checks event/vector alignment and source coverage without
opening original history images.

Captions reuse the existing validated local Qwen caches. No new caption is generated
for a rejected image. Existing shared caches are not modified or deleted. Summaries
are newly generated after all redundant-image source dialogues have been attached.
If a scene attribute was lost from captions, detailed retrieval cannot recover it
from the discarded original pixels.

All dialogue rounds are stored in `dialogues.jsonl`. Text-only rounds have no
standalone visual event and are outside this image-only retrieval/summary pool.
They remain in evaluation denominators when labeled as evidence. Query images are
different from historical images: original question pixels may be passed to the
fast-answer and final-answer LLM, but never to CLIP query rewriting or caption
selection. Supplied dataset query captions are not used.

## Commands

Run from the repository root with the environment used by the existing local
Qwen/CLIP methods. Model inference requires the local checkpoints and CUDA for Qwen.
`prepare` readiness and unit tests do not need CUDA.

Validate source input without building memory or loading models:

```bash
python3 scripts/visual_memory_pipeline.py offline --memory hippomm \
  --hippomm-stage prepare --task Brand_Memory_Test \
  --output results/hippomm_brand/offline
```

Build all Brand history (omit `--task` for all MemEye tasks):

```bash
python3 scripts/visual_memory_pipeline.py offline --memory hippomm \
  --task Brand_Memory_Test --device cuda:0 \
  --output results/hippomm_brand/offline
```

Prepare, answer, and evaluate one question:

```bash
python3 scripts/visual_memory_pipeline.py online --memory hippomm --stage all \
  --task Brand_Memory_Test --limit-questions 1 --device cuda:0 \
  --memory-dir results/hippomm_brand/offline \
  --output results/hippomm_brand/online
```

For the full benchmark, omit `--task` and `--limit-questions` and build the matching
full memory first. Online stages are `prepare`, `run`, `analyze`, or `all`.
`prepare` snapshots sanitized questions and does not call the LLM. `run` executes
both query-dependent retrieval and answering; it resumes completed questions.
`analyze` refuses incomplete responses. Retain the same flags when resuming a run.

Mem-Gallery uses the same commands with `--dataset mem_gallery`. Inputs default to
the existing audited `results/mem_gallery_master_answer_20261021/inputs` directory;
override it with `--hippomm-gallery-inputs` if needed. It must contain the prepared
`memories.jsonl`, `questions.jsonl`, `gold.jsonl`, `protocol.json`, and
`artifact_manifest.json`. Their preparation is documented in `scripts/data/mem_gallery_inputs.py`.
Do not use a MemEye task name with Mem-Gallery.

Without explicit paths, outputs are isolated under
`results/visual_memory_pipeline/hippomm/<dataset>/{offline,online}`.
Use `--output` for this method, not the legacy `--output-root`.
`online --memory hippomm --memory-path <directory>` is also accepted as an explicit
alias for `--memory-dir`; omit neither the method nor the memory path for custom stores.

## Configuration and provenance

| Argument | Default | Phase |
| --- | --- | --- |
| `--hippomm-duplicate-threshold` | 0.9 | offline |
| `--hippomm-confidence-threshold` | 0.75 | online |
| `--hippomm-fallback-threshold` | 0.4 | online |
| `--hippomm-fallback-score` | 0.6 | online |
| `--top-k` | 5 | online |
| `--hippomm-max-new-tokens` | 1024 | both |
| `--hippomm-context-budget` | 120000 | both |

The local Qwen3-VL-8B-Instruct model serves summarization, gating, query rewriting,
caption selection and answering; generation is greedy with seed 42. All call roles,
prompts, raw outputs and token counts are retained under readable call IDs. There is no silent
input truncation. Query rewrites and caption selections get up to three formatting attempts.
A remaining per-question validation failure is recorded in `question_errors.jsonl`;
other questions continue, and analysis refuses final metrics until every question
has a valid answer. Context/output limits never cause silent truncation.

Retrieval prompts never include options, standard answers, clues, or supplied
query-image captions. For MemEye, a successful free-text fast answer is converted
to a letter in a separate final-formatting call; only that call sees the options.
The detailed final reader also sees options. Gold is opened by `analyze` only
(the MemEye source manifest contains mixed fields, but preparation whitelists them).

Offline artifacts: `memory.json`, `build_contract.json`, `events.jsonl`,
`visual_embeddings.npz`, `dialogues.jsonl`, `image_mapping.jsonl`, and audited LLM calls.
Online artifacts: `protocol.json`, sanitized `questions.jsonl`, `qa_responses.jsonl`,
LLM call caches, `per_question.jsonl`, `summary.json`, and `completion_manifest.json`.

Schema 2 uses short sequential event IDs (`E0001`, `E0002`, ...), image references
(`I0001`, ...), and call IDs (`call_000001`, ...). Dataset image labels such as
`D1:IMG_002` are preserved because they carry task meaning. No fingerprint fields
are sent in summary, fast retrieval, caption selection or final-answer prompts.

Code fingerprints live in one `experiment_manifest.json` beside `offline/` and
`online/` for each dataset run. Build/query settings remain in their configuration
files; memory records, answers, call caches and completion records do not carry
per-record signatures. Cache resume matches the complete request directly.
The shared input caches may already use hashes; the adapter consumes those for
source validation and translates them into short references before building memory.

Use a fresh experiment directory after code/settings changes. Schema 1 memories
and their generated summaries are not mixed with the short-ID experiment. The new
memory is rebuilt so earlier fingerprints cannot survive in generated summaries.
Validation checks IDs, counts, task boundaries, vector alignment, complete answer
coverage and referenced calls. It does not provide per-record cryptographic tamper
detection. Original image access is restricted to offline construction; query-time
validation remains possible after the history sources are unavailable.

Older methods that signed the entire shared `visual_memory_pipeline.py` may detect
the new CLI registration as a source-hash change when revalidating old artifacts.
Their hash checks and historical manifests have not been bypassed or rewritten;
validate those historical runs against their original source snapshot.

Source-round recall on the fast path measures the provenance of all supplied
summaries. It does **not** imply that each source fact survived summarization.
Reported storage separates numeric features and deployable memory artifacts;
inference audit logs are additional experiment storage.

## Verification

```bash
python3 -m unittest scripts.tests.test_hippomm \
  scripts.tests.test_caption_visual_pipeline \
  scripts.tests.test_caption_z_reweight_pipeline \
  scripts.tests.test_clip_image_memory \
  scripts.tests.test_clip_image_dialogue \
  scripts.tests.test_full_context_memory \
  scripts.tests.test_memeye_api_pipeline -q
```

Behavioral tests cover source preservation, task isolation, per-event fallback
before global ranking, gold-free prompts, malformed confidence, empty retrieval,
querying after history removal, duplicate/foreign outputs, centralized code checks, readable cache resume, and legacy CLI routing.

## Representative-originals paired experiment

`--memory hippomm_original` is an independent, paired reader ablation. It requires
a completed `hippomm` baseline (`--hippomm-baseline-root`, default
`results/hippomm_short_ids_20261104`). It freezes the baseline's memory, summaries,
routes, query rewrites, candidate scores and selected events. Fast-route answers
are copied unchanged. Only detailed final answers with nonempty retrieval are regenerated by the same
local Qwen3-VL-8B reader, now with the selected representative image files.
This measures the effect of original-pixel access without rerunning retrieval.

The offline phase copies exactly one original source file per retained event,
under short event filenames, with no recompression. It does not store rejected
images. Source dialogue IDs and captions are checked against the baseline before
copying. The current full stores retain 250 images for MemEye and 860 for
Mem-Gallery. Shared datasets and the no-original-image baseline are read-only.

```bash
python3 scripts/visual_memory_pipeline.py offline --memory hippomm_original \
  --dataset memeye --output results/hippomm_original_controlled_20261104/memeye/offline
python3 scripts/visual_memory_pipeline.py online --memory hippomm_original \
  --dataset memeye --device cuda:0 --stage all \
  --memory-dir results/hippomm_original_controlled_20261104/memeye/offline \
  --output results/hippomm_original_controlled_20261104/memeye/online
```

Use `--dataset mem_gallery` and matching output paths for Mem-Gallery. Full runs
reuse all fast answers and all empty-retrieval answers. Only detailed questions
with at least one selected representative are regenerated.
Question images come first in each payload, followed by at most five historical
representatives in the exact baseline ranking order. An explicit `image_order`
list binds each image to its event and original dataset image labels. Redundant
aliases are still approximate associations: a representative cannot establish
the discarded image's exact pixels. The reader uses the same Qwen image
preprocessing and generation settings as the baseline, with a 1,024-token output
limit; stored files retain original bytes, while inference may resize them.

`summary.json` reports paired baseline scores, percentage-point changes,
detailed-subset scores, unchanged source-round recall, image-file bytes and total
memory bytes. Total memory includes the original text/features, the representative
file index and the actual stored images; it excludes model weights and audit logs.
Analysis requires all questions, verifies frozen retrieval fields and unchanged
fast answers, and checks every detailed cache's exact prompt and image paths.

```bash
python3 -m unittest scripts.tests.test_hippomm_original scripts.tests.test_hippomm -q
```

## MemEye GPT-5 / DeepSeek full-backbone run

`--memory hippomm_api --hippomm-backbone gpt5|deepseek` selects an isolated API adapter.
It uses the same 250 CLIP-selected representative images and vectors as the completed
Qwen original-image experiment, but replaces captions with each provider's own verified
437-image caption cache and regenerates summaries, fast decisions, visual query rewrites,
caption selection and answers with that provider. Retrieval thresholds and Top-5 are
unchanged. This is a full-backbone evaluation, so Qwen retrieval/answers are not frozen.
Detailed nonempty answers receive ranked representative originals after question images.

The run supervisor is `reports/hippomm_api_memeye_20261104/run_all.py`. It passes
`--hippomm-max-new-tokens 2048`, seed 42 and four workers per provider. GPT requests
`gpt-5` and requires returned `gpt-5-2025-08-07`; DeepSeek uses the existing audited
DashScope international `deepseek-v4.1-flash` route (an alias, not a pinned weight revision).
The original three query-format attempts are followed, only on failure, by at most
three calls asking for three separate words, with raw outputs retained.

Completion requires all 371 unique questions and a provider/caption/summary/image-payload
audit. Results are under `results/hippomm_api_memeye_20261104/<backbone>/online/`.
Progress is in `reports/hippomm_api_memeye_20261104/RESULTS.txt`. The reported recall
is source-round micro coverage; the fast route sees all task summaries and is not Recall@5.
Code fingerprints live in the central experiment manifest; new model prompts and call
records use readable short IDs. Other pipeline methods and previous outputs are unchanged.

## MemGallery GPT-5 / DeepSeek extension

Use the same `--memory hippomm_api` selector with `--dataset mem_gallery`. An isolated
Gallery adapter preserves the completed MemEye implementation and its signed outputs.
It uses the Qwen experiment's 860 representative images and exact CLIP vectors, but
provider-specific caption caches (979 unique history images each), fresh summaries and
fresh retrieval/answers for all 1,711 questions. Native VS/CD/AR answer instructions
and the existing normalized EM, stemmed token F1, sentence BLEU-1/2/4 scorer are used.

Two retained images have explicit DeepSeek caption rejections in the existing source;
they retain empty captions, provenance and dialogue. No other backbone fills these
captions. New provider rejections are cached without resubmission or alternate routing.
Rejected questions retain empty answers, score zero, and are reported separately from
successful answers. A rejected summary is explicitly unavailable, never fabricated.

Images use original bytes with content-sniffed JPEG/PNG/WebP/static-GIF MIME types.
Only DeepSeek inputs above its Data URI size limit use JPEG encoding at unchanged
dimensions, starting at quality 95 and lowering quality only as needed. Every call
records encoding dimensions and byte sizes; stored originals and CLIP inputs stay intact.

Supervisor: `reports/hippomm_api_memgallery_20261104/run_all.py`; eight workers/provider,
2,048 output tokens, seed 42, CLIP on physical GPU 0. Progress and final metrics are in
that directory's `RESULTS.txt`; after both runs finish, an independent 1,711-ID,
fixed-visual-input and provider audit writes `final_verification.json`.

One retained file is a two-image MPO JPEG container. It is sent with JPEG MIME and
original bytes (primary image at index zero, as in the existing PIL/Qwen loader);
frame metadata is recorded. This is not an animated GIF/WebP input.
