# M²A on MemEye with local Qwen, BGE and SigLIP2

This integration executes the original M²A `ChatAgent`, `MemoryManager`, tools,
SQLite raw store, and Milvus semantic store. It is an explicit model/dataset
adaptation, not a reproduction of the original GPT-4o-mini/LoCoMo scores.

## Pinned sources and models

- Upstream: https://github.com/Little-Fridge/M2A, commit
  `edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8`.
- Untouched upstream files: `scripts/vendor_m2a/upstream/`.
  `UPSTREAM.json` records Git blob IDs and SHA-256 for all 25 files.
- Runnable copies: `scripts/vendor_m2a/runtime/`.
  `scripts/memory/m2a_vendor.py` regenerates these deterministically;
  `scripts/vendor_m2a/PATCHES.json` enumerates every source patch.
- LLM: local `models/Qwen3-VL-8B-Instruct`, greedy inference, seed 42.
- Text embeddings: cached `BAAI/bge-small-en-v1.5`, revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions, L2 normalized.
  Only queries receive `Represent this sentence for searching relevant passages: `.
- Visual embeddings: `google/siglip2-base-patch16-384`, revision
  `f775b65a79762255128c981547af89addcfe0f88`, 768 dimensions, L2 normalized.
  Downloaded to `models/siglip2-base-patch16-384/`; `DOWNLOAD_MANIFEST.json`
  records hashes of the complete local checkpoint and processor/tokenizer files.
- New dependencies are isolated under `.vendor_m2a/`.
  The installed dependency lock is `scripts/vendor_m2a/requirements-lock.txt`.

BGE-small replaces MiniLM without changing the 384-dimensional schema. BGE-base,
BGE-large or BGE-M3 would require changing the schema dimensions and rebuilding
the index. Old embeddings are never reused across encoder changes.

## Method retained

1. Chronological observations are stored in SQLite.
2. ChatAgent decides whether to issue memory update instructions.
3. MemoryManager searches, adds or deletes semantic memories using the original
   tool definitions and prompts. It can revisit original messages using evidence
   ID ranges or timestamps. Updates use delete + add.
4. Memory text and generated caption are indexed by BGE and Milvus BM25. Images
   selected into semantic memories receive SigLIP2 embeddings.
5. Text dense Top-5 and BM25 Top-5 fuse to five text candidates; image queries
   return three visual candidates; the final fusion uses reciprocal ranks with
   k=60. The original unused text-to-image request remains unused.
6. Both agents retain their iterative tool-calling workflows. MemoryManager has
   a 15-call budget, matching the upstream class constant; ChatAgent has five
   query/update calls per respective stage.

## MemEye adaptation and evaluation boundary

- All original dialogue rounds in the selected domain are ingested. A complete
  user/assistant round is one raw observation with source round/session/date.
- The original image pixels are provided. Dataset-provided image captions and
  existing cached captions are not injected; the memory agent generates its own
  semantic text and image descriptions with Qwen.
- Input data are projected from `multi_session_dialogues` only. Annotated QAs,
  answers, options and clues never enter construction.
- Questions and query images come from the frozen
  `data/independent_visual_inputs/qa_manifest.jsonl`; retrieval receives no
  options or gold labels. Options become visible only to a final local MCQ
  reader after the original ChatAgent query graph has completed.
- The final reader sees returned multimodal evidence and the memory agent's
  answer. This additional MCQ formatting stage is explicitly an adaptation.
- Evaluation invokes the read-only graph, avoiding upstream `chat()` appending
  test questions/answers to the raw database. Both storage layers are audited
  before/after QA.
- `--mandatory-query` explicitly executes the first `query_memory` call using
  only the question and optional image, then resumes the upstream ChatAgent
  graph. It enforces the original evaluator's MUST-query instruction for a local
  Qwen model that can otherwise ignore it. This control-flow adaptation must be
  reported separately from the unconstrained, prompt-only variant.
- Historical/query images use an explicit Qwen budget of 262144 pixels for this
  pilot. The question reader emits at most 16 new tokens; agent calls at most
  1200. BGE's 512-token prefix truncation is logged per input.
- Scores use one existing fixed option permutation. They are not four-rotation
  MemEye paper EM. Adaptive retrieval source recall is a union across searches,
  not a fixed Recall@5 or Recall@8 metric.

## Compatibility repairs

The patch manifest is authoritative. Repairs include the LangGraph END sentinel,
core tool/embedding imports, loop-budget enforcement, canonical image-token
paths, a non-searched constant vector required by Milvus's metadata collection,
the BM25 index metric, longer VARCHAR fields to avoid destructive truncation,
and returning success from deletion. These do not enable the unused text-to-image
route or replace the original search/fusion logic.

## Run and validate

Run from `/storage/hewenwen/VisualMemory` with host GPU access:

```bash
python3 -m unittest scripts.tests.test_m2a_local -v
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/evaluation/memeye_m2a.py \
  --task Card_Playlog_Test \
  --mandatory-query \
  --output results/m2a_qwen_bge_card_20261112/new_run
python3 scripts/evaluation/audit_memeye_m2a.py \
  results/m2a_qwen_bge_card_20261112/new_run
```

Always use a fresh output directory. The runner rejects an existing protocol.
`--limit-rounds` and `--limit-questions` are for smoke tests; outputs record whether
the entire task was covered. Do not compare a partial-memory smoke test to full
baselines.

To rerun only QA on an already completed, closed memory database:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/evaluation/memeye_m2a.py \
  --task Card_Playlog_Test --mandatory-query \
  --memory-source results/m2a_qwen_bge_card_20261112/run_v2 \
  --output results/m2a_qwen_bge_card_20261112/new_frozen_memory_qa
```

The source database is copied into the new result directory. Raw/semantic
snapshots and model settings must match the frozen source before evaluation.

Artifacts include protocol/source hashes, the construction manifest, semantic
snapshots, original-message snapshots, actual LLM prompts and tool calls,
embedding/truncation logs, per-search candidates, per-question predictions,
summary, audit and a completion manifest. A launched process or populated database
is not proof of completion. Check final question coverage, errors, evidence IDs
and frozen-memory checks before interpreting a score.

The local dependency bundle can be recreated with
`python3 -m pip install --target .vendor_m2a -r scripts/vendor_m2a/requirements-lock.txt`.
PyTorch, Transformers, sentence-transformers and the local Qwen/BGE checkpoints
are supplied by the existing project environment; exact package versions are
recorded in each run's protocol.

## Measured pilot (2026-11-12)

The completed pilot covers **Card_Playlog_Test: 30 original image/dialogue
rounds and all 48 questions in that task**, rather than the full 371-question
MemEye benchmark. Construction produced **51 semantic memories, all with
images**. The upstream update-only chat implementation also appends an empty
assistant response after each observation, so SQLite has 60 rows: 30 source
observations and 30 empty assistant rows. Test QA never enters that store.

| Query mode | Correct / questions | Accuracy | Questions without semantic search |
| --- | ---: | ---: | ---: |
| Prompt-only Qwen diagnostic (`run_v2`) | 5 / 48 | 10.42% | 42 |
| Explicit mandatory first query (`mandatory_query_v1`) | 20 / 48 | 41.67% | 0 |

The prompt-only run fails the mandatory-retrieval protocol audit and must not
be presented as a successful complete M²A retrieval evaluation. The mandatory
variant uses the **identical frozen raw and semantic memory**, with no answer
feedback or reconstruction. It changes initial query execution explicitly;
this is not an embedding ablation or a reproduction of the original paper's
model/dataset scores.

The mandatory variant passed the independent audit: exact 48-question coverage,
valid evidence IDs, identical source/reused semantic snapshots, unchanged raw
and semantic memories during QA, zero generation errors, zero generation token
limit hits, and zero MCQ format errors. Eight integration/regression tests passed.
Construction used 863 Qwen calls; the mandatory QA run used 256, for 1,119 calls
in the effective construction + QA workflow. Mandatory QA took 237.04 seconds
including model loading on this shared host; this is not a controlled latency
benchmark.

All 48 Card questions are text-only. The separate real SigLIP2/upstream
`hybrid_search` image probe returned pixel-identical content at Top-1 for 3/3
queries. Filename-exact Top-1 was 2/3 because the dataset includes different
filenames containing identical pixels. This is an interface test, not held-out
image-query QA or retrieval accuracy.

Results:

- `results/m2a_qwen_bge_card_20261112/comparison.json`
- `results/m2a_qwen_bge_card_20261112/mandatory_query_v1/summary.json`
- `results/m2a_qwen_bge_card_20261112/mandatory_query_v1/independent_audit.json`
- `results/m2a_qwen_bge_card_20261112/image_smoke_v2/pixel_equivalence_audit.json`

No MiniLM-versus-BGE performance comparison was run. BGE-small's compatibility
and the full adapted workflow were verified; its superiority over MiniLM was
not tested.

## Full MemEye / MemGallery queues (2026-11-12)

The shared runner now accepts `--dataset memeye|memgallery`. The full queues
cover 848 rounds / 371 questions in 8 MemEye tasks, and 3,962 rounds / 1,711
questions in 20 MemGallery scenarios. These are target coverage counts, not
claims that the running experiment is complete.

MemGallery construction reads original dialogue and original pixels. Official
historical image IDs are retained for visual-search answers. Neither historical
nor question-image supplied captions are injected. Its reader uses the existing
official Gallery system/task instructions with 256 output tokens, followed by
the existing normalized EM, stemmed token F1 and BLEU scorer. MemEye retains the
pilot's fixed-permutation MCQ evaluation. No LLM judge is part of this queue.
Both use seed 42, greedy Qwen3-VL-8B, BGE-small, SigLIP2, 262144 image pixels,
1200 agent output tokens and the explicit mandatory first memory query.

Run root: `results/m2a_qwen_full_20261112/`. `plan.json` freezes task coverage,
source hashes, settings and implementation hashes. The Card task reuses the
pilot's closed memory database and reruns its questions. All other scenarios
build isolated memories from their entire history. A four-round / one-image-QA
Gallery smoke test is stored separately and is excluded from all full scores.

```bash
python3 scripts/evaluation/m2a_suite.py prepare --output results/m2a_qwen_full_20261112
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/evaluation/m2a_suite.py run \
  --dataset memeye --output results/m2a_qwen_full_20261112
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/evaluation/m2a_suite.py run \
  --dataset memgallery --output results/m2a_qwen_full_20261112
```

The launch records identify the actual background PIDs and GPU assignments.
Each dataset's `status.json` records its active task and process. The active
attempt's `progress.json` records completed construction rounds or questions;
its live call log and task log expose intermediate progress. The queue only
counts independently audited, complete tasks. Failed attempts remain on disk
and the queue continues other scenarios, ending `completed_with_failures` if
any scenario failed. Reinvoking a stopped queue audits and skips accepted tasks,
then starts a fresh attempt for unfinished tasks; it does not resume a partly
built database. An advisory lock prevents duplicate workers for one dataset.

A dataset-level `completion_manifest.json` is emitted only after every task
passes its audit and question-count checks. It includes weighted aggregate
metrics and links to task completion manifests. Until that exists, inspect
partial results as partial coverage only. Do not edit pinned runner files while
the queue is active: it deliberately stops if their hashes change.

## Evidence-ID compatibility fix (2026-11-13)

Use `scripts/evaluation/m2a_evidence_run.py` for new runs with evidence-ID
validation; it accepts the same arguments as `memeye_m2a.py`. The existing
2026-11-12 queue remains pinned to its original implementation. This opt-in
entry point wraps the native tool/store methods within the new process and
records its code hashes and `evidence_policy` in the signed protocol.

- A two-element group remains an inclusive range: `[[99,103]]` includes all
  five message IDs.
- A singleton becomes a single-point range: `[[83]]` becomes `[[83,83]]`.
- A group of three or more integers is explicitly treated as discrete IDs:
  `[[99,101,103]]` becomes `[[99,99],[101,101],[103,103]]`, never `[99,103]`.
- Invalid JSON, empty groups, strings/floats/booleans, non-positive IDs, reversed
  ranges and references to nonexistent raw messages are rejected by the tool
  before writing a memory. The existing agent loop receives an explanatory
  error and can correct its arguments. Empty top-level evidence lists retain
  the upstream behavior.
- Raw-message fetching applies the same policy. Legacy semantic-memory reads
  normalize the returned object without rewriting the database. This does not
  make an incomplete old run reusable: normal completion and frozen-snapshot
  checks still apply to `--memory-source`.
- `evidence_validation.jsonl` records each normalization and rejection with its
  original value. A completed run includes that log in its artifact manifest.

```bash
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/evaluation/m2a_evidence_run.py \
  --dataset memeye --task Social_Chat_Memory_Test --mandatory-query \
  --output results/m2a_evidence_fix_20261113/social_new_run
python3 scripts/evaluation/audit_memeye_m2a.py \
  results/m2a_evidence_fix_20261113/social_new_run
```

Validation: 22 unit/integration regressions passed. Read-only replay over eight
saved pre-QA snapshots covered 1,000 memories, normalized 42 malformed evidence
lists, and found zero unknown IDs/rejected records. Every replayed normalized
record passed the previously failing range-expansion operation. Source hashes
were checked unchanged. See `results/m2a_evidence_fix_20261113/replay.json` and
`tests.log`. Replay is not a QA rerun or a factuality check; token/context limits
and malformed tool-call JSON are separate failures unaffected by this fix.

The new entry point also completed a real one-question Qwen smoke run using
the frozen Card memory (`qwen_smoke/`): five model calls, zero tool/format/token
errors, unchanged memory, and a passing independent audit. Its answer was
incorrect (0/1); this is an interface regression check, not an accuracy claim.

## Output-length prompt (2026-11-13)

The repaired `m2a_evidence_run.py` entry point now also records and applies
`output_length_policy` from `scripts/memory/m2a_length_prompt.py`:

- Agent calls: at most 200 English words or 300 Chinese characters, retain
  essential facts/IDs, avoid repetition, and close tool JSON/tags completely.
- Open-answer reader: at most 80 English words or 120 Chinese characters,
  retaining requested facts/image IDs and required answer formatting.
- MCQ reader: exactly one uppercase A/B/C/D, no explanation/punctuation; choose
  the best-supported option when uncertain.

The instruction is appended to a copied system message, or prepended as a
system message if none remains in the agent context. Shared conversation
history, tool results and images are not mutated. Actual model logs include the
injected instruction. Generation limits remain 1200/256/16 tokens respectively;
there is no post-generation truncation. This is a prompt constraint, not a hard
guarantee, and cannot resolve input-context overflow by itself. The running old
full queue still uses its frozen original prompts.

`scripts/evaluation/replay_m2a_length_prompt.py` restores three saved failed
requests (Outdoor reader, Health memory agent, Gallery Fashion agent), verifies
the original message fields and image bytes, and adds only this instruction.
It does not execute generated tools or alter any memory database. Outputs go
to a fresh directory under `results/m2a_length_prompt_20261113/` and do not count
as completed benchmark questions.

Measured replay: all three requests completed without token-limit/parse errors.
Output tokens changed from 16/1200/1200 to 2/2/116. The reader returned `A`, the
Health agent returned literal `""` without a tool call, and the Fashion agent
returned a short prose response without a tool call. This verifies avoidance
of the observed generation failures, not successful memory updates or improved
QA accuracy. Existing 22 regression tests also passed.

## Repaired full rerun (2026-11-13)

The user requested another run with both fixes. The old Gallery queue was
stopped, with its previous status and termination record retained in
`results/m2a_qwen_repaired_20261113/superseded_old_queue.json`; its signed
protocols, memories and response logs are preserved. The new root has fresh
memories for every task, including Card, because construction prompts changed.

```bash
python3 scripts/evaluation/m2a_suite.py prepare \
  --output results/m2a_qwen_repaired_20261113 --runner repaired \
  --smoke-first --stop-on-failure
CUDA_VISIBLE_DEVICES=1 python3 -u scripts/evaluation/m2a_suite.py run \
  --output results/m2a_qwen_repaired_20261113 --dataset memeye
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/evaluation/m2a_suite.py run \
  --output results/m2a_qwen_repaired_20261113 --dataset memgallery
```

Each dataset first constructs four real history rounds and answers one question
with an original query image, then runs the independent audit and checks that
semantic memory is nonempty. These smoke results are excluded from full scores.
MemEye starts its full run with Social Chat (53 rounds / 37 questions); Gallery
starts with Fashion (168 rounds / 61 questions), targeting the earlier evidence
and token-limit failures. If a smoke test, full task or audit fails, that dataset
queue stops instead of proceeding through all remaining scenarios. Its sibling
queue can continue. Status files now refresh every five seconds with the active
task's progress. Full coverage remains 8 tasks / 371 questions and 20 scenarios /
1,711 questions respectively; only dataset completion manifests establish that
coverage was actually achieved.
