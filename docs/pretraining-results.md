---
tags:
  - qwen3
  - pretraining
  - results
  - longleaf
created: 2026-06-19
updated: 2026-06-19
status: complete
related:
  - project-overview.md
  - data-pipeline.md
  - case-study-outline.md
---

# RQwen3 — Pretraining Results

## Headline

On 2026-06-19, **RQwen3 finished pretraining**. A 751M-parameter from-scratch Qwen3-0.6B clone reached **step 50,000 / 50,000** with final training loss **2.5186** (perplexity ≈ 12.4), having seen ~13 billion tokens of the curated 6-source corpus — roughly Chinchilla-optimal for the parameter count. The full run took **10 successful 24-hour SLURM submissions across 11 wall-clock days** on UNC Longleaf's `l40-gpu` partition (NVIDIA L40S 48 GB), preceded by **3 OOM crashes and 2 user-cancels on day 1** while the memory wall was being worked out. The base model is saved as `checkpoints/final.pt` and is the artifact downstream tasks (eval, SFT, ablations) will build on.

This doc is the canonical record of how that run unfolded: what the loss did, what the model was actually saying at each milestone, what broke and how it was fixed, and what we'd do differently next time.

---

## Final numbers

| Category | Value |
|---|---|
| **Model** | RQwen3 (751M params, Qwen3-0.6B clone) |
| `d_model × n_layer` | 1024 × 28 |
| `num_heads / num_kv_heads` | 16 / 8 (GQA 2:1) |
| `vocab_size` | 151,936 (Qwen3 BPE) |
| `max_seq_len` | 2,048 |
| **Training config** | |
| Dataset | 6-source curated mix (~13 B tokens) — see [data-pipeline](data-pipeline.md) |
| Effective batch | 128 (`batch_size=2 × grad_accum=64`) |
| Tokens per step | ~262,144 |
| Precision | bf16 autocast (CUDA); SDPA attention (FlashAttention dispatch) |
| LR schedule | Cosine, peak `3e-4` → min `3e-5`, 500-step warmup |
| Optimizer | AdamW (`wd=0.1`, `grad_clip=1.0`) |
| **Result** | |
| Steps completed | **50,000 / 50,000** |
| Final loss | **2.5186** |
| Final perplexity | **≈ 12.4** |
| Starting loss (random init) | 11.88 |
| Total loss drop | **9.36 nats** |
| Wall-clock | 2026-06-08 → 2026-06-19 (11 days) |
| Successful SLURM submissions | 10 |
| Pre-flight crashes / cancels | 3 OOM + 2 cancels (day 1) |
| Final LR | `0.0e+00` (cosine fully completed) |
| Checkpoints saved | 51 total (~429 GB on `/work`) |
| Final artifact | `checkpoints/final.pt` (8.5 GB) |

---

## The training journey

### Phase 0 — The memory wall (2026-06-08)

Three OOM crashes before training started, all on the same L40S, all on the same physical 44 GiB of usable VRAM.

| Job | Wall time | Failure mode |
|---|---|---|
| `53873931` | 1m 06s | OOM in **forward** at `(q @ k.T) * scale` — 1 GiB alloc, 44.10 GiB in use |
| `53874107` | 1m 13s | OOM in **backward** — 4.64 GiB alloc, 41.32 GiB in use |
| `53874149` | 1m 30s | OOM in **backward**, autocast wasn't enough — 4.64 GiB alloc, 40.97 GiB in use |

The smoke test had been green, but it ran at `seq_len=256` and so never touched the production-scale memory profile. Three fixes stacked, in order:

1. **`src/layers/attention.py` — swap naive attention for `F.scaled_dot_product_attention`.** The manual `(q @ k.T) * scale` materializes the full `(B, H, T, T)` matrix in VRAM — 1 GiB per layer at `B=4, H=16, T=2048` in fp32, ×28 layers stacked in the backward graph. SDPA dispatches to FlashAttention on Ada Lovelace GPUs (L40S), which never materializes that matrix. Memory drops from O(T²) to O(T).
2. **`scripts/py/TrainSession.py` — wrap forward + loss in `torch.autocast(dtype=torch.bfloat16)`.** Halves activation memory; lets the L40S's BF16 tensor cores do real work. No GradScaler needed (bf16 has fp32-equivalent dynamic range).
3. **`longleaf/scripts/py/pretrain.py` — halve the micro-batch.** Even with SDPA + bf16, the 151,936-vocab cross-entropy keeps its softmax intermediates in fp32 for numerical stability, and that's ~5 GiB per micro-batch at `B=4`. Drop to `batch_size=2`, bump `grad_accum_steps=32 → 64`, keep effective batch at 128.

Plus a visibility fix that came up immediately: Python buffers stdout when redirected to a file, so the SLURM log was empty for the first few minutes even though GPU utilization was at 98%. Added `export PYTHONUNBUFFERED=1` to `longleaf/pretrain-l40.slurm` so per-step log lines flush immediately. Without this, the first visible loss value wouldn't have appeared until the step-1000 checkpoint flush — a useless feedback loop.

After these four fixes, the first real submission (`53876531`) ran cleanly for 24 hours.

### Phase 1 — Loss collapse (Sub 1 · step 0 → 5,000)

The big drop. Loss starts at **11.88** (essentially random — `ln(152K)` for uniform-over-vocab is ~11.93) and falls to **2.97** in a single 24-hour submission. This is the easy part — the model is just learning word frequencies and basic grammar, the "anyone-can-do-this" gains.

```
step    10/50000 | loss: 11.8807 | lr: 6.00e-06
step   100/50000 | loss:  8.6865 | lr: 6.00e-05
step   500/50000 | loss:  5.3926 | lr: 3.00e-04  ← warmup ends, peak LR
step  5000/50000 | loss:  2.9700 | lr: 2.91e-04
```

### Phase 2 — The middle (Subs 2-7 · step 5K → 35K)

Six submissions, six days. Diminishing returns become the dominant theme: the per-submission drop shrinks from 0.19 (sub 2) to 0.02 (sub 6). Loss-vs-step looks logarithmic. Mid-phase plateaus around step 10K and step 30K both broke within ~1,000 steps.

| Sub | End step | End loss | Δ | Notes |
|---|---|---|---|---|
| 2 | 10,000 | 2.78 | -0.19 | First sample with grammatical multi-sentence prose |
| 3 | 15,000 | 2.69 | -0.09 | Cosine decay enters meaningful territory |
| 4 | 20,000 | 2.68 | -0.03 | First "plateau" — looked stuck mid-submission |
| 5 | 25,000 | 2.60 | -0.08 | Plateau broke, then steady drop |
| 6 | 30,000 | 2.58 | -0.02 | Smallest drop of the run so far |
| 7 | 35,000 | 2.53 | -0.05 | First numerical claims in samples (units correct, value wrong) |

### Phase 3 — The cosine tail (Subs 8-10 · step 35K → 50K)

The end-game. LR is below `1e-4` and decaying toward `min_lr=3e-5`, then below that, and finally to exactly `0` at step 50,000. Loss flattens in the **2.50-2.55** band and stays there — this is the model polishing what it already knows, not learning new things.

Sub 8 also surfaced the only real scheduling surprise of the run: a **18.5-hour queue wait** before it actually started, breaking the "submit at noon → resume tomorrow at noon" cadence we'd settled into. (Saved into memory as a feedback rule: always verify with `squeue`/`sacct` before resubmitting.)

| Sub | End step | End loss | Δ | Notes |
|---|---|---|---|---|
| 8 | 40,000 | 2.54 | +0.01 | Cosine flattening + queue-wait anomaly |
| 9 | 45,000 | 2.50 | -0.04 | First sub-2.5 readings |
| 10 | **50,000** | **2.5186** | +0.02 | `COMPLETED 0:0` — natural exit, not SIGTERM |

Sub 10 ended `COMPLETED 0:0` instead of the `FAILED 0:15` (SIGTERM) the prior nine submissions exited with — the script hit `max_steps` and exited cleanly, writing both `checkpoints/step_50000.pt` and `checkpoints/final.pt`.

---

## Loss trajectory (all 10 submissions)

| Sub | SLURM Job | End step | End loss | Δ from prev | Date ended |
|---|---|---|---|---|---|
| 1 | `53876531` | 5,000 | 2.97 | — | 2026-06-09 |
| 2 | `54045709` | 10,000 | 2.78 | -0.19 | 2026-06-10 |
| 3 | `54161733` | 15,000 | 2.69 | -0.09 | 2026-06-11 |
| 4 | `54287786` | 20,000 | 2.68 | -0.03 | 2026-06-12 |
| 5 | `54543832` | 25,000 | 2.60 | -0.08 | 2026-06-13 |
| 6 | `54615718` | 30,000 | 2.58 | -0.02 | 2026-06-14 |
| 7 | `54715359` | 35,000 | 2.53 | -0.05 | 2026-06-15 |
| 8 | `54802040` | 40,000 | 2.54 | +0.01 | 2026-06-17 |
| 9 | `55153663` | 45,000 | 2.50 | -0.04 | 2026-06-18 |
| **10** | **`55368282`** | **50,000** | **2.5186** | +0.02 | **2026-06-19** |

Plus jobs `53873931`, `53874107`, `53874149` (OOM crashes, 1 min each) and `53874226`, `53874294` (user-cancels during the memory-wall fix and `max_steps=10K → 50K` swap). 15 SLURM job IDs total to produce one base model.

---

## Sample evolution

All samples are responses to the prompt **"The theory of general relativity"** (the prompt set in `pretrain.py`'s `sample_prompt`), captured at the `sample_every=500` print line that fires on each checkpoint boundary.

**Step 500** — warmup just ended, peak LR. Mode-collapse on a single phrase:

> The theory of general relativity, and the other of the most important types of the world. The most important, the most important, the most important, the most important, and the most important, the mo...

**Step 6,000** — model has learned grammar but is still latching onto one token at a time:

> The theory of general relativity is based on the idea that the universe is a continuous, continuous, and continuous, continuous, and continuous, continuous, and continuous, continuous, and continuous,

**Step 10,000** — first coherent multi-sentence prose. Facts are confidently wrong (Einstein, not Planck; 1915, not 1905; that formula is for gravity, not light):

> The theory of general relativity was developed by the German physicist Max Planck in 1905. The theory was based on the idea that the speed of light is inversely proportional to the square of the dista...

**Step 25,000** — reasoning across sentence boundaries appears. Mode-collapse weakens. Still factually mixed:

> The theory of general relativity is that the universe is expanding. The expansion of the universe is caused by the gravitational attraction of the stars and the planets. The expansion of the universe...

**Step 40,000** — qualitatively the biggest jump of the second half: **actually correct cosmology.** "Caused by the expansion of space itself" is real GR — metric expansion of space, not motion through space:

> The theory of general relativity is that the universe is expanding, and that the expansion is accelerating. The expansion is caused by the expansion of space itself. The expansion of space is accelera...

**Step 50,000** — final. Loss number plateaued for the last 10K steps, but the prose continues to consolidate:

> The theory of general relativity is that the universe is expanding, and that the expansion is accelerating. The theory of general relativity is a theory of gravity, and it is based on the idea that th...

What the samples don't show: the loss number stopped moving long before the prose stopped improving. That's a useful lesson — the scalar loss is a coarse summary of what the model is actually learning.

---

## What the final loss means

Final training loss **2.5186** → perplexity **≈ exp(2.5186) ≈ 12.4**. Reference points for ~750M dense models trained from scratch:

| Reference model | Approx. final loss | Notes |
|---|---|---|
| GPT-2 large (774M) | ~2.85 | WebText, 2019 |
| Pythia-410M | ~2.85 | The Pile, 300B tokens |
| **RQwen3 (this run)** | **2.5186** | 13B tokens, ~1 epoch |
| Pythia-1B | ~2.60 | The Pile, 300B tokens |
| Qwen3-0.6B base | ~2.4 | Trained on **~5T tokens** — 400× more data |

Landing **at Pythia-1B's neighborhood** despite being smaller (751M vs 1B) and having seen vastly less data (13B vs 300B tokens) is a clean result for the parameter budget and compute. Below 2.4 would be exceptional at this scale; above 3.0 would suggest something went wrong.

**Important caveat:** base-model loss is **not** user-facing quality. This is a next-token predictor, not an assistant. It will continue grammatical-but-mode-collapsing prompts, fabricate facts, and not follow instructions. Making it useful as an "educational assistant" (the stated project goal) requires the SFT chapter — that's notebook 06 and beyond.

---

## Bugs encountered during training

Beyond the [Seven Ways SLURM Said No](project-overview.md#cluster-setup-journey) cluster-bring-up bugs documented earlier, the dataset-build and pretraining phases surfaced their own:

### Smoke-test typo only caught on CUDA

`longleaf/scripts/py/smoke_pretrain.py:39` used `torch.cuda.get_device_properties(0).total_mem` instead of `total_memory`. The `if device.type == "cuda":` branch is skipped under MPS, so local Mac smoke tests on `MPS` never executed the broken line. Both initial cluster submissions (`38784158` and `38785256`) hit it and died in **1 second** with `AttributeError`. The fix was a one-character edit; the lesson is that branches gated on device type need exercising on every supported device.

### `peS2o` no longer loads in `datasets >= 5.0`

`allenai/peS2o` v2 ships only as a Python loader script (`peS2o.py`). The new `datasets` library refuses to execute it: `RuntimeError: Dataset scripts are no longer supported`. Swapped to `MaLA-LM/peS2o-final` — parquet-native re-export of the same corpus, ungated, with text in the `content` field. Drop-in replacement in `scripts/build_dataset.py`.

### `nampdn-ai/tiny-textbooks` is gated

Trying to stream it raised `DatasetNotFoundError: ... is a gated dataset on the Hub`. Swapped to `HuggingFaceTB/cosmopedia` (`stanford` config) — Phi-1.5-style synthetic textbook corpus, apache-2.0, ungated, parquet. Different content character than `tiny-textbooks` (broader, less STEM-focused than the agent-search hinted) but fits the role of "textbook-quality structured explanations" well enough for this budget.

### StackExchange schema drift

`scripts/build_dataset.py`'s `_stackexchange_extract` looked for `example['chosen']` and `example['question']`. The actual `HuggingFaceH4/stack-exchange-preferences` schema is `{qid, question, answers: [{answer_id, pm_score, selected, text, ...}], date, metadata}`. The old code returned `None` for every example → first resume run downloaded **10.8 million docs and kept 0** after an hour of compute. Fixed to pull the asker-selected answer, falling back to highest `pm_score`. Re-run kept 2.4 M of 2.4 M docs.

### `trust_remote_code` argument removed in `datasets 5.0`

`load_dataset(..., trust_remote_code=True)` now errors with `'trust_remote_code' is not supported anymore`. Dropped the kwarg. All current sources are parquet-native and don't need it.

### Transient HF Hub 503

Sub 3 of the dataset build (`53822136`) crashed mid-stream when `huggingface.co` returned `503 Service Unavailable` for a specific StackExchange parquet. The HF client retries on 5xx, but the underlying error propagated. Re-running fully recovered. Not really a bug — just the cost of streaming from a third-party CDN.

### stdout buffering hid early progress

First submission ran for **8 minutes** with no log output even though GPU utilization was 98% — Python defaults to block-buffering stdout when it's redirected to a file. Added `export PYTHONUNBUFFERED=1` to `longleaf/pretrain-l40.slurm`. With unbuffered output, step-10 logs flush immediately and you know within minutes whether the run is healthy.

### Mac disk full mid-smoke

Local Mac smoke saving an 8.5 GB checkpoint into `checkpoints/` (a symlink to iCloud) failed with `RuntimeError: ios_base::clear` — the disk was at 99 % capacity and iCloud sync was simultaneously trying to upload the partial file. Pointing the smoke checkpoint dir at a non-iCloud path (`checkpoints_local/`) and freeing ~40 GB of HF cache fixed it. Tracked in memory as a project-storage rule.

---

## What we'd do differently

Self-critique, in roughly the order of compute saved:

1. **Smoke at production scale.** The current smoke (`smoke-pretrain-l40.slurm`) uses `seq_len=256, batch_size=2` and finishes in 90 seconds. That's enough to catch typos and venv issues, but **does not** exercise the actual memory peak. A second "load-test" smoke at `seq_len=2048, batch_size=4` (10 steps, 5 min) would have caught the memory wall before the first real submission burned a queue slot.
2. **Default to bf16 + SDPA from day one.** Both are unambiguously the modern standard for transformer training on Ampere/Ada GPUs. Starting in fp32 + naive attention was archaic, and discovering it the hard way cost three job submissions and an evening.
3. **Use a real logger, not `print()`.** Python's default stdout buffering combined with raw `print()` produced the "is it stuck or just buffering?" episode. A logger with file rotation (or even tqdm with `flush=True`) would have made progress visible regardless of buffering, and added timestamps + levels for free.
4. **Checkpoint pruning from the start.** Saving every 1,000 steps for 50,000 steps = 50 checkpoints at 8.5 GB = **425 GB on `/work`**. Keeping `step_1000`, every 5K (`step_5000`, `step_10000`, ...), and the latest 3 would be 18 checkpoints (~153 GB) — same recovery surface, 64 % less storage.

---

## What's next

Concrete follow-ups for the post-pretraining chapter:

- [ ] Pull `final.pt` locally (`make pull` filtered to just `final.pt`) and generate a wider variety of inspection samples
- [ ] Wire up [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and run ARC-easy, HellaSwag, MMLU — get hard numbers to compare to the Pythia/GPT-2 references above
- [ ] Finish [notebooks/06-supervised-finetuning](../notebooks/) — SFT on top of `final.pt` is the bridge from "next-token predictor" to "useful assistant"
- [ ] Optional ablations on top of the pretrained base (score-threshold sweep, source-mix sweep) — cheap with the pipeline now in place
- [ ] Optional: MoE routing exploration in larger Qwen3 variants (Qwen3-30B-A3B)
- [ ] Run the checkpoint-pruning script (see "What we'd do differently" #4) to reclaim ~270 GB of `/work` quota
- [ ] Publish the case-study post — spine in [case-study-outline](case-study-outline.md). This doc is the source of truth for Act 6.

---

## Related

- [project-overview](project-overview.md) — Architecture, full project status, "Seven Ways SLURM Said No" bug log
- [data-pipeline](data-pipeline.md) — The 6-source curated corpus this run consumed
- [case-study-outline](case-study-outline.md) — Act 6 narrative spine; this doc is its data source
