---
tags:
  - qwen3
  - case-study
  - story
  - outline
created: 2026-06-12
status: drafting
supersedes-spine-from: case-study-plan.md
---

# RQwen3 Case Study — Story Outline

## What this doc is

The **structural outline** for the case study post — the 6-act spine, what each act argues, what backs it, what figures and captures it needs.

This pairs with [docs/case-study-plan.md](case-study-plan.md), which owns the orthogonal strategy decisions: audience layering (spine + linked explainers), hosting (own site, not Medium), the hero interactive scope, and the recruiter/grad-school/social variants. **That doc is not superseded** — only its 5-act spine is replaced by the 6-act spine below.

## Where this came from (reconciliation note)

This outline is the merge of two earlier passes:

1. **Story jot** (a `Jot 0.md` brain-dump): led with concepts → data → research notes → training system → cluster → "three building processes." Its strongest instincts were the data-first emphasis, the explicit "training system" section, the personal-motivation paragraph, and the Longleaf-as-UNC-resource framing.
2. **Earlier plan** (case-study-plan.md, 2026-06-08): 5 acts — Why → Build → Data → Scale → Results. Its strongest instincts were the beginner on-ramp, architecture-as-proof via loaded Qwen3-1.7B weights, the "Seven Ways SLURM Said No" bug log, the contingency table, and the "what I'd do differently" close.

**Keeps from the jot:** the explicit *Training System* act, the Longleaf-as-resource framing, the personal motivation paragraph, "trash in = trash out" data framing.

**Keeps from the earlier plan:** architecture as Act 2 (the proof-of-correctness hook earns the reader's trust before data filtering or cluster bugs can land), Results as its own act, the 7-bug story, contingency-aware results, "what I'd do differently" close.

**Drops from the jot:** the "three main processes of building" lump at the end — architecture, pretraining, and post-training are not peers and shouldn't share a section.

---

## Title & hook (one block, opens the post — not its own act)

**Working title:** *I Rebuilt Qwen3 From Scratch and Pretrained It on a University Supercomputer*

**One-sentence pitch:** Reconstructed Qwen3-0.6B's architecture component-by-component (751M params), built a 13B-token curated data pipeline, and pretrained it on UNC's Longleaf A100 cluster.

**Numbers in the header:** 751M params · 13B tokens · 28 layers · A100s. (Concrete numbers are what recruiters skim for.)

**Preempt the 751M vs. nominal 0.6B gap immediately** — a sharp reader will read it as an error. One sentence: ~156M of the 751M is just the 151,936-vocab × 1024-dim embedding table; at this scale embeddings dominate parameter count. Flips "did they miscount?" into "they know where params live in small models."

---

## Act 1 — Why (concepts on-ramp)

**What it argues:** Using models is not understanding them. To understand, I rebuilt one.

**Beats:**
- **Rules vs. patterns.** Two ways to make a computer do a task. Programming = you write every rule explicitly. ML = you show examples and the patterns are learned. This is the load-bearing frame: it's *why* data quality matters, *why* training is a loop, *why* a falling loss curve means anything.
- **Next-token prediction.** The specific pattern this machine learns: a probability distribution over ~152K possible next tokens, training nudges the right one up. This single idea explains the vocab size, the loss function, the data obsession, and what the loss curve represents.
- **Personal motivation paragraph** (from the jot): I don't want to *use* this stuff, I want to *understand it at a low level*. I value the knowledge for its own sake — the case study is the artifact of that value.

**Discipline:** This act is the gloss-in-spine version. Deeper beginner explainers (rules-vs-patterns, next-token, neural network, training loop) live on linked pages per [docs/case-study-plan.md](case-study-plan.md). Keep total linked terms across the whole spine to ~6–10.

**Backing:**
- [notebooks/01-qwen3-overview](../notebooks/) — model family survey
- `notebooks/Basics AI.ipynb` — foundational concepts (exploratory, may quarantine before publish)

**Figures:** none required — this act is text-only on the spine. The hero interactive (next-token picker) sits on the linked explainer.

---

## Act 2 — Build (the architecture, told as proof-of-correctness)

**What it argues:** I rebuilt every layer of the Qwen3 block from scratch. The proof it's actually Qwen3: I loaded the real Qwen3-1.7B weights into my implementation and generated coherent text.

**Beats:**
- **The GPT-2 → Qwen3 component table.** Reframe the third column from "Why It's Better" to **"What It Trades / Why It's Used"** — unhedged "better" reads as reciting blog-post consensus; framing each as an engineering choice with a cost (GQA trades a little quality for big KV-cache savings, etc.) is the most research-flavored signal in the table.
- **One paragraph per component:** RMSNorm, RoPE, GQA (16 query heads ÷ 8 KV heads, 2:1), SwiGLU, QK-Norm, no biases.
- **The proof moment.** Notebook 02 loads real Qwen3-1.7B pretrained weights into the from-scratch implementation and generates coherent text. **Lead with this.** It's the single moment that earns the reader's trust before the data and cluster sections ask them to care.
- **Architecture summary table:** d_model 1024, n_layer 28, num_heads 16, num_kv_heads 8, head_dim 128, intermediate_size 3072, vocab 151,936, max_seq_len 2048 → ~751M params.

**Backing files:**
- [src/config.py](../src/config.py) — `CoreConfig` dataclass
- [src/models/rqwen3.py](../src/models/rqwen3.py) — `RQwen3Transformer` → `LMHeadRQwen3` → `RQwen3`
- [src/layers/normalization.py](../src/layers/normalization.py) — `RMSNorm`
- [src/layers/embeddings.py](../src/layers/embeddings.py) — `RotaryEmbedding`, `apply_rotary_pos_emb`
- [src/layers/attention.py](../src/layers/attention.py) — `GQAttention` (with QK-norm, `store_attn` for viz)
- [src/layers/activations.py](../src/layers/activations.py) — `SwiGLU`
- [src/layers/feedforward.py](../src/layers/feedforward.py)
- [src/layers/block.py](../src/layers/block.py) — `TransformerBlock`
- [notebooks/](../notebooks/) — 02 (weight-load proof), 03 (attention heatmaps), 04 (untrained-vs-trained)

**Figures:**
- Architecture diagram — RQwen3 block structure
- GQA head layout (16 Q, 8 KV, grouping)
- Untrained-vs-trained weight distribution histograms (from notebook 04)
- Generation sample: real Qwen3-1.7B weights loaded into my implementation, prompted, output text

---

## Act 3 — Data (trash in = trash out)

**What it argues:** Pretraining is mostly a data problem. The model goal — an educational STEM assistant — drives every decision in the mix.

**Beats:**
- **The framing line:** trash in = trash out. Why "good data" beats "more data" at this scale.
- **The mix (~13B tokens, 6 sources):** FineWeb-Edu 54%, Wikipedia 15%, OpenWebMath 12%, StackExchange 8%, peS2o 8%, textbooks 4%. One line each on *why this source for this model*.
- **What "good" means here:** quality filtering (score thresholds, length bounds, source-specific rules), exact-hash dedup within each source, deterministic train/val split.
- **Engineering for resumable scale:** pre-tokenized into binary `uint32` memmap shards, O(1) random access, data-offset checkpointing so training resumes at the exact next token after a SIGTERM.
- **The drive-to-understand paragraph** (from the jot) lands here — data is where "I want to understand it at a low level" pays for itself in concrete decisions (which filter? which threshold? why?).
- **What's new (June):** [scripts/stitch_manifest.py](../scripts/stitch_manifest.py) joins shards produced by multi-job dataset builds — added when one build job couldn't finish in the cluster's wall-time limit.

**Backing:**
- [src/data.py](../src/data.py) — `StreamingTokenDataset` (prototyping), `PreTokenizedDataset` (production)
- [scripts/build_dataset.py](../scripts/build_dataset.py) — main builder
- [scripts/stitch_manifest.py](../scripts/stitch_manifest.py) — multi-job manifest stitcher
- [docs/data-pipeline.md](data-pipeline.md) — full spec

**Figures:**
- Source-mix bar chart (or pie, but bar reads better at thumbnail size)
- Pipeline flow diagram: raw HF source → filter → dedup → tokenize → shard
- Optional: shard-format diagram (header + uint32 stream + offsets)

---

## Act 4 — The Training System (CoreConfig + production TrainSession)

**What it argues:** The training loop is a system, not a script. Two abstractions made everything else tractable: `CoreConfig` (one config drives every layer's shape) and `TrainSession` (production train loop with checkpointing, snapshotting, sample generation, and SIGTERM-graceful shutdown).

This act is the most original section vs. the earlier plan — the jot was right to call it out separately.

**Beats:**
- **CoreConfig as consistency layer.** One dataclass drives `RQwen3` instantiation, all sub-layers, the optimizer setup, and the data loader. Why this matters: iterating on a 28-layer model means you change one number and *everything* downstream adjusts. Cost of getting this wrong: shape mismatches at runtime, days of debugging.
- **TrainSession as production loop.** Not just a `for batch in loader` — it wraps:
  - Optimizer (AdamW with weight-decay on 2D params only)
  - Cosine LR schedule with warmup
  - Gradient accumulation (effective batch 128 from physical batch 4)
  - Periodic checkpointing + snapshotting
  - Sample generation at each checkpoint (so you can *see* the model learning)
  - SIGTERM handler → emergency checkpoint before the cluster kills the job
  - `data_offset` tracking so resume continues at the next unseen token, not the start of an epoch
- **Why this is engineering, not glue:** every one of those bullets is a thing you only learn you needed after it bit you.

**Backing:**
- [src/config.py](../src/config.py) — `CoreConfig`
- [src/core.py](../src/core.py) — `CoreBlock` base class
- [src/training.py](../src/training.py) — `TrainConfig`, optimizer, scheduler, loss, checkpointing
- [scripts/py/TrainSession.py](../scripts/py/TrainSession.py) — session manager (modified June 8)
- [longleaf/scripts/py/pretrain.py](../longleaf/scripts/py/pretrain.py) — cluster entry point, auto-resume

**Figures:**
- TrainSession architecture diagram — boxes for CoreConfig → model + optimizer + scheduler → train loop → (checkpoint, snapshot, sample, SIGTERM handler)
- Optional: code snippet of the SIGTERM/data_offset pattern (short, ~10 lines)

---

## Act 5 — Scale (Longleaf + Seven Ways SLURM Said No)

**What it argues:** Going from laptop MPS to a real HPC cluster is not "scale up the batch size." It's a project of its own. The bug log is the proof.

**Beats:**
- **Longleaf intro (from the jot).** What Longleaf is, who built it, what it offers: UNC's HPC cluster, A100 GPUs available to students. Frame this as a resource most CS undergrads don't realize they have. A non-HPC reader needs to know what's powering Act 6 before the bug stories land.
- **SLURM script anatomy** in one annotated figure — show `pretrain.slurm` with callouts for partition, GRES, QOS, wall time, SIGTERM signal.
- **Makefile workflow.** One code block: `make ssh / setup / sync / submit / status / logs / pull`. Why a Makefile instead of typing `sbatch` every time? Because friction is the enemy of iteration.
- **Seven Ways SLURM Said No.** Each bug as problem / fix / lesson — the prose is already in [docs/project-overview.md](project-overview.md), just polish:
  1. PROJECT_ROOT path off by two levels
  2. Venv not activated in SLURM job
  3. Wrong GRES name (`gpu:a100` vs `gpu:nvidia_a100-pcie-40gb`)
  4. Wrong partition (`gpu` = GTX 1080 8GB, not A100)
  5. Module version mismatch (3.11 vs 3.12, cuda 12.1 vs 12.2)
  6. QOS required (`--qos=gpu_access`)
  7. A100 is 40GB PCIe, not 80GB SXM — batch_size 16 → 4, grad_accum 8 → 32
- **Bugs 8+ (June 8–12 work):** the story didn't end at 7. Recent additions show the work continued:
  - L40 partition variants: [longleaf/pretrain-l40.slurm](../longleaf/pretrain-l40.slurm), [longleaf/smoke-pretrain-l40.slurm](../longleaf/smoke-pretrain-l40.slurm) — added when A100 queue times got long
  - Resumable data build: [longleaf/build-data-resume.slurm](../longleaf/build-data-resume.slurm) + [scripts/stitch_manifest.py](../scripts/stitch_manifest.py) — added when a single build job couldn't fit in the wall-time limit
- **Real engineers recognize real pain** — this is the most *hireable* section.

**Backing:**
- [longleaf/](../longleaf/) — all SLURM scripts + setup.sh
- [Makefile](../Makefile) — workflow automation
- [docs/cluster-operations.md](cluster-operations.md) — practical commands
- [docs/slurm-longleaf-guide.md](slurm-longleaf-guide.md) — SLURM fundamentals

**Figures:**
- Annotated `pretrain.slurm` (single screenshot with callouts)
- `squeue` / `sinfo` screenshot — HPC "proof of life"
- Optional: bug-fix timeline (horizontal axis of dates, each bug as a point)

---

## Act 6 — Pretraining and Results (contingency-aware)

**What it argues:** Here's what training looks like, what's run so far, what's queued. Honest snapshot — training continues.

**Beats:**
- **What pretraining actually is** (callback to Act 1's next-token prediction). Loss = how surprised the model is by the right next token. Goal: make it less surprised.
- **What's real to show today (2026-06-12):**
  - Local 200-step run on TinyStories (Apple Silicon MPS, Feb 2026). Real checkpoints in `snapshots/`: untrained.pt → pre-trained.pt.
  - Untrained-vs-trained weight analysis from [notebooks/](../notebooks/) 04 — histograms shift, generation quality jumps.
  - Architecture proof from notebook 02 (carried over from Act 2 — same artifact, different use).
- **What's pending:**
  - Dataset build on Longleaf (build-data + build-data-resume SLURM scripts ready)
  - Full 50K-step pretrain (~13B tokens, ~1 epoch, multiple 24h submissions, auto-resumes)
  - Evaluation via lm-eval-harness (ARC, HellaSwag, MMLU)

**Contingency table** (verbatim from [docs/case-study-plan.md](case-study-plan.md) — works either way):

| Scenario | Results section says |
|----------|---------------------|
| Full 50K run done | Loss curves, lm-eval-harness scores, generations, ablations |
| Partial run | Loss curve to step N, generations at checkpoints, "training continues" framing |
| Cluster stalls | Local 200-step TinyStories run + untrained-vs-trained analysis from notebook 04 |

**Honest framing.** This is a mid-project snapshot. Pretending it's resolved would be worse than saying "training continues — checkpoints and curves go in here as they come out." Recruiters and grad admissions both read "mid-project, here's the rigor" as more credible than "wrapped, here's the bow."

**Figures:**
- Loss curve(s) — local 200-step now, cluster run when available
- Generation samples at checkpoints (step 0, 100, 200, then cluster steps when available) — before/after side-by-side
- Eval table when lm-eval-harness has been run

---

## Closing — What I'd Do Differently / What's Next

**What it argues:** Self-critique = methodology awareness. This is the grad-school signal.

**Beats:**
- **What I'd do differently** — 2–4 honest items. Candidates: pick the score-threshold ablation up front instead of mid-project, settle on one cluster partition before building both A100 and L40 variants, write the data builder as resumable from day one (would've saved the stitch_manifest.py retrofit).
- **What's next (post-training):** Qwen3's 4-stage post-training pipeline (long CoT cold start → reasoning RL → thinking-mode fusion → general RL). My [notebooks/](../notebooks/) 06 (SFT) is in progress; MoE routing in larger Qwen3 variants (30B-A3B) is the next research thread.
- **Links out:** [docs/project-overview.md](project-overview.md), [docs/case-study-plan.md](case-study-plan.md), repo root.

---

## Accuracy log — what changed since the earlier plan

Spot-checks against the current codebase. Fix these before publishing:

- ~~**Step-count conflict still unresolved.**~~ **Resolved 2026-06-15.** All docs now say 50K (matches `pretrain.py`).
- **Notebook filename typo:** `notebooks/Qwen3 Visual Anylasis.ipynb` (Anylasis → Analysis). Rename before publishing or before the README links to it. *(Still open — file rename.)*
- **`src/config.py` defaults are misleading.** Default `d_model=512`, `intermediate_size=6144` — RQwen3 is always instantiated with `d_model=1024, intermediate_size=3072`. Not a bug (defaults are overridden everywhere), but worth either updating the defaults to the real RQwen3 spec or adding a class-level comment that the defaults are unused. *(Still open — code change.)*
- ~~**[docs/project-overview.md](project-overview.md) is out of date.**~~ **Resolved 2026-06-15.** Phase 2 table now reflects L40S as primary, current batch config (`batch_size=2 × grad_accum=64`), bf16/SDPA, and live training progress. L40 variants + `build-data-resume.slurm` + `scripts/stitch_manifest.py` are now mentioned in Phase 3.
- ~~**Cluster pretrain has not actually been run.**~~ **Resolved 2026-06-15.** Full 50K run is in flight on `l40-gpu`: submission 8 of 10, step ~35,170 / 50,000 (70%), loss 11.88 → 2.53. Act 6 should reflect actual numbers at draft time.

---

## How to use this doc

1. **Capture artifacts now** (per [docs/case-study-plan.md](case-study-plan.md) "What to Capture Now"): screenshots of `squeue`/`sinfo`, every loss curve (even failed runs), generation samples at every checkpoint, the bug log as new things break.
2. **Draft Acts 1–5 now** — they don't depend on training outcomes.
3. **Resolve the step-count conflict** before any draft is shown to anyone.
4. **Slot Act 6 in last** — once the cluster run produces something, partial or full.
5. **When publishing,** [docs/case-study-plan.md](case-study-plan.md) governs hosting, audience layering, and explainer pages; this doc governs the spine.
