---
tags:
  - qwen3
  - roadmap
  - planning
  - sft
  - evaluation
created: 2026-06-26
updated: 2026-06-26
status: active
related:
  - "[[project-overview]]"
  - "[[pretraining-results]]"
  - "[[data-pipeline]]"
---

# RQwen3 Roadmap — From Base Model to Chat Assistant

## Where the project is today

Pretraining is **complete** (2026-06-19): 50,000 steps, 13 B tokens, final loss **2.5186** (perplexity ≈ 12.4), saved at `checkpoints/final.pt` on Longleaf `/work`. Full record in [[pretraining-results]]. The case-study writeup of that phase is publishable as-is.

What exists now is a **base model** — a next-token predictor that's fluent but doesn't follow instructions. What this roadmap covers is the path from there to a **working educational chat assistant** that's been evaluated, fine-tuned, and is actually usable.

## What's already done

For context before the forward plan — here's the ground each milestone below is built on. Eight phases shipped, in roughly the order they happened:

| Phase | What it produced | Where it lives | Status |
|---|---|---|---|
| Architecture build | RQwen3 (751M, Qwen3-0.6B clone) — RMSNorm, RoPE, GQA, SwiGLU, QK-Norm, no biases, untied embed/head | `src/layers/`, `src/models/rqwen3.py` | ✓ |
| Weight-loading validation | Loaded real Qwen3-1.7B weights into the from-scratch code; coherent generation = architecture verified | [notebook 02](../notebooks/02-qwen3-from-scratch.ipynb) | ✓ |
| Curated dataset | 13B-token, 6-source pre-tokenized corpus (FineWeb-Edu 54%, Wikipedia 15%, OpenWebMath 12%, StackExchange 8%, peS2o 8%, textbooks 4%) | [[data-pipeline]], `scripts/build_dataset.py` | ✓ |
| Training infrastructure | `CoreConfig`, `TrainSession`, SIGTERM auto-resume, selective weight decay, bf16 + SDPA, gradient accumulation | `src/`, `scripts/py/TrainSession.py`, `longleaf/scripts/py/pretrain.py` | ✓ |
| Cluster setup | "Seven Ways SLURM Said No" debugged; Makefile workflow; L40S partition targeting | `longleaf/`, `Makefile` | ✓ |
| Pretraining run | 50,000 / 50,000 steps, loss 11.88 → **2.5186** (perplexity ≈ 12.4); 10 SLURM submissions over 11 days; `checkpoints/final.pt` on Longleaf `/work` | [[pretraining-results]] | ✓ 2026-06-19 |
| Loss-curve plot | Annotated PNG of the training trajectory across all 10 submissions | `figures/loss_curve.png`, `scripts/py/plot_loss_curve.py` | ✓ 2026-06-25 |
| Case-study writeup | Six-act narrative end-to-end; benchmark comparison; sample evolution; "what's pending" honest framing | `Case Study Story/Whole Story (RQwen3) - Final.md` | ✓ 2026-06-25 |

Everything below is what builds on this — the path from `final.pt` to a working assistant.

## The shape of the path

Three milestones, each a shippable artifact on its own. They're cumulative but not strictly blocking — Milestone A can ship before B starts; B can ship before C starts.

| | Milestone | Output | Active work | Cluster |
|---|---|---|---|---|
| **A** | **Evaluate the base model** | Real benchmark numbers; case-study Act 6 updated with hard evidence | ~5-7 hours | 1 window |
| **B** | **Build the assistant** | Working SFT'd model + CLI chat. The model becomes a thing you can talk to. | ~20-30 hours | 1-2 windows |
| **C** | **Polish & publish** (optional) | DPO-tuned model, web UI, HuggingFace Hub upload, case-study epilogue | ~15-20 hours | 1 window |

**The headline framing for each milestone:**

- **A** turns "I think the loss is good" into "here are the benchmark numbers."
- **B** turns "I built a base model" into "I built a chat assistant."
- **C** turns "here's my project" into "here's a thing recruiters and collaborators can use."

The case study you've been writing pairs naturally with this — each milestone produces a clean addition to the existing writeup or, more likely, its own follow-up post.

---

## What we already have to build on

Before laying out the new work, what's worth naming is that **the foundation from pretraining isn't single-use** — most of the machinery extends to SFT and beyond with light edits:

| Reusable from pretraining | Used for in Milestone B/C |
|---|---|
| `src/config.py` (`CoreConfig`) | Extend with `SFTConfig` (LR, format flags, loss-masking) |
| `src/training.py` (optimizer, scheduler, checkpoint, loss) | SFT loop uses the same plumbing with new LR + masking |
| `scripts/py/TrainSession.py` | SFT session subclass / variant — bf16, SDPA, grad accum all carry over |
| `longleaf/pretrain-l40.slurm` | Template for `sft-l40.slurm` — same partition, same auto-resume |
| `Makefile` (`make sync` / `make submit` / `make pull`) | Same workflow, new targets (`make sft-submit`, `make eval-hellaswag`) |
| `src/data.py` (`PreTokenizedDataset`) | The pre-tokenization pattern adapts to SFT chat data |
| `scripts/py/ChatSession.py` (per README) | Already scaffolded; needs the SFT model + Qwen3 chat template |

The point is that **Milestones B and C are mostly composition of existing pieces**, not new infrastructure. The hard part is the dataset choices, hyperparameters, and getting the chat template right.

---

## Milestone A — Evaluate the base model

**Goal:** turn the cross-tokenizer loss comparison currently in Act 6 into hard, tokenizer-agnostic benchmark numbers. The case study claims `final.pt` "lands near Pythia-1B on a fraction of the data." This milestone makes that claim defensible against a hostile reviewer.

### Stage 0 — Pull & verify `final.pt`

**Output:** verified local copy of the 8.5 GB final checkpoint, loaded and producing samples that match the cluster log at step 50K.

- `make pull` from laptop to bring `final.pt` down from Longleaf `/work`.
- Open a notebook, load the checkpoint into `RQwen3(CoreConfig(...))`, generate from `"The theory of general relativity"` and confirm the output qualitatively matches what cluster step-50K logs showed.
- This is the gate — if the file is corrupt or the load path has drifted, you want to know now.

**Time:** ~30 min. **Risk:** low.

### Stage 1 — Build the eval scaffolding

**Output:** ARC, HellaSwag, MMLU scores comparable to other models in the literature, plus the code to re-run them on any future checkpoint (e.g., the SFT model in Milestone B).

Three pieces:

1. **`RQwen3ForCausalLM(PreTrainedModel)`** — a HuggingFace wrapper around `RQwen3`. lm-eval-harness expects `from_pretrained`, `generate()`, and a `PretrainedConfig`. Roughly 150 lines of glue. **Smoke test against raw-class logits** — a bug in the wrapper means the eval numbers are silently wrong.
2. **Makefile targets** — `make eval-hellaswag`, `make eval-arc`, `make eval-mmlu`, calling lm-eval-harness against the wrapped model.
3. **`longleaf/eval.slurm`** — SLURM script for running eval on an L40S (each benchmark ~30 min on cluster GPU vs. hours on Mac MPS).

### What Milestone A produces

- `src/hf_compat.py` — `RQwen3ForCausalLM` HF-compatible wrapper
- New Makefile targets and `longleaf/eval.slurm`
- A new `docs/eval-results.md` documenting the run and the numbers
- An update to **Act 6 of the case study** replacing the loss-comparison hand-wave with a real benchmark table
- The same scaffolding will get re-run after SFT to detect alignment tax

### Decisions to make in Milestone A

- **Run eval locally on MPS or on the cluster?** Cluster is ~10× faster but adds queue time. Recommended: cluster, since the scaffolding will get reused anyway.
- **Which benchmarks?** Minimum: HellaSwag (most defensible prior). Recommended: HellaSwag + ARC-Challenge + MMLU (matches what every other model card reports). Avoid: niche benchmarks that don't have clean Pythia/GPT-2 references.

**Time estimate:** 5-7 hours active + 1 cluster window. **Risk:** medium (wrapper correctness).

---

## Milestone B — Build the assistant

**Goal:** turn the base model into a chat assistant that follows instructions. This is where the conceptual leap happens — pretraining taught the model to *continue* text; SFT teaches it to *respond*. By the end of this milestone, RQwen3 stops being a curiosity and starts being a tool.

### Stage 2a — Pick / assemble the SFT dataset

**Output:** a curated chat dataset, 50K–200K examples, that aligns with the project's stated goal (educational STEM assistant).

This is the most consequential single decision in the milestone, because the SFT data shapes the assistant's personality more than any other choice.

Three plausible approaches, in increasing custom-ness:

| Approach | Dataset(s) | What it buys |
|---|---|---|
| **Quick & general** | `HuggingFaceH4/ultrachat_200k` or `OpenAssistant/oasst2` | Proven; widely benchmarked; gets you to a working assistant fast |
| **Tuned to project goal** | Tulu-3 SFT mixture | Reasoning- and STEM-leaning, good fit for "educational assistant" goal |
| **Custom STEM mix** | StackExchange Q&A pairs from your own pretraining corpus, reformatted into chat templates, mixed with general SFT data | Most aligned with project goal; most work; most interesting story |

**Recommendation:** start with Tulu-3 SFT mixture for the baseline run; it gives a clean comparison point. Custom STEM mixing becomes a Milestone-C ablation (and its own follow-up post).

For a 751M model, **~100K examples is a defensible budget.** More risks overtraining and capability loss; less risks underbaking the instruction-following behavior.

### Stage 2b — Format with the Qwen3 chat template

**Output:** training data in the form the model expects — `<|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>`.

Two critical implementation details:

- **Reuse the Qwen3 chat template.** Don't invent a new one. The tokenizer already knows `<|im_start|>`, `<|im_end|>`, and the role markers. Apply `tokenizer.apply_chat_template()` to each example.
- **Loss-mask the user turns.** This is the single most common SFT bug. You only want gradient signal on the *assistant's* responses; backpropping on user prompts teaches the model to mimic users, which is wrong. Concretely: in the labels tensor, set every token in a user turn to `-100` (PyTorch's ignore_index for cross-entropy).

### Stage 2c — SFT training loop

**Output:** trained SFT model checkpoint (`checkpoints/sft_final.pt`).

Same `TrainSession` infrastructure as pretraining, with different hyperparameters:

| Parameter | Pretraining | SFT |
|---|---|---|
| Peak LR | 3e-4 | 1e-5 to 5e-5 |
| LR schedule | Cosine, 500-step warmup, 50K steps | Cosine, ~100-step warmup, ~1-3 epochs over SFT data |
| Effective batch | 128 | 64-128 |
| Sequence length | 2048 | 2048 (chat templates can be long) |
| Loss masking | None | **User turns masked** |
| Precision | bf16 autocast + SDPA | Same |
| Cluster target | L40S | L40S |

**Wall time:** 6-12 hours for ~100K examples / 1-3 epochs on a single L40S. Most likely fits in a single 24h SLURM submission.

### Stage 2d — SFT evaluation

**Output:** SFT-specific quality numbers + confirmation that pretraining capability didn't regress.

Two evals to run:

- **MT-Bench** (multi-turn conversation, GPT-4 judged, 1-10 score per category) — current standard for SFT quality.
- **Re-run HellaSwag / MMLU from Milestone A** — the question here is "did SFT hurt base capability?" (the "alignment tax"). If HellaSwag drops by more than ~2 points, the SFT data mix is wrong or the LR is too high.

Optionally: AlpacaEval for single-turn instruction quality.

### Stage 4 (light) — CLI chat

**Output:** a `scripts/py/ChatSession.py` that loads `sft_final.pt`, applies the Qwen3 chat template, and lets you actually talk to the model from the terminal.

Per README, `ChatSession.py` is already scaffolded. The work here is wiring it to the SFT model + handling the chat template (`apply_chat_template` with `add_generation_prompt=True`) + a basic generation loop with stopping on `<|im_end|>`.

### What Milestone B produces

- `data/sft/` — formatted SFT dataset shards
- `scripts/build_sft_dataset.py` — analogous to the pretraining build script
- `scripts/py/SFTSession.py` — SFT training loop (light variant of `TrainSession`)
- `longleaf/sft-l40.slurm` — SFT cluster job
- `checkpoints/sft_final.pt` — the SFT'd model
- An updated `ChatSession.py` you can actually use
- A new `docs/sft-results.md` documenting the SFT run, MT-Bench scores, and the alignment-tax check
- Strong candidate for a **separate "Part 2" case-study post** about SFT specifically

### Decisions to make in Milestone B

- **SFT dataset choice** (Tulu-3 vs. UltraChat vs. custom) — see Stage 2a.
- **Epoch count** (1, 2, or 3) — start with 1, look at MT-Bench, decide whether to push.
- **Whether to do a separate writeup post for SFT** or fold it into a larger Act 7 of the existing case study.

**Time estimate:** 20-30 hours active + 1-2 cluster windows. **Risk:** medium (SFT hyperparams, loss-masking bug, mode collapse).

---

## Milestone C — Polish & publish (optional)

**Goal:** turn the working assistant into a *shareable* artifact other people can find, evaluate, and use. This milestone is incremental polish; the project is already complete after Milestone B for portfolio purposes.

### Stage 3 — DPO (Direct Preference Optimization)

**Output:** a model whose responses lean toward what humans actually prefer, not just what's grammatically correct.

DPO is the current standard for preference learning — simpler than the old PPO/RLHF stack, just a standard supervised loss over `(prompt, chosen_response, rejected_response)` triples.

- **Dataset:** `HuggingFaceH4/ultrafeedback_binarized` or `argilla/ultrafeedback-binarized` are the open standards.
- **Training:** few hundred steps at very low LR (~5e-7), starting from `sft_final.pt`. Cheaper than SFT.
- **Risk:** DPO is finicky. A bad reference model or LR can produce a model worse than the SFT starting point. Always benchmark the DPO output against the SFT input.

**Time:** 8-12 hours active + 1 cluster window.

### Stage 4 (full) — Web UI + HuggingFace Hub

**Output:** something with a public link.

- **Gradio web demo** (~100 lines, local) — talk to the model in a browser
- **Quantization** — convert to int8 or int4 (via `bitsandbytes` or AWQ) so it runs on a laptop without thermal throttling. At 751M, int4 brings the model under 500 MB and ~3× faster on Apple Silicon.
- **HuggingFace Hub upload** — `huggingface-cli upload` the SFT (and DPO) checkpoints with a proper model card
- **Optional: HuggingFace Space** — Gradio Space for a free-hosted public demo

### Stage 5 — Documentation & case-study epilogue

**Output:** the project's published, citable record.

- **HuggingFace model card** — training data disclosure, eval results, intended use, limitations. This is the "model documentation" standard.
- **Case-study epilogue or Part 2/3 posts** — written from this roadmap once the milestones land, with real numbers and the chat demo link.
- **Updated repo README** — eval numbers, chat demo link, model card link, "use this" instructions.

**Time:** 6-8 hours active.

### What Milestone C produces

- `checkpoints/dpo_final.pt`
- Quantized variants for laptop inference
- HF Hub model page
- Public demo (optional)
- HF model card
- Published "Part 2" / "Part 3" case-study writeups

---

## Time and shape — realistic calendar

| Milestone | Active hours | Cluster windows | Sequential? |
|---|---|---|---|
| A | 5-7 | 1 | Blocking — do first |
| B | 20-30 | 1-2 | After A |
| C (optional) | 15-20 | 1 | After B |

**Compressed calendar (all-in push):** ~10 working days for A+B; ~14-16 days through C.

**Realistic calendar (weekends + cluster wait):** A in 1 weekend; B in 2-3 weekends spread across 3-4 weeks (cluster queue is the bottleneck, not the work); C in 2 weekends after that.

---

## Risks worth naming upfront

| Risk | Where it lives | Mitigation |
|---|---|---|
| HF wrapper bugs → silently wrong eval numbers | Milestone A, Stage 1 | Smoke test wrapped logits against raw-class logits before submitting to cluster |
| Loss-masking bug in SFT → model learns to mimic users | Milestone B, Stage 2b | Inspect the labels tensor for a batch; confirm user-turn tokens are `-100` |
| SFT overtraining → capability loss (alignment tax) | Milestone B, Stage 2c-d | Re-run HellaSwag after SFT; if drop > 2 pts, fewer epochs or lower LR |
| DPO produces a worse model than SFT | Milestone C, Stage 3 | Benchmark DPO output against SFT input; rollback if MT-Bench drops |
| Cluster queue length breaks the rhythm | All milestones | The Sub 8 queue-wait incident from pretraining is the precedent. Plan around 12-24h queue waits, submit early in the week |
| iCloud sync on large checkpoints | All milestones | Project-storage memory rule: keep checkpoints on `~/repos/RQwen3/checkpoints/`, not in iCloud-synced paths |

---

## Open questions (your decisions)

These are the choices the roadmap can't make for you. Worth thinking about before starting each milestone:

1. **Milestone A — when do you publish the existing case study?** Two options:
   - **Now,** with the existing "eval queued for next chapter" framing
   - **After Milestone A,** with real benchmark numbers
   - Recommendation in the case-study verification was "publish now." This roadmap doesn't change that recommendation; the case study can ship before Milestone A finishes.

2. **Milestone B — which SFT dataset?**
   - Recommendation: Tulu-3 SFT mixture for the baseline; custom STEM mix as a follow-up ablation.

3. **Milestone B — one post or two?**
   - **One post:** add an Act 7 to the existing case study
   - **Two posts:** a separate "Part 2" specifically about the SFT story (recommended — SFT has its own engineering shape worth a standalone treatment)

4. **Milestone C — go all the way?**
   - DPO + HF Hub upload + Space is meaningfully more credential, but doesn't change whether the project is "done." Optional based on whether you're still building on RQwen3 or pivoting to a new project.

5. **Anything beyond C?**
   - MoE experiment (`MoEFeedForward` on every other layer — the architecture was designed for this)
   - Score-threshold ablation on FineWeb-Edu
   - Source-share ablation on the 6-source data mix
   - These are all backlog items from the pretraining results doc. All become much more interesting research stories once the chat model exists as a baseline.

---

## What this document is for

This is the **canonical planning document** for the post-pretraining phase. As milestones land, update the status block at the top, mark sub-stages complete inline, and link to the artifacts they produced (eval results doc, SFT results doc, HF Hub page).

When a published narrative version is wanted ("Part 2" of the case study), draft it from this — but keep this doc as the working source of truth.
