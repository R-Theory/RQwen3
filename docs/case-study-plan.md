---
tags:
  - qwen3
  - case-study
  - planning
created: 2026-06-07
updated: 2026-06-08
status: planning
---

# RQwen3 Case Study — Presentation Plan

How to package this project for maximum value: job applications, grad school, public visibility, and a learning record — all served by two artifacts: a **written case study** and a **polished repo**.

## Audience Architecture (decided)

The piece serves two audiences that want opposite things from the same paragraph: the recruiter/researcher who wants depth fast, and the curious beginner who needs walking from zero. Resolution: **layered, not linear.**

- **Spine** = the case study page. Technical, uninterrupted, acts 1–5. The expert reads this and never has to see the basics.
- **Explainers** = separate linked pages, one concept each. A glossed term in the spine is a link; clicking it goes to a beginner explanation (optionally with an interactive). The beginner who needs it clicks through; everyone else keeps reading.

This is why hosting is **my own site, not Medium** (decided — see Deliverable 1). Linked explainers + embedded interactives don't work on Medium/Substack.

**Beginner on-ramp** (opens the spine as two short paragraphs; deeper versions live on explainer pages):

1. **Rules vs. patterns** — two ways to make a computer do a task. Programming = you write every rule explicitly. ML = you show examples and the patterns are learned. This is the load-bearing frame: it's *why* data quality is everything (trash in → trash out), why training is a loop, why a falling loss curve means anything.
2. **Next-token prediction** — the specific pattern *this* machine learns. The model outputs a probability distribution over ~152K possible next tokens; training just nudges the right one up. This single idea explains vocab size, the loss function, the data obsession, and what the loss curve represents.

**Linking discipline:** link a term **once**, at first meaningful use, only for terms a motivated non-expert genuinely can't infer. Target ~6–10 linked terms across the whole piece. (neural network, next-token prediction, gradient descent → yes. batch size, checkpoint → no.) Over-linking turns the spine to Swiss cheese and makes the beginner stop trusting that any link matters.

**Interactive scope (v1):** one hero interactive (next-token prediction — let the reader watch the model pick), text explainers for everything else. Interactives are the single most expensive item here — more work than all five acts combined. Ship spine + text explainers + the one hero widget first; add more interactives one at a time after launch. Don't let widgets block publishing.

## The Story

**Title direction:** "I Rebuilt Qwen3 From Scratch and Pretrained It on a University Supercomputer"

One sentence: *Reconstructed Qwen3-0.6B's architecture component-by-component (751M params), built a 13B-token curated data pipeline, and pretrained it on UNC's Longleaf A100 cluster.*

> **Note on the 751M vs. nominal 0.6B gap** — preempt this everywhere it appears, because a sharp reader will read it as an error. The gap is almost all embedding table: 151,936 vocab × 1024 dims ≈ 156M params in token embeddings alone, which dominates parameter count at this scale. Stated, it flips from "did they miscount?" to "they know where params live in small models." One sentence, turned into a flex.

The hook is the **completeness**: most "from scratch" projects stop at a toy model on a laptop. This one goes architecture → data pipeline → real HPC training. That arc is the spine of everything.

### Narrative Arc (5 acts)

1. **Why** — Using models isn't understanding them. Goal: rebuild Qwen3 so every line is mine, then train it for real. *(Opens with the two-paragraph beginner on-ramp: rules vs. patterns → next-token prediction. Expert reads two paragraphs and is already at the architecture; beginner clicks through to explainers.)*
2. **Build** — The architecture, told as "what changed since GPT-2 and why": RMSNorm, RoPE, GQA, SwiGLU, QK-Norm, no biases. Validate by loading real Qwen3-1.7B weights into my implementation and generating text. *(This is the proof the implementation is correct — lead with it.)*
3. **Data** — Pretraining is mostly a data problem. 6-source, ~13B-token curated mix (FineWeb-Edu 54%, Wikipedia, OpenWebMath, StackExchange, peS2o, textbooks), quality-filtered, deduped, pre-tokenized into memmap shards with exact-resume offsets.
4. **Scale** — Apple Silicon → SLURM on Longleaf A100s. Include the 7-bug debugging journey (GRES names, QOS, 40GB vs 80GB A100, etc.) — this is the most relatable, most *hireable* section. Real engineers recognize real pain.
5. **Results** — Loss curves, before/after generations, evals. (Contingency plan below.)

## Deliverable 1: Written Case Study

Long-form post (~2500–4000 words), hosted on **my own site** (decided — required for linked explainers + embedded interactives; Medium/Substack can't do either). Structure mirrors the 5 acts. Rules:

- Every section gets at least one figure. Target figures:
  - Architecture diagram (RQwen3 block structure, GQA head layout)
  - GPT-2 vs Qwen3 component comparison table (already exists in project-overview) — **reframe the "Why It's Better" column to "What It Trades / Why It's Used."** Unhedged "better" reads to a researcher as reciting blog-post consensus; framing each as an engineering choice with a cost (GQA trades a little quality for big KV-cache savings, etc.) is the most research-flavored signal in the table.
  - Attention heatmaps / untrained-vs-trained weight distributions (notebooks 03–04)
  - Data mix pie/bar chart + pipeline flow diagram
  - Loss curve(s) — local 200-step run now, cluster run later
  - Generation samples: untrained vs trained, side by side
- Numbers in headers and intro: **751M params, 13B tokens, 28 layers, A100s**. Concrete numbers are what people remember and what recruiters skim for.
- The bug log becomes a section called something like "Seven Ways SLURM Said No" — each bug as problem/fix/lesson (already written in project-overview, just needs polish).
- End with "What I'd do differently" — this is the grad-school signal (self-critique, methodology awareness).

## Deliverable 2: Repo as Portfolio Piece

The README is the landing page; assume 30-second skim.

- [ ] Lead README with the one-sentence pitch + a hero figure (loss curve or architecture diagram), before any setup instructions
- [ ] Add "Results" section near the top (even partial results)
- [ ] Move Quick Start / Makefile details lower
- [ ] Clean or quarantine exploratory notebooks (`Basics AI.ipynb`, `Rqwen.ipynb`, etc.) into `notebooks/exploratory/` — the numbered 01–07 sequence is the showcase
- [ ] Fix typo: `Qwen3 Visual Anylasis.ipynb`
- [ ] Ensure each numbered notebook has its "What I Learned" section complete
- [ ] Link the case study post from the README and vice versa
- [ ] Add a `figures/` gallery or embed key figures in README

## Deliverable 3: Explainer Pages (beginner layer)

Separate linked pages on the site, one concept each. These are what the spine's glossed terms link to.

- [ ] **Rules vs. patterns** — programming vs. ML, the load-bearing frame
- [ ] **Next-token prediction** — the distribution-over-tokens idea + **hero interactive** (watch the model pick the next token)
- [ ] **What a neural network is** — minimal, only what's needed downstream
- [ ] **What training actually is** — the loop, the loss, gradient descent at a feel-level
- [ ] Keep total linked terms in the spine to ~6–10; each links once at first use

## Results Contingency (works either way)

> **✅ Step-count consistency** — Resolved 2026-06-15. **50K steps** is the canonical figure everywhere: this doc, `project-overview.md`, `cluster-operations.md`, `slurm-longleaf-guide.md`, README, and `longleaf/scripts/py/pretrain.py`. Keep this anchor when adding new content.

The story is **already complete without the 50K-step run**: build → validate against real weights → data pipeline → cluster bring-up → *first* training runs. Results section adapts:

| Scenario | Results section says |
|----------|---------------------|
| Full 50K run done | Loss curves, lm-eval-harness scores (ARC, HellaSwag), generations, ablations |
| Partial run | Loss curve to step N, generations at checkpoints, "training continues" framing |
| Cluster stalls | Local 200-step TinyStories run + untrained-vs-trained analysis from notebook 04 |

Write acts 1–4 **now** — they don't depend on training outcomes. Slot results in last.

## What to Capture Now (cheap now, impossible later)

- [ ] Screenshot `squeue`/`sinfo` output, job logs — HPC "proof of life"
- [ ] Save loss curves from every run, even failed ones
- [ ] Save generation samples at each checkpoint (step 0, 100, 200, ...)
- [ ] Keep the bug log updated as new cluster issues appear
- [ ] Note dates/durations — "3 months, evenings and weekends" is part of the story

## Audience Variants (same core, different emphasis)

- **Recruiters/hiring:** numbers + engineering (acts 2, 4). One-paragraph resume blurb distilled from the intro.
- **Grad school:** methodology + ablations + "what I'd do differently" (acts 3, 5).
- **Public/social:** the arc + figures. Thread version: 1 tweet per act, hero figure first.

## Sequence

1. Now: capture artifacts (see "What to Capture Now"), write acts 1–4 draft, restructure README, **resolve step-count conflict across all files**
2. Build the site skeleton + spine; write the 4 text explainers; build the one hero interactive (next-token)
3. After data build + ablation: add data section figures, ablation results
4. After pretrain (full or partial): results section, hero loss curve, final polish
5. Publish post → update README → distill resume blurb + social thread → add further interactives one at a time