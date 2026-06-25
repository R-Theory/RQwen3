---
tags:
  - qwen3
  - deep-learning
  - transformer
  - from-scratch
created: 2026-03-23
updated: 2026-06-19
status: in-progress
related:
  - data-pipeline.md
  - pretraining-results.md
---

# Qwen3 Analysis — Project Overview

## What This Is

A self-directed deep dive into the Qwen3 model family, focused on **building from first principles**. Instead of just using pre-trained models, this project reconstructs the architecture from scratch, trains it, and studies the results.

The centerpiece is **RQwen3** — a custom 751M-parameter transformer matching Qwen3-0.6B's architecture, built layer by layer to understand every component.

### Focus Areas

- **Architecture**: RMSNorm, RoPE, GQA, SwiGLU — what each innovation does and why
- **MoE design**: Mixture of Experts routing in larger Qwen3 variants
- **Post-training pipeline**: The 4-stage process (long CoT cold start → reasoning RL → thinking mode fusion → general RL)
- **Thinking modes**: How `/think` and `/no_think` tokens control reasoning behavior
- **Weight inspection**: Layer-by-layer analysis comparing untrained vs pretrained models
- **Pre-training at scale**: training RQwen3 on a curated 13B-token dataset

---

## RQwen3 Architecture

Custom implementation matching Qwen3-0.6B scale:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 1024 | Hidden dimension |
| `n_layer` | 28 | Transformer blocks |
| `num_heads` | 16 | Query heads |
| `num_kv_heads` | 8 | KV heads (GQA, 2:1 ratio) |
| `head_dim` | 128 | Per-head dimension |
| `intermediate_size` | 3072 | FFN intermediate (3x expansion) |
| `vocab_size` | 151,936 | Qwen3 tokenizer |
| `max_seq_len` | 2048 | Context window |
| **Total params** | **~751M** | |

### What Changed from GPT-2

| Component | GPT-2 | RQwen3 | Why It's Better |
|-----------|-------|--------|-----------------|
| Normalization | LayerNorm | **RMSNorm** | No mean subtraction, no bias — 50% fewer norm params, faster |
| Position encoding | Learned embeddings | **RoPE** | Rotary embeddings, better length extrapolation, no parameters |
| Attention | Multi-Head (MHA) | **Grouped Query (GQA)** | Smaller KV cache at inference, 2:1 head ratio |
| FFN activation | GELU | **SwiGLU** | Gated activation, consistently better perplexity |
| Attention stability | None | **QK-Norm** | Pre-RoPE normalization prevents attention collapse |
| Biases | Everywhere | **None** | Removed all biases — simpler, regularizing effect |

---

## Project Structure

```
RQwen3/
├── src/                    ← Custom deep learning library
│   ├── config.py           CoreConfig dataclass
│   ├── core.py             CoreBlock base class
│   ├── layers/             Modular components
│   │   ├── normalization.py    RMSNorm
│   │   ├── embeddings.py       RotaryEmbedding
│   │   ├── activations.py      SwiGLU
│   │   ├── attention.py        GQAttention
│   │   ├── feedforward.py      FeedForward
│   │   └── block.py            TransformerBlock
│   ├── models/
│   │   └── rqwen3.py       RQwen3Transformer → LMHeadRQwen3 → RQwen3
│   ├── data.py             StreamingTokenDataset, PreTokenizedDataset
│   ├── training.py         TrainConfig, optimizer, scheduler, loss, checkpointing
│   ├── tokenizers.py       ModelTokenizer wrapper
│   ├── pipeline.py         PipeLine for inference
│   └── utils.py            get_device, print_model_summary, format_param_count
│
├── notebooks/              ← Sequential exploration (01–07)
├── scripts/
│   ├── py/                 Training sessions, chat
│   ├── build_dataset.py    Pre-tokenize & build training data (see data-pipeline.md)
│   └── stitch_manifest.py  Merge shard manifests after a multi-job dataset build
├── checkpoints/            ← Training checkpoints (.pt files)
├── snapshots/              ← Model state snapshots at key points
└── docs/                   ← Documentation (you are here)
```

---

## Notebooks

Sequential learning path — each builds on the previous:

| # | Notebook | What I Built / Learned | Status |
|---|----------|------------------------|--------|
| 01 | `01-qwen3-overview` | Model family survey, architecture comparison across sizes | Complete |
| 02 | `02-qwen3-from-scratch` | Built Qwen3-1.7B by hand, loaded pretrained weights, generated text | Complete |
| 03 | `03-visual-model-analysis` | Attention heatmaps, activation norms, computation graphs | Complete |
| 04 | `04-untrained-vs-trained` | Compared random init vs pretrained — weight distributions, output quality | Complete |
| 05 | `05-train-rqwen3` | First training run — 200 steps on TinyStories using MPS | Complete |
| 06 | `06-supervised-finetuning` | SFT concepts and implementation planning | In Progress |
| 07 | `07-pretrain-rqwen3` | Full pretraining setup, training loop design | Complete |

**Exploratory notebooks** (earlier experiments, not numbered):
- `Basics AI.ipynb` — Foundational concepts
- `My Qwen3.ipynb` — Initial Qwen3 experiments
- `Qwen3 Visual Analysis.ipynb` — Early visualization work
- `RQwen3.ipynb`, `Rqwen.ipynb` — Model prototyping

---

## Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/py/TrainSession.py` | Training session class — optimizer, scheduler, train loop, checkpointing, sample generation |
| `scripts/py/ChatSession.py` | Interactive chat interface for inference |

---

## Training Progress

### Phase 1: Local Training (Complete)

| Detail | Value |
|--------|-------|
| Model | RQwen3 (751M params) |
| Dataset | TinyStories |
| Steps | 200 |
| Hardware | Apple Silicon MPS |
| Date | ~Feb 2026 |

**Artifacts saved:**
- `checkpoints/step_100.pt`, `step_200.pt`, `step_200_final.pt`
- `snapshots/untrained.pt`, `untrained-init.pt`, `pre-train-start.pt`, `pre-trained.pt`

### Phase 2: Pretraining (Complete)

| Detail | Value |
|--------|-------|
| Full pretrain | 50,000 steps on the curated 6-source dataset (~13 B tokens, ~1 epoch) |
| Effective batch | 128 (`batch_size=2 × grad_accum=64`) |
| Tokens per step | ~262K |
| Precision | bf16 autocast on CUDA; SDPA (FlashAttention dispatch) for attention |
| Auto-resume | Yes — SIGTERM checkpoint + scan for latest `.pt` |

The microbatch was reduced from `4 → 2` (with `grad_accum 32 → 64`) to keep the fp32 softmax stable at 151,936 vocab × 2048 seq.

**Result (2026-06-19):** 50,000 / 50,000 steps reached on UNC Longleaf `l40-gpu` (NVIDIA L40S 48 GB). Loss **11.88 → 2.5186** (perplexity ≈ 12.4). Took **10 successful SLURM submissions over ~11 wall-clock days** (2026-06-08 → 2026-06-19), plus 3 OOM crashes and 2 user-cancels during the memory-wall fix on day 1. Base model checkpoint at `checkpoints/final.pt`. Full journey, sample-evolution log, bug list, and self-critique: [pretraining-results](pretraining-results.md).

### Phase 3: Data Pipeline (Complete)

Built a production-grade data preprocessing pipeline. See [data-pipeline](data-pipeline.md) for full details.

- **6 curated sources**: FineWeb-Edu, Wikipedia, OpenWebMath, StackExchange, peS2o (`MaLA-LM/peS2o-final`), textbooks (`HuggingFaceTB/cosmopedia` config `stanford`) — ~13B tokens
- **Quality filtering**: score thresholds, length bounds, source-specific rules
- **Deduplication**: exact SHA-256 hash within each source
- **Pre-tokenization**: binary uint32 shards, memory-mapped for O(1) random access
- **Reproducibility**: deterministic train/val split, data offset checkpointing for exact resume
- **Multi-job build**: `scripts/stitch_manifest.py` re-scans shards and writes a unified `manifest.json`

---

## Next Steps

- [x] Build data pipeline (6-source, filtered, deduplicated, pre-tokenized)
- [x] Build curated dataset (~13B tokens, all 6 sources, stitched via `stitch_manifest.py`)
- [x] Submit full pretrain (50K steps) — **complete**, final loss 2.5186
- [ ] Pull `final.pt` locally and generate inspection samples
- [ ] Set up evaluation suite (lm-evaluation-harness: ARC, HellaSwag, MMLU) — **primary next move**
- [ ] Complete supervised finetuning notebook (06) on top of `final.pt` — **primary next move**
- [ ] Publish case-study post (spine in [case-study-outline](case-study-outline.md))
- [ ] Checkpoint-pruning script (429 GB on `/work` currently — keep `step_1000`, every 5K, latest 3)
- [ ] Run score threshold ablation (score >= 2 vs 3 vs 4) — backlog
- [ ] Run domain mixing ablation (single source vs 6-source mix) — backlog
- [ ] Explore MoE routing in larger Qwen3 models (30B-A3B) — backlog

---

## Related Documentation

- [pretraining-results](pretraining-results.md) — Full pretraining journey, loss trajectory, sample evolution, bugs encountered
- [data-pipeline](data-pipeline.md) — Dataset curation, preprocessing pipeline, storage format
