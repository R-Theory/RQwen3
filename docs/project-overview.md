---
tags:
  - qwen3
  - deep-learning
  - transformer
  - from-scratch
  - longleaf
created: 2026-03-23
status: in-progress
related:
  - "[[cluster-operations]]"
  - "[[slurm-longleaf-guide]]"
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
- **Pre-training at scale**: Training RQwen3 on UNC's Longleaf HPC cluster

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
Qwen3 Analysis/
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
│   ├── py/                 Training sessions, chat, RL experiments
│   └── build_dataset.py    Pre-tokenize & build training data (see [[data-pipeline]])
├── longleaf/               ← HPC cluster infrastructure
│   ├── setup.sh            One-time cluster setup
│   ├── pretrain.slurm      24-hour A100 training job
│   ├── build-data.slurm    CPU-only dataset build job
│   ├── smoke-pretrain.slurm 15-minute smoke test
│   ├── test-gpu.slurm      GPU verification
│   └── scripts/py/         Cluster-specific training scripts
├── checkpoints/            ← Training checkpoints (.pt files)
├── snapshots/              ← Model state snapshots at key points
├── docs/                   ← Documentation (you are here)
├── Makefile                Longleaf workflow automation
└── .rsyncignore            Cluster sync exclusions
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
| 07 | `07-pretrain-rqwen3` | Full pretraining setup, cluster config, training loop design | Complete |

**Exploratory notebooks** (earlier experiments, not numbered):
- `Basics AI.ipynb` — Foundational concepts
- `My Qwen3.ipynb` — Initial Qwen3 experiments
- `Qwen3 Visual Anylasis.ipynb` — Early visualization work
- `RQwen3.ipynb`, `Rqwen.ipynb` — Model prototyping

---

## Training Scripts

| Script | Purpose | Runs On |
|--------|---------|---------|
| `scripts/py/TrainSession.py` | Training session class — optimizer, scheduler, train loop, checkpointing, sample generation | Both |
| `scripts/py/ChatSession.py` | Interactive chat interface for inference | Local |
| `scripts/py/RLModel0.py` | Reinforcement learning experiments | Local |
| `longleaf/scripts/py/pretrain.py` | Full pretraining — 10K steps, FineWeb-Edu, auto-resume, SIGTERM handling | Cluster |
| `longleaf/scripts/py/smoke_pretrain.py` | Smoke test — 20 steps, TinyStories, quick validation | Cluster |

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

### Phase 2: Cluster Training (In Progress)

| Detail | Value |
|--------|-------|
| Cluster | UNC Longleaf HPC |
| Partition | `a100-gpu` (8 nodes, 24x A100 40GB PCIe) |
| Full pretrain | 10,000 steps on FineWeb-Edu |
| Effective batch | 128 (batch_size=4 × grad_accum=32) |
| Tokens per step | ~262K |
| Wall time | 24 hours per submission |
| Auto-resume | Yes — SIGTERM checkpoint + scan for latest `.pt` |

**Current status:** Smoke test job `38785256` submitted to a100-gpu, waiting in queue (estimated start ~Mar 24 7:25 AM). Full pretrain ready to submit after smoke test passes.

### Phase 3: Data Pipeline (Complete)

Built a production-grade data preprocessing pipeline. See [[data-pipeline]] for full details.

- **6 curated sources**: FineWeb-Edu, Wikipedia, OpenWebMath, StackExchange, peS2o, textbooks (~13B tokens)
- **Quality filtering**: score thresholds, length bounds, source-specific rules
- **Deduplication**: exact SHA-256 hash within each source
- **Pre-tokenization**: binary uint32 shards, memory-mapped for O(1) random access
- **Reproducibility**: deterministic train/val split, data offset checkpointing for exact resume

---

## Cluster Setup Journey

Chronological log of getting the training pipeline working on Longleaf. Every one of these was a real bug that had to be diagnosed and fixed.

### Bug 1: PROJECT_ROOT Path

**Problem:** `pretrain.py` had `os.path.join(os.path.dirname(__file__), "..")` — only goes up 1 level to `longleaf/scripts/`, not the project root.

**Fix:** Changed to `"..", "..", ".."` (3 levels: `py/` → `scripts/` → `longleaf/` → root).

**Lesson:** Always trace the actual directory levels when building relative paths.

### Bug 2: Venv Not Activated

**Problem:** `pretrain.slurm` had the venv activation commented out with a placeholder path.

**Fix:** Uncommented and set to `/work/users/t/r/treese20/envs/qwen3/bin/activate`.

**Lesson:** SLURM jobs don't inherit your shell environment. Everything must be explicit.

### Bug 3: Wrong GRES Name

**Problem:** Used `gpu:a100:1` — SLURM rejected it with "Requested node configuration is not available."

**Fix:** Ran `sinfo` to discover the actual name: `gpu:nvidia_a100-pcie-40gb:1`.

**Lesson:** GRES names are cluster-specific. Always check with `sinfo -p <partition> -o "%G"`.

### Bug 4: Wrong Partition

**Problem:** Used `--partition=gpu` — that partition has GTX 1080s with only 8GB VRAM.

**Fix:** Changed to `--partition=a100-gpu`.

**Lesson:** Partition names aren't standardized. `sinfo` shows what's available.

### Bug 5: Module Version Mismatch

**Problem:** SLURM scripts had `python/3.11` and `cuda/12.1`, but `setup.sh` installed `python/3.12` and `cuda/12.2`.

**Fix:** Updated all SLURM scripts and `.bashrc` to match: `python/3.12`, `cuda/12.2`.

**Lesson:** Keep module versions consistent between setup and job scripts.

### Bug 6: QOS Required

**Problem:** Submitting without `--qos` failed with "Invalid qos specification."

**Fix:** Ran `sacctmgr show qos` to find available QOS values, added `--qos=gpu_access`.

**Lesson:** Some partitions require a specific QOS. Check with `sacctmgr`.

### Bug 7: A100 is 40GB, Not 80GB

**Problem:** `batch_size=16` was configured for 80GB A100 SXM — Longleaf has 40GB PCIe.

**Fix:** Reduced `batch_size` from 16 → 4 and increased `grad_accum_steps` from 8 → 32 (effective batch stays 128).

**Lesson:** A100 comes in two variants. Always check actual VRAM with `nvidia-smi` or `sinfo`.

---

## Makefile Workflow

All cluster interaction goes through `make`:

```bash
# First time setup
make ssh           # Authenticate to Longleaf
make setup         # Create dirs + venv on cluster

# Daily workflow
make sync          # Push code changes
make submit        # Sync + submit pretrain job
make submit JOB=smoke-pretrain  # Submit smoke test
make status        # Check job queue
make logs          # Tail latest log
make pull          # Download checkpoints, snapshots, logs

# Utilities
make test          # Submit GPU verification
make cancel JOB=38785256  # Cancel a job
make remote-ls     # List files on cluster
make sync-dry      # Preview what would sync
```

See [[cluster-operations]] for detailed commands and troubleshooting.

---

## Next Steps

- [x] Build data pipeline (6-source, filtered, deduplicated, pre-tokenized)
- [ ] Build dataset on Longleaf (`sbatch longleaf/build-data.slurm`)
- [ ] Run score threshold ablation (score >= 2 vs 3 vs 4) to validate filtering
- [ ] Submit full pretrain (50K steps, ~13B tokens, curated mixed dataset)
- [ ] Set up evaluation suite (lm-evaluation-harness: ARC, HellaSwag, MMLU)
- [ ] Run domain mixing ablation (single source vs 6-source mix)
- [ ] Complete supervised finetuning notebook (06)
- [ ] Explore MoE routing in larger Qwen3 models (30B-A3B)

---

## Related Documentation

- [[data-pipeline]] — Dataset curation, preprocessing pipeline, storage format
- [[cluster-operations]] — Practical commands for monitoring and troubleshooting cluster jobs
- [[slurm-longleaf-guide]] — SLURM fundamentals, script anatomy, auto-resume system
- O-Vault: `Projects/resources/LongLeaf/00-07` — Comprehensive Longleaf reference series
- O-Vault: `Topics/AI/ML & DL/research-notes/AI Models/Qwen3/` — Architectural analysis notes
