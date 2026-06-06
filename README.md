# Qwen3 Analysis

Building the Qwen3 architecture from scratch and training at scale. A hands-on deep dive into modern transformer innovations: RoPE, GQA, SwiGLU, RMSNorm, and MoE design.

**Status:** In Progress — Smoke test submitted to UNC Longleaf cluster, full pretrain ready to launch

## Quick Start

### Local Development

```bash
# From the Study Grounds root directory
source .venv/bin/activate
pip install -r requirements.txt

# Launch notebooks
jupyter lab --notebook-dir="Heutagogy/projects/project-repos/Qwen3 Analysis/"
```

### Cluster Training

```bash
make ssh                        # Authenticate to Longleaf
make setup                      # One-time: create venv on cluster
make submit JOB=smoke-pretrain  # Smoke test (20 steps, 15 min)
make submit                     # Full pretrain (10K steps, 24h)
make status                     # Check job queue
make logs                       # Monitor output
make pull                       # Download checkpoints
```

## Architecture: RQwen3

Custom 751M-parameter transformer matching Qwen3-0.6B:

| Parameter | Value |
|-----------|-------|
| `d_model` | 1024 |
| `n_layer` | 28 |
| `num_heads` | 16 |
| `num_kv_heads` | 8 (GQA, 2:1 ratio) |
| `head_dim` | 128 |
| `intermediate_size` | 3072 |
| `vocab_size` | 151,936 |
| `max_seq_len` | 2048 |
| **Total params** | **~751M** |

**Key innovations over GPT-2:** RMSNorm, RoPE, Grouped Query Attention, SwiGLU activation, QK-Norm, no biases anywhere.

## Project Structure

### `src/` — Custom Deep Learning Library

| Module | Contents |
|--------|----------|
| `config.py` | `CoreConfig` dataclass — architecture settings |
| `core.py` | `CoreBlock` base class — device, params, freeze/unfreeze |
| `layers/normalization.py` | `RMSNorm` |
| `layers/embeddings.py` | `RotaryEmbedding` (RoPE) |
| `layers/activations.py` | `SwiGLU` |
| `layers/attention.py` | `GQAttention` (Grouped Query Attention) |
| `layers/feedforward.py` | `FeedForward` |
| `layers/block.py` | `TransformerBlock` |
| `models/rqwen3.py` | `RQwen3Transformer`, `LMHeadRQwen3`, `RQwen3` |
| `data.py` | `StreamingTokenDataset`, `PreTokenizedDataset` |
| `training.py` | `TrainConfig`, optimizer, scheduler, loss, checkpointing |
| `tokenizers.py` | `ModelTokenizer` wrapper |
| `pipeline.py` | `PipeLine` for inference |
| `utils.py` | `get_device`, `print_model_summary`, `format_param_count` |

### `notebooks/` — Sequential Exploration

| # | Notebook | Focus | Status |
|---|----------|-------|--------|
| 01 | `01-qwen3-overview` | Model family survey, architecture comparison | Complete |
| 02 | `02-qwen3-from-scratch` | Built Qwen3-1.7B by hand, loaded pretrained weights | Complete |
| 03 | `03-visual-model-analysis` | Attention heatmaps, activation norms | Complete |
| 04 | `04-untrained-vs-trained` | Random init vs pretrained weight analysis | Complete |
| 05 | `05-train-rqwen3` | First training run (200 steps, MPS) | Complete |
| 06 | `06-supervised-finetuning` | SFT concepts and implementation | In Progress |
| 07 | `07-pretrain-rqwen3` | Full pretraining setup for cluster | Complete |

### `scripts/` — Training & Inference

| Script | Purpose |
|--------|---------|
| `scripts/py/TrainSession.py` | Training session — optimizer, scheduler, train loop, checkpointing |
| `scripts/py/ChatSession.py` | Interactive chat with trained models |
| `scripts/build_dataset.py` | Build pre-tokenized training data from 6 curated sources |
| `longleaf/scripts/py/pretrain.py` | Full pretraining (50K steps, curated dataset, auto-resume) |
| `longleaf/scripts/py/smoke_pretrain.py` | Smoke test (20 steps, TinyStories, quick validation) |

### `longleaf/` — HPC Cluster Infrastructure

| File | Purpose |
|------|---------|
| `Makefile` | Workflow automation (sync, submit, status, logs, pull) |
| `longleaf/pretrain.slurm` | 24-hour A100 training job with graceful shutdown |
| `longleaf/build-data.slurm` | CPU-only dataset build job (16 cores, 64GB, 12h) |
| `longleaf/smoke-pretrain.slurm` | 15-minute smoke test |
| `longleaf/test-gpu.slurm` | GPU verification |
| `longleaf/setup.sh` | One-time cluster environment setup |
| `.rsyncignore` | Sync exclusions (checkpoints, data, notebooks, etc.) |

## Training Progress

### Local (Complete)

- 200 steps on TinyStories, Apple Silicon MPS
- Checkpoints: `step_100.pt`, `step_200.pt`
- Snapshots: `untrained.pt` → `pre-trained.pt`

### Data Pipeline (Complete)

Built a 6-source, ~13B-token curated dataset: FineWeb-Edu (54%), Wikipedia (15%), OpenWebMath (12%), StackExchange (8%), peS2o (8%), textbooks (4%). Quality-filtered, deduplicated, pre-tokenized into binary shards. See [docs/data-pipeline.md](docs/data-pipeline.md).

### Cluster (In Progress)

- **Cluster:** UNC Longleaf HPC, A100 40GB PCIe
- **Data build:** `sbatch longleaf/build-data.slurm` (CPU-only, ~4-12h)
- **Full pretrain:** 50,000 steps, curated mixed dataset, effective batch 128, multiple 24h submissions

## Makefile Commands

```
make help             Show all commands
make ssh              Authenticate to Longleaf
make setup            One-time cluster setup
make sync             Push code to cluster
make submit           Sync + submit pretrain job
make submit JOB=x     Sync + submit specific job
make test             Sync + submit GPU test
make status           Check job queue
make logs             Tail latest log
make pull             Download checkpoints, snapshots, logs
make cancel JOB=id    Cancel a job
make remote-ls        List files on cluster
```

## Documentation

| Doc | What It Covers |
|-----|----------------|
| [docs/project-overview.md](docs/project-overview.md) | Full project documentation, architecture, progress log |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Dataset curation, preprocessing pipeline, storage format |
| [docs/cluster-operations.md](docs/cluster-operations.md) | Practical cluster commands, queue monitoring, troubleshooting |
| [docs/slurm-longleaf-guide.md](docs/slurm-longleaf-guide.md) | SLURM fundamentals, script anatomy, auto-resume system |

## What's Next

- [ ] Build dataset on Longleaf (`sbatch longleaf/build-data.slurm`)
- [ ] Run score threshold ablation to validate filtering
- [ ] Full pretrain (50K steps, ~13B tokens, curated mixed dataset)
- [ ] Set up evaluation suite (lm-evaluation-harness)
- [ ] Complete supervised finetuning notebook
- [ ] Explore MoE routing in larger Qwen3 models
