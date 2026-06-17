# Qwen3 Analysis

Building the Qwen3 architecture from scratch and training at scale. A hands-on deep dive into modern transformer innovations: RoPE, GQA, SwiGLU, RMSNorm, and MoE design.

**Status (2026-06-15):** Pretrain at ~35,170 / 50,000 steps (~70%), loss 11.88 → 2.53.

## Quick Start

### Local Development

```bash
# From the project root
source .venv/bin/activate
pip install -r requirements.txt

# Launch notebooks
jupyter lab
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
| 07 | `07-pretrain-rqwen3` | Full pretraining setup | Complete |

### `scripts/` — Training & Inference

| Script | Purpose |
|--------|---------|
| `scripts/py/TrainSession.py` | Training session — optimizer, scheduler, train loop, checkpointing, bf16 autocast on CUDA |
| `scripts/py/ChatSession.py` | Interactive chat with trained models |
| `scripts/build_dataset.py` | Build pre-tokenized training data from 6 curated sources |
| `scripts/stitch_manifest.py` | Re-scan shards & write a unified `manifest.json` after a multi-job dataset build |

## Training Progress

### Local (Complete)

- 200 steps on TinyStories, Apple Silicon MPS
- Checkpoints: `step_100.pt`, `step_200.pt`
- Snapshots: `untrained.pt` → `pre-trained.pt`

### Data Pipeline (Complete)

Built a 6-source, ~13B-token curated dataset: FineWeb-Edu (54%), Wikipedia (15%), OpenWebMath (12%), StackExchange (8%), peS2o (8%), textbooks (4%). Quality-filtered, deduplicated, pre-tokenized into binary shards. See [docs/data-pipeline.md](docs/data-pipeline.md).

### Pretraining (In Progress)

- **Full pretrain:** 50,000 steps · effective batch 128 (`batch_size=2 × grad_accum=64`) · bf16 autocast · SDPA attention
- **Progress (2026-06-15):** step ~35,170 / 50,000 (~70%); loss 11.88 → 2.53; projected final ~2.44

## Documentation

| Doc | What It Covers |
|-----|----------------|
| [docs/project-overview.md](docs/project-overview.md) | Full project documentation, architecture, progress log |
| [docs/data-pipeline.md](docs/data-pipeline.md) | Dataset curation, preprocessing pipeline, storage format |

## What's Next

- [x] Build curated dataset — ~13B tokens, 6 sources
- [x] Full pretrain (50K steps) — **in flight**, 70% complete
- [ ] Verify final checkpoint at step 50K, generate samples
- [ ] Set up evaluation suite (lm-evaluation-harness)
- [ ] Complete supervised finetuning notebook
- [ ] Checkpoint-pruning script
- [ ] Backlog: score-threshold ablation, domain-mixing ablation, MoE routing in larger Qwen3 models
