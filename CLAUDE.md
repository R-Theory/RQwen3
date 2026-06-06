# Qwen3 Analysis Project

## Context
Sub-project of Study Grounds, located in Heutagogy/projects/project-repos/.
Uses the parent Study Grounds .venv (Python 3.13.5, torch 2.10.0).
Apple Silicon (arm64) with MPS GPU acceleration available.

## What This Project Is
Exploratory analysis of the Qwen3 model family architecture, focused on:
- Mixture of Experts (MoE) design and expert routing
- Four-stage post-training pipeline
- Thinking/non-thinking mode unification
- Weight inspection and layer-by-layer analysis
- Hybrid attention (Qwen3-Next)

## Model Considerations
- Qwen3 models are large. Use small variants for local analysis (Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B).
- Models cache to ~/.cache/huggingface/ (NOT in this directory -- keep models out of iCloud).
- MPS backend works for inference but may have memory limits on smaller Macs.
- For larger models, use model inspection (config, architecture) without loading full weights.

## Key Files
- notebooks/ -- Jupyter notebooks for exploration
- src/data.py -- StreamingTokenDataset (prototyping) + PreTokenizedDataset (production)
- src/training.py -- TrainConfig, optimizer, scheduler, checkpointing (with data_offset)
- src/utils.py -- Shared helpers (device setup, model loading wrappers)
- scripts/build_dataset.py -- Build pre-tokenized dataset from 6 curated HF sources
- scripts/py/TrainSession.py -- Training session manager
- longleaf/build-data.slurm -- CPU-only SLURM job for dataset build
- longleaf/scripts/py/pretrain.py -- Cluster pretraining (auto-detects local pre-tokenized data)
- figures/ -- Saved visualizations

## Data Pipeline

Model goal: educational assistant (STEM: stats, CS, psych, data science).
6 sources (~13B tokens): FineWeb-Edu 54%, Wikipedia 15%, OpenWebMath 12%, StackExchange 8%, peS2o 8%, textbooks 4%.
Pre-tokenized as uint32 memmap shards. See docs/data-pipeline.md for full details.

## Conventions
- Notebooks numbered sequentially: 01-topic.ipynb, 02-topic.ipynb
- Use markdown cells heavily -- this is for learning, not just running code
- Always include a "What I Learned" section at the end of each notebook
