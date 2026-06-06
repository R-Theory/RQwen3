---
tags:
  - longleaf
  - slurm
  - hpc
  - cluster
  - monitoring
  - troubleshooting
created: 2026-03-23
related:
  - "[[slurm-longleaf-guide]]"
  - "[[project-overview]]"
---

# Cluster Operations Guide

Practical guide for using UNC's Longleaf cluster. Based on real experience — every tip here comes from actually running into it.

> [!info] Prerequisites
> - Connected to UNC VPN (Cisco AnyConnect → `vpn.unc.edu`)
> - SSH configured (see O-Vault: `Projects/resources/LongLeaf/01 LongLeaf Login`)
> - One-time setup completed (`make setup`)

---

## 1. Make Commands — Your Daily Workflow

Everything goes through the Makefile. You don't need to remember rsync paths or SSH commands.

```bash
# ── Setup (one time) ─────────────────────────
make ssh              # Authenticate (opens shell, ctrl-d to exit)
make setup            # Create dirs + venv on Longleaf

# ── Code sync ────────────────────────────────
make sync             # Push code to cluster
make sync-dry         # Preview what would sync (no changes)

# ── Jobs ─────────────────────────────────────
make submit           # Sync + submit pretrain.slurm
make submit JOB=smoke-pretrain  # Sync + submit specific job
make test             # Sync + submit GPU smoke test
make status           # Check your job queue
make cancel JOB=38785256  # Cancel by job ID

# ── Logs ─────────────────────────────────────
make logs             # Tail the most recent log
make logs JOB=38785256  # Tail a specific job's log

# ── Results ──────────────────────────────────
make pull             # Download checkpoints, snapshots, logs

# ── Utilities ────────────────────────────────
make remote-ls        # List files on Longleaf
make help             # Show all commands
```

> [!tip] `make submit` automatically runs `make sync` first — no need to sync manually before submitting.

---

## 2. GPU Partitions on Longleaf

These are the actual partition names and GRES strings that work. Discovered via `sinfo`.

| Partition | GPU | VRAM | GPUs/Node | Nodes | GRES String |
|-----------|-----|------|-----------|-------|-------------|
| `a100-gpu` | A100 PCIe | 40 GB | 3 | 8 | `gpu:nvidia_a100-pcie-40gb:1` |
| `l40-gpu` | L40 / L40S | 48 GB | 4–8 | 18 | `gpu:nvidia_l40:1` or `gpu:nvidia_l40s:1` |
| `volta-gpu` | V100 SXM2 | 16 GB | 4 | 16 | `gpu:tesla_v100-sxm2-16gb:1` |
| `gpu` | GTX 1080 | 8 GB | 8 | 3 | `gpu:nvidia_geforce_gtx_1080:1` |

> [!warning] GRES names are NOT intuitive
> You can't just say `gpu:a100:1`. It must be `gpu:nvidia_a100-pcie-40gb:1`. To find the correct name:
> ```bash
> sinfo -p a100-gpu -o "%P %G"
> ```

### Which Partition to Use

| Your Model | Partition | Why |
|------------|-----------|-----|
| RQwen3 (751M) pretraining | `a100-gpu` | Needs 40GB for model + optimizer + activations |
| RQwen3 smoke test | `a100-gpu` | Same model, just fewer steps |
| Small inference tests | `volta-gpu` or `gpu` | Model fits in 16GB or even 8GB for inference only |
| Fine-tuning with QLoRA | `volta-gpu` | 4-bit quantized models fit in 16GB |

### Checking Live Availability

```bash
# Partition summary (idle/allocated/total nodes)
sinfo -p a100-gpu -o "%P %A %C %D %G"

# Per-node status
sinfo -p a100-gpu -N -o "%N %T %G %C"

# How many jobs are waiting
squeue -p a100-gpu -t PENDING | wc -l
```

---

## 3. Understanding the Queue

### Reading squeue Output

```bash
make status
# or
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.6D %R"
```

Output:
```
     JOBID                 NAME    STATE       TIME  NODES NODELIST(REASON)
  38785256         rqwen3-smoke  PENDING       0:00      1 (Priority)
```

| Column | What It Means |
|--------|---------------|
| `JOBID` | Unique ID — use this for `make cancel`, `make logs`, etc. |
| `NAME` | From `#SBATCH --job-name` in your `.slurm` file |
| `STATE` | Current state (see states table below) |
| `TIME` | How long it's been running (`0:00` if pending) |
| `NODES` | Number of nodes allocated |
| `NODELIST(REASON)` | Node name if running, or why it's pending |

### Job States

| State | Code | Meaning | What to Do |
|-------|------|---------|------------|
| PENDING | `PD` | Waiting in queue | Check reason, be patient |
| RUNNING | `R` | Executing on compute node | Monitor with `make logs` |
| COMPLETING | `CG` | Wrapping up | Wait a moment |
| COMPLETED | `CD` | Finished successfully | Check logs, `make pull` |
| FAILED | `F` | Exited with error | Read `.err` log |
| TIMEOUT | `TO` | Hit wall time limit | Normal — re-submit, auto-resume picks up |
| OUT_OF_MEMORY | `OOM` | Exceeded `--mem` | Reduce batch size or increase `--mem` |
| CANCELLED | `CA` | Cancelled by user or admin | Check if intentional |

### Pending Reasons

| Reason | What It Means | What to Do |
|--------|---------------|------------|
| `Priority` | Other jobs have higher priority | Wait — your turn will come |
| `Resources` | Requested resources not currently free | Wait — GPUs are all in use |
| `QOSMaxJobsPerUserLimit` | You've hit your job limit | Cancel old jobs or wait for them to finish |
| `ReqNodeNotAvail` | Requested node type doesn't exist | Check your GRES string with `sinfo` |
| `InvalidQOS` | Wrong or missing QOS | Add `--qos=gpu_access` to your SLURM script |

### Estimating Wait Time

```bash
# Check estimated start time for your job
squeue -j 38785256 --start

# Output:
#   JOBID  PARTITION  NAME      USER     ST  START_TIME           NODES  NODELIST(REASON)
#   38785256  a100-gpu  rqwen3-s  treese20  PD  2026-03-24T07:25:27  1  (Priority)
```

> [!tip] Estimated start times are rough — they change as other jobs finish early or new jobs get submitted.

### Seeing the Full Queue

```bash
# All jobs on a100-gpu (running + pending)
squeue -p a100-gpu -o "%.10i %.20j %.8T %.10M %.6D %R"

# Just running jobs (with elapsed time)
squeue -p a100-gpu -t RUNNING -o "%.10i %.20j %.10M %.10L %R"

# Count pending vs running
echo "Running:"; squeue -p a100-gpu -t RUNNING | wc -l
echo "Pending:"; squeue -p a100-gpu -t PENDING | wc -l
```

---

## 4. Backfill Scheduling

SLURM uses **backfill** to fill gaps in the schedule. This is why **short jobs often start faster** than long ones.

### How It Works

```
Timeline:
  ┌──────────────────────────────────┐
  │  Long job (24h) running...       │  ← Can't preempt this
  ├──────────────────────────────────┤
  │  Gap: 15 minutes before          │  ← Backfill opportunity!
  │  next scheduled long job         │
  └──────────────────────────────────┘

Your 15-min smoke test fits in the gap → starts immediately
Your 24-hour pretrain job doesn't fit → waits for full slot
```

### Why This Matters

Our smoke test (`--time=00:15:00`) has much better backfill odds than the real pretrain (`--time=24:00:00`). This is by design — validate the pipeline quickly, then commit to a long run.

> [!tip] If the queue is really congested, consider reducing `--time` to something realistic. A job that actually takes 6 hours doesn't need a 24-hour wall time — and a shorter request may start sooner via backfill.

---

## 5. Monitoring Running Jobs

### Live Log Tailing

```bash
# Via Makefile (tails the most recent log)
make logs

# Specific job
make logs JOB=38785256

# Direct SSH (if make doesn't work)
ssh longleaf "tail -f /work/users/t/r/treese20/qwen3-analysis/longleaf/logs/38785256.out"
```

### What to Look for in Logs

A healthy training run shows:

```
========================================
Job ID:       38785256
Node:         g141604
GPU:          0
Start time:   Mon Mar 24 07:25:27 EDT 2026
========================================

PyTorch 2.x.x, CUDA available: True, GPU: NVIDIA A100-PCIe-40GB

============================================================
  Pretraining RQwen3 (751.0M)
  Dataset: HuggingFaceFW/fineweb-edu
  Device: cuda
============================================================

Training for 10000 steps (batch=4, accum=32, effective batch=128)
  step     10/10000 | loss: 12.1432 | lr: 6.00e-05    ← loss should decrease
  step     20/10000 | loss: 11.8921 | lr: 1.20e-04
  step     30/10000 | loss: 11.5103 | lr: 1.80e-04
  ...
```

**Red flags:**
- `CUDA available: False` → GPU not allocated, check GRES
- `ModuleNotFoundError` → venv not activated
- `CUDA out of memory` → reduce batch_size
- Loss is `nan` → learning rate too high or data issue
- Loss not decreasing after 100+ steps → check learning rate schedule

### Resource Usage

```bash
# While job is running
sstat -j 38785256 --format=JobID,MaxRSS,AveCPU

# After job completes
sacct -j 38785256 --format=JobID,State,ExitCode,MaxRSS,Elapsed,CPUTime

# Detailed breakdown
sacct -j 38785256 --format=JobID,JobName,Partition,State,ExitCode,MaxRSS,MaxVMSize,Elapsed,TotalCPU
```

| Field | What It Tells You |
|-------|-------------------|
| `MaxRSS` | Peak RAM usage — compare to your `--mem` request |
| `Elapsed` | Total wall time used |
| `ExitCode` | `0:0` = success, anything else = error |
| `State` | How the job ended (COMPLETED, FAILED, TIMEOUT, etc.) |

---

## 6. Commands Cheat Sheet

### Submitting & Managing

```bash
# Submit
make submit                           # Sync + submit pretrain
make submit JOB=smoke-pretrain        # Sync + submit smoke test
sbatch longleaf/pretrain.slurm        # Direct (no sync)

# Monitor
make status                           # Your jobs
squeue -u $USER                       # Your jobs (verbose)
squeue -p a100-gpu                    # All A100 jobs
squeue -j 38785256 --start            # Estimated start time
watch -n 30 'squeue -u $USER'         # Auto-refresh every 30s

# Logs
make logs                             # Tail latest
make logs JOB=38785256                # Tail specific
cat longleaf/logs/38785256.err        # Check errors (after pull)

# Control
scancel 38785256                      # Cancel one job
scancel -u $USER                      # Cancel all your jobs
scontrol hold 38785256                # Pause in queue
scontrol release 38785256             # Resume in queue
```

### Cluster Info

```bash
# Partitions
sinfo                                 # All partitions
sinfo -p a100-gpu                     # A100 partition
sinfo -p a100-gpu -N -o "%N %T %G"   # Per-node GPU status

# Resources
sinfo -p a100-gpu -o "%P %A %C %D %G" # Summary
sacctmgr show qos                     # Available QOS values
module avail python                    # Available Python versions
module avail cuda                      # Available CUDA versions

# Usage accounting
sacct -j 38785256                      # Job resource usage
sstat -j 38785256                      # Running job stats
```

### On the Login Node

```bash
# Check your storage
quota -s                               # Home directory quota
du -sh /work/users/t/r/treese20/*     # Work directory sizes

# Environment
source /work/users/t/r/treese20/envs/qwen3/bin/activate
python -c "import torch; print(torch.__version__)"

# Check what's on the cluster
ls -la /work/users/t/r/treese20/qwen3-analysis/
```

---

## 7. Troubleshooting

Real issues we hit and how we fixed them.

### "Requested node configuration is not available"

**Cause:** Wrong GRES name.

```bash
# What we had (wrong):
#SBATCH --gres=gpu:a100:1

# What works:
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
```

**How to find correct names:**
```bash
sinfo -p a100-gpu -o "%P %G"
```

### "Invalid qos specification"

**Cause:** Missing or wrong QOS.

```bash
# Fix: add this to your SLURM script
#SBATCH --qos=gpu_access

# To see available QOS values:
sacctmgr show qos format=name,priority
```

### "ModuleNotFoundError: No module named 'torch'"

**Cause:** Virtual environment not activated in the SLURM script.

```bash
# Add this to your SLURM script:
source /work/users/t/r/treese20/envs/qwen3/bin/activate
```

> [!warning] SLURM jobs do NOT source your `.bashrc`. Everything must be explicit in the job script — module loads, venv activation, environment variables.

### "CUDA out of memory"

**Cause:** Batch size too large for GPU VRAM.

Longleaf A100s are **40GB PCIe**, not 80GB SXM. Memory budget:

| Component | Size |
|-----------|------|
| Model (751M × 4 bytes) | ~3 GB |
| Gradients | ~3 GB |
| Optimizer (AdamW: momentum + variance) | ~6 GB |
| Activations (varies with batch × seq_len) | ~2–10 GB |
| **Total** | **~14–22 GB** |

**Safe config for 40GB:**
```python
batch_size = 4
grad_accum_steps = 32   # effective batch = 128
```

### Module Version Errors (Lmod)

**Cause:** Requesting a module version that doesn't exist.

```bash
# Check what's actually available:
module avail python
module avail cuda

# We use:
module load python/3.12
module load cuda/12.2
```

> [!tip] If `.bashrc` has stale module loads from a previous setup, fix them:
> ```bash
> nano ~/.bashrc   # find and update the version numbers
> ```

### Job Produces No Output

**Cause:** `longleaf/logs/` directory doesn't exist.

```bash
# The SLURM script creates it, but if you're running manually:
mkdir -p longleaf/logs
```

### Loss is NaN

| Possible Cause | Fix |
|----------------|-----|
| Learning rate too high | Reduce `learning_rate` (try `1e-4` instead of `3e-4`) |
| No gradient clipping | Add `max_grad_norm=1.0` to training config |
| Data issue | Check dataset loading — are tokens valid? |
| Numerical overflow | Try mixed precision (`torch.autocast`) |

---

## 8. Best Practices

### Always Smoke Test First

Before committing to a 24-hour job:

```bash
make submit JOB=smoke-pretrain  # 20 steps, 15 minutes
make status                      # Wait for it to start
make logs                        # Watch output
```

A smoke test verifies:
- Environment activates correctly
- All imports resolve
- Data loading works
- Model fits in GPU memory
- Forward/backward pass works
- Checkpoint saving works

If the smoke test passes, the real job will too.

### Efficient Queue Management

- **Submit during off-peak hours** — late night and weekends have shorter queues
- **Use realistic wall times** — don't request 24h for a 6h job (hurts backfill chances)
- **Short wall time = faster start** — 15-minute smoke tests often get backfilled
- **Check queue before submitting** — `squeue -p a100-gpu -t PENDING | wc -l`

### Storage Hygiene

```bash
# Check disk usage
du -sh /work/users/t/r/treese20/*

# Clean old checkpoints (keep latest + final)
ls -lt /work/users/t/r/treese20/qwen3-analysis/checkpoints/

# Pull results regularly
make pull
```

- **Code** lives on `/work` (synced via `make sync`)
- **HF_HOME** set to `/work/.../cache/huggingface` (not home dir)
- **Checkpoints** saved to `/work/.../checkpoints/`
- **Logs** in `longleaf/logs/` — pull with `make pull`

### Multi-Day Training

For training runs longer than 24 hours:

1. SLURM sends `SIGTERM` 120 seconds before wall time
2. `pretrain.py` catches it and saves an emergency checkpoint
3. Re-submit the same job — it auto-resumes from the latest checkpoint
4. Repeat until training completes

```bash
# Day 1: Start fresh
make submit

# Day 2: Auto-resumes from checkpoint
make submit

# Day 3: Continues until done
make submit
```

See [[slurm-longleaf-guide#4. Auto-Resume & Graceful Shutdown]] for details.
