# SLURM & Longleaf Guide

A practical guide for training RQwen3 on UNC's Longleaf cluster. Covers what SLURM is, how our `pretrain.slurm` works, and everything you need to submit and manage jobs.

---

## 1. What is SLURM?

Longleaf is a shared cluster — hundreds of UNC researchers use the same GPUs. You can't just SSH in and run `python train.py`. That would be like walking into a library computer lab and hogging a machine with no time limit.

**SLURM** (Simple Linux Utility for Resource Management) is the **job scheduler** that manages this. It works like a queue:

```
You write a job script
    ↓
sbatch submits it to the queue
    ↓
SLURM waits until your requested resources are free
    ↓
Your script runs on a compute node with the GPU/memory you asked for
    ↓
Output is saved to log files you can read later
```

### Key Terminology

| Term | What it means |
|------|---------------|
| **Job** | A single unit of work you submit (one training run) |
| **Job script** | A `.slurm` file — bash script with resource requests at the top |
| **Login node** | The machine you SSH into. Shared by everyone — do NOT run training here |
| **Compute node** | The actual machine with GPUs where your job runs |
| **Partition** | A pool of nodes grouped by type (e.g., `gpu` = nodes with GPUs) |
| **QoS** | "Quality of Service" — permission level. `gpu_access` is required for GPU jobs |
| **Wall time** | Maximum time your job can run before SLURM kills it |
| **Job ID** | A number SLURM assigns when you submit (e.g., `12345678`) |

### The Two Types of Nodes

```
longleaf.unc.edu  (Login Node)
├── This is where you:
│   ├── Write and edit code
│   ├── Submit jobs with sbatch
│   ├── Check job status with squeue
│   └── Read log files
│
└── You NEVER run training here
    (it's shared, has no GPUs, and admins will kill your process)

Compute Nodes  (Where jobs actually run)
├── SLURM assigns one to your job
├── Has the GPU/memory you requested
├── Your script runs here automatically
└── You don't SSH into these directly
```

---

## 2. Anatomy of a SLURM Script

A `.slurm` file is just a bash script with special `#SBATCH` comments at the top. SLURM reads these comments to know what resources you need. Everything below them is regular bash that runs on the compute node.

```bash
#!/bin/bash

# ═══ PART 1: Resource requests (SLURM reads these) ═══
#SBATCH --job-name=my-training       # label shown in squeue
#SBATCH --partition=a100-gpu          # A100 GPU partition
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1  # 1 A100 40GB
#SBATCH --cpus-per-task=8            # 8 CPU cores
#SBATCH --mem=64G                    # 64GB system RAM
#SBATCH --time=24:00:00              # max 24 hours
#SBATCH --output=logs/%j.out         # stdout → logs/<jobid>.out

# ═══ PART 2: Setup (bash, runs on the compute node) ═══
module load python/3.12              # make Python available
source ~/myenv/bin/activate          # activate virtual environment

# ═══ PART 3: Your actual work ═══
python train.py
```

### Common `#SBATCH` Flags

| Flag | What it controls | Our value | Why |
|------|-----------------|-----------|-----|
| `--job-name` | Name shown in `squeue` | `rqwen3-pretrain` | Easy to identify your job |
| `--partition` | Which pool of nodes | `a100-gpu` | A100 GPU partition |
| `--qos` | Permission level | `gpu_access` | Required for GPU partition |
| `--gres` | GPU type and count | `gpu:nvidia_a100-pcie-40gb:1` | 1 A100 (40GB VRAM) |
| `--cpus-per-task` | CPU cores | `8` | For DataLoader workers |
| `--mem` | System RAM | `64G` | Enough for data loading + model |
| `--time` | Wall time limit | `24:00:00` | Maximum 24 hours |
| `--output` | Where stdout goes | `longleaf/logs/%j.out` | `%j` = job ID |
| `--error` | Where stderr goes | `longleaf/logs/%j.err` | Errors in separate file |
| `--signal` | Pre-kill warning signal | `B:TERM@120` | SIGTERM 120s before wall time |

> **Important:** If your job exceeds `--time` or `--mem`, SLURM kills it immediately. Always set these with a buffer.

---

## 3. Our `pretrain.slurm` — Line by Line

Here's every section of `longleaf/pretrain.slurm` and what it does.

### Job Identity

```bash
#SBATCH --job-name=rqwen3-pretrain
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
```

- **job-name**: What shows up when you run `squeue`. Pick something you'll recognize.
- **partition**: `a100-gpu` targets the A100 partition specifically (8 nodes, 3 GPUs each = 24 total A100s).
- **qos**: `gpu_access` is required on Longleaf to use GPU partitions. Without it, your job gets rejected.

### GPU & Compute

```bash
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

- **gres**: "Generic resource" — `gpu:nvidia_a100-pcie-40gb:1` means "1 A100 40GB GPU." The GRES name is cluster-specific — find yours with `sinfo -p <partition> -o "%G"`.
- **cpus-per-task**: 8 CPU cores. Our DataLoader uses `num_workers=4`, but we request 8 to leave headroom for the main process and data preprocessing.
- **mem**: 64GB system RAM (not GPU memory — GPU VRAM is separate). This covers the dataset streaming, tokenization, and any CPU-side processing.

### Time Limit

```bash
#SBATCH --time=24:00:00
```

Maximum 24 hours. After this, SLURM kills the job. Our training is ~10,000 steps which takes roughly 20-24 hours on an A100 40GB with batch_size=4 — the graceful shutdown system (next section) handles cases where training isn't done.

### Graceful Shutdown

```bash
#SBATCH --signal=B:TERM@120
```

This is the key to multi-day training. It tells SLURM: "Send a SIGTERM signal to my process 120 seconds before the wall time expires."

- `B` = send to the batch script's child process (our Python script)
- `TERM` = the signal type (SIGTERM)
- `@120` = 120 seconds before the `--time` limit

Our `pretrain.py` catches this signal and saves an emergency checkpoint. Without this, SLURM would just kill the process and you'd lose everything since the last regular checkpoint.

### Output Logs

```bash
#SBATCH --output=longleaf/logs/%j.out
#SBATCH --error=longleaf/logs/%j.err
```

- **output**: Everything your script `print()`s goes here. Loss values, sample generations, checkpoint confirmations — all captured.
- **error**: Python tracebacks, warnings, and stderr go here.
- **`%j`**: SLURM replaces this with the job ID, so each run gets its own files (e.g., `longleaf/logs/12345678.out`).

> **The `longleaf/logs/` directory must exist before you submit.** If it doesn't, the job fails silently with no output. The script creates it with `mkdir -p`, but it's good to know why.

### Email Notifications (Optional)

```bash
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=YOUR_ONYEN@email.unc.edu
```

These are commented out with `##` (double hash = inactive). To enable:
1. Remove one `#` so they become `#SBATCH`
2. Replace `YOUR_ONYEN` with your actual ONYEN

You'll get emails when the job starts, finishes, or fails. Useful for long runs so you don't have to keep checking `squeue`.

### Setup Section

```bash
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p longleaf/logs

module purge
module load python/3.12
module load cuda/12.2
```

- **`$SLURM_SUBMIT_DIR`**: The directory you were in when you ran `sbatch`. This ensures paths work correctly.
- **`module purge`**: Clears any previously loaded modules (clean slate).
- **`module load`**: Makes Python and CUDA available. These are system-managed installations — you don't install them yourself.

> Check what versions are available with `module avail python` and `module avail cuda`. The exact version numbers may differ.

```bash
# Activate your environment
source /work/users/t/r/treese20/envs/qwen3/bin/activate
export HF_HOME=/work/users/t/r/treese20/.cache/huggingface
```

This activates the Python environment with PyTorch, transformers, etc. and sets the HuggingFace cache to `/work` (home directory has a small quota).

### GPU Verification

```bash
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, ...')"
```

Diagnostic output that shows up in your `.out` log. If the job fails, this tells you whether the GPU was actually available.

### Memory Budget (A100 40GB)

Longleaf's A100s are **40GB PCIe**, not 80GB SXM4. Batch sizes must be set accordingly:

| Component | Memory |
|-----------|--------|
| Model (751M params × 4 bytes) | ~3 GB |
| Gradients (same size) | ~3 GB |
| Optimizer state (AdamW: momentum + variance) | ~6 GB |
| Activations (batch_size=4, seq_len=2048) | ~2-6 GB |
| **Total** | **~14-18 GB** |

Safe config for 40GB A100:

```python
batch_size = 4              # per-device micro-batch
grad_accum_steps = 32       # effective batch = 4 × 32 = 128
```

### Training

```bash
python longleaf/scripts/py/pretrain.py
```

Runs the training script. Everything else (model creation, data streaming, checkpointing, auto-resume) is handled by `pretrain.py` itself.

---

## 4. Auto-Resume & Graceful Shutdown

This system lets you train for longer than any single job's wall time. Here's how the pieces fit together.

### The Problem

SLURM kills your job after `--time` expires. If training takes 30 hours but your wall time is 24 hours, you lose 6 hours of work unless you save progress.

### The Solution: Three Mechanisms

**1. Regular checkpoints** — `pretrain.py` saves a full checkpoint every 1,000 steps (roughly every 2.4 hours). These go to `checkpoints/step_1000.pt`, `step_2000.pt`, etc.

**2. Emergency checkpoint on SIGTERM** — 120 seconds before wall time, SLURM sends SIGTERM. Our signal handler catches it:

```python
def handle_sigterm(signum, frame):
    # Saves checkpoint at whatever step we're on
    session.save_checkpoint()
    sys.exit(0)
```

This means you lose at most a few minutes of work, not the 2+ hours since the last regular checkpoint.

**3. Auto-resume on next submission** — When `pretrain.py` starts, it scans the checkpoint directory:

```python
def find_latest_checkpoint(checkpoint_dir):
    # Finds step_*.pt with the highest step number
    # Returns the path, or None if no checkpoints exist
```

If a checkpoint is found, it resumes from there (loading model weights, optimizer state, scheduler position, loss history). If not, it starts fresh.

### The Multi-Day Training Workflow

```
Day 1:
  sbatch longleaf/pretrain.slurm
  → Starts fresh at step 0
  → Saves checkpoints at steps 1000, 2000, ..., 9000
  → At 23:58:00, SIGTERM fires → saves step 9500
  → Job exits cleanly

Day 2:
  sbatch longleaf/pretrain.slurm     ← same command
  → Finds checkpoints/step_9500.pt
  → Resumes from step 9500
  → Completes remaining 500 steps
  → Saves final checkpoint
  → Done
```

You just keep re-submitting the same `sbatch` command until training finishes. Each run picks up where the last one left off.

---

## 5. Essential SLURM Commands

### Submitting & Managing Jobs

```bash
# Submit a job
sbatch longleaf/pretrain.slurm

# Check your jobs (running or waiting)
squeue -u $USER

# Cancel a job
scancel 12345678

# Cancel ALL your jobs
scancel -u $USER
```

### Reading `squeue` Output

```
JOBID    PARTITION  NAME              ST  TIME      NODES  NODELIST
12345678 gpu        rqwen3-pretrain   R   03:42:15  1      c0301
```

| Column | Meaning |
|--------|---------|
| JOBID | Your job's unique ID |
| ST | State: `PD` = pending (waiting), `R` = running, `CG` = completing |
| TIME | How long it's been running |
| NODELIST | Which compute node it's on |

### Reading Job Logs

```bash
# While the job is running — live-follow the output
tail -f longleaf/logs/12345678.out

# Read the full output after the job finishes
cat longleaf/logs/12345678.out

# Check for errors
cat longleaf/logs/12345678.err

# See resource usage of a completed job
sacct -j 12345678 --format=JobID,Elapsed,MaxRSS,MaxVMSize
```

Your training logs will look something like:

```
========================================
Job ID:       12345678
Node:         c0301
GPU:          0
Start time:   Wed Mar  5 14:00:00 EST 2026
========================================

PyTorch 2.x.x, CUDA available: True, GPU: NVIDIA A100-PCIe-40GB

No checkpoint found — starting fresh.

============================================================
  Pretraining RQwen3 (751.0M)
  Dataset: HuggingFaceFW/fineweb-edu
  Tokens per step: 4,194,304
  Device: cuda
  Starting from step: 0
============================================================

Training for 10000 steps (batch=4, accum=32, effective batch=128)
LR: 0.0003 -> cosine decay, warmup: 500 steps
  step     10/10000 | loss: 12.1432 | lr: 6.00e-05
  step     20/10000 | loss: 11.8921 | lr: 1.20e-04
  ...
```

### Checking Cluster Status

```bash
# See available partitions and node status
sinfo -p a100-gpu

# See what GPUs are available right now
sinfo -p a100-gpu -o "%N %G %t"
```

---

## 6. Before You Submit — Checklist

Run through this before your first `sbatch`:

- [ ] **SSH into Longleaf**: `ssh longleaf` (or `make ssh`)
- [ ] **Your project code is on `/work`**: `make sync` to push code
- [ ] **Python environment exists**: `source /work/users/t/r/treese20/envs/qwen3/bin/activate && python -c "import torch"`
- [ ] **Correct partition & GRES**: `--partition=a100-gpu`, `--gres=gpu:nvidia_a100-pcie-40gb:1`
- [ ] **Correct QOS**: `--qos=gpu_access`
- [ ] **Correct modules**: `python/3.12`, `cuda/12.2`
- [ ] **Venv activation in SLURM script**: Not commented out, correct path set
- [ ] **Batch size fits GPU**: `batch_size=4` for A100 40GB (not 16)
- [ ] **Logs directory exists**: `mkdir -p longleaf/logs`
- [ ] **Optional — enable email**: Uncomment the `--mail-type` and `--mail-user` lines

Then submit:

```bash
make submit   # syncs code + submits pretrain.slurm
```

---

## 7. Common Issues & Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Job stuck in `PD` forever | `squeue` shows `PD` for hours | Check partition name (`sinfo`), ensure `--qos=gpu_access` is set |
| Job fails with no output | No `.out` or `.err` file created | `longleaf/logs/` directory doesn't exist — run `mkdir -p longleaf/logs` |
| `ModuleNotFoundError: No module named 'torch'` | Error in `.err` log | Venv not activated in the SLURM script — uncomment the activation line |
| `CUDA out of memory` | Error in `.err` log | Reduce `batch_size` in `pretrain.py` — A100 40GB fits `batch_size=4` with RQwen3 |
| `module load` fails | "Module not found" error | Run `module avail python` to see what's actually available, update version |
| Job killed at wall time | Ends abruptly, no "Done" message | Normal — the SIGTERM handler saved a checkpoint. Just re-submit. |
| Job ran but loss isn't going down | Loss stays flat in `.out` log | Check learning rate, try `3e-4`. If loss is `nan`, gradient clipping may need adjustment. |
| Can't find checkpoints | `checkpoints/` is empty | Check `checkpoint_dir` path in `pretrain.py` — make sure it points to a writable location on `/work` |

### Debugging a Failed Job

```bash
# Step 1: Check the error log
cat longleaf/logs/<jobid>.err

# Step 2: Check resource usage (did it run out of memory?)
sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS,Elapsed

# Step 3: Check if the job even started
sacct -j <jobid> --format=JobID,State,Start,End,ExitCode
```

---

## Quick Reference Card

```bash
# ── Submit & Monitor ──────────────────────────
sbatch longleaf/pretrain.slurm       # submit
squeue -u $USER                      # check status
tail -f longleaf/logs/<jobid>.out    # live output
scancel <jobid>                      # cancel

# ── After Job Finishes ────────────────────────
cat longleaf/logs/<jobid>.out        # full output
cat longleaf/logs/<jobid>.err        # errors
sacct -j <jobid>                     # resource usage

# ── Cluster Info ──────────────────────────────
sinfo -p a100-gpu                         # GPU partition status
module avail python                  # available Python versions
module avail cuda                    # available CUDA versions
```
