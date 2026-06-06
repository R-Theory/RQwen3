#!/bin/bash
#
# Pretrain RQwen3 (Qwen3-0.6B scale) on Longleaf
# Submit with: sbatch longleaf/pretrain.slurm
#

# ── Job identity ──────────────────────────────────────────────
#SBATCH --job-name=rqwen3-pretrain
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access

# ── GPU & compute ────────────────────────────────────────────
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ── Time limit ───────────────────────────────────────────────
#SBATCH --time=24:00:00

# ── Graceful shutdown ────────────────────────────────────────
# Send SIGTERM 120 seconds before wall time expires.
# pretrain.py catches this and saves an emergency checkpoint.
#SBATCH --signal=B:TERM@120

# ── Output logs ──────────────────────────────────────────────
# %j = job ID — each run gets its own log file
#SBATCH --output=longleaf/logs/%j.out
#SBATCH --error=longleaf/logs/%j.err

# ── Email notifications (optional — fill in your UNC email) ──
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=YOUR_ONYEN@email.unc.edu

# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

# Print job info for debugging
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "========================================"

# Move to project root (where src/ and longleaf/ live)
cd "$SLURM_SUBMIT_DIR" || exit 1

# Create logs directory if it doesn't exist
mkdir -p longleaf/logs

# Load modules — adjust these to match what's available on Longleaf
module purge
module load python/3.12
module load cuda/12.2

# Activate your environment
source /work/users/t/r/treese20/envs/qwen3/bin/activate

# Cache HF models/datasets on /work (home has small quota)
export HF_HOME=/work/users/t/r/treese20/.cache/huggingface

# Verify GPU is visible
echo ""
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo ""

# ══════════════════════════════════════════════════════════════
# RUN TRAINING
# ══════════════════════════════════════════════════════════════

# The pretrain script handles everything:
#   - Model creation (751M params, Qwen3-0.6B architecture)
#   - Streaming data from HuggingFace (fineweb-edu)
#   - Checkpointing every 1000 steps to checkpoints/
#   - Sample generation every 500 steps
#
# Auto-resume: if checkpoints exist, training picks up from the
# latest one. Just re-submit the same job after wall time.
#
# Graceful shutdown: SLURM sends SIGTERM 120s before wall time.
# The script catches it and saves an emergency checkpoint.

python longleaf/scripts/py/pretrain.py

# ══════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════

echo ""
echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    $?"
echo "========================================"
