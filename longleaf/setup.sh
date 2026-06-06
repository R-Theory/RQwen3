#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Longleaf One-Time Setup — Qwen3 Analysis
# ══════════════════════════════════════════════════════════════
#
# Run this ONCE on Longleaf to create directories, venv, and
# install packages. After this, you only need `make sync` and
# `make submit`.
#
# Usage (from your Mac):
#   make setup
#
# Or manually on Longleaf:
#   bash longleaf/setup.sh
#
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
WORKDIR="/work/users/${USER:0:1}/${USER:1:1}/$USER"
ENVDIR="$WORKDIR/envs/qwen3"

echo "══════════════════════════════════════════════════════════"
echo "  Longleaf Setup — Qwen3 Analysis"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  User:    $USER"
echo "  WORKDIR: $WORKDIR"
echo "  ENVDIR:  $ENVDIR"
echo ""

# ── Step 1: Create directory structure ───────────────────────
echo "[1/5] Creating directory structure on /work..."

mkdir -p "$WORKDIR"/{qwen3-analysis,data,checkpoints,logs,envs,snapshots}
mkdir -p "$WORKDIR/.cache/huggingface"

echo "  Created:"
echo "    $WORKDIR/qwen3-analysis/   (synced code)"
echo "    $WORKDIR/data/             (datasets)"
echo "    $WORKDIR/checkpoints/      (model checkpoints)"
echo "    $WORKDIR/logs/             (SLURM logs)"
echo "    $WORKDIR/envs/             (virtual environments)"
echo "    $WORKDIR/snapshots/        (weight snapshots)"
echo "    $WORKDIR/.cache/huggingface/ (HF model/dataset cache)"
echo ""

# ── Step 2: Load modules ────────────────────────────────────
echo "[2/5] Loading modules..."

module purge 2>/dev/null || true

# Try to load Python 3.11 (exact version may vary on Longleaf)
if module load python/3.11.6 2>/dev/null; then
    echo "  Loaded python/3.11.6"
elif module load python/3.11 2>/dev/null; then
    echo "  Loaded python/3.11"
elif module load python/3.12 2>/dev/null; then
    echo "  Loaded python/3.12"
else
    echo "  WARNING: Could not load Python module. Trying system python3..."
fi

# Try to load CUDA
if module load cuda/12.2 2>/dev/null; then
    echo "  Loaded cuda/12.2"
elif module load cuda/12.1 2>/dev/null; then
    echo "  Loaded cuda/12.1"
elif module load cuda/12 2>/dev/null; then
    echo "  Loaded cuda/12"
else
    echo "  WARNING: Could not load CUDA module (GPU jobs load it at runtime)"
fi

echo "  Python: $(python3 --version 2>&1)"
echo ""

# ── Step 3: Create virtual environment ──────────────────────
echo "[3/5] Setting up virtual environment..."

if [ -d "$ENVDIR" ]; then
    echo "  Venv already exists at $ENVDIR — skipping creation."
else
    echo "  Creating venv at $ENVDIR..."
    python3 -m venv "$ENVDIR"
    echo "  Created."
fi

source "$ENVDIR/bin/activate"
echo "  Activated: $(which python3)"
echo ""

# ── Step 4: Install packages ────────────────────────────────
echo "[4/5] Installing Python packages..."

pip install --upgrade pip --quiet
pip install torch transformers datasets accelerate numpy --quiet

echo "  Installed: torch, transformers, datasets, accelerate, numpy"
echo "  PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# ── Step 5: Configure .bashrc (idempotent) ──────────────────
echo "[5/5] Configuring ~/.bashrc..."

MARKER="# --- Qwen3 Analysis setup ---"

if grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    echo "  .bashrc already configured — skipping."
else
    cat >> ~/.bashrc << 'BASHRC'

# --- Qwen3 Analysis setup ---
export WORKDIR=/work/users/${USER:0:1}/${USER:1:1}/$USER
export HF_HOME=$WORKDIR/.cache/huggingface
module load python/3.12 2>/dev/null
module load cuda/12.2 2>/dev/null
BASHRC
    echo "  Added WORKDIR, HF_HOME, and module loads to ~/.bashrc"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "══════════════════════════════════════════════════════════"
echo ""
echo "  To activate the environment:"
echo "    source $ENVDIR/bin/activate"
echo ""
echo "  To verify GPU (on a compute node, not login node):"
echo "    make test   (from your Mac)"
echo ""
echo "  CUDA will show 'False' on the login node — that's normal."
echo "  It'll be 'True' on compute nodes when you submit GPU jobs."
echo ""
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()} (expected False on login node)')"
echo ""
