#!/bin/bash
# ══════════════════════════════════════════════════════════════
# GPU Smoke Test — verify the full pipeline works
# ══════════════════════════════════════════════════════════════
#
# Requests a GPU for 10 minutes, checks:
#   1. nvidia-smi sees the GPU
#   2. PyTorch sees CUDA
#   3. src/ imports work
#   4. Model can do a forward pass on CUDA
#
# Submit with:   make test
# Check output:  make logs
#
# ══════════════════════════════════════════════════════════════

#SBATCH --job-name=gpu-test
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=longleaf/logs/%j.out
#SBATCH --error=longleaf/logs/%j.err

echo "========================================"
echo "GPU Smoke Test"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Time:      $(date)"
echo "========================================"
echo ""

# Create logs dir if it doesn't exist
mkdir -p longleaf/logs

# ── Setup ────────────────────────────────────────────────────
module purge
module load python/3.11 2>/dev/null || module load python/3.11.6 2>/dev/null
module load cuda/12.1 2>/dev/null || module load cuda/12.2 2>/dev/null

WORKDIR=/work/users/${USER:0:1}/${USER:1:1}/$USER
source "$WORKDIR/envs/qwen3/bin/activate"

PASS=0
FAIL=0

# ── Test 1: nvidia-smi ──────────────────────────────────────
echo "[Test 1] nvidia-smi"
if nvidia-smi; then
    echo "  PASS"
    PASS=$((PASS + 1))
else
    echo "  FAIL — nvidia-smi not available"
    FAIL=$((FAIL + 1))
fi
echo ""

# ── Test 2: PyTorch CUDA ────────────────────────────────────
echo "[Test 2] PyTorch CUDA detection"
python3 -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  VRAM:            {props.total_memory / 1e9:.1f} GB')
    print('  PASS')
else:
    print('  FAIL — CUDA not available')
    exit(1)
"
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
echo ""

# ── Test 3: src/ imports ────────────────────────────────────
echo "[Test 3] Project imports"
python3 -c "
import sys, os
sys.path.insert(0, '.')
from src import CoreConfig, RQwen3, get_device
print(f'  CoreConfig: OK')
print(f'  RQwen3:     OK')
print(f'  get_device: {get_device()}')
print('  PASS')
"
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
echo ""

# ── Test 4: Forward pass on CUDA ────────────────────────────
echo "[Test 4] Forward pass on GPU"
python3 -c "
import sys, os
sys.path.insert(0, '.')
import torch
from src import CoreConfig, RQwen3

# Tiny config — just to test the pipeline
config = CoreConfig(
    d_model=128,
    n_layer=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=32,
    intermediate_size=512,
    max_seq_len=64,
)
model = RQwen3(config).cuda()
params = sum(p.numel() for p in model.parameters())
print(f'  Model params:   {params:,}')

# Random input
x = torch.randint(0, 1000, (1, 32)).cuda()
with torch.no_grad():
    out = model(x)
print(f'  Output shape:   {out.shape}')
print(f'  Output device:  {out.device}')
print('  PASS')
"
if [ $? -eq 0 ]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "========================================"
if [ $FAIL -eq 0 ]; then
    echo "ALL TESTS PASSED ($PASS/$PASS)"
else
    echo "TESTS: $PASS passed, $FAIL failed"
fi
echo "========================================"
