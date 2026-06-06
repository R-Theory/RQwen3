#!/bin/bash
#
# Build pre-tokenized training dataset for RQwen3
# CPU-only job — downloads, filters, deduplicates, and tokenizes
# all 6 data sources into binary shard files.
#
# Submit with: sbatch longleaf/build-data.slurm
#
# Expected runtime: 4-12 hours depending on network speed and source sizes.
# Output: /work/users/t/r/treese20/data/rqwen3-pretrain/v1/

# ── Job identity ──────────────────────────────────────────────
#SBATCH --job-name=build-data
#SBATCH --partition=general
#SBATCH --qos=gpu_access

# ── Compute (CPU only, no GPU) ────────────────────────────────
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ── Time limit ────────────────────────────────────────────────
#SBATCH --time=12:00:00

# ── Output logs ───────────────────────────────────────────────
#SBATCH --output=longleaf/logs/build-data_%j.out
#SBATCH --error=longleaf/logs/build-data_%j.err

# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Start time:   $(date)"
echo "========================================"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p longleaf/logs

module purge
module load python/3.12

source /work/users/t/r/treese20/envs/qwen3/bin/activate

export HF_HOME=/work/users/t/r/treese20/.cache/huggingface
export TOKENIZERS_PARALLELISM=true

# ══════════════════════════════════════════════════════════════
# BUILD DATASET
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR="/work/users/t/r/treese20/data/rqwen3-pretrain/v1"

python scripts/build_dataset.py \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer Qwen/Qwen3-0.6B \
    --seq-len 2048 \
    --shard-size 250000000

# ══════════════════════════════════════════════════════════════
# VERIFY
# ══════════════════════════════════════════════════════════════

echo ""
echo "========================================"
echo "Output contents:"
ls -lh "$OUTPUT_DIR"/train/ | head -20
echo "---"
ls -lh "$OUTPUT_DIR"/val/ | head -10
echo "---"
echo "Manifest:"
cat "$OUTPUT_DIR/manifest.json" | python -m json.tool | head -30
echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    $?"
echo "========================================"
