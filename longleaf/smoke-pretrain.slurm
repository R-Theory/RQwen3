#!/bin/bash
#
# Smoke test: run 20 pretrain steps on TinyStories to verify
# the full pipeline works before committing to a 24-hour job.
#
# Submit with: make submit JOB=smoke-pretrain
#

#SBATCH --job-name=rqwen3-smoke
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=longleaf/logs/%j.out
#SBATCH --error=longleaf/logs/%j.err

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start time:   $(date)"
echo "========================================"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p longleaf/logs

module purge
module load python/3.12
module load cuda/12.2

source /work/users/t/r/treese20/envs/qwen3/bin/activate
export HF_HOME=/work/users/t/r/treese20/.cache/huggingface

echo ""
nvidia-smi
echo ""
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
echo ""

python longleaf/scripts/py/smoke_pretrain.py

echo ""
echo "========================================"
echo "End time:     $(date)"
echo "Exit code:    $?"
echo "========================================"
