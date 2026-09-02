#!/bin/bash
#SBATCH --job-name=r007_steer
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# R007 (D013): Stage 1 val propagation smoke, then the frozen Stage 2 test run.
# set -e: Stage 2 never runs if the mandatory propagation checks fail.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "===== STAGE 1: val propagation smoke ====="
uv run python src/steer_termination.py --smoke --run-id r007_steer_smoke

echo "===== STAGE 2: frozen test run (once) ====="
uv run python src/steer_termination.py --run-id r007_steer_test
echo "done"
