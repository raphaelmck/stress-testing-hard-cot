#!/bin/bash
#SBATCH --job-name=r001_checks
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# R001 pre-launch GPU gates (runbook steps A, B, C) as one batch job.
#   A  --smoke   real Qwen3-32B numerical checks on 10 examples
#   B  --stress  representative ~1.5-2.5k-token batch at the configured budget
#   C  --stress  longest selected example (~16.5k tokens) alone at batch 1
# `a100l` is the 80 GB A100 (confirmed present in `main` via sinfo); plain `a100`
# may be the 40 GB variant and must not be used. h100 exists only in
# short-unkillable on this cluster.
#
# set -e: if A fails, B/C are not run and nothing downstream is launched.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "HF_HOME=$HF_HOME"

echo "===== A: smoke ====="
uv run python src/extract_task1_activations.py --smoke --run-id r001_qwen32b_smoke

echo "===== B/C: stress ====="
uv run python src/extract_task1_activations.py --stress --run-id r001_qwen32b_stress \
    --token-budget 16384 --max-batch 8

echo "checks done"
