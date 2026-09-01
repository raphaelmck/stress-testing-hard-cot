#!/bin/bash
#SBATCH --job-name=r001_extract
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# R001: Qwen3-32B last-real-prefix-token activation extraction.
#
# VERIFY BEFORE FIRST USE:
#   sinfo -o "%20P %10G" | sort -u      # confirm the gres name for an 80 GB card
# Qwen3-32B needs 65.5 GB in bf16. Use h100 or a100l (80 GB); plain `a100` may be
# the 40 GB variant, on which this job will ABORT by design rather than offload.
#
# Resumable: if this job is pre-empted or times out, resubmit it unchanged. It
# skips rows already written to shards and continues.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "HF_HOME=$HF_HOME"

uv run python src/extract_task1_activations.py \
    --run-id r001_qwen32b \
    --token-budget 16384 \
    --max-batch 8

# Probe fitting needs no GPU, but running it here means the job produces the
# complete R001 artifact set in one go.
uv run python src/fit_task1_probe.py --run-id r001_qwen32b

echo "done"
