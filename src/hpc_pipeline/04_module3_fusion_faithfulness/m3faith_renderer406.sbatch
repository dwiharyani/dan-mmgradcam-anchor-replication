#!/bin/bash
#SBATCH --job-name=m3-render406
#SBATCH --partition=cpu_x440
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3render406_%A_%a.log

set -euo pipefail

source "$HOME/mm_gradcam/MM_GradCAM_HPC/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    export CASE=fresh
else
    export CASE=warm
fi

python -u \
  "$HOME/mm_gradcam/module3_aligned/scripts/m3faith_renderer406.py"
