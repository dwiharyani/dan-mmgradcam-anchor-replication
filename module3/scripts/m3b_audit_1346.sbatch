#!/bin/bash
#SBATCH --job-name=m3b-audit1346
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_audit1346_%j.log

set -euo pipefail

BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
M3="$HOME/mm_gradcam/module3_aligned"

source "$BASE/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

python -u "$M3/scripts/m3b_audit_1346.py"
