#!/bin/bash
#SBATCH --job-name=m3faith
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3faith_%A_%a.log
set -euo pipefail
BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
M3="$HOME/mm_gradcam/module3_aligned"
source "$BASE/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
export MPLBACKEND=Agg
OUT="$M3/outputs/module3_faithfulness_primary_qc693_v1/raw"
printf -v I4 '%04d' "$SLURM_ARRAY_TASK_ID"
if [ -s "$OUT/faith_${I4}.csv" ] && [ -s "$OUT/faith_${I4}_audit.json" ]; then
    echo "SKIP: Existing completed result for index $SLURM_ARRAY_TASK_ID"
    exit 0
fi
python -u "$M3/scripts/module3_faithfulness.py" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --out-dir "$M3/outputs/module3_faithfulness_primary_qc693_v1/raw"
