#!/bin/bash
#SBATCH --job-name=m3b-off
#SBATCH --partition=cpu_x440
#SBATCH --array=0-13%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_off_%A_%a.log

set -euo pipefail

BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
SRC="$HOME/mm_gradcam/module3_sources"
M3="$HOME/mm_gradcam/module3_aligned"
RUN="$SRC/module2_runtime"

source "$BASE/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

START=$((SLURM_ARRAY_TASK_ID * 50))
printf -v CHUNK "%04d" "$START"

OUT="$M3/outputs/m3b_full694_onednn_off_chunks/trace_${CHUNK}.npz"

if [ -e "$OUT" ]; then
    echo "ERROR: Output already exists: $OUT"
    exit 1
fi

echo "=== MODULE 3B FULL 694 ==="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Start index: $START"
echo "oneDNN disabled"
echo "Output: $OUT"

python -u "$M3/scripts/m3b_generate_trace_support.py" \
    --m2-dir "$M3/inputs/module2_s9534" \
    --m3a-dir "$M3/outputs/module3a_v1" \
    --runtime-root "$RUN" \
    --module2-common "$SRC/module2_review/scripts/module2_common.py" \
    --start "$START" \
    --limit 50 \
    --out "$OUT"

echo "SUCCESS: $OUT"
