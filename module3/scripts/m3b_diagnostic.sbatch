#!/bin/bash
#SBATCH --job-name=m3b-diag
#SBATCH --partition=cpu_x440
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_diag_%A_%a.log

set -euo pipefail

BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
SRC="$HOME/mm_gradcam/module3_sources"
M3="$HOME/mm_gradcam/module3_aligned"
RUN="$SRC/module2_runtime"

IDS=(1162 551 780)
CID="${IDS[$SLURM_ARRAY_TASK_ID]}"

source "$BASE/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

echo "=== DIAGNOSTIC ECG $CID ==="
echo "TensorFlow oneDNN disabled"
echo "Original QC tolerances retained"

python -u "$M3/scripts/m3b_generate_trace_support.py" \
  --m2-dir "$M3/inputs/module2_s9534" \
  --m3a-dir "$M3/outputs/module3a_v1" \
  --runtime-root "$RUN" \
  --module2-common "$SRC/module2_review/scripts/module2_common.py" \
  --only-ids "$CID" \
  --out "$M3/outputs/m3b_diagnostic/trace_${CID}_onednn_off.npz"

echo "PASS: ECG $CID"
