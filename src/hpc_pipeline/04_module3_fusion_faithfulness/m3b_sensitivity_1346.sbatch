#!/bin/bash
#SBATCH --job-name=m3b-sens1346
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_sens1346_%j.log

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

OUT="$M3/outputs/m3b_sensitivity_1346/trace_1346.npz"

if [ -e "$OUT" ]; then
    echo "Output already exists"
    exit 1
fi

python -u "$M3/scripts/m3b_generate_trace_support.py" \
    --m2-dir "$M3/inputs/module2_s9534" \
    --m3a-dir "$M3/outputs/module3a_v1" \
    --runtime-root "$RUN" \
    --module2-common "$SRC/module2_review/scripts/module2_common.py" \
    --only-ids 1346 \
    --atol 3e-5 \
    --rtol 1e-5 \
    --cam-atol 0.005 \
    --cam-rtol 0.001 \
    --out "$OUT"

echo "SENSITIVITY PASS: ECG 1346"
