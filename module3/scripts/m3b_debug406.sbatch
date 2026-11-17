#!/bin/bash
#SBATCH --job-name=m3b-check406
#SBATCH --partition=cpu_x440
#SBATCH --array=0-1%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_debug406_%A_%a.log

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

IDX=$(python -c 'import numpy as np,sys; z=np.load(sys.argv[1]); a=np.flatnonzero(z["cohort_index"]==406); assert len(a)==1; print(int(a[0]))' "$M3/outputs/module3a_v1/module3a_maps.npz")

echo "ECG 406 index in Module3A: $IDX"

if (( IDX < 600 || IDX >= 650 )); then
    echo "ERROR: ECG 406 outside expected chunk 0600"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    MODE=fresh
    ARGS=(--only-ids 406)
else
    MODE=warm
    ARGS=(--start 600 --limit "$((IDX - 600 + 1))")
fi

OUT="$M3/outputs/m3b_406_debug/${MODE}.npz"

echo "MODE=$MODE"

if [ -e "$OUT" ]; then
    echo "ERROR: Existing output $OUT"
    exit 1
fi

python -u "$M3/scripts/m3b_generate_trace_support_debug406.py" \
    --m2-dir "$M3/inputs/module2_s9534" \
    --m3a-dir "$M3/outputs/module3a_v1" \
    --runtime-root "$RUN" \
    --module2-common "$SRC/module2_review/scripts/module2_common.py" \
    "${ARGS[@]}" \
    --out "$OUT"

echo "PASS MODE=$MODE"
