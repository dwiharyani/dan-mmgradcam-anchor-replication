#!/bin/bash
#SBATCH --job-name=m3b-40
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_balanced40_%j.log

set -euo pipefail

BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
SRC="$HOME/mm_gradcam/module3_sources"
M3="$HOME/mm_gradcam/module3_aligned"
RUN="$SRC/module2_runtime"

source "$BASE/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

OUT="$M3/outputs/m3b_trace_support_balanced40.npz"

test -f "$M3/inputs/m3b_balanced40_ids.txt"
test ! -e "$OUT"

python -u "$M3/scripts/m3b_generate_trace_support.py" \
  --m2-dir "$M3/inputs/module2_s9534" \
  --m3a-dir "$M3/outputs/module3a_v1" \
  --runtime-root "$RUN" \
  --module2-common "$SRC/module2_review/scripts/module2_common.py" \
  --only-ids "$(cat "$M3/inputs/m3b_balanced40_ids.txt")" \
  --out "$OUT"
