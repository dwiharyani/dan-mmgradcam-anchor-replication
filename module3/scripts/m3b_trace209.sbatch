#!/bin/bash
#SBATCH --job-name=m3b-trace209
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3b_trace209_%j.log
set -euo pipefail
BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
SRC="$HOME/mm_gradcam/module3_sources"
M3="$HOME/mm_gradcam/module3_aligned"
RUN="$SRC/module2_runtime"
source "$BASE/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
# Model and scaler are links to frozen originals and must exist.
test -f "$RUN/models/MMGradCAM_2d_trained_model.keras"
test -f "$RUN/results/module2/standard_scaler.npz"
python -u "$M3/scripts/m3b_generate_trace_support.py" \
  --m2-dir "$M3/inputs/module2_s9534" \
  --m3a-dir "$M3/outputs/module3a_v1" \
  --runtime-root "$RUN" \
  --module2-common "$SRC/module2_review/scripts/module2_common.py" \
  --only-ids 209 \
  --out "$M3/outputs/m3b_trace_support_209.npz"
