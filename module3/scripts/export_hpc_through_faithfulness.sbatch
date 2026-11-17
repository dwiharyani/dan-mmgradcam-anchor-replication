#!/bin/bash
#SBATCH --job-name=m3-export
#SBATCH --partition=cpu_x440
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3_export_%j.log
set -euo pipefail
M3="$HOME/mm_gradcam/module3_aligned"
source "$HOME/mm_gradcam/MM_GradCAM_HPC/.venv/bin/activate"
export OMP_NUM_THREADS=2
python -u "$M3/scripts/summarize_692_exploratory.py" \
  --raw-dir "$M3/outputs/module3_faithfulness_primary_qc693_v1/raw" \
  --m3b-dir "$M3/outputs/module3b_primary_qc693_v1" \
  --out-dir "$M3/outputs/module3_faithfulness_primary_qc693_v1/summary_692_exploratory"
python -u "$M3/scripts/export_hpc_through_faithfulness.py"
