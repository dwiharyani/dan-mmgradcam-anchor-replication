#!/bin/bash
# Run on the user's Mac terminal, after the export Slurm job has COMPLETED.
set -euo pipefail
HOST="g202527350@hpcmaster-1.kfupm.edu.sa"
REMOTE='~/mm_gradcam/module3_aligned/exports/DAN_MMGradCAM_Sampai_Faithfulness_HPC.zip'
scp "$HOST:$REMOTE" "$HOME/Desktop/"
echo "Saved: $HOME/Desktop/DAN_MMGradCAM_Sampai_Faithfulness_HPC.zip"
