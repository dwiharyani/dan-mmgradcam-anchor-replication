#!/bin/bash
#SBATCH --job-name=cam_shift11
#SBATCH --output=module1_alignment/results/gradcam_shift_sensitivity/slurm_%j.out
#SBATCH --error=module1_alignment/results/gradcam_shift_sensitivity/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

python -u module1_alignment/src/gradcam_shift_sensitivity.py

echo "GRAD-CAM SHIFT SENSITIVITY COMPLETED"
