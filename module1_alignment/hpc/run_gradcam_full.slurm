#!/bin/bash
#SBATCH --job-name=gradcam_full
#SBATCH --output=module1_alignment/results/gradcam_full/slurm_%j.out
#SBATCH --error=module1_alignment/results/gradcam_full/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

python -u module1_alignment/src/generate_gradcam_full.py
