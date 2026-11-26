#!/bin/bash
#SBATCH --job-name=projection_sensitivity
#SBATCH --output=module1_alignment/results/projection_sensitivity/slurm_%j.out
#SBATCH --error=module1_alignment/results/projection_sensitivity/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

python -u module1_alignment/src/projection_sensitivity_full.py

echo "PROJECTION SENSITIVITY COMPLETED"
