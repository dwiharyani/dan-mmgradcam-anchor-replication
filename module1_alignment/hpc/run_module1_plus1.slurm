#!/bin/bash
#SBATCH --job-name=module1_plus1
#SBATCH --output=module1_alignment/results/alignment_plus1px/slurm_%j.out
#SBATCH --error=module1_alignment/results/alignment_plus1px/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

export ALIGNMENT_OUTPUT="alignment_plus1px"
export ALIGNMENT_SHIFT="1"

echo "Starting +1 pixel temporal alignment"

python -u module1_alignment/src/temporal_alignment_full.py

echo "Starting 2D Grad-CAM projection"

python -u module1_alignment/src/projection_full.py

echo "Starting agreement and disagreement analysis"

python -u module1_alignment/src/agreement_disagreement_full.py

echo "PLUS-1-PIXEL ANALYSIS COMPLETED"
