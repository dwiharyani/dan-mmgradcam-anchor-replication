#!/bin/bash
#SBATCH --job-name=ecg_align100
#SBATCH --output=module1_alignment/results/temporal_validation_100/slurm_%j.out
#SBATCH --error=module1_alignment/results/temporal_validation_100/slurm_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

python -u module1_alignment/src/validate_signal_image_mapping_100.py
