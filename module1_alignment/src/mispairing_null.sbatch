#!/bin/bash
#SBATCH --job-name=m1_null
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=module1_alignment/results/within_class_mispairing/slurm_%j.out
#SBATCH --error=module1_alignment/results/within_class_mispairing/slurm_%j.err

cd "$HOME/Alignment-XAI"

python -u module1_alignment/src/within_class_mispairing.py
