#!/bin/bash
#SBATCH --job-name=module1_full
#SBATCH --output=module1_alignment/results/full_analysis_logs/slurm_%j.out
#SBATCH --error=module1_alignment/results/full_analysis_logs/slurm_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e

cd ~/Alignment-XAI

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nahida_ipynb_env

for SHIFT in 0 1
do
    if [ "$SHIFT" -eq 0 ]; then
        export ALIGNMENT_OUTPUT="alignment_0px"
    else
        export ALIGNMENT_OUTPUT="alignment_plus1px"
    fi

    export ALIGNMENT_SHIFT="$SHIFT"

    echo "======================================"
    echo "Running alignment shift: $SHIFT pixels"
    echo "======================================"

    python -u module1_alignment/src/temporal_alignment_full.py

    python -u module1_alignment/src/projection_full.py

    python -u module1_alignment/src/agreement_disagreement_full.py

done

echo "FULL MODULE 1 ANALYSIS COMPLETED"
