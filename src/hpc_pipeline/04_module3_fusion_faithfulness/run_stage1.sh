#!/usr/bin/env bash
# Setup + audit Module3A, does not submit GPU jobs, nor edit M1/M2 originals.
set -euo pipefail
BASE="$HOME/mm_gradcam/MM_GradCAM_HPC"
M3="$HOME/mm_gradcam/module3_aligned"
M2ZIP="${1:-$HOME/mm_gradcam/module3_sources/Dan-mmgradcam.zip}"
PRIMARY="$BASE/results/dan_module1_full_20261015_191138/module1_primary_exploratory.csv"
for f in "$M2ZIP" "$PRIMARY" "$HOME/mm_gradcam/m1_input/alignment_0px/valid_temporal_mask.npy" "$HOME/mm_gradcam/m1_input/alignment_0px/gradcam_1d_aligned.npy" "$HOME/mm_gradcam/m1_input/gradcam_full/gradcam_metadata.csv"; do
  if [[ ! -f "$f" ]]; then echo "Missing: $f" >&2; exit 1; fi
done
source "$BASE/.venv/bin/activate"
python "$M3/scripts/m3_extract_s9534.py" --zip "$M2ZIP" --out "$M3/inputs/module2_s9534"
python "$M3/scripts/m3a_prepare.py" \
  --m2-dir "$M3/inputs/module2_s9534" \
  --primary "$PRIMARY" \
  --m1-valid "$HOME/mm_gradcam/m1_input/alignment_0px/valid_temporal_mask.npy" \
  --m1-metadata "$HOME/mm_gradcam/m1_input/gradcam_full/gradcam_metadata.csv" \
  --m1-1d "$HOME/mm_gradcam/m1_input/alignment_0px/gradcam_1d_aligned.npy" \
  --highlight 209 \
  --out "$M3/outputs/module3a_v1"
echo "Module3A input audit and maps are ready. Review $M3/outputs/module3a_v1/module3a_summary.json"
