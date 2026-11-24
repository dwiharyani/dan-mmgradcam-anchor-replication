#!/bin/bash
#SBATCH --job-name=m3f-cmp406
#SBATCH --partition=cpu_x440
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/g202527350/mm_gradcam/module3_aligned/logs/m3faith_cmp406_%A_%a.log

set -euo pipefail

source "$HOME/mm_gradcam/MM_GradCAM_HPC/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4

CASES=(raw_first cast_first raw_after1d cast_after1d)
export CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"

python - <<'PY'
import os, sys, importlib.util, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

home = Path.home() / "mm_gradcam"
m3 = home / "module3_aligned"
src = home / "module3_sources"
run = src / "module2_runtime"

records = pd.read_csv(
    m3 / "inputs/module2_s9534/noise_stability_records.csv"
)
row = records.loc[records.cohort_index == 406].iloc[0]

with np.load(
    m3 / "inputs/module2_s9534/noise_stability_maps.npz",
    allow_pickle=False
) as z:
    idx = np.flatnonzero(z["cohort_index"].astype(int) == 406)
    assert len(idx) == 1
    ref = np.asarray(z["out2_base"][int(idx[0])], dtype=np.float64)

os.chdir(run)
fp = src / "module2_review/scripts/module2_common.py"
sys.path.insert(0, str(fp.parent))
spec = importlib.util.spec_from_file_location("m2_compare406", fp)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

raw = cm.load_ecg(str(row.ecg_csv_filename))
case = os.environ["CASE"]
x = np.asarray(raw, dtype=np.float64) if case.startswith("cast") else raw

print("CASE:", case)
print("Original dtype:", np.asarray(raw).dtype)
print("Input dtype:", np.asarray(x).dtype)

renderer = cm.Renderer()

if case.endswith("after1d"):
    inp1 = ((x - cm.MEAN) / cm.SCALE).astype(np.float32)[None, :, None]
    cm.run_model("1d", cm.m1, inp1)

rgb = renderer.base(x)
img_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()

inp2 = (cm.to_2d(rgb[None]) / 255.0).astype(np.float32)
pred = np.asarray(cm.run_model("2d", cm.m2, inp2)[0], dtype=np.float64)

err = float(np.max(np.abs(pred - ref)))

print("Image SHA256:", img_hash)
print("Archived probabilities:", ref)
print("Recomputed probabilities:", pred)
print("Maximum error:", err)
print("QC PASS:", bool(np.allclose(pred, ref, atol=1e-5, rtol=1e-5)))
PY
