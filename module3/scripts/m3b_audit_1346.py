import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "mm_gradcam"
M3 = ROOT / "module3_aligned"
SRC = ROOT / "module3_sources"
RUN = SRC / "module2_runtime"
CID = 1346

m2_path = M3 / "inputs/module2_s9534/noise_stability_maps.npz"
m3_path = M3 / "outputs/module3a_v1/module3a_maps.npz"

with np.load(m2_path, allow_pickle=False) as z:
    ids = z["cohort_index"].astype(int)
    idx = np.flatnonzero(ids == CID)
    assert len(idx) == 1
    pos = int(idx[0])
    reference = np.asarray(z["out2_base"][pos], dtype=np.float64)
    saved_cam = np.asarray(z["base2"][pos], dtype=np.float64)

with np.load(m3_path, allow_pickle=False) as z:
    idx = np.flatnonzero(z["cohort_index"].astype(int) == CID)
    assert len(idx) == 1
    valid = z["valid_600"][int(idx[0])].astype(bool)

records = pd.read_csv(
    M3 / "inputs/module2_s9534/noise_stability_records.csv"
)
assert int(records.iloc[pos]["cohort_index"]) == CID

os.chdir(RUN)
fp = SRC / "module2_review/scripts/module2_common.py"
sys.path.insert(0, str(fp.parent))

spec = importlib.util.spec_from_file_location("module2_common_audit", fp)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

row = records.iloc[pos]
raw = cm.load_ecg(str(row["ecg_csv_filename"]))
rgb = cm.Renderer().base(raw)
img = (cm.to_2d(rgb[None]) / 255.0).astype(np.float32)

prob = np.asarray(
    cm.run_model("2d", cm.m2, img)[0],
    dtype=np.float64
)

print("\n=== ECG 1346 PROBABILITY AUDIT ===")
print("Class:", row["class_name"])
print("Archived:", reference)
print("Recomputed:", prob)
print("Absolute difference:", np.abs(prob - reference))
print("Maximum difference:", np.max(np.abs(prob - reference)))
print("Archived prediction:", cm.CLASSES[int(np.argmax(reference))])
print("Recomputed prediction:", cm.CLASSES[int(np.argmax(prob))])
print("Original probability QC passed:",
      np.allclose(prob, reference, atol=1e-5, rtol=1e-5))

native = cm.gradcam(
    "2d", cm.m2, cm.L2, img, int(np.argmax(prob))
)[0]

reproduced_cam, _ = cm.project_2d(native, rgb, valid)

a = np.asarray(reproduced_cam)[valid]
b = saved_cam[valid]
diff = np.abs(a - b)

print("\n=== ECG 1346 GRAD-CAM AUDIT ===")
print("Valid columns:", valid.sum())
print("Maximum CAM difference:", np.max(diff))
print("Mean CAM difference:", np.mean(diff))
print("Original CAM QC passed:",
      np.allclose(a, b, atol=0.005, rtol=0.001))
