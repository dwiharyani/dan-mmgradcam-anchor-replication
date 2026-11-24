"""DAN Module 2 noise-stability run (smoke test, gate, full 1,054-record run; resumable).

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
from module2_common import *   # defines paths, test manifest, models, perturbation/render/Grad-CAM functions

# ---- notebook cell 7 ----
# 8. Self-test: one 2D input, then 31 inputs (the size that used to crash), then one 1D input
selftest()

# ---- notebook cell 9 ----
# 10. Smoke test (2 records per class) -> gate -> full 1,054-record run (about an hour)
RUN_FULL = True
smoke = run(test.groupby("class_name", sort=False).head(2), tag="smoke8_v2")   # 2 per class
gate = ((smoke.status_joint == "evaluable").all() and smoke.base_coverage.min() >= COVERAGE_MIN
        and smoke.n_joint_accepted.min() >= 3)
print("\nSMOKE GATE:", "PASS" if gate else "FAIL - not starting the full run; send me the summary above")
if RUN_FULL and gate:
    results = run(test, tag="full_v2")
