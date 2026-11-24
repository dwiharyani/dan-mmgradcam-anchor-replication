# Project scope and scientific validity

| Component | Achieved | Claim boundary |
|---|---|---|
| Original author study | External anchor publication | Credit the authors. This is not its official software release. |
| 1D and 2D CNN reproduction | Independently trained in the experimental HPC workspace | Training files and model weights not included here; not identical-paper metrics. |
| Paper-style 1D + 2D Grad-CAM visual overlays | Generated for 1,054 test ECGs | Recreates visual composition, not original pixel values or fully calibrated time axis. |
| Exact 1D/2D ECG pairing | 1,054 paired records | Same-record identity established using explicit manifests. |
| Signal-image waveform correspondence | Audited across 1,054 | 810 AUTO_PASS; 244 manual-review according to heuristic QC, not ground-truth proof. |
| Primary quantitative Module 1 analysis | 727 correctly classified, strict-QC ECGs | Provisional and conditional on selection; no causal claims. |
| Projection sensitivity | 727 ECGs × 3 alternatives | Descriptive robustness only; x-time calibration not independently confirmed from source image-render code. |
| Random-shift/null or Module 2 stability | Not complete | Do not claim as finished or experimentally supported. |
| Module 3 reliability-weighted method | Proposed, not complete | Do not claim clinical or predictive improvement. |

No clinical validation, medical device functionality, prospective patient evaluation, or diagnostic recommendation is claimed. The analysis is a research illustration of interpretability comparisons.
