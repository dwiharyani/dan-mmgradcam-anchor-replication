# DAN Module 2 - noise stability (exploratory)

Question: do the 1D and 2D Grad-CAM maps of the same ECG stay in place under mild perturbations of the raw signal?

## Method
1. Perturb the raw Lead-II ECG once (30 perturbations): 6 amplitude scalings (0.8-1.2), 12 baseline drifts (0.3 Hz sine, 5% and 10% of the record peak-to-peak, 6 phases), 12 Gaussian-noise draws (5% and 10% of the record standard deviation).
2. Regenerate both inputs from the same perturbed signal: the 1D tensor (scaled with the 1D scaler) and the 2D image (re-rendered).
3. Run the fixed pretrained 1D and 2D models (not retrained). Grad-CAM targets each branch's own original prediction.
4. Keep a perturbation for a branch only if that branch keeps its prediction. Prediction-retention rates (PRR) are reported separately.
5. On a shared 600-column grid compute mean mu_b, std sigma_b, S_b = 1 - sigma_b / (max sigma_b + eps), H_b = mu_b S_b and R = sqrt(H_1D H_2D). Records with fewer than 2 accepted maps are "unevaluable".

## Provenance and caveats
- **Scaler:** recomputed from every non-test ECG (9,592) because `standard_scaler.npz` was unavailable; the original used 9,534. 1D test accuracy 0.9469 vs 0.9478 in this repository.
- **2D images:** drawn as `plt.figure(figsize=(15,4), dpi=72); plt.plot(x); plt.axis("off"); plt.tight_layout()`, inferred from the geometry of real images (1080x288, waveform x=58-1021). Re-rendered images give 2D accuracy 0.925 vs 0.9668 on the real images, so 2D stability is for re-rendered images.
- **Validation split:** the 1,059 validation records are reconstructed by matching validation images to ECG renders; confidence is reported by the script.
- **Perturbation settings:** not yet validated on the validation split.
- Stability and deletion tests are exploratory and do not establish faithfulness.

## Run (requires data and models locally, none included here)
```bash
python scripts/run_dan_module2_noise_stability.py
python scripts/run_dan_module2_validation_split.py
python scripts/run_dan_module2_convergence_pilot.py
python scripts/run_dan_module2_deletion_test.py
python scripts/plot_dan_module2_figures.py
```
Scripts were exported from a notebook; each imports `module2_common.py` and defaults to `data/` and `results/module2/` paths. Aggregate numbers: `reports/module2_summary.json`.
