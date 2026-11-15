# Reproducibility

## Required inputs (not included)

The scripts and notebook require the project's local ECG dataset, image dataset,
pretrained model files, anchor repository assets, and (for saved-output analyses)
the Module 2 result arrays/CSVs. These large or restricted inputs are not bundled.

## Default paths

The shared setup in `scripts/module2_common.py` expects:
- `data/ECGData`
- `data/ImageData-label2/label1`
- `models/MMGradCAM_1d_trained_model.keras`
- `models/MMGradCAM_2d_trained_model.keras`
- `results/module2/`

Run from the workspace/repository root, or edit the paths in
`module2_common.py` to match the local layout.

## Suggested execution order

1. Review the data/model paths and Python environment in `scripts/module2_common.py`.
2. Run `scripts/run_dan_module2_noise_stability.py`. It performs a smoke test
   and only starts the full run if its gate passes.
3. Run `scripts/run_dan_module2_validation_split.py` only if validation-split
   reconstruction is needed. Do not use its reconstructed matches unless the
   known-pair matching accuracy gate passes.
4. Run `scripts/run_dan_module2_convergence_pilot.py` for the perturbation-count
   pilot; interpret it as exploratory.
5. Run `scripts/plot_dan_module2_figures.py` after the required saved outputs
   exist.
6. Run `scripts/run_dan_module2_deletion_test.py` only after the saved Module 2
   maps and required model/data inputs are available.

The notebook in `notebooks/` is also included as the interactive record of the
analysis. Script and notebook outputs may require environment-specific paths.

## Provenance caveats

- The 1D scaler was recomputed from 9,592 records; the intended original
  non-test cohort was 9,534.
- 2D images were re-rendered from raw Lead-II ECG; re-rendered-image accuracy
  was 0.925 versus 0.9668 on the real images.
- Perturbation settings have not been validated on the validation split.
