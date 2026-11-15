# Module 2 scripts

These six Python files were supplied for the Module 2 contribution.

## Files and intended roles

- `module2_common.py` — shared setup, paths, data/model loading, renderer,
  perturbation generation, Grad-CAM and common functions.
- `run_dan_module2_noise_stability.py` — smoke test and resumable full
  1,054-record noise-stability run.
- `run_dan_module2_validation_split.py` — reconstructs/matches the validation
  split; its matching procedure includes a reliability gate and should not be
  treated as validated unless that gate passes.
- `run_dan_module2_convergence_pilot.py` — perturbation-count convergence pilot.
- `run_dan_module2_deletion_test.py` — paired deletion and block-size
  sensitivity analyses using saved Module 2 maps.
- `plot_dan_module2_figures.py` — analyses saved outputs and generates figures.

## Run requirements

Run from the repository/workspace root so the relative paths in
`module2_common.py` resolve. The scripts expect the dataset, model weights,
anchor repository assets, and saved Module 2 outputs described in the
reproducibility documentation. None of those restricted/large inputs are
bundled here.

`module2_common.py` currently uses these default paths:
- `data/ECGData`
- `data/ImageData-label2/label1`
- `models/`
- `results/module2/`

Review/edit those paths for the target environment before execution. Run the
main noise-stability script's smoke test first; only proceed to the full run
when its gate passes. The validation reconstruction script must pass its
known-pair matching check before using reconstructed validation matches.

All six files passed Python syntax parsing when this ZIP was assembled. This
does not guarantee runtime success because runtime depends on the local data,
weights, package versions, and expected saved-result files.
