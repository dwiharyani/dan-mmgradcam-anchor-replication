# DAN-MMGradCAM: code and notebooks (COE 691 final project)

Branch `coe691-final` of this repository holds the complete COE 691 submission code. `main.ipynb` clones this branch and runs the whole analysis (Module 1, Module 2, Module 3 and the faithfulness evaluation) in one notebook; the six notebooks in `notebooks/` run the same analysis part by part.

**Data in this branch.** Unlike the other branches, `coe691-final` includes `data/archived/`: derived per-record outputs of our experiments (Grad-CAM, stability and fused maps on a 600-column time grid, predictions and deletion results) computed from the public Chapman–Shaoxing data (CC0). It contains no raw ECG signals, no images and no model weights.

Six Jupyter notebooks reproduce every table and figure of the report from the archived outputs of our HPC experiments. All notebooks are already executed and saved with their outputs.

## Contents

```
main.ipynb   clones this branch and runs all parts in one notebook
notebooks/   01 → 06, executed (run in this order)
src/         dan_figures.py, dan_column_figures.py   (plotting functions imported by the notebooks)
src/hpc_pipeline/   the scripts that produced data/archived on the KFUPM HPC (not needed to run the notebooks)
data/archived/   archived HPC outputs read by the notebooks (about 58 MB)
requirements.txt
```

| Notebook | Reproduces |
|---|---|
| 01_data_split_and_replication | patient-separated split, replication accuracy (Table 1), explanation cohort (939 ECGs) |
| 02_module1_intensity_aware_agreement | Module 1 agreement vs mispairing null (Table 2, Figs. 2–3), binary-mask baseline |
| 03_module2_prediction_preserving_stability | prediction retention and explanation stability (Fig. 4) |
| 04_module3_reliability_fusion | Module 3A / 3B audits and cohort metrics, QRS geometry check (Figs. 5–6) |
| 05_faithfulness_and_comparative_evaluation | QC of 692 deletion results, paired ΔAUC with bootstrap CIs (Table 3, Fig. 7), component study (Table 4) |
| 06_pipeline_and_reproducibility_audit | pipeline overview, SHA-256 check of the 1399 archived files, run configuration, documented failures |

## Code on GitHub

* Anchor re-run (cohort, frozen split, 1D/2D CNN training, test-split manifests): `main` branch of <https://github.com/dwiharyani/dan-mmgradcam-anchor-replication>.
* Grad-CAM temporal alignment and Module 1 alignment: `module1-nahida` branch of the same repository.

The same scripts are included in `src/hpc_pipeline/`.

## Trained models

The 1D and 2D CNN weights used by `src/hpc_pipeline/` are release assets of this repository: <https://github.com/dwiharyani/dan-mmgradcam-anchor-replication/releases/tag/v0.2.0-hpc-trained-models>. The notebooks and `main.ipynb` do not need them.

| Asset | Source checkpoint | SHA-256 |
|---|---|---|
| `MMGradCAM_1d_trained_model.keras` | 1D CNN after the last of 100 epochs | `b47e339d6003f0a065ae60eac9ea2c8aef9ff05fd9884f5d357aaba199374887` |
| `MMGradCAM_2d_trained_model.keras` | 2D CNN checkpoint with the best validation accuracy | `049c51ebe13c8f764ccc67c981a239265f825ee3f44ed429807a9c00a03adc96` |

## How to run

**Main notebook (recommended).** Open `main.ipynb` anywhere (VS Code, Jupyter or Google Colab) and run all cells. The first cell clones this branch into `./DAN-MMGradCAM`; it needs `git` and the packages in `requirements.txt` (preinstalled in Colab).

**Part-by-part notebooks.**

Python 3.10 or newer. Total run time is about 4 minutes; seeds are fixed, so the outputs are identical on every run.

**VS Code or Jupyter (Mac / Linux)**

```bash
cd DAN-MMGradCAM
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows: `python -m venv .venv` then `.venv\Scripts\python -m pip install -r requirements.txt`.

Open `notebooks/01_data_split_and_replication.ipynb`, select the `.venv` kernel (top right in VS Code), click **Run All**, save, then repeat for 02 … 06 in order (notebook 03 reads a table written by 02). The first code cell should print `repository root: DAN-MMGradCAM`.

Command-line alternative: `.venv/bin/python -m pip install jupyter` and then `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0*.ipynb`.

**Google Colab**

Upload the ZIP, run `!unzip -q DAN-MMGradCAM_code.zip -d /content` once, open the notebooks from `/content/DAN-MMGradCAM/notebooks/` and run all cells in order. All libraries are preinstalled.

**Troubleshooting**

* `ModuleNotFoundError: No module named 'numpy'`: the packages were installed into a different Python than the notebook kernel. Run `.venv/bin/python -m pip install -r requirements.txt` (or `%pip install numpy pandas scipy scikit-learn matplotlib` in a notebook cell), then restart the kernel.
* `FileNotFoundError: DAN-MMGradCAM folder not found`: open the notebook from inside the extracted folder, keeping `notebooks/`, `src/` and `data/` side by side.

## HPC pipeline (src/hpc_pipeline)

These scripts produced every file in `data/archived/` (Module 2 on Kaggle, all other stages on the KFUPM HPC). They need the raw Chapman–Shaoxing ECGs, the trained models and TensorFlow 2.19–2.21, so they are included for completeness and are not run by the notebooks.

| Folder | Stage | Main scripts |
|---|---|---|
| `01_data_and_training/` | cohort, frozen patient-separated split, 1D/2D CNN training (Adam 3e-4, batch 16, 100 epochs, seed 42) | `build_final_1d.py`, `build_frozen_split.py`, `train_1d_author_code.py`, `train_2d_author_code.py` |
| `02_gradcam_and_module1/` | Grad-CAM for both CNNs, temporal alignment, Module 1 | `run_gradcam_1d_author_code.py`, `run_gradcam_2d_author_code.py`, `module1_alignment_branch/` |
| `03_module2_stability/` | 30 prediction-preserving perturbations, re-rendering, μ / S / H maps (run on Kaggle) | `run_dan_module2_noise_stability.py`, `module2_common.py` |
| `04_module3_fusion_faithfulness/` | Module 3A, Module 3B, paired block deletion (SLURM array 0–692), summary | `m3a_prepare.py`, `m3b_generate_trace_support.py`, `m3b_apply.py`, `module3_faithfulness.py`, `summarize_692_exploratory.py` |

## Data and external resources

* Chapman–Shaoxing ECG database (Zheng et al., *Sci. Data* 7:48, 2020). Raw ECGs, rendered images and model weights are not redistributed; `data/archived/` holds the outputs of the HPC runs that the analysis needs.
* The CNN architectures and training procedure follow the MM-GradCAM anchor (Duranay et al., *Sci. Rep.* 16:7919, 2026); all weights were retrained by us on the KFUPM HPC.
* Notebook 06 keeps `RUN_ON_HPC = False`; the HPC smoke test needs the trained models and raw data.
