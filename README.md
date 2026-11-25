# DAN-MMGradCAM: reliability of paired 1D-signal and 2D-image Grad-CAM explanations for ECG arrhythmia classification

COE 691 Explainable Artificial Intelligence, final project. Team: Dwi Haryani, Nahida Zakir, Ayesha Mateen (contributions in `CONTRIBUTIONS.md`).

The project extends our survey *Reliability of 1D–2D ECG Explanations for Cardiac Arrhythmia Classification*. It takes MM-GradCAM (Duranay et al., 2026), which fuses the Grad-CAM of a 1D signal CNN and a 2D image CNN for the same ECG, and adds three modules:

| Module | Question | Notebook |
|---|---|---|
| 1. Intensity-aware, directional agreement against a mispairing null | Do both CNNs use the same evidence? (RQ1) | `02_module1_intensity_aware_agreement.ipynb` |
| 2. Prediction-preserving, branch-specific stability (30 perturbations) | Is each explanation stable? (RQ2) | `03_module2_prediction_preserving_stability.ipynb` |
| 3. Reliability-weighted fusion `R = sqrt(H1D · H2D)` and waveform-conditioned `R_trace = R · F_trace^λ` | Is the fused map faithful? (RQ3) | `04_module3_reliability_fusion.ipynb`, `05_faithfulness_and_comparative_evaluation.ipynb` |

The report is `report/DAN-MMGradCAM_COE691_report.pdf` (IEEE ICASSP two-column format).

## Folder structure

```
report/                  main.tex, spconf.sty (ICASSP style), report PDF
notebooks/               six executed notebooks (outputs saved); run in order 01 -> 06
src/dan_figures.py       plotting functions used by the notebooks
src/dan_case_figures.py  single-ECG figure helpers
src/hpc_pipeline/        the scripts that produced the archived outputs on the KFUPM HPC
  01_data_and_training/          cohort, frozen split, 1D/2D CNN training (anchor code replication)
  02_gradcam_and_module1/        Grad-CAM, temporal alignment, Module 1 (incl. module1_alignment_branch/)
  03_module2_stability/          Module 2 perturbation stability
  04_module3_fusion_faithfulness/ Module 3A, 3B, deletion faithfulness, SLURM job files
data/archived/           outputs of the HPC runs used as notebook inputs (see below)
results/tables/          CSV tables written by the notebooks
results/figures/         figures written by the notebooks (the report uses these)
tools/execute_notebook.py  runs a notebook top to bottom and saves outputs (fallback when Jupyter is absent)
```

## 1. Install

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers the notebooks (numpy, pandas, scipy, scikit-learn, matplotlib, Pillow, jupyter). The HPC scripts in `src/hpc_pipeline/` additionally need TensorFlow (2.19–2.21), h5py and opencv-python, the trained models and the raw data.

## 2. Data

* **Dataset.** Chapman–Shaoxing 12-lead ECG database (Zheng et al., *Sci. Data* 7:48, 2020), public at <https://figshare.com/collections/ChapmanECG/4560497>. We use 10-s Lead II at 500 Hz and the anchor's four classes (AFib, GSVT, SB, SR). The frozen cohort has 10 588 recordings from 10 588 patients, split into 8475 training, 1059 validation and 1054 test recordings (`data/archived/split/`).
* **What is included.** Raw ECGs, rendered images and model weights are *not* redistributed. The package contains the outputs of every HPC stage that the analysis needs (about 60 MB):

| `data/archived/` | Content | Produced by |
|---|---|---|
| `split/` | test and training manifests, test-split metadata | `01_data_and_training/build_frozen_split.py` |
| `replication/` | test predictions of the 1D and 2D CNNs | `02_gradcam_and_module1/run_gradcam_*_author_code.py` |
| `module1/` | min–max baseline agreement, projection sensitivity, ECG 209 Module 1 panels, ECG 209 waveform mask | Module 1 branch scripts |
| `module2/` | `noise_stability_maps.npz` (µ, S, H per branch, R, Grad-CAM on the 600-column grid for 1054 recordings), record table, run configuration | `03_module2_stability/run_dan_module2_noise_stability.py` |
| `module3a/`, `module3b/`, `module3b_sensitivity/` | fused maps and audits (694, 693 and 694 recordings) | `m3a_prepare.py`, `m3b_generate_trace_support.py`, `m3b_apply.py` |
| `faithfulness_cohort/raw/` | 692 per-recording deletion results with QC audits; `hpc_paired_auc_summary.csv` | `module3_faithfulness.py`, `summarize_692_exploratory.py` |
| `faithfulness_component/` | 100-recording component deletion study | `03_module2_stability/run_dan_module2_deletion_test.py` |
| `provenance/` | SHA-256 manifest written on the HPC, provenance record, ECG 406 failure logs | export script |

Notebook 06 checks every archived file against the HPC SHA-256 manifest (1399 of 1399 match).

## 3. Run the notebooks (reproduces every number and figure in the report)

```bash
cd notebooks
jupyter notebook          # open 01 ... 06 and choose Kernel > Restart & Run All
# or, without the browser:
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

Run them in order: notebook 03 reads `results/tables/module1_per_record.csv` written by 02. Total run time is about 5 minutes on a laptop (notebook 04 takes about 2 minutes). Bootstrap and permutation seeds are fixed, so the outputs are identical on every run.

If Jupyter is not available, `python tools/execute_notebook.py notebooks/0*.ipynb` executes the same cells in order and saves the outputs into the notebooks.

| Notebook | Reproduces |
|---|---|
| 01 | split sizes, replication accuracy (Table 1), explanation cohort (939) |
| 02 | Module 1 agreement and mispairing null (Table 2, Figs. 2–3), min–max baseline comparison |
| 03 | prediction retention and stability (Fig. 4) |
| 04 | Module 3A/3B audits and cohort metrics, QRS geometry check (Fig. 5) |
| 05 | QC of 692 deletion results, paired ΔAUC with bootstrap CIs (Table 3, Fig. 6), component study (Table 4), comparative summary |
| 06 | stage-by-stage pipeline, SHA-256 integrity check, run configuration, documented failures, HPC smoke test |

## 4. Re-running the HPC pipeline (optional, needs data, models and TensorFlow)

The archived outputs were produced on the KFUPM HPC (CPU partition) with the scripts in `src/hpc_pipeline/`, in this order:

1. `01_data_and_training/`: `build_final_1d.py` → `build_frozen_split.py` → `build_image_manifest.py` → `train_1d_author_code.py`, `train_2d_author_code.py` (Adam 3e-4, batch 16, 100 epochs, seed 42).
2. `02_gradcam_and_module1/`: `run_gradcam_1d_author_code.py`, `run_gradcam_2d_author_code.py`, then the alignment and Module 1 scripts (`module1_alignment_branch/hpc/*.slurm`).
3. `03_module2_stability/run_dan_module2_noise_stability.py` (smoke test first, then the full 1054 recordings; paths are set in `module2_common.py`).
4. `04_module3_fusion_faithfulness/run_stage1.sh` (Module 3A), `m3b_full694_onednn_off.sbatch` + `m3b_apply.py` (Module 3B), `module3_faithfulness.sbatch` (SLURM array 0–692), `summarize_692_exploratory.py`.

Set `RUN_ON_HPC = True` in notebook 06 to run a one-recording smoke test of stage 4 on the HPC. Module 3 used TensorFlow 2.19 with `TF_ENABLE_ONEDNN_OPTS=0`; Module 2 used TensorFlow 2.21.

## 5. Known reproducibility issues (also in the report)

* The matplotlib ECG renderer keeps state between calls. For ECG 406 the 2D baseline probability could not be reproduced (error 5.6e-5 > 1e-5), so it is excluded from the deletion study (692 of 693). 2D deletion results are reported as exploratory.
* ECG 1346 failed the Module 3B reproduction check and is excluded from the primary Module 3B cohort (693); the sensitivity run with 694 recordings is included.

## 6. External resources and acknowledgements

* MM-GradCAM anchor: F. M. Duranay et al., *Sci. Rep.* 16:7919, 2026. CNN architectures, image rendering and training procedure follow the authors' code; all weights were retrained by us.
* Chapman–Shaoxing database: J. Zheng et al., *Sci. Data* 7:48, 2020.
* Libraries: TensorFlow/Keras, NumPy, SciPy, pandas, scikit-learn, matplotlib, Pillow.
* Claude (Anthropic) was used to help write analysis and plotting code, the notebooks and the report text; all results were checked by the team.
* Computing: KFUPM HPC.
