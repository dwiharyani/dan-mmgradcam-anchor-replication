# Reproduction and execution contract

This is a **curated post-training code release**, not a dataset or model checkpoint archive. All commands assume a workspace root containing the original data, trained models/Grad-CAM outputs, manifests, and `scripts/` from this repository. Run commands from that root. The upstream source is linked in README; author code is not redistributed.

## Source requirements — expected paths

| Path in workspace | Input | Public GitHub? |
|---|---|---|
| `data/1d/mm_gradcam_leadII_10588.h5` | Numerical 1D ECG, `X` = (10588, 5000) and labels | **No** |
| `data/2d_author_clean/Testdata/{AFib,GSVT,SB,SR}/hasta_*.png` | Author-derived 2D ECG images | **No** |
| `splits/final_cohort_10588.csv` | Cohort-index-to-patient mapping | **No** |
| `results/gradcam_1d_author_code/heatmaps_1d.npy` | 1D CAM (1054, 5000) | **No** |
| `results/gradcam_2d_author_code/heatmaps_2d_resized.npy` | 2D CAM (1054, 300, 600), from native 9x18 | **No** |
| `results/gradcam_1d_author_code/gradcam_1d_manifest.csv` | 1D CAM identity / row index | **No** |
| `results/gradcam_2d_author_code/gradcam_2d_manifest.csv` | 2D CAM identity / row index | **No** |
| `splits/gradcam_test_pairs_verified.csv` | Paired identity indices created by matching script | **No** |
| `results/temporal_validation_full_20261015_163719/temporal_alignment_validation.csv` | Independent ECG-to-image correspondence QC | **No** |

**Original source of numerical 1D ECG:** 2019/2020 Chapman–Shaoxing dataset on Figshare; **author 2D images:** Kaggle `fatmamurat/imagedata-label2`. Both describe the same underlying ECG records for the paired test cohort. The author's separate `label-2` Kaggle entry should **not** be assumed to contain numerical CSVs without checking its current contents.

## Independent anchor reproduction

1. Retrieve the dataset through its source. Use the linked author repositories to prepare and independently train the signal and image CNN branches. **This archive does not contain those training scripts, weights, or datasets.**
2. Generate Grad-CAM for both branches and preserve per-record patient ID, cohort index, class and CAM array row index.
3. Execute in workspace root:

```bash
python scripts/verify_ecg_gradcam_pairs.py
python scripts/mmgradcam_alignment_preflight.py \
  --out results/alignment_preflight
python scripts/run_ecg_temporal_alignment_provisional.py --mode inspect
python scripts/run_all_mmgradcam_alignment.py \
  --expected 1054 --figures representative \
  --out results/mmgradcam_anchor_replication
```

Check the manifest before comparing two CAMs: **do not** pair by 1D row number = 2D row number, and do not silently match by class alone. In the tested cohort 1,041 of 1,054 array row indices differed, although the corresponding patient could be matched exactly.

## Independent temporal validation

```bash
python scripts/validate_ecg_temporal_alignment_hpc.py --self-test
python scripts/validate_ecg_temporal_alignment_hpc.py --all \
  --out results/temporal_validation_full_20261015_163719
```

The waveform correspondence audit is **not** proof that the original 2D rendering uses precisely x0=58, x1=1021 or constant time spacing. It reports automated QRS-like detection and manual review flags. Original full-run counts were 810 `AUTO_PASS` and 244 `MANUAL_REVIEW`; a stricter downstream QC yielded 779 records independent of branch correctness.

## DAN Module 1, full exploratory evaluation

```bash
python scripts/run_dan_module1_exploratory.py \
  --validation results/temporal_validation_full_20261015_163719/temporal_alignment_validation.csv \
  --qc-figures 16 --out results/dan_module1_full_release

python scripts/run_dan_module1_projection_sensitivity.py \
  --pilot results/dan_module1_full_release/module1_primary_exploratory.csv \
  --qc-figures 12 --out results/dan_module1_projection_full_release

python scripts/plot_dan_module1_paper_figures.py \
  --module results/dan_module1_full_release \
  --sensitivity results/dan_module1_projection_full_release \
  --out results/dan_module1_visuals_release
```

**Output directories for Module 1 and projection sensitivity must be empty/new**: both scripts intentionally reject existing nonempty output folders. Set another `--out` suffix to repeat a run. Figure exporter overwrites files only in its explicitly provided figure output directory.

## Runtime

Use the existing HPC Python environment for original TensorFlow/Keras CNN training. Analysis scripts use `numpy`, `pandas`, `scipy`, `Pillow`, `matplotlib`, `h5py` as needed (see root requirements file). Run substantial data work on a compute node when required by the HPC site's policy. No GPU or CNN retraining is required **if** the prerequisite CAM arrays and manifests already exist.

## Numerical guardrails

- Compare CAMs only for **identical ECG and identical target class**; different-target class examples belong to a separate stratum.
- Primary analysis additionally requires correct prediction from both branches and strict independent waveform QC.
- Separate per-record min–max normalization over the same valid ECG x columns.
- Report 6-pixel waveform mask (primary), 24-pixel mask (sensitivity), full-height mean (alternative).
- Do not describe shared saliency as faithfulness, do not describe post-hoc fusion as validated improvement over anchor.
- A percentile bootstrap of individual ECG records does not adjust for repeated patients, if any.
