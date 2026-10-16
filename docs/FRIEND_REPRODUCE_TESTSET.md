# Reproduce the exact MM-GradCAM anchor-replication test cohort

This repository documents an **independent replication** of the 1D/2D MM-GradCAM anchor pipeline, followed by exploratory DAN-MMGradCAM explanation agreement analysis. It does not contain ECG datasets.

## What exactly is fixed?

- **Full filtered cohort:** 10,588 ECG recordings.
- **Held-out classification test cohort:** exactly **1,054 identical ECG records** in both representations; **AFib 222, GSVT 222, SB 388, SR 222**.
- **Exploratory DAN primary subset:** 727 of the 1,054 passed both-correct and strict temporal-QC criteria. **Do not** evaluate classification accuracy using only these 727.
- The exact test list is in [`reproducibility/test_split_1054.csv`](../reproducibility/test_split_1054.csv). Each row contains the original public ECG CSV filename, Kaggle 2D image filename, correct class, cohort index, and the distinct 1D/2D order indices.
- ECG 1D and 2D must be paired by their **record identity** (`cohort_index` and corresponding filenames), never by array position. These two orders differ for most cases.

## Steps for a collaborator (Mac, Linux, or HPC)

### 1. Clone the repository

```bash
git clone https://github.com/dwiharyani/dan-mmgradcam-anchor-replication.git
cd dan-mmgradcam-anchor-replication
python3 tools/verify_exact_test_split.py
```

Expect `PASS | 1054 test samples` and a valid SHA-256 checksum.

### 2. Download both original sources independently

Download and extract the sources in a local directory (do not upload them to GitHub):

- **Numerical 1D:** [Chapman–Shaoxing ECGData.zip, Figshare](https://figshare.com/articles/dataset/ECGData_zip/8361860), 12-lead CSVs. Lead II at 500 Hz/10 s (5,000 values) is the branch input.
- **ECG 2D images:** [Author-provided ImageData-label2, Kaggle](https://www.kaggle.com/datasets/fatmamurat/imagedata-label2), images with names like `hasta_<cohort_index>.png` under one of the four rhythm classes.

The `reproducibility/` manifest is a **file selection list**, not raw data. The same `hasta_N.png` is matched to index `N` of the audited cohort; do not construct a new random test split.

### 3. Verify exactly the same source files exist

Replace example paths with your own extracted folders:

```bash
python3 tools/verify_exact_test_split.py \
  --ecg-csv-dir ~/datasets/ECGData \
  --image-root ~/datasets/ImageData-label2
```

This checks that all 1,054 CSVs and PNGs are available, *not* that their numerical preprocessing has been reproduced.

If the Kaggle images are mixed into one full dataset rather than split into `Testdata`, create a local 1,054-image symlink view without duplicating or uploading anything:

```bash
python3 tools/prepare_test_images.py \
  --kaggle-root ~/datasets/ImageData-label2 \
  --out data/2d_test/Testdata
python3 tools/verify_exact_test_split.py \
  --image-root data/2d_test/Testdata
```

### 4. Download the *same trained checkpoints*

Model release: [v0.2.0-hpc-trained-models](https://github.com/dwiharyani/dan-mmgradcam-anchor-replication/releases/tag/v0.2.0-hpc-trained-models).

With GitHub CLI:

```bash
gh release download v0.2.0-hpc-trained-models \
  --repo dwiharyani/dan-mmgradcam-anchor-replication \
  --dir models_local
```

`models_local/` is local and should not be committed to Git. Alternatively, download both `.keras` files in a browser.

### 5. Evaluate both models with the same input preparation

Use the exact test entries in the manifest, and the same preparation as the original independent replication's 1D and 2D training/CAM scripts (see `scripts/hpc/src/`):

- **1D input:** Lead-II CSV source, branch-equivalent filtering/scaling/normalization and tensor shape; do **not** simply use 12-lead raw CSV or guess a normalizer.
- **2D input:** same resize dimensions, RGB order, pixel scaling, image loader and model tensor layout as the actual 2D training/evaluation code; do **not** independently invent image preprocessing.
- **Labels:** AFib=0, GSVT=1, SB=2, SR=3 in the released implementation; confirm mapping in training code.
- Preserve test manifest `test_order_1d` and `test_order_2d` explicitly; do not assume they match.
- Use the **same released model weights**. Retraining can produce different results even with the same test data.

**Important limitation:** A fixed test split and downloaded models are *necessary but not sufficient* for bit-for-bit equality. The published code must be checked to extract and reuse preprocessing, inference batch logic, TensorFlow/Keras versions, and prediction-to-class mapping. This starter toolkit validates the split; **it does not claim to be a fully validated turnkey 1D+2D inference evaluator**.

### 6. Compare expected classification test results

Reported on the 1,054 held-out examples of the independent replication (verify with source results before treating as benchmark):

| Branch | Accuracy | Macro F1 | UAR |
|---|---:|---:|---:|
| 1D CNN | 0.947818 | 0.945723 | 0.945290 |
| 2D CNN | 0.966793 | 0.963263 | 0.962513 |

Discrepancies should prompt a check of the exact evaluation checkpoint, preprocessing, split order, TensorFlow version, and 2D image loader. The 2D `best_model.keras` is the **checkpoint used for testing** according to the training script; Grad-CAM model-path provenance should be checked separately.

### 7. If reproducing DAN-MMGradCAM Module 1

This requires generating 1D and 2D Grad-CAM maps, exact pairing, independent waveform QC, and applying strict exclusion criteria. The 727-case cohort is *derived*, not an alternate test split. The current quantitative time mapping is provisional; report agreement/disagreement as exploratory. Read [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for full analysis commands and limitations.

## Public file safety

The exported CSV contains **public source record filenames** and indices but no ECG samples, images, per-record predictions, patient_id field or saliency arrays. **Review whether record filenames may be published under the source providers' terms before pushing the manifest to a public repository.** Do not commit datasets, original labels/metadata sheets, credentials, trained `.keras` files (already in Releases), or HDF5/NPY arrays.
