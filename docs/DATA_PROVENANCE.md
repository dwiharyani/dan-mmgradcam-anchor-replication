# Dataset provenance and release boundaries

**Numerical 1D source:** Zheng et al.'s Chapman–Shaoxing ECG recordings: [ECGData.zip, Figshare](https://figshare.com/articles/dataset/ECGData_zip/8361860) (10,646 original records, 12 leads, 500 Hz, nominal 10 s); the study cohort reduced to 10,588 after excluding 58 incomplete/problematic source records. The Figshare entry currently specifies CC0; check its live terms prior to reuse.

**Original author-provided 2D images:** [fatmamurat/imagedata-label2](https://www.kaggle.com/datasets/fatmamurat/imagedata-label2). This independent replication uses image files paired to the same underlying records. Do not assume that the similarly named Kaggle `label-2` provides the numerical CSV source. Independently check the 2D dataset's license for any republication of images or derivations.

**Test cohort:** 1,054 held-out, 4 classes: AFib 222, GSVT 222, SB 388, SR 222. The numerical 1D and image 2D CAMs are paired by verified exact record identity rather than within-class order.

**Public repository is DATA-FREE:** no raw ECG, image dataset folder, waveform arrays, HDF5, CAM arrays, individual-sample indices, patient mappings, CNN weights, training logs, raw splits, or per-record CSV outputs. Curated visual examples and summary-level figures only.

**Publication:** Murat Duranay et al. (2026), *Scientific Reports*, 16, article 7919. DOI: [10.1038/s41598-026-38654-w](https://doi.org/10.1038/s41598-026-38654-w).
