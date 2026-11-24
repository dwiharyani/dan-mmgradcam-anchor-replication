# HPC-trained models and curated code

**Project type:** independent replication of the 2026 anchor MM-GradCAM paper, followed by exploratory DAN-MMGradCAM analysis.

## Model checkpoints

Trained model binaries are distributed as **GitHub Release assets**, not inside Git history or the `data/` directory.

**[Download 1D and 2D trained models](https://github.com/dwiharyani/dan-mmgradcam-anchor-replication/releases/tag/v0.2.0-hpc-trained-models)**

Both `.keras` model files originate from this project's independent HPC experiments. See the release asset names and the `reports/transfer_manifest.json` file for provenance. Load only model files you trust. The original paper authors' model repositories are linked in the main README.

## Code from HPC

`/scripts/hpc/` contains 19 selectively exported scripts from the actual HPC workspace. Original paths such as `src/` are preserved underneath that folder. Code not selected for the release, including legacy class-order pairing and older fusion implementations, is intentionally excluded.

The remaining post-hoc research code is already in `/scripts/`. Scripts are not claimed to constitute a complete standalone training environment unless the required model-training entry points and their dependencies are present.

## Results included

The repository publishes seven aggregate research figures under `figures/results/`, selected anchor-style alignment figures under `figures/alignment/`, and aggregated JSON summaries under `reports/`. No dataset, original ECG recordings, numeric Grad-CAM arrays, per-record metrics or per-patient identifiers are uploaded.

**Limitations:** 2D CAM native grid 9×18, interpolated for analysis; absolute image-to-time registration is still provisional, and Module 1 numerical agreement is exploratory.
