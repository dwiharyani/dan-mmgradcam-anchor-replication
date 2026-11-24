# DAN-MMGradCAM — Module 3 paired deletion pilot (HPC)

This package is an **adaptation**, not an exact reproduction, of the supplied visible excerpt from `run_dan_module2_deletion_test.py` (the excerpt only exposed selected portions of the 833-line source).

## Experiment and scientific safeguards

- Primary cohort: **693** strict-QC ECG IDs from `module3b_primary_qc693_v1`, exactly matched with Module 3A by `cohort_index`.
- Test methods: Module 3A `R`, original fusion `Fbase`, and Module 3B `Rtrace` at lambda **0.5, 1, 2**. Add `Fmu_matched30` only for records with `full_joint_30=True`.
- Two CNN branches: 1D standardized signal and 2D image **rerendered after the same raw ECG deletion**.
- Target score: original ground-truth class probability; baseline must predict that class on **both** branches.
- Geometry: 600-column attribution mapped to 5000 ECG samples with X0=58, X1=1021, original width 1080. Reject invalid/nonfinite maps.
- Blocking and perturbation: 50 samples = 0.1 second at 500 Hz, high-mean greedy nonoverlapping blocks with stride 5; delete fractions 0.05, 0.10, 0.15, 0.20, 0.25, 0.30; replace each block with linear interpolation between samples immediately outside it.
- Random negative control: 5 fixed-seed, nonoverlapping randomly selected block sets per ECG/fraction, shared across methods and branches. **This is a documented implementation choice; exact random-block equivalence to the unseen part of the Module 2 script has not been verified.**
- QC: compare recomputed **both** branches' baseline probabilities with original `noise_stability_maps.npz`, using `np.allclose(atol=1e-5, rtol=1e-5)`; no silent relaxation. If a job fails, investigate before analysis. Note that individually isolated inference succeeded for ECG1162 in Module3B while batch failed; this experiment therefore processes **one ECG per process**.
- Outputs: one raw CSV and audit JSON per ECG. No overwrite of previous outputs. Separate paired AUC summary aggregates per ECG (not per perturbation row).
- Analytical interpretation: larger deletion probability-drop AUC indicates the selected windows have stronger score impact under this perturbation. It is not causal proof of faithfulness; no novelty or superiority claims without sanity checks and geometry-matched controls.
- Provenance limitation persists: archive s9534 reports TensorFlow 2.21, HPC runs 2.19 with oneDNN disabled. Exact archive-producing helper/2D renderer and scaler fit cohort are not independently verified.

## Transfer files to HPC

Download the ZIP from ChatGPT to your Mac and transfer it to your HPC login host with your existing SSH/SCP access. Example **on your Mac** (replace the hostname):

```bash
scp module3_faithfulness_bundle.zip g202527350@YOUR_HPC_LOGIN_HOST:~/mm_gradcam/module3_aligned/
```

Then **on HPC**:

```bash
M3="$HOME/mm_gradcam/module3_aligned"
mkdir -p "$M3/scripts" "$M3/logs" "$M3/outputs/module3_faithfulness_primary_qc693_v1/raw"
unzip -o "$M3/module3_faithfulness_bundle.zip" \
  module3_faithfulness.py module3_faithfulness_summary.py module3_faithfulness.sbatch \
  -d "$M3/scripts"
python -m py_compile "$M3/scripts/module3_faithfulness.py" "$M3/scripts/module3_faithfulness_summary.py"
bash -n "$M3/scripts/module3_faithfulness.sbatch"
```

## First: **pilot only**, 8 widely spread indices

```bash
M3="$HOME/mm_gradcam/module3_aligned"
sbatch --array=0,100,200,300,400,500,600,692%2 "$M3/scripts/module3_faithfulness.sbatch"
```

After Slurm returns the JOB_ID, check:

```bash
sacct -j JOB_ID -X --format=JobID%20,State%15,ExitCode,Elapsed
find "$M3/outputs/module3_faithfulness_primary_qc693_v1/raw" -name 'faith_????.csv' | wc -l
for p in "$M3"/logs/m3faith_JOB_ID_*.log; do tail -n 10 "$p"; done
```

For a quick check of the pilot without a bootstrap summary yet:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path.home()/'mm_gradcam/module3_aligned/outputs/module3_faithfulness_primary_qc693_v1/raw'
files=sorted(p.glob('faith_????.csv'))
print('Pilot files:',len(files))
for f in files:
    df=pd.read_csv(f)
    print(f.name,'ECG:',df.cohort_index.iat[0], 'rows:',len(df),
          'methods:',sorted(df.method.unique()))
PY
```

## After inspecting pilot QC, run full primary cohort

**Only** if pilot passed and method/latency are acceptable:

```bash
M3="$HOME/mm_gradcam/module3_aligned"
sbatch --array=0-692%8 "$M3/scripts/module3_faithfulness.sbatch"
```

The Slurm script skips indices that already have **both** a CSV and audit JSON from the pilot. If an index has only a partial file, diagnose it rather than treating it as complete. HPC array concurrency can be lowered if needed.

## Summary, only when all 693 strict-QC jobs pass

```bash
M3="$HOME/mm_gradcam/module3_aligned"
python "$M3/scripts/module3_faithfulness_summary.py" \
  --raw-dir "$M3/outputs/module3_faithfulness_primary_qc693_v1/raw" \
  --m3b-dir "$M3/outputs/module3b_primary_qc693_v1" \
  --out-dir "$M3/outputs/module3_faithfulness_primary_qc693_v1/summary" \
  --expected-records 693
```

Summary outputs:

- `per_ecg_branch_auc.csv`
- `paired_auc_comparisons.csv` with class-wise/branch-wise paired differences vs Module 3A `R` and percentile bootstrapped 95% intervals (2000 ECG-level resamples)
- `faithfulness_summary.json`

The analysis with relaxed QC for ECG1346 (`n=694`) is **not implemented by this strict-primary pilot package**; it should be a separate, documented sensitivity extension *after* primary QC succeeds, rather than silently applying a different tolerance across all 693 ECGs.

## Important next step not covered

This is deletion faithfulness-style evidence only. Model-parameter randomization sanity, geometry-matched negative controls, robustness across block sizes, and 2D rendering equivalence audits remain outstanding. Do **not** conflate deletion AUC with an established improvement of explanation quality.
