from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 3: DIRECTIONAL DISAGREEMENT ANALYSIS
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
MASS = BASE / "mass_based_agreement"

OUT = BASE / "directional_disagreement"
OUT.mkdir(parents=True, exist_ok=True)

TOP_FRACTION = 0.15
THRESHOLD = 1.0 / 3.0

# ============================================================
# 1. LOAD STEP 2 OUTPUTS
# ============================================================

p = np.load(MASS / "mass_1d.npy")
q = np.load(MASS / "mass_2d.npy")

mass_valid = np.load(
    MASS / "mass_valid_recordings.npy"
).astype(bool)

valid = np.load(
    ALIGN / "valid_temporal_mask.npy"
).astype(bool)

metrics = pd.read_csv(
    MASS / "mass_agreement_metrics.csv"
)

assert p.shape == (1054, 600)
assert q.shape == p.shape
assert valid.shape == p.shape
assert len(metrics) == len(p)

assert np.array_equal(
    mass_valid,
    metrics["mass_valid"].to_numpy()
)

N, W = p.shape

# ============================================================
# 2. INITIALIZE OUTPUTS
# ============================================================

direction_ratio = np.full((N, W), np.nan)

# Direction labels:
# -2 = invalid or nonsalient
# -1 = 2D dominant
#  0 = balanced
#  1 = 1D dominant

direction_labels = np.full(
    (N, W), -2, dtype=np.int8
)

salient_masks = np.zeros(
    (N, W), dtype=bool
)

records = []

# ============================================================
# 3. TOP-K SELECTION
# ============================================================

def top_k_indices(values, indices, k):

    # Stable ordering for deterministic handling of ties.
    order = np.argsort(
        -values[indices],
        kind="stable"
    )

    return indices[order[:k]]

# ============================================================
# 4. DIRECTIONAL DISAGREEMENT
# ============================================================

for i in range(N):

    if not mass_valid[i]:
        continue

    mask = valid[i]
    indices = np.flatnonzero(mask)

    n_valid = len(indices)

    k = int(np.ceil(
        TOP_FRACTION * n_valid
    ))

    assert np.isclose(
        p[i, mask].sum(), 1.0
    )

    assert np.isclose(
        q[i, mask].sum(), 1.0
    )

    # Select salient positions separately.
    top1 = top_k_indices(
        p[i], indices, k
    )

    top2 = top_k_indices(
        q[i], indices, k
    )

    # Union of salient positions.
    salient = np.union1d(top1, top2)

    # Exclude positions with zero combined mass.
    denominator = p[i, salient] + q[i, salient]

    salient = salient[denominator > 0]

    salient_masks[i, salient] = True

    if len(salient) == 0:
        continue

    # Signed directional ratio.
    ratio = (
        p[i, salient] - q[i, salient]
    ) / (
        p[i, salient] + q[i, salient]
    )

    assert np.all(ratio >= -1.0 - 1e-10)
    assert np.all(ratio <= 1.0 + 1e-10)

    direction_ratio[i, salient] = ratio

    dominant1 = ratio > THRESHOLD
    dominant2 = ratio < -THRESHOLD

    balanced = (
        ~dominant1 & ~dominant2
    )

    direction_labels[
        i, salient[dominant1]
    ] = 1

    direction_labels[
        i, salient[dominant2]
    ] = -1

    direction_labels[
        i, salient[balanced]
    ] = 0

    n1 = int(dominant1.sum())
    n2 = int(dominant2.sum())
    nb = int(balanced.sum())

    ns = len(salient)

    assert n1 + n2 + nb == ns

    records.append({

        "cohort_index":
            metrics.iloc[i]["cohort_index"],

        "class_name":
            metrics.iloc[i]["class_name"],

        "both_correct":
            metrics.iloc[i]["both_correct"],

        "valid_columns": n_valid,

        "salient_columns": ns,

        "top_k": k,

        "1d_dominant_columns": n1,

        "2d_dominant_columns": n2,

        "balanced_columns": nb,

        # Percentages relative to salient positions.
        "1d_dominant_pct_salient":
            100.0 * n1 / ns,

        "2d_dominant_pct_salient":
            100.0 * n2 / ns,

        "balanced_pct_salient":
            100.0 * nb / ns,

        # Percentages relative to the entire valid ECG.
        "1d_dominant_pct_valid":
            100.0 * n1 / n_valid,

        "2d_dominant_pct_valid":
            100.0 * n2 / n_valid,

        "balanced_pct_valid":
            100.0 * nb / n_valid,

        "salient_pct_valid":
            100.0 * ns / n_valid,

        "mean_direction_ratio":
            float(ratio.mean()),

        "shared_mass":
            metrics.iloc[i]["shared_mass"]

    })

df = pd.DataFrame(records)

# ============================================================
# 5. VERIFY PERCENTAGES
# ============================================================

salient_cols = [
    "1d_dominant_pct_salient",
    "2d_dominant_pct_salient",
    "balanced_pct_salient"
]

valid_cols = [
    "1d_dominant_pct_valid",
    "2d_dominant_pct_valid",
    "balanced_pct_valid"
]

assert np.allclose(
    df[salient_cols].sum(axis=1),
    100.0
)

assert np.allclose(
    df[valid_cols].sum(axis=1),
    df["salient_pct_valid"]
)

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

np.save(
    OUT / "direction_ratio.npy",
    direction_ratio
)

np.save(
    OUT / "direction_labels.npy",
    direction_labels
)

np.save(
    OUT / "salient_masks.npy",
    salient_masks
)

df.to_csv(
    OUT / "directional_metrics.csv",
    index=False
)

# ============================================================
# 7. OVERALL SUMMARY
# ============================================================

summary_cols = (
    salient_cols + valid_cols +
    ["salient_pct_valid", "mean_direction_ratio"]
)

overall = df[summary_cols].agg([
    "count", "mean", "median", "std"
])

overall.to_csv(
    OUT / "directional_overall_summary.csv"
)

print("\n" + "=" * 70)
print("STEP 3: DIRECTIONAL DISAGREEMENT")
print("=" * 70)

print("\nValid recordings:", len(df))

print("\nOVERALL SUMMARY")
print(overall.round(4).to_string())

# ============================================================
# 8. CLASS-WISE SUMMARY
# ============================================================

class_summary = df.groupby(
    "class_name"
)[summary_cols].agg([
    "mean", "median", "std"
])

class_summary.to_csv(
    OUT / "directional_classwise_summary.csv"
)

print("\nCLASS-WISE MEANS (% OF VALID ECG)")

print(
    df.groupby("class_name")[valid_cols]
    .mean().round(4).to_string()
)

print("\nCLASS-WISE MEANS (% OF SALIENT ECG)")

print(
    df.groupby("class_name")[salient_cols]
    .mean().round(4).to_string()
)

# ============================================================
# 9. PREDICTION CORRECTNESS
# ============================================================

correctness_summary = df.groupby(
    "both_correct"
)[summary_cols].agg([
    "mean", "median", "std"
])

correctness_summary.to_csv(
    OUT / "directional_correctness_summary.csv"
)

print("\nPREDICTION CORRECTNESS SUMMARY")

print(
    df.groupby("both_correct")[valid_cols]
    .mean().round(4).to_string()
)

# ============================================================
# 10. AFIB COHORT 209
# ============================================================

print("\nAFIB COHORT 209")

print(
    df.loc[df["cohort_index"] == 209]
    .to_string(index=False)
)

print("\nAll verification checks passed.")
print("Results saved in:", OUT)
print("STEP 3 COMPLETED")
