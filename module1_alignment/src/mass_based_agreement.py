from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 2: MASS-BASED ATTRIBUTION AGREEMENT
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
GRAD = BASE / "gradcam_full"

OUT = BASE / "mass_based_agreement"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

# ============================================================
# 1. LOAD ORIGINAL ATTRIBUTION PROFILES
# ============================================================

g1 = np.load(
    ALIGN / "gradcam_1d_aligned.npy"
).astype(np.float64)

g2 = np.load(
    ALIGN / "gradcam_2d_projected.npy"
).astype(np.float64)

valid = np.load(
    ALIGN / "valid_temporal_mask.npy"
).astype(bool)

metadata = pd.read_csv(
    GRAD / "gradcam_metadata.csv"
)

original = pd.read_csv(
    ALIGN / "agreement_metrics.csv"
)

geometry = pd.read_csv(
    ALIGN / "temporal_geometry.csv"
)

assert g1.shape == (1054, 600)
assert g2.shape == g1.shape
assert valid.shape == g1.shape

assert np.isfinite(g1).all()
assert np.isfinite(g2).all()

assert (g1 >= 0).all()
assert (g2 >= 0).all()

assert metadata["cohort_index"].is_unique

assert np.array_equal(
    metadata["cohort_index"],
    original["cohort_index"]
)

assert np.array_equal(
    metadata["cohort_index"],
    geometry["cohort_index"]
)

assert np.array_equal(
    valid.sum(axis=1),
    geometry["valid_columns"].to_numpy()
)

N, W = g1.shape

# ============================================================
# 2. LOAD CLASSIFICATION PREDICTIONS
# ============================================================

pred1 = pd.read_csv(
    BASE / "1d_test_predictions.csv"
)

pred2 = pd.read_csv(
    BASE / "2d_test_predictions.csv"
)

assert pred1["cohort_index"].is_unique
assert pred2["cohort_index"].is_unique

pred1 = pred1.set_index("cohort_index").loc[
    metadata["cohort_index"]
]

pred2 = pred2.set_index("cohort_index").loc[
    metadata["cohort_index"]
]

assert np.array_equal(
    pred1["true_class"].to_numpy(),
    metadata["class_name"].to_numpy()
)

assert np.array_equal(
    pred2["true_class"].to_numpy(),
    metadata["class_name"].to_numpy()
)

correct1 = (
    pred1["true_class"] ==
    pred1["predicted_class"]
).to_numpy()

correct2 = (
    pred2["true_class"] ==
    pred2["predicted_class"]
).to_numpy()

both_correct = correct1 & correct2

# ============================================================
# 3. INITIALIZE OUTPUT ARRAYS
# ============================================================

p = np.full((N, W), np.nan)
q = np.full((N, W), np.nan)

shared_mass_maps = np.full((N, W), np.nan)

mass_valid = np.zeros(N, dtype=bool)

records = []

# ============================================================
# 4. MASS NORMALIZATION AND AGREEMENT
# ============================================================

for i in range(N):

    mask = valid[i]

    a = g1[i, mask]
    b = g2[i, mask]

    sum1 = float(a.sum())
    sum2 = float(b.sum())

    zero_mass_1d = sum1 <= EPS
    zero_mass_2d = sum2 <= EPS

    eligible = (
        mask.any()
        and not zero_mass_1d
        and not zero_mass_2d
    )

    overlap = np.nan
    jaccard = np.nan
    tv = np.nan

    if eligible:

        # Normalize each attribution distribution.
        pi = a / sum1
        qi = b / sum2

        assert np.isclose(pi.sum(), 1.0)
        assert np.isclose(qi.sum(), 1.0)

        # Shared attribution mass.
        shared = np.minimum(pi, qi)

        overlap = float(shared.sum())

        # Weighted Jaccard.
        jaccard = overlap / (2.0 - overlap)

        # Total variation distance.
        tv = 0.5 * np.abs(pi - qi).sum()

        # Independent mathematical checks.
        assert -EPS <= overlap <= 1.0 + EPS
        assert -EPS <= jaccard <= 1.0 + EPS

        assert np.isclose(
            tv,
            1.0 - overlap,
            atol=1e-10
        )

        direct_jaccard = (
            np.minimum(pi, qi).sum()
            / np.maximum(pi, qi).sum()
        )

        assert np.isclose(
            jaccard,
            direct_jaccard,
            atol=1e-10
        )

        # Store valid temporal positions only.
        p[i, mask] = pi
        q[i, mask] = qi

        shared_mass_maps[i, mask] = shared

        mass_valid[i] = True

    records.append({
        "cohort_index": int(
            metadata.iloc[i]["cohort_index"]
        ),
        "class_name": metadata.iloc[i]["class_name"],
        "valid_columns": int(mask.sum()),
        "sum_1d_attribution": sum1,
        "sum_2d_attribution": sum2,
        "zero_mass_1d": zero_mass_1d,
        "zero_mass_2d": zero_mass_2d,
        "mass_valid": eligible,
        "shared_mass": overlap,
        "mass_weighted_jaccard": jaccard,
        "total_variation": tv,
        "correct_1d": bool(correct1[i]),
        "correct_2d": bool(correct2[i]),
        "both_correct": bool(both_correct[i]),
        "original_mean_agreement":
            original.iloc[i]["mean_agreement"],
        "original_mean_disagreement":
            original.iloc[i]["mean_disagreement"],
        "original_weighted_jaccard":
            original.iloc[i]["weighted_jaccard"],
        "original_valid_similarity":
            original.iloc[i]["valid_similarity"]
    })

df = pd.DataFrame(records)

# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

np.save(OUT / "mass_1d.npy", p)
np.save(OUT / "mass_2d.npy", q)

np.save(
    OUT / "shared_mass_maps.npy",
    shared_mass_maps
)

np.save(
    OUT / "mass_valid_recordings.npy",
    mass_valid
)

df.to_csv(
    OUT / "mass_agreement_metrics.csv",
    index=False
)

# ============================================================
# 6. OVERALL RESULTS
# ============================================================

eligible_df = df[df["mass_valid"]].copy()

metric_cols = [
    "shared_mass",
    "mass_weighted_jaccard",
    "total_variation"
]

print("\n" + "=" * 70)
print("STEP 2: MASS-BASED ATTRIBUTION AGREEMENT")
print("=" * 70)

print("\nCOHORT SUMMARY")

print("Total recordings:", len(df))
print("Valid mass pairs:", len(eligible_df))
print("Excluded mass pairs:", N - len(eligible_df))

print(
    "Zero-mass 1D:",
    df["zero_mass_1d"].sum()
)

print(
    "Zero-mass 2D:",
    df["zero_mass_2d"].sum()
)

print(
    "Both CNNs correct:",
    df["both_correct"].sum()
)

print(
    "Both correct and mass-valid:",
    eligible_df["both_correct"].sum()
)

print("\nOVERALL MASS-BASED RESULTS")

overall = eligible_df[metric_cols].agg([
    "count", "mean", "median",
    "std", "min", "max"
])

print(overall.round(6).to_string())

overall.to_csv(
    OUT / "mass_overall_summary.csv"
)

# ============================================================
# 7. CLASS-WISE RESULTS
# ============================================================

class_summary = eligible_df.groupby(
    "class_name"
)[metric_cols].agg([
    "count", "mean", "median", "std"
])

print("\nCLASS-WISE RESULTS")
print(class_summary.round(6).to_string())

class_summary.to_csv(
    OUT / "mass_classwise_summary.csv"
)

# ============================================================
# 8. PREDICTION CORRECTNESS SUBGROUPS
# ============================================================

correctness_summary = eligible_df.groupby(
    "both_correct"
)[metric_cols].agg([
    "count", "mean", "median", "std"
])

print("\nPREDICTION CORRECTNESS RESULTS")

print(
    correctness_summary.round(6).to_string()
)

correctness_summary.to_csv(
    OUT / "mass_correctness_summary.csv"
)

# ============================================================
# 9. AFIB COHORT 209 CASE STUDY
# ============================================================

print("\nAFIB COHORT 209")

print(
    df.loc[
        df["cohort_index"] == 209
    ].to_string(index=False)
)

print("\nAll mathematical checks passed.")
print("Results saved in:", OUT)
print("STEP 2 COMPLETED")
