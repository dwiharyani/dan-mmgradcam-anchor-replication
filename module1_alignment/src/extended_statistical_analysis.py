from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from scipy.stats import (
    kruskal,
    mannwhitneyu,
    wilcoxon
)

# ============================================================
# STEP 6: EXTENDED STATISTICAL ANALYSIS
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"

MASS = BASE / "mass_based_agreement"
DIRECTION = BASE / "directional_disagreement"
ENTROPY = BASE / "attribution_entropy"
NULL = BASE / "within_class_mispairing"

OUT = BASE / "extended_statistical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 2000
rng = np.random.default_rng(SEED)

# ============================================================
# 1. LOAD AND MERGE EXPERIMENTAL RESULTS
# ============================================================

mass = pd.read_csv(
    MASS / "mass_agreement_metrics.csv"
)

direction = pd.read_csv(
    DIRECTION / "directional_metrics.csv"
)

entropy = pd.read_csv(
    ENTROPY / "entropy_metrics.csv"
)

mass = mass.loc[mass["mass_valid"]].copy()

direction_cols = [
    "cohort_index",
    "1d_dominant_pct_valid",
    "2d_dominant_pct_valid",
    "balanced_pct_valid",
    "1d_dominant_pct_salient",
    "2d_dominant_pct_salient",
    "balanced_pct_salient"
]

entropy_cols = [
    "cohort_index",
    "entropy_1d",
    "entropy_2d",
    "entropy_difference"
]

df = mass.merge(
    direction[direction_cols],
    on="cohort_index",
    validate="one_to_one"
).merge(
    entropy[entropy_cols],
    on="cohort_index",
    validate="one_to_one"
)

assert len(df) == 1040
assert df["cohort_index"].is_unique

df.to_csv(
    OUT / "extended_recording_metrics.csv",
    index=False
)

METRICS = [
    "shared_mass",
    "mass_weighted_jaccard",
    "1d_dominant_pct_valid",
    "2d_dominant_pct_valid",
    "balanced_pct_valid",
    "entropy_1d",
    "entropy_2d",
    "entropy_difference"
]

assert np.isfinite(df[METRICS].to_numpy()).all()

# ============================================================
# 2. STATISTICAL UTILITIES
# ============================================================

def bootstrap_ci(values, statistic=np.median):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    estimates = np.array([
        statistic(
            rng.choice(
                values,
                size=len(values),
                replace=True
            )
        )
        for _ in range(N_BOOT)
    ])

    return tuple(
        np.percentile(estimates, [2.5, 97.5])
    )


def holm_adjust(p_values):

    p = np.asarray(p_values, dtype=float)

    order = np.argsort(p)
    sorted_p = p[order]

    m = len(p)

    adjusted_sorted = np.maximum.accumulate(
        sorted_p * (m - np.arange(m))
    )

    adjusted_sorted = np.minimum(
        adjusted_sorted, 1.0
    )

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted

    return adjusted


def rank_biserial(u, n1, n2):

    return (
        2.0 * u / (n1 * n2) - 1.0
    )

# ============================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

bootstrap_records = []

groups = [("All", df)]

groups += list(
    df.groupby("class_name")
)

groups += [
    (
        "Both correct",
        df[df["both_correct"]]
    ),
    (
        "At least one incorrect",
        df[~df["both_correct"]]
    )
]

for group_name, group in groups:

    for metric in METRICS:

        values = group[metric].to_numpy()

        low, high = bootstrap_ci(values)

        bootstrap_records.append({

            "group": group_name,
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "median": np.median(values),
            "std": values.std(ddof=1),
            "median_ci_lower": low,
            "median_ci_upper": high

        })

bootstrap_df = pd.DataFrame(
    bootstrap_records
)

bootstrap_df.to_csv(
    OUT / "bootstrap_confidence_intervals.csv",
    index=False
)

print("\n" + "=" * 75)
print("OVERALL BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 75)

print(
    bootstrap_df[
        bootstrap_df["group"] == "All"
    ].round(6).to_string(index=False)
)

# ============================================================
# 4. KRUSKAL-WALLIS CLASS COMPARISONS
# ============================================================

kw_records = []

for metric in METRICS:

    groups = [
        group[metric].to_numpy()
        for _, group in df.groupby("class_name")
    ]

    test = kruskal(*groups)

    n = len(df)
    k = len(groups)

    effect = max(
        0.0,
        (test.statistic - k + 1) / (n - k)
    )

    kw_records.append({

        "metric": metric,
        "H": test.statistic,
        "p_value": test.pvalue,
        "epsilon_squared": effect

    })

kw_df = pd.DataFrame(kw_records)

kw_df["p_holm"] = holm_adjust(
    kw_df["p_value"]
)

kw_df["significant"] = (
    kw_df["p_holm"] < 0.05
)

kw_df.to_csv(
    OUT / "kruskal_wallis_extended.csv",
    index=False
)

print("\n" + "=" * 75)
print("KRUSKAL-WALLIS CLASS COMPARISONS")
print("=" * 75)

print(kw_df.to_string(index=False))

# ============================================================
# 5. PAIRWISE CLASS COMPARISONS
# ============================================================

pairwise_records = []

classes = sorted(
    df["class_name"].unique()
)

for metric in METRICS:

    records = []

    for class1, class2 in combinations(classes, 2):

        a = df.loc[
            df["class_name"] == class1,
            metric
        ].to_numpy()

        b = df.loc[
            df["class_name"] == class2,
            metric
        ].to_numpy()

        test = mannwhitneyu(
            a, b,
            alternative="two-sided"
        )

        records.append({

            "metric": metric,
            "class1": class1,
            "class2": class2,
            "n1": len(a),
            "n2": len(b),
            "median1": np.median(a),
            "median2": np.median(b),
            "U": test.statistic,
            "p_value": test.pvalue,
            "rank_biserial":
                rank_biserial(
                    test.statistic,
                    len(a),
                    len(b)
                )

        })

    temp = pd.DataFrame(records)

    temp["p_holm"] = holm_adjust(
        temp["p_value"]
    )

    temp["significant"] = (
        temp["p_holm"] < 0.05
    )

    pairwise_records.append(temp)

pairwise_df = pd.concat(
    pairwise_records,
    ignore_index=True
)

pairwise_df.to_csv(
    OUT / "pairwise_class_comparisons.csv",
    index=False
)

print("\nSIGNIFICANT SHARED-MASS CLASS COMPARISONS")

print(
    pairwise_df[
        (pairwise_df["metric"] == "shared_mass")
        & pairwise_df["significant"]
    ].to_string(index=False)
)

# ============================================================
# 6. PREDICTION CORRECTNESS COMPARISONS
# ============================================================

correctness_records = []

for metric in METRICS:

    a = df.loc[
        df["both_correct"], metric
    ].to_numpy()

    b = df.loc[
        ~df["both_correct"], metric
    ].to_numpy()

    test = mannwhitneyu(
        a, b,
        alternative="two-sided"
    )

    correctness_records.append({

        "metric": metric,
        "n_correct": len(a),
        "n_incorrect": len(b),
        "median_correct": np.median(a),
        "median_incorrect": np.median(b),
        "median_difference":
            np.median(a) - np.median(b),
        "U": test.statistic,
        "p_value": test.pvalue,
        "rank_biserial":
            rank_biserial(
                test.statistic,
                len(a),
                len(b)
            )

    })

correctness_df = pd.DataFrame(
    correctness_records
)

correctness_df["p_holm"] = holm_adjust(
    correctness_df["p_value"]
)

correctness_df["significant"] = (
    correctness_df["p_holm"] < 0.05
)

correctness_df.to_csv(
    OUT / "correctness_statistical_comparison.csv",
    index=False
)

print("\n" + "=" * 75)
print("PREDICTION CORRECTNESS COMPARISONS")
print("=" * 75)

print(
    correctness_df.round(6).to_string(index=False)
)

# ============================================================
# 7. HOLM CORRECTION OF NULL EXPERIMENTS
# ============================================================

null_summaries = []

for cohort in [
    "all_mass_valid",
    "both_correct_mass_valid"
]:

    summary = pd.read_csv(
        NULL / cohort / "null_classwise_summary.csv"
    )

    # Correct across all 12 class-metric tests
    # within each cohort.

    summary["p_holm"] = holm_adjust(
        summary["permutation_p_value"]
    )

    summary["significant"] = (
        summary["p_holm"] < 0.05
    )

    null_summaries.append(summary)

null_corrected = pd.concat(
    null_summaries,
    ignore_index=True
)

null_corrected.to_csv(
    OUT / "null_holm_corrected.csv",
    index=False
)

print("\n" + "=" * 75)
print("HOLM-CORRECTED NULL EXPERIMENTS")
print("=" * 75)

print(
    null_corrected[[
        "cohort",
        "class_name",
        "metric",
        "median_difference",
        "permutation_p_value",
        "p_holm",
        "significant"
    ]].to_string(index=False)
)

# ============================================================
# 8. OVERALL STRATIFIED NULL COMPARISON
# ============================================================

overall_records = []

for cohort in [
    "all_mass_valid",
    "both_correct_mass_valid"
]:

    archives = []

    for cls in classes:

        with np.load(
            NULL / cohort / f"{cls}_null.npz"
        ) as archive:

            archives.append({
                "actual": archive["actual"],
                "null": archive["null"],
                "metric_names":
                    archive["metric_names"]
            })

    names = archives[0]["metric_names"]

    for m, metric in enumerate(names):

        actual_parts = []
        null_parts = []

        for archive in archives:

            actual = archive["actual"][:, m]
            null = archive["null"][:, :, m]

            keep = np.isfinite(actual)

            actual_parts.append(actual[keep])
            null_parts.append(null[:, keep])

        actual = np.concatenate(actual_parts)

        null = np.concatenate(
            null_parts, axis=1
        )

        assert np.isfinite(null).all()

        n_perm = null.shape[0]

        observed_median = np.median(actual)

        null_medians = np.median(
            null, axis=1
        )

        p_value = (
            1 + np.sum(
                null_medians >= observed_median
            )
        ) / (n_perm + 1)

        individual_null_mean = null.mean(axis=0)

        paired_difference = (
            actual - individual_null_mean
        )

        ci_low, ci_high = bootstrap_ci(
            paired_difference,
            statistic=np.mean
        )

        overall_records.append({

            "cohort": cohort,
            "metric": metric,
            "n": len(actual),
            "n_permutations": n_perm,
            "actual_mean": actual.mean(),
            "actual_median": observed_median,
            "null_mean": null.mean(),
            "null_median": np.median(null),
            "mean_difference":
                paired_difference.mean(),
            "conditional_ci_lower": ci_low,
            "conditional_ci_upper": ci_high,
            "permutation_p_value": p_value

        })

overall_null = pd.DataFrame(
    overall_records
)

overall_null["p_holm"] = (
    overall_null.groupby("cohort")[
        "permutation_p_value"
    ].transform(holm_adjust)
)

overall_null.to_csv(
    OUT / "overall_stratified_null.csv",
    index=False
)

print("\n" + "=" * 75)
print("OVERALL STRATIFIED NULL COMPARISON")
print("=" * 75)

print(
    overall_null.round(6).to_string(index=False)
)

# ============================================================
# 9. PAIRED DIRECTIONAL COMPARISON
# ============================================================

difference = (
    df["1d_dominant_pct_valid"]
    - df["2d_dominant_pct_valid"]
).to_numpy()

test = wilcoxon(
    difference,
    alternative="two-sided"
)

low, high = bootstrap_ci(difference)

direction_stats = pd.DataFrame([{

    "n": len(difference),

    "mean_difference_percentage_points":
        difference.mean(),

    "median_difference_percentage_points":
        np.median(difference),

    "median_ci_lower": low,

    "median_ci_upper": high,

    "wilcoxon_statistic": test.statistic,

    "p_value": test.pvalue

}])

direction_stats.to_csv(
    OUT / "directional_paired_comparison.csv",
    index=False
)

print("\nDIRECTIONAL DOMINANCE COMPARISON")

print(
    direction_stats.to_string(index=False)
)

print("\n" + "=" * 75)
print("STEP 6 COMPLETED")
print("Results saved in:", OUT)
print("=" * 75)
