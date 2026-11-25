from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    wilcoxon
)

# ============================================================
# STEP 7: AGREEMENT FORMULATION AND THRESHOLD SENSITIVITY
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"

ALIGN = BASE / "alignment_0px"
MASS = BASE / "mass_based_agreement"
DIRECTION = BASE / "directional_disagreement"

OUT = BASE / "agreement_formulation_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 2000

rng = np.random.default_rng(SEED)

TOP_FRACTIONS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25
]

THRESHOLDS = [
    0.20,
    1.0 / 3.0,
    0.50,
    0.60,
    2.0 / 3.0
]

# ============================================================
# 1. LOAD EXPERIMENTAL RESULTS
# ============================================================

mass = pd.read_csv(
    MASS / "mass_agreement_metrics.csv"
)

original = pd.read_csv(
    ALIGN / "agreement_metrics.csv"
)

baseline_direction = pd.read_csv(
    DIRECTION / "directional_metrics.csv"
)

p = np.load(
    MASS / "mass_1d.npy"
)

q = np.load(
    MASS / "mass_2d.npy"
)

valid = np.load(
    ALIGN / "valid_temporal_mask.npy"
).astype(bool)

mass_valid = np.load(
    MASS / "mass_valid_recordings.npy"
).astype(bool)

assert p.shape == (1054, 600)
assert q.shape == p.shape
assert valid.shape == p.shape

assert len(mass) == len(p)

assert np.array_equal(
    mass["cohort_index"],
    original["cohort_index"]
)

assert np.array_equal(
    mass_valid,
    mass["mass_valid"].to_numpy()
)

assert mass_valid.sum() == 1040

# ============================================================
# EXPERIMENT A: COMPARISON OF AGREEMENT FORMULATIONS
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT A: AGREEMENT FORMULATION COMPARISON")
print("=" * 75)

df = mass.loc[mass_valid].copy()

comparison_pairs = [

    (
        "original_mean_agreement",
        "shared_mass"
    ),

    (
        "original_weighted_jaccard",
        "mass_weighted_jaccard"
    ),

    (
        "original_mean_disagreement",
        "total_variation"
    )
]

# ============================================================
# 2. CORRELATION ANALYSIS
# ============================================================

correlation_records = []

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

    for metric1, metric2 in comparison_pairs:

        subset = group.copy()

        # Weighted Jaccard comparison requires the
        # original nondegenerate similarity cohort.
        if metric1 == "original_weighted_jaccard":

            subset = subset[
                subset["original_valid_similarity"]
            ]

        x = subset[metric1].to_numpy(dtype=float)
        y = subset[metric2].to_numpy(dtype=float)

        keep = np.isfinite(x) & np.isfinite(y)

        x = x[keep]
        y = y[keep]

        if len(x) < 3:
            continue

        if np.ptp(x) == 0 or np.ptp(y) == 0:
            continue

        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)

        correlation_records.append({

            "group": group_name,

            "original_metric": metric1,

            "extended_metric": metric2,

            "n": len(x),

            "pearson_r": pearson.statistic,

            "pearson_p": pearson.pvalue,

            "spearman_rho": spearman.statistic,

            "spearman_p": spearman.pvalue

        })

correlations = pd.DataFrame(
    correlation_records
)

correlations.to_csv(
    OUT / "formulation_correlations.csv",
    index=False
)

print("\nOVERALL FORMULATION CORRELATIONS")

print(
    correlations[
        correlations["group"] == "All"
    ].to_string(index=False)
)

# ============================================================
# 3. CLASS-WISE COMPARISON ON MATCHED RECORDINGS
# ============================================================

matched = df.loc[
    df["original_valid_similarity"]
].copy()

assert len(matched) == 1033

comparison_metrics = [

    "original_mean_agreement",

    "shared_mass",

    "original_weighted_jaccard",

    "mass_weighted_jaccard",

    "original_mean_disagreement",

    "total_variation"

]

class_comparison = matched.groupby(
    "class_name"
)[comparison_metrics].agg([
    "count", "mean", "median", "std"
])

class_comparison.to_csv(
    OUT / "matched_classwise_formulations.csv"
)

print("\nMATCHED CLASS-WISE MEDIANS")

print(
    matched.groupby("class_name")[
        comparison_metrics
    ].median().round(6).to_string()
)

# ============================================================
# 4. CLASS RANKINGS
# ============================================================

class_medians = matched.groupby(
    "class_name"
)[comparison_metrics].median()

ranking = class_medians.rank(
    ascending=False,
    method="average"
)

ranking.to_csv(
    OUT / "class_rankings.csv"
)

print("\nCLASS RANKINGS")
print("Rank 1 = highest metric value")
print(ranking.to_string())

# ============================================================
# EXPERIMENT B: DIRECTIONAL THRESHOLD SENSITIVITY
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT B: DIRECTIONAL THRESHOLD SENSITIVITY")
print("=" * 75)

# ============================================================
# 5. TOP-K SELECTION
# ============================================================

def top_k(values, indices, k):

    order = np.argsort(
        -values[indices],
        kind="stable"
    )

    return indices[order[:k]]

# ============================================================
# 6. RUN ALL 25 CONFIGURATIONS
# ============================================================

records = []

eligible_indices = np.flatnonzero(
    mass_valid
)

for top_fraction in TOP_FRACTIONS:

    for threshold in THRESHOLDS:

        dominance_ratio = (
            (1.0 + threshold)
            / (1.0 - threshold)
        )

        for i in eligible_indices:

            indices = np.flatnonzero(
                valid[i]
            )

            n_valid = len(indices)

            k = int(
                np.ceil(top_fraction * n_valid)
            )

            top1 = top_k(
                p[i], indices, k
            )

            top2 = top_k(
                q[i], indices, k
            )

            salient = np.union1d(
                top1, top2
            )

            denominator = (
                p[i, salient] + q[i, salient]
            )

            salient = salient[
                denominator > 0
            ]

            if len(salient) == 0:
                continue

            ratio = (
                p[i, salient] - q[i, salient]
            ) / (
                p[i, salient] + q[i, salient]
            )

            dominant1 = ratio > threshold

            dominant2 = ratio < -threshold

            balanced = (
                ~dominant1 & ~dominant2
            )

            n1 = int(dominant1.sum())
            n2 = int(dominant2.sum())
            nb = int(balanced.sum())

            ns = len(salient)

            assert n1 + n2 + nb == ns

            records.append({

                "cohort_index":
                    mass.iloc[i]["cohort_index"],

                "class_name":
                    mass.iloc[i]["class_name"],

                "both_correct":
                    mass.iloc[i]["both_correct"],

                "top_fraction":
                    top_fraction,

                "threshold":
                    threshold,

                "dominance_ratio":
                    dominance_ratio,

                "valid_columns":
                    n_valid,

                "salient_columns":
                    ns,

                "1d_dominant_pct_valid":
                    100.0 * n1 / n_valid,

                "2d_dominant_pct_valid":
                    100.0 * n2 / n_valid,

                "balanced_pct_valid":
                    100.0 * nb / n_valid,

                "1d_dominant_pct_salient":
                    100.0 * n1 / ns,

                "2d_dominant_pct_salient":
                    100.0 * n2 / ns,

                "balanced_pct_salient":
                    100.0 * nb / ns,

                "salient_pct_valid":
                    100.0 * ns / n_valid,

                "dominance_difference":
                    100.0 * (n1 - n2) / n_valid

            })

        print(
            f"Completed: top={100*top_fraction:.0f}% "
            f"| threshold={threshold:.4f}",
            flush=True
        )

sensitivity = pd.DataFrame(records)

assert len(sensitivity) == 25 * 1040

assert np.allclose(

    sensitivity[
        "1d_dominant_pct_salient"
    ]

    + sensitivity[
        "2d_dominant_pct_salient"
    ]

    + sensitivity[
        "balanced_pct_salient"
    ],

    100.0

)

sensitivity.to_csv(
    OUT / "directional_sensitivity_recordings.csv",
    index=False
)

# ============================================================
# 7. VERIFY BASELINE REPRODUCTION
# ============================================================

baseline = sensitivity.loc[

    np.isclose(
        sensitivity["top_fraction"], 0.15
    )

    & np.isclose(
        sensitivity["threshold"], 1.0 / 3.0
    )

].copy()

baseline = baseline.merge(

    baseline_direction,

    on="cohort_index",

    suffixes=("_new", "_old"),

    validate="one_to_one"

)

assert len(baseline) == 1040

check_cols = [

    "1d_dominant_pct_valid",

    "2d_dominant_pct_valid",

    "balanced_pct_valid",

    "1d_dominant_pct_salient",

    "2d_dominant_pct_salient",

    "balanced_pct_salient"

]

for col in check_cols:

    assert np.allclose(

        baseline[col + "_new"],

        baseline[col + "_old"],

        atol=1e-8

    ), f"Baseline mismatch: {col}"

print("\nBASELINE REPRODUCTION PASSED")

# ============================================================
# 8. BOOTSTRAP AND HOLM UTILITIES
# ============================================================

def bootstrap_mean_ci(values):

    values = np.asarray(
        values, dtype=float
    )

    estimates = np.array([

        rng.choice(
            values,
            size=len(values),
            replace=True
        ).mean()

        for _ in range(N_BOOT)

    ])

    return np.percentile(
        estimates, [2.5, 97.5]
    )


def holm_adjust(values):

    values = np.asarray(
        values, dtype=float
    )

    order = np.argsort(values)

    adjusted = np.maximum.accumulate(

        values[order] *

        (len(values) - np.arange(len(values)))

    )

    adjusted = np.minimum(
        adjusted, 1.0
    )

    result = np.empty_like(adjusted)

    result[order] = adjusted

    return result

# ============================================================
# 9. OVERALL SENSITIVITY STATISTICS
# ============================================================

summary_records = []

for (fraction, threshold), group in sensitivity.groupby(
    ["top_fraction", "threshold"]
):

    differences = group[
        "dominance_difference"
    ].to_numpy()

    if np.all(differences == 0):

        statistic = 0.0
        p_value = 1.0

    else:

        test = wilcoxon(
            differences,
            alternative="two-sided"
        )

        statistic = test.statistic
        p_value = test.pvalue

    low, high = bootstrap_mean_ci(
        differences
    )

    summary_records.append({

        "top_fraction": fraction,

        "threshold": threshold,

        "dominance_ratio":
            group["dominance_ratio"].iloc[0],

        "n": len(group),

        "mean_1d_dominant_pct_valid":
            group["1d_dominant_pct_valid"].mean(),

        "mean_2d_dominant_pct_valid":
            group["2d_dominant_pct_valid"].mean(),

        "mean_balanced_pct_valid":
            group["balanced_pct_valid"].mean(),

        "mean_1d_dominant_pct_salient":
            group["1d_dominant_pct_salient"].mean(),

        "mean_2d_dominant_pct_salient":
            group["2d_dominant_pct_salient"].mean(),

        "mean_balanced_pct_salient":
            group["balanced_pct_salient"].mean(),

        "mean_dominance_difference":
            differences.mean(),

        "median_dominance_difference":
            np.median(differences),

        "mean_difference_ci_lower": low,

        "mean_difference_ci_upper": high,

        "pct_recordings_1d_dominates_more":
            100.0 * np.mean(differences > 0),

        "wilcoxon_statistic": statistic,

        "p_value": p_value

    })

summary = pd.DataFrame(
    summary_records
)

summary["p_holm"] = holm_adjust(
    summary["p_value"]
)

summary["significant"] = (
    summary["p_holm"] < 0.05
)

summary.to_csv(
    OUT / "directional_sensitivity_summary.csv",
    index=False
)

# ============================================================
# 10. CLASS-WISE SENSITIVITY
# ============================================================

class_sensitivity = sensitivity.groupby(

    [
        "class_name",
        "top_fraction",
        "threshold"
    ]

)[

    [
        "1d_dominant_pct_valid",

        "2d_dominant_pct_valid",

        "balanced_pct_valid",

        "dominance_difference"
    ]

].agg([
    "mean", "median", "std"
])

class_sensitivity.to_csv(
    OUT / "directional_sensitivity_classwise.csv"
)

# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("OVERALL DIRECTIONAL SENSITIVITY")
print("=" * 75)

display_cols = [

    "top_fraction",

    "threshold",

    "mean_1d_dominant_pct_valid",

    "mean_2d_dominant_pct_valid",

    "mean_balanced_pct_valid",

    "mean_dominance_difference",

    "mean_difference_ci_lower",

    "mean_difference_ci_upper",

    "p_holm",

    "significant"

]

print(
    summary[display_cols]
    .round(6)
    .to_string(index=False)
)

print("\nROBUSTNESS SUMMARY")

print(
    "Configurations evaluated:",
    len(summary)
)

print(
    "Configurations with higher mean 1D dominance:",

    (
        summary["mean_dominance_difference"] > 0
    ).sum()
)

print(
    "Configurations with higher mean 2D dominance:",

    (
        summary["mean_dominance_difference"] < 0
    ).sum()
)

print(
    "Configurations with positive mean difference "
    "and significant paired test:",

    (
        (summary["mean_dominance_difference"] > 0)

        & summary["significant"]

    ).sum()
)

print(
    "Minimum mean dominance difference:",

    summary["mean_dominance_difference"].min()
)

print(
    "Maximum mean dominance difference:",

    summary["mean_dominance_difference"].max()
)

print("\nAll verification checks passed.")

print("Results saved in:", OUT)

print("STEP 7 COMPLETED")
