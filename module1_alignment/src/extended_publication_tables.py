from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 8B: CONSOLIDATED PUBLICATION TABLES
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"

ALIGN = BASE / "alignment_0px"
STATS = BASE / "extended_statistical_analysis"
SENS = BASE / "agreement_formulation_sensitivity"
VALIDATION = BASE / "temporal_validation_100"

OUT = BASE / "extended_publication_tables"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["AFib", "GSVT", "SB", "SR"]

# ============================================================
# 1. LOAD EXPERIMENTAL RESULTS
# ============================================================

original = pd.read_csv(
    ALIGN / "agreement_metrics.csv"
)

extended = pd.read_csv(
    STATS / "extended_recording_metrics.csv"
)

bootstrap = pd.read_csv(
    STATS / "bootstrap_confidence_intervals.csv"
)

null = pd.read_csv(
    STATS / "overall_stratified_null.csv"
)

class_null = pd.read_csv(
    STATS / "null_holm_corrected.csv"
)

correctness = pd.read_csv(
    STATS / "correctness_statistical_comparison.csv"
)

correlations = pd.read_csv(
    SENS / "formulation_correlations.csv"
)

sensitivity = pd.read_csv(
    SENS / "directional_sensitivity_summary.csv"
)

validation = pd.read_csv(
    VALIDATION / "signal_image_correspondence.csv"
)

assert len(original) == 1054
assert len(extended) == 1040
assert len(validation) == 100
assert len(sensitivity) == 25

# ============================================================
# 2. TABLE EXPORT UTILITY
# ============================================================

def save_table(df, name, caption):

    csv_path = OUT / f"{name}.csv"
    tex_path = OUT / f"{name}.tex"

    df.to_csv(csv_path, index=False)

    latex = df.to_latex(
        index=False,
        escape=True,
        float_format="%.4f",
        caption=caption,
        label=f"tab:{name}",
        position="t"
    )

    tex_path.write_text(
        latex,
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)
    print(df.to_string(index=False))

    print("\nSaved:", csv_path.name)
    print("Saved:", tex_path.name)


def format_p(p):

    if p < 0.001:
        return f"{p:.2e}"

    return f"{p:.4f}"


def format_ci(low, high):

    return f"[{low:.4f}, {high:.4f}]"

# ============================================================
# TABLE 1: OVERALL ATTRIBUTION METRICS
# ============================================================

records = []

original_metrics = [
    ("mean_agreement", "Min-max agreement"),
    ("mean_disagreement", "Min-max disagreement"),
    ("spearman", "Spearman"),
    ("dice_top15", "Dice (top 15%)"),
    ("iou_top15", "IoU (top 15%)"),
    ("weighted_jaccard", "Weighted Jaccard")
]

for metric, label in original_metrics:

    if metric in [
        "mean_agreement",
        "mean_disagreement"
    ]:

        values = original[metric].dropna()

    else:

        values = original.loc[
            original["valid_similarity"],
            metric
        ].dropna()

    records.append({
        "Metric": label,
        "N": len(values),
        "Mean": values.mean(),
        "Median": values.median(),
        "SD": values.std()
    })

extended_metrics = [
    ("shared_mass", "Shared mass"),
    ("mass_weighted_jaccard", "Mass Jaccard"),
    ("total_variation", "Total variation"),
    ("entropy_1d", "1D entropy"),
    ("entropy_2d", "2D entropy")
]

for metric, label in extended_metrics:

    values = extended[metric].dropna()

    records.append({
        "Metric": label,
        "N": len(values),
        "Mean": values.mean(),
        "Median": values.median(),
        "SD": values.std()
    })

table1 = pd.DataFrame(records)

save_table(
    table1,
    "01_overall_agreement",
    "Overall cross-representation attribution metrics."
)

# ============================================================
# TABLE 2: CLASS-WISE EXTENDED RESULTS
# ============================================================

records = []

groups = [
    (cls, extended[
        extended["class_name"] == cls
    ])
    for cls in CLASSES
]

groups.append(("All", extended))

for cls, group in groups:

    records.append({

        "Class": cls,

        "N": len(group),

        "Shared mass":
            group["shared_mass"].median(),

        "Mass Jaccard":
            group["mass_weighted_jaccard"].median(),

        "1D dominant (%)":
            group["1d_dominant_pct_valid"].mean(),

        "2D dominant (%)":
            group["2d_dominant_pct_valid"].mean(),

        "Balanced (%)":
            group["balanced_pct_valid"].mean(),

        "1D entropy":
            group["entropy_1d"].median(),

        "2D entropy":
            group["entropy_2d"].median()

    })

table2 = pd.DataFrame(records)

save_table(
    table2,
    "02_classwise_extended",
    "Class-wise mass agreement, directional dominance "
    "and normalized attribution entropy. Directional "
    "percentages refer to valid temporal positions."
)

# ============================================================
# TABLE 3: OVERALL CHANCE CALIBRATION
# ============================================================

metric_labels = {
    "shared_mass": "Shared mass",
    "minmax_agreement": "Min-max agreement",
    "dice_top15": "Dice (top 15%)"
}

records = []

for _, row in null.iterrows():

    cohort = (
        "All"
        if row["cohort"] == "all_mass_valid"
        else "Both correct"
    )

    records.append({

        "Cohort": cohort,

        "Metric":
            metric_labels[row["metric"]],

        "N":
            int(row["n"]),

        "Actual":
            row["actual_mean"],

        "Null":
            row["null_mean"],

        "Difference":
            row["mean_difference"],

        "95% CI":
            format_ci(
                row["conditional_ci_lower"],
                row["conditional_ci_upper"]
            ),

        "Adjusted p":
            format_p(row["p_holm"])

    })

table3 = pd.DataFrame(records)

save_table(
    table3,
    "03_overall_chance_calibration",
    "Observed and within-class mispaired agreement. "
    "Differences are means; confidence intervals are "
    "conditional on the generated null reference."
)

# ============================================================
# TABLE 4: CLASS-WISE CHANCE CALIBRATION
# ============================================================

primary = class_null[
    class_null["cohort"] == "all_mass_valid"
]

records = []

for cls in CLASSES:

    subset = primary[
        primary["class_name"] == cls
    ]

    for _, row in subset.iterrows():

        records.append({

            "Class": cls,

            "Metric":
                metric_labels[row["metric"]],

            "N":
                int(row["n_recordings"]),

            "Actual median":
                row["actual_median"],

            "Null median":
                row["null_median"],

            "Difference":
                row["median_difference"],

            "Above null95 (%)":
                row["pct_above_null95"],

            "Adjusted p":
                format_p(row["p_holm"])

        })

table4 = pd.DataFrame(records)

save_table(
    table4,
    "04_classwise_chance_calibration",
    "Class-wise observed and mispaired agreement "
    "with Holm-adjusted empirical p-values."
)

# ============================================================
# TABLE 5: PREDICTION CORRECTNESS
# ============================================================

correctness_labels = {

    "shared_mass": "Shared mass",

    "mass_weighted_jaccard": "Mass Jaccard",

    "1d_dominant_pct_valid": "1D dominant (%)",

    "2d_dominant_pct_valid": "2D dominant (%)",

    "balanced_pct_valid": "Balanced (%)",

    "entropy_1d": "1D entropy",

    "entropy_2d": "2D entropy",

    "entropy_difference": "Entropy difference"

}

records = []

for _, row in correctness.iterrows():

    records.append({

        "Metric":
            correctness_labels[row["metric"]],

        "Both correct":
            row["median_correct"],

        "Incorrect":
            row["median_incorrect"],

        "Difference":
            row["median_difference"],

        "Rank-biserial":
            row["rank_biserial"],

        "Adjusted p":
            format_p(row["p_holm"])

    })

table5 = pd.DataFrame(records)

save_table(
    table5,
    "05_prediction_correctness",
    "Prediction-correctness comparisons between "
    "969 jointly correct and 71 other mass-valid "
    "recordings. Values are medians."
)

# ============================================================
# TABLE 6: AGREEMENT FORMULATION COMPARISON
# ============================================================

overall_corr = correlations[
    correlations["group"] == "All"
]

comparison_labels = {

    "original_mean_agreement":
        "Agreement vs shared mass",

    "original_weighted_jaccard":
        "Original vs mass Jaccard",

    "original_mean_disagreement":
        "Disagreement vs total variation"

}

records = []

for _, row in overall_corr.iterrows():

    records.append({

        "Comparison":
            comparison_labels[row["original_metric"]],

        "N":
            int(row["n"]),

        "Pearson":
            row["pearson_r"],

        "Spearman":
            row["spearman_rho"]

    })

table6 = pd.DataFrame(records)

save_table(
    table6,
    "06_formulation_comparison",
    "Recording-level correlations between original "
    "and mass-based attribution metrics."
)

# ============================================================
# TABLE 7: DIRECTIONAL THRESHOLD SENSITIVITY
# ============================================================

# Compact matrix containing all 25 configurations.

matrix = sensitivity.pivot(
    index="top_fraction",
    columns="dominance_ratio",
    values="mean_dominance_difference"
).sort_index().sort_index(axis=1)

matrix.index = [
    f"{x * 100:.0f}%"
    for x in matrix.index
]

matrix.columns = [
    f"{x:g}:1"
    for x in matrix.columns
]

table7 = matrix.reset_index()

table7 = table7.rename(
    columns={"index": "Top fraction"}
)

save_table(
    table7,
    "07_directional_sensitivity",
    "Mean 1D-minus-2D dominance difference in "
    "percentage points across saliency fractions "
    "and dominance ratios."
)

# ============================================================
# TABLE 8: TEMPORAL ALIGNMENT VALIDATION
# ============================================================

validation_metrics = [

    (
        "zero_shift_correlation",
        "Zero-shift correlation"
    ),

    (
        "best_correlation",
        "Best-shift correlation"
    ),

    (
        "correlation_improvement",
        "Correlation improvement"
    )

]

records = []

for metric, label in validation_metrics:

    values = validation[metric]

    records.append({

        "Metric": label,

        "N": len(values),

        "Mean": values.mean(),

        "Median": values.median(),

        "SD": values.std()

    })

table8 = pd.DataFrame(records)

save_table(
    table8,
    "08_temporal_validation",
    "Signal-to-image temporal correspondence on "
    "100 ECG recordings."
)

# ============================================================
# 9. SAVE KEY RESULTS IN A SINGLE TEXT FILE
# ============================================================

summary = []

summary.append("MODULE 1 - FINAL EXPERIMENTAL SUMMARY")
summary.append("=" * 60)

summary.append(
    f"Original cohort: {len(original)}"
)

summary.append(
    f"Original nondegenerate pairs: "
    f"{original['valid_similarity'].sum()}"
)

summary.append(
    f"Mass-valid pairs: {len(extended)}"
)

summary.append(
    f"Mean shared mass: "
    f"{extended['shared_mass'].mean():.6f}"
)

summary.append(
    f"Median shared mass: "
    f"{extended['shared_mass'].median():.6f}"
)

summary.append(
    f"Median 1D entropy: "
    f"{extended['entropy_1d'].median():.6f}"
)

summary.append(
    f"Median 2D entropy: "
    f"{extended['entropy_2d'].median():.6f}"
)

summary.append(
    f"Configurations favoring 1D dominance: "
    f"{(sensitivity['mean_dominance_difference'] > 0).sum()}"
)

summary.append(
    f"Configurations favoring 2D dominance: "
    f"{(sensitivity['mean_dominance_difference'] < 0).sum()}"
)

summary.append(
    f"Best alignment shifts: "
    f"{validation['best_shift_pixels'].value_counts().to_dict()}"
)

summary_text = "\n".join(summary)

(OUT / "module1_final_summary.txt").write_text(
    summary_text,
    encoding="utf-8"
)

print("\n" + summary_text)

print("\n" + "=" * 75)
print("STEP 8B COMPLETED")
print("Publication tables saved in:", OUT)
print("=" * 75)
