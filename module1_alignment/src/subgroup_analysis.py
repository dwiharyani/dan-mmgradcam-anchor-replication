from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment"
OUT = BASE / "results/subgroup_analysis"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "mean_agreement",
    "mean_disagreement",
    "spearman",
    "dice_top15",
    "iou_top15",
    "weighted_jaccard",
]

metrics = pd.read_csv(
    BASE / "results/alignment_0px/agreement_metrics.csv"
)

pred1 = pd.read_csv(
    BASE / "results/1d_test_predictions.csv"
)

pred2 = pd.read_csv(
    BASE / "results/2d_test_predictions.csv"
)

print("1D prediction columns:", pred1.columns.tolist())
print("2D prediction columns:", pred2.columns.tolist())

# Accept common prediction column names.
def standardize(df, suffix):
    df = df.copy()

    assert "cohort_index" in df.columns, (
        f"Missing cohort_index in {suffix} predictions"
    )

    candidates = [
        "predicted_class",
        "pred_class",
        "prediction",
        "pred_1d",
        "pred_2d",
    ]

    column = next(
        (c for c in candidates if c in df.columns),
        None
    )

    if column is None:
        raise ValueError(
            f"Cannot identify prediction column: {df.columns.tolist()}"
        )

    df = df.rename(columns={
        column: f"pred_{suffix}"
    })

    assert df["cohort_index"].is_unique

    return df[["cohort_index", f"pred_{suffix}"]]


pred1 = standardize(pred1, "1d")
pred2 = standardize(pred2, "2d")

assert metrics["cohort_index"].is_unique

df = metrics.merge(
    pred1, on="cohort_index", how="left", validate="one_to_one"
).merge(
    pred2, on="cohort_index", how="left", validate="one_to_one"
)

if df[["pred_1d", "pred_2d"]].isna().any().any():
    raise ValueError("Missing predictions for some ECGs")

df["correct_1d"] = df["pred_1d"] == df["class_name"]
df["correct_2d"] = df["pred_2d"] == df["class_name"]

df["both_correct"] = (
    df["correct_1d"] & df["correct_2d"]
)

df["nondegenerate"] = ~(
    df["degenerate_1d"] | df["degenerate_2d"]
)

df["subgroup"] = np.where(
    df["both_correct"],
    "Both correct",
    "At least one incorrect"
)

df.to_csv(
    OUT / "subgroup_manifest.csv",
    index=False
)

print("\nSUBGROUP COUNTS")
print("Total:", len(df))
print("Nondegenerate:", df["nondegenerate"].sum())
print("Both correct:", df["both_correct"].sum())
print("Both correct and nondegenerate:",
      (df["both_correct"] & df["nondegenerate"]).sum())

print("\nCORRECTNESS COMBINATIONS")
print(pd.crosstab(
    df["correct_1d"],
    df["correct_2d"]
))

# Main comparison: nondegenerate pairs only.
valid = df[df["nondegenerate"]].copy()

records = []
tests = []

for group_name, group in [
    ("All 1054", df),
    ("Nondegenerate", valid),
    ("Both correct", valid[valid["both_correct"]]),
    ("At least one incorrect", valid[~valid["both_correct"]]),
]:
    for metric in METRICS:
        values = group[metric].dropna()

        records.append({
            "group": group_name,
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
        })

for metric in METRICS:
    x = valid.loc[valid["both_correct"], metric].dropna()
    y = valid.loc[~valid["both_correct"], metric].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    u, p = mannwhitneyu(
        x, y, alternative="two-sided"
    )

    tests.append({
        "metric": metric,
        "n_both_correct": len(x),
        "n_incorrect": len(y),
        "mean_both_correct": x.mean(),
        "mean_incorrect": y.mean(),
        "u_statistic": u,
        "p_value": p,
        "rank_biserial": 2*u/(len(x)*len(y))-1,
    })

summary = pd.DataFrame(records)
tests = pd.DataFrame(tests)

summary.to_csv(
    OUT / "subgroup_summary.csv",
    index=False
)

tests.to_csv(
    OUT / "correct_vs_incorrect_tests.csv",
    index=False
)

print("\nSUBGROUP SUMMARY")
print(
    summary.pivot(
        index="metric",
        columns="group",
        values="mean"
    ).round(4).to_string()
)

print("\nBOTH CORRECT VS AT LEAST ONE INCORRECT")
print(tests.round(6).to_string(index=False))

print("\nSaved to:", OUT)
