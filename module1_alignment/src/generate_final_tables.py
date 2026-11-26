from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
OUT = BASE / "final_report"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "mean_agreement",
    "mean_disagreement",
    "spearman",
    "dice_top15",
    "iou_top15",
    "weighted_jaccard",
]

CLASSES = ["AFib", "GSVT", "SB", "SR"]

metrics = pd.read_csv(
    BASE / "alignment_0px/agreement_metrics.csv"
)

subgroups = pd.read_csv(
    BASE / "subgroup_analysis/subgroup_manifest.csv"
)

kruskal = pd.read_csv(
    BASE / "statistical_analysis/primary_1033_kruskal.csv"
)

pairwise = pd.read_csv(
    BASE / "statistical_analysis/primary_1033_pairwise.csv"
)

correctness = pd.read_csv(
    BASE / "subgroup_analysis/correct_vs_incorrect_tests.csv"
)

shift = pd.read_csv(
    BASE / "gradcam_shift_sensitivity/shift_metrics.csv"
)

assert len(metrics) == 1054
assert metrics["cohort_index"].is_unique
assert len(subgroups) == 1054
assert subgroups["cohort_index"].is_unique
assert len(shift) == 1054 * 11

# Table 1: Dataset and explanation validity
valid = ~(
    metrics["degenerate_1d"].astype(bool) |
    metrics["degenerate_2d"].astype(bool)
)

table1 = pd.DataFrame([
    ["Total test ECGs", len(metrics)],
    ["Nondegenerate pairs", int(valid.sum())],
    ["Degenerate pairs", int((~valid).sum())],
    ["Both models correct", int(subgroups["both_correct"].sum())],
    ["Both correct and nondegenerate",
     int((subgroups["both_correct"] &
          subgroups["nondegenerate"]).sum())],
], columns=["measure", "count"])

# Table 2: Class-wise results for primary 1033
table2 = (
    metrics.loc[valid]
    .groupby("class_name")[METRICS]
    .agg(["mean", "median", "std"])
)

table2 = table2.reindex(CLASSES)
table2.columns = [
    f"{metric}_{stat}"
    for metric, stat in table2.columns
]
table2 = table2.reset_index()

# Table 3: Omnibus statistical tests
table3 = kruskal.copy()

# Table 4: Pairwise comparisons
table4 = pairwise.copy()

# Table 5: Classification subgroup comparisons
table5 = correctness.copy()

# Table 6: Temporal shift sensitivity
table6 = (
    shift.groupby("shift_pixels")[METRICS]
    .mean()
    .reset_index()
    .sort_values("shift_pixels")
)

tables = {
    "table1_dataset_summary": table1,
    "table2_classwise_metrics": table2,
    "table3_kruskal": table3,
    "table4_pairwise": table4,
    "table5_correctness_subgroups": table5,
    "table6_shift_sensitivity": table6,
}

for name, df in tables.items():
    df.to_csv(OUT / f"{name}.csv", index=False)

with pd.ExcelWriter(OUT / "module1_final_tables.xlsx") as writer:
    for name, df in tables.items():
        sheet = name.replace("table", "T")[:31]
        df.to_excel(writer, sheet_name=sheet, index=False)

print("FINAL TABLES GENERATED")
print("Output:", OUT)
print("\nFiles:")
for name in tables:
    print("-", name + ".csv")
print("- module1_final_tables.xlsx")
