from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path.home() / "Alignment-XAI"
RESULTS = ROOT / "module1_alignment/results"
OUT = RESULTS / "statistical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["AFib", "GSVT", "SB", "SR"]
METRICS = [
    "mean_agreement",
    "mean_disagreement",
    "spearman",
    "dice_top15",
    "iou_top15",
    "weighted_jaccard",
]

df = pd.read_csv(
    RESULTS / "alignment_0px/agreement_metrics.csv"
)

assert len(df) == 1054
assert df["cohort_index"].is_unique
assert set(df["class_name"]) == set(CLASSES)

df["valid_similarity"] = (
    df["valid_similarity"].astype(bool)
)

# Primary analysis: nondegenerate paired explanations.
primary = df[df["valid_similarity"]].copy()

# Secondary analysis: all recordings, for metrics defined
# under the documented zero-preserving convention.
all_records = df.copy()

print("Total ECGs:", len(df))
print("Primary nondegenerate pairs:", len(primary))
print("Degenerate pairs:", len(df) - len(primary))

assert len(primary) == 1033


def analyze(data, metrics, analysis_name):
    descriptive = []
    omnibus = []
    pairwise = []

    for metric in metrics:
        groups = {
            cls: data.loc[
                data["class_name"] == cls, metric
            ].dropna().to_numpy(dtype=float)
            for cls in CLASSES
        }

        for cls, values in groups.items():
            descriptive.append({
                "analysis": analysis_name,
                "metric": metric,
                "class_name": cls,
                "n": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values, ddof=1),
                "q1": np.percentile(values, 25),
                "q3": np.percentile(values, 75),
            })

        h, p = kruskal(*groups.values())

        n_total = sum(len(v) for v in groups.values())
        k = len(CLASSES)

        epsilon_squared = max(
            0.0, (h - k + 1) / (n_total - k)
        )

        omnibus.append({
            "analysis": analysis_name,
            "metric": metric,
            "kruskal_h": h,
            "p_value": p,
            "epsilon_squared": epsilon_squared,
        })

        comparisons = []

        for c1, c2 in combinations(CLASSES, 2):
            x, y = groups[c1], groups[c2]

            u, p_raw = mannwhitneyu(
                x, y, alternative="two-sided",
                method="asymptotic"
            )

            # Rank-biserial effect size:
            # positive means class 1 tends to be larger.
            rank_biserial = 2 * u / (len(x) * len(y)) - 1

            comparisons.append({
                "analysis": analysis_name,
                "metric": metric,
                "class_1": c1,
                "class_2": c2,
                "n_1": len(x),
                "n_2": len(y),
                "u_statistic": u,
                "p_raw": p_raw,
                "rank_biserial": rank_biserial,
                "median_difference": (
                    np.median(x) - np.median(y)
                ),
            })

        adjusted = multipletests(
            [r["p_raw"] for r in comparisons],
            alpha=0.05,
            method="holm"
        )

        for r, reject, p_adj in zip(
            comparisons, adjusted[0], adjusted[1]
        ):
            r["p_holm"] = p_adj
            r["significant_0_05"] = bool(reject)
            pairwise.append(r)

    descriptive = pd.DataFrame(descriptive)
    omnibus = pd.DataFrame(omnibus)
    pairwise = pd.DataFrame(pairwise)

    descriptive.to_csv(
        OUT / f"{analysis_name}_descriptive.csv",
        index=False
    )
    omnibus.to_csv(
        OUT / f"{analysis_name}_kruskal.csv",
        index=False
    )
    pairwise.to_csv(
        OUT / f"{analysis_name}_pairwise.csv",
        index=False
    )

    print(f"\n{'=' * 65}")
    print(analysis_name.upper())
    print(f"{'=' * 65}")

    print("\nKRUSKAL-WALLIS RESULTS")
    print(omnibus.round(6).to_string(index=False))

    print("\nSIGNIFICANT PAIRWISE COMPARISONS")
    sig = pairwise[pairwise["significant_0_05"]]

    if sig.empty:
        print("None")
    else:
        print(
            sig[
                ["metric", "class_1", "class_2",
                 "p_holm", "rank_biserial"]
            ].round(6).to_string(index=False)
        )


analyze(
    primary,
    METRICS,
    "primary_1033"
)

analyze(
    all_records,
    ["mean_agreement", "mean_disagreement"],
    "sensitivity_1054"
)

print("\nStatistical analysis completed.")
print("Results saved to:", OUT)
