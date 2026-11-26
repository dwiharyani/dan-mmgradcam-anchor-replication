from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, kruskal

# ============================================================
# STEP 4: NORMALIZED ATTRIBUTION ENTROPY
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
MASS = BASE / "mass_based_agreement"

OUT = BASE / "attribution_entropy"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD MASS-NORMALIZED ATTRIBUTIONS
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

# ============================================================
# 2. NORMALIZED SHANNON ENTROPY
# ============================================================

def normalized_entropy(values):

    values = np.asarray(values, dtype=np.float64)

    n = len(values)

    assert n > 1
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert np.isclose(values.sum(), 1.0)

    # Convention: 0 * log(0) = 0
    positive = values[values > 0]

    entropy = -np.sum(
        positive * np.log(positive)
    )

    entropy /= np.log(n)

    assert -1e-10 <= entropy <= 1.0 + 1e-10

    return float(np.clip(entropy, 0, 1))

# ============================================================
# 3. MATHEMATICAL VERIFICATION
# ============================================================

uniform = np.ones(100) / 100

concentrated = np.zeros(100)
concentrated[0] = 1.0

assert np.isclose(
    normalized_entropy(uniform), 1.0
)

assert np.isclose(
    normalized_entropy(concentrated), 0.0
)

# ============================================================
# 4. RECORDING-LEVEL ENTROPY
# ============================================================

records = []

for i in range(len(p)):

    if not mass_valid[i]:
        continue

    mask = valid[i]

    entropy1 = normalized_entropy(
        p[i, mask]
    )

    entropy2 = normalized_entropy(
        q[i, mask]
    )

    records.append({

        "cohort_index":
            metrics.iloc[i]["cohort_index"],

        "class_name":
            metrics.iloc[i]["class_name"],

        "both_correct":
            metrics.iloc[i]["both_correct"],

        "valid_columns":
            int(mask.sum()),

        "entropy_1d":
            entropy1,

        "entropy_2d":
            entropy2,

        "entropy_difference":
            entropy2 - entropy1,

        "shared_mass":
            metrics.iloc[i]["shared_mass"]

    })

df = pd.DataFrame(records)

assert len(df) == mass_valid.sum()

df.to_csv(
    OUT / "entropy_metrics.csv",
    index=False
)

# ============================================================
# 5. OVERALL DESCRIPTIVE STATISTICS
# ============================================================

cols = [
    "entropy_1d",
    "entropy_2d",
    "entropy_difference"
]

overall = df[cols].agg([
    "count", "mean", "median",
    "std", "min", "max"
])

overall.to_csv(
    OUT / "entropy_overall_summary.csv"
)

print("\n" + "=" * 70)
print("STEP 4: NORMALIZED ATTRIBUTION ENTROPY")
print("=" * 70)

print("\nValid recordings:", len(df))

print("\nOVERALL SUMMARY")
print(overall.round(6).to_string())

# ============================================================
# 6. CLASS-WISE ANALYSIS
# ============================================================

class_summary = df.groupby(
    "class_name"
)[cols].agg([
    "count", "mean", "median", "std"
])

class_summary.to_csv(
    OUT / "entropy_classwise_summary.csv"
)

print("\nCLASS-WISE MEDIANS")

print(
    df.groupby("class_name")[cols]
    .median().round(6).to_string()
)

# ============================================================
# 7. PAIRED STATISTICAL COMPARISON
# ============================================================

differences = df["entropy_difference"].to_numpy()

if np.all(differences == 0):

    statistic = 0.0
    p_value = 1.0

else:

    test = wilcoxon(
        differences,
        alternative="two-sided"
    )

    statistic = float(test.statistic)
    p_value = float(test.pvalue)

# Recording-level bootstrap confidence interval
rng = np.random.default_rng(42)

bootstrap = np.array([
    np.median(
        rng.choice(
            differences,
            size=len(differences),
            replace=True
        )
    )
    for _ in range(2000)
])

ci_low, ci_high = np.percentile(
    bootstrap, [2.5, 97.5]
)

stats = {

    "n": len(df),

    "median_entropy_1d":
        df["entropy_1d"].median(),

    "median_entropy_2d":
        df["entropy_2d"].median(),

    "median_paired_difference":
        np.median(differences),

    "ci_lower": ci_low,

    "ci_upper": ci_high,

    "wilcoxon_statistic": statistic,

    "wilcoxon_p_value": p_value,

    "pct_2d_higher_entropy":
        100 * np.mean(differences > 0)

}

pd.DataFrame([stats]).to_csv(
    OUT / "entropy_paired_statistics.csv",
    index=False
)

print("\nPAIRED STATISTICAL COMPARISON")

for key, value in stats.items():
    print(f"{key}: {value}")

# ============================================================
# 8. CLASS DIFFERENCES
# ============================================================

class_tests = []

for metric in cols:

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

    class_tests.append({

        "metric": metric,

        "H": test.statistic,

        "p_value": test.pvalue,

        "epsilon_squared": effect

    })

class_tests = pd.DataFrame(class_tests)

class_tests.to_csv(
    OUT / "entropy_class_statistics.csv",
    index=False
)

print("\nCLASS DIFFERENCE TESTS")
print(class_tests.to_string(index=False))

# ============================================================
# 9. PREDICTION CORRECTNESS
# ============================================================

correctness = df.groupby(
    "both_correct"
)[cols].agg([
    "count", "mean", "median", "std"
])

correctness.to_csv(
    OUT / "entropy_correctness_summary.csv"
)

print("\nPREDICTION CORRECTNESS MEDIANS")

print(
    df.groupby("both_correct")[cols]
    .median().round(6).to_string()
)

# ============================================================
# 10. AFIB COHORT 209
# ============================================================

print("\nAFIB COHORT 209")

print(
    df.loc[df["cohort_index"] == 209]
    .to_string(index=False)
)

print("\nAll mathematical checks passed.")
print("Results saved in:", OUT)
print("STEP 4 COMPLETED")
