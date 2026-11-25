from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# STEP 5: WITHIN-CLASS MISPAIRING NULL
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
MASS = BASE / "mass_based_agreement"

OUT = BASE / "within_class_mispairing"
OUT.mkdir(parents=True, exist_ok=True)

N_PERM = 1000
SEED = 42
EPS = 1e-8

# ============================================================
# 1. LOAD ATTRIBUTION ARRAYS
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

metrics = pd.read_csv(
    MASS / "mass_agreement_metrics.csv"
)

mass_valid = np.load(
    MASS / "mass_valid_recordings.npy"
).astype(bool)

assert g1.shape == (1054, 600)
assert g2.shape == g1.shape
assert valid.shape == g1.shape
assert len(metrics) == len(g1)

assert np.array_equal(
    mass_valid,
    metrics["mass_valid"].to_numpy()
)

classes = metrics["class_name"].to_numpy()
both_correct = metrics["both_correct"].to_numpy(dtype=bool)

METRIC_NAMES = [
    "shared_mass",
    "minmax_agreement",
    "dice_top15"
]

# ============================================================
# 2. METRIC CALCULATION
# ============================================================

def minmax(x):

    lo = x.min()
    hi = x.max()

    if hi - lo < EPS:
        return np.zeros_like(x), False

    return (x - lo) / (hi - lo), True


def calculate_metrics(a, b, mask):

    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return np.full(3, np.nan)

    # Mass-based overlap
    if a.sum() > EPS and b.sum() > EPS:

        p = a / a.sum()
        q = b / b.sum()

        overlap = np.minimum(p, q).sum()

    else:
        overlap = np.nan

    # Original min-max agreement
    a_norm, valid_a = minmax(a)
    b_norm, valid_b = minmax(b)

    agreement = np.sqrt(
        a_norm * b_norm
    ).mean()

    # Top-15% Dice
    if valid_a and valid_b:

        k = int(np.ceil(0.15 * len(a)))

        top_a = np.argsort(
            -a_norm, kind="stable"
        )[:k]

        top_b = np.argsort(
            -b_norm, kind="stable"
        )[:k]

        intersection = len(
            np.intersect1d(top_a, top_b)
        )

        dice = intersection / k

    else:
        dice = np.nan

    return np.array([
        overlap,
        agreement,
        dice
    ])


# ============================================================
# 3. GENERATE PERMUTATIONS WITHOUT SELF-PAIRING
# ============================================================

def derangement(indices, rng):

    while True:

        permuted = rng.permutation(indices)

        if np.all(permuted != indices):
            return permuted


# ============================================================
# 4. RUN NULL EXPERIMENT FOR EACH COHORT
# ============================================================

def run_experiment(cohort_name, eligible):

    rng = np.random.default_rng(SEED)

    folder = OUT / cohort_name
    folder.mkdir(parents=True, exist_ok=True)

    summary_records = []
    recording_records = []
    permutation_records = []

    print("\n" + "=" * 70)
    print("COHORT:", cohort_name)
    print("Eligible recordings:", eligible.sum())
    print("=" * 70)

    for cls in sorted(np.unique(classes)):

        indices = np.flatnonzero(
            eligible & (classes == cls)
        )

        n = len(indices)

        if n < 2:
            continue

        # Require identical valid masks within each class
        # so paired and mispaired scores use the same grid.
        if not np.all(
            valid[indices] == valid[indices[0]]
        ):
            raise ValueError(
                f"Different temporal masks in {cls}. "
                "Inspect geometry before continuing."
            )

        mask = valid[indices[0]]

        actual = np.stack([
            calculate_metrics(
                g1[i], g2[i], mask
            )
            for i in indices
        ])

        null = np.full(
            (N_PERM, n, 3),
            np.nan
        )

        # ----------------------------------------------------
        # Within-class derangements
        # ----------------------------------------------------

        for r in range(N_PERM):

            shuffled = derangement(
                indices, rng
            )

            assert np.all(
                shuffled != indices
            )

            assert np.all(
                classes[shuffled] == cls
            )

            for j, (i, partner) in enumerate(
                zip(indices, shuffled)
            ):

                null[r, j] = calculate_metrics(
                    g1[i],
                    g2[partner],
                    mask
                )

            if (r + 1) % 100 == 0:

                print(
                    f"{cls}: "
                    f"{r + 1}/{N_PERM} permutations",
                    flush=True
                )

        np.savez_compressed(
            folder / f"{cls}_null.npz",
            indices=indices,
            actual=actual,
            null=null,
            metric_names=np.array(METRIC_NAMES)
        )

        # ----------------------------------------------------
        # Summarize each metric
        # ----------------------------------------------------

        for m, metric in enumerate(METRIC_NAMES):

            observed = actual[:, m]
            null_values = null[:, :, m]

            # Keep the same finite observed cohort.
            keep = np.isfinite(observed)

            observed = observed[keep]
            null_values = null_values[:, keep]

            if len(observed) == 0:
                continue

            if not np.isfinite(null_values).all():
                raise ValueError(
                    f"Nonfinite null scores: {cls}, {metric}"
                )

            pooled_null = null_values.ravel()

            null_median = np.median(pooled_null)

            null95 = np.percentile(
                pooled_null, 95
            )

            actual_median = np.median(observed)

            # One cohort-level statistic per permutation.
            perm_medians = np.median(
                null_values, axis=1
            )

            p_value = (
                1 + np.sum(
                    perm_medians >= actual_median
                )
            ) / (N_PERM + 1)

            above95 = (
                observed > null95
            )

            summary_records.append({

                "cohort": cohort_name,

                "class_name": cls,

                "metric": metric,

                "n_recordings": len(observed),

                "n_permutations": N_PERM,

                "n_null_pairs": len(pooled_null),

                "actual_mean": observed.mean(),

                "actual_median": actual_median,

                "null_mean": pooled_null.mean(),

                "null_median": null_median,

                "median_difference":
                    actual_median - null_median,

                "null_95th_percentile": null95,

                "pct_above_null95":
                    100 * above95.mean(),

                "permutation_p_value": p_value

            })

            kept_indices = indices[keep]

            for j, i in enumerate(kept_indices):

                individual_null = null_values[:, j]

                recording_records.append({

                    "cohort_index":
                        metrics.iloc[i]["cohort_index"],

                    "class_name": cls,

                    "metric": metric,

                    "actual": observed[j],

                    "mean_individual_null":
                        individual_null.mean(),

                    "actual_minus_mean_null":
                        observed[j] -
                        individual_null.mean(),

                    "class_null95": null95,

                    "above_class_null95":
                        bool(above95[j])

                })

            for r in range(N_PERM):

                permutation_records.append({

                    "class_name": cls,

                    "metric": metric,

                    "permutation": r,

                    "null_median":
                        perm_medians[r]

                })

    summary = pd.DataFrame(summary_records)
    recording = pd.DataFrame(recording_records)
    permutations = pd.DataFrame(permutation_records)

    summary.to_csv(
        folder / "null_classwise_summary.csv",
        index=False
    )

    recording.to_csv(
        folder / "null_recording_metrics.csv",
        index=False
    )

    permutations.to_csv(
        folder / "null_permutation_statistics.csv",
        index=False
    )

    print("\nCLASS-WISE NULL RESULTS")

    print(
        summary.round(6).to_string(index=False)
    )

    print("\nOVERALL RECORDING-LEVEL SUMMARY")

    overall = recording.groupby("metric").agg(

        n=("actual", "count"),

        actual_median=("actual", "median"),

        mean_individual_null=(
            "mean_individual_null", "mean"
        ),

        mean_paired_difference=(
            "actual_minus_mean_null", "mean"
        ),

        fraction_above_class_null95=(
            "above_class_null95", "mean"
        )

    )

    overall.to_csv(
        folder / "null_overall_summary.csv"
    )

    print(overall.round(6).to_string())


# ============================================================
# 5. PRIMARY AND CORRECTNESS SUBGROUP ANALYSES
# ============================================================

run_experiment(
    "all_mass_valid",
    mass_valid
)

run_experiment(
    "both_correct_mass_valid",
    mass_valid & both_correct
)

print("\nSTEP 5 COMPLETED")
print("Results saved in:", OUT)
