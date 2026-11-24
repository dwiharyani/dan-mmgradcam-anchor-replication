"""Paired deletion (faithfulness-style) test and block-size sensitivity audit on saved Module 2 maps.

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
from module2_common import *   # defines paths, test manifest, models, perturbation/render/Grad-CAM functions

# ---- notebook cell 31 ----

# ============================================================
# CORRECTED PAIRED DELETION TEST — DAN-MMGradCAM
# Uses saved Module 2 maps; does NOT rerun Module 2.
# Writes all new results to a separate directory.
# ============================================================

import os, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

# -------------------- 1. Configuration ----------------------

OUT_DIR = "results/module2/faithfulness_corrected_v1"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 20261115
N_PER_CLASS = 25
BLOCK = 50                 # 50 samples = 0.1 seconds at 500 Hz
RANDOM_REPEATS = 5
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BOOTSTRAPS = 2000

X0, X1, ORIGINAL_WIDTH, GRID_WIDTH = 58, 1021, 1080, 600
rng = np.random.default_rng(SEED)

# -------------------- 2. Validate saved outputs --------------

required = [
    "cohort_index", "out1_base", "out2_base",
    "R", "H1", "H2", "base1", "base2"
]
missing = [k for k in required if k not in z.files]
assert not missing, f"Missing saved arrays: {missing}"

assert len(df) == len(z["cohort_index"])
assert np.array_equal(
    df["cohort_index"].to_numpy(),
    z["cohort_index"]
), "DataFrame and saved map order do not match."

assert N == 5000
assert MEAN.shape == (N,), f"Unexpected MEAN shape: {MEAN.shape}"
assert SCALE.shape == (N,), f"Unexpected SCALE shape: {SCALE.shape}"

# Only records correctly classified by BOTH branches.
eligible = df[
    (df.pred_1d == df.class_name) &
    (df.pred_2d == df.class_name)
].copy()

# Balanced sample: 25 records from each class.
selected_parts = []
for cls in CLASSES:
    group = eligible[eligible.class_name == cls]
    assert len(group) >= N_PER_CLASS, (
        f"Only {len(group)} eligible records for {cls}; "
        f"cannot select {N_PER_CLASS}."
    )
    selected_parts.append(
        group.sample(N_PER_CLASS, random_state=SEED)
    )

sample = (
    pd.concat(selected_parts)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

print("Selected records:", len(sample))
print(sample.class_name.value_counts().to_dict())

# Saved-array index lookup.
z_index = {
    int(c): i for i, c in enumerate(z["cohort_index"])
}

# -------------------- 3. Coordinate mapping ------------------

# Map the 600-column attribution grid to the 5000 ECG samples.
# This uses the verified original image geometry.
def map_to_signal(attribution):
    a = np.asarray(attribution, dtype=np.float64)
    valid = np.isfinite(a)

    assert valid.any(), "Attribution map contains no valid values."

    sample_pixels = (
        X0 + np.linspace(0, 1, N) * (X1 - X0)
    )
    sample_columns = (
        (sample_pixels + 0.5) * GRID_WIDTH / ORIGINAL_WIDTH - 0.5
    )

    return np.interp(
        sample_columns,
        np.flatnonzero(valid),
        a[valid]
    )

# -------------------- 4. Contiguous block selection ----------

def choose_top_blocks(scores, n_blocks, block_size):
    """Greedily select high-scoring non-overlapping contiguous blocks."""
    scores = np.asarray(scores)
    candidates = []

    stride = 5
    for start in range(0, len(scores) - block_size + 1, stride):
        score = float(scores[start:start + block_size].mean())
        candidates.append((score, start))

    candidates.sort(reverse=True)

    occupied = np.zeros(len(scores), dtype=bool)
    selected = []

    for _, start in candidates:
        end = start + block_size
        if not occupied[start:end].any():
            selected.append(start)
            occupied[start:end] = True
            if len(selected) == n_blocks:
                break

    assert len(selected) == n_blocks, "Could not select enough non-overlapping blocks."
    return selected


def choose_random_blocks(n_blocks, block_size, seed):
    """Random non-overlapping blocks, matched in number and length."""
    local_rng = np.random.default_rng(seed)
    starts = np.arange(0, N - block_size + 1)
    local_rng.shuffle(starts)

    occupied = np.zeros(N, dtype=bool)
    selected = []

    for start in starts:
        end = start + block_size
        if not occupied[start:end].any():
            selected.append(int(start))
            occupied[start:end] = True
            if len(selected) == n_blocks:
                break

    assert len(selected) == n_blocks
    return selected


def occlude_blocks(x, starts, block_size):
    """
    Replace each selected block by linear interpolation between
    its neighbouring ECG samples, rather than setting samples to zero.
    """
    out = np.asarray(x, dtype=np.float64).copy()

    for start in starts:
        end = start + block_size

        left = out[start - 1] if start > 0 else out[end]
        right = out[end] if end < len(out) else left

        out[start:end] = np.linspace(
            left, right, block_size + 2
        )[1:-1]

    return out


# -------------------- 5. Model evaluation --------------------

def predict_1d(x):
    inp = ((x - MEAN) / SCALE).astype(np.float32)[None, :, None]
    return run_model("1d", m1, inp)[0]


def predict_2d(x, renderer):
    rgb = renderer.pert(x)
    image = (to_2d(rgb[None]) / 255.0).astype(np.float32)
    return run_model("2d", m2, image)[0]


# -------------------- 6. Methods and branches ----------------

METHODS = {
    "1d": ["R", "H1", "base1"],
    "2d": ["R", "H2", "base2"]
}

records = []
renderer = Renderer()

print("\nStarting corrected deletion experiment...")

try:
    for record_no, row in sample.iterrows():

        cohort = int(row.cohort_index)
        zi = z_index[cohort]
        x = load_ecg(row.ecg_csv_filename)

        assert x.shape == (N,), (
            f"Unexpected ECG length for cohort {cohort}: {x.shape}"
        )

        target = CLASSES.index(row.class_name)

        # Original model inputs and baseline predictions.
        base1 = predict_1d(x)

        rgb0 = renderer.base(x)
        image0 = (to_2d(rgb0[None]) / 255.0).astype(np.float32)
        base2 = run_model("2d", m2, image0)[0]

        saved1 = z["out1_base"][zi]
        saved2 = z["out2_base"][zi]

        assert np.allclose(base1, saved1, atol=1e-5), (
            f"1D baseline mismatch: cohort {cohort}"
        )
        assert np.allclose(base2, saved2, atol=1e-5), (
            f"2D baseline mismatch: cohort {cohort}"
        )

        # Map arrays from the saved Module 2 results.
        maps = {
            name: map_to_signal(z[name][zi])
            for name in ["R", "H1", "H2", "base1", "base2"]
        }

        # Confirm all selected records have finite mapped scores.
        assert all(np.isfinite(v).all() for v in maps.values())

        # Evaluate both branches.
        for branch, methods in METHODS.items():

            original_prob = (
                base1[target] if branch == "1d" else base2[target]
            )

            for method in methods:
                scores = maps[method]

                for fraction in FRACTIONS:
                    n_samples = int(round(N * fraction))
                    n_blocks = max(1, int(round(n_samples / BLOCK)))

                    # Keep the actual deleted fraction close to target.
                    actual_fraction = n_blocks * BLOCK / N

                    top_starts = choose_top_blocks(
                        scores, n_blocks, BLOCK
                    )
                    x_top = occlude_blocks(x, top_starts, BLOCK)

                    if branch == "1d":
                        top_prob = predict_1d(x_top)[target]
                    else:
                        top_prob = predict_2d(x_top, renderer)[target]

                    records.append({
                        "cohort_index": cohort,
                        "class_name": row.class_name,
                        "branch": branch,
                        "method": method,
                        "control": "attribution",
                        "requested_fraction": fraction,
                        "actual_fraction": actual_fraction,
                        "deleted_samples": n_blocks * BLOCK,
                        "baseline_probability": original_prob,
                        "perturbed_probability": float(top_prob),
                        "probability_drop": float(original_prob - top_prob)
                    })

                    # Matched random contiguous-block controls.
                    for rep in range(RANDOM_REPEATS):
                        random_starts = choose_random_blocks(
                            n_blocks, BLOCK,
                            seed=SEED + cohort * 1000 +
                                 int(fraction * 10000) + rep
                        )

                        x_random = occlude_blocks(
                            x, random_starts, BLOCK
                        )

                        if branch == "1d":
                            random_prob = predict_1d(x_random)[target]
                        else:
                            random_prob = predict_2d(
                                x_random, renderer
                            )[target]

                        records.append({
                            "cohort_index": cohort,
                            "class_name": row.class_name,
                            "branch": branch,
                            "method": method,
                            "control": "random",
                            "requested_fraction": fraction,
                            "actual_fraction": actual_fraction,
                            "deleted_samples": n_blocks * BLOCK,
                            "baseline_probability": original_prob,
                            "perturbed_probability": float(random_prob),
                            "probability_drop": float(
                                original_prob - random_prob
                            )
                        })

        if (record_no + 1) % 10 == 0:
            print(
                f"Processed {record_no + 1}/{len(sample)} records",
                flush=True
            )
            gc.collect()

finally:
    plt.close(renderer.fig)
    gc.collect()

# -------------------- 7. Save raw results --------------------

results_df = pd.DataFrame(records)

raw_path = os.path.join(OUT_DIR, "deletion_raw_results.csv")
results_df.to_csv(raw_path, index=False)

print("\nRaw results saved:", raw_path)

# -------------------- 8. Summary and paired bootstrap --------

def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan, np.nan

    result = bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=BOOTSTRAPS,
        method="percentile",
        random_state=np.random.default_rng(seed)
    )

    return (
        float(values.mean()),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high)
    )


summary_rows = []

for (branch, method, fraction), group in results_df.groupby(
    ["branch", "method", "requested_fraction"]
):

    attr = (
        group[group.control == "attribution"]
        .set_index("cohort_index")["probability_drop"]
    )

    random = (
        group[group.control == "random"]
        .groupby("cohort_index")["probability_drop"]
        .mean()
    )

    paired = pd.concat(
        [attr.rename("attribution"), random.rename("random")],
        axis=1
    ).dropna()

    mean_a, lo_a, hi_a = bootstrap_mean_ci(
        paired.attribution, SEED
    )
    mean_r, lo_r, hi_r = bootstrap_mean_ci(
        paired.random, SEED + 1
    )
    mean_diff, lo_diff, hi_diff = bootstrap_mean_ci(
        paired.attribution - paired.random, SEED + 2
    )

    summary_rows.append({
        "branch": branch,
        "method": method,
        "fraction": fraction,
        "n_records": len(paired),
        "attribution_mean_drop": mean_a,
        "attribution_ci_low": lo_a,
        "attribution_ci_high": hi_a,
        "random_mean_drop": mean_r,
        "random_ci_low": lo_r,
        "random_ci_high": hi_r,
        "paired_difference": mean_diff,
        "difference_ci_low": lo_diff,
        "difference_ci_high": hi_diff
    })

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, "deletion_summary_bootstrap.csv")
summary.to_csv(summary_path, index=False)

print("\nSUMMARY")
display(summary.round(4))

# -------------------- 9. Plot deletion curves ----------------

for branch in ["1d", "2d"]:
    fig, ax = plt.subplots(figsize=(9, 6))

    branch_summary = summary[summary.branch == branch]

    for method in METHODS[branch]:
        sub = branch_summary[branch_summary.method == method].sort_values(
            "fraction"
        )

        ax.plot(
            sub.fraction * 100,
            sub.attribution_mean_drop,
            marker="o",
            label=f"{method} — attribution"
        )

        ax.plot(
            sub.fraction * 100,
            sub.random_mean_drop,
            marker="s",
            linestyle="--",
            label=f"{method} — random"
        )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("ECG signal deleted (%)")
    ax.set_ylabel("Drop in true-class probability")
    ax.set_title(f"Corrected Deletion Faithfulness — {branch.upper()}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    plot_path = os.path.join(
        OUT_DIR, f"deletion_faithfulness_{branch}.png"
    )
    fig.savefig(plot_path, dpi=200)
    plt.show()
    plt.close(fig)

print("\nCompleted.")
print("Output directory:", OUT_DIR)
print("Raw results:", raw_path)
print("Summary:", summary_path)


# ---- notebook cell 34 ----

# ============================================================
# AUDIT 2: DELETION SENSITIVITY TO BLOCK SIZE
# 25, 50, and 100 ECG samples
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcdefaults()
import matplotlib.pyplot as plt

SENS_DIR = "results/module2/deletion_sensitivity_v1"
os.makedirs(SENS_DIR, exist_ok=True)

SEED = 20261116
N_PER_CLASS = 10
BLOCK_SIZES = [25, 50, 100]
FRACTIONS = [0.10, 0.20, 0.30]
RANDOM_REPEATS = 5
BOOTSTRAPS = 2000

# Use a balanced subset of jointly correct records.
eligible = df[
    (df.pred_1d == df.class_name) &
    (df.pred_2d == df.class_name)
].copy()

parts = []
for cls in CLASSES:
    group = eligible[eligible.class_name == cls]
    assert len(group) >= N_PER_CLASS, f"Not enough records for {cls}"
    parts.append(group.sample(N_PER_CLASS, random_state=SEED))

sample = (
    pd.concat(parts)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

cohort_to_zi = {
    int(c): i for i, c in enumerate(z["cohort_index"])
}

# Exact mapping from the 600-grid to the 5000 ECG samples.
col_centres = (
    (np.arange(600) + 0.5) * 1080 / 600 - 0.5
)
geom_valid = (col_centres >= 58) & (col_centres <= 1021)

sample_pixels = np.linspace(58, 1021, N)
sample_columns = (
    (sample_pixels + 0.5) * 600 / 1080 - 0.5
)

def map_to_ecg(map_600):
    a = np.asarray(map_600, dtype=float)
    valid = geom_valid & np.isfinite(a)
    assert valid.any(), "Map has no valid ECG support."
    return np.interp(
        sample_columns,
        np.arange(600)[valid],
        a[valid]
    )

def select_top_blocks(scores, n_blocks, block_size):
    scores = np.asarray(scores)
    candidates = []

    for start in range(0, N - block_size + 1, 5):
        candidates.append((
            float(scores[start:start + block_size].mean()),
            start
        ))

    candidates.sort(reverse=True)

    occupied = np.zeros(N, dtype=bool)
    selected = []

    for _, start in candidates:
        end = start + block_size
        if not occupied[start:end].any():
            selected.append(start)
            occupied[start:end] = True
            if len(selected) == n_blocks:
                break

    assert len(selected) == n_blocks
    return selected

def select_random_blocks(n_blocks, block_size, seed):
    local_rng = np.random.default_rng(seed)
    starts = np.arange(N - block_size + 1)
    local_rng.shuffle(starts)

    occupied = np.zeros(N, dtype=bool)
    selected = []

    for start in starts:
        end = start + block_size
        if not occupied[start:end].any():
            selected.append(int(start))
            occupied[start:end] = True
            if len(selected) == n_blocks:
                break

    assert len(selected) == n_blocks
    return selected

def delete_blocks(x, starts, block_size):
    out = np.asarray(x, dtype=np.float64).copy()

    for start in starts:
        end = start + block_size

        left = out[start - 1] if start > 0 else out[end]
        right = out[end] if end < N else left

        out[start:end] = np.linspace(
            left, right, block_size + 2
        )[1:-1]

    return out

def predict_1d(x):
    inp = ((x - MEAN) / SCALE).astype(np.float32)[None, :, None]
    return run_model("1d", m1, inp)[0]

def predict_2d(x, renderer):
    rgb = renderer.pert(x)
    image = (to_2d(rgb[None]) / 255.0).astype(np.float32)
    return run_model("2d", m2, image)[0]

METHODS = {
    "1d": ["R", "H1", "base1"],
    "2d": ["R", "H2", "base2"]
}

records = []
renderer = Renderer()

print("Sensitivity analysis: 40 records, 3 block sizes.")

try:
    for ri, row in sample.iterrows():

        cohort = int(row.cohort_index)
        zi = cohort_to_zi[cohort]
        x = load_ecg(row.ecg_csv_filename)
        target = CLASSES.index(row.class_name)

        assert len(x) == N

        # Verify original predictions against saved baselines.
        p1 = predict_1d(x)
        rgb0 = renderer.base(x)
        img0 = (to_2d(rgb0[None]) / 255.0).astype(np.float32)
        p2 = run_model("2d", m2, img0)[0]

        assert np.allclose(p1, z["out1_base"][zi], atol=1e-5), (
            f"1D baseline mismatch: {cohort}"
        )
        assert np.allclose(p2, z["out2_base"][zi], atol=1e-5), (
            f"2D baseline mismatch: {cohort}"
        )

        baseline = {"1d": float(p1[target]), "2d": float(p2[target])}

        # Convert all attribution maps into ECG coordinates.
        maps = {
            name: map_to_ecg(z[name][zi])
            for name in ["R", "H1", "H2", "base1", "base2"]
        }

        for branch, methods in METHODS.items():

            for block_size in BLOCK_SIZES:
                for fraction in FRACTIONS:

                    n_blocks = max(
                        1,
                        int(round(fraction * N / block_size))
                    )
                    actual_fraction = n_blocks * block_size / N

                    # Generate random controls once and reuse them
                    # across methods for a paired comparison.
                    random_sets = [
                        select_random_blocks(
                            n_blocks,
                            block_size,
                            SEED + cohort * 100000 +
                            block_size * 100 + int(fraction * 100) +
                            rep
                        )
                        for rep in range(RANDOM_REPEATS)
                    ]

                    random_drops = []

                    for starts in random_sets:
                        xr = delete_blocks(x, starts, block_size)

                        if branch == "1d":
                            pr = predict_1d(xr)[target]
                        else:
                            pr = predict_2d(xr, renderer)[target]

                        random_drops.append(
                            baseline[branch] - float(pr)
                        )

                    for method in methods:

                        top_starts = select_top_blocks(
                            maps[method], n_blocks, block_size
                        )
                        xtop = delete_blocks(x, top_starts, block_size)

                        if branch == "1d":
                            pt = predict_1d(xtop)[target]
                        else:
                            pt = predict_2d(xtop, renderer)[target]

                        attr_drop = baseline[branch] - float(pt)

                        records.append({
                            "cohort_index": cohort,
                            "class_name": row.class_name,
                            "branch": branch,
                            "method": method,
                            "block_size": block_size,
                            "requested_fraction": fraction,
                            "actual_fraction": actual_fraction,
                            "attribution_drop": attr_drop,
                            "random_drop_mean": float(np.mean(random_drops)),
                            "paired_difference": (
                                attr_drop - float(np.mean(random_drops))
                            ),
                            "random_drops": ";".join(
                                map(str, random_drops)
                            )
                        })

        if (ri + 1) % 5 == 0:
            print(f"Processed {ri + 1}/{len(sample)} records")
            gc.collect()

finally:
    plt.close(renderer.fig)
    gc.collect()

raw = pd.DataFrame(records)
raw_path = os.path.join(SENS_DIR, "sensitivity_raw.csv")
raw.to_csv(raw_path, index=False)

# Paired bootstrap over records.
def boot_ci(values, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(BOOTSTRAPS, len(values)), replace=True
    ).mean(axis=1)

    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975))
    )

summary_rows = []

for (branch, method, block_size, fraction), g in raw.groupby(
    ["branch", "method", "block_size", "requested_fraction"]
):
    mean, low, high = boot_ci(
        g["paired_difference"], SEED + block_size
    )

    attr_mean, attr_low, attr_high = boot_ci(
        g["attribution_drop"], SEED + 1
    )

    random_mean, random_low, random_high = boot_ci(
        g["random_drop_mean"], SEED + 2
    )

    summary_rows.append({
        "branch": branch,
        "method": method,
        "block_size": block_size,
        "fraction": fraction,
        "n_records": len(g),
        "attribution_mean_drop": attr_mean,
        "random_mean_drop": random_mean,
        "paired_difference": mean,
        "difference_ci_low": low,
        "difference_ci_high": high
    })

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(SENS_DIR, "sensitivity_summary.csv")
summary.to_csv(summary_path, index=False)

print("\nSENSITIVITY SUMMARY")
display(summary.round(4))

# Plot R(u) sensitivity across block sizes.
for branch in ["1d", "2d"]:
    fig, ax = plt.subplots(figsize=(9, 6))

    sub = summary[
        (summary.branch == branch) &
        (summary.method == "R")
    ]

    for block_size in BLOCK_SIZES:
        s = sub[sub.block_size == block_size].sort_values("fraction")

        ax.plot(
            s.fraction * 100,
            s.attribution_mean_drop,
            marker="o",
            label=f"R attribution, block={block_size}"
        )

        ax.plot(
            s.fraction * 100,
            s.random_mean_drop,
            marker="s",
            linestyle="--",
            label=f"Random, block={block_size}"
        )

    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("ECG signal deleted (%)")
    ax.set_ylabel("Drop in true-class probability")
    ax.set_title(f"Deletion Sensitivity — {branch.upper()}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        os.path.join(SENS_DIR, f"sensitivity_{branch}.png"),
        dpi=200
    )
    plt.show()
    plt.close(fig)

print("\nCompleted.")
print("Raw:", raw_path)
print("Summary:", summary_path)
print("Output directory:", SENS_DIR)

