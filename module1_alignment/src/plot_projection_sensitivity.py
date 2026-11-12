from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "Alignment-XAI/module1_alignment/results"
INFILE = BASE / "projection_sensitivity/projection_metrics.csv"
OUT = BASE / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)

METRICS = {
    "mean_agreement": "Mean agreement",
    "mean_disagreement": "Mean disagreement",
    "spearman": "Spearman correlation",
    "dice_top15": "Top-15% Dice",
    "iou_top15": "Top-15% IoU",
    "weighted_jaccard": "Weighted Jaccard"
}

SIZES = [6, 13, 24]
COLORS = ["#79B5D4", "#D47C9E", "#B58A6A"]
rng = np.random.default_rng(42)
N_BOOT = 2000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#FFFCFB",
    "axes.facecolor": "#FFFCFB",
    "savefig.facecolor": "#FFFCFB"
})

assert len(df) == 1054 * 3
assert not df.duplicated(["cohort_index", "mask_size"]).any()

def bootstrap_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    assert len(x) > 0

    estimates = np.empty(N_BOOT)
    for i in range(N_BOOT):
        estimates[i] = rng.choice(
            x, size=len(x), replace=True
        ).mean()

    return (
        x.mean(),
        np.percentile(estimates, 2.5),
        np.percentile(estimates, 97.5)
    )

def save(fig, name):
    fig.savefig(OUT / f"{name}.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)

# FIGURE 04: PROJECTION SENSITIVITY
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
summary = []

for ax, (metric, title) in zip(axes.flat, METRICS.items()):
    means, lows, highs = [], [], []

    for size in SIZES:
        x = df.loc[df["mask_size"] == size, metric]
        mean, lo, hi = bootstrap_mean(x)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)

        summary.append({
            "metric": metric,
            "mask_size": size,
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi
        })

    ax.plot(SIZES, means, color="#596B80", linewidth=1)

    for j, size in enumerate(SIZES):
        ax.errorbar(
            size, means[j],
            yerr=[
                [means[j] - lows[j]],
                [highs[j] - means[j]]
            ],
            fmt="o",
            color=COLORS[j],
            markersize=7,
            capsize=4
        )

    ax.set_xticks(SIZES)
    ax.set_xlabel("Vertical waveform-mask size (CAM pixels)")
    ax.set_ylabel("Mean metric value")
    ax.set_title(title)
    ax.grid(alpha=0.2)

fig.suptitle(
    "04 | Projection Sensitivity: Agreement and Disagreement",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "04_projection_sensitivity")

pd.DataFrame(summary).to_csv(
    OUT / "04_projection_bootstrap.csv", index=False
)

# FIGURE 05: WITHIN-ECG PAIRED CHANGES
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
changes = []

for ax, (metric, title) in zip(axes.flat, METRICS.items()):
    wide = df.pivot(
        index="cohort_index",
        columns="mask_size",
        values=metric
    )

    for j, size in enumerate([6, 24]):
        paired = wide[[13, size]].dropna()
        delta = (paired[size] - paired[13]).to_numpy()

        mean, lo, hi = bootstrap_mean(delta)

        ax.errorbar(
            mean, j,
            xerr=[
                [mean - lo],
                [hi - mean]
            ],
            fmt="o",
            color=COLORS[j * 2],
            markersize=7,
            capsize=4
        )

        changes.append({
            "metric": metric,
            "comparison": f"{size} vs 13",
            "n": len(delta),
            "mean_difference": mean,
            "ci_lower": lo,
            "ci_upper": hi
        })

    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks([0, 1], ["6 vs 13", "24 vs 13"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired difference (95% bootstrap CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)

fig.suptitle(
    "05 | Within-ECG Changes in Agreement and Disagreement",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "05_within_ecg_paired_changes")

pd.DataFrame(changes).to_csv(
    OUT / "05_paired_changes.csv", index=False
)

print("FIGURES 04 AND 05 GENERATED")
print("Output:", OUT)
