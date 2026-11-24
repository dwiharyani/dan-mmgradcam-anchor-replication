from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
OUT = BASE / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"

CLASSES = ["AFib", "GSVT", "SB", "SR"]
COLORS = {
    "AFib": "#D47C9E",
    "GSVT": "#79B5D4",
    "SB": "#B58A6A",
    "SR": "#B89BB0",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#FFFCFB",
    "axes.facecolor": "#FFFCFB",
    "savefig.facecolor": "#FFFCFB",
})

metrics = pd.read_csv(
    BASE / "alignment_0px/agreement_metrics.csv"
)

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
)

subgroups = pd.read_csv(
    BASE / "subgroup_analysis/subgroup_manifest.csv"
)

geometry = pd.read_csv(
    BASE / "alignment_0px/temporal_geometry.csv"
).set_index("cohort_index")

g1 = np.load(
    BASE / "alignment_0px/normalized_1d.npy",
    mmap_mode="r"
)
g2 = np.load(
    BASE / "alignment_0px/normalized_2d.npy",
    mmap_mode="r"
)
agreement = np.load(
    BASE / "alignment_0px/agreement_maps.npy",
    mmap_mode="r"
)
disagreement = np.load(
    BASE / "alignment_0px/disagreement_maps.npy",
    mmap_mode="r"
)
valid_mask = np.load(
    BASE / "alignment_0px/valid_temporal_mask.npy",
    mmap_mode="r"
)

assert len(metrics) == len(metadata) == len(subgroups) == 1054
assert metrics["cohort_index"].is_unique
assert metadata["cohort_index"].is_unique
assert subgroups["cohort_index"].is_unique

assert g1.shape == g2.shape == agreement.shape == disagreement.shape == (1054, 600)

# Preserve exact array row correspondence.
metadata = metadata.reset_index(drop=True)
metrics = metadata[["cohort_index"]].merge(
    metrics, on="cohort_index", validate="one_to_one"
)

valid_metrics = metrics[
    ~(metrics["degenerate_1d"] | metrics["degenerate_2d"])
].copy()

assert len(valid_metrics) == 1033


def save(fig, filename):
    fig.savefig(
        OUT / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        OUT / f"{filename}.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================================================
# FIGURE 01 — COHORT OVERVIEW
# =====================================================
print("Generating Figure 01...", flush=True)

both_correct = int(subgroups["both_correct"].sum())
nondegenerate = int(subgroups["nondegenerate"].sum())

counts = metrics["class_name"].value_counts().reindex(CLASSES)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

labels = [
    "Both models correct",
    "At least one incorrect",
    "Nondegenerate explanations",
    "Degenerate explanations",
]
values = [
    both_correct,
    1054 - both_correct,
    nondegenerate,
    1054 - nondegenerate,
]

axes[0].barh(
    labels, values,
    color=["#D47C9E", "#B58A6A", "#79B5D4", "#B89BB0"]
)
axes[0].invert_yaxis()
axes[0].set_xlabel("Number of ECG recordings")
axes[0].set_title("Cohort and explanation validity")

for i, v in enumerate(values):
    axes[0].text(v + 8, i, str(v), va="center")

axes[1].bar(
    CLASSES,
    counts.values,
    color=[COLORS[c] for c in CLASSES]
)
axes[1].set_title("Distribution by arrhythmia class")
axes[1].set_ylabel("Number of ECG recordings")

fig.suptitle(
    "01 | Cohort and Grad-CAM Overview",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "01_cohort_overview")


# =====================================================
# FIGURE 02 — CLASS-WISE DISTRIBUTIONS
# =====================================================
print("Generating Figure 02...", flush=True)

plot_metrics = [
    ("mean_agreement", "Mean agreement"),
    ("mean_disagreement", "Mean disagreement"),
    ("spearman", "Spearman correlation"),
    ("dice_top15", "Top-15% Dice"),
    ("iou_top15", "Top-15% IoU"),
    ("weighted_jaccard", "Weighted Jaccard"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (metric, title) in zip(axes.flat, plot_metrics):
    arrays = [
        valid_metrics.loc[
            valid_metrics["class_name"] == cls, metric
        ].dropna().to_numpy()
        for cls in CLASSES
    ]

    violin = ax.violinplot(
        arrays,
        positions=np.arange(1, 5),
        showmedians=True,
        showextrema=False
    )

    for body, cls in zip(violin["bodies"], CLASSES):
        body.set_facecolor(COLORS[cls])
        body.set_edgecolor(COLORS[cls])
        body.set_alpha(0.5)

    violin["cmedians"].set_color("#25334A")

    ax.set_xticks(range(1, 5), CLASSES)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "02 | Explanation Agreement and Disagreement by ECG Class",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "02_classwise_distributions")


# =====================================================
# FIGURE 03 — BOOTSTRAP MEDIAN CONFIDENCE INTERVALS
# =====================================================
print("Generating Figure 03...", flush=True)

rng = np.random.default_rng(42)
BOOTSTRAPS = 2000

records = []

for metric, title in plot_metrics:
    for cls in CLASSES:
        x = valid_metrics.loc[
            valid_metrics["class_name"] == cls, metric
        ].dropna().to_numpy()

        boot = np.empty(BOOTSTRAPS)

        for b in range(BOOTSTRAPS):
            sample = rng.choice(x, size=len(x), replace=True)
            boot[b] = np.median(sample)

        records.append({
            "metric": metric,
            "class_name": cls,
            "n": len(x),
            "median": np.median(x),
            "ci_lower": np.percentile(boot, 2.5),
            "ci_upper": np.percentile(boot, 97.5),
        })

ci = pd.DataFrame(records)
ci.to_csv(OUT / "03_bootstrap_intervals.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (metric, title) in zip(axes.flat, plot_metrics):
    subset = ci[ci["metric"] == metric]

    for j, cls in enumerate(CLASSES):
        r = subset[subset["class_name"] == cls].iloc[0]

        ax.errorbar(
            r["median"], j,
            xerr=[
                [r["median"] - r["ci_lower"]],
                [r["ci_upper"] - r["median"]]
            ],
            fmt="o",
            color=COLORS[cls],
            capsize=4
        )

    ax.set_yticks(range(4), CLASSES)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Median with 95% bootstrap CI")
    ax.grid(axis="x", alpha=0.2)

fig.suptitle(
    "03 | Class-wise Bootstrap Confidence Intervals",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "03_bootstrap_intervals")


# =====================================================
# FIGURE 06 — RECORD-LEVEL RELATIONSHIPS
# =====================================================
print("Generating Figure 06...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

relationships = [
    ("mean_agreement", "spearman", "Agreement vs Spearman"),
    ("mean_disagreement", "spearman", "Disagreement vs Spearman"),
    ("mean_agreement", "dice_top15", "Agreement vs Dice"),
    ("mean_disagreement", "dice_top15", "Disagreement vs Dice"),
]

for ax, (xcol, ycol, title) in zip(axes.flat, relationships):
    for cls in CLASSES:
        subset = valid_metrics[
            valid_metrics["class_name"] == cls
        ]

        ax.scatter(
            subset[xcol], subset[ycol],
            s=12,
            alpha=0.4,
            color=COLORS[cls],
            label=cls
        )

    ax.set_xlabel(xcol.replace("_", " ").title())
    ax.set_ylabel(ycol.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.2)

axes[0, 0].legend()

fig.suptitle(
    "06 | Record-level Attribution Relationships",
    fontsize=16, fontweight="bold"
)
fig.tight_layout()
save(fig, "06_record_level_relationships")


# =====================================================
# FIGURE 07 — REPRESENTATIVE PAIRED ECG TRACES
# =====================================================
print("Generating Figure 07...", flush=True)

representatives = []

# Select a recording close to the class median Dice.
for cls in CLASSES:
    subset = valid_metrics[
        valid_metrics["class_name"] == cls
    ].copy()

    median_dice = subset["dice_top15"].median()

    subset["distance"] = (
        subset["dice_top15"] - median_dice
    ).abs()

    record = subset.sort_values(
        ["distance", "cohort_index"]
    ).iloc[0]

    representatives.append(record)

fig, axes = plt.subplots(
    4, 3, figsize=(18, 14),
    gridspec_kw={"width_ratios": [1, 1.7, 1.7]}
)

for row_idx, record in enumerate(representatives):
    cls = record["class_name"]
    cohort = int(record["cohort_index"])

    idx = metadata.index[
        metadata["cohort_index"] == cohort
    ][0]

    filename = metadata.loc[idx, "ecg_csv_filename"]

    signal = pd.read_csv(
        ECG_ROOT / filename
    )["II"].to_numpy(dtype=float)

    geom = geometry.loc[cohort]

    width = int(geom["image_width"])
    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])

    x_original = (
        (np.arange(600) + 0.5) * width / 600 - 0.5
    )

    time = 10 * (x_original - x0) / (x1 - x0)
    mask = valid_mask[idx].astype(bool)

    t_signal = np.arange(len(signal)) / 500.0

    ax0, ax1, ax2 = axes[row_idx]

    ax0.plot(
        t_signal, signal,
        color=COLORS[cls],
        linewidth=0.65
    )
    ax0.set_title(f"{cls} | Cohort {cohort}")
    ax0.set_xlim(0, 10)

    ax1.plot(
        time[mask], g1[idx][mask],
        color="#D47C9E",
        linewidth=1,
        label="1D Grad-CAM"
    )
    ax1.plot(
        time[mask], g2[idx][mask],
        color="#79B5D4",
        linewidth=1,
        label="2D projected Grad-CAM"
    )
    ax1.set_title("Branch-level attribution")
    ax1.set_ylim(0, 1.05)

    ax2.plot(
        time[mask], agreement[idx][mask],
        color="#9B1C31",
        linewidth=1,
        label="Agreement"
    )
    ax2.plot(
        time[mask], disagreement[idx][mask],
        color="#174A8B",
        linewidth=1,
        label="Disagreement"
    )
    ax2.set_title("Agreement and disagreement")
    ax2.set_ylim(0, 1.05)

    # Display recording-level agreement/disagreement scores.
    agreement_score = float(record["mean_agreement"])
    disagreement_score = float(record["mean_disagreement"])

    score_text = (
        f"Agreement = {agreement_score:.4f}\n"
        f"Disagreement = {disagreement_score:.4f}"
    )

    ax2.text(
        0.98, 0.96,
        score_text,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#DDDDDD",
            alpha=0.9
        )
    )

    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.15)

    if row_idx == 0:
        ax1.legend(fontsize=8)
        ax2.legend(fontsize=8)

    if row_idx == 3:
        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("ECG time (seconds)")

fig.suptitle(
    "07 | Representative Paired Attribution Traces",
    fontsize=17, fontweight="bold"
)
fig.tight_layout()
save(fig, "07_representative_attribution_traces")

print("\nPUBLICATION FIGURES GENERATED")
print("Output:", OUT)
