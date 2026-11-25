from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# STEP 8A: EXTENDED MODULE 1 PUBLICATION FIGURES
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"

STATS = BASE / "extended_statistical_analysis"
SENS = BASE / "agreement_formulation_sensitivity"
NULL = BASE / "within_class_mispairing"

OUT = BASE / "extended_publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["AFib", "GSVT", "SB", "SR"]

C1 = "#B44A73"
C2 = "#2878B5"
C3 = "#999999"

CLASS_COLORS = [
    "#4477AA",
    "#EE7733",
    "#228833",
    "#AA3377"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white"
})

# ============================================================
# 1. LOAD EXPERIMENTAL RESULTS
# ============================================================

df = pd.read_csv(
    STATS / "extended_recording_metrics.csv"
)

null = pd.read_csv(
    STATS / "null_holm_corrected.csv"
)

overall_null = pd.read_csv(
    STATS / "overall_stratified_null.csv"
)

sensitivity = pd.read_csv(
    SENS / "directional_sensitivity_summary.csv"
)

correlations = pd.read_csv(
    SENS / "formulation_correlations.csv"
)

assert len(df) == 1040
assert len(sensitivity) == 25

def save_figure(fig, name):

    fig.savefig(
        OUT / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", name)

# ============================================================
# FIGURE 1: MASS, DIRECTION AND ENTROPY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    constrained_layout=True
)

# ------------------------------------------------------------
# A. Shared attribution mass
# ------------------------------------------------------------

ax = axes[0, 0]

values = [
    df.loc[df["class_name"] == c, "shared_mass"]
    .to_numpy()
    for c in CLASSES
]

parts = ax.violinplot(
    values,
    showmedians=True,
    showextrema=False
)

for body, color in zip(
    parts["bodies"], CLASS_COLORS
):
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.55)

parts["cmedians"].set_color("black")

ax.set_xticks(range(1, 5), CLASSES)
ax.set_ylabel("Shared attribution mass")
ax.set_ylim(0, 1)
ax.set_title(
    "(a) Class-wise shared attribution mass",
    loc="left",
    weight="bold"
)

ax.grid(axis="y", alpha=0.15)

# ------------------------------------------------------------
# B. Directional disagreement
# ------------------------------------------------------------

ax = axes[0, 1]

direction = df.groupby("class_name")[
    [
        "1d_dominant_pct_salient",
        "2d_dominant_pct_salient",
        "balanced_pct_salient"
    ]
].mean().reindex(CLASSES)

x = np.arange(4)

a = direction[
    "1d_dominant_pct_salient"
].to_numpy()

b = direction[
    "2d_dominant_pct_salient"
].to_numpy()

c = direction[
    "balanced_pct_salient"
].to_numpy()

ax.bar(
    x, a,
    color=C1,
    label="1D-dominant"
)

ax.bar(
    x, b,
    bottom=a,
    color=C2,
    label="2D-dominant"
)

ax.bar(
    x, c,
    bottom=a+b,
    color=C3,
    label="Balanced"
)

for i in range(4):

    for val, bottom in [
        (a[i], 0),
        (b[i], a[i]),
        (c[i], a[i]+b[i])
    ]:

        ax.text(
            i,
            bottom + val/2,
            f"{val:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=9
        )

ax.set_xticks(x, CLASSES)
ax.set_ylim(0, 100)
ax.set_ylabel("Salient positions (%)")

ax.set_title(
    "(b) Directional attribution dominance",
    loc="left",
    weight="bold"
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10)
)

# ------------------------------------------------------------
# C. Normalized attribution entropy
# ------------------------------------------------------------

ax = axes[1, 0]

for i, cls in enumerate(CLASSES):

    group = df[df["class_name"] == cls]

    for metric, offset, color in [
        ("entropy_1d", -0.17, C1),
        ("entropy_2d",  0.17, C2)
    ]:

        bp = ax.boxplot(
            group[metric],
            positions=[i + offset],
            widths=0.28,
            patch_artist=True,
            showfliers=False
        )

        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.6)

        bp["medians"][0].set_color("black")

ax.plot([], [], color=C1, lw=7, label="1D")
ax.plot([], [], color=C2, lw=7, label="2D")

ax.set_xticks(x, CLASSES)
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("Normalized Shannon entropy")

ax.set_title(
    "(c) Attribution concentration",
    loc="left",
    weight="bold"
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.15)

# ------------------------------------------------------------
# D. Prediction correctness
# ------------------------------------------------------------

ax = axes[1, 1]

correct = df.loc[
    df["both_correct"], "shared_mass"
]

incorrect = df.loc[
    ~df["both_correct"], "shared_mass"
]

bp = ax.boxplot(
    [correct, incorrect],
    patch_artist=True,
    showfliers=False,
    widths=0.5
)

for box, color in zip(
    bp["boxes"], ["#228833", "#CC6677"]
):

    box.set_facecolor(color)
    box.set_alpha(0.55)

for med in bp["medians"]:
    med.set_color("black")

ax.set_xticks(
    [1, 2],
    [
        f"Both correct\n(n={len(correct)})",
        f"At least one incorrect\n(n={len(incorrect)})"
    ]
)

ax.set_ylim(0, 1)
ax.set_ylabel("Shared attribution mass")

ax.set_title(
    "(d) Association with prediction correctness",
    loc="left",
    weight="bold"
)

ax.grid(axis="y", alpha=0.15)

save_figure(
    fig,
    "01_mass_direction_entropy"
)

# ============================================================
# FIGURE 2: CHANCE-CALIBRATED AGREEMENT
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    constrained_layout=True
)

primary = null[
    null["cohort"] == "all_mass_valid"
]

metric_names = [
    ("shared_mass", "Shared attribution mass"),
    ("minmax_agreement", "Min-max agreement"),
    ("dice_top15", "Top-15% Dice")
]

for panel, (metric, title) in enumerate(
    metric_names
):

    ax = axes.flat[panel]

    data = primary[
        primary["metric"] == metric
    ].set_index("class_name").reindex(CLASSES)

    x = np.arange(4)
    w = 0.35

    ax.bar(
        x-w/2,
        data["actual_median"],
        width=w,
        color=C2,
        label="Actual pairing"
    )

    ax.bar(
        x+w/2,
        data["null_median"],
        width=w,
        color="#BBBBBB",
        label="Mispaired reference"
    )

    for i, (_, row) in enumerate(
        data.iterrows()
    ):

        if row["p_holm"] < 0.05:

            height = max(
                row["actual_median"],
                row["null_median"]
            )

            ax.text(
                i,
                height + 0.015,
                "*",
                ha="center",
                fontsize=14
            )

    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Median score")

    ax.set_ylim(
        0,
        0.60 if metric == "shared_mass" else 0.30
    )

    ax.set_title(
        f"({chr(97+panel)}) {title}",
        loc="left",
        weight="bold"
    )

    ax.grid(axis="y", alpha=0.15)

    if panel == 0:
        ax.legend(frameon=False)

# ------------------------------------------------------------
# D. Percentage above class-specific null95
# ------------------------------------------------------------

ax = axes[1, 1]

data = primary[
    primary["metric"] == "shared_mass"
].set_index("class_name").reindex(CLASSES)

bars = ax.bar(
    np.arange(4),
    data["pct_above_null95"],
    color=CLASS_COLORS
)

ax.axhline(
    5,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Nominal 5% reference"
)

for bar in bars:

    h = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        h + 0.5,
        f"{h:.1f}%",
        ha="center"
    )

ax.set_xticks(np.arange(4), CLASSES)
ax.set_ylim(0, 25)

ax.set_ylabel("Recordings above null95 (%)")

ax.set_title(
    "(d) Recording-specific shared mass",
    loc="left",
    weight="bold"
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.15)

save_figure(
    fig,
    "02_chance_calibrated_agreement"
)

# ============================================================
# FIGURE 3: FORMULATION AND THRESHOLD SENSITIVITY
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9),
    constrained_layout=True
)

pairs = [
    (
        "original_mean_agreement",
        "shared_mass",
        "Min-max agreement",
        "Shared attribution mass"
    ),
    (
        "original_weighted_jaccard",
        "mass_weighted_jaccard",
        "Original weighted Jaccard",
        "Mass-based weighted Jaccard"
    ),
    (
        "original_mean_disagreement",
        "total_variation",
        "Min-max disagreement",
        "Total variation distance"
    )
]

for panel, (m1, m2, xlabel, ylabel) in enumerate(
    pairs
):

    ax = axes.flat[panel]

    subset = df.copy()

    if m1 == "original_weighted_jaccard":

        subset = subset[
            subset["original_valid_similarity"]
        ]

    for cls, color in zip(
        CLASSES, CLASS_COLORS
    ):

        group = subset[
            subset["class_name"] == cls
        ]

        ax.scatter(
            group[m1],
            group[m2],
            s=10,
            color=color,
            alpha=0.45,
            edgecolors="none",
            label=cls
        )

    corr = correlations[
        (correlations["group"] == "All")
        & (correlations["original_metric"] == m1)
        & (correlations["extended_metric"] == m2)
    ].iloc[0]

    ax.text(
        0.03,
        0.97,
        (
            f"Pearson r = {corr['pearson_r']:.3f}\n"
            f"Spearman rho = {corr['spearman_rho']:.3f}\n"
            f"n = {int(corr['n'])}"
        ),
        transform=ax.transAxes,
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none"
        )
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"({chr(97+panel)}) Formulation comparison",
        loc="left",
        weight="bold"
    )

    ax.grid(alpha=0.12)

    if panel == 0:
        ax.legend(frameon=False, loc="lower right")

# ------------------------------------------------------------
# D. Directional threshold sensitivity
# ------------------------------------------------------------

ax = axes[1, 1]

matrix = sensitivity.pivot(
    index="top_fraction",
    columns="dominance_ratio",
    values="mean_dominance_difference"
).sort_index().sort_index(axis=1)

limit = np.abs(matrix.to_numpy()).max()

im = ax.imshow(
    matrix.to_numpy(),
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-limit,
        vcenter=0,
        vmax=limit
    ),
    aspect="auto"
)

for i in range(matrix.shape[0]):

    for j in range(matrix.shape[1]):

        value = matrix.iloc[i, j]

        ax.text(
            j, i,
            f"{value:+.2f}",
            ha="center",
            va="center",
            color=(
                "white"
                if abs(value) > limit*0.65
                else "black"
            )
        )

ax.set_xticks(
    range(matrix.shape[1]),
    [f"{v:g}:1" for v in matrix.columns]
)

ax.set_yticks(
    range(matrix.shape[0]),
    [f"{v*100:.0f}%" for v in matrix.index]
)

ax.set_xlabel("Required dominance ratio")
ax.set_ylabel("Top saliency fraction")

ax.set_title(
    "(d) Directional threshold sensitivity",
    loc="left",
    weight="bold"
)

# Outline baseline: top 15%, ratio 2:1
from matplotlib.patches import Rectangle

ax.add_patch(
    Rectangle(
        (0.5, 1.5),
        1, 1,
        fill=False,
        edgecolor="black",
        linewidth=2.0
    )
)

fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="1D minus 2D dominance (pp)"
)

save_figure(
    fig,
    "03_formulation_threshold_sensitivity"
)

print("\n" + "=" * 65)
print("STEP 8A COMPLETED")
print("Publication figures saved in:")
print(OUT)
print("=" * 65)
