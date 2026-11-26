from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

# ============================================================
# STEP 1: PATHS AND CONFIGURATION
# ============================================================

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
GRAD = BASE / "gradcam_full"

OUT = BASE / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = (
    Path.home() / "XAI/imageData-Label2/label1/Testdata"
)

COHORT = 209
WINDOW_SECONDS = 0.6

# ============================================================
# STEP 2: LOAD METADATA AND ORIGINAL INPUTS
# ============================================================

metadata = pd.read_csv(
    GRAD / "gradcam_metadata.csv"
)

matches = np.flatnonzero(
    metadata["cohort_index"].to_numpy() == COHORT
)

assert len(matches) == 1

idx = int(matches[0])
row = metadata.iloc[idx]

assert row["class_name"] == "AFib"

ecg = pd.read_csv(
    ECG_ROOT / row["ecg_csv_filename"]
)["II"].to_numpy(dtype=float)

assert ecg.shape == (5000,)

t_ecg = np.arange(5000) / 500.0

image = np.asarray(
    Image.open(
        IMAGE_ROOT / row["image_relative_path"]
    ).convert("RGB")
)

# ============================================================
# STEP 3: LOAD ORIGINAL GRAD-CAM
# ============================================================

raw1 = np.asarray(
    np.load(
        GRAD / "gradcam_1d.npy",
        mmap_mode="r"
    )[idx],
    dtype=float
)

raw2 = np.asarray(
    np.load(
        GRAD / "gradcam_2d.npy",
        mmap_mode="r"
    )[idx],
    dtype=float
)

assert raw1.shape == (5000,)
assert raw2.shape == (300, 600)

raw1 = np.clip(raw1, 0, None)
raw2 = np.clip(raw2, 0, None)

raw1 = raw1 / (raw1.max() + 1e-8)
raw2 = raw2 / (raw2.max() + 1e-8)

# ============================================================
# STEP 4: LOAD ACTUAL ALIGNED EXPERIMENTAL RESULTS
# ============================================================

def load_record(filename):
    return np.asarray(
        np.load(
            ALIGN / filename,
            mmap_mode="r"
        )[idx],
        dtype=float
    )

g1 = load_record("normalized_1d.npy")
g2 = load_record("normalized_2d.npy")

agreement = load_record("agreement_maps.npy")
disagreement = load_record("disagreement_maps.npy")

valid = np.asarray(
    np.load(
        ALIGN / "valid_temporal_mask.npy",
        mmap_mode="r"
    )[idx],
    dtype=bool
)

assert g1.shape == (600,)
assert g2.shape == (600,)

assert np.allclose(
    agreement[valid],
    np.sqrt(g1[valid] * g2[valid]),
    atol=1e-5
)

assert np.allclose(
    disagreement[valid],
    np.abs(g1[valid] - g2[valid]),
    atol=1e-5
)

mean_A = float(agreement[valid].mean())
mean_D = float(disagreement[valid].mean())

# ============================================================
# STEP 5: RECONSTRUCT TEMPORAL ALIGNMENT
# ============================================================

geometry = pd.read_csv(
    ALIGN / "temporal_geometry.csv"
).set_index("cohort_index")

geom = geometry.loc[COHORT]

width = float(geom["image_width"])
x0 = float(geom["waveform_x0"])
x1 = float(geom["waveform_x1"])

u = np.arange(600)

x_original = (
    (u + 0.5) * width / 600 - 0.5
)

time = (
    10.0 * (x_original - x0) / (x1 - x0)
)

# ============================================================
# STEP 6: AUTOMATIC REGION SELECTION
# ============================================================

# Find the number of columns corresponding to 0.6 seconds.
step_seconds = np.median(
    np.diff(time)
)

window = max(
    3,
    int(round(WINDOW_SECONDS / step_seconds))
)

candidates = []

for start in range(600 - window + 1):

    end = start + window

    if not np.all(valid[start:end]):
        continue

    region_A = float(
        agreement[start:end].mean()
    )

    region_D = float(
        disagreement[start:end].mean()
    )

    candidates.append({
        "start": start,
        "end": end,
        "mean_A": region_A,
        "mean_D": region_D
    })

assert candidates, "No valid regions found."

# ------------------------------------------------------------
# PINK: Highest observed shared attribution
# ------------------------------------------------------------

best_agreement = max(
    candidates,
    key=lambda r: r["mean_A"]
)

# ------------------------------------------------------------
# BLUE: Highest observed disagreement
# Must not overlap the pink region.
# ------------------------------------------------------------

remaining = [
    r for r in candidates
    if (
        r["end"] <= best_agreement["start"]
        or
        r["start"] >= best_agreement["end"]
    )
]

assert remaining, "No nonoverlapping regions found."

best_disagreement = max(
    remaining,
    key=lambda r: r["mean_D"]
)

regions = [
    (
        best_agreement,
        "#F5A3BA",
        "Highest observed shared attribution"
    ),
    (
        best_disagreement,
        "#A5D6F5",
        "Highest observed disagreement"
    )
]

# ============================================================
# STEP 7: REGION UTILITIES
# ============================================================

def region_times(region):

    start = time[region["start"]]
    end = time[region["end"] - 1]

    return float(start), float(end)


def shade_region(ax):

    for region, color, label in regions:

        start, end = region_times(region)

        is_agreement = region is best_agreement

        region_color = (
            "#F2ACC9" if is_agreement else "#B0B7BF"
        )

        region_label = "R2" if is_agreement else "R1"

        ax.axvspan(
            start,
            end,
            color=region_color,
            alpha=0.25,
            linewidth=0,
            zorder=0
        )

        ax.text(
            (start + end) / 2,
            0.96,
            region_label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#333333",
            zorder=15,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=1
            )
        )


def region_summary(region, label):

    start, end = region_times(region)

    return (
        f"{label}: {start:.2f}-{end:.2f} s | "
        f"A={region['mean_A']:.3f}, "
        f"D={region['mean_D']:.3f}"
    )




# ============================================================
# STEP 8: COMPACT PUBLICATION FIGURE
# ============================================================

PINK = "#BC407D"
BLUE = "#579BD5"
DARK_BLUE = "#315B7D"
GREY = "#B0B7BF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42
})

fig = plt.figure(
    figsize=(10, 8.5),
    constrained_layout=True
)

gs = fig.add_gridspec(
    4, 2,
    height_ratios=[1.5, 1.2, 1.2, 0.45]
)

# ============================================================
# PANEL (a): ORIGINAL 1D AND 2D GRAD-CAM
# ============================================================

left = gs[0, 0].subgridspec(
    2, 1,
    height_ratios=[4, 0.65],
    hspace=0.04
)

# Original 1D ECG

ax1 = fig.add_subplot(left[0])

ax1.plot(
    t_ecg,
    ecg,
    color=DARK_BLUE,
    linewidth=0.8
)

shade_region(ax1)

ax1.set_title(
    "(a) 1D Grad-CAM",
    loc="left",
    fontweight="bold"
)

ax1.set_ylabel("ECG amplitude")
ax1.set_xlim(0, 10)
ax1.tick_params(labelbottom=False)
ax1.grid(alpha=0.12)

# Original 1D Grad-CAM heatmap

ax_strip = fig.add_subplot(
    left[1],
    sharex=ax1
)

ax_strip.imshow(
    raw1[np.newaxis, :],
    aspect="auto",
    extent=[0, 10, 0, 1],
    cmap="RdPu",
    vmin=0,
    vmax=1
)

for region, color, label in regions:

    start, end = region_times(region)

    ax_strip.axvspan(
        start,
        end,
        color="#777777",
        alpha=0.10,
        linewidth=0
    )

ax_strip.set_yticks([])
ax_strip.set_ylabel("CAM")
ax_strip.set_xlabel("Time (s)")
ax_strip.set_xticks([0, 2, 4, 6, 8, 10])

# Original 2D ECG and spatial Grad-CAM

ax2 = fig.add_subplot(gs[0, 1])

extent = [
    -0.5,
    image.shape[1] - 0.5,
    image.shape[0] - 0.5,
    -0.5
]

ax2.imshow(
    image,
    extent=extent,
    aspect="auto"
)

im = ax2.imshow(
    raw2,
    extent=extent,
    cmap="Blues",
    alpha=0.45,
    vmin=0,
    vmax=1,
    aspect="auto"
)

ax2.set_title(
    "2D Grad-CAM",
    loc="left",
    fontweight="bold"
)

ax2.set_xticks([])
ax2.set_yticks([])

for region, color, label in regions:

    start = x_original[region["start"]]
    end = x_original[region["end"] - 1]

    region_label = (
        "R2" if region is best_agreement else "R1"
    )

    for boundary in [start, end]:

        ax2.axvline(
            boundary,
            color="#444444",
            linewidth=0.9,
            alpha=0.85,
            zorder=10
        )

    ax2.text(
        (start + end) / 2,
        0.96,
        region_label,
        transform=ax2.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        zorder=15,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=1
        )
    )

cbar = fig.colorbar(
    im,
    ax=ax2,
    fraction=0.045,
    pad=0.02
)

cbar.set_label("Attribution")
cbar.set_ticks([0, 0.5, 1])

# ============================================================
# PANEL (b): ALIGNED ATTRIBUTION PROFILES
# ============================================================

ax3 = fig.add_subplot(gs[1, :])

shade_region(ax3)

ax3.plot(
    time[valid],
    g1[valid],
    color=PINK,
    linewidth=1.5,
    label="Aligned 1D Grad-CAM"
)

ax3.plot(
    time[valid],
    g2[valid],
    color=BLUE,
    linewidth=1.6,
    label="Projected 2D Grad-CAM"
)

ax3.set_title(
    "(b) Aligned attribution profiles",
    loc="left",
    fontweight="bold"
)

ax3.set_ylabel("Normalized\nattribution")
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 1.08)
ax3.set_xticks([0, 2, 4, 6, 8, 10])
ax3.set_yticks([0, 0.5, 1])
ax3.tick_params(labelbottom=False)

ax3.legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.01),
    frameon=False,
    ncol=2,
    fontsize=9
)

ax3.grid(alpha=0.12)

# ============================================================
# PANEL (c): LOCAL AGREEMENT AND DISAGREEMENT
# ============================================================

ax4 = fig.add_subplot(
    gs[2, :],
    sharex=ax3
)

shade_region(ax4)

ax4.plot(
    time[valid],
    agreement[valid],
    color=PINK,
    linewidth=1.5,
    label="Agreement"
)

ax4.plot(
    time[valid],
    disagreement[valid],
    color=DARK_BLUE,
    linewidth=1.5,
    label="Disagreement"
)

ax4.set_title(
    "(c) Local agreement and disagreement",
    loc="left",
    fontweight="bold"
)

ax4.set_ylabel("Local score")
ax4.set_xlabel("ECG time (seconds)")
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 1.08)
ax4.set_xticks([0, 2, 4, 6, 8, 10])
ax4.set_yticks([0, 0.5, 1])

ax4.legend(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.01),
    frameon=False,
    ncol=2,
    fontsize=9
)

ax4.grid(alpha=0.12)

# ============================================================
# REGIONAL AND RECORDING-LEVEL SUMMARY
# ============================================================

ax5 = fig.add_subplot(gs[3, :])
ax5.axis("off")

pink_text = region_summary(
    best_agreement,
    "R2 (Agreement)"
)

blue_text = region_summary(
    best_disagreement,
    "R1 (Disagreement)"
)

ax5.text(
    0.5, 0.85,
    blue_text,
    ha="center",
    color="#333333",
    fontsize=9
)

ax5.text(
    0.5, 0.48,
    pink_text,
    ha="center",
    color="#333333",
    fontsize=9
)

ax5.text(
    0.5, 0.08,
    f"Mean agreement = {mean_A:.4f}  |  "
    f"Mean disagreement = {mean_D:.4f}",
    ha="center",
    color="#222222",
    fontsize=9,
    fontweight="medium"
)

# ============================================================
# STEP 9: SAVE FIGURES
# ============================================================

png_path = OUT / "Figure5_AFib209_compact_v6.png"
pdf_path = OUT / "Figure5_AFib209_compact_v6.pdf"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.05
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    pad_inches=0.05
)

plt.close(fig)

# ============================================================
# STEP 10: PRINT RESULTS
# ============================================================

print("\nCOMPACT FIGURE - AFIB COHORT 209")
print("=" * 60)

print(pink_text)
print(blue_text)

print(f"Mean agreement: {mean_A:.4f}")
print(f"Mean disagreement: {mean_D:.4f}")

print("\nSaved files:")
print(png_path)
print(pdf_path)

print("\nFigure generation completed successfully.")
