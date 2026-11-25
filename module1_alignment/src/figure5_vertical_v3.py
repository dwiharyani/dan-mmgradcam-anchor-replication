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

        ax.axvspan(
            start,
            end,
            color=color,
            alpha=0.35,
            zorder=0
        )

        ax.axvline(
            start,
            color=color,
            linestyle="--",
            linewidth=0.9
        )

        ax.axvline(
            end,
            color=color,
            linestyle="--",
            linewidth=0.9
        )


def region_summary(region, label):

    start, end = region_times(region)

    return (
        f"{label}: {start:.2f}-{end:.2f} s | "
        f"A={region['mean_A']:.3f}, "
        f"D={region['mean_D']:.3f}"
    )


# ============================================================
# STEP 8: FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42
})

fig = plt.figure(
    figsize=(13, 17),
    constrained_layout=True
)

gs = fig.add_gridspec(
    6, 1,
    height_ratios=[
        1.4,
        1.7,
        1.2,
        1.3,
        1.3,
        0.7
    ]
)

fig.suptitle(
    "Cross-Representation Grad-CAM Analysis | "
    "AFib Cohort 209",
    fontsize=17,
    fontweight="bold"
)

# ============================================================
# PANEL A: ORIGINAL ECG + 1D GRAD-CAM
# ============================================================

subA = gs[0].subgridspec(
    2, 1,
    height_ratios=[4, 0.6],
    hspace=0.08
)

axA = fig.add_subplot(subA[0])

axA.plot(
    t_ecg,
    ecg,
    color="#17375E",
    linewidth=0.75
)

shade_region(axA)

axA.set_title(
    "A | Original 1D ECG Signal and 1D Grad-CAM",
    loc="left",
    fontweight="bold"
)

axA.set_ylabel(
    "ECG amplitude\n(dataset units)"
)

axA.set_xlim(0, 10)
axA.grid(alpha=0.12)

axStrip = fig.add_subplot(
    subA[1],
    sharex=axA
)

axStrip.imshow(
    raw1[np.newaxis, :],
    aspect="auto",
    extent=[0, 10, 0, 1],
    cmap="Reds",
    vmin=0,
    vmax=1
)

axStrip.set_yticks([])
axStrip.set_ylabel(
    "1D CAM",
    rotation=0,
    labelpad=30
)

axStrip.set_xlabel("ECG time (seconds)")

# ============================================================
# PANEL B: ORIGINAL 2D ECG + 2D GRAD-CAM
# ============================================================

axB = fig.add_subplot(gs[1])

axB.imshow(
    image,
    aspect="auto"
)

im = axB.imshow(
    raw2,
    extent=[
        0,
        image.shape[1],
        image.shape[0],
        0
    ],
    cmap="turbo",
    alpha=0.45,
    vmin=0,
    vmax=1,
    aspect="auto"
)

axB.set_title(
    "B | Original 2D ECG Image and Spatial Grad-CAM",
    loc="left",
    fontweight="bold"
)

axB.set_xticks([])
axB.set_yticks([])

# Highlight the corresponding original-image x coordinates.
for region, color, label in regions:

    start = x_original[region["start"]]
    end = x_original[region["end"] - 1]

    axB.axvspan(
        start,
        end,
        color=color,
        alpha=0.18
    )

fig.colorbar(
    im,
    ax=axB,
    fraction=0.025,
    pad=0.01,
    label="Normalized 2D Grad-CAM"
)

# ============================================================
# PANEL C: PROJECTED 2D ATTRIBUTION
# ============================================================

axC = fig.add_subplot(gs[2])

axC.plot(
    time[valid],
    g2[valid],
    color="#168BDF",
    linewidth=1.6
)

shade_region(axC)

axC.set_title(
    "C | Waveform-Guided Projection onto ECG Time",
    loc="left",
    fontweight="bold"
)

axC.set_ylabel(
    "Projected 2D\nattribution"
)

axC.set_xlabel("ECG time (seconds)")
axC.set_xlim(0, 10)
axC.set_ylim(0, 1.05)

axC.grid(alpha=0.12)

# ============================================================
# PANEL D: ALIGNED ATTRIBUTION PROFILES
# ============================================================

axD = fig.add_subplot(gs[3])

axD.plot(
    time[valid],
    g1[valid],
    color="#E83E8C",
    linewidth=1.4,
    label="1D Grad-CAM"
)

axD.plot(
    time[valid],
    g2[valid],
    color="#168BDF",
    linewidth=1.5,
    label="2D projected Grad-CAM"
)

shade_region(axD)

axD.set_title(
    "D | Temporally Aligned Attribution Profiles",
    loc="left",
    fontweight="bold"
)

axD.set_ylabel(
    "Normalized attribution"
)

axD.set_xlim(0, 10)
axD.set_ylim(0, 1.05)

axD.legend(
    loc="upper right",
    frameon=False,
    ncol=2
)

axD.grid(alpha=0.12)

# ============================================================
# PANEL E: AGREEMENT AND DISAGREEMENT
# ============================================================

axE = fig.add_subplot(gs[4])

axE.plot(
    time[valid],
    agreement[valid],
    color="#B91C3C",
    linewidth=1.4,
    label="Agreement"
)

axE.plot(
    time[valid],
    disagreement[valid],
    color="#174A8B",
    linewidth=1.4,
    label="Disagreement"
)

shade_region(axE)

axE.set_title(
    "E | Local Agreement and Disagreement",
    loc="left",
    fontweight="bold"
)

axE.set_ylabel("Local score")
axE.set_xlabel("ECG time (seconds)")

axE.set_xlim(0, 10)
axE.set_ylim(0, 1.05)

axE.legend(
    loc="upper right",
    frameon=False,
    ncol=2
)

axE.grid(alpha=0.12)

# ============================================================
# PANEL F: REGION INTERPRETATION
# ============================================================

axF = fig.add_subplot(gs[5])
axF.axis("off")

pink_text = region_summary(
    best_agreement,
    "PINK: Highest shared attribution"
)

blue_text = region_summary(
    best_disagreement,
    "BLUE: Highest disagreement"
)

axF.text(
    0.5,
    0.78,
    pink_text,
    transform=axF.transAxes,
    ha="center",
    fontsize=11,
    color="#A51D4B",
    fontweight="bold"
)

axF.text(
    0.5,
    0.48,
    blue_text,
    transform=axF.transAxes,
    ha="center",
    fontsize=11,
    color="#174A8B",
    fontweight="bold"
)

axF.text(
    0.5,
    0.15,
    (
        f"Whole-recording mean agreement = {mean_A:.4f}"
        f"   |   Mean disagreement = {mean_D:.4f}"
    ),
    transform=axF.transAxes,
    ha="center",
    fontsize=11,
    fontweight="bold"
)

# ============================================================
# STEP 9: SAVE FIGURES AND REGION METRICS
# ============================================================

records = []

for region, color, label in regions:

    start, end = region_times(region)

    records.append({
        "cohort_index": COHORT,
        "class_name": row["class_name"],
        "region": label,
        "start_seconds": start,
        "end_seconds": end,
        "mean_agreement": region["mean_A"],
        "mean_disagreement": region["mean_D"]
    })

pd.DataFrame(records).to_csv(
    OUT / "Figure5_AFib209_region_metrics.csv",
    index=False
)

png_path = OUT / "Figure5_AFib209_vertical_v3.png"
pdf_path = OUT / "Figure5_AFib209_vertical_v3.pdf"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# STEP 10: PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FIGURE 5 - AFIB COHORT 209")
print("=" * 65)

print(region_summary(
    best_agreement,
    "Highest shared attribution"
))

print(region_summary(
    best_disagreement,
    "Highest disagreement"
))

print(f"\nWhole-recording agreement: {mean_A:.4f}")
print(f"Whole-recording disagreement: {mean_D:.4f}")

print("\nSaved files:")
print(png_path)
print(pdf_path)

print("\nFigure generation completed successfully.")
