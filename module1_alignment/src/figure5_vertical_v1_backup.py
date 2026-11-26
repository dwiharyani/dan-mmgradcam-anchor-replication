from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
GRAD = BASE / "gradcam_full"
OUT = BASE / "publication_figures"
OUT.mkdir(parents=True, exist_ok=True)

ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

COHORT = 209
metadata = pd.read_csv(GRAD / "gradcam_metadata.csv")
matches = np.flatnonzero(metadata["cohort_index"].to_numpy() == COHORT)
assert len(matches) == 1
idx = int(matches[0])
row = metadata.iloc[idx]
assert row["class_name"] == "AFib"

# Load original ECG and ECG image
ecg = pd.read_csv(
    ECG_ROOT / row["ecg_csv_filename"]
)["II"].to_numpy(dtype=float)
assert len(ecg) == 5000
t_ecg = np.arange(5000) / 500

image = np.asarray(
    Image.open(IMAGE_ROOT / row["image_relative_path"]).convert("RGB")
)

# Load original Grad-CAM and saved aligned outputs
raw1 = np.load(GRAD / "gradcam_1d.npy", mmap_mode="r")[idx]
raw2 = np.load(GRAD / "gradcam_2d.npy", mmap_mode="r")[idx]

g1 = np.load(ALIGN / "normalized_1d.npy", mmap_mode="r")[idx]
g2 = np.load(ALIGN / "normalized_2d.npy", mmap_mode="r")[idx]
a = np.load(ALIGN / "agreement_maps.npy", mmap_mode="r")[idx]
d = np.load(ALIGN / "disagreement_maps.npy", mmap_mode="r")[idx]
valid = np.load(ALIGN / "valid_temporal_mask.npy", mmap_mode="r")[idx].astype(bool)

geometry = pd.read_csv(ALIGN / "temporal_geometry.csv")
geom = geometry.set_index("cohort_index").loc[COHORT]
width = float(geom["image_width"])
x0 = float(geom["waveform_x0"])
x1 = float(geom["waveform_x1"])

u = np.arange(600)
x_original = (u + 0.5) * width / 600 - 0.5
time = 10 * (x_original - x0) / (x1 - x0)

assert np.allclose(
    a[valid], np.sqrt(g1[valid] * g2[valid]), atol=1e-5
)
assert np.allclose(
    d[valid], np.abs(g1[valid] - g2[valid]), atol=1e-5
)

mean_a = float(np.mean(a[valid]))
mean_d = float(np.mean(d[valid]))

# Normalize original Grad-CAM for visualization
raw1 = np.asarray(raw1, dtype=float)
raw2 = np.asarray(raw2, dtype=float)
raw1 = raw1 / (np.max(raw1) + 1e-8)
raw2 = raw2 / (np.max(raw2) + 1e-8)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(13, 17))
gs = fig.add_gridspec(
    5, 1,
    height_ratios=[1.3, 1.7, 0.9, 1.2, 1.2],
    hspace=0.55
)

# A: Raw ECG + 1D attribution
ax = fig.add_subplot(gs[0])
ax.plot(t_ecg, ecg, color="#183153", lw=0.7)
ax.scatter(
    t_ecg[::8], ecg[::8],
    c=raw1[::8], cmap="Reds",
    s=8, vmin=0, vmax=1, alpha=0.8
)
ax.set_title("A | Original 1D ECG with 1D Grad-CAM", loc="left", weight="bold")
ax.set_ylabel("ECG amplitude\n(dataset units)")
ax.set_xlim(0, 10)
ax.grid(alpha=0.12)

# B: Original ECG image + 2D Grad-CAM
ax = fig.add_subplot(gs[1])
ax.imshow(image, aspect="auto")
ax.imshow(
    raw2,
    extent=(0, image.shape[1], image.shape[0], 0),
    cmap="jet", alpha=0.43, vmin=0, vmax=1,
    aspect="auto"
)
ax.set_title("B | Original 2D ECG with 2D Grad-CAM", loc="left", weight="bold")
ax.set_xticks([])
ax.set_yticks([])

# C: Explain mapping with actual geometry
ax = fig.add_subplot(gs[2])
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.axhline(0.65, color="#E83E8C", lw=5, alpha=0.7)
ax.axhline(0.35, color="#168BDF", lw=5, alpha=0.7)
ax.text(0.1, 0.79, "1D Grad-CAM: 5000 samples", fontsize=11)
ax.text(0.1, 0.46, "2D projected Grad-CAM: 600 columns", fontsize=11)
ax.set_title(
    "C | Temporal Alignment onto a Common ECG Timeline",
    loc="left", weight="bold"
)
ax.set_xlabel("ECG time (seconds)")
ax.set_yticks([])
ax.grid(axis="x", alpha=0.15)

# D: Aligned attribution curves
ax = fig.add_subplot(gs[3])
ax.plot(time[valid], g1[valid], color="#E83E8C",
        lw=1.3, label="1D Grad-CAM")
ax.plot(time[valid], g2[valid], color="#168BDF",
        lw=1.5, label="2D projected Grad-CAM")
ax.set_title("D | Aligned Attribution Profiles", loc="left", weight="bold")
ax.set_ylabel("Normalized attribution")
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 10)
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.15)

# E: Agreement and disagreement
ax = fig.add_subplot(gs[4])
ax.plot(time[valid], a[valid], color="#B91C3C",
        lw=1.4, label="Agreement")
ax.plot(time[valid], d[valid], color="#174A8B",
        lw=1.4, label="Disagreement")
ax.set_title(
    "E | Local Agreement and Disagreement",
    loc="left", weight="bold"
)
ax.set_ylabel("Local score")
ax.set_xlabel("ECG time (seconds)")
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 10)
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.15)

fig.suptitle(
    "Cross-Representation Grad-CAM Analysis | AFib Cohort 209",
    fontsize=17, weight="bold", y=0.995
)

fig.text(
    0.5, 0.015,
    f"Whole-recording mean agreement = {mean_a:.4f}   |   "
    f"Mean disagreement = {mean_d:.4f}",
    ha="center", fontsize=12, weight="bold"
)

png = OUT / "Figure5_AFib209_vertical.png"
pdf = OUT / "Figure5_AFib209_vertical.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Saved:", png)
print("Saved:", pdf)
print(f"Mean agreement: {mean_a:.4f}")
print(f"Mean disagreement: {mean_d:.4f}")
