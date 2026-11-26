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

meta = pd.read_csv(GRAD / "gradcam_metadata.csv")
matches = np.flatnonzero(meta["cohort_index"].to_numpy() == COHORT)
assert len(matches) == 1
idx = int(matches[0])
row = meta.iloc[idx]

ecg = pd.read_csv(
    ECG_ROOT / row["ecg_csv_filename"]
)["II"].to_numpy(dtype=float)
image = np.asarray(
    Image.open(IMAGE_ROOT / row["image_relative_path"]).convert("RGB")
)

raw1 = np.asarray(
    np.load(GRAD / "gradcam_1d.npy", mmap_mode="r")[idx], dtype=float
)
raw2 = np.asarray(
    np.load(GRAD / "gradcam_2d.npy", mmap_mode="r")[idx], dtype=float
)

g1 = np.asarray(np.load(ALIGN / "normalized_1d.npy", mmap_mode="r")[idx])
g2 = np.asarray(np.load(ALIGN / "normalized_2d.npy", mmap_mode="r")[idx])
a = np.asarray(np.load(ALIGN / "agreement_maps.npy", mmap_mode="r")[idx])
d = np.asarray(np.load(ALIGN / "disagreement_maps.npy", mmap_mode="r")[idx])
valid = np.asarray(
    np.load(ALIGN / "valid_temporal_mask.npy", mmap_mode="r")[idx], dtype=bool
)

geom = pd.read_csv(ALIGN / "temporal_geometry.csv").set_index("cohort_index").loc[COHORT]
width = float(geom["image_width"])
x0 = float(geom["waveform_x0"])
x1 = float(geom["waveform_x1"])

u = np.arange(600)
x_original = (u + 0.5) * width / 600 - 0.5
time = 10.0 * (x_original - x0) / (x1 - x0)
t_ecg = np.arange(5000) / 500.0

assert raw1.shape == (5000,)
assert raw2.shape == (300, 600)
assert np.allclose(a[valid], np.sqrt(g1[valid] * g2[valid]), atol=1e-5)
assert np.allclose(d[valid], np.abs(g1[valid] - g2[valid]), atol=1e-5)

mean_a = float(a[valid].mean())
mean_d = float(d[valid].mean())

# Display normalization only; saved arrays remain unchanged.
raw1 = raw1 / (raw1.max() + 1e-8)
raw2 = raw2 / (raw2.max() + 1e-8)

# Select fixed-duration windows from the actual saved maps.
window = 36  # approximately 0.6 seconds
candidates = []
for s in range(len(u) - window + 1):
    e = s + window
    if not valid[s:e].all():
        continue
    candidates.append((s, e, float(a[s:e].mean()), float(d[s:e].mean())))

assert candidates

# Shared attribution is rewarded; disagreement is penalized.
agree = max(candidates, key=lambda z: z[2] - z[3])

# Disagreement interval must not overlap the agreement interval.
eligible = [
    z for z in candidates
    if z[1] <= agree[0] or z[0] >= agree[1]
]
assert eligible
disagree = max(eligible, key=lambda z: z[3])

# Do not call an interval "high agreement" unless both branches
# actually show substantial shared attribution.
strong_agreement = agree[2] >= 0.25 and agree[3] <= 0.25

regions = [
    (agree, "#E9A6BC", "Shared attribution" if strong_agreement
     else "Relative agreement"),
    (disagree, "#A5D6F5", "High disagreement"),
]

def shade(ax):
    for reg, color, _ in regions:
        start, end = time[reg[0]], time[reg[1]-1]
        ax.axvspan(start, end, color=color, alpha=0.28, zorder=0)
        ax.axvline(start, color=color, ls="--", lw=0.8)
        ax.axvline(end, color=color, ls="--", lw=0.8)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(12.5, 15.5), constrained_layout=True)
gs = fig.add_gridspec(
    5, 1,
    height_ratios=[1.3, 1.6, 1.3, 1.35, 1.35]
)

# A — ECG plus 1D attribution strip
sub = gs[0].subgridspec(2, 1, height_ratios=[4, 0.65], hspace=0.05)
axA = fig.add_subplot(sub[0])
axA.plot(t_ecg, ecg, color="#17375E", lw=0.75)
shade(axA)
axA.set_title("A  |  1D ECG signal and Grad-CAM", loc="left", weight="bold")
axA.set_ylabel("ECG amplitude\n(dataset units)")
axA.set_xlim(0, 10)
axA.tick_params(labelbottom=False)

axStrip = fig.add_subplot(sub[1], sharex=axA)
axStrip.imshow(
    raw1[None, :], aspect="auto",
    extent=[0, 10, 0, 1], cmap="Reds", vmin=0, vmax=1
)
axStrip.set_yticks([])
axStrip.set_xlabel("ECG time (s)")
axStrip.set_ylabel("1D CAM", rotation=0, labelpad=30, va="center")

# B — 2D ECG with spatial Grad-CAM
axB = fig.add_subplot(gs[1])
axB.imshow(image, aspect="auto")
im = axB.imshow(
    raw2, extent=[0, image.shape[1], image.shape[0], 0],
    cmap="turbo", vmin=0, vmax=1, alpha=0.45, aspect="auto"
)
axB.set_title("B  |  2D ECG image and spatial Grad-CAM", loc="left", weight="bold")
axB.set_xticks([])
axB.set_yticks([])
fig.colorbar(im, ax=axB, fraction=0.025, pad=0.01,
             label="Normalized 2D Grad-CAM")

# C — Actual mapping geometry and projected 2D attribution
axC = fig.add_subplot(gs[2])
axC.plot(time[valid], g2[valid], color="#168BDF", lw=1.6)
shade(axC)
axC.set_xlim(0, 10)
axC.set_ylim(0, 1.05)
axC.set_ylabel("Projected 2D\nattribution")
axC.set_xlabel("Common ECG time (s)")
axC.set_title(
    "C  |  Waveform-guided 2D projection onto the ECG timeline",
    loc="left", weight="bold"
)
axC.text(
    0.01, 0.96,
    f"Original waveform columns: {int(x0)}–{int(x1)}"
    "  |  Common grid: 600 columns",
    transform=axC.transAxes, va="top", fontsize=9,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

# D — Aligned comparison
axD = fig.add_subplot(gs[3])
axD.plot(time[valid], g1[valid], color="#E83E8C", lw=1.3,
         label="1D Grad-CAM")
axD.plot(time[valid], g2[valid], color="#168BDF", lw=1.5,
         label="2D projected Grad-CAM")
shade(axD)
axD.set_xlim(0, 10)
axD.set_ylim(0, 1.05)
axD.set_ylabel("Normalized attribution")
axD.set_title("D  |  Temporally aligned attribution profiles",
              loc="left", weight="bold")
axD.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)

# E — Agreement and disagreement
axE = fig.add_subplot(gs[4])
axE.plot(time[valid], a[valid], color="#B91C3C", lw=1.4,
         label="Agreement")
axE.plot(time[valid], d[valid], color="#174A8B", lw=1.4,
         label="Disagreement")
shade(axE)
axE.set_xlim(0, 10)
axE.set_ylim(0, 1.05)
axE.set_ylabel("Local score")
axE.set_xlabel("ECG time (s)")
axE.set_title("E  |  Agreement and disagreement",
              loc="left", weight="bold")
axE.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)

for ax in [axA, axC, axD, axE]:
    ax.grid(alpha=0.12)

fig.suptitle(
    "Cross-Representation Grad-CAM | AFib Cohort 209",
    fontsize=16, weight="bold"
)

def interval_text(reg, label):
    return (
        f"{label}: {time[reg[0]]:.2f}–{time[reg[1]-1]:.2f} s"
        f"  (A={reg[2]:.3f}, D={reg[3]:.3f})"
    )

print(interval_text(agree, regions[0][2]))
print(interval_text(disagree, regions[1][2]))
print(f"Whole recording: A={mean_a:.4f}, D={mean_d:.4f}")

# Export the selected intervals for verification and reporting.
pd.DataFrame([
    {
        "region": label,
        "start_s": time[reg[0]],
        "end_s": time[reg[1]-1],
        "mean_agreement": reg[2],
        "mean_disagreement": reg[3],
    }
    for reg, _, label in regions
]).to_csv(OUT / "Figure5_AFib209_regions.csv", index=False)

fig.savefig(OUT / "Figure5_AFib209_vertical_v2.png",
            dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Figure5_AFib209_vertical_v2.pdf",
            bbox_inches="tight")
plt.close(fig)

print("Figure saved successfully.")
