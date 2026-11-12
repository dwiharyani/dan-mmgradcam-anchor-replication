from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
OUT = BASE / "publication_figures/scientific_gradcam"
OUT.mkdir(parents=True, exist_ok=True)

ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
).reset_index(drop=True)

geometry = pd.read_csv(
    BASE / "alignment_0px/temporal_geometry.csv"
).set_index("cohort_index")

cam1 = np.load(
    BASE / "gradcam_full/gradcam_1d.npy",
    mmap_mode="r"
)

cam2 = np.load(
    BASE / "gradcam_full/gradcam_2d.npy",
    mmap_mode="r"
)

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

valid = np.load(
    BASE / "alignment_0px/valid_temporal_mask.npy",
    mmap_mode="r"
)

metrics = pd.read_csv(
    BASE / "alignment_0px/agreement_metrics.csv"
).set_index("cohort_index")

REPRESENTATIVES = {
    "AFib": 209,
    "GSVT": 109,
    "SB": 1,
    "SR": 264
}

def normalize(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def save(fig, name):
    fig.savefig(OUT / f"{name}.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf",
                bbox_inches="tight")
    plt.close(fig)

for cls, cohort in REPRESENTATIVES.items():

    idx = metadata.index[
        metadata["cohort_index"] == cohort
    ][0]

    row = metadata.loc[idx]

    signal = pd.read_csv(
        ECG_ROOT / row["ecg_csv_filename"]
    )["II"].to_numpy(dtype=float)

    image_path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))

    h, w = image.shape[:2]

    c1 = normalize(cam1[idx])
    c2 = normalize(cam2[idx])

    geom = geometry.loc[cohort]
    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])

    t = np.arange(5000) / 500.0

    x_original = (
        (np.arange(600) + 0.5) * w / 600 - 0.5
    )

    cam_time = 10 * (x_original - x0) / (x1 - x0)
    mask = valid[idx].astype(bool)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1.2, 1.2, 1.5],
        hspace=0.45,
        wspace=0.25
    )

    # (a) Original signal + 1D Grad-CAM
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(t, signal, color="#344054", lw=0.7)

    threshold = np.quantile(c1, 0.85)
    selected = c1 >= threshold

    ax_a.scatter(
        t[selected], signal[selected],
        c=c1[selected],
        cmap="jet",
        vmin=0, vmax=1,
        s=5
    )
    ax_a.set_title("(a) Signal form: 1D Grad-CAM highlights")
    ax_a.set_xlim(0, 10)
    ax_a.set_xlabel("Time (seconds)")
    ax_a.set_ylabel("ECG amplitude")

    # (b) Original image + 2D Grad-CAM
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(image)
    ax_b.imshow(
        c2, cmap="jet", alpha=0.45,
        extent=(0, w, h, 0),
        vmin=0, vmax=1
    )
    ax_b.set_title("(b) Image form: 2D Grad-CAM overlay")
    ax_b.axis("off")

    # (c) 1D Grad-CAM temporal attribution
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(
        t, c1,
        color="#9B1C31", lw=1
    )
    ax_c.fill_between(
        t, c1,
        color="#9B1C31", alpha=0.18
    )
    ax_c.set_title("(c) 1D Grad-CAM temporal importance")
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel("Time (seconds)")

    # (d) Original spatial 2D Grad-CAM
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.imshow(c2, cmap="jet", vmin=0, vmax=1)
    ax_d.set_title("(d) Original 2D Grad-CAM heatmap")
    ax_d.axis("off")

    # (e) Aligned explanations on common ECG timeline
    ax_e = fig.add_subplot(gs[2, :])

    ax_e.plot(
        cam_time[mask],
        g1[idx][mask],
        color="#D47C9E",
        label="Aligned 1D Grad-CAM",
        lw=1.2
    )
    ax_e.plot(
        cam_time[mask],
        g2[idx][mask],
        color="#79B5D4",
        label="Projected 2D Grad-CAM",
        lw=1.2
    )
    ax_e.plot(
        cam_time[mask],
        agreement[idx][mask],
        color="#9B1C31",
        label="Agreement",
        lw=1.1
    )
    ax_e.plot(
        cam_time[mask],
        disagreement[idx][mask],
        color="#174A8B",
        label="Disagreement",
        lw=1.1
    )

    ax_e.set_xlim(0, 10)
    ax_e.set_ylim(0, 1.05)
    ax_e.set_xlabel("Approximate ECG time (seconds)")
    ax_e.set_ylabel("Normalized attribution")
    ax_e.set_title(
        "(e) Temporally aligned explanations, agreement and disagreement"
    )
    ax_e.legend(ncol=4, fontsize=9)
    ax_e.grid(alpha=0.15)

    m = metrics.loc[cohort]

    fig.suptitle(
        f"{cls} | Cohort {cohort} | "
        f"Agreement={m['mean_agreement']:.4f} | "
        f"Disagreement={m['mean_disagreement']:.4f}",
        fontsize=16,
        fontweight="bold"
    )

    save(fig, f"{cls}_cohort_{cohort}_scientific_gradcam")
    print(f"Generated: {cls} cohort {cohort}", flush=True)

print("\nSCIENTIFIC GRAD-CAM FIGURES GENERATED")
print("Saved to:", OUT)
