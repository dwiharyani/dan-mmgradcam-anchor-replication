from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
OUT = BASE / "publication_figures/gradcam_heatmaps"
OUT.mkdir(parents=True, exist_ok=True)

ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
).reset_index(drop=True)

cam1 = np.load(
    BASE / "gradcam_full/gradcam_1d.npy",
    mmap_mode="r"
)

cam2 = np.load(
    BASE / "gradcam_full/gradcam_2d.npy",
    mmap_mode="r"
)

assert cam1.shape == (1054, 5000)
assert cam2.shape == (1054, 300, 600)
assert metadata["cohort_index"].is_unique

REPRESENTATIVES = {
    "AFib": 209,
    "GSVT": 109,
    "SB": 1,
    "SR": 264
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white"
})

def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

for cls, cohort in REPRESENTATIVES.items():

    matches = metadata.index[
        metadata["cohort_index"] == cohort
    ]

    if len(matches) != 1:
        raise ValueError(f"Missing cohort {cohort}")

    idx = int(matches[0])
    row = metadata.loc[idx]

    assert row["class_name"] == cls

    signal = pd.read_csv(
        ECG_ROOT / row["ecg_csv_filename"]
    )["II"].to_numpy(dtype=float)

    assert signal.shape == (5000,)

    image_path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))

    h, w = image.shape[:2]

    attribution1 = normalize(cam1[idx])
    attribution2 = normalize(cam2[idx])

    t = np.arange(5000) / 500.0

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 13),
        gridspec_kw={
            "height_ratios": [1, 1.3, 1.5, 1.5]
        }
    )

    # ----------------------------------------
    # 1. ORIGINAL 1D ECG
    # ----------------------------------------
    axes[0].plot(
        t, signal,
        color="#344054",
        linewidth=0.8
    )
    axes[0].set_title("Original 1D ECG Signal")
    axes[0].set_xlim(0, 10)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].grid(alpha=0.15)

    # ----------------------------------------
    # 2. 1D GRAD-CAM OVERLAY
    # ----------------------------------------
    points = np.array([
        t, signal
    ]).T.reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]],
        axis=1
    )

    collection = LineCollection(
        segments,
        cmap="jet",
        norm=Normalize(0, 1)
    )

    collection.set_array(
        (attribution1[:-1] + attribution1[1:]) / 2
    )
    collection.set_linewidth(2.2)

    axes[1].add_collection(collection)
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(signal.min(), signal.max())
    axes[1].set_title("1D Grad-CAM Highlighted ECG")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(alpha=0.15)

    fig.colorbar(
        collection,
        ax=axes[1],
        label="Normalized 1D Grad-CAM importance"
    )

    # ----------------------------------------
    # 3. ORIGINAL 2D ECG IMAGE
    # ----------------------------------------
    axes[2].imshow(image)
    axes[2].set_title("Original 2D ECG Image")
    axes[2].axis("off")

    # ----------------------------------------
    # 4. 2D GRAD-CAM HEATMAP OVERLAY
    # ----------------------------------------
    axes[3].imshow(image)

    heatmap = axes[3].imshow(
        attribution2,
        cmap="jet",
        alpha=0.45,
        interpolation="bilinear",
        extent=(0, w, h, 0),
        vmin=0,
        vmax=1
    )

    axes[3].set_xlim(0, w)
    axes[3].set_ylim(h, 0)
    axes[3].set_title("2D Grad-CAM Heatmap Overlay")
    axes[3].axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[3],
        label="Normalized 2D Grad-CAM importance"
    )

    fig.suptitle(
        f"{cls} | Cohort {cohort} | Grad-CAM Explanations",
        fontsize=17,
        fontweight="bold"
    )

    fig.tight_layout()

    filename = f"{cls}_cohort_{cohort}_gradcam_heatmaps"

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

    print(f"Generated: {filename}", flush=True)

print("\nALL FOUR GRAD-CAM HEATMAP FIGURES GENERATED")
print("Saved to:", OUT)
