from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results/gradcam_pilot"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"
OUTPUT = PILOT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

cam1 = np.load(PILOT / "gradcam_1d.npy")
cam2 = np.load(PILOT / "gradcam_2d.npy")
metadata = pd.read_csv(PILOT / "gradcam_metadata.csv")

assert len(metadata) == len(cam1) == len(cam2)

for i, row in metadata.iterrows():
    image_path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(image_path) as im:
        image = im.convert("L").resize((600, 300))
        image = np.asarray(image)

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9),
        layout="constrained"
    )

    time = np.linspace(0, 10, 5000)

    axes[0].plot(time, cam1[i])
    axes[0].set_title("1D Grad-CAM")
    axes[0].set_xlabel("ECG time (seconds)")
    axes[0].set_ylabel("Importance")
    axes[0].set_xlim(0, 10)

    axes[1].imshow(image, cmap="gray", aspect="auto")
    axes[1].set_title("Original 2D ECG Image")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray", aspect="auto")
    axes[2].imshow(
        cam2[i], cmap="jet", alpha=0.45,
        vmin=0, vmax=1, aspect="auto"
    )
    axes[2].set_title("2D Grad-CAM Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{row['class_name']} | Cohort {row['cohort_index']}",
        fontsize=14
    )

    filename = OUTPUT / (
        f"{row['class_name']}_cohort_{row['cohort_index']}.png"
    )

    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print("Saved:", filename)

print("\nAll pilot figures generated.")
