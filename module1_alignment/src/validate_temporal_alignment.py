from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results/gradcam_pilot"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

OUT = PILOT / "temporal_validation"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(PILOT / "gradcam_metadata.csv")
geometry = pd.read_csv(PILOT / "temporal_geometry.csv")

for _, row in metadata.iterrows():
    cohort = int(row["cohort_index"])

    geom = geometry.loc[
        geometry["cohort_index"] == cohort
    ].iloc[0]

    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])

    signal = pd.read_csv(
        ECG_ROOT / row["ecg_csv_filename"]
    )["II"].to_numpy(dtype=np.float32)

    with Image.open(
        IMAGE_ROOT / row["image_relative_path"]
    ) as im:
        image = np.asarray(im.convert("RGB"))

    assert signal.shape == (5000,)
    assert np.isfinite(signal).all()

    height, width = image.shape[:2]

    # Map ECG sample positions to image horizontal coordinates.
    x = x0 + np.arange(5000) * (x1 - x0) / 4999

    # Normalize vertical signal position for visual comparison.
    # This is NOT a calibrated amplitude-to-pixel mapping.
    lo, hi = np.percentile(signal, [1, 99])

    if hi <= lo:
        raise ValueError(f"Invalid signal range: {cohort}")

    y = height * (
        0.85 - 0.70 * (signal - lo) / (hi - lo)
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6),
        layout="constrained"
    )

    axes[0].imshow(image)
    axes[0].set_title("Original 2D ECG image")
    axes[0].set_xlim(0, width)

    axes[1].imshow(image)
    axes[1].plot(
        x, y, color="red", linewidth=0.7, alpha=0.7
    )
    axes[1].set_title(
        "Original 1D ECG overlaid using provisional time mapping"
    )
    axes[1].set_xlim(0, width)

    for ax in axes:
        ax.set_ylim(height, 0)
        ax.set_xlabel("Original image x-coordinate (pixels)")

    fig.suptitle(
        f"{row['class_name']} | Cohort {cohort}\n"
        "Qualitative temporal check — vertical scale is not calibrated"
    )

    path = OUT / f"{row['class_name']}_cohort_{cohort}_validation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print("Saved:", path, flush=True)

print("\nQualitative temporal validation figures generated.")
