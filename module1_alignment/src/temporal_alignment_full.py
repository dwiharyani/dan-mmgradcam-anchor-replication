from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results" / os.environ.get(
    "ALIGNMENT_OUTPUT", "alignment_0px"
)
PILOT.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

cam1 = np.load(ROOT / "module1_alignment/results/gradcam_full/gradcam_1d.npy", mmap_mode="r")
cam2 = np.load(ROOT / "module1_alignment/results/gradcam_full/gradcam_2d.npy", mmap_mode="r")
metadata = pd.read_csv(ROOT / "module1_alignment/results/gradcam_full/gradcam_metadata.csv")

assert cam1.shape == (1054, 5000)
assert cam2.shape == (1054, 300, 600)
assert len(metadata) == 1054

aligned_maps = []
valid_masks = []
geometry = []

for i, row in metadata.iterrows():

    path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"))

    height, width = rgba.shape[:2]
    rgb = rgba[:, :, :3].astype(np.int16)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    columns = np.flatnonzero(ink.any(axis=0))

    if len(columns) < width * 0.6:
        raise ValueError(
            f"Insufficient waveform detected: {path}"
        )

    x0 = int(columns[0])
    x1 = int(columns[-1])

    # Convert 600 CAM column centers to original image coordinates.
    target_width = cam2.shape[2]

    x_original = (
        (np.arange(target_width) + 0.5)
        * width / target_width - 0.5
    )

    valid = (x_original >= x0) & (x_original <= x1)

    # Map image x-coordinate to 1D ECG sample index.
    shift = int(os.environ.get("ALIGNMENT_SHIFT", "0"))
    fraction = (x_original - shift - x0) / (x1 - x0)
    sample_positions = np.clip(fraction, 0, 1) * 4999

    aligned = np.interp(
        sample_positions,
        np.arange(5000),
        cam1[i]
    ).astype(np.float32)

    aligned[~valid] = 0

    aligned_maps.append(aligned)
    valid_masks.append(valid)

    geometry.append({
        "cohort_index": int(row["cohort_index"]),
        "class_name": row["class_name"],
        "image_width": width,
        "image_height": height,
        "waveform_x0": x0,
        "waveform_x1": x1,
        "valid_columns": int(valid.sum())
    })

    print(
        f"{row['class_name']} | cohort {row['cohort_index']} "
        f"| x0={x0}, x1={x1} "
        f"| valid columns={valid.sum()}",
        flush=True
    )

aligned_maps = np.stack(aligned_maps)
valid_masks = np.stack(valid_masks)

assert aligned_maps.shape == (1054, 600)
assert valid_masks.shape == (1054, 600)
assert np.isfinite(aligned_maps).all()

np.save(PILOT / "gradcam_1d_aligned.npy", aligned_maps)
np.save(PILOT / "valid_temporal_mask.npy", valid_masks)

pd.DataFrame(geometry).to_csv(
    PILOT / "temporal_geometry.csv",
    index=False
)

print("\nTemporal alignment completed.")
print("Aligned 1D shape:", aligned_maps.shape)
print("Valid mask shape:", valid_masks.shape)
print("Results saved to:", PILOT)
print("NOTE: Alignment remains provisional until time-axis calibration.")
