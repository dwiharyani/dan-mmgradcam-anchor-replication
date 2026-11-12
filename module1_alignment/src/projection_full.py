from pathlib import Path
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter1d

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results" / os.environ.get(
    "ALIGNMENT_OUTPUT", "alignment_0px"
)
PILOT.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

cam2 = np.load(ROOT / "module1_alignment/results/gradcam_full/gradcam_2d.npy", mmap_mode="r")
valid = np.load(PILOT / "valid_temporal_mask.npy")
metadata = pd.read_csv(ROOT / "module1_alignment/results/gradcam_full/gradcam_metadata.csv")

assert cam2.shape == (1054, 300, 600)
assert valid.shape == (1054, 600)

projected = []
masks = []

for i, row in metadata.iterrows():

    path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Detect the blue ECG waveform.
    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    # Resize binary trace support to the CAM resolution.
    mask_image = Image.fromarray(
        ink.astype(np.uint8) * 255,
        mode="L"
    )

    mask_image = mask_image.resize(
        (600, 300),
        Image.Resampling.BILINEAR
    )

    mask = np.asarray(mask_image) > 14

    # Expand vertically to accommodate broad 2D Grad-CAM.
    mask = maximum_filter1d(
        mask,
        size=13,
        axis=0
    )

    # Exclude columns outside the aligned ECG interval.
    mask[:, ~valid[i]] = False

    count = mask.sum(axis=0)

    g2 = np.zeros(600, dtype=np.float32)

    supported = (count > 0) & valid[i]

    g2[supported] = (
        (cam2[i] * mask).sum(axis=0)[supported]
        / count[supported]
    )

    if not supported[valid[i]].all():
        print(
            f"WARNING: {row['class_name']} has "
            f"{np.sum(valid[i] & ~supported)} unsupported columns"
        )

    projected.append(g2)
    masks.append(mask)

    print(
        f"{row['class_name']} | "
        f"supported columns={supported.sum()} | "
        f"projected max={g2.max():.4f}",
        flush=True
    )

projected = np.stack(projected)
masks = np.stack(masks)

assert projected.shape == (1054, 600)
assert masks.shape == (1054, 300, 600)
assert np.isfinite(projected).all()

np.save(PILOT / "gradcam_2d_projected.npy", projected)
np.save(PILOT / "waveform_masks.npy", masks)

print("\n2D-to-1D projection completed.")
print("Projected CAM shape:", projected.shape)
print("Waveform mask shape:", masks.shape)
print("Saved to:", PILOT)
