from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter1d
from scipy.stats import spearmanr

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"
OUT = BASE / "projection_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
).reset_index(drop=True)

cam2 = np.load(
    BASE / "gradcam_full/gradcam_2d.npy",
    mmap_mode="r"
)

g1 = np.load(
    BASE / "alignment_0px/gradcam_1d_aligned.npy",
    mmap_mode="r"
)

valid = np.load(
    BASE / "alignment_0px/valid_temporal_mask.npy",
    mmap_mode="r"
)

baseline = np.load(
    BASE / "alignment_0px/gradcam_2d_projected.npy",
    mmap_mode="r"
)

assert len(metadata) == 1054
assert cam2.shape == (1054, 300, 600)
assert g1.shape == valid.shape == baseline.shape == (1054, 600)
assert metadata["cohort_index"].is_unique

EPS = 1e-8
TOP_FRACTION = 0.15
SIZES = [6, 13, 24]

def normalize(values):
    x = np.asarray(values, dtype=np.float64)
    difference = np.ptp(x)

    if difference < EPS:
        return np.zeros_like(x), True

    return (x - x.min()) / difference, False

def similarity(x, y):
    n = len(x)
    k = max(1, int(np.ceil(n * TOP_FRACTION)))

    top1 = set(np.argsort(-x, kind="stable")[:k])
    top2 = set(np.argsort(-y, kind="stable")[:k])

    intersection = len(top1 & top2)
    union = len(top1 | top2)

    return {
        "spearman": float(spearmanr(x, y).statistic),
        "dice_top15": intersection / k,
        "iou_top15": intersection / union,
        "weighted_jaccard": float(
            np.minimum(x, y).sum()
            / (np.maximum(x, y).sum() + EPS)
        )
    }

records = []
baseline_max_error = 0.0

for i, row in metadata.iterrows():

    image_path = IMAGE_ROOT / row["image_relative_path"]

    with Image.open(image_path) as im:
        rgb = np.asarray(
            im.convert("RGB"),
            dtype=np.int16
        )

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    mask_image = Image.fromarray(
        ink.astype(np.uint8) * 255,
        mode="L"
    )

    mask_image = mask_image.resize(
        (600, 300),
        Image.Resampling.BILINEAR
    )

    base_mask = np.asarray(mask_image) > 14

    temporal_mask = valid[i].astype(bool)
    x, bad1 = normalize(g1[i][temporal_mask])

    for size in SIZES:

        mask = maximum_filter1d(
            base_mask,
            size=size,
            axis=0
        )

        mask[:, ~temporal_mask] = False

        count = mask.sum(axis=0)
        supported = (count > 0) & temporal_mask

        projected = np.zeros(600, dtype=np.float32)

        projected[supported] = (
            (cam2[i] * mask).sum(axis=0)[supported]
            / count[supported]
        )

        if size == 13:
            error = float(
                np.max(np.abs(projected - baseline[i]))
            )
            baseline_max_error = max(
                baseline_max_error, error
            )

        y, bad2 = normalize(projected[temporal_mask])

        agreement = np.sqrt(x * y)
        disagreement = np.abs(x - y)

        if bad1 or bad2:
            sim = {
                "spearman": np.nan,
                "dice_top15": np.nan,
                "iou_top15": np.nan,
                "weighted_jaccard": np.nan
            }
        else:
            sim = similarity(x, y)

        records.append({
            "cohort_index": int(row["cohort_index"]),
            "class_name": row["class_name"],
            "mask_size": size,
            "mean_agreement": float(agreement.mean()),
            "mean_disagreement": float(disagreement.mean()),
            "degenerate_1d": bool(bad1),
            "degenerate_2d": bool(bad2),
            "supported_columns": int(supported.sum()),
            **sim
        })

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/1054",
            flush=True
        )

df = pd.DataFrame(records)

assert len(df) == 1054 * 3
assert baseline_max_error < 1e-5, (
    f"Baseline projection mismatch: {baseline_max_error}"
)

df.to_csv(
    OUT / "projection_metrics.csv",
    index=False
)

METRICS = [
    "mean_agreement",
    "mean_disagreement",
    "spearman",
    "dice_top15",
    "iou_top15",
    "weighted_jaccard"
]

summary = (
    df.groupby("mask_size")[METRICS]
    .mean()
    .reset_index()
)

summary.to_csv(
    OUT / "projection_summary.csv",
    index=False
)

print("\nPROJECTION SENSITIVITY RESULTS")
print(summary.round(5).to_string(index=False))
print("\nMaximum baseline reconstruction error:",
      baseline_max_error)
print("\nSaved to:", OUT)
