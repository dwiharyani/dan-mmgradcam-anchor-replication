from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
FULL = BASE / "gradcam_full"
ZERO = BASE / "alignment_0px"
OUT = BASE / "gradcam_shift_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(FULL / "gradcam_metadata.csv")
cam1 = np.load(FULL / "gradcam_1d.npy", mmap_mode="r")
g2 = np.load(ZERO / "gradcam_2d_projected.npy", mmap_mode="r")
valid = np.load(ZERO / "valid_temporal_mask.npy", mmap_mode="r")
geometry = pd.read_csv(ZERO / "temporal_geometry.csv")

assert len(metadata) == 1054
assert cam1.shape == (1054, 5000)
assert g2.shape == valid.shape == (1054, 600)

assert metadata["cohort_index"].is_unique
assert geometry["cohort_index"].is_unique

geometry = geometry.set_index("cohort_index").loc[
    metadata["cohort_index"]
].reset_index()

EPS = 1e-8
SHIFTS = range(-5, 6)
TOP_FRACTION = 0.15

def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if np.ptp(x) < EPS:
        return np.zeros_like(x), True
    return (x - x.min()) / np.ptp(x), False

def metrics(x, y):
    n = len(x)
    k = max(1, int(np.ceil(n * TOP_FRACTION)))

    top1 = set(np.argsort(-x, kind="stable")[:k])
    top2 = set(np.argsort(-y, kind="stable")[:k])

    intersection = len(top1 & top2)
    union = len(top1 | top2)

    return {
        "spearman": float(spearmanr(x, y).statistic),
        "dice_top15": 2 * intersection / (2 * k),
        "iou_top15": intersection / union,
        "weighted_jaccard": float(
            np.minimum(x, y).sum() /
            (np.maximum(x, y).sum() + EPS)
        )
    }

records = []

for i, row in metadata.iterrows():
    cohort = int(row["cohort_index"])
    geom = geometry.iloc[i]

    width = int(geom["image_width"])
    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])

    x_original = (
        (np.arange(600) + 0.5) * width / 600 - 0.5
    )

    mask = valid[i].astype(bool)
    y, bad2 = normalize(g2[i][mask])

    for shift in SHIFTS:
        fraction = (
            x_original - shift - x0
        ) / (x1 - x0)

        sample_positions = np.clip(
            fraction, 0, 1
        ) * 4999

        aligned = np.interp(
            sample_positions,
            np.arange(5000),
            cam1[i]
        )

        x, bad1 = normalize(aligned[mask])

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
            sim = metrics(x, y)

        records.append({
            "cohort_index": cohort,
            "class_name": row["class_name"],
            "shift_pixels": shift,
            "mean_agreement": float(agreement.mean()),
            "mean_disagreement": float(disagreement.mean()),
            "degenerate_1d": bool(bad1),
            "degenerate_2d": bool(bad2),
            **sim
        })

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/1054", flush=True)

df = pd.DataFrame(records)

df.to_csv(OUT / "shift_metrics.csv", index=False)

summary = df.groupby("shift_pixels")[
    [
        "mean_agreement",
        "mean_disagreement",
        "spearman",
        "dice_top15",
        "iou_top15",
        "weighted_jaccard"
    ]
].mean()

summary.to_csv(OUT / "shift_summary.csv")

print("\nGRAD-CAM SHIFT SENSITIVITY RESULTS")
print(summary.round(4).to_string())
print("\nSaved to:", OUT)
