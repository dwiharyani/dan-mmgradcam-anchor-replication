from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results" / os.environ.get(
    "ALIGNMENT_OUTPUT", "alignment_0px"
)
PILOT.mkdir(parents=True, exist_ok=True)

g1 = np.load(PILOT / "gradcam_1d_aligned.npy")
g2 = np.load(PILOT / "gradcam_2d_projected.npy")
valid = np.load(PILOT / "valid_temporal_mask.npy")
metadata = pd.read_csv(ROOT / "module1_alignment/results/gradcam_full/gradcam_metadata.csv")

assert g1.shape == g2.shape == valid.shape == (1054, 600)

EPS = 1e-8
TOP_FRACTION = 0.15


def normalize(values, mask):
    """Min-max normalization over valid ECG columns."""
    x = np.asarray(values, dtype=np.float64)
    selected = x[mask]

    lo = selected.min()
    hi = selected.max()

    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)

    output = np.zeros_like(x, dtype=np.float32)
    output[mask] = ((selected - lo) / (hi - lo)).astype(np.float32)
    return output


def calculate_metrics(x, y, fraction=0.15):
    """Compare normalized temporal attributions."""
    n = len(x)
    k = max(1, int(np.ceil(n * fraction)))

    top1 = set(np.argsort(-x, kind="stable")[:k])
    top2 = set(np.argsort(-y, kind="stable")[:k])

    intersection = len(top1 & top2)
    union = len(top1 | top2)

    dice = 2 * intersection / (len(top1) + len(top2))
    iou = intersection / union

    weighted_jaccard = (
        np.minimum(x, y).sum()
        / (np.maximum(x, y).sum() + EPS)
    )

    spearman = spearmanr(x, y).statistic

    return {
        "spearman": float(spearman),
        "dice_top15": float(dice),
        "iou_top15": float(iou),
        "weighted_jaccard": float(weighted_jaccard),
    }


agreement_maps = []
disagreement_maps = []
normalized_1d = []
normalized_2d = []
records = []

for i, row in metadata.iterrows():

    mask = valid[i].astype(bool)

    x = normalize(g1[i], mask)
    y = normalize(g2[i], mask)

    degenerate_1d = np.ptp(g1[i][mask]) < EPS
    degenerate_2d = np.ptp(g2[i][mask]) < EPS

    agreement = np.zeros(600, dtype=np.float32)
    disagreement = np.zeros(600, dtype=np.float32)

    agreement[mask] = np.sqrt(x[mask] * y[mask])
    disagreement[mask] = np.abs(x[mask] - y[mask])

    if degenerate_1d or degenerate_2d:
        metrics = {
            "spearman": np.nan,
            "dice_top15": np.nan,
            "iou_top15": np.nan,
            "weighted_jaccard": np.nan,
        }
    else:
        metrics = calculate_metrics(x[mask], y[mask])

    records.append({
        "cohort_index": int(row["cohort_index"]),
        "class_name": row["class_name"],
        "mean_agreement": float(agreement[mask].mean()),
        "mean_disagreement": float(disagreement[mask].mean()),
        **metrics,
        "degenerate_1d": bool(degenerate_1d),
        "degenerate_2d": bool(degenerate_2d),
        "valid_similarity": bool(
            not degenerate_1d and not degenerate_2d
        ),
    })

    normalized_1d.append(x)
    normalized_2d.append(y)
    agreement_maps.append(agreement)
    disagreement_maps.append(disagreement)

    print(
        f"{row['class_name']} | "
        f"Spearman={metrics['spearman']:.4f} | "
        f"Dice={metrics['dice_top15']:.4f} | "
        f"IoU={metrics['iou_top15']:.4f}",
        flush=True
    )

np.save(PILOT / "normalized_1d.npy", np.stack(normalized_1d))
np.save(PILOT / "normalized_2d.npy", np.stack(normalized_2d))
np.save(PILOT / "agreement_maps.npy", np.stack(agreement_maps))
np.save(PILOT / "disagreement_maps.npy", np.stack(disagreement_maps))

results = pd.DataFrame(records)
results.to_csv(PILOT / "agreement_metrics.csv", index=False)

print("\nMODULE 1 — EXPLORATORY AGREEMENT RESULTS")
print(results.round(4).to_string(index=False))

print("\nSaved to:", PILOT)
print("NOTE: Temporal calibration remains provisional.")
