from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results/gradcam_pilot"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

OUT = ROOT / "module1_alignment/results/temporal_validation_100"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(ROOT / "module1_alignment/data/temporal_validation_100.csv")
# Waveform boundaries are detected independently for each image.
geometry = None

MAX_SHIFT_MS = 200
FS = 500


def extract_image_trace(image, x0, x1):
    """Extract a continuous approximate trace from blue image pixels."""
    rgb = image.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    width = ink.shape[1]
    trace = np.full(width, np.nan, dtype=np.float64)

    previous_y = None

    for x in range(x0, x1 + 1):
        ys = np.flatnonzero(ink[:, x])

        if len(ys) == 0:
            continue

        if previous_y is None:
            chosen = float(np.median(ys))
        else:
            chosen = float(ys[np.argmin(np.abs(ys - previous_y))])

        trace[x] = chosen
        previous_y = chosen

    valid = np.flatnonzero(np.isfinite(trace))

    if len(valid) < 0.6 * (x1 - x0 + 1):
        raise ValueError("Insufficient detected waveform pixels")

    x = np.arange(x0, x1 + 1)

    filled = np.interp(x, valid, trace[valid])
    filled = median_filter(filled, size=3)

    # Image y-axis points downward.
    return x, -filled


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    std = x.std()

    if std < 1e-8:
        raise ValueError("Constant waveform")

    return (x - x.mean()) / std


def shifted_correlation(a, b, shift):
    """Correlation over overlapping samples, without wraparound."""
    if shift > 0:
        aa, bb = a[:-shift], b[shift:]
    elif shift < 0:
        aa, bb = a[-shift:], b[:shift]
    else:
        aa, bb = a, b

    if len(aa) < 100:
        return np.nan

    return float(pearsonr(aa, bb).statistic)


records = []

for _, row in metadata.iterrows():
    cohort = int(row["cohort_index"])

    # Detect waveform boundaries from the original ECG image.
    with Image.open(
        IMAGE_ROOT / row["image_relative_path"]
    ) as im:
        rgba = np.asarray(im.convert("RGBA"))

    rgb = rgba[..., :3].astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    columns = np.flatnonzero(ink.any(axis=0))

    if len(columns) < 0.6 * rgba.shape[1]:
        raise ValueError(f"Insufficient waveform: cohort {cohort}")

    x0 = int(columns[0])
    x1 = int(columns[-1])

    signal = pd.read_csv(
        ECG_ROOT / row["ecg_csv_filename"]
    )["II"].to_numpy(dtype=np.float64)

    assert signal.shape == (5000,)
    assert np.isfinite(signal).all()

    with Image.open(
        IMAGE_ROOT / row["image_relative_path"]
    ) as im:
        image = np.asarray(im.convert("RGB"))

    x_pixels, image_trace = extract_image_trace(
        image, x0, x1
    )

    # Current assumed image-to-time mapping.
    t_image = 10.0 * (x_pixels - x0) / (x1 - x0)

    # Sample the raw ECG at the corresponding image times.
    t_signal = np.arange(5000) / FS

    signal_on_image = np.interp(
        t_image, t_signal, signal
    )

    a = standardize(signal_on_image)
    b = standardize(image_trace)

    zero_corr = shifted_correlation(a, b, 0)

    ms_per_pixel = 10000.0 / (x1 - x0)
    max_shift_px = int(round(MAX_SHIFT_MS / ms_per_pixel))

    shifts = np.arange(-max_shift_px, max_shift_px + 1)

    correlations = np.array([
        shifted_correlation(a, b, int(s))
        for s in shifts
    ])

    best_index = int(np.nanargmax(correlations))
    best_shift = int(shifts[best_index])
    best_corr = float(correlations[best_index])
    best_shift_ms = best_shift * ms_per_pixel

    records.append({
        "cohort_index": cohort,
        "class_name": row["class_name"],
        "zero_shift_correlation": zero_corr,
        "best_correlation": best_corr,
        "best_shift_pixels": best_shift,
        "best_shift_ms": best_shift_ms,
        "correlation_improvement": best_corr - zero_corr,
    })

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 7),
        layout="constrained"
    )

    axes[0].plot(
        t_image, a,
        label="Original 1D ECG",
        linewidth=1
    )

    axes[0].plot(
        t_image, b,
        label="Extracted 2D waveform",
        linewidth=1,
        alpha=0.75
    )

    axes[0].set_title("Waveform correspondence at current mapping")
    axes[0].set_xlabel("ECG time (seconds)")
    axes[0].set_ylabel("Standardized amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        shifts * ms_per_pixel,
        correlations
    )

    axes[1].axvline(
        0, color="gray", linestyle="--",
        label="Current mapping"
    )

    axes[1].axvline(
        best_shift_ms,
        color="red", linestyle=":",
        label=f"Best shift: {best_shift_ms:.1f} ms"
    )

    axes[1].set_title("Correlation versus horizontal time shift")
    axes[1].set_xlabel("Horizontal shift (ms)")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"{row['class_name']} | Cohort {cohort}\n"
        f"Zero-shift r={zero_corr:.3f} | "
        f"Best r={best_corr:.3f} | "
        f"Shift={best_shift_ms:.1f} ms\n"
        "Exploratory trace-extraction validation"
    )

    filename = OUT / (
        f"{row['class_name']}_cohort_{cohort}_correspondence.png"
    )

    fig.savefig(filename, dpi=160)
    plt.close(fig)

    print(
        f"{row['class_name']} | "
        f"zero r={zero_corr:.4f} | "
        f"best r={best_corr:.4f} | "
        f"shift={best_shift_ms:.2f} ms",
        flush=True
    )

results = pd.DataFrame(records)

results.to_csv(
    OUT / "signal_image_correspondence.csv",
    index=False
)

print("\nSIGNAL-IMAGE CORRESPONDENCE RESULTS")
print(results.round(4).to_string(index=False))
print("\nSaved to:", OUT)
