from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr

ROOT = Path.home() / "Alignment-XAI"
SRC = ROOT / "module1_alignment/src"
sys.path.insert(0, str(SRC))

# Reuse the existing waveform extraction logic without
# importing the previous script, which executes on import.
from scipy.ndimage import median_filter

MANIFEST = ROOT / "module1_alignment/data/temporal_validation_100.csv"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"
OUT = ROOT / "module1_alignment/results/temporal_shift_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

SHIFTS = [-1, 0, 1]
SEGMENTS = {
    "beginning": (0.0, 10.0 / 3.0),
    "middle": (10.0 / 3.0, 20.0 / 3.0),
    "end": (20.0 / 3.0, 10.0),
}


def detect_trace(image):
    rgb = image.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    columns = np.flatnonzero(ink.any(axis=0))

    if len(columns) < 0.6 * image.shape[1]:
        raise ValueError("Insufficient waveform pixels")

    x0, x1 = int(columns[0]), int(columns[-1])
    trace = np.full(image.shape[1], np.nan)
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
        raise ValueError("Insufficient trace coverage")

    x = np.arange(x0, x1 + 1)
    y = np.interp(x, valid, trace[valid])
    y = -median_filter(y, size=3)

    return x, y, x0, x1


def correlation(a, b):
    if len(a) < 30:
        return np.nan
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(pearsonr(a, b).statistic)


manifest = pd.read_csv(MANIFEST)
assert len(manifest) == 100
assert manifest["cohort_index"].is_unique

records = []
errors = []

for _, row in manifest.iterrows():
    cohort = int(row["cohort_index"])

    try:
        ecg = pd.read_csv(
            ECG_ROOT / row["ecg_csv_filename"]
        )
        signal = ecg["II"].to_numpy(dtype=np.float64)

        if signal.shape != (5000,) or not np.isfinite(signal).all():
            raise ValueError("Invalid Lead-II signal")

        with Image.open(
            IMAGE_ROOT / row["image_relative_path"]
        ) as im:
            image = np.asarray(im.convert("RGB"))

        x, image_trace, x0, x1 = detect_trace(image)

        # Keep the same image pixels for every shift.
        # Compare signal values at adjusted image coordinates.
        time = 10.0 * (x - x0) / (x1 - x0)

        for shift in SHIFTS:
            shifted_time = (
                10.0 * (x - shift - x0) / (x1 - x0)
            )

            # Avoid extrapolation outside the ECG interval.
            valid = (
                (shifted_time >= 0) &
                (shifted_time <= 10)
            )

            # Common support across all three shifts.
            valid &= (x >= x0 + 1) & (x <= x1 - 1)

            signal_values = np.interp(
                shifted_time,
                np.arange(5000) / 500.0,
                signal
            )

            for segment, (start, end) in SEGMENTS.items():
                mask = (
                    valid &
                    (time >= start) &
                    (time < end if end < 10 else time <= end)
                )

                r = correlation(
                    signal_values[mask],
                    image_trace[mask]
                )

                records.append({
                    "cohort_index": cohort,
                    "class_name": row["class_name"],
                    "segment": segment,
                    "shift_pixels": shift,
                    "correlation": r,
                })

    except Exception as exc:
        errors.append({
            "cohort_index": cohort,
            "error": str(exc)
        })

    print(f"Processed cohort {cohort}", flush=True)

results = pd.DataFrame(records)
results.to_csv(OUT / "shift_correlations.csv", index=False)

if errors:
    pd.DataFrame(errors).to_csv(
        OUT / "validation_errors.csv", index=False
    )

print("\nCLASS-WISE CORRELATIONS")
summary = results.groupby(
    ["class_name", "shift_pixels"]
)["correlation"].median().unstack()

print(summary.round(4).to_string())

print("\nSEGMENT-WISE CORRELATIONS")
segments = results.groupby(
    ["segment", "shift_pixels"]
)["correlation"].median().unstack()

print(segments.round(4).to_string())

# Determine preferred shift for each ECG and segment.
best = results.loc[
    results.groupby(
        ["cohort_index", "segment"]
    )["correlation"].idxmax()
]

print("\nPREFERRED SHIFT DISTRIBUTION")
print(
    pd.crosstab(
        best["segment"],
        best["shift_pixels"]
    ).to_string()
)

print("\nSuccessful ECGs:",
      results["cohort_index"].nunique())
print("Failed ECGs:", len(errors))
print("Saved to:", OUT)
