from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import find_peaks, correlate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results/gradcam_pilot"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
IMAGE_ROOT = Path.home() / "XAI/imageData-Label2/label1/Testdata"

OUT = PILOT / "quantitative_validation"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(PILOT / "gradcam_metadata.csv")
geometry = pd.read_csv(PILOT / "temporal_geometry.csv")

FS = 500
MIN_PEAK_DISTANCE = 0.25
MATCH_TOLERANCE_MS = 100


def detect_signal_peaks(signal):
    """Detect prominent QRS candidates from absolute baseline deviations."""
    baseline = pd.Series(signal).rolling(
        window=151, center=True, min_periods=1
    ).median().to_numpy()

    activity = np.abs(signal - baseline)

    peaks, _ = find_peaks(
        activity,
        distance=int(FS * MIN_PEAK_DISTANCE),
        prominence=max(float(np.percentile(activity, 90) * 0.5), 1e-6)
    )

    return peaks


def detect_image_peaks(image, x0, x1):
    """Detect QRS candidates from vertical waveform excursions."""
    rgb = image[..., :3].astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ink = (
        (b - r >= 22) &
        (g - r >= 14) &
        (b - g >= 12)
    )

    h, w = ink.shape
    ys = np.arange(h)

    activity = np.zeros(w, dtype=np.float64)

    # Use vertical extent of waveform ink at each column.
    for x in range(x0, x1 + 1):
        yy = ys[ink[:, x]]
        if len(yy):
            activity[x] = np.max(np.abs(yy - np.median(yy)))

    # Fill empty columns and smooth locally.
    activity = pd.Series(activity).rolling(
        window=7, center=True, min_periods=1
    ).max().to_numpy()

    peaks, _ = find_peaks(
        activity,
        distance=int((x1 - x0) * MIN_PEAK_DISTANCE / 10),
        prominence=max(float(np.percentile(activity[x0:x1+1], 85) * 0.3), 1e-6)
    )

    return peaks[(peaks >= x0) & (peaks <= x1)]


def match_peaks(t1, t2, tolerance=0.1):
    """Greedy one-to-one matching of nearest peaks within tolerance."""
    candidates = []

    for i, a in enumerate(t1):
        for j, b in enumerate(t2):
            delta = abs(float(a - b))
            if delta <= tolerance:
                candidates.append((delta, i, j))

    candidates.sort()

    used1, used2 = set(), set()
    matches = []

    for delta, i, j in candidates:
        if i not in used1 and j not in used2:
            used1.add(i)
            used2.add(j)
            matches.append((i, j, delta))

    return matches


records = []

for _, row in metadata.iterrows():
    cohort = int(row["cohort_index"])
    geom = geometry.loc[
        geometry["cohort_index"] == cohort
    ].iloc[0]

    x0 = int(geom["waveform_x0"])
    x1 = int(geom["waveform_x1"])

    ecg = pd.read_csv(ECG_ROOT / row["ecg_csv_filename"])
    signal = ecg["II"].to_numpy(dtype=np.float32)

    with Image.open(IMAGE_ROOT / row["image_relative_path"]) as im:
        image = np.asarray(im.convert("RGB"))

    p1 = detect_signal_peaks(signal)
    p2 = detect_image_peaks(image, x0, x1)

    t1 = p1 / FS
    t2 = 10.0 * (p2 - x0) / (x1 - x0)

    matches = match_peaks(t1, t2, MATCH_TOLERANCE_MS / 1000)

    errors = np.array([m[2] * 1000 for m in matches])

    if len(errors):
        median = float(np.median(errors))
        p95 = float(np.percentile(errors, 95))
        maximum = float(errors.max())
    else:
        median = p95 = maximum = float("nan")

    match_fraction = (
        len(matches) / max(len(p1), len(p2), 1)
    )

    records.append({
        "cohort_index": cohort,
        "class_name": row["class_name"],
        "signal_peaks": len(p1),
        "image_peaks": len(p2),
        "matched_peaks": len(matches),
        "match_fraction": match_fraction,
        "median_error_ms": median,
        "p95_error_ms": p95,
        "max_error_ms": maximum,
    })

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(np.arange(len(signal)) / FS, signal, lw=0.7,
            color="#34495e", label="Original Lead-II ECG")

    ax.scatter(t1, signal[p1], color="#c05270", s=24,
               label="1D detected peaks", zorder=3)

    for j, t in enumerate(t2):
        ax.axvline(t, color="#277faf", alpha=0.3, lw=0.8)

    ax.set_xlim(0, 10)
    ax.set_xlabel("ECG time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"{row['class_name']} | Cohort {cohort} | "
        f"Matched {len(matches)} peaks | "
        f"Median error {median:.1f} ms"
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(
        OUT / f"{row['class_name']}_cohort_{cohort}_peaks.png",
        dpi=150
    )
    plt.close(fig)

    print(
        f"{row['class_name']} | "
        f"1D peaks={len(p1)} | 2D peaks={len(p2)} | "
        f"matched={len(matches)} | "
        f"median={median:.2f} ms | p95={p95:.2f} ms",
        flush=True
    )

results = pd.DataFrame(records)
results.to_csv(OUT / "temporal_validation_metrics.csv", index=False)

print("\nQUANTITATIVE TEMPORAL VALIDATION")
print(results.round(3).to_string(index=False))
print("\nSaved to:", OUT)
print("NOTE: Automated peak correspondence is heuristic, not ground truth.")
