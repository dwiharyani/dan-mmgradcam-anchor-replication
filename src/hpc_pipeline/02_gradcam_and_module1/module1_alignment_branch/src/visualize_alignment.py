from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
PILOT = ROOT / "module1_alignment/results/gradcam_pilot"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
OUT = PILOT / "alignment_figures"
OUT.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(PILOT / "gradcam_metadata.csv")
metrics = pd.read_csv(PILOT / "agreement_metrics.csv")

g1 = np.load(PILOT / "normalized_1d.npy")
g2 = np.load(PILOT / "normalized_2d.npy")
agreement = np.load(PILOT / "agreement_maps.npy")
disagreement = np.load(PILOT / "disagreement_maps.npy")
valid = np.load(PILOT / "valid_temporal_mask.npy").astype(bool)

assert g1.shape == g2.shape == (4, 600)
assert agreement.shape == disagreement.shape == (4, 600)
assert valid.shape == (4, 600)

for i, row in metadata.iterrows():

    # Original ECG waveform
    ecg_path = ECG_ROOT / row["ecg_csv_filename"]
    ecg = pd.read_csv(ecg_path)
    signal = ecg["II"].to_numpy(dtype=np.float32)

    assert signal.shape == (5000,)
    assert np.isfinite(signal).all()

    # Original ECG time
    t_signal = np.arange(5000) / 500.0

    # CAM column centers mapped to provisional ECG time
    geometry = pd.read_csv(PILOT / "temporal_geometry.csv")
    geom = geometry.loc[
        geometry["cohort_index"] == row["cohort_index"]
    ].iloc[0]

    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])
    image_width = float(geom["image_width"])

    x_original = (
        (np.arange(600) + 0.5) * image_width / 600 - 0.5
    )

    t_cam = 10.0 * (x_original - x0) / (x1 - x0)
    mask = valid[i]

    result = metrics.loc[
        metrics["cohort_index"] == row["cohort_index"]
    ].iloc[0]

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1, 1, 1]},
        layout="constrained"
    )

    # Panel 1: Original ECG
    axes[0].plot(
        t_signal, signal,
        color="#34495e", linewidth=0.8
    )
    axes[0].set_title("Original Lead-II ECG")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.2)

    # Panel 2: Aligned explanations
    axes[1].plot(
        t_cam[mask], g1[i, mask],
        label="1D Grad-CAM",
        color="#c05270",
        linewidth=1.5
    )

    axes[1].plot(
        t_cam[mask], g2[i, mask],
        label="2D Grad-CAM projected",
        color="#277faf",
        linewidth=1.5
    )

    axes[1].set_title("Aligned 1D and 2D Grad-CAM")
    axes[1].set_ylabel("Normalized attribution")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    # Panel 3: Agreement
    axes[2].plot(
        t_cam[mask], agreement[i, mask],
        color="#2e8b57",
        linewidth=1.5
    )
    axes[2].fill_between(
        t_cam[mask], agreement[i, mask],
        color="#2e8b57", alpha=0.25
    )
    axes[2].set_title("Agreement A(u)")
    axes[2].set_ylabel("Agreement")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.2)

    # Panel 4: Disagreement
    axes[3].plot(
        t_cam[mask], disagreement[i, mask],
        color="#c0392b",
        linewidth=1.5
    )
    axes[3].fill_between(
        t_cam[mask], disagreement[i, mask],
        color="#c0392b", alpha=0.25
    )
    axes[3].set_title("Disagreement D(u)")
    axes[3].set_ylabel("Disagreement")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel(
        "ECG time (seconds) — provisional alignment"
    )
    axes[3].grid(alpha=0.2)

    for ax in axes:
        ax.set_xlim(0, 10)

    fig.suptitle(
        f"Module 1 | {row['class_name']} | "
        f"Cohort {row['cohort_index']}\n"
        f"Spearman={result['spearman']:.3f} | "
        f"Dice={result['dice_top15']:.3f} | "
        f"IoU={result['iou_top15']:.3f} | "
        f"Weighted Jaccard={result['weighted_jaccard']:.3f}\n"
        "EXPLORATORY — Temporal calibration not yet validated",
        fontsize=12
    )

    filename = OUT / (
        f"{row['class_name']}_cohort_"
        f"{row['cohort_index']}_alignment.png"
    )

    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", filename, flush=True)

print("\nAll alignment figures generated successfully.")
