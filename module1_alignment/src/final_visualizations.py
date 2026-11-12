from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
OUT = BASE / "final_figures"
OUT.mkdir(parents=True, exist_ok=True)

metrics = pd.read_csv(
    BASE / "alignment_0px/agreement_metrics.csv"
)

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
)

subgroups = pd.read_csv(
    BASE / "subgroup_analysis/subgroup_manifest.csv"
)

g1 = np.load(
    BASE / "alignment_0px/normalized_1d.npy",
    mmap_mode="r"
)
g2 = np.load(
    BASE / "alignment_0px/normalized_2d.npy",
    mmap_mode="r"
)
agreement = np.load(
    BASE / "alignment_0px/agreement_maps.npy",
    mmap_mode="r"
)
disagreement = np.load(
    BASE / "alignment_0px/disagreement_maps.npy",
    mmap_mode="r"
)
valid = np.load(
    BASE / "alignment_0px/valid_temporal_mask.npy",
    mmap_mode="r"
)

assert len(metrics) == len(metadata) == 1054
assert all(
    a.shape == (1054, 600)
    for a in [g1, g2, agreement, disagreement, valid]
)

assert metadata["cohort_index"].is_unique
assert metrics["cohort_index"].is_unique

# Match every metric to its original array row.
metrics = metadata[["cohort_index"]].merge(
    metrics,
    on="cohort_index",
    validate="one_to_one"
)

valid_metrics = metrics[
    ~(metrics["degenerate_1d"] |
      metrics["degenerate_2d"])
].copy()

CLASSES = ["AFib", "GSVT", "SB", "SR"]

# --------------------------------------------------
# FIGURE 1: CLASS-WISE BOXPLOTS
# --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric, title in [
    (axes[0], "mean_agreement", "Agreement"),
    (axes[1], "mean_disagreement", "Disagreement"),
]:
    values = [
        valid_metrics.loc[
            valid_metrics["class_name"] == cls, metric
        ].dropna().values
        for cls in CLASSES
    ]

    ax.boxplot(values, tick_labels=CLASSES)
    ax.set_title(f"Class-wise {title}")
    ax.set_ylabel(f"Mean {title.lower()}")
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "classwise_agreement_disagreement.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    OUT / "classwise_agreement_disagreement.pdf",
    bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------
# FIGURE 2: AGREEMENT VS DISAGREEMENT
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for cls in CLASSES:
    subset = valid_metrics[
        valid_metrics["class_name"] == cls
    ]

    ax.scatter(
        subset["mean_agreement"],
        subset["mean_disagreement"],
        s=16,
        alpha=0.55,
        label=cls
    )

ax.set_xlabel("Mean agreement")
ax.set_ylabel("Mean disagreement")
ax.set_title("Agreement vs. Disagreement Across ECGs")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(
    OUT / "agreement_vs_disagreement.png",
    dpi=300, bbox_inches="tight"
)
fig.savefig(
    OUT / "agreement_vs_disagreement.pdf",
    bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------
# FIGURE 3: REPRESENTATIVE ECG EXPLANATIONS
# --------------------------------------------------
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"

representatives = []

for cls in CLASSES:
    subset = valid_metrics[
        valid_metrics["class_name"] == cls
    ]

    high_agreement = subset.loc[
        subset["mean_agreement"].idxmax()
    ]

    high_disagreement = subset.loc[
        subset["mean_disagreement"].idxmax()
    ]

    for label, record in [
        ("high_agreement", high_agreement),
        ("high_disagreement", high_disagreement),
    ]:
        cohort = int(record["cohort_index"])

        idx = metadata.index[
            metadata["cohort_index"] == cohort
        ][0]

        filename = metadata.loc[
            idx, "ecg_csv_filename"
        ]

        ecg = pd.read_csv(ECG_ROOT / filename)
        signal = ecg["II"].to_numpy(dtype=float)

        if signal.shape != (5000,):
            raise ValueError(f"Invalid ECG: {filename}")

        # Use saved waveform boundaries to map valid
        # CAM columns to approximate ECG time.
        geometry = pd.read_csv(
            BASE / "alignment_0px/temporal_geometry.csv"
        )
        geom = geometry.set_index("cohort_index").loc[cohort]

        width = int(geom["image_width"])
        x0 = float(geom["waveform_x0"])
        x1 = float(geom["waveform_x1"])

        x_original = (
            (np.arange(600) + 0.5) * width / 600 - 0.5
        )
        cam_time = 10 * (x_original - x0) / (x1 - x0)

        mask = valid[idx].astype(bool)
        t_signal = np.arange(5000) / 500.0

        fig, axes = plt.subplots(
            4, 1, figsize=(13, 10),
            sharex=True
        )

        axes[0].plot(t_signal, signal, linewidth=0.7)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(
            f"{cls} | Cohort {cohort} | {label.replace('_', ' ')}"
        )

        axes[1].plot(
            cam_time[mask], g1[idx][mask],
            label="1D Grad-CAM"
        )
        axes[1].plot(
            cam_time[mask], g2[idx][mask],
            label="2D projected Grad-CAM"
        )
        axes[1].set_ylabel("Attribution")
        axes[1].legend()

        axes[2].fill_between(
            cam_time[mask],
            agreement[idx][mask],
            alpha=0.5
        )
        axes[2].set_ylabel("Agreement")

        axes[3].fill_between(
            cam_time[mask],
            disagreement[idx][mask],
            alpha=0.5
        )
        axes[3].set_ylabel("Disagreement")
        axes[3].set_xlabel("ECG time (seconds)")

        for ax in axes:
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 10)

        fig.tight_layout()

        stem = f"{cls}_{label}_cohort_{cohort}"
        fig.savefig(
            OUT / f"{stem}.png",
            dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            OUT / f"{stem}.pdf",
            bbox_inches="tight"
        )
        plt.close(fig)

        representatives.append({
            "class_name": cls,
            "selection": label,
            "cohort_index": cohort,
            "mean_agreement": record["mean_agreement"],
            "mean_disagreement": record["mean_disagreement"],
        })

pd.DataFrame(representatives).to_csv(
    OUT / "representative_ecgs.csv",
    index=False
)

print("Final visualizations generated.")
print("Output:", OUT)
print("\nSelected representative ECGs:")
print(pd.DataFrame(representatives).to_string(index=False))
