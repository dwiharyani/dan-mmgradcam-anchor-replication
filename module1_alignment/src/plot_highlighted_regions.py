from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path.home() / "Alignment-XAI"
BASE = ROOT / "module1_alignment/results"
ALIGN = BASE / "alignment_0px"
OUT = BASE / "publication_figures/highlighted_regions"
ECG_ROOT = Path.home() / "XAI/1D-ECG-Signal"
OUT.mkdir(parents=True, exist_ok=True)

RECORDS = {"AFib": 209, "GSVT": 109, "SB": 1, "SR": 264}

# Region selection settings
WINDOW_SECONDS = 0.6
MIN_AGREEMENT = 0.25
MAX_AGREEMENT_REGION_DISAGREEMENT = 0.25

COLORS = {
    "ecg": "#183153",
    "1d": "#E83E8C",
    "2d": "#168BDF",
    "agreement": "#B91C3C",
    "disagreement": "#174A8B",
}

metadata = pd.read_csv(
    BASE / "gradcam_full/gradcam_metadata.csv"
).reset_index(drop=True)

geometry = pd.read_csv(
    ALIGN / "temporal_geometry.csv"
).set_index("cohort_index")

metrics = pd.read_csv(
    ALIGN / "agreement_metrics.csv"
).set_index("cohort_index")

g1 = np.load(ALIGN / "normalized_1d.npy", mmap_mode="r")
g2 = np.load(ALIGN / "normalized_2d.npy", mmap_mode="r")
agreement = np.load(ALIGN / "agreement_maps.npy", mmap_mode="r")
disagreement = np.load(ALIGN / "disagreement_maps.npy", mmap_mode="r")
valid = np.load(ALIGN / "valid_temporal_mask.npy", mmap_mode="r")

assert len(metadata) == len(g1) == len(g2) == 1054
assert g1.shape == g2.shape == agreement.shape == disagreement.shape
assert g1.shape[1] == 600
assert metadata["cohort_index"].is_unique

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

def find_regions(t, a, d, mask):
    """
    Select fixed-duration, nonoverlapping windows.
    Agreement requires high A and relatively low D.
    Disagreement maximizes mean D.
    """
    step = np.median(np.diff(t[mask]))
    nwin = max(3, int(round(WINDOW_SECONDS / step)))
    candidates = []

    for start in range(len(t) - nwin + 1):
        end = start + nwin
        if not np.all(mask[start:end]):
            continue

        candidates.append({
            "start": start,
            "end": end,
            "a": float(np.mean(a[start:end])),
            "d": float(np.mean(d[start:end])),
        })

    if not candidates:
        raise ValueError("No valid temporal windows found")

    # Prefer genuine shared-high-attribution regions.
    eligible = [
        x for x in candidates
        if x["a"] >= MIN_AGREEMENT
        and x["d"] <= MAX_AGREEMENT_REGION_DISAGREEMENT
    ]

    if eligible:
        best_a = max(eligible, key=lambda x: (x["a"], -x["d"]))
        strong_agreement = True
    else:
        # Fallback is labeled as relative agreement,
        # not falsely claimed to be strong agreement.
        best_a = max(candidates, key=lambda x: (x["a"] - x["d"], x["a"]))
        strong_agreement = False

    # Exclude windows overlapping the agreement interval.
    remaining = [
        x for x in candidates
        if x["end"] <= best_a["start"]
        or x["start"] >= best_a["end"]
    ]

    if not remaining:
        raise ValueError("No nonoverlapping disagreement window")

    best_d = max(remaining, key=lambda x: x["d"])
    return best_a, best_d, strong_agreement


def plot_record(cls, cohort):
    matches = metadata.index[
        metadata["cohort_index"] == cohort
    ]

    if len(matches) != 1:
        raise ValueError(f"Missing cohort {cohort}")

    idx = int(matches[0])
    row = metadata.iloc[idx]

    assert row["class_name"] == cls

    ecg_path = ECG_ROOT / row["ecg_csv_filename"]
    signal = pd.read_csv(ecg_path)["II"].to_numpy(dtype=float)
    assert signal.shape == (5000,)

    t_ecg = np.arange(5000) / 500.0

    geom = geometry.loc[cohort]
    width = float(geom["image_width"])
    x0 = float(geom["waveform_x0"])
    x1 = float(geom["waveform_x1"])

    x_original = (
        (np.arange(600) + 0.5) * width / 600 - 0.5
    )
    t = 10.0 * (x_original - x0) / (x1 - x0)

    mask = np.asarray(valid[idx], dtype=bool)
    a = np.asarray(agreement[idx], dtype=float)
    d = np.asarray(disagreement[idx], dtype=float)
    y1 = np.asarray(g1[idx], dtype=float)
    y2 = np.asarray(g2[idx], dtype=float)

    assert np.allclose(
        a[mask], np.sqrt(np.clip(y1[mask] * y2[mask], 0, None)),
        atol=1e-5
    )
    assert np.allclose(
        d[mask], np.abs(y1[mask] - y2[mask]),
        atol=1e-5
    )

    reg_a, reg_d, strong = find_regions(t, a, d, mask)

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10.5),
        gridspec_kw={"height_ratios": [1.0, 1.2, 1.2]},
        sharex=True,
        layout="constrained"
    )

    fig.suptitle(
        f"{cls} | Cohort {cohort} — Paired ECG Attribution",
        fontsize=17, fontweight="bold"
    )

    # Panel A: Original ECG
    axes[0].plot(
        t_ecg, signal, color=COLORS["ecg"], lw=0.75
    )
    axes[0].set_title(
        "A | Original 1D ECG Signal (Lead II)",
        loc="left", fontweight="bold"
    )
    axes[0].set_ylabel("ECG amplitude\n(dataset units)")

    # Panel B: Both attribution profiles
    axes[1].plot(
        t[mask], y1[mask],
        color=COLORS["1d"], lw=1.2,
        label="1D CNN Grad-CAM"
    )
    axes[1].plot(
        t[mask], y2[mask],
        color=COLORS["2d"], lw=1.4,
        label="2D CNN Grad-CAM (projected)"
    )
    axes[1].set_title(
        "B | Grad-CAM Attribution by Branch",
        loc="left", fontweight="bold"
    )
    axes[1].set_ylabel("Normalized\nattribution")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel C: Derived agreement and disagreement
    axes[2].plot(
        t[mask], a[mask],
        color=COLORS["agreement"], lw=1.3,
        label="Agreement A(u)"
    )
    axes[2].plot(
        t[mask], d[mask],
        color=COLORS["disagreement"], lw=1.4,
        label="Disagreement D(u)"
    )
    axes[2].set_title(
        "C | Local Agreement and Disagreement",
        loc="left", fontweight="bold"
    )
    axes[2].set_ylabel("Local score")
    axes[2].set_xlabel("Approximate ECG time (seconds)")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].legend(loc="upper left", fontsize=9)

    regions = [
        (reg_a, "#F6B4C8", "Agreement candidate"),
        (reg_d, "#B5DDF6", "Disagreement"),
    ]

    for ax in axes:
        for region, color, _ in regions:
            left = t[region["start"]]
            right = t[region["end"] - 1]
            ax.axvspan(left, right, color=color, alpha=0.45)
            ax.axvline(left, color=color, ls="--", lw=1)
            ax.axvline(right, color=color, ls="--", lw=1)
        ax.set_xlim(0, 10)
        ax.grid(alpha=0.15)

    full_a = float(a[mask].mean())
    full_d = float(d[mask].mean())

    # Cross-check published recording-level scores.
    saved = metrics.loc[cohort]
    assert np.isclose(
        full_a, float(saved["mean_agreement"]), atol=1e-4
    )
    assert np.isclose(
        full_d, float(saved["mean_disagreement"]), atol=1e-4
    )

    def description(region):
        return (
            f"{t[region['start']]:.2f}–"
            f"{t[region['end']-1]:.2f} s\n"
            f"Mean A = {region['a']:.3f}\n"
            f"Mean D = {region['d']:.3f}"
        )

    label_a = (
        "Shared-high attribution"
        if strong else
        "Relative agreement only (weak shared evidence)"
    )

    fig.text(
        0.17, 0.025,
        f"{label_a}\n{description(reg_a)}",
        ha="center", va="bottom", fontsize=10,
        color="#8C1630"
    )
    fig.text(
        0.51, 0.025,
        f"High disagreement\n{description(reg_d)}",
        ha="center", va="bottom", fontsize=10,
        color="#174A8B"
    )
    fig.text(
        0.84, 0.025,
        f"Whole-recording scores\n"
        f"Mean A = {full_a:.4f}\n"
        f"Mean D = {full_d:.4f}",
        ha="center", va="bottom", fontsize=10,
        fontweight="bold"
    )

    stem = f"{cls}_cohort_{cohort}_highlighted"
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        f"{cls} cohort {cohort}: "
        f"agreement A={reg_a['a']:.4f}, D={reg_a['d']:.4f}; "
        f"disagreement A={reg_d['a']:.4f}, D={reg_d['d']:.4f}; "
        f"strong agreement={strong}",
        flush=True
    )

    return {
        "class_name": cls,
        "cohort_index": cohort,
        "strong_agreement_found": strong,
        "agreement_start_s": t[reg_a["start"]],
        "agreement_end_s": t[reg_a["end"]-1],
        "agreement_region_mean_A": reg_a["a"],
        "agreement_region_mean_D": reg_a["d"],
        "disagreement_start_s": t[reg_d["start"]],
        "disagreement_end_s": t[reg_d["end"]-1],
        "disagreement_region_mean_A": reg_d["a"],
        "disagreement_region_mean_D": reg_d["d"],
        "whole_mean_A": full_a,
        "whole_mean_D": full_d,
    }


records = []
for cls, cohort in RECORDS.items():
    records.append(plot_record(cls, cohort))

pd.DataFrame(records).to_csv(
    OUT / "selected_regions.csv", index=False
)

print("\nSaved figures to:", OUT)
