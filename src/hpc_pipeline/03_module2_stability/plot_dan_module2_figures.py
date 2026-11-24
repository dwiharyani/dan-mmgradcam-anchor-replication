"""Module 2 analysis and paper-style figures from saved outputs.

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
from module2_common import *   # defines paths, test manifest, models, perturbation/render/Grad-CAM functions

# ---- notebook cell 10 ----
# 11. Analysis of the saved full v2 run (reads files only; no models, no GPU)
TAG = "full_v2"
D = os.path.join(OUT, f"module2_{TAG}")
df = pd.read_csv(f"{D}/noise_stability_records.csv")
z = np.load(f"{D}/noise_stability_maps.npz")

# ---- 0. integrity: files, maps and manifest must line up by cohort_index / filename ----
assert len(df) == len(test) and df.cohort_index.is_unique
assert set(df.cohort_index) == set(test.cohort_index)
assert (z["cohort_index"] == df.cohort_index.to_numpy()).all()
assert (z["filename"] == np.array(df.ecg_csv_filename.tolist())).all()
m = test.set_index("cohort_index").loc[df.cohort_index]
assert (m.ecg_csv_filename.to_numpy() == df.ecg_csv_filename.to_numpy()).all()
assert (m.class_name.to_numpy() == df.class_name.to_numpy()).all()
v = z["valid_600"]
assert np.isnan(z["R"][~v]).all()
print("integrity OK | rows:", len(df), "| records with any NaN R on valid columns:",
      int(np.isnan(np.where(v, z["R"], 0)).any(1).sum()))

# ---- 1. map quality (Module 1-aligned pipeline) ----
print("\nEvaluable 1D / 2D / joint:", (df.status_1d == "evaluable").sum(),
      (df.status_2d == "evaluable").sum(), (df.status_joint == "evaluable").sum())
print("Degenerate perturbed maps 1D / 2D:", int(df.n_degenerate_1d.sum()), int(df.n_degenerate_2d.sum()))
print("Degenerate BASE maps 1D / 2D:", int(df.base_degenerate_1d.sum()), int(df.base_degenerate_2d.sum()))
print("Base mask coverage: min", round(df.base_coverage.min(), 3), "| bad_mask records (<0.75):", int(df.bad_mask.sum()),
      df[df.bad_mask].class_name.value_counts().to_dict())

# ---- 2. PRR by perturbation family (this is what validates the magnitudes) ----
fam = sorted(c for c in df.columns if c.startswith(("prr_1d_", "prr_2d_")))
print("\nPRR by family (overall):")
print(df[fam].mean().round(4).to_string())
for b in ["1d", "2d"]:
    print(f"\nPRR by family x class ({b.upper()}):")
    print(df.groupby("class_name")[[c for c in fam if c.startswith(f"prr_{b}_")]].mean().round(3).to_string())

# ---- 3. PRR distribution ----
for c in ["prr_1d", "prr_2d"]:
    x = df[c]
    print(f"\n{c}: min={x.min():.3f} p5={x.quantile(.05):.3f} median={x.median():.3f} | "
          f"<0.9: {(x < .9).sum()}  <0.5: {(x < .5).sum()}  ==1.0: {(x == 1).sum()}")
print("n_joint_accepted: min", df.n_joint_accepted.min(), "| records with <10:", (df.n_joint_accepted < 10).sum())

# ---- 4. base predictions and branch disagreement ----
print("\nBase accuracy 1D / 2D:", round((df.pred_1d == df.class_name).mean(), 4),
      round((df.pred_2d == df.class_name).mean(), 4))
print("Branch disagreement:", round(df.branch_disagree.mean(), 4))
print(df.groupby("class_name").branch_disagree.mean().round(4).to_string())

# ---- 5. stability scores ----
print("\nCorrelation S1_mean vs S2_mean:", round(df.S1_mean.corr(df.S2_mean), 3))
print(df[["S1_mean", "S2_mean", "H1_mean", "H2_mean", "R_mean"]].describe().round(4).to_string())
print("\nBy class:")
print(df.groupby("class_name")[["prr_1d", "prr_2d", "n_joint_accepted", "S1_mean", "S2_mean", "R_mean"]].mean().round(4))
print("\nStability by whether the branches agree on the base prediction:")
print(df.groupby("branch_disagree")[["S1_mean", "S2_mean", "R_mean"]].mean().round(4).to_string())

# ---- 6. mean maps per class on Module 1's 600-column grid ----
g = np.linspace(0, 10, 600)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), sharex=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for cls in sorted(df.class_name.unique()):
        i = (df.class_name == cls).to_numpy()
        for a, k, t in zip(ax, ["R", "S1", "S2"], ["R(u)", "S_1D(u)", "S_2D(u)"]):
            a.plot(g, np.nanmean(z[k][i], 0), label=cls); a.set_title(t + " - class mean"); a.set_xlabel("time (s)")
ax[0].legend(); plt.tight_layout(); plt.savefig(f"{D}/class_mean_maps.png", dpi=120); plt.show()

# ---- notebook cell 16 ----
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# --- Load saved results (No models, no 40-min run!) ---
TAG = "full_v2"
D = f"results/module2/module2_{TAG}"
df = pd.read_csv(f"{D}/noise_stability_records.csv")
z = np.load(f"{D}/noise_stability_maps.npz")

ECG_ROOT = "data/ECGData"
IMG_ROOT = "data/ImageData-label2/label1"

def load_ecg(fn):
    return pd.read_csv(os.path.join(ECG_ROOT, fn), usecols=["II"])["II"].to_numpy(dtype=np.float64)

# --- Styling ---
CL = ["AFib", "GSVT", "SB", "SR"]  # <--- THIS WAS MISSING
CLS = {"AFib": "#d1658b", "GSVT": "#5599b8", "SB": "#a68a73", "SR": "#9b719c"}
BG = "#FFFBF7"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": "white", "savefig.facecolor": BG})

# --- Helper to map 600-col grid to 5000-sample signal ---
def map_600_to_5000(arr_600):
    return np.interp(np.linspace(0, 1, 5000), np.linspace(0, 1, 600), arr_600)

# Select one representative example per class
selected_indices = {}
for c in CL:
    # Pick the record with median R_mean for this class
    class_df = df[df.class_name == c].sort_values("R_mean")
    idx = class_df.index[len(class_df) // 2]
    selected_indices[c] = idx

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle("Anchor-style ECG alignment gallery", fontsize=26, fontweight='bold', x=0.05, ha='left', y=0.98)
fig.text(0.05, 0.94, "Module 2: Noise Stability & Reliability-weighted attribution R(u)", fontsize=14, color='#7f8c8d')

for ax, cls in zip(axes, ["AFib", "GSVT", "SB"]): # Using 3 classes for layout, add SR if you want 4
    idx = selected_indices[cls]
    row = df.iloc[idx]
    
    x = load_ecg(row.ecg_csv_filename)
    
    # Get Grad-CAM and Stability maps (Module 2 focus)
    base1 = z['base1'][idx] # 1D Grad-CAM
    base2 = z['base2'][idx] # 2D Grad-CAM
    R_map = z['R'][idx]     # Reliability Map R(u)
    
    # Map to 5000 samples
    base1_5000 = map_600_to_5000(base1)
    base2_5000 = gaussian_filter1d(map_600_to_5000(np.nan_to_num(base2)), sigma=20)
    R_5000 = gaussian_filter1d(map_600_to_5000(np.nan_to_num(R_map)), sigma=30)
    
    # Plot raw ECG
    ax.plot(np.linspace(0, 10, 5000), x, color='black', linewidth=1, zorder=2)
    
    # Overlay 2D Grad-CAM (Heatmap)
    masked_heatmap = np.ma.masked_where(base2_5000 < 0.2, base2_5000)
    ax.imshow(masked_heatmap[np.newaxis, :], aspect='auto', cmap='jet', 
              extent=[0, 10, x.min(), x.max()], alpha=0.5, zorder=1)
    
    # Overlay 1D Grad-CAM (Red dots for top 15%)
    thr = np.percentile(base1_5000, 85)
    marker_idx = np.where(base1_5000 > thr)[0]
    if len(marker_idx) > 0:
        ax.scatter(np.linspace(0, 10, 5000)[marker_idx], x[marker_idx], 
                   color='red', s=10, zorder=3, edgecolors='none')
        
    # Add Reliability Strip R(u) at the bottom
    ax_R = ax.inset_axes([0, -0.25, 1, 0.15])
    ax_R.imshow(R_5000[np.newaxis, :], aspect='auto', cmap='magma', 
                extent=[0, 10, 0, 1], vmin=0, vmax=np.nanpercentile(R_5000, 99))
    ax_R.set_yticks([]); ax_R.set_xticks([])
    ax_R.set_ylabel("R(u)", rotation=0, ha='right', va='center', fontsize=10, labelpad=15)
    
    ax.set_title(f"{cls}  ·  PRR 1D: {row.prr_1d:.2f} / 2D: {row.prr_2d:.2f}  ·  Median R: {row.R_mean:.3f}", 
                 loc='left', fontweight='bold', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", fc=CLS[cls], ec="none", alpha=0.8), color='white')
    ax.set_yticks([]); ax.set_xticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
os.makedirs("results/module2/figures_m2", exist_ok=True)
plt.savefig("results/module2/figures_m2/module2_gallery.png", dpi=300, bbox_inches='tight')
plt.show()

# ---- notebook cell 18 ----
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# --- 1. Load Data ---
TAG = "full_v2"
D = f"results/module2/module2_{TAG}"
df = pd.read_csv(f"{D}/noise_stability_records.csv")
z = np.load(f"{D}/noise_stability_maps.npz")

ECG_ROOT = "data/ECGData"
def load_ecg(fn):
    return pd.read_csv(os.path.join(ECG_ROOT, fn), usecols=["II"])["II"].to_numpy(dtype=np.float64)

# --- 2. Styling (Dwi's Exact Palette) ---
BG = "#FFFBF7"       # Soft cream background
INK = "#2D3047"      # Dark slate text
MUTE = "#767A8C"     # Gray text
EDGE = "#E8E0DA"     # Light border
CLS = {"AFib": "#D06B9A", "GSVT": "#4A93B5", "SB": "#A98568", "SR": "#A47A8A"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "white", 
    "savefig.facecolor": BG, "axes.edgecolor": EDGE,
    "text.color": INK, "font.size": 11, "font.family": "sans-serif"
})

def map_600_to_5000(arr_600):
    return np.interp(np.linspace(0, 1, 5000), np.linspace(0, 1, 600), arr_600)

def draw_card(fig, x, y, w, h):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", 
                                  transform=fig.transFigure, fc="white", ec=EDGE, lw=1.2, zorder=0))

# --- 3. Select Records (Median R for each class) ---
classes_to_plot = ["AFib", "GSVT", "SB"]
selected_indices = {}
for c in classes_to_plot:
    class_df = df[df.class_name == c].sort_values("R_mean")
    selected_indices[c] = class_df.index[len(class_df) // 2]

# --- 4. Build the Figure ---
fig = plt.figure(figsize=(16, 10))
fig.text(0.04, 0.94, "Anchor-style ECG alignment gallery", fontsize=26, weight="bold", color=INK)
fig.text(0.04, 0.90, "Module 2: Noise Stability & Reliability-weighted attribution R(u)", fontsize=13, color=MUTE)

for i, cls in enumerate(classes_to_plot):
    idx = selected_indices[cls]
    row = df.iloc[idx]
    x = load_ecg(row.ecg_csv_filename)
    
    # Get maps
    base1 = map_600_to_5000(z['base1'][idx])
    base2 = gaussian_filter1d(map_600_to_5000(np.nan_to_num(z['base2'][idx])), sigma=15)
    R_map = gaussian_filter1d(map_600_to_5000(np.nan_to_num(z['R'][idx])), sigma=25)
    
    # Calculate card position
    y0 = 0.68 - i * 0.23
    draw_card(fig, 0.04, y0, 0.92, 0.20)
    
    # Add Class Badge
    fig.text(0.06, y0 + 0.16, cls, color="white", weight="bold", fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.4", fc=CLS[cls], ec="none"))
    fig.text(0.14, y0 + 0.16, "Paired ECG  ·  1D markers + 2D CAM", fontsize=11, color=INK)
    
    # Add Plot Axis inside the card
    ax = fig.add_axes([0.10, y0 + 0.02, 0.82, 0.12])
    
    # Plot raw ECG
    ax.plot(np.linspace(0, 10, 5000), x, color=INK, linewidth=0.8, zorder=3)
    
    # Overlay 2D Heatmap (smooth and subtle)
    masked = np.ma.masked_where(base2 < 0.15, base2)
    ax.imshow(masked[np.newaxis, :], aspect='auto', cmap='Spectral_r', 
              extent=[0, 10, x.min(), x.max()], alpha=0.6, zorder=2)
    
    # Overlay 1D Red Dots (Top 15%)
    thr = np.percentile(base1, 85)
    marker_idx = np.where(base1 > thr)[0]
    ax.scatter(np.linspace(0, 10, 5000)[marker_idx], x[marker_idx], 
               color="#E0182D", s=8, zorder=4, edgecolors='none')
    
    # Clean up axes
    ax.set_xlim(0, 10)
    ax.set_ylim(x.min(), x.max())
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    
    # Add R(u) strip below the ECG
    ax_R = fig.add_axes([0.10, y0 + 0.02, 0.82, 0.025])
    ax_R.imshow(R_map[np.newaxis, :], aspect='auto', cmap='magma', 
                extent=[0, 10, 0, 1], vmin=0, vmax=np.nanpercentile(R_map, 99))
    ax_R.set_xticks([]); ax_R.set_yticks([])
    for spine in ax_R.spines.values(): spine.set_visible(False)
    fig.text(0.06, y0 + 0.02, "R(u)", color=MUTE, fontsize=10, weight="bold")



os.makedirs("results/module2/figures_m2", exist_ok=True)
plt.savefig("results/module2/figures_m2/module_2.png", dpi=300, bbox_inches='tight')
plt.close()

# ---- notebook cell 20 ----
import matplotlib.patches as patches

# Create figure
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#f8f9fa')

# --- Header ---
fig.text(0.05, 0.92, "DAN / MMGradCAM", color='#d1658b', fontweight='bold', fontsize=14)
fig.text(0.05, 0.88, "02 | Noise Stability & Reliability Overview", fontweight='bold', fontsize=28, color='#2c3e50')
fig.text(0.05, 0.84, f"{len(df):,} held-out ECG records  ·  30 perturbations each (scale, drift, noise)  ·  Module 2 focus", 
         fontsize=14, color='#7f8c8d')

# --- Top KPI Boxes ---
box_style = dict(boxstyle="round,pad=0.4", fc="white", ec="#dcdde1", lw=1.5)
fig.text(0.05, 0.75, f"RECORDS\n{len(df)}", fontsize=20, fontweight='bold', color='#d1658b', bbox=box_style, va='top')
fig.text(0.25, 0.75, f"MEAN PRR (1D)\n{df.prr_1d.mean():.3f}", fontsize=20, fontweight='bold', color='#5599b8', bbox=box_style, va='top')
fig.text(0.45, 0.75, f"MEAN PRR (2D)\n{df.prr_2d.mean():.3f}", fontsize=20, fontweight='bold', color='#a68a73', bbox=box_style, va='top')
fig.text(0.65, 0.75, f"MEDIAN S (1D)\n{df.S1_mean.median():.3f}", fontsize=20, fontweight='bold', color='#9b719c', bbox=box_style, va='top')
fig.text(0.85, 0.75, f"MEDIAN R(u)\n{df.R_mean.median():.3f}", fontsize=20, fontweight='bold', color='#2c3e50', bbox=box_style, va='top')

# --- Left Bar Chart (PRR by Perturbation) ---
ax1 = fig.add_axes([0.05, 0.15, 0.4, 0.45])
fams = ['scale', 'drift_0.05', 'drift_0.1', 'noise_0.05', 'noise_0.1']
labels = ['Scale', 'Drift 5%', 'Drift 10%', 'Noise 5%', 'Noise 10%']
prr_1d = [df[f'prr_1d_{f}'].mean() for f in fams]
prr_2d = [df[f'prr_2d_{f}'].mean() for f in fams]

x = np.arange(len(fams))
width = 0.35
ax1.bar(x - width/2, prr_1d, width, label='1D Model', color='#5599b8')
ax1.bar(x + width/2, prr_2d, width, label='2D Model', color='#d1658b')

ax1.set_title("Prediction Retention Rate (PRR) by Perturbation", fontweight='bold', loc='left', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0.8, 1.0)
ax1.legend(frameon=False)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# --- Right Boxplots (Stability by Rhythm) ---
ax2 = fig.add_axes([0.55, 0.15, 0.4, 0.45])
rhythms = ['AFib', 'GSVT', 'SB', 'SR']
data_s1 = [df[df.class_name == r]['S1_mean'] for r in rhythms]
data_s2 = [df[df.class_name == r]['S2_mean'] for r in rhythms]

positions = np.arange(len(rhythms))
bp1 = ax2.boxplot(data_s1, positions=positions - 0.2, widths=0.3, patch_artist=True, showfliers=False)
bp2 = ax2.boxplot(data_s2, positions=positions + 0.2, widths=0.3, patch_artist=True, showfliers=False)

for patch in bp1['boxes']: patch.set_facecolor('#5599b8')
for patch in bp2['boxes']: patch.set_facecolor('#d1658b')

ax2.set_title("Explanation Stability (S) by Rhythm", fontweight='bold', loc='left', fontsize=14)
ax2.set_xticks(positions)
ax2.set_xticklabels(rhythms)
ax2.set_ylabel("Mean Stability Score")
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ['1D Stability', '2D Stability'], frameon=False, loc='lower right')

# --- Footer ---
fig.text(0.05, 0.04, "Prediction-preserving perturbations only  ·  PRR reported separately  ·  Stability is not a faithfulness claim", 
         fontsize=10, color='#95a5a6')

plt.savefig("results/module2/figures_m2/module2_dashboard.png", dpi=300, bbox_inches='tight')
plt.show()

# ---- notebook cell 22 ----
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
import gc

# --- 1. Load Data ---
TAG = "full_v2"
D = f"results/module2/module2_{TAG}"
df = pd.read_csv(f"{D}/noise_stability_records.csv")
z = np.load(f"{D}/noise_stability_maps.npz")
ECG_ROOT = "data/ECGData"
FIG_DIR = "results/module2/figures_m2"
os.makedirs(FIG_DIR, exist_ok=True)

def load_ecg(fn):
    return pd.read_csv(os.path.join(ECG_ROOT, fn), usecols=["II"])["II"].to_numpy(dtype=np.float64)

# --- 2. Styling (Dwi's Exact Palette) ---
BG = "#FFFBF7"       
INK = "#2D3047"      
MUTE = "#767A8C"     
EDGE = "#E8E0DA"     
CLS = {"AFib": "#D06B9A", "GSVT": "#4A93B5", "SB": "#A98568", "SR": "#A47A8A"}
CL = ["AFib", "GSVT", "SB", "SR"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "white", 
    "savefig.facecolor": BG, "axes.edgecolor": EDGE,
    "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE,
    "text.color": INK, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False
})

def save_fig(fig, name):
    path = f"{FIG_DIR}/{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.png")

def draw_card(fig, x, y, w, h):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", 
                                  transform=fig.transFigure, fc="white", ec=EDGE, lw=1.2, zorder=0))

# =====================================================================
# FIGURE 1: DASHBOARD OVERVIEW
# =====================================================================
def make_dashboard():
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.04, 0.92, "DAN / MMGradCAM", color=CLS["AFib"], weight="bold", fontsize=14)
    fig.text(0.04, 0.88, "02 | Noise Stability Overview", fontsize=28, weight="bold")
    fig.text(0.04, 0.84, f"{len(df):,} held-out ECG records  ·  30 perturbations each  ·  Module 2 focus", fontsize=13, color=MUTE)

    # KPI Cards
    kpis = [
        ("RECORDS", f"{len(df):,}", CLS["AFib"]),
        ("MEDIAN S (1D)", f"{df.S1_mean.median():.3f}", CLS["GSVT"]),
        ("MEDIAN S (2D)", f"{df.S2_mean.median():.3f}", CLS["SB"]),
        ("MEDIAN R", f"{df.R_mean.median():.3f}", CLS["SR"]),
        ("MEAN PRR", f"{df.prr_1d.mean():.2f} / {df.prr_2d.mean():.2f}", INK)
    ]
    for k, (title, val, col) in enumerate(kpis):
        x = 0.04 + k * 0.19
        draw_card(fig, x, 0.70, 0.17, 0.11)
        fig.text(x + 0.02, 0.76, title, color=MUTE, weight="bold", fontsize=9)
        fig.text(x + 0.02, 0.72, val, color=col, weight="bold", fontsize=22)

    # Chart 1: PRR by Family
    ax1 = fig.add_axes([0.06, 0.12, 0.40, 0.48])
    fams = ['scale', 'drift_0.05', 'drift_0.1', 'noise_0.05', 'noise_0.1']
    labels = ['Scale', 'Drift 5%', 'Drift 10%', 'Noise 5%', 'Noise 10%']
    x = np.arange(len(fams))
    ax1.bar(x - 0.2, [df[f'prr_1d_{f}'].mean() for f in fams], 0.4, label='1D', color=CLS["GSVT"])
    ax1.bar(x + 0.2, [df[f'prr_2d_{f}'].mean() for f in fams], 0.4, label='2D', color=CLS["AFib"])
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylim(0.85, 1.0); ax1.legend(frameon=False)
    ax1.set_title("Prediction Retention (PRR) by Perturbation", loc='left', fontweight='bold')
    ax1.grid(axis='y', color=EDGE)

    # Chart 2: Stability by Rhythm
    ax2 = fig.add_axes([0.56, 0.12, 0.38, 0.48])
    for j, c in enumerate(CL):
        bp = ax2.boxplot([df[df.class_name == c]['S1_mean'], df[df.class_name == c]['S2_mean']], 
                         positions=[j-0.15, j+0.15], widths=0.25, patch_artist=True, showfliers=False)
        bp['boxes'][0].set(facecolor=CLS["GSVT"], alpha=0.8)
        bp['boxes'][1].set(facecolor=CLS["AFib"], alpha=0.8)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(CL)
    ax2.set_title("Explanation Stability (S) by Rhythm", loc='left', fontweight='bold')
    ax2.set_ylabel("Stability Score (S)")
    ax2.grid(axis='y', color=EDGE)
    
    fig.text(0.04, 0.02, "Solid boxes = 1D Model  ·  Pale boxes = 2D Model  ·  Stability is not a faithfulness claim", color=MUTE, fontsize=9)
    save_fig(fig, "01_dashboard_overview")

# =====================================================================
# FIGURE 2: DISTRIBUTIONS
# =====================================================================
def make_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes.ravel()
    
    ax[0].hist([df.prr_1d, df.prr_2d], bins=np.linspace(0.1, 1.0, 25), color=[CLS["GSVT"], CLS["AFib"]], label=["1D", "2D"])
    ax[0].set_title("Prediction Retention Rate (PRR)")
    ax[0].legend(frameon=False); ax[0].set_yscale('log')
    
    ax[1].hist([df.S1_mean, df.S2_mean], bins=30, color=[CLS["GSVT"], CLS["AFib"]], label=["S1 (1D)", "S2 (2D)"])
    ax[1].set_title("Mean Stability per Record")
    ax[1].legend(frameon=False)
    
    ax[2].hist(df.R_mean, bins=30, color=CLS["SR"])
    ax[2].set_title("Mean Reliability R(u) per Record")
    
    ax[3].hist(df.n_joint_accepted, bins=np.arange(0, 32, 1), color=CLS["SB"])
    ax[3].set_title("Jointly Accepted Perturbations (out of 30)")
    
    for a in ax: a.grid(axis='y', color=EDGE)
    fig.tight_layout()
    save_fig(fig, "02_distributions")

# =====================================================================
# FIGURE 3: SCATTER S1 vs S2
# =====================================================================
def make_scatter():
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in CL:
        d = df[df.class_name == c]
        ax.scatter(d.S1_mean, d.S2_mean, s=20, alpha=0.6, color=CLS[c], label=c, edgecolors='none')
    ax.set_xlabel("1D Stability (S1)")
    ax.set_ylabel("2D Stability (S2)")
    ax.set_title(f"Stability Comparison: 1D vs 2D (r = {df.S1_mean.corr(df.S2_mean):.2f})", fontweight='bold')
    ax.legend(frameon=False)
    ax.grid(color=EDGE)
    save_fig(fig, "03_scatter_S1_vs_S2")

# =====================================================================
# FIGURE 4: CLASS MEAN CURVES (Over Time)
# =====================================================================
def make_class_curves():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    W_, FS_ = 600, 500.0
    TS = np.clip(((np.arange(W_) + .5) * 1080 / W_ - .5 - 58) / (1021 - 58), 0, 1) * (5000 - 1)
    TSEC = TS / FS_
    v = np.isfinite(z["R"]).any(0)
    
    for c in CL:
        i = (df.class_name == c).to_numpy()
        for a, k, t in zip(ax, ["R", "S1", "S2"], ["R(u)", "S 1D(u)", "S 2D(u)"]):
            m = np.nanmean(z[k][i], 0)
            a.plot(TSEC[v], m[v], color=CLS[c], label=c)
            a.set_title(f"{t}  (Class Mean)")
            a.set_xlabel("Time (s)")
            a.grid(color=EDGE)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "04_class_mean_curves")

# =====================================================================
# FIGURE 5: RECORD-BY-TIME HEATMAPS
# =====================================================================
def make_heatmaps():
    o = df.assign(_i=np.arange(len(df))).sort_values(["class_name", "R_mean"])
    idx = o._i.to_numpy()
    v = np.isfinite(z["R"]).any(0)
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    for a, k, t in zip(ax, ["R", "S1", "S2"], ["R(u)", "S 1D(u)", "S 2D(u)"]):
        M = z[k][idx][:, v]
        hi = np.nanpercentile(M, 99)
        im = a.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=hi, interpolation="nearest")
        a.set_title(f"{t} (Sorted by class & R)")
        a.set_xlabel("Time Grid (600 cols)")
        fig.colorbar(im, ax=a, fraction=.04)
        
    cum = np.cumsum([0] + [int((o.class_name == c).sum()) for c in CL])
    for a in ax:
        for e in cum[1:-1]: a.axhline(e, color="white", lw=1.5, linestyle="--")
    ax[0].set_yticks((cum[:-1] + cum[1:]) / 2)
    ax[0].set_yticklabels(CL)
    fig.tight_layout()
    save_fig(fig, "05_record_by_time_heatmaps")

# =====================================================================
# FIGURE 6: PER-RECORD PANELS (FIXED)
# =====================================================================
def make_panels():
    ok = df[(df.pred_1d == df.class_name) & (df.pred_2d == df.class_name)]
    t_axis = np.linspace(0, 10, 600) # Explicit time axis for the 600 columns
    
    for c in ["AFib", "SR"]:
        d = ok[ok.class_name == c].sort_values("R_mean")
        for label, r in [("Most Stable", d.iloc[-1]), ("Median", d.iloc[len(d)//2]), ("Least Stable", d.iloc[0])]:
            i = int(np.where(df.cohort_index.to_numpy() == r.cohort_index)[0][0])
            x = load_ecg(r.ecg_csv_filename)
            
            fig = plt.figure(figsize=(14, 7))
            gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.4)
            
            # 1. Top: Raw ECG (0 to 10 seconds)
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(np.arange(5000)/500, x, color=INK, lw=1)
            ax0.set_xlim(0, 10)
            ax0.set_title(f"{c} · {label} · Cohort {int(r.cohort_index)} · R={r.R_mean:.3f}", loc='left', fontweight='bold')
            ax0.set_xticks([]); ax0.set_yticks([])
            
            # 2. Middle: 2D CAM (stretched via extent)
            ax1 = fig.add_subplot(gs[1])
            ax1.imshow(z['base2'][i][np.newaxis, :], aspect='auto', cmap='jet', interpolation='bilinear', extent=[0, 10, 0, 1])
            ax1.set_ylabel("2D CAM", rotation=0, ha='right', va='center')
            ax1.set_xticks([]); ax1.set_yticks([])
            
            # 3. Stability S (using explicit time axis)
            ax2 = fig.add_subplot(gs[2])
            ax2.plot(t_axis, z['S1'][i], color=CLS["GSVT"], label="S1 (1D)")
            ax2.plot(t_axis, z['S2'][i], color=CLS["AFib"], label="S2 (2D)")
            ax2.set_xlim(0, 10)
            ax2.set_ylabel("Stability S", rotation=0, ha='right', va='center')
            ax2.legend(frameon=False, ncol=2, loc="upper right")
            ax2.set_xticks([])
            
            # 4. Reliability R(u) (using explicit time axis)
            ax3 = fig.add_subplot(gs[3])
            ax3.fill_between(t_axis, z['R'][i], color=CLS["SR"], alpha=0.6)
            ax3.plot(t_axis, z['R'][i], color=CLS["SR"])
            ax3.set_xlim(0, 10)
            ax3.set_ylabel("R(u)", rotation=0, ha='right', va='center')
            ax3.set_xlabel("Time (seconds)")
            
            save_fig(fig, f"06_panel_{c}_{label.replace(' ', '_')}")
            gc.collect()

# =====================================================================
# FIGURE 7: BEAUTIFUL ANCHOR GALLERY
# =====================================================================
def make_gallery():
    classes_to_plot = ["AFib", "GSVT", "SB"]
    selected_indices = {}
    for c in classes_to_plot:
        class_df = df[df.class_name == c].sort_values("R_mean")
        selected_indices[c] = class_df.index[len(class_df) // 2]

    fig = plt.figure(figsize=(16, 10))
    fig.text(0.04, 0.94, "Anchor-style ECG alignment gallery", fontsize=26, weight="bold", color=INK)
    fig.text(0.04, 0.90, "Module 2: Noise Stability & Reliability-weighted attribution R(u)", fontsize=13, color=MUTE)

    for i, cls in enumerate(classes_to_plot):
        idx = selected_indices[cls]
        row = df.iloc[idx]
        x = load_ecg(row.ecg_csv_filename)
        
        base2 = z['base2'][idx]
        R_map = z['R'][idx]
        
        y0 = 0.68 - i * 0.23
        draw_card(fig, 0.04, y0, 0.92, 0.20)
        
        fig.text(0.06, y0 + 0.16, cls, color="white", weight="bold", fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.4", fc=CLS[cls], ec="none"))
        fig.text(0.14, y0 + 0.16, "Paired ECG  ·  1D markers + 2D CAM", fontsize=11, color=INK)
        
        ax = fig.add_axes([0.10, y0 + 0.03, 0.82, 0.11])
        ax.plot(np.linspace(0, 10, 5000), x, color=INK, linewidth=0.8, zorder=3)
        ax.set_xlim(0, 10)
        ax.set_ylim(x.min(), x.max())
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        
        ax.imshow(np.ma.masked_where(base2 < 0.15, base2)[np.newaxis, :], 
                  aspect='auto', cmap='Spectral_r', alpha=0.6, zorder=2,
                  extent=[0, 10, x.min(), x.max()], interpolation='bilinear')
        
        base1_5000 = np.interp(np.linspace(0, 10, 5000), np.linspace(0, 10, 600), z['base1'][idx])
        thr = np.percentile(base1_5000, 85)
        marker_idx = np.where(base1_5000 > thr)[0]
        ax.scatter(np.linspace(0, 10, 5000)[marker_idx], x[marker_idx], 
                   color="#E0182D", s=8, zorder=4, edgecolors='none')
        
        ax_R = fig.add_axes([0.10, y0 + 0.03, 0.82, 0.02])
        ax_R.imshow(R_map[np.newaxis, :], aspect='auto', cmap='magma', 
                    extent=[0, 10, 0, 1], vmin=0, vmax=np.nanpercentile(R_map, 99), interpolation='bilinear')
        ax_R.set_xticks([]); ax_R.set_yticks([])
        for spine in ax_R.spines.values(): spine.set_visible(False)
        fig.text(0.06, y0 + 0.03, "R(u)", color=MUTE, fontsize=10, weight="bold")

    fig.text(0.04, 0.02, "Paper-like visual composition; provisional temporal co-registration; not a quantitative agreement validation.", color=MUTE, fontsize=10)
    save_fig(fig, "07_anchor_gallery")

# --- EXECUTE ALL ---
print("Generating all Module 2 figures...")
make_dashboard()
make_distributions()
make_scatter()
make_class_curves()
make_heatmaps()
make_panels()
make_gallery()
print("All figures generated successfully in results/module2/figures_m2/")

# ---- notebook cell 35 ----
# ============================================================
# MODULE 2 — PAPER FIGURES + RESEARCH CONCLUSION
# Creates figures from existing outputs and downloads a ZIP.
# ============================================================

import os
import glob
import zipfile
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------- PATHS --------------------

ROOT = "results/module2"

SENS_DIR = os.path.join(ROOT, "deletion_sensitivity_v1")
FAITH_DIR = os.path.join(ROOT, "faithfulness_corrected_v1")
MAP_DIR = os.path.join(ROOT, "mapping_audit_v1")

OUT = os.path.join(ROOT, "module2_paper_figures")
os.makedirs(OUT, exist_ok=True)

sens_path = os.path.join(SENS_DIR, "sensitivity_summary.csv")
map_path_candidates = glob.glob(os.path.join(MAP_DIR, "*.csv"))

assert os.path.exists(sens_path), f"Missing: {sens_path}"
assert map_path_candidates, "Mapping audit CSV not found."

sens = pd.read_csv(sens_path)
mapping = pd.read_csv(map_path_candidates[0])

# -------------------- STYLE --------------------

plt.rcParams.update({
    "figure.figsize": (9, 5.5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.bbox": "tight"
})

COLORS = {
    "R": "#D46A94",
    "H1": "#4D94B7",
    "H2": "#82A65B",
    "base1": "#AE8266",
    "base2": "#9670B8"
}

def savefig(name):
    plt.savefig(os.path.join(OUT, name), dpi=300)
    plt.close()

# ============================================================
# FIGURE 1 — MAPPING AUDIT
# ============================================================

required = {
    "map",
    "mean_mass_outside_geometry",
    "mean_top10_percent_outside_geometry"
}
assert required.issubset(mapping.columns), (
    f"Unexpected mapping columns: {mapping.columns.tolist()}"
)

m = mapping.copy()

plt.figure()
x = np.arange(len(m))
w = 0.36

plt.bar(
    x - w/2,
    m["mean_mass_outside_geometry"],
    w,
    label="Mean attribution mass outside ECG region"
)
plt.bar(
    x + w/2,
    m["mean_top10_percent_outside_geometry"],
    w,
    label="Top-10% attribution outside ECG region"
)

plt.xticks(x, m["map"])
plt.ylabel("Attribution proportion")
plt.title("Module 2 — Attribution-to-signal mapping audit")
plt.legend()
savefig("01_mapping_audit.png")

# ============================================================
# FIGURE 2 — 1D DELETION SENSITIVITY
# ============================================================

required = {
    "branch", "method", "block_size", "fraction",
    "paired_difference"
}
assert required.issubset(sens.columns), (
    f"Unexpected sensitivity columns: {sens.columns.tolist()}"
)

def plot_sensitivity(branch, filename, title):
    d = sens[sens["branch"].str.lower() == branch].copy()

    plt.figure(figsize=(9, 5.5))

    methods = list(d["method"].dropna().unique())

    for method in methods:
        for block in sorted(d["block_size"].unique()):
            q = d[
                (d["method"] == method) &
                (d["block_size"] == block)
            ].sort_values("fraction")

            if q.empty:
                continue

            x = q["fraction"].to_numpy() * 100
            y = q["paired_difference"].to_numpy()

            plt.plot(
                x, y, marker="o",
                label=f"{method}, block={block}"
            )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("ECG signal deleted (%)")
    plt.ylabel("Attribution drop − random drop")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    savefig(filename)

plot_sensitivity(
    "1d",
    "02_deletion_sensitivity_1d.png",
    "Module 2 — 1D deletion sensitivity"
)

# ============================================================
# FIGURE 3 — 2D DELETION SENSITIVITY
# ============================================================

plot_sensitivity(
    "2d",
    "03_deletion_sensitivity_2d.png",
    "Module 2 — 2D deletion sensitivity"
)

# ============================================================
# FIGURE 4 — PAIRED DIFFERENCE WITH CONFIDENCE INTERVALS
# ============================================================

low_col = next(
    (c for c in sens.columns if c in
     ["difference_ci_low", "diff_ci_low", "ci_low"]),
    None
)
high_col = next(
    (c for c in sens.columns if c in
     ["difference_ci_high", "diff_ci_high", "ci_high"]),
    None
)

if low_col and high_col:
    for branch in ["1d", "2d"]:
        d = sens[sens["branch"].str.lower() == branch].copy()
        d = d.sort_values(["method", "block_size", "fraction"])

        if d.empty:
            continue

        labels = (
            d["method"].astype(str) +
            " | " + d["block_size"].astype(str) +
            " | " + (d["fraction"] * 100).round().astype(int).astype(str) + "%"
        )

        y = d["paired_difference"].to_numpy()
        lo = d[low_col].to_numpy()
        hi = d[high_col].to_numpy()

        plt.figure(figsize=(10, max(5, len(d) * 0.24)))
        positions = np.arange(len(d))

        plt.errorbar(
            y, positions,
            xerr=[y - lo, hi - y],
            fmt="o",
            capsize=2
        )

        plt.axvline(0, color="black", linewidth=1)
        plt.yticks(positions, labels, fontsize=7)
        plt.xlabel("Paired difference (attribution − random)")
        plt.title(f"Module 2 — {branch.upper()} paired differences with confidence intervals")
        savefig(f"04_paired_ci_{branch}.png")

# ============================================================
# FIGURE 5 — MAPPING MASS SUMMARY
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    mapping["map"],
    mapping["mean_mass_outside_geometry"]
)

plt.ylabel("Mean attribution mass outside ECG region")
plt.title("Module 2 — Attribution outside valid ECG geometry")
savefig("05_outside_geometry.png")

# ============================================================
# RESEARCH-PAPER CONCLUSION
# ============================================================




# ============================================================
# COPY ORIGINAL RESULT TABLES INTO FIGURE PACKAGE
# ============================================================

for folder in [FAITH_DIR, SENS_DIR, MAP_DIR]:
    if os.path.isdir(folder):
        for file in glob.glob(os.path.join(folder, "*.csv")):
            target = os.path.join(OUT, os.path.basename(file))
            if not os.path.exists(target):
                import shutil
                shutil.copy2(file, target)

# ============================================================
# ZIP AND DOWNLOAD
# ============================================================

zip_path = os.path.join(ROOT, "module2_paper_figures.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in glob.glob(os.path.join(OUT, "*")):
        z.write(file, arcname=os.path.basename(file))

print("\nCompleted.")
print("Figures and conclusion:", OUT)
print("ZIP:", zip_path)

