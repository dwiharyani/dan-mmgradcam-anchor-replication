"""Convergence of stability scores vs number of perturbations (pilot). Use validation records for any quoted result.

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
from module2_common import *   # defines paths, test manifest, models, perturbation/render/Grad-CAM functions

# ---- notebook cell 14 ----
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Setup ---
N_PILOT_RECORDS = 15
MAX_PERT = 50

# Select a balanced mix of records from the test set
pilot_records = test.groupby("class_name").head(4).sample(frac=1, random_state=42).head(N_PILOT_RECORDS)

# We need a modified perturbation generator to go up to 50
def make_pilot_perturbations(x, rng):
    p2p, sd, t = np.ptp(x), x.std(), np.arange(N) / FS
    P = []
    # 10 scales
    for s in [0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2, 1.3]:
        P.append(x * s)
    # 20 drifts
    for a in [0.05, 0.10]:
        for ph in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            P.append(x + a * p2p * np.sin(2 * np.pi * 0.3 * t + ph))
    # 20 noises
    for a in [0.05, 0.10]:
        for k in range(10):
            P.append(x + rng.normal(0, a * sd, N))
    return np.stack(P)

S1_curves, S2_curves = [], []
rend = Renderer()

print(f"Running pilot on {N_PILOT_RECORDS} records up to {MAX_PERT} perturbations...")

# --- 2. Run the Pilot ---
for idx, r in pilot_records.iterrows():
    x = load_ecg(r.ecg_csv_filename)
    rng = np.random.default_rng(int(r.cohort_index))

    # Generate 50 perturbations
    P = make_pilot_perturbations(x, rng)
    xs = np.concatenate([x[None], P])

    # Prepare inputs exactly as in Module 2
    X1_ = ((xs - MEAN) / SCALE).astype(np.float32)[..., None]
    rgbs = np.stack([rend.base(x)] + [rend.pert(p) for p in P])
    I2 = to_2d(rgbs)
    I2n = (I2 / 255.0).astype(np.float32)

    # Predictions
    pr1 = run_model("1d", m1, X1_)
    pr2 = run_model("2d", m2, I2n)
    k1, k2 = int(pr1[0].argmax()), int(pr2[0].argmax())
    keep1 = pr1.argmax(1)[1:] == k1
    keep2 = pr2.argmax(1)[1:] == k2

    # Grad-CAMs
    cam1 = gradcam("1d", m1, L1, X1_, k1)
    cam2 = gradcam("2d", m2, L2, I2n, k2)

    # Valid columns
    mask0 = trace_mask_at_2d_grid(rgbs[0], dilate_y=MASK_DILATE_Y)
    valid = GEOM_VALID & (mask0.sum(axis=0) > 0)

    # Project to 600-grid
    G1 = np.stack([project_1d(c, valid) for c in cam1])
    G2 = np.stack([project_2d(c, rgbs[i], valid)[0] for i, c in enumerate(cam2)])

    # Calculate running stability for N=2 to 50
    s1_run, s2_run = [], []
    for n in range(2, MAX_PERT + 1):
        acc1 = keep1[:n] & np.array([np.isfinite(g[valid]).all() for g in G1[1:n+1]])
        acc2 = keep2[:n] & np.array([np.isfinite(g[valid]).all() for g in G2[1:n+1]])

        if acc1.sum() >= 2:
            mu1, sd1 = G1[1:n+1][acc1][:, valid].mean(0), G1[1:n+1][acc1][:, valid].std(0)
            S1 = 1 - sd1 / (sd1.max() + 1e-8)
            s1_run.append(S1.mean())
        else:
            s1_run.append(np.nan)

        if acc2.sum() >= 2:
            mu2, sd2 = G2[1:n+1][acc2][:, valid].mean(0), G2[1:n+1][acc2][:, valid].std(0)
            S2 = 1 - sd2 / (sd2.max() + 1e-8)
            s2_run.append(S2.mean())
        else:
            s2_run.append(np.nan)

    S1_curves.append(s1_run)
    S2_curves.append(s2_run)
    print(f"  Processed record {r.cohort_index} ({r.class_name})")

plt.close(rend.fig)

# --- 3. Plotting ---
mean_S1 = np.nanmean(S1_curves, axis=0)
mean_S2 = np.nanmean(S2_curves, axis=0)
x_axis = np.arange(2, MAX_PERT + 1)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, mean_S1, label="1D Stability (S1)", color="#5599b8", lw=2)
plt.plot(x_axis, mean_S2, label="2D Stability (S2)", color="#d1658b", lw=2)

# Add a vertical line where Module 2 actually cut off
plt.axvline(x=30, color="red", linestyle="--", label="Chosen N = 30 (Module 2)")

plt.title("Convergence of Stability Scores vs. Number of Perturbations", fontweight="bold")
plt.xlabel("Number of Perturbations (N)")
plt.ylabel("Mean Stability Score (S)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save for your paper
os.makedirs("results/module2/figures_m2", exist_ok=True)
plt.savefig("results/module2/figures_m2/convergence_pilot.png", dpi=300)
plt.show()

# ---- notebook cell 27 ----
import numpy as np
import matplotlib.pyplot as plt
import os

# --- FIX: Redefine Renderer with the missing 'pert' method ---
class Renderer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot(np.arange(5000), np.zeros(5000))
        self.ax.axis("off")
    def base(self, x):
        self.ax.set_autoscaley_on(True)
        self.line.set_ydata(x)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.tight_layout()
        self.ylim = self.ax.get_ylim()
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
    def pert(self, x):
        self.line.set_ydata(x)
        self.ax.set_ylim(*self.ylim)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

N_PILOT_RECORDS = 15
MAX_PERT = 50

# Use the newly created validation set
pilot_records = val_df.groupby("class_name").head(4).sample(frac=1, random_state=42).head(N_PILOT_RECORDS)

def make_pilot_perturbations(x, rng):
    p2p, sd, t = np.ptp(x), x.std(), np.arange(N) / FS
    P = []
    for s in [0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2, 1.3]: P.append(x * s)
    for a in [0.05, 0.10]:
        for ph in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            P.append(x + a * p2p * np.sin(2 * np.pi * 0.3 * t + ph))
    for a in [0.05, 0.10]:
        for k in range(10): P.append(x + rng.normal(0, a * sd, N))
    return np.stack(P)

S1_curves, S2_curves = [], []
rend = Renderer()

print(f"Running VALIDATION pilot on {N_PILOT_RECORDS} records...")

for idx, r in pilot_records.iterrows():
    x = load_ecg(r.ecg_csv_filename)
    rng = np.random.default_rng(int(r.cohort_index))
    P = make_pilot_perturbations(x, rng)
    xs = np.concatenate([x[None], P])
    X1_ = ((xs - MEAN) / SCALE).astype(np.float32)[..., None]
    rgbs = np.stack([rend.base(x)] + [rend.pert(p) for p in P])
    I2 = to_2d(rgbs)
    I2n = (I2 / 255.0).astype(np.float32)

    pr1 = run_model("1d", m1, X1_)
    pr2 = run_model("2d", m2, I2n)
    k1, k2 = int(pr1[0].argmax()), int(pr2[0].argmax())
    keep1 = pr1.argmax(1)[1:] == k1
    keep2 = pr2.argmax(1)[1:] == k2

    cam1 = gradcam("1d", m1, L1, X1_, k1)
    cam2 = gradcam("2d", m2, L2, I2n, k2)

    mask0 = trace_mask_at_2d_grid(rgbs[0], dilate_y=MASK_DILATE_Y)
    valid = GEOM_VALID & (mask0.sum(axis=0) > 0)
    G1 = np.stack([project_1d(c, valid) for c in cam1])
    G2 = np.stack([project_2d(c, rgbs[i], valid)[0] for i, c in enumerate(cam2)])

    s1_run, s2_run = [], []
    for n in range(2, MAX_PERT + 1):
        acc1 = keep1[:n] & np.array([np.isfinite(g[valid]).all() for g in G1[1:n+1]])
        acc2 = keep2[:n] & np.array([np.isfinite(g[valid]).all() for g in G2[1:n+1]])
        
        if acc1.sum() >= 2:
            mu1, sd1 = G1[1:n+1][acc1][:, valid].mean(0), G1[1:n+1][acc1][:, valid].std(0)
            s1_run.append((1 - sd1 / (sd1.max() + 1e-8)).mean())
        else: s1_run.append(np.nan)

        if acc2.sum() >= 2:
            mu2, sd2 = G2[1:n+1][acc2][:, valid].mean(0), G2[1:n+1][acc2][:, valid].std(0)
            s2_run.append((1 - sd2 / (sd2.max() + 1e-8)).mean())
        else: s2_run.append(np.nan)

    S1_curves.append(s1_run)
    S2_curves.append(s2_run)
    print(f"  Processed {r.cohort_index} ({r.class_name})")

plt.close(rend.fig)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(np.arange(2, MAX_PERT + 1), np.nanmean(S1_curves, axis=0), label="1D Stability (S1)", color="#5599b8", lw=2)
plt.plot(np.arange(2, MAX_PERT + 1), np.nanmean(S2_curves, axis=0), label="2D Stability (S2)", color="#d1658b", lw=2)
plt.axvline(x=30, color="red", linestyle="--", label="Chosen N = 30 (Module 2)")
plt.title("Convergence of Stability Scores vs. Number of Perturbations (Validation Set)", fontweight="bold")
plt.xlabel("Number of Perturbations (N)")
plt.ylabel("Mean Stability Score (S)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("results/module2/figures_m2", exist_ok=True)
plt.savefig("results/module2/figures_m2/validation_convergence_pilot.png", dpi=300)
plt.show()
