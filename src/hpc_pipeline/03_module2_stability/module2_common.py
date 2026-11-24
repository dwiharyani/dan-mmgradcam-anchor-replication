"""Module 2 common setup: paths, ECG loader, provisional 1D scaler, pretrained models, noise-stability definitions.

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
# ---- notebook cell 0 ----
# 1. Setup: imports and paths
import os, json, time, pickle, datetime, contextlib, subprocess, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5000                                              # 10 s at 500 Hz, Lead II
CLASSES = ["AFib", "GSVT", "SB", "SR"]

ECG_ROOT = "data/ECGData"
IMG_ROOT = "data/ImageData-label2/label1"
MODEL1D  = "models/MMGradCAM_1d_trained_model.keras"
MODEL2D  = "models/MMGradCAM_2d_trained_model.keras"
REPO_URL = "https://github.com/dwiharyani/dan-mmgradcam-anchor-replication.git"
REPO_DIR = "results/module2/dan-mmgradcam-anchor-replication"
OUT      = "results/module2"
SCALER_PATH = "results/module2/standard_scaler.npz"
os.makedirs(OUT, exist_ok=True)

for p in [ECG_ROOT, IMG_ROOT, MODEL1D, MODEL2D]:
    print("OK     " if os.path.exists(p) else "MISSING", p)
print("TensorFlow", tf.__version__, "| GPUs:", [d.name for d in tf.config.list_physical_devices("GPU")])

# ---- notebook cell 1 ----
# 2. Repo + test manifest
if not os.path.exists(REPO_DIR):
    r = subprocess.run(["git", "clone", REPO_URL, REPO_DIR], capture_output=True, text=True)
    print("clone ok" if r.returncode == 0 else r.stderr)
else:
    print("repo already present")

test = pd.read_csv(f"{REPO_DIR}/reproducibility/test_split_1054.csv")
ecg_files = sorted(f for f in os.listdir(ECG_ROOT) if f.lower().endswith(".csv"))

assert len(test) == 1054 and test["cohort_index"].is_unique and test["ecg_csv_filename"].is_unique
missing = set(test["ecg_csv_filename"]) - set(ecg_files)
assert not missing, f"{len(missing)} test ECG files missing"
print("ECG CSV files in folder:", len(ecg_files), "| test records:", len(test))
print(test["class_name"].value_counts().to_dict())
print("test columns:", test.columns.tolist())

# ---- notebook cell 2 ----
# 3. ECG loader (Lead II) + check against the signals saved by your earlier run
def load_ecg(fn):
    a = pd.read_csv(os.path.join(ECG_ROOT, fn), usecols=["II"])["II"].to_numpy(dtype=np.float64)
    assert a.shape == (N,) and np.isfinite(a).all(), (fn, a.shape)
    return a

SAVED_TEST = "results/module2/X_test_raw.npy"          # written by your earlier session
if os.path.exists(SAVED_TEST):
    Xt_saved = np.load(SAVED_TEST, mmap_mode="r")
    assert Xt_saved.shape == (len(test), N), Xt_saved.shape
    idx = np.linspace(0, len(test) - 1, 25).astype(int)
    d = max(float(np.abs(load_ecg(test["ecg_csv_filename"].iloc[i]) - Xt_saved[i]).max()) for i in idx)
    print("max |diff| vs saved test signals (25 records):", d)
    assert d < 1e-2, "load_ecg does NOT reproduce the signals used earlier - stop and tell Claude"
    print("load_ecg reproduces the earlier signals")
else:
    x = load_ecg(test["ecg_csv_filename"].iloc[0])
    print("No saved test signals to compare with; shape check only:", x.shape)

# ---- notebook cell 3 ----
# 4. 1D scaler (PROVISIONAL: fitted on 9,592 records, intended cohort is 9,534)
# Loads the saved scaler. It only refits if the file is missing, and never silently.
REFIT_SCALER = False  # set True to rebuild the provisional scaler (~11 min)
if REFIT_SCALER or not os.path.exists(SCALER_PATH):
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    fit_files = sorted(set(ecg_files) - set(test["ecg_csv_filename"]))
    assert len(fit_files) == 9592, len(fit_files)       # = 10,646 - 1,054
    X = np.stack(Parallel(n_jobs=-1, batch_size=64)(delayed(load_ecg)(f) for f in tqdm(fit_files)))
    mean, var = X.mean(0), X.var(0)
    scale = np.sqrt(var); scale[scale == 0] = 1.0
    np.savez(SCALER_PATH, mean=mean, scale=scale, var=var)
    print("REFIT scaler on", X.shape[0], "records (provisional)")
    del X
sc = np.load(SCALER_PATH)
MEAN, SCALE = sc["mean"], sc["scale"]
assert MEAN.shape == (N,) and SCALE.shape == (N,)
SCALER_N_FIT = 9592
print("scaler loaded:", MEAN.shape, "| PROVISIONAL - fitted on", SCALER_N_FIT, "records, intended 9,534")

# ---- notebook cell 5 ----
# 6. Pretrained models (not retrained)
def last_conv(model, layer_type):
    return next(l.name for l in reversed(model.layers) if isinstance(l, layer_type))

m1 = tf.keras.models.load_model(MODEL1D)
m2 = tf.keras.models.load_model(MODEL2D)
L1 = last_conv(m1, tf.keras.layers.Conv1D)
L2 = last_conv(m2, tf.keras.layers.Conv2D)
print("1D:", m1.input_shape, "->", m1.output_shape, "| last Conv1D:", L1)
print("2D:", m2.input_shape, "->", m2.output_shape, "| last Conv2D:", L2)
assert m1.input_shape[1:] == (N, 1) and m2.input_shape[1:] == (300, 600, 1)

# ---- notebook cell 6 ----
# 7. MODULE 2 v2 definitions (no computation happens in this cell)
# [M1] = copied verbatim from scripts/run_dan_module1_exploratory.py
import os, json, time, pickle, datetime, contextlib 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter1d

# ---------------- 0. preflight ----------------
_missing = [n for n in ["N", "load_ecg", "test", "OUT", "MEAN", "SCALE", "SCALER_N_FIT",
                       "m1", "m2", "L1", "L2", "last_conv"] if n not in globals()]
assert not _missing, f"Run the earlier cells first. Not defined: {_missing}"
assert N == 5000 and len(test) == 1054
assert test["ecg_csv_filename"].is_unique and test["cohort_index"].is_unique

# ---------------- 1. configuration ----------------
EPS, FS, DRIFT_HZ = 1e-8, 500.0, 0.3
CLASSES = ["AFib", "GSVT", "SB", "SR"]
SCALES = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]
DRIFTS = [0.05, 0.10]
DRIFT_PHASES = np.linspace(0, 2 * np.pi, 6, endpoint=False)
NOISES = [0.05, 0.10]
NOISE_SEEDS = 6
N_PERT = 30
assert len(SCALES) + len(DRIFTS) * len(DRIFT_PHASES) + len(NOISES) * NOISE_SEEDS == N_PERT

X0, X1, W = 58, 1021, 600        # [M1] defaults; 600 = Module 1 grid columns
MASK_DILATE_Y, COVERAGE_MIN = 6, 0.75
MIN_ACCEPTED, CKPT_EVERY = 2, 25
FAMILIES = ["scale"] + [f"drift_{a}" for a in DRIFTS] + [f"noise_{a}" for a in NOISES]
BS = {"1d": 32, "2d": 8}
FORCE_CPU = {"1d": False, "2d": False}

CONFIG = json.loads(json.dumps(dict(
    version="v2-module1-aligned", FS=FS, DRIFT_HZ=DRIFT_HZ, SCALES=SCALES, DRIFTS=DRIFTS,
    DRIFT_PHASES=DRIFT_PHASES.tolist(), NOISES=NOISES, NOISE_SEEDS=NOISE_SEEDS,
    X0=X0, X1=X1, W=W, MASK_DILATE_Y=MASK_DILATE_Y, COVERAGE_MIN=COVERAGE_MIN,
    MIN_ACCEPTED=MIN_ACCEPTED)))

# ---------------- 2. Module 1 functions [M1] (verbatim) ----------------
def grayscale_norm(values, valid):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite attribution")
    subset = a[valid]
    if subset.size == 0:
        raise ValueError("No valid temporal columns")
    lo, hi = float(subset.min()), float(subset.max())
    if hi - lo <= 1e-10:
        return None
    out = np.zeros(a.shape, dtype=np.float32)
    out[valid] = ((subset - lo) / (hi - lo)).astype(np.float32)
    return out

def trace_mask_at_2d_grid(original_rgb, target_h=300, target_w=600, dilate_y=6):
    im = np.asarray(original_rgb, dtype=np.int16)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    ink = ((B - R >= 14) & (B - G >= 5) & (G - R >= 4) &
           (R < 240) & (G < 250))
    resized = Image.fromarray(ink.astype(np.uint8) * 255, 'L').resize(
        (target_w, target_h), Image.Resampling.BILINEAR)
    mask = (np.asarray(resized, dtype=np.uint8) > 14)
    mask = maximum_filter1d(mask, size=2 * dilate_y + 1, axis=0)
    return mask

def numeric_register_1d(x, x0, x1, original_width=1080, target_w=600):
    x_orig = (np.arange(target_w, dtype=np.float64) + .5) * original_width / target_w - .5
    valid = (x_orig >= x0) & (x_orig <= x1)
    t = np.clip((x_orig - x0) / (x1 - x0), 0, 1)
    out = np.interp(t * (len(x) - 1), np.arange(len(x)), x)
    return out, valid

_, GEOM_VALID = numeric_register_1d(np.zeros(N), X0, X1)

# ---------------- 3. perturbations ----------------
def make_perturbations(x, rng):
    p2p, sd, t = np.ptp(x), x.std(), np.arange(N) / FS
    names, fams, P = [], [], []
    for s in SCALES:
        names.append(f"scale_{s}"); fams.append("scale"); P.append(x * s)
    for a in DRIFTS:
        for ph in DRIFT_PHASES:
            names.append(f"drift_{a}_{ph:.3f}"); fams.append(f"drift_{a}")
            P.append(x + a * p2p * np.sin(2 * np.pi * DRIFT_HZ * t + ph))
    for a in NOISES:
        for k in range(NOISE_SEEDS):
            names.append(f"noise_{a}_{k}"); fams.append(f"noise_{a}")
            P.append(x + rng.normal(0, a * sd, N))
    P = np.stack(P)
    assert P.shape == (N_PERT, N)
    return names, fams, P

# ---------------- 4. 2D rendering (unchanged) ----------------
class Renderer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 4), dpi=72)
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot(np.arange(N), np.zeros(N))
        self.ax.axis("off")
    def base(self, x):
        self.ax.set_autoscaley_on(True)
        self.line.set_ydata(x)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.tight_layout()
        self.ylim = self.ax.get_ylim()
        return self._grab()
    def pert(self, x):
        self.line.set_ydata(x)
        self.ax.set_ylim(*self.ylim)
        return self._grab()
    def _grab(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

def to_2d(rgb):
    with tf.device("/CPU:0"):
        g = tf.image.rgb_to_grayscale(tf.cast(np.asarray(rgb), tf.float32))
        return tf.image.resize(g, [300, 600]).numpy()

# ---------------- 5. model calls (chunked, auto-fallback) ----------------
def _chunked(key, fn, x):
    while True:
        bs = BS[key]
        try:
            dev = tf.device("/CPU:0") if FORCE_CPU[key] else contextlib.nullcontext()
            with dev:
                return np.concatenate([fn(x[i:i + bs]) for i in range(0, len(x), bs)])
        except tf.errors.OpError as e:
            if bs > 1:
                BS[key] = max(1, bs // 2)
                print(f"[{key}] {type(e).__name__}; retrying with batch {BS[key]}")
            elif not FORCE_CPU[key]:
                FORCE_CPU[key] = True
                print(f"[{key}] batch 1 failed on GPU; retrying on CPU")
            else:
                raise

def run_model(key, model, x):
    return _chunked(key, lambda xb: model(tf.convert_to_tensor(xb), training=False).numpy(), x)

_GM = {}
def gradcam(key, model, layer, x, k):
    """Raw ReLU'd Grad-CAM at native resolution (same target/layer/weights as the authors' scripts)."""
    if key not in _GM:
        _GM[key] = tf.keras.Model(model.inputs, [model.get_layer(layer).output, model.output])
    gm = _GM[key]
    def one(xb):
        with tf.GradientTape() as tape:
            conv, pred = gm(tf.convert_to_tensor(xb), training=False)
            loss = tf.reduce_sum(pred[:, k])
        g = tape.gradient(loss, conv)
        w = tf.reduce_mean(g, axis=tuple(range(1, len(conv.shape) - 1)), keepdims=True)
        return tf.nn.relu(tf.reduce_sum(w * conv, axis=-1)).numpy()
    return _chunked(key, one, x)

# ---------------- 6. CAM -> Module 1 grid (600 columns) ----------------
def _max_norm(c):                      # authors: divide by max, zeros if max == 0
    m = float(np.max(c))
    return c / m if m > 0 else np.zeros_like(c, dtype=np.float32)

def _norm_or_nan(raw, valid):
    try:
        g = grayscale_norm(raw, valid)
    except ValueError:
        g = None
    if g is None:
        return np.full(W, np.nan, np.float32)
    g[~valid] = np.nan                 # outside valid columns: NaN (Module 1 stores 0)
    return g

def project_1d(cam_native, valid):
    c = _max_norm(np.asarray(cam_native, np.float64))
    full = np.interp(np.arange(N), np.linspace(0, N, len(c)), c)      # authors' 5000-sample resize
    reg, _ = numeric_register_1d(full, X0, X1)
    return _norm_or_nan(reg, valid)

def project_2d(cam_native, rgb, valid):
    c = _max_norm(np.asarray(cam_native, np.float64))
    r = tf.image.resize(c[..., None].astype(np.float32), [300, 600], method="bilinear").numpy()[..., 0]
    r = np.clip(r, 0.0, 1.0)
    mask = trace_mask_at_2d_grid(rgb, dilate_y=MASK_DILATE_Y)
    support = mask.sum(axis=0)
    raw = (r * mask).sum(axis=0) / (support + EPS)
    return _norm_or_nan(raw, valid), support

def stability(maps):                   # maps: (n, n_valid_columns)
    mu, sd = maps.mean(0), maps.std(0)
    S = 1 - sd / (sd.max() + EPS)
    return mu, S, mu * S

# ---------------- 8. one record ----------------
def eval_record(x, rng, rend):
    names, fams, P = make_perturbations(x, rng)
    xs = np.concatenate([x[None], P])                      # index 0 = original
    X1_ = ((xs - MEAN) / SCALE).astype(np.float32)[..., None]
    rgbs = np.stack([rend.base(x)] + [rend.pert(p) for p in P])       # (31, 288, 1080, 3)
    assert rgbs.shape[1:] == (288, 1080, 3), rgbs.shape
    I2 = to_2d(rgbs)
    I2n = (I2 / 255.0).astype(np.float32)

    pr1 = run_model("1d", m1, X1_)
    pr2 = run_model("2d", m2, I2n)
    k1, k2 = int(pr1[0].argmax()), int(pr2[0].argmax())
    keep1 = pr1.argmax(1)[1:] == k1
    keep2 = pr2.argmax(1)[1:] == k2

    cam1 = gradcam("1d", m1, L1, X1_, k1)
    cam2 = gradcam("2d", m2, L2, I2n, k2)

    # valid columns come from the ORIGINAL (unperturbed) rendering, as in Module 1
    mask0 = trace_mask_at_2d_grid(rgbs[0], dilate_y=MASK_DILATE_Y)
    valid = GEOM_VALID & (mask0.sum(axis=0) > 0)
    coverage = float(valid.sum() / max(1, GEOM_VALID.sum()))

    G1 = np.stack([project_1d(c, valid) for c in cam1])               # (31, 600)
    G2 = np.stack([project_2d(c, rgbs[i], valid)[0] for i, c in enumerate(cam2)])

    lo, hi = rend.ylim
    clip = ((P < lo) | (P > hi)).mean(1)
    return dict(k1=k1, k2=k2, keep1=keep1, keep2=keep2, names=names, fams=np.array(fams),
                G1=G1, G2=G2, valid=valid, coverage=coverage, clip=clip,
                out1=pr1[0], out2=pr2[0])

def summarize(r, ev):
    k1, k2, keep1, keep2, valid = ev["k1"], ev["k2"], ev["keep1"], ev["keep2"], ev["valid"]
    G1, G2 = ev["G1"], ev["G2"]
    ok = lambda G: np.array([np.isfinite(g[valid]).all() and valid.any() for g in G])
    ok1, ok2 = ok(G1), ok(G2)                       # non-degenerate maps (index 0 = original)
    acc1 = keep1 & ok1[1:]
    acc2 = keep2 & ok2[1:]
    joint = acc1 & acc2
    n1, n2, nj = int(acc1.sum()), int(acc2.sum()), int(joint.sum())

    row = dict(
        cohort_index=int(r.cohort_index), ecg_csv_filename=r.ecg_csv_filename,
        class_name=r.class_name, pred_1d=CLASSES[k1], pred_2d=CLASSES[k2],
        branch_disagree=bool(k1 != k2), n_perturb=N_PERT,
        n_kept_1d=int(keep1.sum()), n_kept_2d=int(keep2.sum()),
        n_degenerate_1d=int((~ok1[1:]).sum()), n_degenerate_2d=int((~ok2[1:]).sum()),
        n_accepted_1d=n1, n_accepted_2d=n2, n_joint_accepted=nj,
        prr_1d=float(keep1.mean()), prr_2d=float(keep2.mean()),
        base_degenerate_1d=bool(not ok1[0]), base_degenerate_2d=bool(not ok2[0]),
        base_coverage=ev["coverage"], bad_mask=bool(ev["coverage"] < COVERAGE_MIN),
        n_valid_cols=int(valid.sum()),
        clip_frac_mean=float(ev["clip"].mean()), clip_frac_max=float(ev["clip"].max()),
        S1_mean=np.nan, H1_mean=np.nan, S2_mean=np.nan, H2_mean=np.nan, R_mean=np.nan,
        status_1d="unevaluable", status_2d="unevaluable", status_joint="unevaluable")
    for fam in FAMILIES:
        idx = ev["fams"] == fam
        row[f"prr_1d_{fam}"] = float(keep1[idx].mean())
        row[f"prr_2d_{fam}"] = float(keep2[idx].mean())

    arr = {k: np.full(W, np.nan, np.float32) for k in
           ["mu1", "S1", "H1", "mu2", "S2", "H2", "H1j", "H2j", "R", "base1", "base2"]}
    arr["base1"], arr["base2"] = G1[0].astype(np.float32), G2[0].astype(np.float32)

    def put(keys, vals):
        for k, v in zip(keys, vals):
            arr[k][valid] = v

    if n1 >= MIN_ACCEPTED:
        mu, S, H = stability(G1[1:][acc1][:, valid])
        put(["mu1", "S1", "H1"], [mu, S, H])
        row.update(S1_mean=float(S.mean()), H1_mean=float(H.mean()), status_1d="evaluable")
    if n2 >= MIN_ACCEPTED:
        mu, S, H = stability(G2[1:][acc2][:, valid])
        put(["mu2", "S2", "H2"], [mu, S, H])
        row.update(S2_mean=float(S.mean()), H2_mean=float(H.mean()), status_2d="evaluable")
    if nj >= MIN_ACCEPTED:
        _, _, H1j = stability(G1[1:][joint][:, valid])
        _, _, H2j = stability(G2[1:][joint][:, valid])
        R = np.sqrt(H1j * H2j)
        put(["H1j", "H2j", "R"], [H1j, H2j, R])
        row.update(R_mean=float(R.mean()), status_joint="evaluable")

    return dict(row=row, arr=arr, valid=valid, keep1=keep1, keep2=keep2,
                deg1=~ok1[1:], deg2=~ok2[1:], names=ev["names"],
                out1=ev["out1"].astype(np.float32), out2=ev["out2"].astype(np.float32))

# ---------------- 9. save / run ----------------
def save_outputs(records, out_dir, done):
    df = pd.DataFrame([rc["row"] for rc in records])
    df.to_csv(f"{out_dir}/noise_stability_records.csv", index=False)
    np.savez_compressed(
        f"{out_dir}/noise_stability_maps.npz",
        cohort_index=df["cohort_index"].to_numpy(dtype=np.int64),
        filename=np.array(df["ecg_csv_filename"].tolist(), dtype=str),
        pert_names=np.array(records[0]["names"]),
        valid_600=np.stack([rc["valid"] for rc in records]),
        keep_1d=np.stack([rc["keep1"] for rc in records]),
        keep_2d=np.stack([rc["keep2"] for rc in records]),
        degenerate_1d=np.stack([rc["deg1"] for rc in records]),
        degenerate_2d=np.stack([rc["deg2"] for rc in records]),
        out1_base=np.stack([rc["out1"] for rc in records]),
        out2_base=np.stack([rc["out2"] for rc in records]),
        **{k: np.stack([rc["arr"][k] for rc in records]) for k in records[0]["arr"]})
    with open(f"{out_dir}/run_metadata.json", "w") as f:
        json.dump(dict(config=CONFIG, n_records=len(df), done=done,
                       time=datetime.datetime.now().isoformat(timespec="seconds"),
                       tf=tf.__version__, batch_sizes=BS, force_cpu=FORCE_CPU,
                       scaler_n_fit=SCALER_N_FIT,
                       note="PROVISIONAL: scaler fitted on 9,592 records (intended 9,534); 2D rendering "
                            "not fully verified; perturbation magnitudes not validated"), f, indent=2)
    return df

def _atomic_pickle(obj, path):
    with open(path + ".tmp", "wb") as f:
        pickle.dump(obj, f)
    os.replace(path + ".tmp", path)

def run(rows, tag, resume=True):
    assert rows["ecg_csv_filename"].is_unique and rows["cohort_index"].is_unique
    out_dir = os.path.join(OUT, f"module2_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.pkl")
    records, done = [], set()
    if resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        assert d["config"] == CONFIG, f"Config changed since checkpoint; use a new tag or delete {ckpt}"
        records = d["records"]
        done = {rc["row"]["cohort_index"] for rc in records}
        print(f"Resuming: {len(done)} records already done")
    rend, t0, todo = Renderer(), time.time(), 0
    try:
        for r in rows.itertuples(index=False):
            if int(r.cohort_index) in done:
                continue
            x = load_ecg(r.ecg_csv_filename)                   # paired by filename
            assert x.shape == (N,), x.shape
            ev = eval_record(x, np.random.default_rng(int(r.cohort_index)), rend)
            records.append(summarize(r, ev))
            todo += 1
            if todo % CKPT_EVERY == 0:
                _atomic_pickle(dict(config=CONFIG, records=records), ckpt)
                el = time.time() - t0
                print(f"{len(records)}/{len(rows)} | {el / todo:.1f}s/record | "
                      f"~{el / todo * (len(rows) - len(records)) / 60:.0f} min left", flush=True)
    finally:
        plt.close(rend.fig)
        if records:
            _atomic_pickle(dict(config=CONFIG, records=records), ckpt)
    df = save_outputs(records, out_dir, done=len(records))
    print("\n========== MODULE 2 v2 SUMMARY ==========")
    print("Output dir:", out_dir, "| records:", len(df))
    print("Evaluable 1D / 2D / joint:", (df.status_1d == "evaluable").sum(),
          (df.status_2d == "evaluable").sum(), (df.status_joint == "evaluable").sum())
    print("Mean PRR 1D / 2D:", round(df.prr_1d.mean(), 4), round(df.prr_2d.mean(), 4))
    print("Degenerate perturbed maps (total) 1D / 2D:", int(df.n_degenerate_1d.sum()), int(df.n_degenerate_2d.sum()))
    print("Degenerate BASE maps 1D / 2D:", int(df.base_degenerate_1d.sum()), int(df.base_degenerate_2d.sum()))
    print("Base mask coverage: min", round(df.base_coverage.min(), 3), "| bad_mask records:", int(df.bad_mask.sum()))
    print("Batch sizes in use:", BS, "| CPU fallback:", FORCE_CPU)
    print(df.groupby("class_name")[["prr_1d", "prr_2d", "n_joint_accepted", "S1_mean", "S2_mean", "R_mean"]]
          .mean().round(4))
    return df

def selftest():
    x = load_ecg(test.iloc[0]["ecg_csv_filename"])
    rend = Renderer()
    rgb = rend.base(x)
    img = (to_2d(rgb[None]) / 255.0).astype(np.float32)
    print("2D, 1 input :", run_model("2d", m2, img).round(3))
    print("2D, 31 inputs:", run_model("2d", m2, np.repeat(img, 31, 0)).shape)
    print("1D, 1 input :", run_model("1d", m1, ((x - MEAN) / SCALE).astype(np.float32)[None, :, None]).round(3))
    cov = (GEOM_VALID & (trace_mask_at_2d_grid(rgb).sum(0) > 0)).sum() / GEOM_VALID.sum()
    print("Trace-mask coverage on our rendered image (Module 1 requires >= 0.75):", round(float(cov), 3))
    plt.close(rend.fig)

print("Module 2 v2 definitions loaded | geometric valid columns:", int(GEOM_VALID.sum()), "of", W)
