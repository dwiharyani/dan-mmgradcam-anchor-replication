"""Rebuild the 1,059 validation records by matching validation images to ECG renders.

Exported from the Module 2 Kaggle notebook. Paths below default to a local layout
(data/ECGData, data/ImageData-label2/label1, models/, results/module2/); edit them to match your workspace.
Scaler is PROVISIONAL (fit on 9,592 records; original cohort 9,534). 2D images are re-rendered (see docs/MODULE2_NOISE_STABILITY.md).
Exploratory; not a faithfulness claim. No data or models are included in this repository.
"""
from module2_common import *   # defines paths, test manifest, models, perturbation/render/Grad-CAM functions

# ---- notebook cell 12 ----
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

L2 = Path("data/ImageData-label2/label1")
img = pd.DataFrame([dict(folder=f.relative_to(L2).parts[0], class_name=f.parent.name,
                         cohort_index=int(re.search(r"hasta_(\d+)", f.name).group(1)))
                    for f in L2.rglob("*.png")])
print(img.groupby(["folder", "class_name"]).size().unstack(fill_value=0))
print("unique cohort indices:", img.cohort_index.nunique(), "of", len(img))

train = pd.read_csv(f"{REPO_DIR}/reproducibility/train_split.csv")
tr_ids, te_ids = set(train.cohort_index), set(test.cohort_index)
te_img = set(img[img.folder == "Testdata"].cohort_index)
tr_img = set(img[img.folder == "Traindata"].cohort_index)
print("\nTestdata ids == your test records:", te_img == te_ids, "| overlap:", len(te_img & te_ids))
print("Traindata ids:", len(tr_img), "| contain all train_split ids:", tr_ids <= tr_img)
val_ids = sorted(tr_img - tr_ids)
print("Traindata minus train_split (candidate validation ids):", len(val_ids))

known = pd.concat([train[["cohort_index", "class_name", "ecg_csv_filename"]],
                   test[["cohort_index", "class_name", "ecg_csv_filename"]]])
print("\nIs cohort_index tied to filename order? (known pairs = train + test)")
for cls, g in known.groupby("class_name"):
    r = spearmanr(g.cohort_index, g.ecg_csv_filename.rank()).statistic
    print(f"  {cls}: n={len(g)}  spearman(cohort_index, filename order) = {r:.3f}")
print("cohort_index range per class:", known.groupby("class_name").cohort_index.agg(["min", "max"]).to_dict("index"))

# ---- notebook cell 13 ----
# 12. Rebuild the 1,059 validation records: match each validation image to its ECG file
# Known so far (verified): Traindata ids = 9,534 = your 8,475 train + 1,059 validation. The ECG FILENAMES of the
# 1,059 are unknown, and cohort_index is not tied to filename order. Here we match images to ECG renders instead.
import re, warnings
from pathlib import Path
from PIL import Image
from scipy.optimize import linear_sum_assignment

L2 = Path("data/ImageData-label2/label1")
imgs = pd.DataFrame([dict(folder=f.relative_to(L2).parts[0], class_name=f.parent.name,
                          cohort_index=int(re.search(r"hasta_(\d+)", f.name).group(1)))
                     for f in L2.rglob("*.png")])
train = pd.read_csv(f"{REPO_DIR}/reproducibility/train_split.csv")
val_img = imgs[(imgs.folder == "Traindata") & ~imgs.cohort_index.isin(train.cohort_index)].reset_index(drop=True)
assert len(val_img) == 1059, len(val_img)
leftover = sorted(set(ecg_files) - set(train.ecg_csv_filename) - set(test.ecg_csv_filename))
print("validation images:", len(val_img), "| leftover ECG files:", len(leftover), "(= 1,059 + 58 ECGs without an image)")
assert len(leftover) == 1117, len(leftover)

N_KNOWN_CHECK = 300
BINS = slice(12, 204)                       # 5-px bins covering image columns 58..1021 (the waveform)
def ink_profile(rgb):
    im = np.asarray(rgb)[..., :3].astype(np.int16)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    ink = (B - R >= 14) & (B - G >= 5) & (G - R >= 4) & (R < 240) & (G < 250)     # Module 1's ink rule
    cnt = ink.sum(0)
    ctr = np.where(cnt > 0, (ink * np.arange(ink.shape[0])[:, None]).sum(0) / np.maximum(cnt, 1), np.nan)
    ctr = ctr[: (len(ctr) // 5) * 5].reshape(-1, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(ctr, 1)[BINS]
    ok = np.isfinite(v)
    if ok.sum() < 20:
        return None
    v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    v = v - v.mean()
    s = v.std()
    return v / s if s > 0 else None

def profiles(items, loader):
    out, bad = [], []
    for k in items:
        p = loader(k)
        (out if p is not None else bad).append(p if p is not None else k)
    return out, bad

rend = Renderer()
ZERO = np.zeros(BINS.stop - BINS.start)
render_raw = lambda fn: ink_profile(rend.base(load_ecg(fn)))
render_prof = lambda fn: (lambda p: ZERO if p is None else p)(render_raw(fn))
real_prof = lambda path: (lambda p: ZERO if p is None else p)(ink_profile(np.asarray(Image.open(path).convert("RGB"))))
cls_dir = lambda c, i: f"{IMG_ROOT}/{c}/hasta_{i}.png"

# ---- A. validation of the method on KNOWN pairs (train): can we find the right image among 1,359? ----
known = train.sample(N_KNOWN_CHECK, random_state=0).reset_index(drop=True)
Pk_ecg = np.stack([render_prof(f) for f in known.ecg_csv_filename])
Pk_img = np.stack([real_prof(f"{IMG_ROOT}/{p}") for p in known.image_relative_path])
Pv_img = np.stack([real_prof(cls_dir(c, i)) for c, i in zip(val_img.class_name, val_img.cohort_index)])
pool_img = np.concatenate([Pk_img, Pv_img])                         # candidates: known + validation images
corr = Pk_ecg @ pool_img.T / Pk_ecg.shape[1]
best = corr.argmax(1)
acc = float((best == np.arange(len(known))).mean())
srt = np.sort(corr, 1)
print(f"\nKNOWN-PAIR CHECK: top-1 accuracy over {len(known)} train ECGs = {acc:.4f}")
print("  correct-pair correlation: median", round(float(np.median(corr[np.arange(len(known)), np.arange(len(known))])), 3),
      "| gap to runner-up (min / p5):", round(float((srt[:, -1] - srt[:, -2]).min()), 3),
      round(float(np.quantile(srt[:, -1] - srt[:, -2], .05)), 3))

if acc < 0.99:
    print("\nMATCHING NOT RELIABLE (accuracy < 0.99). Do NOT use it - ask Dwi for val_author_code.csv instead.")
else:
    # ---- B. match the 1,117 leftover ECGs to the 1,059 validation images (one-to-one) ----
    Pl_ecg, bad_ecg = profiles(leftover, render_raw)
    ok_files = [f for f in leftover if f not in set(bad_ecg)]
    Pl_ecg = np.stack(Pl_ecg)
    C = 1 - Pl_ecg @ Pv_img.T / Pl_ecg.shape[1]                     # cost = 1 - Pearson r
    r, c = linear_sum_assignment(C)
    m = pd.DataFrame(dict(ecg_csv_filename=[ok_files[i] for i in r],
                          cohort_index=val_img.cohort_index.to_numpy()[c],
                          class_name=val_img.class_name.to_numpy()[c], cost=C[r, c]))
    row_best = C.argmin(1)[r] == c                                   # does Hungarian agree with the plain nearest image?
    s2 = np.sort(C[r], 1)
    m["agrees_with_nearest"] = row_best
    m["margin"] = s2[:, 1] - s2[:, 0]
    thr = float(np.quantile(1 - corr[np.arange(len(known)), np.arange(len(known))], .99))
    m["confident"] = (m.cost <= thr) & m.agrees_with_nearest
    unmatched = sorted(set(ok_files) - set(m.ecg_csv_filename)) + list(bad_ecg)
    print(f"\nMATCHED {len(m)} validation records | unmatched ECGs: {len(unmatched)} (expected 58)")
    print("  confident:", int(m.confident.sum()), "| agree with nearest image:", int(m.agrees_with_nearest.sum()),
          "| cost p50/p95/max:", np.round(np.quantile(m.cost, [.5, .95, 1]), 3), "| known-pair cost p99:", round(thr, 3))
    print("  class counts:", m.class_name.value_counts().to_dict())
    m.sort_values("cohort_index").to_csv(f"{OUT}/val_reconstruction.csv", index=False)
    pd.Series(unmatched, name="ecg_csv_filename").to_csv(f"{OUT}/ecg_without_image.csv", index=False)
    print("saved:", f"{OUT}/val_reconstruction.csv", "and", f"{OUT}/ecg_without_image.csv")
plt.close(rend.fig)
