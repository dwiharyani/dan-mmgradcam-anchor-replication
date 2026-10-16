from pathlib import Path
import hashlib

import h5py
import numpy as np
import pandas as pd


ROOT = Path("data/1d/ECGData")
META = Path("data/1d/Diagnostics.xlsx")
EXCLUSIONS = Path("splits/final_exclusions_58.csv")

OUTPUT = Path(
    "data/1d/mm_gradcam_leadII_10588.h5"
)

MANIFEST = Path(
    "splits/final_cohort_10588.csv"
)

RHYTHM_TO_CLASS = {
    "AFIB": "AFib",
    "AF": "AFib",

    "ST": "GSVT",
    "SVT": "GSVT",
    "AT": "GSVT",
    "AVNRT": "GSVT",
    "AVRT": "GSVT",
    "SAAWR": "GSVT",

    "SB": "SB",

    "SR": "SR",
    "SA": "SR",
    "SI": "SR",
}

CLASS_TO_ID = {
    "AFib": 0,
    "GSVT": 1,
    "SB": 2,
    "SR": 3,
}

EXPECTED = {
    "AFib": 2218,
    "GSVT": 2260,
    "SB": 3888,
    "SR": 2222,
}


meta = pd.read_excel(META)

exclude = pd.read_csv(EXCLUSIONS)

exclude_ids = set(
    exclude["patient_id"]
    .astype(str)
    .str.strip()
)

print("=" * 72)
print("MM-GRADCAM FINAL 1D COHORT")
print("=" * 72)

print("Metadata :", len(meta))
print("Excluded :", len(exclude_ids))

assert len(meta) == 10646
assert len(exclude_ids) == 58


signals = []
labels = []
patient_ids = []
rhythms = []
classes = []
manifest_rows = []


for source_row, r in meta.iterrows():

    pid = str(r["FileName"]).strip()
    rhythm = str(r["Rhythm"]).strip().upper()

    if pid in exclude_ids:
        continue

    cls = RHYTHM_TO_CLASS[rhythm]

    path = ROOT / f"{pid}.csv"

    if not path.exists():
        raise RuntimeError(
            f"Missing ECG: {path}"
        )

    df = pd.read_csv(path)

    if len(df) != 5000:
        raise RuntimeError(
            f"{pid}: expected 5000 rows, "
            f"got {len(df)}"
        )

    if "II" not in df.columns:
        raise RuntimeError(
            f"{pid}: Lead II missing"
        )

    x = pd.to_numeric(
        df["II"],
        errors="coerce"
    ).to_numpy(dtype=np.float32)

    if x.shape != (5000,):
        raise RuntimeError(
            f"{pid}: bad shape {x.shape}"
        )

    if not np.isfinite(x).all():
        raise RuntimeError(
            f"{pid}: NaN/Inf in Lead II"
        )

    # IMPORTANT:
    # no scaling
    # no filtering
    # no normalization
    # raw Lead II exactly as stored

    signals.append(x)
    labels.append(CLASS_TO_ID[cls])
    patient_ids.append(pid)
    rhythms.append(rhythm)
    classes.append(cls)

    manifest_rows.append({
        "cohort_index": len(signals) - 1,
        "source_row": source_row,
        "patient_id": pid,
        "rhythm": rhythm,
        "class_name": cls,
        "class_id": CLASS_TO_ID[cls],
        "csv_path": str(path),
    })


X = np.stack(signals).astype(
    np.float32
)

y = np.asarray(
    labels,
    dtype=np.int64
)

manifest = pd.DataFrame(
    manifest_rows
)


print("\nFinal shape:")
print("X:", X.shape)
print("y:", y.shape)

print("\nClass counts:")
counts = manifest[
    "class_name"
].value_counts()

print(counts)


assert X.shape == (10588, 5000)
assert y.shape == (10588,)

for cls, expected in EXPECTED.items():

    actual = int(
        counts.get(cls, 0)
    )

    assert actual == expected, (
        f"{cls}: "
        f"{actual} != {expected}"
    )


# ------------------------------------------------------------
# Save HDF5
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

string_dtype = h5py.string_dtype(
    encoding="utf-8"
)

with h5py.File(
    OUTPUT,
    "w"
) as f:

    f.create_dataset(
        "X",
        data=X,
        dtype="float32"
    )

    f.create_dataset(
        "y",
        data=y,
        dtype="int64"
    )

    f.create_dataset(
        "patient_id",
        data=np.asarray(
            patient_ids,
            dtype=object
        ),
        dtype=string_dtype
    )

    f.create_dataset(
        "original_rhythm",
        data=np.asarray(
            rhythms,
            dtype=object
        ),
        dtype=string_dtype
    )

    f.create_dataset(
        "class_name",
        data=np.asarray(
            classes,
            dtype=object
        ),
        dtype=string_dtype
    )

    f.attrs[
        "source"
    ] = "Chapman-Shaoxing"

    f.attrs[
        "lead"
    ] = "II"

    f.attrs[
        "sampling_frequency_hz"
    ] = 500

    f.attrs[
        "signal_length"
    ] = 5000

    f.attrs[
        "preprocessing"
    ] = "none"

    f.attrs[
        "n_samples"
    ] = 10588


manifest.to_csv(
    MANIFEST,
    index=False
)


# ------------------------------------------------------------
# SHA256
# ------------------------------------------------------------

sha = hashlib.sha256()

with open(OUTPUT, "rb") as fh:

    while True:

        block = fh.read(
            1024 * 1024
        )

        if not block:
            break

        sha.update(block)


digest = sha.hexdigest()


print("\n" + "=" * 72)
print("FINAL AUDIT")
print("=" * 72)

print("Records       :", len(manifest))
print("Signal shape  :", X.shape)
print("Lead          : II")
print("Sampling rate : 500 Hz")
print("Preprocessing : NONE")

print("\nHDF5:")
print(OUTPUT.resolve())

print("\nManifest:")
print(MANIFEST.resolve())

print("\nSHA256:")
print(digest)

print("\nSTATUS: PASS")
