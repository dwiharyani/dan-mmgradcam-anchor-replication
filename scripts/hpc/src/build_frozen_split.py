from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PAIRS = Path("splits/mm_pairs.csv")
OUTPUT = Path("splits/frozen_split_author_code.csv")


CLASS_ORDER = [
    "AFib",
    "GSVT",
    "SB",
    "SR",
]

TEST_COUNTS = {
    "AFib": 222,
    "GSVT": 222,
    "SB": 388,
    "SR": 222,
}

EXPECTED_TOTAL = {
    "AFib": 2218,
    "GSVT": 2260,
    "SB": 3888,
    "SR": 2222,
}


df = pd.read_csv(PAIRS)

assert len(df) == 10588
assert df["cohort_index"].nunique() == 10588


# ============================================================
# 1. AUTHOR TEST SET
#    Notebook logic:
#    class_indices = np.where(y_true == class_label)[0]
#    test_indices = class_indices[:N]
# ============================================================

test_parts = []
train_pool_parts = []

for cls in CLASS_ORDER:

    cls_df = (
        df[df["class_name"] == cls]
        .sort_values("cohort_index")
        .copy()
    )

    assert len(cls_df) == EXPECTED_TOTAL[cls]

    n_test = TEST_COUNTS[cls]

    test_cls = cls_df.iloc[:n_test].copy()
    pool_cls = cls_df.iloc[n_test:].copy()

    test_parts.append(test_cls)
    train_pool_parts.append(pool_cls)


# The notebook concatenates classes in this order:
# AFib -> GSVT -> SB -> SR
test_df = pd.concat(
    test_parts,
    ignore_index=True
)

pool_df = pd.concat(
    train_pool_parts,
    ignore_index=True
)


assert len(test_df) == 1054
assert len(pool_df) == 9534


# ============================================================
# 2. AUTHOR TRAIN / VALIDATION SPLIT
#
# Notebook:
# train_test_split(
#     X_train_scaled,
#     y_train_encoded,
#     test_size=0.111,
#     random_state=42
# )
#
# We split ROW INDICES here so the exact patient membership is
# frozen before loading any signals or images.
# ============================================================

pool_indices = np.arange(len(pool_df))

train_idx, val_idx = train_test_split(
    pool_indices,
    test_size=0.111,
    random_state=42,
    shuffle=True,
)

train_df = pool_df.iloc[train_idx].copy()
val_df = pool_df.iloc[val_idx].copy()


# ============================================================
# 3. LABEL SPLITS
# ============================================================

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# Useful order marker
train_df["split_order"] = np.arange(len(train_df))
val_df["split_order"] = np.arange(len(val_df))
test_df["split_order"] = np.arange(len(test_df))


out = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True,
)


# ============================================================
# 4. AUDIT
# ============================================================

print("=" * 72)
print("FROZEN AUTHOR-CODE SPLIT")
print("=" * 72)

print("\nOverall:")
print(out["split"].value_counts())

print("\nTest class counts:")
print(
    test_df["class_name"]
    .value_counts()
    .reindex(CLASS_ORDER)
)

print("\nValidation class counts:")
print(
    val_df["class_name"]
    .value_counts()
    .reindex(CLASS_ORDER)
)

print("\nTrain class counts:")
print(
    train_df["class_name"]
    .value_counts()
    .reindex(CLASS_ORDER)
)


# Expected exact sizes from sklearn train_test_split
assert len(train_df) == 8475
assert len(val_df) == 1059
assert len(test_df) == 1054

expected_test = {
    "AFib": 222,
    "GSVT": 222,
    "SB": 388,
    "SR": 222,
}

actual_test = (
    test_df["class_name"]
    .value_counts()
    .to_dict()
)

assert actual_test == expected_test


# Every patient exactly once
assert len(out) == 10588
assert out["patient_id"].nunique() == 10588
assert out["cohort_index"].nunique() == 10588


# 1D/2D already paired
assert out["image_path"].notna().all()
assert (
    out["class_name"] == out["image_class"]
).all()


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUTPUT,
    index=False
)


# Separate convenient files
train_df.to_csv(
    "splits/train_author_code.csv",
    index=False
)

val_df.to_csv(
    "splits/val_author_code.csv",
    index=False
)

test_df.to_csv(
    "splits/test_author_code.csv",
    index=False
)


print("\n" + "=" * 72)
print("EXPECTED NOTEBOOK SIZES")
print("=" * 72)

print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))
print("Total      :", len(out))

print("\nOutput:")
print(OUTPUT.resolve())

print("\nSTATUS: PASS")
