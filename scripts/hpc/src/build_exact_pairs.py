from pathlib import Path
import re
import pandas as pd

COHORT = Path("splits/final_cohort_10588.csv")
IMAGE_ROOT = Path("data/2d")
OUT = Path("splits/mm_pairs.csv")

VALID_CLASSES = {"AFib", "GSVT", "SB", "SR"}
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

cohort = pd.read_csv(COHORT)

assert len(cohort) == 10588
assert cohort["cohort_index"].tolist() == list(range(10588))

images = []

for p in IMAGE_ROOT.rglob("*"):

    if not p.is_file():
        continue

    if p.suffix.lower() not in VALID_EXT:
        continue

    m = re.fullmatch(r"hasta_(\d+)", p.stem)

    if not m:
        continue

    image_id = int(m.group(1))

    image_class = None

    for part in p.parts:
        if part in VALID_CLASSES:
            image_class = part
            break

    if image_class is None:
        raise RuntimeError(
            f"Cannot determine class from path: {p}"
        )

    images.append({
        "cohort_index": image_id,
        "image_class": image_class,
        "image_path": str(p.resolve()),
    })


img = pd.DataFrame(images)

print("=" * 72)
print("IMAGE AUDIT")
print("=" * 72)

print("Images     :", len(img))
print("Unique IDs :", img["cohort_index"].nunique())
print("Min ID     :", img["cohort_index"].min())
print("Max ID     :", img["cohort_index"].max())

print("\nImage class counts:")
print(img["image_class"].value_counts())

assert len(img) == 10588
assert img["cohort_index"].nunique() == 10588
assert img["cohort_index"].min() == 0
assert img["cohort_index"].max() == 10587


pairs = cohort.merge(
    img,
    on="cohort_index",
    how="left",
    validate="one_to_one",
)


print("\n" + "=" * 72)
print("PAIR AUDIT")
print("=" * 72)

print("Pairs:", len(pairs))

missing = pairs["image_path"].isna().sum()

print("Missing images:", missing)

mismatch = pairs[
    pairs["class_name"] != pairs["image_class"]
]

print("Class mismatches:", len(mismatch))

if len(mismatch):

    print("\nFirst mismatches:")

    print(
        mismatch[
            [
                "cohort_index",
                "patient_id",
                "class_name",
                "image_class",
                "image_path",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


assert missing == 0
assert len(mismatch) == 0


print("\nPaired class counts:")
print(pairs["class_name"].value_counts())


expected = {
    "AFib": 2218,
    "GSVT": 2260,
    "SB": 3888,
    "SR": 2222,
}

actual = pairs["class_name"].value_counts().to_dict()

assert actual == expected


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

pairs.to_csv(
    OUT,
    index=False
)


print("\n" + "=" * 72)
print("FINAL MULTIMODAL PAIR AUDIT")
print("=" * 72)

print("1D records     :", len(cohort))
print("2D images      :", len(img))
print("Paired records :", len(pairs))
print("Missing        :", missing)
print("Class mismatch :", len(mismatch))

print("\nOutput:")
print(OUT.resolve())

print("\nSTATUS: PASS")
