from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
from .common import CLASS_NAMES, canonical_label

IMAGE_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
SPLIT_ALIASES = {
    "train":"train", "training":"train",
    "val":"val", "valid":"val", "validation":"val",
    "test":"test", "testing":"test",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", default="splits/image_manifest.csv")
    p.add_argument("--id-regex", default=r"(\d+)")
    args = p.parse_args()

    root = Path(args.root)
    rows = []
    rx = re.compile(args.id_regex)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        split = ""
        class_token = None
        if len(parts) >= 3 and parts[0].lower() in SPLIT_ALIASES:
            split = SPLIT_ALIASES[parts[0].lower()]
            class_token = parts[1]
        elif len(parts) >= 2:
            # Look upward for a recognized class directory.
            for token in reversed(parts[:-1]):
                try:
                    canonical_label(token)
                    class_token = token
                    break
                except Exception:
                    pass
        if class_token is None:
            print("SKIP (no recognized class directory):", path)
            continue
        try:
            label = canonical_label(class_token)
        except Exception:
            print("SKIP (unknown class):", path)
            continue
        m = rx.search(path.stem)
        patient_id = m.group(1) if m else ""
        rows.append({
            "image_path": str(path.resolve()),
            "split": split,
            "label": label,
            "class_name": CLASS_NAMES[label],
            "patient_id": patient_id,
            "filename": path.name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable images found.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.groupby(["split","class_name"], dropna=False).size())
    print("Saved:", args.output, "rows:", len(df))

if __name__ == "__main__":
    main()
