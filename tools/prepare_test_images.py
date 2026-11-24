#!/usr/bin/env python3
"""Create a LOCAL exact 1054-image Testdata view from downloaded Kaggle images.

Never downloads or uploads images. Output must be inside ignored data/ folder.
Default uses relative symlinks, avoiding 1054 duplicate image copies.
"""
import argparse
import csv
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='reproducibility/test_split_1054.csv')
    ap.add_argument('--kaggle-root', required=True,
                    help='Unzipped Kaggle author image dataset root')
    ap.add_argument('--out', default='data/2d_test/Testdata')
    ap.add_argument('--mode', choices=['symlink', 'copy'], default='symlink')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    source = Path(args.kaggle_root).resolve()
    if not source.is_dir():
        raise SystemExit('Kaggle input root not found: ' + str(source))
    if 'data' not in out.parts:
        raise SystemExit('STOP: output must be inside local data/ folder (Git-ignored)')
    with Path(args.manifest).open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1054:
        raise SystemExit('STOP: manifest must have 1054 rows')
    wanted = {r['image_relative_path'] for r in rows}
    found = {}
    for f in source.rglob('*.png'):
        if f.name.startswith('hasta_'):
            # Parent folder identifies original rhythm class; ignore test/train layout.
            cls = f.parent.name
            key = f'{cls}/{f.name}'
            if key in wanted:
                if key in found:
                    raise SystemExit('STOP: ambiguous duplicate image ' + key)
                found[key] = f
    absent = sorted(wanted - set(found))
    if absent:
        raise SystemExit(f'STOP: missing {len(absent)} Kaggle images, e.g. {absent[:5]}')
    for rel in sorted(wanted):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        f = found[rel]
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() and dest.resolve() == f:
                continue
            raise SystemExit('STOP: destination already exists: ' + str(dest))
        if args.mode == 'symlink':
            dest.symlink_to(Path(os.path.relpath(f, dest.parent)))
        else:
            shutil.copy2(f, dest)
    print(f'PASS: prepared exactly {len(wanted)} local ECG test images at {out}')
    print('No images added to Git; keep data/ ignored.')


if __name__ == '__main__':
    main()
