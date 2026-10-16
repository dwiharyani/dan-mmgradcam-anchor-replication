#!/usr/bin/env python3
"""Audit published test manifest and optionally locate source ECG/image files.

No ML package dependency. Does NOT infer preprocessing or claim evaluation accuracy.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED = {'AFib': 222, 'GSVT': 222, 'SB': 388, 'SR': 222}
FIELDS = ['test_order_1d', 'test_order_2d', 'cohort_index',
          'class_name', 'ecg_csv_filename', 'image_relative_path']


def fail(msg):
    raise SystemExit('FAIL: ' + msg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='reproducibility/test_split_1054.csv')
    p.add_argument('--metadata', default='reproducibility/test_split_1054_metadata.json')
    p.add_argument('--ecg-csv-dir', help='Folder with downloaded Figshare CSVs, recursively searchable')
    p.add_argument('--image-root', help='Author Kaggle image root, e.g. Testdata/')
    args = p.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        fail(f'manifest not found: {manifest}. The owner must export it from HPC first.')
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            fail(f'manifest columns differ from schema: {reader.fieldnames}')
        rows = list(reader)
    if len(rows) != 1054:
        fail(f'expected 1054 ECG records, received {len(rows)}')
    cnt = Counter(r['class_name'] for r in rows)
    if dict(cnt) != EXPECTED:
        fail(f'class distribution not expected: {cnt}')
    if any(set(r) != set(FIELDS) for r in rows):
        fail('bad manifest row')
    for col in ('test_order_1d','test_order_2d','cohort_index'):
        try:
            indices = [int(r[col]) for r in rows]
        except ValueError:
            fail('non-integer ' + col)
        if len(indices) != len(set(indices)):
            fail('duplicate ' + col)
        if col.startswith('test_order') and sorted(indices) != list(range(1054)):
            fail(f'{col} is not a complete 0..1053 permutation')
    names = set()
    for r in rows:
        idx, cls = int(r['cohort_index']), r['class_name']
        im = f'{cls}/hasta_{idx}.png'
        if r['image_relative_path'] != im:
            fail(f'inconsistent 2D path: {r}')
        name = r['ecg_csv_filename']
        if ('/' in name or '\\' in name or not name.lower().endswith('.csv')):
            fail(f'invalid 1D filename: {name}')
        if name in names:
            fail(f'duplicate 1D CSV filename: {name}')
        names.add(name)
    meta = Path(args.metadata)
    if meta.is_file():
        metadata = json.loads(meta.read_text(encoding='utf-8'))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != metadata.get('sha256_csv'):
            fail(f'manifest SHA256 mismatch: expected {metadata.get("sha256_csv")} got {digest}')
        print('Manifest checksum: PASS')
    else:
        print('WARNING: metadata JSON absent; checksum not verified')
    print('Manifest: PASS | 1054 test samples | AFib 222, GSVT 222, SB 388, SR 222')
    if args.ecg_csv_dir:
        base = Path(args.ecg_csv_dir)
        if not base.is_dir():
            fail(f'ECG csv dir does not exist: {base}')
        files = {}
        for pth in base.rglob('*.csv'):
            if pth.name in names:
                if pth.name in files:
                    fail(f'duplicate source CSV basename: {pth.name}')
                files[pth.name] = pth
        missing = sorted(names - set(files))
        print(f'1D Figshare source files: {len(files)}/1054 located')
        if missing:
            print('First missing 1D CSVs:', missing[:8])
            fail('missing input files, download/extract Figshare dataset')
    if args.image_root:
        base = Path(args.image_root)
        if not base.is_dir():
            fail(f'2D root not found: {base}')
        options = [base, base / 'Testdata']
        candidates = [root for root in options if root.is_dir()]
        matches = [(r, root / r['image_relative_path'])
                   for r in rows for root in candidates
                   if (root / r['image_relative_path']).is_file()]
        found = {r['cohort_index'] for r,_ in matches}
        missing = [r['image_relative_path'] for r in rows if r['cohort_index'] not in found]
        print(f'2D Kaggle source files: {len(found)}/1054 located')
        if missing:
            print('First missing 2D images:', missing[:8])
            fail('missing image files, download/extract author Kaggle image dataset')
    print('VERIFY COMPLETE: exact split is reconstructable from public source files.')
    print('NOTE: Identical predictions also require author-equivalent preprocessing and class mapping.')


if __name__ == '__main__':
    main()
