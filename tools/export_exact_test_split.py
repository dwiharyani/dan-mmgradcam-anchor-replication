#!/usr/bin/env python3
"""Export a dataset-free exact, identity-checked 1D/2D held-out ECG manifest.

Run from the HPC project root, or set --root. Publishes names of files in the
public Chapman/Figshare and author/Kaggle datasets, NOT patient_id, ECG data,
images, predictions or activation maps. Inspect these public record filenames
against your data sharing requirements before publicly uploading the manifest.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

EXPECTED = {'AFib': 222, 'GSVT': 222, 'SB': 388, 'SR': 222}
COLUMNS = ['test_order_1d', 'test_order_2d', 'cohort_index', 'class_name',
           'ecg_csv_filename', 'image_relative_path']


def require(condition, message):
    if not condition:
        raise SystemExit('STOP: ' + message)


def read_csv(path, needed):
    require(path.is_file(), f'file not found: {path}')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        require(rd.fieldnames is not None, f'empty CSV: {path}')
        missing = sorted(set(needed) - set(rd.fieldnames))
        require(not missing, f'{path.name} missing columns {missing}; found {rd.fieldnames}')
        return list(rd)


def hashed(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def integer(value, label):
    try:
        n = int(str(value))
    except (ValueError, TypeError):
        raise SystemExit(f'STOP: non-integer {label}: {value!r}')
    require(n >= 0, f'negative {label}: {n}')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='MM_GradCAM_HPC root')
    ap.add_argument('--pairs', default='splits/gradcam_test_pairs_verified.csv')
    ap.add_argument('--cohort', default='splits/final_cohort_10588.csv')
    ap.add_argument('--image-root', default='data/2d_author_clean/Testdata')
    ap.add_argument('--out', default='reproducibility/test_split_1054.csv')
    ap.add_argument('--skip-image-existence-check', action='store_true',
                    help='Only if original Kaggle image files are not locally available')
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    pairs_path = root / args.pairs
    cohort_path = root / args.cohort
    img_root = root / args.image_root
    out = root / args.out
    require(root.is_dir(), f'missing HPC root {root}')
    pairs = read_csv(pairs_path, ('cohort_index', 'true_class', 'patient_id',
                                  'cam_row_1d', 'cam_row_2d'))
    cohort = read_csv(cohort_path, ('cohort_index', 'patient_id', 'class_name', 'csv_path'))
    require(len(pairs) == 1054, f'expected 1054 verified test pairs, got {len(pairs)}')
    require(len(cohort) == 10588, f'expected 10588 final cohort records, got {len(cohort)}')
    cohort_by_key = {}
    for row in cohort:
        idx = integer(row['cohort_index'], 'cohort_index')
        require(idx not in cohort_by_key, f'duplicate cohort index {idx}')
        cohort_by_key[idx] = row
    out_rows, found_indexes, found_names = [], set(), set()
    counts = Counter()
    for p in pairs:
        idx = integer(p['cohort_index'], 'pair cohort_index')
        require(idx not in found_indexes, f'duplicate test cohort index {idx}')
        found_indexes.add(idx)
        require(idx in cohort_by_key, f'test cohort index {idx} missing from final cohort')
        c = cohort_by_key[idx]
        cls = p['true_class'].strip()
        require(cls in EXPECTED, f'unknown class {cls}')
        require(c['class_name'].strip() == cls, f'class mismatch at cohort {idx}')
        require(c['patient_id'].strip() == p['patient_id'].strip(),
                f'identity mismatch between pair and cohort at {idx}')
        csv_name = Path(c['csv_path'].replace('\\', '/')).name
        require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+\.csv', csv_name, flags=re.I)),
                f'unexpected ECG source filename at cohort {idx}: {csv_name}')
        require(csv_name not in found_names, f'duplicate ECG CSV filename {csv_name}')
        found_names.add(csv_name)
        image_rel = f'{cls}/hasta_{idx}.png'
        if not args.skip_image_existence_check:
            require((img_root / image_rel).is_file(),
                    f'2D image missing: {img_root / image_rel}')
        if p.get('image_path_2d', '').strip():
            actual_path = p['image_path_2d'].replace('\\', '/')
            require(actual_path.endswith(image_rel),
                    f'2D image mapping mismatch for cohort {idx}: {actual_path}')
        out_rows.append({
            'test_order_1d': integer(p['cam_row_1d'], 'cam_row_1d'),
            'test_order_2d': integer(p['cam_row_2d'], 'cam_row_2d'),
            'cohort_index': idx,
            'class_name': cls,
            'ecg_csv_filename': csv_name,
            'image_relative_path': image_rel,
        })
        counts[cls] += 1
    require(dict(counts) == EXPECTED, f'incorrect class counts: {dict(counts)}')
    for col in ['test_order_1d', 'test_order_2d']:
        require(sorted(r[col] for r in out_rows) == list(range(1054)),
                f'{col} not 0..1053 permutation')
    out_rows.sort(key=lambda r: r['test_order_1d'])
    require(out_rows[0]['test_order_1d'] == 0, 'test order must start at 0')
    require(not out.is_symlink(), 'refusing to write via symlink')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(out_rows)
    meta = {
        'schema_version': 1,
        'status': 'independent_anchor_replication_exact_test_split',
        'test_records': len(out_rows),
        'class_counts': EXPECTED,
        'sha256_csv': hashed(out),
        'manifest_columns': COLUMNS,
        'image_name_rule': '{class_name}/hasta_{cohort_index}.png',
        'notes': [
            'Public source filenames only; no ECG signal, ECG image, patient_id, or predictions included.',
            'Test order 1D/2D may differ. Do NOT match representations by row position.',
            'Preprocessing, class mapping and pretrained weights must also match for metric reproduction.',
            'The 727 primary analysis subset requires predictions and temporal QC, not just this manifest.',
        ],
    }
    meta_out = out.with_name(out.stem + '_metadata.json')
    meta_out.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print('EXPORT PASSED | exact pairs:', len(out_rows), '| counts:', dict(counts))
    print('Manifest:', out)
    print('SHA256:', meta['sha256_csv'])
    print('Metadata:', meta_out)
    print('Dataset files copied: ZERO')


if __name__ == '__main__':
    main()
