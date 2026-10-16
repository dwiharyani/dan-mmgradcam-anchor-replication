#!/usr/bin/env python3
"""Verify exact 1D/2D test-record identity and preserve per-array row indices.

Run from MM_GradCAM_HPC project root. Never overwrites splits/mm_pairs.csv.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

SIGNAL = Path('results/gradcam_1d_author_code/gradcam_1d_manifest.csv')
IMAGE = Path('results/gradcam_2d_author_code/gradcam_2d_manifest.csv')
CAM1 = Path('results/gradcam_1d_author_code/heatmaps_1d.npy')
CAM2 = Path('results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
OUT = Path('splits/gradcam_test_pairs_verified.csv')


def check(condition, message):
    if not condition:
        raise SystemExit('FAIL: ' + message)


def inspect_keys(df, label, idx_col):
    for col in ('patient_id', 'cohort_index', 'true_class', idx_col):
        check(col in df.columns, f'{label}: missing column {col}')
    df['patient_id'] = df['patient_id'].astype('string').str.strip()
    df['true_class'] = df['true_class'].astype('string').str.strip()
    check(bool(df['patient_id'].notna().all() and df['patient_id'].str.len().gt(0).all()),
          f'{label}: empty patient IDs')
    for col in ('cohort_index', idx_col):
        df[col] = pd.to_numeric(df[col], errors='raise').astype('int64')
    check(bool(df[['patient_id', 'cohort_index', idx_col]].notna().all().all()),
          f'{label}: missing keys')
    check(bool(df['patient_id'].is_unique), f'{label}: duplicate patient IDs')
    check(bool(df['cohort_index'].is_unique), f'{label}: duplicate cohort indices')
    check(bool(df[idx_col].is_unique), f'{label}: duplicate heatmap row indices')
    check(sorted(df[idx_col].tolist()) == list(range(len(df))),
          f'{label}: heatmap row indices are not exactly 0..N-1')
    check(bool(df['true_class'].isin(['AFib', 'GSVT', 'SB', 'SR']).all()),
          f'{label}: unknown class labels')


def main():
    for f in (SIGNAL, IMAGE, CAM1, CAM2):
        check(f.is_file(), f'missing file: {f}')
    s = pd.read_csv(SIGNAL)
    i = pd.read_csv(IMAGE)
    inspect_keys(s, '1D', 'test_order')
    inspect_keys(i, '2D', 'row_index')
    shape1 = np.load(CAM1, mmap_mode='r').shape
    shape2 = np.load(CAM2, mmap_mode='r').shape
    check(shape1 == (len(s), 5000), f'1D CAM shape {shape1} does not match ({len(s)}, 5000)')
    check(shape2 == (len(i), 300, 600), f'2D CAM shape {shape2} does not match ({len(i)}, 300, 600)')
    print('1D manifest:', len(s), '| heatmaps:', shape1)
    print('2D manifest:', len(i), '| heatmaps:', shape2)

    # Confirm the path used for a 2D image agrees with its cohort and class.
    if 'image_filename' in i.columns:
        names = i['image_filename'].astype(str).str.replace('\\', '/', regex=False)
        extracted = names.str.extract(r'(?:^|/)hasta_(\d+)\.[^.]+$')[0]
        check(bool(extracted.notna().all()), '2D image filenames do not all match hasta_<index>.<ext>')
        check(bool((extracted.astype('int64') == i['cohort_index']).all()),
              '2D image filename cohort IDs do not match manifest')
        classes = names.str.extract(r'(?:^|/)(AFib|GSVT|SB|SR)/hasta_\d+\.[^.]+$')[0]
        check(bool(classes.notna().all() and (classes == i['true_class']).all()),
              '2D image filename classes do not match manifest true_class')
        print('2D filename -> cohort_index/class: PASS')

    # Join on cohort_index, and independently confirm patient ID + class.
    pairs = s.merge(i, on='cohort_index', how='inner', suffixes=('_1d', '_2d'), validate='one_to_one')
    print('Matched cohort_index count:', len(pairs))
    missing_in_2d = s.loc[~s['cohort_index'].isin(i['cohort_index']),
                          ['cohort_index', 'patient_id', 'true_class']]
    missing_in_1d = i.loc[~i['cohort_index'].isin(s['cohort_index']),
                          ['cohort_index', 'patient_id', 'true_class']]
    if len(missing_in_2d) or len(missing_in_1d):
        print('Missing from 2D test:', len(missing_in_2d))
        print(missing_in_2d.head(5).to_string(index=False))
        print('Missing from 1D test:', len(missing_in_1d))
        print(missing_in_1d.head(5).to_string(index=False))
        raise SystemExit('STOP: test cohorts differ; do not align by row or class order')
    check(len(pairs) == len(s) == len(i) == 1054, 'Expected 1054 exact pairs; inspect the cohort')
    wrong_id = pairs['patient_id_1d'] != pairs['patient_id_2d']
    wrong_class = pairs['true_class_1d'] != pairs['true_class_2d']
    if bool(wrong_id.any() or wrong_class.any()):
        cols = ['cohort_index', 'patient_id_1d', 'patient_id_2d', 'true_class_1d', 'true_class_2d']
        print(pairs.loc[wrong_id | wrong_class, cols].head(10).to_string(index=False))
        raise SystemExit('STOP: patient IDs or true classes differ')
    pairs = pairs.sort_values('test_order').reset_index(drop=True)

    out = pd.DataFrame({
        'patient_id': pairs['patient_id_1d'],
        'cohort_index': pairs['cohort_index'].astype('int64'),
        'true_class': pairs['true_class_1d'],
        'cam_row_1d': pairs['test_order'].astype('int64'),
        'cam_row_2d': pairs['row_index'].astype('int64'),
        'pred_class_1d': pairs['pred_class'] if 'pred_class' in pairs else '',
        'pred_class_2d': pairs['pred_from_model'] if 'pred_from_model' in pairs else '',
        'image_path_2d': pairs['image_path'] if 'image_path' in pairs else '',
    })
    check(bool(out['patient_id'].is_unique and out['cohort_index'].is_unique),
          'Paired outputs must be one-to-one')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    check(not OUT.exists(), f'{OUT} already exists; refusing to overwrite')
    out.to_csv(OUT, index=False)
    print('\nSTATUS: PASS — all 1054 ECG records matched on cohort, patient ID, and class')
    print('Different 1D vs 2D heatmap row indices:', int((out['cam_row_1d'] != out['cam_row_2d']).sum()))
    if 'pred_match_manifest' in i.columns:
        mismatch = i['pred_match_manifest'].astype(str).str.lower().eq('false').sum()
        print('Pre-existing 2D prediction/manifest mismatches:', int(mismatch))
    print('Saved:', OUT)
    print(out.head(5).to_string(index=False))
    print('NEXT: inspect image x-axis/time mapping BEFORE resampling the 2D heatmap')


if __name__ == '__main__':
    main()
