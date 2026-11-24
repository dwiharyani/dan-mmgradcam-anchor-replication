#!/usr/bin/env python3
"""Batch MM-GradCAM 1D-2D ECG visual alignment for all verified test records.

Designed for the already-produced KFUPM HPC 1D/2D Grad-CAM arrays and exact
patient-level pairing. No model training, no Grad-CAM recomputation, no GPU.

Outputs:
 * combined/: ONE ECG waveform with 1D red saliency dots + 2D color CAM
 * figures/: optional annotated comparisons (representative / all / none)
 * alignment_manifest.csv: record/array index/target/audit/output paths
 * cam1d_on_2d_x_600_PROVISIONAL.npy: raw 1D CAM on the 2D horizontal grid
 * waveform_mask_600.npy: columns within the visible ECG trace region
 * summary.json: provenance and scientific validity warnings

Do NOT call the numerical registration 'validated temporal alignment' until
ECG-image rendering code confirms linear t=0..10s across x=58..1021.
For 63 prediction-discordant records, Grad-CAM targets differ by class.

The visual renderer is imported from the previously reviewed script
`run_anchor_mmgradcam_paper_figure.py`, shipped alongside this package.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import run_anchor_mmgradcam_paper_figure as visuals

CLASSES = ('AFib', 'GSVT', 'SB', 'SR')
NEEDED = ('patient_id', 'cohort_index', 'true_class', 'cam_row_1d',
          'cam_row_2d', 'pred_class_1d', 'pred_class_2d', 'image_path_2d')


def as_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else root / p


def safe_stem(r) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{r.true_class}_{r.patient_id}')


def make_selected_indices(pairs: pd.DataFrame, mode: str) -> set:
    """Choose QC figures, NOT the processed ECG sample set."""
    if mode == 'none':
        return set()
    if mode == 'all':
        return set(range(len(pairs)))
    selected = set()
    for cls in CLASSES:
        group = pairs[pairs.true_class == cls]
        correct = group[(group.pred_class_1d == cls) & (group.pred_class_2d == cls)]
        discordant_or_error = group[(group.pred_class_1d != cls) | (group.pred_class_2d != cls)]
        if not correct.empty:
            selected.add(int(correct.index[0]))
        if not discordant_or_error.empty:
            selected.add(int(discordant_or_error.index[0]))
    return selected


def read_inputs(args, root: Path):
    pairs = pd.read_csv(as_path(root, args.pairs))
    absent = set(NEEDED) - set(pairs.columns)
    if absent:
        raise ValueError('Verified pairs missing fields: ' + ', '.join(sorted(absent)))
    expected = args.expected
    if len(pairs) != expected:
        raise ValueError(f'Expected {expected} pairs, found {len(pairs)}')
    if pairs['patient_id'].isna().any() or pairs['patient_id'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing patient IDs')
    if pairs.patient_id.duplicated().any() or pairs.cohort_index.duplicated().any():
        raise ValueError('Patient/cohort IDs are not unique')
    if not pairs.true_class.isin(CLASSES).all():
        raise ValueError('Unknown true ECG class')
    if not pairs.pred_class_1d.isin(CLASSES).all() or not pairs.pred_class_2d.isin(CLASSES).all():
        raise ValueError('Unknown model prediction class')
    c1 = np.load(as_path(root, args.cam1), mmap_mode='r', allow_pickle=False)
    c2 = np.load(as_path(root, args.cam2), mmap_mode='r', allow_pickle=False)
    if c1.shape != (expected, 5000) or c2.shape != (expected, 300, 600):
        raise ValueError(f'Expected ({expected},5000) and ({expected},300,600); got {c1.shape}, {c2.shape}')
    for name in ('cam_row_1d', 'cam_row_2d'):
        indices = pd.to_numeric(pairs[name], errors='raise').astype(int).to_numpy()
        if sorted(indices.tolist()) != list(range(expected)):
            raise ValueError(f'{name} must be a permutation of 0..{expected-1}')
    for name in ('cohort_index', 'cam_row_1d', 'cam_row_2d'):
        pairs[name] = pairs[name].astype(int)
    mf1 = as_path(root, args.manifest1)
    mf2 = as_path(root, args.manifest2)
    if args.strict_manifests and (not mf1.exists() or not mf2.exists()):
        raise FileNotFoundError('Source Grad-CAM manifests required for patient ID and confidence audit')
    m1 = pd.read_csv(mf1) if mf1.is_file() else None
    m2 = pd.read_csv(mf2) if mf2.is_file() else None
    if m1 is not None and len(m1) != expected:
        raise ValueError('1D CAM manifest length mismatch')
    if m2 is not None and len(m2) != expected:
        raise ValueError('2D CAM manifest length mismatch')
    return pairs.reset_index(drop=True), c1, c2, m1, m2


def provisional_1d_cam_on_2d(cam1: np.ndarray, *, w=1080, left=58, right=1021, target_w=600):
    """Keep raw CAM values: explicit, provisional x-position registration.

    This assumes (not yet verified) t=0..10s linear across original waveform
    area, then positions registered CAM samples on the 600-column 2D grid.
    """
    x_on_orig = (np.arange(target_w, dtype=np.float64) + .5) * w / target_w - .5
    mask = (x_on_orig >= left) & (x_on_orig <= right)
    t = np.clip((x_on_orig - left) / (right - left), 0., 1.)
    sample_positions = np.arange(cam1.size, dtype=np.float64)
    out = np.zeros((target_w,), dtype=np.float32)
    vals = np.interp(t * (cam1.size - 1), sample_positions,
                     np.asarray(cam1, dtype=np.float64))
    out[mask] = vals[mask].astype(np.float32)
    return out, mask.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description='All verified ECG MM-GradCAM author-style visual+provisional numeric alignment')
    ap.add_argument('--root', default='.')
    ap.add_argument('--pairs', default='splits/gradcam_test_pairs_verified.csv')
    ap.add_argument('--cam1', default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    ap.add_argument('--cam2', default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    ap.add_argument('--manifest1', default='results/gradcam_1d_author_code/gradcam_1d_manifest.csv')
    ap.add_argument('--manifest2', default='results/gradcam_2d_author_code/gradcam_2d_manifest.csv')
    ap.add_argument('--strict-manifests', action='store_true', default=False)
    ap.add_argument('--expected', type=int, default=1054, help='Fixed at 1054 for real experiment; set smaller only for synthetic tests')
    ap.add_argument('--out', default='results/mmgradcam_all_paper_style_v1')
    ap.add_argument('--figures', choices=['none', 'representative', 'all'], default='representative',
                    help='Annotated figures; ALL 1054 combined ECGs are always generated')
    ap.add_argument('--threshold', type=float, default=.23)
    ap.add_argument('--heat-opacity', type=float, default=.70)
    ap.add_argument('--left', type=int, default=58)
    ap.add_argument('--right', type=int, default=1021)
    args = ap.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.heat_opacity <= 1:
        ap.error('threshold and heat-opacity must both be between 0 and 1')
    if not 0 <= args.left < args.right < 1080:
        ap.error('invalid horizontal trace bounds for 1080-pixel ECG')

    root = Path(args.root).resolve()
    pairs, c1, c2, mf1, mf2 = read_inputs(args, root)
    out = as_path(root, args.out)
    # Default refuses overwrite: it could wipe research outputs from earlier runs.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}; choose another --out name')
    (out/'combined').mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(parents=True, exist_ok=True)

    want_fig = make_selected_indices(pairs, args.figures)
    aligned = np.zeros((len(pairs), 600), dtype=np.float32)
    valid = np.zeros((len(pairs), 600), dtype=np.uint8)
    records = []
    wrong_preds = int((pairs.pred_class_1d != pairs.pred_class_2d).sum())
    print(f'Paired records: {len(pairs)} | different prediction targets: {wrong_preds}', flush=True)
    print('All combined ECG images will be generated; annotated figures:', args.figures, flush=True)
    print('Output:', out, flush=True)

    for k, r in pairs.iterrows():
        img_file = as_path(root, str(r.image_path_2d))
        if img_file.stem != f'hasta_{int(r.cohort_index)}' or img_file.parent.name != str(r.true_class):
            raise ValueError(f'2D filename is inconsistent with verified pair: {r.patient_id} | {img_file}')
        if not img_file.is_file():
            raise FileNotFoundError(img_file)
        with Image.open(img_file) as f:
            original = np.asarray(f.convert('RGB'))
        if original.shape != (288, 1080, 3):
            raise ValueError(f'Unexpected ECG image geometry: {img_file} {original.shape}')
        cam1 = np.asarray(c1[int(r.cam_row_1d)], dtype=np.float32)
        cam2 = np.asarray(c2[int(r.cam_row_2d)], dtype=np.float32)
        if not np.isfinite(cam1).all() or not np.isfinite(cam2).all():
            raise ValueError(f'Non-finite Grad-CAM values for {r.patient_id}')
        # The checked paper-style renderer preserves the ECG waveform and uses
        # red 1D saliency dots + 2D heatmap, not grayscale vertical stripes.
        one_ecg, ndots, coverage = visuals.make_image(
            original, cam1, cam2, left=args.left, right=args.right,
            cam_threshold=args.threshold, heat_opacity=args.heat_opacity
        )
        basename = safe_stem(r)
        saved_png = out/'combined'/f'{basename}_mmgradcam_one_ecg.png'
        one_ecg.save(saved_png)
        aligned[k], valid[k] = provisional_1d_cam_on_2d(
            cam1, w=original.shape[1], left=args.left, right=args.right, target_w=600
        )

        conf = visuals.prediction_info(r, mf1, mf2)  # also checks patient identity
        fig_file = ''
        if k in want_fig:
            fig_file = out/'figures'/f'{basename}_paper_layout.png'
            visuals.annotated_figure(original, one_ecg, r, conf, fig_file)

        records.append({
            'aligned_row': int(k),
            'patient_id': str(r.patient_id),
            'cohort_index': int(r.cohort_index),
            'true_class': str(r.true_class),
            'cam_row_1d': int(r.cam_row_1d),
            'cam_row_2d': int(r.cam_row_2d),
            'pred_class_1d': str(r.pred_class_1d),
            'pred_class_2d': str(r.pred_class_2d),
            'same_predicted_class': bool(r.pred_class_1d == r.pred_class_2d),
            'both_predictions_correct': bool(r.pred_class_1d == r.true_class and r.pred_class_2d == r.true_class),
            'confidence_1d_pct': conf.get('pct1'),
            'confidence_2d_pct': conf.get('pct2'),
            'waveform_x_start': int(args.left),
            'waveform_x_end': int(args.right),
            'blue_trace_coverage': float(coverage),
            'number_of_red_1d_markers': int(ndots),
            'visual_png': str(saved_png.relative_to(root)),
            'annotated_png': str(fig_file.relative_to(root)) if fig_file else '',
            'numerical_registration_status': 'PROVISIONAL_UNVERIFIED_TIME_AXIS',
        })
        if (k+1) % 100 == 0 or k+1 == len(pairs):
            print(f'Processed {k+1}/{len(pairs)}', flush=True)

    manifest = pd.DataFrame(records)
    np.save(out/'cam1d_on_2d_x_600_PROVISIONAL.npy', aligned)
    np.save(out/'waveform_mask_600.npy', valid)
    manifest.to_csv(out/'alignment_manifest.csv', index=False)
    disagree = manifest.loc[~manifest.same_predicted_class]
    disagree.to_csv(out/'different_prediction_targets.csv', index=False)
    summary = {
        'status': 'PASS_PAPER_STYLE_VISUAL_ALIGNMENT; TEMPORAL_MAPPING_PROVISIONAL',
        'total_ecg_pairs': len(pairs),
        'generated_combined_pngs': int(manifest.visual_png.notna().sum()),
        'generated_annotated_figures': int(manifest.annotated_png.ne('').sum()),
        'predictions_disagree': int(len(disagree)),
        'both_predictions_correct': int(manifest.both_predictions_correct.sum()),
        'classes': manifest.true_class.value_counts().to_dict(),
        'one_d_source': args.cam1, 'two_d_source': args.cam2,
        'patient_pairs_source': args.pairs,
        'paper_style': 'Original ECG waveform + red 1D CAM markers + 2D CAM heatmap',
        'numerical_1d_alignment': 'Raw 1D CAM interpolated to 600 image columns using x=left..right',
        'numeric_output_shape': list(aligned.shape),
        'visual_threshold': args.threshold,
        'visual_heat_opacity': args.heat_opacity,
        'estimated_time_x_bounds_original_pixels': [args.left, args.right],
        'temporal_mapping_validated': False,
        'note': 'Paper-style reimplementation using independently generated CAMs. Not pixel-identical to authors. '
                '2D CAM spatial resize for display; waveform x bounds estimated. Not a quantitative agreement test. '
                'Pairs with different predicted classes explain different class targets.',
    }
    (out/'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('\nSTATUS: PASS — all paper-style ECG figures produced', flush=True)
    print('Visual figures:', out/'combined', flush=True)
    print('Provisional numeric 1D registration:', out/'cam1d_on_2d_x_600_PROVISIONAL.npy', flush=True)
    print('Manifest:', out/'alignment_manifest.csv', flush=True)
    print('Prediction target disagreements:', len(disagree), flush=True)
    print('IMPORTANT: numerical time mapping is provisional until independently verified.', flush=True)


if __name__ == '__main__':
    main()
