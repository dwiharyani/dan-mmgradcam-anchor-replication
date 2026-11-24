#!/usr/bin/env python3
"""Paper-layout MM-GradCAM figure from previously generated paired ECG inputs.

Reconstructs the *visual composition* described in Duranay et al., Scientific
Reports 2026, Fig. 5 (e) and Fig. 7: original ECG waveform + a 2D Grad-CAM
color field + red dot markers denoting high 1D Grad-CAM attribution, displayed
on ONE shared ECG image, together with branch-specific predictions.

This corrects the previous grayscale vertical-stripe overlay, which hid the ECG
waveform. It does NOT recover the authors' original rendered PNGs, prove
calibrated time alignment, or compute quantitative branch agreement.

Run from the root of MM_GradCAM_HPC:
python run_anchor_mmgradcam_paper_figure.py --selection representative \\
    --out results/anchor_mmgradcam_paper_style_smoke
python run_anchor_mmgradcam_paper_figure.py --selection all \\
    --out results/anchor_mmgradcam_paper_style
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CLASSES = ("AFib", "GSVT", "SB", "SR")
REFERENCE = "https://www.nature.com/articles/s41598-026-38654-w"


def norm(a):
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("CAM has non-finite values")
    lo, hi = float(np.min(a)), float(np.max(a))
    return np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)


def trace_pixels(rgb):
    """Detect blue ECG ink; only for author-prepared clean blue-on-white images.

    Returns a per-pixel soft mask and a waveform y value per x-column.
    Aborts if the image does not look like the author's blue ECG.
    """
    rgb = rgb.astype(np.float32)
    rr, gg, bb = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue_delta = np.minimum(bb - rr, bb - gg)
    # Matplotlib-like light blue ECG: blue channel much higher than red & green.
    hard = (bb - rr > 22) & (bb - gg > 8) & (rr < 240)
    col_count = hard.sum(axis=0)
    valid = col_count > 0
    if valid.sum() < 0.65 * rgb.shape[1]:
        raise RuntimeError(
            f"Blue ECG trace detection covers {valid.mean():.1%} columns; "
            "cannot place accurate 1D dots. Inspect source image first."
        )
    # Weighted vertical center of the blue trace. One y value per column.
    weights = np.maximum(0.0, blue_delta - 7.0) * hard
    ys = np.arange(rgb.shape[0], dtype=np.float32)[:, None]
    sums = weights.sum(axis=0)
    ycol = np.divide((weights * ys).sum(axis=0), sums, out=np.full(rgb.shape[1], np.nan), where=sums > 0)
    xs = np.arange(rgb.shape[1])
    present = np.where(np.isfinite(ycol))[0]
    ycol = np.interp(xs, present, ycol[present])
    # Soft mask for superimposing the ECG trace above the heatmap.
    soft = np.clip((bb - rr - 8) / 60.0, 0, 1) * np.clip((bb - gg - 3) / 35.0, 0, 1)
    soft *= (rr < 245)
    return soft.astype(np.float32), ycol, hard


def select_cases(pairs, mode):
    if mode == "all":
        return pairs
    indices = []
    for cls in CLASSES:
        group = pairs[pairs.true_class == cls]
        both_correct = group[(group.pred_class_1d == cls) & (group.pred_class_2d == cls)]
        an_error = group[(group.pred_class_1d != cls) | (group.pred_class_2d != cls)]
        if not both_correct.empty: indices.append(both_correct.index[0])
        if not an_error.empty: indices.append(an_error.index[0])
    return pairs.loc[list(dict.fromkeys(indices))]


def make_image(ecg_rgb, cam1, cam2, *, left=58, right=1021,
               cam_threshold=0.23, heat_opacity=0.70):
    """One shared ECG background, transparent 2D heatmap, red 1D dot markers."""
    from matplotlib import colormaps

    h, w = ecg_rgb.shape[:2]
    blue_alpha, waveform_y, blue_mask = trace_pixels(ecg_rgb)
    cam1 = norm(cam1).ravel()
    cam2 = norm(cam2)
    if cam1.shape != (5000,) or cam2.shape != (300, 600):
        raise ValueError(f"Unexpected CAM shapes {cam1.shape}, {cam2.shape}")
    if not 0 <= left < right < w:
        raise ValueError(f"Invalid estimated time x limits: {(left, right)} / {w}")

    # Map 2D CAM to original image canvas as a visual spatial field (not a
    # calibrated temporal reprojection). Keep white ECG and show high response.
    from PIL import Image as PILImage
    cmap_img = PILImage.fromarray((cam2 * 255).astype('uint8'), 'L').resize((w, h), PILImage.Resampling.BILINEAR)
    heat = np.asarray(cmap_img, dtype=np.float32) / 255.0
    colored = colormaps['jet'](heat)[..., :3].astype(np.float32) * 255.0
    # Mild pink-toned low-activation background, colored strong activation.
    original = ecg_rgb.astype(np.float32)
    base_tint = np.array([239., 220., 216.], dtype=np.float32)
    base = 0.84 * original + 0.16 * base_tint
    strength = (np.clip(heat, 0, 1) ** 1.20) * heat_opacity
    fused = base * (1 - strength[..., None]) + colored * strength[..., None]
    # ECG line should remain plainly visible on top of the colored heatmap.
    line_color = np.array([68., 31., 45.], dtype=np.float32)
    alpha = np.minimum(0.90, blue_alpha * 1.2)
    fused = fused * (1 - alpha[..., None]) + line_color * alpha[..., None]
    composed = Image.fromarray(np.uint8(np.clip(fused, 0, 255)), 'RGB')
    draw = ImageDraw.Draw(composed)

    # Only highlight signal-form 1D CAM at true trace y(x), not vertical stripes.
    # Threshold 0.23 follows stored 1D Grad-CAM threshold convention.
    candidates = np.arange(left, right + 1, 3, dtype=int)
    sample_pos = np.clip((candidates - left) / max(1, (right - left)) * (len(cam1) - 1), 0, len(cam1) - 1)
    salient = np.interp(sample_pos, np.arange(len(cam1)), cam1) >= cam_threshold
    dots = 0
    for x in candidates[salient]:
        if not blue_mask[:, x].any():
            continue
        y = int(round(waveform_y[x]))
        r = 2.1
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(205, 36, 47))
        dots += 1
    if dots == 0:
        raise RuntimeError("No signal-form saliency markers were placed on the ECG trace")
    return composed, dots, float(blue_mask.any(axis=0).mean())


def prediction_info(r, man1, man2):
    """Use actual XAI-run prediction (not stale 2D training manifest)."""
    c1 = r.pred_class_1d
    c2 = r.pred_class_2d
    out = {'pred1': c1, 'pred2': c2, 'pct1': None, 'pct2': None}
    if man1 is not None:
        row = man1.iloc[int(r.cam_row_1d)]
        if str(row.patient_id) != str(r.patient_id):
            raise ValueError('1D Grad-CAM manifest identity mismatch')
        out['pct1'] = float(row.get('confidence', np.nan)) * 100
    if man2 is not None:
        row = man2.iloc[int(r.cam_row_2d)]
        if str(row.patient_id) != str(r.patient_id):
            raise ValueError('2D Grad-CAM manifest identity mismatch')
        if str(row.pred_from_model) != str(c2):
            raise ValueError('2D prediction differs from paired XAI manifest')
        out['pct2'] = float(row[f'prob_{c2}']) * 100
    return out


def annotated_figure(original, fusion, case, conf, output):
    """Single paper-like figure with original ECG above ONE fused ECG image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14.6, 6.2), dpi=120, facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[6.5, 1.5], height_ratios=[1.0, 1.35],
                          left=.035, right=.985, top=.855, bottom=.11, wspace=.02, hspace=.34)
    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[1, 0]); axr = fig.add_subplot(gs[1, 1])
    for a in (ax0, ax1, axr): a.axis('off')
    ax0.imshow(original); ax0.set_title('Original ECG signal (same record)', fontsize=14, fontweight='semibold', pad=7)
    ax1.imshow(fusion); ax1.set_title('MM-GradCAM — shared ECG image: 1D red markers + 2D heatmap',
                                    fontsize=14, fontweight='semibold', pad=7)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1)
    def box(y, label, pred, pct, color):
        from matplotlib.patches import FancyBboxPatch
        bb = FancyBboxPatch((.055,y),.89,.38,boxstyle='round,pad=0.012,rounding_size=0.04',
                             facecolor=color, edgecolor='#AAA0A0', lw=.8, transform=axr.transAxes)
        axr.add_patch(bb)
        txt = f'{label}\nActual: {case.true_class}\nPredicted: {pred}'
        if pct is not None and np.isfinite(pct): txt += f'\nConfidence: {pct:.2f}%'
        axr.text(.50, y+.19, txt, transform=axr.transAxes, ha='center', va='center',
                 fontsize=10.5, color='#241F2A', fontweight='medium')
    box(.55, '1D signal CNN', conf['pred1'], conf['pct1'], '#F5DCE3')
    box(.065, '2D image CNN', conf['pred2'], conf['pct2'], '#DCEAF8')
    fig.suptitle(f'{case.true_class}  |  {case.patient_id}', fontsize=15, fontweight='bold', y=.985)
    if case.pred_class_1d != case.pred_class_2d:
        note = 'Different predicted classes: the two Grad-CAM maps explain DIFFERENT class targets.'
    else:
        note = 'Visual baseline: paper-like composite; not quantitative agreement or validated ECG-time alignment.'
    fig.text(.5, .025, note, ha='center', fontsize=9.6, color='#73545C')
    fig.savefig(output, dpi=140, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--pairs', default='splits/gradcam_test_pairs_verified.csv')
    ap.add_argument('--cam1', default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    ap.add_argument('--cam2', default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    ap.add_argument('--manifest1', default='results/gradcam_1d_author_code/gradcam_1d_manifest.csv')
    ap.add_argument('--manifest2', default='results/gradcam_2d_author_code/gradcam_2d_manifest.csv')
    ap.add_argument('--out', default='results/anchor_mmgradcam_paper_style')
    ap.add_argument('--selection', choices=['representative','all'], default='representative')
    ap.add_argument('--threshold', type=float, default=.23)
    ap.add_argument('--heat-opacity', type=float, default=.70)
    ap.add_argument('--estimated-left', type=int, default=58)
    ap.add_argument('--estimated-right', type=int, default=1021)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    resolve = lambda p: (Path(p) if Path(p).is_absolute() else root / p)
    pf = resolve(args.pairs)
    p = pd.read_csv(pf)
    required = ['patient_id','cohort_index','true_class','cam_row_1d','cam_row_2d',
                'pred_class_1d','pred_class_2d','image_path_2d']
    if p.duplicated('patient_id').any() or p.duplicated('cohort_index').any():
        raise ValueError('Duplicate ECG records in verified pairing file')
    if not set(required).issubset(p): raise ValueError('Missing required pair fields')
    a = np.load(resolve(args.cam1), mmap_mode='r')
    b = np.load(resolve(args.cam2), mmap_mode='r')
    if len(p)!=1054 or a.shape!=(1054,5000) or b.shape!=(1054,300,600):
        raise ValueError(f'Unexpected cohort/CAM shapes {len(p)}, {a.shape}, {b.shape}')
    for col in ['cam_row_1d','cam_row_2d']:
        if sorted(p[col].tolist()) != list(range(1054)):
            raise ValueError(f'{col} not a one-to-one permutation of CAM array positions')
    chosen = select_cases(p, args.selection)
    mf1 = resolve(args.manifest1); mf2 = resolve(args.manifest2)
    m1 = pd.read_csv(mf1) if mf1.exists() else None
    m2 = pd.read_csv(mf2) if mf2.exists() else None
    out = resolve(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output folder already contains files: {out}. Choose a new --out path.')
    (out/'combined').mkdir(parents=True, exist_ok=True)
    (out/'figures').mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (_,r) in enumerate(chosen.iterrows(), 1):
        image_path = resolve(str(r.image_path_2d))
        if image_path.stem != f'hasta_{r.cohort_index}' or image_path.parent.name != r.true_class:
            raise ValueError(f'2D image path identity differs: {r.patient_id}')
        with Image.open(image_path) as im: original = np.asarray(im.convert('RGB'))
        if original.shape[:2] != (288, 1080):
            raise ValueError(f'Unexpected ECG source shape: {original.shape}; do not assume x-limits')
        cc, dots, coverage = make_image(original, a[int(r.cam_row_1d)], b[int(r.cam_row_2d)],
                                       left=args.estimated_left, right=args.estimated_right,
                                       cam_threshold=args.threshold, heat_opacity=args.heat_opacity)
        conf = prediction_info(r,m1,m2)
        basename = re.sub(r'[^A-Za-z0-9_-]','_', f'{r.true_class}_{r.patient_id}')
        png = out/'combined'/f'{basename}_mmgradcam_one_ecg.png'
        cc.save(png)
        fig_file = ''
        if args.selection == 'representative' or n<=8:
            fig_file = out/'figures'/f'{basename}_paper_layout.png'
            annotated_figure(original,cc,r,conf,fig_file)
        rows.append({'patient_id':r.patient_id,'cohort_index':int(r.cohort_index),
                     'true_class':r.true_class,'pred_1d':r.pred_class_1d,'pred_2d':r.pred_class_2d,
                     'same_predicted_class':r.pred_class_1d==r.pred_class_2d,
                     'cam_row_1d':int(r.cam_row_1d),'cam_row_2d':int(r.cam_row_2d),
                     'blue_trace_x_coverage':coverage,'red_markers':dots,
                     'composite_png':str(png.relative_to(root)),
                     'figure_png':str(fig_file.relative_to(root)) if fig_file else ''})
        if n%100==0 or n==len(chosen): print(f'Processed {n}/{len(chosen)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    with (out/'summary.json').open('w') as f:
        json.dump({'status':'PAPER_STYLE_VISUAL_COMPOSITION_PASS',
                   'source_paper':REFERENCE,'total_pairs':len(p),
                   'rendered':len(chosen),'note':'One ECG image with 2D heatmap over signal and red 1D salient markers. Reimplementation, not original PNG and not pixel-identical.',
                   'one_d_markers_threshold':args.threshold,
                   'x_axis':f'Provisional {args.estimated_left}..{args.estimated_right} (render time mapping not verified)',
                   'spatial_registration':'2D Grad-CAM resized to original ECG bitmap, display-only',
                   'same_target_class_not_guaranteed':True,
                   'prediction_disagreements':int((p.pred_class_1d!=p.pred_class_2d).sum())},f,indent=2)
    print('STATUS: PAPER-STYLE ONE-ECG VISUALIZATION COMPLETED')
    print('Output:',out)
    print('NOTE: This is a paper-like re-render, not quantitatively calibrated alignment.')

if __name__=='__main__':main()
