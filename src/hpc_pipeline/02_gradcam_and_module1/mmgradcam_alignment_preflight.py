#!/usr/bin/env python3
"""Non-destructive coordinate/geometry preflight for paired MM-GradCAM.

Run from the existing MM_GradCAM_HPC directory. Does not perform alignment or
fusion and deliberately does not assume full image width represents ECG time.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_checked(path, dims, label):
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    a = np.load(path, mmap_mode='r', allow_pickle=False)
    if a.ndim != dims:
        raise ValueError(f'{label}: expected {dims} dimensions, got {a.shape}')
    return a


def resolve_image(project, val):
    p = Path(str(val))
    if not p.is_absolute():
        p = project / p
    return p


def clean(s):
    return re.sub(r'[^\w.-]', '_', str(s))[:85]


def draw_one(row, project, heat1, heat2, folder):
    i1, i2 = int(row.cam_row_1d), int(row.cam_row_2d)
    h1 = np.asarray(heat1[i1], dtype='float32')
    h2 = np.asarray(heat2[i2], dtype='float32')
    if not np.isfinite(h1).all() or not np.isfinite(h2).all():
        raise ValueError(f'Nonfinite CAM values for {row.patient_id}')
    image_path = resolve_image(project, row.image_path_2d)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as img:
        base = np.asarray(img.convert('RGB'))
        orig_size = img.size
        orig_mode = img.mode
    bh, bw = base.shape[:2]
    ch, cw = h2.shape
    # For preflight, resampling the CAM to its native image for DISPLAY ONLY is allowed;
    # no ECG time correspondence is asserted by this operation.
    if (bh, bw) != (ch, cw):
        cam_visual = np.asarray(Image.fromarray(h2).resize((bw,bh),resample=Image.Resampling.BILINEAR))
    else:
        cam_visual = h2
    lo, hi = float(cam_visual.min()), float(cam_visual.max())
    saliency_vis = (cam_visual-lo)/(hi-lo+1e-8)

    fig = plt.figure(figsize=(12.0, 8.7), dpi=130, layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.1, 2.1])
    a0 = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2])
    seconds = np.linspace(0,10,h1.size)
    a0.plot(seconds, h1, linewidth=0.8, color='#a03a69')
    a0.set_xlim(0,10); a0.set_ylabel('1D CAM'); a0.set_xlabel('ECG time (seconds)')
    a0.set_title('1D heatmap: 5,000 sample indices mapped to 10 s')
    a0.grid(alpha=.25)

    a1.imshow(base, origin='upper', extent=[0,bw,bh,0], interpolation='nearest', aspect='auto')
    a1.set_title('Original 2D ECG image — IMAGE PIXELS, not verified ECG time')
    a2.imshow(base, origin='upper', extent=[0,bw,bh,0], interpolation='nearest', aspect='auto')
    a2.imshow(saliency_vis, origin='upper', extent=[0,bw,bh,0], interpolation='bilinear',
              cmap='turbo', alpha=np.minimum(saliency_vis*.60, .60), aspect='auto')
    a2.set_title('Original 2D image + Grad-CAM (display only; no temporal alignment)')
    for a in (a1,a2):
        a.set_xlim(0,bw); a.set_ylim(bh,0)
        a.set_xlabel('Horizontal position in image (pixels)')
        a.set_ylabel('Image Y (pixels)')
        for x in np.linspace(0,bw,11):
            a.axvline(x, color='red', linestyle='--', alpha=.30, linewidth=.8)
        a.set_xticks(np.linspace(0,bw,11))
    fig.suptitle(f"PAIR CHECK | {row.true_class} | {row.patient_id} | 1D CAM row={i1} / 2D CAM row={i2}",
                 fontsize=10, fontweight='bold')
    filename = folder / (clean(f'{row.true_class}_{row.patient_id}')+'.png')
    fig.savefig(filename, dpi=140)
    plt.close(fig)
    return {'patient_id':row.patient_id, 'true_class':row.true_class,
            'cam_row_1d':i1, 'cam_row_2d':i2, 'image_path':str(image_path),
            'image_width':bw, 'image_height':bh, 'image_original_mode':orig_mode,
            'image_original_size':str(orig_size), 'heatmap2d_width':cw,
            'heatmap2d_height':ch, 'preview_file':str(filename)}


def main():
    p=argparse.ArgumentParser(description='Preflight geometry; no alignment/fusion is performed.')
    p.add_argument('--project',default='.')
    p.add_argument('--pairs',default='splits/gradcam_test_pairs_verified.csv')
    p.add_argument('--heatmap-1d',default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    p.add_argument('--heatmap-2d',default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    p.add_argument('--out',default='results/alignment_preflight')
    args=p.parse_args()
    project=Path(args.project).resolve()
    def local(v):
        x=Path(v)
        return x if x.is_absolute() else project/x
    pairs=pd.read_csv(local(args.pairs))
    required=['patient_id','true_class','cohort_index','cam_row_1d','cam_row_2d','image_path_2d']
    missing=[c for c in required if c not in pairs.columns]
    if missing: raise ValueError('Missing required pair columns: '+', '.join(missing))
    if pairs['patient_id'].duplicated().any() or pairs['cohort_index'].duplicated().any():
        raise ValueError('Pair file has duplicate patient_id or cohort_index.')
    heat1=load_checked(local(args.heatmap_1d),2,'1D CAM array')
    heat2=load_checked(local(args.heatmap_2d),3,'2D CAM array')
    if not (len(pairs)==len(heat1)==len(heat2)==1054):
        raise ValueError(f'Expected 1054 pairs/heatmaps, got {len(pairs)}, {heat1.shape}, {heat2.shape}')
    if not (pairs['cam_row_1d'].between(0,len(heat1)-1).all() and
            pairs['cam_row_2d'].between(0,len(heat2)-1).all()):
        raise ValueError('Out of range CAM row index')
    if not (pairs['cam_row_1d'].nunique()==1054 and pairs['cam_row_2d'].nunique()==1054):
        raise ValueError('Nonunique CAM row indices')
    folder=local(args.out)
    folder.mkdir(parents=True,exist_ok=True)
    chosen=[]
    for cls in ('AFib','GSVT','SB','SR'):
        class_rows=pairs[pairs['true_class']==cls]
        if class_rows.empty: raise ValueError('No verified paired test ECG for '+cls)
        chosen.append(class_rows.iloc[0])
    # Add one example of prediction disagreement, if present.
    if {'pred_class_1d','pred_class_2d'} <= set(pairs.columns):
        bad = pairs[pairs['pred_class_1d'] != pairs['pred_class_2d']]
        if not bad.empty and bad.iloc[0]['patient_id'] not in {r['patient_id'] for r in chosen}:
            chosen.append(bad.iloc[0])
    records=[draw_one(row,project,heat1,heat2,folder) for row in chosen]
    outcsv=folder/'geometry_samples.csv'
    pd.DataFrame(records).to_csv(outcsv,index=False)
    report=folder/'README.txt'
    report.write_text(
       f'PRE-ALIGNMENT IMAGE/TIME-AXIS AUDIT\n'
       f'Pairs verified: {len(pairs)}\n'
       f'1D CAM array shape: {heat1.shape}\n'
       f'2D CAM array shape: {heat2.shape}\n'
       f'Diagnostic figures generated: {len(records)}\n\n'
       'No alignment, fusion or sampling of 2D CAM onto ECG time was performed.\n'
       'Image X values are PIXELS and NOT YET ECG seconds.\n'
       'Visually inspect original images to find actual waveform start/end columns,\n'
       'and check whether each image depicts one continuous 10-second trace.\n'
       'Do not infer those bounds solely from Grad-CAM activation.\n',encoding='utf-8')
    print('STATUS: PREFLIGHT PASS (not a completed temporal alignment)')
    print('Verified pairs:',len(pairs))
    print('CAM arrays:',heat1.shape,heat2.shape)
    print('Original image geometry:', {(r['image_width'],r['image_height']) for r in records})
    print('Generated figures:')
    for r in records: print('  ',r['preview_file'])
    print('Geometry CSV:',outcsv)
    print('Interpretation notes:',report)
    print('NEXT: inspect image x-axis/time mapping before 1D->2D registration.')

if __name__=='__main__':
    main()
