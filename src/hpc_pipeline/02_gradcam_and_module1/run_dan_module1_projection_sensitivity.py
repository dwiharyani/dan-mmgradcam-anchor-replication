#!/usr/bin/env python3
"""DAN-MMGradCAM Module 1: 2D-to-time projection sensitivity, NO retraining.

Runs three projection methods on EXACTLY the already measured primary pilot ECGs:
  waveform_mask_6px : existing paper-proposal waveform mask with 6px vertical dilation
  waveform_mask_24px: broader 24px vertical dilation around the same waveform
  full_height_mean  : average CAM over all 300 image rows for each x column

This is a PROVISIONAL sensitivity analysis, NOT fully calibrated agreement.
Place alongside run_dan_module1_exploratory.py in MM_GradCAM_HPC root.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import run_dan_module1_exploratory as core

METRICS = [
    'mean_agreement_A', 'mean_disagreement_D', 'spearman', 'pearson',
    'dice_top_fraction', 'jaccard_top_fraction', 'weighted_jaccard',
]
METHODS = ['waveform_mask_6px', 'waveform_mask_24px', 'full_height_mean']


def as_path(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def finite_float(x):
    return float(x) if x is not None and np.isfinite(x) else None


def plot_qc(root, r, rgb, t, valid, g1, curves, scores_by_method, out_file):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.4),
                             gridspec_kw={'height_ratios': [1.15, 1, 1]})
    axes[0].imshow(rgb)
    axes[0].set_title(f'{r.true_class} | {r.patient_id} | ORIGINAL 2D ECG')
    axes[0].axis('off')
    colors = {'waveform_mask_6px':'#237da9',
              'waveform_mask_24px':'#9b64ac',
              'full_height_mean':'#b38b35'}
    axes[1].plot(t[valid], g1[valid], c='#c75073', lw=1.2, label='1D CAM (common)')
    for meth in METHODS:
        axes[1].plot(t[valid], curves[meth][valid], lw=1.4,
                     color=colors[meth], label=meth)
    axes[1].set_ylim(-.04, 1.08)
    axes[1].set_ylabel('Normalized saliency')
    axes[1].legend(loc='upper right', fontsize=8, ncol=2)
    for meth in METHODS:
        axes[2].plot(t[valid], scores_by_method[meth]['difference'][valid],
                     c=colors[meth], lw=1.25, label=f'{meth}: D(u)')
    axes[2].set_ylim(-.04, 1.08)
    axes[2].set_xlabel('Provisional ECG time (seconds) — not final time calibration')
    axes[2].set_ylabel('D(u)')
    axes[2].legend(loc='upper right', fontsize=8)
    fig.suptitle('DAN-MMGradCAM Module 1 | 2D Projection Sensitivity — EXPLORATORY', y=.995,
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out_file, dpi=130, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Compare three 2D CAM projection methods on exact Module 1 pilot cases')
    ap.add_argument('--root', default='.')
    ap.add_argument('--pilot', default='results/dan_module1_pilot_24_v1/module1_primary_exploratory.csv')
    ap.add_argument('--cam1', default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    ap.add_argument('--cam2', default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    ap.add_argument('--out', default='results/dan_module1_projection_sensitivity_v1')
    ap.add_argument('--x0', type=int, default=58)
    ap.add_argument('--x1', type=int, default=1021)
    ap.add_argument('--baseline-dilate-y', type=int, default=6)
    ap.add_argument('--wide-dilate-y', type=int, default=24)
    ap.add_argument('--top-fraction', type=float, default=0.15)
    ap.add_argument('--qc-figures', type=int, default=8)
    ap.add_argument('--reproduce-tolerance', type=float, default=0.0002)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = as_path(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output folder is not empty: {out}. Please choose a new --out.')
    if args.baseline_dilate_y < 0 or args.wide_dilate_y <= args.baseline_dilate_y:
        raise ValueError('The broader mask must use larger vertical dilation than the baseline.')
    if not (0 < args.top_fraction < 1):
        raise ValueError('top-fraction must be in (0,1)')
    pilot_path = as_path(root, args.pilot)
    df = pd.read_csv(pilot_path)
    required = ['patient_id','cohort_index','true_class','cam_row_1d','cam_row_2d',
                'image_path_2d'] + METRICS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Pilot primary CSV missing columns: {missing}')
    if df.empty or df.patient_id.duplicated().any() or df.cohort_index.duplicated().any():
        raise ValueError('Pilot records empty or contain duplicate identities')
    if 'processing_note' in df and not (df.processing_note=='EXPLORATORY_MEASURED').all():
        raise ValueError('Expected exclusively originally measured primary pilot cases')
    if 'both_correct' in df and not df.both_correct.fillna(False).astype(bool).all():
        raise ValueError('Pilot contains samples not correctly predicted by both branches')
    if 'strict_waveform_qc' in df and not df.strict_waveform_qc.fillna(False).astype(bool).all():
        raise ValueError('Pilot contains samples without strict waveform QC')

    ca1 = np.load(as_path(root,args.cam1), mmap_mode='r', allow_pickle=False)
    ca2 = np.load(as_path(root,args.cam2), mmap_mode='r', allow_pickle=False)
    if len(ca1.shape)!=2 or ca1.shape[1]!=5000 or ca2.shape[1:]!=(300,600):
        raise ValueError(f'Unexpected CAM shapes: {ca1.shape}, {ca2.shape}')
    if ca1.shape[0]!=ca2.shape[0]:
        raise ValueError('CAM array lengths differ')
    if np.min(df.cam_row_1d)<0 or np.max(df.cam_row_1d)>=len(ca1) or np.min(df.cam_row_2d)<0 or np.max(df.cam_row_2d)>=len(ca2):
        raise ValueError('CAM row index out of bounds')

    out.mkdir(parents=True,exist_ok=True)
    (out/'qc_figures').mkdir(exist_ok=True)
    details = []
    trace_details = []
    curves_arrays = {m:[] for m in METHODS}
    by_class_fig = {c:0 for c in ('AFib','GSVT','SB','SR')}
    plotted=0
    max_plots=max(0,args.qc_figures)
    t=np.linspace(0.,10.,600)
    max_baseline_diff=0.

    for j,r in enumerate(df.itertuples(index=False),start=1):
        f=as_path(root,str(r.image_path_2d))
        if f.stem!=f'hasta_{int(r.cohort_index)}' or f.parent.name!=r.true_class:
            raise ValueError(f'2D file identity mismatch: {f}')
        with Image.open(f) as im:
            rgb=np.asarray(im.convert('RGB'))
        if rgb.shape!=(288,1080,3):
            raise ValueError(f'Original ECG dimensions mismatch for {r.patient_id}: {rgb.shape}')
        c1=np.asarray(ca1[int(r.cam_row_1d)],dtype=np.float64)
        c2=np.asarray(ca2[int(r.cam_row_2d)],dtype=np.float64)
        if not np.isfinite(c1).all() or not np.isfinite(c2).all():
            raise ValueError(f'Nonfinite CAM for {r.patient_id}')
        g1_raw, geometrically_valid=core.numeric_register_1d(c1,args.x0,args.x1)
        m6=core.trace_mask_at_2d_grid(rgb,dilate_y=args.baseline_dilate_y)
        m24=core.trace_mask_at_2d_grid(rgb,dilate_y=args.wide_dilate_y)
        s6=m6.sum(axis=0)
        s24=m24.sum(axis=0)
        valid_original=geometrically_valid & (s6>0)
        valid_common=valid_original & (s24>0)
        if int(valid_common.sum()) < 350:
            raise ValueError(f'Too few comparable columns for {r.patient_id}: {valid_common.sum()}')
        # Auditing identity with Module 1 baseline uses its ORIGINAL valid mask.
        g1_original=core.grayscale_norm(g1_raw,valid_original)
        g2_original_raw=(c2*m6).sum(axis=0)/(s6+core.EPS)
        g2_original=core.grayscale_norm(g2_original_raw,valid_original)
        baseline_orig=core.scores(g1_original,g2_original,valid_original,args.top_fraction)
        if baseline_orig is None:
            raise ValueError(f'Cannot reproduce Module 1 baseline for {r.patient_id}')
        _, _, orig_metrics=baseline_orig
        for name in METRICS:
            expected=float(getattr(r,name))
            actual=float(orig_metrics[name])
            if not np.isfinite(expected) or not np.isfinite(actual):
                raise ValueError(f'Nonfinite baseline metric {name} for {r.patient_id}')
            diff=abs(expected-actual)
            max_baseline_diff=max(max_baseline_diff,diff)
            if diff>args.reproduce_tolerance:
                raise ValueError(f'Module 1 baseline not reproduced: {r.patient_id} {name}: saved={expected:.7f}, new={actual:.7f}; difference={diff:.7f}')

        # Compare projected traces on IDENTICAL coordinates across all methods.
        # Normalize EACH method separately, exactly like baseline's valid min-max norm.
        g1=core.grayscale_norm(g1_raw,valid_common)
        raw_projections={
          METHODS[0]:g2_original_raw,
          METHODS[1]:(c2*m24).sum(axis=0)/(s24+core.EPS),
          METHODS[2]:c2.mean(axis=0),
        }
        scores_by_method={}
        traces={}
        for method in METHODS:
            g2=core.grayscale_norm(raw_projections[method],valid_common)
            if g2 is None or g1 is None:
                raise ValueError(f'Constant projection {method} for {r.patient_id}')
            res=core.scores(g1,g2,valid_common,args.top_fraction)
            if res is None:
                raise ValueError(f'Invalid similarity metrics {method} for {r.patient_id}')
            a,d,mets=res
            item={'patient_id':r.patient_id,'cohort_index':int(r.cohort_index),'true_class':r.true_class,
                  'method':method, 'valid_columns':int(valid_common.sum()),
                  'waveform_support_count_min':int(s6[valid_common].min()),
                  'waveform_support_count_median':float(np.median(s6[valid_common])),
                  'waveform_support_count_max':int(s6[valid_common].max())}
            item.update(mets)
            details.append(item)
            traces[method]=g2
            scores_by_method[method]={'agreement':a,'difference':d}
            curves_arrays[method].append(g2)
        corr_baseline_vs_full=float(np.corrcoef(traces[METHODS[0]][valid_common],traces[METHODS[2]][valid_common])[0,1])
        corr_baseline_vs_wide=float(np.corrcoef(traces[METHODS[0]][valid_common],traces[METHODS[1]][valid_common])[0,1])
        trace_details.append({'patient_id':r.patient_id,'true_class':r.true_class,'cohort_index':int(r.cohort_index),
                              'baseline_vs_full_2dtrace_pearson':corr_baseline_vs_full,
                              'baseline_vs_wide_2dtrace_pearson':corr_baseline_vs_wide,
                              'baseline_support_cv':float(s6[valid_common].std()/max(float(s6[valid_common].mean()),core.EPS))})
        if plotted<max_plots and by_class_fig.get(r.true_class,0)<max(1,int(np.ceil(max_plots/4))):
            stem=re.sub('[^A-Za-z0-9_.-]','_',f'{r.true_class}_{r.patient_id}')
            plot_qc(root,r,rgb,t,valid_common,g1,traces,scores_by_method,
                    out/'qc_figures'/f'{stem}_projection_sensitivity.png')
            plotted+=1
            by_class_fig[r.true_class]=by_class_fig.get(r.true_class,0)+1
        print(f'Sensitivity compared {j}/{len(df)}: {r.true_class} {r.patient_id}',flush=True)

    long=pd.DataFrame(details)
    long.to_csv(out/'projection_metrics_long.csv',index=False)
    pd.DataFrame(trace_details).to_csv(out/'projection_trace_diagnostics.csv',index=False)
    # Paired table using same patient_id and metrics for all methods
    wide=long.pivot(index=['patient_id','cohort_index','true_class'],columns='method',values=METRICS)
    wide.columns=[f'{met}__{method}' for met,method in wide.columns.to_flat_index()]
    wide=wide.reset_index()
    for method in METHODS[1:]:
        for met in METRICS:
            wide[f'delta_{met}__{method}_vs_baseline']=(wide[f'{met}__{method}']-wide[f'{met}__{METHODS[0]}'])
    wide.to_csv(out/'paired_metric_differences.csv',index=False)
    long.groupby('method')[METRICS].agg(['count','mean','median','std']).round(6).to_csv(out/'overall_by_method.csv')
    long.groupby(['true_class','method'])[METRICS].agg(['count','mean','median']).round(6).to_csv(out/'by_class_and_method.csv')
    np.savez_compressed(out/'projected_cam2_600_by_method.npz',
                        **{method:np.stack(arr,axis=0).astype(np.float32) for method,arr in curves_arrays.items()})
    sumry={'status':'EXPLORATORY_PROJECTION_SENSITIVITY_COMPLETE_NOT_FULLY_TIME_CALIBRATED',
           'pilot_source':str(args.pilot),'pilot_ecg_count':len(df),
           'methods':METHODS,'same_ecg_and_same_valid_x_for_all_methods':True,
           'baseline_original_metrics_reproduced':True,
           'max_absolute_baseline_metric_difference':finite_float(max_baseline_diff),
           'mask_dilation_300px_grid':{'baseline':args.baseline_dilate_y,'broader':args.wide_dilate_y},
           'top_fraction':args.top_fraction,'x0_assumed':args.x0,'x1_assumed':args.x1,
           'per_method_median_metrics':{
               method:{m:finite_float(long.loc[long.method==method,m].median()) for m in METRICS}
               for method in METHODS},
           'per_method_vs_baseline_median_change':{
               method:{m:finite_float(wide[f'delta_{m}__{method}_vs_baseline'].median()) for m in METRICS}
               for method in METHODS[1:]},
           'important_limits':['This is a sensitivity test on the fixed pilot cohort; not 1054-case result',
             '2D Grad-CAM native resolution 9x18 is interpolated to 300x600',
             'Shared ECG timeline assumes unverified image-generation mapping',
             'Predicted class must agree; pilot source contains both-correct, strict-QC cases',
             'Differences in metrics can originate from mask geometry, 2D model, or interpolation; this test does not prove causality or faithfulness',
             'Each projection is min-max normalized on identical valid columns prior to scoring']}
    (out/'summary.json').write_text(json.dumps(sumry,indent=2)+'\n',encoding='utf-8')
    print('\nSTATUS: PROJECTION SENSITIVITY COMPLETE (EXPLORATORY)',flush=True)
    print('Patients:',len(df),'baseline reproduced; max difference:',max_baseline_diff,flush=True)
    print('Median scores:',json.dumps(sumry['per_method_median_metrics'],indent=2),flush=True)
    print('Output:',out.resolve(),flush=True)


if __name__=='__main__':
    main()
