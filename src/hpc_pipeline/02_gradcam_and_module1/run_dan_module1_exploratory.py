#!/usr/bin/env python3
"""DAN-MMGradCAM Module 1: conservative exploratory 1D/2D agreement analysis.

INPUTS: existing paired 1D and 2D Grad-CAM, ORIGINAL blue ECG bitmaps,
        and the independent 1D-versus-image waveform correspondence audit.

Scientific restrictions:
- No model training, no new Grad-CAM, no replacement of existing results.
- Different predicted class targets are NEVER compared as explanatory agreement.
- Strict main-analysis cohort also requires BOTH predictions correct and waveform QC.
- Quantitative mapping is provisional until a direct image time-axis rendering
  calibration is available. This is exploratory, not a final validated result.
- 2D CAM is resized from low-resolution (9x18) features, so peak precision in
  the original image and CAM feature-map precision are not interchangeable.

Run from ~/mm_gradcam/MM_GradCAM_HPC, after .venv activation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter1d
from scipy.stats import spearmanr

CLASSES = ("AFib", "GSVT", "SB", "SR")
EPS = 1e-8


def at(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def check(cond, message):
    if not cond:
        raise ValueError(message)


def grayscale_norm(values, valid):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Non-finite attribution")
    subset = a[valid]
    if subset.size == 0:
        raise ValueError("No valid temporal columns")
    lo, hi = float(subset.min()), float(subset.max())
    if hi - lo <= 1e-10:
        return None
    out = np.zeros(a.shape, dtype=np.float32)
    out[valid] = ((subset - lo) / (hi - lo)).astype(np.float32)
    return out


def trace_mask_at_2d_grid(original_rgb, target_h=300, target_w=600, dilate_y=6):
    """Detect blue ECG stroke independently in ORIGINAL image, then form M(v,u).

    2D heatmaps are defined on an RGB ECG image, not on a synthetic heatmap
    overlay. The binary waveform mask is dilated VERTICALLY by 6 pixels in the
    300 px CAM grid as a tolerant spatial support for the low-res 2D CAM.
    """
    im = np.asarray(original_rgb, dtype=np.int16)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    ink = ((B - R >= 14) & (B - G >= 5) & (G - R >= 4) &
           (R < 240) & (G < 250))
    resized = Image.fromarray(ink.astype(np.uint8) * 255, 'L').resize(
        (target_w, target_h), Image.Resampling.BILINEAR)
    mask = (np.asarray(resized, dtype=np.uint8) > 14)
    mask = maximum_filter1d(mask, size=2 * dilate_y + 1, axis=0)
    return mask


def numeric_register_1d(x, x0, x1, original_width=1080, target_w=600):
    """Map 5000-sample 1D Grad-CAM to original image x within x0..x1.

    PROVISIONAL AFFINE ASSUMPTION: first ECG sample -> x0, last -> x1.
    """
    x_orig = (np.arange(target_w, dtype=np.float64) + .5) * original_width / target_w - .5
    valid = (x_orig >= x0) & (x_orig <= x1)
    t = np.clip((x_orig - x0) / (x1 - x0), 0, 1)
    out = np.interp(t * (len(x) - 1), np.arange(len(x)), x)
    return out, valid


def topk_dice_jaccard(a, b, valid, fraction=.15):
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    k = max(1, int(np.ceil(len(x) * fraction)))
    ax = np.argsort(-x, kind='stable')[:k]
    by = np.argsort(-y, kind='stable')[:k]
    overlap = len(set(ax.tolist()) & set(by.tolist()))
    return (2. * overlap / (2 * k), overlap / (2 * k - overlap))


def scores(g1, g2, valid, top_fraction=.15):
    x, y = g1[valid].astype(np.float64), g2[valid].astype(np.float64)
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    a = np.sqrt(np.maximum(g1, 0) * np.maximum(g2, 0)).astype(np.float32)
    d = np.abs(g1 - g2).astype(np.float32)
    dice, jaccard = topk_dice_jaccard(g1, g2, valid, top_fraction)
    spearman = float(spearmanr(x, y).statistic)
    pearson = float(np.corrcoef(x, y)[0, 1])
    weighted_j = float(np.minimum(x, y).sum() / (np.maximum(x, y).sum() + EPS))
    metrics = {
        'mean_agreement_A': float(a[valid].mean()),
        'mean_disagreement_D': float(d[valid].mean()),
        'weighted_jaccard': weighted_j,
        'spearman': spearman,
        'pearson': pearson,
        'dice_top_fraction': float(dice),
        'jaccard_top_fraction': float(jaccard),
        'mean_1d_saliency': float(x.mean()),
        'mean_2d_saliency': float(y.mean()),
    }
    return a, d, metrics


def qc_plot(row, rgb, g1, g2, agreement, difference, valid, filename, top_fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.linspace(0, 10., len(g1))
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.2), gridspec_kw={'height_ratios':[1.4,1,1]})
    ax[0].imshow(rgb)
    ax[0].set_title(f"Original 2D ECG | {row.true_class} | {row.patient_id}")
    ax[0].axis('off')
    ax[1].plot(t[valid], g1[valid], color='#C05270', lw=1.3, label='1D CAM, normalized')
    ax[1].plot(t[valid], g2[valid], color='#277FAF', lw=1.3, label='2D CAM projected onto waveform')
    ax[1].legend(loc='upper right', fontsize=9)
    ax[1].set_ylim(-.05, 1.1)
    ax[1].set_ylabel('Saliency')
    ax[2].plot(t[valid], agreement[valid], color='#875D3D', lw=1.3, label='A(u): shared importance')
    ax[2].plot(t[valid], difference[valid], color='#B43D3D', lw=1.2, label='D(u): difference')
    ax[2].set_ylim(-.05, 1.1)
    ax[2].set_xlabel('ECG time, seconds — PROVISIONAL affine calibration')
    ax[2].set_ylabel('Score')
    ax[2].legend(loc='upper right', fontsize=9)
    fig.suptitle(
        f'DAN-MMGradCAM Module 1 — EXPLORATORY / NOT FINAL VALIDATED\n'
        f'Target: {row.pred_class_1d}; QC: {row.status}; top fraction={top_fraction}',
        y=.993, fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(filename, dpi=125, facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Exploratory Module 1 analysis with independently audited waveform correspondence')
    p.add_argument('--root', default='.')
    p.add_argument('--pairs', default='splits/gradcam_test_pairs_verified.csv')
    p.add_argument('--validation', default='results/temporal_validation_full_20261015_163719/temporal_alignment_validation.csv')
    p.add_argument('--cam1', default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    p.add_argument('--cam2', default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    p.add_argument('--out', default='results/dan_module1_exploratory_v1')
    p.add_argument('--limit', type=int, default=0, help='0 = all pairs, N = first N pairs for debugging only')
    p.add_argument('--pilot-per-class', type=int, default=0, help='0 = full cohort, N = take N records per class for a balanced pilot')
    p.add_argument('--expected-count', type=int, default=1054, help='Expected input cohort size; change only for synthetic tests')
    p.add_argument('--qc-figures', type=int, default=8)
    p.add_argument('--p95-max-ms', type=float, default=50.)
    p.add_argument('--median-max-ms', type=float, default=25.)
    p.add_argument('--top-fraction', type=float, default=.15)
    p.add_argument('--x0', type=int, default=58)
    p.add_argument('--x1', type=int, default=1021)
    p.add_argument('--mask-dilate-y', type=int, default=6)
    args = p.parse_args()
    check(0 < args.top_fraction < 1, 'top-fraction must be between 0 and 1')
    check(0 <= args.x0 < args.x1 < 1080, 'Invalid x0/x1')
    check(args.limit >= 0 and args.pilot_per_class >= 0 and args.qc_figures >= 0 and args.mask_dilate_y >= 0, 'Nonnegative integer required')
    check(not (args.limit and args.pilot_per_class), 'Use either --limit or --pilot-per-class, not both')
    expected = args.expected_count
    check(expected > 0, '--expected-count must be positive')
    root = Path(args.root).resolve()
    pairs = pd.read_csv(at(root, args.pairs))
    val = pd.read_csv(at(root, args.validation))
    check(len(pairs) == expected, f'Expected {expected} verified ECG test pairs')
    check(len(val) == expected, f'Expected independent full audit for {expected} ECGs')
    key = ['patient_id', 'cohort_index', 'true_class']
    for name, df in (('pairs', pairs), ('validation', val)):
        check(not df[key].isna().any().any(), name + ' missing match keys')
        check(not df.duplicated(key).any(), name + ' duplicate patient/cohort/class keys')
    df = pairs.merge(val[key + ['status','match_fraction','median_abs_time_error_ms',
                                 'p95_abs_time_error_ms','true_minus_wrong_match_fraction',
                                 'x0_assumed','x1_assumed']], on=key,
                     validate='one_to_one', how='left', indicator=True)
    check((df._merge=='both').all(), 'Unmatched validation rows')
    check(len(df)==expected, 'Joined cohort size mismatch')
    check((df['x0_assumed']==args.x0).all() and (df['x1_assumed']==args.x1).all(),
          'x0/x1 differ from those used by independent waveform validation')
    for col in ['cam_row_1d','cam_row_2d']:
        ix = df[col].astype(int).to_numpy()
        check(sorted(ix.tolist())==list(range(expected)), col+' not permutation of all original CAM rows')
        df[col] = ix
    check(df.patient_id.is_unique and df.cohort_index.is_unique, 'Nonunique paired identities')
    check(df.true_class.isin(CLASSES).all(), 'Unknown rhythm class')
    check(df.pred_class_1d.isin(CLASSES).all() and df.pred_class_2d.isin(CLASSES).all(), 'Invalid model predictions')
    c1 = np.load(at(root, args.cam1), mmap_mode='r',allow_pickle=False)
    c2 = np.load(at(root, args.cam2), mmap_mode='r',allow_pickle=False)
    check(c1.shape==(expected,5000), 'Bad 1D CAM array shape')
    check(c2.shape==(expected,300,600), 'Bad 2D CAM array shape')
    df['same_predicted_target'] = (df.pred_class_1d == df.pred_class_2d)
    df['both_correct'] = (df.pred_class_1d == df.true_class) & (df.pred_class_2d == df.true_class)
    df['strict_waveform_qc'] = (
        (df.status == 'AUTO_PASS') &
        (df.median_abs_time_error_ms <= args.median_max_ms) &
        (df.p95_abs_time_error_ms <= args.p95_max_ms) &
        (df.true_minus_wrong_match_fraction > 0))
    df['primary_eligible'] = df['both_correct'] & df['strict_waveform_qc']
    df['analysis_group'] = np.select([
        ~df.same_predicted_target,
        df.same_predicted_target & ~df.both_correct,
        df.both_correct & ~df.strict_waveform_qc,
        df.primary_eligible],
        ['DIFFERENT_TARGET_NO_COMPARISON', 'SAME_TARGET_BUT_NOT_BOTH_CORRECT',
         'BOTH_CORRECT_BUT_QC_NOT_STRICT','PRIMARY_EXPLORATORY'],
        default='INTERNAL_ERROR')

    if args.pilot_per_class:
        df = pd.concat([df[df.true_class == c].head(args.pilot_per_class) for c in CLASSES],ignore_index=True)
    elif args.limit:
        df = df.iloc[:args.limit].copy()
    else:
        df = df.copy()
    out = at(root, args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory exists and is not empty: {out}. Choose a fresh --out folder.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'qc_figures').mkdir(exist_ok=True)
    arrays = np.full((4, len(df), 600), np.nan, dtype=np.float32)
    cols = ['mean_agreement_A','mean_disagreement_D','weighted_jaccard',
            'spearman','pearson','dice_top_fraction','jaccard_top_fraction',
            'mean_1d_saliency','mean_2d_saliency']
    for col in cols:
        df[col] = np.nan
    df['waveform_mask_column_coverage'] = np.nan
    df['processing_note'] = ''
    figure_counts={c:0 for c in CLASSES}
    problems=[]
    for j, r in enumerate(df.itertuples(index=False)):
        if not r.primary_eligible:
            df.iat[j, df.columns.get_loc('processing_note')] = 'Not included in primary exploratory quantitative comparison'
            continue
        f = at(root, str(r.image_path_2d))
        check(f.is_file(), f'Missing 2D ECG image {f}')
        check(f.stem==f'hasta_{r.cohort_index}' and f.parent.name==r.true_class, f'Image identity mismatch: {f}')
        with Image.open(f) as im:
            rgb=np.asarray(im.convert('RGB'))
        check(rgb.shape==(288,1080,3), f'Unexpected original image shape {rgb.shape}')
        mask2d = trace_mask_at_2d_grid(rgb,dilate_y=args.mask_dilate_y)
        cam1=np.asarray(c1[r.cam_row_1d], dtype=np.float64)
        cam2=np.asarray(c2[r.cam_row_2d], dtype=np.float64)
        check(np.isfinite(cam1).all() and np.isfinite(cam2).all(),f'Nonfinite CAM for {r.patient_id}')
        projected_1d, geom_valid = numeric_register_1d(cam1, args.x0, args.x1)
        support = mask2d.sum(axis=0)
        valid = geom_valid & (support>0)
        coverage=float((valid & geom_valid).sum()/max(1,geom_valid.sum()))
        df.iat[j, df.columns.get_loc('waveform_mask_column_coverage')] = coverage
        if coverage<.75:
            problems.append({'patient_id':r.patient_id,'reason':f'Insufficient waveform-mask support {coverage:.3f}'})
            df.iat[j, df.columns.get_loc('processing_note')] = 'BAD_WAVEFORM_MASK'
            continue
        g2_raw = (cam2*mask2d).sum(axis=0)/(support+EPS)
        g1 = grayscale_norm(projected_1d, valid)
        g2 = grayscale_norm(g2_raw, valid)
        if g1 is None or g2 is None:
            problems.append({'patient_id':r.patient_id,'reason':'Constant CAM after projection/normalization'})
            df.iat[j, df.columns.get_loc('processing_note')] = 'DEGENERATE_CAM'
            continue
        res = scores(g1,g2,valid,args.top_fraction)
        if res is None:
            problems.append({'patient_id':r.patient_id,'reason':'Cannot compute rank/linear similarity'})
            df.iat[j, df.columns.get_loc('processing_note')] = 'DEGENERATE_METRICS'
            continue
        a,d,metrics=res
        arrays[0,j]=g1; arrays[1,j]=g2; arrays[2,j]=a; arrays[3,j]=d
        for k,v in metrics.items():
            df.iat[j,df.columns.get_loc(k)]=v
        df.iat[j,df.columns.get_loc('processing_note')]='EXPLORATORY_MEASURED'
        if sum(figure_counts.values())<args.qc_figures and figure_counts[r.true_class] < max(1, int(np.ceil(args.qc_figures/len(CLASSES)))):
            stem=re.sub('[^A-Za-z0-9_.-]','_',f'{r.true_class}_{r.patient_id}')
            qc_plot(r,rgb,g1,g2,a,d,valid,out/'qc_figures'/f'{stem}_module1.png',args.top_fraction)
            figure_counts[r.true_class] += 1
        if (j+1)%100==0 or j+1==len(df):
            print(f'Module 1 inspected: {j+1}/{len(df)}',flush=True)
    for name,a in zip(['g1_normalized_600.npy','g2_projected_normalized_600.npy',
                       'agreement_A_600.npy','disagreement_D_600.npy'],arrays):
        np.save(out/name,a)
    df.to_csv(out/'module1_all_records.csv',index=False)
    measured=df[df.processing_note=='EXPLORATORY_MEASURED'].copy()
    measured.to_csv(out/'module1_primary_exploratory.csv',index=False)
    pd.DataFrame(problems).to_csv(out/'processing_issues.csv',index=False)
    groups = df.groupby(['true_class','analysis_group'],dropna=False).size().rename('n').reset_index()
    groups.to_csv(out/'cohort_flow_by_class.csv', index=False)
    numerical_metrics = measured.groupby('true_class')[['mean_agreement_A','mean_disagreement_D','weighted_jaccard','spearman','dice_top_fraction']].agg(['count','median','mean']).round(6)
    numerical_metrics.to_csv(out/'class_metrics_exploratory.csv')
    summary = {
        'status':'EXPLORATORY_ONLY_PENDING_FULL_TEMPORAL_CALIBRATION',
        'total_pairs_in_full_test':expected,
        'pairs_evaluated_this_run':len(df),
        'control_flow_counts': {str(k):int(v) for k,v in df.analysis_group.value_counts().items()},
        'waveform_auto_pass':int((df.status=='AUTO_PASS').sum()),
        'strict_waveform_qc':int(df.strict_waveform_qc.sum()),
        'both_predictions_correct':int(df.both_correct.sum()),
        'primary_selected':int(df.primary_eligible.sum()),
        'primary_measured':len(measured),
        'primary_processing_issues':len(problems),
        'selected_by_class': {str(k):int(v) for k,v in measured.true_class.value_counts().items()},
        'top_fraction':args.top_fraction,
        'qc_cutoffs':{'median_abs_time_error_ms_max':args.median_max_ms,
                      'p95_abs_time_error_ms_max':args.p95_max_ms,
                      'wrong_patient_margin_positive':True,
                      'auto_pass_required':True,
                      'both_class_predictions_correct_required':True},
        'projection':'2D CAM mean under blue-waveform mask dilated vertically; original ECG bitmap at 1080x288 resized onto CAM 600x300',
        '1d_registration':'linear x0..x1 from independent waveform correspondence audit; assumes 0..10s; pending confirmation from original image creation code',
        'normalization':'each 1D and 2D projected CAM separately min-max normalized over valid x columns',
        'metric_restrictions':'Different classification targets and QC-ineligible records are not compared; NaN stored instead.',
        'important_limitations':['Not reproduction of original author numerical agreement',
                                 'Not quantitatively validated absolute time mapping',
                                 'Grad-CAM 2D native resolution is 9x18; 300x600 map is interpolated',
                                 'Auto-pass and cutoffs are QC rules, not a clinical gold standard',
                                 'Geometric mean is a shared-saliency operator, not proof of faithfulness'],
        'data_sources':{'pairs':args.pairs,'time_qc':args.validation,'CAM_1d':args.cam1,'CAM_2d':args.cam2}
    }
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('STATUS: EXPLORATORY MODULE 1 COMPLETED; NOT FULLY CALIBRATED',flush=True)
    print('Both correct:',summary['both_predictions_correct'],'QC strict:',summary['strict_waveform_qc'],flush=True)
    print('Primary selected:',summary['primary_selected'],'measured:',len(measured),flush=True)
    print('By class:',summary['selected_by_class'],flush=True)
    print('Output:',out,flush=True)


if __name__=='__main__':
    main()
