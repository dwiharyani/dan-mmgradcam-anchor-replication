#!/usr/bin/env python3
"""Conservative 1D-signal versus 2D-image temporal validation for MM-GradCAM.

Run in the existing MM_GradCAM_HPC project root. READS existing original
HDF5 signal X, the 10,588-record cohort index and verified 1,054 paired image
paths. It does NOT use the 1D/2D Grad-CAM maps to assess temporal validity,
thus avoids circular validation from overlay images.

1) Audit source identity and 1D label for every selected ECG.
2) Independently locate prominent QRS-like signal peaks in H5 X.
3) Extract prominent narrow vertical blue waveform features from the ORIGINAL
   ECG image and locate their peaks.
4) Project signal sample indices to image x-coordinate using the provisional
   time mapping and evaluate one-to-one peak correspondence.
5) Produce per-record timing errors and QC plots; NEVER modify existing data.

NOTE: This is an independent *image/signal correspondence* test, not proof
that the image was generated with that exact mapping. Automated R-peak/QRS
heuristics must be visually audited. A model's Grad-CAM explanation class
may differ even if the underlying ECG is paired correctly.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks

CLASSES = {0: 'AFib', 1: 'GSVT', 2: 'SB', 3: 'SR'}


def local(root, name):
    p = Path(name)
    return p if p.is_absolute() else root / p


def original_blue_trace(im):
    """Detect a cyan/blue raster line; do not use any Grad-CAM overlay image.

    The original 2D ECG files in this project are light-blue waveforms on a
    white background. In case rendering changed, fail for manual inspection.
    """
    rgb = im.astype(np.int16)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Reject near-white pixels and keep blue-dominant waveform pixels.
    return ((B - R >= 14) & (B - G >= 5) & (G - R >= 4)
            & (R < 240) & (G < 250))


def image_qrs_candidates(original, x0, x1):
    mask = original_blue_trace(original)
    H, W = mask.shape
    mask[:5, :] = False
    mask[H - 5:, :] = False
    columns = mask[:, x0:x1 + 1]
    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    spans = np.zeros(len(xs), dtype=np.float64)
    support = np.zeros(len(xs), dtype=np.int32)
    centers = np.full(len(xs), np.nan, dtype=np.float64)
    for k in range(len(xs)):
        yp = np.flatnonzero(columns[:, k])
        if yp.size:
            spans[k] = yp[-1] - yp[0] + 1
            support[k] = yp.size
            centers[k] = np.median(yp)
    frac_ink = np.mean(support > 0)
    sm = gaussian_filter1d(spans, 1.4)
    min_dist = max(10, int((x1 - x0) * 0.19 / 10))  # about 190 ms
    if sm.max() > 0:
        pks, props = find_peaks(sm, distance=min_dist,
                                height=max(11., sm.max() * .25),
                                prominence=max(8., sm.max() * .12))
    else:
        pks = np.array([], dtype=int)
    return xs[pks], xs, spans, sm, centers, frac_ink


def signal_qrs_candidates(signal, sampling_rate=500):
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size != 5000 or not np.isfinite(sig).all():
        raise ValueError(f'Signal must be finite 5000 samples, got {sig.shape}')
    baseline = median_filter(sig, size=151, mode='nearest')
    residual = sig - baseline
    candidates = []
    # Evaluate both polarities independently. Some ECG leads have inverted QRS.
    for polarity in (1, -1):
        f = gaussian_filter1d(polarity * residual, 3.0)
        magnitude = float(np.percentile(np.abs(f), 99.9))
        if magnitude < 1e-9:
            candidates.append((polarity, np.array([], dtype=np.int64), f))
            continue
        peaks, _ = find_peaks(f, distance=int(.25 * sampling_rate),
                              prominence=max(magnitude * .19, float(np.std(f)) * .55),
                              height=magnitude * .24)
        candidates.append((polarity, peaks, f))
    return candidates


def gated_match(pred, obs, tolerance_px):
    # Monotonic one-to-one matching: never re-use a peak from the same image.
    # Only candidates within tolerance are considered. Greedy matching by
    # smallest absolute error is globally sorted to avoid nearest-neighbor
    # reuse at different predicted peaks.
    d = np.abs(pred[:, None] - obs[None, :]) if len(pred) and len(obs) else np.empty((len(pred), len(obs)))
    candidates = np.argwhere(d <= tolerance_px)
    candidates = sorted(candidates, key=lambda ij: float(d[ij[0], ij[1]]))
    used_p, used_o = set(), set()
    matched = []
    for pi, oi in candidates:
        a, b = int(pi), int(oi)
        if a in used_p or b in used_o:
            continue
        used_p.add(a)
        used_o.add(b)
        matched.append((a, b, float(obs[b] - pred[a])))
    return sorted(matched)


def validate_one(pair, X, h5_y, cohort, root, x0, x1, fs, tolerance_ms):
    pid = str(pair.patient_id)
    idx = int(pair.cohort_index)
    if idx not in cohort.index:
        raise ValueError(f'cohort_index missing in source cohort: {idx}')
    record = cohort.loc[idx]
    if str(record.patient_id) != pid or str(record.class_name) != str(pair.true_class):
        raise ValueError(f'Identity mismatch for index {idx}: {pid}')
    if 'class_id' in cohort.columns and int(h5_y[idx]) != int(record.class_id):
        raise ValueError(f'H5 y/class_id mismatch for {pid}: H5 {h5_y[idx]} vs cohort {record.class_id}')
    img_path = local(root, str(pair.image_path_2d))
    if img_path.stem != f'hasta_{idx}' or img_path.parent.name != str(pair.true_class):
        raise ValueError(f'Image file does not match cohort/class: {img_path}')
    with Image.open(img_path) as im:
        arr = np.asarray(im.convert('RGB'))
    if arr.shape != (288, 1080, 3):
        raise ValueError(f'Unexpected image shape: {arr.shape} for {pid}')
    sig = np.asarray(X[idx], dtype=np.float64)
    img_pks, xs, spans, img_profile, centers, coverage = image_qrs_candidates(arr, x0, x1)
    choices = signal_qrs_candidates(sig, fs)
    all_candidates = []
    for polarity, sig_pks, filt in choices:
        pred_pks = x0 + sig_pks * (x1 - x0) / 4999.
        tol_pix = tolerance_ms / 1000. * (x1 - x0) / (5000 / fs)
        matched = gated_match(pred_pks, img_pks, tol_pix)
        errors_ms = np.asarray([m[2] * 1000 * (5000 / fs) / (x1 - x0) for m in matched])
        precision = len(matched) / max(len(sig_pks), len(img_pks), 1)
        median_error = float(np.median(np.abs(errors_ms))) if errors_ms.size else float('nan')
        p95_error = float(np.percentile(np.abs(errors_ms), 95)) if errors_ms.size else float('nan')
        all_candidates.append((len(matched), precision, -median_error if np.isfinite(median_error) else -1e9,
                               polarity, sig_pks, pred_pks, matched, errors_ms, filt, median_error, p95_error))
    best = max(all_candidates, key=lambda c: (c[0], c[1], c[2]))
    nmatch, precision, _, pol, sig_pks, pred_pks, matched, err_ms, filtered, med, p95 = best
    status = 'AUTO_PASS' if (len(sig_pks) >= 5 and len(img_pks) >= 5 and nmatch >= 5
                             and precision >= .75 and med <= 50 and p95 <= 100
                             and coverage >= .70) else 'MANUAL_REVIEW'
    record = {
        'patient_id': pid, 'cohort_index': idx, 'true_class': pair.true_class,
        'image_path_2d': str(pair.image_path_2d),
        'status': status, 'polarity_for_peak_detector': 'positive' if pol == 1 else 'negative',
        'signal_peak_count': int(len(sig_pks)), 'image_peak_count': int(len(img_pks)),
        'matched_peak_count': int(nmatch), 'match_fraction': float(precision),
        'median_abs_time_error_ms': med, 'p95_abs_time_error_ms': p95,
        'image_trace_ink_column_coverage': float(coverage),
        'x0_assumed': x0, 'x1_assumed': x1,
    }
    diagnostic = (sig, sig_pks, filtered, arr, img_pks, xs, spans, img_profile,
                  pred_pks, matched, pol)
    return record, diagnostic


def plot_qc(record, diagnostic, out_dir, fs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    signal, sig_pks, filtered, img, img_pks, xs, spans, profile, pred_pks, matched, pol = diagnostic
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), dpi=115, constrained_layout=True)
    axes[0].imshow(img)
    axes[0].set_title('Original 2D ECG — predicted H5 signal peaks (magenta) versus image-derived peaks (green)')
    for x in pred_pks:
        axes[0].axvline(x, color='#D33175', linewidth=.75, alpha=.52, linestyle='--')
    for x in img_pks:
        axes[0].axvline(x, color='#267A52', linewidth=.7, alpha=.50)
    axes[0].set_xlim(0, 1080)
    axes[0].set_xlabel('Original image x-coordinate (pixels)')
    axes[1].plot(np.arange(len(signal)) / fs, signal, color='#405C7B', linewidth=.7)
    if len(sig_pks): axes[1].scatter(sig_pks / fs, signal[sig_pks], color='#D33175', s=14)
    axes[1].set_xlim(0, len(signal)/fs)
    axes[1].set_title('Independent HDF5 1D signal and detected QRS candidates')
    axes[1].set_xlabel('Seconds')
    axes[2].plot(xs, profile, color='#3A7A69', linewidth=1., label='Image vertical blue-pixel span')
    if len(img_pks):
        axes[2].scatter(img_pks, np.interp(img_pks, xs, profile), color='#267A52', s=22, label='Detected image peaks')
    for x in pred_pks:
        axes[2].axvline(x, color='#D33175', linewidth=.65, alpha=.5, linestyle='--')
    axes[2].set_xlim(xs[0],xs[-1]); axes[2].set_title('Image feature profile and predicted 1D peak positions')
    axes[2].set_xlabel('Original image x-coordinate (pixels)')
    axes[2].legend(loc='upper right', fontsize=8)
    err = record['median_abs_time_error_ms']; p95 = record['p95_abs_time_error_ms']
    fig.suptitle(f"{record['true_class']} | {record['patient_id']} | {record['status']} | "
                 f"matched {record['matched_peak_count']}/{max(record['signal_peak_count'],record['image_peak_count'])} | "
                 f"median {err:.1f} ms; p95 {p95:.1f} ms", fontsize=12, weight='bold')
    name = re.sub(r'[^A-Za-z0-9_.-]+','_',f"{record['true_class']}_{record['patient_id']}")
    fig.savefig(out_dir/(name+'_peak_validation.png'), dpi=125)
    plt.close(fig)


def run_validation(args):
    root = Path(args.root).resolve()
    h5_path = local(root, args.h5)
    pairs = pd.read_csv(local(root,args.pairs))
    cohort_csv = pd.read_csv(local(root,args.cohort))
    required = ('cohort_index', 'patient_id', 'true_class', 'image_path_2d')
    for col in required:
        if col not in pairs.columns: raise ValueError(f'Pair file missing required {col}')
    for col in ('cohort_index', 'patient_id', 'class_name'):
        if col not in cohort_csv.columns: raise ValueError(f'Cohort file missing required {col}')
    if pairs.patient_id.duplicated().any() or pairs.cohort_index.duplicated().any():
        raise ValueError('Duplicate IDs in verified pair manifest')
    if len(pairs) != 1054 or len(cohort_csv) != 10588:
        raise ValueError(f'Unexpected count: {len(pairs)} test pairs / {len(cohort_csv)} cohort')
    if not cohort_csv.cohort_index.eq(np.arange(len(cohort_csv))).all():
        raise ValueError('Cohort index not sequential 0..10587. Cannot use as H5 row index.')
    cohort = cohort_csv.set_index('cohort_index',verify_integrity=True)
    if args.all:
        subset = pairs.copy()
    else:
        selected=[]
        for cls in ('AFib','GSVT','SB','SR'):
            s = pairs[pairs.true_class.eq(cls)]
            if len(s) < args.per_class: raise ValueError(f'Insufficient {cls} samples')
            selected.append(s.head(args.per_class))
        subset = pd.concat(selected,ignore_index=True)
    out = local(root,args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory exists and is not empty: {out}; set another --out path')
    qc_dir = out/'qc_figures'
    qc_dir.mkdir(parents=True,exist_ok=True)
    # Deterministic same-class wrong-patient image for a negative control.
    # Its QRS-like pattern should not match more closely than the true image.
    wrong_image_by_pid = {}
    for cls, grp in pairs.groupby('true_class'):
        grp = grp.sort_values('cohort_index').reset_index(drop=True)
        for k in range(len(grp)):
            wrong_image_by_pid[str(grp.iloc[k].patient_id)] = str(grp.iloc[(k+1)%len(grp)].image_path_2d)
    rows=[]
    qc_count=0
    qc_by_class=Counter()
    with h5py.File(h5_path,'r') as h5:
        for k in ('X','y'):
            if k not in h5: raise ValueError(f'HDF5 missing dataset key {k}')
        X,y = h5['X'],h5['y']
        if X.shape != (10588,5000) or y.shape != (10588,):
            raise ValueError(f'Unexpected H5 shapes: X={X.shape}, y={y.shape}')
        for k, rec in enumerate(subset.itertuples(index=False),1):
            result,diag = validate_one(rec,X,y,cohort,root,args.x0,args.x1,args.fs,args.match_tolerance_ms)
            # Negative control: compare this signal's peak coordinates to an
            # unrelated image from the SAME rhythm class. Strong regularity can
            # produce accidental matching; such records need manual review.
            wrong_path = local(root, wrong_image_by_pid[str(rec.patient_id)])
            with Image.open(wrong_path) as wrong_im:
                wrong_arr = np.asarray(wrong_im.convert('RGB'))
            wrong_pks, _, _, _, _, _ = image_qrs_candidates(wrong_arr, args.x0, args.x1)
            mapped_sig_peaks = diag[8]
            tol_px = args.match_tolerance_ms/1000 * (args.x1-args.x0) / (5000/args.fs)
            neg_pairs = gated_match(mapped_sig_peaks, wrong_pks, tol_px)
            wrong_frac = len(neg_pairs) / max(len(mapped_sig_peaks),len(wrong_pks),1)
            result['wrong_patient_same_class_match_fraction'] = float(wrong_frac)
            result['true_minus_wrong_match_fraction'] = float(result['match_fraction']-wrong_frac)
            if result['status']=='AUTO_PASS' and (result['match_fraction']-wrong_frac)<.10:
                result['status']='MANUAL_REVIEW'
                result['review_reason']='matched peaks not clearly stronger than wrong-patient control'
            elif result['status']=='AUTO_PASS':
                result['review_reason']='automatic criteria and negative control satisfied'
            else:
                result['review_reason']='insufficient or imprecise QRS correspondence'
            rows.append(result)
            # First few per class; all severe cases if requested.
            choose = (not args.all) or (qc_by_class[str(rec.true_class)] < 2 or
                    result['status']=='MANUAL_REVIEW')
            if choose and qc_count < args.qc_max:
                plot_qc(result,diag,qc_dir,args.fs)
                qc_count+=1
                qc_by_class[str(rec.true_class)]+=1
            if k%100==0 or k==len(subset): print(f'Validated {k}/{len(subset)}',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(out/'temporal_alignment_validation.csv',index=False)
    class_summary = df.groupby('true_class').agg(
        records=('patient_id','count'), automatic_pass=('status',lambda s:int((s=='AUTO_PASS').sum())),
        median_error_ms=('median_abs_time_error_ms','median'),
        median_match_fraction=('match_fraction','median')).reset_index()
    class_summary.to_csv(out/'class_summary.csv',index=False)
    cnt = Counter(df.status)
    def json_safe(v):
        f = float(v)
        return f if np.isfinite(f) else None
    summary = {
        'stage':'independent_1d_vs_original_2d_waveform_peak_correspondence',
        'not_gradcam_to_gradcam_comparison':True,
        'not_automatically_proof_of_exact_rendering_mapping':True,
        'hdf5_source':str(h5_path), 'test_pairs_file':args.pairs,
        'source_cohort':args.cohort, 'x0_assumed':args.x0,'x1_assumed':args.x1,
        'n':len(df),'class_counts':df.true_class.value_counts().to_dict(),
        'status_counts':dict(cnt),'auto_pass_fraction':float((df.status=='AUTO_PASS').mean()),
        'median_record_median_error_ms':json_safe(df.median_abs_time_error_ms.median()),
        'median_record_match_fraction':json_safe(df.match_fraction.median()),
        'median_wrong_patient_match_fraction':json_safe(df.wrong_patient_same_class_match_fraction.median()),
        'fraction_true_match_stronger_than_control':float((df.true_minus_wrong_match_fraction>0).mean()),
        'qc_figures':qc_count,
        'interpretation':'AUTO_PASS = heuristic QRS peak correspondence at assumed x-range that also exceeds a same-class wrong-patient control. Review QC before declaring image-time registration validated. MANUAL_REVIEW is not necessarily a biological error.'
    }
    (out/'alignment_validation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('\nRESULTS:',out)
    print('Waveform peak matching:',dict(cnt))
    print('Auto-pass fraction:',f"{summary['auto_pass_fraction']:.1%}")
    print('Median matched-vs-wrong match fractions:',
          round(summary['median_record_match_fraction'],3),
          round(summary['median_wrong_patient_match_fraction'],3))
    print('Median error across records:',summary['median_record_median_error_ms'],'ms')
    print('STATUS: AUTOMATED WAVEFORM CORRESPONDENCE AUDIT COMPLETE — CHECK QC; TIME CALIBRATION STILL NEEDS REVIEW')


def self_test():
    # Synthetic 10s signal with 14 distinct R peaks and raster ECG drawn from
    # *the same* 1D vector. Also test mismatched raster shifted by 1.3 seconds.
    from PIL import ImageDraw
    n=5000; x0,x1=58,1021; fs=500
    rng=np.random.default_rng(42)
    r_times=np.array([.37,.98,1.59,2.28,2.97,3.50,4.12,4.85,5.34,6.06,6.78,7.39,8.20,8.88,9.44])
    x=np.arange(n)/fs
    sig=.012*np.sin(x*2*np.pi*.7)
    for t in r_times: sig+=1.6*np.exp(-.5*((x-t)/.011)**2)
    sig+=rng.normal(0,.002,n)
    def raster(v):
        xx=np.linspace(0,4999,x1-x0+1)
        xx_y=145-92*np.interp(xx,np.arange(n),v)
        im=Image.new('RGB',(1080,288),'white')
        ImageDraw.Draw(im).line([(x0+i,int(y)) for i,y in enumerate(xx_y)],fill=(65,154,205),width=2)
        return np.asarray(im)
    for label,img in (('MATCHED',raster(sig)),('SHIFTED',raster(np.roll(sig,650)))):
        ip,_,_,_,_,coverage=image_qrs_candidates(img,x0,x1)
        sp=signal_qrs_candidates(sig)
        choices=[]
        for pol,pk,_ in sp:
            mapped=x0+pk*(x1-x0)/4999
            ma=gated_match(mapped,ip,100/1000*(x1-x0)/10)
            choices.append((len(ma),pol,len(pk),len(ip),[abs(m[2])*1000*10/(x1-x0) for m in ma]))
        best=max(choices,key=lambda x:x[0])
        print(label,'matches=',best[0], 'signal peaks=',best[2],'image peaks=',best[3],
              'median ms=',round(float(np.median(best[4])),2) if best[4] else None,'coverage=',round(coverage,3))
        if label=='MATCHED' and best[0]<10:
            raise AssertionError('Synthetic positive control failed')
        if label=='SHIFTED' and best[0]>=10:
            raise AssertionError('Synthetic negative control unexpectedly passed')
    print('SELF-TEST PASS')


def main():
    p=argparse.ArgumentParser(description='Independent 1D H5 signal vs original 2D raster ECG alignment audit')
    p.add_argument('--root',default='.')
    p.add_argument('--h5',default='data/1d/mm_gradcam_leadII_10588.h5')
    p.add_argument('--cohort',default='splits/final_cohort_10588.csv')
    p.add_argument('--pairs',default='splits/gradcam_test_pairs_verified.csv')
    p.add_argument('--out',default='results/temporal_validation_rpeak_v1')
    p.add_argument('--all',action='store_true',help='Analyze all 1,054 pairs; otherwise sample per class')
    p.add_argument('--per-class',type=int,default=6)
    p.add_argument('--qc-max',type=int,default=25)
    p.add_argument('--x0',type=int,default=58)
    p.add_argument('--x1',type=int,default=1021)
    p.add_argument('--fs',type=int,default=500)
    p.add_argument('--match-tolerance-ms',type=float,default=100)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test(); return
    if not (0<=args.x0<args.x1<1080):p.error('x bounds invalid')
    if args.fs != 500:p.error('Data in this experiment are fixed 500 Hz; confirm independently before changing this value')
    if args.per_class <= 0 or args.qc_max<=0 or args.match_tolerance_ms<=0:p.error('positive counts and tolerance required')
    run_validation(args)


if __name__=='__main__':
    main()
