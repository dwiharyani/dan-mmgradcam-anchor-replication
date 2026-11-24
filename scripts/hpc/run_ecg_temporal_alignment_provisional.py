#!/usr/bin/env python3
"""Provisional temporal alignment of author-style 1D and 2D ECG Grad-CAM.

PURPOSE
  Verify per-image waveform x extents and map each 1D Grad-CAM onto the 600-column
  2D CAM x grid for the EXACT record pairs already verified by patient_id.

SCIENTIFIC GUARDRAIL
  This is PROVISIONAL until the image-generation code confirms that the image
  waveform extends linearly from t=0 to t=10 s between detected x bounds.
  It never performs explanation fusion, disagreement, or faithfulness testing.

INPUTS
  splits/gradcam_test_pairs_verified.csv
  results/gradcam_1d_author_code/heatmaps_1d.npy  (1054, 5000)
  results/gradcam_2d_author_code/heatmaps_2d_resized.npy (1054, 300, 600)
  data/2d_author_clean/Testdata/<class>/hasta_<index>.png

EXAMPLE
  python run_ecg_temporal_alignment_provisional.py --mode inspect
  python run_ecg_temporal_alignment_provisional.py --mode align-provisional

No overwriting of original Grad-CAM data or splits/mm_pairs.csv.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def fullpath(root: Path, s: str) -> Path:
    p = Path(str(s))
    return p if p.is_absolute() else root / p


def identify_waveform_bounds(impath: Path):
    with Image.open(impath) as im:
        rgba = np.asarray(im.convert('RGBA'), dtype=np.uint8)
        width, height = im.size
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3]
    # Prefer transparency when the canvas really is transparent outside the trace.
    active_alpha = alpha >= 35
    alpha_ratio = float(np.mean(active_alpha))
    if alpha_ratio < 0.50:
        ink = active_alpha
        method = 'transparent_alpha'
    else:
        # Source visualizations inspected in preflight use a blue line on white;
        # identify the line by its characteristic blue hue, not saliency values.
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        ink = (b-r >= 22) & (g-r >= 14) & (b-g >= 12) & (alpha >= 20)
        method = 'blue_waveform'
    ink_cols = np.flatnonzero(ink.any(axis=0))
    if ink_cols.size < 0.60 * width:
        raise ValueError(f'Waveform detector found ink in only {ink_cols.size}/{width} columns using {method}: {impath}')
    x0, x1 = int(ink_cols[0]), int(ink_cols[-1])
    # Sanity checks anchored to preflight visual geometry; no silent full-canvas fallback.
    if not (width == 1080 and height == 288):
        raise ValueError(f'Unexpected image geometry {width}x{height} for {impath}')
    if not (25 <= x0 <= 110 and 960 <= x1 <= 1055 and 850 <= x1-x0 <= 1030):
        raise ValueError(f'Unexpected visible waveform x range [{x0}, {x1}] in {impath}')
    return dict(image_w=width, image_h=height, x_start=x0, x_end=x1,
                ink_coverage=round(ink_cols.size/width, 4), detection=method)


def load_checked(root: Path, pairs_arg: str, cam1_arg: str, cam2_arg: str):
    pairs = pd.read_csv(fullpath(root, pairs_arg))
    needed = ['patient_id','cohort_index','true_class','cam_row_1d','cam_row_2d','image_path_2d']
    miss = set(needed)-set(pairs.columns)
    if miss: raise ValueError('Missing paired manifest columns: '+', '.join(sorted(miss)))
    c1 = np.load(fullpath(root, cam1_arg), mmap_mode='r', allow_pickle=False)
    c2 = np.load(fullpath(root, cam2_arg), mmap_mode='r', allow_pickle=False)
    if not (len(pairs)==1054 and c1.shape==(1054,5000) and c2.shape==(1054,300,600)):
        raise ValueError(f'Unexpected pair/CAM dimensions: {len(pairs)}, {c1.shape}, {c2.shape}')
    if pairs['patient_id'].isna().any() or pairs['patient_id'].duplicated().any() or pairs['cohort_index'].duplicated().any():
        raise ValueError('Patient/cohort IDs missing or duplicated')
    for key,n in [('cam_row_1d',c1.shape[0]),('cam_row_2d',c2.shape[0])]:
        ind = pairs[key].astype(int).to_numpy()
        if set(ind)!=set(range(n)):
            raise ValueError(f'{key} does not use all 1054 CAM indices exactly once')
    return pairs,c1,c2


def paint_examples(root: Path, out: Path, geometry: pd.DataFrame, c1, c2, cam_on_x, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=[]
    for cl in ('AFib','GSVT','SB','SR'):
        z=geometry[geometry['true_class']==cl]
        if len(z): examples.append(z.iloc[0])
    if {'pred_class_1d','pred_class_2d'} <= set(geometry.columns):
        disagreement=geometry[geometry['pred_class_1d']!=geometry['pred_class_2d']]
        if len(disagreement) and disagreement.iloc[0]['patient_id'] not in [str(r.patient_id) for r in examples]:
            examples.append(disagreement.iloc[0])
    plot_folder=out/'qc_figures'
    plot_folder.mkdir(parents=True,exist_ok=True)
    for r in examples:
        a=int(r.cam_row_1d); b=int(r.cam_row_2d)
        original=Image.open(fullpath(root,r.image_path_2d)).convert('RGB')
        h2=np.asarray(c2[b],dtype=np.float32)
        if not np.isfinite(h2).all(): raise ValueError('Nonfinite 2D CAM: '+str(r.patient_id))
        norm=lambda v: (v-np.min(v))/(np.max(v)-np.min(v)+1e-8)
        resized_cam=np.asarray(Image.fromarray(norm(h2)).resize((r.image_w,r.image_h),Image.Resampling.BILINEAR))
        x0,x1=int(r.x_start),int(r.x_end)
        ticks=[0,2,4,6,8,10]
        tick_x=[x0+(x1-x0)*t/10 for t in ticks]
        fig,axs=plt.subplots(3,1,figsize=(12,8.0),gridspec_kw={'height_ratios':[1.0,2.2,2.2]},layout='constrained')
        t1=np.linspace(0,10,5000)
        axs[0].plot(t1,np.asarray(c1[a]),color='#AC4770',lw=0.8)
        axs[0].set_xlim(0,10)
        axs[0].set_ylabel('1D CAM')
        axs[0].set_title('Original 1D Grad-CAM (ECG time)')
        axs[0].grid(alpha=.2)
        axs[1].imshow(original,extent=[0,r.image_w,r.image_h,0],aspect='auto')
        axs[1].imshow(resized_cam,extent=[0,r.image_w,r.image_h,0],alpha=.35,cmap='turbo',aspect='auto')
        axs[1].set_title('Original ECG image + 2D Grad-CAM (image-space)')
        axs[2].imshow(original,extent=[0,r.image_w,r.image_h,0],aspect='auto')
        raw=cam_on_x[int(r.name)]
        band=np.tile(raw[None,:],(r.image_h,1))
        mapped=np.array(Image.fromarray(band).resize((r.image_w,r.image_h),Image.Resampling.NEAREST), copy=True)
        # Use detected trace interval mask so image margins are uncolored.
        mapped[:,:x0]=0; mapped[:,x1+1:]=0
        axs[2].imshow(mapped,extent=[0,r.image_w,r.image_h,0],cmap='Reds',
                      alpha=np.clip(mapped*0.65,0,0.65),vmin=0,vmax=1,aspect='auto')
        axs[2].set_title('1D Grad-CAM registered to 2D image x-axis (PROVISIONAL)')
        for ax in axs[1:]:
            ax.set_xlim(0,r.image_w); ax.set_ylim(r.image_h,0)
            ax.set_xticks(tick_x,ticks)
            ax.set_xlabel('Assumed ECG time (s) — requires image-render code confirmation')
            ax.axvline(x0,color='black',ls=':',alpha=.45,lw=1)
            ax.axvline(x1,color='black',ls=':',alpha=.45,lw=1)
        fig.suptitle(f'PROVISIONAL 1D–2D TEMPORAL REGISTRATION | {r.true_class} | {r.patient_id}\n'
                     f'1D row {a} / 2D row {b} | x in original PNG: {x0}–{x1} / {r.image_w}',fontsize=11)
        target=plot_folder/(re.sub(r'[^\w.-]','_',f'{r.true_class}_{r.patient_id}')+'.png')
        fig.savefig(target,dpi=125)
        plt.close(fig)
        print('QC:',target)


def main():
    p=argparse.ArgumentParser(description='Auditable provisional ECG 1D->2D CAM temporal alignment')
    p.add_argument('--project',default='.')
    p.add_argument('--mode',choices=['inspect','align-provisional'],default='inspect')
    p.add_argument('--pairs',default='splits/gradcam_test_pairs_verified.csv')
    p.add_argument('--cam1',default='results/gradcam_1d_author_code/heatmaps_1d.npy')
    p.add_argument('--cam2',default='results/gradcam_2d_author_code/heatmaps_2d_resized.npy')
    p.add_argument('--out',default='results/temporal_alignment_provisional')
    args=p.parse_args()
    root=Path(args.project).resolve()
    out=fullpath(root,args.out)
    out.mkdir(parents=True,exist_ok=True)
    pairs,c1,c2=load_checked(root,args.pairs,args.cam1,args.cam2)
    all_records=[]; errors=[]
    for k,row in pairs.iterrows():
        try:
            im=fullpath(root,row.image_path_2d)
            if not im.is_file(): raise FileNotFoundError(im)
            info=identify_waveform_bounds(im)
            rec={**row.to_dict(),**info}
            all_records.append(rec)
        except Exception as e:
            errors.append(f'pair_row={k} patient={row.patient_id}: {e}')
        if (k+1)%200 == 0: print(f'Geometry inspected: {k+1}/{len(pairs)}')
    if errors:
        (out/'geometry_errors.txt').write_text('\n'.join(errors)+'\n')
        print('GEOMETRY AUDIT FAIL:',len(errors),'errors. First 10:')
        for e in errors[:10]: print(e)
        print('See:',out/'geometry_errors.txt')
        raise SystemExit('STOP: no alignment generated; inspect geometry audit errors')
    g=pd.DataFrame.from_records(all_records)
    g.to_csv(out/'waveform_geometry_audit.csv',index=False)
    print('GEOMETRY AUDIT PASS:',len(g),'verified waveform extents')
    print('X start: min/median/max',int(g.x_start.min()),float(g.x_start.median()),int(g.x_start.max()))
    print('X end  : min/median/max',int(g.x_end.min()),float(g.x_end.median()),int(g.x_end.max()))
    print('Ink coverage: min/median/max',float(g.ink_coverage.min()),float(g.ink_coverage.median()),float(g.ink_coverage.max()))
    if args.mode=='inspect':
        print('STATUS: INSPECT PASS (NO alignment generated)')
        print('NEXT: verify image rendering logic, then run --mode align-provisional for visual QC')
        return
    # Keep rows in sorted pairing manifest order for clear array <-> CSV correspondence.
    registered=np.zeros((len(g),c2.shape[2]),dtype=np.float32)
    in_waveform=np.zeros_like(registered,dtype=np.uint8)
    for k,r in g.iterrows():
        signal=np.asarray(c1[int(r.cam_row_1d)],dtype=np.float32)
        if not np.isfinite(signal).all(): raise ValueError('Nonfinite 1D CAM '+str(r.patient_id))
        cam_width=c2.shape[2]
        x_orig=(np.arange(cam_width,dtype=np.float64)+.5)*(float(r.image_w)/cam_width)-.5
        x0,x1=float(r.x_start),float(r.x_end)
        mask=(x_orig>=x0)&(x_orig<=x1)
        frac=np.clip((x_orig-x0)/(x1-x0),0.0,1.0)
        value=np.interp(frac*(signal.size-1),np.arange(signal.size),signal).astype(np.float32)
        # Preserve source saliency scale; no per-sample renormalization.
        registered[k,mask]=value[mask]
        in_waveform[k,mask]=1
    np.save(out/'cam1d_registered_on_2d_grid_600.npy',registered)
    np.save(out/'waveform_valid_x_600.npy',in_waveform)
    g.insert(0,'aligned_row',np.arange(len(g),dtype=int))
    g['registered_1d_cam_row']='aligned_row'
    g.to_csv(out/'aligned_manifest.csv',index=False)
    meta={
        'status':'PROVISIONAL_PENDING_IMAGE_RENDER_TIME_AXIS_VERIFICATION',
        'not_validated_for_fusion_or_agreement':True,
        'pairs':len(g),
        'source_1d_shape':list(c1.shape),'source_2d_shape':list(c2.shape),
        'aligned_cam_1d_shape':list(registered.shape),
        'image_pixel_mapping':'original_x=(resized_cam_column+0.5)*original_width/600-0.5',
        'ecg_time_assumption':'0 to 10 seconds linearly across detected waveform x_start..x_end',
        'no_fusion_performed':True,
        'explanation_cam_values_normalized':False,
        'prediction_disagreements_excluded':False,
    }
    (out/'alignment_status.json').write_text(json.dumps(meta,indent=2)+'\n')
    paint_examples(root,out,g,c1,c2,registered,in_waveform)
    print('STATUS: PROVISIONAL TEMPORAL ALIGNMENT SAVED (NOT YET VALIDATED)')
    print('Saved:',out)
    print('Next: inspect QC figures and confirm from image-render code that x0..x1 represents 0..10 seconds linearly.')

if __name__=='__main__':
    main()
