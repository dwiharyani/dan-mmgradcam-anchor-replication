#!/usr/bin/env python3
"""DAN-MMGradCAM Module 3A: read-only integration/audit of matched Module 2 output.

Does not execute models or new perturbations. Does not modify Module 1/2 files.
Module 3A R(u) is exactly the saved jointly accepted R map from Module 2.
No-stability F_mu is a controlled ablation ONLY with 30 jointly accepted perturbations,
because mu1 and mu2 are otherwise calculated on branch-specific accepted sets.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def finite_corr(x, y):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12: return np.nan
    return float(np.corrcoef(x[ok],y[ok])[0,1])


def support_iou(x,y,percent=.1):
    if len(x)!=len(y) or len(x)<2: return np.nan
    k=max(1,int(np.ceil(percent*len(x))))
    a=set(np.argsort(-x,kind='stable')[:k].tolist())
    b=set(np.argsort(-y,kind='stable')[:k].tolist())
    return float(len(a&b)/len(a|b))


def peaknorm(a):
    a=np.asarray(a,dtype=float)
    peak=np.nanmax(a) if np.isfinite(a).any() else 0
    return a/peak if peak>0 else a


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m2-dir',required=True,help='New s9534 Module2 results folder')
    ap.add_argument('--primary',required=True,help='Frozen Module1 727-primary CSV')
    ap.add_argument('--m1-valid',required=True,help='Module1 valid_temporal_mask.npy')
    ap.add_argument('--m1-metadata',required=True,help='Module1 gradcam_metadata.csv')
    ap.add_argument('--m1-1d',default=None,help='Optional Module1 aligned 1D Grad-CAM array')
    ap.add_argument('--highlight',type=int,default=209)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    out=Path(a.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    p=Path(a.m2_dir).expanduser()
    paths=[p/'noise_stability_maps.npz',p/'noise_stability_records.csv',p/'run_metadata.json',Path(a.primary).expanduser(),Path(a.m1_valid).expanduser(),Path(a.m1_metadata).expanduser()]
    if a.m1_1d:paths.append(Path(a.m1_1d).expanduser())
    for f in paths:
        if not f.is_file():raise FileNotFoundError(f)
    meta=json.loads((p/'run_metadata.json').read_text())
    csv=pd.read_csv(p/'noise_stability_records.csv')
    pri=pd.read_csv(Path(a.primary).expanduser())
    m1meta=pd.read_csv(Path(a.m1_metadata).expanduser())
    for name,d in [('module2',csv),('primary',pri),('Module1 metadata',m1meta)]:
        if 'cohort_index' not in d:raise ValueError(f'{name}: missing cohort_index')
        if d.cohort_index.duplicated().any():raise ValueError(f'{name}: duplicate cohort_index; resolve before continuing')
    primary=set(map(int,pri.cohort_index));m1_ids=m1meta.cohort_index.to_numpy(dtype=int)
    m1_lookup={int(v):i for i,v in enumerate(m1_ids)}
    valid1=np.load(Path(a.m1_valid).expanduser(),mmap_mode='r')
    if valid1.shape!=(len(m1meta),600):raise ValueError(f'Module1 valid shape {valid1.shape} != ({len(m1meta)},600)')
    m1g1=np.load(Path(a.m1_1d).expanduser(),mmap_mode='r') if a.m1_1d else None
    if m1g1 is not None and m1g1.shape!=(len(m1meta),600):raise ValueError(f'Module1 1D shape {m1g1.shape} mismatch')
    with np.load(p/'noise_stability_maps.npz',allow_pickle=False) as z:
        for key in ['cohort_index','filename','valid_600','mu1','mu2','S1','S2','H1j','H2j','R','base1','base2','keep_1d','keep_2d','degenerate_1d','degenerate_2d']:
            if key not in z:raise ValueError(f'Missing Module2 array {key}')
        ids=z['cohort_index'].astype(int)
        if len(set(ids))!=len(ids):raise ValueError('Module2 duplicate IDs')
        if not np.array_equal(ids,csv.cohort_index.to_numpy(dtype=int)):raise ValueError('Module2 NPZ/CSV row order mismatch')
        if not np.array_equal(z['filename'],csv.ecg_csv_filename.astype(str).to_numpy()):raise ValueError('Module2 filenames mismatch')
        masks=z['valid_600'].astype(bool)
        if masks.shape!=(len(ids),600):raise ValueError('Unexpected valid mask shape')
        need=['mu1','mu2','S1','S2','H1j','H2j','R','base1','base2']
        vals={k:z[k].astype(np.float32) for k in need}
        if any(v.shape!=(len(ids),600) for v in vals.values()):raise ValueError('Unexpected M2 temporal array shape')
        records=[];keep_idx=[]
        def iso_at(idx,m):
            return float(np.nanmean(vals['R'][idx,m]))
        for i,rec in csv.iterrows():
            cid=int(ids[i]);valid=masks[i]; inprimary=cid in primary
            if cid not in m1_lookup:raise ValueError(f'Missing Module1 metadata for ID {cid}')
            j=m1_lookup[cid];mask_match=bool(np.array_equal(valid,np.asarray(valid1[j]).astype(bool)))
            label_match=(str(rec['class_name'])==str(m1meta.iloc[j]['class_name'])) if 'class_name' in m1meta else True
            bothcorrect=(rec.pred_1d==rec.class_name and rec.pred_2d==rec.class_name)
            jointn=int(rec.n_joint_accepted)
            formula=np.sqrt(np.clip(vals['H1j'][i,valid],0,None).astype(float)*np.clip(vals['H2j'][i,valid],0,None).astype(float))
            rawr=vals['R'][i,valid].astype(float)
            finiteR=bool(valid.any() and np.isfinite(rawr).all() and np.isfinite(formula).all())
            err=float(np.max(np.abs(formula-rawr))) if finiteR else np.nan
            eligible=bool(inprimary and bothcorrect and jointn>=2 and finiteR and err<5e-6 and mask_match and label_match)
            fulljoint=bool(eligible and jointn==30 and bool(z['keep_1d'][i].all() and z['keep_2d'][i].all() and not z['degenerate_1d'][i].any() and not z['degenerate_2d'][i].any()))
            m1_corr=np.nan;m1_mae=np.nan
            if m1g1 is not None and valid.any():
                g1x=peaknorm(m1g1[j,valid]);g1y=peaknorm(vals['base1'][i,valid]);m1_corr=finite_corr(g1x,g1y);m1_mae=float(np.nanmean(np.abs(g1x-g1y)))
            row={'cohort_index':cid,'class_name':str(rec.class_name),'in_primary':inprimary,'both_correct_m2':bothcorrect,'mask_equal':mask_match,'class_equal':label_match,'n_joint_accepted':jointn,'eligible_3a':eligible,'full_joint_30':fulljoint,'r_formula_max_error':err,'r_mean':iso_at(i,valid) if finiteR else np.nan,'m1_m2_baseline_1d_pearson':m1_corr,'m1_m2_baseline_1d_mae':m1_mae}
            if eligible:
                keep_idx.append(i)
                r=vals['R'][i,valid].astype(float)
                f_base=np.sqrt(np.clip(vals['base1'][i,valid],0,None).astype(float)*np.clip(vals['base2'][i,valid],0,None).astype(float))
                row['fbase_mean']=float(f_base.mean())
                if fulljoint:
                    f_mu=np.sqrt(np.clip(vals['mu1'][i,valid],0,None).astype(float)*np.clip(vals['mu2'][i,valid],0,None).astype(float))
                    row['fmu_mean']=float(f_mu.mean());row['r_fmu_intensity_ratio']=float(r.sum()/f_mu.sum()) if f_mu.sum()>0 else np.nan
                    row['r_fmu_shape_pearson']=finite_corr(r,f_mu)
                    row['r_fmu_top10_iou']=support_iou(r,f_mu)
                    row['matched_h1_error']=float(np.max(np.abs(vals['H1j'][i,valid]-vals['mu1'][i,valid]*vals['S1'][i,valid])))
                    row['matched_h2_error']=float(np.max(np.abs(vals['H2j'][i,valid]-vals['mu2'][i,valid]*vals['S2'][i,valid])))
            records.append(row)
        df=pd.DataFrame(records)
        if not df.in_primary.sum()==len(primary):raise ValueError(f'Only {int(df.in_primary.sum())}/{len(primary)} frozen primary IDs found in Module2')
        if not df.loc[df.in_primary,'mask_equal'].all():raise ValueError('Module1 vs Module2 valid mask mismatch: abort')
        if not df.loc[df.in_primary,'class_equal'].all():raise ValueError('Module1 vs Module2 class mismatch: abort')
        ix=np.array(keep_idx,dtype=int)
        if len(ix)==0:raise ValueError('No eligible primary ECG after audit')
        mask_eligible=masks[ix]
        fbase=np.sqrt(np.clip(vals['base1'][ix],0,None)*np.clip(vals['base2'][ix],0,None))
        mu=np.sqrt(np.clip(vals['mu1'][ix],0,None)*np.clip(vals['mu2'][ix],0,None))
        fulljoint_flags=df.set_index('cohort_index').loc[ids[ix],'full_joint_30'].to_numpy(dtype=bool)
        mu[~fulljoint_flags,:]=np.nan
        fbase[~mask_eligible]=np.nan
        np.savez_compressed(out/'module3a_maps.npz',cohort_index=ids[ix],valid_600=mask_eligible,R=vals['R'][ix],Fbase=fbase,Fmu_matched30=mu,H1j=vals['H1j'][ix],H2j=vals['H2j'][ix],base1=vals['base1'][ix],base2=vals['base2'][ix],S1=vals['S1'][ix],S2=vals['S2'][ix],full_joint_30=fulljoint_flags)
    df.to_csv(out/'module3a_record_audit.csv',index=False)
    df[df.eligible_3a].to_csv(out/'module3a_eligible_records.csv',index=False)
    df[df.full_joint_30].to_csv(out/'module3a_matched30_records.csv',index=False)
    byclass=df[df.eligible_3a].groupby('class_name').agg(n=('cohort_index','size'),median_R_mean=('r_mean','median'),n_matched30=('full_joint_30','sum')).reset_index()
    byclass.to_csv(out/'module3a_by_class.csv',index=False)
    summary={'experiment':'Module3A from s9534 archived maps','scaler_n_fit_reported':meta.get('scaler_n_fit'), 'source_note':meta.get('note'), 'n_module2':int(len(df)),'n_module1_frozen_primary':int(len(primary)),'n_primary_eligible_m2':int(df.eligible_3a.sum()),'n_primary_matched30_clean':int(df.full_joint_30.sum()),'n_m2_both_correct':int(df.both_correct_m2.sum()),'n_m1_m2_mask_match':int(df.mask_equal.sum()),'max_R_formula_error':float(df.r_formula_max_error.max()),'median_r_mean_eligible':float(df.loc[df.eligible_3a,'r_mean'].median()),'median_matched30_ratio_r_over_mu':float(df.loc[df.full_joint_30,'r_fmu_intensity_ratio'].median()),'n_m1_baseline_1d_audited':int(df.m1_m2_baseline_1d_pearson.notna().sum()),'limitations':['R and baselines are Module2 on-trace (rerendered image) maps, NOT Gaussian G12 Module1 maps','same scaler provenance in Module2 metadata does not by itself prove 2D spatial pipeline equality','No-stability matched Fmu is valid only when 30/30 perturbations are accepted by BOTH branches','Map effect metrics are not faithfulness tests']}
    (out/'module3a_summary.json').write_text(json.dumps(summary,indent=2))
    with np.load(out/'module3a_maps.npz',allow_pickle=False) as q:
        hits=np.where(q['cohort_index']==a.highlight)[0]
        if len(hits)==1:
            k=int(hits[0]);v=q['valid_600'][k];xx=np.arange(600)
            fig,axes=plt.subplots(3,1,figsize=(12,7.7),sharex=True)
            axes[0].plot(xx[v],q['base1'][k,v],color='#ad6938',lw=1.7,label='Baseline 1D (Module 2)')
            axes[0].plot(xx[v],q['base2'][k,v],color='#c96798',lw=1.7,label='Baseline 2D on-trace (Module 2)')
            axes[0].legend(loc='upper right',fontsize=8);axes[0].set_ylabel('Relative CAM')
            axes[1].plot(xx[v],q['S1'][k,v],color='#ad6938',label='1D stability')
            axes[1].plot(xx[v],q['S2'][k,v],color='#c96798',label='2D stability');axes[1].legend(loc='upper right',fontsize=8);axes[1].set_ylabel('Local stability')
            axes[2].plot(xx[v],q['Fbase'][k,v],color='#888888',lw=1.6,label='Original fusion')
            if q['full_joint_30'][k]:axes[2].plot(xx[v],q['Fmu_matched30'][k,v],color='#d39729',lw=1.7,label='No-stability (matched 30)')
            axes[2].plot(xx[v],q['R'][k,v],color='#7858a9',lw=2,label='Module3A R(u) — native values')
            axes[2].legend(loc='upper right',fontsize=8);axes[2].set_ylabel('Fusion intensity');axes[2].set_xlabel('Temporal grid column (0–599; not calibrated seconds)')
            for ax in axes:ax.set_xlim(0,599);ax.set_ylim(0,1.02);ax.grid(alpha=.14)
            fig.suptitle(f'Module3A — ECG {a.highlight} | archived s9534 maps, no display renormalization')
            fig.tight_layout();fig.savefig(out/f'ECG_{a.highlight}_module3a.png',dpi=180);plt.close(fig)
    print('=== MODULE 3A INPUT AUDIT & FUSION ===')
    for k,v in summary.items():
        if k!='limitations':print(k,':',v)
    print('OUTPUT:',out)

if __name__=='__main__':main()
