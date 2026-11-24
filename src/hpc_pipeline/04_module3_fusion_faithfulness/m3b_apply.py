#!/usr/bin/env python3
"""DAN-MMGradCAM Module 3B exploratory extension: apply SAME-pipeline F_trace to saved M3A R.

R_trace(u) = R(u) * F_trace(u)**lambda, lambda pre-specified e.g. 0.5,1,2.
No normalization or display rescaling. Scientific advantage not established until
paired deletion/random/sanity evaluations are completed on identical ECGs.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROVENANCE='module2_rerendered_baseline_2d_spatial'

def pearson(x,y):
    if len(x)<2 or np.std(x)<1e-12 or np.std(y)<1e-12:return np.nan
    return float(np.corrcoef(x,y)[0,1])

def iou(x,y):
    k=max(1,int(np.ceil(.1*len(x))))
    s=set(np.argsort(-x,kind='stable')[:k]);t=set(np.argsort(-y,kind='stable')[:k])
    return len(s&t)/len(s|t)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m3a-dir',required=True)
    ap.add_argument('--trace-npz',nargs='+',required=True,help='One or more M2 trace-support NPZ files (chunks accepted)')
    ap.add_argument('--lambdas',default='0.5,1.0,2.0',help='Prespecified sensitivity values')
    ap.add_argument('--highlight',type=int,default=209)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    lambdas=[float(s.strip()) for s in a.lambdas.split(',')]
    if not lambdas or any((x<=0 or x>8 or not np.isfinite(x)) for x in lambdas):raise ValueError('lambda values must be finite 0 < lambda <= 8')
    if len(set(lambdas))!=len(lambdas):raise ValueError('Duplicate lambdas')
    out=Path(a.out).expanduser();out.mkdir(parents=True,exist_ok=True)
    with np.load(Path(a.m3a_dir).expanduser()/'module3a_maps.npz',allow_pickle=False) as q:
        ids=q['cohort_index'].astype(int);main_lookup={int(v):i for i,v in enumerate(ids)}
        R=q['R'].astype(np.float32);valid=q['valid_600'].astype(bool)
    tf_by_id={};source_ids={}
    for fn in a.trace_npz:
        p=Path(fn).expanduser()
        with np.load(p,allow_pickle=False) as tr:
            required={'origin','cohort_index','F_trace','valid_600'}
            if not required.issubset(tr.files):raise ValueError(f'{p}: missing {required-set(tr.files)}')
            origin=str(tr['origin'].item())
            if origin!=PROVENANCE:raise ValueError(f'Wrong F_trace origin {origin}. Required {PROVENANCE}')
            tid=tr['cohort_index'].astype(int);arr=tr['F_trace'];mask=tr['valid_600'].astype(bool)
            if arr.shape!=(len(tid),600) or mask.shape!=arr.shape:raise ValueError(f'{p}: shapes not compatible')
            for j,cid in enumerate(tid):
                cid=int(cid)
                if cid in tf_by_id:raise ValueError(f'Duplicate trace ECG {cid} across chunks')
                if cid not in main_lookup:raise ValueError(f'Trace ECG {cid} not present in M3A primary eligible set')
                if not np.array_equal(mask[j],valid[main_lookup[cid]]):raise ValueError(f'ECG {cid} spatial support mask differs from M3A')
                x=arr[j,mask[j]]
                if not np.isfinite(x).all() or np.any(x<-1e-6) or np.any(x>1+1e-6):raise ValueError(f'ECG {cid}: invalid F_trace values')
                tf_by_id[cid]=arr[j].astype(np.float32);source_ids[cid]=str(p)
    if not tf_by_id:raise ValueError('No verified spatial trace support')
    selected=np.asarray([v for v in ids if int(v) in tf_by_id],dtype=np.int64)
    rrows=np.asarray([main_lookup[int(v)] for v in selected],dtype=int)
    arr=np.stack([tf_by_id[int(v)] for v in selected]);r=R[rrows];m=valid[rrows]
    results={'cohort_index':selected,'valid_600':m,'R':r,'F_trace':arr}
    rows=[]
    for lam in lambdas:
        name=f'Rtrace_lambda_{str(lam).replace(".","p")}'
        f=np.power(np.clip(arr,0,1),lam).astype(np.float32)
        value=(r*f).astype(np.float32)
        value[~m]=np.nan
        results[name]=value
        for i,cid in enumerate(selected):
            v=m[i];x=r[i,v];y=value[i,v]
            rows.append({'cohort_index':int(cid),'lambda':lam,'R_mean':float(x.mean()),'Rtrace_mean':float(y.mean()),'Rtrace_over_R_mean':float(y.sum()/x.sum()) if x.sum()>0 else np.nan,'shape_Pearson':pearson(x,y),'top10_IoU':iou(x,y),'F_trace_mean':float(arr[i,v].mean())})
    np.savez_compressed(out/'module3b_maps.npz',**results)
    df=pd.DataFrame(rows);df.to_csv(out/'module3b_record_metrics.csv',index=False)
    summary={'n_trace_records':len(selected),'n_module3a_available':len(ids),'lambda_values':lambdas,'source':'F_trace computed from SAME Module2 2D CNN/renderer baseline spatial CAM','primary_method':'Module3A R, unchanged','hypothesis':'Multiplying R by waveform-corridor spatial mass fraction may focus explanations; must test against M3A with same deletion protocol','limitations':['R_trace <= R pointwise by construction; lower absolute intensity alone is not evidence of greater reliability','Source F_trace is an unperturbed 2D spatial CAM feature, not a new stability estimate','Need evaluate paired deletion, sanity, and sensitivity; do not claim performance enhancement from descriptive map metrics']}
    (out/'module3b_summary.json').write_text(json.dumps(summary,indent=2))
    loc=np.where(selected==a.highlight)[0]
    if len(loc)==1:
        i=int(loc[0]);v=m[i];xx=np.arange(600)
        fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        axes[0].plot(xx[v],arr[i,v],color='#3f806d',lw=1.8,label='F_trace: on-waveform CAM spatial mass fraction')
        axes[0].legend(loc='upper right',fontsize=8);axes[0].set_ylabel('Spatial support F_trace')
        axes[1].plot(xx[v],r[i,v],color='#7858a9',lw=2.0,label='Module3A R(u)')
        for lam in lambdas:
            key=f'Rtrace_lambda_{str(lam).replace(".","p")}'
            axes[1].plot(xx[v],results[key][i,v],lw=1.4,label=f'Module3B lambda={lam:g}')
        axes[1].legend(loc='upper right',fontsize=8);axes[1].set_ylabel('Native attribution intensity');axes[1].set_xlabel('Temporal grid column (0–599)')
        for ax in axes:ax.set_ylim(0,1.02);ax.set_xlim(0,599);ax.grid(alpha=.15)
        fig.suptitle(f'ECG {a.highlight}: matched-pipeline Module3B — no color/intensity rescaling')
        fig.tight_layout();fig.savefig(out/f'ECG_{a.highlight}_module3b.png',dpi=180);plt.close(fig)
    print('=== MODULE 3B SPATIAL SUPPORT FUSION ===');print('Matched-pipeline records:',len(selected));print('Provenance:',PROVENANCE)
    print('By lambda:')
    print(df.groupby('lambda').agg(n=('cohort_index','size'),median_ratio=('Rtrace_over_R_mean','median'),median_top10_IoU=('top10_IoU','median')).to_string())
    print('OUTPUT:',out)

if __name__=='__main__':main()
