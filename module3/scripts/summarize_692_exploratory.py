#!/usr/bin/env python3
"""Summarize 692 successful Module 3 faithfulness ECGs without fabricating ECG 406.

This is a *descriptive/exploratory* comparison because the 2D renderer's
history-dependence has been demonstrated and not yet resolved.
No changes to raw files, no inference, no new perturbations.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20261115
FRACTIONS=np.array([0.05,0.10,0.15,0.20,0.25,0.30])
CORE={'R','Fbase','Rtrace_0p5','Rtrace_1p0','Rtrace_2p0'}


def bootstrap_mean(v,seed):
    arr=np.asarray(v,dtype=float)
    assert len(arr)>0 and np.isfinite(arr).all()
    rng=np.random.default_rng(seed)
    values=arr[rng.integers(len(arr),size=(2000,len(arr)))].mean(axis=1)
    return (float(arr.mean()),float(np.percentile(values,2.5)),float(np.percentile(values,97.5)))


def auc(sub):
    g=sub.groupby('fraction',as_index=False)['probability_drop'].mean().sort_values('fraction')
    if len(g)!=len(FRACTIONS) or not np.allclose(g['fraction'],FRACTIONS):
        raise ValueError('Incomplete/invalid deletion fractions')
    integrate=getattr(np,'trapezoid',np.trapz)
    return float(integrate(np.r_[0.,g.probability_drop.to_numpy()],np.r_[0.,FRACTIONS]))


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--raw-dir',required=True)
    a.add_argument('--m3b-dir',required=True)
    a.add_argument('--out-dir',required=True)
    a=a.parse_args()
    src=Path(a.raw_dir)
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    expected=set(range(693))-{602}
    csvs={int(p.stem.split('_')[1]):p for p in src.glob('faith_????.csv')}
    audits={int(p.stem.split('_')[1]):p for p in src.glob('faith_????_audit.json')}
    if set(csvs)!=expected or set(audits)!=expected:
        raise ValueError(f'Unexpected index coverage: CSV missing={sorted(expected-set(csvs))}, audits missing={sorted(expected-set(audits))}, extras={sorted((set(csvs)|set(audits))-expected)}')
    with np.load(Path(a.m3b_dir)/'module3b_maps.npz',allow_pickle=False) as z:
        m3_ids=z['cohort_index'].astype(int)
    if len(m3_ids)!=693 or len(set(m3_ids))!=693:
        raise ValueError('M3B reference must contain 693 distinct IDs')
    audit_rows=[]
    curves=[]
    n_raw=0
    for index in sorted(expected):
        meta=json.loads(audits[index].read_text())
        cid=int(m3_ids[index])
        if int(meta.get('index',-1))!=index or int(meta.get('cohort_index',-1))!=cid:
            raise ValueError(f'Index/ID mismatch {index}')
        if meta.get('strict_baseline_qc') is not True or meta.get('oneDNN_disabled') is not True:
            raise ValueError(f'QC audit invalid {index}')
        df=pd.read_csv(csvs[index])
        if set(df.cohort_index.astype(int))!={cid}:
            raise ValueError(f'CSV ECG ID mismatch {index}')
        if len(df) not in (120,132) or len(df.branch.unique())!=2:
            raise ValueError(f'Unexpected method count or branches {index}')
        if not np.isfinite(df[['probability_drop','baseline_probability','perturbed_probability']].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite values {index}')
        if set(df.block_samples)!={50} or not np.allclose(df.actual_fraction,df.fraction):
            raise ValueError(f'Unexpected block/fraction definition {index}')
        for branch in ['1d','2d']:
            branch_df=df[df.branch==branch]
            method_rows=branch_df[branch_df.control=='attribution']
            random_rows=branch_df[branch_df.control=='random']
            if len(random_rows)!=30 or set(random_rows.method)!={'RANDOM'}:
                raise ValueError(f'Random control mismatch {index}/{branch}')
            if not CORE.issubset(set(method_rows.method)):
                raise ValueError(f'Missing method {index}/{branch}')
            if set(method_rows.method)-CORE != ({'Fmu_matched30'} if meta.get('matched30') else set()):
                raise ValueError(f'Matched-30 eligibility mismatch {index}/{branch}')
            for f in FRACTIONS:
                rs=random_rows[np.isclose(random_rows.fraction,f)]
                if len(rs)!=5 or set(rs['repeat'].astype(int))!=set(range(5)):
                    raise ValueError(f'Random repeats invalid {index}/{branch}/{f}')
            for method,group in branch_df.groupby('method'):
                if method!='RANDOM' and (len(group)!=6 or group.fraction.nunique()!=6):
                    raise ValueError(f'Incomplete method {index}/{branch}/{method}')
                curves.append(dict(index=index,cohort_index=cid,class_name=str(meta['class_name']),
                                   branch=branch,method=method,auc_drop=auc(group)))
        audit_rows.append(dict(index=index,cohort_index=cid,class_name=str(meta['class_name']),
                               matched30=bool(meta['matched30']),n_rows=len(df),
                               baseline_1d_max_error=meta.get('baseline_1d_max_error'),
                               baseline_2d_max_error=meta.get('baseline_2d_max_error')))
        n_raw+=len(df)
    a_df=pd.DataFrame(audit_rows)
    auc_df=pd.DataFrame(curves)
    a_df.to_csv(out/'audit_692.csv',index=False)
    auc_df.to_csv(out/'per_ecg_branch_auc_692.csv',index=False)
    rows=[]
    both=auc_df.groupby(['cohort_index','class_name','method'],as_index=False)['auc_drop'].mean()
    for branch,data in [('1d',auc_df[auc_df.branch=='1d']),('2d',auc_df[auc_df.branch=='2d']),('both_branch_mean',both)]:
        for cl in ['ALL']+sorted(data.class_name.unique()):
            sub=data if cl=='ALL' else data[data.class_name==cl]
            pt=sub.pivot(index='cohort_index',columns='method',values='auc_drop')
            for method in ['Fbase','Rtrace_0p5','Rtrace_1p0','Rtrace_2p0','Fmu_matched30','RANDOM']:
                if method not in pt.columns:continue
                matched=pt[['R',method]].dropna()
                if matched.empty:continue
                d=matched[method].to_numpy()-matched['R'].to_numpy()
                mean,lo,hi=bootstrap_mean(d,SEED+len(d)+len(method)+len(cl)+len(branch))
                rows.append(dict(branch=branch,class_name=cl,comparison=f'{method} - R',
                                 n_ecg=len(matched),mean_delta_auc=mean,
                                 ci95_lower=lo,ci95_upper=hi))
    comp=pd.DataFrame(rows)
    comp.to_csv(out/'paired_auc_692_exploratory.csv',index=False)
    info={
       'status':'EXPLORATORY - 2D rendering history dependence unresolved',
       'n_successful_ecg':692,'missing_primary_index':602,'missing_ecg_id':406,
       'n_raw_rows':int(n_raw),'n_matched30':int(a_df.matched30.sum()),
       'n_by_class':{str(k):int(v) for k,v in a_df.class_name.value_counts().to_dict().items()},
       'lambda_values':[.5,1.,2.], 'block_size_samples':50,
       'deletion_fractions':FRACTIONS.tolist(),'random_repeats':5,
       'auc_integration':'trapezoid of class-probability drop vs fraction, anchored at (0,0)',
       'paired_bootstrap':'2000 ECG-level resamples, percentile 95% CI',
       'positive_delta_interpretation':'Greater probability drop on selected deletion regions only; NOT a proof of explanatory superiority',
       'notes':['Input ECG signals, CNN weights, and Module2 outputs are not modified',
                'Within-model input-image rendering after deletion has not passed a full state-dependence audit',
                'Some 2D deletion results may depend on renderer state; do not publish superiority claims without remediation',
                'Exact Module2 source/runtime provenance and scaler fitting history remain incompletely verified']
    }
    (out/'faithfulness_summary_692_exploratory.json').write_text(json.dumps(info,indent=2)+'\n')
    lines=['DAN-MMGradCAM | preliminary 692-ECG paired deletion summary',
           'IMPORTANT: EXPLORATORY, unresolved history-dependent 2D renderer.',
           'ECGs = 692; primary-index 602 / ECG 406 failed strict baseline QC.','',
           'Both-branch mean paired AUC difference (method - Module3A R):']
    focus=comp[(comp.branch=='both_branch_mean')&(comp.class_name=='ALL')]
    for _,r in focus.iterrows():
        lines.append(f"{r['comparison']}: n={int(r.n_ecg)}, mean={r.mean_delta_auc:+.7f}, CI95=[{r.ci95_lower:+.7f}, {r.ci95_upper:+.7f}]")
    lines+=['','These estimates are not evidence of method superiority before 2D input-rendering audit.']
    (out/'READ_ME_FIRST_692_RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('EXPLORATORY SUMMARY WRITTEN:',out)

if __name__=='__main__':main()
