#!/usr/bin/env python3
"""Audit and summarize strictly paired Module 3 ECG deletion results.

Higher deletion-drop AUC suggests more score-sensitive regions, but is not a
standalone causal faithfulness guarantee. Bootstrap resamples ECG IDs, not rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261115
FRACTIONS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])


def auc_curve(df):
    z = df.groupby('fraction', as_index=False)['probability_drop'].mean()
    z = z.sort_values('fraction')
    if len(z) != 6 or not np.allclose(z['fraction'].to_numpy(), FRACTIONS):
        raise ValueError(f'Unexpected fractions: {z.fraction.tolist()}')
    # Explicit zero point: deleting zero signal must produce zero probability drop.
    xx = np.concatenate(([0.0], z['fraction'].to_numpy(dtype=float)))
    yy = np.concatenate(([0.0], z['probability_drop'].to_numpy(dtype=float)))
    trapezoid = getattr(np, 'trapezoid', np.trapz)
    return float(trapezoid(yy, xx))


def boot_mean(values, seed=SEED, n=2000):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return [None, None, None]
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return [float(a.mean()), float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--expected-records', type=int, default=693)
    ap.add_argument('--m3b-dir', help='Optional original M3B primary directory for ECG ID match')
    a = ap.parse_args()
    source = Path(a.raw_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csvs = sorted(source.glob('faith_[0-9][0-9][0-9][0-9].csv'))
    audits = sorted(source.glob('faith_[0-9][0-9][0-9][0-9]_audit.json'))
    if len(csvs) != a.expected_records or len(audits) != a.expected_records:
        raise ValueError(f'Expected {a.expected_records} CSV+audit files; found {len(csvs)} + {len(audits)}')
    expected_indices = [f'faith_{i:04d}.csv' for i in range(a.expected_records)]
    if [p.name for p in csvs] != expected_indices:
        raise ValueError('Noncontiguous or missing output indices')
    audit_by_idx = {}
    for p in audits:
        key = int(p.stem.split('_')[1])
        val = json.loads(p.read_text())
        if val.get('index') != key or not val.get('strict_baseline_qc'):
            raise ValueError(f'Invalid audit record {p}')
        if val.get('oneDNN_disabled') is not True:
            raise ValueError(f'oneDNN not disabled for {p}')
        audit_by_idx[key] = val
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    ids = df['cohort_index'].unique()
    if len(ids) != a.expected_records:
        raise ValueError(f'Expected {a.expected_records} unique IDs; found {len(ids)}')
    if set(ids) != {x['cohort_index'] for x in audit_by_idx.values()}:
        raise ValueError('Audit IDs do not match raw data IDs')
    if a.m3b_dir:
        with np.load(Path(a.m3b_dir) / 'module3b_maps.npz', allow_pickle=False) as z:
            expected_ids = set(z['cohort_index'].astype(int))
        if set(ids) != expected_ids:
            raise ValueError('Results do not match M3B strict cohort ID set')
    if not np.isfinite(df[['probability_drop', 'baseline_probability',
                           'perturbed_probability']].to_numpy()).all():
        raise ValueError('Nonfinite probability outcomes')
    if (df['block_samples'] != 50).any():
        raise ValueError('Unexpected deletion block size')
    if not np.allclose(df['actual_fraction'], df['fraction']):
        raise ValueError('Deletion fraction mismatch')

    # Validate each record's balanced paired 1D/2D outcomes and random controls.
    for cid, part in df.groupby('cohort_index'):
        for branch in ['1d', '2d']:
            sub = part[part.branch == branch]
            r = sub[sub.control == 'random']
            if len(r) != 6 * 5 or set(r.method) != {'RANDOM'}:
                raise ValueError(f'Random controls incomplete for ECG {cid} branch {branch}')
            if not all(r[r.fraction == f].repeat.nunique() == 5 for f in FRACTIONS):
                raise ValueError(f'Random replicates incomplete for ECG {cid} branch {branch}')
            methods = sub[sub.control == 'attribution']
            if not set(['R', 'Fbase', 'Rtrace_0p5', 'Rtrace_1p0', 'Rtrace_2p0']).issubset(methods.method):
                raise ValueError(f'Missing core methods for ECG {cid} branch {branch}')
            for method, sm in methods.groupby('method'):
                if len(sm) != 6 or sm.fraction.nunique() != 6:
                    raise ValueError(f'Incomplete deletion curve {cid} {branch} {method}')
    # Integrate per-ECG per-branch curves; random repetitions are averaged at each fraction.
    auc_rows = []
    for (cid, cls, branch, method), sub in df.groupby(
        ['cohort_index', 'class_name', 'branch', 'method'], sort=False):
        auc_rows.append(dict(cohort_index=int(cid), class_name=cls,
                             branch=branch, method=method, auc_drop=auc_curve(sub)))
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(out / 'per_ecg_branch_auc.csv', index=False)

    # Compare paired AUCs to R, separately by branch AND after averaging branches per ECG.
    rows = []
    combined = auc.groupby(['cohort_index', 'class_name', 'method'], as_index=False)['auc_drop'].mean()
    for branch, data in [('1d', auc[auc.branch == '1d']),
                         ('2d', auc[auc.branch == '2d']),
                         ('both_branch_mean', combined)]:
        pivot = data.pivot_table(index=['cohort_index', 'class_name'],
                                 columns='method', values='auc_drop')
        for cls in ['ALL', *sorted(data.class_name.unique())]:
            x = pivot if cls == 'ALL' else pivot.loc[pivot.index.get_level_values('class_name') == cls]
            for method in ['Fbase', 'Rtrace_0p5', 'Rtrace_1p0', 'Rtrace_2p0', 'Fmu_matched30', 'RANDOM']:
                if method not in x or 'R' not in x:
                    continue
                matched = x[['R', method]].dropna()
                if matched.empty:
                    continue
                d = matched[method].to_numpy() - matched['R'].to_numpy()
                mean, lo, hi = boot_mean(d, seed=SEED + len(matched))
                rows.append(dict(branch=branch, class_name=cls, comparison=f'{method} - R',
                                 n_ecg=len(matched), mean_delta_auc=mean,
                                 ci95_lower=lo, ci95_upper=hi,
                                 interpretation='Positive delta means greater mean deletion probability drop'))
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(out / 'paired_auc_comparisons.csv', index=False)
    info = dict(n_ecg=len(ids), n_raw_rows=len(df),
                n_matched30=int(sum(v['matched30'] for v in audit_by_idx.values())),
                block_samples=50, fractions=FRACTIONS.tolist(), random_repeats=5,
                ci='Percentile paired bootstrap: 2000 resamples of ECG IDs',
                primary_interpretation='Higher deletion-drop AUC is suggestive, not proof of causal faithfulness',
                provenance_limitation='M2 s9534 TensorFlow 2.21 vs runtime TensorFlow 2.19; helper exact commit not verified')
    (out / 'faithfulness_summary.json').write_text(json.dumps(info, indent=2) + '\n')
    print('AUDIT PASSED:', len(ids), 'unique ECGs,', len(df), 'raw deletion rows')
    print('\n=== BOTH BRANCHES MEAN: PAIRWISE AUC DELTA vs Module3A R ===')
    print(comparisons[(comparisons.branch == 'both_branch_mean') &
                      (comparisons.class_name == 'ALL')].to_string(index=False))
    print('\nOutput:', out)


if __name__ == '__main__':
    main()
