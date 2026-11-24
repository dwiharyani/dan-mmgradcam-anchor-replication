#!/usr/bin/env python3
"""Module 3 paired deletion experiment, adapted from the supplied Module 2 protocol.

Run exactly one strict-QC ECG per Slurm array task. Keeps Module 2 artifacts and
Module 3 attribution maps unchanged. Python/TensorFlow are imported only on compute nodes.
This is a faithfulness-style perturbation test, NOT causal proof of faithfulness.
"""
import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261115
FS = 500
N = 5000
BLOCK = 50
FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
RANDOM_REPEATS = 5
X0, X1, ORIGINAL_WIDTH, GRID_WIDTH = 58, 1021, 1080, 600
STRIDE = 5


def fail(msg):
    raise RuntimeError(msg)


def to_signal(a, valid):
    """Map valid 600-column scores onto 5000 ECG samples, as in Module 2."""
    a = np.asarray(a, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if a.shape != (GRID_WIDTH,) or valid.shape != a.shape:
        fail(f"Unexpected map/mask shape: {a.shape}, {valid.shape}")
    if not valid.any() or not np.isfinite(a[valid]).all():
        fail("Nonfinite or empty valid map")
    sample_pixels = X0 + np.linspace(0, 1, N) * (X1 - X0)
    sample_columns = (sample_pixels + 0.5) * GRID_WIDTH / ORIGINAL_WIDTH - 0.5
    result = np.interp(sample_columns, np.flatnonzero(valid), a[valid])
    if not np.isfinite(result).all():
        fail("Nonfinite interpolated scores")
    return result


def choose_top_blocks(scores, n_blocks):
    """Greedy high-mean nonoverlapping 50-sample blocks, stride 5."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (N,):
        fail("Unexpected score length")
    candidates = [
        (float(scores[s:s + BLOCK].mean()), s)
        for s in range(0, N - BLOCK + 1, STRIDE)
    ]
    candidates.sort(reverse=True)  # Same tie handling as visible Module 2 snippet
    starts = []
    for _, s in candidates:
        if all(s + BLOCK <= old or old + BLOCK <= s for old in starts):
            starts.append(s)
            if len(starts) == n_blocks:
                break
    if len(starts) != n_blocks:
        fail(f"Could not select {n_blocks} top nonoverlapping blocks")
    return sorted(starts)


def choose_random_blocks(n_blocks, seed):
    """Deterministic nonoverlapping random control from same stride-5 grid."""
    rng = np.random.default_rng(seed)
    starts = []
    options = np.arange(0, N - BLOCK + 1, STRIDE)
    for s in rng.permutation(options):
        s = int(s)
        if all(s + BLOCK <= old or old + BLOCK <= s for old in starts):
            starts.append(s)
            if len(starts) == n_blocks:
                break
    if len(starts) != n_blocks:
        fail(f"Could not select {n_blocks} random nonoverlapping blocks")
    return sorted(starts)


def occlude_blocks(x, starts):
    """Linear interpolation between samples just outside each selected block."""
    out = np.asarray(x, dtype=np.float64).copy()
    for start in starts:
        end = start + BLOCK
        left = out[start - 1] if start > 0 else out[end]
        right = out[end] if end < len(out) else left
        out[start:end] = np.linspace(left, right, BLOCK + 2)[1:-1]
    return out


def load_npz(path, keys):
    if not path.is_file():
        fail(f"Missing input: {path}")
    with np.load(path, allow_pickle=False) as z:
        missing = set(keys) - set(z.files)
        if missing:
            fail(f"{path}: missing keys {sorted(missing)}")
        return {k: z[k].copy() for k in keys}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(Path.home() / 'mm_gradcam/module3_aligned'))
    ap.add_argument('--sources', default=str(Path.home() / 'mm_gradcam/module3_sources'))
    ap.add_argument('--index', type=int, required=True,
                    help='Zero-based index within strict-QC 693 ECG M3B file')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    root, sources = Path(args.root), Path(args.sources)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'faith_{args.index:04d}.csv'
    audit_path = out_dir / f'faith_{args.index:04d}_audit.json'
    if path.exists() or audit_path.exists():
        fail(f"Refusing to overwrite existing output for index {args.index}")

    m3a = load_npz(root / 'outputs/module3a_v1/module3a_maps.npz',
                   ['cohort_index', 'valid_600', 'R', 'Fbase', 'Fmu_matched30',
                    'full_joint_30'])
    m3b = load_npz(root / 'outputs/module3b_primary_qc693_v1/module3b_maps.npz',
                   ['cohort_index', 'valid_600', 'R', 'F_trace',
                    'Rtrace_lambda_0p5', 'Rtrace_lambda_1p0',
                    'Rtrace_lambda_2p0'])
    if len(m3b['cohort_index']) != 693 or not 0 <= args.index < 693:
        fail('Expected strict-QC cohort of 693 and index in range 0..692')
    ids = m3b['cohort_index'].astype(int)
    cid = int(ids[args.index])
    a_index = {int(v): i for i, v in enumerate(m3a['cohort_index'])}
    if cid not in a_index or cid == 1346:
        fail(f'Invalid strict-QC record {cid}')
    ai = a_index[cid]
    bi = args.index
    v = m3b['valid_600'][bi].astype(bool)
    if not np.array_equal(v, m3a['valid_600'][ai]):
        fail('M3A and M3B valid mask mismatch')
    if not np.allclose(m3a['R'][ai][v], m3b['R'][bi][v],
                       atol=1e-7, rtol=1e-6):
        fail('M3A and M3B R map mismatch')

    archive = load_npz(root / 'inputs/module2_s9534/noise_stability_maps.npz',
                       ['cohort_index', 'out1_base', 'out2_base'])
    m2_index = {int(v): i for i, v in enumerate(archive['cohort_index'])}
    if cid not in m2_index:
        fail(f'ID {cid} absent from Module 2 archive')
    mi = m2_index[cid]
    records = pd.read_csv(root / 'inputs/module2_s9534/noise_stability_records.csv')
    if not np.array_equal(records['cohort_index'].to_numpy(dtype=int),
                          archive['cohort_index'].astype(int)):
        fail('Module 2 record order mismatch')
    row = records.iloc[mi]

    methods = {
        'R': m3a['R'][ai],
        'Fbase': m3a['Fbase'][ai],
        'Rtrace_0p5': m3b['Rtrace_lambda_0p5'][bi],
        'Rtrace_1p0': m3b['Rtrace_lambda_1p0'][bi],
        'Rtrace_2p0': m3b['Rtrace_lambda_2p0'][bi],
    }
    matched30 = bool(m3a['full_joint_30'][ai])
    if matched30:
        methods['Fmu_matched30'] = m3a['Fmu_matched30'][ai]
    mapped = {name: to_signal(arr, v) for name, arr in methods.items()}

    runtime = sources / 'module2_runtime'
    fp = sources / 'module2_review/scripts/module2_common.py'
    if not runtime.is_dir() or not fp.is_file():
        fail('Missing Module 2 runtime or helper')
    os.chdir(runtime)
    sys.path.insert(0, str(fp.parent))
    spec = importlib.util.spec_from_file_location('module2_common_m3faith', fp)
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)

    target = cm.CLASSES.index(str(row.class_name))
    x = np.asarray(cm.load_ecg(str(row.ecg_csv_filename)), dtype=np.float64)
    if x.shape != (N,):
        fail('Unexpected ECG length')
    renderer = cm.Renderer()
    try:
        def pred_1d(sig):
            inp = ((sig - cm.MEAN) / cm.SCALE).astype(np.float32)[None, :, None]
            return np.asarray(cm.run_model('1d', cm.m1, inp)[0], dtype=np.float64)

        def pred_2d(sig):
            rgb = renderer.pert(sig)
            inp = (cm.to_2d(rgb[None]) / 255.0).astype(np.float32)
            return np.asarray(cm.run_model('2d', cm.m2, inp)[0], dtype=np.float64)

        p1 = pred_1d(x)
        rgb0 = renderer.base(x)
        inp0 = (cm.to_2d(rgb0[None]) / 255.0).astype(np.float32)
        p2 = np.asarray(cm.run_model('2d', cm.m2, inp0)[0], dtype=np.float64)
        ref1 = np.asarray(archive['out1_base'][mi], dtype=np.float64)
        ref2 = np.asarray(archive['out2_base'][mi], dtype=np.float64)
        e1 = float(np.max(np.abs(p1 - ref1)))
        e2 = float(np.max(np.abs(p2 - ref2)))
        print(f'ECG {cid} | class={row.class_name} | 1D err={e1:.8g} | 2D err={e2:.8g}', flush=True)
        if not np.allclose(p1, ref1, atol=1e-5, rtol=1e-5):
            fail(f'1D baseline QC mismatch ECG {cid}: {e1:.8g}')
        if not np.allclose(p2, ref2, atol=1e-5, rtol=1e-5):
            fail(f'2D baseline QC mismatch ECG {cid}: {e2:.8g}')
        if int(np.argmax(p1)) != target or int(np.argmax(p2)) != target:
            fail(f'ECG {cid} is not correctly predicted by both branches')

        original = {'1d': float(p1[target]), '2d': float(p2[target])}
        data = []
        for fraction in FRACTIONS:
            n_blocks = max(1, int(round(int(round(N * fraction)) / BLOCK)))
            actual_fraction = n_blocks * BLOCK / N
            # Precompute all masks BEFORE model inference; identical for both branches.
            selected = {name: choose_top_blocks(s, n_blocks)
                        for name, s in mapped.items()}
            randoms = [choose_random_blocks(
                n_blocks, SEED + cid * 1000 + int(round(fraction * 10000)) + rep)
                for rep in range(RANDOM_REPEATS)]
            for branch, predict in [('1d', pred_1d), ('2d', pred_2d)]:
                for method, starts in selected.items():
                    pred = float(predict(occlude_blocks(x, starts))[target])
                    data.append(dict(cohort_index=cid, class_name=str(row.class_name),
                                     branch=branch, method=method, control='attribution',
                                     repeat=-1, fraction=fraction, actual_fraction=actual_fraction,
                                     block_samples=BLOCK, n_blocks=n_blocks,
                                     baseline_probability=original[branch],
                                     perturbed_probability=pred,
                                     probability_drop=original[branch]-pred))
                for rep, starts in enumerate(randoms):
                    pred = float(predict(occlude_blocks(x, starts))[target])
                    data.append(dict(cohort_index=cid, class_name=str(row.class_name),
                                     branch=branch, method='RANDOM', control='random',
                                     repeat=rep, fraction=fraction, actual_fraction=actual_fraction,
                                     block_samples=BLOCK, n_blocks=n_blocks,
                                     baseline_probability=original[branch],
                                     perturbed_probability=pred,
                                     probability_drop=original[branch]-pred))
            print(f'ECG {cid} | fraction {fraction:.2f} complete', flush=True)

        df = pd.DataFrame.from_records(data)
        if not np.isfinite(df[['probability_drop','perturbed_probability']].to_numpy()).all():
            fail('Nonfinite outcome metrics')
        tmp = path.with_suffix('.csv.tmp')
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        audit = dict(cohort_index=cid, class_name=str(row.class_name),
                     index=args.index, strict_baseline_qc=True, matched30=matched30,
                     baseline_1d_max_error=e1, baseline_2d_max_error=e2,
                     n_rows=len(df), methods=sorted(methods),
                     fractions=FRACTIONS, block_samples=BLOCK,
                     random_repeats=RANDOM_REPEATS, seed=SEED,
                     oneDNN_disabled=os.getenv('TF_ENABLE_ONEDNN_OPTS') == '0',
                     tensorflow_version=cm.tf.__version__,
                     protocol='Module2 linear-interpolation deletion; paired 1D and re-rendered 2D')
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        print(f'PASS ECG={cid} methods={len(methods)} rows={len(df)} path={path}', flush=True)
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(renderer.fig)
        except Exception:
            pass
        gc.collect()


if __name__ == '__main__':
    main()
