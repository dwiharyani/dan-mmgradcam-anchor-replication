#!/usr/bin/env python3
"""DAN-MMGradCAM Module 3 figures, drawn on the original ECG waveform.

  m3a       Reliability-weighted attribution for one ECG:
            mu -> stability S -> H = mu*S -> R = sqrt(H1*H2) -> what changed
  m3b       Waveform-conditioned fusion for one ECG: R vs R_trace(lambda), same scale
  deletion  Cohort deletion faithfulness: drop over random, with bootstrap CI

Colour code (fixed in every figure): 1D brown, 2D pink, fusion violet,
stability teal-grey strip, grey = low. H1, H2, R and R_trace share one absolute colour
scale per figure (no per-panel rescaling), so a pale panel really is weaker.

Usage (from ~/mm_gradcam/module3_aligned):
  python m3_figures.py m3a --maps outputs/module3a_v1/module3a_maps.npz --cohort-index 209
  python m3_figures.py m3b --maps3a outputs/module3a_v1/module3a_maps.npz \
         --maps3b outputs/module3b_pilot_209/module3b_maps.npz --cohort-index 209
  python m3_figures.py deletion --summary <path>/deletion_summary_bootstrap.csv
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

T = 10.0
C1D, C2D, CR = '#C8874A', '#EC7BAE', '#58A0DC'
CGAIN, CLOSS, CBOTH, CLOW = '#2C6CA8', '#E3B04B', CR, '#D6D6D6'
HEAD, INK, INK2 = '#9A582F', '#2B2A30', '#6A6870'
seq = lambda c, dark: LinearSegmentedColormap.from_list('s', [CLOW, c, dark])
CM1, CM2, CMR = seq(C1D, '#8A5528'), seq(C2D, '#B9467C'), seq(CR, '#2C6CA8')
CMS = LinearSegmentedColormap.from_list('stab', ['#F6F6F6', '#C9D6E6', '#7D93B2'])
DEFAULT_X0, DEFAULT_X1 = 58, 1021
import os
PLAIN = os.environ.get('M3_PLAIN') == '1'   # manuscript mode: no in-figure header, title or footnote
if PLAIN:   # same typography as figures.py
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.edgecolor': '#9A9AA0', 'axes.linewidth': .7, 'legend.fontsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'dejavusans'})


# ------------------------------------------------------------------ data
def load_maps(path, cid, keys):
    z = np.load(Path(path).expanduser(), allow_pickle=False)
    if 'cohort_index' not in z.files:
        raise KeyError(f'{path}: no cohort_index array; keys = {z.files}')
    hit = np.flatnonzero(z['cohort_index'] == cid)
    if len(hit) != 1:
        raise ValueError(f'ECG {cid} found {len(hit)} times in {path}')
    i = int(hit[0])
    miss = [k for k in keys if k not in z.files]
    if miss:
        raise KeyError(f'{path}: missing {miss}; available = {z.files}')
    out = {k: np.asarray(z[k][i], float) for k in keys}
    vkey = 'valid_600' if 'valid_600' in z.files else None
    out['valid'] = np.asarray(z[vkey][i], bool) if vkey else np.isfinite(out[keys[0]])
    return out, z.files, i


def trace_from_image(img_path, x0, x1, width=600):
    """Blue ECG trace cropped to the waveform span [x0, x1) and resized to the map grid,
    so map column u and trace column u are the same time point."""
    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert('RGB'), np.int16)
    rgb = rgb[:, x0:x1]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ink = (b - r >= 22) & (g - r >= 14) & (b - g >= 12)
    h = ink.shape[0]
    small = Image.fromarray(ink.astype(np.uint8) * 255, 'L').resize((width, h), Image.Resampling.BILINEAR)
    m = np.asarray(small) > 40
    if m.sum() < width:   # colour rule failed: fall back to darkest pixels
        gray = rgb.mean(2)
        m0 = gray < np.percentile(gray, 3)
        m = np.asarray(Image.fromarray(m0.astype(np.uint8) * 255, 'L').resize((width, h))) > 40
    return m


def polyline(mask):
    pts, prev = [], None
    for x in range(mask.shape[1]):
        ys = np.flatnonzero(mask[:, x])
        if ys.size == 0:
            pts.append((np.nan, np.nan)); prev = None; continue
        lo, hi = float(ys.min()), float(ys.max())
        a, b = (lo, hi) if prev is None or abs(prev - lo) <= abs(prev - hi) else (hi, lo)
        pts += [(x, a), (x, b)]; prev = b
    return np.array(pts, float)


def draw_wave(ax, poly, values, cmap, vmax, lw=(0.8, 3.4), y_off=0.0, mask_cols=None):
    """ECG line whose colour and width follow values[u] on a fixed 0..vmax scale."""
    n = len(values)
    seg, val = [], []
    for k in range(1, len(poly)):
        (xa, ya), (xb, yb) = poly[k - 1], poly[k]
        if np.isnan(xa) or np.isnan(xb):
            continue
        u = int(xb)
        v = values[u] if 0 <= u < n else np.nan
        if mask_cols is not None and not mask_cols[u]:
            v = np.nan
        seg.append([(T * xa / n, -ya + y_off), (T * xb / n, -yb + y_off)])
        val.append(v)
    val = np.array(val, float)
    f = np.clip(np.nan_to_num(val / vmax), 0, 1)
    o = np.argsort(f)
    cols = cmap(f[o]); cols[np.isnan(val[o])] = (0.85, 0.85, 0.85, 1)
    ax.add_collection(LineCollection(np.array(seg)[o], colors=cols,
                                     linewidths=lw[0] + (lw[1] - lw[0]) * f[o], capstyle='round'))


def wave_axes(ax, poly, title, label=None):
    ys = poly[np.isfinite(poly[:, 1]), 1]
    ax.set_xlim(0, T); ax.set_ylim(-(ys.max() + 8), -(ys.min() - 8))
    ax.set_yticks([]); ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.set_title(title, loc='left', fontsize=9.5, color=INK, weight='bold' if PLAIN else 'normal')
    if label:
        ax.text(-0.012, 0.5, label, transform=ax.transAxes, ha='right', va='center',
                fontsize=8.5, weight='bold', color=label[1] if isinstance(label, tuple) else INK)


def strip(ax, values, cmap, vmax, label, valid):
    v = np.where(valid, values, np.nan)
    ax.imshow(v[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=vmax, extent=(0, T, 0, 1),
              interpolation='nearest')
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-0.012, 0.5, label, transform=ax.transAxes, ha='right', va='center', fontsize=7.5, color=INK2)


def topk(x, valid, frac=0.10):
    v = np.where(valid, np.nan_to_num(x), -np.inf)
    k = max(1, int(np.ceil(frac * valid.sum())))
    s = np.zeros(len(x), bool); s[np.argsort(-v)[:k]] = True
    return s & valid


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def side_table(ax, sections):
    ax.axis('off'); y = 0.99
    for head, rows in sections:
        ax.text(0, y, head, fontsize=8.8, weight='bold', color=HEAD, transform=ax.transAxes)
        y -= 0.05
        for k, v in rows:
            ax.text(0.02, y, k, fontsize=7.8, color=INK2, transform=ax.transAxes)
            ax.text(0.99, y, v, fontsize=7.8, weight='bold', color=INK, ha='right', transform=ax.transAxes)
            y -= 0.042
        y -= 0.025


def image_for(args, cid):
    meta = pd.read_csv(Path(args.meta).expanduser())
    row = meta[meta.cohort_index == cid]
    if len(row) != 1:
        raise ValueError(f'ECG {cid} not unique in {args.meta}')
    row = row.iloc[0]
    return Path(args.image_root).expanduser() / str(row.image_relative_path), str(row.class_name)


def header(fig, crumb, title, sub):
    if PLAIN:
        return
    fig.text(.06, .985, crumb, color=HEAD, fontsize=9.5, va='top')
    fig.text(.06, .962, title, color='#26324A', fontsize=15, weight='bold', va='top')
    fig.text(.06, .940, sub, color=INK2, fontsize=8.5, va='top')
    fig.lines.append(Line2D([.06, .94], [.922, .922], transform=fig.transFigure, color='#E4C2CF', lw=1))


def save(fig, out, payload):
    out = Path(out).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    if PLAIN:
        w, h = fig.get_size_inches(); fig.set_size_inches(9.4, 9.4 * h / w)
    fig.savefig(out.with_suffix('.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    out.with_suffix('.json').write_text(json.dumps(payload, indent=2, default=float), encoding='utf8')
    print('Saved:', out.with_suffix('.png'))


# ------------------------------------------------------------------ Module 3A
def fig_m3a(args):
    d, _, _ = load_maps(args.maps, args.cohort_index, ['mu1', 'S1', 'H1', 'mu2', 'S2', 'H2', 'R'])
    v = d['valid']
    if getattr(args, 'trace_mask', None):
        # waveform supplied as a mask covering exactly the valid columns
        mk = np.load(Path(args.trace_mask).expanduser())
        nv = int(v.sum()); rep = mk.shape[1] // nv
        mk = mk[:, :nv * rep]
        # each valid map column is drawn over `rep` waveform pixel columns
        d = {k: (np.repeat(a[v], rep) if k != 'valid' else a) for k, a in d.items()}
        v = np.ones(nv * rep, bool); cls = args.class_name
        args._rep = rep
        poly = polyline(mk)
    else:
        img, cls = image_for(args, args.cohort_index)
        poly = polyline(trace_from_image(img, args.x0, args.x1))

    R0 = np.sqrt(np.clip(d['mu1'], 0, None) * np.clip(d['mu2'], 0, None))   # fusion without stability
    vH = max(np.nanmax(np.where(v, d[k], np.nan)) for k in ('H1', 'H2', 'R'))
    vmu = max(1.0, np.nanmax(np.where(v, d['mu1'], np.nan)), np.nanmax(np.where(v, d['mu2'], np.nan)))
    t0, tR = topk(R0, v), topk(d['R'], v)
    cat = np.full(len(v), np.nan)
    cat[t0 & tR], cat[tR & ~t0], cat[t0 & ~tR] = 3, 1, 2
    mass = lambda a: float(np.nansum(np.where(v, a, 0)))
    ms = lambda a: float(np.nanmean(np.where(v, a, np.nan)))

    fig = plt.figure(figsize=(13, 13.2), facecolor='white')
    header(fig, f'DAN-MMGradCAM  /  ECG {args.cohort_index}  /  {cls}  /  Module 3A',
           'From attribution to reliability-weighted attribution',
           'Same ECG waveform • one time axis • H₁, H₂ and R on one absolute colour scale '
           f'(0–{vH:.2f}); no per-panel rescaling')
    gs = GridSpec(5, 2, figure=fig, width_ratios=[5, 1.25], height_ratios=[1.35, 1.35, 1.0, 1.55, 1.15],
                  hspace=.62, wspace=.12, top=(.975 if PLAIN else .89), bottom=.08, left=.10, right=.97)

    # A: mu with stability strip underneath, per branch
    for r, (mu, S, cm, col, nm) in enumerate(((d['mu1'], d['S1'], CM1, C1D, '1D'),
                                             (d['mu2'], d['S2'], CM2, C2D, '2D'))):
        sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[r, 0], height_ratios=[1, .16], hspace=.05)
        a, s = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        draw_wave(a, poly, np.where(v, mu, np.nan), cm, vmu)
        wave_axes(a, poly, f'A{r + 1}  {nm} perturbation-mean attribution μ{"₁" if r == 0 else "₂"}(u), '
                           f'with local stability S{"₁" if r == 0 else "₂"}(u) underneath')
        a.tick_params(labelbottom=False)
        a.text(-0.012, .5, nm, transform=a.transAxes, ha='right', va='center', fontsize=10,
               weight='bold', color=col)
        strip(s, S, CMS, 1.0, f'S{"₁" if r == 0 else "₂"}  mean {ms(S):.2f}', v)

    # B: H1 and H2 stacked on one axis (same scale)
    b = fig.add_subplot(gs[2, 0])
    ys = poly[np.isfinite(poly[:, 1]), 1]; span = ys.max() - ys.min() + 16
    draw_wave(b, poly, np.where(v, d['H1'], np.nan), CM1, vH, y_off=0)
    draw_wave(b, poly, np.where(v, d['H2'], np.nan), CM2, vH, y_off=-span)
    b.set_xlim(0, T); b.set_ylim(-(ys.max() + 8) - span, -(ys.min() - 8))
    b.set_yticks([]); b.spines[['top', 'right', 'left']].set_visible(False); b.tick_params(labelsize=7.5)
    b.tick_params(labelbottom=False)
    b.set_title('B  Stability-weighted attribution H(u) = μ(u)·S(u)  (unstable parts fade)', loc='left',
                fontsize=9.5, color=INK, weight='bold' if PLAIN else 'normal')
    b.text(-0.012, .75, 'H₁', transform=b.transAxes, ha='right', va='center', fontsize=10, weight='bold', color=C1D)
    b.text(-0.012, .25, 'H₂', transform=b.transAxes, ha='right', va='center', fontsize=10, weight='bold', color=C2D)

    # C: R (main result)
    c = fig.add_subplot(gs[3, 0])
    draw_wave(c, poly, np.where(v, d['R'], np.nan), CMR, vH, lw=(1.0, 4.6))
    wave_axes(c, poly, 'C  Module 3A reliability map  R(u) = √(H₁(u)·H₂(u))' if PLAIN else
              'C  Module 3A reliability map  R(u) = √(H₁(u)·H₂(u))  — high only where both '
              'models attribute and stay stable')
    c.tick_params(labelbottom=False)
    c.text(-0.012, .5, 'R', transform=c.transAxes, ha='right', va='center', fontsize=12, weight='bold', color=CR)

    # D: what changed in the top-10 % after stability weighting
    e = fig.add_subplot(gs[4, 0])
    draw_wave(e, poly, cat, ListedColormap([CGAIN, CLOSS, CBOTH]), 3.0, lw=(0.8, 3.2))
    # draw_wave maps value/vmax to cmap; map categories explicitly instead
    e.cla()
    colmap = {1: CGAIN, 2: CLOSS, 3: CBOTH}
    n = len(v); segs = {k: [] for k in (0, 1, 2, 3)}
    for k in range(1, len(poly)):
        (xa, ya), (xb, yb) = poly[k - 1], poly[k]
        if np.isnan(xa) or np.isnan(xb):
            continue
        cc = cat[int(xb)]; key = 0 if np.isnan(cc) else int(cc)
        segs[key].append([(T * xa / n, -ya), (T * xb / n, -yb)])
    for key, w, col in ((0, .8, CLOW), (3, 2.8, CBOTH), (2, 2.8, CLOSS), (1, 2.8, CGAIN)):
        if segs[key]:
            e.add_collection(LineCollection(segs[key], colors=col, linewidths=w, capstyle='round'))
    wave_axes(e, poly, 'D  What stability weighting changed: top-10 % of √(μ₁μ₂) vs top-10 % of R')
    e.set_xlabel('ECG time within valid waveform (s)', fontsize=8.5)
    e.legend(handles=[Patch(color=CBOTH, label='top-10 % in both'),
                      Patch(color=CGAIN, label='entered after stability'),
                      Patch(color=CLOSS, label='dropped after stability')],
             ncol=3, fontsize=7.3, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.42))

    # colour bars + side
    sp = fig.add_subplot(gs[:3, 1])
    side_table(sp, [
        ('STABILITY', [('Mean S₁ (1D)', f"{ms(d['S1']):.2f}"), ('Mean S₂ (2D)', f"{ms(d['S2']):.2f}")]),
        ('ATTRIBUTION KEPT', [('Σ H₁ / Σ μ₁', f"{100 * mass(d['H1']) / mass(d['mu1']):.0f}%"),
                              ('Σ H₂ / Σ μ₂', f"{100 * mass(d['H2']) / mass(d['mu2']):.0f}%"),
                              ('Σ R / Σ √(μ₁μ₂)', f"{100 * mass(d['R']) / mass(R0):.0f}%")]),
        ('LOCATION CHANGE', [('Top-10 % IoU', f'{iou(t0, tR):.2f}'),
                             ('Columns entered', f'{int((tR & ~t0).sum()) // getattr(args, "_rep", 1)}'),
                             ('Columns dropped', f'{int((t0 & ~tR).sum()) // getattr(args, "_rep", 1)}')]),
    ])
    sc = fig.add_subplot(gs[3:, 1]); sc.axis('off')
    for k, (cm, lab, vm) in enumerate(((CMS, 'Stability S', 1.0), (CM1, 'H₁ (μ₁ panel uses 0–1)', None),
                                      (CM2, 'H₂ (μ₂ panel uses 0–1)', None), (CMR, 'R', None))):
        cax = sc.inset_axes([0.05, 0.86 - k * 0.2, 0.85, 0.05])
        top = vm if vm else vH
        fig.colorbar(plt.cm.ScalarMappable(Normalize(0, top), cm), cax=cax, orientation='horizontal')
        cax.tick_params(labelsize=6.5)
        cax.set_title(lab, fontsize=7, loc='left', color=INK2)
    if not PLAIN: fig.text(.10, .012, 'R(u) is a reliability-weighted attribution, not a probability of clinical correctness; '
             'its usefulness is tested with deletion faithfulness and sanity checks on the cohort.',
             fontsize=7.2, color=INK2)
    save(fig, args.out or f'figures/Module3A_{cls}_{args.cohort_index}',
         {'cohort_index': args.cohort_index, 'class_name': cls, 'scale_vmax_H_R': vH,
          'mean_S1': ms(d['S1']), 'mean_S2': ms(d['S2']),
          'kept_H1_pct': 100 * mass(d['H1']) / mass(d['mu1']),
          'kept_H2_pct': 100 * mass(d['H2']) / mass(d['mu2']),
          'kept_R_vs_nostab_pct': 100 * mass(d['R']) / mass(R0),
          'top10_iou_nostab_vs_R': iou(t0, tR)})


# ------------------------------------------------------------------ Module 3B
def find_lambda_maps(npz, i):
    """Return {lambda: map} and F_trace from a module3b_maps.npz with flexible key names."""
    keys = npz.files
    out = {}
    if 'lambdas' in keys and any(k in keys for k in ('R_trace', 'Rtrace')):
        arr = npz['R_trace' if 'R_trace' in keys else 'Rtrace']
        lam = np.asarray(npz['lambdas'], float)
        a = arr[i]
        if a.shape[0] == len(lam):
            out = {float(l): np.asarray(a[j], float) for j, l in enumerate(lam)}
        else:
            out = {float(l): np.asarray(arr[i, :, j], float) for j, l in enumerate(lam)}
    else:
        for k in keys:
            m = re.search(r'(?:R_?trace|Rtrace).*?(?:lam(?:bda)?)?_?([0-9]+(?:[._p][0-9]+)?)$', k, re.I)
            if m:
                out[float(m.group(1).replace('p', '.').replace('_', '.'))] = np.asarray(npz[k][i], float)
    F = None
    for k in ('F_trace', 'Ftrace', 'F'):
        if k in keys:
            F = np.asarray(npz[k][i], float); break
    if not out:
        raise KeyError(f'No R_trace maps recognised; keys = {keys}. '
                       f'Expected e.g. lambdas + R_trace, or R_trace_lam0.5 ...')
    return dict(sorted(out.items())), F


def fig_m3b(args):
    d, _, _ = load_maps(args.maps3a, args.cohort_index, ['R'])
    v = d['valid']
    z = np.load(Path(args.maps3b).expanduser(), allow_pickle=False)
    i = int(np.flatnonzero(z['cohort_index'] == args.cohort_index)[0]) if 'cohort_index' in z.files else 0
    lam, F = find_lambda_maps(z, i)
    if 'R' in z.files and not np.allclose(np.nan_to_num(z['R'][i]), np.nan_to_num(d['R']), atol=1e-6):
        print('WARNING: R in module3b_maps differs from Module 3A R; using Module 3A R as reference')
    img, cls = image_for(args, args.cohort_index)
    poly = polyline(trace_from_image(img, args.x0, args.x1))
    R = d['R']
    vmax = float(np.nanmax(np.where(v, R, np.nan)))
    mass = lambda a: float(np.nansum(np.where(v, a, 0)))
    tR = topk(R, v)
    stats = {l: (100 * mass(m) / mass(R), iou(tR, topk(m, v))) for l, m in lam.items()}

    rows = 1 + len(lam)
    fig = plt.figure(figsize=(13, 3.0 + 1.55 * rows + 2.4), facecolor='white')
    header(fig, f'DAN-MMGradCAM  /  ECG {args.cohort_index}  /  {cls}  /  Module 3B',
           'Waveform-conditioned fusion  R_trace(u) = R(u)·F_trace(u)^λ',
           f'All panels share the Module 3A colour scale (0–{vmax:.2f}); fading means attribution was removed, '
           'not re-coloured')
    hr = ([.28] if F is not None else []) + [1.0] * rows + [.25, 1.6]
    gs = GridSpec(len(hr), 2, figure=fig, width_ratios=[5, 1.25], height_ratios=hr,
                  hspace=.55, wspace=.12, top=.88, bottom=.07, left=.10, right=.97)
    r0 = 0
    if F is not None:
        s = fig.add_subplot(gs[0, 0]); strip(s, F, CMS, 1.0, 'F_trace(u)', v)
        s.set_title('Waveform support F_trace(u)', loc='left', fontsize=9, color=INK); r0 = 1
    panels = [('Module 3A  R(u)', R, None)] + [(f'Module 3B  λ = {l:g}', m, l) for l, m in lam.items()]
    for k, (ttl, m, l) in enumerate(panels):
        a = fig.add_subplot(gs[r0 + k, 0])
        draw_wave(a, poly, np.where(v, m, np.nan), CMR, vmax, lw=(0.9, 4.2))
        wave_axes(a, poly, ttl)
        a.tick_params(labelbottom=k == len(panels) - 1)
        if l is not None:
            kept, io = stats[l]
            a.text(1.0, 1.02, f'mass kept {kept:.0f}%   •   top-10 % IoU vs R {io:.2f}',
                   transform=a.transAxes, ha='right', fontsize=8, color=INK)
    fig.axes[-1].set_xlabel('ECG time within valid waveform (s)', fontsize=8.5)
    fig.add_subplot(gs[-2, 0]).axis('off')

    # trade-off chart
    t = fig.add_subplot(gs[-1, 0])
    L = list(stats); kept = [stats[l][0] for l in L]; io = [100 * stats[l][1] for l in L]
    t.plot(L, kept, '-o', color=CR, lw=2, ms=6, label='attribution mass kept (%)')
    t.plot(L, io, '-s', color=INK2, lw=2, ms=6, label='top-10 % location kept, IoU×100')
    for l, a_, b_ in zip(L, kept, io):
        t.annotate(f'{a_:.0f}', (l, a_), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7.5, color=CR)
        t.annotate(f'{b_:.0f}', (l, b_), xytext=(0, -13), textcoords='offset points', ha='center', fontsize=7.5, color=INK2)
    t.set_xticks(L); t.set_xlabel('λ', fontsize=9); t.set_ylim(0, 105)
    t.grid(axis='y', color='#EEE'); t.spines[['top', 'right']].set_visible(False)
    t.legend(fontsize=7.5, frameon=False, loc='lower left')
    t.set_title('How much of Module 3A survives as λ grows (this ECG only)', loc='left', fontsize=9.5, color=INK)

    sp = fig.add_subplot(gs[:, 1]); side_table(sp, [
        ('PER λ  (mass kept / top-10 % IoU vs R)',
         [(f'λ = {l:g}', f'{stats[l][0]:.0f}% / {stats[l][1]:.2f}') for l in L])])
    cax = sp.inset_axes([0.05, 0.45, 0.85, 0.025])
    fig.colorbar(plt.cm.ScalarMappable(Normalize(0, vmax), CMR), cax=cax, orientation='horizontal')
    cax.tick_params(labelsize=6.5); cax.set_title('R / R_trace (shared scale)', fontsize=7, loc='left', color=INK2)
    if not PLAIN: fig.text(.10, .02, 'Losing mass is expected by construction (F_trace ≤ 1). Whether waveform conditioning '
             'helps must be decided by deletion faithfulness across ECGs, not by this picture.',
             fontsize=7.2, color=INK2)
    save(fig, args.out or f'figures/Module3B_{cls}_{args.cohort_index}',
         {'cohort_index': args.cohort_index, 'class_name': cls, 'scale_vmax_R': vmax,
          'per_lambda': {str(l): {'mass_kept_pct': stats[l][0], 'top10_iou_vs_R': stats[l][1]} for l in L}})


# ------------------------------------------------------------------ deletion
def fig_deletion(args):
    d = pd.read_csv(Path(args.summary).expanduser())
    names = {'base1': ('Grad-CAM (base)', '#B9B2A6'), 'base2': ('Grad-CAM (base)', '#B9B2A6'),
             'H1': ('H = μ·S', C1D), 'H2': ('H = μ·S', C2D), 'R': ('R (Module 3A)', CR),
             'R_trace': ('R_trace (Module 3B)', '#1B9565')}
    br = [b for b in ('1d', '2d') if b in d.branch.unique()]
    fig, axs = plt.subplots(1, len(br), figsize=(6.2 * len(br), 4.6), sharey=True, facecolor='white')
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, br):
        g = d[d.branch == b]
        for meth in g.method.unique():
            gg = g[g.method == meth].sort_values('fraction')
            lab, col = names.get(meth, (meth, '#888'))
            x = 100 * gg.fraction
            ax.fill_between(x, gg.difference_ci_low, gg.difference_ci_high, color=col, alpha=.15, lw=0)
            ax.plot(x, gg.paired_difference, '-o', color=col, lw=2.2, ms=4.5, label=lab)
            ax.annotate(lab, (x.iloc[-1], gg.paired_difference.iloc[-1]), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=7.5, color=col)
        ax.axhline(0, color='#333', lw=.9)
        ax.text(0.01, 0.004, 'random deletion', transform=ax.get_yaxis_transform(), ha='left', va='bottom', fontsize=7, color='#333')
        n = int(g.n_records.iloc[0]) if 'n_records' in g else None
        ax.set_title(f'{"1D" if b == "1d" else "2D"} model' + (f'  (n = {n})' if n else ''), loc='left',
                     fontsize=10, color=INK)
        ax.set_xlabel('Highest-attribution signal deleted (%)', fontsize=8.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#EEE')
        ax.set_xlim(0, 100 * g.fraction.max() * 1.32)
    axs[0].set_ylabel('Prediction-probability drop\nminus random deletion', fontsize=8.5)
    if not PLAIN: fig.suptitle('Deletion faithfulness: does deleting what the map highlights hurt the model more than random?',
                 x=.06, ha='left', fontsize=11.5, weight='bold', color='#26324A')
    if not PLAIN: fig.text(.06, -.03, 'Above 0 = more faithful than random. Bands = paired bootstrap 95 % CI. '
             'Same records and same random deletions for every method.', fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, .95))
    if PLAIN:
        fig.set_size_inches(9.4, 3.4)
        for ax in axs:
            ax.title.set_weight('bold')
    out = Path(args.out or 'figures/Module3_deletion_faithfulness').expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig); print('Saved:', out.with_suffix('.png'))


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--cohort-index', type=int, default=209)
    common.add_argument('--meta', default='~/mm_gradcam/m1_input/gradcam_full/gradcam_metadata.csv')
    common.add_argument('--image-root', default='~/mm_gradcam/MM_GradCAM_HPC')
    common.add_argument('--x0', type=int, default=DEFAULT_X0, help='left waveform pixel of the 2D image')
    common.add_argument('--x1', type=int, default=DEFAULT_X1, help='right waveform pixel of the 2D image')
    common.add_argument('--out', default=None, help='output path without extension')
    a = sub.add_parser('m3a', parents=[common]); a.add_argument('--maps', required=True)
    a.add_argument('--trace-mask', default=None, help='npy mask of the waveform over the valid columns')
    a.add_argument('--class-name', default='AFib')
    b = sub.add_parser('m3b', parents=[common])
    b.add_argument('--maps3a', required=True); b.add_argument('--maps3b', required=True)
    c = sub.add_parser('deletion'); c.add_argument('--summary', required=True); c.add_argument('--out', default=None)
    args = ap.parse_args()
    {'m3a': fig_m3a, 'm3b': fig_m3b, 'deletion': fig_deletion}[args.cmd](args)


if __name__ == '__main__':
    main()
