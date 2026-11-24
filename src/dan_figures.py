#!/usr/bin/env python3
"""Figures for the DAN-MMGradCAM manuscript (brown = 1D, pink = 2D, blue = shared / fusion).

Plotting functions called by the notebooks. Each function writes PDF + PNG to the given folder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

C1, C2, CB = '#C8874A', '#EC7BAE', '#58A0DC'          # 1D, 2D, shared / fusion
CNAVY, CAMB, CGREY, CNULL = '#2C6CA8', '#E3B04B', '#D3D5D9', '#B4B8BE'
INK, INK2 = '#222226', '#5E5C66'
CLASSES = ['AFib', 'GSVT', 'SB', 'SR']
seq = lambda c, d: LinearSegmentedColormap.from_list('s', ['#DEDEDE', c, d])
CM1, CM2, CMB = seq(C1, '#8A5528'), seq(C2, '#B9467C'), seq(CB, '#2C6CA8')
CMS = LinearSegmentedColormap.from_list('stab', ['#F6F6F6', '#C9D6E6', '#7D93B2'])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
                     'axes.titlepad': 6,
                     'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.edgecolor': '#9A9AA0', 'axes.linewidth': .6, 'legend.fontsize': 6.8,
                     'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


# ------------------------------------------------------------- waveform helpers
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


def wave(ax, poly, W, values, cmap, vmax, lw=(.55, 2.6), off=0.0, cat=None):
    """values: per time column (len n); poly x in [0, W). Colour + width follow intensity."""
    n = len(values)
    seg, val = [], []
    for k in range(1, len(poly)):
        (xa, ya), (xb, yb) = poly[k - 1], poly[k]
        if np.isnan(xa) or np.isnan(xb):
            continue
        u = min(n - 1, int(xb * n / W))
        seg.append([(10 * xa / W, -ya + off), (10 * xb / W, -yb + off)]); val.append(values[u])
    seg, val = np.array(seg), np.array(val, float)
    if cat is not None:   # categorical colours: dict value -> (colour, width)
        for key, (col, w) in cat.items():
            s = seg[val == key]
            if len(s):
                ax.add_collection(LineCollection(s, colors=col, linewidths=w, capstyle='round', zorder=2 + w))
        return
    f = np.clip(np.nan_to_num(val / vmax), 0, 1); o = np.argsort(f)
    ax.add_collection(LineCollection(seg[o], colors=cmap(f[o]), linewidths=lw[0] + (lw[1] - lw[0]) * f[o],
                                     capstyle='round'))


def wave_ax(ax, mask, extra=0, label=None, color=INK, xt=False):
    ys = np.flatnonzero(mask.any(1))
    ax.set_xlim(0, 10); ax.set_ylim(-(ys.max() + 6) - extra, -(ys.min() - 6))
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.tick_params(labelbottom=xt, length=2)
    if label:
        ax.text(-.008, .5, label, transform=ax.transAxes, ha='right', va='center', fontsize=8,
                weight='bold', color=color)


def strip(ax, v, cmap, lab):
    ax.imshow(v[None, :], aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=(0, 10, 0, 1), interpolation='nearest')
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-.008, .5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=6.5, color=INK2)


def cbar(fig, ax, cmap, vmax, label):
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, vmax), cmap), ax=ax, fraction=.014, pad=.008, aspect=12)
    cb.ax.tick_params(labelsize=6, length=1.5); cb.set_label(label, fontsize=6.5); cb.outline.set_linewidth(.4)


def spacer(fig, ax):
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), CMB), ax=ax, fraction=.014, pad=.008, aspect=12)
    cb.ax.set_visible(False)


def topk(x, f):
    k = max(1, int(np.ceil(f * x.size)))
    s = np.zeros(x.size, bool); s[np.argsort(-x, kind='stable')[:k]] = True
    return s


def mass(x):
    x = np.clip(np.nan_to_num(x), 0, None); return x / x.sum()


# ------------------------------------------------------------- figures
def fig_case_module1(z, i, mask, out):
    v = z['valid_600'][i].astype(bool)
    g1, g2 = z['base1'][i][v], z['base2'][i][v]
    p, q = mass(g1), mass(g2)
    sh = np.minimum(p, q)
    a, b = topk(p, .15), topk(q, .15)
    r = (p - q) / (p + q + 1e-12)
    cat = np.zeros(p.size); sal = a | b
    cat[sal & (r > 1 / 3)], cat[sal & (r < -1 / 3)], cat[sal & (np.abs(r) <= 1 / 3)] = 1, 2, 3
    poly, W = polyline(mask), mask.shape[1]
    fig, ax = plt.subplots(5, 1, figsize=(7.1, 6.6), gridspec_kw={'height_ratios': [1, 1, 1, 1, .85],
                                                                     'hspace': .62})
    wave(ax[0], poly, W, g1 / g1.max(), CM1, 1.0); wave_ax(ax[0], mask, label='1D', color=C1)
    ax[0].set_title('(a) 1D Grad-CAM intensity on the ECG waveform', loc='left'); cbar(fig, ax[0], CM1, 1, '$g_1$')
    wave(ax[1], poly, W, g2 / g2.max(), CM2, 1.0); wave_ax(ax[1], mask, label='2D', color=C2)
    ax[1].set_title('(b) 2D Grad-CAM intensity received by the waveform (on-trace projection)', loc='left')
    cbar(fig, ax[1], CM2, 1, '$g_2$')
    wave(ax[2], poly, W, sh / sh.max(), CMB, 1.0, lw=(.55, 3.0)); wave_ax(ax[2], mask, label='shared', color=CB)
    ax[2].set_title(r'(c) Intensity-aware shared attribution $a(u)=\min(p_u,q_u)$', loc='left')
    cbar(fig, ax[2], CMB, 1, 'relative')
    wave(ax[3], poly, W, cat, None, None, cat={0: (CGREY, .55), 3: (CB, 2.3), 2: (C2, 2.3), 1: (C1, 2.3)})
    wave_ax(ax[3], mask, label='direction')
    ax[3].set_title('(d) Direction ($|r_u|>1/3$)', loc='left'); spacer(fig, ax[3])
    ax[3].legend(handles=[Line2D([], [], color=C1, lw=2.3, label='1D-dominant'),
                          Line2D([], [], color=C2, lw=2.3, label='2D-dominant'),
                          Line2D([], [], color=CB, lw=2.3, label='balanced'),
                          Line2D([], [], color=CGREY, lw=1.5, label='not salient')],
                 ncol=4, frameon=False, loc='upper right', bbox_to_anchor=(1, 1.32))
    t = np.arange(p.size) * 10 / p.size
    ax[4].fill_between(t, 0, sh, color=CB, alpha=.35, lw=0, label='shared $\\min(p,q)$')
    ax[4].plot(t, p, color=C1, lw=.9, label='1D mass $p_u$'); ax[4].plot(t, q, color=C2, lw=1.2, label='2D mass $q_u$')
    ax[4].set_xlim(0, 10); ax[4].set_ylim(0, max(p.max(), q.max()) * 1.2)
    ax[4].set_xlabel('ECG time within valid waveform (s)'); ax[4].set_ylabel('mass', fontsize=7)
    ax[4].legend(ncol=3, frameon=False, loc='upper left')
    ax[4].set_title('(e) Attribution mass on the common time axis', loc='left'); spacer(fig, ax[4])
    for k in range(4):
        ax[k].set_xticks(range(0, 11, 2))
    save(fig, out, 'fig_module1_case209_drawn')


def fig_module1_cohort(per, null, out):
    prim = per[per.both_correct & ~per.degenerate]
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.9), gridspec_kw={'wspace': .32, 'hspace': .55})
    ax = axs.ravel()
    pos = np.arange(4)
    for k, (m, ttl) in enumerate((('O', '(a) Shared mass $O$'), ('spearman', r'(b) Spearman $\rho_s$'))):
        obs = [prim[prim.class_name == c][m] for c in CLASSES]
        nul = [null[null.class_name == c][m] for c in CLASSES]
        b1 = ax[k].boxplot(obs, positions=pos - .19, widths=.32, patch_artist=True, showfliers=False)
        b2 = ax[k].boxplot(nul, positions=pos + .19, widths=.32, patch_artist=True, showfliers=False)
        for bp, col, al in ((b1, CB, .75), (b2, CNULL, .55)):
            for el in bp['boxes']:
                el.set(facecolor=col, alpha=al, edgecolor=col, lw=.8)
            for el in bp['medians']:
                el.set(color='#4A4A50', lw=1)
            for key in ('whiskers', 'caps'):
                for el in bp[key]:
                    el.set(color=col, lw=.8)
        ax[k].set_xticks(pos); ax[k].set_xticklabels(CLASSES); ax[k].set_title(ttl, loc='left')
        if m == 'spearman':
            ax[k].axhline(0, color='#999', lw=.5)
    ax[0].legend(handles=[Patch(color=CB, alpha=.75, label='same ECG'), Patch(color=CNULL, alpha=.55, label='mispaired (null)')],
                 frameon=False, loc='upper left', ncol=2, fontsize=6.6, bbox_to_anchor=(0, 1.0))
    ax[0].set_ylim(.12, .78)
    bp = ax[2].boxplot([prim.entropy_1D, prim.entropy_2D], widths=.45, patch_artist=True, showfliers=False)
    for el, col in zip(bp['boxes'], (C1, C2)):
        el.set(facecolor=col, alpha=.8, edgecolor=col)
    for el in bp['medians']:
        el.set(color='#4A4A50')
    ax[2].set_xticks([1, 2]); ax[2].set_xticklabels(['1D', '2D']); ax[2].set_ylim(.6, 1.0)
    ax[2].set_ylabel('normalised entropy'); ax[2].set_title('(c) Map flatness', loc='left')
    w = .26
    for j, (k, col, lab) in enumerate((('dir_1D_dominant', C1, '1D-dominant'), ('dir_2D_dominant', C2, '2D-dominant'),
                                       ('dir_balanced', CB, 'balanced'))):
        med = [prim[prim.class_name == c][k].median() for c in CLASSES]
        ax[3].bar(pos + (j - 1) * w, med, w * .92, color=col, label=lab)
    ax[3].set_xticks(pos); ax[3].set_xticklabels(CLASSES); ax[3].set_ylabel('% of valid time')
    ax[3].set_title('(d) Direction', loc='left'); ax[3].set_ylim(0, 18)
    ax[3].legend(frameon=False, fontsize=6.6, loc='upper center', ncol=3, bbox_to_anchor=(.5, 1.0))
    save(fig, out, 'fig_module1_cohort')


def fig_module2(per, prr, out):
    prim = per[per.both_correct & ~per.degenerate]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={'wspace': .42})
    lab = ['scale', 'drift\n0.05', 'drift\n0.10', 'noise\n0.05', 'noise\n0.10']
    x = np.arange(5)
    ax[0].bar(x - .19, 100 * prr.prr_1d, .36, color=C1, label='1D CNN')
    ax[0].bar(x + .19, 100 * prr.prr_2d, .36, color=C2, label='2D CNN')
    ax[0].set_ylim(90, 100); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=6.4)
    ax[0].set_ylabel('prediction retained (%)'); ax[0].set_title('(a) Prediction retention', loc='left')
    pos = np.arange(4)
    for d, col, key in ((-.19, C1, 'S1_mean'), (.19, C2, 'S2_mean')):
        bp = ax[1].boxplot([prim[prim.class_name == c][key] for c in CLASSES], positions=pos + d, widths=.32,
                           patch_artist=True, showfliers=False)
        for el in bp['boxes']:
            el.set(facecolor=col, alpha=.8, edgecolor=col)
        for el in bp['medians']:
            el.set(color='#4A4A50')
    ax[1].set_xticks(pos); ax[1].set_xticklabels(CLASSES); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel('mean stability'); ax[1].set_title('(b) Stability by class', loc='left')
    for j, (b, col) in enumerate((('S1', C1), ('S2', C2))):
        s_, r_ = prim[f'{b}_salient'], prim[f'{b}_rest']
        for k, (vals, mk) in enumerate(((s_, 'o'), (r_, 's'))):
            med = vals.median(); lo, hi = vals.quantile(.25), vals.quantile(.75)
            xx = j + (k - .5) * .36
            ax[2].errorbar(xx, med, yerr=[[med - lo], [hi - med]], fmt=mk, color=col, ms=6, lw=1.1, capsize=2.5,
                           mfc=col if k == 0 else 'white')
        ax[2].plot([j - .18, j + .18], [s_.median(), r_.median()], color=col, lw=.9, alpha=.7)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(['1D', '2D']); ax[2].set_ylim(0, 1); ax[2].set_xlim(-.6, 1.6)
    ax[2].set_ylabel('stability (median, IQR)'); ax[2].set_title('(c) Salient vs other', loc='left')
    fig.legend(handles=[Patch(color=C1, label='1D CNN'), Patch(color=C2, label='2D CNN'),
                        Line2D([], [], marker='o', color=INK2, lw=0, label='top-15 % salient columns'),
                        Line2D([], [], marker='s', color=INK2, mfc='white', lw=0, label='other columns')],
               ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.02), fontsize=6.8)
    save(fig, out, 'fig_module2_stability')


def fig_case_module3a(z, i, mask, out):
    v = z['valid_600'][i].astype(bool)
    g = {k: z[k][i][v] for k in ('mu1', 'S1', 'H1', 'mu2', 'S2', 'H2', 'R')}
    R0 = np.sqrt(g['mu1'] * g['mu2'])
    vH = max(g['H1'].max(), g['H2'].max(), g['R'].max())
    t0, tR = topk(R0, .1), topk(g['R'], .1)
    cat = np.zeros(R0.size); cat[t0 & tR], cat[tR & ~t0], cat[t0 & ~tR] = 3, 1, 2
    poly, W = polyline(mask), mask.shape[1]
    ys = np.flatnonzero(mask.any(1)); span = ys.max() - ys.min() + 12
    fig = plt.figure(figsize=(7.1, 8.4))
    gs = fig.add_gridspec(11, 1, height_ratios=[1, .14, .42, 1, .14, .42, 1.75, .38, 1.1, .38, 1.0], hspace=.25)
    for r0, (mu, S, cm, col, nm) in ((0, (g['mu1'], g['S1'], CM1, C1, '1D')), (3, (g['mu2'], g['S2'], CM2, C2, '2D'))):
        a = fig.add_subplot(gs[r0]); s = fig.add_subplot(gs[r0 + 1])
        wave(a, poly, W, mu, cm, 1.0); wave_ax(a, mask, label=nm, color=col)
        sub = '1D' if nm == '1D' else '2D'
        a.set_title(f'({"a" if nm == "1D" else "b"}) Perturbation-mean attribution $\\mu_{{{sub}}}(u)$ '
                    f'and local stability $S_{{{sub}}}(u)$ (strip, mean {S.mean():.2f})', loc='left')
        strip(s, S, CMS, f'$S_{{{sub}}}$'); spacer(fig, s)
        cbar(fig, a, cm, 1, f'$\\mu_{{{sub}}}$')
    for k in (2, 5, 7, 9):
        fig.add_subplot(gs[k]).axis('off')
    b = fig.add_subplot(gs[6])
    wave(b, poly, W, g['H1'], CM1, vH); wave(b, poly, W, g['H2'], CM2, vH, off=-span)
    wave_ax(b, mask, extra=span)
    b.text(-.008, .75, '$H_{1D}$', transform=b.transAxes, ha='right', va='center', fontsize=8, color=C1, weight='bold')
    b.text(-.008, .25, '$H_{2D}$', transform=b.transAxes, ha='right', va='center', fontsize=8, color=C2, weight='bold')
    b.set_title(f'(c) Stability-weighted attribution $H_b=\\mu_b S_b$ (shared scale 0–{vH:.2f}; unstable parts fade)',
                loc='left')
    cbar(fig, b, CMS, 1, '$S_b$ (strips)')
    c = fig.add_subplot(gs[8])
    wave(c, poly, W, g['R'], CMB, vH, lw=(.6, 3.4)); wave_ax(c, mask, label='$R$', color=CB)
    c.set_title('(d) Module 3A reliability-weighted attribution $R(u)=\\sqrt{H_{1D}(u)H_{2D}(u)}$ (same scale as c)',
                loc='left')
    cbar(fig, c, CMB, vH, '$H$, $R$')
    d = fig.add_subplot(gs[10])
    wave(d, poly, W, cat, None, None, cat={0: (CGREY, .55), 3: (CB, 2.2), 2: (CAMB, 2.2), 1: (CNAVY, 2.4)})
    wave_ax(d, mask, xt=True); spacer(fig, d)
    d.set_xlabel('ECG time within valid waveform (s)')
    d.set_title('(e) Top-10 % location: $\\sqrt{\\mu_{1D}\\mu_{2D}}$ vs $R$', loc='left')
    d.legend(handles=[Line2D([], [], color=CB, lw=2.2, label='kept'), Line2D([], [], color=CNAVY, lw=2.2, label='entered'),
                      Line2D([], [], color=CAMB, lw=2.2, label='dropped')], ncol=3, frameon=False,
             loc='upper right', bbox_to_anchor=(1, 1.28))
    save(fig, out, 'fig_module3A_case209')


def fig_faith(summ, pairs, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw={'wspace': .7, 'width_ratios': [1, 1, .95]})
    style = {'base1': ('Grad-CAM', CNULL, 'o'), 'base2': ('Grad-CAM', CNULL, 'o'), 'H1': ('$H=\\mu S$', C1, 's'),
             'H2': ('$H=\\mu S$', C2, 's'), 'R': ('$R$ (3A)', CB, 'D')}
    for k, b in enumerate(('1d', '2d')):
        g = summ[summ.branch == b]
        for m in ('base1', 'base2', 'H1', 'H2', 'R'):
            gg = g[g.method == m].sort_values('fraction')
            if gg.empty:
                continue
            lab, col, mk = style[m]; x = 100 * gg.fraction
            ax[k].fill_between(x, gg.difference_ci_low, gg.difference_ci_high, color=col, alpha=.16, lw=0)
            ax[k].plot(x, gg.paired_difference, marker=mk, ms=3.5, color=col, lw=1.4, label=lab)
        ax[k].axhline(0, color=INK, lw=.7); ax[k].text(31, -.006, 'random', fontsize=6, color=INK2, va='top', ha='right')
        ax[k].set_xlabel('top-attributed signal deleted (%)')
        ax[k].set_title(f'({"ab"[k]}) {"1D" if b == "1d" else "2D"} CNN', loc='left'); ax[k].set_ylim(-.2, .16)
        ax[k].legend(frameon=False, loc='lower left', fontsize=6.2)
    ax[0].set_ylabel('probability drop − random')
    lab = {('1d', 'H1 - base1'): '1D: $H$ − Grad-CAM', ('1d', 'R - H1'): '1D: $R$ − $H$',
           ('2d', 'H2 - base2'): '2D: $H$ − Grad-CAM', ('2d', 'R - H2'): '2D: $R$ − $H$'}
    rows = [(k, pairs[(pairs.branch == k[0]) & (pairs.comparison == k[1])].iloc[0]) for k in lab]
    for y, (k, r) in enumerate(rows[::-1]):
        col = C1 if k[0] == '1d' else C2
        ax[2].errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt='o', color=col,
                       ms=4.5, capsize=2, lw=1.1)
    ax[2].axvline(0, color=INK, lw=.7)
    ax[2].set_yticks(range(len(rows))); ax[2].set_yticklabels([lab[k] for k, _ in rows[::-1]], fontsize=6.6)
    ax[2].set_xlabel('paired difference in mean drop (AOPC)'); ax[2].set_title('(c) Component effect, 95 % CI', loc='left')
    save(fig, out, 'fig_component_faithfulness')


CLS_COL = {'AFib': '#4FB3A9', 'GSVT': '#E3B04B', 'SB': '#9B8BD4', 'SR': '#7D8A99'}
CLS_MK = {'AFib': 'o', 'GSVT': 's', 'SB': '^', 'SR': 'D'}


def fig_confusion(p1, p2, out):
    from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots(1, 2, figsize=(3.6, 1.95), gridspec_kw={'wspace': .38})
    for k, (d, cm, nm) in enumerate(((p1, CM1, '1D signal CNN'), (p2, CM2, '2D image CNN'))):
        M = confusion_matrix(d.true_class, d.predicted_class, labels=CLASSES)
        P = M / M.sum(1, keepdims=True)
        ax[k].imshow(P, cmap=LinearSegmentedColormap.from_list('c', ['#FFFFFF', [C1, C2][k], ['#8A5528', '#B9467C'][k]]),
                     vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax[k].text(j, i, f'{M[i, j]}', ha='center', va='center', fontsize=6,
                           color='white' if P[i, j] > .6 else INK)
        acc = np.trace(M) / M.sum()
        ax[k].set_xticks(range(4)); ax[k].set_yticks(range(4))
        ax[k].set_xticklabels(CLASSES); ax[k].set_yticklabels(CLASSES)
        ax[k].set_xlabel('predicted'); ax[k].set_ylabel('true')
        ax[k].set_title(f'({"ab"[k]}) {nm.split()[0]}: {100 * acc:.2f}%', loc='left', fontsize=6.8)
        ax[k].tick_params(labelsize=5.8); ax[k].xaxis.label.set_size(6.2); ax[k].yaxis.label.set_size(6.2)
        for sp in ax[k].spines.values():
            sp.set_visible(False)
        ax[k].tick_params(length=0)
    save(fig, out, 'fig_replication_confusion')


def fig_m1_baseline(mm, out):
    ok = mm[(mm.degenerate_1d.astype(str) == 'False') & (mm.degenerate_2d.astype(str) == 'False')]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw={'wspace': .42, 'width_ratios': [1.25, 1, 1]})
    for c in CLASSES:
        g = ok[ok.class_name == c]
        ax[0].scatter(g.mean_agreement, g.mean_disagreement, s=6, color=CLS_COL[c], marker=CLS_MK[c], alpha=.7,
                      lw=0, label=c)
    ax[0].set_xlabel('mean agreement $\\bar A$'); ax[0].set_ylabel('mean disagreement $\\bar D$')
    ax[0].set_title('(a) Agreement vs disagreement', loc='left')
    ax[0].legend(frameon=False, fontsize=6.4, markerscale=2, loc='lower right', ncol=2)
    for k, (col, key, ttl) in enumerate(((CB, 'mean_agreement', '(b) $\\bar A$ by class'),
                                         (CNULL, 'mean_disagreement', '(c) $\\bar D$ by class'))):
        bp = ax[k + 1].boxplot([ok[ok.class_name == c][key] for c in CLASSES], widths=.55, patch_artist=True,
                               showfliers=False)
        for el in bp['boxes']:
            el.set(facecolor=col, alpha=.75, edgecolor=col)
        for el in bp['medians']:
            el.set(color='#4A4A50')
        ax[k + 1].set_xticklabels(CLASSES); ax[k + 1].set_title(ttl, loc='left')
    save(fig, out, 'fig_module1_minmax_baseline')
    return len(ok)


def recolor_branch_png(src, dst_png):
    """Re-colour a Module 1 branch figure (matplotlib default colours) into the paper theme.
    Each colour is matched as an alpha blend over white and replaced by the theme colour
    with the same alpha, so antialiasing and the 50 % area fills are preserved."""
    from PIL import Image
    a = np.asarray(Image.open(src).convert('RGB')).astype(float)
    W = np.array([255., 255., 255.])
    hexrgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)
    blue, orange = np.array([31., 119., 180.]), np.array([255., 127., 14.])
    dark = (a.sum(2) < 200)
    rows = [y for y in range(a.shape[0]) if dark[y].mean() > .6]
    edges = sorted(set(r for r in rows if r - 1 not in rows))
    panels = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    maps = [[(blue, hexrgb('#5A5A60'))], [(blue, hexrgb(C1)), (orange, hexrgb(C2))],
            [(blue, hexrgb(CB))], [(blue, hexrgb('#AEB3BA'))]]
    out = a.copy()
    for (y0, y1), mp in zip(panels, maps):
        reg = a[y0:y1]
        for src_c, dst_c in mp:
            v = W - src_c
            al = ((W - reg) @ v) / (v @ v)
            res = np.linalg.norm(W - al[..., None] * v - reg, axis=2)
            m = (al > .08) & (al < 1.08) & (res < 22)
            new = W - np.clip(al, 0, 1)[..., None] * (W - dst_c)
            blk = out[y0:y1]; blk[m] = new[m]; out[y0:y1] = blk
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(dst_png)


def fig_m1_examples(srcs, out):
    from PIL import Image, ImageDraw, ImageFont
    tmp = []
    for k, s in enumerate(srcs):
        p = out / f'_tmp_{k}.png'; recolor_branch_png(s, p); tmp.append(Image.open(p))
    w = 1900
    ims = [im.resize((w, int(im.height * w / im.width)), Image.Resampling.LANCZOS) for im in tmp]
    gap = 60
    canvas = Image.new('RGB', (2 * w + gap, max(i.height for i in ims) + 70), 'white')
    dr = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype('DejaVuSans.ttf', 44)
    except OSError:
        f = ImageFont.load_default()
    for k, im in enumerate(ims):
        canvas.paste(im, (k * (w + gap), 70))
        dr.text((k * (w + gap) + 10, 8), ['(a) high agreement', '(b) high disagreement'][k], fill=(30, 30, 34), font=f)
    canvas.save(out / 'fig_module1_examples.png', dpi=(300, 300))
    for k in range(len(srcs)):
        (out / f'_tmp_{k}.png').unlink()
    print('saved fig_module1_examples')


# ------------------------------------------------------------- Module 2 detail figures
def fig_module2_detail(z, rec, per, out):
    """Module 2 in one figure: prediction changes, branch stability, stability over time, salient vs other."""
    from matplotlib.gridspec import GridSpec
    bc = ((rec.pred_1d == rec.class_name) & (rec.pred_2d == rec.class_name)).to_numpy()
    prim = per[per.both_correct & ~per.degenerate]
    fams = ['scale', 'drift_0.05', 'drift_0.1', 'noise_0.05', 'noise_0.1']
    fam_lab = ['scale', 'drift\n0.05', 'drift\n0.10', 'noise\n0.05', 'noise\n0.10']
    fig = plt.figure(figsize=(7.1, 5.3))
    gs = GridSpec(2, 3, figure=fig, hspace=.72, wspace=.45)
    chg = {b: np.array([[100 * (1 - rec[rec.class_name == c][f'prr_{b}_{f}'].mean()) for f in fams] for c in CLASSES])
           for b in ('1d', '2d')}
    vmax = max(m.max() for m in chg.values())
    for k, (b, nm) in enumerate((('1d', '1D CNN'), ('2d', '2D CNN'))):
        ax = fig.add_subplot(gs[0, k]); M = chg[b]
        cmap = LinearSegmentedColormap.from_list('h', ['#FFFFFF', [C1, C2][k], ['#8A5528', '#B9467C'][k]])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6.3,
                        color='white' if M[i, j] > .6 * vmax else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels(fam_lab, fontsize=6); ax.set_yticks(range(4))
        ax.set_yticklabels(CLASSES); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'({"ab"[k]}) {nm[:2]}: % predictions changed', loc='left')
    ax = fig.add_subplot(gs[0, 2]); pos = np.arange(4)
    for d, col, key in ((-.19, C1, 'S1_mean'), (.19, C2, 'S2_mean')):
        bp = ax.boxplot([prim[prim.class_name == c][key] for c in CLASSES], positions=pos + d, widths=.32,
                        patch_artist=True, showfliers=False)
        for el in bp['boxes']:
            el.set(facecolor=col, alpha=.8, edgecolor=col)
        for el in bp['medians']:
            el.set(color='#4A4A50')
    ax.set_xticks(pos); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1); ax.set_ylabel('mean stability')
    ax.set_title('(c) Stability by class', loc='left')
    V = z['valid_600'][0].astype(bool); n = int(V.sum()); t = np.arange(n) * 10 / n
    idx = np.flatnonzero(bc)
    for k, (key, ttl, ylab) in enumerate((('S1', '(d) 1D stability over time', '$S_{1D}(u)$, class mean'),
                                          ('S2', '(e) 2D stability over time', '$S_{2D}(u)$, class mean'))):
        ax = fig.add_subplot(gs[1, k])
        for c in CLASSES:
            ii = [i for i in idx if rec.class_name.iloc[i] == c]
            ax.plot(t, z[key][ii][:, V].mean(0), color=CLS_COL[c], lw=.9, label=c)
        ax.set_xlim(0, 10); ax.set_ylim(.3, 1.0); ax.set_xlabel('time (s)'); ax.set_ylabel(ylab)
        ax.set_title(ttl, loc='left')
        if key == 'S2':
            ax.axvspan(0, 1, color='#EEEEEE', zorder=0, lw=0)
            ax.text(1.15, .97, 'image margin', ha='left', va='top', fontsize=6, color=INK2)
    ax = fig.add_subplot(gs[1, 2])
    for j, (b, col) in enumerate((('S1', C1), ('S2', C2))):
        s_, r_ = prim[f'{b}_salient'], prim[f'{b}_rest']
        for k, (vals, mk) in enumerate(((s_, 'o'), (r_, 's'))):
            med = vals.median(); lo, hi = vals.quantile(.25), vals.quantile(.75)
            ax.errorbar(j + (k - .5) * .36, med, yerr=[[med - lo], [hi - med]], fmt=mk, color=col, ms=6, lw=1.1,
                        capsize=2.5, mfc=col if k == 0 else 'white')
        ax.plot([j - .18, j + .18], [s_.median(), r_.median()], color=col, lw=.9, alpha=.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['1D', '2D']); ax.set_ylim(0, 1); ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('stability (median, IQR)'); ax.set_title('(f) Salient vs other', loc='left')
    fig.legend(handles=[Patch(color=C1, label='1D CNN'), Patch(color=C2, label='2D CNN')] +
               [Line2D([], [], color=CLS_COL[c], lw=1.5, label=c) for c in CLASSES] +
               [Line2D([], [], marker='o', color=INK2, lw=0, label='top-15 % salient'),
                Line2D([], [], marker='s', color=INK2, mfc='white', lw=0, label='other columns')],
               ncol=8, frameon=False, loc='upper center', bbox_to_anchor=(.5, .02), fontsize=6.3, columnspacing=1.0,
               handlelength=1.4)
    save(fig, out, 'fig_module2_detail')


def crop_rows(src, dst, top_frac, bottom_frac):
    from PIL import Image
    im = Image.open(src).convert('RGB'); w, h = im.size
    im.crop((0, int(h * top_frac), w, int(h * bottom_frac))).save(dst)


def fig_module3b(m3b_npz, rec3b, mask, out):
    """Module 3B on ECG 209 (real maps) + cohort distributions (693 ECGs)."""
    from matplotlib.gridspec import GridSpec
    b = np.load(m3b_npz); k = int(np.flatnonzero(b['cohort_index'] == 209)[0]); v = b['valid_600'][k].astype(bool)
    R = b['R'][k][v]; F = b['F_trace'][k][v]
    maps = [('$R(u)$ (Module 3A)', R)] + [(f'$R_{{trace}}(u)$, $\\lambda={l}$', b[f'Rtrace_lambda_{t}'][k][v])
                                         for l, t in (('0.5', '0p5'), ('1', '1p0'), ('2', '2p0'))]
    vmax = float(R.max()); poly, W = polyline(mask), mask.shape[1]
    fig = plt.figure(figsize=(7.1, 7.3))
    gs = GridSpec(7, 3, figure=fig, height_ratios=[.16, 1, 1, 1, 1, .25, 1.35], hspace=.55, wspace=.45)
    s0 = fig.add_subplot(gs[0, :]); strip(s0, F, CMS, '$F_{trace}$'); spacer(fig, s0)
    s0.set_title(f'(a) Waveform support $F_{{trace}}(u)$ of ECG 209 (mean {F.mean():.2f}; darker = more 2D CAM on the trace)', loc='left')
    for r, (lab, m) in enumerate(maps):
        ax = fig.add_subplot(gs[1 + r, :])
        wave(ax, poly, W, m, CMB, vmax, lw=(.5, 3.0)); wave_ax(ax, mask, xt=(r == 3))
        kept = 100 * m.sum() / R.sum()
        ax.set_title(f'({"bcde"[r]}) {lab}' + ('' if r == 0 else f'  —  {kept:.0f} % of the mass of $R$ kept'), loc='left')
        cbar(fig, ax, CMB, vmax, 'shared scale') if r == 0 else spacer(fig, ax)
    fig.axes[-1].set_xlabel('ECG time within valid waveform (s)')
    fig.add_subplot(gs[5, :]).axis('off')
    L = [.5, 1.0, 2.0]
    for c, (key, ttl, yl) in enumerate((('Rtrace_over_R_mean', '(f) Mass kept vs $R$', 'fraction of $R$'),
                                        ('top10_IoU', '(g) Top-10 % IoU vs $R$', 'IoU'))):
        ax = fig.add_subplot(gs[6, c])
        bp = ax.boxplot([rec3b[rec3b['lambda'] == l][key] for l in L], widths=.5, patch_artist=True, showfliers=False)
        for el in bp['boxes']:
            el.set(facecolor=CB, alpha=.75, edgecolor=CB)
        for el in bp['medians']:
            el.set(color='#4A4A50')
        ax.set_xticklabels(['0.5', '1', '2']); ax.set_xlabel('$\\lambda$'); ax.set_ylabel(yl); ax.set_ylim(0, 1)
        ax.set_title(ttl, loc='left')
    ax = fig.add_subplot(gs[6, 2])
    f1 = rec3b[rec3b['lambda'] == .5]
    ax.hist(f1.F_trace_mean, bins=30, color=C2, alpha=.85)
    ax.axvline(f1.F_trace_mean.median(), color='#4A4A50', lw=1)
    ax.set_xlabel('mean $F_{trace}$ per ECG'); ax.set_ylabel('ECGs'); ax.set_title('(h) On-trace 2D CAM share', loc='left')
    save(fig, out, 'fig_module3B')


def fig_faith692(curves, paired, out):
    from matplotlib.gridspec import GridSpec
    sty = {'R': ('$R$ (3A)', CB, 'o', '-'), 'Fbase': ('$F_{base}$', CNULL, 's', '-'),
           'Fmu_matched30': ('$F_{\\mu}$ (matched-30)', '#A0A7B0', 'v', '--'),
           'Rtrace_0p5': ('$R_{trace}$, $\\lambda=0.5$', '#F2B4CF', 'D', '-'),
           'Rtrace_1p0': ('$R_{trace}$, $\\lambda=1$', C2, 'D', '-'),
           'Rtrace_2p0': ('$R_{trace}$, $\\lambda=2$', '#B9467C', 'D', '-'), 'RANDOM': ('random', '#555555', '', ':')}
    fig = plt.figure(figsize=(7.1, 5.0))
    gs = GridSpec(2, 3, figure=fig, hspace=.62, wspace=.5)
    for k, b in enumerate(('1d', '2d')):
        ax = fig.add_subplot(gs[0, k])
        for m, (lab, col, mk, ls) in sty.items():
            if m == 'Fmu_matched30':
                continue   # defined on the 551-record subset only; compared in (c)-(e)
            g = curves[(curves.branch == b) & (curves.method == m)].sort_values('fraction')
            ax.plot(100 * g.fraction, g.probability_drop, ls, color=col, marker=mk, ms=3, lw=1.3, label=lab)
        ax.set_xlabel('signal deleted (%)'); ax.set_ylabel('mean probability drop'); ax.set_ylim(0, .43)
        ax.set_title(f'({"ab"[k]}) {b.upper()} CNN', loc='left')
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, ls=ls, ms=3, lw=1.3, label=l) for m, (l, c, mk, ls) in sty.items() if m != 'Fmu_matched30'],
               loc='upper center', ncol=7, frameon=False, bbox_to_anchor=(.5, .015), fontsize=6.2, columnspacing=.9, handlelength=1.6)
    ax = fig.add_subplot(gs[0, 2])
    comps = ['Fbase - R', 'Fmu_matched30 - R', 'Rtrace_0p5 - R', 'Rtrace_1p0 - R', 'Rtrace_2p0 - R', 'RANDOM - R']
    labs = ['$F_{base}$', '$F_{\\mu}$', '$\\lambda=0.5$', '$\\lambda=1$', '$\\lambda=2$', 'random']
    A = paired[paired.class_name == 'ALL']
    for d, (b, col) in ((-.17, ('1d', C1)), (.17, ('2d', C2))):
        for y, cpn in enumerate(comps):
            r = A[(A.branch == b) & (A.comparison == cpn)].iloc[0]
            ax.errorbar(r.mean_delta_auc, y + d, xerr=[[r.mean_delta_auc - r.ci95_lower], [r.ci95_upper - r.mean_delta_auc]],
                        fmt='o', color=col, ms=3.5, capsize=1.5, lw=1)
    ax.axvline(0, color=INK, lw=.7); ax.set_yticks(range(len(comps))); ax.set_yticklabels(labs, fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlabel('$\\Delta$AUC vs $R$'); ax.set_title('(c) Paired difference', loc='left')
    ax.legend(handles=[Line2D([], [], marker='o', color=C1, lw=0, label='1D'), Line2D([], [], marker='o', color=C2, lw=0, label='2D')],
              frameon=False, fontsize=6.2, loc='center right')
    cm = LinearSegmentedColormap.from_list('dv', ['#B9467C', '#F7D7E6', '#FFFFFF', '#CFE3F5', '#2C6CA8'])
    for k, b in enumerate(('1d', '2d')):
        ax = fig.add_subplot(gs[1, k])
        M = np.array([[paired[(paired.branch == b) & (paired.class_name == c) & (paired.comparison == cpn)].mean_delta_auc.iloc[0]
                       for c in CLASSES] for cpn in comps])
        ax.imshow(M, cmap=cm, vmin=-.15, vmax=.15, aspect='auto')
        for i in range(M.shape[0]):
            for jj in range(M.shape[1]):
                ax.text(jj, i, f'{M[i, jj]:+.3f}', ha='center', va='center', fontsize=6, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(len(comps))); ax.set_yticklabels(labs, fontsize=6.4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f'({"de"[k]}) {b.upper()}: $\\Delta$AUC vs $R$ by class', loc='left')
    ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
    ax.text(0, .95, 'Blue: method above $R$\nPink: method below $R$\n\n$n$ = 692 ECGs\n($F_{\\mu}$: 551 matched-30)\n\n2D results exploratory:\nrenderer state not\nfully audited',
            va='top', fontsize=6.8, color=INK2, transform=ax.transAxes)
    save(fig, out, 'fig_module3_faith692')


