"""Single-column (3.45 in wide) versions of the report figures for the IEEE two-column layout.

Same data, colours and panel letters as dan_figures.py; only the layout and type size change.
Brown = 1D branch, pink = 2D branch, blue = shared / fused attribution.
"""
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from dan_figures import (C1, C2, CB, CNAVY, CAMB, CGREY, CNULL, INK, INK2, CLASSES, CM1, CM2, CMB, CMS, CLS_COL,
                         polyline, wave, wave_ax, strip, topk, save)

WIDTH = 3.45   # inches, IEEE column width


@contextmanager
def small_type():
    rc = {'font.size': 6.2, 'axes.titlesize': 6.6, 'axes.labelsize': 6.2, 'xtick.labelsize': 5.8,
          'ytick.labelsize': 5.8, 'legend.fontsize': 5.8, 'axes.titlepad': 3, 'axes.linewidth': .5,
          'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5}
    with plt.rc_context(rc):
        yield


def _cbar(fig, ax, cmap, vmax, label):
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, vmax), cmap), ax=ax, fraction=.025, pad=.01, aspect=10)
    cb.ax.tick_params(labelsize=4.8, length=1.2, pad=1); cb.set_label(label, fontsize=5.2, labelpad=1)
    cb.outline.set_linewidth(.3)


def _spacer(fig, ax):
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, 1), CMB), ax=ax, fraction=.025, pad=.01, aspect=10)
    cb.ax.set_visible(False)


def _strip(ax, v, cmap, lab):
    strip(ax, v, cmap, lab)
    for t in ax.texts:
        t.set_fontsize(5.4)


def _label(ax, text, color):
    wave_ax(ax, MASK)
    ax.text(-.01, .5, text, transform=ax.transAxes, ha='right', va='center', fontsize=6.4, weight='bold', color=color)


MASK = None


# ------------------------------------------------------------------ Module 2
def fig_module2_column(z, rec, per, out):
    prim = per[per.both_correct & ~per.degenerate]
    bc = ((rec.pred_1d == rec.class_name) & (rec.pred_2d == rec.class_name)).to_numpy()
    fams = ['scale', 'drift_0.05', 'drift_0.1', 'noise_0.05', 'noise_0.1']
    fam_lab = ['scale', 'drift\n.05', 'drift\n.10', 'noise\n.05', 'noise\n.10']
    with small_type():
        fig = plt.figure(figsize=(WIDTH, 4.9))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=.75, wspace=.42)
        chg = {b: np.array([[100 * (1 - rec[rec.class_name == c][f'prr_{b}_{f}'].mean()) for f in fams] for c in CLASSES])
               for b in ('1d', '2d')}
        vmax = max(m.max() for m in chg.values())
        for k, b in enumerate(('1d', '2d')):
            ax = fig.add_subplot(gs[0, k]); M = chg[b]
            cmap = LinearSegmentedColormap.from_list('h', ['#FFFFFF', [C1, C2][k], ['#8A5528', '#B9467C'][k]])
            ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
            for i in range(4):
                for j in range(5):
                    ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=4.9,
                            color='white' if M[i, j] > .6 * vmax else INK)
            ax.set_xticks(range(5)); ax.set_xticklabels(fam_lab, fontsize=4.9); ax.set_yticks(range(4))
            ax.set_yticklabels(CLASSES); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f'({"ab"[k]}) {b.upper()}: % changed', loc='left')
        V = z['valid_600'][0].astype(bool); n = int(V.sum()); t = np.arange(n) * 10 / n
        idx = np.flatnonzero(bc)
        for k, (key, ttl) in enumerate((('S1', '(c) 1D stability over time'), ('S2', '(d) 2D stability over time'))):
            ax = fig.add_subplot(gs[1, k])
            for c in CLASSES:
                ii = [i for i in idx if rec.class_name.iloc[i] == c]
                ax.plot(t, z[key][ii][:, V].mean(0), color=CLS_COL[c], lw=.7, label=c)
            ax.set_xlim(0, 10); ax.set_ylim(.3, 1.0); ax.set_xlabel('time (s)')
            ax.set_ylabel('class-mean $S(u)$'); ax.set_title(ttl, loc='left')
            if key == 'S2':
                ax.axvspan(0, 1, color='#EEEEEE', zorder=0, lw=0)
        ax = fig.add_subplot(gs[2, 0]); pos = np.arange(4)
        for d, col, key in ((-.19, C1, 'S1_mean'), (.19, C2, 'S2_mean')):
            bp = ax.boxplot([prim[prim.class_name == c][key] for c in CLASSES], positions=pos + d, widths=.32,
                            patch_artist=True, showfliers=False, medianprops={'color': '#4A4A50', 'lw': .8},
                            whiskerprops={'lw': .6}, capprops={'lw': .6})
            for el in bp['boxes']:
                el.set(facecolor=col, alpha=.8, edgecolor=col, lw=.6)
        ax.set_xticks(pos); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 1); ax.set_ylabel('mean stability')
        ax.set_title('(e) Stability by class', loc='left')
        ax = fig.add_subplot(gs[2, 1])
        for j, (b, col) in enumerate((('S1', C1), ('S2', C2))):
            s_, r_ = prim[f'{b}_salient'], prim[f'{b}_rest']
            for k, (vals, mk) in enumerate(((s_, 'o'), (r_, 's'))):
                med = vals.median(); lo, hi = vals.quantile(.25), vals.quantile(.75)
                ax.errorbar(j + (k - .5) * .36, med, yerr=[[med - lo], [hi - med]], fmt=mk, color=col, ms=3.6, lw=.8,
                            capsize=1.8, mfc=col if k == 0 else 'white')
            ax.plot([j - .18, j + .18], [s_.median(), r_.median()], color=col, lw=.7, alpha=.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['1D', '2D']); ax.set_ylim(0, 1); ax.set_xlim(-.6, 1.6)
        ax.set_ylabel('stability (median, IQR)'); ax.set_title('(f) Salient vs other', loc='left')
        fig.legend(handles=[Patch(color=C1, label='1D CNN'), Patch(color=C2, label='2D CNN')] +
                   [Line2D([], [], color=CLS_COL[c], lw=1.2, label=c) for c in CLASSES] +
                   [Line2D([], [], marker='o', color=INK2, lw=0, ms=3, label='top-15 %'),
                    Line2D([], [], marker='s', color=INK2, mfc='white', lw=0, ms=3, label='other')],
                   ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(.5, .045), fontsize=5.4,
                   columnspacing=.8, handlelength=1.2)
        save(fig, out, 'col_module2')


# ------------------------------------------------------------------ Module 3A
def fig_module3a_column(z, i, mask, out):
    global MASK
    MASK = mask
    v = z['valid_600'][i].astype(bool)
    g = {k: z[k][i][v] for k in ('mu1', 'S1', 'H1', 'mu2', 'S2', 'H2', 'R')}
    R0 = np.sqrt(g['mu1'] * g['mu2'])
    vH = max(g['H1'].max(), g['H2'].max(), g['R'].max())
    t0, tR = topk(R0, .1), topk(g['R'], .1)
    cat = np.zeros(R0.size); cat[t0 & tR], cat[tR & ~t0], cat[t0 & ~tR] = 3, 1, 2
    poly, W = polyline(mask), mask.shape[1]
    ys = np.flatnonzero(mask.any(1)); span = ys.max() - ys.min() + 12
    with small_type():
        fig = plt.figure(figsize=(WIDTH, 5.3))
        gs = fig.add_gridspec(9, 1, height_ratios=[1, .13, .30, 1, .13, .30, 1.7, .30, 1], hspace=.32)
        for r0, (mu, S, cm, col, nm) in ((0, (g['mu1'], g['S1'], CM1, C1, '1D')), (3, (g['mu2'], g['S2'], CM2, C2, '2D'))):
            a = fig.add_subplot(gs[r0]); s = fig.add_subplot(gs[r0 + 1])
            wave(a, poly, W, mu, cm, 1.0, lw=(.35, 1.9)); _label(a, nm, col)
            a.set_title(f'({"a" if nm == "1D" else "b"}) $\\mu_{{{nm}}}(u)$ and stability $S_{{{nm}}}(u)$ '
                        f'(strip, mean {S.mean():.2f})', loc='left')
            _strip(s, S, CMS, f'$S_{{{nm}}}$'); _spacer(fig, s)
            _cbar(fig, a, cm, 1, f'$\\mu_{{{nm}}}$')
        for k in (2, 5, 7):
            fig.add_subplot(gs[k]).axis('off')
        b = fig.add_subplot(gs[6])
        wave(b, poly, W, g['H1'], CM1, vH, lw=(.35, 1.9)); wave(b, poly, W, g['H2'], CM2, vH, off=-span, lw=(.35, 1.9))
        wave_ax(b, mask, extra=span)
        b.text(-.01, .75, '$H_{1D}$', transform=b.transAxes, ha='right', va='center', fontsize=6.2, color=C1, weight='bold')
        b.text(-.01, .25, '$H_{2D}$', transform=b.transAxes, ha='right', va='center', fontsize=6.2, color=C2, weight='bold')
        b.set_title('(c) $H_b=\\mu_b S_b$ (one scale; unstable parts fade)', loc='left')
        _cbar(fig, b, CMS, 1, '$S_b$')
        c = fig.add_subplot(gs[8])
        wave(c, poly, W, g['R'], CMB, vH, lw=(.4, 2.3)); _label(c, '$R$', CB)
        c.tick_params(labelbottom=True); c.set_xlabel('time (s)', labelpad=1)
        c.set_title('(d) $R(u)=\\sqrt{H_{1D}H_{2D}}$; thick dark = top-10 % of $R$', loc='left')
        _cbar(fig, c, CMB, vH, '$H$, $R$')
        # top-10 % change marked as ticks above panel (d)
        n = len(cat); x = np.arange(n) * 10 / n
        ytop = -(ys.min() - 6)
        for key, col, lab in ((3, CB, 'kept'), (1, CNAVY, 'entered'), (2, CAMB, 'dropped')):
            xx = x[cat == key]
            c.scatter(xx, np.full(xx.size, ytop + 3), s=2.2, marker='|', color=col, lw=.8, clip_on=False, label=lab)
        c.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.42), fontsize=5.2, markerscale=3,
                 handletextpad=.2, columnspacing=.9, title='ticks above (d): top-10 % columns of $R$ vs fusion without stability',
                 title_fontsize=5.2)
        save(fig, out, 'col_module3A')


# ------------------------------------------------------------------ Module 3B
def fig_module3b_column(m3b_npz, rec3b, mask, out):
    global MASK
    MASK = mask
    b = np.load(m3b_npz); k = int(np.flatnonzero(b['cohort_index'] == 209)[0]); v = b['valid_600'][k].astype(bool)
    R = b['R'][k][v]; F = b['F_trace'][k][v]
    maps = [('$R$ (3A)', R)] + [(f'$R_{{trace}}$, $\\lambda={l}$', b[f'Rtrace_lambda_{t}'][k][v])
                               for l, t in (('0.5', '0p5'), ('1', '1p0'), ('2', '2p0'))]
    vmax = float(R.max()); poly, W = polyline(mask), mask.shape[1]
    with small_type():
        fig = plt.figure(figsize=(WIDTH, 5.0))
        gs = GridSpec(7, 3, figure=fig, height_ratios=[.13, 1, 1, 1, 1, .32, 1.35], hspace=.62, wspace=.62)
        s0 = fig.add_subplot(gs[0, :]); _strip(s0, F, CMS, '$F_{trace}$'); _spacer(fig, s0)
        s0.set_title(f'(a) Waveform support $F_{{trace}}(u)$ of ECG 209 (mean {F.mean():.2f})', loc='left')
        for r, (lab, m) in enumerate(maps):
            ax = fig.add_subplot(gs[1 + r, :])
            wave(ax, poly, W, m, CMB, vmax, lw=(.35, 2.1)); wave_ax(ax, mask, xt=(r == 3))
            kept = 100 * m.sum() / R.sum()
            ax.set_title(f'({"bcde"[r]}) {lab}' + ('' if r == 0 else f': {kept:.0f} % of the mass of $R$ kept'), loc='left')
            _cbar(fig, ax, CMB, vmax, 'shared') if r == 0 else _spacer(fig, ax)
        fig.axes[-1].set_xlabel('time (s)', labelpad=1)
        fig.add_subplot(gs[5, :]).axis('off')
        L = [.5, 1.0, 2.0]
        for c, (key, ttl, yl) in enumerate((('Rtrace_over_R_mean', '(f) Mass kept', 'fraction of $R$'),
                                            ('top10_IoU', '(g) Top-10 % IoU', 'IoU with $R$'))):
            ax = fig.add_subplot(gs[6, c])
            bp = ax.boxplot([rec3b[rec3b['lambda'] == l][key] for l in L], widths=.5, patch_artist=True, showfliers=False,
                            medianprops={'color': '#4A4A50', 'lw': .8}, whiskerprops={'lw': .6}, capprops={'lw': .6})
            for el in bp['boxes']:
                el.set(facecolor=CB, alpha=.75, edgecolor=CB, lw=.6)
            ax.set_xticklabels(['0.5', '1', '2']); ax.set_xlabel('$\\lambda$', labelpad=1); ax.set_ylabel(yl); ax.set_ylim(0, 1)
            ax.set_title(ttl, loc='left')
        ax = fig.add_subplot(gs[6, 2])
        f1 = rec3b[rec3b['lambda'] == .5]
        ax.hist(f1.F_trace_mean, bins=25, color=C2, alpha=.85)
        ax.axvline(f1.F_trace_mean.median(), color='#4A4A50', lw=.8)
        ax.set_xlabel('mean $F_{trace}$', labelpad=1); ax.set_ylabel('ECGs'); ax.set_title('(h) On-trace share', loc='left')
        save(fig, out, 'col_module3B')


# ------------------------------------------------------------------ faithfulness
def fig_faith692_column(curves, paired, out):
    sty = {'R': ('$R$', CB, 'o', '-'), 'Fbase': ('$F_{base}$', CNULL, 's', '-'),
           'Rtrace_0p5': ('$\\lambda$=0.5', '#F2B4CF', 'D', '-'), 'Rtrace_1p0': ('$\\lambda$=1', C2, 'D', '-'),
           'Rtrace_2p0': ('$\\lambda$=2', '#B9467C', 'D', '-'), 'RANDOM': ('random', '#555555', '', ':')}
    comps = ['Fbase - R', 'Fmu_matched30 - R', 'Rtrace_0p5 - R', 'Rtrace_1p0 - R', 'Rtrace_2p0 - R', 'RANDOM - R']
    labs = ['$F_{base}$', '$F_{\\mu}$', '$\\lambda$=0.5', '$\\lambda$=1', '$\\lambda$=2', 'random']
    with small_type():
        fig = plt.figure(figsize=(WIDTH, 5.0))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.05, 1.15], hspace=.78, wspace=.5)
        for k, b in enumerate(('1d', '2d')):
            ax = fig.add_subplot(gs[0, k])
            for m, (lab, col, mk, ls) in sty.items():
                g = curves[(curves.branch == b) & (curves.method == m)].sort_values('fraction')
                ax.plot(100 * g.fraction, g.probability_drop, ls, color=col, marker=mk, ms=2, lw=.9, label=lab)
            ax.set_xlabel('signal deleted (%)', labelpad=1); ax.set_ylabel('mean prob. drop'); ax.set_ylim(0, .62)
            ax.set_title(f'({"ab"[k]}) {b.upper()} CNN', loc='left')
        fig.axes[0].legend(ncol=2, frameon=False, fontsize=4.9, loc='upper left', handlelength=1.3, labelspacing=.2, columnspacing=.6)
        ax = fig.add_subplot(gs[1, :])
        A = paired[paired.class_name == 'ALL']
        for d, (b, col) in ((-.17, ('1d', C1)), (.17, ('2d', C2))):
            for y, cpn in enumerate(comps):
                r = A[(A.branch == b) & (A.comparison == cpn)].iloc[0]
                ax.errorbar(r.mean_delta_auc, y + d, xerr=[[r.mean_delta_auc - r.ci95_lower], [r.ci95_upper - r.mean_delta_auc]],
                            fmt='o', color=col, ms=2.6, capsize=1.2, lw=.8)
        ax.axvline(0, color=INK, lw=.6); ax.set_yticks(range(len(comps))); ax.set_yticklabels(labs)
        ax.invert_yaxis(); ax.set_xlabel('$\\Delta$AUC vs $R$ (mean, 95 % CI)', labelpad=1)
        ax.set_title('(c) Paired difference to $R$ (692 ECGs; $F_{\\mu}$: 551)', loc='left')
        ax.legend(handles=[Line2D([], [], marker='o', color=C1, lw=0, ms=3, label='1D'),
                           Line2D([], [], marker='o', color=C2, lw=0, ms=3, label='2D')],
                  frameon=False, loc='lower right')
        cm = LinearSegmentedColormap.from_list('dv', ['#B9467C', '#F7D7E6', '#FFFFFF', '#CFE3F5', '#2C6CA8'])
        for k, b in enumerate(('1d', '2d')):
            ax = fig.add_subplot(gs[2, k])
            M = np.array([[paired[(paired.branch == b) & (paired.class_name == c) & (paired.comparison == cpn)].mean_delta_auc.iloc[0]
                           for c in CLASSES] for cpn in comps])
            ax.imshow(M, cmap=cm, vmin=-.15, vmax=.15, aspect='auto')
            for i in range(M.shape[0]):
                for jj in range(M.shape[1]):
                    ax.text(jj, i, f'{100 * M[i, jj]:+.1f}', ha='center', va='center', fontsize=4.8, color=INK)
            ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, fontsize=5.2); ax.set_yticks(range(len(comps)))
            ax.set_yticklabels(labs, fontsize=5.2); ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f'({"de"[k]}) {b.upper()} by class (×10$^{{-2}}$)', loc='left')
        save(fig, out, 'col_module3_faith692')


# ------------------------------------------------------------------ Module 1 branch (min-max agreement)
M1_METRICS = [('mean_agreement', 'agreement $\\bar A$'), ('mean_disagreement', 'disagreement $\\bar D$'),
              ('spearman', 'Spearman'), ('dice_top15', 'Dice$_{15}$'), ('iou_top15', 'IoU$_{15}$'),
              ('weighted_jaccard', 'weighted Jaccard')]


def fig_module1_classwise_column(nd, out):
    """Class-wise distributions of the six min-max agreement metrics (1033 non-degenerate pairs)."""
    with small_type():
        fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 4.3))
        for ax, (key, lab), letter in zip(axes.ravel(), M1_METRICS, 'abcdef'):
            data = [nd[nd.class_name == c][key].to_numpy() for c in CLASSES]
            vp = ax.violinplot(data, positions=range(4), widths=.8, showextrema=False, showmedians=False)
            for body, c in zip(vp['bodies'], CLASSES):
                body.set_facecolor(CLS_COL[c]); body.set_edgecolor(CLS_COL[c]); body.set_alpha(.55); body.set_linewidth(.4)
            for x, d in enumerate(data):
                q1, med, q3 = np.percentile(d, [25, 50, 75])
                ax.plot([x, x], [q1, q3], color='#4A4A50', lw=1.4, solid_capstyle='butt')
                ax.plot([x - .18, x + .18], [med, med], color='white', lw=1.1)
                ax.plot(x, d.mean(), marker='D', ms=2.4, color=INK, lw=0)
            ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES); ax.set_title(f'({letter}) {lab}', loc='left')
            if key == 'spearman':
                ax.axhline(0, color=CGREY, lw=.5, ls=':')
        fig.legend(handles=[Line2D([], [], color='#4A4A50', lw=1.4, label='IQR'),
                            Line2D([], [], color='#4A4A50', marker='_', ms=6, lw=0, label='median (white)'),
                            Line2D([], [], color=INK, marker='D', ms=2.6, lw=0, label='mean')],
                   ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(.5, .015), fontsize=5.4)
        fig.tight_layout(h_pad=.6, w_pad=.8)
        save(fig, out, 'col_module1_classwise')


def fig_module1_sensitivity_column(shift_summary, proj_boot, out):
    """(a) mean metrics versus horizontal alignment shift; (b) mean metrics versus corridor mask size with 95 % CIs."""
    show = [('mean_agreement', 'agreement', CB), ('mean_disagreement', 'disagreement', C2),
            ('spearman', 'Spearman', C1), ('dice_top15', 'Dice$_{15}$', CNAVY)]
    with small_type():
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.75), gridspec_kw={'width_ratios': [1.35, 1]})
        ax = axes[0]
        for key, lab, col in show:
            ax.plot(shift_summary.index, shift_summary[key], marker='o', ms=2.2, lw=.9, color=col, label=lab)
        ax.axvline(0, color=CGREY, lw=.5, ls=':'); ax.set_xlabel('alignment shift (image px)', labelpad=1)
        ax.set_ylabel('mean over recordings'); ax.set_ylim(0, .48); ax.set_title('(a) Temporal shift', loc='left')
        ax.legend(frameon=False, fontsize=5.0, loc='upper right', bbox_to_anchor=(1, .84), handlelength=1.2, labelspacing=.2)
        ax = axes[1]
        for k, (key, lab, col) in enumerate(show):
            g = proj_boot[proj_boot.metric == key].sort_values('mask_size')
            x = np.arange(len(g)) + (k - 1.5) * .12
            ax.errorbar(x, g['mean'], yerr=[g['mean'] - g.ci_lo, g.ci_hi - g['mean']], fmt='o', ms=2.4, color=col,
                        lw=.8, capsize=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels(['6', '13', '24']); ax.set_xlabel('mask size (px)', labelpad=1)
        ax.set_ylim(0, .48); ax.set_title('(b) Projection mask', loc='left')
        fig.tight_layout(w_pad=.8)
        save(fig, out, 'col_module1_sensitivity')


# ------------------------------------------------------------------ Module 1 case
def fig_module1_case_column(z, i, mask, out):
    """ECG 209 drawn from the archived maps base1/base2 (unperturbed Grad-CAM, aligned 1D, on-trace 2D)."""
    global MASK
    MASK = mask
    from dan_figures import mass
    v = z['valid_600'][i].astype(bool)
    g1, g2 = z['base1'][i][v], z['base2'][i][v]
    p, q = mass(g1), mass(g2)
    sh = np.minimum(p, q)
    r = (p - q) / (p + q + 1e-12)
    sal = topk(p, .15) | topk(q, .15)
    cat = np.zeros(p.size)
    cat[sal & (r > 1 / 3)], cat[sal & (r < -1 / 3)], cat[sal & (np.abs(r) <= 1 / 3)] = 1, 2, 3
    poly, W = polyline(mask), mask.shape[1]
    with small_type():
        fig = plt.figure(figsize=(WIDTH, 4.35))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, .8], hspace=.62)
        a = fig.add_subplot(gs[0]); wave(a, poly, W, g1 / g1.max(), CM1, 1.0, lw=(.35, 1.9)); _label(a, '1D', C1)
        a.set_title('(a) 1D Grad-CAM intensity on the waveform', loc='left'); _cbar(fig, a, CM1, 1, '$g_1$')
        b = fig.add_subplot(gs[1]); wave(b, poly, W, g2 / g2.max(), CM2, 1.0, lw=(.35, 1.9)); _label(b, '2D', C2)
        b.set_title('(b) 2D intensity received by the waveform ($\\pm6$ px)', loc='left'); _cbar(fig, b, CM2, 1, '$g_2$')
        c = fig.add_subplot(gs[2]); wave(c, poly, W, sh / sh.max(), CMB, 1.0, lw=(.35, 2.2)); _label(c, 'shared', CB)
        c.set_title(f'(c) shared intensity $\\min(p_u,q_u)$; $O={sh.sum():.3f}$', loc='left'); _cbar(fig, c, CMB, 1, 'relative')
        d = fig.add_subplot(gs[3])
        wave(d, poly, W, cat, None, None, cat={0: (CGREY, .4), 3: (CB, 1.7), 2: (C2, 1.7), 1: (C1, 1.7)}); _label(d, 'dir.', INK)
        d.set_title('(d) direction', loc='left'); _spacer(fig, d)
        d.legend(handles=[Line2D([], [], color=C1, lw=1.7, label='1D-dominant'), Line2D([], [], color=C2, lw=1.7, label='2D-dominant'),
                          Line2D([], [], color=CB, lw=1.7, label='balanced')],
                 ncol=3, frameon=False, loc='lower right', bbox_to_anchor=(1.0, .93), handlelength=1.0, columnspacing=.7,
                 handletextpad=.3, fontsize=5.4)
        e = fig.add_subplot(gs[4]); t = np.arange(p.size) * 10 / p.size
        e.fill_between(t, 0, sh, color=CB, alpha=.35, lw=0, label='shared')
        e.plot(t, p, color=C1, lw=.6, label='1D mass $p_u$'); e.plot(t, q, color=C2, lw=.8, label='2D mass $q_u$')
        e.set_xlim(0, 10); e.set_ylim(0, max(p.max(), q.max()) * 1.45)
        e.spines[['top', 'right']].set_visible(False)
        e.set_xlabel('time (s)', labelpad=1); e.set_ylabel('mass', labelpad=1)
        e.legend(ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(0, 1.08), handlelength=1.2, columnspacing=.9)
        e.set_title('(e) attribution mass on the common time axis', loc='left', pad=6); _spacer(fig, e)
        save(fig, out, 'col_module1_case209')
