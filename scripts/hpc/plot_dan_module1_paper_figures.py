#!/usr/bin/env python3
"""DAN-MMGradCAM Module 1 | publication-style figures from existing HPC results.

NO model training; NO CAM regeneration; NO invented or synthetic scientific results.
Needs numpy, pandas, matplotlib, PIL. scipy is not required.

Example (run inside MM_GradCAM_HPC):
 python plot_dan_module1_paper_figures.py \
   --module results/dan_module1_full_20261015_191138 \
   --sensitivity results/dan_module1_projection_sensitivity_full_727_v1 \
   --out results/dan_module1_visuals_pastel

Produces numbered PNG (300 dpi), vector PDF, CSV summary and README.
Re-running is safe: it replaces only its own output figures in --out.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import textwrap
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyBboxPatch

# --- Elegant, print-legible soft tones; text uses darker ink variants. ---
BG = '#FFFCFA'
PANEL = '#FFF9F7'
INK = '#303242'
MUTED = '#747584'
GRID = '#E9E4E3'
PINK = '#CE6D96'
PINK_SOFT = '#EBC0D1'
BLUE = '#4B91B2'
BLUE_SOFT = '#B9DAE8'
BROWN = '#A77E60'
BROWN_SOFT = '#DDC3A9'
ROSE = '#A37888'
ROSE_SOFT = '#E6C9CF'
CLASSES = ['AFib', 'GSVT', 'SB', 'SR']
CLASS_COLORS = {'AFib': PINK, 'GSVT': BLUE, 'SB': BROWN, 'SR': ROSE}
CLASS_SOFT = {'AFib': PINK_SOFT, 'GSVT': BLUE_SOFT, 'SB': BROWN_SOFT, 'SR': ROSE_SOFT}
METHODS = ['waveform_mask_6px', 'waveform_mask_24px', 'full_height_mean']
METHOD_LABELS = {'waveform_mask_6px': 'Waveform mask · 6 px',
                 'waveform_mask_24px': 'Waveform mask · 24 px',
                 'full_height_mean': 'Full-height mean'}
METHOD_COLORS = {'waveform_mask_6px': PINK, 'waveform_mask_24px': BLUE,
                 'full_height_mean': BROWN}
M_LABELS = {'spearman':'Spearman correlation',
            'dice_top_fraction':'Top-15% Dice',
            'weighted_jaccard':'Weighted Jaccard',
            'mean_disagreement_D':'Mean disagreement',
            'mean_agreement_A':'Mean shared saliency',
            'pearson':'Pearson correlation'}
KEY_METRICS = ['spearman', 'dice_top_fraction', 'weighted_jaccard']
FOOTER = ('EXPLORATORY · Both branches correct + strict temporal QC · '
          'Provisional 2D-to-time calibration · Not a faithfulness claim')
SEED = 20261015


def rc_style():
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#D8D4D4',
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.spines.right': False, 'axes.spines.top': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': BG, 'figure.facecolor': BG,
        'axes.facecolor': 'white',
    })


def create_canvas(title, subtitle, figsize=(16,9), foot=FOOTER):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    fig.text(.055, .953, 'DAN  /  MMGradCAM', fontsize=10.6, weight='bold', color=PINK)
    fig.text(.055, .909, title, fontsize=22.5, weight='bold', color=INK)
    fig.text(.055, .876, subtitle, fontsize=10.8, color=MUTED)
    fig.add_artist(plt.Line2D([.055,.945],[.855,.855],transform=fig.transFigure,
                              lw=1, color=GRID))
    fig.text(.055, .032, foot, fontsize=8.5, color=MUTED)
    return fig


def axis_clean(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, pad=8)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=.85, alpha=.78)
        ax.set_axisbelow(True)


def panel(ax, title):
    ax.set_title(title, loc='left', fontsize=12, color=INK, pad=17, weight='bold')


def save(fig, out, name, made, dpi=300):
    png = out / (name + '.png')
    pdf = out / (name + '.pdf')
    fig.savefig(png, dpi=dpi, bbox_inches='tight', pad_inches=.24, facecolor=BG)
    fig.savefig(pdf, bbox_inches='tight', pad_inches=.24, facecolor=BG)
    plt.close(fig)
    made.extend([png.name, pdf.name])
    print('CREATED:', png, flush=True)


def median_ci(x, reps=1000, seed=SEED):
    """Percentile bootstrap of within-record median (resampling ECGs)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if len(x) == 1:
        return med, med, med
    rng = np.random.default_rng(int(seed))
    # Bound memory; use small chunks even for very large cohorts.
    chunks = []
    for start in range(0, reps, 150):
        b = min(150, reps-start)
        indices = rng.integers(0, len(x), size=(b, len(x)))
        chunks.append(np.median(x[indices], axis=1))
    boot = np.concatenate(chunks)
    lo, hi = np.percentile(boot, [2.5,97.5])
    return med, float(lo), float(hi)


def verify_and_read(module_dir, sens_dir):
    paths = [module_dir/'module1_primary_exploratory.csv',
             module_dir/'module1_all_records.csv',
             module_dir/'summary.json',
             sens_dir/'projection_metrics_long.csv',
             sens_dir/'summary.json']
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Required existing HPC result not found: ' + str(path))
    primary = pd.read_csv(paths[0]); all_df = pd.read_csv(paths[1]); proj = pd.read_csv(paths[3])
    ms = json.loads(paths[2].read_text()); ss = json.loads(paths[4].read_text())
    required_primary = ['patient_id', 'cohort_index', 'true_class', 'processing_note'] + KEY_METRICS + ['mean_disagreement_D', 'mean_agreement_A', 'pearson']
    required_proj = ['patient_id','cohort_index','true_class','method'] + KEY_METRICS + ['mean_disagreement_D']
    for name,df,req in [('Primary',primary,required_primary),('Projection',proj,required_proj)]:
        missing = [c for c in req if c not in df]
        if missing:
            raise ValueError(name + ' input missing columns: ' + str(missing))
    if len(primary) != int(ms['primary_measured']):
        raise ValueError('Module 1 count disagreement with summary')
    if len(primary) != int(ss['pilot_ecg_count']):
        raise ValueError('Projection sensitivity subset does not match Module 1 measured count')
    if primary.duplicated(['patient_id','cohort_index']).any():
        raise ValueError('Duplicate identities in primary CSV')
    if not (primary['processing_note']=='EXPLORATORY_MEASURED').all():
        raise ValueError('Primary file contains unmeasured rows')
    if not set(primary['true_class']).issubset(CLASSES):
        raise ValueError('Unexpected rhythm class')
    if len(all_df) != int(ms['pairs_evaluated_this_run']):
        raise ValueError('All-records CSV length differs from Module 1 summary')
    if len(proj) != len(primary)*len(METHODS):
        raise ValueError('Projection sensitivity CSV has unexpected count')
    if set(proj['method']) != set(METHODS):
        raise ValueError('Projection method labels differ from expected')
    if proj.duplicated(['patient_id','cohort_index','method']).any():
        raise ValueError('Duplicate projection identities')
    a = set(zip(primary['patient_id'].astype(str), primary['cohort_index'].astype(int)))
    b = set(zip(proj['patient_id'].astype(str), proj['cohort_index'].astype(int)))
    if a != b:
        raise ValueError('Projection and primary patient cohorts differ')
    compare = proj[proj.method==METHODS[0]][['patient_id','cohort_index']+KEY_METRICS].merge(
        primary[['patient_id','cohort_index']+KEY_METRICS], on=['patient_id','cohort_index'], suffixes=('_p','_main'), validate='one_to_one')
    for metric in KEY_METRICS:
        err = np.abs(compare[metric+'_p']-compare[metric+'_main']).max()
        if err > 1.e-3:
            raise ValueError('Baseline metric reproduction mismatch for ' + metric + ': ' + str(err))
    return primary, all_df, proj, ms, ss


def fig01_overview(primary, all_df, ms, out, made):
    n=len(primary); total=len(all_df)
    fig=create_canvas('01  |  Cohort & explanation agreement',
                      f'{total:,} held-out ECG records   ·   {n:,} paired attributions in the exploratory primary cohort')
    # decorative numbers: true observed values
    kpis=[('PRIMARY COHORT', f'{n:,}', PINK),
          ('MEDIAN SPEARMAN', f'{primary.spearman.median():.3f}', BLUE),
          ('MEDIAN TOP-15% DICE',f'{primary.dice_top_fraction.median():.3f}', BROWN)]
    for i,(lbl,val,color) in enumerate(kpis):
        x=.055+i*.303
        rect=FancyBboxPatch((x,.710),.275,.113,boxstyle='round,pad=0.007,rounding_size=0.012',
                            facecolor='white',edgecolor=GRID,lw=.9,transform=fig.transFigure)
        fig.add_artist(rect)
        fig.text(x+.020,.779,lbl,color=MUTED,fontsize=9,weight='bold')
        fig.text(x+.020,.731,val,color=color,fontsize=24,weight='bold')
    ax1=fig.add_axes([.075,.185,.39,.430])
    ax2=fig.add_axes([.580,.185,.345,.430])
    panel(ax1,'Composition of all held-out records')
    flow=ms.get('control_flow_counts',{})
    groups=[('Primary / eligible',int(flow.get('PRIMARY_EXPLORATORY',0)),PINK),
            ('Both correct / QC excluded',int(flow.get('BOTH_CORRECT_BUT_QC_NOT_STRICT',0)),BROWN),
            ('Different targets',int(flow.get('DIFFERENT_TARGET_NO_COMPARISON',0)),BLUE),
            ('Same target / not both correct',int(flow.get('SAME_TARGET_BUT_NOT_BOTH_CORRECT',0)),ROSE)]
    assert sum(v for _,v,_ in groups)==total, 'Cohort counts do not add up'
    ax1.set_xlim(0,total*1.14)
    for i,(label,num,color) in enumerate(groups):
        y=3-i
        ax1.barh(y,num,height=.52,color=color,alpha=.80)
        ax1.text(8,y,f'{num:,}',va='center',color='white' if num>100 else INK,fontsize=10,weight='bold')
        ax1.text(total*1.125,y,f'{100*num/total:.1f}%',va='center',ha='right',color=MUTED,fontsize=9.5)
    ax1.set_yticks([3,2,1,0],[x[0] for x in groups],fontsize=9.7)
    ax1.set_xticks([])
    ax1.spines[:].set_visible(False)
    ax1.grid(False)
    panel(ax2,'Primary cohort by rhythm')
    all_counts = all_df.true_class.value_counts()
    counts=primary.true_class.value_counts()
    for i,c in enumerate(CLASSES):
        eligible=int(counts.get(c,0)); total_c=int(all_counts.get(c,0)); excluded=total_c-eligible
        ax2.bar(i,eligible,color=CLASS_COLORS[c],width=.61,zorder=3)
        ax2.bar(i,excluded,bottom=eligible,color=CLASS_SOFT[c],width=.61,zorder=3)
        ax2.text(i,eligible/2,str(eligible),ha='center',va='center',fontsize=11,weight='bold',color='white')
        ax2.text(i,total_c+8,f'{100*eligible/total_c:.0f}%',ha='center',fontsize=9,color=MUTED)
    ax2.set_xticks(range(4),CLASSES)
    ax2.set_ylim(0,max(all_counts)*1.18)
    ax2.set_ylabel('ECG recordings')
    axis_clean(ax2)
    fig.text(.57,.108,'Solid = included   /   Pale = not eligible',fontsize=9.5,color=MUTED)
    save(fig,out,'01_overview',made)


def fig02_distributions(primary,out,made):
    n=len(primary)
    fig=create_canvas('02  |  Similarity distributions by class',
                      f'Individual ECG values and class medians   ·   n = {n:,}   ·   no inferential class ranking',
                      figsize=(16,10))
    gs=fig.add_gridspec(2,2,left=.080,right=.94,top=.805,bottom=.125,wspace=.22,hspace=.42)
    metrics=['spearman','dice_top_fraction','weighted_jaccard','mean_disagreement_D']
    rng=np.random.default_rng(SEED)
    for ix,metric in enumerate(metrics):
        ax=fig.add_subplot(gs[ix//2,ix%2])
        panel(ax,M_LABELS[metric])
        for j,c in enumerate(CLASSES):
            values=primary.loc[primary.true_class==c,metric].dropna().to_numpy(dtype=float)
            if len(values)<2:
                continue
            v=ax.violinplot([values],positions=[j],widths=.70,showmeans=False,showmedians=False,showextrema=False)
            for body in v['bodies']:
                body.set_facecolor(CLASS_SOFT[c]);body.set_edgecolor(CLASS_COLORS[c]);body.set_linewidth(.9);body.set_alpha(.68)
            jitter=rng.normal(0,.065,size=len(values))
            ax.scatter(np.full(len(values),j)+jitter,values,s=8,alpha=.26,color=CLASS_COLORS[c],rasterized=True,zorder=3)
            q1,med,q3=np.percentile(values,[25,50,75])
            ax.plot([j,j],[q1,q3],color=CLASS_COLORS[c],lw=5,solid_capstyle='round',zorder=4)
            ax.scatter([j],[med],s=49,color='white',edgecolor=INK,lw=1.0,zorder=5)
        ax.set_xticks(range(4),[f'{c}  (n={sum(primary.true_class==c)})' for c in CLASSES],fontsize=9)
        if metric in ('spearman',):
            ax.axhline(0,color=INK,lw=.9,ls='--',alpha=.33)
        if metric == 'dice_top_fraction':
            ax.axhline(.15,color=BROWN,lw=1.15,ls=':',alpha=.6)
            ax.text(3.45,.151,'independent uniform\ntop-15% reference',ha='right',va='bottom',fontsize=8,color=BROWN)
        ax.set_ylabel('Per-ECG score',fontsize=9.7)
        axis_clean(ax)
    save(fig,out,'02_distribution_by_class',made)


def fig03_class_ci(primary,out,made):
    fig=create_canvas('03  |  Class-wise median attribution similarity',
                      'Dots = class median  ·  horizontal intervals = 95% percentile bootstrap CI (ECG resampling)',figsize=(16,9.6))
    gs=fig.add_gridspec(1,3,left=.078,right=.95,top=.79,bottom=.16,wspace=.23)
    for k,metric in enumerate(KEY_METRICS):
        ax=fig.add_subplot(gs[k]);panel(ax,M_LABELS[metric])
        vals=[]
        for j,c in enumerate(CLASSES):
            x=primary.loc[primary.true_class==c,metric]
            med,lo,hi=median_ci(x,seed=SEED+37*j+k)
            vals.append((med,lo,hi))
            ax.errorbar(med,3-j,xerr=[[med-lo],[hi-med]],fmt='o',color=CLASS_COLORS[c],
                        markerfacecolor=CLASS_SOFT[c],markeredgecolor=CLASS_COLORS[c],
                        markersize=10,elinewidth=2,capsize=4,zorder=3)
            ax.text(med,3-j+.22,f'{med:.3f}',ha='center',fontsize=9.2,color=INK)
        lo_min=min(x[1] for x in vals); hi_max=max(x[2] for x in vals)
        span=max(.035,hi_max-lo_min)
        ax.set_xlim(lo_min-.38*span,hi_max+.38*span)
        ax.set_ylim(-.65,3.70)
        ax.set_yticks([3,2,1,0],CLASSES,fontsize=10.5)
        if metric=='spearman':ax.axvline(0,ls='--',lw=1,color=GRID)
        ax.set_xlabel('Median score (95% bootstrap CI)',fontsize=9)
        axis_clean(ax,'x')
    save(fig,out,'03_class_medians_ci',made)


def fig04_projection(proj, out,made):
    fig=create_canvas('04  |  Projection sensitivity: absolute scores',
                      f'Identical {proj.patient_id.nunique():,} ECGs per method   ·   each projection normalized separately over common valid columns',figsize=(16,9.5))
    gs=fig.add_gridspec(1,3,left=.070,right=.95,top=.78,bottom=.18,wspace=.20)
    for k,metric in enumerate(KEY_METRICS):
        ax=fig.add_subplot(gs[k]);panel(ax,M_LABELS[metric])
        medians=[]
        for j,method in enumerate(METHODS):
            x=proj.loc[proj.method==method,metric].to_numpy(float)
            med,lo,hi=median_ci(x,seed=SEED+k*37+j)
            medians.append(med)
            ax.errorbar(j,med,yerr=[[med-lo],[hi-med]],fmt='o',markersize=11,
                        markerfacecolor=METHOD_COLORS[method],markeredgecolor='white',markeredgewidth=1.5,
                        color=METHOD_COLORS[method],elinewidth=1.8,capsize=4,zorder=4)
            ax.text(j,hi+.009,f'{med:.3f}',ha='center',fontsize=10,color=INK,weight='bold')
        ax.plot(range(3),medians,color='#CFCCD0',lw=2,zorder=2)
        ax.set_xticks(range(3),['6 px\nmask','24 px\nmask','Full\nheight'],fontsize=10)
        ylow=min(medians)-.04; yhigh=max(medians)+.065
        ax.set_ylim(min(0,ylow),yhigh)
        ax.set_ylabel('Median score (95% bootstrap CI)',fontsize=9)
        axis_clean(ax)
    save(fig,out,'04_projection_median_scores',made)


def fig05_paired_changes(proj,out,made):
    fig=create_canvas('05  |  Paired effect of changing projection',
                      'Each point is the median within-ECG change relative to 6 px mask   ·   95% paired bootstrap CI',
                      figsize=(16,9.6))
    gs=fig.add_gridspec(1,3,left=.078,right=.952,top=.79,bottom=.18,wspace=.22)
    for k,metric in enumerate(KEY_METRICS):
        ax=fig.add_subplot(gs[k]);panel(ax,M_LABELS[metric])
        mat=proj.pivot(index=['patient_id','cohort_index'],columns='method',values=metric)
        if mat.isna().any().any() or mat.shape[1]!=3:
            raise ValueError('Incomplete paired projection table for '+metric)
        delta_24=(mat[METHODS[1]]-mat[METHODS[0]]).to_numpy(float)
        delta_full=(mat[METHODS[2]]-mat[METHODS[0]]).to_numpy(float)
        allv=np.concatenate([delta_24,delta_full])
        ax.axvline(0,color=INK,lw=1.35,ls='--',alpha=.4)
        for pos,data,method in [(1,delta_24,METHODS[1]),(0,delta_full,METHODS[2])]:
            med,lo,hi=median_ci(data,seed=SEED+123+k+pos)
            col=METHOD_COLORS[method]
            ax.errorbar(med,pos,xerr=[[med-lo],[hi-med]],fmt='o',color=col,markersize=12,
                        markeredgecolor='white',markeredgewidth=1.3,elinewidth=2.7,capsize=5,zorder=4)
            ax.text(med,pos+.23,f'{med:+.4f}',ha='center',fontsize=10,color=INK,weight='bold')
        ax.set_yticks([1,0],['24 px − 6 px','Full height − 6 px'])
        ax.set_ylim(-.6,1.7)
        vscale=max(.03,abs(np.nanpercentile(allv,5)),abs(np.nanpercentile(allv,95)))
        ax.set_xlim(-vscale*1.25,vscale*1.25)
        ax.set_xlabel('Median paired difference',fontsize=9.5)
        axis_clean(ax,'x')
    save(fig,out,'05_projection_paired_changes',made)


def fig06_scatter(primary,out,made):
    fig=create_canvas('06  |  Record-level explanation similarity',
                      'Each dot is one paired ECG   ·   correlation and region overlap are different properties',figsize=(14,9.5))
    ax=fig.add_axes([.10,.16,.79,.62])
    for c in CLASSES:
        d=primary[primary.true_class==c]
        ax.scatter(d.spearman,d.dice_top_fraction,s=32,color=CLASS_COLORS[c],alpha=.46,
                   edgecolors='white',linewidths=.28,label=f'{c}   n={len(d)}',rasterized=True)
        ax.scatter([d.spearman.median()],[d.dice_top_fraction.median()],marker='D',s=107,
                   color=CLASS_COLORS[c],edgecolors=INK,linewidths=.6,zorder=5)
    ax.axvline(0,lw=1,color=INK,alpha=.25,ls='--')
    ax.axhline(.15,lw=1.15,color=BROWN,alpha=.6,ls=':',label='Uniform independent top-15% reference ≈ 0.15')
    ax.set_xlabel('Spearman correlation (1D CAM vs projected 2D CAM)',fontsize=11,labelpad=13)
    ax.set_ylabel('Top-15% Dice overlap',fontsize=11,labelpad=12)
    axis_clean(ax)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False,fontsize=9.4)
    save(fig,out,'06_spearman_vs_dice_scatter',made)


def fig07_representative(primary,all_df,module_dir,root,out,made):
    paths={x.name:module_dir/x.name for x in [Path('g1_normalized_600.npy'),
            Path('g2_projected_normalized_600.npy'),Path('agreement_A_600.npy'),Path('disagreement_D_600.npy')]}
    if not all(p.is_file() for p in paths.values()):
        print('SKIP fig07: Missing arrays in Module 1 directory',flush=True)
        return
    cams={k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in paths.items()}
    n_all=len(all_df)
    if not all(z.shape==(n_all,600) for z in cams.values()):
        raise ValueError('Expected Module 1 CAM arrays with shape (total_records,600)')
    all_lookup={ (str(r.patient_id),int(r.cohort_index)):i for i,r in enumerate(all_df.itertuples(index=False))}
    fig=create_canvas('07  |  Representative paired attribution traces',
                      'One near-median Top-15% Dice example per rhythm   ·   same patient identity across branches',
                      figsize=(18,13.0))
    gs=fig.add_gridspec(4,3,left=.065,right=.958,top=.800,bottom=.112,
                        wspace=.18,hspace=.39,width_ratios=[1.40,2.00,2.00])
    def pcam(k):return cams[k]
    x_orig=(np.arange(600)+.5)*1080/600-.5
    valid=(x_orig>=58)&(x_orig<=1021)
    t=np.linspace(0,10,600)
    for ci,c in enumerate(CLASSES):
        d=primary[primary.true_class==c].copy()
        target=d.dice_top_fraction.median()
        r=d.loc[(d.dice_top_fraction-target).abs().idxmin()]
        key=(str(r.patient_id),int(r.cohort_index))
        if key not in all_lookup:
            raise ValueError('CAM array row lookup failed: '+str(key))
        j=all_lookup[key]
        axim=fig.add_subplot(gs[ci,0])
        raw=Path(str(r.image_path_2d))
        raw=raw if raw.is_absolute() else root/raw
        if raw.exists():
            with Image.open(raw) as im:
                arr=np.asarray(im.convert('RGB'))
            axim.imshow(arr)
        else:
            axim.text(.5,.5,'ECG image unavailable',ha='center',va='center',transform=axim.transAxes,color=MUTED)
        axim.set_title(f'{c}  ·  sample {int(r.cohort_index)}',loc='left',color=CLASS_COLORS[c],weight='bold',fontsize=11,pad=8)
        axim.axis('off')
        axg=fig.add_subplot(gs[ci,1]); axa=fig.add_subplot(gs[ci,2])
        g1=np.asarray(pcam('g1_normalized_600.npy')[j])
        g2=np.asarray(pcam('g2_projected_normalized_600.npy')[j])
        ag=np.asarray(pcam('agreement_A_600.npy')[j]); diff=np.asarray(pcam('disagreement_D_600.npy')[j])
        axg.plot(t[valid],g1[valid],color=PINK,lw=1.25,label='1D attribution')
        axg.plot(t[valid],g2[valid],color=BLUE,lw=1.4,label='2D projected')
        axa.plot(t[valid],ag[valid],color=BROWN,lw=1.4,label='Shared saliency')
        axa.plot(t[valid],diff[valid],color=ROSE,lw=1.4,label='Disagreement')
        for ax in (axg,axa):
            ax.set_ylim(-.045,1.06);ax.set_xlim(t[valid][0],t[valid][-1]);ax.set_yticks([0,.5,1])
            axis_clean(ax)
            if ci!=3: ax.set_xticklabels([])
            else:ax.set_xlabel('Approximate time (s) · provisional')
        if ci==0:
            axg.set_title('Branch-level saliency',loc='left',fontsize=11,pad=8,weight='bold')
            axa.set_title('Shared saliency vs disagreement',loc='left',fontsize=11,pad=8,weight='bold')
            axg.legend(loc='upper right',frameon=False,fontsize=8,ncol=2)
            axa.legend(loc='upper right',frameon=False,fontsize=8,ncol=2)
        axg.text(.015,.08,f'ρ={r.spearman:.3f}     Dice={r.dice_top_fraction:.3f}',
                 transform=axg.transAxes,fontsize=8.7,color=INK,va='bottom',
                 bbox=dict(facecolor=BG,edgecolor='none',alpha=.86,pad=3))
    save(fig,out,'07_representative_ecg_traces',made,dpi=260)


def save_summary(primary, proj, out, made):
    metrics=['mean_agreement_A','mean_disagreement_D','spearman','pearson','dice_top_fraction','weighted_jaccard']
    all_summary=primary[metrics].agg(['count','mean','median','std']).T.reset_index().rename(columns={'index':'metric'})
    all_summary.to_csv(out/'module1_overall_metrics.csv',index=False)
    cls=primary.groupby('true_class')[metrics].agg(['count','mean','median','std']).round(6)
    cls.to_csv(out/'module1_class_metrics.csv')
    pm=proj.groupby('method')[metrics].agg(['count','mean','median','std']).round(6)
    pm.to_csv(out/'module1_projection_metrics.csv')
    made.extend(['module1_overall_metrics.csv','module1_class_metrics.csv','module1_projection_metrics.csv'])


def main():
    p=argparse.ArgumentParser(description='Paper-ready pastel pink-blue-brown visuals for DAN-MMGradCAM Module 1')
    p.add_argument('--root',default='.',help='HPC project root (default: current directory)')
    p.add_argument('--module',default='results/dan_module1_full_20261015_191138',help='Finished Module 1 directory')
    p.add_argument('--sensitivity',default='results/dan_module1_projection_sensitivity_full_727_v1',help='Finished sensitivity directory')
    p.add_argument('--out',default='results/dan_module1_visuals_pastel',help='Visual output directory; safe to rerun')
    p.add_argument('--skip-examples',action='store_true',help='Skip figure 07 with original ECG and CAM traces')
    args=p.parse_args()
    root=Path(args.root).expanduser().resolve()
    def resolve(s):
        x=Path(s).expanduser(); return x if x.is_absolute() else root/x
    module_dir=resolve(args.module);sens_dir=resolve(args.sensitivity);out=resolve(args.out)
    rc_style()
    primary,all_df,proj,ms,ss=verify_and_read(module_dir,sens_dir)
    out.mkdir(parents=True,exist_ok=True)
    made=[]
    print('VERIFIED:',len(all_df),'all ECG; primary:',len(primary),'; projection rows:',len(proj),flush=True)
    fig01_overview(primary,all_df,ms,out,made)
    fig02_distributions(primary,out,made)
    fig03_class_ci(primary,out,made)
    fig04_projection(proj,out,made)
    fig05_paired_changes(proj,out,made)
    fig06_scatter(primary,out,made)
    if not args.skip_examples:
        fig07_representative(primary,all_df,module_dir,root,out,made)
    save_summary(primary,proj,out,made)
    note=(
      'DAN-MMGradCAM Module 1 — Publication-style visualization exports\n'
      'Source: existing HPC Module 1 and projection sensitivity metrics.\n'
      f'All test ECG = {len(all_df)}; exploratory eligible = {len(primary)}.\n'
      'Numerical agreement is exploratory pending independent absolute image-time calibration.\n'
      '2D Grad-CAM original grid is 9x18, interpolated to 300x600.\n'
      'Selected cohort includes only cases correctly classified by both branches with strict waveform QC.\n'
      'Bootstrap CIs use resampling of ECG records; no correction for patient clustering, if any.\n'
      'The dashed Dice ~0.15 line illustrates independent uniform position selection, not a structure-preserving null test.\n'
      'Per-record paired changes are not the same as differences between two overall medians.\n'
      'Repeated execution overwrites files in this visualization folder only; existing Module 1 data untouched.\n'
      'Generated filenames:\n'+'\n'.join('  - '+f for f in made)+'\n')
    (out/'READ_ME_FIRST.txt').write_text(note,encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'status':'EXPLORATORY_ONLY',
      'source_module':str(module_dir),'source_projection':str(sens_dir),
      'n_total':len(all_df),'n_primary':len(primary),'files':made},indent=2),encoding='utf-8')
    print('\nSUCCESS: publication-style figures generated at',out.resolve(),flush=True)
    print('PNG, vector PDF, and metric CSV:',len(made),'files. No training performed.',flush=True)

if __name__=='__main__':
    main()
