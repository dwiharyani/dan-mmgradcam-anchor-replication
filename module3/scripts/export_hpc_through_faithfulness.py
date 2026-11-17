#!/usr/bin/env python3
"""Read-only snapshot of actual HPC code/results through Module3 paired deletion.

Run via Slurm after summarize_692_exploratory.py, not on the shared login node.
Exclude raw ECG, 2D image datasets, venvs, model checkpoints, passwords.
"""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from datetime import datetime,timezone

HOME=Path.home()
ROOT=HOME/'mm_gradcam'
M3=ROOT/'module3_aligned'
SOURCES=ROOT/'module3_sources'
EXPORT=M3/'exports'/'DAN_MMGradCAM_Sampai_Faithfulness_HPC.zip'
SUMMARY=M3/'outputs/module3_faithfulness_primary_qc693_v1/summary_692_exploratory'
REPORT=M3/'DAN_MMGradCAM_Sampai_Faithfulness_Laporan.pdf'
SKIP_PARTS={'.git','.venv','__pycache__','.mypy_cache','.pytest_cache', 'data','models','results'}
CODE_EXT={'.py','.sh','.sbatch','.md','.txt','.yaml','.yml','.json','.toml','.tex','.bib'}


def git_status(repo):
    if not repo.is_dir():return {'unavailable':'repository directory not found'}
    try:
        head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        branch=subprocess.check_output(['git','-C',str(repo),'rev-parse','--abbrev-ref','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip())
        return dict(head=head,branch=branch,dirty=dirty)
    except Exception as exc:return {'unavailable':str(exc)}


def main():
    raw=M3/'outputs/module3_faithfulness_primary_qc693_v1/raw'
    csvs=list(raw.glob('faith_????.csv'))
    audits=list(raw.glob('faith_????_audit.json'))
    if len(csvs)!=692 or len(audits)!=692:
        raise SystemExit(f'STOP: Faithfulness expected 692 CSV and 692 audit JSON; found {len(csvs)} and {len(audits)}')
    if not (SUMMARY/'paired_auc_692_exploratory.csv').is_file():
        raise SystemExit('STOP: exploratory paired summary missing; first run summarize_692_exploratory.py')
    if not REPORT.is_file():
        raise SystemExit(f'STOP: report PDF missing: {REPORT}; upload it first')
    primary=M3/'outputs/module3b_primary_qc693_v1/module3b_maps.npz'
    sensitivity=M3/'outputs/module3b_sensitivity_694_v1/module3b_maps.npz'
    m3a=M3/'outputs/module3a_v1/module3a_maps.npz'
    for p in [primary,sensitivity,m3a]:
        if not p.is_file():raise SystemExit(f'STOP: mandatory map missing: {p}')
    import numpy as np
    with np.load(primary,allow_pickle=False) as z:
        if len(z['cohort_index'])!=693:raise SystemExit('Wrong M3B strict count')
    with np.load(sensitivity,allow_pickle=False) as z:
        if len(z['cohort_index'])!=694:raise SystemExit('Wrong M3B sensitivity count')
    with np.load(m3a,allow_pickle=False) as z:
        if len(z['cohort_index'])!=694:raise SystemExit('Wrong M3A count')

    file_index={}  # zip path -> source Path
    def add_file(path, arc):
        path=Path(path)
        if not path.is_file():return
        arc=str(arc).replace('\\','/')
        if arc in file_index and file_index[arc]!=path:
            raise RuntimeError(f'Archive path collision: {arc}')
        if path.stat().st_size > 300*1024*1024:
            raise SystemExit(f'STOP: unexpected very large file (>300MB): {path}')
        file_index[arc]=path

    def add_dir(path,arc,filter_code=False):
        if not path.is_dir():return
        for file in sorted(path.rglob('*')):
            if not file.is_file() or file.is_symlink():continue
            rel=file.relative_to(path)
            if any(s in SKIP_PARTS or s.startswith('.') for s in rel.parts):continue
            if filter_code and file.suffix.lower() not in CODE_EXT:continue
            add_file(file,Path(arc)/rel)

    # Actual scripts from the HPC run, not a fabricated replacement.
    add_dir(M3/'scripts','live_code/module3/scripts',filter_code=True)
    add_dir(SOURCES/'module2_review/scripts','live_code/module2/scripts',filter_code=True)
    # Capture tracked source-code snapshots of the user's Module3 checkout.
    repo=SOURCES/'module3_git'
    if (repo/'.git').exists():
        try:
            files=subprocess.check_output(['git','-C',str(repo),'ls-files','-z']).split(b'\0')
            for rawname in files:
                if not rawname:continue
                rel=Path(os.fsdecode(rawname))
                if any(s in SKIP_PARTS or s.startswith('.') for s in rel.parts):continue
                if rel.suffix.lower() not in CODE_EXT:continue
                p=repo/rel
                if p.is_file() and not p.is_symlink():add_file(p,Path('live_code/module3_git_tracked')/rel)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print('WARNING: git unavailable; exporting source files directly:', exc)
            add_dir(repo, 'live_code/module3_git_snapshot', filter_code=True)
    # Saved Module2 maps/record metadata used by Module3. No ECG datasets.
    add_dir(M3/'inputs/module2_s9534','inputs/module2_s9534')
    # Scientific outputs up through faithfulness, including all raw 692 ECG curves.
    for folder in ['module3a_v1','module3b_primary_qc693_v1','module3b_sensitivity_694_v1',
                   'module3_faithfulness_primary_qc693_v1','m3b_full694_onednn_off_chunks',
                   'm3b_individual_300_349','m3b_sensitivity_1346']:
        add_dir(M3/'outputs'/folder,Path('outputs')/folder)
    # Selected failure and renderer diagnostics, not all Slurm logs.
    for glob in ['m3faith_2409801_602.log','m3faith_2410627_602.log',
                 'm3b_check406_2410671_*.log','m3b_debug406_2410679_*.log']:
        for f in sorted((M3/'logs').glob(glob)):
            add_file(f,Path('diagnostics')/f.name)
    # Scaler source file is small; follow link but archive under its descriptive name.
    scaler=SOURCES/'module2_runtime/results/module2/standard_scaler.npz'
    if scaler.is_file(): add_file(scaler,'inputs/reproducibility/standard_scaler.npz')
    add_file(REPORT,'documentation/DAN_MMGradCAM_Sampai_Faithfulness_Laporan.pdf')
    # Package user-visible handoff guide if it exists.
    for f in ['MODULE3_FAITHFULNESS_GUIDE.md','README_Module3_HPC.md']:
        matches=list((M3/'scripts').glob(f))
        for p in matches:add_file(p,Path('documentation')/f)
    if len(file_index)<100:
        raise SystemExit(f'STOP: Only {len(file_index)} files found, expected 692 raw+audit pairs')
    readme='''DAN-MMGradCAM | Reproducibility archive through Module3 Faithfulness

Snapshot was exported from the user's HPC filesystem. This archive contains LIVE code,
Module2 s9534 input maps, Module3A maps, Module3B strict/sensitivity maps,
692 completed faithfulness ECG raw CSV and audit JSON, and an EXPLORATORY
paired AUC summary computed on those 692 ECGs. It also includes the PDF report.

STATUS: Module3A 694 eligible; Module3B strict 693; Module3B sensitivity 694;
faithfulness strict QC 692 SUCCESS + 1 FAILED (index 602, ECG 406).
Missing ECG 406 was NOT fabricated, imputed, or silently accepted.

SCIENTIFIC LIMITATIONS: The live Matplotlib renderer demonstrably generates
history-dependent ECG406 image bytes (fresh vs warm). This impacts 2D baseline
reproduction and may influence perturbed-image deletion. Until audited, use
2D faithfulness as EXPLORATORY and do not claim superiority or causal faithfulness.
Module2 s9534 reports TF 2.21.0 whereas HPC module3 used TF 2.19.0.
Scaler fit count is not independently verified. Source-producing commit for s9534
has not been independently established.

ARCHIVE EXCLUSIONS: Raw ECG waveforms, image dataset, full model checkpoints, .venv,
credentials, complete Git history are NOT included. This is a code/results archive,
not a standalone end-to-end training environment.

Use MANIFEST_SHA256.json to check exported individual file integrity.
Do not claim 693 faithfulness results: only 692 passed strict QC.
'''
    prov={'created_utc':datetime.now(timezone.utc).isoformat(),
          'faithfulness_success':692,'faithfulness_failed_ecg':406,
          'failed_index':602,'module3a_eligible':694,'module3b_strict':693,
          'module3b_sensitivity':694,'source_git_module2':git_status(SOURCES/'module2_review'),
          'source_git_module3':git_status(repo),'included_files':len(file_index),
          'model_checkpoints_included':False,'raw_ecg_included':False}
    EXPORT.parent.mkdir(parents=True,exist_ok=True)
    if EXPORT.exists():raise SystemExit(f'STOP: archive already exists (no overwrite): {EXPORT}')
    manifest={}
    with zipfile.ZipFile(EXPORT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
        z.writestr('README_FIRST.txt',readme)
        z.writestr('PROVENANCE.json',json.dumps(prov,indent=2)+'\n')
        for arc,path in sorted(file_index.items()):
            z.write(path,arc)
            h=hashlib.sha256()
            with path.open('rb') as stream:
                while True:
                    chunk=stream.read(1024*1024)
                    if not chunk:break
                    h.update(chunk)
            manifest[arc]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
        z.writestr('MANIFEST_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    print(f'EXPORT SUCCESS: {EXPORT}')
    print(f'FILES: {len(file_index)} | ZIP SIZE MiB: {EXPORT.stat().st_size/(1024**2):.2f}')
    print('FAITHFULNESS: 692 completed strict-QC, ECG406 failed and preserved as failure.')

if __name__=='__main__':main()
