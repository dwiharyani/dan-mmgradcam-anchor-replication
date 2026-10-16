#!/usr/bin/env python3
"""Refuse accidental publication of raw ECG/data/models or suspicious secrets."""
import argparse
from pathlib import Path
import subprocess
import sys
import re

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_ROOTS={'scripts','figures','docs','tools','.github','reports'}
ALLOWED_SINGLE={'README.md','requirements.txt','.gitignore','.gitattributes','LICENSE_NOTICE.md'}
BANNED_EXT={'.csv','.tsv','.npy','.npz','.h5','.hdf5','.keras','.pt','.pth','.pkl','.pickle','.zip','.mat','.edf','.db','.sqlite','.parquet','.pem','.key','.env'}
BANNED_NAMES={'id_rsa','id_ed25519','credentials.json','secrets.json'}
BANNED_DIRS={'data','datasets','splits','results','models','checkpoints','weights','logs','.venv','venv'}
PATTERNS=[
    (re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'), 'private key'),
    (re.compile(r'(?i)(?:github_pat_|ghp_|gho_)[A-Za-z0-9_]{12,}'), 'GitHub token'),
    (re.compile(r'(?im)^\s*(?:AWS_SECRET_ACCESS_KEY|HF_TOKEN|WANDB_API_KEY|OPENAI_API_KEY)\s*[:=]'), 'credential assignment'),
    (re.compile(r'/home/g\d{6,}(/|\b)'), 'institutional home path'),
]

def paths_from_git(all_files=False):
    if all_files:
        try:
            out=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT)
        except subprocess.CalledProcessError:
            raise SystemExit('Run git add before --all; no Git repository yet.')
        return [Path(x.decode()) for x in out.split(b'\0') if x]
    return [p.relative_to(ROOT) for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]

def audit(paths):
    errs=[]
    for path in paths:
        parts=path.parts
        if not parts: continue
        if parts[0] not in ALLOWED_ROOTS and path.as_posix() not in ALLOWED_SINGLE:
            errs.append(f'Unexpected top-level file: {path}')
        if (parts[0] in BANNED_DIRS or (parts[0] != 'figures' and set(parts) & BANNED_DIRS) or path.suffix.lower() in BANNED_EXT or path.name.lower() in BANNED_NAMES):
            errs.append(f'Data/model/secret-like path: {path}')
        if path.parts[0]=='figures' and path.suffix.lower()!='.png':
            errs.append(f'Only curated PNG images allowed in figures/: {path}')
        f=ROOT/path
        if not f.is_file(): continue
        if f.stat().st_size>12_000_000:
            errs.append(f'Unusually large file (over 12MB): {path}')
        if f.suffix.lower() in {'.py','.md','.yml','.yaml','.txt'}:
            txt=f.read_text('utf-8',errors='replace')
            for pat,why in PATTERNS:
                if pat.search(txt): errs.append(f'Suspicious {why} in {path}')
    return errs

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--all',action='store_true',help='Check files already tracked by git')
    args=p.parse_args()
    paths=paths_from_git(args.all)
    errs=audit(paths)
    print('Audited',len(paths),'files')
    if errs:
        print('REJECT:');print('\n'.join(' - '+x for x in errs));sys.exit(1)
    print('PASS: No blocked paths or likely secrets found; manual review is still required.')
