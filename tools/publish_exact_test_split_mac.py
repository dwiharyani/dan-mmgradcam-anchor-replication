#!/usr/bin/env python3
"""Transfer verified test-split filenames from HPC and publish ONLY curated files.

From local Git checkout:
  python3 tools/publish_exact_test_split_mac.py fetch
  python3 tools/publish_exact_test_split_mac.py publish

Requires Mac ssh/scp/git and previously configured GitHub origin/main.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HOST = 'g202527350@hpcmaster-1.kfupm.edu.sa'
REMOTE_ROOT = '~/mm_gradcam/MM_GradCAM_HPC'
REMOTE_OUT = 'reproducibility/test_split_1054.csv'
FILENAMES = [
    'reproducibility/test_split_1054.csv',
    'reproducibility/test_split_1054_metadata.json',
    'tools/export_exact_test_split.py',
    'tools/verify_exact_test_split.py',
    'tools/publish_exact_test_split_mac.py',
    'tools/prepare_test_images.py',
    'docs/FRIEND_REPRODUCE_TESTSET.md',
    'README.md',
]
EXPECTED_GITHUB = 'dwiharyani/dan-mmgradcam-anchor-replication'


def run(args, **kwargs):
    print('$', ' '.join(str(x) for x in args), flush=True)
    subprocess.run(args, check=True, **kwargs)


def output(args):
    return subprocess.check_output(args, text=True).strip()


def root_of_repo():
    root = Path(output(['git', 'rev-parse', '--show-toplevel'])).resolve()
    if not (root / '.git').exists() and not (root / '.git').is_file():
        raise SystemExit('Expected a Git checkout')
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['fetch', 'publish'])
    p.add_argument('--host', default=HOST)
    args = p.parse_args()
    root = root_of_repo()
    origin = output(['git', 'remote', 'get-url', 'origin'])
    if EXPECTED_GITHUB not in origin:
        raise SystemExit(f'STOP: unexpected GitHub origin: {origin}')
    if args.mode == 'fetch':
        exporter = root / 'tools/export_exact_test_split.py'
        if not exporter.is_file():
            raise SystemExit('STOP: tools/export_exact_test_split.py missing. Extract toolkit into Git repo root first.')
        remote_cmd = f'python3 - --root {REMOTE_ROOT} --out {REMOTE_OUT}'
        print('Running standard-library-only exporter on HPC; no dataset is copied.', flush=True)
        run(['ssh', '-T', args.host, remote_cmd], input=exporter.read_bytes())
        local = root / 'reproducibility'
        local.mkdir(parents=True, exist_ok=True)
        for name in ['test_split_1054.csv', 'test_split_1054_metadata.json']:
            run(['scp', f'{args.host}:{REMOTE_ROOT}/reproducibility/{name}', str(local / name)])
        run([sys.executable, str(root / 'tools/verify_exact_test_split.py'),
             '--manifest', str(local / 'test_split_1054.csv'),
             '--metadata', str(local / 'test_split_1054_metadata.json')])
        print('\nFETCH OK. Review manifest and run "publish" when ready. No GitHub changes yet.')
        return
    branch = output(['git', 'branch', '--show-current'])
    if branch != 'main':
        raise SystemExit('STOP: expected main branch; currently on ' + branch)
    pre_staged = output(['git', 'diff', '--cached', '--name-only', '-z'])
    if pre_staged:
        raise SystemExit('STOP: Git index already has staged files. Commit/unstage them first; avoid mixing uploads.')
    local = root / 'reproducibility'
    run([sys.executable, str(root / 'tools/verify_exact_test_split.py'),
         '--manifest', str(local / 'test_split_1054.csv'),
         '--metadata', str(local / 'test_split_1054_metadata.json')])
    readme = root / 'README.md'
    if not readme.is_file():
        raise SystemExit('STOP: README.md missing')
    marker = '## Reproduce the exact 1,054-ECG held-out test set'
    text = readme.read_text(encoding='utf-8')
    if marker not in text:
        extra = (f'\n\n{marker}\n\n'
                 'To reproduce the **independent anchor MM-GradCAM** evaluation, '
                 'use the exact record-aligned test manifest (AFib 222, GSVT 222, '
                 'SB 388, SR 222) and the same trained checkpoints. '                 'The original ECG datasets are **not hosted here**.\n\n'
                 '- [Exact 1,054-record test manifest](reproducibility/test_split_1054.csv)\n'
                 '- [Step-by-step guide for collaborators](docs/FRIEND_REPRODUCE_TESTSET.md)\n'
                 '- [Trained models (GitHub Release v0.2.0)]'
                 '(https://github.com/dwiharyani/dan-mmgradcam-anchor-replication/releases/tag/v0.2.0-hpc-trained-models)\n'
                 '\nNumerical preprocessing, label mapping and inference settings must also '
                 'match; the fixed split alone does not guarantee identical metrics.\n')
        readme.write_text(text.rstrip() + extra, encoding='utf-8')
        print('README.md: appended reproducibility section')
    for rel in FILENAMES:
        if not (root / rel).is_file():
            raise SystemExit(f'STOP: required curated file absent {rel}')
    run(['git', 'add', '-f', '--', *FILENAMES], cwd=root)
    staged = [s for s in output(['git', 'diff', '--cached', '--name-only']).splitlines() if s]
    bad = [s for s in staged if s not in FILENAMES]
    if bad:
        raise SystemExit('STOP: unexpectedly staged entries: ' + repr(bad))
    if not staged:
        print('No differences to publish; GitHub already matches these files.')
        return
    print('\n=== ONLY THESE PATHS WILL BE COMMITTED ===')
    run(['git', 'diff', '--cached', '--stat'], cwd=root)
    answer = input('Type PUBLISH to commit and push curated manifest/code: ').strip()
    if answer != 'PUBLISH':
        run(['git', 'restore', '--staged', '--', *staged], cwd=root)
        raise SystemExit('Cancelled. No push performed.')
    run(['git', 'commit', '-m', 'Add exact 1054-record test split reproduction manifest'], cwd=root)
    run(['git', 'push', 'origin', 'main'], cwd=root)
    print('\nSUCCESS: reproducibility manifest and helper code uploaded; dataset excluded.')


if __name__ == '__main__':
    main()
