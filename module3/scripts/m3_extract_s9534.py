#!/usr/bin/env python3
"""Extract only the trusted-format Module2 s9534 numerical artifacts from user ZIP.

Never extract or unpickle checkpoint.pkl; never execute source inside the archive.
"""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

FILES=['noise_stability_maps.npz','noise_stability_records.csv','run_metadata.json','combined_R_D_maps.npz','combined_R_D_records.csv','agreement_from_module2_base_maps.csv']
PREFIX='Dan-mmgradcam/module2_full_v2_s9534/'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--zip',required=True)
    a.add_argument('--out',required=True)
    x=a.parse_args()
    src=Path(x.zip).expanduser();dst=Path(x.out).expanduser();dst.mkdir(parents=True,exist_ok=True)
    if not src.is_file():raise FileNotFoundError(src)
    manifest={'source_archive':str(src),'source_sha256':sha256(src),'format':'Module2 s9534 numerical artifacts','files':{}}
    with zipfile.ZipFile(src) as z:
        members=set(z.namelist())
        for name in FILES:
            member=PREFIX+name
            if member not in members:raise ValueError(f'Archive does not contain {member}')
            target=dst/name
            with z.open(member) as origin:
                if target.exists():
                    import tempfile,os
                    with tempfile.NamedTemporaryFile(dir=dst,delete=False) as temp:
                        shutil.copyfileobj(origin,temp)
                        tmp=Path(temp.name)
                    if sha256(tmp)!=sha256(target):
                        tmp.unlink()
                        raise RuntimeError(f'Existing file differs: {target}. Refusing overwrite; use new output folder')
                    tmp.unlink()
                else:
                    with open(target,'wb') as out:shutil.copyfileobj(origin,out)
            manifest['files'][name]={'bytes':target.stat().st_size,'sha256':sha256(target)}
    mp=dst/'input_manifest.json'
    mp.write_text(json.dumps(manifest,indent=2))
    print('Extracted/verified',len(FILES),'data files from s9534. Skipped untrusted pickle/checkpoints.')
    print('OUTPUT:',dst)

if __name__=='__main__':main()
